"""Physical blanking: transverse impulse, exact stop and irreversible loss."""

from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.nanopulser import NanoPulser, electrostatic_deflection_rad
from temsim.physics import core
from temsim.physics.aperture_clipping import clip_segment


def _drift_state(**settings):
    settings.setdefault("voltage_v", 500.0)
    return SimpleNamespace(
        nanopulser=NanoPulser(**settings),
        beam_voltage_kv=300.0,
        lenses=[], stigmators=[], corrector_elements=[], apertures=[],
        sample=SimpleNamespace(z_mm=100.0),
        projector_mode="diffraction", step_mm=7.0, history_step_mm=40.0,
        acceleration_enabled=False, acceleration_backend="CPU",
    )


def _trace(state, start=0.0, stop=100.0):
    # Three on-axis-pupil rays, followed by an off-axis ray that should hit
    # the stop even when the electric field is off.
    x = np.asarray((-0.05, 0.0, 0.05, 0.2)) * 1.0e-3
    zero = np.zeros_like(x)
    return core.propagate(
        state, start, stop, x, zero, zero, zero,
        include_spherical_aberration=False, include_hexapole=False,
        checkpoint_z_mm=(stop,), return_checkpoints=True,
    )


@pytest.mark.parametrize("beam_kv", (80.0, 200.0, 300.0))
def test_voltage_deflection_matches_relativistic_lorentz_impulse(beam_kv):
    # Independent dimensional calculation of q E L / (p v).
    charge, mass, speed = 1.602176634e-19, 9.1093837015e-31, 299792458.0
    kinetic = beam_kv * 1000.0 * charge
    total = kinetic + mass * speed**2
    momentum = np.sqrt(total**2 - (mass * speed**2)**2) / speed
    velocity = momentum * speed**2 / total
    field_v_m, length_m = 500.0 / 0.001, 0.010
    expected = charge * field_v_m * length_m / (momentum * velocity)
    actual = electrostatic_deflection_rad(500.0, 10.0, 1.0, beam_kv)
    assert actual == pytest.approx(expected, rel=1.0e-14)
    assert electrostatic_deflection_rad(-500.0, 10.0, 1.0, beam_kv) == -actual
    assert electrostatic_deflection_rad(500.0, 10.0, 2.0, beam_kv) == actual / 2.0


def test_blank_field_hits_exact_stop_and_records_irreversible_loss():
    state = _drift_state(installed=True, blanked=True)
    z, x, tx, y, ty, checkpoints = _trace(state)
    assert state.nanopulser.stop_z_mm in z
    expected_angle = electrostatic_deflection_rad(500.0, 10.0, 1.0, 300.0)
    expected_displacement = expected_angle * 0.080
    assert checkpoints.tx_rad[-1, 1] == pytest.approx(expected_angle, abs=1.0e-16)
    assert checkpoints.x_m[-1, 1] == pytest.approx(expected_displacement, abs=1.0e-16)
    alive, blocked_z, blocked_key = clip_segment(state, z, x, y)
    assert not np.any(alive)
    np.testing.assert_array_equal(blocked_z, 60.0)
    assert blocked_key == ["nanopulser_aperture"] * 4

    # Downstream recentering cannot revive intercepted electrons.
    downstream_z = np.asarray((80.0, 100.0))
    downstream_xy = np.zeros((2, 4))
    survived, stopped_z, stopped_key = clip_segment(
        state, downstream_z, downstream_xy, downstream_xy,
        alive, blocked_z, blocked_key,
    )
    assert not np.any(survived)
    np.testing.assert_array_equal(stopped_z, blocked_z)
    assert stopped_key == blocked_key


@pytest.mark.parametrize("blanked,voltage_v", ((False, 500.0), (True, 0.0)))
def test_open_or_zero_voltage_transmits_pupil_but_stop_remains(blanked, voltage_v):
    state = _drift_state(installed=True, blanked=blanked, voltage_v=voltage_v)
    z, x, _tx, y, _ty, _checkpoints = _trace(state)
    alive, blocked_z, blocked_key = clip_segment(state, z, x, y)
    np.testing.assert_array_equal(alive, (True, True, True, False))
    assert np.all(np.isnan(blocked_z[:3]))
    assert blocked_z[-1] == 60.0
    assert blocked_key == ["", "", "", "nanopulser_aperture"]


def test_uninstalled_device_does_not_change_propagation_or_clip():
    state = _drift_state(installed=False, blanked=True)
    inactive_plan = core.build_propagation_plan(state, 0.0, 100.0)
    z, x, _tx, y, _ty, _checkpoints = _trace(state)
    alive, _, _ = clip_segment(state, z, x, y)
    assert np.all(alive)
    del state.nanopulser
    baseline_plan = core.build_propagation_plan(state, 0.0, 100.0)
    assert inactive_plan.signature == baseline_plan.signature


def test_rotation_and_post_blanker_segment_do_not_reapply_impulse():
    state = _drift_state(installed=True, blanked=True, azimuth_deg=90.0)
    full = _trace(state)[5]
    assert full.tx_rad[-1, 1] == pytest.approx(0.0, abs=1.0e-16)
    assert full.ty_rad[-1, 1] > 0.0
    downstream = _trace(state, start=80.0)[5]
    assert downstream.tx_rad[-1, 1] == 0.0
    assert downstream.ty_rad[-1, 1] == 0.0


def test_profile_roundtrip_excludes_instrument_geometry():
    original = NanoPulser(
        installed=True, blanked=True, voltage_v=-400.0, azimuth_deg=45.0,
        z_mm=470.0, stop_z_mm=510.0,
    )
    payload = original.to_dict()
    assert payload == {
        "installed": True, "blanked": True,
        "voltage_v": -400.0, "azimuth_deg": 45.0,
    }
    restored = NanoPulser.from_dict(payload | {"z_mm": 9999.0})
    assert restored.to_dict() == payload
    assert restored.z_mm != 9999.0


def test_blanking_invalidates_optics_and_downstream_product_caches():
    from temsim.calculation_cache import calculation_signatures

    state = _drift_state(installed=True, blanked=False)
    state.to_dict = lambda: {
        "electron_gun": {}, "sample": {},
        "nanopulser": state.nanopulser.to_dict(),
    }
    opened_products = calculation_signatures(state)
    opened_plan = core.build_propagation_plan(state, 0.0, 100.0)
    state.nanopulser.blanked = True
    blanked_products = calculation_signatures(state)
    blanked_plan = core.build_propagation_plan(state, 0.0, 100.0)
    for product in opened_products:
        assert opened_products[product] != blanked_products[product], product
    assert opened_plan.signature != blanked_plan.signature
    prefix_nodes = core.propagation_plan_common_prefix_nodes(opened_plan, blanked_plan)
    assert prefix_nodes == 0 or opened_plan.z_mm[prefix_nodes - 1] < state.nanopulser.z_mm


def test_aperture_display_records_include_installed_physical_stop():
    from temsim.simulation_pipeline import aperture_stop_records

    state = _drift_state(installed=True)
    state.electron_gun = SimpleNamespace()
    records = aperture_stop_records(state)
    assert len(records) == 1
    assert records[0]["key"] == "nanopulser_aperture"
    assert records[0]["z_mm"] == 60.0
    assert records[0]["diameter_mm"] == 0.2
    assert records[0]["enabled"] is True


@pytest.mark.parametrize("settings", (
    {"voltage_v": float("nan")}, {"plate_gap_mm": 0.0},
    {"stop_z_mm": 21.0}, {"installed": "false"},
))
def test_invalid_blank_geometry_or_voltage_is_rejected(settings):
    with pytest.raises(ValueError):
        NanoPulser(**settings).validate()
