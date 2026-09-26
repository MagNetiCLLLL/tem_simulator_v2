"""Physical field routing and conductor events, with small analytical fixtures."""

from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.electron_gun.field_emission import FieldEmissionGun
from temsim.optics.electron_gun import tracing
from temsim.physics.relativistic_lorentz import (
    RelativisticPhaseSpace, kinetic_energy_ev_from_momentum,
    momentum_from_kinetic_energy_ev,
)


def test_flat_source_selects_geometry_without_changing_emission():
    gun = FieldEmissionGun()
    before = gun.emit(9)
    gun.validate()
    after = gun.emit(9)
    for name in ("x_m", "y_m", "tx_rad", "ty_rad", "energy_offset_ev", "weight", "ray_id"):
        np.testing.assert_array_equal(getattr(before, name), getattr(after, name))
    assert gun.uses_geometry_electric_field
    assert gun.to_dict()["integrator"]["method"] == "static_discrete_gradient"
    # Accelerator gaps are vacuum with a solved field, not compact drift windows.
    assert gun.integration_step_mm_at(np.array([100., 110.])) == gun.trace_step_mm


def test_closed_field_failure_cannot_select_an_analytic_fallback(monkeypatch):
    import temsim.physics.closed_gun_field as fields
    def failed(_gun):
        raise ValueError("deliberate solved-field failure")
    monkeypatch.setattr(fields, "closed_field", failed)
    with pytest.raises(ValueError, match="deliberate solved-field failure"):
        FieldEmissionGun().electric_field


def test_installed_monochromator_keeps_its_added_electric_and_magnetic_fields(monkeypatch):
    import temsim.physics.closed_gun_field as fields
    base = SimpleNamespace(
        potential_v_at_global_positions=lambda p: 1000.*p[:, 2],
        potential_rise_v_at_global_positions=lambda p: 1000.*p[:, 2],
        field_at_global_positions_v_per_m=lambda p: np.tile([0., 0., -1000.], (len(p), 1)),
    )
    monkeypatch.setattr(fields, "closed_field", lambda gun: base)
    gun = FieldEmissionGun()
    gun.install_monochromator()
    z = gun.monochromator.wien.optical_reference_from_tip_mm*1e-3
    p = np.array([[1e-5, 0., z]])
    actual = gun.electric_field
    assert actual.base_field is base
    wien = gun.monochromator.field_provider
    np.testing.assert_allclose(actual.field_at_global_positions_v_per_m(p),
                               base.field_at_global_positions_v_per_m(p)+wien.field_at_global_positions_v_per_m(p))
    np.testing.assert_allclose(actual.potential_rise_v_at_global_positions(p),
                               base.potential_rise_v_at_global_positions(p)+wien.potential_v_at_global_positions(p))
    assert gun.magnetic_field.wien_field.element is gun.monochromator.wien
    assert gun.local_wien_reference_energy_ev == pytest.approx(gun.emitter.emission_energy_ev+1000.*z)


def liner(start=.37, stop=.45, radius=.003, key="grounded_outlet:test"):
    return {"key": key, "start_m": start, "stop_m": stop,
            "inner_m": radius, "outer_m": radius+.001}


def clip(rows, previous, current, *, preceding=np.nan, eligible=True):
    previous, current = np.asarray([previous], float), np.asarray([current], float)
    alive, blocked, keys = np.array([not np.isfinite(preceding)]), np.array([preceding]), ["body" if np.isfinite(preceding) else ""]
    tracing._clip_grounded_liner(rows, previous, current, np.array([eligible]), alive, blocked, keys)
    return bool(alive[0]), float(blocked[0]), keys[0]


def test_grounded_liner_stops_at_first_radial_intersection():
    alive, z, key = clip([liner()], [0., 0., .38], [.004, 0., .4])
    assert not alive and key == "grounded_outlet:test"
    assert z == pytest.approx(395.)
    assert clip([liner()], [0., 0., .38], [.002, 0., .4])[0]


def test_grounded_liner_bore_step_is_a_physical_face():
    rows = [liner(stop=.4, radius=.005), liner(start=.4, key="narrow_tube")]
    alive, z, key = clip(rows, [.004, 0., .39], [.004, 0., .41])
    assert not alive and key == "narrow_tube"
    assert z == pytest.approx(400.)


def test_liner_interception_preserves_earliest_stop_and_completed_rays():
    assert clip([liner()], [0., 0., .38], [.004, 0., .4], preceding=390.)[1:] == (390., "body")
    assert clip([liner()], [0., 0., .38], [.004, 0., .4], preceding=399.)[1] == pytest.approx(395.)
    assert clip([liner()], [0., 0., .38], [.004, 0., .4], eligible=False)[0]


def test_liner_handles_zero_axial_motion_and_reverse_motion():
    alive, z, _ = clip([liner()], [0., 0., .4], [.004, 0., .4])
    assert not alive and z == pytest.approx(400.)
    alive, z, _ = clip([liner()], [0., 0., .4], [.004, 0., .38])
    assert not alive and z == pytest.approx(385.)


def test_discrete_gradient_retains_actual_work_without_nominal_energy_projection(monkeypatch):
    # Independent uniform electrostatic field fixture, not a full-gun claim.
    electric = SimpleNamespace(
        potential_v_at_global_positions=lambda p: 2e5*p[:, 2],
        potential_rise_v_at_global_positions=lambda p: 2e5*p[:, 2],
        field_at_global_positions_v_per_m=lambda p: np.tile([0., 0., -2e5], (len(p), 1)),
    )
    magnetic = SimpleNamespace(field_at_global_positions_t=lambda p: np.zeros_like(p))
    initial = RelativisticPhaseSpace(np.zeros((1, 3)), momentum_from_kinetic_energy_ev(np.array([100.]), np.array([[.1, 0., 1.]])))
    def prohibited(*args, **kwargs):
        raise AssertionError("geometric integration must not project nominal energy")
    monkeypatch.setattr(tracing, "_enforce_static_field_energy", prohibited)
    final, _ = tracing._surface_step(initial, 1e-13, np.array([True]), magnetic, electric)
    expected = 100.+electric.potential_v_at_global_positions(final.position_m)
    np.testing.assert_allclose(kinetic_energy_ev_from_momentum(final.momentum_kg_m_per_s), expected, rtol=0., atol=1e-9)
    np.testing.assert_allclose(final.momentum_kg_m_per_s[:, :2], initial.momentum_kg_m_per_s[:, :2], rtol=1e-12, atol=0.)
