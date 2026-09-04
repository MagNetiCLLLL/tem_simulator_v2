"""Ordinary gun-tilt blanking uses installed coils and existing stops."""

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.calculation_cache import calculation_signatures
from temsim.optics.column import default_state
from temsim.profile_io import apply_profile_values, read_profile, save_profile


ALIGNMENT_FIELDS = (
    "upper_field_x_mt", "upper_field_y_mt",
    "lower_field_x_mt", "lower_field_y_mt", "enabled",
)


@pytest.mark.parametrize("gun_name", ("FEG", "FEG + Mono", "Thermionic"))
def test_ordinary_gun_tilt_physically_stops_beam_and_restores_source(gun_name):
    state = default_state()
    AssemblyCatalog().apply(
        state, AssemblySelection(gun_name, "C3 + Probe Corrector", "Energy Filter"),
    )
    gun = state.electron_gun
    original_geometry = tuple((part.key, part.center_z_mm) for part in state._resolved_assembly.parts)
    original_alignment = tuple(getattr(gun.deflector, name) for name in ALIGNMENT_FIELDS)
    opened = gun.trace_to_exit(128)
    assert np.any(opened.exit_bundle.alive)
    before = calculation_signatures(state)
    state.beam_blanked = True
    after = calculation_signatures(state)
    blocked = gun.trace_to_exit(128)
    assert not np.any(blocked.exit_bundle.alive)
    # Electrons that otherwise exit must hit physical gun components; neither
    # an optional module nor a synthetic loss flag accounts for this change.
    for index in np.flatnonzero(opened.exit_bundle.alive):
        assert blocked.blocked_key[index] in {
            gun.deflector.key, gun.stigmator.key, gun.c1_aperture.key,
        }
        assert gun.deflector.upper_center_from_tip_mm < blocked.blocked_z_mm[index] < gun.exit_plane_z_mm
    assert all(before[key] != after[key] for key in before)
    assert original_alignment == tuple(getattr(gun.deflector, name) for name in ALIGNMENT_FIELDS)
    assert original_geometry == tuple((part.key, part.center_z_mm) for part in state._resolved_assembly.parts)
    state.beam_blanked = False
    restored = gun.trace_to_exit(128)
    np.testing.assert_array_equal(restored.exit_bundle.alive, opened.exit_bundle.alive)
    for name in ("x_m", "y_m", "tx_rad", "ty_rad"):
        np.testing.assert_array_equal(getattr(restored.exit_bundle, name), getattr(opened.exit_bundle, name))
    assert calculation_signatures(state) == before


def test_blanking_drive_preserves_nonzero_disabled_gun_alignment():
    state = default_state()
    component = state.electron_gun.deflector
    for name, value in zip(ALIGNMENT_FIELDS, (0.12, -0.24, 0.31, -0.42, False)):
        setattr(component, name, value)
    expected = tuple(getattr(component, name) for name in ALIGNMENT_FIELDS)
    positions = np.asarray(((0.0, 0.0, component.upper_center_from_tip_mm * 1.0e-3),))
    np.testing.assert_array_equal(component.field_at_global_positions_t(positions), 0.0)
    state.beam_blanked = True
    np.testing.assert_allclose(
        component.field_at_global_positions_t(positions),
        ((0.0, component.blanking_field_y_mt * 1.0e-3, 0.0),),
    )
    state.beam_blanked = False
    np.testing.assert_array_equal(component.field_at_global_positions_t(positions), 0.0)
    assert tuple(getattr(component, name) for name in ALIGNMENT_FIELDS) == expected


def test_ordinary_blanking_persists_in_json_toml_and_across_gun_switches(tmp_path):
    state = default_state()
    state.electron_gun.deflector.upper_field_x_mt = 0.123
    state.beam_blanked = True
    payload = state.to_dict()
    component_key = state.electron_gun.deflector.key
    assert payload["electron_gun"]["components"][component_key]["beam_blanked"] is True
    assert "blanking_field_y_mt" not in payload["electron_gun"]["components"][component_key]
    restored = type(state).from_dict(payload)
    assert restored.beam_blanked
    assert restored.electron_gun.deflector.upper_field_x_mt == 0.123
    path = tmp_path / "blanked.toml"
    save_profile(path, state, AssemblyCatalog().default_selection())
    _, values = read_profile(path)
    restored.beam_blanked = False
    assert apply_profile_values(restored, values) == []
    assert restored.beam_blanked
    restored.select_electron_gun("thermionic")
    assert restored.beam_blanked
    restored.beam_blanked = False
    restored.select_electron_gun("cold_feg")
    assert not restored.beam_blanked
    assert restored.electron_gun.deflector.upper_field_x_mt == 0.123


def test_ordinary_and_optional_blank_controls_are_independent():
    state = default_state()
    for installed in (False, True):
        state.nanopulser.installed = installed
        state.nanopulser.blanked = True
        state.beam_blanked = True
        state.beam_blanked = False
        assert state.nanopulser.blanked is True
        state.beam_blanked = True
        state.nanopulser.blanked = False
        assert state.beam_blanked is True


def test_low_voltage_feg_blanking_finishes_finite_boris_trace():
    state = default_state()
    state.electron_gun.accelerator.high_tension_kv = 30.0
    state.beam_blanked = True
    blocked = state.electron_gun.trace_to_exit(49)
    assert not np.any(blocked.exit_bundle.alive)
    assert np.all(np.isfinite(blocked.blocked_z_mm))
    assert all(key for key in blocked.blocked_key)
    assert not any("stalled" in key for key in blocked.blocked_key)


def test_ordinary_blanker_pipeline_returns_zero_and_reopens_without_stale_cache():
    from temsim.simulation_pipeline import calculate

    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, catalog.default_selection())
    state.electron_gun.emitter.ray_count = 49
    state.step_mm = 0.1
    state.sample.eds_enabled = True
    state.beam_blanked = True
    progress = []
    blocked = calculate(state, progress_callback=lambda c, total, label: progress.append((c, total, label)))
    assert not state.nanopulser.installed
    assert not np.any(blocked.simulation.incident.alive)
    assert blocked.simulation.metrics["sample_surviving_current_pa"] == 0.0
    assert blocked.wave_imaging is None
    assert blocked.specimen_interactions is None
    assert blocked.energy_filter.eels_transmitted_current_pa == 0.0
    assert progress[-1][0] == progress[-1][1]
    state.beam_blanked = False
    state.sample.eds_enabled = False
    opened = calculate(state, existing_result=blocked)
    assert np.any(opened.simulation.incident.alive)
    assert opened.simulation.metrics["sample_surviving_current_pa"] > 0.0
    assert opened.simulation.metrics["column_segment_cache"]["mode"] != "full_incident"
    assert "column" in opened.calculated_products
