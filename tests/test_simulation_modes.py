"""Model selection tests use synthetic fields, not measured microscope data."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.optics.model import State
from temsim.simulation_modes import switch_mode, is_ideal, normalise_profiles, mode_key


def _state():
    return State.from_dict(default_state().to_dict())


def test_mode_switch_preserves_current_controls_geometry_and_resolution():
    state = _state()
    initial = deepcopy(state.to_dict())
    positions = [(lens.key, lens.z_mm) for lens in state.lenses]
    original = state.lenses[0].percent
    switch_mode(state, "ideal")
    assert state.lenses[0].percent == original
    state.lenses[0].percent = 12.5
    state.probe_aberrations = {"c3_mm": 5, "c1_mm": .001}
    switch_mode(state, "analytical")
    state.lenses[0].percent = 25
    switch_mode(state, "ideal")
    assert state.lenses[0].percent == 25
    assert state.probe_aberrations["c3_mm"] == 5
    assert [(lens.key, lens.z_mm) for lens in state.lenses] == positions
    assert state.step_mm == initial["step_mm"]
    assert state.electron_gun.emitter.ray_count == _state().electron_gun.emitter.ray_count
    switch_mode(state, "custom")
    assert state.lenses[0].percent == 25
    assert state.probe_aberrations == initial["probe_aberrations"]


def test_modes_roundtrip_and_legacy_states_keep_custom():
    state = _state()
    switch_mode(state, "ideal")
    restored = State.from_dict(state.to_dict())
    assert is_ideal(restored)
    assert restored.simulation_mode_profiles == state.simulation_mode_profiles
    restored.simulation_mode_profiles["custom"]["probe_aberrations"]["c3_mm"] = 7
    assert restored.simulation_mode_profiles != state.simulation_mode_profiles
    old = state.to_dict()
    old.pop("simulation_mode")
    old.pop("simulation_mode_profiles")
    assert mode_key(State.from_dict(old)) == "custom"


@pytest.mark.parametrize("key", ["nonlinear_material", "coupled_multiphysics", "invalid"])
def test_unavailable_or_unconfigured_modes_reject_without_mutation(key):
    state = _state()
    before = deepcopy(state.to_dict())
    with pytest.raises(ValueError):
        switch_mode(state, key)
    assert state.to_dict() == before


def test_linear_mode_rejects_missing_recipes_without_mutation():
    state = _state()
    before = deepcopy(state.to_dict())
    with pytest.raises(ValueError, match="not ready"):
        switch_mode(state, "linear_geometry")
    assert state.to_dict() == before


def test_configured_linear_mode_runs_geometry_field_and_never_falls_back():
    from temsim.physics.lens_field_provider import active_mapped_providers, FieldMapError
    state = _state()
    # Isolate one installed projector. These NI/mu values are synthetic test
    # inputs, not measured coil data or a production material assignment.
    for lens in state.lenses:
        lens.enabled = lens.key == "projector_lens_1"
    state.lens_field_map_descriptors = {"projector_lens_1": {
        "solver": "axisymmetric_linear_fem", "relative_permeability": 1,
        "ampere_turns": 100, "radial_nodes": 16, "axial_nodes": 24, "padding_factor": 2,
    }}
    switch_mode(state, "linear_geometry")
    # The installed tapered-pole profile is now supported without changing its
    # geometry. Synthetic NI/mu inputs do not calibrate a real projector.
    assert active_mapped_providers(state)[0].lens_key == "projector_lens_1"
    # A deliberately synthetic air-core variant also remains supported.
    from dataclasses import replace
    from temsim.magnetic_circuits import MAGNETIC_BODIES
    state._resolved_assembly = replace(state._resolved_assembly, parts=tuple(
        replace(part, data={**part.data, "magnetic_circuit_topology": "air_core"})
        if part.key == "projector_lens_1" else part
        for part in state._resolved_assembly.parts
        if part.data.get("mechanical_profile") not in MAGNETIC_BODIES
    ))
    providers = active_mapped_providers(state)
    assert len(providers) == 1
    assert providers[0].lens_key == "projector_lens_1"
    assert np.all(np.isfinite(providers[0].magnetic_field_t([state.projector_lens_p1.z_mm])))
    state.lens_field_map_descriptors.clear()
    with pytest.raises(FieldMapError, match="not ready"):
        active_mapped_providers(state)


def test_inactive_shelves_do_not_break_return_to_cached_identity():
    from temsim.calculation_cache import calculation_signatures
    state = _state()
    before = calculation_signatures(state)
    switch_mode(state, "ideal")
    ideal = calculation_signatures(state)
    assert before["incident"] != ideal["incident"]
    switch_mode(state, "custom")
    assert calculation_signatures(state) == before
    switch_mode(state, "ideal")
    state.simulation_mode_profiles["custom"]["probe_aberrations"]["c3_mm"] = 500
    assert calculation_signatures(state) == ideal


def test_ideal_and_analytical_bypass_saved_map_and_return_to_custom_map():
    from temsim.physics.lens_field_provider import (
        lens_geometry_binding, bind_imported_lens_field_map, resolve_runtime_lens_field_provider,
        active_mapped_providers, MagneticFieldMap, CoordinateRegistration, FieldMapProvenance,
        MappedLensFieldProvider,
    )
    state = _state()
    lens = state.projector_lens_p1
    binding = lens_geometry_binding(state, lens.key, lens)
    shape = (5, 5)
    field = MagneticFieldMap("axisymmetric_rz", (np.linspace(0, .01, 5), np.linspace(2, 3, 5)),
                             (np.zeros(shape), np.full(shape, .1)), CoordinateRegistration(),
                             binding.geometry_fingerprint, 100, 1,
                             FieldMapProvenance("fem", "", "0"*64, "Synthetic uniform test field"))
    bind_imported_lens_field_map(state, lens.key, field, native_provider=lens)
    assert isinstance(resolve_runtime_lens_field_provider(state, lens.key, lens), MappedLensFieldProvider)
    for mode in ("ideal", "analytical"):
        switch_mode(state, mode)
        assert active_mapped_providers(state) == ()
        provider = resolve_runtime_lens_field_provider(state, lens.key, lens)
        assert not isinstance(provider, MappedLensFieldProvider)
        np.testing.assert_allclose(provider.magnetic_field_t([lens.z_mm]), lens.magnetic_field_t([lens.z_mm]))
    switch_mode(state, "custom")
    assert isinstance(resolve_runtime_lens_field_provider(state, lens.key, lens), MappedLensFieldProvider)


def test_ideal_aberrations_keep_defocus_and_do_not_run_field_fit(monkeypatch):
    from temsim.optics import aberrations, field_aberrations
    from temsim.physics.chromatic import configured_objective_chromatic_focal_mm
    from temsim.physics.core import hexapole_field_components, spherical_aberration_kick_m3
    state = _state()
    switch_mode(state, "ideal")
    state.probe_aberrations = {"mode": "field_derived", "c1_mm": .002, "c3_mm": 999, "a2_mm": 5}
    state.chromatic_aberration_enabled = True
    monkeypatch.setattr(field_aberrations, "derive_field_aberrations", lambda *a: pytest.fail("Ideal must not fit"))
    aberrations.prepare_field_aberration_diagnostics(state)
    coefficients = aberrations.active_effective_aberrations(state, "probe")
    assert coefficients.c1_mm == .002
    assert coefficients.c3_mm == coefficients.a2_mm == coefficients.cc_mm == 0
    assert configured_objective_chromatic_focal_mm(state) is None
    z = np.linspace(0, 3000, 300)
    assert not np.any(spherical_aberration_kick_m3(z, state))
    assert not np.any(hexapole_field_components(z, state))
    assert state.probe_aberrations["c3_mm"] == 999


def test_ideal_propagation_is_linear_and_reference_energy_achromatic():
    from temsim.physics.core import propagate, build_propagation_plan, propagation_plan_common_prefix_nodes
    lens = _state().projector_lens_p1
    state = SimpleNamespace(lenses=[lens], stigmators=[], corrector_elements=[], simulation_mode="ideal",
                            step_mm=.25, history_step_mm=.5, beam_voltage_kv=300,
                            acceleration_enabled=False, acceleration_backend="CPU")
    z0, z1 = lens.z_mm-20, lens.z_mm+20
    x = np.array([1e-5, 2e-5, 1e-5])
    zero = np.zeros(3)
    energy = np.array([0., 0., 10000.])
    result = propagate(state, z0, z1, x, zero, zero, zero, energy_offset_ev=energy)
    for values in result[1:]:
        np.testing.assert_allclose(values[:, 1], 2*values[:, 0], rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(values[:, 2], values[:, 0], rtol=1e-12, atol=1e-15)
    np.testing.assert_array_equal(energy, [0, 0, 10000])
    ideal_plan = build_propagation_plan(state, z0, z1)
    state.simulation_mode = "analytical"
    analytical_plan = build_propagation_plan(state, z0, z1)
    assert ideal_plan.solver_signature != analytical_plan.solver_signature
    assert propagation_plan_common_prefix_nodes(ideal_plan, analytical_plan) == 0
    real = propagate(state, z0, z1, x, zero, zero, zero, energy_offset_ev=energy)
    assert not np.allclose(real[1][:, 2], real[1][:, 0], rtol=1e-7, atol=1e-14)


def test_profile_stores_model_and_shelves_and_old_profile_restores_custom(tmp_path):
    from temsim.profile_io import save_profile, read_profile, apply_profile_values
    from temsim.assembly_catalog import AssemblyCatalog
    state = _state()
    switch_mode(state, "ideal")
    state.lenses[0].percent = 13.5
    path = tmp_path / "mode.toml"
    save_profile(path, state, AssemblyCatalog().default_selection())
    _, values = read_profile(path)
    restored = _state()
    assert apply_profile_values(restored, values) == []
    assert mode_key(restored) == "ideal"
    assert restored.lenses[0].percent == 13.5
    assert restored.simulation_mode_profiles == state.simulation_mode_profiles
    apply_profile_values(restored, {})
    assert mode_key(restored) == "custom"


def test_main_menu_switch_preserves_result_cache_and_accuracy(qtbot, monkeypatch):
    from temsim.gui.main_window import MainWindow
    monkeypatch.setattr(MainWindow, "INITIAL_PREVIEW_DELAY_MS", 100000)
    window = MainWindow()
    qtbot.addWidget(window)
    window.preview_timer.stop()
    assert [action.text() for action in window.menuBar().actions()][:3] == ["File", "View", "Simulation"]
    assert is_ideal(window.state)
    menu = window.simulation_menu
    assert not menu.mode_actions["nonlinear_material"].isEnabled()
    assert not menu.mode_actions["coupled_multiphysics"].isEnabled()
    assert not menu.mode_actions["linear_geometry"].isEnabled()
    cached = object()
    window.calculations._high_cache["retained"] = cached
    before = (window.high_rays.value(), window.high_step.value(), [lens.percent for lens in window.state.lenses])
    monkeypatch.setattr(window.operating_presets, "submit", lambda *a, **k: pytest.fail("No preset solve"))
    menu.mode_actions["analytical"].trigger()
    window.preview_timer.stop()
    assert mode_key(window.state) == "analytical"
    assert window.simulation_mode_label.text() == "Analytical Field"
    assert window.calculations._high_cache["retained"] is cached
    assert before == (window.high_rays.value(), window.high_step.value(), [lens.percent for lens in window.state.lenses])


def test_profiles_cannot_restore_geometry():
    with pytest.raises(ValueError, match="unsupported"):
        normalise_profiles({"ideal": {"component_placements": {}}})
