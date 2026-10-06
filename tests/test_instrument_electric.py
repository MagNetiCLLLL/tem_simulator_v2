"""Canonical E-domain/cache contracts; tiny fields are explicitly unsolved."""
from collections import OrderedDict
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.physics.closed_gun_field import ClosedGunField, closed_field_request, mesh_axes
from temsim.physics.planar_gun_field import request_digest
from temsim.physics.instrument_electric import (
    InstrumentElectricField, capture_instrument_electric_field,
    configure_instrument_electric_domain, instrument_electric_end_mm,
)
from temsim.physics.gun_field_environment import instrument_gun_field_context
from temsim.test_electron_scene import _prepare_electric_provider, prepare_test_electron_scene


@pytest.fixture
def tiny_solver(monkeypatch):
    from temsim.physics import closed_gun_field as module
    calls = []
    def build(request, directory):
        calls.append(request)
        r = np.linspace(0., request['domain']['outer_radius_m'], 3)
        z = np.linspace(request['domain']['entrance_m'], request['domain']['exit_m'], 5)
        voltage = np.broadcast_to(z*1000., (len(r), len(z))).copy()
        return ClosedGunField(request, r, z, voltage, {'request_sha256': request_digest(request)})
    monkeypatch.setattr(module, '_MEMORY_FIELDS', OrderedDict())
    monkeypatch.setattr(module, '_build_request_field', build)
    monkeypatch.setattr(module, '_trim_disk_cache', lambda *args: None)
    return calls


def test_tip_capture_and_all_cutoffs_use_one_exact_cached_solution(tiny_solver):
    state = default_state()
    gun = state.electron_gun
    original = closed_field_request(gun)
    with instrument_gun_field_context(state):
        main = gun.electric_field
        actual_request = closed_field_request(gun)
    captured = capture_instrument_electric_field(state)
    assert captured.provider is main
    assert captured.request_identity == request_digest(actual_request)
    assert captured.physical_identity and captured.numerical_identity
    assert not hasattr(gun, '_instrument_electric_end_mm')
    assert closed_field_request(gun) == original
    positions = np.array([[0., 0., .1], [1e-6, 2e-6, .45], [1e-5, 0., 2.7]])
    for stop in (.2, 1.7, 3.0264):
        _, field, _ = _prepare_electric_provider(state, stop)
        assert field is main
        np.testing.assert_array_equal(captured.interpolate(positions)[0], field.potential_v_at_global_positions(positions))
        np.testing.assert_array_equal(captured.interpolate(positions)[1], field.field_at_global_positions_v_per_m(positions))
    assert len(tiny_solver) == 1
    assert not captured.is_constant_on_interval(550., 3000.)


def test_fixed_mesh_keeps_physical_end_face_without_sub_ulp_duplicate():
    state = default_state()
    gun = state.electron_gun
    with instrument_gun_field_context(state):
        request = closed_field_request(gun)
    end = instrument_electric_end_mm(state)*1e-3
    assert request['domain']['exit_m'] == end
    r, z = mesh_axes(request)
    assert z[-1] == end and np.all(np.diff(z) > 0.)
    assert len(r)*len(z) < 1_000_000


def test_continuous_curvature_request_uses_same_fixed_mechanical_endpoint():
    from temsim.physics.continuous_gun_field import continuous_field_request
    state = default_state()
    gun = state.electron_gun
    gun.emitter.curvature_nm_inv = .02
    with instrument_gun_field_context(state):
        request = continuous_field_request(gun)
    assert request['domain']['exit_m'] == instrument_electric_end_mm(state)*1e-3
    assert request['source_admission'] == 'continuous_curvature_classical_tip'
    assert gun.emitter.curvature_nm_inv == .02


def test_electrode_change_invalidates_solution_but_observation_cutoff_does_not(tiny_solver):
    state = default_state()
    before = capture_instrument_electric_field(state)
    state.electron_gun.extractor.voltage_kv += .1
    after = capture_instrument_electric_field(state)
    assert before.provider is not after.provider
    assert before.request_identity != after.request_identity
    assert before.physical_identity != after.physical_identity
    with pytest.raises(ValueError, match='outside the fixed instrument'):
        _prepare_electric_provider(state, after.bounds_m[1, 2]+.001)
    assert len(tiny_solver) == 2


def test_downstream_pure_magnetic_tuning_keeps_full_electric_solution(tiny_solver):
    from temsim.component_keys import DIFFRACTION_LENS
    state = default_state()
    before = capture_instrument_electric_field(state)
    next(lens for lens in state.lenses if lens.key == DIFFRACTION_LENS).percent += 1.
    after = capture_instrument_electric_field(state)
    assert before.provider is after.provider
    assert before.request_identity == after.request_identity
    assert after.bounds_m[1, 2] > 3.
    with instrument_gun_field_context(state):
        assert state.electron_gun.electric_field is after.provider
        assert state.electron_gun._instrument_magnetic_query_upper_m < after.bounds_m[1, 2]
    assert len(tiny_solver) == 1


def test_field_capture_does_not_copy_executed_particle_population(tiny_solver):
    class MustNotCopy:
        def __deepcopy__(self, memo):
            raise AssertionError('Executed particle histories are not field inputs')
    state = default_state()
    executed = MustNotCopy()
    state.electron_gun._trace_cache = executed
    result = capture_instrument_electric_field(state)
    assert result.gun_snapshot._trace_cache is None
    assert state.electron_gun._trace_cache is executed


def test_scene_captures_immutable_actual_instrument_column_inputs(tiny_solver):
    from temsim.instrument_snapshot import decode_instrument
    from temsim.magnetic_field_scene import prepare_magnetic_scene
    from temsim.immutable_json import json_digest
    state = default_state()
    magnetic = prepare_magnetic_scene(state)
    scene = prepare_test_electron_scene(state, magnetic, z_limits_mm=(0., 3000.))
    assert scene._column_handoff_z_m == state.electron_gun.exit_plane_z_mm*1e-3
    assert scene._column_identity == json_digest(scene._column_input_graph)
    with pytest.raises(TypeError):
        scene._column_input_graph['new'] = 'input'
    restored = decode_instrument(scene._column_input_graph)
    assert restored.electron_gun.high_tension_kv == state.electron_gun.high_tension_kv
    assert not hasattr(restored.electron_gun, '_instrument_electric_end_mm')
    state.electron_gun.high_tension_kv -= 1.
    assert restored.electron_gun.high_tension_kv != state.electron_gun.high_tension_kv


def test_unsupported_real_instrument_model_is_not_silently_omitted(tiny_solver):
    from temsim.magnetic_field_scene import prepare_magnetic_scene
    state = default_state()
    magnetic = prepare_magnetic_scene(state)
    state.unknown_model = object()
    with pytest.raises(TypeError, match='Unregistered working-point model'):
        prepare_test_electron_scene(state, magnetic, z_limits_mm=(0., 3000.))


def test_compute_policy_changes_execution_identity_without_changing_captured_fields(tiny_solver):
    from temsim.magnetic_field_scene import prepare_magnetic_scene
    from temsim.diagnostic_execution_identity import trajectory_execution_identity
    from temsim.magnetic_test_particle import TestElectronSettings
    state = default_state()
    state.acceleration_backend, state.acceleration_enabled = "CPU", False
    magnetic = prepare_magnetic_scene(state)
    cpu = prepare_test_electron_scene(state, magnetic, z_limits_mm=(0., 3000.))
    state.acceleration_backend, state.acceleration_enabled = "Require GPU", True
    gpu = prepare_test_electron_scene(state, magnetic, z_limits_mm=(0., 3000.))
    assert cpu.physical_identity == gpu.physical_identity
    assert cpu.numerical_identity == gpu.numerical_identity
    assert cpu._column_identity != gpu._column_identity
    assert cpu.transport_identity != gpu.transport_identity
    settings = TestElectronSettings()
    assert trajectory_execution_identity(cpu, settings) != trajectory_execution_identity(gpu, settings)
    assert len(tiny_solver) == 1


def test_stale_magnetic_capture_cannot_rebuild_a_different_column(tiny_solver):
    from temsim.magnetic_field_scene import prepare_magnetic_scene
    state = default_state()
    magnetic = prepare_magnetic_scene(state)
    state.lenses[0].percent += 1.
    with pytest.raises(ValueError, match='differs from the instrument inputs'):
        prepare_test_electron_scene(state, magnetic, z_limits_mm=(0., 3000.))


def test_app_display_crop_is_verified_and_expanded_to_complete_transport_field(tiny_solver):
    from temsim.magnetic_field_scene import prepare_magnetic_scene
    state = default_state()
    complete = prepare_magnetic_scene(state)
    cropped = prepare_magnetic_scene(state, z_limits_mm=(1000., 1700.))
    assert len(cropped._sources) < len(complete._sources)
    scene = prepare_test_electron_scene(state, cropped, z_limits_mm=(0., 3000.))
    assert scene.magnetic_scene.physical_identity == complete.physical_identity
    assert len(scene.magnetic_scene._sources) == len(complete._sources)
    assert scene._column_input_graph is not None
    assert any('display crop is expanded' in note for note in scene.notes)
    # A matching view window cannot hide an old command in an overlapping coil.
    lens = next(lens for lens in state.lenses if lens.key in cropped.source_keys)
    lens.percent += 1.
    with pytest.raises(ValueError, match='differs from the instrument inputs'):
        prepare_test_electron_scene(state, cropped, z_limits_mm=(0., 3000.))


def test_unknown_real_scene_identity_retains_actual_fields_without_column_replay(tiny_solver):
    from temsim.magnetic_field_scene import prepare_magnetic_scene
    state = default_state()
    magnetic = replace(prepare_magnetic_scene(state), physical_identity=None, numerical_identity=None)
    scene = prepare_test_electron_scene(state, magnetic, z_limits_mm=(0., 3000.))
    assert scene._column_input_graph is scene._column_handoff_z_m is scene._column_identity is None
    assert scene.diagnostic_fields_at_global_position((0., 0., .001)) is not None
    assert any('retains full time-domain' in note for note in scene.notes)


def test_record_rejects_queries_outside_fixed_domain(tiny_solver):
    record = capture_instrument_electric_field(default_state())
    for position in ((0., 0., record.bounds_m[1, 2]+1e-9),
                     (record.bounds_m[1, 0], record.bounds_m[1, 0], .1), (0., 0., np.nan)):
        with pytest.raises(ValueError):
            record.interpolate([position])


def test_historical_surface_model_keeps_declared_field_and_exact_zero_extension(monkeypatch):
    from temsim.optics.electron_gun.tip_surface import load_tip_surface_reference
    from temsim.physics.grounded_tip_field import GroundedTipField, field_request
    state = default_state()
    gun = state.electron_gun
    gun.emitter.surface_model = load_tip_surface_reference()
    before = field_request(gun)
    historical = GroundedTipField.__new__(GroundedTipField)
    historical.r, historical.z = np.array([0., .01]), np.array([0., .45])
    monkeypatch.setattr('temsim.physics.grounded_tip_field.grounded_field', lambda gun: historical)
    captured = capture_instrument_electric_field(state)
    assert captured.base_field is historical
    assert field_request(captured.gun_snapshot) == before
    assert not hasattr(captured.gun_snapshot, '_instrument_electric_end_mm')
    assert captured.is_constant_on_interval(450., 3000.)
    assert not captured.is_constant_on_interval(449., 3000.)
    assert any('Historical surface-model' in note for note in captured.notes)
    historical.field_at_global_positions_v_per_m = lambda p: np.ones_like(p)
    assert not captured.is_constant_on_interval(450., 3000.)


def test_potential_rise_gauge_and_wien_field_are_added_once():
    class Base:
        def potential_v_at_global_positions(self, points):
            return np.full(points.shape[:-1], -299999.)
        def potential_rise_v_at_global_positions(self, points):
            return np.ones(points.shape[:-1])
        def field_at_global_positions_v_per_m(self, points):
            return np.broadcast_to((0., 0., 2.), points.shape)
    class Wien:
        def potential_v_at_global_positions(self, points):
            return np.full(points.shape[:-1], 3.)
        def field_at_global_positions_v_per_m(self, points):
            return np.broadcast_to((4., 0., 0.), points.shape)
    base = Base()
    provider = SimpleNamespace(base_field=base, wien_field=Wien())
    record = InstrumentElectricField(provider, base, None,
                np.array(((-1.,-1.,0.),(1.,1.,1.))), None, None, None, ())
    potential, electric = record.interpolate([[0.,0.,.1]])
    np.testing.assert_array_equal(potential, [4.])
    np.testing.assert_array_equal(electric, [[4.,0.,2.]])


def _filter_state():
    from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
    state = default_state()
    AssemblyCatalog().apply(state, AssemblySelection("FEG", "C3", "Energy Filter"))
    return state


@pytest.mark.parametrize("curvature", [0., .02])
def test_installed_filter_fixed_electric_domain_reaches_declared_axial_handoff(curvature):
    from temsim.physics.particle_sections import section_limits
    from temsim.physics.closed_gun_field import _physical_liner_rows
    state = _filter_state()
    gun, assembly = state.electron_gun, state._resolved_assembly
    gun.emitter.curvature_nm_inv = curvature
    handoff = assembly.part("energy_filter").center_z_mm
    assert handoff == assembly.part("energy_filter_entrance_aperture").center_z_mm
    assert instrument_electric_end_mm(state) == handoff == section_limits(state)[1]
    assert handoff < assembly.exit_z_mm
    assert max(row.end_z_mm for row in assembly.vacuum_liner_segments) == handoff
    physical = _physical_liner_rows(gun, None)
    assert physical[-1]["stop_m"] == handoff * 1e-3
    with instrument_gun_field_context(state):
        request = closed_field_request(gun)
    assert request["domain"]["exit_m"] == handoff * 1e-3
    assert request["grounded_liner"] == physical
    assert not hasattr(gun, "_instrument_electric_end_mm")


def test_filter_inlet_carrier_interval_is_covered_by_same_cached_electric_field(tiny_solver):
    state = _filter_state()
    inlet = state._resolved_assembly.part("energy_filter_entrance_aperture")
    captured = capture_instrument_electric_field(state)
    assert captured.bounds_m[1, 2] == inlet.center_z_mm * 1e-3
    for stop in (1.7, inlet.start_z_mm * 1e-3, (inlet.center_z_mm - .1) * 1e-3, inlet.center_z_mm * 1e-3):
        _, provider, _ = _prepare_electric_provider(state, stop)
        assert provider is captured.provider
        captured.interpolate([[0., 0., stop]])
    with pytest.raises(ValueError, match="outside the fixed instrument"):
        _prepare_electric_provider(state, inlet.center_z_mm * 1e-3 + 1e-6)
    assert len(tiny_solver) == 1


def test_filter_domain_does_not_follow_observation_or_carrier_envelope():
    state = _filter_state()
    assembly = state._resolved_assembly
    end = instrument_electric_end_mm(state)
    state.camera.inserted = not state.camera.inserted
    state.step_mm *= 2.
    state._resolved_assembly = replace(assembly, parts=tuple(
        replace(part, start_z_mm=part.start_z_mm - 1., length_mm=part.length_mm + 1.)
        if part.key == "energy_filter_entrance_aperture" else part for part in assembly.parts))
    assert instrument_electric_end_mm(state) == end


def test_filter_domain_rejects_truncated_liner_instead_of_silently_shortening():
    state = _filter_state()
    gun = state.electron_gun
    rows = list(gun._grounded_outlet_liner_segments)
    last = max(range(len(rows)), key=lambda index: rows[index].end_z_mm)
    rows[last] = replace(rows[last], end_z_mm=rows[last].end_z_mm - .1)
    gun._grounded_outlet_liner_segments = tuple(rows)
    with pytest.raises(ValueError, match="not covered by the connected grounded liner"):
        instrument_electric_end_mm(state)


def test_filter_field_request_still_rejects_disconnected_actual_liner():
    state = _filter_state()
    gun = state.electron_gun
    rows = list(gun._grounded_outlet_liner_segments)
    last = max(range(len(rows)), key=lambda index: rows[index].end_z_mm)
    rows[last] = replace(rows[last], start_z_mm=rows[last].start_z_mm + .1)
    gun._grounded_outlet_liner_segments = tuple(rows)
    # Endpoint coverage alone is insufficient: the ordinary field request
    # must retain its complete continuity validation before any solver runs.
    assert instrument_electric_end_mm(state) == state.energy_filter.entrance_z_mm
    with instrument_gun_field_context(state), pytest.raises(ValueError, match="contiguous and non-overlapping"):
        closed_field_request(gun)


def test_filter_domain_rejects_inconsistent_resolved_interface():
    state = _filter_state()
    state._resolved_assembly = replace(state._resolved_assembly, parts=tuple(
        replace(part, center_z_mm=part.center_z_mm + 1.) if part.key == "energy_filter" else part
        for part in state._resolved_assembly.parts))
    with pytest.raises(ValueError, match="handoff must coincide"):
        instrument_electric_end_mm(state)
