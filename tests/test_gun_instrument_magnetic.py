"""Shared instrument B in production gun execution; tiny step fixtures only.

These checks establish provider/cache/compiled-law parity, not full microscope
trajectory qualification or the separate electric-domain parity requirement.
"""
from dataclasses import replace

import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.component_keys import CONDENSER_LENS_1, CONDENSER_LENS_2, DIFFRACTION_LENS, PROJECTOR_LENS_2
from temsim.physics.gun_field_environment import instrument_gun_field_context, gun_transport_magnetic_field
from temsim.physics.instrument_magnetic import capture_instrument_magnetic_field
from temsim.physics.compiled_magnetic_field import prepare_compiled_magnetic_sources, compiled_magnetic_batch


def _key(state):
    with instrument_gun_field_context(state):
        return state.electron_gun._cache_key(9)


@pytest.mark.parametrize('key', [CONDENSER_LENS_2, DIFFRACTION_LENS, PROJECTOR_LENS_2])
def test_downstream_magnetic_tuning_keeps_executed_gun_dependency(key):
    from temsim.physics.particle_sections import gun_dependency_signature
    from temsim.instrument_snapshot import encode_instrument, decode_instrument
    state = default_state()
    before = _key(state)
    signature = gun_dependency_signature(state)
    # Reconstruct the same physical inputs, as continuation in a new process
    # does; no runtime field bindings are exported to create this equality.
    state = decode_instrument(encode_instrument(state))
    next(lens for lens in state.lenses if lens.key == key).percent += 1.
    assert _key(state) == before
    assert gun_dependency_signature(state) == signature
    with instrument_gun_field_context(state):
        gun = state.electron_gun
        assert gun._instrument_magnetic_query_upper_m == pytest.approx(.55)
        assert gun._instrument_electric_end_mm > 3000.
        assert all(start <= 550. for start, _ in gun._instrument_magnetic_supports_mm)


def test_overlapping_lens_tail_invalidates_executed_gun_dependency():
    state = default_state()
    before = _key(state)
    c1 = next(lens for lens in state.lenses if lens.key == CONDENSER_LENS_1)
    c1.percent += 1.
    assert _key(state) != before
    assert not hasattr(state.electron_gun, "_instrument_magnetic_field")


def test_query_extension_changes_dependencies_when_it_admits_another_field():
    state = default_state()
    before = _key(state)
    state.electron_gun._gun_field_exit_extension_mm = 300.
    extended = _key(state)
    assert extended != before
    next(lens for lens in state.lenses if lens.key == CONDENSER_LENS_2).percent += 1.
    assert _key(state) != extended


def test_dispatch_binds_full_instrument_once_and_restores_standalone(monkeypatch):
    from temsim.optics.electron_gun.source import trace_source_to_exit
    state = default_state()
    gun = state.electron_gun
    gun.deflector.upper_field_y_mt = .3
    reference = capture_instrument_magnetic_field(state)
    point = np.array([[1e-5, -2e-5, .418], [1e-5, 2e-5, .449]])
    sentinel = object()
    def traced(count=None, **kwargs):
        assert count == 9
        field = gun_transport_magnetic_field(gun)
        np.testing.assert_array_equal(field.field_at_global_positions_t(point),
                                      reference.field_at_global_positions_t(point))
        assert np.any(field.field_at_global_positions_t(point)[:, 2] != 0.)
        return sentinel
    monkeypatch.setattr(gun, "trace_to_exit", traced)
    assert trace_source_to_exit(state, 9) is sentinel
    assert not hasattr(gun, "_instrument_magnetic_field")
    np.testing.assert_array_equal(gun_transport_magnetic_field(gun).field_at_global_positions_t(point),
                                  gun.magnetic_field.field_at_global_positions_t(point))


def test_context_cleans_on_trace_failure_and_setup_failure(monkeypatch):
    from temsim.physics import gun_field_environment as environment
    state = default_state()
    gun = state.electron_gun
    with pytest.raises(RuntimeError, match="trace failure"):
        with instrument_gun_field_context(state):
            raise RuntimeError("trace failure")
    assert not any(name.startswith("_instrument_") for name in vars(gun))
    def fail(*args):
        raise ValueError("invalid field request")
    monkeypatch.setattr(environment, "_gun_magnetic_dependency_identity", fail)
    with pytest.raises(ValueError, match="invalid field request"):
        with instrument_gun_field_context(state):
            pass
    assert not any(name.startswith("_instrument_") for name in vars(gun))


def test_unknown_overlapping_provider_cannot_reuse_a_gun_trace(monkeypatch):
    from temsim.optics.electron_gun import field_emission
    from temsim.optics.electron_gun.source import trace_source_to_exit
    state = default_state()
    field = capture_instrument_magnetic_field(state)
    sources = list(field._sources)
    index = next(i for i, source in enumerate(sources) if source.key == CONDENSER_LENS_1)
    source = sources[index]
    sources[index] = replace(source, identity=replace(source.identity, numerical_identity=None))
    unknown = replace(field, _sources=tuple(sources), numerical_identity=None)
    monkeypatch.setattr("temsim.physics.instrument_magnetic.capture_instrument_magnetic_field", lambda state: unknown)
    executions = []
    def trace(gun, count):
        assert gun_transport_magnetic_field(gun) is unknown
        executions.append(object())
        return executions[-1]
    monkeypatch.setattr(field_emission, "trace_feg_to_exit", trace)
    first = trace_source_to_exit(state, 9)
    second = trace_source_to_exit(state, 9)
    assert first is not second and len(executions) == 2
    assert state.electron_gun._trace_cache is None


def test_instrument_binding_is_not_a_serialized_hardware_input():
    from temsim.instrument_snapshot import capture_instrument_snapshot
    state = default_state()
    before = capture_instrument_snapshot(state).digest
    with instrument_gun_field_context(state):
        assert capture_instrument_snapshot(state).digest == before


def test_packed_full_sum_matches_native_without_display_radius_clipping():
    state = default_state()
    state.electron_gun.deflector.upper_field_y_mt = .37
    state.electron_gun.stigmator.gradient_t_per_m = 7.
    field = capture_instrument_magnetic_field(state)
    packed = prepare_compiled_magnetic_sources(field._sources)
    assert packed is not None
    points = []
    for source in field._sources:
        if source.known_zero:
            continue
        for z in np.linspace(source.bounds_m[0, 2], source.bounds_m[1, 2], 5):
            points.extend(([1e-5, -2e-5, z], [1.1*source.radius_m, 0., z]))
    points = np.asarray(points)
    expected = field.field_at_global_positions_t(points)
    actual = compiled_magnetic_batch(points, *packed)
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=1e-12)


def test_closed_compiled_step_keeps_full_instrument_magnetic_force():
    from temsim.physics.closed_gun_field import ClosedGunField, closed_field_request
    from temsim.physics.relativistic_lorentz import RelativisticPhaseSpace, momentum_from_kinetic_energy_ev
    from temsim.physics.static_energy_lorentz import static_energy_step
    from temsim.physics.grounded_particle_step import try_step
    state = default_state()
    state.electron_gun.deflector.upper_field_y_mt = .3
    magnetic = capture_instrument_magnetic_field(state)
    request = closed_field_request(state.electron_gun)
    r = np.linspace(0., request["domain"]["outer_radius_m"], 3)
    z = np.linspace(0., request["domain"]["exit_m"], 5)
    voltage = np.broadcast_to(z*3e5/z[-1], (3, 5)).copy()
    electric = ClosedGunField(request, r, z, voltage, {})
    points = np.array([[1e-5, -2e-5, .401], [-2e-5, 1e-5, .418], [1e-5, 1e-5, .449]])
    energy = .3+electric.potential_v_at_global_positions(points)
    phase = RelativisticPhaseSpace(points, momentum_from_kinetic_energy_ev(energy,
                                   np.tile([.001, -.002, 1.], (len(points), 1))))
    dt = 1e-15
    actual = try_step(phase, dt, magnetic, electric, 1e-11, 64)
    assert actual is not None
    electric.compiled_particle_steps = False
    reference = static_energy_step(phase, dt, magnetic, electric)
    np.testing.assert_allclose(actual.position_m, reference.position_m, rtol=1e-13, atol=1e-22)
    np.testing.assert_allclose(actual.momentum_kg_m_per_s, reference.momentum_kg_m_per_s, rtol=1e-11, atol=1e-35)


def test_narrow_instrument_support_limits_spatial_steps():
    state = default_state()
    gun = state.electron_gun
    gun._instrument_magnetic_supports_mm = ((410., 410.01),)
    assert gun.integration_step_mm_at([409.999]) == pytest.approx(.001)
    assert gun.integration_step_mm_at([410.005]) == pytest.approx(.01/16.)


def test_diagnostic_kernel_consumes_the_shared_magnetic_laws(monkeypatch):
    from test_diagnostic_scene_identity import tiny_field, scene_from
    from temsim.magnetic_field_scene import prepare_magnetic_scene
    from temsim.physics.closed_gun_field import closed_field_request
    from temsim.test_electron_compiled import prepare_compiled_fields, compiled_fields
    state = default_state()
    state.electron_gun.stigmator.gradient_t_per_m = 7.
    electric = tiny_field(closed_field_request(state.electron_gun))
    magnetic = prepare_magnetic_scene(state)
    scene = scene_from(monkeypatch, electric, magnetic=magnetic, stop_mm=500.)
    packed = prepare_compiled_fields(scene)
    assert packed is not None
    for z in (.402, .418, .449, .5):
        point = np.array([1e-5, -2e-5, z])
        reference = scene.diagnostic_fields_at_global_position(point)
        valid, b, e, potential = compiled_fields(point, packed)
        assert valid and reference is not None
        np.testing.assert_allclose(b, reference[0], rtol=2e-7, atol=1e-12)
        np.testing.assert_allclose(e, reference[1], rtol=2e-13, atol=1e-9)
        assert potential == pytest.approx(reference[2], rel=2e-15, abs=1e-10)


# Reuse only a mathematical analytic-E fixture, never claim a production E solve.
from test_analytic_particle_step import gun, _phase, _reference


def test_analytic_compiled_boris_retains_axial_and_transverse_instrument_fields(gun):
    from temsim.physics import analytic_particle_step as stepping
    state = default_state()
    state.electron_gun.deflector.upper_field_y_mt = .3
    magnetic = capture_instrument_magnetic_field(state)
    electric = gun.electric_field
    phase, invariant = _phase(gun, [402., 418., 449.])
    active = np.ones(len(invariant), bool)
    reference, reference_dt = _reference(gun, phase, 1e-14, active, magnetic, electric, invariant)
    actual = stepping.try_analytic_step(gun, phase, 1e-14, active, magnetic, electric, invariant)
    assert actual is not None
    result, dt = actual
    assert dt == reference_dt
    np.testing.assert_allclose(result.position_m, reference.position_m, rtol=2e-12, atol=1e-20)
    np.testing.assert_allclose(result.momentum_kg_m_per_s, reference.momentum_kg_m_per_s, rtol=2e-10, atol=1e-34)
    execution = stepping.prepare_analytic_execution(gun, magnetic, electric, len(active))
    assert execution is not None and execution.begin_step(phase, active)
    assert execution.time_step(1e-14, .025) > 0.


@pytest.mark.parametrize('z_mm', [409.999, 410.005])
def test_compiled_batch_step_bound_resolves_instrument_support_like_outer_loop(gun, z_mm):
    from temsim.physics.analytic_particle_batch import _spatial_dt
    from temsim.physics.relativistic_lorentz import velocity_from_momentum_m_per_s
    gun._instrument_magnetic_supports_mm = ((410., 410.01),)
    phase, _ = _phase(gun, [z_mm])
    expected = gun.integration_step_mm_at([z_mm])*1e-3 / float(
        velocity_from_momentum_m_per_s(phase.momentum_kg_m_per_s)[0, 2])
    actual = _spatial_dt(phase.position_m, phase.momentum_kg_m_per_s, np.array([0]),
        np.asarray(gun.field_supports_mm), gun.trace_step_mm, gun.drift_step_mm,
        np.asarray(gun._instrument_magnetic_supports_mm))
    assert actual == pytest.approx(expected, rel=1e-14)


def test_query_guard_rejects_invalid_start_and_leaves_standalone_schedule():
    from temsim.physics.gun_transport_domain import bounded_gun_time_step
    from temsim.physics.relativistic_lorentz import SPEED_OF_LIGHT_M_PER_S as c
    dt = bounded_gun_time_step(.55, .449, 1e-3)
    assert 0. < dt < (.55-.449)/c
    assert .449+c*dt < .55
    assert bounded_gun_time_step(np.inf, .449, 1e-3) == 1e-3
    with pytest.raises(ValueError, match='outside.*query envelope'):
        bounded_gun_time_step(.55, .55, 1e-15)


@pytest.mark.parametrize('direction', [[.01, 0., 1.], [.2, .1, -1.]])
def test_native_trials_and_plane_refinement_only_query_inside_envelope(direction):
    """Instrumented mathematical fields inspect every DG trial B query."""
    from temsim.physics.gun_transport_domain import bounded_gun_time_step
    from temsim.physics.relativistic_lorentz import RelativisticPhaseSpace, momentum_from_kinetic_energy_ev
    from temsim.optics.electron_gun.tracing import _surface_step, _surface_plane_crossing
    upper = .449001
    queries = []
    class Electric:
        def field_at_global_positions_v_per_m(self, positions):
            return np.zeros_like(positions)
        def potential_v_at_global_positions(self, positions):
            return np.zeros(np.asarray(positions).shape[:-1])
    class Magnetic:
        def field_at_global_positions_t(self, positions):
            queries.extend(np.asarray(positions)[:, 2])
            assert np.max(np.asarray(positions)[:, 2]) < upper
            return np.tile([0., 0., .01], (len(positions), 1))
    electric, magnetic = Electric(), Magnetic()
    initial = RelativisticPhaseSpace(np.array([[1e-8, 0., .449]]),
        momentum_from_kinetic_energy_ev([300000.], [direction]))
    dt = bounded_gun_time_step(upper, initial.position_m[0, 2], 1e-3)
    result, accepted = _surface_step(initial, dt, np.array([True]), magnetic, electric)
    assert accepted <= dt and result.position_m[0, 2] < upper
    plane = .5*(initial.position_m[0, 2]+result.position_m[0, 2])
    crossing = _surface_plane_crossing(plane, initial.position_m[0],
        initial.momentum_kg_m_per_s[0], 0., accepted, electric, magnetic)
    assert crossing.position_m[0, 2] == pytest.approx(plane, abs=1e-14)
    assert len(queries) >= 6 and max(queries) < upper
