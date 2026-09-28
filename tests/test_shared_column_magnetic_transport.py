"""Finite magnetic-force fixtures; no microscope-calibration claim."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.model import DeflectorPair
from temsim.physics import core
from temsim.physics.acceleration import momentum_profile
from temsim.physics.ray_device_cache import plan_identity
from temsim.physics.ray_integrator import NUMBA_AVAILABLE


@pytest.fixture(autouse=True)
def isolated_magnetic_fixture_has_explicit_constant_electric_field(monkeypatch):
    """These mathematical gun-magnet fixtures contain no source/electrodes."""
    from temsim.physics import instrument_electric
    original = instrument_electric.capture_instrument_electric_field
    def capture(state):
        if type(getattr(state, 'electron_gun', None)) is not SimpleNamespace:
            return original(state)
        return SimpleNamespace(
            gun_snapshot=SimpleNamespace(exit_plane_z_mm=0.),
            base_field=SimpleNamespace(z=()), numerical_identity='constant-E magnetic fixture',
            potential_rise_v_at_global_positions=lambda points: np.zeros(len(points)),
            is_constant_on_interval=lambda *_args: True)
    monkeypatch.setattr(instrument_electric, 'capture_instrument_electric_field', capture)


def state_with_coil(*, step=.7, mode="analytical"):
    component = DeflectorPair("Fixture", "finite_fixture", 5., 15.,
                             .3, -.2, 0., 0., thickness_mm=2.)
    state = SimpleNamespace(
        lenses=[], stigmators=[], corrector_elements=[], deflectors=[component],
        beam_voltage_kv=300., step_mm=step, history_step_mm=1.,
        acceleration_enabled=False, acceleration_backend="CPU", simulation_mode=mode,
        projector_mode="diffraction", equivalent_image_lenses_enabled=False,
        sample=SimpleNamespace(z_mm=20.), simulation_time_s=0.,
    )
    return state, (5., .0003, -.0002)


def plan_for(state, event, *, checkpoints=(4., 5., 6., 10.)):
    return core.build_propagation_plan(state, 0., 10., (event,),
        include_spherical_aberration=False, checkpoint_z_mm=checkpoints)


@pytest.mark.parametrize("step", [.7, 5.])
def test_finite_coil_has_no_field_leak_or_remaining_thin_kick(step):
    state, event = state_with_coil(step=step)
    plan = plan_for(state, event)
    assert not np.any(plan.kick_x_rad) and not np.any(plan.kick_y_rad)
    assert 4. in plan.z_mm and 6. in plan.z_mm
    intervals = plan.dipole_by_t.reshape(-1, 3)
    middle = .5*(plan.z_mm[:-1]+plan.z_mm[1:])
    np.testing.assert_array_equal(intervals[(middle < 4.) | (middle > 6.)], 0.)
    np.testing.assert_array_equal(intervals[:, 0], intervals[:, 1])
    np.testing.assert_array_equal(intervals[:, 1], intervals[:, 2])
    result = core.execute_propagation_plan(state, plan, *(np.zeros(1),)*4)[-1]
    np.testing.assert_array_equal(result.x_m[0], 0.)
    np.testing.assert_array_equal(result.tx_rad[0], 0.)
    # Independent integration of a uniform transverse force, including its
    # interior displacement. A thin kick at the centre cannot satisfy this.
    np.testing.assert_allclose(result.tx_rad[:, 0], [0., .00015, .0003, .0003], atol=1e-18)
    np.testing.assert_allclose(result.x_m[:, 0], [0., .0003*.00025, .0003*.001, .0003*.005], atol=1e-19)
    np.testing.assert_allclose(result.y_m[:, 0], result.x_m[:, 0]*(-2./3.), atol=1e-19)


def test_empty_event_plan_preserves_homogeneous_transfer_query_and_unmatched_action():
    state, _ = state_with_coil()
    plan = core.build_propagation_plan(state, 0., 10., (), include_spherical_aberration=False)
    assert not np.any(plan.dipole_bx_t) and not np.any(plan.dipole_by_t)
    event = (5.123, .001, 0.)
    plan = plan_for(state, event, checkpoints=(10.,))
    assert not np.any(plan.dipole_bx_t) and not np.any(plan.dipole_by_t)
    result = core.execute_propagation_plan(state, plan, *(np.zeros(1),)*4)[-1]
    assert result.tx_rad[-1, 0] == pytest.approx(.001)
    assert result.x_m[-1, 0] == pytest.approx((10.-event[0])*1e-6, abs=1e-19)


@pytest.mark.parametrize("mode", ["analytical", "ideal"])
def test_fixed_magnetic_field_uses_particle_momentum_and_keeps_ideal_reference(mode):
    state, event = state_with_coil(mode=mode)
    plan = plan_for(state, event, checkpoints=(10.,))
    energy = np.array([-100000., 0., 100000.])
    result = core.execute_propagation_plan(state, plan, *(np.zeros(3),)*4,
        energy_offset_ev=energy)[-1]
    p = momentum_profile(state, np.array([0.]), None if mode == "ideal" else energy)[0]
    expected = event[1]*core.electron(state)[1]/p
    np.testing.assert_allclose(result.tx_rad[-1], np.broadcast_to(expected, (3,)), rtol=2e-15, atol=1e-18)


def test_finite_coil_resume_midfield_preserves_exact_state_and_clock():
    state, event = state_with_coil(step=.2)
    plan = plan_for(state, event)
    source = (np.zeros(2), np.array([1e-4, -2e-4]), np.zeros(2), np.array([2e-4, 1e-4]))
    full = core.execute_propagation_plan(state, plan, *source,
        initial_time_s=np.array([0., 1e-9]), return_flight_times=True)[-1]
    mid = int(np.flatnonzero(full.z_mm == 5.)[0])
    resumed = core.execute_propagation_plan(state, plan,
        full.x_m[mid], full.tx_rad[mid], full.y_m[mid], full.ty_rad[mid],
        start_index=int(plan.checkpoint_index[mid]), include_initial_plane_kicks=False,
        initial_time_s=full.flight_time_s[mid], return_flight_times=True)[-1]
    for name in ("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s"):
        np.testing.assert_array_equal(getattr(resumed, name)[-1], getattr(full, name)[-1])


def test_interval_forcing_change_invalidates_common_prefix():
    state, event = state_with_coil()
    plan = plan_for(state, event)
    changed_field = plan.dipole_by_t.copy()
    interval = int(np.flatnonzero(changed_field.reshape(-1, 3)[:, 1])[0])
    changed_field[3*interval+1] *= 1.01
    changed = replace(plan, dipole_by_t=changed_field)
    assert core.propagation_plan_common_prefix_nodes(plan, changed) <= interval


@pytest.mark.parametrize("engine", ["numba", "serial", "cuda", "mapped"])
@pytest.mark.parametrize("timed", [False, True])
def test_finite_forcing_and_clock_backend_parity(monkeypatch, engine, timed):
    if engine in ("numba", "serial") and not NUMBA_AVAILABLE:
        pytest.skip("Numba unavailable")
    if engine == "cuda":
        from temsim.physics.compute_backend import cuda_capability
        if not cuda_capability().available:
            pytest.skip("CUDA hardware unavailable")
    state, event = state_with_coil(step=.2)
    plan = plan_for(state, event)
    values = []
    for backend in (core.BACKEND_CPU, core.BACKEND_CUDA if engine == "cuda" else core.BACKEND_NUMBA):
        monkeypatch.setattr(core, "choose_ray_backend", lambda *_a, **_k: (backend, None))
        if len(values) and engine == "serial":
            state._optical_tuning = True
            state.acceleration_enabled = True
            state.acceleration_backend = "Auto"
        if len(values) and engine == "mapped":
            # An overlapping zero imported map must leave the driven orbit
            # unchanged while exercising the mapped interval's full derivative.
            zero_map = SimpleNamespace(scale=1.,
                field_map=SimpleNamespace(field_support_mm=(0., 10.)),
                field_at_global_positions_t=lambda points: np.zeros_like(points))
            plan = replace(plan, mapped_fields=(zero_map,))
        values.append(core.execute_propagation_plan(state, plan, *(np.zeros(3),)*4,
            energy_offset_ev=np.array([-100000., 0., 100000.]),
            initial_time_s=np.array([0., np.nan, 1e-9]) if timed else None,
            return_flight_times=timed)[-1])
    if engine == "cuda":
        assert "CUDA GPU" in state.active_backend
    if engine == "serial":
        assert state._tuning_kernel == "serial_numba"
    for name in ("x_m", "tx_rad", "y_m", "ty_rad") + (("flight_time_s",) if timed else ()):
        np.testing.assert_allclose(getattr(values[0], name), getattr(values[1], name),
                                   rtol=3e-14, atol=1e-23, equal_nan=True)


@pytest.mark.parametrize("index", [19, 20, 21])
def test_device_identity_binds_force_and_reference_momentum(index):
    from test_ray_device_residency import inputs
    before = inputs()
    after = list(before)
    after[index] = after[index].copy()
    after[index][0] += .25
    assert plan_identity(before) != plan_identity(after)


def test_tensor_force_scales_with_momentum_against_independent_matrix_exponential():
    from scipy.linalg import expm
    from test_stigmator_tensor import tensor_inputs
    from temsim.physics.ray_integrator import vectorised_rk4
    inputs = list(tensor_inputs())
    inputs[5] = np.array([.5, 1., 2.])
    inputs[21] = np.array([1.])
    actual = vectorised_rk4(*inputs)
    for ray, ratio in enumerate(inputs[5]):
        generator = np.array([[0., 1., 0., 0.], [-200.*ratio, 0., -80.*ratio, 0.],
                              [0., 0., 0., 1.], [-80.*ratio, 0., 200.*ratio, 0.]])
        expected = expm(generator*.01) @ np.array(inputs[10:14])[:, ray]
        np.testing.assert_allclose(np.array(actual[4:])[:, 0, ray], expected, rtol=8e-10, atol=1e-15)


def test_extended_gun_dipole_changes_orbit_but_not_linear_transfer():
    from temsim.optics.electron_gun.alignment import GunDeflector
    state, _ = state_with_coil(step=.1)
    state.deflectors = []
    coil = GunDeflector(5., 8., 20., 10., 5., 8., 2.,
                        soft_edge_mm=.5, upper_field_y_mt=.01)
    state.electron_gun = SimpleNamespace(deflector=coil)
    driven = core.propagate(state, 0., 10., *(np.zeros(1),)*4,
        checkpoint_z_mm=(10.,), return_checkpoints=True)[-1]
    matrix = core.transfer(state, 0., 10.)
    complex_matrix = core.complex_transfer(state, 0., 10.)
    assert abs(driven.tx_rad[-1, 0]) > 1e-6
    coil.upper_field_y_mt = 0.
    undriven = core.propagate(state, 0., 10., *(np.zeros(1),)*4,
        checkpoint_z_mm=(10.,), return_checkpoints=True)[-1]
    assert undriven.tx_rad[-1, 0] == 0.
    np.testing.assert_allclose(matrix, core.transfer(state, 0., 10.), atol=2e-15)
    np.testing.assert_allclose(complex_matrix, core.complex_transfer(state, 0., 10.), atol=2e-15)
    np.testing.assert_allclose(matrix, [[1., .01], [0., 1.]], atol=2e-15)


@pytest.mark.parametrize("rotation", [0., 22.5, 45.])
def test_gun_quadrupole_uses_one_shared_coefficient_in_field_queries_and_plans(rotation):
    from temsim.optics.electron_gun.alignment import GunStigmator
    state, _ = state_with_coil(step=.2)
    state.deflectors = []
    state.electron_gun = SimpleNamespace(stigmator=GunStigmator(
        5., 4., 20., 10., 2., soft_edge_mm=.5,
        gradient_t_per_m=.02, rotation_deg=rotation))
    angle = np.deg2rad(2.*rotation)
    q, p, _ = core.electron(state)
    expected = (q/p*.02*np.cos(angle), -q/p*.02*np.cos(angle),
                q/p*.02*np.sin(angle))
    _, sx, sy = core.fields(np.array([5.]), state)
    skew = core.skew_quadrupole_field(np.array([5.]), state)
    np.testing.assert_allclose([sx[0], sy[0], skew[0]], expected, atol=1e-12)
    plan = core.build_propagation_plan(state, 0., 10., checkpoint_z_mm=(5.,))
    index = int(np.flatnonzero(plan.z_mm == 5.)[0])
    np.testing.assert_allclose([plan.sx_m2[index], plan.sy_m2[index], plan.sxy_m2[index]],
                               expected, atol=1e-12)


def test_paused_wave_preparation_rejects_finite_gun_dipole_before_propagation(monkeypatch):
    from temsim.optics.electron_gun.alignment import GunDeflector
    from temsim.physics import column_wave
    state, _ = state_with_coil(step=.5)
    state.apertures = []
    state.deflectors = []
    state.electron_gun = SimpleNamespace(deflector=GunDeflector(
        5., 8., 20., 10., 5., 8., 2., soft_edge_mm=.5, upper_field_y_mt=.01))
    monkeypatch.setattr(column_wave, "_vacuum_segments", lambda *_args: ())
    with pytest.raises(ValueError, match="coherent development is paused"):
        column_wave._prepare_column(state, 0., 10., .5)


def test_paused_wave_preparation_rejects_overlap_even_when_coil_center_is_outside(monkeypatch):
    from temsim.physics import column_wave
    state, _ = state_with_coil(step=.2)
    state.apertures = []
    monkeypatch.setattr(column_wave, "_vacuum_segments", lambda *_args: ())
    # The powered coil spans 4--6 mm while its centre lies beyond this segment.
    # Its original wave event builder produces no affine event here.
    with pytest.raises(ValueError, match="coherent development is paused"):
        column_wave._prepare_column(state, 0., 4.5, .2)


def test_alignment_uses_complete_xy_map_for_shared_gun_skew():
    from temsim.optics.direct_alignment import _LiveFirstOrderModel
    from temsim.optics.electron_gun.alignment import GunStigmator
    from temsim.physics.first_order import trace_transverse_transfer

    state, _ = state_with_coil(step=.2)
    state.deflectors = []
    state.electron_gun = SimpleNamespace(stigmator=GunStigmator(
        5., 4., 20., 10., 2., soft_edge_mm=.5,
        gradient_t_per_m=.02, rotation_deg=22.5))
    model = _LiveFirstOrderModel(state, 0., 10., (), step_mm=.2)
    assert model.full_field_transfer
    expected = trace_transverse_transfer(state, 0., 10., maximum_step_mm=.2)
    np.testing.assert_allclose(model.matrix([]), expected.matrix, atol=1e-14)
    assert abs(expected.matrix[0, 1]) > 1e-6


def test_diffraction_observer_keeps_shared_gun_dipole_reference_offset():
    from temsim.optics.direct_alignment import diffraction_transfer
    from temsim.optics.electron_gun.alignment import GunDeflector
    from temsim.physics.first_order import trace_transverse_transfer

    state, _ = state_with_coil(step=.1)
    state.sample = SimpleNamespace(z_mm=0.)
    state.deflectors = []
    state.electron_gun = SimpleNamespace(deflector=GunDeflector(
        5., 8., 20., 10., 5., 8., 2., soft_edge_mm=.5, upper_field_y_mt=.01))
    expected = trace_transverse_transfer(state, 0., 10., maximum_step_mm=.025)
    actual = diffraction_transfer(state, 10.)
    np.testing.assert_allclose(actual.matrix, expected.matrix, atol=1e-14)
    np.testing.assert_array_equal(actual.position_offset_m, expected.position_offset_m)
    np.testing.assert_array_equal(actual.angle_offset_rad, expected.angle_offset_rad)
    assert abs(actual.angle_offset_rad[0]) > 1e-6
