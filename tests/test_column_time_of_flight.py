"""Time-of-flight fixtures use explicit test origins, never downstream sources."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from temsim.optics.column import default_state
from temsim.physics import core
from temsim.physics.ray_integrator import NUMBA_AVAILABLE


def _state(monkeypatch):
    state = default_state()
    for component in (*state.lenses, *state.stigmators, *state.corrector_elements):
        component.enabled = False
    state.acceleration_backend = "CPU"
    state.acceleration_enabled = False
    state.step_mm = .25
    state.history_step_mm = 1.
    monkeypatch.setattr(core, "fields", lambda z, _s: (np.zeros_like(z),)*3)
    return state


def _speed(energy_ev):
    energy = np.asarray(energy_ev)*core.E
    rest = core.M*core.C**2
    return core.C*np.sqrt(energy*(energy+2.*rest))/(energy+rest)


def _propagate(state, source, **kwargs):
    return core.propagate(state, 1500., 1505., *source,
                          include_spherical_aberration=False, include_hexapole=False,
                          checkpoint_z_mm=(1502., 1505.), return_checkpoints=True, **kwargs)


@pytest.mark.parametrize("mode", ["custom", "ideal"])
def test_straight_tilted_and_energy_offset_time_is_analytic_and_float64(monkeypatch, mode):
    state = _state(monkeypatch)
    state.simulation_mode = mode
    tx, ty = np.array([0., .03, -.2]), np.array([0., -.04, .1])
    source = np.zeros(3), tx, np.zeros(3), ty
    initial = np.array([0., 2e-9, 4e-9])
    offsets = np.array([0., -100000., 50000.])
    result = _propagate(state, source, initial_time_s=initial,
                        energy_offset_ev=offsets, return_flight_times=True)
    expected = initial+5e-3*np.sqrt(1.+tx*tx+ty*ty)/_speed(state.beam_voltage_kv*1000.+offsets)
    np.testing.assert_allclose(result[5][-1], expected, rtol=3e-15, atol=1e-23)
    np.testing.assert_allclose(result[-1].flight_time_s[-1], expected, rtol=3e-15, atol=1e-23)
    assert result[5].dtype == result[-1].flight_time_s.dtype == np.float64
    assert not result[-1].flight_time_s.flags.writeable
    without = _propagate(state, source, energy_offset_ev=offsets)
    assert len(without) == 6 and without[-1].flight_time_s is None
    for old, timed in zip(without[:5], result[:5]):
        np.testing.assert_array_equal(old, timed)


def test_unknown_origins_remain_unknown_independently_per_ray(monkeypatch):
    state = _state(monkeypatch)
    source = (np.zeros(2),)*4
    unknown = _propagate(state, source, return_flight_times=True)
    assert np.isnan(unknown[5]).all() and np.isnan(unknown[-1].flight_time_s).all()
    mixed = _propagate(state, source, initial_time_s=np.array([np.nan, 0.]), return_flight_times=True)
    assert np.isnan(mixed[5][:, 0]).all()
    assert np.isfinite(mixed[5][:, 1]).all()
    plain = core.propagate(state, 1500., 1501., *source)
    timed = core.propagate(state, 1500., 1501., *source, return_flight_times=True)
    assert len(plain) == 5 and len(timed) == 6


@pytest.mark.parametrize("initial", [np.array([-1.]), np.array([np.inf]), np.array([-np.inf]), np.zeros(2), 0.])
def test_invalid_initial_times_are_not_silently_reinterpreted(monkeypatch, initial):
    state = _state(monkeypatch)
    with pytest.raises(ValueError, match="Initial flight times"):
        _propagate(state, (np.zeros(1),)*4, initial_time_s=initial, return_flight_times=True)


def _gaussian(monkeypatch, state):
    momentum = core.electron(state)[1]
    def rates(z):
        offset = (np.asarray(z)-1.5025)/.0011
        g = 220.*np.exp(-.5*offset**2)
        return g, -offset*g/.0011
    def fields(z, _state):
        g, _ = rates(np.asarray(z)*1e-3)
        return -2.*momentum*g/core.E, np.zeros_like(g), np.zeros_like(g)
    monkeypatch.setattr(core, "fields", fields)
    return rates


def test_time_uses_rk_stage_path_and_converges_under_step_refinement(monkeypatch):
    state = _state(monkeypatch)
    rates = _gaussian(monkeypatch, state)
    speed = _speed(state.beam_voltage_kv*1000.)
    initial = np.array([2e-6, .03, -1e-6, -.02, 0.])
    def rhs(z, values):
        x, tx, y, ty, _ = values
        g, dg = rates(z)
        return tx, 2.*g*ty+dg*y, ty, -2.*g*tx-dg*x, np.sqrt(1.+tx*tx+ty*ty)/speed
    reference = solve_ivp(rhs, (1.5, 1.505), initial, method="DOP853", rtol=2e-12,
                          atol=np.array([1e-16]*4+[1e-25])).y[-1, -1]
    errors = []
    for step in (.4, .2, .1):
        state.step_mm = step
        result = _propagate(state, tuple(np.array([v]) for v in initial[:4]),
                            initial_time_s=np.zeros(1), return_flight_times=True)
        errors.append(abs(result[5][-1, 0]-reference))
    assert errors[0]/errors[1] > 8.
    assert errors[1]/errors[2] > 8.
    assert errors[-1] < 1e-20


def test_checkpoint_resume_preserves_time_and_does_not_repeat_plane_kicks(monkeypatch):
    state = _state(monkeypatch)
    _gaussian(monkeypatch, state)
    plan = core.build_propagation_plan(state, 1500., 1505., events=((1502., .03, -.02),),
        checkpoint_z_mm=(1502., 1505.), include_spherical_aberration=False, include_hexapole=False)
    initial = tuple(np.array([v]) for v in (2e-6, .01, -1e-6, -.02))
    full = core.execute_propagation_plan(state, plan, *initial,
        initial_time_s=np.array([3e-9]), return_flight_times=True)[-1]
    resumed = core.execute_propagation_plan(state, plan, full.x_m[0], full.tx_rad[0],
        full.y_m[0], full.ty_rad[0], start_index=int(plan.checkpoint_index[0]),
        include_initial_plane_kicks=False, initial_time_s=full.flight_time_s[0],
        return_flight_times=True)[-1]
    for name in ("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s"):
        np.testing.assert_array_equal(getattr(full, name)[-1], getattr(resumed, name)[-1])


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba optional")
def test_numba_and_numpy_time_agree_without_changing_ray_dynamics(monkeypatch):
    state = _state(monkeypatch)
    _gaussian(monkeypatch, state)
    source = tuple(np.array([v, -v]) for v in (2e-6, .01, -1e-6, -.02))
    results = []
    for backend in (core.BACKEND_CPU, core.BACKEND_NUMBA):
        monkeypatch.setattr(core, "choose_ray_backend", lambda *_a, **_k: (backend, None))
        results.append(_propagate(state, source, initial_time_s=np.array([0., np.nan]),
            energy_offset_ev=np.array([3000., -1000.]), return_flight_times=True)[-1])
    for name in ("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s"):
        np.testing.assert_allclose(getattr(results[0], name), getattr(results[1], name),
                                   rtol=5e-12, atol=1e-24, equal_nan=True)


def test_mapped_uniform_magnetic_field_time_follows_circular_arc(monkeypatch):
    # Reuse the independent, explicitly synthetic field-map fixture.
    from test_vector_field_transport import _state as mapped_state, _map
    state = mapped_state()
    _map(state, (.03, 0., 0.))
    result = core.propagate(state, 0., 1., *(np.zeros(1),)*4,
        initial_time_s=np.zeros(1), return_flight_times=True,
        include_spherical_aberration=False, include_hexapole=False)
    curvature = core.E*.03/core.electron(state)[1]
    expected = np.arcsin(curvature*1e-3)/curvature/_speed(state.beam_voltage_kv*1000.)
    assert result[5][-1, 0] == pytest.approx(expected, rel=2e-9)
    plan = core.build_propagation_plan(state, 0., 1., checkpoint_z_mm=(.5, 1.),
        include_spherical_aberration=False, include_hexapole=False)
    full = core.execute_propagation_plan(state, plan, *(np.zeros(1),)*4,
        initial_time_s=np.zeros(1), return_flight_times=True)[-1]
    resumed = core.execute_propagation_plan(state, plan, full.x_m[0], full.tx_rad[0],
        full.y_m[0], full.ty_rad[0], initial_time_s=full.flight_time_s[0],
        start_index=int(plan.checkpoint_index[0]), include_initial_plane_kicks=False,
        return_flight_times=True)[-1]
    np.testing.assert_array_equal(full.flight_time_s[-1], resumed.flight_time_s[-1])


def test_deferred_out_of_domain_map_ray_does_not_keep_a_valid_time():
    from test_vector_field_transport import _state as mapped_state, _map
    state = mapped_state()
    _map(state, (100., 0., 0.))
    with np.errstate(over="ignore", invalid="ignore"):
        result = core.propagate(state, 0., 1., *(np.zeros(1),)*4,
            initial_time_s=np.zeros(1), return_flight_times=True,
            defer_nonfinite_until_clipping=True,
            include_spherical_aberration=False, include_hexapole=False)
    assert np.isnan(result[1][-1, 0])
    assert np.isnan(result[5][-1, 0])


def test_time_device_buffers_reuse_optics_but_upload_updated_clocks():
    from test_ray_device_residency import FakeCUDA, inputs
    from temsim.physics.ray_device_cache import RayDeviceCache, last_device_receipt
    from temsim.physics.ray_integrator import vectorised_rk4
    class TimeKernel:
        def __getitem__(self, _launch):
            def run(*arrays):
                values = tuple(a.array for a in arrays)
                result = vectorised_rk4(*values[:19], initial_time_s=values[19], inverse_speed=values[20])
                for target, value in zip(arrays[21:], result, strict=True):
                    target.array[...] = value
            return run
    base = inputs()
    times = np.arange(7)*1e-9
    cuda, cache = FakeCUDA(), RayDeviceCache()
    first = cache.execute(cuda, TimeKernel(), (*base, times, np.full(7, 5e-9)))
    allocations = cuda.allocations
    second = cache.execute(cuda, TimeKernel(), (*base, times+1e-9, np.full(7, 5e-9)))
    assert cuda.allocations == allocations and last_device_receipt()["plan_reused"]
    np.testing.assert_allclose(second[8]-first[8], 1e-9, rtol=1e-15)
    assert first[8].dtype == first[9].dtype == np.float64


def test_small_cuda_time_and_legacy_kernel_match_cpu_on_available_hardware(monkeypatch):
    from temsim.physics.compute_backend import cuda_capability
    if not cuda_capability().available:
        pytest.skip("CUDA hardware unavailable")
    state = _state(monkeypatch)
    _gaussian(monkeypatch, state)
    source = tuple(np.linspace(v, -v, 7) for v in (2e-6, .01, -1e-6, -.02))
    initial = np.linspace(0., 3e-9, 7)
    initial[2] = np.nan
    cpu = _propagate(state, source, initial_time_s=initial, return_flight_times=True)
    monkeypatch.setattr(core, "choose_ray_backend", lambda *_a, **_k: (core.BACKEND_CUDA, None))
    gpu = _propagate(state, source, initial_time_s=initial, return_flight_times=True)
    assert "CUDA GPU" in state.active_backend
    for name in ("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s"):
        np.testing.assert_allclose(getattr(gpu[-1], name), getattr(cpu[-1], name),
                                   rtol=5e-12, atol=1e-24, equal_nan=True)
    legacy = _propagate(state, source)
    assert len(legacy) == 6 and legacy[-1].flight_time_s is None
    for name in ("x_m", "tx_rad", "y_m", "ty_rad"):
        np.testing.assert_allclose(getattr(legacy[-1], name), getattr(gpu[-1], name),
                                   rtol=5e-12, atol=1e-24)
