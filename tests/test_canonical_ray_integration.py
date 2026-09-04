"""Accuracy gates for the physical ray solver, independent of lens presets."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from temsim.optics.column import default_state
from temsim.physics import core
from temsim.physics.ray_integrator import NUMBA_AVAILABLE


def _isolated_state():
    state = default_state()
    for component in (*state.lenses, *state.stigmators, *state.corrector_elements):
        component.enabled = False
    state.acceleration_enabled = False
    state.acceleration_backend = "CPU"
    state.history_step_mm = 1.0
    return state


def _gaussian_field(monkeypatch, state):
    """A known smooth field with nonzero field at the starting plane."""
    momentum = core.electron(state)[1]
    centre_m, sigma_m, peak_g_m1 = 1.505, 0.0011, 220.0

    def rates(z_m):
        offset = (np.asarray(z_m) - centre_m) / sigma_m
        g = peak_g_m1 * np.exp(-0.5 * offset**2)
        return g, -offset * g / sigma_m

    def fields(z_mm, _state):
        g, _dg = rates(np.asarray(z_mm) * 1.0e-3)
        return -2.0 * momentum * g / core.E, np.zeros_like(g), np.zeros_like(g)

    monkeypatch.setattr(core, "fields", fields)
    return rates


def _last_checkpoint(state, initial, start, stop, **kwargs):
    result = core.propagate(
        state, start, stop,
        *(np.asarray((value,)) for value in initial),
        include_spherical_aberration=False,
        include_hexapole=False,
        checkpoint_z_mm=(stop,), return_checkpoints=True,
        **kwargs,
    )
    checkpoint = result[5]
    return np.asarray((
        checkpoint.x_m[-1, 0], checkpoint.tx_rad[-1, 0],
        checkpoint.y_m[-1, 0], checkpoint.ty_rad[-1, 0],
    ))


def test_gaussian_lens_converges_fourth_order_against_lab_frame_ode(monkeypatch):
    state = _isolated_state()
    rates = _gaussian_field(monkeypatch, state)
    initial = np.asarray((2.0e-6, 0.001, -1.0e-6, -0.0007))
    start, stop = 1503.8, 1508.3

    # Independent reference: physical slopes and the analytic field
    # derivative, rather than the production canonical transformation.
    def laboratory_rhs(z_m, values):
        x, tx, y, ty = values
        g, dg = rates(z_m)
        return tx, 2.0 * g * ty + dg * y, ty, -2.0 * g * tx - dg * x

    reference = solve_ivp(
        laboratory_rhs, (start * 1e-3, stop * 1e-3), initial,
        method="DOP853", rtol=2.0e-12, atol=1.0e-16,
    ).y[:, -1]
    errors = []
    for step_mm in (0.2, 0.1, 0.05):
        state.step_mm = step_mm
        measured = _last_checkpoint(state, initial, start, stop)
        errors.append(np.linalg.norm((measured - reference) * (1.0, 0.001, 1.0, 0.001)))
    assert errors[0] / errors[1] > 12.0
    assert errors[1] / errors[2] > 12.0
    assert errors[-1] < 1.0e-13


def test_thin_deflector_occurs_at_its_exact_physical_plane(monkeypatch):
    state = _isolated_state()
    monkeypatch.setattr(
        core, "fields", lambda z, _s: (np.zeros_like(z),) * 3
    )
    event_z, target_z = 1504.123456, 1508.3
    expected = (target_z - event_z) * 1.0e-3 * 0.001
    for step_mm in (0.3, 0.1):
        state.step_mm = step_mm
        actual = _last_checkpoint(
            state, (0.0, 0.0, 0.0, 0.0), 1503.8, target_z,
            events=((event_z, 0.001, 0.0),),
        )
        assert actual[0] == pytest.approx(expected, abs=1.0e-17)
        assert actual[1] == pytest.approx(0.001, abs=1.0e-17)


def test_midpoint_change_invalidates_the_cached_interval(monkeypatch):
    state = _isolated_state()
    state.step_mm = 1.0
    peak = [0.0]

    def fields(z_mm, _state):
        magnetic = np.zeros_like(z_mm)
        magnetic[np.isclose(z_mm, 1501.5, rtol=0, atol=1e-9)] = peak[0]
        return magnetic, np.zeros_like(z_mm), np.zeros_like(z_mm)

    monkeypatch.setattr(core, "fields", fields)
    first = core.build_propagation_plan(
        state, 1500.0, 1505.0,
        include_spherical_aberration=False, include_hexapole=False,
    )
    peak[0] = 0.1
    second = core.build_propagation_plan(
        state, 1500.0, 1505.0,
        include_spherical_aberration=False, include_hexapole=False,
    )
    np.testing.assert_array_equal(first.magnetic_t, second.magnetic_t)
    assert first.signature != second.signature
    assert core.propagation_plan_common_prefix_nodes(first, second) <= 1


def test_checkpoint_resume_inside_magnetic_field_preserves_physical_slopes(monkeypatch):
    state = _isolated_state()
    _gaussian_field(monkeypatch, state)
    state.step_mm = 0.1
    plan = core.build_propagation_plan(
        state, 1503.8, 1508.3,
        include_spherical_aberration=False, include_hexapole=False,
        checkpoint_z_mm=(1505.1, 1508.3),
    )
    source = tuple(np.asarray((value,)) for value in (2e-6, 0.001, -1e-6, -0.0007))
    full = core.execute_propagation_plan(state, plan, *source)[5]
    checkpoint_source = (
        full.x_m[0], full.tx_rad[0], full.y_m[0], full.ty_rad[0],
    )
    resumed = core.execute_propagation_plan(
        state, plan, *checkpoint_source,
        start_index=int(plan.checkpoint_index[0]),
        include_initial_plane_kicks=False,
    )[5]
    for name in ("x_m", "tx_rad", "y_m", "ty_rad"):
        np.testing.assert_array_equal(getattr(resumed, name)[-1], getattr(full, name)[-1])


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba unavailable")
def test_numba_and_numpy_agree_for_nonzero_source_field_and_energy_spread(monkeypatch):
    state = _isolated_state()
    _gaussian_field(monkeypatch, state)
    state.step_mm = 0.1
    plan = core.build_propagation_plan(
        state, 1503.8, 1508.3,
        include_spherical_aberration=False, include_hexapole=False,
        checkpoint_z_mm=(1508.3,),
    )
    results = []
    for backend in (core.BACKEND_CPU, core.BACKEND_NUMBA):
        monkeypatch.setattr(core, "choose_ray_backend", lambda *_a, **_k: (backend, None))
        results.append(core.execute_propagation_plan(
            state, plan,
            np.asarray((2e-6, -3e-6)), np.asarray((0.001, -0.0002)),
            np.asarray((-1e-6, 3e-6)), np.asarray((-0.0007, 0.0008)),
            energy_offset_ev=np.asarray((-0.3, 0.5)),
        )[5])
    for name in ("x_m", "tx_rad", "y_m", "ty_rad"):
        np.testing.assert_allclose(
            getattr(results[0], name), getattr(results[1], name),
            rtol=5e-12, atol=1e-17,
        )
