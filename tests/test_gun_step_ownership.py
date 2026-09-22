"""Small ownership checks; no complete gun trace or curved-tip mesh solve."""
from types import SimpleNamespace

import numpy as np
import pytest

from test_analytic_particle_step import gun, _phase
from temsim.optics.electron_gun import tracing
from temsim.physics import analytic_particle_step as compiled
from temsim.physics.relativistic_lorentz import (
    RelativisticPhaseSpace, momentum_from_kinetic_energy_ev,
)


def _protect(phase):
    originals = phase.position_m.copy(), phase.momentum_kg_m_per_s.copy()
    phase.position_m.setflags(write=False)
    phase.momentum_kg_m_per_s.setflags(write=False)
    return originals


def _assert_owned_advance(phase, originals, advanced, active):
    for old, expected, new in zip(
            (phase.position_m, phase.momentum_kg_m_per_s), originals,
            (advanced.position_m, advanced.momentum_kg_m_per_s), strict=True):
        np.testing.assert_array_equal(old, expected)
        assert not np.shares_memory(old, new)
        np.testing.assert_array_equal(new[~active], expected[~active])
    assert advanced.time_s > phase.time_s


@pytest.mark.parametrize("prepared", [False, True])
def test_analytic_step_keeps_readonly_input_and_prior_outputs_independent(gun, prepared):
    if prepared and compiled._compiled_step is None:
        pytest.skip("Numba optional")
    gun.compiled_particle_steps = prepared
    phase, invariant = _phase(gun, [13., 402., 418.])
    active = np.array([True, False, True])
    magnetic, electric = gun.magnetic_field, gun.electric_field
    execution = (compiled.prepare_analytic_execution(gun, magnetic, electric, len(active))
                 if prepared else None)
    if prepared:
        assert execution is not None
    originals = _protect(phase)
    if execution is not None:
        assert execution.begin_step(phase, active)
    first, _ = tracing._analytic_step(gun, phase, 1e-15, active, magnetic, electric,
                                      invariant[active], execution=execution)
    _assert_owned_advance(phase, originals, first, active)
    first_values = _protect(first)
    if execution is not None:
        assert execution.begin_step(first, active)
    second, _ = tracing._analytic_step(gun, first, 1e-15, active, magnetic, electric,
                                       invariant[active], execution=execution)
    _assert_owned_advance(first, first_values, second, active)
    _assert_owned_advance(phase, originals, second, active)


def test_surface_step_accepts_readonly_input_without_aliasing_it():
    # A uniform static field exercises the discrete-gradient step and event
    # ownership, without claiming to validate a solved curved-tip boundary.
    electric = SimpleNamespace(
        potential_v_at_global_positions=lambda xyz: -1e3 * xyz[..., 2],
        field_at_global_positions_v_per_m=lambda xyz:
            np.broadcast_to([0., 0., 1e3], np.shape(xyz)).copy())
    magnetic = SimpleNamespace(field_at_global_positions_t=lambda xyz: np.zeros_like(xyz))
    phase = RelativisticPhaseSpace(np.array([[0., 0., 0.], [1e-8, 0., 1e-7]]),
        momentum_from_kinetic_energy_ev([100., 100.], [[0., 0., 1.], [.01, 0., 1.]]))
    active = np.array([True, False])
    originals = _protect(phase)
    advanced, dt = tracing._surface_step(phase, 1e-15, active, magnetic, electric)
    assert dt == 1e-15
    _assert_owned_advance(phase, originals, advanced, active)


@pytest.mark.parametrize("with_medium", [False, True])
def test_five_step_cancellation_retains_prior_states_and_independent_history(gun, monkeypatch, with_medium):
    from temsim.physics import residual_medium
    from temsim.vacuum import Medium, ResolvedMedium

    gun.history_step_mm = gun.trace_step_mm
    gun._vacuum_regions = (() if not with_medium else (
        ResolvedMedium("ownership_fixture", "Ownership fixture", 0., gun.exit_plane_z_mm,
                       Medium(phase="vacuum")),))
    inputs, boundary_positions, medium_checks = [], [], []
    original_step = tracing._analytic_step
    original_clip = tracing._clip_body_bores
    original_medium = residual_medium.MediumTransport.advance

    def observe_step(gun, phase, dt, active, *args, **kwargs):
        inputs.append((phase, phase.position_m.copy(), phase.momentum_kg_m_per_s.copy(),
                       active, active.copy()))
        return original_step(gun, phase, dt, active, *args, **kwargs)

    def observe_bore(gun, old, new, alive, completed, blocked_z, blocked_key):
        original_clip(gun, old, new, alive, completed, blocked_z, blocked_key)
        boundary_positions.append(new)
        if with_medium and len(inputs) == 2:
            # Deterministic stop fixture, solely to exercise clipping of the
            # medium segment independently of the accepted solver endpoint.
            alive[1] = False
            blocked_z[1] = .5 * (old[1, 2] + new[1, 2]) * 1000.
            blocked_key[1] = "fixture_aperture"

    def observe_medium(self, start, end, direction, energy, **kwargs):
        active = inputs[-1][3]
        assert not np.shares_memory(self.alive, active)
        assert not np.shares_memory(end, boundary_positions[-1])
        assert not np.shares_memory(start, end)
        if len(inputs) == 2:
            assert start[1, 2] < end[1, 2] < boundary_positions[-1][1, 2]
        output = original_medium(self, start, end, direction, energy, **kwargs)
        if len(inputs) == 3:
            # Inject one owned-medium removal, rather than sampling a rare gas
            # event. Mutating medium.alive must leave this step's active mask.
            self.alive[0] = False
            self.blocked_z[0] = .5 * (start[0, 2] + end[0, 2]) * 1000.
            self.blocked_key[0] = "medium_removal:ownership_fixture"
            assert active[0]
        medium_checks.append(True)
        return output

    monkeypatch.setattr(tracing, "_analytic_step", observe_step)
    monkeypatch.setattr(tracing, "_clip_body_bores", observe_bore)
    if with_medium:
        monkeypatch.setattr(residual_medium.MediumTransport, "advance", observe_medium)
    with pytest.raises(RuntimeError, match="Superseded") as cancelled:
        tracing.trace_feg_to_exit(gun, 9, cancelled=lambda: len(inputs) >= 5)
    assert len(inputs) == 5
    for phase, position, momentum, active, mask in inputs:
        np.testing.assert_array_equal(phase.position_m, position)
        np.testing.assert_array_equal(phase.momentum_kg_m_per_s, momentum)
        np.testing.assert_array_equal(active, mask)

    # Cancellation intentionally publishes no result. Inspect the abandoned
    # trace frame to check its retained snapshots before the frame is released.
    frame = cancelled.value.__traceback__
    while frame.tb_frame.f_code is not tracing.trace_feg_to_exit.__code__:
        frame = frame.tb_next
    pending = frame.tb_frame.f_locals
    positions, momenta = pending["history_position"], pending["history_momentum"]
    expected = [(row[1], row[2]) for row in inputs]
    expected.append((pending["phase"].position_m, pending["phase"].momentum_kg_m_per_s))
    assert len(positions) == len(momenta) == len(expected) == 6
    for index, (position, momentum) in enumerate(zip(positions, momenta, strict=True)):
        np.testing.assert_array_equal(position, expected[index][0])
        np.testing.assert_array_equal(momentum, expected[index][1])
        for state, *_ in inputs:
            assert not np.shares_memory(position, state.position_m)
            assert not np.shares_memory(momentum, state.momentum_kg_m_per_s)
        if index:
            assert not np.shares_memory(position, positions[index - 1])
            assert not np.shares_memory(momentum, momenta[index - 1])
    if with_medium:
        assert len(medium_checks) == 5
        assert np.isfinite(pending["blocked_time_s"][0])
        assert pending["blocked_key"][0] == "medium_removal:ownership_fixture"
