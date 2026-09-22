"""Small deterministic checks for bounded, event-preserving gun batches."""
from threading import Event

import numpy as np
import pytest

from test_analytic_particle_step import gun, _phase
from temsim.optics.electron_gun import tracing
from temsim.physics import analytic_particle_batch as batching
from temsim.physics import analytic_particle_step as stepping
from temsim.physics.relativistic_lorentz import velocity_from_momentum_m_per_s


pytestmark = pytest.mark.skipif(batching._compiled_batch is None, reason="Numba optional")


def _execution(gun, count):
    return stepping.prepare_analytic_execution(gun, gun.magnetic_field, gun.electric_field, count)


def _reference_step(gun, execution, phase, active, energy):
    velocity = velocity_from_momentum_m_per_s(phase.momentum_kg_m_per_s[active])
    dt = gun.integration_step_mm_at(phase.position_m[active, 2]*1000.)*1e-3/max(float(np.max(velocity[:, 2])), 1.)
    assert execution.begin_step(phase, active)
    dt = execution.time_step(dt, tracing.ANALYTIC_MAXIMUM_RELATIVE_IMPULSE)
    advanced, dt = execution.step(dt, energy[active])
    advanced.momentum_kg_m_per_s[active] = execution.project_momentum(
        advanced.position_m[active], advanced.momentum_kg_m_per_s[active], energy[active])
    return advanced, dt


@pytest.mark.parametrize("z", [[.01, 1., 5., 13., 14., 41., 77., 402., 418., 433., 550.], [13.]*1150])
def test_outer_projection_retains_numpy_operation_order(gun, z):
    phase, energy = _phase(gun, z)
    actual = phase.momentum_kg_m_per_s.copy()
    expected = tracing._enforce_static_field_energy(gun, phase.position_m, actual, energy)
    terms = stepping._field_parameters(gun.electric_field, gun.magnetic_field)[0]
    batching._outer_projection(phase.position_m, actual, np.arange(len(z)), energy, terms)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("count,maximum", [(9, 1), (9, 7), (9, 100), (1150, 7)])
def test_batch_matches_every_reference_step_and_owns_its_arrays(gun, count, maximum):
    phase, energy = _phase(gun, 13., count=count)
    active = np.ones(count, bool)
    active[::7] = False
    passed = np.zeros(count, bool)
    batch = batching.prepare_analytic_batch(gun, _execution(gun, count), Event().is_set)
    assert batch is not None
    actual = batch.advance(phase, active, energy, passed, passed,
                          maximum_steps=maximum, impulse=.025)
    assert actual is not None
    advanced, old_x, old_p, old_time, dt, steps = actual
    assert steps == min(maximum, batching.MAX_BATCH_STEPS)
    assert batch.accepted_steps == batch.maximum_batch_steps == steps
    assert batch.completed_batches == 1
    reference, execution = phase, _execution(gun, count)
    for _ in range(steps):
        previous = reference
        reference, reference_dt = _reference_step(gun, execution, reference, active, energy)
    assert dt == reference_dt
    assert advanced.time_s == reference.time_s
    assert old_time == previous.time_s
    for left, right in ((advanced.position_m, reference.position_m),
                        (advanced.momentum_kg_m_per_s, reference.momentum_kg_m_per_s),
                        (old_x, previous.position_m), (old_p, previous.momentum_kg_m_per_s)):
        np.testing.assert_array_equal(left, right)
        assert not np.shares_memory(left, phase.position_m)
        assert not np.shares_memory(left, phase.momentum_kg_m_per_s)
    np.testing.assert_array_equal(advanced.position_m[~active], phase.position_m[~active])


def test_unknown_cancel_callback_and_custom_step_fall_back(gun, monkeypatch):
    execution = _execution(gun, 9)
    assert batching.prepare_analytic_batch(gun, execution, lambda: False) is None
    monkeypatch.setattr(gun, "integration_step_mm_at", lambda _z: .2)
    assert batching.prepare_analytic_batch(gun, execution, None) is None


def test_original_callbacks_are_not_bypassed_and_current_fields_remain_live(gun, monkeypatch):
    phase, energy = _phase(gun, 13., count=9)
    active, passed = np.ones(9, bool), np.zeros(9, bool)
    batch = batching.prepare_analytic_batch(gun, _execution(gun, 9), None)
    monkeypatch.setattr(tracing, "_clip_body_bores", lambda *_a: None)
    assert batch.advance(phase, active, energy, passed, passed, maximum_steps=5, impulse=.025) is None
    monkeypatch.undo()
    gun.extractor.voltage_kv += .2
    result = batch.advance(phase, active, energy, passed, passed, maximum_steps=5, impulse=.025)
    reference, execution = phase, _execution(gun, 9)
    for _ in range(result[-1]):
        reference, _ = _reference_step(gun, execution, reference, active, energy)
    np.testing.assert_array_equal(result[0].position_m, reference.position_m)
    np.testing.assert_array_equal(result[0].momentum_kg_m_per_s, reference.momentum_kg_m_per_s)


def _exact_result(left, right):
    from dataclasses import fields, is_dataclass
    if isinstance(left, np.ndarray):
        assert left.dtype == right.dtype
        np.testing.assert_array_equal(left, right)
    elif is_dataclass(left):
        for field in fields(left):
            _exact_result(getattr(left, field.name), getattr(right, field.name))
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _exact_result(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _exact_result(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("stop", ["none", "aperture", "bore", "aligned_refined"])
def test_complete_nine_ray_trace_preserves_history_events_identity_and_tof(gun, stop):
    if stop == "aperture":
        gun.dpa_aperture.radius_mm = 0.
    elif stop == "bore":
        gun.extractor.mechanical_clear_bore_diameter_mm = 1e-8
    elif stop == "aligned_refined":
        gun.trace_step_mm *= .5
        gun.deflector.upper_field_x_mt = .08
        gun.deflector.lower_field_y_mt = -.03
        gun.stigmator.gradient_t_per_m = 2.
        gun.stigmator.rotation_deg = 37.
    gun.compiled_particle_batches = False
    reference = tracing.trace_feg_to_exit(gun, 9, cancelled=Event().is_set)
    gun.compiled_particle_batches = True
    actual = tracing.trace_feg_to_exit(gun, 9, cancelled=Event().is_set)
    _exact_result(actual, reference)
    if stop in ("aperture", "bore"):
        assert any(actual.blocked_key)


@pytest.mark.parametrize("event", ["dpa", "c1", "exit", "bore", "backstream"])
def test_every_boundary_candidate_returns_to_original_event_processing(event):
    old = np.array([[0., 0., 1.]])
    new = np.array([[0., 0., 1.1]])
    momentum = np.array([[0., 0., 1.]])
    planes, bores = np.full(3, 10.), np.array([[2000., 1., 1.]])
    if event in ("dpa", "c1", "exit"):
        planes[("dpa", "c1", "exit").index(event)] = 1.05
    elif event == "bore":
        bores = np.array([[1100., .1, .5]])
        new[0, 0] = .001
    else:
        momentum[0, 2] = -1.
    assert batching._boundary_candidate(old, new, momentum, np.array([0]),
        bores, planes, np.zeros(1, bool), np.zeros(1, bool))


def test_unsupported_physics_and_custom_aperture_are_not_admitted(gun, monkeypatch):
    execution = _execution(gun, 9)
    gun._vacuum_regions = (object(),)
    assert batching.prepare_analytic_batch(gun, execution, None) is None
    gun._vacuum_regions = ()
    monkeypatch.setattr(gun.dpa_aperture, "transmission_mask", lambda x, y: np.ones_like(x, bool))
    assert batching.prepare_analytic_batch(gun, execution, None) is None


def test_event_cancellation_is_observed_at_the_next_bounded_boundary(gun, monkeypatch):
    cancellation, accepted = Event(), []
    original = batching._compiled_batch
    def execute(*args):
        result = original(*args)
        accepted.append(result[-1])
        cancellation.set()
        return result
    monkeypatch.setattr(batching, "_compiled_batch", execute)
    with pytest.raises(RuntimeError, match="Superseded"):
        tracing.trace_feg_to_exit(gun, 9, cancelled=cancellation.is_set)
    assert accepted == [1]  # Initial history step cannot be crossed.


def test_custom_execution_hooks_keep_per_step_fallback(gun, monkeypatch):
    phase, energy = _phase(gun, 13., count=9)
    execution = _execution(gun, 9)
    batch = batching.prepare_analytic_batch(gun, execution, None)
    monkeypatch.setattr(execution, "project_momentum", lambda *_a: None)
    active, passed = np.ones(9, bool), np.zeros(9, bool)
    assert batch.advance(phase, active, energy, passed, passed, maximum_steps=5, impulse=.025) is None
