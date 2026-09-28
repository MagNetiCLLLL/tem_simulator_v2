"""Bounded scalar performance observations; no new physical approximation."""
from collections import Counter
from dataclasses import fields
import io
import pickle
from queue import Queue
from threading import Event
import time
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.electron_execution_performance import (
    ExecutionPerformance, MAX_EVENTS, validate_performance_payload,
)
from temsim.test_electron_execution import (
    ElectronExecutionBackend, ElectronExecutionCancelled, ElectronExecutionError, ElectronExecutionPolicy,
    _ProfiledCommand, _dumps, _receive, _send,
)


def test_submillisecond_durations_do_not_use_coarse_policy_clock(monkeypatch):
    from temsim import electron_execution_performance as performance
    ticks = iter((10., 10.001, 10.001002))
    monkeypatch.setattr(performance, "perf_counter", lambda: next(ticks))
    recorder = performance.ExecutionPerformance(1, "trace", scope="child")
    with recorder.stage("short_acceptance"):
        pass
    assert recorder.snapshot()["stages"][0]["seconds"] == pytest.approx(2e-6)
    assert recorder.started == 10.


def test_recording_is_scalar_bounded_and_nested_timings_are_explicit():
    recorder = ExecutionPerformance(4, "trace", scope="child")
    with recorder.stage("particle_transport"):
        with recorder.stage("compiler_warmup"):
            recorder.increment("trajectory_executions")
    payload = recorder.snapshot()
    assert validate_performance_payload(payload) is payload
    assert payload["actual_backend"] == "reference"
    stages = {row["stage"]: row for row in payload["stages"]}
    assert stages["particle_transport"]["seconds"] >= stages["compiler_warmup"]["seconds"]
    assert all(row["nested"] for row in stages.values())
    assert "do not sum" in payload["accounting"]
    for index in range(MAX_EVENTS + 10):
        recorder.events.record("bounded", index=index)
    assert len(recorder.snapshot()["events"]) == MAX_EVENTS


@pytest.mark.parametrize("invalid", [np.zeros(2), float("nan"), float("inf"), "x"*1025])
def test_ipc_performance_rejects_arrays_and_unbounded_or_nonfinite_values(invalid):
    payload = ExecutionPerformance(1, "trace", scope="child").snapshot()
    payload["events"].append({"value": invalid})
    with pytest.raises(ValueError):
        validate_performance_payload(payload)


@pytest.mark.parametrize("change", [
    lambda p: p["counts"].update(field_solver_calls=-1),
    lambda p: p["counts"].update(field_solver_calls=True),
    lambda p: p["counts"].update(field_solver_calls=2**64),
    lambda p: p["stages"].append({"seconds": 1.}),
    lambda p: p.update(actual_backend="invented_gpu"),
    lambda p: p.update(extra={}),
])
def test_ipc_performance_schema_is_strict(change):
    payload = ExecutionPerformance(1, "trace", scope="child").snapshot()
    change(payload)
    with pytest.raises(ValueError):
        validate_performance_payload(payload)


def test_numba_compile_observer_distinguishes_load_build_and_existing_overload():
    recorder = ExecutionPerformance(1, "trace", scope="child")
    dispatcher = SimpleNamespace(_cache_hits=Counter(), _cache_misses=Counter(), overloads={})
    result = object()

    def compile(original, signature):
        if signature not in original.overloads:
            (original._cache_hits if signature == "disk" else original._cache_misses)[signature] += 1
            original.overloads[signature] = result
        return result

    observed = recorder._compile_wrapper(compile)
    assert observed(dispatcher, "disk") is result
    assert observed(dispatcher, "compile") is result
    assert observed(dispatcher, "compile") is result
    counts = recorder.snapshot()["counts"]
    assert counts["compiler_calls"] == 3
    assert counts["compiler_cache_hits"] == counts["compiler_cache_misses"] == 1
    assert counts["compiler_new_signatures"] == 2


def test_prepare_observes_actual_solver_and_restores_methods_even_on_exception(monkeypatch):
    from temsim.physics import closed_gun_field as closed
    from temsim.physics.axisymmetric_cut_field import AxisymmetricCutField
    calls = []

    def construct(self, *args, **kwargs):
        calls.append("solver")

    def access(_gun):
        return object()

    monkeypatch.setattr(AxisymmetricCutField, "__init__", construct)
    monkeypatch.setattr(closed, "closed_field", access)
    recorder = ExecutionPerformance(1, "prepare", scope="child")
    with pytest.raises(RuntimeError, match="fixture"):
        with recorder.observe():
            closed.closed_field(None)
            AxisymmetricCutField()
            raise RuntimeError("fixture failure")
    assert AxisymmetricCutField.__init__ is construct
    assert closed.closed_field is access
    counts = recorder.snapshot()["counts"]
    assert calls == ["solver"] and counts["field_solver_calls"] == 1
    assert counts["field_accessor_calls"] == counts["field_memory_cache_hits"] == 1
    assert counts["field_disk_load_hits"] == 0


def test_wire_observations_preserve_exact_original_message():
    message = (7, _ProfiledCommand(("ready",)))
    assert pickle.loads(_dumps(message)) == message
    recorder = ExecutionPerformance(7, "ready", scope="parent")
    pipe = io.BytesIO()
    _send(pipe, message, performance=recorder)
    pipe.seek(0)
    assert _receive(pipe, performance_getter=lambda: recorder) == message
    counts = recorder.snapshot()["counts"]
    assert counts["serialized_bytes"] == counts["received_bytes"] == len(_dumps(message))


@pytest.fixture
def protocol_backend(monkeypatch):
    backend = ElectronExecutionBackend(measure_performance=True)
    backend._responses = Queue()
    backend._process = SimpleNamespace(stdin=io.BytesIO(), pid=-1, poll=lambda: None)
    monkeypatch.setattr(backend, "_start", lambda: None)
    monkeypatch.setattr(backend, "_stop_process", lambda: None)
    yield backend
    backend.close()


def test_performance_frame_precedes_unchanged_result_and_toggle_is_serial(protocol_backend):
    backend = protocol_backend
    child = ExecutionPerformance(1, "ready", scope="child")
    backend._responses.put((1, "performance", child.snapshot()))
    backend._responses.put((1, "ok", "ready"))
    assert backend.start() == "ready"
    report = backend.last_performance
    assert report["request_kind"] == "ready" and report["sequence"] == 1
    assert report["child_measurement_received"] and report["diagnostic"] is None
    assert report["counts"]["trajectory_executions"] == 0
    assert {s["stage"] for s in report["stages"]} >= {"process_reuse", "result_acceptance"}
    backend.measure_performance = False
    backend._responses.put((2, "ok", "unmeasured"))
    assert backend.start() == "unmeasured" and backend.last_performance is None
    backend.measure_performance = True
    backend._responses.put((3, "performance", child.snapshot()))
    backend._responses.put((3, "ok", "measured again"))
    assert backend.start() == "measured again"
    assert backend.last_performance["sequence"] == 3


@pytest.mark.parametrize("frames", [
    [(1, "ok", "missing observations")],
    [(2, "performance", "wrong sequence")],
    [(1, "performance", {"events": np.ones(2)})],
])
def test_invalid_or_absent_performance_is_explicit_protocol_failure(protocol_backend, frames):
    for frame in frames:
        protocol_backend._responses.put(frame)
    with pytest.raises(ElectronExecutionError):
        protocol_backend.start()
    assert protocol_backend.last_performance["diagnostic"] is not None


def small_executed_scene():
    """Explicit constant-potential fixture; no physical instrument qualification."""
    from temsim.magnetic_field_scene import MagneticSceneField
    from temsim.physics.closed_gun_field import ClosedGunField
    from temsim.test_electron_scene import TestElectronScene
    bounds = np.array(((-1e-3, -1e-3, 0.), (1e-3, 1e-3, 1e-4)))
    bounds.setflags(write=False)
    field = ClosedGunField({}, np.linspace(0., 1e-3, 3), np.linspace(0., 1e-4, 3), np.zeros((3, 3)))
    magnetic = MagneticSceneField(bounds, 1e-4, (), (), (), (), bounds)
    return TestElectronScene(magnetic, field, field, bounds, bounds, (0., 0., 0.), .3,
        1e-4, ("Constant-potential protocol/performance fixture only.",), (), (), (), _flat_cathode=True)


def test_real_process_probe_toggle_and_reference_share_unchanged_prepared_scene(tmp_path):
    from temsim.magnetic_test_particle import TestElectronSettings
    scene = small_executed_scene()
    settings = TestElectronSettings(kinetic_energy_ev=.3, max_path_length_m=1e-6,
                                   step_m=1e-7, max_steps=128)
    backend = ElectronExecutionBackend(measure_performance=True, diagnostic_root=tmp_path,
        policy=ElectronExecutionPolicy(startup_timeout_s=30., prepare_timeout_s=30., trace_timeout_s=60.))
    try:
        remote = backend.install_prepared(scene)
        assert backend.last_performance["counts"]["field_solver_calls"] == 0
        reference = backend.trace(remote, settings, use_compiled=False)
        assert backend.last_performance["actual_backend"] == "reference"
        assert backend.last_performance["counts"]["trajectory_executions"] == 1
        measured = backend.trace(remote, settings)
        report = backend.last_performance
        assert report["actual_backend"] == "compiled_blocks"
        assert report["counts"]["field_solver_calls"] == report["counts"]["exact_result_cache_hits"] == 0
        backend.measure_performance = False
        plain = backend.trace(remote, settings)
        assert backend.last_performance is None and backend.owns_scene(remote)
        for item in fields(measured):
            a, b = getattr(measured, item.name), getattr(plain, item.name)
            if isinstance(a, np.ndarray):
                np.testing.assert_array_equal(a, b)
                assert not b.flags.writeable
        np.testing.assert_allclose(measured.positions_m, reference.positions_m, rtol=1e-10, atol=1e-15)
        backend.measure_performance = True
        backend.trace(remote, settings)
        assert backend.last_performance["counts"]["trajectory_executions"] == 1
        assert backend.last_performance["child_measurement_received"]
    finally:
        backend.close()


@pytest.mark.parametrize("measured", [False, True])
def test_rapid_edit_during_trace_send_drains_frame_and_keeps_prepared_fields(tmp_path, monkeypatch, measured):
    from temsim import test_electron_execution as execution
    from temsim.magnetic_test_particle import TestElectronSettings
    cancellation = Event()
    backend = ElectronExecutionBackend(measure_performance=measured, diagnostic_root=tmp_path)
    settings = TestElectronSettings(max_path_length_m=1e-6, step_m=1e-7, max_steps=128)
    original = execution._send
    delayed = []

    def cancel_during_write(stream, payload, **kwargs):
        command = payload[1]
        command = command.command if isinstance(command, execution._ProfiledCommand) else command
        if command[0] == "trace" and not delayed:
            delayed.append(True)
            cancellation.set()
            time.sleep(.05)
        return original(stream, payload, **kwargs)

    try:
        remote = backend.install_prepared(small_executed_scene())
        pid = backend.process_id
        monkeypatch.setattr(execution, "_send", cancel_during_write)
        with pytest.raises(ElectronExecutionCancelled) as caught:
            backend.trace(remote, settings, cancelled=cancellation.is_set)
        assert not caught.value.fields_lost
        assert backend.process_id == pid and backend.owns_scene(remote)
        cancellation.clear()
        result = backend.trace(remote, settings)
        assert result.completed and result.reason == "path_limit"
        assert backend.process_id == pid
    finally:
        backend.close()


def test_cancelled_trace_with_stuck_sender_remains_bounded(tmp_path, monkeypatch):
    from temsim import test_electron_execution as execution
    from temsim.magnetic_test_particle import TestElectronSettings
    entered, release = Event(), Event()
    backend = ElectronExecutionBackend(diagnostic_root=tmp_path,
        policy=ElectronExecutionPolicy(cancel_grace_s=.1, terminate_grace_s=.05))
    original = execution._send
    pending = None

    def blocked_send(stream, payload, **kwargs):
        if payload[1][0] == "trace":
            entered.set()
            if not release.wait(5.):
                raise RuntimeError("Test sender watchdog expired")
        return original(stream, payload, **kwargs)

    try:
        remote = backend.install_prepared(small_executed_scene())
        monkeypatch.setattr(execution, "_send", blocked_send)
        started = time.perf_counter()
        with pytest.raises(ElectronExecutionCancelled) as caught:
            backend.trace(remote, TestElectronSettings(), cancelled=entered.is_set)
        assert time.perf_counter()-started < 2.
        assert caught.value.fields_lost and backend.process_id is None
        assert caught.value.diagnostic["reason"] == "send_timeout"
        pending = backend._command_write
        assert pending is not None and pending.poll() is None
    finally:
        release.set()
        if pending is not None:
            pending.thread.join(2.)
            assert not pending.thread.is_alive()
        backend.close()
