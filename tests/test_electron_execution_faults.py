"""Bounded lifecycle fault injection, distinct from numerical validation.

Real subprocess fixtures use only scripted IPC, sleeps, exits and stderr. The
existing test_test_electron_execution.py retains the actual field/integration
tests. Every potentially blocking call here has an independent safety watchdog.
"""
from dataclasses import replace
import io
import json
from pathlib import Path
from queue import Queue
from threading import Event, Thread, Timer
import time
from types import SimpleNamespace

import pytest

from temsim.magnetic_test_particle import TestElectronSettings
from temsim.test_electron_execution import (
    ElectronExecutionBackend, ElectronExecutionCancelled, ElectronExecutionError,
    ElectronExecutionPolicy,
)


def bounded_call(backend, call, *, timeout=15.):
    """A failed bounded-wait implementation must fail the test, not hang pytest."""
    outcomes = []

    def run():
        try:
            outcomes.append((True, call()))
        except BaseException as error:
            outcomes.append((False, error))

    worker = Thread(target=run, name="fault-test-request", daemon=True)
    worker.start()
    worker.join(timeout)
    timed_out = worker.is_alive()
    if timed_out:
        # A test-only escape route also wakes old implementations that rely
        # entirely on the result queue instead of checking process liveness.
        if backend._responses is not None:
            backend._responses.put(("disconnected", "Fault-test safety watchdog"))
        closer = Thread(target=backend.close, daemon=True)
        closer.start()
        closer.join(3.)
        worker.join(3.)
    assert not timed_out, "Backend exceeded the independent fault-test watchdog"
    assert not worker.is_alive()
    assert len(outcomes) == 1
    success, value = outcomes[0]
    if not success:
        raise value
    return value


@pytest.fixture
def fault_backend(monkeypatch, tmp_path):
    mode = ["normal"]
    marker = tmp_path / "entered-stage.txt"
    helper = Path(__file__).with_name("electron_fault_worker.py")
    monkeypatch.setattr(ElectronExecutionBackend, "_child_command", lambda self, executable, identity:
                        [executable, str(helper), identity, mode[0], str(marker)])
    policy = ElectronExecutionPolicy(startup_timeout_s=10., prepare_timeout_s=5.,
        trace_timeout_s=5., send_timeout_s=5., cancel_grace_s=.12,
        terminate_grace_s=.3, poll_interval_s=.01)
    backend = ElectronExecutionBackend(policy=policy, diagnostic_root=tmp_path / "diagnostics")
    yield backend, mode, marker
    # close should itself be bounded; use a safety thread so lifecycle
    # regressions cannot strand the remainder of the test suite.
    closer = Thread(target=backend.close, daemon=True)
    closer.start()
    closer.join(3.)
    assert not closer.is_alive(), "Closing an owned diagnostic process stalled"


def install_fixture(backend):
    return bounded_call(backend, lambda: backend.install_prepared(None))


def entered_trace(marker):
    try:
        return marker.read_text(encoding="utf-8") == "trace"
    except (FileNotFoundError, PermissionError):
        return False


def assert_failure_record(backend, error):
    assert isinstance(error.diagnostic, dict) and error.diagnostic
    assert backend.last_diagnostic == error.diagnostic
    assert backend.last_diagnostic_path
    report = Path(backend.last_diagnostic_path)
    assert report.is_file()
    assert isinstance(json.loads(report.read_text(encoding="utf-8")), dict)
    return report


@pytest.mark.parametrize("stage,code", [("prepare", 17), ("trace", 19)])
def test_child_exit_during_work_is_reported_and_invalidates_fields(fault_backend, stage, code):
    backend, mode, _ = fault_backend
    mode[0] = f"exit_{stage}"
    if stage == "prepare":
        operation = lambda: backend.prepare(None, None, z_limits_mm=(0., 10.))
    else:
        remote = install_fixture(backend)
        operation = lambda: backend.trace(remote, TestElectronSettings())
    with pytest.raises(ElectronExecutionError) as captured:
        bounded_call(backend, operation)
    assert captured.value.fields_lost
    assert backend.process_id is None
    report = assert_failure_record(backend, captured.value)
    evidence = report.read_text(encoding="utf-8")
    assert str(code) in evidence
    assert stage in evidence.lower()


@pytest.mark.parametrize("mode_name", ["nan_position", "wrong_direction_shape", "forged_execution_identity"])
def test_corrupt_trajectory_payload_is_rejected_before_returning_to_gui(fault_backend, mode_name):
    backend, mode, _ = fault_backend
    mode[0] = mode_name
    remote = install_fixture(backend)
    with pytest.raises(ElectronExecutionError) as caught:
        bounded_call(backend, lambda: backend.trace(remote, TestElectronSettings()))
    assert caught.value.fields_lost and not backend.owns_scene(remote)
    assert caught.value.diagnostic["reason"] == "protocol_error"
    assert_failure_record(backend, caught.value)


def test_invalid_scene_support_bounds_never_become_an_active_handle(fault_backend):
    backend, mode, _ = fault_backend
    mode[0] = "invalid_bounds"
    with pytest.raises(ElectronExecutionError) as caught:
        install_fixture(backend)
    assert caught.value.fields_lost and backend._remote_scene is None
    assert_failure_record(backend, caught.value)


def test_alive_child_ignoring_cancel_is_killed_after_grace_and_never_accepts_prefix(fault_backend):
    backend, mode, marker = fault_backend
    mode[0] = "ignore_cancel"
    remote = install_fixture(backend)
    process = backend._process
    started = time.monotonic()
    with pytest.raises(ElectronExecutionCancelled) as captured:
        bounded_call(backend, lambda: backend.trace(remote, TestElectronSettings(),
                                                    cancelled=lambda: entered_trace(marker)))
    assert time.monotonic() - started < 3.
    assert entered_trace(marker)
    assert captured.value.fields_lost
    assert process.poll() is not None and backend.process_id is None
    assert not backend.owns_scene(remote)
    assert_failure_record(backend, captured.value)


def test_cooperative_cancel_keeps_fields_and_accepts_following_electron(fault_backend):
    backend, mode, marker = fault_backend
    mode[0] = "cooperative_cancel"
    remote = install_fixture(backend)
    pid = backend.process_id
    with pytest.raises(ElectronExecutionCancelled) as captured:
        bounded_call(backend, lambda: backend.trace(remote, TestElectronSettings(),
                                                    cancelled=lambda: entered_trace(marker)))
    assert not captured.value.fields_lost
    assert backend.process_id == pid and backend.owns_scene(remote)
    result = bounded_call(backend, lambda: backend.trace(remote, TestElectronSettings()))
    assert result.completed and result.reason == "path_limit"


@pytest.mark.parametrize("mode_name", ["bad_response_shape", "wrong_sequence", "partial_frame", "oversized_frame"])
def test_corrupt_or_mismatched_response_stops_owner(fault_backend, mode_name):
    backend, mode, _ = fault_backend
    mode[0] = mode_name
    with pytest.raises(ElectronExecutionError) as captured:
        bounded_call(backend, lambda: backend.prepare(None, None, z_limits_mm=(0., 10.)))
    assert captured.value.fields_lost
    assert backend.process_id is None
    assert_failure_record(backend, captured.value)


@pytest.mark.parametrize("mode_name", ["wrong_owner", "empty_token", "not_metadata"])
def test_prepared_metadata_must_belong_to_this_live_worker(fault_backend, mode_name):
    backend, mode, _ = fault_backend
    mode[0] = mode_name
    with pytest.raises(ElectronExecutionError) as captured:
        install_fixture(backend)
    assert captured.value.fields_lost
    assert backend._scene_token is None and backend.process_id is None


@pytest.mark.parametrize("mode_name", ["bad_progress", "stale_progress", "unrequested_progress", "terminal_prefix"])
def test_invalid_progress_or_final_prefix_cannot_be_accepted(fault_backend, mode_name):
    backend, mode, _ = fault_backend
    mode[0] = mode_name
    remote = install_fixture(backend)
    observed = []
    progress = None if mode_name == "unrequested_progress" else observed.append
    with pytest.raises(ElectronExecutionError):
        bounded_call(backend, lambda: backend.trace(remote, TestElectronSettings(), progress=progress))
    assert not observed


def test_stderr_flood_is_drained_and_failure_evidence_survives(fault_backend):
    backend, mode, _ = fault_backend
    mode[0] = "stderr_flood"
    remote = install_fixture(backend)
    with pytest.raises(ElectronExecutionError) as captured:
        bounded_call(backend, lambda: backend.trace(remote, TestElectronSettings()))
    assert captured.value.fields_lost
    report = assert_failure_record(backend, captured.value)
    artifacts = [path for path in report.parent.rglob("*") if path.is_file()]
    text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in artifacts)
    assert "FAULT_STDERR_FLOOD_FINISHED" in text
    assert "FAULT_STDERR_FLOOD_FINISHED" in captured.value.diagnostic["stderr_tail"]
    assert "29" in report.read_text(encoding="utf-8")
    logs = list(report.parent.glob("stderr.log*"))
    assert 1 <= len(logs) <= 3
    assert all(path.stat().st_size <= 256*1024 for path in logs)
    assert report.stat().st_size <= 256*1024
    assert backend.process_id is None


def test_uncaught_python_traceback_is_preserved_in_failure_evidence(fault_backend):
    backend, mode, _ = fault_backend
    mode[0] = "python_traceback"
    remote = install_fixture(backend)
    with pytest.raises(ElectronExecutionError) as captured:
        bounded_call(backend, lambda: backend.trace(remote, TestElectronSettings()))
    report = assert_failure_record(backend, captured.value)
    text = "\n".join(path.read_text(encoding="utf-8", errors="replace")
                     for path in report.parent.rglob("*") if path.is_file())
    assert "Traceback (most recent call last)" in text
    assert "INJECTED_PYTHON_TRACEBACK_FROM_SCRIPTED_WORKER" in text


def test_cancellation_during_preparation_kills_child_without_waiting_for_trace_grace(fault_backend):
    backend, mode, marker = fault_backend
    mode[0] = "ignore_prepare"
    with pytest.raises(ElectronExecutionCancelled) as captured:
        bounded_call(backend, lambda: backend.prepare(None, None, z_limits_mm=(0., 10.),
                                                    cancelled=marker.exists))
    assert captured.value.fields_lost and backend.process_id is None
    assert_failure_record(backend, captured.value)


def test_large_command_send_to_unresponsive_reader_is_bounded(fault_backend):
    backend, mode, _ = fault_backend
    mode[0] = "ignore_input"
    backend.policy = replace(backend.policy, send_timeout_s=.3)
    with pytest.raises(ElectronExecutionError) as captured:
        bounded_call(backend, lambda: backend.prepare(b"x" * (512*1024), None,
                                                    z_limits_mm=(0., 10.)), timeout=5.)
    assert captured.value.fields_lost and backend.process_id is None
    assert captured.value.diagnostic.get("reason") == "send_timeout"


def test_cancellation_can_interrupt_blocked_command_send(fault_backend):
    backend, mode, marker = fault_backend
    mode[0] = "ignore_input"
    with pytest.raises(ElectronExecutionCancelled) as captured:
        bounded_call(backend, lambda: backend.prepare(b"x" * (512*1024), None,
                                                    z_limits_mm=(0., 10.), cancelled=marker.exists), timeout=5.)
    assert captured.value.fields_lost and backend.process_id is None


def test_cancelled_parent_serializer_keeps_cpu_admission_blocked_until_it_exits(fault_backend, monkeypatch):
    from temsim import test_electron_execution as execution
    from temsim.cpu_resources import numerical_job

    backend, _, _ = fault_backend
    entered, release = Event(), Event()
    original = execution._dumps

    def slow_serializer(value):
        entered.set()
        if not release.wait(8.):
            raise RuntimeError("Serializer safety watchdog")
        return original(value)

    monkeypatch.setattr(execution, "_dumps", slow_serializer)
    pending = None
    try:
        with pytest.raises(ElectronExecutionCancelled):
            bounded_call(backend, lambda: backend.start(cancelled=entered.is_set), timeout=5.)
        pending = backend._command_write
        assert pending is not None and pending.poll() is None
        assert backend.process_id is None
        with pytest.raises(RuntimeError, match="command writer"):
            with numerical_job(1):
                pytest.fail("An unfinished serializer must not overlap new numerical work")
    finally:
        release.set()
        if pending is not None:
            pending.thread.join(3.)
            assert not pending.thread.is_alive()
    with numerical_job(1) as receipt:
        assert receipt.numerical_thread_budget == 1


def test_long_normal_trace_without_prefix_is_allowed_within_stage_budget(fault_backend):
    backend, mode, _ = fault_backend
    mode[0] = "slow_trace"
    remote = install_fixture(backend)
    started = time.monotonic()
    result = bounded_call(backend, lambda: backend.trace(remote, TestElectronSettings()))
    assert result.completed and result.reason == "path_limit"
    assert time.monotonic() - started >= .2
    assert backend.owns_scene(remote)


def test_stage_timeout_is_separate_from_cancellation_grace(fault_backend):
    backend, mode, _ = fault_backend
    mode[0] = "ignore_cancel"
    # Constructor policy replacement is explicit configuration, not a short
    # global deadline imposed on every legitimate field solve.
    backend.policy = replace(backend.policy, trace_timeout_s=.15)
    remote = install_fixture(backend)
    started = time.monotonic()
    with pytest.raises(ElectronExecutionError) as captured:
        bounded_call(backend, lambda: backend.trace(remote, TestElectronSettings()))
    assert time.monotonic() - started < 3.
    assert captured.value.fields_lost
    assert "time" in str(captured.value).lower()
    assert_failure_record(backend, captured.value)


def test_explicit_retry_uses_new_process_and_released_cpu_admission(fault_backend):
    backend, mode, _ = fault_backend
    mode[0] = "exit_trace"
    old = install_fixture(backend)
    with pytest.raises(ElectronExecutionError):
        bounded_call(backend, lambda: backend.trace(old, TestElectronSettings()))
    mode[0] = "normal"
    current = install_fixture(backend)
    assert old.process_identity != current.process_identity
    assert not backend.owns_scene(old) and backend.owns_scene(current)
    assert bounded_call(backend, lambda: backend.trace(current, TestElectronSettings())).completed


def test_cpu_admission_reopens_only_after_ignored_cancel_child_has_stopped(fault_backend):
    """A waiting live thread cannot inherit the old RLock owner's thread id."""
    from temsim import cpu_resources

    backend, mode, marker = fault_backend
    mode[0] = "ignore_cancel"
    remote = install_fixture(backend)
    process = backend._process
    cancel_request, stop_probe = Event(), Event()
    probe_attempted, probe_acquired = Event(), Event()
    request_outcomes, probe_outcomes = [], []
    request_budgets_after = []

    def trace():
        try:
            request_outcomes.append(backend.trace(
                remote, TestElectronSettings(), cancelled=cancel_request.is_set))
        except BaseException as error:
            request_outcomes.append(error)
        finally:
            request_budgets_after.append(cpu_resources._ACTIVE_BUDGET.get())

    def probe_cancelled():
        # Called after a failed lock acquisition, or immediately after entry.
        # Paired with probe_acquired below, this proves the probe is contending.
        probe_attempted.set()
        return stop_probe.is_set()

    def probe():
        observation = {}
        try:
            with cpu_resources.numerical_job(1, cancelled=probe_cancelled) as receipt:
                observation.update(exit_code_at_admission=process.poll(),
                    budget=receipt.numerical_thread_budget, numba_threads=receipt.numba_threads,
                    active_budget=cpu_resources._ACTIVE_BUDGET.get())
                probe_acquired.set()
        except BaseException as error:
            observation["error"] = error
        finally:
            observation["budget_after"] = cpu_resources._ACTIVE_BUDGET.get()
            probe_outcomes.append(observation)

    def request_safety():
        cancel_request.set()
        if backend._responses is not None:
            backend._responses.put(("disconnected", "CPU admission test safety watchdog"))
        Thread(target=backend.close, name="cpu-admission-safety-close", daemon=True).start()

    request_thread = Thread(target=trace, name="cpu-admission-active-request", daemon=True)
    probe_thread = Thread(target=probe, name="cpu-admission-persistent-probe", daemon=True)
    request_watchdog = Timer(8., request_safety)
    probe_watchdog = Timer(10., stop_probe.set)
    request_watchdog.daemon = probe_watchdog.daemon = True
    request_watchdog.start()
    probe_watchdog.start()
    request_thread.start()
    try:
        deadline = time.monotonic() + 4.
        while not entered_trace(marker) and request_thread.is_alive() and time.monotonic() < deadline:
            time.sleep(.01)
        assert entered_trace(marker), "Fixture did not reach its deliberately unresponsive trace"
        assert request_thread.is_alive() and process.poll() is None

        # Start before cancellation, while the owner's identity is still live.
        # Starting only after it exits could reuse that identity and falsely
        # permit recursive admission to a leaked RLock.
        probe_thread.start()
        assert probe_attempted.wait(1.), "Probe never attempted CPU admission"
        assert request_thread.ident != probe_thread.ident
        assert not probe_acquired.wait(.15), "Another numerical job entered while the old child was running"
        assert request_thread.is_alive() and process.poll() is None

        cancel_request.set()
        request_thread.join(3.)
        assert not request_thread.is_alive(), "Ignored cancellation exceeded cleanup watchdog"
        assert probe_acquired.wait(2.), "CPU admission was not released after terminating the child"
        probe_thread.join(1.)
        assert not probe_thread.is_alive()
        assert len(request_outcomes) == 1
        assert isinstance(request_outcomes[0], ElectronExecutionCancelled)
        assert request_outcomes[0].fields_lost
        assert request_budgets_after == [None]
        assert len(probe_outcomes) == 1 and "error" not in probe_outcomes[0]
        observation = probe_outcomes[0]
        assert observation["exit_code_at_admission"] is not None
        assert observation["budget"] == observation["numba_threads"] == observation["active_budget"] == 1
        assert observation["budget_after"] is None
        assert process.poll() is not None and backend.process_id is None
        assert cpu_resources._ACTIVE_BUDGET.get() is None
    finally:
        cancel_request.set()
        stop_probe.set()
        if request_thread.is_alive():
            request_safety()
        request_thread.join(1.)
        if probe_thread.ident is not None:
            probe_thread.join(1.)
        request_watchdog.cancel()
        probe_watchdog.cancel()


def test_dead_process_poll_without_reader_eof_is_not_an_infinite_wait(monkeypatch, tmp_path):
    """Simulate an inherited output handle keeping EOF from reaching the reader."""
    backend = ElectronExecutionBackend(
        policy=ElectronExecutionPolicy(poll_interval_s=.01), diagnostic_root=tmp_path)
    backend._identity, backend._scene_token = "test-process", "test-scene"
    backend._responses = Queue()
    process = SimpleNamespace(stdin=io.BytesIO(), stdout=io.BytesIO(), stderr=None,
                              pid=-1, poll=lambda: 31, wait=lambda **kwargs: 31,
                              terminate=lambda: None, kill=lambda: None)
    backend._process = process
    monkeypatch.setattr(backend, "_start", lambda: None)
    try:
        with pytest.raises(ElectronExecutionError) as captured:
            bounded_call(backend, lambda: backend._request(("ready",)), timeout=2.)
        assert captured.value.fields_lost
        assert backend.process_id is None
        assert "31" in json.dumps(captured.value.diagnostic)
    finally:
        backend.close()


@pytest.mark.parametrize("response", [None, (), ("bad",), {}, (1,), (1, "ok"),
                                       (1, "ok", "extra", "item"), ("disconnected",),
                                       (True, "ok", "value"), (1.0, "ok", "value"), (1, None, "value")])
def test_malformed_response_shape_is_a_protocol_error_not_a_raw_unpack_exception(monkeypatch, tmp_path, response):
    backend = ElectronExecutionBackend(diagnostic_root=tmp_path)
    backend._responses = Queue()
    backend._responses.put(response)
    ended = [False]
    process = SimpleNamespace(stdin=io.BytesIO(), stdout=io.BytesIO(), stderr=None, pid=-1,
        poll=lambda: -15 if ended[0] else None, wait=lambda **kwargs: -15,
        terminate=lambda: ended.__setitem__(0, True), kill=lambda: ended.__setitem__(0, True))
    backend._process = process
    monkeypatch.setattr(backend, "_start", lambda: None)
    try:
        with pytest.raises(ElectronExecutionError):
            bounded_call(backend, lambda: backend._request(("ready",)), timeout=2.)
        assert ended[0]
    finally:
        backend.close()
