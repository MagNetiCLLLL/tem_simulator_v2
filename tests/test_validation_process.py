"""Synthetic process-cleanup failures; these fixtures never launch or kill a PID."""
import io
import subprocess
from types import SimpleNamespace

import pytest

from temsim import validation_process as module


class Child:
    def __init__(self, pid=4101, code=None, *, direct_stops=True, on_wait=None):
        self.pid, self.returncode = pid, code
        self.direct_stops, self.on_wait = direct_stops, on_wait
        self.waits, self.actions = [], []
        self.stdin = self.stdout = self.stderr = None

    def __enter__(self):
        raise AssertionError("Popen context management may hide an unbounded cleanup wait")

    def __exit__(self, *_args):
        raise AssertionError("Popen.__exit__ must never perform an unbounded wait")

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        assert timeout is not None and timeout >= 0, "Every wait must be bounded"
        self.waits.append(timeout)
        if self.on_wait is not None:
            self.on_wait(self)
        if self.returncode is None:
            raise subprocess.TimeoutExpired(["synthetic", str(self.pid)], timeout)
        return self.returncode

    def terminate(self):
        self.actions.append("terminate")
        if self.direct_stops:
            self.returncode = -15

    def kill(self):
        self.actions.append("kill")
        if self.direct_stops:
            self.returncode = -9


@pytest.fixture(autouse=True)
def isolated_cleanup_registry(monkeypatch):
    monkeypatch.setattr(module, "_FAILED_CLEANUPS", [])


def windows(monkeypatch, child, *, utility_code=0, utility_error=None, utility_stubborn=False):
    calls, utilities = [], []

    def popen(command, **options):
        calls.append((command, options))
        if command[0] != "taskkill":
            return child
        if utility_error is not None:
            raise utility_error

        def finish_tree(utility):
            if utility.returncode == 0:
                child.returncode = -9

        utility = Child(pid=5101, code=utility_code,
                        direct_stops=not utility_stubborn, on_wait=finish_tree)
        utilities.append(utility)
        return utility

    monkeypatch.setattr(module, "os", SimpleNamespace(name="nt"))
    proxy = SimpleNamespace(Popen=popen, CREATE_NEW_PROCESS_GROUP=512, CREATE_NO_WINDOW=0x08000000,
        DEVNULL=subprocess.DEVNULL, PIPE=subprocess.PIPE, CompletedProcess=subprocess.CompletedProcess,
        TimeoutExpired=subprocess.TimeoutExpired, SubprocessError=subprocess.SubprocessError)
    monkeypatch.setattr(module, "subprocess", proxy)
    return calls, utilities


def unix(monkeypatch, child, killpg):
    calls = []

    def popen(command, **options):
        calls.append((command, options))
        return child

    monkeypatch.setattr(module, "os", SimpleNamespace(name="posix", killpg=killpg))
    monkeypatch.setattr(module, "signal", SimpleNamespace(SIGKILL=9))
    monkeypatch.setattr(module, "subprocess", SimpleNamespace(Popen=popen,
        PIPE=subprocess.PIPE, CompletedProcess=subprocess.CompletedProcess,
        TimeoutExpired=subprocess.TimeoutExpired, SubprocessError=subprocess.SubprocessError))
    return calls


def test_normal_exit_keeps_completed_process_contract_and_callers_log(monkeypatch):
    child = Child(code=7)
    log = io.StringIO()
    child.stdout = log  # An intentionally strict ownership fixture.
    calls, _ = windows(monkeypatch, child)
    command = ["fixture-validation"]
    result = module.run_bounded(command, timeout=1., stdout=log)
    assert result.args == command and result.returncode == 7
    assert not log.closed and len(calls) == 1
    assert child.waits == [1.]


def test_owned_pipe_is_closed_after_normal_exit(monkeypatch):
    child = Child(code=0)
    child.stdout = io.BytesIO()
    windows(monkeypatch, child)
    module.run_bounded(["fixture"], timeout=1., stdout=subprocess.PIPE)
    assert child.stdout.closed


def test_windows_timeout_kills_exact_owned_tree_and_returns_124(monkeypatch):
    child = Child()
    calls, utilities = windows(monkeypatch, child)
    result = module.run_bounded(["fixture-validation"], timeout=.25)
    assert result.returncode == 124 and child.poll() is not None
    assert calls[0][1]["creationflags"] == 512
    assert calls[1][0] == ["taskkill", "/PID", "4101", "/T", "/F"]
    assert calls[1][1]["creationflags"] == 0x08000000
    assert all(value is not None for value in child.waits + utilities[0].waits)
    assert not module._FAILED_CLEANUPS


@pytest.mark.parametrize("kind", ["launch_error", "nonzero", "utility_timeout"])
def test_taskkill_failure_is_bounded_and_blocks_next_validation_even_after_root_exit(monkeypatch, kind):
    child = Child()
    kwargs = ({"utility_error": OSError("taskkill unavailable")} if kind == "launch_error"
              else {"utility_code": 5} if kind == "nonzero" else {"utility_code": None})
    calls, utilities = windows(monkeypatch, child, **kwargs)
    with pytest.raises(module.ValidationCleanupError, match="4101") as failure:
        module.run_bounded(["fixture-validation"], timeout=.25)
    assert not failure.value.tree_termination_confirmed
    assert child.poll() is not None  # Direct child stopped; tree is still unconfirmed.
    spawned = len(calls)
    with pytest.raises(module.ValidationCleanupError):
        module.run_bounded(["another-validation"], timeout=.25)
    assert len(calls) == spawned
    assert all(value is not None for value in child.waits)
    assert all(value is not None for utility in utilities for value in utility.waits)


def test_successful_tree_kill_with_unreaped_child_never_returns_false_124(monkeypatch):
    child = Child(direct_stops=False)
    calls, _ = windows(monkeypatch, child)
    # Tree command reports success but the exact owned handle stays alive.
    monkeypatch.setattr(module, "_taskkill_tree", lambda pid, deadline: 0)
    with pytest.raises(module.ValidationCleanupError) as failure:
        module.run_bounded(["fixture"], timeout=.25)
    assert failure.value.tree_termination_confirmed
    assert child.actions == ["terminate", "kill"]
    assert child.poll() is None
    assert all(value <= 15. for value in child.waits)
    with pytest.raises(module.ValidationCleanupError):
        module.run_bounded(["next"], timeout=.25)
    assert len(calls) == 1


def test_confirmed_tree_late_reap_allows_new_validation_without_signalling_old_pid(monkeypatch):
    child = Child(direct_stops=False)
    calls, _ = windows(monkeypatch, child)
    killed = []
    monkeypatch.setattr(module, "_taskkill_tree", lambda pid, deadline: killed.append(pid) or 0)
    with pytest.raises(module.ValidationCleanupError):
        module.run_bounded(["fixture"], timeout=.25)
    child.returncode = -9  # OS later confirms that this owned process has ended.
    result = module.run_bounded(["next"], timeout=.25)
    assert result.returncode == -9
    assert len(calls) == 2 and killed == [child.pid]
    assert not module._FAILED_CLEANUPS


def test_taskkill_utility_itself_can_fail_to_stop_without_hidden_context_wait(monkeypatch):
    child = Child(direct_stops=False)
    calls, utilities = windows(monkeypatch, child, utility_code=None, utility_stubborn=True)
    with pytest.raises(module.ValidationCleanupError):
        module.run_bounded(["fixture"], timeout=.25)
    assert len(module._FAILED_CLEANUPS) == 2
    assert utilities[0].actions == ["terminate", "kill"]
    with pytest.raises(module.ValidationCleanupError, match="5101"):
        module.run_bounded(["next"], timeout=.25)
    assert len(calls) == 2


def test_cleanup_deadline_is_shared_and_external_log_survives_failure(monkeypatch):
    child = Child(direct_stops=False)
    log = io.StringIO()
    child.stdout = log
    windows(monkeypatch, child)
    clock = [100.]
    monkeypatch.setattr(module, "monotonic", lambda: clock[0])

    def consume_deadline(pid, deadline):
        assert pid == child.pid and deadline == 115.
        clock[0] = deadline
        raise subprocess.TimeoutExpired(["taskkill"], 15.)

    monkeypatch.setattr(module, "_taskkill_tree", consume_deadline)
    with pytest.raises(module.ValidationCleanupError, match="4101"):
        module.run_bounded(["fixture"], timeout=.25, stdout=log)
    assert child.waits == [.25, 0., 0., 0.]
    assert not log.closed
    assert child.actions == ["terminate", "kill"]


def test_unix_permission_failure_is_explicit_and_never_waits_unbounded(monkeypatch):
    child = Child()
    signalled = []

    def killpg(pid, sig):
        signalled.append((pid, sig))
        raise PermissionError("synthetic denied group termination")

    calls = unix(monkeypatch, child, killpg)
    with pytest.raises(module.ValidationCleanupError, match="denied group termination"):
        module.run_bounded(["fixture"], timeout=.25)
    assert signalled == [(child.pid, 9)]
    assert calls[0][1]["start_new_session"] is True
    assert child.poll() is not None
    with pytest.raises(module.ValidationCleanupError):
        module.run_bounded(["next"], timeout=.25)
    assert len(calls) == 1


@pytest.mark.parametrize("already_gone", [False, True])
def test_unix_confirmed_group_cleanup_returns_timeout_code(monkeypatch, already_gone):
    child = Child()

    def killpg(pid, sig):
        assert pid == child.pid and sig == 9
        child.returncode = -9
        if already_gone:
            raise ProcessLookupError("group already finished")

    unix(monkeypatch, child, killpg)
    assert module.run_bounded(["fixture"], timeout=.25).returncode == 124
    assert not module._FAILED_CLEANUPS


@pytest.mark.parametrize("timeout", [None, 0, -1, float("nan"), float("inf"), True])
def test_unbounded_or_invalid_request_is_rejected_before_spawning(monkeypatch, timeout):
    calls, _ = windows(monkeypatch, Child())
    with pytest.raises(ValueError, match="finite and positive"):
        module.run_bounded(["fixture"], timeout=timeout)
    assert not calls
