"""Admission after failed process cleanup; scripted process objects, no physics."""
import io
import subprocess
from threading import Event, Thread

import pytest

from temsim import cpu_resources as resources
from temsim.test_electron_execution import ElectronExecutionBackend


class UnreapedProcess:
    def __init__(self, pid):
        self.pid = pid
        self.returncode = None
        self.poll_error = None
        self.actions = []
        self.stdin, self.stdout, self.stderr = io.BytesIO(), io.BytesIO(), None

    def poll(self):
        if self.poll_error is not None:
            raise self.poll_error
        return self.returncode

    def terminate(self):
        self.actions.append("terminate")

    def kill(self):
        self.actions.append("kill")

    def wait(self, *, timeout):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("scripted-unreaped-process", timeout)
        return self.returncode


@pytest.fixture
def isolated_unreaped_registry(monkeypatch):
    # No real child is created here. Preserve any pre-existing registry for
    # other tests rather than clearing an actual process owner's evidence.
    registry = []
    monkeypatch.setattr(resources, "_UNREAPED_NUMERICAL_PROCESSES", registry)
    return registry


def admission_from_other_live_thread():
    """Bound lock regressions while the previous (main) owner is still alive."""
    stop = Event()
    outcomes = []

    def run():
        observation = {}
        try:
            with resources.numerical_job(1, cancelled=stop.is_set) as receipt:
                observation["budget"] = receipt.numerical_thread_budget
                observation["active_budget"] = resources._ACTIVE_BUDGET.get()
        except BaseException as error:
            observation["error"] = error
        finally:
            observation["budget_after"] = resources._ACTIVE_BUDGET.get()
            outcomes.append(observation)

    worker = Thread(target=run, name="unreaped-process-admission-probe", daemon=True)
    worker.start()
    worker.join(3.)
    timed_out = worker.is_alive()
    if timed_out:
        stop.set()
        worker.join(1.)
    assert not timed_out, "Admission leaked the numerical lock instead of returning a bounded result"
    assert not worker.is_alive() and len(outcomes) == 1
    return outcomes[0]


def test_failed_terminate_and_kill_blocks_other_jobs_until_confirmed_dead(tmp_path, isolated_unreaped_registry):
    process = UnreapedProcess(81234)
    backend = ElectronExecutionBackend(diagnostic_root=tmp_path)
    backend._process = process
    backend._identity, backend._scene_token = "scripted-owner", "scripted-fields"
    try:
        with resources.numerical_job(1):
            assert backend._stop_process() is None
            assert backend._process is process
            assert backend._cleanup_error
            assert resources._ACTIVE_BUDGET.get() == 1
        assert process.actions == ["terminate", "kill"]
        assert isolated_unreaped_registry == [process]
        assert resources._ACTIVE_BUDGET.get() is None

        blocked = admission_from_other_live_thread()
        assert isinstance(blocked.get("error"), RuntimeError)
        assert "81234" in str(blocked["error"])
        assert "budget" not in blocked and blocked["budget_after"] is None
        assert backend._process is process

        process.returncode = -9
        admitted = admission_from_other_live_thread()
        assert "error" not in admitted
        assert admitted["budget"] == admitted["active_budget"] == 1
        assert admitted["budget_after"] is None
        assert isolated_unreaped_registry == []
    finally:
        process.returncode = -9
        backend.close()


def test_all_unreaped_owners_must_stop_and_duplicate_registration_is_idempotent(isolated_unreaped_registry):
    first, second = UnreapedProcess(101), UnreapedProcess(202)
    resources.retain_unreaped_numerical_process(first)
    resources.retain_unreaped_numerical_process(first)
    resources.retain_unreaped_numerical_process(second)
    assert isolated_unreaped_registry == [first, second]
    first.returncode = 0
    with pytest.raises(RuntimeError, match="PID 202"):
        with resources.numerical_job(1):
            pytest.fail("A still-running numerical child must block admission")
    assert isolated_unreaped_registry == [second]
    assert resources._ACTIVE_BUDGET.get() is None
    second.returncode = 0
    admitted = admission_from_other_live_thread()
    assert "error" not in admitted and admitted["budget"] == 1
    assert admitted["budget_after"] is None
    assert isolated_unreaped_registry == []


def test_poll_failure_remains_blocked_without_entering_numerical_libraries(monkeypatch, isolated_unreaped_registry):
    process = UnreapedProcess(303)
    process.poll_error = OSError("Scripted process status unavailable")
    resources.retain_unreaped_numerical_process(process)
    monkeypatch.setattr(resources, "numerical_thread_budget",
                        lambda *args: pytest.fail("A blocked job must not initialize numerical work"))
    with pytest.raises(RuntimeError, match="PID 303"):
        with resources.numerical_job(1):
            pytest.fail("An unknown process state must not be treated as dead")
    assert isolated_unreaped_registry == [process]
    assert resources._ACTIVE_BUDGET.get() is None


def test_registration_does_not_need_the_numerical_lease(isolated_unreaped_registry):
    process = UnreapedProcess(404)
    registered = Event()
    with resources.numerical_job(1):
        def register():
            resources.retain_unreaped_numerical_process(process)
            registered.set()
        worker = Thread(target=register, name="unreaped-cleanup-registration", daemon=True)
        worker.start()
        registered_while_admitted = registered.wait(1.)
    # Always release the lease and join before assertions, including a broken
    # implementation that accidentally used the numerical lock to register.
    worker.join(1.)
    assert registered_while_admitted, "Cleanup registration must not deadlock against the active request"
    assert not worker.is_alive()
    assert isolated_unreaped_registry == [process]
    process.returncode = 0
    admitted = admission_from_other_live_thread()
    assert "error" not in admitted and admitted["budget_after"] is None


def test_invalid_process_cannot_silently_disable_admission_guard(isolated_unreaped_registry):
    with pytest.raises(TypeError, match="poll"):
        resources.retain_unreaped_numerical_process(object())
    assert isolated_unreaped_registry == []
