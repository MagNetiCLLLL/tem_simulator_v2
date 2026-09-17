"""Deterministic ownership/transition barriers; these are software fixtures."""
from threading import Event
from types import SimpleNamespace

import pytest

from temsim.gui.calculation_controller import PreparationWorker, CalculationController
from temsim.gui.calculation_request import PreparationCancelled
from temsim.gui.job_coordinator import JobCoordinator, CoordinatedPool, ResourceClaim
from temsim.job_events import JobEvents


class Lease:
    def __init__(self):
        self.calls = 0
    def close(self):
        self.calls += 1


def make_worker(operation, *, generation=1, quality="Preview"):
    lease = Lease()
    request = SimpleNamespace(quality=quality, prepare=operation, _input_assets=lease)
    worker = PreparationWorker(generation, request, Event())
    worker.job_input_identity = "captured-fixture"
    worker.request_id = "request-fixture"
    return worker, lease


def wait_entry(qtbot, event, coordinator):
    try:
        qtbot.waitUntil(event.is_set, timeout=5000)
    except Exception as exc:
        # Real timeout evidence includes all outstanding owners and stacks;
        # the five-second threshold is intentionally unchanged.
        raise AssertionError(coordinator.diagnostic_snapshot(include_stacks=True)) from exc


@pytest.mark.parametrize("scenario", ["success", "failure", "cancel_preparation", "rejection", "queued_cancel", "closure"])
def test_one_terminal_one_release_every_preparation_exit(qtbot, scenario):
    coordinator = JobCoordinator(ram_budget_bytes=2**30)
    pool = CoordinatedPool(coordinator=coordinator)
    entered, release = Event(), Event()
    published, errors, finished = [], [], []
    def prepare(cancel):
        entered.set()
        assert release.wait(10), "test barrier not released"
        if cancel.is_set():
            raise PreparationCancelled()
        if scenario == "failure":
            raise ValueError("controlled preparation failure")
        return "prepared fixture"
    worker, lease = make_worker(prepare)
    worker.signals.prepared.connect(lambda *args: published.append(args))
    worker.signals.error.connect(lambda *args: errors.append(args))
    worker.signals.finished.connect(lambda *args: finished.append(args))
    if scenario == "rejection":
        worker.resource_claim = ResourceClaim(2**31)
    try:
        identifier = pool.start(worker)
        if scenario == "queued_cancel":
            pool.clear()
        elif scenario != "rejection":
            wait_entry(qtbot, entered, coordinator)
            if scenario in {"closure", "cancel_preparation"}:
                pool.clear()
                assert coordinator.statistics()["reserved_working_bytes"] > 0
            release.set()
        qtbot.waitUntil(lambda: len(coordinator.history) == 1, timeout=10000)
        coordinator._done(identifier, None)  # A duplicate late delivery is inert.
        rows = [row for row in coordinator.events.snapshot() if row.get("job_id") == identifier]
        assert sum(row["event"] == "terminal" for row in rows) == 1
        assert sum(row["event"] == "cleanup" for row in rows) == 1
        assert lease.calls == 1
        assert not coordinator.has_owner(pool)
        assert coordinator.statistics()["reserved_working_bytes"] == 0
        assert bool(published) == (scenario == "success")
        assert bool(errors) == (scenario in {"failure", "rejection"})
        assert len(finished) == (0 if scenario == "success" else 1)
        if entered.is_set():
            names = [row["event"] for row in rows]
            assert names.index("enqueue") < names.index("admission") < names.index("worker_entry") < names.index("terminal")
            assert next(row for row in rows if row["event"] == "worker_entry")["thread_id"] != rows[0]["thread_id"]
    finally:
        release.set()
        pool.clear()
        assert pool.waitForDone(10000)


def test_small_preparation_does_not_reserve_whole_solver_budget(qtbot):
    coordinator = JobCoordinator(ram_budget_bytes=512*1024**2)
    pool = CoordinatedPool(coordinator=coordinator)
    worker, lease = make_worker(lambda cancel: None)
    assert worker.resource_claim.working_bytes < 512*1024**2
    pool.start(worker)
    qtbot.waitUntil(lambda: bool(coordinator.history), timeout=10000)
    assert coordinator.history[-1]["status"] == "finished"
    assert lease.calls == 1


@pytest.mark.parametrize("attempt", range(5))
def test_warm_fifo_high_and_independent_owner_survive_live_cancellation(qtbot, attempt):
    coordinator = JobCoordinator(ram_budget_bytes=2**30)
    live = CoordinatedPool(coordinator=coordinator)
    detached = CoordinatedPool(coordinator=coordinator)
    entered, release = Event(), Event()
    order = []
    first, first_lease = make_worker(lambda cancel: (entered.set(), release.wait(10)))
    high, high_lease = make_worker(lambda cancel: order.append("high"), generation=2, quality="High accuracy")
    other, other_lease = make_worker(lambda cancel: order.append("experiment"), generation=3, quality="Study")
    try:
        live.start(first)
        wait_entry(qtbot, entered, coordinator)
        live.start(high)
        detached.start(other)
        live.clear(lambda worker: worker.quality != "High accuracy")
        assert not high.cancel_event.is_set() and not other.cancel_event.is_set()
        release.set()
        qtbot.waitUntil(lambda: len(coordinator.history) == 3, timeout=10000)
        assert order == ["high", "experiment"]
        assert [first_lease.calls, high_lease.calls, other_lease.calls] == [1, 1, 1]
        assert [row["status"] for row in coordinator.history] == ["cancelled", "finished", "finished"]
    finally:
        release.set()
        live.clear()
        detached.clear()
        assert live.waitForDone(10000) and detached.waitForDone(10000)


def test_finished_controller_request_cannot_publish_twice(qtbot, monkeypatch):
    controller = CalculationController(persistent_cache_enabled=False)
    generation = controller._begin_request("Preview")
    seen = []
    monkeypatch.setattr(controller, "_cache_tuning_result", lambda *args: None)
    controller.result_ready.connect(lambda *args: seen.append(args))
    result = SimpleNamespace(external_inputs=None)
    controller._accept_result(generation, "Preview", result, 0.)
    controller._accept_finished(generation, "Preview")
    controller._accept_result(generation, "Preview", result, 0.)
    controller._accept_finished(generation, "Preview")
    assert len(seen) == 1
    rows = controller.pool.coordinator.events.snapshot()
    assert sum(row["event"] == "request_terminal" and row.get("generation") == generation
               and row.get("owner") == "CalculationController" for row in rows) >= 1


def test_trace_is_bounded_scalar_evidence():
    trace = JobEvents(maximum=3)
    for i in range(5):
        trace.record("fixture", index=i)
    assert [row["index"] for row in trace.snapshot()] == [2, 3, 4]
    with pytest.raises(TypeError):
        trace.record("bad", payload=object())


def test_admission_precedes_already_posted_display_cleanup(qtbot):
    """A zero timer must not wait behind arbitrary queued plot disposal."""
    from PySide6.QtCore import QCoreApplication, QObject, QEvent
    coordinator = JobCoordinator(ram_budget_bytes=2**30)
    pool = CoordinatedPool(coordinator=coordinator)
    observed = []
    event_type = QEvent.Type(QEvent.registerEventType())
    class DisplayCleanup(QObject):
        def event(self, event):
            if event.type() == event_type:
                observed.append(any(row["event"] == "admission" for row in coordinator.events.snapshot()))
                return True
            return super().event(event)
    cleanup = DisplayCleanup(coordinator)
    QCoreApplication.postEvent(cleanup, QEvent(event_type))
    worker, lease = make_worker(lambda cancel: None)
    pool.start(worker)
    try:
        qtbot.waitUntil(lambda: bool(observed), timeout=5000)
        assert observed == [True]
    finally:
        assert pool.waitForDone(10000)
    assert lease.calls == 1


def test_dispatch_owner_survives_python_collection_until_qt_disposal(qtbot):
    import gc
    import weakref
    from PySide6.QtCore import QCoreApplication, QEvent
    coordinator = JobCoordinator(ram_budget_bytes=2**30)
    reference = weakref.ref(coordinator)
    assert coordinator.parent() is QCoreApplication.instance()
    pool = CoordinatedPool(coordinator=coordinator)
    worker, lease = make_worker(lambda cancel: None)
    pool.start(worker)
    qtbot.waitUntil(lambda: bool(coordinator.history), timeout=5000)
    del worker, pool, coordinator
    gc.collect()
    assert reference() is not None  # Native pool/queued receivers have a Qt owner.
    assert lease.calls == 1
    reference().deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    gc.collect()
    from shiboken6 import isValid
    assert reference() is None or not isValid(reference())
