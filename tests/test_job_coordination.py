"""Scheduling/ownership fixtures; no substitute for physical-chain tests."""
from threading import Event, get_ident
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QObject, QRunnable, Signal, QTimer

from temsim.gui.job_coordinator import JobCoordinator, CoordinatedPool, ResourceClaim


class Signals(QObject):
    failed = Signal(str)
    finished = Signal()


class Worker(QRunnable):
    def __init__(self, operation, *, size=1024):
        super().__init__()
        self.operation = operation
        self.cancel_event = Event()
        self.resource_claim = ResourceClaim(size)
        self.signals = Signals()

    def run(self):
        self.operation()
        self.signals.finished.emit()


def test_fifo_across_controllers_keeps_ui_responsive_and_releases_failures(qtbot):
    coordinator = JobCoordinator(ram_budget_bytes=2**30)
    a, b = CoordinatedPool(coordinator=coordinator), CoordinatedPool(coordinator=coordinator)
    entered, release = Event(), Event()
    completed, failures, ticks = [], [], []
    first = Worker(lambda: (entered.set(), release.wait(5), completed.append("first")))
    broken = Worker(lambda: (_ for _ in ()).throw(ValueError("fixture failure")))
    broken.signals.failed.connect(failures.append)
    last = Worker(lambda: completed.append("last"))
    timer = QTimer()
    timer.setInterval(5)
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start()
    try:
        a.start(first)
        b.start(broken)
        a.start(last)
        qtbot.waitUntil(entered.is_set, timeout=10000)
        qtbot.waitUntil(lambda: len(ticks) >= 3)
        stats = coordinator.statistics()
        assert stats["running"] == 1 and stats["queued"] == 2
        assert stats["reserved_working_bytes"] == 1024
        release.set()
        qtbot.waitUntil(lambda: not coordinator.queue and not coordinator.active, timeout=10000)
        assert completed == ["first", "last"]
        assert failures == ["fixture failure"]
        assert [row["status"] for row in coordinator.history] == ["finished", "failed", "finished"]
        assert coordinator.statistics()["reserved_working_bytes"] == 0
    finally:
        release.set()
        timer.stop()
        assert a.waitForDone(10000) and b.waitForDone(10000)


def test_oversized_job_rejects_without_execution_and_cancelled_queue_finishes(qtbot):
    coordinator = JobCoordinator(ram_budget_bytes=2**20)
    pool = CoordinatedPool(coordinator=coordinator)
    ran, errors, finished = [], [], []
    worker = Worker(lambda: ran.append(True), size=2**21)
    worker.signals.failed.connect(errors.append)
    pool.start(worker)
    cancelled = Worker(lambda: ran.append(True))
    cancelled.signals.finished.connect(lambda: finished.append(True))
    pool.start(cancelled)
    cancelled.cancel_event.set()
    qtbot.waitUntil(lambda: not coordinator.queue)
    assert not ran and len(errors) == 1 and finished == [True]
    assert coordinator.statistics()["reserved_working_bytes"] == 0


def test_shared_retained_buffers_are_charged_once(qtbot):
    class Owner:
        def __init__(self, value):
            self.value = value
        def roots(self):
            return self.value
    coordinator = JobCoordinator()
    data = np.zeros(2**20, dtype=np.uint8)
    a, b = Owner(data), Owner(data[::2])
    baseline = coordinator._retained_bytes()
    coordinator.register_retained(a, "roots")
    one = coordinator._retained_bytes()
    coordinator.register_retained(b, "roots")
    two = coordinator._retained_bytes()
    assert one - baseline >= data.nbytes
    assert two - one < 1024


def test_queue_cancellation_allows_finished_callback_to_enqueue(qtbot):
    coordinator = JobCoordinator(ram_budget_bytes=2**30)
    cancelled_pool = CoordinatedPool(coordinator=coordinator)
    retained_pool = CoordinatedPool(coordinator=coordinator)
    order = []
    cancelled = Worker(lambda: pytest.fail("Cancelled worker must not run"))
    retained = Worker(lambda: order.append("already queued"))
    followup = Worker(lambda: order.append("new submission"))
    cancelled.signals.finished.connect(lambda: cancelled_pool.start(followup))
    cancelled_pool.start(cancelled)
    retained_pool.start(retained)
    try:
        cancelled_pool.clear()
        assert [job.worker for job in coordinator.queue] == [retained, followup]
        qtbot.waitUntil(lambda: len(coordinator.history) == 3, timeout=10_000)
        assert order == ["already queued", "new submission"]
        assert [row["status"] for row in coordinator.history] == ["cancelled", "finished", "finished"]
    finally:
        assert cancelled_pool.waitForDone(10_000)
        assert retained_pool.waitForDone(10_000)


def test_live_edits_preserve_high_captured_job_and_reject_old_preview(qtbot, monkeypatch):
    from temsim.gui.calculation_controller import CalculationController
    controller = CalculationController(persistent_cache_enabled=False)
    recorded = []
    monkeypatch.setattr(controller, "_cache_result", lambda result: None)
    monkeypatch.setattr(controller, "_cache_tuning_result", lambda quality, result: None)
    controller.result_ready.connect(lambda quality, result, duration: recorded.append((quality, result)))
    high = controller._begin_request("High accuracy")
    token = controller._requests[high]["cancel"]
    preview = controller._begin_request("Preview")
    stale_token = controller._requests[preview]["cancel"]
    latest = controller._begin_request("Preview")
    assert not token.is_set() and stale_token.is_set()
    result = SimpleNamespace(external_inputs=None)
    controller._accept_result(preview, "Preview", result, 0.)
    controller._accept_result(high, "High accuracy", result, 0.)
    controller._accept_result(latest, "Preview", result, 0.)
    assert [quality for quality, _ in recorded] == ["High accuracy", "Preview"]
    controller.invalidate_pending()
    assert not token.is_set()
    controller.invalidate_pending(include_explicit=True)
    assert token.is_set() and not controller.has_pending_requests


def test_widget_cycles_are_collected_on_gui_thread_while_worker_runs(qtbot):
    import gc
    from temsim.gui.garbage_collection import install_gui_gc
    from PySide6.QtCore import QCoreApplication
    app = QCoreApplication.instance()
    install_gui_gc(app)
    owner_thread = get_ident()
    finalized, release, entered = [], Event(), Event()
    class Cycle:
        def __init__(self):
            self.reference = self
        def __del__(self):
            finalized.append(get_ident())
    coordinator = JobCoordinator(ram_budget_bytes=2**30)
    pool = CoordinatedPool(coordinator=coordinator)
    def operation():
        entered.set()
        for _ in range(3000):
            Cycle()
        release.wait(5)
    pool.start(Worker(operation))
    try:
        qtbot.waitUntil(entered.is_set, timeout=10000)
        qtbot.waitUntil(lambda: bool(finalized), timeout=10000)
        assert set(finalized) == {owner_thread}
    finally:
        release.set()
        assert pool.waitForDone(10000)


def test_handled_worker_error_is_failed_without_duplicate_notifications(qtbot):
    coordinator = JobCoordinator(ram_budget_bytes=2**30)
    pool = CoordinatedPool(coordinator=coordinator)
    worker = Worker(lambda: None)
    messages = []
    worker.signals.failed.connect(messages.append)
    def handled():
        worker.signals.failed.emit('handled numerical failure')
        worker.signals.finished.emit()
    worker.run = handled
    pool.start(worker)
    qtbot.waitUntil(lambda: bool(coordinator.history), timeout=10000)
    assert coordinator.history[-1]['status'] == 'failed'
    assert messages == ['handled numerical failure']
    assert coordinator.statistics()['reserved_working_bytes'] == 0
