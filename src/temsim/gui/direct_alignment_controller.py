"""Single-worker background controller for coupled Direct Alignment solves."""

from __future__ import annotations

from time import perf_counter
from threading import Event

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from temsim.alignment_transaction import AlignmentRequest, solve_alignment_candidate


class DirectAlignmentWorkerSignals(QObject):
    result = Signal(int, str, object, float)
    error = Signal(int, str, str)
    finished = Signal(int, str)


class DirectAlignmentWorker(QRunnable):
    def __init__(self, generation: int, request, cancellation) -> None:
        super().__init__()
        self.generation = int(generation)
        self.key = request.key
        self.request = request
        self.cancellation = cancellation
        self.signals = DirectAlignmentWorkerSignals()

    def run(self) -> None:
        started = perf_counter()
        try:
            result = solve_alignment_candidate(self.request, cancelled=self.cancellation.is_set)
            self.signals.result.emit(
                self.generation,
                self.key,
                result,
                perf_counter() - started,
            )
        except Exception as exc:
            self.signals.error.emit(
                self.generation, self.key, str(exc)
            )
        finally:
            self.signals.finished.emit(self.generation, self.key)


class DirectAlignmentController(QObject):
    """Solve on an independent state snapshot and reject stale generations."""

    started = Signal(str, float)
    result_ready = Signal(str, object, float)
    failed = Signal(str, str)
    finished = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self._generation = 0
        self._cancellation = Event()

    def submit(self, state, key: str, target: float, *, revision=0) -> None:
        # Failed capture must not cancel a previous valid request.
        request = AlignmentRequest.capture(state, key, target, revision=revision)
        self._cancellation.set()
        self._cancellation = Event()
        self._generation += 1
        generation = self._generation
        self.pool.clear()
        worker = DirectAlignmentWorker(
            generation, request, self._cancellation
        )
        worker.signals.result.connect(self._accept_result)
        worker.signals.error.connect(self._accept_error)
        worker.signals.finished.connect(self._accept_finished)
        self.pool.start(worker)
        self.started.emit(str(key), float(target))

    def invalidate_pending(self) -> None:
        """Ignore queued/running results after another state edit."""

        self._generation += 1
        self._cancellation.set()
        self.pool.clear()

    def _accept_result(
        self, generation: int, key: str, result, duration: float
    ) -> None:
        if generation == self._generation:
            self.result_ready.emit(key, result, duration)

    def _accept_error(
        self, generation: int, key: str, message: str
    ) -> None:
        if generation == self._generation:
            self.failed.emit(key, message)

    def _accept_finished(self, generation: int, key: str) -> None:
        if generation == self._generation:
            self.finished.emit(key)
