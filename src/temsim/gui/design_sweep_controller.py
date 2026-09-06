"""Single-worker Qt controller for detached Design Explorer sweeps."""

from __future__ import annotations

from threading import Event
from time import perf_counter

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from temsim.design_sweep_execution import (
    SweepCalculationCache,
    execute_parameter_sweep,
)


class _SweepWorkerSignals(QObject):
    progress = Signal(int, object)
    result = Signal(int, object, float)
    failed = Signal(int, str)
    finished = Signal(int)


class _SweepWorker(QRunnable):
    def __init__(
        self,
        generation,
        recipe,
        sweep,
        *,
        catalog,
        artifact_store,
        calculation_cache,
        tolerance_rules,
        cancel_event,
    ) -> None:
        super().__init__()
        self.generation = int(generation)
        self.recipe = recipe
        self.sweep = sweep
        self.catalog = catalog
        self.artifact_store = artifact_store
        self.calculation_cache = calculation_cache
        self.tolerance_rules = tuple(tolerance_rules)
        self.cancel_event = cancel_event
        self.signals = _SweepWorkerSignals()

    def run(self) -> None:
        started = perf_counter()
        try:
            result = execute_parameter_sweep(
                self.recipe,
                self.sweep,
                catalog=self.catalog,
                artifact_store=self.artifact_store,
                calculation_cache=self.calculation_cache,
                tolerance_rules=self.tolerance_rules,
                progress_callback=lambda row: self.signals.progress.emit(
                    self.generation, row
                ),
                cancel_requested=self.cancel_event.is_set,
            )
            self.signals.result.emit(
                self.generation, result, perf_counter() - started
            )
        except Exception as exc:
            self.signals.failed.emit(self.generation, str(exc))
        finally:
            self.signals.finished.emit(self.generation)


class DesignSweepController(QObject):
    """Run one bounded sweep without superseding main-window calculations."""

    started = Signal(int)
    progress_changed = Signal(object)
    result_ready = Signal(object, float)
    failed = Signal(str)
    finished = Signal()

    def __init__(
        self,
        parent=None,
        *,
        catalog,
        artifact_store=None,
    ) -> None:
        super().__init__(parent)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self._catalog = catalog
        self._artifact_store = artifact_store
        # One rolling result is sufficient for adjacent-point dependency
        # reuse and avoids retaining a second potentially multi-GiB bundle.
        self._cache = SweepCalculationCache(maximum_results=1)
        self._cancel_event = Event()
        self._generation = 0
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def set_catalog(self, catalog) -> None:
        if self._running:
            raise RuntimeError("Cannot replace the catalog during a sweep")
        self._catalog = catalog
        self._cache.clear()

    def seed_completed_results(self, results) -> None:
        """Offer existing main-window results to the scoped sweep cache."""

        if self._running:
            return
        for result in tuple(results):
            self._cache.put(result)

    def submit(self, recipe, sweep, *, tolerance_rules=()) -> None:
        if self._running:
            raise RuntimeError("A design sweep is already running")
        self._generation += 1
        generation = self._generation
        self._cancel_event = Event()
        self._running = True
        worker = _SweepWorker(
            generation,
            recipe,
            sweep,
            catalog=self._catalog,
            artifact_store=self._artifact_store,
            calculation_cache=self._cache,
            tolerance_rules=tuple(tolerance_rules),
            cancel_event=self._cancel_event,
        )
        worker.signals.progress.connect(self._accept_progress)
        worker.signals.result.connect(self._accept_result)
        worker.signals.failed.connect(self._accept_failure)
        worker.signals.finished.connect(self._accept_finished)
        self.started.emit(len(sweep.points))
        self.pool.start(worker)

    def cancel(self) -> None:
        if self._running:
            self._cancel_event.set()

    def invalidate_pending(self) -> None:
        self._cancel_event.set()
        self._generation += 1
        self._running = False
        self.pool.clear()

    def _accept_progress(self, generation, progress) -> None:
        if int(generation) == self._generation:
            self.progress_changed.emit(progress)

    def _accept_result(self, generation, result, duration) -> None:
        if int(generation) == self._generation:
            self.result_ready.emit(result, float(duration))

    def _accept_failure(self, generation, message) -> None:
        if int(generation) == self._generation:
            self.failed.emit(str(message))

    def _accept_finished(self, generation) -> None:
        if int(generation) == self._generation:
            self._running = False
            self.finished.emit()


__all__ = ("DesignSweepController",)
