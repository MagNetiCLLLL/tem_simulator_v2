"""Transactional worker for the independent Interactive Calculation page."""
from threading import Event

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from temsim.interactive_calculation import build_bank, read_bank


class _Signals(QObject):
    progress = Signal(int, int, int, int, str)
    result = Signal(int, str, object)
    failed = Signal(int, str)
    finished = Signal(int)


class _Worker(QRunnable):
    def __init__(self, generation, kind, operation, event):
        super().__init__()
        self.generation, self.kind = generation, kind
        self.operation, self.event = operation, event
        self.signals = _Signals()

    def run(self):
        try:
            result = self.operation(self.event.is_set, self.signals.progress.emit)
            if not self.event.is_set():
                self.signals.result.emit(self.generation, self.kind, result)
        except Exception as exc:
            if not self.event.is_set():
                self.signals.failed.emit(self.generation, str(exc))
        finally:
            self.signals.finished.emit(self.generation)


class InteractiveController(QObject):
    busy_changed = Signal(bool)
    progress = Signal(int, int, int, int, str)
    bank_ready = Signal(object)
    readout_ready = Signal(object)
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.bank = None
        self.busy = False
        self._generation = 0
        self._event = Event()
        self._kind = ""
        self._pending_readout = None

    def _start(self, kind, operation):
        if self.busy:
            raise RuntimeError("Wait for the current interactive operation")
        self._generation += 1
        self._event = Event()
        self._kind = kind
        self.busy = True
        worker = _Worker(self._generation, kind, operation, self._event)
        worker.signals.progress.connect(self.progress)
        worker.signals.result.connect(self._result)
        worker.signals.failed.connect(self._failed)
        worker.signals.finished.connect(self._finished)
        self.busy_changed.emit(True)
        self.pool.start(worker)

    def build(self, state, plan, seeds=()):
        self._pending_readout = None
        self._start("build", lambda cancel, progress: build_bank(
            state, plan, seeds=seeds, retained_roots=(self.bank,), progress=progress, cancelled=cancel))

    def read(self, coordinates):
        if self.bank is None:
            raise ValueError("Build a complete bank first")
        if self.busy:
            if self._kind != "read":
                raise RuntimeError("Wait for bank construction to finish")
            self._pending_readout = dict(coordinates)
            self._event.set()
            return
        self._start("read", lambda cancel, progress: read_bank(self.bank, coordinates, cancelled=cancel))

    def cancel(self):
        self._pending_readout = None
        self._event.set()

    def _result(self, generation, kind, result):
        if generation != self._generation or self._event.is_set():
            return
        if kind == "build":
            self.bank = result
            self.bank_ready.emit(result)
        else:
            self.readout_ready.emit(result)

    def _failed(self, generation, message):
        if generation == self._generation:
            self.failed.emit(message)

    def _finished(self, generation):
        if generation != self._generation:
            return
        pending = self._pending_readout
        self._pending_readout = None
        self.busy = False
        self.busy_changed.emit(False)
        if pending is not None:
            self.read(pending)

    def shutdown(self):
        self.cancel()
        self._generation += 1
        self.pool.clear()
        self.pool.waitForDone(3000)
