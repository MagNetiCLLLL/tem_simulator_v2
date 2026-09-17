"""Deferred archive verification, isolated from live instrument inputs."""
from PySide6.QtCore import QObject, QRunnable, Signal


class Signals(QObject):
    ready = Signal(object)
    failed = Signal(str)
    finished = Signal()


class ArchiveLoader(QRunnable):
    def __init__(self, record, cancelled, *, portable_inputs=False):
        super().__init__()
        self.record, self.cancelled = record, cancelled
        self.job_input_identity = record.digest
        from temsim.gui.job_coordinator import ResourceClaim
        # The package's bounded unpacked size accounts for pending array load;
        # portable input capture adds bounded JSON/hex and verification copies.
        budget = getattr(record, "maximum_unpacked_bytes", 0)
        self.resource_claim = ResourceClaim(max(512*1024**2, int(budget)))
        self.portable_inputs = portable_inputs
        self.signals = Signals()

    def run(self):
        try:
            if self.cancelled.is_set():
                return
            if self.portable_inputs:
                from temsim.working_point_export import make_portable_inputs
                point = make_portable_inputs(self.record)
            else:
                point = self.record.load()
            if not self.cancelled.is_set():
                self.signals.ready.emit(point)
        except Exception as exc:
            if not self.cancelled.is_set():
                self.signals.failed.emit(str(exc))
        finally:
            self.signals.finished.emit()
