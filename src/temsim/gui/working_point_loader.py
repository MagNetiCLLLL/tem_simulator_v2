"""Deferred archive verification, isolated from live instrument inputs."""
from dataclasses import replace
from PySide6.QtCore import QObject, QRunnable, Signal


class Signals(QObject):
    ready = Signal(object)
    failed = Signal(str)
    finished = Signal()


class ArchiveLoader(QRunnable):
    def __init__(self, record, cancelled, *, portable_inputs=False, maximum_unpacked_bytes):
        super().__init__()
        from temsim.working_point import WorkingPointArchiveIndex
        self.maximum_unpacked_bytes = int(maximum_unpacked_bytes)
        unpacked = 0
        if isinstance(record, WorkingPointArchiveIndex):
            unpacked = record.unpacked_size_bytes
            if unpacked > self.maximum_unpacked_bytes:
                raise ValueError("Working-point package exceeds the configured memory budget")
            # The validated package size is also the load cap: a larger file
            # replacement cannot allocate beyond this job's reservation.
            record = replace(record, maximum_unpacked_bytes=unpacked)
        self.record, self.cancelled = record, cancelled
        self.job_input_identity = record.digest
        from temsim.gui.job_coordinator import ResourceClaim
        # The package's bounded unpacked size accounts for pending array load;
        # portable input capture adds bounded JSON/hex and verification copies.
        self.resource_claim = ResourceClaim(max(512*1024**2, 2*unpacked))
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
