"""Background assembly/preset calculations on isolated microscope states."""

from __future__ import annotations

from time import perf_counter
from threading import Event
from temsim import input_io

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.operating_modes import apply_operating_mode_pair, compatible_modes


class _Signals(QObject):
    result = Signal(int, object, object, object, float)
    error = Signal(int, str)
    finished = Signal(int)


class _Worker(QRunnable):
    def __init__(self, generation, state_type, payload, catalog, selection,
                 condenser_key, projector_key, load_assembly):
        super().__init__()
        self.generation = generation
        self.state_type = state_type
        self.payload = payload
        self.catalog = catalog
        self.selection = selection
        self.condenser_key = condenser_key
        self.projector_key = projector_key
        self.load_assembly = load_assembly
        self.cancel_event = Event()
        self.signals = _Signals()

    @input_io.using_state_inputs
    def run(self):
        started = perf_counter()
        try:
            state = self.state_type.from_dict(self.payload)
            state._tuning_cancelled = self.cancel_event.is_set
            if self.load_assembly:
                self.catalog.apply(state, self.selection)
            else:
                # State deserialization starts with the default module root.
                # The solve must use the selected instrument's geometry too.
                apply_physical_layout_to_state(
                    state, assembly_root=self.catalog.root,
                    preserve_operating_parameters=True,
                )
            condenser_modes = compatible_modes(
                "condenser", self.selection.column, self.selection.recording
            )
            projector_modes = compatible_modes(
                "projector", self.selection.column, self.selection.recording
            )
            available = (
                self.condenser_key in {mode.key for mode in condenser_modes}
                and self.projector_key in {mode.key for mode in projector_modes}
            )
            if not available and not self.load_assembly:
                raise ValueError("Selected assembly has no matching optical preset")
            result = None
            if available:
                result = apply_operating_mode_pair(
                    state, self.condenser_key, self.projector_key,
                    column_name=self.selection.column,
                    recording_name=self.selection.recording,
                )
            apply_physical_layout_to_state(state, assembly_root=self.catalog.root)
            if self.cancel_event.is_set():
                return
            if hasattr(state, "_tuning_cancelled"):
                del state._tuning_cancelled
            self.signals.result.emit(
                self.generation, state, self.selection, result,
                perf_counter() - started,
            )
        except Exception as exc:
            self.signals.error.emit(self.generation, str(exc))
        finally:
            self.signals.finished.emit(self.generation)


class OperatingPresetController(QObject):
    result_ready = Signal(object, object, object, float)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        from temsim.gui.job_coordinator import CoordinatedPool
        self.pool = CoordinatedPool(self)
        self.pool.setMaxThreadCount(1)
        self._generation = 0

    def submit(self, state, catalog, selection, condenser_key, projector_key,
               *, load_assembly=False):
        self.invalidate_pending()
        worker = _Worker(
            self._generation, type(state), state.to_dict(), catalog, selection,
            condenser_key, projector_key, load_assembly,
        )
        from temsim.immutable_json import json_digest
        worker.job_input_identity = json_digest(dict(payload=worker.payload, selection=selection,
            condenser=condenser_key, projector=projector_key, load_assembly=load_assembly))
        self._active_cancel = worker.cancel_event
        worker.signals.result.connect(self._accept_result)
        worker.signals.error.connect(self._accept_error)
        worker.signals.finished.connect(self._accept_finished)
        self.pool.start(worker)

    def invalidate_pending(self):
        if getattr(self, "_active_cancel", None) is not None:
            self._active_cancel.set()
        self._generation += 1
        self.pool.clear()

    def _accept_result(self, generation, state, selection, result, duration):
        if generation == self._generation:
            self.result_ready.emit(state, selection, result, duration)

    def _accept_error(self, generation, message):
        if generation == self._generation:
            self.failed.emit(message)

    def _accept_finished(self, generation):
        if generation == self._generation:
            self.finished.emit()
