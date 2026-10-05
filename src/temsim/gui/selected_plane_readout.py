"""Debounced, captured-optics readout for the selected Ray Diagram plane."""

from collections import OrderedDict
from html import escape
import math
from threading import Event

from PySide6.QtCore import QObject, QRunnable, QTimer, Qt, Signal, Slot
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from temsim.gui.job_coordinator import CoordinatedPool, ResourceClaim
from temsim.gui.plane_equations import plane_equation_tooltip
from temsim.cpu_resources import numerical_job
from temsim.physics.selected_plane import (
    SelectedPlaneDiagnostic,
    calculate_selected_plane,
    plane_status_without_trace,
)


class _Signals(QObject):
    solved = Signal(int, int, object)
    failed = Signal(int, str)
    finished = Signal(int)


class _PlaneWorker(QRunnable):
    def __init__(self, generation, result_generation, result, z_mm, event):
        super().__init__()
        self.generation = generation
        self.result_generation = result_generation
        self.existing_result = result
        self.z_mm = z_mm
        self.event = event
        self.job_input_identity = f"selected-plane:{result_generation}:{z_mm.hex()}"
        self.resource_claim = ResourceClaim(256 * 1024**2)
        self.signals = _Signals()

    def run(self):
        try:
            with numerical_job(1, cancelled=self.event.is_set):
                diagnostic = calculate_selected_plane(
                    self.existing_result, self.z_mm, cancelled=self.event.is_set
                )
            if not self.event.is_set():
                self.signals.solved.emit(
                    self.generation, self.result_generation, diagnostic
                )
        except Exception as exc:
            if not self.event.is_set():
                self.signals.failed.emit(self.generation, str(exc))
        finally:
            self.signals.finished.emit(self.generation)


class SelectedPlaneReadout(QWidget):
    """Show a bounded first-order diagnostic without retracing on the UI thread.

    Exact Z results are cached only within one explicit result publication.
    Re-publishing even the same result object invalidates the cache, because
    callers can have changed its arrays or captured state in place.
    """

    tooltip_changed = Signal(str)
    CACHE_LIMIT = 64
    DEBOUNCE_MS = 180
    _NAMES = {
        "image": "Image plane",
        "diffraction": "Diffraction plane",
        "mixed": "Mixed plane",
        "degenerate": "Degenerate transfer",
        "specimen": "Specimen plane",
        "upstream": "Upstream of specimen",
        "not_calculated": "Outside calculated range",
        "unavailable": "Plane diagnostic unavailable",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("selectedPlaneReadout")
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.label = QLabel("Plane state: choose an axial Z in a calculated Ray Diagram")
        self.label.setObjectName("selectedPlaneState")
        self.label.setWordWrap(False)
        self.label.setMinimumWidth(0)
        self.label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.label.setTextFormat(Qt.TextFormat.PlainText)
        self.label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self.label.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self._set_equation_tooltip()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)
        self.pool = CoordinatedPool(self)
        self.pool.setMaxThreadCount(1)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._start_pending)
        self._result = None
        self._result_generation = 0
        self._generation = 0
        self._selected_z_mm = None
        self._diagnostic = None
        self._cache = OrderedDict()
        self._worker = None
        self._pending = None
        self._stale = False
        self._closed = False

    def _cancel_request(self):
        self._generation += 1
        self.timer.stop()
        self._pending = None
        if self._worker is not None:
            self._worker.event.set()
        self.pool.clear()

    def set_result(self, result):
        # Shutdown processes queued Qt events while waiting for the worker.
        # A late calculation publication must not restart this observer.
        if self._closed:
            return
        self._cancel_request()
        self._result_generation += 1
        self._result = result
        self._selected_z_mm = None
        self._diagnostic = None
        self._cache.clear()
        self._stale = False
        self._show_waiting()

    def _set_equation_tooltip(self, kind="pending", *, stale=False, detail=""):
        tooltip = plane_equation_tooltip(kind, stale=stale)
        if detail:
            tooltip += f"<p>{escape(str(detail))}</p>"
        self.label.setToolTip(tooltip)
        self.tooltip_changed.emit(tooltip)

    def _show_waiting(self):
        self.label.setText("Plane state: choose an axial Z in a calculated Ray Diagram")
        self.label.setStyleSheet("color: #94a3b8;")
        self._set_equation_tooltip()

    def select_z(self, z_mm):
        if self._closed or self._result is None or z_mm is None:
            return
        try:
            if isinstance(z_mm, (bool, str, bytes)):
                raise ValueError("Selected axial Z must be numeric")
            z_mm = float(z_mm)
            if not math.isfinite(z_mm):
                raise ValueError("Selected axial Z must be finite")
        except (TypeError, ValueError, OverflowError) as exc:
            self._cancel_request()
            self._selected_z_mm = None
            self._display(SelectedPlaneDiagnostic(float("nan"), "unavailable", detail=str(exc)))
            return
        if z_mm == self._selected_z_mm:
            return
        self._cancel_request()
        self._selected_z_mm = z_mm
        self._diagnostic = None
        if z_mm in self._cache:
            self._cache.move_to_end(z_mm)
            self._display(self._cache[z_mm])
            return
        if self._stale:
            self._display_previous_pending()
            return
        try:
            diagnostic = plane_status_without_trace(self._result, z_mm)
        except Exception as exc:
            diagnostic = SelectedPlaneDiagnostic(z_mm, "unavailable", detail=str(exc))
        if diagnostic is not None:
            self._display(diagnostic)
            return
        self.label.setText(f"Plane state | Z {z_mm:.9g} mm | evaluating captured optics…")
        self.label.setStyleSheet("color: #94a3b8;")
        self._set_equation_tooltip()
        self._pending = (self._generation, self._result_generation, self._result, z_mm)
        self.timer.start(self.DEBOUNCE_MS)

    def _start_pending(self):
        if self._closed or self._stale or self._pending is None or self._worker is not None:
            return
        generation, result_generation, result, z_mm = self._pending
        self._pending = None
        if generation != self._generation or result_generation != self._result_generation:
            return
        worker = _PlaneWorker(generation, result_generation, result, z_mm, Event())
        self._worker = worker
        worker.signals.solved.connect(self._solved)
        worker.signals.failed.connect(self._failed)
        worker.signals.finished.connect(self._finished)
        self.pool.start(worker)

    @Slot(int, int, object)
    def _solved(self, generation, result_generation, diagnostic):
        if (self._closed or self._stale or generation != self._generation
                or result_generation != self._result_generation):
            return
        if float(diagnostic.z_mm) != self._selected_z_mm:
            return
        self._cache[self._selected_z_mm] = diagnostic
        self._cache.move_to_end(self._selected_z_mm)
        while len(self._cache) > self.CACHE_LIMIT:
            self._cache.popitem(last=False)
        self._display(diagnostic)

    @Slot(int, str)
    def _failed(self, generation, error):
        worker = self._worker
        if (self._closed or self._stale or generation != self._generation
                or worker is None or worker.result_generation != self._result_generation):
            return
        self._display(SelectedPlaneDiagnostic(
            self._selected_z_mm, "unavailable", detail=str(error)
        ))

    @Slot(int)
    def _finished(self, generation):
        if self._worker is not None and generation == self._worker.generation:
            self._worker = None
        if self._pending is not None and not self.timer.isActive():
            self._start_pending()

    def _display(self, diagnostic):
        self._diagnostic = diagnostic
        prefix = "Previous optics — inputs changed | " if self._stale else ""
        name = self._NAMES.get(diagnostic.kind, "Plane diagnostic unavailable")
        position = f"Z {diagnostic.z_mm:.9g} mm" if math.isfinite(diagnostic.z_mm) else "Invalid axial Z"
        text = f"{prefix}{position} | {name}"
        residuals = []
        if diagnostic.image_residual_m_per_rad is not None:
            residuals.append(f"image ‖B‖ {diagnostic.image_residual_m_per_rad:.3g} m/rad")
        if diagnostic.diffraction_residual is not None:
            residuals.append(f"diffraction ‖A‖ {diagnostic.diffraction_residual:.3g}")
        if residuals:
            text += " | " + "; ".join(residuals)
        self.label.setText(text)
        colour = "#fbbf24" if self._stale else "#a5f3fc"
        self.label.setStyleSheet(f"color: {colour};")
        # The equations remain symbolic; numerical residuals stay in the readout.
        # Retain boundary/failure reasons without inserting captured values into
        # the mathematical expressions or treating an unavailable plane as solved.
        detail = (str(diagnostic.detail).strip()
                  if diagnostic.kind in {"upstream", "not_calculated", "unavailable"}
                  else "")
        self._set_equation_tooltip(diagnostic.kind, stale=self._stale, detail=detail)

    def _display_previous_pending(self):
        self.label.setText(
            f"Previous optics — inputs changed | Z {self._selected_z_mm:.9g} mm | "
            "recalculate Ray Diagram to evaluate this plane"
        )
        self.label.setStyleSheet("color: #fbbf24;")
        self._set_equation_tooltip(stale=True)

    def mark_stale(self):
        self._cancel_request()
        self._stale = True
        if self._diagnostic is not None:
            self._display(self._diagnostic)
        elif self._selected_z_mm is not None:
            self._display_previous_pending()
        else:
            self.label.setText("Previous optics — inputs changed | recalculate Ray Diagram")
            self.label.setStyleSheet("color: #fbbf24;")
            self._set_equation_tooltip(stale=True)

    def clear(self):
        self.set_result(None)

    def shutdown(self):
        self._cancel_request()
        self._closed = True
        return self.pool.waitForDone(3_000)
