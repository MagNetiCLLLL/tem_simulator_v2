"""Lazy, bounded field-line presentation of a captured magnetic-field state."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import Event

import numpy as np

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal, Slot, Qt
from PySide6.QtWidgets import QCheckBox, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from temsim.gui.input_policy import WheelSafeComboBox as QComboBox
from temsim.gui.input_policy import WheelSafeDoubleSpinBox as QDoubleSpinBox
from temsim.gui.magnetic_field_canvas import MagneticFieldCanvas
from temsim.gui.magnetic_test_electron import TestElectronController


@dataclass(frozen=True)
class _Presentation:
    geometry: object
    labels: tuple
    categories: tuple
    scene: object = None


class _Signals(QObject):
    finished = Signal(object, object, object)


class _FieldLinesWorker(QRunnable):
    def __init__(self, key, state, z_limits_mm, prepared_scene=None):
        super().__init__()
        self.key, self.state, self.z_limits_mm = key, state, z_limits_mm
        self.prepared_scene = prepared_scene
        self.signals = _Signals()
        self.cancelled = Event()

    def run(self):
        geometry, error = None, None
        try:
            from temsim.cpu_resources import numerical_job
            from temsim.magnetic_field_scene import prepare_magnetic_scene
            from temsim.magnetic_field_lines import build_field_lines
            with numerical_job(1, cancelled=self.cancelled.is_set):
                scene = self.prepared_scene
                if scene is None:
                    scene = prepare_magnetic_scene(self.state, z_limits_mm=self.z_limits_mm)
                if not self.cancelled.is_set():
                    geometry = build_field_lines(
                        scene, reference_t=self.key[1], density=self.key[2],
                        max_lines=640, max_steps=64)
                    geometry = _Presentation(geometry, tuple(
                        (region.label, tuple(np.mean(region.bounds_m, axis=0)))
                        for region in getattr(scene, "source_regions", ())),
                        tuple(sorted(set(getattr(scene, "source_categories", ())))), scene)
        except Exception as exc:
            error = str(exc)
        if self.cancelled.is_set():
            geometry, error = None, None
        self.signals.finished.emit(self.key, geometry, error)


class MagneticField3DPage(QWidget):
    """Camera changes never call a field provider or a particle calculation."""

    settings_changed = Signal()
    view_range_changed = Signal(object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = None
        self._prepared_scene = None
        self._records = ()
        self._z_limits_mm = None
        self._generation = 0
        self._active = False
        self._worker = None
        self._cache = OrderedDict()
        self._current_geometry = None
        self._reference_initialized = False
        self._peak_t = 0.
        self._electron_mode = False
        self._field_status = "Select 3D field lines to view the captured combined magnetic field."
        self._field_tooltip = ""
        self._request_timer = QTimer(self)
        self._request_timer.setSingleShot(True)
        self._request_timer.setInterval(100)
        self._request_timer.timeout.connect(self._request_geometry)

        self.density = QComboBox()
        self.density.setObjectName("magnetic3DDensity")
        # Standard leaves headroom for a stronger field to add lines while the
        # reference stays fixed, instead of saturating at the initial peak.
        for label, factor in (("Sparse", .25), ("Standard", .5), ("Dense", 1.)):
            self.density.addItem(label, factor)
        self.density.setCurrentIndex(1)
        self.reference = QDoubleSpinBox()
        self.reference.setObjectName("magnetic3DReference")
        self.reference.setRange(1e-9, 1e4)
        self.reference.setDecimals(9)
        self.reference.setValue(1.)
        self.reference.setSuffix(" T")
        self.reference.setKeyboardTracking(False)
        self.reference.setToolTip(
            "Fixed |B| reference for density and colour. Stronger fields add lines "
            "on a logarithmic scale until the display budget saturates. It stays fixed across hardware adjustments.")
        self.match_reference = QPushButton("Use current peak")
        self.match_reference.setToolTip("Rescale only the display reference to the current combined field peak.")
        self.fit_button = QPushButton("Fit")
        self.fit_button.setToolTip("Fit all visible electrons and magnetic field lines at their actual relative scale. Hide the B background to fit electrons alone. Linked views share the fitted physical ranges.")
        self.link_view = QCheckBox("Link Ray Diagram")
        self.link_view.setObjectName("magneticViewLinkRayDiagram")
        self.link_view.setChecked(True)
        self.link_view.setToolTip("Share physical Z and projected U ranges with Ray Diagram. Uncheck to pan and zoom this view independently.")

        self.canvas = MagneticFieldCanvas()
        self.canvas.setObjectName("magneticField3DCanvas")
        self.canvas.view_range_changed.connect(self._canvas_view_range_changed)
        self.status = QLabel("Select 3D field lines to view the captured combined magnetic field.")
        self.status.setObjectName("magnetic3DStatus")
        self.status.setWordWrap(False)
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.status.setStyleSheet("color: #94a3b8;")
        self.canvas.setToolTip(
            "Combined magnetic field. Arrows point along +B, not electron motion. "
            "Colour and line density use a fixed logarithmic |B| scale. "
            "Rotate Ray Diagram to change this projection. Drag to pan; wheel to zoom. "
            "Wheel over an axis to zoom only that direction. Axis values are physical distances. "
            "Transverse dimensions are enlarged for viewing; physical geometry is unchanged.")
        self._field_canvas_tooltip = self.canvas.toolTip()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.status)
        # The parent owns the compact toolbar and advanced display controls.
        for control in (self.density, self.reference, self.match_reference, self.fit_button, self.link_view):
            control.setParent(self)
            control.hide()
        self.density.currentIndexChanged.connect(self._controls_changed)
        self.reference.valueChanged.connect(self._reference_changed)
        self.match_reference.clicked.connect(self._use_current_peak)
        self.fit_button.clicked.connect(self.canvas.fit_view)
        self.electron = TestElectronController(self)
        self.electron.paths_changed.connect(self._show_electron_paths)
        self.electron.status_changed.connect(self._electron_status_changed)
        self.electron.background_changed.connect(self._background_changed)

    def set_electron_mode(self, enabled):
        self._electron_mode = bool(enabled)
        self.canvas.set_electron_mode(enabled)
        self.canvas.setToolTip(
            "Coloured paths: independent virtual electrons in captured electric and magnetic fields. "
            "Colours match the electron list. Arrows follow motion; circles mark starts and squares mark endpoints. "
            "Background lines follow +B. Rotate Ray Diagram to change projection; drag to pan, wheel to zoom. "
            "Wheel over an axis to zoom only that direction. Axis values are physical distances. "
            "Transverse dimensions are enlarged for viewing."
            if enabled else self._field_canvas_tooltip)
        self.canvas.set_field_lines_visible(not enabled or self.electron.background.isChecked())
        self.electron.set_active(enabled and self._active)
        self._show_electron_paths(self.electron.visible_paths() if enabled else ())
        if enabled:
            self._electron_status_changed(self.electron.status_text)
        else:
            self.status.setText(self._field_status)
            self.status.setToolTip(self._field_tooltip)

    def _show_electron_paths(self, paths):
        if not self._electron_mode:
            self.canvas.clear_electron_paths()
        else:
            self.canvas.set_electron_paths(paths)

    def _electron_status_changed(self, message):
        if self._electron_mode:
            self.status.setText(message)
            self.status.setToolTip(
                message + "\nIndependent virtual electrons in captured electric and magnetic fields. "
                "Extraction, gun lens, acceleration and hardware interception are included; "
                "specimen, detector and electron-electron interactions are excluded. Rotation follows Ray Diagram.")

    def _set_field_status(self, message, tooltip=""):
        self._field_status, self._field_tooltip = message, tooltip
        if not self._electron_mode:
            self.status.setText(message)
            self.status.setToolTip(tooltip)

    def _background_changed(self, visible):
        self.canvas.set_field_lines_visible(not self._electron_mode or visible)
        if self._active and visible:
            self._request_timer.start(0)

    def set_projection_angle(self, angle_deg):
        self.canvas.set_projection_angle(angle_deg)

    def view_range_mm(self):
        return self.canvas.view_range_mm()

    def set_view_range_mm(self, x_range, y_range, *, emit=False):
        self.canvas.set_view_range_mm(x_range, y_range, emit=emit)
        # Navigation updates the explicit "Start at view centre" action only.
        # It never changes the current electron's physical initial conditions.
        self.electron.set_axial_range_mm(*self.canvas.view_range_mm()[0])

    def _canvas_view_range_changed(self, x_range, y_range):
        self.electron.set_axial_range_mm(*x_range)
        self.view_range_changed.emit(x_range, y_range)

    def set_wheel_scale_factor(self, factor):
        self.canvas.set_wheel_scale_factor(factor)

    def set_axial_range_mm(self, lower, upper):
        self.canvas.set_axial_range_mm(lower, upper)
        self.electron.set_axial_range_mm(lower, upper)

    def _reference_changed(self):
        self._reference_initialized = True
        self._controls_changed()

    def _current_peak(self):
        return self._peak_t

    def _use_current_peak(self):
        peak = self._current_peak()
        if peak > 0:
            self.reference.setValue(peak)

    def _controls_changed(self):
        self.settings_changed.emit()
        if self._active:
            self._request_timer.start(100)

    def update_snapshot(self, state, records, z_limits_mm, *, peak_t=None, prepared_scene=None):
        self.invalidate()
        self._state, self._records, self._z_limits_mm = state, tuple(records), z_limits_mm
        self._prepared_scene = prepared_scene
        self.electron.set_captured_scene(state, prepared_scene, z_limits_mm)
        self._peak_t = (float(peak_t) if peak_t is not None else
                        max((float(row.peak_t) for row in records), default=0.))
        if not self._reference_initialized and self._peak_t > 0:
            self._reference_initialized = True
            self.reference.setValue(self._peak_t)
        if self._active:
            self._request_timer.start(0)

    def invalidate(self, message="Magnetic field view pending a current calculation snapshot."):
        self._generation += 1
        self._state = None
        self._prepared_scene = None
        self._cache.clear()
        self._current_geometry = None
        self._request_timer.stop()
        if self._worker is not None:
            self._worker.cancelled.set()
        self.canvas.clear()
        self.electron.invalidate()
        self._set_field_status(message)

    def set_active(self, active):
        self._active = bool(active)
        self.electron.set_active(self._electron_mode and self._active)
        if self._active:
            self._request_timer.start(0)
        else:
            self._request_timer.stop()
            if self._worker is not None:
                self._worker.cancelled.set()

    def _key(self):
        return (self._generation, float(self.reference.value()), float(self.density.currentData()))

    def _request_geometry(self):
        if not self._active or self._state is None:
            return
        if self._electron_mode and not self.electron.background.isChecked():
            return
        key = self._key()
        if key in self._cache:
            self._show_geometry(self._cache[key])
            self._cache.move_to_end(key)
            return
        self._current_geometry = None
        self.canvas.clear_field_lines()
        self._set_field_status("Building magnetic field lines from the captured field…")
        if self._worker is not None:
            if self._worker.key != key:
                self._worker.cancelled.set()
            return
        worker = _FieldLinesWorker(key, self._state, self._z_limits_mm, self._prepared_scene)
        self._worker = worker
        # Destruction only signals cancellation; it never blocks the GUI on a
        # numerical job currently holding the process-wide CPU admission lock.
        worker.destruction_connection = self.destroyed.connect(
            lambda *_args, event=worker.cancelled: event.set())
        worker.signals.finished.connect(self._finished)
        QThreadPool.globalInstance().start(worker)

    @Slot(object, object, object)
    def _finished(self, key, geometry, error):
        if self._worker is not None:
            QObject.disconnect(self._worker.destruction_connection)
        self._worker = None
        if geometry is not None and key[0] == self._generation:
            self._cache[key] = geometry
            while len(self._cache) > 4:
                self._cache.popitem(last=False)
        if not self._active or self._state is None:
            return
        if key != self._key():
            self._request_timer.start(0)
        elif error:
            self._set_field_status(f"3D field lines unavailable: {error}")
        elif geometry is not None:
            self._show_geometry(geometry)
        else:
            self._request_timer.start(0)

    def _show_geometry(self, presentation):
        geometry = presentation.geometry
        if self._prepared_scene is None and presentation.scene is not None:
            self._prepared_scene = presentation.scene
            self.electron.set_captured_scene(self._state, presentation.scene, self._z_limits_mm)
        self._current_geometry = geometry
        # Fit the visible field lines, not the long near-zero tails of the
        # provider's numerical support. This changes no field or path data.
        display_bounds = geometry.bounds_m
        if len(geometry.segments_m):
            points = geometry.segments_m.reshape(-1, 3)
            lower, upper = points.min(axis=0), points.max(axis=0)
            padding = np.maximum((upper - lower) * .08,
                                 (geometry.bounds_m[1] - geometry.bounds_m[0]) * .002)
            display_bounds = np.stack((lower - padding, upper + padding))
        labels = presentation.labels
        self.canvas.set_geometry(
            geometry.segments_m, geometry.strengths_t,
            reference_t=geometry.reference_t, bounds_m=display_bounds,
            direction_segments_m=geometry.direction_segments_m, labels=labels)
        notes = "\n".join(geometry.notes)
        kinds = ", ".join(presentation.categories)
        self._set_field_status(
            f"{geometry.line_count} field lines | Combined magnetic field"
            + (f" | {kinds}" if kinds else "")
            + (" | No lines at this reference/density" if not geometry.line_count else ""),
            f"Fixed logarithmic reference {geometry.reference_t:.6g} T\n"
            f"Field domain Z {geometry.bounds_m[0, 2] * 1000:.6g}–"
            f"{geometry.bounds_m[1, 2] * 1000:.6g} mm\n" + notes)
