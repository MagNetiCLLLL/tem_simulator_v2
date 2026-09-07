"""Explicit background validation, separate from microscope calculations."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Event

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Qt, QRectF
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QProgressBar,
    QTableWidget, QTableWidgetItem, QTabWidget, QFileDialog, QSplitter, QCheckBox,
)
from temsim.gui.input_policy import WheelSafeDoubleSpinBox as QDoubleSpinBox, WheelSafeComboBox as QComboBox
from temsim.magnetic_validation import (
    ValidationOptions, ValidationCache, snapshot_state,
    input_signature, run_validation, report_payload,
)


class _Signals(QObject):
    progress = Signal(int, int, str)
    result = Signal(object)
    failed = Signal(str)
    finished = Signal()


class _Worker(QRunnable):
    def __init__(self, state, key, options, cancel):
        super().__init__()
        self.state, self.key, self.options, self.cancel = state, key, options, cancel
        self.signals = _Signals()

    def run(self):
        try:
            result = run_validation(self.state, self.key, self.options,
                                    progress=self.signals.progress.emit, cancelled=self.cancel.is_set)
            if not self.cancel.is_set():
                self.signals.result.emit(result)
        except Exception as exc:
            self.signals.failed.emit(str(exc))
        finally:
            self.signals.finished.emit()


class MagneticValidationPage(QWidget):
    """No model changes or result signals are sent to the imaging pipeline."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = None
        self._key = None
        self._report = None
        self._cache = ValidationCache()
        self._worker = None
        self._closed = False
        self._cancel = Event()
        self.run_button = QPushButton("Check mesh and boundary")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.setToolTip("Stop after the current field solve; completed image results are unaffected.")
        self.export_button = QPushButton("Export validation...")
        self.export_button.setEnabled(False)
        self.export_button.setToolTip("Export the displayed study and its original inputs, including retained earlier results. This does not certify an external software comparison.")
        self.relative = QDoubleSpinBox()
        self.relative.setRange(.0001, 20)
        self.relative.setDecimals(4)
        self.relative.setValue(1)
        self.relative.setSuffix(" %")
        self.relative.setToolTip("Relative change target. Absolute floors: field 1 uT; focal length 10 nm; rotation 0.0001 deg; test-beam radius 0.0001 um. User targets, not certified accuracy.")
        self.refinement = QDoubleSpinBox()
        self.refinement.setRange(1.1, 2)
        self.refinement.setSingleStep(.1)
        self.refinement.setValue(1.5)
        self.refinement.setToolTip("Two mesh refinements followed by two boundary expansions. The boundary study retains the existing interior mesh.")
        self.radius = QDoubleSpinBox()
        self.radius.setDecimals(3)
        self.radius.setRange(.001, 1000)
        self.radius.setValue(1)
        self.radius.setSuffix(" um")
        self.radius.setToolTip("Radius of a local collimated paraxial test bundle, not the user's illumination beam.")
        controls = QHBoxLayout()
        for widget in (QLabel("Target change"), self.relative, QLabel("Refinement"), self.refinement,
                       QLabel("Test radius"), self.radius):
            controls.addWidget(widget)
        controls.addStretch()
        actions = QHBoxLayout()
        for widget in (self.run_button, self.cancel_button, self.export_button):
            actions.addWidget(widget)
        actions.addStretch()
        self.status = QLabel("Not checked")
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.status.setWordWrap(True)
        self.status.setToolTip("Independent snapshots only. No preset optimisation or image recalculation. A field solve can converge without mesh/domain convergence. Cs/Cc and independent-software validation are not covered by these local paraxial checks.")
        self.progress = QProgressBar()
        self.progress.setRange(0, 5)
        self.progress.setFormat("%v / %m cases completed")
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(("Comparison", "Observable", "Value", "Change", "Allowed change", "Status"))
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.curves = pg.PlotWidget(background="#070b18")
        self.curves.setLabel("bottom", "Axial position (mm)")
        self.curves.setLabel("left", "Bz (T)")
        self.curves.showGrid(x=True, y=True, alpha=.15)
        self.curves.addLegend()
        self.validation_splitter = QSplitter(Qt.Orientation.Vertical)
        self.validation_splitter.setObjectName("magneticValidationSplitter")
        self.validation_splitter.addWidget(self.table)
        self.validation_splitter.addWidget(self.curves)
        self.validation_splitter.setSizes([250, 250])
        self.scene_plot = pg.PlotWidget(background="#070b18")
        self.scene_plot.setLabel("bottom", "Axial position Z (mm)")
        self.scene_plot.setLabel("left", "Radius R (mm)")
        self.scene_plot.setAspectLocked(True)
        for plot in (self.curves, self.scene_plot):
            for axis in ("bottom", "left"):
                plot.getAxis(axis).enableAutoSIPrefix(False)
        self.scene_plot.showGrid(x=True, y=True, alpha=.15)
        self.image = pg.ImageItem(axisOrder="row-major")
        self.scene_plot.addItem(self.image)
        self.colourbar = pg.ColorBarItem(values=(0, 1), colorMap=pg.colormap.get("viridis"), interactive=False)
        self.colourbar.setImageItem(self.image, insert_in=self.scene_plot.getPlotItem())
        self.layer = QComboBox()
        self.layer.addItems(("Field strength |B| (T)", "Material regions"))
        self.flux = QCheckBox("Magnetic flux lines")
        self.flux.setChecked(True)
        self.flux.setToolTip("Contours of R*A_phi: poloidal magnetic flux per radian. This is a computed field, not Gaussian display support.")
        fit = QPushButton("Fit material")
        fit.clicked.connect(self.fit_material)
        scene_controls = QHBoxLayout()
        for widget in (self.layer, self.flux, fit):
            scene_controls.addWidget(widget)
        scene_controls.addStretch()
        self.scene_hint = QLabel("Vacuum · magnetic material · excitation coil")
        self.scene_hint.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        scene = QWidget()
        scene_layout = QVBoxLayout(scene)
        scene_layout.addLayout(scene_controls)
        scene_layout.addWidget(self.scene_hint)
        scene_layout.addWidget(self.scene_plot, 1)
        self.tabs = QTabWidget()
        self.tabs.setObjectName("magneticValidationTabs")
        self.tabs.addTab(self.validation_splitter, "Convergence")
        self.tabs.addTab(scene, "Magnetic R-Z")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(controls)
        layout.addLayout(actions)
        layout.addWidget(self.status)
        layout.addWidget(self.progress)
        layout.addWidget(self.tabs, 1)
        self._contours = []
        self.run_button.clicked.connect(self.start)
        self.cancel_button.clicked.connect(self.cancel)
        self.export_button.clicked.connect(self.export)
        self.layer.currentIndexChanged.connect(self.redraw_scene)
        self.flux.toggled.connect(self.redraw_scene)
        for control in (self.relative, self.refinement, self.radius):
            control.valueChanged.connect(self.refresh_identity)

    @property
    def running(self):
        return self._worker is not None

    def options(self):
        return ValidationOptions(relative_tolerance=self.relative.value()/100,
                                 mesh_factor=self.refinement.value(), boundary_factor=self.refinement.value(),
                                 test_radius_um=self.radius.value())

    def set_context(self, state, key):
        self._state, self._key = state, key
        self.refresh_identity()

    def refresh_identity(self, *_args):
        if self._state is None or not self._key or self.running:
            return
        signature = input_signature(self._state, self._key, self.options())
        cached = self._cache.get(signature)
        if cached is not None:
            if cached is not self._report:
                self.display_report(cached)
            else:
                self._set_report_status()
        elif self._report is not None:
            self.status.setText(f"Inputs changed — previous {self._report.selected_key} study retained; run a new check")
        else:
            self.status.setText("Not checked · Cs/Cc and external validation: not checked")

    def start(self):
        if self.running or self._state is None or self._closed:
            return
        try:
            options = self.options().validate()
            signature = input_signature(self._state, self._key, options)
            cached = self._cache.get(signature)
            if cached is not None:
                self.display_report(cached)
                return
            snapshot = snapshot_state(self._state)
        except Exception as exc:
            self.status.setText(str(exc))
            return
        self._cancel = Event()
        self._worker = _Worker(snapshot, self._key, options, self._cancel)
        self._worker.signals.progress.connect(self._progress)
        self._worker.signals.result.connect(self._result)
        self._worker.signals.failed.connect(self._failure)
        self._worker.signals.finished.connect(self._finished)
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setValue(0)
        self.status.setText("Preparing detached validation study")
        QThreadPool.globalInstance().start(self._worker)

    def _progress(self, done, total, label):
        if not self._closed:
            self.progress.setRange(0, total)
            self.progress.setValue(done)
            self.status.setText(label)

    def _result(self, report):
        if self._closed or self._cancel.is_set():
            return
        self._cache.put(report)
        current = input_signature(self._state, self._key, self.options())
        if report.signature == current:
            self.display_report(report)
        else:
            self.status.setText("Study completed for earlier inputs; cached without replacing the current view")

    def _failure(self, message):
        if not self._closed:
            self.status.setText(message)

    def _finished(self):
        self._worker = None
        if not self._closed:
            self.run_button.setEnabled(True)
            self.cancel_button.setEnabled(False)

    def cancel(self):
        self._cancel.set()
        self.cancel_button.setEnabled(False)
        self.status.setText("Cancelling after current field solve; previous study retained")

    def shutdown(self):
        self._closed = True
        self._cancel.set()

    def _set_report_status(self):
        report = self._report
        status = "Sampled checks within tolerance" if report.passed else "Not all checks passed — inspect changes / unavailable values"
        self.status.setText(report.selected_key + " · " + status + " · Cs/Cc and external validation: not checked")

    def display_report(self, report):
        new_geometry = self._report is None or report.signature != self._report.signature
        self._report = report
        self._set_report_status()
        self.progress.setValue(5)
        self.export_button.setEnabled(True)
        self.table.setRowCount(len(report.comparisons))
        def fmt(value, unit):
            return "Unavailable" if value is None else f"{value:.6g} {unit}"
        for row, item in enumerate(report.comparisons):
            values = (item.stage, item.metric, fmt(item.value, item.unit), fmt(item.absolute_change, item.unit),
                      fmt(item.allowed_change, item.unit), "Within target" if item.passed else "Unavailable" if item.passed is None else "Outside target")
            for col, value in enumerate(values):
                cell = QTableWidgetItem(value)
                cell.setToolTip("Absolute difference <= absolute floor + relative target * refined magnitude. Near-zero relative differences are not divided by an artificial epsilon.")
                self.table.setItem(row, col, cell)
        self.table.resizeColumnsToContents()
        self.curves.clear()
        for i, case in enumerate(report.cases):
            self.curves.plot(report.z_m*1000, case.axis_field_t, pen=pg.mkPen(pg.intColor(i, 5), width=1.5), name=case.label)
        self.redraw_scene()
        if new_geometry:
            self.fit_material()

    def redraw_scene(self, *_args):
        if self._report is None:
            return
        scene = self._report.scene
        material = self.layer.currentIndex() == 1
        self.colourbar.setVisible(not material)
        data = scene.regions if material else scene.magnitude_t
        self.image.setImage(data, autoLevels=False)
        if material:
            self.image.setLookupTable(np.array(((7, 11, 24, 255), (120, 150, 190, 255), (245, 158, 11, 255)), np.ubyte))
            self.image.setLevels((0, 2))
        else:
            self.image.setLookupTable(pg.colormap.get("viridis").getLookupTable())
            self.colourbar.setLevels((0, max(float(np.max(data)), 1e-12)))
        # Pixel centres match the uniformly sampled physical coordinates.
        dz, dr = float(scene.z_m[1]-scene.z_m[0]), float(scene.r_m[1]-scene.r_m[0])
        rect = QRectF((scene.z_m[0]-.5*dz)*1000, (scene.r_m[0]-.5*dr)*1000,
                      dz*len(scene.z_m)*1000, dr*len(scene.r_m)*1000)
        self.image.setRect(rect)
        for item in self._contours:
            item.setParentItem(None)
            self.scene_plot.scene().removeItem(item)
        self._contours.clear()
        if self.flux.isChecked():
            flux = scene.flux_per_radian_wb
            low, high = float(np.min(flux)), float(np.max(flux))
            if high-low > 1e-25:
                for level in np.linspace(low, high, 14)[1:-1]:
                    contour = pg.IsocurveItem(data=flux, level=level, pen=pg.mkPen("#eeeeee", width=.7), axisOrder="row-major")
                    contour.setParentItem(self.image)
                    self._contours.append(contour)
        self.scene_hint.setText("Final study case · Blue: magnetic material · Orange: coil · Dark: vacuum" if material else
                                "Final study case · |B| in T · White: flux contours · Not OEM calibration")

    def fit_material(self):
        if self._report is not None:
            low, high, radius = self._report.scene.material_bounds_mm
            self.scene_plot.setRange(xRange=(low, high), yRange=(0, radius), padding=.08)

    def export(self):
        if self._report is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export validation and reference inputs", "magnetic_validation.json", "JSON (*.json)")
        if not path:
            return
        try:
            Path(path).write_text(json.dumps(report_payload(self._report), indent=2, allow_nan=False), encoding="utf-8")
        except (OSError, ValueError) as exc:
            self.status.setText(str(exc))
