"""Read-only experiment plots/exports and explicit captured-input resumption."""
from pathlib import Path
from threading import Event

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, QRunnable, Signal, Slot
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QComboBox, QTabWidget)

from temsim.gui.job_coordinator import CoordinatedPool, ResourceClaim
from temsim.gui.input_policy import WheelSafeSpinBox
from temsim.experiment_records import load_experiment, save_experiment, export_experiment_table, pareto_points
from temsim.immutable_json import json_digest


class _Signals(QObject):
    result = Signal(object)
    failed = Signal(str)
    finished = Signal()


class _FileWorker(QRunnable):
    def __init__(self, operation, path, record, cancel_event):
        super().__init__()
        self.operation, self.path, self.payload = operation, path, record
        self.cancel_event = cancel_event
        self.signals = _Signals()
        # A bounded 256 MiB JSON record may coexist with decoded/frozen copies.
        self.resource_claim = ResourceClaim(2 * 1024**3)
        self.job_input_identity = json_digest(dict(operation=operation, path=str(path),
            experiment=record[2].execution_identity if record else None))

    def run(self):
        try:
            if self.cancel_event.is_set():
                return
            if self.operation == "load":
                value = load_experiment(self.path)
            elif self.operation == "save":
                save_experiment(self.path, *self.payload)
                value = self.path
            else:
                recipe, sweep, result, rules = self.payload
                export_experiment_table(self.path, recipe, sweep, result)
                value = self.path
            if not self.cancel_event.is_set():
                self.signals.result.emit(value)
        except Exception as exc:
            self.signals.failed.emit(str(exc))
        finally:
            self.signals.finished.emit()


class ExperimentTools(QWidget):
    resume_requested = Signal(object, object, object, object)
    record_loaded = Signal(object)
    error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.record = None
        self.pending = None
        self.busy = False
        self.worker = None
        self.pool = CoordinatedPool(self)
        self.cancel_event = Event()
        self.pool.coordinator.register_retained(self, "retained_roots")
        layout = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.identity = QLabel("No completed experiment")
        self.identity.setWordWrap(True)
        layout.addWidget(self.identity)
        self.load_button = QPushButton("Load experiment…")
        self.save_button = QPushButton("Save experiment…")
        self.export_button = QPushButton("Export table / manifest…")
        self.resume_button = QPushButton("Resume captured experiment")
        for button in (self.load_button, self.save_button, self.export_button, self.resume_button):
            bar.addWidget(button)
        self.load_button.clicked.connect(lambda: self._file("load"))
        self.save_button.clicked.connect(lambda: self._file("save"))
        self.export_button.clicked.connect(lambda: self._file("export"))
        self.resume_button.clicked.connect(self._resume)
        layout.addLayout(bar)
        self.plot_export = QPushButton("Export displayed plot…")
        self.plot_export.clicked.connect(self._export_plot)
        layout.addWidget(self.plot_export)
        choices = QHBoxLayout()
        self.metric = QComboBox()
        self.objective_a = QComboBox()
        self.objective_b = QComboBox()
        self.direction_a = QComboBox()
        self.direction_b = QComboBox()
        for combo in (self.metric, self.objective_a, self.objective_b):
            combo.setMinimumContentsLength(12)
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.direction_a.addItems(("min", "max"))
        self.direction_b.addItems(("max", "min"))
        choices.addWidget(QLabel("Plot"))
        choices.addWidget(self.metric)
        choices.addWidget(self.objective_a)
        choices.addWidget(self.direction_a)
        choices.addWidget(self.objective_b)
        choices.addWidget(self.direction_b)
        compare = QPushButton("Show trade-offs")
        choices.addWidget(compare)
        compare.clicked.connect(self._compare)
        self.metric.currentIndexChanged.connect(self._plot)
        layout.addLayout(choices)
        self.plot = pg.PlotWidget()
        self.plot.setObjectName("experimentResponsePlot")
        layout.addWidget(self.plot)
        self.notice = QLabel("Plots show executed points. Missing or failed points are not interpolated; no universal optimum is selected.")
        self.notice.setWordWrap(True)
        layout.addWidget(self.notice)
        self._buttons()

    def retained_roots(self):
        return (self.record, self.pending)

    def bind_pending(self, recipe, sweep, rules):
        self.pending = (recipe, sweep, rules)

    def accept_result(self, result):
        if self.pending is not None:
            recipe, sweep, rules = self.pending
            if result.recipe_digest == recipe.digest and tuple(result.sweep_axes) == tuple(sweep.axes):
                self.set_record((recipe, sweep, result, rules))
                self.pending = None

    def set_record(self, record):
        self.record = record
        result = record[2]
        self.identity.setText(f"Captured experiment: {result.execution_identity or 'historical identity unavailable'}; numerical qualification: not established")
        for combo in (self.metric, self.objective_a, self.objective_b):
            combo.blockSignals(True)
            combo.clear()
            for definition in result.metric_definitions:
                combo.addItem(f"{definition.label} ({definition.unit})", definition.key)
            combo.blockSignals(False)
        current = self.objective_b.findData("sample_surviving_current_pa")
        self.objective_b.setCurrentIndex(max(0, current))
        self._buttons()
        self._plot()

    def set_busy(self, busy):
        self.busy = bool(busy)
        self._buttons()

    def _buttons(self):
        self.load_button.setEnabled(not self.busy and self.worker is None)
        for button in (self.save_button, self.export_button):
            button.setEnabled(self.record is not None and self.worker is None)
        self.resume_button.setEnabled(self.record is not None and not self.busy and self.worker is None)

    def _resume(self):
        if self.record is not None:
            recipe, sweep, result, rules = self.record
            self.resume_requested.emit(recipe, sweep, rules, result)

    def _file(self, operation):
        if self.worker is not None:
            return
        if operation == "load":
            path, _ = QFileDialog.getOpenFileName(self, "Load captured experiment", "", "Experiment (*.temexp)")
        else:
            if self.record is None:
                return
            extension, label = ("temexp", "Experiment") if operation == "save" else ("csv", "Scalar table")
            path, _ = QFileDialog.getSaveFileName(self, label, f"experiment.{extension}", f"{label} (*.{extension})")
        if not path:
            return
        self.cancel_event = Event()
        self.worker = _FileWorker(operation, Path(path), self.record, self.cancel_event)
        self.worker.signals.result.connect(self._file_result)
        self.worker.signals.failed.connect(self.error.emit)
        self.worker.signals.finished.connect(self._file_finished)
        self.pool.start(self.worker)
        self._buttons()

    @Slot(object)
    def _file_result(self, value):
        if self.worker.operation == "load":
            self.set_record(value)
            self.record_loaded.emit(value)
        else:
            self.notice.setText(f"Saved {value}. Viewing/exporting did not change instrument controls.")

    @Slot()
    def _file_finished(self):
        self.worker = None
        self._buttons()

    def _plot(self, *_):
        self.plot.clear()
        if self.record is None:
            return
        recipe, sweep, result, _ = self.record
        metric = self.metric.currentData()
        if len(sweep.axes) not in {1, 2}:
            self.plot.setTitle("Response plots require one or two selected parameters")
            return
        first = sweep.axes[0]
        rows = sorted(result.point_results, key=lambda row: row.coordinates[first.path])
        x = [row.coordinates[first.path] for row in rows]
        self.plot.setLabel("bottom", first.path, units=first.unit)
        self.plot.setTitle(self.metric.currentText())
        if len(sweep.axes) == 1:
            y = [row.metrics.get(metric, np.nan) if row.status == "COMPLETE" else np.nan for row in rows]
            self.plot.setLabel("left", self.metric.currentText())
            if any(np.isfinite(y)):
                self.plot.plot(x, y, pen=pg.mkPen("#38bdf8") if sweep.kind != "normal_perturbations" else None, connect="finite")
                good = np.isfinite(y)
                self.plot.plot(np.asarray(x)[good], np.asarray(y)[good], symbol="o", pen=None)
        else:
            second = sweep.axes[1]
            self.plot.setLabel("left", second.path, units=second.unit)
            finite = [row.metrics[metric] for row in rows if row.status == "COMPLETE" and metric in row.metrics]
            lo, hi = (min(finite), max(finite)) if finite else (0., 1.)
            cmap = pg.colormap.get("viridis")
            spots = []
            for row in rows:
                value = row.metrics.get(metric)
                valid = row.status == "COMPLETE" and value is not None
                color = cmap.map((value-lo)/max(hi-lo, 1.e-30), mode="qcolor") if valid else "#ef4444"
                spots.append(dict(pos=(row.coordinates[first.path], row.coordinates[second.path]),
                    brush=pg.mkBrush(color), symbol="o" if valid else "x", data=f"Point {row.point_index+1}: {value if valid else row.failure_reason}"))
            self.plot.addItem(pg.ScatterPlotItem(spots, size=11, hoverable=True))
            self.plot.setTitle(f"{self.metric.currentText()} · colour range {lo:g} to {hi:g}; crosses: unavailable")

    def _compare(self):
        if self.record is None:
            return
        objectives = {self.objective_a.currentData(): self.direction_a.currentText(), self.objective_b.currentData(): self.direction_b.currentText()}
        if len(objectives) != 2:
            self.notice.setText("Choose two distinct objectives for a trade-off comparison")
            return
        indices = pareto_points(self.record[2], objectives)
        self.notice.setText("Nondominated points for the selected objectives: " + (", ".join(str(i+1) for i in indices) or "none with both metrics") + ". Failed/incomplete observations remain in the table; this is not a universal optimum.")

    def _export_plot(self):
        if self.record is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export displayed response", "experiment.png", "Plot (*.png)")
        if not path:
            return
        try:
            from pyqtgraph.exporters import ImageExporter
            self._plot()
            ImageExporter(self.plot.plotItem).export(path)
            from temsim.experiment_records import _atomic_json
            from temsim.immutable_json import thaw_json
            from hashlib import sha256
            output = Path(path)
            _atomic_json(output.with_suffix(output.suffix + ".manifest.json"), dict(execution_identity=self.record[2].execution_identity,
                output_file=output.name, output_sha256=sha256(output.read_bytes()).hexdigest(),
                metric=self.metric.currentData(), units={d.key: d.unit for d in self.record[2].metric_definitions},
                experiment_inputs=thaw_json(self.record[2].execution_inputs),
                scope="Executed points; no interpolation or numerical qualification"))
        except Exception as exc:
            self.error.emit(str(exc))

    def shutdown(self):
        self.cancel_event.set()
        self.pool.clear()
        return self.pool.waitForDone(3000)
