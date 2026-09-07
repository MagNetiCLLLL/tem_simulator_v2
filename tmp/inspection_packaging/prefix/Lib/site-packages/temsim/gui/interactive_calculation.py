"""Explicit range planning and independent cached signal inspection."""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, QSettings, QTimer, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QLineEdit, QHeaderView, QSplitter, QFormLayout,
    QGroupBox, QProgressBar, QFileDialog,
    QSlider,
)
from temsim.gui.input_policy import WheelSafeComboBox as QComboBox
from temsim.gui.input_policy import WheelSafeDoubleSpinBox as QDoubleSpinBox
from temsim.gui.input_policy import WheelSafeSpinBox as QSpinBox
from temsim.gui.interactive_controller import InteractiveController
from temsim.gui.aligned_control_table import AlignedControlTable
from temsim.interactive_calculation import (
    CalculationRange, InteractivePlan, available_controls, validate_plan,
)


_SPLITTER_SETTINGS_KEY = "interactive_calculation/control_splitter"
_LIVE_REFRESH_MS = 50
_BANK_DEBOUNCE_MS = 180


def _label(text):
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse
                                  | Qt.TextInteractionFlag.TextSelectableByKeyboard)
    return label


class InteractiveCalculationPage(QWidget):
    capture_requested = Signal()
    build_requested = Signal()
    operation_finished = Signal()
    tuning_changed = Signal(object)
    high_accuracy_requested = Signal()
    rays_requested = Signal()
    readout_updated = Signal(object)
    readout_status_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("interactiveCalculationPage")
        self.controller = InteractiveController(self)
        self.source_state = None
        self.current_state = lambda: self.source_state
        self.seeds = ()
        self.controls = ()
        self.live_widgets = {}
        self.live_plan = None
        self._live_mode = False
        self._readout = None
        self._failed = False
        self.operation_allowed = lambda: True
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(_BANK_DEBOUNCE_MS)
        self.timer.timeout.connect(self._read)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setObjectName("interactiveCalculationSplitter")
        left, middle = QWidget(), QWidget()
        left.setObjectName("interactiveRangePanel")
        middle.setObjectName("interactiveExcitationPanel")
        middle.setMinimumWidth(280)
        self.splitter.addWidget(left)
        self.splitter.addWidget(middle)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        layout = QVBoxLayout(self)
        layout.addWidget(self.splitter)
        form = QVBoxLayout(left)
        self.tuning_quality = _label("Live tuning: Preview (49 rays). Change quality in the toolbar.")
        form.addWidget(self.tuning_quality)
        self.capture = QPushButton("Capture current settings")
        self.capture.clicked.connect(self.capture_requested)
        form.addWidget(self.capture)
        self.capture_info = _label("Capture settings and enter ranges. Tune rays first; calculate high accuracy once when ready.")
        form.addWidget(self.capture_info)
        self.choice = QComboBox()
        self.choice.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.choice.setMinimumContentsLength(18)
        add = QPushButton("Add range")
        add.clicked.connect(self._add_range)
        buttons = QHBoxLayout()
        buttons.addWidget(self.choice, 1)
        buttons.addWidget(add)
        form.addLayout(buttons)
        self.ranges = QTableWidget(0, 6)
        self.ranges.setHorizontalHeaderLabels(["Parameter / unit", "Minimum", "Maximum", "Bank samples", "Bank update", "Reference"])
        self.ranges.horizontalHeader().moveSection(5, 1)
        self.ranges.setWordWrap(False)
        self.ranges.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in (1, 2, 3, 4, 5):
            self.ranges.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        form.addWidget(self.ranges)
        remove = QPushButton("Remove selected range")
        remove.clicked.connect(self._remove_range)
        form.addWidget(remove)
        self.live_start = QPushButton("Start live tuning")
        self.live_start.setToolTip("Uses continuous values inside the declared ranges and updates current settings and Ray Diagram. Does not build a high-accuracy bank.")
        self.live_start.clicked.connect(self.start_live_tuning)
        form.addWidget(self.live_start)
        self.final_calculation = QPushButton("Run high-accuracy once at current settings")
        self.final_calculation.clicked.connect(self._request_high_accuracy)
        form.addWidget(self.final_calculation)
        advanced = QGroupBox("Advanced bank (optional)")
        self.advanced_bank = advanced
        advanced.setCheckable(True)
        advanced.setChecked(False)
        advanced_layout = QVBoxLayout(advanced)
        advanced_body = QWidget()
        advanced_form = QVBoxLayout(advanced_body)
        advanced_layout.addWidget(advanced_body)
        advanced_body.hide()
        advanced.toggled.connect(advanced_body.setVisible)
        advanced.toggled.connect(lambda visible: self.ranges.setColumnHidden(3, not visible))
        advanced.toggled.connect(lambda visible: self.ranges.setColumnHidden(4, not visible))
        self.ranges.setColumnHidden(3, True)
        self.ranges.setColumnHidden(4, True)
        form.addWidget(advanced)
        self.budget = QDoubleSpinBox()
        self.budget.setRange(0.25, 20.0)
        self.budget.setValue(6.0)
        self.budget.setSuffix(" GiB")
        budget_layout = QFormLayout()
        budget_layout.addRow("Pinned cache limit", self.budget)
        advanced_form.addLayout(budget_layout)
        self.range_summary = _label("Minimum and maximum are required. Optical samples are exact nodes; readout ranges are continuous.")
        advanced_form.addWidget(self.range_summary)
        self.build = QPushButton("Build high-accuracy bank (advanced)")
        self.build.setEnabled(False)
        self.build.clicked.connect(self.build_requested)
        self.cancel = QPushButton("Cancel")
        self.cancel.setEnabled(False)
        self.cancel.clicked.connect(self._cancel)
        actions = QHBoxLayout()
        actions.addWidget(self.build)
        actions.addWidget(self.cancel)
        advanced_form.addLayout(actions)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        advanced_form.addWidget(self.progress)
        self.status = _label("No interactive bank. Main-window results are not changed by this page.")
        form.addWidget(self.status)
        self.live_status = _label("")
        form.addWidget(self.live_status)
        self.status.setToolTip("Lens and pre-specimen aperture changes use precomputed nodes. Post-specimen stops reuse ray coordinates and repropagate coherent waves. Range edits do not change an existing bank. Capture again after changing the sample or assembly.")
        self.export = QPushButton("Export range plan...")
        self.export.clicked.connect(self._export)
        advanced_form.addWidget(self.export)
        self.live_heading = _label("Excitation / live controls")
        self.live_heading.setParent(middle)
        self.live_heading.move(9, 9)
        self.live_heading.adjustSize()
        self.live_heading.setToolTip("Enter ranges on the left, then start live tuning or build an advanced bank.")
        self.live_table = AlignedControlTable(self.ranges, middle)
        # The main Ray Diagram owns the only live ray plot. This separate
        # widget is reparented into its Cached signals subpage by the workspace.
        self.readout_panel = QWidget(self)
        self.readout_panel.setObjectName("interactiveReadoutPanel")
        self.readout_panel.hide()
        right_layout = QVBoxLayout(self.readout_panel)
        self.result_status = _label("No signal readout")
        right_layout.addWidget(self.result_status)
        self.signal_table = QTableWidget(0, 3)
        self.signal_table.setHorizontalHeaderLabels(["Physical detector", "Source fraction", "Current (pA)"])
        self.signal_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        right_layout.addWidget(self.signal_table, 1)
        self.notes = _label("Images: choose Advanced bank in Illuminating Image or Scanning Image.")
        right_layout.addWidget(self.notes)
        self.controller.busy_changed.connect(self._busy)
        self.controller.progress.connect(self._progress)
        self.controller.bank_ready.connect(self._bank_ready)
        self.controller.readout_ready.connect(self._readout_ready)
        self.controller.failed.connect(self.show_error)
        self.splitter.setSizes([460, 340])
        # Old full-page splitters include the now-removed duplicate ray plot.
        saved = QSettings().value(_SPLITTER_SETTINGS_KEY)
        if saved is not None:
            self.splitter.restoreState(saved)
        self.splitter.splitterMoved.connect(lambda *_: QSettings().setValue(
            _SPLITTER_SETTINGS_KEY, self.splitter.saveState()))

    def _add_control_row(self, control, widget):
        self.live_table.add_control(control, widget)

    @property
    def busy(self):
        return self.controller.busy

    def set_source(self, state, seeds=()):
        if self.busy:
            raise RuntimeError("Cancel the current operation before capturing settings")
        self.timer.stop()
        if self._live_mode:
            self.live_status.clear()
            self._live_mode = False
            self.live_plan = None
            self.live_widgets = {}
            self.live_table.clear_controls()
            self.live_heading.setText("Excitation / live controls")
            self.live_heading.setToolTip("Captured settings: enter new tuning ranges on the left.")
        self.source_state, self.seeds = state, tuple(seeds)
        self.controls = available_controls(state)
        self.choice.clear()
        for control in self.controls:
            self.choice.addItem(f"{control.label} ({control.unit})", control)
        self.ranges.setRowCount(0)
        self.live_table.sync_draft()
        self.build.setEnabled(True)
        self.capture_info.setText(f"Final calculation: {state.electron_gun.ray_count:,} rays | step {state.step_mm:g} mm. Live tuning uses a small ray bundle.")
        self.status.setText("New settings captured. Previous bank remains available until a replacement completes.")
        self._mark_readout_previous("New settings captured")

    def set_tuning_quality(self, quality):
        from temsim.physics.optical_tuning import TUNING_PROFILES
        p = TUNING_PROFILES[quality]
        self.tuning_quality.setText(f"Live tuning: {quality} ({p.rays} rays). Change quality in the toolbar.")

    def start_live_tuning(self):
        try:
            if self.busy:
                raise ValueError("Finish or cancel the bank operation first")
            plan = self.plan(precompute=False, live_only=True)
            if not self.operation_allowed():
                raise ValueError("Finish the active calculation before starting live tuning")
            self.timer.stop()
            self._failed = False
            self.live_plan = plan
            self._live_mode = True
            self.live_heading.setText("Live controls")
            self.live_heading.setToolTip("Live tuning updates current settings and Ray Diagram.")
            self.live_table.clear_controls()
            self.live_widgets = {}
            for axis in plan.ranges:
                row = QWidget()
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setMinimumWidth(90)
                slider.setToolTip(f"{axis.control.label} ({axis.control.unit})")
                slider.setRange(0, 10000)
                slider.setTracking(True)
                slider.sliderReleased.connect(self._flush_live_tuning)
                value = QDoubleSpinBox()
                value.setDecimals(9)
                value.setRange(axis.minimum, axis.maximum)
                value.setSuffix(f" {axis.control.unit}")
                value.setSingleStep((axis.maximum-axis.minimum)/1000)
                value.setValue(min(max(axis.control.current, axis.minimum), axis.maximum))
                slider.setValue(round(10000*(value.value()-axis.minimum)/(axis.maximum-axis.minimum)))
                slider.valueChanged.connect(lambda i, w=value, a=axis: w.setValue(
                    a.minimum + i*(a.maximum-a.minimum)/10000))
                def changed(v, s=slider, a=axis):
                    s.blockSignals(True)
                    s.setValue(round(10000*(v-a.minimum)/(a.maximum-a.minimum)))
                    s.blockSignals(False)
                    self._queue_read()
                value.valueChanged.connect(changed)
                row_layout.addWidget(slider, 1)
                row_layout.addWidget(value)
                self.live_widgets[axis.control.identity] = value
                self._add_control_row(axis.control, row)
            self.live_status.setText("Live tuning | waiting for rays")
            self.rays_requested.emit()
            self.status.setText("Live tuning: lenses and apertures only; no specimen calculation."
                                + (" Detector rows: Advanced bank only."
                                   if len(plan.ranges) < self.ranges.rowCount() else ""))
            self.timer.start(_LIVE_REFRESH_MS)
        except (ValueError, RuntimeError) as exc:
            self.show_error(str(exc), affects_readout=False)

    def _flush_live_tuning(self):
        # Release submits the final value even inside the current throttle window.
        if self._live_mode and self.timer.isActive():
            self.timer.stop()
            self._read()

    def _request_high_accuracy(self):
        # Flush the final slider value before the explicit single calculation.
        self._flush_live_tuning()
        self.high_accuracy_requested.emit()

    def display_tuning_status(self, result):
        """Report the shared ray frame without drawing or changing cached signals."""
        if self._live_mode:
            self.live_status.setText(f"{result.simulation.metrics.get('tuning_quality', 'Preview')} | rays updated; sample signals not calculated")

    def _add_range(self):
        control = self.choice.currentData()
        if control is None:
            return
        # The selection list can predate live edits. Read the value when the
        # unit is added; keep this reference distinct from user-entered bounds.
        current = {c.identity: c for c in available_controls(self.current_state())}
        if control.identity not in current:
            self.show_error("This component changed; capture the current settings again", affects_readout=False)
            return
        control = current[control.identity]
        if any(self.ranges.item(i, 0).data(Qt.ItemDataRole.UserRole).identity == control.identity
               for i in range(self.ranges.rowCount())):
            self.show_error("This parameter already has a range", affects_readout=False)
            return
        row = self.ranges.rowCount()
        self.ranges.insertRow(row)
        name = control.label.removesuffix(" / Excitation")
        item = QTableWidgetItem(f"{name} ({control.unit})")
        item.setData(Qt.ItemDataRole.UserRole, control)
        item.setToolTip(f"{control.label}\nReference when added: {control.current:.9g} {control.unit}")
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.ranges.setItem(row, 0, item)
        reference = QTableWidgetItem(f"{control.current:.9g}")
        reference.setFlags(reference.flags() & ~Qt.ItemFlag.ItemIsEditable)
        reference.setToolTip(f"Reference when added: {control.current:.9g} {control.unit}")
        self.ranges.setItem(row, 5, reference)
        reference_display = QLineEdit(reference.text())
        reference_display.setReadOnly(True)
        reference_display.setMaximumWidth(92)
        reference_display.setToolTip(reference.toolTip())
        self.ranges.setCellWidget(row, 5, reference_display)
        for col in (1, 2):
            edit = QLineEdit()
            edit.setPlaceholderText("Required")
            edit.setMaximumWidth(96)
            self.ranges.setCellWidget(row, col, edit)
        samples = QSpinBox()
        samples.setRange(2, 129)
        samples.setValue(5)
        samples.setEnabled(control.stage == "optical")
        self.ranges.setCellWidget(row, 3, samples)
        stage = QTableWidgetItem("Precompute" if control.stage == "optical" else "Readout")
        stage.setFlags(stage.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.ranges.setItem(row, 4, stage)
        self.live_table.sync_draft()

    def _remove_range(self):
        if self.ranges.currentRow() >= 0:
            self.ranges.removeRow(self.ranges.currentRow())
            self.live_table.sync_draft()

    def plan(self, *, precompute=True, live_only=False):
        axes = []
        for row in range(self.ranges.rowCount()):
            control = self.ranges.item(row, 0).data(Qt.ItemDataRole.UserRole)
            # Detector geometry belongs to detached bank readout. Its draft
            # endpoints must not prevent unrelated live optical tuning.
            if live_only and control.group == "detector":
                continue
            lo, hi = (self.ranges.cellWidget(row, c).text().strip() for c in (1, 2))
            if not lo or not hi:
                raise ValueError(f"Range {row + 1}: minimum and maximum are required")
            axes.append(CalculationRange(control, float(lo), float(hi), self.ranges.cellWidget(row, 3).value()))
        if live_only and not axes:
            raise ValueError("Add a lens or aperture range for live tuning. Detector rows use Advanced bank.")
        plan = InteractivePlan(tuple(axes), int(self.budget.value() * 1024**3), precompute)
        if self.source_state is None:
            raise ValueError("Capture the current settings first")
        validate_plan(self.source_state, plan)
        return plan

    def start_build(self):
        try:
            self.timer.stop()
            self._live_mode = False
            plan = self.plan()
            self._failed = False
            self.progress.setValue(0)
            self.progress.setFormat("Preparing range cache")
            self.range_summary.setText(f"{plan.point_count} optical combinations | {self.budget.value():g} GiB cache limit | no lens interpolation")
            self.status.setText("Building independent range cache; previous results retained.")
            self._mark_readout_previous("Building a replacement bank")
            self.controller.build(self.source_state, plan, self.seeds)
        except (ValueError, RuntimeError, TypeError) as exc:
            self.show_error(str(exc))

    def _busy(self, busy):
        for widget in (self.capture, self.build, self.ranges, self.choice, self.budget,
                       self.live_start, self.final_calculation):
            widget.setEnabled(not busy)
        self.cancel.setEnabled(busy)
        if not busy:
            self.operation_finished.emit()

    def _progress(self, index, count, done, total, stage):
        self.progress.setValue(round(1000 * (index + done / max(total, 1)) / count))
        self.progress.setFormat(f"%p% | {index + 1}/{count}: {stage}")

    def _bank_ready(self, bank):
        self._live_mode = False
        self.live_status.clear()
        self.live_heading.setText("Cached controls")
        self.live_heading.setToolTip("Explore the completed range, separately from draft ranges and live settings.")
        self.seeds = ()
        self.progress.setValue(1000)
        self.status.setText(f"Ready: {len(bank.points)} optical points | {bank.retained_bytes / 1024**3:.3f} GiB pinned. Main results unchanged.")
        self.live_table.clear_controls()
        self.live_widgets = {}
        for axis in bank.plan.ranges:
            if axis.control.stage == "optical":
                widget = QComboBox()
                for value in axis.values:
                    widget.addItem(f"{value:.9g} {axis.control.unit}", value)
                widget.currentIndexChanged.connect(self._queue_read)
            else:
                widget = QDoubleSpinBox()
                widget.setDecimals(9)
                widget.setRange(axis.minimum, axis.maximum)
                widget.setSingleStep((axis.maximum - axis.minimum) / 100)
                widget.setValue(min(max(axis.control.current, axis.minimum), axis.maximum))
                widget.setSuffix(f" {axis.control.unit}")
                widget.valueChanged.connect(self._queue_read)
            self.live_widgets[axis.control.identity] = widget
            self._add_control_row(axis.control, widget)
        self._mark_readout_previous("New bank ready; preparing readout")
        self.timer.start(_BANK_DEBOUNCE_MS)

    def _queue_read(self, *_):
        if self._live_mode:
            self.live_status.setText("Live tuning | updating rays")
            # Throttle instead of debounce: sustained motion must not postpone
            # every update until the mouse stops. Read the latest values only.
            if not self.timer.isActive():
                self.timer.start(_LIVE_REFRESH_MS)
            return
        self._mark_readout_previous("Bank settings changed; updating")
        self.timer.start(_BANK_DEBOUNCE_MS)

    def _mark_readout_previous(self, reason):
        prefix = "Previous bank readout" if self._readout is not None else "No bank readout"
        message = f"{prefix} | {reason}"
        self.result_status.setText(message)
        self.readout_status_changed.emit(message)

    def _bank_coordinates(self):
        return {key: widget.currentData() if isinstance(widget, QComboBox) else widget.value()
                for key, widget in self.live_widgets.items()}

    def _read(self):
        if not self.live_widgets:
            return
        if self._live_mode:
            self.tuning_changed.emit(tuple((axis, self.live_widgets[axis.control.identity].value())
                                          for axis in self.live_plan.ranges))
            return
        if not self.operation_allowed():
            self.show_error("Finish the main calculation or alignment before reading the interactive bank")
            return
        coordinates = self._bank_coordinates()
        try:
            self._mark_readout_previous("Updating selected bank settings")
            self.controller.read(coordinates)
        except (ValueError, RuntimeError) as exc:
            self.show_error(str(exc))

    def _readout_ready(self, result):
        # A slider can change while a readout finishes, before the debounce
        # timer submits its replacement. Never label that older setting ready.
        if not self._live_mode and self.live_widgets and self._bank_coordinates() != dict(result.coordinates):
            return
        self._readout = result
        self.result_status.setText("Advanced bank | readout ready")
        self.result_status.setToolTip("\n".join(f"{key}: {value:.9g}" for key, value in result.coordinates.items()))
        self.signal_table.setRowCount(len(result.detector_fractions))
        for row, (key, fraction) in enumerate(result.detector_fractions.items()):
            for col, text in enumerate((key, f"{fraction:.9g}", f"{result.detector_current_pa[key]:.9g}")):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.signal_table.setItem(row, col, item)
        self.notes.setText("TEM: " + ("available" if result.wave is not None else "unavailable")
                           + " | STEM: " + ("available" if result.stem is not None else "unavailable")
                           + ". Images: choose Advanced bank in Illuminating Image or Scanning Image.")
        self.notes.setMaximumHeight(60)
        self.notes.setToolTip("\n".join(result.notes))
        # This event is the only successful bank-image publication. A later
        # busy=False notification must not mark the completed image pending.
        self.readout_updated.emit(result)

    def _cancel(self):
        self.controller.cancel()
        self.status.setText("Cancellation requested; previous complete bank retained.")
        if not self._live_mode:
            self._mark_readout_previous("Bank operation cancelled")

    def show_error(self, message, *, affects_readout=True):
        self._failed = True
        self.status.setText(str(message))
        self.progress.setFormat("Stopped: " + str(message))
        if affects_readout and not self._live_mode:
            self._mark_readout_previous("No new result; " + str(message))

    def _export(self):
        try:
            plan = self.plan()
            path, _ = QFileDialog.getSaveFileName(self, "Export range plan", "", "JSON (*.json)")
            if path:
                data = {"schema": "interactive-range-v1", "cache_budget_bytes": plan.cache_budget_bytes,
                        "optical_points": plan.point_count, "ranges": [
                            {"control": r.control.identity, "unit": r.control.unit, "minimum": r.minimum,
                             "maximum": r.maximum, "points": r.points, "stage": r.control.stage} for r in plan.ranges]}
                Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
        except (OSError, ValueError) as exc:
            self.show_error(str(exc), affects_readout=False)

    def shutdown(self):
        self.timer.stop()
        self.controller.shutdown()
