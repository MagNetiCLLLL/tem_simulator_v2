"""Manual alignment-to-actuator controls over the instrument's live state."""
from __future__ import annotations

from copy import copy
import math

from PySide6.QtCore import Qt, QSignalBlocker, Signal
from PySide6.QtWidgets import (
    QCheckBox, QFormLayout, QGroupBox, QLabel, QLayout, QLineEdit, QScrollArea,
    QPushButton, QSizePolicy, QSplitter, QToolButton, QTreeWidget, QTreeWidgetItem,
    QVBoxLayout, QWidget,
)

from temsim.gui.input_policy import WheelSafeComboBox, WheelSafeDoubleSpinBox
from temsim.hardware_tuning import TUNING_TASKS, resolve_task
from temsim.hardware_tuning_feedback import (
    baseline_difference, captured_hardware_values, observe_retained_beam, recorded_waists,
)
from temsim.runtime_parameters import convert_runtime_value, validate_runtime_assignment


def _label(text=""):
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse
                                  | Qt.TextInteractionFlag.TextSelectableByKeyboard)
    label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
    return label


class HardwareTuningPanel(QWidget):
    """Choosing a task is read-only; edits use the ordinary runtime validator.

    Bindings own labels and parameter identities, never another actuator value.
    The main window routes accepted edits through its existing invalidation and
    preview policy. No alignment solver or navigation action is called here.
    """

    runtime_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("hardwareTuningPanel")
        self._state = None
        self._generation = 0
        self._signature = None
        self._tasks = {task.key: task for task in TUNING_TASKS}
        self._items = {}
        self.editors = {}
        self._baseline = {}
        self._retained = {}
        self._captured_controls = {}
        self._result_stale = {}
        self._observations = {}
        self._feedback_baseline = None
        self._revision = 0
        self._plane_selected = False

        self.search = QLineEdit()
        self.search.setObjectName("hardwareTuningSearch")
        self.search.setPlaceholderText("Find an alignment…")
        self.tree = QTreeWidget()
        self.tree.setObjectName("hardwareTuningAlignments")
        self.tree.setHeaderHidden(True)
        self.tree.setMinimumWidth(150)
        self.tree.setUniformRowHeights(True)
        categories = {}
        for task in TUNING_TASKS:
            if task.category not in categories:
                group = QTreeWidgetItem(self.tree, [task.category])
                group.setFlags(group.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                categories[task.category] = group
            item = QTreeWidgetItem(categories[task.category], [task.label])
            item.setData(0, Qt.ItemDataRole.UserRole, task.key)
            item.setToolTip(0, task.description)
            self._items[task.key] = item
        self.tree.expandAll()
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(_label("Alignment"))
        left_layout.addWidget(self.search)
        left_layout.addWidget(self.tree, 1)

        self.title = _label()
        self.title.setObjectName("hardwareTuningTitle")
        self.title.setStyleSheet("font-size: 16px; font-weight: 600;")
        self.context = _label()
        self.description = _label()
        self.description.setObjectName("hardwareTuningDescription")
        self.status = _label()
        self.status.setObjectName("hardwareTuningStatus")
        self.status.hide()
        self._host = QWidget()
        self._form = QVBoxLayout(self._host)
        self._form.setContentsMargins(0, 0, 0, 0)
        self._form.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self.feedback_toggle = QToolButton()
        self.feedback_toggle.setObjectName("hardwareTuningFeedbackToggle")
        self.feedback_toggle.setText("Executed beam feedback (read only)")
        self.feedback_toggle.setCheckable(True)
        self.feedback_toggle.setChecked(True)
        self.feedback = QWidget()
        feedback_layout = QFormLayout(self.feedback)
        feedback_layout.setContentsMargins(0, 0, 0, 0)
        feedback_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        feedback_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.result_selection = WheelSafeComboBox()
        self.result_selection.setObjectName("hardwareTuningResultSelection")
        self.result_selection.addItem("Ray result", "ray")
        self.result_selection.addItem("Retained High accuracy", "high")
        self.observation_z = WheelSafeDoubleSpinBox()
        self.observation_z.setObjectName("hardwareTuningObservationZ")
        self.observation_z.setRange(-1e6, 1e6)
        self.observation_z.setDecimals(6)
        self.observation_z.setSuffix(" mm")
        self.observation_z.setKeyboardTracking(False)
        self.observation_z.setToolTip("Read retained histories at this Z. Defaults to the captured sample centre; changing Z never calculates or changes the instrument.")
        feedback_layout.addRow(_label("Result"), self.result_selection)
        feedback_layout.addRow(_label("Observation Z"), self.observation_z)
        self.feedback_status = _label()
        self.feedback_status.setObjectName("hardwareTuningFeedbackStatus")
        self.feedback_metrics = _label()
        self.feedback_metrics.setObjectName("hardwareTuningFeedbackMetrics")
        self.feedback_definitions = _label(
            "Weights: source-current fractions, without renormalizing transmission after stops. "
            "Centroid: weighted X/Y. Mean direction: projected angles of the weighted mean unit direction. "
            "RMS radius and D95: radius RMS and twice the 95% radius about the centroid. "
            "Angular RMS / 95%: angular deviations from the mean direction; not an aperture edge."
        )
        self.feedback_definitions.setObjectName("hardwareTuningFeedbackDefinitions")
        self.feedback_captured = _label()
        self.feedback_captured.setObjectName("hardwareTuningCapturedHardware")
        self.feedback_waists = _label()
        self.feedback_waists.setObjectName("hardwareTuningWaists")
        self.baseline_button = QPushButton("Use displayed readout as baseline")
        self.baseline_button.setObjectName("hardwareTuningSetBaseline")
        self.feedback_delta = _label()
        self.feedback_delta.setObjectName("hardwareTuningBaselineDelta")
        for widget in (self.feedback_status, self.feedback_metrics,
                       self.feedback_waists, self.baseline_button, self.feedback_delta,
                       ):
            feedback_layout.addRow(widget)
        self.feedback_details_toggle = QToolButton()
        self.feedback_details_toggle.setObjectName("hardwareTuningFeedbackDetailsToggle")
        self.feedback_details_toggle.setText("Definitions and captured hardware")
        self.feedback_details_toggle.setCheckable(True)
        self.feedback_details = QWidget()
        details_layout = QVBoxLayout(self.feedback_details)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.addWidget(self.feedback_definitions)
        details_layout.addWidget(self.feedback_captured)
        self.feedback_details.hide()
        self.feedback_details_toggle.toggled.connect(self.feedback_details.setVisible)
        feedback_layout.addRow(self.feedback_details_toggle)
        feedback_layout.addRow(self.feedback_details)
        content_layout.addWidget(self.feedback_toggle)
        content_layout.addWidget(self.feedback)
        content_layout.addWidget(_label("Current applied hardware controls"))
        content_layout.addWidget(self._host)
        self.scroll = QScrollArea()
        self.scroll.setObjectName("hardwareTuningControlsScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.scroll.setWidget(content)
        right = QWidget()
        right.setMinimumWidth(240)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        for widget in (self.title, self.context, self.description, self.status):
            right_layout.addWidget(widget)
        right_layout.addWidget(self.scroll, 1)
        right_layout.addWidget(_label(
            "Edit a value and press Enter or leave the field. Changes update the "
            "current instrument and follow the existing calculation settings."
        ))

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setObjectName("hardwareTuningSplitter")
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(left)
        self.splitter.addWidget(right)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([260, 740])
        layout = QVBoxLayout(self)
        layout.addWidget(_label(
            "Choose an alignment to edit its hardware controls here. "
            "These are manual adjustments; shared controls retain the same values across tasks."
        ))
        layout.addWidget(self.splitter, 1)
        self.tree.currentItemChanged.connect(self._selection_changed)
        self.search.textChanged.connect(self._filter)
        self.feedback_toggle.toggled.connect(self.feedback.setVisible)
        self.result_selection.currentIndexChanged.connect(self._refresh_feedback)
        self.observation_z.valueChanged.connect(self._plane_changed)
        self.baseline_button.clicked.connect(self._set_feedback_baseline)
        self.select_task(TUNING_TASKS[0].key)

    @property
    def selected_key(self):
        item = self.tree.currentItem()
        return item.data(0, Qt.ItemDataRole.UserRole) if item else None

    def select_task(self, key):
        self.tree.setCurrentItem(self._items[key])

    def _filter(self, text):
        text = text.strip().casefold()
        for key, item in self._items.items():
            task = self._tasks[key]
            item.setHidden(text not in f"{task.label} {task.category} {task.description}".casefold())
        for index in range(self.tree.topLevelItemCount()):
            group = self.tree.topLevelItem(index)
            group.setHidden(all(group.child(i).isHidden() for i in range(group.childCount())))
        if text:
            self.tree.expandAll()

    def _selection_changed(self, *_):
        self._signature = None
        self.status.hide()
        self.refresh_values()

    def set_state(self, state):
        if state is not self._state:
            self._signature = None
            self.status.hide()
            self._result_stale.update({key: True for key in self._retained})
        self._state = state
        self.refresh_values()

    def showEvent(self, event):
        self.refresh_values()
        super().showEvent(event)

    def refresh_values(self, *, force=False):
        task = self._tasks.get(self.selected_key)
        if task is None:
            return
        self.title.setText(task.label)
        self.description.setText(task.description)
        self.context.setText(
            "Current settings · "
            f"Illumination: {getattr(self._state, 'illumination_mode', 'unavailable')} · "
            f"Projection: {getattr(self._state, 'projector_mode', 'unavailable')}"
        )
        groups = resolve_task(self._state, task) if self._state is not None else ()
        signature = (id(self._state), task.key, tuple(
            (group.binding, id(group.target.obj) if group.target else None,
             group.fields, group.notice) for group in groups
        ))
        if signature != self._signature:
            self._signature = signature
            self._rebuild(groups)
        for group in groups:
            if group.target is None:
                continue
            for field in group.fields:
                identity = (group.target.key, field.name)
                editor = self.editors[identity]
                if (not force and isinstance(editor, QLineEdit)
                        and editor.hasFocus() and editor.isModified()):
                    continue
                value = getattr(group.target.obj, field.name)
                with QSignalBlocker(editor):
                    if isinstance(editor, QCheckBox):
                        editor.setChecked(value)
                    elif isinstance(editor, WheelSafeComboBox):
                        editor.setCurrentIndex(editor.findData(value))
                    else:
                        editor.setText(str(value))
                        editor.setModified(False)
                self._baseline[identity] = value
        self._refresh_feedback()

    def publish_result(self, result, quality):
        """Accept an already published execution; this method never calculates."""
        self._observations.clear()
        record = (result, str(quality), self._revision)
        captured_state = getattr(result, "state_snapshot", None)
        controls = {}
        for task in TUNING_TASKS:
            try:
                controls[task.key] = captured_hardware_values(captured_state, task.key)
            except (ValueError, TypeError, AttributeError, KeyError):
                controls[task.key] = ()
        self._retained["ray"] = record
        self._captured_controls["ray"] = controls
        self._result_stale["ray"] = False
        if str(quality).strip().lower() == "high accuracy":
            self._retained["high"] = record
            self._captured_controls["high"] = controls
            self._result_stale["high"] = False
        if not self._plane_selected:
            z = getattr(getattr(getattr(result, "state_snapshot", None), "sample", None), "z_mm", None)
            try:
                z = float(z)
            except (TypeError, ValueError, OverflowError):
                z = math.nan
            if math.isfinite(z):
                with QSignalBlocker(self.observation_z):
                    self.observation_z.setValue(z)
        self._refresh_feedback()

    def mark_result_stale(self, key="ray"):
        self._result_stale[key] = True
        self._refresh_feedback()

    def clear_results(self):
        self._retained.clear()
        self._captured_controls.clear()
        self._result_stale.clear()
        self._observations.clear()
        self._feedback_baseline = None
        self._plane_selected = False
        self._refresh_feedback()

    def capture_result_presentation(self):
        """Capture read-only presentation for transactional result-file loading."""
        return dict(retained=dict(self._retained), controls=dict(self._captured_controls),
                    stale=dict(self._result_stale), observations=dict(self._observations),
                    baseline=self._feedback_baseline, revision=self._revision,
                    selection=self.result_selection.currentData(), z_mm=self.observation_z.value(),
                    plane_selected=self._plane_selected, expanded=self.feedback_toggle.isChecked())

    def restore_result_presentation(self, saved):
        """Restore presentation only; never restore or apply instrument inputs."""
        self._retained = dict(saved["retained"])
        self._captured_controls = dict(saved["controls"])
        self._result_stale = dict(saved["stale"])
        self._observations = dict(saved["observations"])
        self._feedback_baseline = saved["baseline"]
        self._revision = saved["revision"]
        self._plane_selected = saved["plane_selected"]
        with QSignalBlocker(self.result_selection), QSignalBlocker(self.observation_z):
            self.select_result(saved["selection"])
            self.observation_z.setValue(saved["z_mm"])
        self.feedback_toggle.setChecked(saved["expanded"])
        self._refresh_feedback()

    def set_revision(self, revision):
        self._revision = int(revision)
        self._refresh_feedback()

    def select_result(self, key):
        index = self.result_selection.findData(key)
        if index >= 0:
            self.result_selection.setCurrentIndex(index)

    def _plane_changed(self, *_):
        self._plane_selected = True
        self._refresh_feedback()

    def _observation(self):
        record = self._retained.get(self.result_selection.currentData())
        if record is None:
            return None
        key = (id(record[0]), self.observation_z.value())
        if key not in self._observations:
            # Only detached scalars are cached. No extra physical histories or
            # solver state are retained when observation planes are visited.
            if len(self._observations) >= 16:
                self._observations.clear()
            self._observations[key] = observe_retained_beam(record[0], key[1])
        return self._observations[key]

    def _set_feedback_baseline(self):
        self._feedback_baseline = self._observation()
        self._refresh_feedback()

    def _refresh_feedback(self, *_):
        key = self.result_selection.currentData()
        record = self._retained.get(key)
        row = self._observation()
        self.baseline_button.setEnabled(row is not None and row.centroid_x_um is not None)
        if record is None or row is None:
            self.feedback_status.setText(f"Live revision {self._revision} · No retained result; run a calculation to obtain feedback.")
            for label in (self.feedback_metrics, self.feedback_captured, self.feedback_waists, self.feedback_delta):
                label.clear()
            return
        result, quality, revision = record
        captured = self._captured_controls.get(key, {}).get(self.selected_key, ())
        try:
            live = captured_hardware_values(self._state, self.selected_key)
        except (ValueError, TypeError, AttributeError, KeyError):
            live = ()
        changed = captured != live
        stale = self._result_stale.get(key, False) or changed
        self.feedback_status.setText(
            f"Live revision {self._revision} · {quality} result {row.result_id[:12]} · accepted at revision {revision}\n"
            + ("STALE — applied inputs changed or recalculation is pending.\n" if stale else "Captured execution.\n")
            + f"Z {row.z_mm:.6g} mm · {row.population} · {row.status}\n{row.reason}"
        )
        self.feedback_status.setToolTip(
            f"Request: {row.result_id}\nModel: {row.model_id}\nManifest: {row.manifest_id}\n{row.provenance}")
        self.feedback_status.setStyleSheet("color: #fbbf24;" if stale else "")

        def shown(value):
            return "unavailable" if value is None else f"{value:.6g}"

        self.feedback_metrics.setText(
            f"Positive-weight paths: {row.ray_count} · Source transmission: {shown(None if row.source_fraction is None else row.source_fraction * 100)}% · Current: {shown(row.current_pa)} pA\n"
            f"Centroid X/Y: {shown(row.centroid_x_um)} / {shown(row.centroid_y_um)} µm\n"
            f"Mean direction X/Y: {shown(row.direction_x_mrad)} / {shown(row.direction_y_mrad)} mrad\n"
            f"RMS radius: {shown(row.rms_radius_um)} µm · D95: {shown(row.diameter95_um)} µm\n"
            f"Angular RMS: {shown(row.angular_rms_mrad)} mrad · Angular 95%: {shown(row.angular95_mrad)} mrad\n"
            + row.provenance
        )
        self.feedback_captured.setText("Hardware used by this execution:\n" + (
            "\n".join(f"{group} · {label}: {value}" + (f" {unit}" if unit else "")
                      for _, _, group, label, unit, value in captured)
            if captured else "Unavailable for this task in the captured instrument."
        ))
        task = self._tasks.get(self.selected_key)
        focus = task is not None and "focus" in task.key
        self.feedback_waists.setVisible(focus)
        if focus:
            waists = recorded_waists(result, {item[0] for item in captured})
            self.feedback_waists.setText("Recorded optical-reference RMS waists:\n" + (
                "\n".join(f"{component}: Z {z:.6g} mm, RMS radius {radius:.6g} µm" for component, z, radius in waists)
                + "\nCurrent-weighted retained-node markers; at least five paths per bracket. Branch identity is not retained in these markers; this is not an image-focus calibration."
                if waists else "Unavailable: no qualifying waist marker for these lenses is retained. No focus position is extrapolated."
            ))
        if self._feedback_baseline is None:
            self.feedback_delta.setText("No baseline selected.")
        else:
            baseline = self._feedback_baseline
            delta, reason = baseline_difference(row, baseline)
            text = f"Baseline {baseline.result_id[:12]} · {baseline.population} · Z {baseline.z_mm:.6g} mm\n"
            if delta is not None:
                text += (f"Δ centroid X/Y: {delta['centroid_x_um']:.6g} / {delta['centroid_y_um']:.6g} µm\n"
                         f"Δ mean direction X/Y: {delta['direction_x_mrad']:.6g} / {delta['direction_y_mrad']:.6g} mrad\n")
            self.feedback_delta.setText(text + reason)

    def _rebuild(self, groups):
        self._generation += 1
        generation = self._generation
        self.editors.clear()
        self._baseline.clear()
        while self._form.count():
            item = self._form.takeAt(0)
            if item.widget():
                item.widget().hide()
                item.widget().deleteLater()
        if not groups:
            self._form.addWidget(_label("No editable hardware is available for this selection."))
        for group in groups:
            box = QGroupBox(group.binding.label)
            box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
            form = QFormLayout(box)
            form.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            if group.notice:
                form.addRow(_label(group.notice))
            if group.target is not None:
                for field in group.fields:
                    identity = (group.target.key, field.name)
                    value = getattr(group.target.obj, field.name)
                    if isinstance(value, bool):
                        editor = QCheckBox()
                        editor.toggled.connect(
                            lambda _=None, k=identity, g=generation: self._commit(k, g))
                    elif field.choices:
                        editor = WheelSafeComboBox()
                        for label, choice in field.choices:
                            editor.addItem(label, choice)
                        editor.currentIndexChanged.connect(
                            lambda _=None, k=identity, g=generation: self._commit(k, g))
                    else:
                        editor = QLineEdit()
                        editor.editingFinished.connect(
                            lambda k=identity, g=generation: self._commit(k, g))
                    editor.setObjectName(f"hardware_{identity[0]}_{identity[1]}")
                    editor.setToolTip(f"{group.binding.label} · {field.label}" +
                                      (f" ({field.unit})" if field.unit else ""))
                    self.editors[identity] = editor
                    label = field.label + (f" ({field.unit})" if field.unit else "")
                    form.addRow(_label(label), editor)
            self._form.addWidget(box)
        self._form.addStretch(1)
        self._form.activate()

    def _commit(self, identity, generation):
        if generation != self._generation or self._state is None:
            return
        editor = self.editors[identity]
        try:
            groups = resolve_task(self._state, self._tasks[self.selected_key])
            target, field = next(
                (group.target, field) for group in groups if group.target is not None
                for field in group.fields if (group.target.key, field.name) == identity
            )
            current = getattr(target.obj, field.name)
            if current != self._baseline.get(identity):
                raise ValueError("This parameter changed elsewhere. Review its current value and edit again.")
            if isinstance(editor, QCheckBox):
                value = editor.isChecked()
            elif isinstance(editor, WheelSafeComboBox):
                value = editor.currentData()
            else:
                value = convert_runtime_value(current, editor.text())
            value = validate_runtime_assignment(target, field.name, value)
            if value == current:
                self.refresh_values(force=True)
                return
            # Validate the component's coupled drives before touching live
            # state. Some setters also update a derived lower-coil gain.
            candidate = copy(target.obj)
            setattr(candidate, field.name, value)
            validate = getattr(candidate, "validate", None)
            if validate is not None:
                gun = self._state.electron_gun
                surface = getattr(gun.emitter, "surface_model", None) is not None
                electrode = any(target.obj is obj for obj in (
                    gun.extractor, gun.electrostatic_lens,
                ))
                if surface and electrode:
                    validate(grounded=True)
                else:
                    validate()
            setattr(target.obj, field.name, value)
        except (ValueError, TypeError, AttributeError, OverflowError, StopIteration) as exc:
            self.status.setText(str(exc) or "This control is no longer available in the current instrument.")
            self.status.setStyleSheet("color: #fca5a5;")
            self.status.show()
            self.refresh_values(force=True)
            return
        self.status.setText(f"Applied · {field.label}: {current} → {value}" +
                            (f" {field.unit}" if field.unit else ""))
        self.status.setStyleSheet("color: #86efac;")
        self.status.show()
        self._result_stale.update({key: True for key in self._retained})
        self.refresh_values(force=True)
        self.runtime_changed.emit(f"{target.key}.{field.name}")
