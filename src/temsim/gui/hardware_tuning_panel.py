"""Manual alignment-to-actuator controls over the instrument's live state."""
from __future__ import annotations

from copy import copy

from PySide6.QtCore import Qt, QSignalBlocker, Signal
from PySide6.QtWidgets import (
    QCheckBox, QFormLayout, QGroupBox, QLabel, QLayout, QLineEdit, QScrollArea,
    QSizePolicy, QSplitter, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from temsim.gui.input_policy import WheelSafeComboBox
from temsim.hardware_tuning import TUNING_TASKS, resolve_task
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
        self.scroll = QScrollArea()
        self.scroll.setObjectName("hardwareTuningControlsScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.scroll.setWidget(self._host)
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
        self.refresh_values(force=True)
        self.runtime_changed.emit(f"{target.key}.{field.name}")
