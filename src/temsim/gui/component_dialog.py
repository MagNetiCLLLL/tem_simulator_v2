"""Review a component insertion, placement or independent copy before staging it."""

from pathlib import Path
import math

from PySide6.QtCore import Qt
from PySide6.QtGui import QTextOption
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QFrame, QHBoxLayout,
    QLabel, QLineEdit, QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from temsim.gui.input_policy import WheelSafeComboBox


class _PathReadout(QPlainTextEdit):
    """Wrap Windows paths at any character while copying their exact text."""

    def __init__(self, text=""):
        super().__init__(text)
        self.setReadOnly(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setWordWrapMode(QTextOption.WrapMode.WrapAnywhere)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFixedHeight(self.fontMetrics().lineSpacing() * 3 + 6)
        self.setStyleSheet("background: transparent; border: none; padding: 0px;")

    def text(self):
        return self.toPlainText()

    def setText(self, text):
        self.setPlainText(text)


class ComponentDialog(QDialog):
    """``submit(values)`` validates/stages a candidate; an error keeps all input."""

    def __init__(self, action, *, source_path, part=None, target_paths=(),
                 module_origins=None, parts_for_target=None, document_for_target=None,
                 submit=None, parent=None):
        super().__init__(parent)
        if action not in {"new", "place", "copy"}:
            raise ValueError("Unknown component operation")
        self.action = action
        self.source_path = Path(source_path).resolve()
        self.part = dict(part or {})
        self._origins = {Path(path).resolve(): float(origin)
                         for path, origin in (module_origins or {}).items()}
        self._parts_for_target = parts_for_target or (lambda _path: ())
        self._document_for_target = document_for_target
        self._target_parts = ()
        self._last_target = self.source_path
        self._submit = submit
        self.result_value = None
        self._target_error = ""
        self._changing = False
        self._coordinate_mode = "local"
        self.setObjectName("componentOperationDialog")
        self.setWindowTitle({"new": "New component", "place": "Place component",
                             "copy": "Copy to assembly"}[action])
        self.resize(620, 680 if action == "new" else 540)
        layout = QVBoxLayout(self)
        self.source_label = _PathReadout(f"Source file: {self.source_path}")
        self.source_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.source_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.source_label)
        content = QWidget()
        form = QFormLayout(content)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        target_row = QHBoxLayout()
        self.target = WheelSafeComboBox()
        self.target.setObjectName("componentTargetFile")
        self.target.setMinimumContentsLength(18)
        self.target.setSizeAdjustPolicy(WheelSafeComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        paths = dict.fromkeys((self.source_path, *(Path(path).resolve() for path in target_paths)))
        for path in paths:
            self.target.addItem(str(path), str(path))
        self.target.setCurrentIndex(self.target.findData(str(self.source_path)))
        self.browse_button = QPushButton("Browse…")
        self.browse_button.clicked.connect(self._browse)
        target_row.addWidget(self.target, 1)
        target_row.addWidget(self.browse_button)
        form.addRow("Destination file", target_row)
        self.target.setEnabled(action == "copy")
        self.browse_button.setVisible(action == "copy")
        self.target_label = _PathReadout()
        self.target_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.target_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self.target_label)
        self.key_edit = QLineEdit()
        self.key_edit.setObjectName("componentKey")
        self.name_edit = QLineEdit()
        self.name_edit.setObjectName("componentName")
        stem = "new_component" if action == "new" else str(self.part.get("key", "component")) + "_copy"
        keys = {part["key"] for part in self._parts_for_target(self.source_path)}
        key, number = stem, 2
        while key in keys:
            key, number = f"{stem}_{number}", number + 1
        self.key_edit.setText(self.part.get("key", "") if action == "place" else key)
        self.name_edit.setText("New component" if action == "new" else
                               str(self.part.get("name", "Component")) + (" copy" if action == "copy" else ""))
        self.key_edit.setReadOnly(action == "place")
        self.name_edit.setReadOnly(action == "place")
        form.addRow("Component key", self.key_edit)
        form.addRow("Name", self.name_edit)
        self.parent_part = WheelSafeComboBox()
        self.parent_part.setObjectName("componentParent")
        self.parent_part.setMinimumContentsLength(18)
        self.parent_part.setSizeAdjustPolicy(WheelSafeComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        form.addRow("Belongs to", self.parent_part)
        self.parent_part.setEnabled(action != "place")
        self.shape = WheelSafeComboBox()
        self.shape.setObjectName("componentShape")
        for label, value in (("Tube / circular cylinder", "tube"), ("Box", "box"),
                             ("Elliptic cylinder", "elliptic_cylinder")):
            self.shape.addItem(label, value)
        form.addRow("Shape", self.shape)
        form.setRowVisible(self.shape, action == "new")
        self.values = {}
        for field, label, default in (("length_mm", "Length (mm)", 10),
                                      ("inner_diameter_mm", "Inner diameter (mm)", 10),
                                      ("outer_diameter_mm", "Outer diameter (mm)", 20),
                                      ("width_mm", "Width X (mm)", 20),
                                      ("height_mm", "Height Y (mm)", 20)):
            editor = QLineEdit(str(default))
            editor.setObjectName("component_" + field)
            self.values[field] = editor
            form.addRow(label, editor)
        self._form = form
        self.coordinates = WheelSafeComboBox()
        self.coordinates.setObjectName("componentCoordinates")
        self.coordinates.addItem("Local Z · destination file", "local")
        self.coordinates.addItem("Global Z · installed instrument", "global")
        form.addRow("Coordinate system", self.coordinates)
        self.center = QLineEdit(format(float(self.part.get("local_center_z_mm", 0)), ".15g"))
        self.center.setObjectName("componentCenterZ")
        form.addRow("Centre Z (mm)", self.center)
        self.coordinate_note = QLabel()
        self.coordinate_note.setWordWrap(True)
        form.addRow(self.coordinate_note)
        self.placement_context = QLabel()
        self.placement_context.setObjectName("componentPlacementContext")
        self.placement_context.setWordWrap(True)
        form.addRow(self.placement_context)
        self.include_children = QCheckBox("Include child components")
        self.include_children.setObjectName("componentIncludeChildren")
        self.include_children.setChecked(True)
        self.include_children.setVisible(action != "new")
        form.addRow(self.include_children)
        self.summary = QLabel()
        self.summary.setObjectName("componentOperationSummary")
        self.summary.setWordWrap(True)
        form.addRow(self.summary)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)
        note = QLabel("Confirmation updates the 3D draft and preview. Save writes the destination file. "
                      "New components and copies are independent mechanical CAD geometry; no optical device is installed.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.error_label = QLabel()
        self.error_label.setObjectName("componentOperationError")
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #ffb4a9")
        self.error_label.hide()
        layout.addWidget(self.error_label)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.confirm_button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.confirm_button.setText({"new": "Add to draft", "place": "Place in draft", "copy": "Copy to draft"}[action])
        self.buttons.accepted.connect(self._confirm)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.target.currentIndexChanged.connect(self._target_changed)
        self.coordinates.currentIndexChanged.connect(self._coordinates_changed)
        self.shape.currentIndexChanged.connect(self._shape_changed)
        self.center.textChanged.connect(self._update_summary)
        self.parent_part.currentIndexChanged.connect(self._update_summary)
        self.include_children.toggled.connect(self._update_summary)
        for editor in self.values.values():
            editor.textChanged.connect(self._update_summary)
        self._target_changed()
        self._shape_changed()

    @staticmethod
    def _number(text, label):
        try:
            value = float(text)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{label} must be a finite number") from exc
        if not math.isfinite(value):
            raise ValueError(f"{label} must be a finite number")
        return value

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose destination assembly file", str(self.source_path.parent),
                                             "Instrument TOML (*.toml)")
        if path:
            resolved = str(Path(path).resolve())
            index = self.target.findData(resolved)
            if index < 0:
                self.target.addItem(resolved, resolved)
                index = self.target.count() - 1
            self.target.setCurrentIndex(index)

    def _target_changed(self, *_):
        path = Path(self.target.currentData()).resolve()
        self.target_label.setText(str(path))
        self.target.setToolTip(str(path))
        blocked = self.parent_part.blockSignals(True)
        self.parent_part.clear()
        self.parent_part.addItem("Top level", None)
        try:
            self._target_parts = tuple(self._parts_for_target(path))
            for part in self._target_parts:
                self.parent_part.addItem(f"{part.get('name', part['key'])} · {part['key']}", part["key"])
            document = self._document_for_target(path) if self._document_for_target else {}
            ports = document.get("ports", {})
            if "entrance" in ports and "exit" in ports:
                self.placement_context.setText(
                    f"File extent: local Z {float(ports['entrance']['local_z_mm']):.12g} to "
                    f"{float(ports['exit']['local_z_mm']):.12g} mm (entrance → exit).")
            else:
                self.placement_context.setText("File entrance and exit are unavailable.")
            self._target_error = ""
        except Exception as exc:
            self._target_error = str(exc)
            self._target_parts = ()
            self.placement_context.setText("Destination could not be read: " + str(exc))
        if self.action == "place":
            self.parent_part.setCurrentIndex(max(0, self.parent_part.findData(self.part.get("parent_key"))))
        self.parent_part.blockSignals(blocked)
        available = path in self._origins
        self.coordinates.model().item(1).setEnabled(available)
        self._changing = True
        if not available and self.coordinates.currentData() == "global":
            # Global values cannot silently acquire a different frame meaning.
            self.coordinates.setCurrentIndex(0)
            try:
                local = self._number(self.center.text(), "Centre Z") - self._origins[self._last_target]
                self.center.setText(format(local, ".15g"))
            except (ValueError, KeyError):
                pass
            self._coordinate_mode = "local"
        self._changing = False
        self._last_target = path
        self.coordinate_note.setText(
            f"Global Z = local Z + {self._origins[path]:.12g} mm (resolved destination origin)."
            if available else "Global position unavailable: this file is not resolved in the installed instrument. Use local Z.")
        self._update_summary()

    def _coordinates_changed(self, *_):
        if self._changing:
            return
        mode = self.coordinates.currentData()
        origin = self._origins.get(Path(self.target.currentData()).resolve())
        try:
            value = self._number(self.center.text(), "Centre Z")
            if origin is not None and mode != self._coordinate_mode:
                self.center.setText(format(value + origin if mode == "global" else value - origin, ".15g"))
        except ValueError:
            pass
        self._coordinate_mode = mode
        self._update_summary()

    def _shape_changed(self, *_):
        shape = self.shape.currentData()
        for field, editor in self.values.items():
            self._form.setRowVisible(editor, self.action == "new" and
                (field == "length_mm" or (field in {"inner_diameter_mm", "outer_diameter_mm"}) == (shape == "tube")))
        self._update_summary()

    def values_to_submit(self):
        if self._target_error:
            raise ValueError(self._target_error)
        target = Path(self.target.currentData()).resolve()
        key, name = self.key_edit.text().strip(), self.name_edit.text().strip()
        if not key or not name:
            raise ValueError("Component key and name are required")
        center = self._number(self.center.text(), "Centre Z")
        origin = self._origins.get(target)
        mode = self.coordinates.currentData()
        if mode == "global":
            if origin is None:
                raise ValueError("Global coordinates require a resolved destination file")
            center -= origin
        result = dict(action=self.action, target_path=target, key=key, name=name,
                      center_z_mm=center, coordinate_system=mode, module_origin_z_mm=origin,
                      parent_key=self.parent_part.currentData(), include_children=self.include_children.isChecked())
        if self.action == "new":
            result["shape"] = self.shape.currentData()
            for field, editor in self.values.items():
                if not editor.isHidden():
                    result[field] = self._number(editor.text(), field)
        return result

    def _update_summary(self, *_):
        try:
            values = self.values_to_submit()
            local = values["center_z_mm"]
            origin = values["module_origin_z_mm"]
            position = f"Destination centre: local Z {local:.12g} mm"
            if origin is not None:
                position += f" / global Z {local + origin:.12g} mm"
            if self.action == "place":
                position = f"Current local Z {float(self.part['local_center_z_mm']):.12g} mm → " + position
            if self.action == "new":
                length = values["length_mm"]
                start, end = local - length / 2, local + length / 2
            else:
                shift = local - float(self.part["local_center_z_mm"])
                start = float(self.part["local_start_z_mm"]) + shift
                end = float(self.part["local_end_z_mm"]) + shift
            position += f".\nComponent envelope: local Z {start:.12g} to {end:.12g} mm."
            parent = next((part for part in self._target_parts if part["key"] == values["parent_key"]), None)
            if parent is not None:
                position += (f"\nParent {parent.get('name', parent['key'])}: local Z "
                             f"{float(parent['local_start_z_mm']):.12g} to {float(parent['local_end_z_mm']):.12g} mm.")
            overlap = [part for part in self._target_parts if not (self.action == "place" and part["key"] == self.part.get("key"))
                       and float(part["local_start_z_mm"]) < end and float(part["local_end_z_mm"]) > start]
            if overlap:
                position += f"\nAxial envelopes overlap {len(overlap)} existing components; this does not establish solid interference."
            position += "\nConfirm to review the 3D draft. Save writes the destination file."
            self.summary.setText(position)
        except ValueError:
            self.summary.setText("Enter valid dimensions and a centre position to review the destination.")

    def _confirm(self):
        try:
            values = self.values_to_submit()
            self.result_value = self._submit(values) if self._submit is not None else values
        except Exception as exc:
            self.error_label.setText(str(exc))
            self.error_label.show()
            return
        self.accept()
