"""Interactive 2-D axisymmetric section editor for simple annular parts."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import pyqtgraph as pg
from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDialog, QFormLayout, QFrame, QGraphicsRectItem, QGroupBox, QHBoxLayout,
    QLabel, QLayout, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from temsim.gui.input_policy import WheelSafeComboBox, WheelSafeDoubleSpinBox
from temsim.part_geometry import geometry_from_part


class GeometryEditorDialog(QDialog):
    """Preview dimension changes; only Apply invokes the persistence callback."""

    def __init__(
        self, part: Mapping, apply_changes: Callable[[dict], None], *,
        parent=None, neighbours=(),
    ):
        super().__init__(parent)
        self.setObjectName("partGeometryEditor")
        self.setWindowTitle("Part dimensions · 2D axisymmetric section")
        self.setMinimumSize(900, 600)
        self.resize(1100, 700)
        self._apply_changes = apply_changes
        self._source_part = dict(part)
        self._simulation_mode = None
        self._simulation_descriptors = None
        self._semantic_by_key = {}
        self._semantic_dimension = "length_mm"
        self._baseline = geometry_from_part(part)
        self._draft = self._baseline
        self._history = [self._draft]
        self._history_index = 0
        self._numeric_origin = None
        self._drag_origin = None
        self._syncing = False
        self._input_error = False
        self._neighbours = []
        for neighbour in neighbours:
            try:
                geometry = geometry_from_part(neighbour)
            except (TypeError, ValueError, KeyError, OverflowError):
                continue
            if geometry.key != self._draft.key:
                self._neighbours.append(geometry)

        self.setStyleSheet("""
            QDialog#partGeometryEditor { background: #101827; color: #e5edf8; }
            #partGeometryEditor QLabel { color: #dbe6f4; }
            #partGeometryEditor QGroupBox {
                color: #dbe6f4; border: 1px solid #33445c; border-radius: 7px;
                margin-top: 12px; padding: 15px 12px 10px;
            }
            #partGeometryEditor QGroupBox::title { subcontrol-origin: margin; left: 12px; }
            #partGeometryEditor QDoubleSpinBox, #partGeometryEditor QComboBox {
                background: #1c2a3e; color: #f1f5fb; border: 1px solid #425773;
                border-radius: 4px; padding: 5px;
            }
            #partGeometryEditor QPushButton {
                background: #223249; color: #e5edf8; border: 1px solid #425773;
                border-radius: 5px; padding: 7px 13px;
            }
            #partGeometryEditor QPushButton:hover { background: #30445f; }
            #partGeometryEditor QPushButton:disabled { color: #697c95; border-color: #2a394c; }
            #partGeometryEditor QPushButton#geometryApply {
                background: #1677ce; border-color: #268fe9; color: white;
            }
            #partGeometryEditor QPushButton#geometryApply:disabled { background: #233d55; color: #7890a7; }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(12)
        heading = QLabel(self._baseline.name)
        heading.setStyleSheet("font-size: 21px; font-weight: 600;")
        layout.addWidget(heading)
        description = self._label(
            "2D axisymmetric section · Z and R in millimetres\n"
            "Drag a handle or enter a dimension. The axial centre stays fixed; adjacent clearances may change."
        )
        description.setStyleSheet("color: #9db0c8;")
        layout.addWidget(description)

        body = QHBoxLayout()
        body.setSpacing(18)
        canvas = QVBoxLayout()
        toolbar = QHBoxLayout()
        self.status_label = QLabel("Applied dimensions")
        toolbar.addWidget(self.status_label)
        toolbar.addStretch()
        self.undo_button = self._button("Undo", self.undo)
        self.redo_button = self._button("Redo", self.redo)
        self.fit_button = self._button("Fit view", self.fit_view)
        for button in (self.undo_button, self.redo_button, self.fit_button):
            toolbar.addWidget(button)
        canvas.addLayout(toolbar)
        self.plot = pg.PlotWidget(background="#0b1220")
        self.plot.setObjectName("partGeometryPlot")
        self.plot.setMinimumSize(480, 320)
        self.plot.setLabel("bottom", "Axial position Z", units="mm", color="#9bb0cb")
        self.plot.setLabel("left", "Radius R", units="mm", color="#9bb0cb")
        self.plot.showGrid(x=True, y=True, alpha=0.16)
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()
        self.plot.getViewBox().setAspectLocked(True, 1.0)
        self.plot.getViewBox().disableAutoRange()
        for axis in ("left", "bottom"):
            self.plot.getAxis(axis).setTextPen("#9bb0cb")
            self.plot.getAxis(axis).setPen("#3c506c")
            self.plot.getAxis(axis).enableAutoSIPrefix(False)
        canvas.addWidget(self.plot, 1)
        legend = self._label(
            "Blue: selected · Grey: neighbours · Squares: drag · Dashed: vacuum\n"
            "Equal scale on both axes · Scroll to zoom · Drag the background to pan"
        )
        legend.setStyleSheet("color: #9db0c8; font-size: 11px;")
        canvas.addWidget(legend)
        body.addLayout(canvas, 1)

        sidebar = QWidget()
        sidebar.setObjectName("geometryParameterContent")
        sidebar.setStyleSheet("QWidget#geometryParameterContent { background: #101827; }")
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(0, 0, 0, 0)
        side.setSizeConstraint(QLayout.SizeConstraint.SetMinAndMaxSize)
        dimensions = QGroupBox("Dimensions")
        form = QFormLayout(dimensions)
        form.setSpacing(10)
        self.dimension_controls = {}
        self.dimension_labels = {}
        for name, label in (
            ("length_mm", "Axial length"),
            ("inner_diameter_mm", "Inner diameter"),
            ("outer_diameter_mm", "Outer diameter"),
            ("thickness_mm", "Radial thickness"),
        ):
            control = WheelSafeDoubleSpinBox()
            control.setObjectName("geometry_" + name)
            control.setDecimals(6)
            control.setRange(0.0, 1.0e9)
            control.setSingleStep(0.1)
            control.setSuffix(" mm")
            control.setMinimumSize(132, 30)
            control.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            control.setKeyboardTracking(True)
            control.valueChanged.connect(lambda value, field=name: self._dimension_changed(field, value))
            control.editingFinished.connect(self._finish_numeric_edit)
            self.dimension_controls[name] = control
            label_widget = self._label(label)
            self.dimension_labels[name] = label_widget
            control.installEventFilter(self)
            control.lineEdit().installEventFilter(self)
            form.addRow(label_widget, control)
        self.thickness_anchor = WheelSafeComboBox()
        self.thickness_anchor.setObjectName("geometryThicknessAnchor")
        self.thickness_anchor.setMinimumHeight(30)
        self.thickness_anchor.addItem("Keep inner diameter", "inner")
        self.thickness_anchor.addItem("Keep outer diameter", "outer")
        form.addRow("Thickness edit", self.thickness_anchor)
        thickness_note = self._label("Radial thickness = (outer diameter − inner diameter) / 2.")
        thickness_note.setStyleSheet("color: #9db0c8; font-size: 11px;")
        form.addRow(thickness_note)
        self.dimension_source = self._label("")
        self.dimension_source.setObjectName("geometryDimensionSource")
        form.addRow(self.dimension_source)
        side.addWidget(dimensions)

        fixed = QGroupBox("Reference dimensions · read-only")
        fixed_form = QFormLayout(fixed)
        self.center_label = QLabel(f"{self._baseline.center_z_mm:.6g} mm")
        self.start_label = QLabel()
        self.end_label = QLabel()
        self.vacuum_label = QLabel(f"Ø {self._baseline.vacuum_inner_diameter_mm:.6g} mm")
        self.material_label = self._label(str(self._baseline.material_class))
        fixed_form.addRow("Axial centre Z", self.center_label)
        fixed_form.addRow("Start Z", self.start_label)
        fixed_form.addRow("End Z", self.end_label)
        fixed_form.addRow("Vacuum aperture", self.vacuum_label)
        fixed_form.addRow("Material class", self.material_label)
        side.addWidget(fixed)
        side.addWidget(self._label(
            "The vacuum aperture is independent of mechanical diameters. "
            "Apply validates and saves the part dimensions."
        ))
        self.parameter_details = self._label("")
        self.parameter_details.setObjectName("geometryParameterMeaning")
        self.parameter_details.setTextFormat(Qt.TextFormat.PlainText)
        self.parameter_details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.parameter_details.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        side.addWidget(self.parameter_details)
        side.addStretch(1)
        self.parameters_scroll = QScrollArea()
        self.parameters_scroll.setObjectName("geometryParametersScroll")
        self.parameters_scroll.setFixedWidth(342)
        self.parameters_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.parameters_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.parameters_scroll.setStyleSheet("QScrollArea { background: #101827; border: none; }")
        self.parameters_scroll.setWidgetResizable(True)
        self.parameters_scroll.setWidget(sidebar)
        body.addWidget(self.parameters_scroll)
        layout.addLayout(body, 1)

        self.error_label = self._label("")
        self.error_label.setObjectName("geometryInlineError")
        self.error_label.setStyleSheet("color: #ffb5ab; background: #422832; padding: 9px; border-radius: 5px;")
        self.error_label.hide()
        layout.addWidget(self.error_label)
        footer = QHBoxLayout()
        footer_note = QLabel("Changes remain a preview until you apply them.")
        footer.addWidget(footer_note)
        footer.addStretch()
        self.revert_button = self._button("Revert", self.revert)
        self.close_button = self._button("Close", self.reject)
        self.apply_button = self._button("Apply", self.apply)
        self.apply_button.setObjectName("geometryApply")
        for button in (self.revert_button, self.close_button, self.apply_button):
            footer.addWidget(button)
        layout.addLayout(footer)

        for sequence, action in ((QKeySequence.StandardKey.Undo, self.undo), (QKeySequence.StandardKey.Redo, self.redo)):
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.activated.connect(action)
        self._build_scene()
        self._sync()
        self._refresh_parameter_semantics()
        QTimer.singleShot(0, self.fit_view)

    def set_simulation_context(self, mode=None, descriptors=None, by_key=None):
        self._simulation_mode = mode
        self._simulation_descriptors = descriptors
        self._semantic_by_key = dict(by_key or {})
        self._refresh_parameter_semantics()

    def _semantic_path(self, dimension):
        if dimension == "thickness_mm":
            return ("derived", self._baseline.key, "radial_thickness_mm")
        field = {"inner_diameter_mm": "mechanical_inner_diameter_mm",
                 "outer_diameter_mm": "mechanical_outer_diameter_mm"}.get(dimension, dimension)
        return ("parts", self._baseline.key, field)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.FocusIn:
            for name, control in self.dimension_controls.items():
                if watched is control or watched is control.lineEdit():
                    self._semantic_dimension = name
                    self._refresh_parameter_semantics()
                    break
        return super().eventFilter(watched, event)

    def _refresh_parameter_semantics(self):
        if not hasattr(self, "parameter_details"):
            return
        from temsim.parameter_semantics import describe_parameter
        from temsim.parameter_impact import describe_parameter_impact

        for name, control in self.dimension_controls.items():
            meaning = describe_parameter(self._source_part, self._semantic_path(name), by_key=self._semantic_by_key)
            self.dimension_labels[name].setText(meaning.label)
            tooltip = "\n".join(filter(None, (meaning.label, "Category: " + meaning.category_label,
                "Source: " + meaning.source_label + ". " + meaning.source_note, meaning.description)))
            self.dimension_labels[name].setToolTip(tooltip)
            control.setToolTip(tooltip)
            handles = ({"length_mm": ("start", "end"),
                        "inner_diameter_mm": ("inner_top", "inner_bottom"),
                        "outer_diameter_mm": ("outer_top", "outer_bottom")}.get(name, ()))
            for handle in handles:
                gesture = ("Drag axially; the centre stays fixed." if name == "length_mm"
                           else "Drag vertically; the opposite side mirrors this handle.")
                self.handles[handle].setToolTip(tooltip + "\n" + gesture)
        path = self._semantic_path(self._semantic_dimension)
        meaning = describe_parameter(self._source_part, path, by_key=self._semantic_by_key)
        self.dimension_source.setText("Source: " + meaning.source_label)
        self.dimension_source.setToolTip(meaning.source_note)
        impact = describe_parameter_impact(self._source_part, path, by_key=self._semantic_by_key,
            simulation_mode=self._simulation_mode, descriptors=self._simulation_descriptors)
        mode = getattr(self._simulation_mode, "value", self._simulation_mode)
        details = [meaning.label, "Category: " + meaning.category_label,
                   "Source: " + meaning.source_label + ". " + meaning.source_note, meaning.description,
                   f"Simulation mode: {mode if mode is not None else 'not connected'}. "
                   f"Impact: {impact.label}. {impact.detail}"]
        if impact.affected_results:
            details.append("Affected results: " + ", ".join(impact.affected_results))
        self.parameter_details.setText("\n".join(filter(None, details)))
        self.parameter_details.setToolTip(self.parameter_details.text())
        for name, label in (("local_center_z_mm", self.center_label),
                            ("local_start_z_mm", self.start_label), ("local_end_z_mm", self.end_label),
                            ("vacuum_inner_diameter_mm", self.vacuum_label)):
            reference = describe_parameter(self._source_part, ("parts", self._baseline.key, name),
                                           by_key=self._semantic_by_key)
            label.setToolTip("\n".join(filter(None, (reference.label, reference.source_label,
                                                   reference.source_note, reference.description))))

    @staticmethod
    def _label(text):
        label = QLabel(text)
        label.setWordWrap(True)
        return label

    @staticmethod
    def _button(text, slot):
        button = QPushButton(text)
        button.setAutoDefault(False)
        button.clicked.connect(slot)
        return button

    @property
    def geometry(self):
        return self._draft

    def _build_scene(self):
        self.material_items = []
        for geometry in self._neighbours:
            self._material_rects(geometry, "#48566a", "#2a3546", 0)
        self.material_items = self._material_rects(self._draft, "#74c5ff", "#234e74", 2)
        self.center_line = pg.InfiniteLine(
            pos=self._draft.center_z_mm, angle=90,
            pen=pg.mkPen("#71839c", width=1, style=Qt.PenStyle.DashLine),
        )
        self.plot.addItem(self.center_line)
        self.axis_line = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen("#576c85", width=1))
        self.plot.addItem(self.axis_line)
        self.vacuum_lines = []
        for sign in (-1, 1):
            line = pg.InfiniteLine(
                pos=sign * self._draft.vacuum_inner_diameter_mm / 2, angle=0,
                pen=pg.mkPen("#4fd1c5", width=1, style=Qt.PenStyle.DashLine),
            )
            self.plot.addItem(line)
            self.vacuum_lines.append(line)
        self.handles = {}
        for name in ("start", "end", "inner_top", "inner_bottom", "outer_top", "outer_bottom"):
            radial = name.startswith(("inner", "outer"))
            color = "#f5ba64" if radial else "#82ccff"
            handle = pg.TargetItem(
                size=13, symbol="s", pen=pg.mkPen(color, width=2),
                brush="#101827", hoverPen="#ffffff", hoverBrush=color,
            )
            handle.setZValue(20)
            if radial:
                diameter = "Inner" if name.startswith("inner") else "Outer"
                handle.setToolTip(
                    f"{diameter} diameter: drag vertically; the opposite side mirrors this handle."
                )
            else:
                handle.setToolTip("Drag axially to change length while keeping the centre fixed.")
            handle.sigPositionChanged.connect(lambda item, key=name: self._handle_moved(key, item))
            handle.sigPositionChangeFinished.connect(self._finish_drag)
            self.plot.addItem(handle)
            self.handles[name] = handle

    def _material_rects(self, geometry, outline, fill, layer):
        result = []
        for sign in (-1, 1):
            item = QGraphicsRectItem()
            item.setPen(pg.mkPen(outline, width=1.5))
            item.setBrush(pg.mkBrush(fill))
            item.setZValue(layer)
            item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self.plot.addItem(item)
            result.append(item)
        self._set_rects(result, geometry)
        return result

    @staticmethod
    def _set_rects(items, geometry):
        inner = geometry.inner_diameter_mm / 2
        outer = geometry.outer_diameter_mm / 2
        for item, bottom in zip(items, (-outer, inner)):
            item.setRect(geometry.start_z_mm, bottom, geometry.length_mm, outer - inner)

    def _set_error(self, message="", *, invalid_input=False):
        self._input_error = invalid_input
        self.error_label.setText(message)
        self.error_label.setVisible(bool(message))
        self._update_buttons()

    def _sync(self):
        self._syncing = True
        try:
            geometry = self._draft
            for name, control in self.dimension_controls.items():
                control.setValue(getattr(geometry, name))
            self.start_label.setText(f"{geometry.start_z_mm:.6g} mm")
            self.end_label.setText(f"{geometry.end_z_mm:.6g} mm")
            self._set_rects(self.material_items, geometry)
            # Separate inner/outer handles axially so thin windings remain
            # easy to select even when their radial separation is a few pixels.
            inner_z = geometry.start_z_mm + geometry.length_mm * 0.35
            outer_z = geometry.start_z_mm + geometry.length_mm * 0.65
            positions = {
                "start": (geometry.start_z_mm, 0), "end": (geometry.end_z_mm, 0),
                "inner_top": (inner_z, geometry.inner_diameter_mm / 2),
                "inner_bottom": (inner_z, -geometry.inner_diameter_mm / 2),
                "outer_top": (outer_z, geometry.outer_diameter_mm / 2),
                "outer_bottom": (outer_z, -geometry.outer_diameter_mm / 2),
            }
            for name, position in positions.items():
                self.handles[name].setPos(position)
            self.handles["start"].movable = geometry.center_fraction > 0
            self.handles["end"].movable = geometry.center_fraction < 1
        finally:
            self._syncing = False
        self._update_buttons()

    def _update_buttons(self):
        dirty = self._draft != self._baseline
        self.status_label.setText("Unapplied preview" if dirty or self._input_error else "Applied dimensions")
        self.apply_button.setEnabled(dirty and not self._input_error)
        self.revert_button.setEnabled(dirty or self._input_error or bool(self.error_label.text()))
        self.undo_button.setEnabled(self._history_index > 0 or self._numeric_origin is not None)
        self.redo_button.setEnabled(self._history_index + 1 < len(self._history))

    def _dimension_changed(self, name, value):
        if self._syncing:
            return
        if self._numeric_origin is None:
            self._numeric_origin = self._draft
        try:
            self._draft = self._draft.with_dimension(
                name, value, thickness_anchor=self.thickness_anchor.currentData()
            )
        except (TypeError, ValueError, OverflowError) as exc:
            self._set_error(str(exc), invalid_input=True)
            return
        self._set_error()
        self._sync()
        self.fit_view()

    def _remember(self, origin):
        if origin is not None and self._draft != origin:
            self._history = self._history[:self._history_index + 1]
            self._history.append(self._draft)
            self._history_index += 1
        self._update_buttons()

    def _finish_numeric_edit(self):
        origin, self._numeric_origin = self._numeric_origin, None
        self._remember(origin)

    def _handle_moved(self, name, item):
        if self._syncing:
            return
        if self._drag_origin is None:
            self._finish_numeric_edit()
            self._drag_origin = self._draft
        origin = self._drag_origin
        try:
            if name == "start":
                value = (origin.center_z_mm - item.pos().x()) / origin.center_fraction
                dimension = "length_mm"
            elif name == "end":
                value = (item.pos().x() - origin.center_z_mm) / (1 - origin.center_fraction)
                dimension = "length_mm"
            else:
                dimension = "inner_diameter_mm" if name.startswith("inner") else "outer_diameter_mm"
                value = abs(item.pos().y()) * 2
            self._draft = origin.with_dimension(dimension, value)
        except (TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
            self._set_error(str(exc))
        else:
            self._set_error()
        self._sync()

    def _finish_drag(self, _item):
        origin, self._drag_origin = self._drag_origin, None
        self._remember(origin)

    def undo(self):
        self._finish_numeric_edit()
        if self._history_index > 0:
            self._history_index -= 1
            self._draft = self._history[self._history_index]
            self._set_error()
            self._sync()

    def redo(self):
        self._finish_numeric_edit()
        if self._history_index + 1 < len(self._history):
            self._history_index += 1
            self._draft = self._history[self._history_index]
            self._set_error()
            self._sync()

    def revert(self):
        self._draft = self._baseline
        self._numeric_origin = self._drag_origin = None
        self._history = [self._baseline]
        self._history_index = 0
        self._set_error()
        self._sync()
        self.fit_view()

    def apply(self):
        self._finish_numeric_edit()
        if self._input_error or self._draft == self._baseline:
            return
        try:
            self._apply_changes(dict(self._draft.updates_from(self._baseline)))
        except Exception as exc:
            self._set_error(f"Unable to apply dimensions: {exc}")
            return
        self._baseline = self._draft
        self._set_error()
        self._sync()

    def reject(self):
        self.revert()
        super().reject()

    def fit_view(self):
        geometries = [self._draft, *self._neighbours]
        start = min(item.start_z_mm for item in geometries)
        end = max(item.end_z_mm for item in geometries)
        radius = max(item.outer_diameter_mm for item in geometries) / 2
        span = max(end - start, 1.0)
        margin = max(span * 0.12, radius * 0.2, 1.0)
        self.plot.setRange(
            xRange=(start - margin, end + margin),
            yRange=(-radius - margin, radius + margin), padding=0,
        )


__all__ = ("GeometryEditorDialog",)
