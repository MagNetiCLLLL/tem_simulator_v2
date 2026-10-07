"""Rotatable geometric sections of the saved assembly, independent of optics.

The plane passes through the nominal column axis. Its transverse coordinate is
U = X cos(a) + Y sin(a); V = -X sin(a) + Y cos(a) is zero on the plane.
Geometry is built only when visible and angle changes reuse the same meshes.
"""

from collections import OrderedDict
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton,
    QSlider, QVBoxLayout, QWidget,
)

from temsim.assembly_section import assembly_section_from_model
from temsim.cell_geometry import CellPhysicalContext
from temsim.gui.assembly_geometry_cache import AssemblyGeometryCache


class AssemblySectionPage(QWidget):
    component_selected = Signal(str)
    edit_part_requested = Signal(str, float)
    projection_angle_changed = Signal(float)

    def __init__(self, parent=None, *, geometry_cache=None):
        super().__init__(parent)
        self.setObjectName("assemblySectionPage")
        self._geometry_cache = geometry_cache if geometry_cache is not None else AssemblyGeometryCache()
        self._assembly = None
        self._runtime_values = {}
        self._cell_context = CellPhysicalContext()
        self._cell_error = None
        self._assembly_model = None
        self._model = None
        self._section = None
        self._fingerprint = None
        self._parts = {}
        self._editable_keys = set()
        self._selected_key = None
        self._angle_deg = 0.0
        self._last_ray_angle_deg = 0.0
        self._pending = False
        self._geometry_pending = False
        self._has_fitted = False
        self._sections = OrderedDict()
        self._curves = []
        self.mesh_builds = 0
        self.section_builds = 0
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self._flush)

        self.xz_button = QPushButton("X-Z")
        self.yz_button = QPushButton("Y-Z")
        for button in (self.xz_button, self.yz_button):
            button.setCheckable(True)
        self.xz_button.setChecked(True)
        self.angle_slider = QSlider(Qt.Orientation.Horizontal)
        self.angle_slider.setRange(0, 3600)
        self.angle_slider.setPageStep(50)
        self.angle_slider.setMinimumWidth(100)
        self.angle_slider.setMaximumWidth(260)
        self.angle_spin = QDoubleSpinBox()
        self.angle_spin.setRange(0.0, 360.0)
        self.angle_spin.setDecimals(1)
        self.angle_spin.setSingleStep(1.0)
        self.angle_spin.setSuffix(" °")
        self.angle_spin.setKeyboardTracking(False)
        self.follow_ray_diagram = QCheckBox("Follow Ray Diagram")
        self.follow_ray_diagram.setChecked(True)
        self.follow_ray_diagram.setToolTip(
            "Share the observation angle with Ray Diagram. Uncheck for an independent section. "
            "This changes the view, not the physical placement or particle calculation."
        )
        self.component_combo = QComboBox()
        self.component_combo.setMinimumWidth(160)
        self.component_combo.setMaximumWidth(350)
        self.component_combo.setToolTip("Select a component to highlight its section contours.")
        self.fit_all = QPushButton("Fit assembly")
        self.fit_selected = QPushButton("Fit selected")
        self.edit_part = QPushButton("Open in 3D Parts")
        self.edit_part.setEnabled(False)
        self.envelopes = QCheckBox("Schematic envelopes")
        self.envelopes.setChecked(True)
        self.envelopes.setToolTip("Include approximate envelopes where no detailed solid is configured.")
        self.equal_scale = QCheckBox("Equal scale")
        self.equal_scale.setToolTip("Use the same millimetre scale horizontally and vertically.")
        self.plane_label = QLabel()
        self.selection_label = QLabel("Select a component")
        self.status = QLabel("Saved assembly section | no calculation required")
        for label in (self.plane_label, self.selection_label, self.status):
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.view = pg.PlotWidget(background="#070b17")
        self.view.setObjectName("assemblySectionView")
        self.view.setLabel("bottom", "Axial position Z", units="mm")
        self.view.setLabel("left", "Section coordinate X", units="mm")
        self.view.getAxis("bottom").enableAutoSIPrefix(False)
        self.view.getAxis("left").enableAutoSIPrefix(False)
        self.view.showGrid(x=True, y=True, alpha=0.12)
        self.view.setMenuEnabled(False)
        self.view.getViewBox().setAspectLocked(False)
        self.view.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen("#677181", style=Qt.PenStyle.DashLine)))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        angles = QHBoxLayout()
        for widget in (self.xz_button, self.yz_button, QLabel("Angle"), self.angle_slider,
                       self.angle_spin, self.follow_ray_diagram):
            angles.addWidget(widget)
        angles.addStretch(1)
        actions = QHBoxLayout()
        for widget in (self.component_combo, self.fit_all, self.fit_selected, self.edit_part):
            actions.addWidget(widget)
        actions.addStretch(1)
        options = QHBoxLayout()
        options.addWidget(self.envelopes)
        options.addWidget(self.equal_scale)
        options.addStretch(1)
        layout.addLayout(angles)
        layout.addLayout(actions)
        layout.addLayout(options)
        layout.addWidget(self.plane_label)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.selection_label)
        layout.addWidget(self.status)

        self.xz_button.clicked.connect(lambda: self._set_angle(0.0, user=True))
        self.yz_button.clicked.connect(lambda: self._set_angle(90.0, user=True))
        self.angle_slider.valueChanged.connect(lambda value: self._set_angle(value / 10.0, user=True))
        self.angle_spin.valueChanged.connect(lambda value: self._set_angle(value, user=True))
        self.follow_ray_diagram.toggled.connect(self._follow_changed)
        self.component_combo.currentIndexChanged.connect(self._component_changed)
        self.fit_all.clicked.connect(self._fit_all)
        self.fit_selected.clicked.connect(self.fit_current_selection)
        self.edit_part.clicked.connect(self._open_selected_part)
        self.envelopes.toggled.connect(self._display_section)
        self.equal_scale.toggled.connect(lambda checked: self.view.getViewBox().setAspectLocked(checked))
        self._update_plane_label()

    def set_assembly(self, assembly, runtime_values=None):
        self._assembly = assembly
        self._runtime_values = deepcopy(runtime_values or {})
        self._geometry_pending = True
        self._schedule()

    def set_runtime_values(self, runtime_values):
        self.set_assembly(self._assembly, runtime_values)

    def set_cell_context(self, context, error=None):
        self._cell_context = context
        self._cell_error = error
        self._geometry_pending = True
        self._schedule()

    def set_projection_angle(self, angle):
        """Receive a Ray Diagram viewing angle without echoing a user signal."""
        angle = float(angle)
        if not np.isfinite(angle):
            return
        self._last_ray_angle_deg = float(np.clip(angle, 0.0, 360.0))
        if self.follow_ray_diagram.isChecked():
            self._set_angle(self._last_ray_angle_deg)

    def _follow_changed(self, checked):
        if checked:
            self._set_angle(self._last_ray_angle_deg)

    def _set_angle(self, angle, *, user=False):
        angle = round(float(np.clip(angle, 0.0, 360.0)), 1)
        changed = angle != self._angle_deg
        self._angle_deg = angle
        for widget, value in ((self.angle_slider, round(angle * 10)), (self.angle_spin, angle)):
            with QSignalBlocker(widget):
                widget.setValue(value)
        self.xz_button.setChecked(angle % 360.0 == 0.0)
        self.yz_button.setChecked(angle == 90.0)
        self._update_plane_label()
        if changed:
            self._schedule()
            if user and self.follow_ray_diagram.isChecked():
                self.projection_angle_changed.emit(angle)

    def _update_plane_label(self):
        angle = self._angle_deg
        axis = "X" if angle % 360.0 == 0.0 else "Y" if angle == 90.0 else "U"
        self.view.setLabel("left", f"Section coordinate {axis}", units="mm")
        self.plane_label.setText(f"{axis}-Z section | {angle:g}° | Through the nominal column axis")
        self.plane_label.setToolTip(
            "Intersections with saved three-dimensional meshes, not a projection of all surfaces. "
            "U = X cos(angle) + Y sin(angle); V = -X sin(angle) + Y cos(angle) = 0. "
            "The plane always contains the nominal Z axis; "
            "offset parts outside this plane have no contour. Angle does not rotate hardware."
        )

    def _schedule(self):
        self._pending = True
        if self.isVisible() and not self._refresh_timer.isActive():
            self._refresh_timer.start(60)

    def showEvent(self, event):
        super().showEvent(event)
        if self._pending:
            self._refresh_timer.start(0)

    def hideEvent(self, event):
        self._refresh_timer.stop()
        super().hideEvent(event)

    def _flush(self):
        if not self._pending or not self.isVisible():
            return
        self._pending = False
        try:
            if self._assembly is None:
                self._clear_model("No saved assembly available")
                return
            if self._geometry_pending:
                base = self._geometry_cache.model_for(self._assembly, self._runtime_values)
                fingerprint = (id(base), self._cell_context.signature(), self._cell_error)
                if fingerprint != self._fingerprint:
                    model = replace(base, meshes=base.meshes + self._cell_context.meshes(),
                                    errors=base.errors + ((self._cell_error,) if self._cell_error else ()))
                    self._assembly_model = base
                    self._model, self._fingerprint = model, fingerprint
                    self.mesh_builds += 1
                    self._sections.clear()
                    self._populate_components()
                self._geometry_pending = False
            if self._model is None:
                return
            key = self._angle_deg % 360.0
            section = self._sections.get(key)
            if section is None:
                section = assembly_section_from_model(self._model, key)
                self._sections[key] = section
                self.section_builds += 1
                while len(self._sections) > 8:
                    self._sections.popitem(last=False)
            self._sections.move_to_end(key)
            self._section = section
            self._display_section()
        except Exception as exc:
            self._clear_model(f"Assembly section unavailable: {exc}")

    def _clear_model(self, message):
        self._model = self._section = self._fingerprint = None
        self._assembly_model = None
        self._parts = {}
        self._editable_keys = set()
        self._sections.clear()
        self._geometry_pending = True
        self._clear_curves()
        with QSignalBlocker(self.component_combo):
            self.component_combo.clear()
        self.edit_part.setEnabled(False)
        self.selection_label.setText("Select a component")
        self.status.setText(message)

    def _populate_components(self):
        self._parts = {part.key: part for part in self._assembly.parts}
        self._editable_keys = set(self._parts)
        self._parts.update({part.key: part for part in self._cell_context.parts()})
        self._editable_keys.update(layer.key for layer in self._cell_context.layers)
        for liner in getattr(self._assembly, "vacuum_liner_segments", ()):
            self._parts.setdefault(liner.key, SimpleNamespace(
                key=liner.key, name=liner.name, center_z_mm=(liner.start_z_mm + liner.end_z_mm) / 2,
                data={},
            ))
        with QSignalBlocker(self.component_combo):
            self.component_combo.clear()
            self.component_combo.addItem("Select a component…", None)
            for part in sorted(self._parts.values(), key=lambda p: (p.center_z_mm, p.key)):
                self.component_combo.addItem(part.name, part.key)
                self.component_combo.setItemData(self.component_combo.count() - 1,
                                                f"{part.key} | Z {part.center_z_mm:g} mm",
                                                Qt.ItemDataRole.ToolTipRole)
        self.focus_component(self._selected_key)

    def _clear_curves(self):
        for _section, curve in self._curves:
            self.view.removeItem(curve)
        self._curves.clear()

    def _display_section(self, *_):
        if self._section is None:
            return
        self._clear_curves()
        for mesh in self._section.meshes:
            if not self.envelopes.isChecked() and not mesh.is_exact:
                continue
            coordinates = mesh.segments_mm.reshape(-1, 2)
            if not len(coordinates):
                continue
            curve = pg.PlotCurveItem(coordinates[:, 0], coordinates[:, 1], connect="pairs",
                                     antialias=False)
            curve.setClickable(True, width=7)
            curve.sigClicked.connect(lambda _curve, _event, key=mesh.key: self._surface_selected(key))
            curve.setToolTip(f"{mesh.key} / {mesh.region}\n{mesh.material_class}\n{mesh.description}")
            self.view.addItem(curve)
            self._curves.append((mesh, curve))
        self._style_selection()
        if not self._has_fitted and self._curves:
            self._fit_all()
            self._has_fitted = True
        count = len({mesh.key for mesh, _curve in self._curves})
        approximate = len({mesh.key for mesh, _curve in self._curves if not mesh.is_exact})
        errors = f" | {len(self._section.errors)} unavailable" if self._section.errors else ""
        self.status.setText(
            f"Saved mesh section | {count} intersected parts | {approximate} schematic envelopes | "
            f"contours only | +Z downstream{errors}"
        )
        self.status.setToolTip("\n".join((
            "True plane intersections of the configured meshes, with angular tessellation approximation.",
            "Configured geometry is not a claim of OEM dimensional accuracy. Holes remain open.",
            "Schematic envelopes use dashed contours. Unsaved 3D Parts drafts are excluded.",
            "No particle, field or image calculation is started by this view.",
            *self._section.notes, *self._section.errors,
        )))

    def _related_keys(self, key):
        keys = {key} if key else set()
        while True:
            more = {part.key for part in self._parts.values()
                    if part.data.get("parent_key") in keys
                    or keys.intersection(part.data.get("magnetic_lens_keys", ()))}
            if more <= keys:
                return keys
            keys.update(more)

    def _style_selection(self):
        selected = self._related_keys(self._selected_key)
        for mesh, curve in self._curves:
            color = tuple(int(round(np.clip(channel, 0, 1) * 255)) for channel in mesh.color[:3])
            curve.setPen(pg.mkPen(color, width=2.5 if mesh.key in selected else 1.3,
                                 style=Qt.PenStyle.SolidLine if mesh.is_exact else Qt.PenStyle.DashLine))

    def focus_component(self, part_or_key):
        self._selected_key = getattr(part_or_key, "key", part_or_key)
        with QSignalBlocker(self.component_combo):
            index = self.component_combo.findData(self._selected_key)
            self.component_combo.setCurrentIndex(max(0, index))
        selected = self._parts.get(self._selected_key)
        self.edit_part.setEnabled(self._selected_key in self._editable_keys)
        self.edit_part.setText("Edit cell / windows" if selected and selected.data.get("cell_context")
                               else "Open in 3D Parts")
        self.selection_label.setText(f"{selected.name} | Z {selected.center_z_mm:g} mm"
                                     if selected else "Select a component")
        self._style_selection()

    def _component_changed(self, index):
        self._surface_selected(self.component_combo.itemData(index))

    def _surface_selected(self, key):
        self.focus_component(key)
        if key in self._editable_keys:
            self.component_selected.emit(key)

    def _open_selected_part(self):
        part = self._parts.get(self._selected_key)
        if part is not None and part.key in self._editable_keys:
            self.edit_part_requested.emit(part.key, float(part.center_z_mm))

    def _fit_all(self):
        self._fit_curves(self._curves)

    def _fit_curves(self, curves):
        bounds = [mesh.segments_mm.reshape(-1, 2) for mesh, _curve in curves if len(mesh.segments_mm)]
        if not bounds:
            return False
        coordinates = np.concatenate(bounds)
        low, high = coordinates.min(axis=0), coordinates.max(axis=0)
        span = np.maximum(high - low, 1e-6)
        self.view.setRange(xRange=(low[0] - .05 * span[0], high[0] + .05 * span[0]),
                           yRange=(low[1] - .05 * span[1], high[1] + .05 * span[1]), padding=0)
        return True

    def fit_current_selection(self):
        self._flush()
        keys = self._related_keys(self._selected_key)
        fitted = self._fit_curves([(mesh, curve) for mesh, curve in self._curves if mesh.key in keys])
        if not fitted:
            self.selection_label.setText("No visible intersection for this component at the current angle.")
        return fitted
