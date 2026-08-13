"""Central finite-sample editor and ball-and-stick structure workspace."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication, QVector3D
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from temsim.specimen.geometry import (
    IDENTITY_QUATERNION_WXYZ,
    build_sample_geometry_snapshot,
    quaternion_from_euler_xyz_deg,
    quaternion_from_zone_axes,
    quaternion_multiply,
    quaternion_to_matrix,
    sample_orientation_quaternion,
    set_sample_orientation,
)
from temsim.specimen.presets import available_specimen_presets
from temsim.specimen.virtual import resolve_virtual_interactions


try:
    if os.environ.get("TEMSIM_DISABLE_OPENGL", "").strip() == "1":
        raise ImportError("OpenGL disabled by TEMSIM_DISABLE_OPENGL")
    import pyqtgraph.opengl as gl
except Exception as _opengl_import_error:  # pragma: no cover - platform dependent
    gl = None
    OPENGL_IMPORT_ERROR = str(_opengl_import_error)
else:
    OPENGL_IMPORT_ERROR = None


if gl is not None:
    class _EditableGLView(gl.GLViewWidget):
        orientation_dragged = Signal(float, float)

        def __init__(self, parent=None):
            super().__init__(parent)
            self.edit_orientation = False
            self._edit_position = None

        def mousePressEvent(self, event):
            if self.edit_orientation and event.button() == Qt.MouseButton.LeftButton:
                self._edit_position = event.position()
                event.accept()
                return
            super().mousePressEvent(event)

        def mouseMoveEvent(self, event):
            if self.edit_orientation and self._edit_position is not None:
                position = event.position()
                delta = position - self._edit_position
                self._edit_position = position
                self.orientation_dragged.emit(float(delta.x()), float(delta.y()))
                event.accept()
                return
            super().mouseMoveEvent(event)

        def mouseReleaseEvent(self, event):
            if self.edit_orientation and self._edit_position is not None:
                self._edit_position = None
                event.accept()
                return
            super().mouseReleaseEvent(event)


def _rectangle_lines(bounds, z):
    x0, x1, y0, y1 = bounds
    return np.asarray(
        (
            (x0, y0, z),
            (x1, y0, z),
            (x1, y1, z),
            (x0, y1, z),
            (x0, y0, z),
        ),
        dtype=float,
    )


def _box_lines(centre, size):
    cx, cy, cz = centre
    hx, hy, hz = (0.5 * value for value in size)
    corners = np.asarray(
        [
            (cx + sx * hx, cy + sy * hy, cz + sz * hz)
            for sz in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sx in (-1.0, 1.0)
        ]
    )
    edges = (
        (0, 1), (0, 2), (1, 3), (2, 3),
        (4, 5), (4, 6), (5, 7), (6, 7),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    points = []
    for first, second in edges:
        points.extend((corners[first], corners[second]))
    return np.asarray(points, dtype=float)


def _cell_lines(vectors):
    vectors = np.asarray(vectors, dtype=float)
    corners = np.asarray(
        [
            i * vectors[0] + j * vectors[1] + k * vectors[2]
            for k in (0.0, 1.0)
            for j in (0.0, 1.0)
            for i in (0.0, 1.0)
        ]
    )
    corners -= 0.5 * np.sum(vectors, axis=0)
    edges = (
        (0, 1), (0, 2), (1, 3), (2, 3),
        (4, 5), (4, 6), (5, 7), (6, 7),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    result = []
    for first, second in edges:
        result.extend((corners[first], corners[second]))
    return np.asarray(result)


def _region_outline(region, z=0.0):
    sx, sy = region.size_nm
    if region.kind == "ellipse":
        phase = np.linspace(0.0, 2.0 * math.pi, 129)
        xy = np.column_stack((0.5 * sx * np.cos(phase), 0.5 * sy * np.sin(phase)))
    else:
        xy = np.asarray(
            ((-0.5 * sx, -0.5 * sy), (0.5 * sx, -0.5 * sy),
             (0.5 * sx, 0.5 * sy), (-0.5 * sx, 0.5 * sy),
             (-0.5 * sx, -0.5 * sy)),
            dtype=float,
        )
    angle = math.radians(region.rotation_deg)
    rotation = np.asarray(
        ((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle)))
    )
    xy = xy @ rotation.T + np.asarray(region.centre_nm)
    return np.column_stack((xy, np.full(xy.shape[0], float(z))))


def _atomic_colours(numbers):
    """Return ASE/Jmol element colours as RGBA floats."""

    from ase.data.colors import jmol_colors

    result = []
    for number in np.asarray(numbers, dtype=int):
        if 0 < number < len(jmol_colors):
            red, green, blue = jmol_colors[number]
        else:
            red, green, blue = (0.35, 0.8, 0.65)
        result.append((float(red), float(green), float(blue), 1.0))
    return np.asarray(result, dtype=float)


def _atomic_radii_nm(numbers):
    """Return reduced covalent radii for a conventional ball-stick view."""

    from ase.data import covalent_radii

    radii = []
    for number in np.asarray(numbers, dtype=int):
        radius_angstrom = (
            float(covalent_radii[number])
            if 0 < number < len(covalent_radii)
            else 1.0
        )
        radii.append(max(0.42 * radius_angstrom * 0.1, 0.018))
    return np.asarray(radii, dtype=float)


def _bond_line_data(positions, bonds, colours):
    """Split every bond at its midpoint so each half matches its atom."""

    positions = np.asarray(positions, dtype=float)
    bonds = np.asarray(bonds, dtype=int)
    colours = np.asarray(colours, dtype=float)
    if bonds.size == 0:
        return np.empty((0, 3), dtype=float), np.empty((0, 4), dtype=float)
    first = positions[bonds[:, 0]]
    second = positions[bonds[:, 1]]
    middle = 0.5 * (first + second)
    points = np.stack((first, middle, middle, second), axis=1).reshape(-1, 3)
    line_colours = np.stack(
        (
            colours[bonds[:, 0]],
            colours[bonds[:, 0]],
            colours[bonds[:, 1]],
            colours[bonds[:, 1]],
        ),
        axis=1,
    ).reshape(-1, 4)
    return points, line_colours


class ElementLegend(QScrollArea):
    """Element-colour key shown beside the sample structure."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sampleElementLegend")
        self.setWidgetResizable(True)
        self.setMinimumWidth(150)
        self.setMaximumWidth(230)
        self.body = QWidget()
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(8, 8, 8, 8)
        self.setWidget(self.body)
        self.set_atomic_numbers(())

    def set_atomic_numbers(self, numbers):
        while self.body_layout.count():
            item = self.body_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        title = QLabel("Displayed atoms")
        title.setStyleSheet("font-weight: 600;")
        self.body_layout.addWidget(title)
        values, counts = np.unique(np.asarray(numbers, dtype=int), return_counts=True)
        if not values.size:
            empty = QLabel("No atomic structure")
            empty.setWordWrap(True)
            empty.setStyleSheet("color: #64748b;")
            self.body_layout.addWidget(empty)
        else:
            from ase.data import atomic_names, chemical_symbols

            colours = _atomic_colours(values)
            for number, count, colour in zip(values, counts, colours):
                row = QWidget()
                layout = QHBoxLayout(row)
                layout.setContentsMargins(0, 2, 0, 2)
                swatch = QLabel("●")
                red, green, blue = (
                    int(round(255.0 * float(value))) for value in colour[:3]
                )
                swatch.setStyleSheet(
                    f"color: rgb({red}, {green}, {blue}); font-size: 20pt;"
                )
                symbol = chemical_symbols[int(number)]
                name = atomic_names[int(number)].title()
                text = QLabel(f"{symbol} — {name}\n{int(count):,} shown")
                text.setWordWrap(True)
                layout.addWidget(swatch)
                layout.addWidget(text, 1)
                self.body_layout.addWidget(row)
        note = QLabel("Colours: ASE/Jmol convention")
        note.setWordWrap(True)
        note.setStyleSheet("color: #64748b; font-size: 9pt;")
        self.body_layout.addStretch(1)
        self.body_layout.addWidget(note)


class SampleSceneView(QWidget):
    """OpenGL sample scene with a deterministic 2-D fallback."""

    orientation_dragged = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.opengl_available = False
        self.opengl_detail = OPENGL_IMPORT_ERROR
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        platform_name = str(QGuiApplication.platformName()).lower()
        platform_supports_gl = platform_name not in {"offscreen", "minimal"}
        if gl is not None and platform_supports_gl:
            try:
                self.view = _EditableGLView(self)
                self.view.setObjectName("sampleOpenGlView")
                self.view.setBackgroundColor(QColor("#050816"))
                self.view.orientation_dragged.connect(self.orientation_dragged)
                self.opengl_available = True
                self.opengl_detail = "pyqtgraph.opengl / PyOpenGL"
            except Exception as exc:  # pragma: no cover - driver dependent
                self.opengl_detail = f"OpenGL initialisation failed: {exc}"
                self.view = self._fallback_plot()
        else:
            if not platform_supports_gl:
                self.opengl_detail = (
                    f"Qt platform {platform_name!r} has no supported OpenGL widget"
                )
                if OPENGL_IMPORT_ERROR:
                    self.opengl_detail += (
                        f"; OpenGL import unavailable: {OPENGL_IMPORT_ERROR}"
                    )
            self.view = self._fallback_plot()
        layout.addWidget(self.view, 1)
        self._items = []

    def _fallback_plot(self):
        plot = pg.PlotWidget(background="#050816")
        plot.setObjectName("sampleFallback2DView")
        plot.setLabel("bottom", "Laboratory X", units="nm")
        plot.setLabel("left", "Laboratory Y", units="nm")
        plot.showGrid(x=True, y=True, alpha=0.2)
        plot.getViewBox().setAspectLocked(True)
        return plot

    def set_edit_orientation(self, enabled):
        if self.opengl_available:
            self.view.edit_orientation = bool(enabled)

    def clear(self):
        if self.opengl_available:
            for item in self._items:
                try:
                    self.view.removeItem(item)
                except Exception:
                    pass
        else:
            self.view.clear()
        self._items = []

    def _add_gl_line(self, positions, colour, width=2.0, mode="line_strip"):
        item = gl.GLLinePlotItem(
            pos=np.asarray(positions, dtype=float),
            color=np.asarray(colour, dtype=float),
            width=float(width),
            antialias=True,
            mode=mode,
        )
        self.view.addItem(item)
        self._items.append(item)

    def _add_gl_atoms(self, positions, numbers):
        positions = np.asarray(positions, dtype=float)
        colours = _atomic_colours(numbers)
        radii = _atomic_radii_nm(numbers)
        if len(positions) <= 3_000:
            template = gl.MeshData.sphere(rows=5, cols=8, radius=1.0)
            vertices = np.asarray(template.vertexes(), dtype=np.float32)
            faces = np.asarray(template.faces(), dtype=np.int32)
            vertex_count = vertices.shape[0]
            expanded_vertices = (
                positions[:, None, :]
                + radii[:, None, None] * vertices[None, :, :]
            ).reshape(-1, 3).astype(np.float32, copy=False)
            expanded_faces = (
                faces[None, :, :]
                + np.arange(len(positions), dtype=np.int32)[:, None, None]
                * vertex_count
            ).reshape(-1, 3)
            vertex_colours = np.repeat(
                colours.astype(np.float32),
                vertex_count,
                axis=0,
            )
            mesh_data = gl.MeshData(
                vertexes=expanded_vertices,
                faces=expanded_faces,
                vertexColors=vertex_colours,
            )
            item = gl.GLMeshItem(
                meshdata=mesh_data,
                smooth=True,
                drawEdges=False,
                shader="shaded",
                glOptions="opaque",
            )
        else:
            # Large user-selected display windows use GPU point sprites as an
            # explicit level of detail; colours and physical diameters remain
            # element-specific while avoiding millions of triangle faces.
            item = gl.GLScatterPlotItem(
                pos=positions,
                color=colours,
                size=2.0 * radii,
                pxMode=False,
            )
        self.view.addItem(item)
        self._items.append(item)

    def display_snapshot(self, snapshot, *, draft_quaternion=None):
        self.clear()
        target_rotation = quaternion_to_matrix(
            draft_quaternion
            if draft_quaternion is not None
            else snapshot.orientation_quaternion_wxyz
        )
        if self.opengl_available:
            self._display_gl(snapshot, target_rotation)
        else:
            self._display_2d(snapshot, target_rotation)

    @staticmethod
    def _oriented_atoms(snapshot, target_rotation):
        positions = np.asarray(snapshot.atom_positions_nm, dtype=float)
        if not positions.size:
            return positions
        current = np.asarray(snapshot.orientation_matrix, dtype=float)
        return positions @ current @ np.asarray(target_rotation, dtype=float).T

    def _display_gl(self, snapshot, target_rotation):
        cx, cy, cz = snapshot.centre_nm
        sx, sy, sz = snapshot.size_nm
        if snapshot.atom_display_size_nm is not None:
            display_size = np.asarray(snapshot.atom_display_size_nm, dtype=float)
            display_centre = np.asarray(snapshot.atom_display_centre_nm, dtype=float)
        else:
            display_size = np.asarray((sx, sy, sz), dtype=float)
            display_centre = np.asarray((cx, cy, cz), dtype=float)
        scale = max(float(np.max(display_size)), 1.0e-3)
        if max(sx, sy, sz) <= 16.0 * scale:
            box = _box_lines((0.0, 0.0, 0.0), (sx, sy, sz))
            box = box @ np.asarray(target_rotation, dtype=float).T
            box += np.asarray((cx, cy, cz))
            self._add_gl_line(
                box,
                (0.3, 0.75, 0.95, 0.9),
                mode="lines",
            )
        if snapshot.atom_display_size_nm is not None:
            self._add_gl_line(
                _box_lines(display_centre, display_size),
                (0.75, 0.55, 1.0, 0.85),
                width=2.0,
                mode="lines",
            )
        beam_x, beam_y = (
            snapshot.current_probe_nm
            if snapshot.current_probe_nm is not None
            else tuple(display_centre[:2])
        )
        self._add_gl_line(
            ((beam_x, beam_y, -0.8 * scale), (beam_x, beam_y, 0.8 * scale)),
            (1.0, 0.85, 0.2, 0.9),
            width=3.0,
        )
        if snapshot.scan_fov_bounds_nm is not None:
            self._add_gl_line(
                _rectangle_lines(snapshot.scan_fov_bounds_nm, 0.52 * sz),
                (0.2, 1.0, 0.45, 0.95),
                width=3.0,
            )
        if snapshot.calculation_roi_bounds_nm is not None:
            self._add_gl_line(
                _rectangle_lines(snapshot.calculation_roi_bounds_nm, 0.56 * sz),
                (1.0, 0.45, 0.75, 0.9),
                width=2.0,
            )
        for region in snapshot.regions:
            if not region.enabled:
                continue
            self._add_gl_line(
                _region_outline(region, 0.58 * sz),
                (1.0, 0.35 + 0.55 * region.density, 0.2, 0.9),
                width=2.0,
            )
        if snapshot.current_probe_nm is not None:
            px, py = snapshot.current_probe_nm
            probe = gl.GLScatterPlotItem(
                pos=np.asarray(((px, py, 0.6 * sz),)),
                color=np.asarray(((1.0, 1.0, 0.2, 1.0),)),
                size=max(scale * 0.012, 3.0),
                pxMode=False,
            )
            self.view.addItem(probe)
            self._items.append(probe)
        if snapshot.atom_positions_nm.size:
            positions = self._oriented_atoms(snapshot, target_rotation).copy()
            positions[:, 0] += cx
            positions[:, 1] += cy
            colours = _atomic_colours(snapshot.atomic_numbers)
            bond_points, bond_colours = _bond_line_data(
                positions,
                snapshot.atom_bond_pairs,
                colours,
            )
            if bond_points.size:
                self._add_gl_line(
                    bond_points,
                    bond_colours,
                    width=2.5,
                    mode="lines",
                )
            self._add_gl_atoms(positions, snapshot.atomic_numbers)
        if snapshot.cell_vectors_nm.shape == (3, 3):
            current = np.asarray(snapshot.orientation_matrix, dtype=float)
            target_cell = (
                np.asarray(snapshot.cell_vectors_nm, dtype=float)
                @ current
                @ np.asarray(target_rotation, dtype=float).T
            )
            cell = _cell_lines(target_cell) + display_centre
            self._add_gl_line(
                cell,
                (0.75, 0.55, 1.0, 0.9),
                width=2.0,
                mode="lines",
            )
        self.view.opts["distance"] = 1.8 * scale
        self.view.opts["center"] = QVector3D(*display_centre)
        self.view.update()

    def _display_2d(self, snapshot, target_rotation):
        cx, cy, _cz = snapshot.centre_nm
        sx, sy, sz = snapshot.size_nm
        if snapshot.atom_display_size_nm is not None:
            display_size = np.asarray(snapshot.atom_display_size_nm, dtype=float)
            display_centre = np.asarray(snapshot.atom_display_centre_nm, dtype=float)
        else:
            display_size = np.asarray((sx, sy, sz), dtype=float)
            display_centre = np.asarray((cx, cy, 0.0), dtype=float)
        scale = max(float(np.max(display_size[:2])), 1.0e-3)
        if max(sx, sy) <= 16.0 * scale:
            box = _box_lines((0.0, 0.0, 0.0), snapshot.size_nm)
            box = box @ np.asarray(target_rotation, dtype=float).T
            self.view.plot(
                box[:, 0] + cx,
                box[:, 1] + cy,
                pen=pg.mkPen("#38bdf8", width=2),
                connect="pairs",
            )
        if snapshot.atom_display_size_nm is not None:
            display_box = _box_lines(display_centre, display_size)
            self.view.plot(
                display_box[:, 0],
                display_box[:, 1],
                pen=pg.mkPen("#c084fc", width=2),
                connect="pairs",
            )
        if snapshot.scan_fov_bounds_nm is not None:
            x0, x1, y0, y1 = snapshot.scan_fov_bounds_nm
            self.view.plot(
                (x0, x1, x1, x0, x0),
                (y0, y0, y1, y1, y0),
                pen=pg.mkPen("#22c55e", width=2),
            )
        if snapshot.calculation_roi_bounds_nm is not None:
            x0, x1, y0, y1 = snapshot.calculation_roi_bounds_nm
            self.view.plot(
                (x0, x1, x1, x0, x0),
                (y0, y0, y1, y1, y0),
                pen=pg.mkPen("#f472b6", width=2),
            )
        for region in snapshot.regions:
            if not region.enabled:
                continue
            outline = _region_outline(region)
            self.view.plot(
                outline[:, 0],
                outline[:, 1],
                pen=pg.mkPen("#fb923c", width=2),
            )
        if snapshot.atom_positions_nm.size:
            positions = self._oriented_atoms(snapshot, target_rotation)
            positions = positions.copy()
            positions[:, 0] += cx
            positions[:, 1] += cy
            bonds = np.asarray(snapshot.atom_bond_pairs, dtype=int)
            if bonds.size:
                bond_points = positions[bonds].reshape(-1, 3)
                self.view.plot(
                    bond_points[:, 0],
                    bond_points[:, 1],
                    pen=pg.mkPen("#94a3b8", width=1.5),
                    connect="pairs",
                )
            colours = _atomic_colours(snapshot.atomic_numbers)
            radii = _atomic_radii_nm(snapshot.atomic_numbers)
            radius_scale = max(float(np.max(radii)), 1.0e-12)
            spots = [
                {
                    "pos": (float(position[0]), float(position[1])),
                    "size": float(6.0 + 8.0 * radius / radius_scale),
                    "brush": pg.mkBrush(*(255.0 * colour).astype(int)),
                    "pen": pg.mkPen("#0f172a", width=0.5),
                }
                for position, radius, colour in zip(positions, radii, colours)
            ]
            scatter = pg.ScatterPlotItem(spots=spots, pxMode=True)
            self.view.addItem(scatter)
        if snapshot.cell_vectors_nm.shape == (3, 3):
            current = np.asarray(snapshot.orientation_matrix, dtype=float)
            target_cell = (
                np.asarray(snapshot.cell_vectors_nm, dtype=float)
                @ current
                @ np.asarray(target_rotation, dtype=float).T
            )
            cell = _cell_lines(target_cell) + display_centre
            self.view.plot(
                cell[:, 0],
                cell[:, 1],
                pen=pg.mkPen("#c084fc", width=2),
                connect="pairs",
            )
        half_span = 0.55 * max(float(display_size[0]), float(display_size[1]), 1.0e-3)
        self.view.setRange(
            xRange=(display_centre[0] - half_span, display_centre[0] + half_span),
            yRange=(display_centre[1] - half_span, display_centre[1] + half_span),
            padding=0.02,
        )


class SamplePage(QWidget):
    """Edit one live sample and inspect the exact calculated sample snapshot."""

    parameters_changed = Signal(str)
    error = Signal(str)

    @staticmethod
    def _double_control(
        object_name: str,
        minimum: float,
        maximum: float,
        *,
        decimals: int = 6,
        suffix: str = "",
    ) -> QDoubleSpinBox:
        control = QDoubleSpinBox()
        control.setObjectName(object_name)
        control.setDecimals(decimals)
        control.setRange(minimum, maximum)
        control.setSuffix(suffix)
        control.setKeyboardTracking(False)
        return control

    @staticmethod
    def _integer_control(
        object_name: str,
        minimum: int,
        maximum: int,
        *,
        step: int = 1,
    ) -> QSpinBox:
        control = QSpinBox()
        control.setObjectName(object_name)
        control.setRange(minimum, maximum)
        control.setSingleStep(step)
        control.setKeyboardTracking(False)
        return control

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("samplePage")
        self._state = None
        self._result = None
        self._snapshot = None
        self._updating = False
        self._draft_quaternion = IDENTITY_QUATERNION_WXYZ

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(6, 6, 6, 6)

        identity = QGroupBox("Sample state and finite envelope")
        identity_form = QFormLayout(identity)
        identity_form.setRowWrapPolicy(
            QFormLayout.RowWrapPolicy.WrapLongRows
        )
        self.inserted = QCheckBox("Inserted (interactions enabled)")
        self.inserted.setObjectName("sampleInsertedControl")
        self.mode = QComboBox()
        self.mode.setObjectName("sampleModeControl")
        self.mode.addItem("Real sample (CIF / crystal)", "atomic")
        self.mode.addItem("Virtual sample", "virtual")
        identity_form.addRow("Holder", self.inserted)
        identity_form.addRow("Mode", self.mode)
        self.scalar_controls = {}
        for field, label, suffix, minimum, maximum in (
            ("size_x_nm", "Size X", " nm", 1.0e-6, 1.0e9),
            ("size_y_nm", "Size Y", " nm", 1.0e-6, 1.0e9),
            ("thickness_nm", "Thickness", " nm", 1.0e-6, 1.0e9),
            ("centre_x_nm", "Sample centre X", " nm", -1.0e9, 1.0e9),
            ("centre_y_nm", "Sample centre Y", " nm", -1.0e9, 1.0e9),
            ("scan_origin_x_nm", "Scan origin X", " nm", -1.0e9, 1.0e9),
            ("scan_origin_y_nm", "Scan origin Y", " nm", -1.0e9, 1.0e9),
        ):
            control = QDoubleSpinBox()
            control.setObjectName(f"sample_{field}")
            control.setDecimals(6)
            control.setRange(minimum, maximum)
            control.setSuffix(suffix)
            control.setKeyboardTracking(False)
            control.valueChanged.connect(
                lambda value, name=field: self._set_scalar(name, value)
            )
            identity_form.addRow(label, control)
            self.scalar_controls[field] = control
        controls_layout.addWidget(identity)

        real = QGroupBox("Real sample structure and orientation")
        real.setObjectName("realSampleControls")
        real_layout = QVBoxLayout(real)
        source_form = QFormLayout()
        source_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.preset = QComboBox()
        self.preset.setObjectName("samplePresetControl")
        self.preset.addItem("Default TOML preset", "")
        for key, preset_name in available_specimen_presets():
            self.preset.addItem(preset_name, key)
        path_row = QHBoxLayout()
        self.cif_path = QLineEdit()
        self.cif_path.setObjectName("sampleCifPath")
        browse = QPushButton("Import CIF...")
        browse.setObjectName("sampleImportCif")
        browse.clicked.connect(self._browse_cif)
        self.cif_path.editingFinished.connect(self._cif_edited)
        path_row.addWidget(self.cif_path, 1)
        path_row.addWidget(browse)
        path_widget = QWidget()
        path_widget.setLayout(path_row)
        source_form.addRow("TOML preset", self.preset)
        source_form.addRow("Custom CIF / MCIF", path_widget)
        real_layout.addLayout(source_form)

        axes_form = QFormLayout()
        axes_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.zone_controls = self._axis_row("zoneAxis", (0, 0, 1))
        self.in_plane_controls = self._axis_row("inPlaneAxis", (1, 0, 0))
        axes_form.addRow("Zone axis [uvw] -> +Z", self.zone_controls[0])
        axes_form.addRow("In-plane [uvw] -> +X", self.in_plane_controls[0])
        self.structure_atom_limit = QSpinBox()
        self.structure_atom_limit.setObjectName("sampleStructureAtomLimit")
        self.structure_atom_limit.setRange(100, 50_000)
        self.structure_atom_limit.setSingleStep(500)
        self.structure_atom_limit.setValue(2_500)
        self.structure_atom_limit.setSuffix(" atoms")
        self.structure_atom_limit.setKeyboardTracking(False)
        self.structure_atom_limit.setToolTip(
            "Soft rendering limit only. If the finite sample / scan ROI "
            "contains more atoms, a centred repeated-CIF display window is "
            "used without changing the multislice calculation ROI."
        )
        axes_form.addRow("Structure display limit", self.structure_atom_limit)
        real_layout.addLayout(axes_form)
        apply_zone = QPushButton("Align zone axis")
        apply_zone.setObjectName("sampleApplyZoneAxis")
        apply_zone.clicked.connect(self._apply_zone_axis)
        real_layout.addWidget(apply_zone)

        tilt_row = QHBoxLayout()
        self.tilt_controls = []
        for axis in "XYZ":
            control = QDoubleSpinBox()
            control.setObjectName(f"sampleFineTilt{axis}")
            control.setRange(-360.0, 360.0)
            control.setDecimals(4)
            control.setSuffix(" deg")
            tilt_row.addWidget(QLabel(axis))
            tilt_row.addWidget(control)
            self.tilt_controls.append(control)
        apply_tilt = QPushButton("Apply incremental tilt")
        apply_tilt.clicked.connect(self._apply_incremental_tilt)
        tilt_row.addWidget(apply_tilt)
        real_layout.addLayout(tilt_row)

        self.edit_orientation = QCheckBox("Mouse-drag edits sample orientation")
        self.edit_orientation.setObjectName("sampleEditOrientation")
        self.edit_orientation.setToolTip(
            "Off: mouse orbits the camera. On: left-drag edits a draft sample orientation; Apply commits it."
        )
        self.apply_draft = QPushButton("Apply draft orientation")
        self.apply_draft.setEnabled(False)
        self.apply_draft.clicked.connect(self._commit_draft_orientation)
        draft_row = QHBoxLayout()
        draft_row.addWidget(self.edit_orientation)
        draft_row.addWidget(self.apply_draft)
        real_layout.addLayout(draft_row)

        inelastic = QGroupBox("Real inelastic collisions")
        inelastic.setObjectName("sampleRealInelasticControls")
        inelastic_form = QFormLayout(inelastic)
        inelastic_form.setRowWrapPolicy(
            QFormLayout.RowWrapPolicy.WrapLongRows
        )
        self.real_inelastic_enabled = QCheckBox(
            "Material IMFP + Poisson event transport"
        )
        self.real_inelastic_enabled.setObjectName(
            "sampleRealInelasticEnabled"
        )
        self.real_inelastic_enabled.setToolTip(
            "Adds physical energy-loss populations (zero loss, plasmon, "
            "core ionisation and plural scattering) without inventing "
            "elastic diffraction beams."
        )
        self.inelastic_scalar_controls = {
            "real_plasmon_mean_free_path_nm": self._double_control(
                "sampleRealPlasmonMfp", 0.0, 1.0e9, suffix=" nm"
            ),
            "real_ionisation_mean_free_path_nm": self._double_control(
                "sampleRealIonisationMfp", 0.0, 1.0e9, suffix=" nm"
            ),
            "real_other_inelastic_mean_free_path_nm": self._double_control(
                "sampleRealOtherInelasticMfp", 0.0, 1.0e9, suffix=" nm"
            ),
            "real_absorption_mean_free_path_nm": self._double_control(
                "sampleRealAbsorptionMfp", 0.0, 1.0e9, suffix=" nm"
            ),
            "real_plasmon_energy_ev": self._double_control(
                "sampleRealPlasmonEnergy", 0.0, 1.0e9, suffix=" eV"
            ),
            "real_ionisation_energy_ev": self._double_control(
                "sampleRealIonisationEnergy", 0.0, 1.0e9, suffix=" eV"
            ),
            "real_other_inelastic_energy_ev": self._double_control(
                "sampleRealOtherInelasticEnergy", 1.0e-6, 1.0e9, suffix=" eV"
            ),
        }
        for field in (
            "real_plasmon_mean_free_path_nm",
            "real_ionisation_mean_free_path_nm",
            "real_plasmon_energy_ev",
            "real_ionisation_energy_ev",
        ):
            self.inelastic_scalar_controls[field].setSpecialValueText(
                "Material default"
            )
        for field in (
            "real_other_inelastic_mean_free_path_nm",
            "real_absorption_mean_free_path_nm",
        ):
            self.inelastic_scalar_controls[field].setSpecialValueText(
                "Disabled"
            )
        self.inelastic_scalar_controls[
            "real_absorption_mean_free_path_nm"
        ].setToolTip(
            "Effective removal from the tracked transmitted beam. This is "
            "not literal surface adsorption of a 60-300 keV TEM electron."
        )
        inelastic_form.addRow("Calculate", self.real_inelastic_enabled)
        for label, field in (
            ("Plasmon / low-loss MFP", "real_plasmon_mean_free_path_nm"),
            ("Core-ionisation MFP", "real_ionisation_mean_free_path_nm"),
            ("Other inelastic MFP", "real_other_inelastic_mean_free_path_nm"),
            ("Effective absorption MFP", "real_absorption_mean_free_path_nm"),
            ("Plasmon loss", "real_plasmon_energy_ev"),
            ("Ionisation loss", "real_ionisation_energy_ev"),
            ("Other representative loss", "real_other_inelastic_energy_ev"),
        ):
            inelastic_form.addRow(
                label, self.inelastic_scalar_controls[field]
            )
        self.inelastic_summary = QLabel(
            "Material inelastic probabilities appear after state binding."
        )
        self.inelastic_summary.setObjectName(
            "sampleRealInelasticSummary"
        )
        self.inelastic_summary.setWordWrap(True)
        self.inelastic_summary.setStyleSheet(
            "color: #64748b; font-weight: 600;"
        )
        inelastic_form.addRow("Resolved model", self.inelastic_summary)
        real_layout.addWidget(inelastic)

        wave = QGroupBox("High-accuracy wave calculation")
        wave.setObjectName("sampleWaveControls")
        wave_form = QFormLayout(wave)
        wave_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.tem_wave_enabled = QCheckBox("TEM image / diffraction")
        self.tem_wave_enabled.setObjectName("sampleTemWaveEnabled")
        self.stem_wave_enabled = QCheckBox("STEM detector images")
        self.stem_wave_enabled.setObjectName("sampleStemWaveEnabled")
        self.multislice_enabled = QCheckBox("Multislice propagation")
        self.multislice_enabled.setObjectName("sampleMultisliceEnabled")
        self.atomistic_enabled = QCheckBox("Lobato IAM potential")
        self.frozen_enabled = QCheckBox("Frozen-phonon ensemble")
        self.frozen_configurations = self._integer_control(
            "sampleFrozenPhononConfigurations", 1, 64
        )
        self.frozen_sigma = self._double_control(
            "sampleFrozenPhononSigma", 0.0, 1.0e6, suffix=" Å"
        )
        self.frozen_sigma.setSpecialValueText("Preset value")
        self.frozen_seed = self._integer_control(
            "sampleFrozenPhononSeed", 0, 2_147_483_647
        )
        self.wave_grid = self._integer_control(
            "sampleWaveGridPixels", 0, 8192, step=32
        )
        self.wave_grid.setSpecialValueText("Preset default")
        self.wave_scalar_controls = {
            "wave_field_of_view_angstrom": self._double_control(
                "sampleWaveFieldOfView", 0.0, 1.0e9, suffix=" Å"
            ),
            "wave_slice_thickness_angstrom": self._double_control(
                "sampleWaveSliceThickness", 1.0e-6, 1.0e9, suffix=" Å"
            ),
            "wave_defocus_nm": self._double_control(
                "sampleWaveDefocus", -1.0e9, 1.0e9, suffix=" nm"
            ),
            "wave_bandwidth_fraction": self._double_control(
                "sampleWaveBandwidth", 1.0e-6, 1.0, decimals=5
            ),
            "wave_probe_padding_factor": self._double_control(
                "sampleWaveProbePadding", 0.0, 1.0e6, decimals=4
            ),
        }
        self.wave_scalar_controls[
            "wave_field_of_view_angstrom"
        ].setSpecialValueText("Preset default")
        self.tail_enabled = QCheckBox("Approximate Rutherford high-angle tail")
        self.element_sigma = QLineEdit()
        self.element_sigma.setObjectName("sampleElementThermalRms")
        self.element_sigma.setPlaceholderText('{"Si": 0.075, "O": 0.09}')
        self.element_sigma.setToolTip(
            "Required for a custom CIF frozen-phonon calculation unless a positive global RMS override is set."
        )
        self.tail_atomic_number = QSpinBox()
        self.tail_atomic_number.setRange(1, 118)
        self.tail_density = QDoubleSpinBox()
        self.tail_density.setRange(0.0, 1.0e9)
        self.tail_density.setDecimals(6)
        self.tail_density.setSuffix(" atoms/nm2")
        self.tail_screening = QDoubleSpinBox()
        self.tail_screening.setRange(1.0e-6, 500.0)
        self.tail_screening.setDecimals(6)
        self.tail_screening.setSuffix(" mrad")
        self.tail_maximum = QDoubleSpinBox()
        self.tail_maximum.setRange(1.0e-6, 500.0)
        self.tail_maximum.setDecimals(6)
        self.tail_maximum.setSuffix(" mrad")
        wave_form.addRow("Calculate", self.tem_wave_enabled)
        wave_form.addRow("", self.stem_wave_enabled)
        wave_form.addRow(self.multislice_enabled)
        wave_form.addRow(self.atomistic_enabled)
        wave_form.addRow("Grid", self.wave_grid)
        wave_form.addRow(
            "Field of view",
            self.wave_scalar_controls["wave_field_of_view_angstrom"],
        )
        wave_form.addRow(
            "Target slice thickness",
            self.wave_scalar_controls["wave_slice_thickness_angstrom"],
        )
        wave_form.addRow(
            "Additional defocus",
            self.wave_scalar_controls["wave_defocus_nm"],
        )
        wave_form.addRow(
            "Bandwidth fraction",
            self.wave_scalar_controls["wave_bandwidth_fraction"],
        )
        wave_form.addRow(
            "Probe padding factor",
            self.wave_scalar_controls["wave_probe_padding_factor"],
        )
        wave_form.addRow(self.frozen_enabled)
        wave_form.addRow("Configurations", self.frozen_configurations)
        wave_form.addRow("Global RMS sigma", self.frozen_sigma)
        wave_form.addRow("Random seed", self.frozen_seed)
        wave_form.addRow("Per-element RMS JSON", self.element_sigma)
        wave_form.addRow(self.tail_enabled)
        wave_form.addRow("Tail atomic number Z", self.tail_atomic_number)
        wave_form.addRow("Tail areal density", self.tail_density)
        wave_form.addRow("Tail screening angle", self.tail_screening)
        wave_form.addRow("Tail maximum angle", self.tail_maximum)
        real_layout.addWidget(wave)
        controls_layout.addWidget(real)
        self.real_group = real

        virtual = QGroupBox("Virtual interaction channels (absolute probabilities)")
        virtual.setObjectName("virtualSampleControls")
        virtual_layout = QVBoxLayout(virtual)
        self.diffraction_enabled = QCheckBox(
            "Plot enabled virtual interaction channels in Ray Diagram"
        )
        self.diffraction_enabled.setObjectName(
            "sampleVirtualInteractionsEnabled"
        )
        self.diffraction_enabled.setToolTip(
            "Applies only to Virtual sample mode. Real-sample diffraction "
            "and scattering are calculated by the high-accuracy wave model, "
            "not by manually defined ray branches."
        )
        virtual_layout.addWidget(self.diffraction_enabled)
        self.interaction_table = self._table(
            ("On", "Name", "Kind", "Probability", "Parameters (JSON)")
        )
        virtual_layout.addWidget(self.interaction_table)
        interaction_buttons = QHBoxLayout()
        add_interaction = QPushButton("Add interaction")
        remove_interaction = QPushButton("Remove selected")
        apply_interactions = QPushButton("Apply interactions")
        add_interaction.clicked.connect(self._add_interaction)
        remove_interaction.clicked.connect(
            lambda: self._remove_selected(self.interaction_table)
        )
        apply_interactions.clicked.connect(self._apply_interactions)
        for button in (add_interaction, remove_interaction, apply_interactions):
            interaction_buttons.addWidget(button)
        virtual_layout.addLayout(interaction_buttons)

        region_label = QLabel(
            "Finite regions: rectangle, ellipse, or grayscale map (NPY/PNG/TIFF). Outside is vacuum."
        )
        region_label.setWordWrap(True)
        virtual_layout.addWidget(region_label)
        self.region_table = self._table(
            ("On", "Name", "Kind", "Density", "Parameters (JSON)")
        )
        virtual_layout.addWidget(self.region_table)
        region_buttons = QHBoxLayout()
        add_region = QPushButton("Add region")
        remove_region = QPushButton("Remove selected")
        apply_regions = QPushButton("Apply regions")
        add_region.clicked.connect(self._add_region)
        remove_region.clicked.connect(lambda: self._remove_selected(self.region_table))
        apply_regions.clicked.connect(self._apply_regions)
        for button in (add_region, remove_region, apply_regions):
            region_buttons.addWidget(button)
        virtual_layout.addLayout(region_buttons)
        self.virtual_probe_convolution = QCheckBox(
            "Convolve density with the calculated probe"
        )
        self.virtual_probe_convolution.setObjectName(
            "sampleVirtualProbeConvolution"
        )
        virtual_layout.addWidget(self.virtual_probe_convolution)
        controls_layout.addWidget(virtual)
        self.virtual_group = virtual

        controls_layout.addStretch(1)

        self.controls_scroll = QScrollArea()
        self.controls_scroll.setObjectName("sampleControlsScrollArea")
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setMinimumWidth(390)
        self.controls_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.controls_scroll.setWidget(controls)

        self.scene = SampleSceneView()
        self.scene.orientation_dragged.connect(self._orientation_dragged)
        self.edit_orientation.toggled.connect(self.scene.set_edit_orientation)
        self.scene_status = QLabel()
        self.scene_status.setWordWrap(True)
        self.scene_status.setStyleSheet("color: #475569;")
        scene_page = QWidget()
        scene_layout = QVBoxLayout(scene_page)
        scene_layout.addWidget(self.scene_status)
        structure_row = QHBoxLayout()
        structure_row.setContentsMargins(0, 0, 0, 0)
        structure_row.addWidget(self.scene, 1)
        self.element_legend = ElementLegend()
        structure_row.addWidget(self.element_legend)
        scene_layout.addLayout(structure_row, 1)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.controls_scroll)
        splitter.addWidget(scene_page)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes((420, 1000))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        self.inserted.toggled.connect(lambda value: self._set_bool("inserted", value))
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.preset.currentIndexChanged.connect(self._preset_changed)
        self.diffraction_enabled.toggled.connect(
            lambda value: self._set_bool("diffraction_enabled", value)
        )
        for control, field in (
            (self.tem_wave_enabled, "wave_enabled"),
            (self.stem_wave_enabled, "stem_wave_enabled"),
            (self.multislice_enabled, "wave_multislice_enabled"),
            (self.atomistic_enabled, "wave_atomistic_enabled"),
            (self.frozen_enabled, "wave_frozen_phonon_enabled"),
            (self.real_inelastic_enabled, "real_inelastic_enabled"),
            (self.tail_enabled, "real_high_angle_tail_enabled"),
            (
                self.virtual_probe_convolution,
                "virtual_probe_convolution_enabled",
            ),
        ):
            control.toggled.connect(
                lambda value, name=field: self._set_bool(name, value)
            )
        self.wave_grid.valueChanged.connect(
            lambda value: self._set_integer("wave_grid_pixels", value)
        )
        self.frozen_configurations.valueChanged.connect(
            lambda value: self._set_integer(
                "wave_frozen_phonon_configurations", value
            )
        )
        self.frozen_seed.valueChanged.connect(
            lambda value: self._set_integer("wave_frozen_phonon_seed", value)
        )
        self.frozen_sigma.valueChanged.connect(
            lambda value: self._set_scalar(
                "wave_frozen_phonon_sigma_angstrom", value
            )
        )
        for field, control in self.wave_scalar_controls.items():
            control.valueChanged.connect(
                lambda value, name=field: self._set_scalar(name, value)
            )
        for field, control in self.inelastic_scalar_controls.items():
            control.valueChanged.connect(
                lambda value, name=field: self._set_scalar(name, value)
            )
        self.element_sigma.editingFinished.connect(
            self._element_sigma_edited
        )
        self.structure_atom_limit.valueChanged.connect(
            lambda _value: self.refresh_snapshot()
        )
        self.tail_atomic_number.valueChanged.connect(
            lambda value: self._set_integer("real_tail_atomic_number", value)
        )
        for control, field in (
            (self.tail_density, "real_tail_areal_density_atoms_nm2"),
            (self.tail_screening, "real_tail_screening_angle_mrad"),
            (self.tail_maximum, "real_tail_max_angle_mrad"),
        ):
            control.valueChanged.connect(
                lambda value, name=field: self._set_scalar(name, value)
            )

    @staticmethod
    def _table(headers):
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)
        table.setMinimumHeight(150)
        return table

    @staticmethod
    def _axis_row(prefix, values):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        controls = []
        for index, value in enumerate(values):
            control = QSpinBox()
            control.setObjectName(f"{prefix}{index}")
            control.setRange(-999, 999)
            control.setValue(value)
            layout.addWidget(control)
            controls.append(control)
        return widget, tuple(controls)

    def set_state(self, state):
        self._state = state
        self._updating = True
        try:
            sample = state.sample
            self.inserted.setChecked(bool(sample.inserted))
            index = self.mode.findData(str(sample.specimen_mode).lower())
            self.mode.setCurrentIndex(max(index, 0))
            for field, control in self.scalar_controls.items():
                control.setValue(float(getattr(sample, field)))
            preset_index = self.preset.findData(
                str(sample.specimen_preset_key)
            )
            self.preset.setCurrentIndex(
                preset_index if preset_index >= 0 else 0
            )
            self.cif_path.setText(str(sample.cif_path))
            for controls, values in (
                (self.zone_controls[1], sample.zone_axis_uvw),
                (self.in_plane_controls[1], sample.in_plane_axis_uvw),
            ):
                for control, value in zip(controls, values):
                    control.setValue(int(value))
            self.diffraction_enabled.setChecked(
                bool(sample.diffraction_enabled)
            )
            self.tem_wave_enabled.setChecked(bool(sample.wave_enabled))
            self.stem_wave_enabled.setChecked(bool(sample.stem_wave_enabled))
            self.multislice_enabled.setChecked(
                bool(sample.wave_multislice_enabled)
            )
            self.atomistic_enabled.setChecked(bool(sample.wave_atomistic_enabled))
            self.frozen_enabled.setChecked(bool(sample.wave_frozen_phonon_enabled))
            self.wave_grid.setValue(int(sample.wave_grid_pixels))
            for field, control in self.wave_scalar_controls.items():
                control.setValue(float(getattr(sample, field)))
            self.frozen_configurations.setValue(
                int(sample.wave_frozen_phonon_configurations)
            )
            self.frozen_sigma.setValue(
                float(sample.wave_frozen_phonon_sigma_angstrom)
            )
            self.frozen_seed.setValue(int(sample.wave_frozen_phonon_seed))
            self.real_inelastic_enabled.setChecked(
                bool(sample.real_inelastic_enabled)
            )
            for field, control in self.inelastic_scalar_controls.items():
                control.setValue(float(getattr(sample, field)))
            self.tail_enabled.setChecked(bool(sample.real_high_angle_tail_enabled))
            self.virtual_probe_convolution.setChecked(
                bool(sample.virtual_probe_convolution_enabled)
            )
            self.element_sigma.setText(
                json.dumps(
                    sample.wave_frozen_phonon_sigma_by_element_angstrom,
                    sort_keys=True,
                )
            )
            self.tail_atomic_number.setValue(int(sample.real_tail_atomic_number))
            self.tail_density.setValue(float(sample.real_tail_areal_density_atoms_nm2))
            self.tail_screening.setValue(float(sample.real_tail_screening_angle_mrad))
            self.tail_maximum.setValue(float(sample.real_tail_max_angle_mrad))
            self._load_interaction_table(sample.virtual_interactions)
            self._load_region_table(sample.virtual_regions)
            self._draft_quaternion = sample_orientation_quaternion(sample)
            self.apply_draft.setEnabled(False)
            self._update_mode_controls()
            self._update_wave_controls()
            self._refresh_inelastic_summary()
        finally:
            self._updating = False
        self.refresh_snapshot()

    def _set_scalar(self, name, value):
        if self._updating or self._state is None:
            return
        if name in {"size_x_nm", "size_y_nm", "thickness_nm"} and value <= 0.0:
            self.error.emit(f"{name} must be positive.")
            return
        setattr(self._state.sample, name, float(value))
        self._changed(f"sample.{name}")

    def _set_bool(self, name, value):
        if self._updating or self._state is None:
            return
        setattr(self._state.sample, name, bool(value))
        if name in {
            "wave_enabled",
            "stem_wave_enabled",
            "wave_multislice_enabled",
            "wave_atomistic_enabled",
            "wave_frozen_phonon_enabled",
            "real_inelastic_enabled",
        }:
            self._update_wave_controls()
        self._changed(f"sample.{name}")

    def _set_integer(self, name, value):
        if self._updating or self._state is None:
            return
        if name == "wave_grid_pixels" and 0 < int(value) < 32:
            self.error.emit("wave_grid_pixels must be 0 or at least 32.")
            self._updating = True
            try:
                self.wave_grid.setValue(
                    int(self._state.sample.wave_grid_pixels)
                )
            finally:
                self._updating = False
            return
        setattr(self._state.sample, name, int(value))
        self._changed(f"sample.{name}")

    def _preset_changed(self):
        if self._updating or self._state is None:
            return
        self._state.sample.specimen_preset_key = str(
            self.preset.currentData() or ""
        )
        self._changed("sample.specimen_preset_key")

    def _element_sigma_edited(self):
        if self._updating or self._state is None:
            return
        try:
            values = json.loads(self.element_sigma.text().strip() or "{}")
            if not isinstance(values, dict):
                raise ValueError("Per-element RMS values must be a JSON object.")
            converted = {}
            for symbol, value in values.items():
                sigma = float(value)
                if not math.isfinite(sigma) or sigma <= 0.0:
                    raise ValueError(
                        f"Frozen-phonon RMS for {symbol} must be positive."
                    )
                converted[str(symbol)] = sigma
        except Exception as exc:
            self.error.emit(str(exc))
            return
        self._state.sample.wave_frozen_phonon_sigma_by_element_angstrom = converted
        self._changed(
            "sample.wave_frozen_phonon_sigma_by_element_angstrom"
        )

    def _mode_changed(self):
        if self._updating or self._state is None:
            return
        self._state.sample.specimen_mode = str(self.mode.currentData())
        self._update_mode_controls()
        self._update_wave_controls()
        self._changed("sample.specimen_mode")

    def _update_mode_controls(self):
        atomic = str(self.mode.currentData()) == "atomic"
        self.real_group.setVisible(atomic)
        self.virtual_group.setVisible(not atomic)

    def _update_wave_controls(self):
        atomic = str(self.mode.currentData()) == "atomic"
        illumination = str(
            getattr(self._state, "illumination_mode", "TEM")
            if self._state is not None
            else "TEM"
        ).upper()
        tem_available = atomic and illumination == "TEM"
        stem_available = atomic and illumination == "STEM"
        self.tem_wave_enabled.setEnabled(tem_available)
        self.stem_wave_enabled.setEnabled(stem_available)
        self.tem_wave_enabled.setToolTip(
            "Calculate the local specimen-to-Objective image and exit-wave "
            "diffraction diagnostic."
            if tem_available
            else "TEM wave imaging requires Real sample mode and Microprobe "
            "(TEM) illumination."
        )
        self.stem_wave_enabled.setToolTip(
            "Calculate raster detector images with the STEM wave model."
            if stem_available
            else "STEM wave detector imaging requires Real sample mode and "
            "Nanoprobe (STEM) illumination."
        )

        inelastic_enabled = (
            atomic and self.real_inelastic_enabled.isChecked()
        )
        self.real_inelastic_enabled.setEnabled(atomic)
        for control in self.inelastic_scalar_controls.values():
            control.setEnabled(inelastic_enabled)

        multislice = atomic and self.multislice_enabled.isChecked()
        atomistic = multislice and self.atomistic_enabled.isChecked()
        frozen = atomistic and self.frozen_enabled.isChecked()
        self.multislice_enabled.setEnabled(atomic)
        self.atomistic_enabled.setEnabled(multislice)
        self.frozen_enabled.setEnabled(atomistic)
        for control in (
            self.frozen_configurations,
            self.frozen_sigma,
            self.frozen_seed,
            self.element_sigma,
        ):
            control.setEnabled(frozen)

    def _refresh_inelastic_summary(self):
        if self._state is None:
            return
        try:
            from temsim.specimen.inelastic import real_inelastic_distribution

            distribution = real_inelastic_distribution(self._state)

            def mfp(value):
                return (
                    f"{float(value):.6g} nm"
                    if math.isfinite(float(value)) else "disabled"
                )

            channel_text = ", ".join(
                f"{channel.label} {100.0 * channel.probability:.5g}%"
                for channel in distribution.channels
            )
            self.inelastic_summary.setText(
                f"{distribution.material_name}; total λ {mfp(distribution.total_inelastic_mean_free_path_nm)}, "
                f"plasmon λ {mfp(distribution.plasmon_mean_free_path_nm)}, "
                f"ionisation λ {mfp(distribution.ionisation_mean_free_path_nm)}; "
                f"t/λ {distribution.mean_inelastic_events:.6g}. "
                f"{channel_text}; effective absorption "
                f"{100.0 * distribution.absorbed_probability:.5g}%."
            )
            self.inelastic_summary.setToolTip(
                "\n".join(
                    (
                        f"Model: {distribution.model}",
                        f"Reference: {distribution.reference}",
                        f"Applicability: {distribution.applicability}",
                        *distribution.warnings,
                    )
                )
            )
        except Exception as exc:
            self.inelastic_summary.setText(
                f"Inelastic model unavailable: {exc}"
            )
            self.inelastic_summary.setToolTip(str(exc))

    def _changed(self, name):
        self._refresh_inelastic_summary()
        self.refresh_snapshot()
        self.parameters_changed.emit(name)

    def _browse_cif(self):
        path, _selected = QFileDialog.getOpenFileName(
            self,
            "Import crystallographic structure",
            self.cif_path.text(),
            "Crystallographic files (*.cif *.mcif);;All files (*)",
        )
        if path:
            self.cif_path.setText(path)
            self._cif_edited()

    def _cif_edited(self):
        if self._updating or self._state is None:
            return
        self._state.sample.cif_path = self.cif_path.text().strip()
        self._changed("sample.cif_path")

    def _apply_zone_axis(self):
        if self._state is None:
            return
        path = Path(self.cif_path.text()).expanduser()
        try:
            if not path.is_file():
                raise ValueError("Select an existing CIF before aligning a zone axis.")
            from ase.io import read

            atoms = read(path)
            zone = tuple(control.value() for control in self.zone_controls[1])
            in_plane = tuple(control.value() for control in self.in_plane_controls[1])
            quaternion = quaternion_from_zone_axes(
                np.asarray(atoms.cell.array, dtype=float),
                zone,
                in_plane,
            )
            set_sample_orientation(self._state.sample, quaternion)
            self._state.sample.zone_axis_uvw = zone
            self._state.sample.in_plane_axis_uvw = in_plane
            self._draft_quaternion = quaternion
        except Exception as exc:
            self.error.emit(str(exc))
            return
        self._changed("sample.specimen_orientation_quaternion_wxyz")

    def _apply_incremental_tilt(self):
        if self._state is None:
            return
        delta = quaternion_from_euler_xyz_deg(
            tuple(control.value() for control in self.tilt_controls)
        )
        quaternion = quaternion_multiply(
            delta,
            sample_orientation_quaternion(self._state.sample),
        )
        set_sample_orientation(self._state.sample, quaternion)
        self._draft_quaternion = quaternion
        for control in self.tilt_controls:
            control.setValue(0.0)
        self._changed("sample.specimen_orientation_quaternion_wxyz")

    def _orientation_dragged(self, dx, dy):
        if self._state is None or not self.edit_orientation.isChecked():
            return
        delta = quaternion_from_euler_xyz_deg((0.25 * dy, 0.25 * dx, 0.0))
        self._draft_quaternion = quaternion_multiply(delta, self._draft_quaternion)
        self.apply_draft.setEnabled(True)
        if self._snapshot is not None:
            self.scene.display_snapshot(
                self._snapshot,
                draft_quaternion=self._draft_quaternion,
            )
        self.scene_status.setText(
            "Draft physical orientation changed. Apply commits it; calculations still use the last applied orientation."
        )

    def _commit_draft_orientation(self):
        if self._state is None:
            return
        set_sample_orientation(self._state.sample, self._draft_quaternion)
        self.apply_draft.setEnabled(False)
        self._changed("sample.specimen_orientation_quaternion_wxyz")

    def _load_interaction_table(self, rows):
        self.interaction_table.setRowCount(0)
        for raw in rows or ():
            self._append_table_row(
                self.interaction_table,
                (
                    bool(raw.get("enabled", True)),
                    str(raw.get("name", "Interaction")),
                    str(raw.get("kind", "diffuse_ring")),
                    float(raw.get("probability", 0.0)),
                    json.dumps(
                        {
                            key: value
                            for key, value in raw.items()
                            if key not in {"enabled", "name", "kind", "probability"}
                        },
                        sort_keys=True,
                    ),
                ),
            )

    def _load_region_table(self, rows):
        self.region_table.setRowCount(0)
        for raw in rows or ():
            self._append_table_row(
                self.region_table,
                (
                    bool(raw.get("enabled", True)),
                    str(raw.get("name", "Region")),
                    str(raw.get("kind", "rectangle")),
                    float(raw.get("density", 1.0)),
                    json.dumps(
                        {
                            key: value
                            for key, value in raw.items()
                            if key not in {"enabled", "name", "kind", "density"}
                        },
                        sort_keys=True,
                    ),
                ),
            )

    @staticmethod
    def _append_table_row(table, values):
        row = table.rowCount()
        table.insertRow(row)
        enabled = QTableWidgetItem()
        enabled.setFlags(enabled.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        enabled.setCheckState(
            Qt.CheckState.Checked if values[0] else Qt.CheckState.Unchecked
        )
        table.setItem(row, 0, enabled)
        for column, value in enumerate(values[1:], 1):
            table.setItem(row, column, QTableWidgetItem(str(value)))

    def _add_interaction(self):
        self._append_table_row(
            self.interaction_table,
            (True, "New diffuse channel", "gaussian_diffuse", 0.05, '{"sigma_mrad": 10.0}'),
        )

    def _add_region(self):
        self._append_table_row(
            self.region_table,
            (
                True,
                "New region",
                "rectangle",
                1.0,
                json.dumps(
                    {
                        "centre_x_nm": 0.0,
                        "centre_y_nm": 0.0,
                        "size_x_nm": 100.0,
                        "size_y_nm": 100.0,
                        "rotation_deg": 0.0,
                    }
                ),
            ),
        )

    @staticmethod
    def _remove_selected(table):
        rows = sorted({index.row() for index in table.selectedIndexes()}, reverse=True)
        for row in rows:
            table.removeRow(row)

    @staticmethod
    def _cell_text(table, row, column):
        item = table.item(row, column)
        return "" if item is None else item.text().strip()

    def _apply_interactions(self):
        if self._state is None:
            return
        try:
            rows = []
            for row in range(self.interaction_table.rowCount()):
                enabled = self.interaction_table.item(row, 0).checkState() == Qt.CheckState.Checked
                parameters = json.loads(self._cell_text(self.interaction_table, row, 4) or "{}")
                if not isinstance(parameters, dict):
                    raise ValueError(f"Interaction row {row + 1}: parameters must be a JSON object.")
                parameters.update(
                    {
                        "enabled": enabled,
                        "name": self._cell_text(self.interaction_table, row, 1),
                        "kind": self._cell_text(self.interaction_table, row, 2),
                        "probability": float(self._cell_text(self.interaction_table, row, 3)),
                    }
                )
                rows.append(parameters)
            original = self._state.sample.virtual_interactions
            self._state.sample.virtual_interactions = rows
            try:
                resolve_virtual_interactions(
                    self._state.sample,
                    beam_energy_kv=self._state.beam_voltage_kv,
                )
            except Exception:
                self._state.sample.virtual_interactions = original
                raise
        except Exception as exc:
            self.error.emit(str(exc))
            return
        self._changed("sample.virtual_interactions")

    def _apply_regions(self):
        if self._state is None:
            return
        try:
            rows = []
            for row in range(self.region_table.rowCount()):
                parameters = json.loads(self._cell_text(self.region_table, row, 4) or "{}")
                if not isinstance(parameters, dict):
                    raise ValueError(f"Region row {row + 1}: parameters must be a JSON object.")
                parameters.update(
                    {
                        "enabled": self.region_table.item(row, 0).checkState() == Qt.CheckState.Checked,
                        "name": self._cell_text(self.region_table, row, 1),
                        "kind": self._cell_text(self.region_table, row, 2),
                        "density": float(self._cell_text(self.region_table, row, 3)),
                    }
                )
                rows.append(parameters)
            original = self._state.sample.virtual_regions
            self._state.sample.virtual_regions = rows
            try:
                build_sample_geometry_snapshot(self._state.sample, load_atoms=False)
            except Exception:
                self._state.sample.virtual_regions = original
                raise
        except Exception as exc:
            self.error.emit(str(exc))
            return
        self._changed("sample.virtual_regions")

    def refresh_snapshot(self, calculation_result=None):
        if self._state is None:
            return
        result = calculation_result or self._result
        stem = getattr(result, "stem_scan", None) if result is not None else None
        sample = (
            getattr(result, "state_snapshot", None).sample
            if result is not None and getattr(result, "state_snapshot", None) is not None
            else self._state.sample
        )
        probe = getattr(stem, "probe_state", None)
        try:
            snapshot = build_sample_geometry_snapshot(
                sample,
                scan_x_um=(stem.scan_x_um if stem is not None else None),
                scan_y_um=(stem.scan_y_um if stem is not None else None),
                current_probe_nm=(probe.centroid_nm if probe is not None else None),
                probe_padding_nm=(
                    float(probe.radius_99_nm)
                    * float(getattr(sample, "wave_probe_padding_factor", 3.0))
                    if probe is not None
                    else 0.0
                ),
                load_atoms=True,
                maximum_display_atoms=self.structure_atom_limit.value(),
            )
        except Exception as exc:
            self.scene_status.setText(f"Sample geometry unavailable: {exc}")
            self.scene.clear()
            self.element_legend.set_atomic_numbers(())
            return
        self._snapshot = snapshot
        self.scene.display_snapshot(snapshot, draft_quaternion=self._draft_quaternion)
        self.element_legend.set_atomic_numbers(snapshot.atomic_numbers)
        backend = (
            f"3-D OpenGL ({self.scene.opengl_detail})"
            if self.scene.opengl_available
            else f"safe 2-D fallback ({self.scene.opengl_detail})"
        )
        mode = "Real sample" if snapshot.mode == "atomic" else "Virtual sample"
        atom_detail = ""
        if snapshot.atomic_numbers.size:
            display_size = snapshot.atom_display_size_nm
            window = (
                f" in {display_size[0]:.6g} x {display_size[1]:.6g} x "
                f"{display_size[2]:.6g} nm"
                if display_size is not None
                else ""
            )
            if not self.scene.opengl_available:
                render_model = "2-D projected ball-stick fallback"
            elif snapshot.atomic_numbers.size <= 3_000:
                render_model = "shaded mesh spheres"
            else:
                render_model = "point-sphere level of detail"
            atom_detail = (
                f" | repeated CIF: {snapshot.atomic_numbers.size:,} atoms, "
                f"{snapshot.atom_bond_pairs.shape[0]:,} bonds{window} | "
                f"{render_model}"
            )
        warning = " | ".join(snapshot.warnings)
        self.scene_status.setText(
            f"{mode} | {'INSERTED' if snapshot.inserted else 'RETRACTED'} | {backend}{atom_detail}"
            + (f"\n{warning}" if warning else "")
        )

    def display_result(self, result, stem_frame=None):
        """Refresh specimen geometry; detector images belong to the STEM page."""

        self._result = result
        self.refresh_snapshot(result)
