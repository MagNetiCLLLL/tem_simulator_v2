"""Central finite-sample editor and ball-and-stick structure workspace."""

from __future__ import annotations

from temsim.gui.input_policy import (
    WheelSafeComboBox as QComboBox,
    WheelSafeDoubleSpinBox as QDoubleSpinBox,
    WheelSafeSpinBox as QSpinBox,
)

import json
import hashlib
import math
import os
from dataclasses import fields
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QColor, QGuiApplication, QMatrix4x4, QVector3D
from PySide6.QtWidgets import (
    QCheckBox,
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
from temsim.specimen.reference_catalog import (
    available_reference_samples, apply_reference_sample, get_reference_sample,
    refresh_reference_samples,
)
from temsim.specimen.source import active_cif_path
from temsim.specimen.rutherford import resolve_tail_material
from temsim.specimen.support import (
    available_support_materials,
    available_support_meshes,
)
from temsim.gui.sample_scene_labels import sample_scene_labels
from temsim.gui.sample_display_source import resolve_sample_display_source


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


def _disk_lines(centre, size, *, samples=96):
    """Return line pairs for a finite elliptical disk/cylinder envelope."""

    cx, cy, cz = centre
    radius_x = 0.5 * float(size[0])
    radius_y = 0.5 * float(size[1])
    half_z = 0.5 * float(size[2])
    phase = np.linspace(0.0, 2.0 * math.pi, int(samples) + 1)
    rings = []
    for z in (cz - half_z, cz + half_z):
        ring = np.column_stack(
            (
                cx + radius_x * np.cos(phase),
                cy + radius_y * np.sin(phase),
                np.full(phase.size, z),
            )
        )
        rings.extend(np.column_stack((ring[:-1], ring[1:])).reshape(-1, 3))
    for angle in np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False):
        x = cx + radius_x * math.cos(float(angle))
        y = cy + radius_y * math.sin(float(angle))
        rings.extend(((x, y, cz - half_z), (x, y, cz + half_z)))
    return np.asarray(rings, dtype=float)


def _sample_envelope_lines(snapshot):
    if snapshot.envelope_shape == "disk":
        return _disk_lines((0.0, 0.0, 0.0), snapshot.size_nm)
    return _box_lines((0.0, 0.0, 0.0), snapshot.size_nm)


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
        self._snapshot = None
        self._has_fitted = False
        self._render_key = None
        self._atom_item = self._bond_item = self._beam_item = self._probe_item = None
        self._atom_spots = []
        self._base_rotation = np.eye(3)
        self._shown_rotation = np.eye(3)
        self._shown_probe = None
        self.model_builds = 0
        self.model_reuses = 0

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
        self._render_key = None
        self._atom_item = self._bond_item = self._beam_item = self._probe_item = None
        self._atom_spots = []

    def _add_gl_line(self, positions, colour, width=2.0, mode="line_strip"):
        positions = np.asarray(positions, dtype=float)
        colours = np.asarray(colour, dtype=float)
        # pyqtgraph treats every ndarray colour as a per-vertex buffer. A
        # length-four ndarray is NOT a uniform RGBA and leaves later vertices
        # without colours, making almost all of a wireframe disappear.
        if colours.shape == (4,):
            colours = tuple(float(value) for value in colours)
        elif colours.shape != (len(positions), 4):
            raise ValueError("Line colours must be one RGBA or one RGBA per vertex.")
        item = gl.GLLinePlotItem(
            pos=positions,
            color=colours,
            width=float(width),
            antialias=True,
            mode=mode,
        )
        self.view.addItem(item)
        self._items.append(item)
        return item

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
        return item

    @staticmethod
    def _geometry_key(snapshot):
        # Cached atom arrays are immutable, so identity avoids hashing them on
        # each scan/probe update. The retained snapshot keeps those identities
        # alive until a different model is installed. Empty arrays are equal.
        parts = []
        for field in fields(snapshot):
            if field.name == "warnings":
                continue
            value = getattr(snapshot, field.name)
            if field.name == "current_probe_nm":
                value = value is not None
            elif isinstance(value, np.ndarray):
                identity_array = field.name in {"atom_positions_nm", "atomic_numbers", "atom_bond_pairs"}
                value = (value.shape, value.dtype.str,
                         (id(value) if value.size else None) if identity_array else value.tobytes())
            parts.append(value)
        return tuple(parts)

    def display_snapshot(self, snapshot, *, draft_quaternion=None):
        target_rotation = quaternion_to_matrix(
            draft_quaternion
            if draft_quaternion is not None
            else snapshot.orientation_quaternion_wxyz
        )
        key = self._geometry_key(snapshot)
        if key == self._render_key:
            self.model_reuses += 1
            self._update_retained_model(snapshot, target_rotation)
        else:
            self.clear()
            self._base_rotation = target_rotation.copy()
            if self.opengl_available:
                self._display_gl(snapshot, target_rotation)
            else:
                self._display_2d(snapshot, target_rotation)
            self._render_key = key
            self.model_builds += 1
        self._snapshot = snapshot
        self._shown_rotation = target_rotation.copy()
        self._shown_probe = snapshot.current_probe_nm
        if not self._has_fitted:
            self.fit_full_sample()

    def _update_retained_model(self, snapshot, target_rotation):
        """Move draft lattice/probe objects without rebuilding meshes or bonds."""
        if not np.array_equal(target_rotation, self._shown_rotation):
            if self.opengl_available:
                delta = target_rotation @ self._base_rotation.T
                centre = np.asarray(snapshot.centre_nm, dtype=float)
                transform = np.eye(4)
                transform[:3, :3] = delta
                transform[:3, 3] = centre - delta @ centre
                matrix = QMatrix4x4(*transform.ravel().tolist())
                for item in (self._atom_item, self._bond_item):
                    if item is not None:
                        item.setTransform(matrix)
            elif self._atom_item is not None:
                positions = self._oriented_atoms(snapshot, target_rotation).copy()
                positions[:, :2] += snapshot.centre_nm[:2]
                for spot, position in zip(self._atom_spots, positions):
                    spot["pos"] = tuple(position[:2])
                self._atom_item.setData(spots=self._atom_spots)
                if self._bond_item is not None:
                    points = positions[snapshot.atom_bond_pairs].reshape(-1, 3)
                    self._bond_item.setData(points[:, 0], points[:, 1])
        if self.opengl_available and snapshot.current_probe_nm != self._shown_probe:
            size = snapshot.atom_display_size_nm or snapshot.size_nm
            centre = snapshot.atom_display_centre_nm or snapshot.centre_nm
            scale = max(float(np.max(size)), 1.0e-3)
            px, py = snapshot.current_probe_nm or tuple(centre[:2])
            if self._beam_item is not None:
                self._beam_item.setData(pos=np.asarray(
                    ((px, py, -0.8 * scale), (px, py, 0.8 * scale)), dtype=float))
            if self._probe_item is not None:
                half = max(scale * 0.012, 1.0e-3)
                z = 0.6 * snapshot.size_nm[2]
                self._probe_item.setData(pos=np.asarray(
                    ((px - half, py, z), (px + half, py, z),
                     (px, py - half, z), (px, py + half, z)), dtype=float))
        self.view.update()

    def fit_full_sample(self):
        """Fit physical dimensions without changing or regenerating atoms."""
        if self._snapshot is not None:
            self._fit_region(self._snapshot.centre_nm, self._snapshot.size_nm)

    def fit_local_region(self):
        """Fit the requested local material region, not its capped atom subset."""
        if self._snapshot is None:
            return
        bounds = getattr(self._snapshot, "local_material_bounds_nm", None)
        if bounds is None:
            self.fit_full_sample()
            return
        limits = np.asarray(bounds, dtype=float).reshape(3, 2)
        self._fit_region(np.mean(limits, axis=1), np.diff(limits, axis=1).ravel())

    def _fit_region(self, centre, size):
        centre = np.asarray(centre, dtype=float)
        size = np.maximum(np.asarray(size, dtype=float), 1.0e-3)
        if self.opengl_available:
            # GLViewWidget uses a horizontal field of view. Fit a bounding
            # sphere using the smaller viewport angle, without axis stretching.
            half_angle = math.radians(float(self.view.opts["fov"]) * 0.5)
            aspect = max(self.view.height(), 1) / max(self.view.width(), 1)
            half_angle = min(half_angle, math.atan(math.tan(half_angle) * aspect))
            self.view.opts["distance"] = float(0.6 * np.linalg.norm(size) / math.sin(half_angle))
            self.view.opts["center"] = QVector3D(*centre)
            self.view.update()
        else:
            self.view.setRange(
                xRange=(centre[0] - 0.55 * size[0], centre[0] + 0.55 * size[0]),
                yRange=(centre[1] - 0.55 * size[1], centre[1] + 0.55 * size[1]),
                padding=0.02,
                disableAutoRange=True,
            )
        self._has_fitted = True

    @staticmethod
    def _local_outline(snapshot):
        bounds = getattr(snapshot, "local_material_bounds_nm", None)
        if bounds is None:
            return np.empty((0, 3))
        limits = np.asarray(bounds, dtype=float).reshape(3, 2)
        return _box_lines(np.mean(limits, axis=1), np.diff(limits, axis=1).ravel())

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
        # The finite material envelope is in laboratory coordinates. Crystal
        # zone alignment rotates the lattice, not the solver's clipping solid.
        box = _sample_envelope_lines(snapshot) + np.asarray((cx, cy, cz))
        self._add_gl_line(box, (0.22, 0.74, 0.97, 0.95), mode="lines")
        local = self._local_outline(snapshot)
        if local.size:
            self._add_gl_line(local, (0.96, 0.45, 0.71, 0.9), mode="lines")
        if getattr(snapshot, "atom_display_capped", False):
            self._add_gl_line(
                _box_lines(display_centre, display_size),
                (0.98, 0.75, 0.14, 0.75),
                width=1.0,
                mode="lines",
            )
        beam_x, beam_y = (
            snapshot.current_probe_nm
            if snapshot.current_probe_nm is not None
            else tuple(display_centre[:2])
        )
        self._beam_item = self._add_gl_line(
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
            half = max(scale * 0.012, 1.0e-3)
            self._probe_item = self._add_gl_line(
                ((px - half, py, 0.6 * sz), (px + half, py, 0.6 * sz),
                 (px, py - half, 0.6 * sz), (px, py + half, 0.6 * sz)),
                (1.0, 0.85, 0.2, 0.9), width=2.0, mode="lines",
            )
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
                self._bond_item = self._add_gl_line(
                    bond_points,
                    bond_colours,
                    width=2.5,
                    mode="lines",
                )
            self._atom_item = self._add_gl_atoms(positions, snapshot.atomic_numbers)
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
        box = _sample_envelope_lines(snapshot)
        self.view.plot(
            box[:, 0] + cx, box[:, 1] + cy,
            pen=pg.mkPen("#38bdf8", width=2), connect="pairs",
            name="Full sample",
        )
        local = self._local_outline(snapshot)
        if local.size:
            self.view.plot(
                local[:, 0], local[:, 1],
                pen=pg.mkPen("#f472b6", width=2), connect="pairs",
                name="Local region",
            )
        if getattr(snapshot, "atom_display_capped", False):
            display_box = _box_lines(display_centre, display_size)
            self.view.plot(
                display_box[:, 0],
                display_box[:, 1],
                pen=pg.mkPen("#fbbf24", width=1, style=Qt.PenStyle.DashLine),
                connect="pairs",
                name="Displayed subset",
            )
        if snapshot.scan_fov_bounds_nm is not None:
            x0, x1, y0, y1 = snapshot.scan_fov_bounds_nm
            self.view.plot(
                (x0, x1, x1, x0, x0),
                (y0, y0, y1, y1, y0),
                pen=pg.mkPen("#22c55e", width=2),
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
                self._bond_item = self.view.plot(
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
            self._atom_spots = spots
            self._atom_item = scatter


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
        self._reference_revision = None
        self._result = None
        self._snapshot = None
        self._refresh_pending = False
        self._pending_calculation_result = None
        self._eds_result = None
        self._elastic_result = None
        self._specimen_interactions = None
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
        self.mode.addItem("Reference CIF", "reference")
        self.mode.addItem("Open CIF", "atomic")
        self.envelope_shape = QComboBox()
        self.envelope_shape.setObjectName("sampleEnvelopeShapeControl")
        self.envelope_shape.addItem("Circular disk", "disk")
        self.envelope_shape.addItem("Rectangle", "rectangle")
        identity_form.addRow("Holder", self.inserted)
        identity_form.addRow("Structure source", self.mode)
        identity_form.addRow("Envelope", self.envelope_shape)
        self.scalar_controls = {}
        self.scalar_labels = {}
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
            label_widget = QLabel(label)
            identity_form.addRow(label_widget, control)
            self.scalar_controls[field] = control
            self.scalar_labels[field] = label_widget
        controls_layout.addWidget(identity)

        real = QGroupBox("Crystal structure — CIF / MCIF")
        real.setObjectName("realSampleControls")
        real_layout = QVBoxLayout(real)
        source_form = QFormLayout()
        source_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.source_note = QLabel("Reference and imported samples use their actual CIF structure.")
        self.source_note.setWordWrap(True)
        self.source_note.setObjectName("sampleRealSourceNote")
        self.source_note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        real_layout.addWidget(self.source_note)
        self.preset = QComboBox()
        self.preset.setObjectName("sampleReferenceControl")
        self.preset.setMinimumContentsLength(18)
        self.preset.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.reference_sample = self.preset
        try:
            references = available_reference_samples()
        except Exception as exc:
            references = ()
            self.source_note.setText(f"Reference catalog unavailable: {exc}")
        for reference in references:
            self.preset.addItem(reference.name, reference.key)
        reference_row = QHBoxLayout()
        reference_row.setContentsMargins(0, 0, 0, 0)
        reference_row.addWidget(self.preset, 1)
        self.refresh_references = QPushButton("Refresh")
        self.refresh_references.setObjectName("sampleRefreshReferences")
        self.refresh_references.setToolTip(
            "Reload CIF / MCIF files and optional metadata from configs/reference_samples."
        )
        self.refresh_references.clicked.connect(self._refresh_references)
        reference_row.addWidget(self.refresh_references)
        self.reference_source_widget = QWidget()
        self.reference_source_widget.setLayout(reference_row)
        source_form.addRow("Reference CIF", self.reference_source_widget)
        path_row = QHBoxLayout()
        self.cif_path = QLineEdit()
        self.cif_path.setObjectName("sampleCifPath")
        browse = QPushButton("Open CIF...")
        browse.setObjectName("sampleImportCif")
        browse.clicked.connect(self._browse_cif)
        self.cif_browse = browse
        self.cif_path.editingFinished.connect(self._cif_edited)
        path_row.addWidget(self.cif_path, 1)
        path_row.addWidget(browse)
        path_widget = QWidget()
        path_widget.setLayout(path_row)
        self.cif_source_widget = path_widget
        source_form.addRow("Imported CIF / MCIF", path_widget)
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
        apply_zone = QPushButton("Align CIF zone axis")
        apply_zone.setObjectName("sampleApplyZoneAxis")
        apply_zone.clicked.connect(self._apply_zone_axis)
        self.apply_zone = apply_zone
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
        real_layout.addLayout(tilt_row)
        real_layout.addWidget(apply_tilt)

        self.edit_orientation = QCheckBox("Mouse-drag edits sample orientation")
        self.edit_orientation.setObjectName("sampleEditOrientation")
        self.edit_orientation.setToolTip(
            "Off: mouse orbits the camera. On: left-drag edits a draft sample orientation; Apply commits it."
        )
        self.apply_draft = QPushButton("Apply draft orientation")
        self.apply_draft.setEnabled(False)
        self.apply_draft.clicked.connect(self._commit_draft_orientation)
        real_layout.addWidget(self.edit_orientation)
        real_layout.addWidget(self.apply_draft)

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

        wave = QGroupBox("Wave imaging settings")
        wave.setObjectName("sampleWaveControls")
        wave_form = QFormLayout(wave)
        wave_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.tem_wave_enabled = QCheckBox("TEM image / diffraction")
        self.tem_wave_enabled.setObjectName("sampleTemWaveEnabled")
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
        self.wave_grid.setToolTip(
            "Grid at the configured wave FOV. Larger STEM illumination windows "
            "add pixels to preserve this spacing, within the memory limit."
        )
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
        self.wave_scalar_controls["wave_field_of_view_angstrom"].setToolTip(
            "Wave window, not specimen diameter. STEM may enlarge it for the "
            "scan and defocused probe; vacuum outside the specimen is retained."
        )
        self.tail_enabled = QCheckBox("Approximate Rutherford high-angle tail")
        self.tail_material_source = QComboBox()
        self.tail_material_source.setObjectName("sampleTailMaterialSource")
        self.tail_material_source.addItem("Auto from structure", "structure")
        self.tail_material_source.addItem("Manual", "manual")
        self.tail_screening_source = QComboBox()
        self.tail_screening_source.setObjectName("sampleTailScreeningSource")
        self.tail_screening_source.addItem("Auto (Molière)", "moliere")
        self.tail_screening_source.addItem("Manual", "manual")
        self.tail_material_summary = QLabel()
        self.tail_material_summary.setObjectName("sampleTailMaterialSummary")
        self.tail_material_summary.setWordWrap(True)
        self.tail_material_summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
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
        stem_location = QLabel("STEM imaging: Scanning Image → Scanning Parameters")
        stem_location.setObjectName("sampleStemImageLocation")
        stem_location.setWordWrap(True)
        stem_location.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        stem_location.setToolTip(
            "Enable STEM detector images in Scanning Image. "
            "The wave-propagation settings below are shared by TEM and STEM."
        )
        wave_form.addRow(stem_location)
        wave_form.addRow(self.multislice_enabled)
        wave_form.addRow(self.atomistic_enabled)
        self.illumination_button = QPushButton("Configure illumination pupil / source modes…")
        self.illumination_button.setObjectName("configureIllumination")
        self.illumination_button.clicked.connect(self._configure_illumination)
        self.illumination_summary = QLabel()
        self.illumination_summary.setWordWrap(True)
        wave_form.addRow(self.illumination_button)
        wave_form.addRow(self.illumination_summary)
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
        wave_form.addRow("Tail material", self.tail_material_source)
        wave_form.addRow(self.tail_material_summary)
        wave_form.addRow("Manual atomic number Z", self.tail_atomic_number)
        wave_form.addRow("Manual areal density", self.tail_density)
        wave_form.addRow("Tail screening", self.tail_screening_source)
        wave_form.addRow("Manual screening angle", self.tail_screening)
        wave_form.addRow("Tail maximum angle", self.tail_maximum)
        controls_layout.addWidget(real)
        self.real_group = real

        controls_layout.addWidget(wave)
        self.wave_group = wave

        eds = QGroupBox("EDS signal and specimen support")
        eds.setObjectName("sampleEdsControls")
        eds_form = QFormLayout(eds)
        eds_form.setRowWrapPolicy(
            QFormLayout.RowWrapPolicy.WrapLongRows
        )
        self.eds_enabled = QCheckBox("Enable explicit EDS acquisition")
        self.eds_enabled.setObjectName("sampleEdsEnabled")
        self.eds_support_material = QComboBox()
        self.eds_support_material.setObjectName(
            "sampleEdsSupportMaterial"
        )
        for key, name in available_support_materials():
            self.eds_support_material.addItem(name, key)
        self.eds_support_mesh = QComboBox()
        self.eds_support_mesh.setObjectName("sampleEdsSupportMesh")
        for key, name in available_support_meshes():
            self.eds_support_mesh.addItem(name, key)
        self.eds_solid_angle = QComboBox()
        self.eds_solid_angle.setObjectName("sampleEdsSolidAngle")
        self.eds_solid_angle.addItem(
            "Installed holder-conditioned acceptance",
            "installed_holder",
        )
        self.eds_solid_angle.addItem(
            "Unshadowed reference acceptance", "unshadowed"
        )
        self.eds_transport = QComboBox()
        self.eds_transport.setObjectName("sampleEdsTransportMode")
        self.eds_transport.addItem(
            "Elastic Monte Carlo (finite 3-D geometry)",
            "elastic_monte_carlo",
        )
        self.eds_transport.addItem(
            "Straight primary reference", "straight_primary"
        )
        self.eds_transport.setToolTip(
            "Elastic mode traces seeded event-by-event 3-D paths through the "
            "finite sample and grid. The current screened-Rutherford provider "
            "is provisional for Z>30 and does not model crystal channeling."
        )
        self.eds_scalar_controls = {
            "eds_support_offset_x_um": self._double_control(
                "sampleEdsSupportOffsetX",
                -1.0e6,
                1.0e6,
                suffix=" um",
            ),
            "eds_support_offset_y_um": self._double_control(
                "sampleEdsSupportOffsetY",
                -1.0e6,
                1.0e6,
                suffix=" um",
            ),
            "eds_support_rotation_deg": self._double_control(
                "sampleEdsSupportRotation",
                -360.0,
                360.0,
                suffix=" deg",
            ),
            "eds_detector_efficiency": self._double_control(
                "sampleEdsDetectorEfficiency",
                0.0,
                1.0,
                decimals=5,
            ),
            "eds_spectrum_max_energy_ev": self._double_control(
                "sampleEdsMaximumEnergy",
                1.0,
                1.0e7,
                suffix=" eV",
            ),
            "eds_spectrum_bin_width_ev": self._double_control(
                "sampleEdsBinWidth",
                1.0e-3,
                1.0e6,
                suffix=" eV",
            ),
            "eds_energy_resolution_fwhm_ev": self._double_control(
                "sampleEdsEnergyResolution",
                0.0,
                1.0e6,
                suffix=" eV FWHM",
            ),
        }
        self.eds_scalar_controls[
            "eds_energy_resolution_fwhm_ev"
        ].setSpecialValueText("Ideal line spectrum")
        self.eds_poisson_enabled = QCheckBox("Sample Poisson counts")
        self.eds_poisson_enabled.setObjectName("sampleEdsPoissonEnabled")
        self.eds_poisson_seed = self._integer_control(
            "sampleEdsPoissonSeed", 0, 2_147_483_647
        )
        self.eds_elastic_seed = self._integer_control(
            "sampleEdsElasticSeed", 0, 2_147_483_647
        )
        self.eds_elastic_max_events = self._integer_control(
            "sampleEdsElasticMaximumEvents", 1, 1_000_000
        )
        self.eds_elastic_max_events.setToolTip(
            "Safety guard only. A nonzero event-limit fraction is reported "
            "and means the transport result is truncated."
        )
        self.eds_acquire = QPushButton("Calculate point EDS")
        self.eds_acquire.setObjectName("sampleEdsAcquirePoint")
        self.eds_acquire.setToolTip(
            "Runs only on this button. Editing a support or mechanical "
            "component does not automatically recalculate the EDS spectrum."
        )
        self.eds_summary = QLabel(
            "No EDS point acquisition has been calculated."
        )
        self.eds_summary.setObjectName("sampleEdsSummary")
        self.eds_summary.setWordWrap(True)
        self.eds_summary.setStyleSheet(
            "color: #64748b; font-weight: 600;"
        )
        self.eds_lines = self._table(
            ("Source", "Element", "Transition", "Energy / counts")
        )
        self.eds_lines.setObjectName("sampleEdsLineTable")
        self.eds_lines.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        eds_form.addRow("Calculate", self.eds_enabled)
        eds_form.addRow("Support material", self.eds_support_material)
        eds_form.addRow("Grid mesh", self.eds_support_mesh)
        eds_form.addRow(
            "Grid offset X",
            self.eds_scalar_controls["eds_support_offset_x_um"],
        )
        eds_form.addRow(
            "Grid offset Y",
            self.eds_scalar_controls["eds_support_offset_y_um"],
        )
        eds_form.addRow(
            "Grid rotation",
            self.eds_scalar_controls["eds_support_rotation_deg"],
        )
        eds_form.addRow("Acceptance", self.eds_solid_angle)
        eds_form.addRow("Electron paths", self.eds_transport)
        eds_form.addRow("Elastic seed", self.eds_elastic_seed)
        eds_form.addRow(
            "Maximum events / trajectory", self.eds_elastic_max_events
        )
        eds_form.addRow(
            "Ideal scalar efficiency",
            self.eds_scalar_controls["eds_detector_efficiency"],
        )
        eds_form.addRow(
            "Spectrum maximum",
            self.eds_scalar_controls["eds_spectrum_max_energy_ev"],
        )
        eds_form.addRow(
            "Bin width",
            self.eds_scalar_controls["eds_spectrum_bin_width_ev"],
        )
        eds_form.addRow(
            "Energy resolution",
            self.eds_scalar_controls[
                "eds_energy_resolution_fwhm_ev"
            ],
        )
        eds_form.addRow("Shot noise", self.eds_poisson_enabled)
        eds_form.addRow("Poisson seed", self.eds_poisson_seed)
        eds_form.addRow(self.eds_acquire)
        eds_form.addRow("Result", self.eds_summary)
        eds_form.addRow(self.eds_lines)
        self.eds_group = eds
        # EDS has a dedicated top-level page. Keep this legacy construction
        # temporarily for saved UI-object compatibility, but never display it
        # inside the central Sample editor.
        self.eds_group.hide()

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
        self.scene_status.setStyleSheet("color: #94a3b8;")
        self.scene_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        scene_page = QWidget()
        scene_layout = QVBoxLayout(scene_page)
        scene_header = QHBoxLayout()
        scene_header.addWidget(self.scene_status, 1)
        self.fit_full_sample_button = QPushButton("Fit full sample")
        self.fit_full_sample_button.setObjectName("sampleFitFullSample")
        self.fit_full_sample_button.clicked.connect(self.scene.fit_full_sample)
        self.fit_local_region_button = QPushButton("Fit local region")
        self.fit_local_region_button.setObjectName("sampleFitLocalRegion")
        self.fit_local_region_button.clicked.connect(self.scene.fit_local_region)
        scene_header.addWidget(self.fit_full_sample_button)
        scene_header.addWidget(self.fit_local_region_button)
        scene_layout.addLayout(scene_header)
        self.full_sample_label = QLabel()
        self.local_region_label = QLabel()
        self.atom_display_label = QLabel()
        for label, name, colour in (
            (self.full_sample_label, "sampleFullOutlineLabel", "#38bdf8"),
            (self.local_region_label, "sampleLocalRegionLabel", "#f472b6"),
            (self.atom_display_label, "sampleAtomicDetailLabel", "#e5e7eb"),
        ):
            label.setObjectName(name)
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            label.setStyleSheet(f"color: {colour};")
            scene_layout.addWidget(label)
        self.full_sample_label.setToolTip(
            "The blue wireframe is the full finite sample in laboratory coordinates. "
            "It is not resized to match the local atom display. Use Fit full sample to see all of it."
        )
        self._atom_display_help = (
            "Spheres show equilibrium CIF atoms inside the local material region. "
            "If the rendering limit is reached, the amber box marks the displayed subset; "
            "the pink local-region boundary is unchanged. This is not a frozen-phonon configuration."
        )
        self.atom_display_label.setToolTip(self._atom_display_help)
        structure_row = QHBoxLayout()
        structure_row.setContentsMargins(0, 0, 0, 0)
        structure_row.addWidget(self.scene, 1)
        self.element_legend = ElementLegend()
        structure_row.addWidget(self.element_legend)
        scene_layout.addLayout(structure_row, 1)
        self.eds_trajectory_plot = pg.PlotWidget()
        self.eds_trajectory_plot.setObjectName("sampleEdsElasticTrajectoryPlot")
        self.eds_trajectory_plot.setMinimumHeight(220)
        self.eds_trajectory_plot.setTitle(
            "Elastic trajectories: X-Z projection (+Z downward)"
        )
        self.eds_trajectory_plot.setLabel("bottom", "X displacement", units="nm")
        self.eds_trajectory_plot.setLabel("left", "Z", units="nm")
        self.eds_trajectory_plot.showGrid(x=True, y=True, alpha=0.22)
        self.eds_trajectory_plot.getViewBox().invertY(True)
        self.eds_trajectory_plot.setToolTip(
            "Representative elastic histories projected onto X-Z. Red points "
            "are elastic collisions. Green: forward; orange: reverse; blue: "
            "lateral; red/purple: event/path safety limit."
        )
        self.eds_trajectory_plot.hide()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("samplePageSplitter")
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
        self.envelope_shape.currentIndexChanged.connect(
            self._envelope_shape_changed
        )
        self.preset.currentIndexChanged.connect(self._preset_changed)
        for control, field in (
            (self.tem_wave_enabled, "wave_enabled"),
            (self.multislice_enabled, "wave_multislice_enabled"),
            (self.atomistic_enabled, "wave_atomistic_enabled"),
            (self.frozen_enabled, "wave_frozen_phonon_enabled"),
            (self.real_inelastic_enabled, "real_inelastic_enabled"),
            (self.tail_enabled, "real_high_angle_tail_enabled"),
            (self.eds_enabled, "eds_enabled"),
            (self.eds_poisson_enabled, "eds_poisson_enabled"),
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
        for field, control in self.eds_scalar_controls.items():
            control.valueChanged.connect(
                lambda value, name=field: self._set_scalar(name, value)
            )
        self.eds_support_material.currentIndexChanged.connect(
            lambda _index: self._set_eds_choice(
                "eds_support_material_key",
                self.eds_support_material.currentData(),
            )
        )
        self.eds_support_mesh.currentIndexChanged.connect(
            lambda _index: self._set_eds_choice(
                "eds_support_mesh_key",
                self.eds_support_mesh.currentData(),
            )
        )
        self.eds_solid_angle.currentIndexChanged.connect(
            lambda _index: self._set_eds_choice(
                "eds_solid_angle_mode",
                self.eds_solid_angle.currentData(),
            )
        )
        self.eds_transport.currentIndexChanged.connect(
            lambda _index: self._set_eds_choice(
                "eds_transport_mode", self.eds_transport.currentData()
            )
        )
        self.eds_poisson_seed.valueChanged.connect(
            lambda value: self._set_integer("eds_poisson_seed", value)
        )
        self.eds_elastic_seed.valueChanged.connect(
            lambda value: self._set_integer("eds_elastic_seed", value)
        )
        self.eds_elastic_max_events.valueChanged.connect(
            lambda value: self._set_integer("eds_elastic_max_events", value)
        )
        self.eds_acquire.clicked.connect(self._calculate_eds_point)
        self.element_sigma.editingFinished.connect(
            self._element_sigma_edited
        )
        self.structure_atom_limit.valueChanged.connect(
            lambda _value: self.refresh_snapshot()
        )
        self.tail_atomic_number.valueChanged.connect(
            lambda value: self._set_integer("real_tail_atomic_number", value)
        )
        for control, name in (
            (self.tail_material_source, "real_tail_material_source"),
            (self.tail_screening_source, "real_tail_screening_source"),
        ):
            control.currentIndexChanged.connect(
                lambda _index, combo=control, field=name: self._set_tail_choice(field, combo.currentData())
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
            shape_index = self.envelope_shape.findData(
                str(getattr(sample, "envelope_shape", "rectangle")).lower()
            )
            self.envelope_shape.setCurrentIndex(max(shape_index, 0))
            for field, control in self.scalar_controls.items():
                control.setValue(float(getattr(sample, field)))
            preset_index = self.preset.findData(
                str(sample.reference_sample_key)
            )
            if preset_index < 0:
                self.preset.addItem(f"Missing reference: {sample.reference_sample_key}", sample.reference_sample_key)
                preset_index = self.preset.count() - 1
            self.preset.setCurrentIndex(
                preset_index
            )
            self.cif_path.setText(str(sample.cif_path))
            for controls, values in (
                (self.zone_controls[1], sample.zone_axis_uvw),
                (self.in_plane_controls[1], sample.in_plane_axis_uvw),
            ):
                for control, value in zip(controls, values):
                    control.setValue(int(value))
            self.tem_wave_enabled.setChecked(bool(sample.wave_enabled))
            self._refresh_illumination_summary()
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
            for control, name in (
                (self.tail_material_source, "real_tail_material_source"),
                (self.tail_screening_source, "real_tail_screening_source"),
            ):
                control.setCurrentIndex(control.findData(getattr(sample, name)))
            self.eds_enabled.setChecked(bool(sample.eds_enabled))
            self.eds_poisson_enabled.setChecked(
                bool(sample.eds_poisson_enabled)
            )
            self.eds_poisson_seed.setValue(
                int(sample.eds_poisson_seed)
            )
            self.eds_elastic_seed.setValue(int(sample.eds_elastic_seed))
            self.eds_elastic_max_events.setValue(
                int(sample.eds_elastic_max_events)
            )
            for field, control in self.eds_scalar_controls.items():
                control.setValue(float(getattr(sample, field)))
            for combo, value in (
                (
                    self.eds_support_material,
                    sample.eds_support_material_key,
                ),
                (self.eds_support_mesh, sample.eds_support_mesh_key),
                (self.eds_solid_angle, sample.eds_solid_angle_mode),
                (self.eds_transport, sample.eds_transport_mode),
            ):
                combo_index = combo.findData(str(value))
                combo.setCurrentIndex(
                    combo_index if combo_index >= 0 else 0
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
            self._draft_quaternion = sample_orientation_quaternion(sample)
            self.apply_draft.setEnabled(False)
            self._update_mode_controls()
            self._update_envelope_controls()
            self._update_wave_controls()
            self._update_eds_controls()
            self._refresh_inelastic_summary()
            self._refresh_tail_summary()
        finally:
            self._updating = False
        self._reference_revision = self._reference_source_revision()
        self.refresh_snapshot()

    def _set_scalar(self, name, value):
        if self._updating or self._state is None:
            return
        if name in {"size_x_nm", "size_y_nm", "thickness_nm"} and value <= 0.0:
            self.error.emit(f"{name} must be positive.")
            return
        setattr(self._state.sample, name, float(value))
        if (
            name == "size_x_nm"
            and str(getattr(self._state.sample, "envelope_shape", ""))
            == "disk"
        ):
            self._state.sample.size_y_nm = float(value)
            self._updating = True
            try:
                self.scalar_controls["size_y_nm"].setValue(float(value))
            finally:
                self._updating = False
        self._changed(f"sample.{name}")

    def _envelope_shape_changed(self):
        if self._updating or self._state is None:
            return
        shape = str(self.envelope_shape.currentData())
        self._state.sample.envelope_shape = shape
        if shape == "disk":
            diameter = float(self._state.sample.size_x_nm)
            self._state.sample.size_y_nm = diameter
            self._updating = True
            try:
                self.scalar_controls["size_y_nm"].setValue(diameter)
            finally:
                self._updating = False
        self._update_envelope_controls()
        self._changed("sample.envelope_shape")

    def _update_envelope_controls(self):
        disk = str(self.envelope_shape.currentData()) == "disk"
        self.scalar_labels["size_x_nm"].setText(
            "Diameter" if disk else "Size X"
        )
        self.scalar_labels["size_y_nm"].setVisible(not disk)
        self.scalar_controls["size_y_nm"].setVisible(not disk)

    def _set_bool(self, name, value):
        if self._updating or self._state is None:
            return
        setattr(self._state.sample, name, bool(value))
        if name in {
            "wave_enabled",
            "wave_multislice_enabled",
            "wave_atomistic_enabled",
            "wave_frozen_phonon_enabled",
            "real_inelastic_enabled",
            "eds_enabled",
            "eds_poisson_enabled",
        }:
            self._update_wave_controls()
            self._update_eds_controls()
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

    def _set_eds_choice(self, name, value):
        if self._updating or self._state is None:
            return
        setattr(self._state.sample, name, str(value))
        self._update_eds_controls()
        self._changed(f"sample.{name}")

    def _preset_changed(self):
        if self._updating or self._state is None:
            return
        key = str(self.preset.currentData() or "")
        try:
            apply_reference_sample(self._state.sample, key)
        except Exception as exc:
            self.error.emit(str(exc))
            self.set_state(self._state)
            return
        self.set_state(self._state)
        self._changed("sample.reference_sample_key")

    def _reference_source_revision(self):
        if self._state is None or str(self._state.sample.specimen_mode).strip().lower() != "reference":
            return None
        key = str(self._state.sample.reference_sample_key)
        try:
            reference = get_reference_sample(key)
            revision = [key]
            for path in (reference.cif_path, reference.metadata_path):
                if path is None:
                    revision.append(None)
                else:
                    with path.open("rb") as stream:
                        revision.append((str(path), hashlib.file_digest(stream, "sha256").hexdigest()))
            return tuple(revision)
        except Exception as exc:
            return (key, "unavailable", str(exc))

    def _reference_source_refreshed(self):
        revision = self._reference_source_revision()
        changed = revision != self._reference_revision
        self._reference_revision = revision
        if changed and revision is not None:
            # This is a file-content change rather than a selection change.
            # Preserve the user's orientation/dimensions while invalidating
            # calculations and notifying the normal main-window update path.
            self._changed("sample.reference_source")
            return True
        return False

    def _refresh_references(self):
        key = (str(self._state.sample.reference_sample_key)
               if self._state is not None else str(self.preset.currentData() or ""))
        try:
            refresh_reference_samples()
            references = available_reference_samples()
        except Exception as exc:
            self.source_note.setText(f"Reference catalog unavailable: {exc}")
            self.error.emit(str(exc))
            self._reference_source_refreshed()
            return
        self.preset.blockSignals(True)
        try:
            self.preset.clear()
            for reference in references:
                self.preset.addItem(reference.name, reference.key)
            index = self.preset.findData(key)
            if index < 0:
                self.preset.addItem(f"Missing reference: {key}", key)
                index = self.preset.count() - 1
            self.preset.setCurrentIndex(index)
        finally:
            self.preset.blockSignals(False)
        self._update_mode_controls()
        self._update_wave_controls()
        if not self._reference_source_refreshed():
            self._refresh_tail_summary()
            self.refresh_snapshot()

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
        mode = str(self.mode.currentData())
        try:
            if mode == "reference":
                apply_reference_sample(self._state.sample, self._state.sample.reference_sample_key)
            else:
                self._state.sample.specimen_mode = mode
        except Exception as exc:
            self.error.emit(str(exc))
            self.set_state(self._state)
            return
        self.set_state(self._state)
        self._changed("sample.specimen_mode")

    def _structure_path(self):
        if self._state is None:
            return ""
        try:
            return active_cif_path(self._state.sample)
        except (ValueError, OSError):
            return ""

    def _update_mode_controls(self):
        atomic = str(self.mode.currentData()) == "atomic"
        self.reference_source_widget.setEnabled(not atomic)
        self.cif_source_widget.setEnabled(atomic)
        path = self._structure_path()
        self.apply_zone.setEnabled(bool(path))
        detail = f"Structure: {path}" if path else "No available CIF structure. Select an existing reference or open a CIF."
        if not path and self._state is not None:
            try:
                active_cif_path(self._state.sample)
            except (ValueError, OSError) as exc:
                detail = f"CIF structure unavailable: {exc}"
        self.source_note.setText(f"Structure: {Path(path).name}" if path else detail)
        self.source_note.setToolTip(detail)

    def _update_wave_controls(self):
        illumination = str(
            getattr(self._state, "illumination_mode", "TEM")
            if self._state is not None
            else "TEM"
        ).upper()
        structure_available = bool(self._structure_path())
        tem_available = structure_available and illumination == "TEM"
        self.tem_wave_enabled.setEnabled(tem_available)
        self.tem_wave_enabled.setToolTip(
            "Calculate the local specimen-to-Objective image and exit-wave "
            "diffraction diagnostic."
            if tem_available
            else "TEM wave imaging requires a reference or imported CIF "
            "structure, plus Microprobe (TEM) illumination."
        )

        inelastic_enabled = (
            structure_available and self.real_inelastic_enabled.isChecked()
        )
        self.real_inelastic_enabled.setEnabled(structure_available)
        for control in self.inelastic_scalar_controls.values():
            control.setEnabled(inelastic_enabled)

        multislice = (
            structure_available and self.multislice_enabled.isChecked()
        )
        atomistic = multislice and self.atomistic_enabled.isChecked()
        frozen = atomistic and self.frozen_enabled.isChecked()
        self.multislice_enabled.setEnabled(structure_available)
        self.atomistic_enabled.setEnabled(multislice)
        self.frozen_enabled.setEnabled(atomistic)
        for control in (
            self.frozen_configurations,
            self.frozen_sigma,
            self.frozen_seed,
            self.element_sigma,
        ):
            control.setEnabled(frozen)

    def _update_eds_controls(self):
        enabled = (
            self._state is not None and self.eds_enabled.isChecked()
        )
        material_is_vacuum = (
            str(self.eds_support_material.currentData()) == "vacuum"
        )
        elastic = (
            str(self.eds_transport.currentData()) == "elastic_monte_carlo"
        )
        for control in (
            self.eds_support_material,
            self.eds_solid_angle,
            self.eds_transport,
            self.eds_poisson_enabled,
            self.eds_acquire,
            *self.eds_scalar_controls.values(),
        ):
            control.setEnabled(enabled)
        self.eds_support_mesh.setEnabled(
            enabled and not material_is_vacuum
        )
        self.eds_poisson_seed.setEnabled(
            enabled and self.eds_poisson_enabled.isChecked()
        )
        for control in (
            self.eds_elastic_seed,
            self.eds_elastic_max_events,
        ):
            control.setEnabled(enabled and elastic)

    def _calculate_eds_point(self):
        if self._state is None or self._result is None:
            self.error.emit(
                "Run a column calculation before the explicit EDS acquisition."
            )
            return
        if not self.eds_enabled.isChecked():
            self.eds_summary.setText("EDS acquisition is disabled.")
            return
        try:
            from temsim.component_keys import EDS_DETECTOR_SYSTEM
            from temsim.detector.eds_geometry import (
                EDSDetectorArrayGeometry,
            )
            from temsim.specimen.interaction_engine import (
                run_specimen_interactions,
            )
            from temsim.specimen.interaction_types import (
                SpecimenInteractionRequest,
            )

            assembly = getattr(self._result, "assembly", None)
            if assembly is None:
                raise ValueError(
                    "The current result has no installed EDS geometry."
                )
            part = assembly.part(EDS_DETECTOR_SYSTEM)
            geometry = EDSDetectorArrayGeometry.from_part_data(part.data)
            interactions = run_specimen_interactions(
                self._state,
                getattr(self._result, "simulation", None),
                SpecimenInteractionRequest.eds_point(),
                detector_geometry=geometry,
            )
            spectrum = interactions.eds_spectrum
            if spectrum is None:
                raise RuntimeError(
                    "Specimen interaction engine returned no EDS spectrum"
                )
        except Exception as exc:
            self.error.emit(str(exc))
            self.eds_summary.setText(f"EDS calculation failed: {exc}")
            return
        self._specimen_interactions = interactions
        self._eds_result = spectrum
        self._elastic_result = spectrum.elastic_transport
        self._plot_elastic_trajectories(self._elastic_result)
        ordered = sorted(
            spectrum.lines,
            key=lambda line: line.expected_detected_counts,
            reverse=True,
        )[:24]
        self.eds_lines.setRowCount(len(ordered))
        from ase.data import chemical_symbols

        for row, line in enumerate(ordered):
            values = (
                line.source_key,
                chemical_symbols[line.atomic_number],
                f"{line.subshell} / {line.transition}",
                (
                    f"{line.energy_ev * 1.0e-3:.6g} keV | "
                    f"{line.expected_detected_counts:.6g}"
                ),
            )
            for column, value in enumerate(values):
                self.eds_lines.setItem(
                    row, column, QTableWidgetItem(str(value))
                )
        sampled_text = ""
        if spectrum.sampled_counts is not None:
            sampled_text = (
                f"; sampled {int(np.sum(spectrum.sampled_counts))} counts"
            )
        source_names = ", ".join(
            dict.fromkeys(line.source_key for line in spectrum.lines)
        ) or "vacuum only"
        if self._elastic_result is None:
            transport_text = (
                "Straight-primary reference; no elastic trajectory generation."
            )
        else:
            metrics = self._elastic_result.metrics
            transport_text = (
                f"Elastic MC: {metrics['trajectory_count']} trajectories, "
                f"{metrics['mean_elastic_events_per_trajectory']:.6g} mean "
                "events/trajectory, "
                f"{100.0 * metrics['transmitted_fraction']:.5g}% forward, "
                f"{100.0 * metrics['backscattered_fraction']:.5g}% reverse. "
                "Relativistic screened-Rutherford fallback; elastic energy "
                "loss and inelastic angular deflection are not included."
            )
            if metrics["rutherford_heavy_element_warning"]:
                transport_text += (
                    " Z>30 was encountered, where ELSEPA cross sections are "
                    "recommended for quantitative work."
                )
        self.eds_summary.setText(
            f"EDS point: {spectrum.total_expected_counts:.6g} expected "
            f"counts{sampled_text}; {len(spectrum.lines)} characteristic "
            f"transitions; sources: {source_names}. {transport_text} "
            "Bremsstrahlung is not yet included."
        )
        self.eds_summary.setToolTip(
            "\n".join(
                f"{key}: {value}"
                for key, value in spectrum.metrics.items()
            )
        )

    def _plot_elastic_trajectories(self, transport):
        self.eds_trajectory_plot.clear()
        if transport is None or not transport.trajectories:
            return
        colours = {
            "transmitted": (34, 197, 94, 150),
            "backscattered": (249, 115, 22, 180),
            "lateral_escape": (59, 130, 246, 170),
            "event_limit": (239, 68, 68, 200),
            "path_limit": (168, 85, 247, 200),
        }
        event_x = []
        event_z = []
        for trajectory in transport.trajectories:
            points = trajectory.points_nm
            if points.shape[0] < 2:
                continue
            x_displacement = points[:, 0] - points[0, 0]
            self.eds_trajectory_plot.plot(
                x_displacement,
                points[:, 2],
                pen=pg.mkPen(
                    colours.get(trajectory.outcome, (100, 116, 139, 150)),
                    width=1.15,
                ),
            )
            event_x.extend(
                event.position_nm[0] - points[0, 0]
                for event in trajectory.events
            )
            event_z.extend(event.position_nm[2] for event in trajectory.events)
        if event_x:
            self.eds_trajectory_plot.plot(
                event_x,
                event_z,
                pen=None,
                symbol="o",
                symbolSize=3.5,
                symbolPen=None,
                symbolBrush=pg.mkBrush(239, 68, 68, 155),
            )
        self.eds_trajectory_plot.enableAutoRange()

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
            detail_text = (
                f"{distribution.material_name}; total λ {mfp(distribution.total_inelastic_mean_free_path_nm)}, "
                f"plasmon λ {mfp(distribution.plasmon_mean_free_path_nm)}, "
                f"ionisation λ {mfp(distribution.ionisation_mean_free_path_nm)}; "
                f"t/λ {distribution.mean_inelastic_events:.6g}. "
                f"{channel_text}; effective absorption "
                f"{100.0 * distribution.absorbed_probability:.5g}%."
            )
            self.inelastic_summary.setText(
                f"{distribution.material_name} | total λ "
                f"{mfp(distribution.total_inelastic_mean_free_path_nm)} | "
                f"t/λ {distribution.mean_inelastic_events:.6g} | absorption "
                f"{100.0 * distribution.absorbed_probability:.5g}%"
            )
            self.inelastic_summary.setToolTip(
                detail_text
                + "\n\n"
                + "\n".join(
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

    def _refresh_illumination_summary(self):
        from temsim.physics.illumination import illumination_config, source_nodes
        try:
            config = illumination_config(self._state)
            self.illumination_summary.setText(f"{config['model']} | {len(source_nodes(config))} source mode(s)")
        except (ValueError, TypeError) as exc:
            self.illumination_summary.setText(str(exc))

    def _configure_illumination(self):
        if self._state is None or self._updating:
            return
        from temsim.gui.illumination_dialog import IlluminationDialog
        dialog = IlluminationDialog(self._state.sample.wave_illumination, self._state.beam_voltage_kv, self)
        if dialog.exec():
            self._state.sample.wave_illumination = dialog.config
            self._refresh_illumination_summary()
            self._changed("wave_illumination")

    def _changed(self, name):
        self._eds_result = None
        self._elastic_result = None
        self._specimen_interactions = None
        self.eds_summary.setText(
            "EDS settings or specimen state changed; press Calculate point EDS."
        )
        self.eds_lines.setRowCount(0)
        self.eds_trajectory_plot.clear()
        self._refresh_inelastic_summary()
        self._refresh_tail_summary()
        self.refresh_snapshot()
        self.parameters_changed.emit(name)

    def _set_tail_choice(self, name, value):
        if self._updating or self._state is None:
            return
        setattr(self._state.sample, name, str(value))
        self._changed(f"sample.{name}")

    def _refresh_tail_summary(self):
        manual_material = self.tail_material_source.currentData() == "manual"
        manual_screening = self.tail_screening_source.currentData() == "manual"
        self.tail_atomic_number.setEnabled(manual_material)
        self.tail_density.setEnabled(manual_material)
        self.tail_screening.setEnabled(manual_screening)
        if self._state is None:
            return
        try:
            material = resolve_tail_material(
                self._state.sample, self._state.sample.thickness_nm,
                self._state.beam_voltage_kv,
            )
            from ase.data import chemical_symbols
            rows = [material.provenance]
            for element in material.elements:
                density = (f"{element.number_density_atoms_nm3:.6g} atoms/nm³; "
                           if element.number_density_atoms_nm3 is not None else "")
                rows.append(
                    f"{chemical_symbols[element.atomic_number]} (Z={element.atomic_number}): "
                    f"{density}{element.areal_density_atoms_nm2:.6g} atoms/nm²; "
                    f"screening {element.screening_angle_mrad:.6g} mrad"
                )
            self.tail_material_summary.setText("\n".join(rows))
            self.tail_material_summary.setToolTip("\n".join((*rows, *material.warnings)))
        except Exception as exc:
            message = f"Tail material unavailable: {exc}"
            self.tail_material_summary.setText(message)
            self.tail_material_summary.setToolTip(message)

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
        path = self.cif_path.text().strip()
        self._state.sample.cif_path = path
        self._update_mode_controls()
        self._update_wave_controls()
        self._changed("sample.cif_path")

    def _apply_zone_axis(self):
        if self._state is None:
            return
        try:
            path = Path(active_cif_path(self._state.sample)).expanduser()
            if not path.is_file():
                raise ValueError("Select an existing CIF before aligning a zone axis.")
            from temsim.specimen.cif_io import read_cif_atoms

            atoms = read_cif_atoms(path)
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
            "Orientation draft | Apply to update the specimen"
        )
        self.local_region_label.setText("Local orientation preview | Not a calculated atom state")

    def _commit_draft_orientation(self):
        if self._state is None:
            return
        set_sample_orientation(self._state.sample, self._draft_quaternion)
        self.apply_draft.setEnabled(False)
        self._changed("sample.specimen_orientation_quaternion_wxyz")

    def refresh_snapshot(self, calculation_result=None):
        if self._state is None:
            return
        result = calculation_result or self._result
        if not self.isVisible():
            # Tab switches and hidden live previews retain only the newest
            # requested source; no CIF reads, atoms, bonds or GL work here.
            self._pending_calculation_result = result
            self._refresh_pending = True
            return
        self._refresh_pending = False
        self._pending_calculation_result = None
        atom_error = None
        try:
            source = resolve_sample_display_source(self._state, result)
            sample = source.sample
            geometry_options = dict(
                scan_x_um=source.scan_x_um, scan_y_um=source.scan_y_um,
                current_probe_nm=source.current_probe_nm, probe_padding_nm=source.probe_padding_nm,
                calculation_roi_bounds_nm_override=source.calculation_roi_bounds_nm_override,
                maximum_display_atoms=self.structure_atom_limit.value(),
            )
            try:
                snapshot = build_sample_geometry_snapshot(sample, load_atoms=True, **geometry_options)
            except Exception as exc:
                # Missing/invalid CIF data must not erase valid known material
                # dimensions. If geometry itself is invalid this also raises.
                snapshot = build_sample_geometry_snapshot(sample, load_atoms=False, **geometry_options)
                atom_error = str(exc)
        except Exception as exc:
            self.scene_status.setText(f"Sample geometry unavailable: {exc}")
            self.scene.clear()
            self.scene._snapshot = None
            self._snapshot = None
            for label in (self.full_sample_label, self.local_region_label, self.atom_display_label):
                label.clear()
            self.fit_full_sample_button.setEnabled(False)
            self.fit_local_region_button.setEnabled(False)
            self.element_legend.set_atomic_numbers(())
            return
        self._snapshot = snapshot
        self.scene.display_snapshot(snapshot)
        self.fit_full_sample_button.setEnabled(True)
        self.fit_local_region_button.setEnabled(snapshot.local_material_bounds_nm is not None)
        for label, text in zip(
            (self.full_sample_label, self.local_region_label, self.atom_display_label),
            sample_scene_labels(snapshot, completed_region=source.completed_region),
        ):
            label.setText(text)
        self.local_region_label.setToolTip(source.provenance_detail)
        self.atom_display_label.setStyleSheet(
            "color: #fbbf24;" if snapshot.atom_display_capped or atom_error else "color: #e5e7eb;"
        )
        self.atom_display_label.setToolTip(self._atom_display_help)
        if atom_error:
            self.atom_display_label.setText("Spheres unavailable | Geometry retained")
            self.atom_display_label.setToolTip(f"Atomic preview unavailable: {atom_error}")
        self.element_legend.set_atomic_numbers(snapshot.atomic_numbers)
        backend = (
            f"3-D OpenGL ({self.scene.opengl_detail})"
            if self.scene.opengl_available
            else f"safe 2-D fallback ({self.scene.opengl_detail})"
        )
        mode = "Open CIF" if snapshot.mode == "atomic" else "Reference CIF"
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
        detail_text = (
            f"{mode} | {'INSERTED' if snapshot.inserted else 'RETRACTED'} | "
            f"{backend}{atom_detail}"
            + (f"\n{warning}" if warning else "")
        )
        warning_count = len(snapshot.warnings) + int(atom_error is not None)
        self.scene_status.setText(
            f"{source.provenance_label} | {'inserted' if snapshot.inserted else 'retracted'}"
            + (f" | {warning_count} warning(s)" if warning_count else "")
        )
        self.scene_status.setToolTip(
            source.provenance_detail + "\n" + detail_text
            + (f"\nAtomic preview unavailable: {atom_error}" if atom_error else "")
        )

    def display_result(self, result, stem_frame=None):
        """Refresh specimen geometry; detector images belong to the STEM page."""

        self._result = result
        self.refresh_snapshot(result)

    def showEvent(self, event):
        super().showEvent(event)
        # Defer until the splitter/viewport has its restored size, including
        # first display. Duplicate queued events consume only one refresh.
        if self._refresh_pending:
            QTimer.singleShot(0, self._flush_pending_snapshot)

    def _flush_pending_snapshot(self):
        if self.isVisible() and self._refresh_pending:
            self.refresh_snapshot(self._pending_calculation_result)
