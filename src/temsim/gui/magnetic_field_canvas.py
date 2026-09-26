"""Bounded orthographic 3-D display of already calculated magnetic field lines.

Input positions remain physical XYZ metres (Z is the column axis).  A transverse
display gain is applied only while projecting; this widget never evaluates a
field or changes the model.  The Ray Diagram's projection angle rebuilds
vectorised, colour-batched paths; the field view has no independent rotation.
Pan, zoom, resize and repaint reuse those paths without walking line segments.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPainterPath, QPen, QPixmap, QTransform, QWheelEvent
from PySide6.QtWidgets import QWidget
from pyqtgraph import AxisItem
from pyqtgraph.functions import arrayToQPath, siScale

from temsim.magnetic_field_lines import field_strength_fraction
from temsim.gui.test_electron_types import ElectronPath


_COLOUR_BINS = 24
_MAX_DIRECTION_ARROWS = 160
_MAX_ELECTRON_ARROWS = 12
_ELECTRON_LINE_WIDTH = 1.0
_SELECTED_ELECTRON_LINE_WIDTH = 1.4
_ELECTRON_FIELD_OPACITY = 0.75


@dataclass(frozen=True)
class _ElectronProjection:
    positions_m: np.ndarray
    revision: int
    segments: np.ndarray
    path: QPainterPath
    markers: tuple[tuple[str, np.ndarray], ...]
    arrows: np.ndarray


def _segments(value: object, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (0,):
        array = np.empty((0, 2, 3), dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (2, 3) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite XYZ segments with shape (N, 2, 3)")
    return np.array(array, dtype=np.float64, order="C", copy=True)


def _strength_colour(fraction: float) -> QColor:
    """Sequential blue/cyan/yellow palette shared by lines and local legend."""
    stops = ((0.0, (54, 94, 236)), (0.5, (28, 209, 211)), (1.0, (253, 224, 77)))
    fraction = min(1.0, max(0.0, float(fraction)))
    low, high = (stops[0], stops[1]) if fraction <= 0.5 else (stops[1], stops[2])
    weight = (fraction - low[0]) / (high[0] - low[0])
    return QColor(*(int(round(a + weight * (b - a))) for a, b in zip(low[1], high[1])))


def _paired_path(points: np.ndarray) -> QPainterPath:
    if not points.size:
        return QPainterPath()
    flat = np.ascontiguousarray(points.reshape(-1, 2))
    return arrayToQPath(flat[:, 0], flat[:, 1], connect="pairs", finiteCheck=False)


def _clip_axial_segments(segments: np.ndarray, lower: float, upper: float) -> tuple[np.ndarray, np.ndarray]:
    """Clip display segments to an axial slab without changing physical data."""
    z = segments[:, :, 2]
    selected = np.minimum(z[:, 0], z[:, 1]) <= upper
    selected &= np.maximum(z[:, 0], z[:, 1]) >= lower
    visible = np.array(segments[selected], copy=True)
    if len(visible):
        start = visible[:, 0].copy()
        delta = visible[:, 1] - start
        axial = delta[:, 2]
        low = np.divide(lower - start[:, 2], axial, out=np.zeros_like(axial), where=axial != 0.0)
        high = np.divide(upper - start[:, 2], axial, out=np.ones_like(axial), where=axial != 0.0)
        first = np.clip(np.minimum(low, high), 0.0, 1.0)
        last = np.clip(np.maximum(low, high), 0.0, 1.0)
        visible[:, 0] = start + first[:, None] * delta
        visible[:, 1] = start + last[:, None] * delta
    return visible, selected


class MagneticFieldCanvas(QWidget):
    """A software-rendered 3-D magnetic scene projected like the Ray Diagram.

    Dragging pans; the wheel zooms around the cursor.
    Double click fits the visible volume.  Axial and transverse axes are fitted
    independently, with their actual display enlargement labelled in the footer.
    """

    view_range_changed = Signal(object, object)
    plot_geometry_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(220, 180)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setToolTip(
            "Projection follows the Ray Diagram. Drag: pan. Wheel: zoom about the pointer. "
            "Wheel over the bottom Z axis or left U axis scales only that direction. "
            "Double click: fit. Transverse enlargement changes only the display. "
            "Each virtual electron keeps its own colour: a circle marks its start, "
            "a square its end, and arrows follow chronological order. "
            "The selected electron is highlighted. Transverse display enlargement means "
            "screen angles are not physical electron angles; use the direction readout in Virtual electrons."
        )
        self._segments_m = np.empty((0, 2, 3), dtype=np.float64)
        self._strengths_t = np.empty(0, dtype=np.float64)
        self._direction_segments_m = np.empty((0, 2, 3), dtype=np.float64)
        self._bounds_m = np.array([[-0.001, -0.001, 0.0], [0.001, 0.001, 1.0]])
        self._labels: tuple[tuple[str, np.ndarray], ...] = ()
        self._reference_t = 0.0
        self._transverse_gain = 1.0
        self._projection_angle_deg = 0.0
        self._axial_range_m: tuple[float, float] | None = None
        self._zoom = 1.0
        self._pan = QPointF()
        self._navigation_range_mm = None
        self._horizontal_plot_edges = None
        self._wheel_scale_factor = -1.0 / 8.0  # pyqtgraph ViewBox default, as in Ray Diagram.
        self._tick_axes = {name: AxisItem(orientation=name) for name in ("bottom", "left")}
        self._drag_position: QPointF | None = None
        self._drag_button = Qt.MouseButton.NoButton
        self._drag_axis = None
        self._colour_paths: tuple[tuple[QColor, QPainterPath], ...] = ()
        self._field_projected_arrows = np.empty((0, 2, 2))
        self._field_arrow_screen_key = None
        self._field_arrow_screen_path = QPainterPath()
        self._field_layer_key = None
        self._field_layer_pixmap = None
        self._bounds_path = QPainterPath()
        self._axis_path = QPainterPath()
        self._projected_labels: tuple[tuple[str, np.ndarray], ...] = ()
        self._label_layout_key: tuple | None = None
        self._label_placements: tuple[tuple[str, QPointF, QRectF, QPointF], ...] = ()
        self._basis = np.eye(3, dtype=np.float64)
        self._centre_m = np.mean(self._bounds_m, axis=0)
        self._display_scale_m = 1.0
        self._projection_revision = 0
        self._field_lines_visible = True
        self._electron_mode = False
        self._electron_paths: tuple[ElectronPath, ...] = ()
        self._electron_projections: dict[str, _ElectronProjection] = {}
        self._rebuild_projection()

    def sizeHint(self) -> QSize:
        return QSize(850, 450)

    @property
    def segment_count(self) -> int:
        return len(self._segments_m)

    @property
    def transverse_gain(self) -> float:
        return self._transverse_gain

    @property
    def projection_angle_deg(self) -> float:
        return self._projection_angle_deg

    @property
    def electron_point_count(self) -> int:
        return sum(len(path.positions_m) for path in self._electron_paths)

    @property
    def electron_path_count(self) -> int:
        return len(self._electron_paths)

    def set_electron_paths(self, paths: Iterable[ElectronPath]) -> None:
        """Show the supplied independent electrons in their original time order.

        This is detached display data, not a column source or checkpoint. No
        sorting by Z is allowed: a turning electron can visit a plane repeatedly.
        A subset is a visibility choice; only supplied paths contribute to fit.
        Updating overlays leaves cached magnetic field drawing paths untouched.
        """
        checked = tuple(paths)
        if any(not isinstance(path, ElectronPath) for path in checked):
            raise TypeError("Each displayed electron must be an ElectronPath")
        if len({path.key for path in checked}) != len(checked):
            raise ValueError("Displayed electron path keys must be unique")
        self._electron_paths = checked
        self._rebuild_electron_projection()
        self._update_visible_fit()
        self.update()

    def clear_electron_paths(self) -> None:
        self.set_electron_paths(())

    def set_electron_mode(self, enabled: bool) -> None:
        """Highlight electrons while retaining visible field context at its true scale."""
        if bool(enabled) != self._electron_mode:
            self._electron_mode = bool(enabled)
            self._update_visible_fit()
            self.update()

    def set_field_lines_visible(self, visible: bool) -> None:
        if bool(visible) != self._field_lines_visible:
            self._field_lines_visible = bool(visible)
            self._update_visible_fit()
            self.update()

    def set_geometry(
        self,
        segments_m: np.ndarray,
        strengths_t: np.ndarray,
        *,
        reference_t: float,
        bounds_m: np.ndarray,
        direction_segments_m: np.ndarray | None = None,
        labels: Iterable[tuple[str, np.ndarray]] | None = None,
    ) -> None:
        """Replace a detached scene, validating everything before changing it.

        ``strengths_t`` has one non-negative field magnitude in tesla per segment.
        ``bounds_m`` is the finite ``(minimum XYZ, maximum XYZ)`` display volume.
        Optional arrows point from the first endpoint to the second; their
        direction must be supplied by the field model, never inferred here.
        """
        segments = _segments(segments_m, "segments_m")
        strengths = np.array(strengths_t, dtype=np.float64, copy=True)
        if strengths.shape != (len(segments),) or not np.isfinite(strengths).all() or np.any(strengths < 0.0):
            raise ValueError("strengths_t must be finite non-negative magnitudes with shape (N,)")
        reference = float(reference_t)
        if not math.isfinite(reference) or reference < 0.0:
            raise ValueError("reference_t must be finite and non-negative")
        if reference == 0.0 and np.any(strengths > 0.0):
            raise ValueError("reference_t must be positive for a non-zero field")
        bounds = np.array(bounds_m, dtype=np.float64, copy=True)
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] < bounds[0]):
            raise ValueError("bounds_m must contain finite ordered minimum and maximum XYZ coordinates")
        directions = _segments([] if direction_segments_m is None else direction_segments_m, "direction_segments_m")
        checked_labels = []
        for text, position in (() if labels is None else labels):
            point = np.array(position, dtype=np.float64, copy=True)
            if point.shape != (3,) or not np.isfinite(point).all():
                raise ValueError("label positions must contain three finite XYZ coordinates")
            checked_labels.append((str(text), point))
        for array in (segments, strengths, bounds, directions):
            array.setflags(write=False)
        self._segments_m = segments
        self._strengths_t = strengths
        self._bounds_m = bounds
        self._reference_t = reference
        self._direction_segments_m = directions
        self._labels = tuple(checked_labels)
        self._rebuild_projection()
        self.update()

    def clear_field_lines(self) -> None:
        """Clear obsolete field geometry while retaining an independent electron."""
        self._segments_m = np.empty((0, 2, 3), dtype=np.float64)
        self._strengths_t = np.empty(0, dtype=np.float64)
        self._direction_segments_m = np.empty((0, 2, 3), dtype=np.float64)
        self._reference_t = 0.0
        self._labels = ()
        self._rebuild_projection()
        self.update()

    def clear(self) -> None:
        self._electron_paths = ()
        self._electron_projections = {}
        self.clear_field_lines()

    def fit_view(self) -> None:
        self._navigation_range_mm = None
        self._zoom = 1.0
        self._pan = QPointF()
        # Fit is an explicit navigation choice. Later worker publications and
        # mode changes must retain this viewport until the user fits again.
        self._navigation_range_mm = self.view_range_mm()
        self.update()
        self.view_range_changed.emit(*self._navigation_range_mm)

    def set_wheel_scale_factor(self, factor: float) -> None:
        factor = float(factor)
        if not math.isfinite(factor):
            raise ValueError("wheel scale factor must be finite")
        self._wheel_scale_factor = factor

    def view_range_mm(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Visible physical (Z, U) limits in millimetres; display gain is excluded."""
        if self._navigation_range_mm is not None:
            return self._navigation_range_mm
        plot = self._plot_rect()
        lower = self.screen_to_physical_mm(plot.bottomLeft())
        upper = self.screen_to_physical_mm(plot.topRight())
        return ((lower.x(), upper.x()), (lower.y(), upper.y()))

    def set_view_range_mm(self, x_range, y_range, *, emit=False) -> None:
        """Navigate in physical coordinates without recomputing any field or path.

        Legacy axial fitting may have clipped cached display paths. Leaving that
        mode restores their full projection once; subsequent navigation only
        changes the screen transform. Neither operation alters physical arrays.
        """
        ranges = tuple(tuple(float(value) for value in interval) for interval in (x_range, y_range))
        if any(len(interval) != 2 or not all(math.isfinite(value) for value in interval)
               or interval[1] <= interval[0] for interval in ranges):
            raise ValueError("physical view ranges must be finite and increasing")
        if self._axial_range_m is not None:
            self._axial_range_m = None
            self._rebuild_projection()
        if ranges != self._navigation_range_mm:
            self._navigation_range_mm = ranges
            self._label_layout_key = None
            self.update()
            if emit:
                self.view_range_changed.emit(*ranges)

    def screen_to_physical_mm(self, position: QPointF) -> QPointF:
        projected = self._screen_transform().inverted()[0].map(position)
        centre_u = -float(self._basis[1] @ self._centre_m)
        return QPointF((projected.x() * self._display_scale_m + self._centre_m[2]) * 1e3,
                       (-projected.y() * self._display_scale_m / self._transverse_gain + centre_u) * 1e3)

    def physical_mm_to_screen(self, z_mm: float, u_mm: float) -> QPointF:
        centre_u = -float(self._basis[1] @ self._centre_m)
        projected = QPointF((float(z_mm) * 1e-3 - self._centre_m[2]) / self._display_scale_m,
                            -(float(u_mm) * 1e-3 - centre_u) * self._transverse_gain / self._display_scale_m)
        return self._screen_transform().map(projected)

    def set_transverse_gain(self, gain: float) -> None:
        gain = float(gain)
        if not math.isfinite(gain) or gain <= 0.0:
            raise ValueError("transverse display gain must be positive and finite")
        if gain != self._transverse_gain:
            self._transverse_gain = gain
            self._rebuild_projection()
            self.update()

    def set_projection_angle(self, angle_deg: float) -> None:
        """Use the Ray Diagram's U = X cos(theta) + Y sin(theta) projection.

        Z remains horizontal and positive U points upwards.  The orthogonal
        V = -X sin(theta) + Y cos(theta) coordinate is depth, never mixed into
        either screen coordinate.  Changing angle only projects cached geometry;
        it neither evaluates the field nor alters pan or zoom.
        """
        angle = float(angle_deg)
        if not math.isfinite(angle):
            raise ValueError("projection angle must be finite")
        angle %= 360.0
        if angle != self._projection_angle_deg:
            self._projection_angle_deg = angle
            self._rebuild_projection()
            self.update()

    def set_axial_range_mm(self, lower: float, upper: float) -> None:
        """Fit the Ray Diagram's visible Z interval, supplied in millimetres.

        Only already calculated lines intersecting this interval contribute to
        the transverse fit.  Clipping is a display operation and never traces
        additional field lines or changes the cached physical scene.
        """
        interval = (float(lower) * 1e-3, float(upper) * 1e-3)
        if not all(math.isfinite(value) for value in interval) or interval[1] <= interval[0]:
            raise ValueError("axial range must be finite and increasing")
        if interval != self._axial_range_m:
            self._axial_range_m = interval
            self._rebuild_projection()
            self.fit_view()

    def _project(self, positions_m: np.ndarray) -> np.ndarray:
        display = (np.asarray(positions_m) - self._centre_m) * np.array([self._transverse_gain, self._transverse_gain, 1.0])
        # This small projection must not start a BLAS worker pool in the GUI.
        return np.einsum("...j,ij->...i", display, self._basis[:2], optimize=False) / self._display_scale_m

    def _rebuild_projection(self) -> None:
        angle = math.radians(self._projection_angle_deg)
        cosine, sine = math.cos(angle), math.sin(angle)
        horizontal = np.array([0.0, 0.0, 1.0])
        up = np.array([cosine, sine, 0.0])
        depth = np.array([-sine, cosine, 0.0])
        self._basis = np.stack((horizontal, -up, depth))
        display_bounds = self._bounds_m.copy()
        if self._axial_range_m is not None:
            display_bounds[:, 2] = self._axial_range_m
        self._centre_m = np.mean(display_bounds, axis=0)
        extents = (display_bounds[1] - display_bounds[0]) * np.array([self._transverse_gain, self._transverse_gain, 1.0])
        # A valid zero-field scene may have collapsed bounds. Keep projection
        # finite if an electron overlay is subsequently added to that scene.
        self._display_scale_m = max(float(np.linalg.norm(extents)), 1e-12)
        if self._axial_range_m is None:
            visible, selected = self._segments_m, np.ones(len(self._segments_m), dtype=bool)
        else:
            visible, selected = _clip_axial_segments(self._segments_m, *self._axial_range_m)
        projected = self._project(visible)
        fractions = (field_strength_fraction(self._strengths_t[selected], self._reference_t)
                     if self._reference_t else np.zeros(len(projected)))
        bins = np.minimum((fractions * _COLOUR_BINS).astype(np.int64), _COLOUR_BINS - 1)
        self._colour_paths = tuple(
            (_strength_colour(index / (_COLOUR_BINS - 1)), _paired_path(projected[bins == index]))
            for index in range(_COLOUR_BINS) if np.any(bins == index)
        )
        corners = np.array([[display_bounds[x, 0], display_bounds[y, 1], display_bounds[z, 2]]
                            for x in (0, 1) for y in (0, 1) for z in (0, 1)])
        edges = [(i, i ^ step) for i in range(8) for step in (1, 2, 4) if i < (i ^ step)]
        projected_corners = self._project(corners)
        self._projected_bounds = projected_corners
        self._field_projected_segments = projected
        self._bounds_path = _paired_path(projected_corners[np.asarray(edges)])
        column = np.array([[0.0, 0.0, display_bounds[0, 2]], [0.0, 0.0, display_bounds[1, 2]]])
        self._axis_path = _paired_path(self._project(column[None, :, :]))
        # Direction is a display aid. Bound its overdraw independently of the
        # field-line density, preserving the full supplied physical geometry.
        if self._axial_range_m is None:
            eligible = np.arange(len(self._direction_segments_m))
        else:
            _, direction_selected = _clip_axial_segments(self._direction_segments_m, *self._axial_range_m)
            eligible = np.flatnonzero(direction_selected)
        self._direction_draw_indices = eligible[np.linspace(
            0, len(eligible) - 1, min(len(eligible), _MAX_DIRECTION_ARROWS), dtype=np.int64,
        )]
        arrows = self._project(self._direction_segments_m[self._direction_draw_indices])
        self._field_projected_arrows = arrows
        self._field_arrow_screen_key = None
        self._projected_labels = tuple((text, self._project(position)) for text, position in self._labels)
        self._projection_revision += 1
        self._rebuild_electron_projection()
        self._update_visible_fit()
        self._label_layout_key = None

    def _rebuild_electron_projection(self) -> None:
        projections = {}
        for electron in self._electron_paths:
            positions = electron.positions_m
            previous = self._electron_projections.get(electron.key)
            if (previous is not None and previous.revision == self._projection_revision
                    and (previous.positions_m is positions or np.array_equal(previous.positions_m, positions))):
                projections[electron.key] = previous
                continue
            segments = np.stack((positions[:-1], positions[1:]), axis=1)
            if self._axial_range_m is not None:
                segments, _ = _clip_axial_segments(segments, *self._axial_range_m)
            projected = self._project(segments)
            path = _paired_path(projected)
            markers = []
            if len(positions):
                for name, point in (("start", positions[0]), ("end", positions[-1])):
                    if self._axial_range_m is None or self._axial_range_m[0] <= point[2] <= self._axial_range_m[1]:
                        markers.append((name, self._project(point)))
            count = len(projected)
            indices = np.linspace(0, count - 1, min(count, _MAX_ELECTRON_ARROWS), dtype=np.int64)
            projections[electron.key] = _ElectronProjection(
                positions, self._projection_revision, projected, path, tuple(markers), projected[indices],
            )
        self._electron_projections = projections

    def _update_visible_fit(self) -> None:
        """Refit cached display coordinates without rebuilding field batches."""
        self._projected_extent = np.maximum(np.ptp(self._projected_bounds, axis=0), 1e-9)
        self._projected_centre = np.zeros(2)
        chunks = []
        for projection in self._electron_projections.values():
            chunks.append(projection.segments.reshape(-1, 2))
            if projection.markers:
                chunks.append(np.array([point for _, point in projection.markers]))
        electron = np.concatenate(chunks) if chunks else np.empty((0, 2))
        field = (self._field_projected_segments.reshape(-1, 2) if self._field_lines_visible
                 else np.empty((0, 2)))
        # Fit all visible physical geometry on one U scale. Prioritising a
        # nanometre electron path would otherwise clip almost all field lines.
        points = np.concatenate((field, electron))
        if len(points):
            low, high = np.min(points, axis=0), np.max(points, axis=0)
            if high[1] - low[1] > 1e-12:
                self._projected_extent[1] = high[1] - low[1]
                self._projected_centre[1] = (high[1] + low[1]) * 0.5
            elif self._electron_mode or not self._field_lines_visible:
                # A straight on-axis ray has no transverse span. Centre it in a
                # finite display window; this does not change the physical path.
                self._projected_extent[1] = 1e-6 * self._transverse_gain / self._display_scale_m
                self._projected_centre[1] = (high[1] + low[1]) * 0.5
        if self._axial_range_m is None and len(electron):
            axial = (electron[:, 0] if not self._field_lines_visible else
                     np.concatenate((self._projected_bounds[:, 0], electron[:, 0])))
            span = float(np.ptp(axial))
            if span > 1e-12:
                self._projected_extent[0] = span
                self._projected_centre[0] = float((np.min(axial) + np.max(axial)) * 0.5)
        self._label_layout_key = None

    def _plot_rect(self) -> QRectF:
        rect = QRectF(self.rect()).adjusted(94.0, 8.0, -12.0, -58.0)
        if self._horizontal_plot_edges is not None:
            left, right = self._horizontal_plot_edges
            # During a pending layout event retain a usable plot until the
            # owner supplies the newly measured Ray Diagram pixel edges.
            if 0.0 <= left < right <= self.width() and right-left >= 40.0:
                rect.setLeft(left)
                rect.setRight(right)
        return rect

    def set_horizontal_plot_edges(self, edges: tuple[float, float] | None) -> None:
        """Set local logical-pixel Z-axis edges without changing physical ranges.

        The owner measures these from the linked Ray Diagram ViewBox. None
        restores standalone margins; this never projects or traces geometry.
        """
        checked = None if edges is None else tuple(float(value) for value in edges)
        if checked is not None and (len(checked) != 2 or not all(map(math.isfinite, checked))
                                    or checked[1] <= checked[0]):
            raise ValueError("Horizontal plot edges must be two finite increasing pixel coordinates")
        if checked != self._horizontal_plot_edges:
            self._horizontal_plot_edges = checked
            self._label_layout_key = None
            self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.plot_geometry_changed.emit()

    def moveEvent(self, event) -> None:
        super().moveEvent(event)
        self.plot_geometry_changed.emit()

    def _screen_transform(self) -> QTransform:
        plot = self._plot_rect()
        if self._navigation_range_mm is not None:
            axial, transverse = self._navigation_range_mm
            scale_x = plot.width() * self._display_scale_m * 1e3 / (axial[1] - axial[0])
            scale_y = plot.height() * self._display_scale_m * 1e3 / self._transverse_gain / (transverse[1] - transverse[0])
            centre_u = -float(self._basis[1] @ self._centre_m)
            x = ((axial[0] + axial[1]) * 0.5e-3 - self._centre_m[2]) / self._display_scale_m
            y = -((transverse[0] + transverse[1]) * 0.5e-3 - centre_u) * self._transverse_gain / self._display_scale_m
            return QTransform(scale_x, 0.0, 0.0, scale_y,
                              plot.center().x() - x * scale_x, plot.center().y() - y * scale_y)
        scale_x = max(np.finfo(float).tiny, plot.width() / self._projected_extent[0]) * self._zoom
        scale_y = max(np.finfo(float).tiny, plot.height() / self._projected_extent[1] * 0.94) * self._zoom
        centre = plot.center() + self._pan
        return QTransform(scale_x, 0.0, 0.0, scale_y,
                          centre.x() - self._projected_centre[0] * scale_x,
                          centre.y() - self._projected_centre[1] * scale_y)

    @staticmethod
    def _pen(colour: QColor | str, width: float = 1.0, style: Qt.PenStyle = Qt.PenStyle.SolidLine) -> QPen:
        pen = QPen(QColor(colour), width, style)
        pen.setCosmetic(True)
        return pen

    def _field_arrows_on_screen(self, transform: QTransform) -> QPainterPath:
        """Bounded screen-size direction heads, never enlarged in physical U.

        Anisotropic Z/U scaling must not stretch arrowheads into transverse
        spikes. Repainting a fixed viewport reuses this small display cache.
        """
        key = (transform.m11(), transform.m22(), transform.dx(), transform.dy())
        if key != self._field_arrow_screen_key:
            path = QPainterPath()
            for first, last in self._field_projected_arrows:
                start, end = (transform.map(QPointF(*point)) for point in (first, last))
                dx, dy = end.x()-start.x(), end.y()-start.y()
                length = math.hypot(dx, dy)
                if length < 1e-9:
                    continue
                direction = QPointF(dx/length, dy/length)
                wing = QPointF(-direction.y(), direction.x())*2.
                tip = (start+end)*.5
                back = tip-direction*5.
                path.moveTo(back+wing)
                path.lineTo(tip)
                path.lineTo(back-wing)
            self._field_arrow_screen_path = path
            self._field_arrow_screen_key = key
        return self._field_arrow_screen_path

    def _field_layer(self, transform):
        """Reuse rasterized, unchanged field lines while electrons move.

        Physical field geometry stays untouched. A camera, layout, display-mode
        or device-pixel-ratio change rebuilds this single bounded display layer.
        """
        ratio = self.devicePixelRatioF()
        rect = self._plot_rect()
        key = (self._projection_revision, self.width(), self.height(), ratio,
               transform.m11(), transform.m22(), transform.dx(), transform.dy(),
               rect.x(), rect.y(), rect.width(), rect.height(), self._electron_mode)
        if key != self._field_layer_key:
            layer = QPixmap(max(1, math.ceil(self.width()*ratio)), max(1, math.ceil(self.height()*ratio)))
            layer.setDevicePixelRatio(ratio)
            layer.fill(Qt.GlobalColor.transparent)
            painter = QPainter(layer)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setClipRect(rect)
            painter.setTransform(transform)
            painter.setOpacity(_ELECTRON_FIELD_OPACITY if self._electron_mode else 1.)
            for colour, path in self._colour_paths:
                painter.setPen(self._pen(colour, 1.15))
                painter.drawPath(path)
            arrow_colour = QColor("#b8dbe0")
            arrow_colour.setAlpha(150)
            painter.setPen(self._pen(arrow_colour, 1.))
            painter.resetTransform()
            painter.drawPath(self._field_arrows_on_screen(transform))
            painter.end()
            self._field_layer_pixmap, self._field_layer_key = layer, key
        return self._field_layer_pixmap

    def paintEvent(self, event: object) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#080e1b"))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        field_visible = self._field_lines_visible and self.segment_count
        if not field_visible and not self.electron_point_count:
            painter.setPen(QColor("#9baec9"))
            message = ("Set the virtual electron parameters to display a trajectory" if self._electron_mode
                       else "No magnetic field lines in this volume")
            painter.drawText(self._plot_rect(), Qt.AlignmentFlag.AlignCenter, message)
        else:
            painter.save()
            painter.setClipRect(self._plot_rect())
            transform = self._screen_transform()
            painter.setTransform(transform)
            painter.setPen(self._pen("#27374e", 0.8))
            painter.drawPath(self._bounds_path)
            painter.setPen(self._pen("#728198", 1.0, Qt.PenStyle.DashLine))
            painter.drawPath(self._axis_path)
            if field_visible:
                painter.resetTransform()
                painter.drawPixmap(0, 0, self._field_layer(transform))
                painter.setTransform(transform)
            if self.electron_point_count:
                # Selected paths are painted last so intersections remain clear.
                for electron in sorted(self._electron_paths, key=lambda path: path.selected):
                    path = self._electron_projections[electron.key].path
                    width = _SELECTED_ELECTRON_LINE_WIDTH if electron.selected else _ELECTRON_LINE_WIDTH
                    previous = electron.state == "previous"
                    painter.setOpacity(.45 if previous else 1.)
                    painter.setPen(self._pen(electron.colour, width,
                                            Qt.PenStyle.DashLine if previous else Qt.PenStyle.SolidLine))
                    painter.drawPath(path)
                painter.setOpacity(1.)
            painter.resetTransform()
            self._paint_labels(painter, transform)
            self._paint_electron_markers(painter, transform)
            painter.restore()
        self._paint_axes(painter)
        self._paint_footer(painter)

    def axis_rect(self, name: str) -> QRectF:
        plot = self._plot_rect()
        if name == "left":
            return QRectF(0.0, plot.top(), plot.left(), plot.height())
        if name == "bottom":
            return QRectF(plot.left(), plot.bottom(), plot.width(), 38.0)
        raise ValueError("axis must be left or bottom")

    def _axis_ticks(self, name):
        index = 0 if name == "bottom" else 1
        lower, upper = self.view_range_mm()[index]
        length = self._plot_rect().width() if index == 0 else self._plot_rect().height()
        scale, prefix = siScale(max(abs(lower), abs(upper)) * 1e-3)
        levels = self._tick_axes[name].tickValues(lower, upper, length)
        spacing, values = levels[0] if levels else (upper - lower, [])
        labels = self._tick_axes[name].tickStrings(values, scale * 1e-3, spacing)
        title = f"{'Axial Z' if index == 0 else 'Projected U'} ({prefix}m)"
        return tuple(zip(values, labels)), title

    def _paint_axes(self, painter):
        plot = self._plot_rect()
        painter.setPen(self._pen("#879bb8"))
        painter.drawLine(plot.bottomLeft(), plot.bottomRight())
        painter.drawLine(plot.bottomLeft(), plot.topLeft())
        axial, transverse = self.view_range_mm()
        for name in ("bottom", "left"):
            ticks, title = self._axis_ticks(name)
            for value, label in ticks:
                point = self.physical_mm_to_screen(value, transverse[0]) if name == "bottom" else self.physical_mm_to_screen(axial[0], value)
                if name == "bottom":
                    painter.drawLine(point, point + QPointF(0.0, 4.0))
                    width = painter.fontMetrics().horizontalAdvance(label) + 4.0
                    left = max(2.0, min(point.x() - width / 2, self.width() - width - 2.0))
                    painter.drawText(QRectF(left, point.y()+5, width, 16), Qt.AlignmentFlag.AlignHCenter, label)
                else:
                    painter.drawLine(point, point - QPointF(4.0, 0.0))
                    painter.drawText(QRectF(26, point.y()-8, plot.left()-33, 16), Qt.AlignmentFlag.AlignRight, label)
            painter.setPen(QColor("#dbe6f4"))
            if name == "bottom":
                painter.drawText(QRectF(plot.left(), plot.bottom()+22, plot.width(), 18), Qt.AlignmentFlag.AlignHCenter, title)
            else:
                painter.save()
                painter.translate(15, plot.center().y())
                painter.rotate(-90)
                painter.drawText(QRectF(-plot.height()/2, -9, plot.height(), 18), Qt.AlignmentFlag.AlignHCenter, title)
                painter.restore()
            painter.setPen(self._pen("#879bb8"))

    def _paint_electron_markers(self, painter: QPainter, transform: QTransform) -> None:
        """Constant screen-size arrows retain the supplied chronological direction."""
        for electron in sorted(self._electron_paths, key=lambda path: path.selected):
            if electron.state == "previous":
                continue
            projection = self._electron_projections[electron.key]
            arrows = QPainterPath()
            for first, last in projection.arrows:
                start, end = (transform.map(QPointF(*point)) for point in (first, last))
                dx, dy = end.x() - start.x(), end.y() - start.y()
                length = math.hypot(dx, dy)
                # Sub-pixel integrator steps still supply a valid tangent.
                if length < 1e-9:
                    continue
                ux, uy = dx / length, dy / length
                tip = (start + end) * 0.5
                base = tip - QPointF(ux, uy) * 5.0
                wing = QPointF(-uy, ux) * 2.0
                arrows.moveTo(base + wing)
                arrows.lineTo(tip)
                arrows.lineTo(base - wing)
            painter.setPen(self._pen(QColor(electron.colour).lighter(120), 1.1 if electron.selected else .9))
            painter.drawPath(arrows)
            painter.setPen(self._pen("#f4f7ff" if electron.selected else electron.colour, .8))
            for name, point in projection.markers:
                centre = transform.map(QPointF(*point))
                painter.setBrush(QColor(electron.colour))
                if name == "start" or electron.state == "in_progress":
                    painter.drawEllipse(centre, 3.0, 3.0)
                else:
                    painter.drawRect(QRectF(centre.x()-2.5, centre.y()-2.5, 5.0, 5.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _layout_labels(self, transform: QTransform, metrics: object) -> tuple[tuple[str, QPointF, QRectF, QPointF], ...]:
        """Place sparse context labels without overlap; omit crowded labels.

        Only screen text is repositioned. A short leader retains its association
        with the original physical location. No label changes a field line.
        """
        available = self._plot_rect().adjusted(3.0, 3.0, -3.0, -3.0)
        occupied: list[QRectF] = []
        placements = []
        height = float(metrics.height())
        ascent = float(metrics.ascent())
        for text, position in self._projected_labels:
            anchor = transform.map(QPointF(float(position[0]), float(position[1])))
            if not available.contains(anchor):
                continue
            width = float(metrics.horizontalAdvance(text)) + 6.0
            if width > available.width():
                continue
            for offset in (-height - 6.0, -2.0 * height - 10.0, -3.0 * height - 14.0,
                           8.0, height + 12.0, 2.0 * height + 16.0):
                left = min(max(anchor.x() + 5.0, available.left()), available.right() - width)
                rectangle = QRectF(left, anchor.y() + offset, width, height + 3.0)
                if not available.contains(rectangle) or any(rectangle.intersects(other) for other in occupied):
                    continue
                baseline = QPointF(rectangle.left() + 3.0, rectangle.top() + ascent + 1.0)
                placements.append((text, baseline, rectangle, anchor))
                occupied.append(rectangle.adjusted(-3.0, -3.0, 3.0, 3.0))
                break
        return tuple(placements)

    def _paint_labels(self, painter: QPainter, transform: QTransform) -> None:
        key = (self._projection_revision, transform.m11(), transform.m22(), transform.dx(), transform.dy(),
               self.width(), self.height(), painter.font().toString())
        if key != self._label_layout_key:
            self._label_placements = self._layout_labels(transform, painter.fontMetrics())
            self._label_layout_key = key
        leaders = QPainterPath()
        for _, _, rectangle, anchor in self._label_placements:
            endpoint = QPointF(min(max(anchor.x(), rectangle.left()), rectangle.right()),
                               min(max(anchor.y(), rectangle.top()), rectangle.bottom()))
            leaders.moveTo(anchor)
            leaders.lineTo(endpoint)
        painter.setPen(self._pen("#53637a", 0.7))
        painter.drawPath(leaders)
        background = QColor("#080e1b")
        background.setAlpha(215)
        for text, baseline, rectangle, _ in self._label_placements:
            painter.fillRect(rectangle, background)
            painter.setPen(QColor("#c2d0e2"))
            painter.drawText(baseline, text)

    def _paint_footer(self, painter: QPainter) -> None:
        left, top = 12.0, self.height() - 10.0
        painter.setPen(QColor("#bdcbe0"))
        transform = self._screen_transform()
        effective_gain = self._transverse_gain * abs(transform.m22() / transform.m11())
        caption = f"U({self._projection_angle_deg:g}°)–Z · Ray Diagram projection · transverse {effective_gain:.3g}×"
        if self.width() < 750:
            caption = f"U({self._projection_angle_deg:g}°)–Z · transverse {effective_gain:.3g}×"
        caption += " · angles not to scale"
        painter.drawText(QPointF(left, top), caption)
        left += painter.fontMetrics().horizontalAdvance(caption) + 18.0
        if self._electron_mode or self.electron_point_count:
            self._paint_electron_legend(painter, left, top)
            return
        width = min(90.0, max(20.0, self.width() * 0.1))
        for index in range(_COLOUR_BINS):
            painter.fillRect(QRectF(left + index * width / _COLOUR_BINS, top - 7, width / _COLOUR_BINS + 0.5, 7),
                             _strength_colour(index / (_COLOUR_BINS - 1)))
        painter.setPen(QColor("#9baec9"))
        painter.drawText(QPointF(left + width + 8.0, top), f"log |B|: 0–{self._reference_t:.3g} T")

    def _paint_electron_legend(self, painter: QPainter, left: float, top: float) -> None:
        """One bounded row; the selection list carries the complete legend."""
        metrics = painter.fontMetrics()
        paths = sorted(self._electron_paths, key=lambda path: not path.selected)
        for index, electron in enumerate(paths):
            remaining = self.width()-left-12.0
            reserve = 42.0 if index < len(paths)-1 else 0.0
            if remaining < 70.0+reserve:
                if remaining > 25.0:
                    painter.setPen(QColor("#bdcbe0"))
                    painter.drawText(QPointF(left, top), f"+{len(paths)-index}")
                break
            label_width = max(0, int(min(125.0, remaining-reserve-28.0)))
            label = metrics.elidedText(electron.label, Qt.TextElideMode.ElideRight, label_width)
            painter.setPen(self._pen(electron.colour, _SELECTED_ELECTRON_LINE_WIDTH if electron.selected else _ELECTRON_LINE_WIDTH))
            painter.drawLine(QPointF(left, top-4.0), QPointF(left+16.0, top-4.0))
            painter.setPen(QColor("#ffffff" if electron.selected else "#bdcbe0"))
            painter.drawText(QPointF(left+23.0, top), label)
            left += 34.0+metrics.horizontalAdvance(label)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._drag_position = event.position()
            self._drag_button = event.button()
            self._drag_axis = (0 if self.axis_rect("bottom").contains(event.position()) else
                               1 if self.axis_rect("left").contains(event.position()) else None)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_position is None:
            super().mouseMoveEvent(event)
            return
        delta = event.position() - self._drag_position
        self._drag_position = event.position()
        axial, transverse = self.view_range_mm()
        plot = self._plot_rect()
        dz = -delta.x() / plot.width() * (axial[1] - axial[0]) if self._drag_axis != 1 else 0.0
        du = delta.y() / plot.height() * (transverse[1] - transverse[0]) if self._drag_axis != 0 else 0.0
        self._pan += delta
        self.set_view_range_mm(tuple(value + dz for value in axial), tuple(value + du for value in transverse), emit=True)
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == self._drag_button:
            self._drag_position = None
            self._drag_button = Qt.MouseButton.NoButton
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.fit_view()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        position = event.position()
        axis = (0 if self.axis_rect("bottom").contains(position) else
                1 if self.axis_rect("left").contains(position) else None)
        if axis is None and not self._plot_rect().contains(position):
            event.ignore()
            return
        delta = event.angleDelta().y()
        if not delta:
            delta = event.pixelDelta().y() * 4.0
        if not delta:
            event.ignore()
            return
        # Same scale factor and cursor anchoring as Ray Diagram's pg.ViewBox.
        scale = 1.02 ** (float(delta) * self._wheel_scale_factor)
        anchor = self.screen_to_physical_mm(position)
        ranges = [list(interval) for interval in self.view_range_mm()]
        for index, centre in enumerate((anchor.x(), anchor.y())):
            if axis is None or axis == index:
                ranges[index] = [centre + (value - centre) * scale for value in ranges[index]]
        self._zoom /= scale
        self.set_view_range_mm(*ranges, emit=True)
        event.accept()


__all__ = ["MagneticFieldCanvas"]
