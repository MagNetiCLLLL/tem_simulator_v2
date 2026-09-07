"""A software-rendered, freely rotatable CAD mesh viewport.

Coordinates and distances are in millimetres: x/y are radial and z is axial.
The same triangle depth buffer drives painting and picking, including offscreen
Qt platforms without an OpenGL context. No instrument state is owned here.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import math

import numpy as np
from numba import njit
from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget


@dataclass(frozen=True)
class _Mesh:
    vertices: np.ndarray
    faces: np.ndarray
    key: str
    region: str
    color: np.ndarray
    face_groups: tuple
    surfaces: dict
    edges: tuple


def _value(record, name, default=None):
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _color(value) -> np.ndarray:
    if isinstance(value, (str, QColor)):
        color = QColor(value)
        if not color.isValid():
            raise ValueError("Mesh color must be a valid color")
        return np.array(color.getRgb()[:3], dtype=float)
    values = np.asarray(value, dtype=float)
    if values.ndim != 1 or values.size not in (3, 4) or not np.isfinite(values).all():
        raise ValueError("Mesh color must contain three or four finite channels")
    if np.any(values < 0) or np.any(values > 255):
        raise ValueError("Mesh color channels must be between 0 and 255")
    # Surfaces are deliberately opaque so depth and selection remain unambiguous.
    return values[:3] * (255.0 if values.max() <= 1.0 else 1.0)


def _mesh(record) -> _Mesh:
    vertices = np.array(_value(record, "vertices"), dtype=float, copy=True)
    raw_faces = np.asarray(_value(record, "faces"))
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("Mesh vertices must be a finite N by 3 array")
    if raw_faces.ndim != 2 or raw_faces.shape[1] != 3 or raw_faces.dtype.kind not in "iu":
        raise ValueError("Mesh faces must be an integer M by 3 array")
    if raw_faces.size and (raw_faces.min() < 0 or raw_faces.max() >= len(vertices)):
        raise ValueError("Mesh face index is outside the vertices array")
    faces = np.array(raw_faces, dtype=np.int64, copy=True)
    key = str(_value(record, "key", ""))
    if not key:
        raise ValueError("Every mesh needs a part key")
    raw_groups = _value(record, "face_groups", ())
    groups = () if raw_groups is None else tuple(str(value) for value in raw_groups)
    if groups and len(groups) != len(faces):
        raise ValueError("Mesh face_groups must have one semantic ID per triangle")
    surfaces = {}
    for identity, value in _value(record, "surfaces", {}).items():
        metadata = dict(value)
        metadata["parameter_paths"] = tuple(tuple(path) for path in metadata.get("parameter_paths", ()))
        if metadata.get("normal") is not None:
            normal = np.asarray(metadata["normal"], dtype=float)
            if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) == 0:
                raise ValueError("Surface normal must be a finite nonzero three-vector")
            metadata["normal"] = tuple(normal / np.linalg.norm(normal))
        surfaces[str(identity)] = metadata
    if groups and not set(groups).issubset(surfaces):
        raise ValueError("Mesh face_groups must refer to declared semantic surfaces")
    edges = []
    for raw_edge in _value(record, "edges", ()):
        edge = dict(raw_edge)
        points = np.array(edge["vertices"], dtype=float, copy=True)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
            raise ValueError("Semantic edge vertices must be a finite polyline with at least two points")
        points.setflags(write=False)
        edge.update(id=str(edge["id"]), vertices=points,
                    parameter_paths=tuple(tuple(path) for path in edge.get("parameter_paths", ())))
        edges.append(edge)
    if len({edge["id"] for edge in edges}) != len(edges):
        raise ValueError("Semantic edge IDs must be unique within a mesh")
    vertices.setflags(write=False)
    faces.setflags(write=False)
    return _Mesh(vertices, faces, key, str(_value(record, "region", "body")),
                 _color(_value(record, "color", _value(record, "materialcolor", "#8c9dad"))),
                 groups, surfaces, tuple(edges))


def _view_rotation(direction) -> np.ndarray:
    forward = np.asarray(direction, dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross([0.0, 0.0, 1.0], forward)
    right /= np.linalg.norm(right)
    return np.array([right, np.cross(forward, right), forward])


def _arc_rotation(start, end):
    axis = np.cross(start, end)
    sine = np.linalg.norm(axis)
    cosine = float(np.clip(np.dot(start, end), -1, 1))
    if sine < 1e-12:
        if cosine > 0:
            return np.eye(3)
        axis = np.cross(start, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-9:
            axis = np.cross(start, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    axis /= sine
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + sine * cross + (1 - cosine) * (cross @ cross)


def _section_triangles(triangles, *, with_sources=False, keep_positive_y=True):
    """Clip at world y=0 without reflecting or changing source coordinates."""
    result = []
    sources = []
    sign = 1.0 if keep_positive_y else -1.0
    for source, triangle in enumerate(triangles):
        if np.all(sign * triangle[:, 1] >= 0):
            result.append(triangle)
            sources.append(source)
            continue
        if np.all(sign * triangle[:, 1] < 0):
            continue
        polygon = []
        for previous, current in zip(np.roll(triangle, 1, axis=0), triangle):
            previous_inside, current_inside = sign * previous[1] >= 0, sign * current[1] >= 0
            if previous_inside != current_inside:
                fraction = previous[1] / (previous[1] - current[1])
                polygon.append(previous + fraction * (current - previous))
            if current_inside:
                polygon.append(current)
        for index in range(1, len(polygon) - 1):
            result.append([polygon[0], polygon[index], polygon[index + 1]])
            sources.append(source)
    result = np.asarray(result, dtype=float).reshape(-1, 3, 3)
    return (result, np.asarray(sources, dtype=np.int64)) if with_sources else result


@njit(cache=True, nogil=True)
def _rasterize(triangles, colors, mesh_id, face_offset, pixels, depth, ids, face_ids, depth_epsilon):
    """Fill visible pixels without a Python call or array allocation per face."""
    height, width = depth.shape
    for index in range(len(triangles)):
        a, b, c = triangles[index]
        xmin = max(0, math.ceil(min(a[0], b[0], c[0]) - 0.5))
        xmax = min(width - 1, math.floor(max(a[0], b[0], c[0]) - 0.5))
        ymin = max(0, math.ceil(min(a[1], b[1], c[1]) - 0.5))
        ymax = min(height - 1, math.floor(max(a[1], b[1], c[1]) - 0.5))
        if xmin > xmax or ymin > ymax:
            continue
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denominator) < 1e-12:
            continue
        dax, dbx = (b[1] - c[1]) / denominator, (c[1] - a[1]) / denominator
        for y in range(ymin, ymax + 1):
            wa = ((b[1] - c[1]) * (xmin + 0.5 - c[0]) + (c[0] - b[0]) * (y + 0.5 - c[1])) / denominator
            wb = ((c[1] - a[1]) * (xmin + 0.5 - c[0]) + (a[0] - c[0]) * (y + 0.5 - c[1])) / denominator
            for x in range(xmin, xmax + 1):
                wc = 1.0 - wa - wb
                if wa >= -1e-9 and wb >= -1e-9 and wc >= -1e-9:
                    value = wa * a[2] + wb * b[2] + wc * c[2]
                    # Shared/coincident material boundaries keep a deterministic
                    # first surface instead of flickering with roundoff per pixel.
                    if value > depth[y, x] + depth_epsilon:
                        depth[y, x] = value
                        ids[y, x] = mesh_id
                        face_ids[y, x] = face_offset + index
                        pixels[y, x, 0] = colors[index, 0]
                        pixels[y, x, 1] = colors[index, 1]
                        pixels[y, x, 2] = colors[index, 2]
                wa += dax
                wb += dbx


@njit(cache=True, nogil=True)
def _visible_edge_point(x, y, z, mesh_id, ids, face_ids, triangles, epsilon):
    """Compare exact edge depth with nearby visible triangle planes.

    Pixel-centre depths alone need a slope-dependent tolerance that can expose
    back edges. Evaluating the winning triangles at the actual edge point avoids
    that tolerance, while the neighbourhood includes rasterized silhouettes.
    """
    height, width = ids.shape
    front, own = -np.inf, -np.inf
    for py in range(max(0, int(math.floor(y)) - 1), min(height, int(math.floor(y)) + 2)):
        for px in range(max(0, int(math.floor(x)) - 1), min(width, int(math.floor(x)) + 2)):
            face = face_ids[py, px]
            if face < 0:
                continue
            a, b, c = triangles[face]
            denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if abs(denominator) < 1e-12:
                continue
            wa = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / denominator
            wb = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / denominator
            wc = 1.0 - wa - wb
            if min(wa, wb, wc) < -1e-8:
                continue
            value = wa * a[2] + wb * b[2] + wc * c[2]
            front = max(front, value)
            if ids[py, px] == mesh_id:
                own = max(own, value)
    return np.isfinite(own) and abs(z - own) <= epsilon and z >= front - epsilon


@njit(cache=True, nogil=True)
def _draw_semantic_edge(segments, mesh_id, pixels, ids, face_ids, triangles, epsilon, color, radius):
    height, width = ids.shape
    for segment in segments:
        a, b = segment
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        if length < 1e-9:
            continue
        # Clip the screen segment before sampling so extreme zoom cannot cause
        # millions of offscreen samples on an otherwise small viewport.
        low, high = 0.0, 1.0
        for axis, limit in ((0, width), (1, height)):
            delta = b[axis] - a[axis]
            if abs(delta) < 1e-12:
                if a[axis] < -radius or a[axis] > limit + radius:
                    high = -1.0
            else:
                first, last = (-radius - a[axis]) / delta, (limit + radius - a[axis]) / delta
                low, high = max(low, min(first, last)), min(high, max(first, last))
        if low > high:
            continue
        count = max(1, int(math.ceil(length * (high - low) * 2)))
        for step in range(count + 1):
            fraction = low + (high - low) * step / count
            x, y, z = a + fraction * (b - a)
            if not _visible_edge_point(x, y, z, mesh_id, ids, face_ids, triangles, epsilon):
                continue
            cx, cy = int(math.floor(x)), int(math.floor(y))
            for py in range(max(0, cy - radius), min(height, cy + radius + 1)):
                for px in range(max(0, cx - radius), min(width, cx + radius + 1)):
                    pixels[py, px, :3] = color


def _edge_segments(vertices, section, *, keep_positive_y=True):
    segments = np.stack((vertices[:-1], vertices[1:]), axis=1).copy()
    if not section:
        return segments
    sign = 1.0 if keep_positive_y else -1.0
    segments = segments[np.any(sign * segments[..., 1] >= 0, axis=1)]
    for segment in segments:
        if (sign * segment[0, 1] < 0) != (sign * segment[1, 1] < 0):
            fraction = segment[0, 1] / (segment[0, 1] - segment[1, 1])
            crossing = segment[0] + fraction * (segment[1] - segment[0])
            segment[0 if sign * segment[0, 1] < 0 else 1] = crossing
    return segments


def _topology_identity(item):
    return tuple(str(item.get(field, "")) for field in ("key", "region", "kind", "id"))


def _closest_topology_geometry(mesh, kind, group, point):
    """Keep a restored placement point on its edited semantic surface or seam."""
    if kind == "edge":
        vertices = next(edge["vertices"] for edge in mesh.edges if edge["id"] == group)
        starts, directions = vertices[:-1], np.diff(vertices, axis=0)
        fraction = np.clip(np.einsum("ij,ij->i", point - starts, directions)
                           / np.maximum(np.einsum("ij,ij->i", directions, directions), 1e-30), 0, 1)
        candidates = starts + fraction[:, None] * directions
        index = np.argmin(np.sum((candidates - point) ** 2, axis=1))
        return candidates[index], None
    triangles = mesh.vertices[mesh.faces[np.array([value == group for value in mesh.face_groups])]]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    triangles, normals = triangles[lengths > 1e-20], normals[lengths > 1e-20]
    if not len(triangles):
        return point, None
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    projected = point - np.einsum("ij,ij->i", point - triangles[:, 0], normals)[:, None] * normals
    ab, ac, ap = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], projected - triangles[:, 0]
    aa, bb, cross = np.sum(ab * ab, axis=1), np.sum(ac * ac, axis=1), np.sum(ab * ac, axis=1)
    first, second = np.sum(ap * ab, axis=1), np.sum(ap * ac, axis=1)
    determinant = np.maximum(np.sum(np.cross(ab, ac) ** 2, axis=1), 1e-30)
    u, v = (bb * first - cross * second) / determinant, (aa * second - cross * first) / determinant
    inside = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
    directions = np.roll(triangles, -1, axis=1) - triangles
    fraction = np.clip(np.sum((point - triangles) * directions, axis=2)
                       / np.maximum(np.sum(directions ** 2, axis=2), 1e-30), 0, 1)
    edges = triangles + fraction[..., None] * directions
    closest_edges = np.argmin(np.sum((edges - point) ** 2, axis=2), axis=1)
    candidates = edges[np.arange(len(edges)), closest_edges]
    candidates[inside] = projected[inside]
    index = np.argmin(np.sum((candidates - point) ** 2, axis=1))
    return candidates[index], normals[index]


class PartModelView(QWidget):
    """Opaque triangle meshes with trackball rotation and exact visible picking.

    ``set_meshes`` accepts mappings or objects with vertices, faces, key, region
    and color fields, plus optional face_groups/surfaces/edges semantic metadata.
    ``selection_changed`` only fires for user selections;
    ``set_selection`` can therefore follow an external component tree safely.
    Section mode defaults to removing y<0 surfaces; its retained side can be
    chosen explicitly for a downstream-oriented column view. It does not invent
    material faces across a cut or change the source mesh. Topology selection
    reports module-coordinate hits after part selection callbacks have completed.
    """

    selection_changed = Signal(str, str)
    topology_selection_changed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("partModelView")
        self.setMinimumSize(240, 240)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setToolTip("Left drag: rotate freely. Right drag: pan. Wheel: zoom. Click a visible surface to select it.")
        self._view_title = "3D model"
        self._empty_text = "Select a part to view its 3D model"
        self._meshes = ()
        self._triangles = ()
        self._triangle_sources = ()
        self._section = False
        self._section_keep_positive_y = True
        self._selection = ("", None)
        self._selection_keys = frozenset()
        self._selection_mode = "part"
        self._topology_selection = ()
        self._selecting_topology = False
        self._rotation = _view_rotation([1.0, -1.5, 0.85])
        self._center = np.zeros(3)
        self._radius = 1.0
        self._zoom = 1.0
        self._pan = np.zeros(2)
        self._press_position = None
        self._pressed_button = Qt.MouseButton.NoButton
        self._dragged = False
        self._interactive = False
        self._dirty = True
        self._image = QImage()
        self._depth = np.empty((0, 0))
        self._ids = np.empty((0, 0), dtype=np.int32)
        self._face_ids = np.empty((0, 0), dtype=np.int32)
        self._screen_triangles = np.empty((0, 3, 3))
        self._render_sources = ()
        self._settle_timer = QTimer(self)
        self._settle_timer.setSingleShot(True)
        self._settle_timer.setInterval(120)
        self._settle_timer.timeout.connect(self._settle)

    @property
    def selection(self):
        return self._selection

    @property
    def selection_mode(self):
        return self._selection_mode

    @property
    def topology_selection(self):
        return tuple(dict(item) for item in self._topology_selection)

    def set_selection_mode(self, mode):
        if mode not in ("part", "face", "edge"):
            raise ValueError("Selection mode must be part, face or edge")
        if mode != self._selection_mode:
            self._selection_mode = mode
            self.clear_topology_selection()
            self._invalidate()

    def set_topology_selection(self, payload, *, emit=False):
        """Restore existing semantic IDs, discarding topology removed by edits.

        Programmatic restoration is silent by default, like set_selection. Hit
        points and normals stay in module coordinates, never camera coordinates.
        """
        result, identities = [], set()
        for item in payload or ():
            identity = _topology_identity(item)
            if identity in identities:
                continue
            key, region, kind, group = identity
            mesh = next((mesh for mesh in self._meshes if (mesh.key, mesh.region) == (key, region)
                         and ((kind == "face" and group in mesh.surfaces and group in mesh.face_groups)
                              or (kind == "edge" and any(edge["id"] == group for edge in mesh.edges)))), None)
            if mesh is None:
                continue
            metadata = mesh.surfaces[group] if kind == "face" else next(edge for edge in mesh.edges if edge["id"] == group)
            restored = dict(key=key, region=region, kind=kind, id=group,
                            label=metadata.get("label", group), parameter_paths=metadata.get("parameter_paths", ()))
            for field in ("point", "normal"):
                if item.get(field) is not None:
                    vector = np.asarray(item[field], dtype=float)
                    if vector.shape == (3,) and np.isfinite(vector).all():
                        restored[field] = tuple(float(value) for value in vector)
            if "point" in restored:
                point = np.asarray(restored["point"])
                closest, normal = _closest_topology_geometry(mesh, kind, group, point)
                if np.linalg.norm(closest - point) > max(1e-10, np.ptp(mesh.vertices, axis=0).max() * 1e-12):
                    restored["point"] = tuple(float(value) for value in closest)
                if kind == "face":
                    normal = metadata.get("normal") if metadata.get("normal") is not None else normal
                    if normal is not None:
                        restored["normal"] = tuple(float(value) for value in normal)
            result.append(restored)
            identities.add(identity)
        selection = tuple(result)
        if selection != self._topology_selection:
            self._topology_selection = selection
            self._invalidate()
            if emit:
                self.topology_selection_changed.emit(self.topology_selection)

    def clear_topology_selection(self):
        self.set_topology_selection((), emit=True)

    @property
    def section_enabled(self):
        return self._section

    @property
    def camera_rotation(self):
        return self._rotation.copy()

    @property
    def zoom_factor(self):
        return self._zoom

    def set_meshes(self, records, *, preserve_view=False):
        meshes = tuple(_mesh(record) for record in records)
        self._meshes = meshes
        self._rebuild_triangles()
        self.set_topology_selection(self._topology_selection, emit=not self._selecting_topology)
        if not self._selection_keys.intersection(mesh.key for mesh in meshes):
            self.set_selection(None)
        elif self._selection[1] is not None and not any(
            mesh.key in self._selection_keys and mesh.region == self._selection[1] for mesh in meshes
        ):
            self._selection = (self._selection[0], None)
        if not preserve_view:
            self.fit_all()
        else:
            self._invalidate()

    def _rebuild_triangles(self):
        pairs = tuple(_section_triangles(mesh.vertices[mesh.faces], with_sources=True,
                                        keep_positive_y=self._section_keep_positive_y) if self._section
                      else (mesh.vertices[mesh.faces], np.arange(len(mesh.faces))) for mesh in self._meshes)
        self._triangles = tuple(pair[0] for pair in pairs)
        self._triangle_sources = tuple(pair[1] for pair in pairs)

    def set_selection(self, key, region=None, *, related_keys=()):
        selection = (str(key) if key is not None else "", region)
        keys = frozenset((selection[0], *related_keys))
        if selection != self._selection or keys != self._selection_keys:
            self._selection = selection
            self._selection_keys = keys
            if not self._selecting_topology:
                self.clear_topology_selection()
            self._invalidate()

    def set_section_enabled(self, enabled, *, keep_positive_y=True):
        """Retain a chosen world-Y half; the default editor side is y>=0."""
        enabled = bool(enabled)
        keep_positive_y = bool(keep_positive_y)
        if enabled != self._section or keep_positive_y != self._section_keep_positive_y:
            self._section = enabled
            self._section_keep_positive_y = keep_positive_y
            self._rebuild_triangles()
            self._invalidate()

    def fit_all(self):
        self._fit_meshes(self._meshes)

    def fit_selection(self):
        """Centre and fit selected material while retaining camera rotation."""
        meshes = [mesh for mesh in self._meshes
                  if mesh.key in self._selection_keys
                  and self._selection[1] in (None, mesh.region)]
        if not meshes:
            return False
        self._fit_meshes(meshes)
        return True

    def _fit_meshes(self, meshes):
        self._interactive = False
        self._settle_timer.stop()
        points = [mesh.vertices for mesh in meshes if len(mesh.vertices)]
        if points:
            low = np.min([vertices.min(axis=0) for vertices in points], axis=0)
            high = np.max([vertices.max(axis=0) for vertices in points], axis=0)
            self._center = low + (high - low) * 0.5
            self._radius = max(float(np.linalg.norm((high - low) * 0.5)), 1e-9)
        else:
            self._center, self._radius = np.zeros(3), 1.0
        self._zoom = 1.0
        self._pan = np.zeros(2)
        self._invalidate()

    def set_isometric_view(self):
        self._interactive = False
        self._settle_timer.stop()
        self._rotation = _view_rotation([1.0, -1.5, 0.85])
        self._invalidate()

    def set_front_view(self):
        self._interactive = False
        self._settle_timer.stop()
        self._rotation = _view_rotation([0.0, -1.0, 0.0])
        self._invalidate()

    def set_axial_view(self):
        """Look through the z-axis from the +z end, with x right and y up."""
        self._interactive = False
        self._settle_timer.stop()
        self._rotation = np.eye(3)
        self._invalidate()

    def set_column_view(self):
        """Look from +Y with +X right and downstream +Z vertically down.

        This is a proper camera rotation, not a reflected mesh. Its camera
        axes are (+X, -Z, +Y); screen Y increases downward. Existing section
        clipping remains in world coordinates and is not changed by a view.
        """
        self._interactive = False
        self._settle_timer.stop()
        self._rotation = np.array([[1.0, 0.0, 0.0],
                                   [0.0, 0.0, -1.0],
                                   [0.0, 1.0, 0.0]])
        self._invalidate()

    def set_column_isometric_view(self):
        """Oblique orthographic column view with +Z down and visible depth.

        The preset is deliberately only slightly elevated to keep a long
        column legible. Rolling both camera-plane axes by 180 degrees keeps
        a right-handed rotation; projected physical units are never stretched.
        """
        self._interactive = False
        self._settle_timer.stop()
        self._rotation = _view_rotation([0.65, 1.5, -0.4])
        self._rotation[:2] *= -1.0
        self._invalidate()

    def set_view_labels(self, *, title=None, empty_text=None):
        """Customize presentation text without changing geometry or camera."""
        if title is not None:
            self._view_title = str(title)
        if empty_text is not None:
            self._empty_text = str(empty_text)
        self.update()

    def _scale(self):
        return max(1, min(self.width(), self.height())) * 0.42 / self._radius * self._zoom

    def project_points(self, vertices):
        """Return widget x/y and camera depth (greater is nearer) for vertices."""
        camera = (np.asarray(vertices, dtype=float) - self._center) @ self._rotation.T
        screen = camera.copy()
        screen[..., 0] = camera[..., 0] * self._scale() + self.width() / 2 + self._pan[0]
        screen[..., 1] = -camera[..., 1] * self._scale() + self.height() / 2 + self._pan[1]
        return screen

    def _invalidate(self, *, interactive=False):
        self._dirty = True
        if interactive:
            self._interactive = True
            self._settle_timer.start()
        self.update()

    def _settle(self):
        self._interactive = False
        self._invalidate()

    def resizeEvent(self, event):
        self._invalidate()
        super().resizeEvent(event)

    def _render(self):
        if not self._dirty:
            return
        budget = 100_000 if self._interactive else 900_000
        reduction = min(1.0, math.sqrt(budget / max(1, self.width() * self.height())))
        width, height = max(1, round(self.width() * reduction)), max(1, round(self.height() * reduction))
        sx, sy = width / max(1, self.width()), height / max(1, self.height())
        pixels = np.empty((height, width, 4), dtype=np.uint8)
        pixels[:] = [18, 25, 36, 255]
        depth = np.full((height, width), -np.inf, dtype=np.float64)
        ids = np.full((height, width), -1, dtype=np.int32)
        face_ids = np.full((height, width), -1, dtype=np.int32)
        projected_meshes, render_sources = [], []
        for mesh_id, (triangles, sources) in enumerate(zip(self._triangles, self._triangle_sources)):
            projected = self.project_points(triangles)
            projected[..., 0] *= sx
            projected[..., 1] *= sy
            projected_meshes.append(projected)
            render_sources.extend((mesh_id, int(source)) for source in sources)
        screen_triangles = np.concatenate(projected_meshes) if projected_meshes else np.empty((0, 3, 3))
        topology = {_topology_identity(item) for item in self._topology_selection}
        selected_faces = []
        face_offset = 0
        light = np.array([-0.3, 0.45, 0.84])
        light /= np.linalg.norm(light)
        for mesh_id, (mesh, triangles, projected, sources) in enumerate(zip(
                self._meshes, self._triangles, projected_meshes, self._triangle_sources)):
            normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            lengths = np.linalg.norm(normals, axis=1)
            camera_normals = (normals / np.maximum(lengths[:, None], 1e-20)) @ self._rotation.T
            shades = 0.38 + 0.62 * np.abs(camera_normals @ light)
            selected = (self._selection_mode == "part" and mesh.key in self._selection_keys
                        and self._selection[1] in (None, mesh.region))
            bases = np.tile(mesh.color, (len(triangles), 1))
            chosen = np.array([bool(selected or (mesh.face_groups and
                              (mesh.key, mesh.region, "face", mesh.face_groups[source]) in topology))
                               for source in sources], dtype=bool)
            bases[chosen] = 0.35 * mesh.color + 0.65 * np.array([255, 196, 84])
            selected_faces.extend(face_offset + np.flatnonzero(chosen))
            colors = np.clip(shades[:, None] * bases, 0, 255).astype(np.uint8)
            _rasterize(projected, colors, mesh_id, face_offset, pixels, depth, ids, face_ids,
                       max(1e-12, self._radius * 1e-10))
            face_offset += len(triangles)
        if selected_faces:
            selected_pixels = np.isin(face_ids, selected_faces)
            interior = selected_pixels.copy()
            interior[1:] &= selected_pixels[:-1]
            interior[:-1] &= selected_pixels[1:]
            interior[:, 1:] &= selected_pixels[:, :-1]
            interior[:, :-1] &= selected_pixels[:, 1:]
            pixels[selected_pixels & ~interior, :3] = [255, 211, 116]
        for mesh_id, mesh in enumerate(self._meshes):
            for edge in mesh.edges:
                selected = (mesh.key, mesh.region, "edge", edge["id"]) in topology
                if not selected and self._selection_mode != "edge":
                    continue
                segments = self.project_points(_edge_segments(
                    edge["vertices"], self._section,
                    keep_positive_y=self._section_keep_positive_y,
                ))
                segments[..., 0] *= sx
                segments[..., 1] *= sy
                _draw_semantic_edge(segments, mesh_id, pixels, ids, face_ids, screen_triangles,
                                    max(1e-10, self._radius * 1e-9),
                                    np.array([255, 211, 116] if selected else [113, 136, 162], dtype=np.uint8),
                                    1 if selected else 0)
        self._image = QImage(pixels.data, width, height, pixels.strides[0], QImage.Format.Format_RGBA8888).copy()
        self._depth, self._ids = depth, ids
        self._face_ids, self._screen_triangles = face_ids, screen_triangles
        self._render_sources = tuple(render_sources)
        self._dirty = False

    def pick_at(self, position):
        """Return the frontmost visible (part key, region), or None."""
        if isinstance(position, (tuple, list)):
            position = QPointF(*position)
        if not (0 <= position.x() < self.width() and 0 <= position.y() < self.height()):
            return None
        # Finish the coarse interaction frame before an exact surface pick.
        if self._interactive:
            self._interactive = False
            self._dirty = True
        self._render()
        x = min(self._ids.shape[1] - 1, int(position.x() * self._ids.shape[1] / self.width()))
        y = min(self._ids.shape[0] - 1, int(position.y() * self._ids.shape[0] / self.height()))
        index = self._ids[y, x]
        return None if index < 0 else (self._meshes[index].key, self._meshes[index].region)

    def pick_topology_at(self, position, mode=None):
        """Return the visible semantic face or edge under a widget position.

        Face IDs group analytic surfaces rather than tessellation triangles.
        Edge hits are limited to declared, unoccluded seams within six logical
        pixels. Returned points and normals use the mesh's module coordinates.
        """
        mode = self._selection_mode if mode is None else mode
        if mode not in ("part", "face", "edge"):
            raise ValueError("Selection mode must be part, face or edge")
        if mode == "part":
            return None
        if isinstance(position, (tuple, list)):
            position = QPointF(*position)
        # Also finishes a coarse interaction frame before precise selection.
        self.pick_at(position)
        if not (0 <= position.x() < self.width() and 0 <= position.y() < self.height()):
            return None
        sx, sy = self._ids.shape[1] / self.width(), self._ids.shape[0] / self.height()
        if mode == "edge":
            return self._pick_edge(position, sx, sy)
        x, y = int(position.x() * sx), int(position.y() * sy)
        rendered_face = self._face_ids[y, x]
        if rendered_face < 0:
            return None
        mesh_id, source = self._render_sources[rendered_face]
        mesh = self._meshes[mesh_id]
        if not mesh.face_groups:
            return None
        group = mesh.face_groups[source]
        metadata = mesh.surfaces[group]
        camera = np.array([((x + 0.5) / sx - self.width() / 2 - self._pan[0]) / self._scale(),
                           -((y + 0.5) / sy - self.height() / 2 - self._pan[1]) / self._scale(),
                           self._depth[y, x]])
        point = camera @ self._rotation + self._center
        normal = metadata.get("normal")
        if normal is None:
            a, b, c = mesh.vertices[mesh.faces[source]]
            normal = np.cross(b - a, c - a)
            normal /= max(float(np.linalg.norm(normal)), 1e-20)
        return dict(key=mesh.key, region=mesh.region, kind="face", id=group,
                    label=metadata.get("label", group), parameter_paths=metadata.get("parameter_paths", ()),
                    point=tuple(float(value) for value in point), normal=tuple(float(value) for value in normal))

    def _pick_edge(self, position, sx, sy):
        cursor = np.array([position.x(), position.y()])
        candidates = []
        for mesh_id, mesh in enumerate(self._meshes):
            for edge in mesh.edges:
                segments = _edge_segments(
                    edge["vertices"], self._section,
                    keep_positive_y=self._section_keep_positive_y,
                )
                projected = self.project_points(segments)
                if not len(projected):
                    continue
                directions = projected[:, 1, :2] - projected[:, 0, :2]
                squared = np.sum(directions ** 2, axis=1)
                fraction = np.clip(np.sum((cursor - projected[:, 0, :2]) * directions, axis=1)
                                   / np.maximum(squared, 1e-20), 0, 1)
                closest = projected[:, 0] + fraction[:, None] * (projected[:, 1] - projected[:, 0])
                distances = np.linalg.norm(closest[:, :2] - cursor, axis=1)
                for index in np.flatnonzero((distances <= 6.0) & (squared > 1e-12)):
                    x, y, z = closest[index]
                    if _visible_edge_point(x * sx, y * sy, z, mesh_id, self._ids, self._face_ids,
                                           self._screen_triangles, max(1e-10, self._radius * 1e-9)):
                        point = segments[index, 0] + fraction[index] * (segments[index, 1] - segments[index, 0])
                        payload = dict(key=mesh.key, region=mesh.region, kind="edge", id=edge["id"],
                                       label=edge.get("label", edge["id"]),
                                       parameter_paths=edge.get("parameter_paths", ()),
                                       point=tuple(float(value) for value in point))
                        candidates.append((float(distances[index]), -float(z), payload))
        return min(candidates, key=lambda item: item[:2])[2] if candidates else None

    def paintEvent(self, event):
        self._render()
        painter = QPainter(self)
        painter.drawImage(self.rect(), self._image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QColor("#b8c8dc"))
        section_label = " · Section y ≥ 0" if self._section_keep_positive_y else " · Section y ≤ 0"
        title = self._view_title + " · Orthographic" + (section_label if self._section else "")
        painter.drawText(14, 24, title)
        if not self._meshes:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._empty_text)
        if self.width() >= 450:
            painter.drawText(14, self.height() - 16, "Left drag: rotate   Right drag: pan   Wheel: zoom")
        origin = QPointF(self.width() - 58, self.height() - 56)
        for axis, label, color in ((0, "X", "#f18d83"), (1, "Y", "#89ce9b"), (2, "Z", "#8fb8ff")):
            delta = self._rotation[:, axis] * 30
            endpoint = origin + QPointF(delta[0], -delta[1])
            painter.setPen(QPen(QColor(color), 2))
            painter.drawLine(origin, endpoint)
            painter.drawText(endpoint + QPointF(3, -3), label)
        painter.end()

    def _trackball(self, position):
        radius = max(1.0, min(self.width(), self.height()) * 0.5)
        x, y = (position.x() - self.width() / 2) / radius, (self.height() / 2 - position.y()) / radius
        squared = x * x + y * y
        return np.array([x, y, math.sqrt(1 - squared)]) if squared <= 1 else np.array([x, y, 0.0]) / math.sqrt(squared)

    def mousePressEvent(self, event):
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            self.setFocus()
            self._press_position = event.position()
            self._pressed_button = event.button()
            self._drag_start_rotation = self._rotation.copy()
            self._drag_start_pan = self._pan.copy()
            self._dragged = False
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._press_position is None:
            super().mouseMoveEvent(event)
            return
        delta = event.position() - self._press_position
        self._dragged |= abs(delta.x()) + abs(delta.y()) >= 4
        if self._dragged:
            if self._pressed_button == Qt.MouseButton.LeftButton:
                arc = _arc_rotation(self._trackball(self._press_position), self._trackball(event.position()))
                self._rotation = arc @ self._drag_start_rotation
            elif self._pressed_button == Qt.MouseButton.RightButton:
                self._pan = self._drag_start_pan + [delta.x(), delta.y()]
            self._invalidate(interactive=True)
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._press_position is not None and event.button() == self._pressed_button:
            if event.button() == Qt.MouseButton.LeftButton and not self._dragged:
                if self._selection_mode == "part":
                    picked = self.pick_at(event.position())
                    selection = picked if picked is not None else ("", None)
                    changed = selection != self._selection
                    self.set_selection(*selection)
                    if changed:
                        self.selection_changed.emit(selection[0], selection[1] or "")
                else:
                    picked = self.pick_topology_at(event.position())
                    previous = self.topology_selection
                    additive = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
                    if picked is not None or not additive:
                        selection = (picked["key"], picked["region"]) if picked else ("", None)
                        changed = selection != self._selection
                        # The editor may synchronously replace meshes in its
                        # part-selection callback. Resolve topology afterwards.
                        self._selecting_topology = True
                        try:
                            self.set_selection(*selection)
                            if changed:
                                self.selection_changed.emit(selection[0], selection[1] or "")
                        finally:
                            self._selecting_topology = False
                        selected = []
                        if picked is not None:
                            identity = _topology_identity(picked)
                            if additive:
                                selected = [item for item in previous if _topology_identity(item) != identity]
                                if not any(_topology_identity(item) == identity for item in previous):
                                    selected.append(picked)
                            else:
                                selected = [picked]
                        self.set_topology_selection(selected, emit=True)
            self._press_position = None
            self._pressed_button = Qt.MouseButton.NoButton
            self._settle_timer.start()
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        amount = event.angleDelta().y() / 120.0
        if not amount:
            amount = event.pixelDelta().y() / 40.0
        if amount:
            previous = self._zoom
            self._zoom = float(np.clip(previous * math.exp(np.clip(amount * math.log(1.2), -50, 50)), 1e-3, 1e3))
            anchor = np.array([event.position().x() - self.width() / 2, event.position().y() - self.height() / 2])
            self._pan = anchor - (anchor - self._pan) * (self._zoom / previous)
            self._invalidate(interactive=True)
        event.accept()
