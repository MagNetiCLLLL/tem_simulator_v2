"""Longitudinal cuts through captured, already placed assembly surfaces.

The section plane contains the global Z axis. Its transverse coordinate is
``U = X cos(angle) + Y sin(angle)`` and its normal is
``V = -X sin(angle) + Y cos(angle)``. Only intersections at V = 0 are returned;
projecting an entire solid into the view would incorrectly close its holes or
show bodies which the plane never reaches.

This module performs geometry-only CPU work. It never builds a mesh, reads a
configuration file, evaluates a field or retraces particles. The caller can
reuse the same cached AssemblyModel3D while rotating the cutting plane.
"""

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np

from temsim.assembly_model_3d import AssemblyModel3D
from temsim.part_model_3d import TriangleMesh


_FACE_CHUNK_SIZE = 16384
_EDGE_CORNERS = np.array(((0, 1), (1, 2), (2, 0)), dtype=np.intp)


@dataclass(frozen=True)
class MeshSection:
    """Unfilled material boundary segments, endpoints in (Z, U) millimetres.

    ``segments_mm`` has shape (N, 2, 2). Separate segments deliberately retain
    disconnected boundaries and openings; renderers must not join them with
    an arbitrary polygon. ``is_exact`` carries the source mesh's qualification
    and does not imply that tessellated surfaces are analytic or OEM geometry.
    """

    key: str
    region: str
    segments_mm: np.ndarray
    color: tuple[float, ...]
    material_class: str
    is_exact: bool
    description: str


@dataclass(frozen=True)
class AssemblySection:
    angle_deg: float
    meshes: tuple[MeshSection, ...]
    notes: tuple[str, ...] = ()
    omitted_keys: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


def _parameters(angle_deg, tolerance_mm):
    for value, label in ((angle_deg, "angle_deg"), (tolerance_mm, "tolerance_mm")):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{label} must be a finite number")
    if tolerance_mm <= 0:
        raise ValueError("tolerance_mm must be positive")
    return float(angle_deg) % 360.0, float(tolerance_mm)


def _immutable_segments(segments):
    array = np.asarray(segments, dtype=float).reshape(-1, 2, 2)
    # A bytes buffer protects cached output even from setflags(write=True).
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _project(points, cosine, sine):
    return np.stack((points[..., 2], points[..., 0] * cosine + points[..., 1] * sine), axis=-1)


def _unique_segments(segments, tolerance_mm, *, boundary_only=False):
    """Deduplicate unoriented edges; optionally remove shared coplanar edges.

    Coordinate keys also cover triangle soups with duplicated vertex indices.
    In a coplanar triangulated face, only edges belonging to one triangle are
    retained. This removes internal face diagonals instead of drawing them as
    material boundaries.
    """
    if not len(segments):
        return np.empty((0, 2, 2), dtype=float)
    segments = np.asarray(segments, dtype=float).reshape(-1, 2, 2)
    segments = segments[np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1) > tolerance_mm]
    if not len(segments):
        return segments
    # Translation keeps axial positions far along the column out of the key's
    # magnitude. The quantised keys remain floats to avoid integer overflow
    # with an unusually small requested tolerance.
    origin = segments.reshape(-1, 2).min(axis=0)
    with np.errstate(over="ignore", invalid="ignore"):
        keys = np.rint((segments - origin) / tolerance_mm)
    if not np.all(np.isfinite(keys)):
        raise ValueError("Section coordinates exceed the requested tolerance range")
    reverse = (keys[:, 0, 0] > keys[:, 1, 0]) | (
        (keys[:, 0, 0] == keys[:, 1, 0]) & (keys[:, 0, 1] > keys[:, 1, 1])
    )
    keys[reverse] = keys[reverse, ::-1]
    _, first, counts = np.unique(keys.reshape(-1, 4), axis=0, return_index=True, return_counts=True)
    if boundary_only:
        first = first[counts == 1]
    return segments[first]


def _mesh_arrays(mesh):
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.all(np.isfinite(vertices)):
        raise ValueError("Mesh vertices must be finite XYZ coordinates")
    if faces.ndim != 2 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("Mesh faces must be integer triangle indices")
    if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("Mesh face index is outside the vertex array")
    return vertices, faces


def _crossing_points(vertices, distances, faces):
    # Use the same endpoint order for either triangle sharing an edge, so its
    # intersection is bit-identical rather than depending on face winding.
    edges = np.sort(faces[:, _EDGE_CORNERS], axis=2)
    start_d, end_d = distances[edges[..., 0]], distances[edges[..., 1]]
    crossing = ((start_d < 0) & (end_d > 0)) | ((start_d > 0) & (end_d < 0))
    edges = edges[crossing]
    first, second = edges[:, 0], edges[:, 1]
    fraction = distances[first] / (distances[first] - distances[second])
    return vertices[first] + fraction[:, None] * (vertices[second] - vertices[first])


def mesh_section(mesh: TriangleMesh, angle_deg: float, *, tolerance_mm: float = 1e-8) -> MeshSection:
    """Intersect one globally placed mesh with a rotating longitudinal plane.

    Face traversal uses bounded chunks. Tangent points produce no segment;
    on-plane edges appear once, and coplanar faces contribute their boundary
    without tessellation diagonals. Non-material wireframe guides are empty.
    """
    angle, tolerance = _parameters(angle_deg, tolerance_mm)
    segments, coplanar = [], []
    if not mesh.wireframe:
        vertices, faces = _mesh_arrays(mesh)
        radians = math.radians(angle)
        cosine, sine = math.cos(radians), math.sin(radians)
        distances = -vertices[:, 0] * sine + vertices[:, 1] * cosine
        distances[np.abs(distances) <= tolerance] = 0.0
        if np.any(distances <= 0.0) and np.any(distances >= 0.0):
            for start in range(0, len(faces), _FACE_CHUNK_SIZE):
                chunk = faces[start:start + _FACE_CHUNK_SIZE]
                signed = distances[chunk]
                on_plane = signed == 0.0
                counts = np.count_nonzero(on_plane, axis=1)
                straddles = (signed.min(axis=1) < 0.0) & (signed.max(axis=1) > 0.0)

                crossing = chunk[(counts == 0) & straddles]
                if len(crossing):
                    points = _crossing_points(vertices, distances, crossing).reshape(-1, 2, 3)
                    segments.append(_project(points, cosine, sine))

                one_on = (counts == 1) & straddles
                if np.any(one_on):
                    crossing = chunk[one_on]
                    first = vertices[crossing[on_plane[one_on]]]
                    second = _crossing_points(vertices, distances, crossing)
                    segments.append(_project(np.stack((first, second), axis=1), cosine, sine))

                two_on = counts == 2
                if np.any(two_on):
                    points = vertices[chunk[two_on][on_plane[two_on]]].reshape(-1, 2, 3)
                    segments.append(_project(points, cosine, sine))

                face_on = chunk[counts == 3]
                if len(face_on):
                    points = vertices[face_on[:, _EDGE_CORNERS]].reshape(-1, 2, 3)
                    coplanar.append(_project(points, cosine, sine))

    if coplanar:
        segments.append(_unique_segments(np.concatenate(coplanar), tolerance, boundary_only=True))
    output = _unique_segments(np.concatenate(segments), tolerance) if segments else np.empty((0, 2, 2))
    return MeshSection(
        key=mesh.key, region=mesh.region, segments_mm=_immutable_segments(output),
        color=tuple(mesh.color), material_class=mesh.material_class,
        is_exact=mesh.is_exact, description=mesh.description,
    )


def assembly_section_from_model(
    model: AssemblyModel3D, angle_deg: float, *, tolerance_mm: float = 1e-8,
) -> AssemblySection:
    """Cut cached assembly meshes, preserving qualifications and partial errors.

    A solid that misses this cutting plane is absent from ``meshes``. It is
    not an omitted/unsupported component; ``omitted_keys`` is reserved for
    pre-existing omissions, invalid geometry and non-material guides.
    """
    angle, tolerance = _parameters(angle_deg, tolerance_mm)
    sections, omitted, errors = [], list(model.omitted_keys), list(model.errors)
    for mesh in model.meshes:
        if mesh.wireframe:
            omitted.append(mesh.key)
            continue
        try:
            section = mesh_section(mesh, angle, tolerance_mm=tolerance)
            if len(section.segments_mm):
                sections.append(section)
        except (ValueError, TypeError, IndexError, OverflowError) as exc:
            omitted.append(mesh.key)
            errors.append(f"{mesh.key}: {exc}")
    notes = (
        "True V = 0 section of the configured 3D mesh; angular tessellation and "
        "source-model approximation limits apply. Non-material guides are omitted.",
        *model.notes,
    )
    return AssemblySection(angle, tuple(sections), tuple(dict.fromkeys(notes)),
                           tuple(dict.fromkeys(omitted)), tuple(errors))
