"""Read-only, renderer-neutral meshes of existing module-local geometry.

Coordinates are x/y radial and z axial, all in mm. ``is_exact`` means that the
configured axisymmetric section is represented (up to angular tessellation),
not that the configuration is an OEM CAD model. Unsupported mechanisms retain
an explicitly labelled envelope; no holes, slots or material are invented.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import math
from numbers import Integral, Real

import numpy as np

from temsim.magnetic_circuits import radial_profile_mm
from temsim.magnetic_geometry import objective_layer_intervals_mm
from temsim.mechanical_profiles import MAGNETIC_LENS_MECHANICAL_PROFILES


@dataclass(frozen=True)
class DimensionSpec:
    path: tuple[str | int, ...]
    label: str
    value: float
    unit: str
    editable: bool = True
    reason: str = ""
    meaning: object | None = None

    @property
    def name(self):
        return self.path[2]

    @property
    def units(self):
        return self.unit


@dataclass(frozen=True)
class TriangleMesh:
    vertices: np.ndarray
    faces: np.ndarray
    key: str
    region: str = "body"
    color: tuple[float, ...] = (0.55, 0.61, 0.69, 1.0)
    material_class: str = "Unspecified"
    is_exact: bool = True
    description: str = "Configured axisymmetric body"
    face_groups: np.ndarray | None = None
    surfaces: Mapping = field(default_factory=dict)
    edges: tuple = ()

    @property
    def part_key(self):
        return self.key

    @property
    def region_key(self):
        return self.region


@dataclass(frozen=True)
class PartModel3D:
    part_key: str
    name: str
    meshes: tuple[TriangleMesh, ...]
    fields: tuple[DimensionSpec, ...]
    notes: tuple[str, ...] = ()


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _segments(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or not 3 <= value <= 4096:
        raise ValueError("angular_segments must be an integer from 3 to 4096")
    return int(value)


def _colour(material, exact):
    if not exact:
        return (0.53, 0.61, 0.70, 0.28)
    if "copper" in material or "winding" in material:
        return (0.83, 0.43, 0.20, 1.0)
    if "soft_magnetic" in material:
        return (0.36, 0.49, 0.64, 1.0)
    return (0.64, 0.70, 0.75, 1.0)


def _mesh(vertices, faces, key, region, material, exact, description):
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if not len(faces) or not np.all(np.isfinite(vertices)):
        raise ValueError(f"{key}: geometry must produce a finite nonempty mesh")
    triangles = vertices[faces]
    with np.errstate(over="ignore", invalid="ignore"):
        areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                       triangles[:, 2] - triangles[:, 0]), axis=1)
    if not np.all(np.isfinite(areas)) or np.any(areas <= 0):
        raise ValueError(f"{key}: geometry produces degenerate triangles")
    vertices.setflags(write=False)
    faces.setflags(write=False)
    return TriangleMesh(vertices, faces, key, region, _colour(material, exact),
                        material, exact, description)


def revolve_section(section_mm, *, key, region="body", angular_segments=32,
                    material_class="Unspecified", is_exact=True,
                    description="Configured axisymmetric body"):
    """Revolve a simple closed (z, radius) material boundary into outward faces.

    The closing point is optional. Axis points each have one vertex, so solid
    cylinders and cone tips have triangle fans rather than degenerate quads.
    """
    count = _segments(angular_segments)
    points = []
    for row in section_mm:
        if len(row) != 2:
            raise ValueError("A section requires (z, radius) pairs")
        point = (_number(row[0], "z"), _number(row[1], "radius"))
        if point[1] < 0:
            raise ValueError("Section radii must be nonnegative")
        if not points or point != points[-1]:
            points.append(point)
    if points and points[0] == points[-1]:
        points.pop()
    if len(points) < 3 or len(set(points)) != len(points):
        raise ValueError("A material section needs at least three distinct boundary points")
    # Translation avoids cancellation for short bodies far along the column.
    origin = points[0][0]
    signed_area = sum((a[0] - origin) * b[1] - (b[0] - origin) * a[1]
                      for a, b in zip(points, points[1:] + points[:1]))
    if not math.isfinite(signed_area) or signed_area == 0:
        raise ValueError("A material section must have positive area")
    if signed_area > 0:
        points.reverse()
    angles = np.arange(count) * (2 * np.pi / count)
    vertices, rings, faces = [], [], []
    for z, radius in points:
        start = len(vertices)
        if radius == 0:
            vertices.append((0.0, 0.0, z))
            rings.append([start])
        else:
            vertices.extend(zip(radius * np.cos(angles), radius * np.sin(angles),
                                np.full(count, z)))
            rings.append(list(range(start, start + count)))
    for left, right in zip(rings, rings[1:] + rings[:1]):
        if len(left) == len(right) == 1:
            continue  # An axis edge sweeps zero area.
        for j in range(count):
            following = (j + 1) % count
            if len(left) == 1:
                faces.append((left[0], right[following], right[j]))
            elif len(right) == 1:
                faces.append((left[j], left[following], right[0]))
            else:
                faces.extend(((left[j], left[following], right[following]),
                              (left[j], right[following], right[j])))
    return _mesh(vertices, faces, str(key), str(region), str(material_class),
                 bool(is_exact), description)


def _annulus(start, end, inner, outer):
    if not start < end or not 0 <= inner < outer:
        raise ValueError("Body requires start < end and 0 <= inner radius < outer radius")
    return [(start, inner), (start, outer), (end, outer), (end, inner)]


def _part_index(document):
    parts = document.get("parts", ())
    if not isinstance(parts, (tuple, list)):
        raise ValueError("Module parts must be an array")
    result = {}
    for part in parts:
        if not isinstance(part, Mapping) or not isinstance(part.get("key"), str) or not part["key"]:
            raise ValueError("Every part needs a nonempty key")
        if part["key"] in result:
            raise ValueError(f"Duplicate part key: {part['key']}")
        result[part["key"]] = part
    return result


def _dimensions(part, by_key=None):
    from temsim.parameter_semantics import describe_parameter
    fields = []
    def append(value, path, label, unit):
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                append(item, (*path, index), f"{label} [{index}]", unit)
        elif not isinstance(value, bool) and isinstance(value, Real):
            fields.append(DimensionSpec(path, label, _number(value, str(path)), unit))

    for name, value in part.items():
        unit = next((unit for unit in ("mm", "um", "deg") if name.endswith("_" + unit)), None)
        if unit is None:
            continue
        if not any(token in name for token in ("diameter", "radius", "length", "width", "height", "thickness", "gap", "inset", "angle", "radial_profile", "material_intervals")):
            continue
        append(value, ("parts", part["key"], name),
               name.removesuffix("_" + unit).replace("_", " ").capitalize(), unit)
    parent = (by_key or {}).get(part.get("parent_key"), {})
    source_fields = [f"{side}_yoke_{edge}_local_z_mm" for side in ("upper", "lower") for edge in ("start", "end")]
    # Introspection must work even while the draft's dimensions are invalid.
    split = (part.get("mechanical_profile") in {"magnetic_excitation_coil", "magnetic_lens_yoke"}
             and all(name in parent for name in source_fields))
    if split:
        reason = "Display-envelope length; physical sections use the parent yoke endpoints and coil inset below."
        fields = [replace(field, editable=False, reason=reason) if field.path[-1] == "length_mm" else field
                  for field in fields]
        if part["mechanical_profile"] == "magnetic_excitation_coil":
            source_fields.append("mechanical_coil_axial_inset_mm")
        for name in source_fields:
            if name in parent:
                append(parent[name], ("parts", parent["key"], name),
                       "Parent: " + name.removesuffix("_mm").replace("_", " "), "mm")
    annotated = []
    for item in fields:
        owner = (by_key or {}).get(item.path[1], part)
        meaning = describe_parameter(owner, item.path, by_key=by_key)
        label = meaning.label if len(item.path) == 3 else item.label
        if item.path[1] != part["key"]:
            label = "Parent: " + label
        annotated.append(replace(item, label=label, reason=item.reason or meaning.description, meaning=meaning))
    return tuple(annotated)


def _radius(part, fields):
    for name, factor in fields:
        if name in part:
            return _number(part[name], name) * factor
    return None


def _extent(part):
    start = _number(part["local_start_z_mm"], "local_start_z_mm")
    end = _number(part["local_end_z_mm"], "local_end_z_mm")
    if end < start:
        raise ValueError(f"{part['key']}: axial endpoints are reversed")
    if end == start:
        for name in ("housing_length_mm", "electrode_length_mm", "plate_thickness_mm"):
            if name in part and _number(part[name], name) > 0:
                length = _number(part[name], name)
                return start - length / 2, end + length / 2
    return start, end


def _pole_section(part, start, end):
    get = lambda field: _number(part[field], field)
    length = end - start
    bore, outer, tip = (get(field) / 2 for field in
                        ("mechanical_bore_diameter_mm", "mechanical_outer_diameter_mm", "mechanical_tip_diameter_mm"))
    if not 0 <= bore < tip <= outer:
        raise ValueError(f"{part['key']}: pole requires 0 <= bore < tip <= outer radius")
    nose = get("pole_nose_axial_length_mm") if "pole_nose_axial_length_mm" in part else 0.38 * length
    if not 0 < nose <= length:
        raise ValueError(f"{part['key']}: pole nose must fit its axial length")
    style = part.get("pole_piece_geometry_style", "")
    if style not in {"", "embedded_hourglass_bore", "tapered_bore_pole",
                     "objective_vertical_back_inserted_shank_tapered_nose"}:
        return None
    if style == "objective_vertical_back_inserted_shank_tapered_nose":
        shank = get("pole_mounting_shank_axial_length_mm")
        shank_inner = get("pole_mounting_shank_inner_diameter_mm") / 2
        stem = get("pole_stem_outer_diameter_mm") / 2
        connector = get("pole_vacuum_connector_axial_length_mm") if "pole_vacuum_connector_axial_length_mm" in part else 0
        tail = get("pole_vacuum_connector_outer_diameter_mm") / 2 if connector else stem
        if not (0 <= connector < shank <= length - nose and 0 <= shank_inner < tail <= stem <= outer):
            raise ValueError(f"{part['key']}: invalid pole shank/connector/shoulder dimensions")
        section = [(0, shank_inner), (0, tail)]
        if connector:
            section += [(connector, tail), (connector, stem)]
        section += [(shank, stem), (shank, outer), (length - nose, outer),
                    (length, tip), (length, bore), (shank, bore), (shank, shank_inner)]
    else:
        section = [(0, bore), (0, outer), (length - nose, outer), (length, tip), (length, bore)]
    # The shared C1/C2 cartridge's two pole faces are reversed relative to their
    # optical names, as in the existing physical-layout projection.
    face_at_end = "upper" in part["key"]
    if part["key"] == "condenser_lens_1_lower_pole":
        face_at_end = True
    elif part["key"] == "condenser_lens_2_upper_pole":
        face_at_end = False
    return [(start + z if face_at_end else end - z, radius) for z, radius in section]


def _legacy_part_meshes(part, by_key, count, aperture_index=0, runtime=None):
    from temsim.part_model_apertures import is_strip_aperture, strip_meshes
    if is_strip_aperture(part):
        return strip_meshes(part, count, runtime, aperture_index)
    key, profile = part["key"], part.get("mechanical_profile", "")
    material = str(part.get("material_class", "Unspecified"))
    start, end = _extent(part)
    outer = _radius(part, (("mechanical_outer_diameter_mm", .5), ("mechanical_outer_radius_mm", 1),
                           ("outer_diameter_mm", .5), ("outer_width_mm", .5)))
    inner = _radius(part, (("mechanical_inner_diameter_mm", .5), ("mechanical_bore_diameter_mm", .5),
                           ("mechanical_clear_bore_diameter_mm", .5), ("mechanical_bore_radius_mm", 1),
                           ("bore_diameter_mm", .5), ("clear_bore_diameter_mm", .5),
                           ("inner_diameter_mm", .5)))
    emit = lambda section, region="body", exact=True, description="Configured axisymmetric body": revolve_section(
        section, key=key, region=region, angular_segments=count, material_class=material,
        is_exact=exact, description=description)
    radial = radial_profile_mm(part, end - start)
    if radial is not None:
        section = [(start + z, radius) for z, _, radius in radial]
        section += [(start + z, radius) for z, radius, _ in radial[::-1]]
        return (emit(section, description="Configured magnetic radial profile"),), ()
    if profile == "magnetic_pole_piece" and end > start:
        section = _pole_section(part, start, end)
        if section is not None:
            notes = () if "pole_nose_axial_length_mm" in part else (
                f"{key}: nose uses the existing 0.38 × length reconstruction rule.",)
            if "pole_root_fillet_radius_range_mm" in part:
                notes += (f"{key}: fillet radius range is metadata; the existing physical model has sharp shoulders.",)
            return (emit(section, description="Configured pole section with an open bore"),), notes
    hole_field = next((name for name in ("aperture_hole_diameters_mm", "aperture_hole_diameters_um",
                                       "hole_diameters_mm", "hole_diameters_um") if name in part), None)
    if hole_field:
        holes = part[hole_field]
        if (not isinstance(holes, (list, tuple)) or not holes
                or isinstance(aperture_index, bool) or not isinstance(aperture_index, Integral)
                or not 0 <= aperture_index < len(holes)):
            raise ValueError(f"{key}: aperture_index must select an existing aperture")
        hole = _number(holes[aperture_index], hole_field) * (0.001 if hole_field.endswith("_um") else 1) / 2
        if hole <= 0:
            raise ValueError(f"{key}: the selected aperture diameter must be positive")
        if outer is None:
            return (), (f"{key}: selected existing aperture index={aperture_index}; outer plate envelope is undefined.",)
        if "plate_thickness_mm" in part:
            thickness = _number(part["plate_thickness_mm"], "plate_thickness_mm")
            center = _number(part.get("optical_reference_local_z_mm", part.get("local_center_z_mm", (start + end) / 2)), "aperture center")
            start, end = center - thickness / 2, center + thickness / 2
        description = (f"Selected existing aperture index={aperture_index}, diameter={hole * 2:g} mm; "
                       "outer plate envelope only, multi-hole positions are not defined")
        return (emit(_annulus(start, end, hole, outer), exact=False, description=description),), (f"{key}: {description}.",)
    if profile in MAGNETIC_LENS_MECHANICAL_PROFILES and outer is not None and inner is not None:
        parent = by_key.get(part.get("parent_key"), {})
        intervals = ()
        if profile in {"magnetic_excitation_coil", "magnetic_lens_yoke"}:
            intervals = objective_layer_intervals_mm(parent, parent.get("local_start_z_mm", 0), profile)
        if intervals:
            return tuple(emit(_annulus(a, b, inner, outer), region,
                              description="Parent-defined physical material interval")
                         for region, (a, b) in zip(("upper", "lower"), intervals)), ()
        if part.get("material_intervals_mm") is not None:
            intervals = part["material_intervals_mm"]
            pieces = [emit(_annulus(_number(a, "interval start"), _number(b, "interval end"), inner, outer))
                      for a, b in intervals]
            if not pieces:
                raise ValueError(f"{key}: material intervals must not be empty")
            return (_join_meshes(pieces),), ()
        return (emit(_annulus(start, end, inner, outer)),), ()
    exact = profile in {"circular_aperture", "vacuum_liner", "electrostatic_bias_tube"}
    if outer is None:
        return (), (f"{key}: no defined outer envelope; only existing dimensions are listed.",)
    if inner is None:
        # A vacuum clearance is not a hole in the material (e.g. a cathode or
        # specimen can occupy it). An unknown construction gets a solid envelope.
        inner = 0
        exact = False
    description = "Configured axisymmetric body" if exact else "Envelope only; internal/non-axisymmetric construction is not defined by this surface"
    if start == end:
        # Preserve a true reference plane; never manufacture a visible thickness.
        if not 0 <= inner < outer:
            return (), (f"{key}: reference plane has no material annulus.",)
        angles = np.arange(count) * (2 * np.pi / count)
        vertices = [(outer * np.cos(t), outer * np.sin(t), start) for t in angles]
        if inner == 0:
            vertices.append((0, 0, start))
            faces = [(count, j, (j + 1) % count) for j in range(count)]
        else:
            vertices += [(inner * np.cos(t), inner * np.sin(t), start) for t in angles]
            faces = [(j, (j + 1) % count, count + (j + 1) % count) for j in range(count)]
            faces += [(j, count + (j + 1) % count, count + j) for j in range(count)]
        return (_mesh(vertices, faces, key, "body", material, False, "Reference/envelope plane; no axial thickness"),), (f"{key}: zero-thickness reference/envelope plane.",)
    return (emit(_annulus(start, end, inner, outer), exact=exact, description=description),), (() if exact else (f"{key}: {description}.",))


def _join_meshes(pieces):
    offsets = np.cumsum([0] + [len(piece.vertices) for piece in pieces[:-1]])
    first = pieces[0]
    return _mesh(np.concatenate([piece.vertices for piece in pieces]),
                 np.concatenate([piece.faces + offset for piece, offset in zip(pieces, offsets)]),
                 first.key, first.region, first.material_class, all(piece.is_exact for piece in pieces),
                 "Configured disjoint material intervals")


def _part_meshes(part, by_key, count, aperture_index=0, runtime=None):
    from temsim.part_model_features import apply_model_features, annotate_legacy_mesh, validate_model_3d
    validate_model_3d(part)
    # An explicit new base can model parts whose old display has no usable solid.
    explicit = part.get("model_3d", {}).get("base", {}).get("kind", "existing") != "existing"
    if explicit:
        meshes, notes = (), ()
    else:
        meshes, notes = _legacy_part_meshes(part, by_key, count, aperture_index, runtime)
    meshes = tuple(annotate_legacy_mesh(mesh, part, by_key) for mesh in meshes)
    if "model_3d" in part:
        meshes = apply_model_features(part, meshes, angular_segments=count)
        notes += (f"{part['key']}: explicit 3D solid and feature geometry; the optical/magnetic solver does not infer a new field model from this mesh.",)
    return meshes, notes


def part_dimension_specs(document, part_key, *, runtime_values=None):
    """List existing dimension paths even when the draft cannot produce a mesh."""
    by_key = _part_index(document)
    if part_key not in by_key:
        raise ValueError(f"Unknown part: {part_key}")
    from temsim.part_model_features import feature_dimension_specs
    from temsim.part_model_apertures import operating_dimension_specs
    part = by_key[part_key]
    from temsim.parameter_semantics import describe_parameter
    specs = (_dimensions(part, by_key) + feature_dimension_specs(part)
             + operating_dimension_specs(part, (runtime_values or {}).get(part_key)))
    return tuple(item if item.meaning is not None else replace(
        item, meaning=describe_parameter(by_key.get(item.path[1], part), item.path, by_key=by_key))
        for item in specs)


def part_model_from_document(document, part_key, *, angular_segments=32, include_children=True,
                             aperture_index=0, runtime_values=None):
    """Read one part and optionally its declared descendants/shared bodies.

    Ownership is preserved: a shared cartridge retains its own key and occurs
    once, rather than becoming a filled cylinder for each optical channel.
    """
    by_key = _part_index(document)
    count = _segments(angular_segments)
    if part_key not in by_key:
        raise ValueError(f"Unknown part: {part_key}")
    keys = {part_key}
    if include_children:
        while True:
            more = {key for key, part in by_key.items()
                    if part.get("parent_key") in keys or keys.intersection(part.get("magnetic_lens_keys", ()))}
            if more <= keys:
                break
            keys.update(more)
    meshes, notes = [], []
    for key, part in by_key.items():
        if key in keys:
            items, messages = _part_meshes(part, by_key, count, aperture_index if key == part_key else 0,
                                          (runtime_values or {}).get(key))
            meshes.extend(items)
            notes.extend(messages)
    part = by_key[part_key]
    return PartModel3D(part_key, str(part.get("name", part_key)), tuple(meshes),
                       part_dimension_specs(document, part_key, runtime_values=runtime_values), tuple(notes))


def module_model_from_document(document, *, angular_segments=32, runtime_values=None):
    """Render each declared part once, without duplicating shared ownership."""
    by_key = _part_index(document)
    count = _segments(angular_segments)
    meshes, fields, notes = [], [], []
    for part in by_key.values():
        items, messages = _part_meshes(part, by_key, count, runtime=(runtime_values or {}).get(part["key"]))
        meshes.extend(items)
        fields.extend(part_dimension_specs(document, part["key"], runtime_values=runtime_values))
        notes.extend(messages)
    module = document.get("module", {})
    return PartModel3D(str(module.get("key", "module")), str(module.get("name", "Module")),
                       tuple(meshes), tuple(fields), tuple(notes))
