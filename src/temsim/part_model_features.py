"""Explicit 3-D solids and subtractive features, with Boolean provenance.

Manifold's original mesh IDs carry semantic surface ownership through cuts.
Geometry is constructed relative to the part centre, then transformed by
Rz @ Ry @ Rx @ diag(scale_x, scale_y, 1), and translated back to module mm.
No optical or magnetic constitutive law is inferred from this CAD geometry.
"""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import replace
import math
from numbers import Integral, Real

import numpy as np


def _number(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{label} must be {'positive and ' if positive else ''}finite")
    return result


def _vector(value, size, label, *, positive=False):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{label} requires {size} numbers")
    return np.array([_number(v, label, positive=positive) for v in value])


def _table(value, allowed, label):
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a table")
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown {label} fields: {', '.join(sorted(map(str, unknown)))}")


def default_model_3d(part):
    """A fresh editable configuration that preserves the existing base shape."""
    return {"schema_version": 1, "base": {"kind": "existing"},
            "transform": {"scale_xy": [1.0, 1.0], "offset_mm": [0.0, 0.0, 0.0],
                          "rotation_deg": [0.0, 0.0, 0.0]}, "features": []}


def validate_model_3d(part):
    """Validate schema only; actual Boolean validity is checked when building."""
    if "model_3d" not in part:
        return
    model = part["model_3d"]
    _table(model, ("schema_version", "base", "transform", "features"), "model_3d")
    if type(model.get("schema_version")) is not int or model["schema_version"] != 1:
        raise ValueError("model_3d.schema_version must be 1")
    base = model.get("base", {"kind": "existing"})
    _table(base, ("kind", "width_mm", "height_mm", "length_mm"), "model_3d.base")
    kind = base.get("kind", "existing")
    if kind not in {"existing", "box", "elliptic_cylinder"}:
        raise ValueError("Base kind must be existing, box or elliptic_cylinder")
    if kind == "existing" and set(base) - {"kind"}:
        raise ValueError("An existing base uses the part's dimensions, not separate base dimensions")
    if kind != "existing":
        for name in ("width_mm", "height_mm", "length_mm"):
            _number(base.get(name), "base." + name, positive=True)
    transform = model.get("transform", {})
    _table(transform, ("scale_xy", "offset_mm", "rotation_deg"), "model_3d.transform")
    _vector(transform.get("scale_xy", [1, 1]), 2, "scale_xy", positive=True)
    _vector(transform.get("offset_mm", [0, 0, 0]), 3, "offset_mm")
    _vector(transform.get("rotation_deg", [0, 0, 0]), 3, "rotation_deg")
    features = model.get("features", [])
    if not isinstance(features, list) or len(features) > 64:
        raise ValueError("model_3d.features must be an array of at most 64 features")
    ids = set()
    for feature in features:
        _table(feature, ("id", "kind", "axis", "center_mm", "depth_mm", "diameter_mm",
                         "width_mm", "length_mm", "rotation_deg", "enabled"), "feature")
        identity = feature.get("id")
        if not isinstance(identity, str) or not identity.strip() or identity in ids:
            raise ValueError("Every feature requires a unique nonempty id")
        ids.add(identity)
        if feature.get("kind") not in {"hole", "slot"}:
            raise ValueError("Feature kind must be hole or slot")
        if feature.get("axis", "z") not in {"x", "y", "z"}:
            raise ValueError("Feature axis must be x, y or z")
        if type(feature.get("enabled", True)) is not bool:
            raise ValueError("Feature enabled must be a boolean")
        _vector(feature.get("center_mm", [0, 0, 0]), 3, "feature.center_mm")
        _number(feature.get("depth_mm"), "feature.depth_mm", positive=True)
        _number(feature.get("rotation_deg", 0), "feature.rotation_deg")
        if feature["kind"] == "hole":
            _number(feature.get("diameter_mm"), "feature.diameter_mm", positive=True)
            if "width_mm" in feature or "length_mm" in feature:
                raise ValueError("A hole uses diameter_mm, not slot width/length")
        else:
            width = _number(feature.get("width_mm"), "feature.width_mm", positive=True)
            length = _number(feature.get("length_mm"), "feature.length_mm", positive=True)
            if length < width or "diameter_mm" in feature:
                raise ValueError("A slot requires length_mm >= width_mm and no diameter_mm")


def feature_dimension_specs(part):
    """Numeric paths refer to actual dict keys and feature list indices."""
    if "model_3d" not in part or not isinstance(part["model_3d"], Mapping):
        return ()
    from temsim.part_model_3d import DimensionSpec
    specs = []
    def visit(value, path, label, unit="mm"):
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in {"schema_version", "enabled"}:
                    continue
                next_unit = "deg" if key == "rotation_deg" else "×" if key == "scale_xy" else unit
                visit(item, (*path, key), (label + " / " + key.replace("_", " ")).strip(" /"), next_unit)
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(item, (*path, index), label + f" [{index}]", unit)
        elif isinstance(value, Real) and not isinstance(value, bool):
            specs.append(DimensionSpec(path, label, _number(value, label), unit))
    visit(part["model_3d"], ("parts", part["key"], "model_3d"), "3D model")
    return tuple(specs)


def _normal_data(vertices, faces):
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    norms = np.linalg.norm(normals, axis=1)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("A 3D solid must contain only finite, nondegenerate triangles")
    return triangles.mean(axis=1), normals / norms[:, None]


def _paths(part, names):
    return tuple(("parts", part["key"], name) for name in names if name in part)


def annotate_legacy_mesh(mesh, part, by_key=None):
    """Classify the existing rotational section before any arbitrary transform."""
    if mesh.face_groups is not None:
        return mesh
    centers, normals = _normal_data(mesh.vertices, mesh.faces)
    groups = np.where(np.abs(normals[:, 2]) > 1 - 1e-9,
                      np.where(normals[:, 2] > 0, "axial_positive", "axial_negative"),
                      np.where(np.einsum("ij,ij->i", centers[:, :2], normals[:, :2]) < 0, "inner", "outer"))
    axial = _paths(part, ("length_mm", "pole_nose_axial_length_mm", "pole_mounting_shank_axial_length_mm",
                          "pole_vacuum_connector_axial_length_mm", "plate_thickness_mm"))
    outer = _paths(part, ("mechanical_outer_diameter_mm", "mechanical_outer_radius_mm", "outer_diameter_mm",
                          "outer_width_mm", "mechanical_tip_diameter_mm", "pole_stem_outer_diameter_mm",
                          "pole_vacuum_connector_outer_diameter_mm", "pole_nose_axial_length_mm"))
    inner = _paths(part, ("mechanical_inner_diameter_mm", "mechanical_bore_diameter_mm", "bore_diameter_mm",
                          "mechanical_clear_bore_diameter_mm", "mechanical_bore_radius_mm", "clear_bore_diameter_mm",
                          "inner_diameter_mm", "pole_mounting_shank_inner_diameter_mm"))
    upstream_axial = downstream_axial = axial
    parent = (by_key or {}).get(part.get("parent_key"), {})
    if mesh.region in {"upper", "lower"} and part.get("mechanical_profile") in {
            "magnetic_lens_yoke", "magnetic_excitation_coil"}:
        from temsim.magnetic_circuits import is_custom_mechanical_part
        start_field = mesh.region + "_yoke_start_local_z_mm"
        end_field = mesh.region + "_yoke_end_local_z_mm"
        if is_custom_mechanical_part(part) and "material_intervals_mm" in part:
            index = 0 if mesh.region == "upper" else 1
            upstream_axial = (("parts", part["key"], "material_intervals_mm", index, 0),)
            downstream_axial = (("parts", part["key"], "material_intervals_mm", index, 1),)
        elif start_field in parent and end_field in parent:
            inset = (("mechanical_coil_axial_inset_mm",) if
                     part["mechanical_profile"] == "magnetic_excitation_coil" else ())
            upstream_axial = _paths(parent, (start_field, *inset))
            downstream_axial = _paths(parent, (end_field, *inset))
    upstream_paths = tuple(dict.fromkeys((*upstream_axial, *inner, *outer)))
    downstream_paths = tuple(dict.fromkeys((*downstream_axial, *inner, *outer)))
    surfaces = {"outer": {"label": "Outer surface", "kind": "outer", "parameter_paths": outer},
                "inner": {"label": "Bore / inner surface", "kind": "inner", "parameter_paths": inner},
                "axial_negative": {"label": "Upstream axial face", "kind": "cap", "parameter_paths": upstream_paths},
                "axial_positive": {"label": "Downstream axial face", "kind": "cap", "parameter_paths": downstream_paths}}
    # Profile knots and listed aperture diameters are actual authoritative paths.
    for spec in _legacy_array_specs(part):
        target = "inner" if "diameters" in str(spec.path[2]) or spec.path[-1] == 1 else "outer"
        surfaces[target]["parameter_paths"] += (spec.path,)
    # Separate real profile segments/planes, rather than highlighting every
    # shoulder with the same normal. Ordered segment IDs survive dimension edits.
    triangles = mesh.vertices[mesh.faces]
    groups = groups.astype(object)
    for group in tuple(surfaces):
        indices = np.flatnonzero(groups == group)
        bands = sorted({(float(triangles[index, :, 2].min()), float(triangles[index, :, 2].max()))
                        for index in indices})
        if len(bands) <= 1:
            continue
        for ordinal, (start, end) in enumerate(bands):
            identity = group + ":segment:" + str(ordinal)
            selected = [index for index in indices if
                        (float(triangles[index, :, 2].min()), float(triangles[index, :, 2].max())) == (start, end)]
            groups[selected] = identity
            surfaces[identity] = {**surfaces[group], "label": surfaces[group]["label"] +
                                  (f" at Z {start:g} mm" if start == end else f" section {ordinal + 1}")}
    return _decorate(mesh, groups, surfaces)


def _legacy_array_specs(part):
    from temsim.part_model_3d import _dimensions
    return [spec for spec in _dimensions(part) if len(spec.path) > 3 and spec.path[2] in {
        "magnetic_radial_profile_mm", "aperture_hole_diameters_mm", "aperture_hole_diameters_um",
        "hole_diameters_mm", "hole_diameters_um"}]


def _decorate(mesh, groups, metadata):
    groups = np.array(groups, dtype=str)
    if len(groups) != len(mesh.faces):
        raise ValueError("Surface provenance must have one ID per triangle")
    _, normals = _normal_data(mesh.vertices, mesh.faces)
    surfaces = {}
    for group in sorted(set(groups)):
        if group not in metadata:
            raise ValueError(f"Missing surface provenance: {group}")
        info = dict(metadata[group])
        info.pop("normal", None)
        subset = normals[groups == group]
        if np.all(np.linalg.norm(subset - subset[0], axis=1) < 1e-7):
            info["normal"] = tuple(subset[0])
        surfaces[group] = info
    groups.setflags(write=False)
    return replace(mesh, face_groups=groups, surfaces=surfaces,
                   edges=_semantic_edges(mesh.vertices, mesh.faces, groups, surfaces))


def _semantic_edges(vertices, faces, groups, surfaces):
    adjacency = defaultdict(list)
    for face, group in zip(faces, groups):
        for a, b in zip(face, np.roll(face, -1)):
            adjacency[tuple(sorted((int(a), int(b))))].append(str(group))
    boundaries = defaultdict(set)
    for edge, owners in adjacency.items():
        distinct = tuple(sorted(set(owners)))
        if len(distinct) > 1:
            boundaries[distinct].add(edge)
    records = []
    for pair, segments in sorted(boundaries.items()):
        neighbours = defaultdict(set)
        for a, b in segments:
            neighbours[a].add(b)
            neighbours[b].add(a)
        chains = []
        unused = set(segments)
        while unused:
            candidates = {v for edge in unused for v in edge}
            ends = [v for v in candidates if len(neighbours[v]) != 2]
            first = min(ends or candidates, key=lambda v: tuple(vertices[v]))
            chain, current = [first], first
            while True:
                available = [v for v in neighbours[current] if tuple(sorted((current, v))) in unused]
                if not available:
                    break
                following = min(available, key=lambda v: tuple(vertices[v]))
                unused.remove(tuple(sorted((current, following))))
                chain.append(following)
                current = following
                if current == first or len(neighbours[current]) != 2:
                    break
            chains.append(vertices[chain])
        chains.sort(key=lambda points: tuple(points.mean(axis=0)))
        paths = tuple(dict.fromkeys(path for surface in pair for path in surfaces[surface]["parameter_paths"]))
        label = " / ".join(surfaces[surface]["label"] for surface in pair)
        for index, points in enumerate(chains):
            points.setflags(write=False)
            records.append({"id": "|".join(pair) + f":{index}", "label": label,
                            "vertices": points, "surface_ids": pair, "parameter_paths": paths})
    return tuple(records)


def _manifold_module():
    try:
        import manifold3d
    except ImportError as exc:
        raise ValueError("3D solid features require the manifold3d dependency") from exc
    return manifold3d


def _input_solid(mesh, provenance):
    m = _manifold_module()
    groups = mesh.face_groups
    ordered = sorted(set(groups))
    indices = np.concatenate([np.flatnonzero(groups == group) for group in ordered])
    counts = [int(np.count_nonzero(groups == group)) for group in ordered]
    start_id = m.Manifold.reserve_ids(len(ordered))
    ids = np.arange(start_id, start_id + len(ordered), dtype=np.uint32)
    for identity, group in zip(ids, ordered):
        provenance[int(identity)] = (str(group), mesh.surfaces[str(group)])
    mesh64 = m.Mesh64(np.array(mesh.vertices, dtype=np.float64, order="C", copy=True),
                     np.array(mesh.faces[indices], dtype=np.uint64, order="C", copy=True),
                     run_index=np.asarray(np.cumsum([0, *counts]) * 3, dtype=np.uint64),
                     run_original_id=ids)
    solid = m.Manifold(mesh64)
    if solid.status() != m.Error.NoError or solid.is_empty() or solid.volume() <= 0:
        raise ValueError(f"{mesh.key}: the existing mesh is not a closed manifold solid ({solid.status()})")
    return solid


def _primitive_mesh(part, solid, kind, paths, prefix, *, feature=False):
    from temsim.part_model_3d import _mesh
    raw = solid.to_mesh64()
    vertices, faces = np.asarray(raw.vert_properties)[:, :3], np.asarray(raw.tri_verts, dtype=np.int64)
    centers, normals = _normal_data(vertices, faces)
    if kind == "box":
        axes = np.argmax(np.abs(normals), axis=1)
        labels = ["xyz"[axis] + ("_positive" if normal[axis] > 0 else "_negative") for axis, normal in zip(axes, normals)]
    else:
        labels = np.where(np.abs(normals[:, 2]) > 1 - 1e-9,
                          np.where(normals[:, 2] > 0, "end", "start"), "wall")
    groups = np.array([prefix + ":" + label for label in labels])
    metadata = {}
    for group, label in zip(groups, labels):
        selected_paths = paths
        if not feature and kind == "box":
            axis_field = {"x": "width_mm", "y": "height_mm", "z": "length_mm"}[label[0]]
            selected_paths = tuple(path for path in paths if path[-1] == axis_field)
        elif not feature:
            selected_paths = tuple(path for path in paths if (path[-1] == "length_mm") == (label != "wall"))
        metadata[str(group)] = {"label": prefix.replace(":", " ") + " " + str(label).replace("_", " "),
                                "kind": "cut" if feature else "base", "parameter_paths": selected_paths}
    mesh = _mesh(vertices, faces, part["key"], "body", str(part.get("material_class", "Unspecified")),
                 True, "Explicit 3D solid" if not feature else "Boolean cutting tool")
    return _decorate(mesh, groups, metadata)


def _cut_mesh(part, feature, index, segments):
    m = _manifold_module()
    depth = float(feature["depth_mm"])
    if feature["kind"] == "hole":
        solid = m.Manifold.cylinder(depth, float(feature["diameter_mm"]) / 2,
                                   circular_segments=segments, center=True)
    else:
        radius, span = float(feature["width_mm"]) / 2, float(feature["length_mm"])
        half = span / 2 - radius
        angles = np.linspace(-np.pi / 2, np.pi / 2, max(2, segments // 2) + 1)
        right = np.c_[half + radius * np.cos(angles), radius * np.sin(angles)]
        left = np.c_[-half + radius * np.cos(angles + np.pi), radius * np.sin(angles + np.pi)]
        contour = np.concatenate((right, left)) if half else np.c_[radius * np.cos(np.arange(segments) * 2 * np.pi / segments), radius * np.sin(np.arange(segments) * 2 * np.pi / segments)]
        solid = m.Manifold.extrude(m.CrossSection([contour]), depth).translate((0, 0, -depth / 2))
    prefix = ("parts", part["key"], "model_3d", "features", index)
    paths = tuple(spec.path for spec in feature_dimension_specs(part) if spec.path[:5] == prefix)
    mesh = _primitive_mesh(part, solid, feature["kind"], paths, "feature:" + feature["id"], feature=True)
    # Slot orientation rotates in the transverse plane before aligning its axis.
    angle = math.radians(float(feature.get("rotation_deg", 0)))
    spin = np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    axis = feature.get("axis", "z")
    orient = {"z": np.eye(3), "x": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
              "y": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])}[axis]
    vertices = mesh.vertices @ (orient @ spin).T + np.asarray(feature.get("center_mm", [0, 0, 0]))
    return _decorate(replace(mesh, vertices=vertices), mesh.face_groups, mesh.surfaces)


def _transform(part):
    model = part.get("model_3d", {})
    transform = model.get("transform", {})
    sx, sy = transform.get("scale_xy", [1, 1])
    x, y, z = np.radians(transform.get("rotation_deg", [0, 0, 0]))
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return rz @ ry @ rx @ np.diag([sx, sy, 1]), np.asarray(transform.get("offset_mm", [0, 0, 0]), float)


def apply_model_features(part, legacy_meshes, *, angular_segments=32):
    """Build real closed Boolean differences, preserving source surface IDs."""
    validate_model_3d(part)
    if "model_3d" not in part:
        return tuple(legacy_meshes)
    from temsim.part_model_3d import _mesh, _segments
    count = _segments(angular_segments)
    center = np.array([0, 0, _number(part.get("local_center_z_mm"), "local_center_z_mm")])
    model = part["model_3d"]
    base = model.get("base", {"kind": "existing"})
    if base.get("kind", "existing") == "existing":
        if not legacy_meshes:
            raise ValueError("This part has no existing solid; choose an explicit box or elliptic cylinder")
        meshes = [replace(mesh, vertices=mesh.vertices - center) for mesh in legacy_meshes]
    else:
        m = _manifold_module()
        width, height, length = (float(base[name]) for name in ("width_mm", "height_mm", "length_mm"))
        if base["kind"] == "box":
            solid = m.Manifold.cube((width, height, length), center=True)
        else:
            solid = m.Manifold.cylinder(length, .5, circular_segments=count, center=True).scale((width, height, 1))
        paths = tuple(("parts", part["key"], "model_3d", "base", name) for name in ("width_mm", "height_mm", "length_mm"))
        meshes = [_primitive_mesh(part, solid, base["kind"], paths, "base")]
    features = [(index, feature) for index, feature in enumerate(model.get("features", [])) if feature.get("enabled", True)]
    if features:
        if any(not mesh.is_exact for mesh in meshes):
            raise ValueError("An envelope is not a verified solid; choose an explicit box or elliptic cylinder before cutting")
        provenance = {}
        solids = [_input_solid(mesh, provenance) for mesh in meshes]
        for index, feature in features:
            tool = _input_solid(_cut_mesh(part, feature, index, count), provenance)
            before = sum(solid.volume() for solid in solids)
            results = [solid - tool for solid in solids]
            if any(result.status() != _manifold_module().Error.NoError for result in results):
                raise ValueError(f"Feature {feature['id']} could not produce a manifold solid")
            after = sum(result.volume() for result in results)
            if not math.isfinite(after) or after <= 0:
                raise ValueError(f"Feature {feature['id']} removes the entire solid")
            if before - after <= max(abs(before) * 1e-12, np.finfo(float).tiny):
                raise ValueError(f"Feature {feature['id']} does not remove material from the solid")
            solids = results
        converted = []
        for original, solid in zip(meshes, solids):
            if solid.is_empty():
                continue
            raw = solid.to_mesh64()
            vertices = np.asarray(raw.vert_properties)[:, :3]
            faces = np.asarray(raw.tri_verts, dtype=np.int64)
            groups = np.empty(len(faces), dtype=object)
            metadata = {}
            for start, end, identity in zip(raw.run_index, raw.run_index[1:], raw.run_original_id):
                group, info = provenance[int(identity)]
                groups[int(start) // 3:int(end) // 3] = group
                metadata[group] = info
            mesh = _mesh(vertices, faces, original.key, original.region, original.material_class, True,
                         "Explicit 3D solid with Boolean holes/slots")
            converted.append(_decorate(mesh, groups, metadata))
        meshes = converted
    matrix, offset = _transform(part)
    transform_paths = tuple(spec.path for spec in feature_dimension_specs(part) if spec.path[3] == "transform")
    output = []
    for mesh in meshes:
        vertices = mesh.vertices @ matrix.T + center + offset
        if not np.isfinite(vertices).all():
            raise ValueError("3D transform produces non-finite vertices")
        vertices.setflags(write=False)
        metadata = {key: {**value, "parameter_paths": tuple(dict.fromkeys((*value["parameter_paths"], *transform_paths)))}
                    for key, value in mesh.surfaces.items()}
        output.append(_decorate(replace(mesh, vertices=vertices), mesh.face_groups, metadata))
    return tuple(output)


def feature_placement(part, point_mm, normal):
    """Return local feature centre, cardinal axis and a through-cut depth.

    The non-axis-aligned hit normal chooses its nearest local cardinal axis;
    arbitrary freeform cutting directions are intentionally outside this schema.
    """
    validate_model_3d(part)
    point = _vector(list(point_mm), 3, "point_mm")
    normal = _vector(list(normal), 3, "normal")
    if np.linalg.norm(normal) <= 0:
        raise ValueError("A placement normal must be nonzero")
    matrix, offset = _transform(part)
    center = np.array([0, 0, _number(part["local_center_z_mm"], "local_center_z_mm")])
    local = np.linalg.solve(matrix, point - center - offset)
    axis = int(np.argmax(np.abs(matrix.T @ normal)))
    base = part.get("model_3d", {}).get("base", {"kind": "existing"})
    if base.get("kind", "existing") == "existing":
        diameter = float(part.get("mechanical_outer_diameter_mm", part.get("outer_diameter_mm", part.get("outer_width_mm", 1))))
        radii = np.array([diameter / 2, diameter / 2,
                          max(abs(float(part.get("local_start_z_mm", center[2])) - center[2]),
                              abs(float(part.get("local_end_z_mm", center[2])) - center[2]))])
    else:
        radii = np.array([base["width_mm"], base["height_mm"], base["length_mm"]], float) / 2
    depth = 2 * max(radii[axis], abs(local[axis])) * 1.1 + 1
    local[axis] = 0
    return tuple(float(value) for value in local), "xyz"[axis], float(depth)
