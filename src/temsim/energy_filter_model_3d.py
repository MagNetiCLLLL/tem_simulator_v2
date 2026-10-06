"""Configured Energy Filter surfaces in its existing Cartesian branch frame.

The branch enters along +X, keeps +Y non-dispersive and bends toward -Z.
These are mechanical/reference surfaces, not magnetic field support lengths.
No optical solver, file read, inferred yoke, electrode thickness or camera
package is involved. Dimensions retain the provenance of the source document.
"""
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
import math
from numbers import Integral

import numpy as np

from temsim.part_model_3d import _annulus, _mesh, _number, revolve_section


# Proper rotation: branch +X -> column +Z, branch -Z -> column +X.
# This acts on filter Cartesian coordinates. The ray tracer's historical
# entrance transverse-x -> filter +Z convention is deliberately not changed.
FILTER_TO_COLUMN_ROTATION = ((0., 0., -1.), (0., 1., 0.), (1., 0., 0.))
_PROFILES = frozenset({
    "tapered_sector_prism", "energy_filter_multipole_carrier",
    "xo_energy_slit_assembly", "electrostatic_quadrupole",
    "electrostatic_bias_tube", "fast_electrostatic_shutter",
    "electrostatic_camera_deflector", "zebra_eels_detector",
})


def supports_energy_filter_part(part):
    return ((bool(part.get("branch_path_only")) and part.get("branch") == "energy_filter"
             and part.get("mechanical_profile") in _PROFILES)
            or part.get("key") == "energy_filter_entrance_aperture")


def energy_filter_material_region(part, region):
    """Map known display subregions to the existing single body assignment.

    Display patches do not create separately calibrated material domains.
    Unknown region names still fail the caller's ordinary validation.
    """
    if not supports_energy_filter_part(part):
        return region
    by_profile = {
        "tapered_sector_prism": {"lower_pole_boundary", "upper_pole_boundary", "inner_radial_channel", "outer_radial_channel"},
        "energy_filter_multipole_carrier": {"housing_envelope"},
        "electrostatic_quadrupole": {"housing_envelope"},
        "electrostatic_bias_tube": {"housing_envelope"},
        "xo_energy_slit_assembly": {"lower_blade_edge", "upper_blade_edge"},
        "fast_electrostatic_shutter": {"outer_envelope", "negative_gap_boundary", "positive_gap_boundary"},
        "electrostatic_camera_deflector": {"outer_envelope", "negative_gap_boundary", "positive_gap_boundary"},
        "zebra_eels_detector": {"alignment_active_area"},
    }
    allowed = by_profile.get(part.get("mechanical_profile"), set())
    if part.get("mechanical_profile") == "zebra_eels_detector":
        count = part.get("strip_count")
        if isinstance(count, Integral) and not isinstance(count, bool) and 1 <= count <= 256:
            allowed = allowed | {f"active_strip_{index + 1}" for index in range(count)}
    return "body" if region in allowed else region


def energy_filter_render_values(state):
    """Capture the actual slit position for rendering, without ensuring state."""
    slit = getattr(getattr(state, "energy_filter", None), "energy_slit", None)
    if slit is None:
        return {}
    return {"energy_filter_slit": {
        "gap_m": _number(slit.gap_m, "gap_m"),
        "centre_m": _number(slit.centre_m, "centre_m"),
    }}


def energy_filter_origin_mm(by_key):
    for key in ("energy_filter", "energy_filter_entrance_aperture", "energy_filter_tapered_prism"):
        if key in by_key:
            return _number(by_key[key]["local_center_z_mm"], "branch entrance datum")
    raise ValueError("Energy Filter requires its branch entrance datum")


def _positive(part, field):
    value = _number(part[field], field)
    if value <= 0:
        raise ValueError(f"{part['key']}: {field} must be positive")
    return value


def _sector(by_key):
    prism = by_key["energy_filter_tapered_prism"]
    radius = _positive(prism, "prism_radius_mm")
    angle = math.radians(_positive(prism, "bend_angle_deg"))
    if angle >= math.pi:
        raise ValueError("Energy Filter bend must be less than 180 degrees")
    entrance = _number(prism["path_entrance_mm"], "path_entrance_mm")
    if entrance < 0 or prism.get("path_reference") != "branch_entrance":
        raise ValueError("Prism entrance must follow the branch entrance")
    origin = np.array((entrance, 0., 0.))
    exit_point = origin + (radius * math.sin(angle), 0., -radius * (1. - math.cos(angle)))
    tangent = np.array((math.cos(angle), 0., -math.sin(angle)))
    return prism, radius, angle, origin, exit_point, tangent


def energy_filter_frame_mm(part, by_key):
    """Return origin and proper transverse-X/Y/longitudinal frame in mm."""
    distance = _number(part["path_center_mm"], "path_center_mm")
    if distance < 0:
        raise ValueError("Energy Filter path coordinate must be nonnegative")
    reference = part.get("path_reference")
    if reference == "branch_entrance":
        origin, tangent = np.array((distance, 0., 0.)), np.array((1., 0., 0.))
    elif reference == "prism_exit":
        *_, exit_point, tangent = _sector(by_key)
        origin = exit_point + distance * tangent
    else:
        raise ValueError("Unknown Energy Filter path reference")
    transverse_y = np.array((0., 1., 0.))
    return origin, np.column_stack((np.cross(transverse_y, tangent), transverse_y, tangent))


def energy_filter_component_pose(part, by_key):
    """Local CAD axes are transverse X/Y and longitudinal Z at this part."""
    if part["key"] == "energy_filter_entrance_aperture":
        return np.zeros(3), np.asarray(FILTER_TO_COLUMN_ROTATION).T
    if part.get("mechanical_profile") == "tapered_sector_prism":
        _, _, _, origin, _, _ = _sector(by_key)
        return origin, np.asarray(FILTER_TO_COLUMN_ROTATION).T
    return energy_filter_frame_mm(part, by_key)


def energy_filter_feature_hit(part, by_key, point, normal):
    """Return a picked branch point/normal in the existing CAD schema axes."""
    origin, rotation = energy_filter_component_pose(part, by_key)
    center = np.array((0., 0., _number(part["local_center_z_mm"], "local_center_z_mm")))
    return rotation.T @ (np.asarray(point, dtype=float) - origin) + center, rotation.T @ np.asarray(normal, dtype=float)


def rigid_mesh(mesh, rotation, translation):
    """Transform surfaces and selectable edges/normals without mutating input."""
    rotation, translation = np.asarray(rotation, dtype=float), np.asarray(translation, dtype=float)
    if (rotation.shape != (3, 3) or translation.shape != (3,)
            or not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation))
            or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-12)
            or not math.isclose(float(np.linalg.det(rotation)), 1., rel_tol=0, abs_tol=1e-12)):
        raise ValueError("Energy Filter mount must be a finite right-handed rigid transform")
    vertices = np.asarray(mesh.vertices) @ rotation.T + translation
    vertices.setflags(write=False)
    edges = []
    for edge in mesh.edges:
        points = np.asarray(edge["vertices"]) @ rotation.T + translation
        points.setflags(write=False)
        edges.append({**edge, "vertices": points})
    surfaces = {key: {**value, **({"normal": tuple(rotation @ value["normal"])}
                                  if "normal" in value else {})}
                for key, value in mesh.surfaces.items()}
    return replace(mesh, vertices=vertices, surfaces=surfaces, edges=tuple(edges))


def mount_energy_filter_mesh(mesh, entrance_z_mm):
    return rigid_mesh(mesh, FILTER_TO_COLUMN_ROTATION,
                      (0., 0., _number(entrance_z_mm, "resolved branch entrance Z")))


def _paths(part, fields):
    return tuple(("parts", part["key"], name) for name in fields if name in part)


def _surface(part, region, vertices, faces, fields, description, *, wireframe=False, color=None):
    mesh = _mesh(vertices, faces, part["key"], region,
                 str(part.get("material_class", "Unspecified")), False, description)
    paths = _paths(part, fields)
    groups = np.full(len(mesh.faces), region)
    groups.setflags(write=False)
    metadata = {"label": description, "kind": "reference" if wireframe else "surface", "open_surface": True,
                "parameter_paths": paths}
    triangles = mesh.vertices[mesh.faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    if np.allclose(normals, normals[0], rtol=0, atol=1e-12):
        metadata["normal"] = tuple(normals[0])
    counts = Counter(tuple(sorted((int(a), int(b)))) for face in mesh.faces
                     for a, b in zip(face, np.roll(face, -1)))
    edges = []
    for index, (a, b) in enumerate(pair for pair, n in sorted(counts.items()) if n == 1):
        points = mesh.vertices[[a, b]]
        points.setflags(write=False)
        edges.append(dict(id=f"{region}:boundary:{index}", label=description,
                          vertices=points, surface_ids=(region,), parameter_paths=paths))
    return replace(mesh, face_groups=groups, surfaces={region: metadata}, edges=tuple(edges),
                   wireframe=wireframe, color=color or mesh.color)


def _quad(part, region, vertices, fields, description, **kwargs):
    return _surface(part, region, vertices, ((0, 1, 2), (0, 2, 3)), fields, description, **kwargs)


def _strip_faces(count):
    return [(2 * i, 2 * i + 1, 2 * i + 3) for i in range(count - 1)] + [
        (2 * i, 2 * i + 3, 2 * i + 2) for i in range(count - 1)]


def _prism_meshes(part, by_key, count):
    _, radius, angle, origin, exit_point, tangent = _sector(by_key)
    gap, half_width = _positive(part, "pole_gap_mm"), _positive(part, "radial_clear_half_width_mm")
    if half_width >= radius:
        raise ValueError("Prism radial channel must not cross its curvature centre")
    angles = np.linspace(0., angle, max(3, math.ceil(count * angle / (2 * math.pi)) + 1))
    center = origin + (0., 0., -radius)
    point = lambda r, y, a: center + (r * np.sin(a), y, r * np.cos(a))
    fields = ("prism_radius_mm", "bend_angle_deg", "path_entrance_mm", "pole_gap_mm", "radial_clear_half_width_mm")
    meshes = []
    for label, y in (("lower_pole_boundary", -gap / 2), ("upper_pole_boundary", gap / 2)):
        vertices = [point(r, y, a) for a in angles for r in (radius - half_width, radius + half_width)]
        faces = _strip_faces(len(angles))
        if y > 0:
            faces = [face[::-1] for face in faces]
        meshes.append(_surface(part, label, vertices, faces, fields,
                               "Configured pole-gap boundary; pole thickness and external yoke unknown",
                               color=(.50, .62, .76, .65)))
    for label, r in (("inner_radial_channel", radius - half_width), ("outer_radial_channel", radius + half_width)):
        vertices = [point(r, y, a) for a in angles for y in (-gap / 2, gap / 2)]
        meshes.append(_surface(part, label, vertices, _strip_faces(len(angles)), fields,
                               "Radial vacuum-channel boundary; no material wall inferred", wireframe=True))
    # One existing reference orbit, including the unmodelled gaps, explains the
    # path connection without manufacturing a vacuum tube or a filled volume.
    downstream = max((_number(row["path_center_mm"], "path_center_mm") for row in by_key.values()
                      if row.get("path_reference") == "prism_exit" and "path_center_mm" in row), default=0.)
    path = np.vstack(((0., 0., 0.), [point(radius, 0., a) for a in angles], exit_point + tangent * downstream))
    path.setflags(write=False)
    edge = dict(id="reference_centerline", label="Configured branch reference path; not material",
                vertices=path, surface_ids=(), parameter_paths=_paths(part, fields[:3]))
    meshes[2] = replace(meshes[2], edges=meshes[2].edges + (edge,))
    return tuple(meshes)


def slit_opening_mm(part, runtime=None):
    """Use supplied physical operating gap, otherwise show the travel limit."""
    if runtime is not None and not isinstance(runtime, Mapping):
        raise ValueError("Slit operating values must be a table")
    runtime = runtime or {}
    maximum = _positive(part, "maximum_gap_mm")
    gap = _number(runtime["gap_m"], "gap_m") * 1000 if "gap_m" in runtime else maximum
    center = _number(runtime.get("centre_m", 0.), "centre_m") * 1000
    if not 0 <= gap <= maximum:
        raise ValueError("Slit operating gap exceeds configured mechanical travel")
    return gap, center


def slit_render_dependency(part, runtime=None):
    opening = slit_opening_mm(part, runtime)
    # A supplied value additionally owns its selectable parameter path, even
    # when its numerical value happens to equal the offline travel limit.
    return opening, tuple(key for key in ("gap_m", "centre_m") if key in (runtime or {}))


def _slit_meshes(part, runtime):
    gap, center = slit_opening_mm(part, runtime)
    height, thickness = _positive(part, "clear_height_mm"), _positive(part, "blade_thickness_mm")
    fields = ("maximum_gap_mm", "clear_height_mm", "blade_thickness_mm", "path_center_mm")
    meshes = []
    for label, x in (("lower_blade_edge", center - gap / 2), ("upper_blade_edge", center + gap / 2)):
        vertices = ((x, -height/2, -thickness/2), (x, height/2, -thickness/2),
                    (x, height/2, thickness/2), (x, -height/2, thickness/2))
        if label == "upper_blade_edge":
            vertices = vertices[::-1]
        mesh = _quad(part, label, vertices, fields,
                     f"Two-blade slit inner edge; gap {gap:g} mm; outer blade extent unknown",
                     color=(.70, .73, .78, 1.))
        if runtime and ("gap_m" in runtime or "centre_m" in runtime):
            extra = tuple(("runtime", part["key"], key) for key in ("gap_m", "centre_m") if key in runtime)
            metadata = {key: {**value, "parameter_paths": value["parameter_paths"] + extra}
                        for key, value in mesh.surfaces.items()}
            mesh = replace(mesh, surfaces=metadata,
                           edges=tuple({**edge, "parameter_paths": edge["parameter_paths"] + extra} for edge in mesh.edges))
        meshes.append(mesh)
    return tuple(meshes)


def _annular_envelope(part, count):
    from temsim.part_model_features import annotate_legacy_mesh
    if part["mechanical_profile"] == "energy_filter_multipole_carrier":
        inner, outer = _positive(part, "mechanical_bore_radius_mm"), _positive(part, "mechanical_outer_radius_mm")
    else:
        inner, outer = _positive(part, "clear_bore_diameter_mm") / 2, _positive(part, "mechanical_outer_diameter_mm") / 2
    length = _positive(part, "housing_length_mm")
    mesh = revolve_section(_annulus(-length/2, length/2, inner, outer), key=part["key"], region="housing_envelope",
                           angular_segments=count, is_exact=False,
                           description="Configured housing envelope with open bore; internal electrodes/poles not dimensioned")
    local = {**part, "local_start_z_mm": -length/2, "local_end_z_mm": length/2, "local_center_z_mm": 0.}
    mesh = annotate_legacy_mesh(mesh, local)
    # Zero main-column packing length must never own these longitudinal faces.
    surfaces = {key: {**value, "parameter_paths": tuple(path for path in value["parameter_paths"] if path[-1] != "length_mm")}
                for key, value in mesh.surfaces.items()}
    for key, value in surfaces.items():
        if value.get("kind") == "cap":
            value["parameter_paths"] += _paths(part, ("housing_length_mm", "path_center_mm"))
    from temsim.part_model_features import _decorate
    return (_decorate(mesh, mesh.face_groups, surfaces),)


def _deflection_guides(part, count):
    length, gap = _positive(part, "electrode_length_mm"), _positive(part, "electrode_gap_mm")
    radius = _positive(part, "mechanical_outer_diameter_mm") / 2
    if gap >= radius * 2:
        raise ValueError("Electrode gap must fit the declared outer envelope")
    fields = ("electrode_length_mm", "electrode_gap_mm", "mechanical_outer_diameter_mm", "path_center_mm")
    angles = np.linspace(0., 2 * math.pi, count + 1)
    vertices = [(radius * np.cos(a), radius * np.sin(a), z) for a in angles for z in (-length/2, length/2)]
    meshes = [_surface(part, "outer_envelope", vertices, _strip_faces(len(angles)), fields,
                       "Provisional outer envelope only; electrode thickness and profile unknown", wireframe=True)]
    # The chord is only a clipping extent for a gap boundary guide, not a
    # measured electrode width or an electrode solid.
    half_width = math.sqrt(radius**2 - (gap/2)**2)
    for label, y in (("negative_gap_boundary", -gap/2), ("positive_gap_boundary", gap/2)):
        meshes.append(_quad(part, label, ((-half_width, y, -length/2), (half_width, y, -length/2),
                                         (half_width, y, length/2), (-half_width, y, length/2)), fields,
                            "Electrode gap boundary clipped to envelope; not electrode material", wireframe=True))
    return tuple(meshes)


def _zebra_meshes(part):
    count = part["strip_count"]
    if isinstance(count, bool) or not isinstance(count, Integral) or not 1 <= count <= 256:
        raise ValueError("Zebra strip count must be a positive bounded integer")
    width, height = _positive(part, "strip_active_width_mm"), _positive(part, "strip_active_height_mm")
    pitch = _positive(part, "provisional_strip_center_pitch_mm")
    if pitch < height:
        raise ValueError("Zebra provisional strip pitch causes overlap")
    fields = ("strip_active_width_mm", "strip_active_height_mm", "provisional_strip_center_pitch_mm", "path_center_mm")
    meshes = []
    for index in range(count):
        y = (index - (count - 1) / 2) * pitch
        meshes.append(_quad(part, f"active_strip_{index + 1}",
                            ((-width/2, y-height/2, 0.), (width/2, y-height/2, 0.),
                             (width/2, y+height/2, 0.), (-width/2, y+height/2, 0.)), fields,
                            "Spectrum active strip; strip pitch and height retain configuration provenance",
                            color=(.38, .76, .55, 1.)))
    width, height = _positive(part, "alignment_active_width_mm"), _positive(part, "alignment_active_height_mm")
    meshes.append(_quad(part, "alignment_active_area", ((-width/2, -height/2, 0.), (width/2, -height/2, 0.),
                                                       (width/2, height/2, 0.), (-width/2, height/2, 0.)),
                        ("alignment_active_width_mm", "alignment_active_height_mm", "path_center_mm"),
                        "Alignment active-area outline at common readout datum; relative sensor layout unknown",
                        wireframe=True, color=(.95, .72, .35, 1.)))
    return tuple(meshes)


def energy_filter_meshes(part, by_key, count, runtime=None, aperture_index=0):
    """Shared part/assembly adapter; output is in branch Cartesian mm."""
    if part["key"] == "energy_filter_entrance_aperture":
        from temsim.part_model_apertures import strip_meshes
        meshes, notes = strip_meshes(part, count, runtime, aperture_index)
        # The existing strip is expressed in column-local XYZ. Undo the source
        # datum and inverse-mount it, preserving the current entrance aperture
        # and historical entrance-ray transverse signs in the main assembly.
        inverse = np.asarray(FILTER_TO_COLUMN_ROTATION).T
        datum = energy_filter_origin_mm(by_key)
        return tuple(rigid_mesh(mesh, inverse, inverse @ (0., 0., -datum)) for mesh in meshes), notes
    if not math.isclose(_number(part["local_center_z_mm"], "branch axial datum"),
                        energy_filter_origin_mm(by_key), rel_tol=0, abs_tol=1e-8):
        raise ValueError("Branch-path component must share the Energy Filter entrance datum")
    profile = part["mechanical_profile"]
    if profile == "tapered_sector_prism":
        meshes = _prism_meshes(part, by_key, count)
    else:
        origin, rotation = energy_filter_frame_mm(part, by_key)
        if profile in {"energy_filter_multipole_carrier", "electrostatic_quadrupole", "electrostatic_bias_tube"}:
            meshes = _annular_envelope(part, count)
        elif profile == "xo_energy_slit_assembly":
            meshes = _slit_meshes(part, runtime)
        elif profile in {"fast_electrostatic_shutter", "electrostatic_camera_deflector"}:
            meshes = _deflection_guides(part, count)
        elif profile == "zebra_eels_detector":
            meshes = _zebra_meshes(part)
        else:
            raise ValueError("Unsupported Energy Filter geometry")
        meshes = tuple(rigid_mesh(mesh, rotation, origin) for mesh in meshes)
    notes = [f"{part['key']}: {mesh.description}." for mesh in meshes]
    if profile == "xo_energy_slit_assembly" and (not runtime or "gap_m" not in runtime):
        notes.append(f"{part['key']}: no runtime slit gap supplied; blade edges show the configured maximum travel opening.")
    notes.append(f"{part['key']}: branch path positions and unpublished mechanical dimensions are provisional, not OEM measurements.")
    return meshes, tuple(dict.fromkeys(notes))
