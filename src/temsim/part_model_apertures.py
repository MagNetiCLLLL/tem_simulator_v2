"""Thin aperture-plate previews, separate from their mechanism envelopes.

The existing manifests identify perforated strips but do not dimension their
transverse outlines. Reuse the 2-D schematic reach only for a labelled preview;
only the configured thickness and an explicitly supplied opening are physical
dimensions. No carrier bore is substituted for an unknown working aperture.
"""

from collections.abc import Mapping

import numpy as np


def is_strip_aperture(part):
    return part.get("aperture_plate_form") == "perforated_strip"


def plate_thickness_field(part):
    return "plate_thickness_mm" if "plate_thickness_mm" in part else "active_length_mm"


def dimension_semantics(part):
    if not is_strip_aperture(part):
        return {}
    return {
        "length_mm": ("Mechanism envelope length", "Axial space allocated to the aperture mechanism, not the thin plate thickness."),
        "mechanical_outer_diameter_mm": ("Mechanism envelope diameter", "Outer envelope of the aperture mechanism, not a measured strip width or plate diameter."),
        "outer_diameter_mm": ("Mechanism envelope diameter", "Outer envelope of the aperture mechanism, not a measured strip width or plate diameter."),
        "mechanical_bore_diameter_mm": ("Carrier bore diameter", "Clear bore of the carrier, separate from the smaller operating aperture opening."),
        "bore_diameter_mm": ("Carrier bore diameter", "Clear bore of the carrier, separate from the operating aperture opening."),
        "vacuum_inner_diameter_mm": ("Beam passage (vacuum)", "Shared vacuum-passage constraint. This is neither the working aperture opening nor the plate's width."),
        plate_thickness_field(part): ("Aperture plate thickness", "Configured physical plate thickness along Z; the 3D preview uses this value at the optical reference plane."),
        "maximum_radius_mm": ("Maximum working opening radius", "Operating aperture radius limit; not the plate's outer radius."),
    }


def _opening(part, runtime, aperture_index=0):
    from temsim.part_model_3d import _number
    # Explicit hole arrays retain their user-selected index, where available.
    for field in ("aperture_hole_diameters_mm", "aperture_hole_diameters_um",
                  "hole_diameters_mm", "hole_diameters_um"):
        if field in part:
            from numbers import Integral
            holes = part[field]
            if (not isinstance(holes, (list, tuple)) or not holes
                    or isinstance(aperture_index, bool) or not isinstance(aperture_index, Integral)
                    or not 0 <= aperture_index < len(holes)):
                raise ValueError(f"{part['key']}: aperture_index must select an existing aperture")
            diameter = _number(holes[aperture_index], field) * (0.001 if field.endswith("_um") else 1)
            if diameter <= 0:
                raise ValueError("The selected aperture diameter must be positive")
            return diameter, (("parts", part["key"], field, aperture_index),), (0., 0.)
    for source, prefix in ((runtime or {}, "runtime"), (part, "parts")):
        if not isinstance(source, Mapping):
            raise ValueError("Aperture operating values must be a table")
        for field, factor in (("diameter_mm", 1), ("opening_diameter_mm", 1),
                              ("radius_mm", 2), ("aperture_radius_mm", 2)):
            if field in source:
                diameter = _number(source[field], field) * factor
                if diameter < 0:
                    raise ValueError("Working aperture diameter must be nonnegative")
                paths = [(prefix, part["key"], field)]
                offsets = []
                for name in ("offset_x_mm", "offset_y_mm"):
                    offsets.append(_number(source.get(name, 0.), name))
                    if name in source:
                        paths.append((prefix, part["key"], name))
                return diameter, tuple(paths), tuple(offsets)
    return None, (), (0., 0.)


def operating_dimension_specs(part, runtime):
    if not is_strip_aperture(part):
        return ()
    from temsim.part_model_3d import DimensionSpec
    diameter, paths, _ = _opening(part, runtime)
    if not paths or paths[0][0] != "runtime":
        return ()
    return (DimensionSpec(paths[0], "Working opening diameter", diameter, "mm", False,
                          "Current operating opening (diameter = 2 × radius). Change it with the aperture's Opening diameter control; it is separate from the carrier bore and is not saved as a mechanical dimension."),)


def strip_meshes(part, count, runtime=None, aperture_index=0):
    from temsim.part_model_3d import _mesh, _number
    from temsim.part_model_features import _decorate, _manifold_module, _normal_data

    thickness_field = plate_thickness_field(part)
    if thickness_field not in part:
        raise ValueError(f"{part['key']}: aperture plate thickness is not defined")
    thickness = _number(part[thickness_field], thickness_field)
    if thickness <= 0:
        raise ValueError("Aperture plate thickness must be positive")
    center = _number(part.get("optical_reference_local_z_mm", part.get("local_center_z_mm")), "aperture center")
    diameter, opening_paths, (dx, dy) = _opening(part, runtime, aperture_index)
    # These are the same schematic reach rules as the 2-D physical layout.
    vacuum = .5 * _number(part.get("vacuum_inner_diameter_mm", 0.), "vacuum diameter")
    carrier = .5 * _number(part.get("mechanical_bore_diameter_mm", part.get("bore_diameter_mm", 0.)), "carrier bore diameter")
    envelope = .5 * _number(part.get("mechanical_outer_diameter_mm", part.get("outer_diameter_mm", 0.)), "mechanism envelope diameter")
    if min(vacuum, carrier, envelope) < 0:
        raise ValueError("Aperture clearances and envelope must be nonnegative")
    envelope = max(envelope, vacuum + 1.)
    margin = max(.75, .15 * max(vacuum, 1.))
    reach = max(vacuum + margin, carrier + margin)
    joint = max(reach, min(.60 * envelope, vacuum + 12.))
    # The outline remains schematic, including when the operating hole moves.
    radius = .5 * (diameter or 0.)
    half_width = max(reach, abs(dx) + radius + margin)
    lower, upper = min(-reach, dy - radius - margin), max(joint, dy + radius + margin)
    m = _manifold_module()
    solid = m.Manifold.cube((2 * half_width, upper - lower, thickness), center=True).translate((0., .5 * (lower + upper), 0.))
    if radius > 0:
        cutter = m.Manifold.cylinder(2 * thickness, radius, circular_segments=count, center=True).translate((dx, dy, 0.))
        solid = solid - cutter
    if solid.status() != m.Error.NoError or solid.is_empty():
        raise ValueError(f"{part['key']}: aperture preview could not produce a valid thin plate")
    raw = solid.to_mesh64()
    vertices = np.asarray(raw.vert_properties)[:, :3].copy()
    vertices[:, 2] += center
    opening_note = ("working opening is not specified in this file" if diameter is None else
                    f"working opening diameter {diameter:g} mm" + (" (runtime)" if opening_paths[0][0] == "runtime" else ""))
    description = f"Schematic aperture strip; configured thickness {thickness:g} mm; {opening_note}"
    if opening_paths and len(opening_paths[0]) == 4:
        description += f"; selected existing aperture index={aperture_index}; multi-hole positions are not defined"
    mesh = _mesh(vertices, raw.tri_verts, part["key"], "body",
                 str(part.get("material_class", part.get("aperture_plate_material", "Unspecified"))), False, description)
    centers, normals = _normal_data(mesh.vertices, mesh.faces)
    cap = np.abs(normals[:, 2]) > 1 - 1e-9
    hole = np.einsum("ij,ij->i", centers[:, :2] - [dx, dy], normals[:, :2]) < 0
    groups = np.where(cap, np.where(normals[:, 2] > 0, "plate_positive", "plate_negative"),
                      np.where(hole, "working_opening", "schematic_outline"))
    thickness_path = (("parts", part["key"], thickness_field),)
    metadata = {
        "plate_positive": {"label": "Aperture plate face (+Z)", "kind": "cap", "parameter_paths": (*thickness_path, *opening_paths)},
        "plate_negative": {"label": "Aperture plate face (-Z)", "kind": "cap", "parameter_paths": (*thickness_path, *opening_paths)},
        "working_opening": {"label": "Working aperture opening", "kind": "inner", "parameter_paths": (*opening_paths, *thickness_path)},
        "schematic_outline": {"label": "Schematic strip edge (transverse dimensions unspecified)", "kind": "outer", "parameter_paths": thickness_path},
    }
    notes = (description + ".",
             "Transverse strip dimensions are not specified; outline is schematic. "
             "Mechanism envelope length/diameter and carrier bore do not define the thin plate's outer shape or working opening.")
    return (_decorate(mesh, groups, metadata),), notes
