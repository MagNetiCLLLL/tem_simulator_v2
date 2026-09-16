"""Physical tip surfaces with an explicitly bounded, selectable emitting cap."""
from dataclasses import replace
import math
import numpy as np


def tip_dimension_overrides(part, specs, runtime):
    """Expose applied geometry beside saved defaults when they differ."""
    from temsim.tip_emission_view import emission_dimensions
    dimensions = emission_dimensions(part, runtime)
    if dimensions is not None:
        from temsim.part_model_3d import DimensionSpec
        from temsim.optics.electron_gun.tip_assembly import PART_FIELDS
        specs = tuple(replace(spec, label="Reference: " + spec.label,
            reason="Saved tip definition. The active analytic emission uses the operating curvature and source FWHM below.")
            if spec.path[0] == "parts" and spec.path[2] in PART_FIELDS else spec for spec in specs)
        active = tuple(DimensionSpec((kind, part["key"], name), label, dimensions[name], unit,
            editable=False, reason="Current launch surface. Edit Tip model / emission or the operating controls.")
            for kind, name, label, unit in (
                ("runtime", "curvature_nm_inv", "Active emission curvature", "nm⁻¹"),
                ("derived", "emission_radius_nm", "Emission radius (∞ = flat)", "nm"),
                ("runtime", "virtual_source_fwhm_nm", "Projected source FWHM", "nm"),
                ("derived", "emission_support_diameter_nm", "Emission support diameter", "nm"),
                ("derived", "emission_depth_nm", "Emission edge depth", "nm")))
        return active + specs
    model = (runtime or {}).get("tip_surface_model")
    if not model or not part.get("tip_particle_model"):
        return specs
    from temsim.part_model_3d import DimensionSpec
    fields = {"tip_radius_nm": ("geometry", "apex_radius_nm", 1.),
              "tip_cone_half_angle_deg": ("geometry", "cone_half_angle_deg", 1.),
              "length_mm": ("geometry", "shank_length_um", .001),
              "emission_cap_half_angle_deg": ("emission", "cap_half_angle_deg", 1.),
              "emission_maximum_angle_deg": ("emission", "maximum_angle_deg", 1.)}
    result = []
    for spec in specs:
        entry = fields.get(spec.path[-1])
        if entry is not None:
            section, name, scale = entry
            active = model[section][name]*scale
            if active != spec.value:
                result.append(replace(spec, label="Saved default: " + spec.label))
                result.append(DimensionSpec(("runtime", part["key"], "tip_surface_model", section, name),
                    "Active: " + spec.label, active, spec.unit, editable=False,
                    reason="Applied tip override used by this view and the next calculation. Edit Tip model / emission, or save the geometry draft to apply TOML defaults."))
                continue
        result.append(spec)
    return tuple(result)


def tip_meshes(part, count, runtime=None):
    from temsim.optics.electron_gun.tip_assembly import model_from_part, tip_apex_z_mm
    from temsim.optics.electron_gun.tip_surface import TipSurfaceModel
    from temsim.optics.electron_gun.tip_patch import patch_dimensions
    from temsim.part_model_3d import revolve_section
    from temsim.part_model_features import annotate_legacy_mesh
    model = model_from_part(part)
    active = not part.get("mechanical_only", False)
    if runtime is not None and "tip_surface_model" in runtime:
        active = runtime["tip_surface_model"] is not None
        if active:
            model = TipSurfaceModel.from_dict(runtime["tip_surface_model"])
    g, e = model.geometry, model.emission
    radius = g.apex_radius_nm*1e-6
    apex = tip_apex_z_mm(part)
    cap = math.radians(e.cap_half_angle_deg)
    theta = np.unique(np.r_[np.linspace(0, math.pi/2-math.radians(g.cone_half_angle_deg), 33),
                            np.linspace(0, cap, 17)])
    section = [(apex-2*radius*math.sin(t/2)**2, radius*math.sin(t)) for t in theta]
    length = g.shank_length_um*.001
    section += [(apex-length, float(g.radius_m(-length*.001)*1000)), (apex-length, 0.)]
    whole = revolve_section(section, key=part["key"], angular_segments=count,
                            material_class=g.material, description="Physical spherical apex and tangent cone")
    if not active:
        from temsim.tip_emission_view import emission_note
        note = emission_note(part, runtime)
        return (whole,), ((note, "Reference body from TOML; independent of the active analytic emission.") if note else
                         ("Physical tip geometry. Curved-surface emission is inactive; this view does not define a new source.",))
    boundary = apex-2*radius*math.sin(cap/2)**2
    mask = np.min(whole.vertices[whole.faces, 2], axis=1) >= boundary-4*np.spacing(max(abs(apex), radius))
    # Keep one watertight solid for copying, export and Boolean operations.
    # Emission is face provenance/colour, not a second open material body.
    whole = annotate_legacy_mesh(whole, part, {})
    groups = whole.face_groups.astype(object).copy()
    groups[mask] = "emitting_cap"
    groups.setflags(write=False)
    paths = tuple(("parts", part["key"], field) for field in ("tip_radius_nm", "emission_cap_half_angle_deg"))
    paths += (("runtime", part["key"], "tip_surface_model", "geometry", "apex_radius_nm"),
              ("runtime", part["key"], "tip_surface_model", "emission", "cap_half_angle_deg"))
    phi = np.linspace(0, 2*np.pi, count+1)
    edge = np.column_stack((radius*math.sin(cap)*np.cos(phi), radius*math.sin(cap)*np.sin(phi), np.full(phi.size, boundary)))
    whole = replace(whole, face_groups=groups,
        surfaces={**whole.surfaces, "emitting_cap": {"label": "Emitting apex surface", "kind": "emission",
            "parameter_paths": paths, "color": (1., .72, .12, 1.)}},
        edges=(*whole.edges, {"id": "emission_boundary", "vertices": edge, "label": "Emission boundary", "parameter_paths": paths}))
    dims = patch_dimensions(g, e.cap_half_angle_deg)
    note = (f"Gold: emitting apex cap only · diameter {dims['projected_diameter_nm']:.5g} nm · "
            f"depth {dims['cap_depth_nm']:.5g} nm · area {dims['surface_area_nm2']:.5g} nm². "
            "The remaining tip does not emit. Cap angle and electron emission angle are distinct.")
    return (whole,), (note,)
