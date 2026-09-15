"""Physical tip definitions owned by the same TOML part as its placement.

Flat scalar fields are editable by the existing assembly/3D editor. The
shared source definition is resolved before geometry and emission are applied.
"""
import math
from numbers import Real

from .tip_surface import TipSurfaceModel, TipGeometry, SurfaceEmission, TipFieldNumerics

MODEL = "curved_surface_particles"
GEOMETRY_FIELDS = {"tip_shape", "tip_material", "tip_radius_nm", "tip_cone_half_angle_deg", "length_mm"}
EMISSION_FIELDS = {
    "emission_flux_electrons_per_nm2_s", "emission_cap_half_angle_deg",
    "emission_maximum_angle_deg", "emission_energy_distribution",
    "emission_normal_mean_energy_ev", "emission_tangential_mean_energy_ev",
    "emission_kinetic_mean_ev", "emission_kinetic_sigma_ev",
}
NUMERICAL_FIELDS = {
    "tip_field_radial_nodes", "tip_field_axial_nodes", "tip_apex_cells_per_radius",
    "tip_field_outer_radius_factor", "tip_field_residual_tolerance",
}
OPTIONAL_NUMERICAL_FIELDS = {"emission_directions_per_position",
    "tip_field_axis_core_fraction", "tip_field_electrode_cells_per_bore", "tip_field_electrode_corner_cells"}
REQUIRED_PART_FIELDS = GEOMETRY_FIELDS | EMISSION_FIELDS | NUMERICAL_FIELDS | {"tip_particle_model"}
PART_FIELDS = REQUIRED_PART_FIELDS | OPTIONAL_NUMERICAL_FIELDS


def is_tip_part(part):
    return part.get("tip_particle_model") == MODEL


def model_from_part(part):
    if "tip_particle_model" not in part:
        return None  # historical assembly, no implicit conversion
    if not is_tip_part(part):
        raise ValueError("Unknown TOML tip particle model")
    missing = REQUIRED_PART_FIELDS - part.keys()
    if missing:
        raise ValueError("Missing tip assembly fields: " + ", ".join(sorted(missing)))
    for key in (PART_FIELDS & part.keys()) - {"tip_particle_model", "tip_shape", "tip_material", "emission_energy_distribution"}:
        value = part[key]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"Tip {key} must be a finite number")
    model = TipSurfaceModel(
        geometry=TipGeometry(part["tip_shape"], part["tip_material"], part["tip_radius_nm"],
                             part["tip_cone_half_angle_deg"], part["length_mm"]*1000),
        emission=SurfaceEmission(None, part["emission_cap_half_angle_deg"],
            part["emission_normal_mean_energy_ev"], part["emission_tangential_mean_energy_ev"],
            part["emission_flux_electrons_per_nm2_s"], part["emission_maximum_angle_deg"],
            part["emission_energy_distribution"], part["emission_kinetic_mean_ev"], part["emission_kinetic_sigma_ev"],
            directions_per_position=part.get("emission_directions_per_position", 1)),
        field_numerics=TipFieldNumerics(part["tip_field_radial_nodes"], part["tip_field_axial_nodes"],
            part["tip_apex_cells_per_radius"], part["tip_field_outer_radius_factor"],
            part["tip_field_residual_tolerance"], part.get("accelerator_ring_thickness_mm", 1.0),
            axis_core_fraction=part.get("tip_field_axis_core_fraction", .01),
            electrode_cells_per_bore=part.get("tip_field_electrode_cells_per_bore", 8),
            electrode_corner_cells=part.get("tip_field_electrode_corner_cells", 0)),
    ).validate()
    return model


def tip_apex_z_mm(part):
    # A copied solid is geometry only. It never creates another electron source.
    if part.get("mechanical_part_role") == "custom_mechanical_copy":
        return float(part["optical_reference_local_z_mm"])
    return 0.0


def derived_envelope(model, apex_z_mm=0.0):
    length = model.geometry.shank_length_um*.001
    return {"local_start_z_mm": apex_z_mm-length, "local_center_z_mm": apex_z_mm-.5*length,
            "local_end_z_mm": apex_z_mm, "optical_reference_local_z_mm": apex_z_mm,
            "outer_diameter_mm": float(2*model.geometry.radius_m(-length*.001)*1000)}


def validate_tip_part(part):
    model = model_from_part(part)
    if model is None:
        return
    for key, value in derived_envelope(model, tip_apex_z_mm(part)).items():
        if not math.isclose(float(part[key]), value, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f"{part['key']}.{key} must follow the tip shape ({value:g}); edit length, curvature radius or cone angle. The apex is the gun origin.")
    if part.get("model_3d"):
        raise ValueError("Tip particle geometry supports its declared cap/cone shape; independent CAD overrides are not consumed by the extraction field")


def apply_tip_part(emitter, part):
    model = model_from_part(part)
    if model is None:
        return
    signature = tuple((key, part.get(key)) for key in sorted(PART_FIELDS | OPTIONAL_NUMERICAL_FIELDS))
    previous = getattr(emitter, "_tip_assembly_signature", None)
    if previous != signature:
        # Reloading dimensions must not activate a different source family in
        # an explicit historical planar or coherent profile.
        if previous is not None and (emitter.surface_model is None
                or emitter.surface_model.coherence is not None or emitter.coherence is not None):
            emitter._tip_assembly_signature = signature
            return
        emitter.surface_model = model
        emitter.coherence = None
        emitter.tip_radius_nm = model.geometry.apex_radius_nm
        emitter.tip_cone_half_angle_deg = model.geometry.cone_half_angle_deg
        emitter.emitter_material = model.geometry.material
        emitter._tip_assembly_signature = signature


def complete_tip_updates(document, updates):
    """Keep the apex fixed and update dependent mechanical extents, atomically."""
    result = dict(updates)
    for original in document.get("parts", ()):
        if not is_tip_part(original):
            continue
        prefix = ("parts", original["key"])
        if not any(path[:2] == prefix for path in updates):
            continue
        part = dict(original)
        part.update({path[2]: value for path, value in updates.items() if path[:2] == prefix and len(path) == 3})
        model = model_from_part(part)
        if any(prefix+(key,) in updates for key in GEOMETRY_FIELDS):
            for key, value in derived_envelope(model, tip_apex_z_mm(original)).items():
                if prefix+(key,) in updates and not math.isclose(float(updates[prefix+(key,)]), value, rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError(f"Tip {key} is derived from its cap/cone geometry")
                result[prefix+(key,)] = value
    return result


def validate_electrical_defaults(parts):
    by_key = {part["key"]: part for part in parts}
    for part in parts:
        if "default_voltage_reference" in part and part["default_voltage_reference"] not in {"tip", "extractor", "ground"}:
            raise ValueError("Gun-lens voltage reference must be tip, extractor or ground")
        if "electrode_thickness_mm" in part:
            value = part["electrode_thickness_mm"]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("Accelerator electrode thickness must be finite and positive")
            centers = part.get("stage_centers_z_mm", ())
            if centers and (min(centers)-value/2 < part["local_start_z_mm"]
                            or max(centers)+value/2 > part["local_end_z_mm"]
                            or any(b-a <= value for a, b in zip(centers, centers[1:]))):
                raise ValueError("Accelerator electrodes must fit their assembly and remain separated")
        for field in ("default_voltage_kv", "default_high_tension_kv"):
            if field not in part:
                continue
            value = part[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{part['key']}.{field} must be a finite number")
            if field == "default_high_tension_kv" and not 30 <= value <= 300:
                raise ValueError("Default accelerating voltage must be in [30, 300] kV")
            if part["key"] == "feg_extractor" and not 0 <= value <= 20:
                raise ValueError("Default extraction voltage must be in [0, 20] kV")
    if "feg_extractor" in by_key and "feg_accelerator" in by_key:
        ext = by_key["feg_extractor"].get("default_voltage_kv", 4.)
        ht = by_key["feg_accelerator"].get("default_high_tension_kv", 300.)
        if not ext < ht:
            raise ValueError("Extraction voltage must be smaller than the final accelerating voltage")
