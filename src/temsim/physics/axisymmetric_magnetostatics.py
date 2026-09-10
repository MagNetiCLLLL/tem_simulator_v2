"""Axisymmetric A-phi finite elements on authoritative R-Z geometry.

The weak form integrates nu * [(d_r A + A/r)(d_r v + v/r)
+ d_z A d_z v] r = J_phi v r. Linear triangular elements use three-point
quadrature. A=0 on the axis and remote boundary; domain and mesh convergence
are separate checks. Linear recipes use explicit constant permeability;
nonlinear recipes use sourced, single-valued B-H tables and a joint current
vector. Neither is an instrument-specific material calibration.
"""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math

import numpy as np

MU0 = 4e-7 * math.pi
SOLVER_VERSION = "axisymmetric-aphi-bh-material-regions-v4"


def _merged_axis(values):
    axis = np.unique(np.asarray(values, float))
    tolerance = max(float(np.ptp(axis)), 1e-12) * 1e-12
    return axis[np.r_[True, np.diff(axis) > tolerance]]


@dataclass(frozen=True)
class MagnetostaticSolution:
    r_m: np.ndarray
    z_m: np.ndarray
    a_phi_tm: np.ndarray
    br_t: np.ndarray
    bz_t: np.ndarray
    relative_residual: float
    iterations: int = 1
    peak_material_t: float = 0.0


def solve_axisymmetric(r_m, z_m, relative_permeability, current_density_a_m2):
    """Linear reference interface; cached mesh is shared with nonlinear FEM."""
    from temsim.physics.nonlinear_magnetostatics import solve_linear
    return solve_linear(r_m, z_m, relative_permeability, current_density_a_m2)


def _part_mask(part, r, z):
    """Positive-radius counterpart of the supported physical pole profiles."""
    data = part["data"]
    start, end = float(part["start_z_mm"]) * 1e-3, float(part["end_z_mm"]) * 1e-3
    from temsim.magnetic_circuits import radial_profile_mm
    profile = radial_profile_mm(data, (end-start)*1e3)
    if profile is not None:
        offset = (z-start)*1e3
        inner = np.interp(offset, profile[:, 0], profile[:, 1])*1e-3
        outer = np.interp(offset, profile[:, 0], profile[:, 2])*1e-3
        return (z >= start) & (z <= end) & (r >= inner) & (r <= outer)
    inner = float(data.get("mechanical_inner_diameter_mm", data.get("mechanical_bore_diameter_mm", 0))) * 0.5e-3
    outer = float(data["mechanical_outer_diameter_mm"]) * 0.5e-3
    outer_radius = np.full_like(r, outer)
    if data.get("mechanical_profile") == "magnetic_pole_piece":
        inner = float(data["mechanical_bore_diameter_mm"]) * 0.5e-3
        tip = float(data["mechanical_tip_diameter_mm"]) * 0.5e-3
        upper = "upper" in part["key"]
        distance = z - start if upper else end - z
        length = end - start
        nose = float(data.get("pole_nose_axial_length_mm", 0)) * 1e-3 or 0.38 * length
        outer_radius = np.minimum(outer_radius, tip + (outer - tip) * (length - distance) / nose)
        style = data.get("pole_piece_geometry_style", "")
        if style == "objective_vertical_back_inserted_shank_tapered_nose":
            shank = float(data["pole_mounting_shank_axial_length_mm"]) * 1e-3
            connector = float(data.get("pole_vacuum_connector_axial_length_mm", 0)) * 1e-3
            outer_radius = np.where(distance < shank, float(data["pole_stem_outer_diameter_mm"]) * 0.5e-3, outer_radius)
            outer_radius = np.where(distance < connector, float(data["pole_vacuum_connector_outer_diameter_mm"]) * 0.5e-3, outer_radius)
            inner = np.where(distance < shank, float(data["pole_mounting_shank_inner_diameter_mm"]) * 0.5e-3, inner)
        elif style not in {"", "embedded_hourglass_bore", "tapered_bore_pole"}:
            raise ValueError(f"Unsupported magnetostatic pole profile: {style}")
    intervals = data.get("material_intervals_mm", ((start*1e3, end*1e3),))
    axial = np.logical_or.reduce([(z >= a*1e-3) & (z <= b*1e-3) for a, b in intervals])
    return axial & (r >= inner) & (r <= outer_radius)


def _material_region_masks(part, assignments, r, z):
    """Partition an existing part mask; assignments never create new geometry."""
    whole = _part_mask(part, r, z)
    if not (set(assignments) - {"body"}):
        return ((whole, assignments.get("body")),)
    intervals = part["data"].get("material_intervals_mm", ())
    if len(intervals) != 2 or "magnetic_radial_profile_mm" in part["data"]:
        raise ValueError(f"{part['key']}: upper/lower materials require existing split material intervals")
    return tuple((whole & (z >= start*1e-3) & (z <= end*1e-3),
                  assignments.get(region, assignments.get("body")))
                 for region, (start, end) in zip(("upper", "lower"), intervals))


@lru_cache(maxsize=12)
def _solve_bound_geometry(geometry_json, settings_json):
    from temsim.physics.lens_field_provider import CoordinateRegistration, MagneticFieldMap, FieldMapProvenance
    geometry, settings = json.loads(geometry_json), json.loads(settings_json)
    topology = geometry["lens_assembly"].get("magnetic_circuit_topology", "")
    if topology == "monolithic_saturated_insert" and settings.get("solver") != "axisymmetric_nonlinear_fem":
        raise ValueError("Monolithic saturation inserts require a nonlinear B-H solve. The linear solver cannot model their operating principle; use a field map at a documented operating point")
    nonlinear = settings.get("solver") == "axisymmetric_nonlinear_fem"
    mur, turns = float(settings.get("relative_permeability", 1)), float(settings.get("ampere_turns", 0))
    nr, nz, padding = int(settings.get("radial_nodes", 40)), int(settings.get("axial_nodes", 80)), float(settings.get("padding_factor", 2))
    if not (mur > 0 and math.isfinite(mur) and math.isfinite(turns) and 12 <= nr <= 256 and 16 <= nz <= 512 and 1.5 <= padding <= 10):
        raise ValueError("Invalid permeability, ampere-turns, mesh or boundary padding")
    parts = geometry["lens_assembly"]["parts"]
    from temsim.magnetic_circuits import radial_profile_mm
    from temsim.part_materials import is_magnetostatic_body, validated_material_regions
    iron = [p for p in parts if is_magnetostatic_body(p["data"])]
    if topology == "air_core" and iron:
        raise ValueError("Air-core circuit still contains magnetic bodies; change the declared geometry first")
    neighbours = geometry["lens_assembly"].get("magnetostatic_neighbours", [])
    coils = [p for p in parts if p["data"].get("mechanical_profile") == "magnetic_excitation_coil"]
    driven = coils if nonlinear else [p for p in coils if p["data"].get("field_source_key", geometry.get("lens_key")) == geometry.get("lens_key")]
    if not driven:
        raise ValueError("The selected lens has no explicit excitation-coil geometry")
    fractions = []
    for coil in driven:
        owner = coil["data"].get("field_source_key", geometry.get("lens_key"))
        owned = [p for p in driven if p["data"].get("field_source_key", geometry.get("lens_key")) == owner]
        values = [p["data"].get("coil_ampere_turn_fraction") for p in owned]
        if all(value is None for value in values):
            fraction = 1/len(owned)
        elif (any(value is None for value in values)
              or not all(math.isfinite(float(value)) for value in values)
              or not math.isclose(sum(abs(float(value)) for value in values), 1, rel_tol=0, abs_tol=1e-10)):
            raise ValueError("Declare every coil ampere-turn fraction; absolute fractions must sum to one per channel")
        else:
            fraction = float(coil["data"]["coil_ampere_turn_fraction"])
        amplitude = float(settings["channel_ampere_turns"][owner]) if nonlinear else turns
        if not math.isfinite(amplitude):
            raise ValueError("Coil ampere-turns must be finite")
        fractions.append(fraction*amplitude)
    used = iron + coils
    radius = max(float(p["data"]["mechanical_outer_diameter_mm"]) * .5e-3 for p in used)
    low, high = min(p["start_z_mm"] for p in used) * 1e-3, max(p["end_z_mm"] for p in used) * 1e-3
    centre, half = (low + high) / 2, (high - low) / 2
    iron.extend(neighbours)
    assignments = {p["key"]: validated_material_regions(p["data"]) for p in iron + coils}
    # Add authoritative material boundaries to the grid; a narrow gap must not
    # vanish because the generic mesh happens to skip it.
    radial_edges = [0.0]
    axial_edges = []
    for p in used + list(neighbours):
        intervals = p["data"].get("material_intervals_mm", ((p["start_z_mm"], p["end_z_mm"]),))
        axial_edges.extend(value*1e-3 for interval in intervals for value in interval)
        radial_edges.extend(float(v) * .5e-3 for k, v in p["data"].items()
                            if k in {"mechanical_bore_diameter_mm", "mechanical_tip_diameter_mm", "mechanical_inner_diameter_mm", "mechanical_outer_diameter_mm"})
        profile = radial_profile_mm(p["data"], p["end_z_mm"]-p["start_z_mm"])
        if profile is not None:
            axial_edges.extend((profile[:, 0]+p["start_z_mm"])*1e-3)
            radial_edges.extend(profile[:, 1:].ravel()*1e-3)
    radial_edges = [value for value in radial_edges if 0 <= value <= radius*padding]
    axial_edges = [value for value in axial_edges if centre-half*padding <= value <= centre+half*padding]
    r = _merged_axis(np.r_[np.linspace(0, radius * padding, nr), radial_edges])
    z = _merged_axis(np.r_[np.linspace(centre - half * padding, centre + half * padding, nz), axial_edges])
    if "validation_grid_axes_m" in settings:
        # Validation alone may extend a domain without moving its interior mesh.
        supplied = settings["validation_grid_axes_m"]
        if not isinstance(supplied, list) or len(supplied) != 2:
            raise ValueError("Validation grid requires explicit R and Z axes")
        axes = tuple(np.asarray(axis, float) for axis in supplied)
        bounds = ((0., radius*padding), (centre-half*padding, centre+half*padding))
        for axis, bound in zip(axes, bounds):
            if (axis.ndim != 1 or not 4 <= axis.size <= 4096 or not np.all(np.isfinite(axis))
                    or np.any(np.diff(axis) <= 0) or not np.allclose(axis[[0, -1]], bound, rtol=1e-10, atol=1e-14)):
                raise ValueError("Validation grid must be increasing and match the physical solve domain")
        r, z = _merged_axis(np.r_[axes[0], radial_edges]), _merged_axis(np.r_[axes[1], axial_edges])
        if r.size*z.size > 250000:
            raise ValueError("Validation grid exceeds the 250000-node resource limit")
    def iron_mask(r, z):
        return np.logical_or.reduce([_part_mask(p, r, z) for p in iron]) if iron else np.zeros_like(r, bool)
    def permeability(r, z):
        result = np.ones_like(r)
        occupied = np.zeros_like(r, dtype=bool)
        material_values = settings.get("material_permeabilities", {})
        for part in iron:
            value = float(material_values.get(part["data"].get("material_class", ""), mur))
            for mask, assignment in _material_region_masks(part, assignments[part["key"]], r, z):
                response = float(assignment["relative_permeability"]) if assignment is not None else value
                if np.any(mask & occupied & (result != response)):
                    raise ValueError("Overlapping magnetic bodies have conflicting material assignments")
                result = np.where(mask, response, result)
                occupied |= mask
        return result
    def current(r, z):
        result = np.zeros_like(r)
        for coil, fraction in zip(driven, fractions):
            data = coil["data"]
            intervals = data.get("material_intervals_mm", ((coil["start_z_mm"], coil["end_z_mm"]),))
            area = .5e-3 * (float(data["mechanical_outer_diameter_mm"]) - float(data["mechanical_inner_diameter_mm"])) * sum(b-a for a, b in intervals) * 1e-3
            if area <= 0:
                raise ValueError("Coil winding area must be positive")
            mask = _part_mask(coil, r, z)
            if np.any(mask & iron_mask(r, z)):
                raise ValueError("Coil and magnetic material overlap; resolve the mechanical geometry before a field solve")
            result += mask * float(fraction) / area
        return result
    if nonlinear:
        from temsim.physics.nonlinear_magnetostatics import solve_nonlinear
        materials = [settings["bh_material"]]
        overrides = settings.get("material_bh_overrides", {})
        material_classes = sorted(overrides)
        materials.extend(overrides[name] for name in material_classes)
        indices = {name: index+1 for index, name in enumerate(material_classes)}
        from temsim.magnetic_materials import validate_bh_material
        assigned_indices = {}
        for index, material in enumerate(materials):
            assigned_indices.setdefault(json.dumps(validate_bh_material(material), sort_keys=True), index)
        for part_assignments in assignments.values():
            for assignment in part_assignments.values():
                if assignment["magnetic_response"] == "bh":
                    encoded = json.dumps(assignment["bh_material"], sort_keys=True)
                    if encoded not in assigned_indices:
                        assigned_indices[encoded] = len(materials)
                        materials.append(assignment["bh_material"])
        def material_ids(r, z):
            result = np.full_like(r, -1, dtype=int)
            occupied = np.zeros_like(r, dtype=bool)
            for part in iron:
                default_index = indices.get(part["data"].get("material_class", ""), 0)
                for mask, assignment in _material_region_masks(part, assignments[part["key"]], r, z):
                    index = default_index
                    if assignment is not None:
                        index = (assigned_indices[json.dumps(assignment["bh_material"], sort_keys=True)]
                                 if assignment["magnetic_response"] == "bh" else -1)
                    if np.any(mask & occupied & (result != index)):
                        raise ValueError("Overlapping magnetic bodies have conflicting material assignments")
                    result[mask] = index
                    occupied |= mask
            return result
        solution = solve_nonlinear(r, z, material_ids, current, materials,
                                   relative_tolerance=float(settings.get("relative_tolerance", 1e-7)),
                                   max_iterations=int(settings.get("max_iterations", 80)))
    else:
        solution = solve_axisymmetric(r, z, permeability, current)
    fingerprint = hashlib.sha256(geometry_json.encode()).hexdigest()
    source_hash = hashlib.sha256((SOLVER_VERSION + geometry_json + settings_json).encode()).hexdigest()
    note = f"{SOLVER_VERSION}; topology={topology or 'legacy unspecified'}; linear mu_r={mur:g}; channel={geometry.get('lens_key', 'test')}; NI={turns:g} at 100%; residual={solution.relative_residual:.4g}; mesh/domain convergence not checked; not calibrated"
    if nonlinear:
        note = f"{SOLVER_VERSION}; static isotropic B-H; material={settings['bh_material']['label']}; joint NI={settings['channel_ampere_turns']}; residual={solution.relative_residual:.4g}; Newton iterations={solution.iterations}; peak material B={solution.peak_material_t:.6g} T; no hysteresis; mesh/domain convergence not checked; not calibrated"
    assigned_labels = [f"{key}/{region}={value['material_key']}"
                       for key, regions in sorted(assignments.items())
                       for region, value in sorted(regions.items())]
    if assigned_labels:
        note += "; explicit part materials: " + ", ".join(assigned_labels)
    field_map = MagneticFieldMap("axisymmetric_rz", (r, z), (solution.br_t, solution.bz_t), CoordinateRegistration(),
                                 fingerprint, 100.0, 1, FieldMapProvenance("fem", "", source_hash, note))
    for array in (solution.a_phi_tm, solution.br_t, solution.bz_t):
        array.setflags(write=False)
    return field_map, solution


def solve_geometry_field_map(binding, settings):
    """Deterministic geometry/settings cache; never optimise excitation presets."""
    return solve_geometry_field_details(binding, settings)[0]


def solve_geometry_field_details(binding, settings):
    """The same cached solve plus immutable potential and residual diagnostics."""
    from temsim.excitation_calibration import validate_excitation_recipe
    validate_excitation_recipe(settings)
    return _solve_bound_geometry(binding.canonical_geometry_json,
                                 json.dumps(dict(settings), sort_keys=True, allow_nan=False))
