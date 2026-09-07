"""Read-only explanations of existing parameter consumers, never a field solve.

``active`` describes a known scientific input route in the selected mode. It
does not promise a changed numerical result: a disabled component, zero current,
overlapping narrower bore, or equal material response can mask an edit. Recipe
checks here do not replace full assembly validation or numerical convergence.
CAD feature meshes are deliberately excluded from all scientific effects.
"""
from collections.abc import Mapping
from dataclasses import dataclass, replace
import math

from temsim.component_keys import APERTURE_KEYS, FIXED_APERTURE_KEYS
from temsim.magnetic_circuits import circuit_channels, optical_owner, part_data
from temsim.part_materials import MATERIAL_PROFILES, is_magnetostatic_body
from temsim.parameter_semantics import describe_parameter
from temsim.simulation_modes import MODE_BY_KEY


@dataclass(frozen=True)
class ParameterImpact:
    effects: tuple[str, ...]
    active: bool | None
    status: str
    label: str
    detail: str
    affected_results: tuple[str, ...]


@dataclass(frozen=True)
class ComponentImpactSummary:
    simulation_mode: str | None
    mode_label: str
    status: str
    label: str
    detail: str
    active_effects: tuple[str, ...]
    ignored_cad_parameters: tuple[str, ...]
    parameter_impacts: tuple[tuple[tuple, ParameterImpact], ...]


_COIL = "magnetic_excitation_coil"
_LENS = "magnetic_lens_assembly"
_LINEAR = "axisymmetric_linear_fem"
_NONLINEAR = "axisymmetric_nonlinear_fem"
_AXIAL_EXTENTS = {"length_mm", "local_start_z_mm", "local_end_z_mm"}
_SPLIT_FIELDS = {f"{side}_yoke_{edge}_local_z_mm" for side in ("upper", "lower") for edge in ("start", "end")}
_FEM_FIELDS = _AXIAL_EXTENTS | {
    "mechanical_inner_diameter_mm", "mechanical_outer_diameter_mm",
    "mechanical_bore_diameter_mm", "mechanical_tip_diameter_mm",
    "magnetic_radial_profile_mm", "material_intervals_mm",
    "pole_nose_axial_length_mm", "pole_mounting_shank_axial_length_mm",
    "pole_vacuum_connector_axial_length_mm", "pole_stem_outer_diameter_mm",
    "pole_vacuum_connector_outer_diameter_mm", "pole_mounting_shank_inner_diameter_mm",
    "pole_piece_geometry_style", "coil_ampere_turn_fraction",
}
_APERTURE_RUNTIME = {
    "radius_mm", "radius_um", "diameter_mm", "diameter_um",
    "offset_x_mm", "offset_y_mm", "offset_x_um", "offset_y_um", "z_mm",
    "enabled", "installed", "slit_width_mm", "slit_width_um",
}
_MATERIAL_FIELDS = {"material_regions", "material_class", "aperture_plate_material"}
_RESULT_LABELS = {
    "3d_preview": "3D preview", "physical_layout": "Physical layout",
    "ray_transport": "Ray trajectories", "beam_transmission": "Beam transmission",
    "magnetic_field": "Magnetic field", "optical_transfer": "Optical transfer",
    "field_map_validity": "Field-map validity", "material_model": "Material model",
    "material_definition": "Saved material definition", "input_validation": "Input validation",
}


def _impact(status, label, detail, *, effects=("display", "geometry"), results=("3d_preview", "physical_layout")):
    active = True if status == "active" else None if status == "unknown" else False
    return ParameterImpact(tuple(effects), active, status, label, detail,
                           tuple(_RESULT_LABELS.get(result, result) for result in results))


def _row(part):
    return part_data(part) if part is not None else {}


def _context(part, path, by_key):
    row = _row(part)
    index = {str(key): _row(value) for key, value in (by_key or {}).items()}
    if row.get("key"):
        index[str(row["key"])] = row
    path = (path,) if isinstance(path, str) else tuple(path)
    kind = path[0] if path else ""
    if kind in {"parts", "runtime", "derived"} and len(path) >= 3:
        target = str(path[1])
        if target in index:
            row = index[target]
        elif kind == "parts" and target != str(row.get("key", "")):
            return {}, index, path, kind, ""
        field = str(path[2])
    elif kind == "geometry" and len(path) >= 2:
        field = str(path[1])
    else:
        field = str(path[0]) if path else ""
    return row, index, path, kind, field


def _is_aperture(row):
    return (row.get("key") in APERTURE_KEYS or row.get("aperture_plate_form") == "perforated_strip"
            or row.get("mechanical_profile") in {"circular_aperture", "fixed_differential_pumping_aperture"})


def _owners(row, index):
    owners = set()
    if row.get("field_source_key"):
        owners.add(str(row["field_source_key"]))
    owner = optical_owner(row, index)
    if owner:
        owners.add(owner)
    owners.update(str(key) for key in row.get("magnetic_lens_keys", ()))
    overlap = str(row.get("mechanical_overlap_group", ""))
    if overlap.endswith("_assembly") and overlap[:-9] in index:
        owners.add(overlap[:-9])
    return tuple(sorted({channel for key in owners for channel in circuit_channels(index, key)}))


def _recipe_issue(recipe):
    """Validate the existing operator inputs without loading maps or solving."""
    try:
        solver = recipe.get("solver")
        ni = float(recipe["ampere_turns"])
        if not math.isfinite(ni):
            raise ValueError("finite ampere-turns required")
        if solver == _NONLINEAR:
            from temsim.physics.nonlinear_circuits import operator_settings
            operator_settings(recipe)
            if ni < 0:
                raise ValueError("nonnegative ampere-turns required; use polarity for direction")
        elif solver == _LINEAR:
            mu = float(recipe["relative_permeability"])
            nr, nz = int(recipe.get("radial_nodes", 40)), int(recipe.get("axial_nodes", 80))
            padding = float(recipe.get("padding_factor", 2))
            if not (math.isfinite(mu) and mu > 0 and 12 <= nr <= 256 and 16 <= nz <= 512
                    and math.isfinite(padding) and 1.5 <= padding <= 10):
                raise ValueError("valid permeability, mesh and padding inputs required")
        else:
            return "No supported generated-field recipe"
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return str(exc)
    return ""


def _magnetic_impact(row, index, mode, descriptors, *, material=False):
    source = "Assigned material response" if material else "Authoritative mechanical geometry"
    if mode in {"ideal", "analytical"}:
        return _impact("inactive", "Magnetic geometry inactive in this mode",
                       f"{MODE_BY_KEY[mode].label} uses the existing analytic/paraxial lens model. "
                       f"{source} does not rebuild its magnetic field; focusing and magnetic rotation remain.")
    try:
        owners = _owners(row, index)
    except ValueError as exc:
        return _impact("unknown", "Magnetic ownership needs verification", str(exc))
    if not owners:
        return _impact("unknown", "Magnetic owner not connected",
                       "Provide the optical parent/shared-circuit context to identify the field recipe.")
    descriptors = descriptors or {}
    recipes = {key: descriptors.get(key, {}) for key in owners}
    expected = _LINEAR if mode == "linear_geometry" else _NONLINEAR if mode == "nonlinear_material" else None
    required, generated, imported, analytic = [], [], [], []
    for key, recipe in recipes.items():
        if not isinstance(recipe, Mapping):
            required.append(f"{key}: invalid field recipe")
            continue
        solver = recipe.get("solver")
        if expected and solver != expected:
            required.append(f"{key}: configure {expected}")
        elif solver in {_LINEAR, _NONLINEAR}:
            issue = _recipe_issue(recipe)
            if solver == _LINEAR and index.get(key, {}).get("magnetic_circuit_topology") == "monolithic_saturated_insert":
                issue = "saturation insert requires nonlinear B-H"
            if issue:
                required.append(f"{key}: {issue}")
            else:
                generated.append(key)
        elif any(name in recipe for name in ("source_path", "geometry_fingerprint", "map_type")):
            imported.append(key)
        elif not recipe or solver in {None, "analytic", "analytical"}:
            analytic.append(key)
        else:
            required.append(f"{key}: unrecognized field recipe")
    if required:
        return _impact("configuration_required", "Field configuration required", "; ".join(required)
                       + ". Full assembly validity and a successful solve are still required.")
    # Match existing joint B-H/shared linear operator restrictions without
    # pretending that local descriptor readiness validates an entire column.
    nonlinear = [key for key in generated if recipes[key].get("solver") == _NONLINEAR]
    if nonlinear:
        from temsim.physics.nonlinear_circuits import operator_settings
        joint = [recipe for recipe in descriptors.values() if isinstance(recipe, Mapping)
                 and recipe.get("solver") == _NONLINEAR]
        issues = [_recipe_issue(recipe) for recipe in joint]
        if any(issues) or any(operator_settings(recipe) != operator_settings(joint[0]) for recipe in joint):
            return _impact("configuration_required", "Joint B-H configuration required",
                           "All joint B-H channels need valid, identical material/mesh/operator settings; ampere-turns may differ.")
        missing = [key for key in owners if recipes[key].get("solver") != _NONLINEAR]
        if missing:
            return _impact("configuration_required", "Shared B-H channels missing",
                           "Configure every shared-circuit channel for the joint B-H solve: " + ", ".join(missing))
    if len(generated) > 1 and not nonlinear:
        signatures = {(float(recipes[key]["relative_permeability"]),
                       tuple(sorted((name, float(value)) for name, value in
                                    recipes[key].get("material_permeabilities", {}).items())),
                       int(recipes[key].get("radial_nodes", 40)), int(recipes[key].get("axial_nodes", 80)),
                       float(recipes[key].get("padding_factor", 2))) for key in generated}
        if len(signatures) > 1:
            return _impact("configuration_required", "Shared linear operators disagree",
                           "Shared channels require identical material, mesh and boundary settings; excitation may differ.")
    if generated:
        detail = (f"{source} enters the existing axisymmetric "
                  f"{'joint static B-H' if nonlinear else 'linear FEM'} route for {', '.join(generated)}. "
                  "This reports supported inputs, not a converged or experimentally calibrated result. "
                  "Ampere-turns remain independent prescribed inputs; coil dimensions change winding area and current density.")
        if imported:
            detail += " Imported channels also require new matching maps after edits: " + ", ".join(imported) + "."
        if analytic:
            detail += " Analytic channels do not solve this geometry: " + ", ".join(analytic) + "."
        return _impact("active", "Active magnetic material" if material else "Active magnetic geometry", detail,
                       effects=("display", "geometry", "magnetic_field"),
                       results=("3d_preview", "physical_layout", "magnetic_field", "ray_transport", "optical_transfer"))
    if imported:
        return _impact("configuration_required", "Matching imported field map required",
                       "Editing this geometry/material invalidates an earlier imported map for " + ", ".join(imported)
                       + ". The map is not regenerated from the edit; Custom may use an explicit provisional analytic fallback. "
                       "Map-file availability and the current geometry fingerprint must be checked separately.",
                       effects=("display", "geometry", "field_map_identity"),
                       results=("3d_preview", "physical_layout", "field_map_validity"))
    return _impact("inactive", "Custom analytic field ignores mechanical edits",
                   "No generated or imported field recipe is configured for " + ", ".join(analytic)
                   + ". The existing analytic field is retained; mechanical edits do not reconstruct it.")


def _wall_effect(row, field):
    if field not in _AXIAL_EXTENTS | {"vacuum_inner_diameter_mm", "axial_vacuum_context_only"}:
        return False
    if "vacuum_inner_diameter_mm" not in row or (row.get("axial_vacuum_context_only") and field != "axial_vacuum_context_only"):
        return False
    try:
        start, end = float(row["local_start_z_mm"]), float(row["local_end_z_mm"])
        return math.isfinite(start) and math.isfinite(end) and end > start
    except (KeyError, TypeError, ValueError):
        return False


def _include_wall_impact(impact, row, field):
    """A material extent can also delimit a vacuum segment, in every mode."""
    if not _wall_effect(row, field):
        return impact
    detail = (impact.detail + " The same axial extent also bounds this part's vacuum-ID segment in the column-wall profile. "
              "That beam-clearance route is active in every mode; an overlapping narrower/equal bore can mask its numerical effect.")
    effects = tuple(dict.fromkeys((*impact.effects, "beam_clearance")))
    results = tuple(dict.fromkeys((*impact.affected_results, "Ray trajectories", "Beam transmission")))
    if impact.status == "inactive":
        return replace(impact, active=True, status="active", label="Vacuum active; magnetic geometry inactive",
                       effects=effects, detail=detail, affected_results=results)
    label = "Vacuum and magnetic geometry active" if impact.status == "active" else impact.label
    return replace(impact, label=label, effects=effects, detail=detail, affected_results=results)


def _describe_parameter_impact(part, path, *, by_key=None, simulation_mode="ideal", descriptors=None):
    """Describe only verified consumers of a manifest, runtime or derived path.

    Paths may be ``('parts', key, field, ...)``, ``('runtime', key, field)``
    or ``('derived', key, 'radial_thickness_mm')``. Unknown context stays unknown.
    """
    row, index, path, kind, field = _context(part, path, by_key)
    if not row:
        return _impact("unknown", "Source part not connected", "The parameter's source part is unavailable; its physical route needs verification.")
    if field == "model_3d":
        return _impact("unsupported", "CAD display only",
                       "The saved model_3d base, transforms and hole/slot features change the 3D mesh only. "
                       "They are not consumed by beam clipping, magnetic FEM or field-map geometry identity; no electrical/thermal coupling is inferred.",
                       effects=("display",), results=("3d_preview",))
    if simulation_mode not in MODE_BY_KEY or not MODE_BY_KEY[simulation_mode].available:
        return _impact("unknown", "Simulation context not connected",
                       "Select a connected, implemented simulation mode before assessing physical participation.", effects=("display",))
    mode = simulation_mode
    profile = row.get("mechanical_profile")
    aperture = _is_aperture(row)
    if kind == "runtime":
        if aperture and field in _APERTURE_RUNTIME:
            if field == "enabled" and row.get("key") in FIXED_APERTURE_KEYS:
                return _impact("inactive", "Fixed aperture cannot retract",
                               "This hardware remains inserted; its enabled flag is normalized to true. Installation and opening remain separate controls.", effects=("operating",))
            return _impact("active", "Active aperture operating control",
                           "The runtime opening/offset/position and installation state enter the aperture transmission mask. "
                           "The mechanical carrier bore and envelope are separate; a disabled or absent stop does not clip rays.",
                           effects=("display", "operating", "beam_clearance"), results=("physical_layout", "ray_transport", "beam_transmission"))
        if profile == _LENS and field in {"percent", "excitation_percent", "polarity", "enabled"}:
            if mode in {"linear_geometry", "nonlinear_material"}:
                configured = _magnetic_impact(row, index, mode, descriptors)
                if configured.status != "active":
                    return replace(configured, effects=("operating",))
            return _impact("active", "Active lens operating control",
                           "Excitation and polarity control the selected lens model. Linear maps scale their reference field; "
                           "B-H routes rebuild the joint operating point. Mechanical dimensions do not infer ampere-turns.",
                           effects=("operating", "magnetic_field"), results=("magnetic_field", "ray_transport", "optical_transfer"))
        return _impact("unknown", "Runtime route needs verification",
                       "This runtime field has no verified consumer in the parameter-impact registry.", effects=("operating",))
    if kind == "derived" and field == "radial_thickness_mm":
        derived = describe_parameter_impact(row, ("parts", row["key"], "mechanical_outer_diameter_mm"),
                    by_key=index, simulation_mode=mode, descriptors=descriptors)
        return replace(derived, detail="Radial thickness is (mechanical OD - ID) / 2; editing changes the chosen diameter boundary. " + derived.detail)
    semantics = describe_parameter(row, path, by_key=index)
    if field in _MATERIAL_FIELDS:
        snapshot_metadata = (field == "material_regions" and len(path) > 4
                             and path[4] not in {"relative_permeability", "bh_material", "magnetic_response", "material_key"})
        if snapshot_metadata or (field == "material_class" and profile in {_COIL, "magnetic_lens_housing"}):
            return _impact("inactive", "Material role/source metadata",
                           "This identifies the material role or its source; it is not an independent constitutive-law input. "
                           "It remains part of saved material/map identity. Assigned region responses control magnetic behavior.",
                           effects=("display", "material_definition", "field_map_identity"),
                           results=("3d_preview", "material_definition", "field_map_validity"))
        if profile not in MATERIAL_PROFILES or field == "aperture_plate_material":
            return _impact("inactive", "Material definition only",
                           "This component has no modeled magnetostatic solid. The saved material definition does not change "
                           "magnetic fields, beam cutoffs, scattering, conductivity or temperature.",
                           effects=("display", "material_definition"), results=("3d_preview", "material_definition"))
        impact = _magnetic_impact(row, index, mode, descriptors, material=True)
        if profile == _COIL and impact.status == "active":
            return _impact("active", "Prescribed nonmagnetic winding model",
                           "Coil material assignments are validated as the explicit static relative-permeability-one model. "
                           "All supported winding material choices have the same magnetic response; changing copper to aluminum "
                           "does not change the field or prescribed ampere-turns. Conductivity, heating and eddy currents are not modeled.",
                           effects=("display", "material_definition", "magnetic_model"),
                           results=("3d_preview", "material_model", "field_map_validity"))
        return impact
    if aperture:
        if field == "maximum_radius_mm":
            return _impact("active", "Operating range constraint",
                           "This sets the permitted runtime opening range; it does not itself set the current aperture diameter.",
                           effects=("operating",), results=("input_validation",))
        direct_hole = (row.get("key") == "projection_chamber_dpa_aperture" and field == "mechanical_bore_diameter_mm"
                       or row.get("key") == "nanopulser_aperture" and field == "aperture_radius_mm")
        if direct_hole or field == "optical_reference_local_z_mm":
            return _impact("active", "Active hard-aperture geometry",
                           "This field is explicitly bound to the physical stop's opening or optical plane and changes the transmission mask.",
                           effects=("display", "geometry", "beam_clearance"), results=("physical_layout", "ray_transport", "beam_transmission"))
        if field in {"plate_thickness_mm", "active_length_mm", "mechanical_bore_diameter_mm", "bore_diameter_mm"}:
            return _impact("inactive", "Plate/carrier geometry; separate operating opening",
                           "The plate thickness or carrier bore is displayed as saved geometry. Current beam clipping uses the separate "
                           "runtime opening at a hard-edge plane; no finite-thickness plate propagation is modeled.")
    split_parent = profile == _LENS and field in _SPLIT_FIELDS | {"mechanical_coil_axial_inset_mm"}
    magnetic_geometry = split_parent or (semantics.category != "envelope" and field in _FEM_FIELDS and profile in MATERIAL_PROFILES)
    if _wall_effect(row, field) and not magnetic_geometry:
        detail = ("The part's vacuum ID and axial extent enter the resolved column-wall clearance profile in every mode. "
                  "Overlapping parts and the continuous tube use the narrower bore, so an edit may be masked. "
                  "An aperture envelope length is not its plate thickness or operating opening.")
        if profile in {_COIL, "magnetic_lens_yoke"} and semantics.category == "envelope":
            detail += " The split material uses its parent/explicit intervals; this child envelope does not resize the winding or yoke intervals."
        return _impact("active", "Vacuum-wall geometry input",
                       detail,
                       effects=("display", "geometry", "beam_clearance"), results=("physical_layout", "ray_transport", "beam_transmission"))
    if field == "vacuum_inner_diameter_mm":
        return _impact("inactive" if row.get("axial_vacuum_context_only") else "unknown", "Vacuum context only" if row.get("axial_vacuum_context_only") else "Vacuum extent needs verification",
                       "Context-only vacuum rows do not narrow the wall profile; a finite resolved axial segment is needed to verify other rows.")
    if semantics.category == "envelope":
        detail = "This is a mechanical/display envelope, not a solved material or aperture opening."
        parent = index.get(str(row.get("parent_key", "")), {})
        if profile in {_COIL, "magnetic_lens_yoke"} and _SPLIT_FIELDS.issubset(parent):
            detail += " The actual upper/lower material intervals come from the parent yoke endpoints and coil inset; changing this child envelope does not resize them."
        return _impact("inactive", "Envelope geometry only", detail)
    if magnetic_geometry:
        if profile == "magnetic_lens_housing" and not is_magnetostatic_body(row):
            impact = _impact("inactive", "Nonmagnetic housing geometry",
                             "This housing is not a magnetic FEM domain without an explicit magnetic material assignment. Geometry is retained for display and map identity.")
        else:
            impact = _magnetic_impact(row, index, mode, descriptors)
        return _include_wall_impact(impact, row, field)
    if semantics.category == "vacuum":
        return _impact("unknown", "Vacuum route needs verification", "Only the resolved vacuum-ID/axial-segment consumer is verified here; this field may describe a liner or another constraint.")
    return _impact("unknown", "Physical route needs verification",
                   f"{semantics.label}: saved/display semantics are known, but this field's scientific consumer has not been verified.")


def describe_parameter_impact(part, path, *, by_key=None, simulation_mode="ideal", descriptors=None):
    """Inspect an existing input route; incomplete or invalid drafts stay unknown.

    This is an explanation for editors, not a substitute for save validation.
    No map file is read, field solved, runtime state changed or cache updated.
    """
    try:
        return _describe_parameter_impact(part, path, by_key=by_key,
                    simulation_mode=simulation_mode, descriptors=descriptors)
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as exc:
        return _impact("unknown", "Parameter route needs verification",
                       "The draft/context is insufficient to verify this route: " + str(exc), effects=("display",))


def _cad_leaf_paths(value, prefix="model_3d"):
    if isinstance(value, Mapping):
        return tuple(path for name, item in value.items() if name not in {"schema_version", "id"}
                     for path in _cad_leaf_paths(item, prefix + "." + str(name)))
    if isinstance(value, (list, tuple)):
        return tuple(path for index, item in enumerate(value) for path in _cad_leaf_paths(item, f"{prefix}[{index}]"))
    return (prefix,)


def component_impact_summary(part, *, by_key=None, simulation_mode="ideal", descriptors=None):
    """Compact component report, including every CAD field ignored by physics."""
    row = _row(part)
    impacts = tuple((("parts", row.get("key", ""), field), describe_parameter_impact(
                        row, ("parts", row.get("key", ""), field), by_key=by_key,
                        simulation_mode=simulation_mode, descriptors=descriptors))
                    for field in row if field not in {"key", "name", "parent_key", "order", "branch"})
    ignored = _cad_leaf_paths(row["model_3d"]) if "model_3d" in row else ()
    index = {str(key): _row(value) for key, value in (by_key or {}).items()}
    key = str(row.get("key", ""))
    index[key] = row
    displayed = {key}
    # Same closure as part_model_from_document: descendants and shared bodies
    # are shown together, even though this report's field rows stay part-local.
    while True:
        more = set()
        for name, child in index.items():
            parent = child.get("parent_key")
            shared = child.get("magnetic_lens_keys", ())
            shared = [item for item in shared if isinstance(item, str)] if isinstance(shared, (tuple, list)) else ()
            if (isinstance(parent, str) and parent in displayed) or displayed.intersection(shared):
                more.add(name)
        if more <= displayed:
            break
        displayed.update(more)
    for name in sorted(displayed - {key}):
        if "model_3d" in index[name]:
            ignored += _cad_leaf_paths(index[name]["model_3d"], f"parts.{name}.model_3d")
    known = [impact for _, impact in impacts if impact.status != "unknown"]
    status = ("configuration_required" if any(item.status == "configuration_required" for item in known)
              else "active" if any(item.active for item in known)
              else "inactive" if known else "unknown")
    connected = isinstance(simulation_mode, str) and simulation_mode in MODE_BY_KEY and MODE_BY_KEY[simulation_mode].available
    if not connected:
        status = "unknown"
    label = MODE_BY_KEY[simulation_mode].label if connected else "Simulation context not connected"
    effects = tuple(sorted({effect for _, item in impacts if item.active for effect in item.effects
                            if effect not in {"display", "geometry"}}))
    notes = []
    for _, impact in impacts:
        if impact.status != "unknown" and impact.detail not in notes:
            notes.append(impact.detail)
    if ignored:
        notes.append("Ignored by beam/magnetic solvers: " + ", ".join(ignored) + ".")
    return ComponentImpactSummary(simulation_mode, label, status, f"{label}: {status.replace('_', ' ')}",
                                  "\n".join(notes) or "No verified scientific parameter route for this component.",
                                  effects, ignored, impacts)
