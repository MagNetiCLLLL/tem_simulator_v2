"""Machine-queryable geometry participation and axisymmetric-model admission."""

from dataclasses import asdict
import json

from temsim.parameter_impact import component_impact_summary

GEOMETRY_EFFECTS_SCHEMA = "geometry-consumers-v1"
GEOMETRY_POLICIES = ("require_full_geometry", "authoritative_dimensions", "legacy_authoritative_dimensions")


def geometry_effects(part, **context):
    summary = component_impact_summary(part, **context)
    parameters = []
    for path, impact in summary.parameter_impacts:
        parameters.append({"path": list(path), **asdict(impact),
                           "display": "display" in impact.effects,
                           "field": bool(impact.active and "magnetic_field" in impact.effects),
                           "vacuum_or_stop": bool(impact.active and "beam_clearance" in impact.effects),
                           "scientific_input": bool(impact.active and set(impact.effects) - {"display", "geometry"})})
    return {"schema": GEOMETRY_EFFECTS_SCHEMA, "mode": summary.simulation_mode,
            "status": summary.status, "parameters": parameters,
            "unsupported_cad_paths": list(summary.ignored_cad_parameters),
            "scope": "Mechanical envelope, material intervals, field support and optical reference remain separate inputs"}


def field_geometry_admission(state, binding, descriptor):
    """Run before provider cache lookup; an unsupported new CAD edit must fail.

    Existing recipes retain a named legacy approximation. New research recipes
    require represented geometry unless the user explicitly selects dimensions.
    """
    policy = descriptor.get("geometry_policy", "legacy_authoritative_dimensions")
    if policy not in GEOMETRY_POLICIES:
        raise ValueError("Unknown field geometry policy")
    geometry = json.loads(binding.canonical_geometry_json)["lens_assembly"]
    participating = {row["key"] for name in ("parts", "magnetostatic_neighbours") for row in geometry.get(name, ())}
    unsupported = sorted(part.key for part in getattr(getattr(state, "_resolved_assembly", None), "parts", ())
                         if part.key in participating and part.data.get("model_3d"))
    if unsupported and policy == "require_full_geometry":
        raise ValueError("Axisymmetric field solver cannot consume model_3d features/transforms on " + ", ".join(unsupported)
                         + "; explicitly choose authoritative_dimensions approximation or provide a matching 3D field model")
    report = {"schema": GEOMETRY_EFFECTS_SCHEMA, "policy": policy, "ignored_model_3d_parts": unsupported,
            "status": "explicit_approximation" if unsupported else "authoritative_dimensions_represented",
            "legacy_policy": policy == "legacy_authoritative_dimensions"}
    key = json.loads(binding.canonical_geometry_json).get("lens_key", "unknown")
    state._geometry_effects_diagnostics = {**getattr(state, "_geometry_effects_diagnostics", {}), key: report}
    return report


def admit_state_geometry(state):
    """Check new strict CAD edits before any whole-calculation cache reuse."""
    from temsim.simulation_modes import uses_field_maps
    if not uses_field_maps(state) or not any(p.data.get("model_3d") for p in getattr(getattr(state,"_resolved_assembly",None),"parts",())):
        return
    from temsim.physics.lens_field_provider import lens_geometry_binding
    from temsim.physics.nonlinear_circuits import resolve_nonlinear_provider
    for lens in state.lenses:
        recipe = state.lens_field_map_descriptors.get(lens.key,{})
        if recipe.get("geometry_policy") != "require_full_geometry":
            continue
        binding = lens_geometry_binding(state,lens.key,lens)
        if recipe.get("solver") == "axisymmetric_nonlinear_fem":
            resolve_nonlinear_provider(state,lens.key,lens,binding,prepare_only=True)
        elif recipe.get("solver") == "axisymmetric_linear_fem":
            field_geometry_admission(state,binding,recipe)
