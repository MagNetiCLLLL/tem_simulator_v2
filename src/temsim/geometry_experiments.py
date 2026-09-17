"""Detached TOML candidates using the existing editor and assembly validation."""
from hashlib import sha256
import math
from pathlib import Path

from temsim import input_io
from temsim.design_experiments import ParameterSweep, SweepAxis, SweepPoint
from temsim.immutable_json import json_digest, thaw_json


def plan_geometry_sweep(recipe, module, part, dimension, values, *, optimization=None, maximum_points=64):
    if module not in {"gun", "column", "recording", "beam_blanker"}:
        raise ValueError("Choose a selected assembly module")
    if not part or not dimension.endswith(("_mm", "_um", "_nm", "_deg")):
        raise ValueError("Choose one existing scalar part dimension")
    if recipe.instrument_snapshot is None:
        raise ValueError("Geometry experiments require a complete instrument capture")
    values = tuple(float(value) for value in values)
    if not 1 <= len(values) <= maximum_points:
        raise ValueError("Geometry candidate count exceeds its numerical budget")
    axis = SweepAxis(f"geometry[{part}].{dimension}", values, dimension.rsplit("_", 1)[-1])
    if optimization is not None:
        optimization = validate_optimization(optimization)
    assumptions = dict(module=module, part=part, dimension=dimension,
        controls="Optimize selected controls for each candidate" if optimization else "Fixed-control comparison",
        optimization=optimization, definition_policy="Independent materialized TOML; live files unchanged")
    points = tuple(SweepPoint(i, {axis.path: value}, recipe.state_payload,
        json_digest(recipe.state_payload)) for i, value in enumerate(values))
    return ParameterSweep(recipe.digest, (axis,), points, "geometry", assumptions)


def validate_optimization(document):
    from temsim.alignment_constraints import ConstrainedAlignment, BeamConstraint
    if document.get("key") not in {"microprobe_illumination", "nanoprobe_convergence"}:
        raise ValueError("Geometry optimization requires a registered condenser target")
    target = float(document["target"])
    if not math.isfinite(target):
        raise ValueError("Alignment target must be finite")
    options = document["options"]
    options = ConstrainedAlignment(tuple(BeamConstraint(**row) for row in options["constraints"]),
        options["bounds"], options["maximum_evaluations"], options["minimum_effective_samples"], options["check_topology"])
    return dict(key=document["key"], target=target, options=options.to_dict())


@input_io.using_state_inputs
def candidate_state(recipe, point, sweep, base_archive, *, cancelled=lambda: False):
    from temsim.design_sweep_execution import rebuild_recipe_state, _runtime_operating_values, _prepare_high_accuracy_request, restore_operating_values
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.part_model_document import PartModelDocument
    from temsim.column.state_layout import apply_physical_layout_to_state
    from temsim.calculation_manifest import capture_external_input_identities
    from temsim.instrument_snapshot import capture_instrument_snapshot

    state, selection = rebuild_recipe_state(recipe)
    _prepare_high_accuracy_request(state, recipe)
    before = _runtime_operating_values(state)
    settings = sweep.assumptions
    resolved = state._resolved_assembly
    paths = dict(resolved.selected_module_paths)
    module = settings["module"]
    if module not in paths:
        raise ValueError("The selected module is not installed in the captured assembly")
    path = Path(resolved.root) / paths[module]
    input_io.bind_archive(state, base_archive)
    with input_io.input_scope(state):
        draft = PartModelDocument(path)
        value = point.coordinates[sweep.axes[0].path]
        draft.set_dimension(("parts", settings["part"], settings["dimension"]), value)
        text = draft.independent_copy_text()
    payload = thaw_json(base_archive)
    original = input_io.archive_for(base_archive).original_path(path)
    row = next((r for r in payload["files"] if Path(r["path"]) == original), None)
    if row is None:
        raise ValueError("Candidate module is missing from the input archive")
    content = text.encode("utf-8")
    row.update(content_hex=content.hex(), sha256=sha256(content).hexdigest())
    payload["digest"] = json_digest({k: v for k, v in payload.items() if k != "digest"})
    input_io.bind_archive(state, payload)
    # Tip curvature is derived from the edited radius, not an independent
    # operating control to restore over TOML geometry.
    if settings["part"] == "feg_tip" and settings["dimension"] == "tip_radius_nm":
        for parameters in before.values():
            parameters.pop("curvature_nm_inv", None)
    with input_io.input_scope(state):
        catalog = AssemblyCatalog(root=resolved.root)
        candidate = catalog.apply(state, selection, preserve_operating_parameters=True)
        restore_operating_values(state, before)
        apply_physical_layout_to_state(state, preserve_operating_parameters=True,
            assembly_root=catalog.root, assembly=candidate)
        actual = _runtime_operating_values(state)
        differences = [f"{key}.{name}: {value!r} -> {actual.get(key, {}).get(name)!r}"
            for key, fields in before.items() for name, value in fields.items() if actual.get(key, {}).get(name) != value]
        if differences:
            raise ValueError("Candidate geometry cannot preserve captured operating controls: " + "; ".join(differences))
        # Never treat an imported map's mismatch as permission to fall back.
        fields = validate_active_fields(state)
        evidence = dict(candidate_module=str(original), candidate_toml=text, candidate_toml_sha256=row["sha256"],
            archive_identity=state._archive_resolver.identity, controls=settings["controls"],
            field_dependencies=fields, external_inputs=capture_external_input_identities(state),
            numerical_status="NOT_RUN", validation="Existing component graph, module clearances and assembled physical layout")
        if settings.get("optimization"):
            try:
                state, alignment = optimize_candidate(state, settings["optimization"], cancelled=cancelled)
            except Exception as exc:
                exc.evidence = dict(evidence, alignment=getattr(exc, "evidence", {}))
                raise
            evidence["alignment"] = alignment
        evidence["operating_controls"] = _runtime_operating_values(state)
        evidence["candidate_input_identity"] = capture_instrument_snapshot(state).digest
    return state, selection, evidence


def validate_active_fields(state):
    from temsim.simulation_modes import uses_field_maps
    from temsim.physics.lens_field_provider import resolve_runtime_lens_field_provider, GeometryAwareAnalyticFieldProvider
    from temsim.component_keys import CONDENSER_LENS_KEYS
    from temsim.geometry_effects import admit_state_geometry
    admit_state_geometry(state)
    if not uses_field_maps(state):
        return dict(status="Configured analytical model retained; changed pole geometry is not a magnetostatic field solution")
    descriptors = getattr(state, "lens_field_map_descriptors", {})
    keys = set(descriptors) | set(getattr(state, "_lens_field_map_bindings", {}))
    reports = {}
    for lens in state.lenses:
        if not lens.enabled or lens.key not in keys:
            continue
        native = state.condenser_system[lens.key] if lens.key in CONDENSER_LENS_KEYS else lens
        provider = resolve_runtime_lens_field_provider(state, lens.key, native)
        if isinstance(provider, GeometryAwareAnalyticFieldProvider):
            raise ValueError(f"{lens.key}: requested field map cannot consume candidate geometry: {provider.fallback_reason}")
        reports[lens.key] = getattr(state, "_field_provider_diagnostics", {}).get(lens.key, {})
    return reports


def optimize_candidate(state, settings, *, cancelled):
    from temsim.alignment_transaction import AlignmentRequest, solve_alignment_candidate, AlignmentCommitGate
    from temsim.alignment_constraints import BeamConstraint, ConstrainedAlignment
    options = settings["options"]
    options = ConstrainedAlignment(tuple(BeamConstraint(**v) for v in options["constraints"]),
        options["bounds"], options["maximum_evaluations"], options["minimum_effective_samples"], options["check_topology"])
    request = AlignmentRequest.capture(state, settings["key"], settings["target"], revision=0, options=options)
    candidate = solve_alignment_candidate(request, cancelled=cancelled)
    report = thaw_json(candidate.validation)
    report.pop("candidate_snapshot", None)
    report.update(solved_controls=dict(candidate.result.strengths), target=request.target, key=request.key)
    if candidate.status != "READY_TO_APPLY":
        error = ValueError(f"Candidate alignment failed: {candidate.status}; {candidate.result.message}")
        error.evidence = report
        raise error
    return AlignmentCommitGate().apply(state, candidate, revision=0), report
