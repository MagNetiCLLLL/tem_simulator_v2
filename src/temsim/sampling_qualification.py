"""Resumable scalar evidence around the existing classical comparison engine.

No live-state writes, replacement source, adaptive tolerance or ray-array journal.
Time cancellation is cooperative at existing solver boundaries, not a hard kill.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from time import perf_counter

from temsim.immutable_json import freeze_json, thaw_json, json_digest
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.sampling_convergence import (
    AXES, ConvergenceRequest, SamplingCancelled, _thresholds, compare_runs,
    run_convergence, write_evidence,
)

SCHEMA = "incident-qualification-plan-v1"
RECEIPT_SCHEMA = "incident-qualification-journal-v1"
MAX_RECEIPT_BYTES = 64 * 1024**2
PLAN_AXES = {k: v for k, v in AXES.items() if k != "joint_steps"}
PRODUCT_AXES = {"spatial", "directions", "energies"}
OBSERVABLES = {
    "transmission": 1e-12, "plane_current_a": 1e-18,
    "diameter95_m": 1e-15, "alpha95_rad": 1e-12, "alpha99_rad": 1e-12,
    "centroid_x_m": 1e-12, "centroid_y_m": 1e-12,
    "chief_slope_x": 1e-12, "chief_slope_y": 1e-12,
}


def qualification_thresholds():
    result = _thresholds()
    result["absolute"] = dict(OBSERVABLES)
    result["definition"] = "exact-plane-current-chief-direction-containment-v1"
    return freeze_json(result)


def validate_thresholds(limits):
    if set(limits.get("absolute", {})) != set(OBSERVABLES):
        raise ValueError("Qualification requires all declared physical observables")
    values = [limits.get("relative"), limits.get("focus_tolerance_nm"), *limits["absolute"].values()]
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("Thresholds and absolute floors must be finite and positive")
    for name in ("minimum_transmitted_samples", "minimum_effective_samples"):
        if type(limits.get(name)) is not int or limits[name] < 16:
            raise ValueError("Qualification support screen cannot be below 16 samples")
    if limits.get("definition") != "exact-plane-current-chief-direction-containment-v1":
        raise ValueError("Unrecognized qualification observable definition")


def design_identity(snapshot):
    """Exclude only exposed numerical values; unknown fields stay identity-bound.

    A changed quadrature *method* remains significant (including None -> product).
    This is a plan identity, never permission to reuse a propagation cache.
    """
    graph = thaw_json(snapshot.graph)
    nodes = graph["nodes"]
    root = nodes[graph["root"]["ref"]]["attributes"]
    root.pop("step_mm", None)
    gun = nodes[root["electron_gun"]["ref"]]["attributes"]
    gun.pop("trace_step_mm", None)
    emitter = nodes[gun["emitter"]["ref"]]["attributes"]
    emitter.pop("ray_count", None)
    # These public properties are persisted through their canonical backing
    # attributes, not duplicated as independently configurable source inputs.
    quadrature = emitter.get("_emission_quadrature")
    if isinstance(quadrature, dict) and "ref" in quadrature:
        attrs = nodes[quadrature["ref"]]["attributes"]
        for key in PRODUCT_AXES:
            attrs.pop(key, None)
    surface = emitter.get("_surface_model")
    if isinstance(surface, dict) and "ref" in surface:
        attrs = nodes[surface["ref"]]["attributes"]
        field = attrs.get("field_numerics")
        if isinstance(field, dict) and "ref" in field:
            attrs = nodes[field["ref"]]["attributes"]
            for key in ("radial_nodes", "axial_nodes", "apex_cells_per_radius", "outer_radius_factor"):
                attrs.pop(key, None)
    return json_digest(dict(graph=graph, external_inputs=snapshot.external_inputs))


@dataclass(frozen=True)
class QualificationPlan:
    checkpoint: object
    axes: tuple[str, ...]
    refinements: int = 2
    maximum_comparisons: int = 12
    wall_seconds: float = 120.
    maximum_rays: int = 4096
    maximum_checkpoint_bytes: int = 512 * 1024**2
    spatial_samples: int = 9
    direction_samples: int = 9
    energy_samples: int = 9
    check_topology: bool = False
    checkpoint_spacing_mm: float = 2.
    thresholds: object = None

    def __post_init__(self):
        object.__setattr__(self, "axes", tuple(self.axes))
        if not self.axes or len(set(self.axes)) != len(self.axes) or not set(self.axes) <= PLAN_AXES.keys():
            raise ValueError("Select distinct supported independent axes")
        if "emission_samples" in self.axes and PRODUCT_AXES.intersection(self.axes):
            raise ValueError("Joint emission sampling and product quadrature are different methods; use separate plans")
        if type(self.refinements) is not int or not 2 <= self.refinements <= 8:
            raise ValueError("Declare two to eight refinements (at least three settings)")
        if type(self.maximum_comparisons) is not int or not 1 <= self.maximum_comparisons <= 128:
            raise ValueError("Comparison budget must be between 1 and 128")
        if isinstance(self.wall_seconds, bool) or not math.isfinite(self.wall_seconds) or not 1 <= self.wall_seconds <= 7200:
            raise ValueError("Declare a total wall-time budget from 1 to 7200 seconds")
        for value in (self.spatial_samples, self.direction_samples, self.energy_samples):
            if type(value) is not int or not 3 <= value <= 65536:
                raise ValueError("Product sampling factors must be integers from 3 to 65536")
        limits = qualification_thresholds() if self.thresholds is None else freeze_json(self.thresholds)
        validate_thresholds(limits)
        object.__setattr__(self, "thresholds", limits)

    def manifest(self):
        from dataclasses import fields
        result = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "checkpoint"}
        result.update(schema=SCHEMA, checkpoint_id=self.checkpoint.digest,
            snapshot_id=self.checkpoint.snapshot.digest,
            implementation=self.checkpoint.snapshot.implementation,
            seed_policy="deterministic existing emitter; no randomized replication",
            scope="classical incident transport through specimen entrance only",
            memory_scope="retained checkpoint/workspace estimate, not total process RSS",
            time_scope="cumulative; cooperative cancellation at solver boundaries")
        return freeze_json(result)

    @property
    def identity(self):
        return json_digest(self.manifest())

    @classmethod
    def from_manifest(cls, checkpoint, manifest):
        from dataclasses import fields
        names = {f.name for f in fields(cls)} - {"checkpoint"}
        result = cls(checkpoint, **{key: manifest[key] for key in names})
        if result.manifest() != freeze_json(manifest):
            raise ValueError("Selected checkpoint or implementation is incompatible with this plan")
        return result

    def request(self, axis, level):
        return ConvergenceRequest(self.checkpoint, axis, self.maximum_rays,
            self.spatial_samples, self.direction_samples, self.energy_samples,
            self.check_topology, self.checkpoint_spacing_mm, self.maximum_checkpoint_bytes,
            refinement_level=level, threshold_policy=self.thresholds)


def _seal(record):
    result = {k: v for k, v in record.items() if k != "digest"}
    result["digest"] = json_digest(result)
    return freeze_json(result)


def load_journal(path):
    path = Path(path)
    if path.stat().st_size > MAX_RECEIPT_BYTES:
        raise ValueError("Qualification receipt exceeds 64 MiB")
    result = json.loads(path.read_text(encoding="utf-8"))
    _verify_digest(result)
    if result.get("schema") != RECEIPT_SCHEMA:
        raise ValueError("Unrecognized qualification receipt")
    return freeze_json(result)


def _verify_digest(record):
    if record.get("digest") != json_digest({k: v for k, v in record.items() if k != "digest"}):
        raise ValueError("Qualification evidence checksum mismatch")


def _verify_completed(plan, report, axis, level, expected_design):
    """Recompute numerical verdicts, not a trusted saved PASS label."""
    from temsim.working_point_evidence import assess_evidence
    from temsim.topology_evidence import compare_topology
    verdict, _ = assess_evidence(plan.checkpoint, report,
                                implementation=plan.checkpoint.snapshot.implementation)
    if verdict != "MATCHING_NUMERICAL_COMPARISON" or report["axis"] != axis or report.get("refinement_level") != level:
        raise ValueError("Comparison does not match the declared input, method or refinement")
    if freeze_json(report["thresholds"]) != plan.thresholds:
        raise ValueError("Changed tolerance policy cannot reuse this comparison")
    if report["maximum_rays"] != plan.maximum_rays or report["maximum_checkpoint_bytes"] != plan.maximum_checkpoint_bytes:
        raise ValueError("Comparison resource settings do not match the plan")
    request = plan.request(axis, level)
    baseline = request.prepare()
    from temsim.sampling_convergence import refine_state, _sampling_rule
    from temsim.optics.beam_path_audit import optical_component_planes
    start, end = baseline.electron_gun.exit_plane_z_mm, baseline.sample.upper_surface_z_mm
    planes = sorted({float(start), float(end), *(float(z) for _, z in optical_component_planes(baseline) if start < z < end)})
    if len(report["runs"]) != 2:
        raise ValueError("Comparison must contain both executed populations")
    for index, run in enumerate(report["runs"]):
        if index == 1:
            refine_state(baseline, axis)
        snapshot = capture_instrument_snapshot(baseline)
        if run["snapshot_id"] != snapshot.digest or design_identity(snapshot) != expected_design:
            raise ValueError("A physical, quadrature method or numerical input changed")
        if run.get("input_snapshot", {}).get("graph") != snapshot.graph:
            raise ValueError("Actual comparison input graph is missing or incompatible")
        if [row["plane_z_mm"] for row in run["planes"]] != planes:
            raise ValueError("Comparison must retain every exact named physical plane")
        if freeze_json(run["sampling_rule"]) != freeze_json(_sampling_rule(baseline)):
            raise ValueError("Source quadrature method, conditional law or seed policy changed")
        if plan.check_topology != (run["topology"] is not None):
            raise ValueError("Topology participation changed")
    comparison = compare_runs(report["runs"][0]["planes"], report["runs"][1]["planes"], plan.thresholds)
    topology = compare_topology(report["runs"][0]["topology"], report["runs"][1]["topology"])
    comparison["topology"] = topology
    if topology["status"] == "UNRESOLVED":
        comparison["status"] = "UNRESOLVED"
    if freeze_json(comparison) != freeze_json(report["comparison"]):
        raise ValueError("Saved numerical verdict does not match its scalar evidence")
    return comparison["status"]


def run_qualification(plan, *, previous=None, path=None, cancelled=lambda: False,
                      progress=lambda message: None, evidence=lambda report: None):
    """Run only unfinished, compatible comparisons; journal every boundary.

    Completed comparisons survive cancel/error/budget stops. A interrupted
    process has an unknown elapsed budget and is read-only, not a free retry.
    """
    start = perf_counter()
    manifest = plan.manifest()
    if path is not None and previous is None and Path(path).exists():
        raise ValueError("Receipt already exists; explicitly resume it or choose a new file")
    # Preflight all declared refinements without executing particles. Unsupported
    # later axes must not spend earlier runs before their incompatibility is known.
    designs = set()
    for axis in plan.axes:
        for level in range(plan.refinements):
            state = plan.request(axis, level).prepare()
            designs.add(design_identity(capture_instrument_snapshot(state)))
            del state
    if len(designs) != 1:
        raise ValueError("Axes use different quadrature methods. Capture an explicit product-sampling input design before combining them")
    design_id = designs.pop()
    if previous is not None:
        previous = freeze_json(previous)
        _verify_digest(previous)
        if previous.get("schema") != RECEIPT_SCHEMA or previous.get("plan_id") != plan.identity or previous.get("plan") != manifest:
            raise ValueError("Inputs, implementation, tolerance or budget changed; start a separate plan")
        if previous.get("physical_design_id") != design_id:
            raise ValueError("Physical design identity changed")
        record = thaw_json(previous)
        if record["status"] == "RUNNING":
            raise ValueError("Interrupted execution has unknown elapsed time; retained evidence is read-only. Start a new budgeted plan")
    else:
        record = dict(schema=RECEIPT_SCHEMA, plan_id=plan.identity, plan=thaw_json(manifest),
            physical_design_id=design_id, numerical_recipe_id=plan.identity,
            created_at_utc=datetime.now(timezone.utc).isoformat(), status="NOT_RUN",
            completed=[], attempts=[], elapsed_s=0., physical_validation="NOT_ESTABLISHED",
            full_simulator_qualification="NOT_ESTABLISHED")
    prior_elapsed = float(record["elapsed_s"])
    if not math.isfinite(prior_elapsed) or prior_elapsed < 0:
        raise ValueError("Invalid recorded wall-time use")
    schedule = [(axis, level) for axis in plan.axes for level in range(plan.refinements)]
    if len(record["completed"]) > len(schedule) or len(record["attempts"]) > plan.maximum_comparisons:
        raise ValueError("Invalid recorded evaluation budget")
    for index, report in enumerate(record["completed"]):
        axis, level = schedule[index]
        status = _verify_completed(plan, freeze_json(report), axis, level, design_id)
        if status != "STABLE_FOR_CHECKED_AXIS":
            if index != len(record["completed"])-1:
                raise ValueError("An unresolved comparison cannot be followed by qualified work")
            record.update(status="UNRESOLVED", reason="Retained comparison remains unresolved")
            return _seal(record)

    def publish(status, reason):
        record.update(status=status, reason=reason, elapsed_s=prior_elapsed+perf_counter()-start)
        result = _seal(record)
        if path is not None:
            size = len(json.dumps(thaw_json(result), allow_nan=False).encode("utf-8"))
            if size > MAX_RECEIPT_BYTES:
                raise ValueError("Scalar qualification journal exceeded its 64 MiB bound")
            write_evidence(result, path)
        return result

    def deadline():
        return prior_elapsed+perf_counter()-start >= plan.wall_seconds

    for axis, level in schedule[len(record["completed"]):]:
        if cancelled():
            return publish("CANCELLED", "Completed evidence retained; no partial comparison admitted")
        if deadline() or len(record["attempts"]) >= plan.maximum_comparisons:
            return publish("BUDGET_EXHAUSTED", "Declared cumulative time or comparison budget exhausted")
        attempt = dict(axis=axis, refinement_level=level, status="RUNNING")
        record["attempts"].append(attempt)
        publish("RUNNING", f"{axis}: refinement {level+1}/{plan.refinements}")
        try:
            progress(f"{axis}: refinement {level+1}/{plan.refinements}")
            report = run_convergence(plan.request(axis, level),
                cancelled=lambda: cancelled() or deadline(), progress=progress)
            status = _verify_completed(plan, report, axis, level, design_id)
        except SamplingCancelled:
            attempt["status"] = "CANCELLED"
            return publish("BUDGET_EXHAUSTED" if deadline() else "CANCELLED",
                           "Stopped at a solver boundary; completed comparisons retained")
        except Exception as exc:
            attempt.update(status="FAILED", error_type=type(exc).__name__, reason=str(exc))
            return publish("FAILED", f"{type(exc).__name__}: {exc}")
        attempt["status"] = "COMPLETE"
        record["completed"].append(thaw_json(report))
        publish("IN_PROGRESS", "Completed scalar evidence saved")
        evidence(report)
        if status != "STABLE_FOR_CHECKED_AXIS":
            return publish("UNRESOLVED", "Insufficient or unstable evidence; tolerances unchanged")
    return publish("NUMERICALLY_CHECKED_FOR_DECLARED_SCOPE",
        "All declared multi-level comparisons passed; not physical, OEM, image or full-model qualification")
