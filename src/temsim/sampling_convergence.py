"""Bounded, detached classical convergence checks using the existing audit.

Only independently exposed numerical controls are refined. A stable pair is
not whole-model qualification; source support, current and optics stay fixed.
"""
from __future__ import annotations

from dataclasses import dataclass, replace, asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from temsim import input_io
from time import perf_counter
import tomllib

import numpy as np

from temsim.immutable_json import freeze_json, json_digest, thaw_json
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.sampling_diagnostics import sampling_summary
from temsim.working_point import snapshot_changes

AXES = {
    "gun_step": "Gun integration step / 2",
    "column_step": "Column integration step / 2",
    "emission_samples": "Total emission samples x 2 (joint quadrature)",
    "spatial": "Tip positions x 2 (fixed direction and energy factors)",
    "directions": "Local directions x 2 (fixed position and energy factors)",
    "energies": "Conditional energy samples x 2 (fixed position and direction factors)",
    "field_radial": "Grounded gun radial mesh nodes x 2",
    "field_axial": "Grounded gun axial mesh nodes x 2",
    "field_apex": "Grounded gun apex cells per radius x 2",
    "field_domain": "Grounded gun outer numerical boundary factor x 2",
    "checkpoint_spacing": "Crossover bracket spacing / 2 (fixed transport step)",
    "joint_steps": "Joint gun + column steps / 2 (after both separate checks)",
}
UNAVAILABLE_AXES = {
    "Other sources": "Independent product quadrature requires the existing classical Cold FEG tip; unsupported source laws remain explicit.",
    "Other field grids": "Gun mesh refinements require an active grounded surface model. Imported fixed maps cannot be refined without their generating solver.",
    "Full qualification": "Individual comparisons do not qualify all axes, crossover topology, physical source calibration or sample/detector physics.",
}
_PRODUCT_AXES = {"spatial", "directions", "energies"}
_FIELD_AXES = {"field_radial": "radial_nodes", "field_axial": "axial_nodes", "field_apex": "apex_cells_per_radius",
               "field_domain": "outer_radius_factor"}


class SamplingCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class ConvergenceRequest:
    checkpoint: object
    axis: str
    maximum_rays: int = 4096
    spatial_samples: int = 9
    direction_samples: int = 9
    energy_samples: int = 9
    check_topology: bool = False
    checkpoint_spacing_mm: float = 2.0
    maximum_checkpoint_bytes: int = 512*1024**2
    prior_evidence: tuple = ()
    refinement_level: int = 0
    threshold_policy: object = None

    def prepare(self):
        if self.axis not in AXES:
            raise ValueError("This independent refinement is not implemented")
        if type(self.refinement_level) is not int or not 0 <= self.refinement_level <= 7:
            raise ValueError("Refinement level must be between zero and seven")
        if type(self.maximum_rays) is not int or not 9 <= self.maximum_rays <= 65536:
            raise ValueError("Choose a ray budget from 9 to 65536")
        if type(self.check_topology) is not bool:
            raise ValueError("Topology participation must be explicit")
        if not np.isfinite(self.checkpoint_spacing_mm) or self.checkpoint_spacing_mm <= 0:
            raise ValueError("Crossover bracket spacing must be finite and positive")
        if type(self.maximum_checkpoint_bytes) is not int or not 1024**2 <= self.maximum_checkpoint_bytes <= 32*1024**3:
            raise ValueError("Choose a checkpoint memory bound from 1 MiB to 32 GiB")
        if self.axis == "checkpoint_spacing" and not self.check_topology:
            raise ValueError("Bracket refinement requires crossover topology checking")
        state = self.checkpoint.snapshot.restore()
        from temsim.optics.electron_gun.source_policy import require_physical_gun_source
        require_physical_gun_source(state.electron_gun)
        emitter = state.electron_gun.emitter
        surface = getattr(emitter, "surface_model", None)
        if getattr(emitter, "coherence", None) is not None or (surface is not None and surface.coherence is not None):
            raise ValueError("Coherent tip-to-column development is paused; this assistant runs classical particles only")
        if bool(getattr(getattr(state, "vacuum_map", None), "enabled", False)):
            raise ValueError("The incident audit does not qualify active vacuum scattering; the saved on/off choice is preserved")
        if self.axis in {"gun_step", "joint_steps"} and not hasattr(state.electron_gun, "trace_step_mm"):
            raise ValueError("This source has no supported gun-step control")
        if self.axis == "joint_steps":
            from temsim.working_point_evidence import assess_evidence
            checked = set()
            for report in self.prior_evidence:
                verdict, _ = assess_evidence(self.checkpoint, report, implementation=self.checkpoint.snapshot.implementation)
                if verdict == "MATCHING_NUMERICAL_COMPARISON" and not report.get("selection_to_baseline_changes"):
                    checked.add(report["axis"])
            if not {"gun_step", "column_step"} <= checked:
                raise ValueError("Run the separate gun-step and column-step checks for these exact inputs before the joint check")
        if self.axis in _PRODUCT_AXES:
            from temsim.optics.electron_gun.emitter import ColdFieldEmitter, EmissionQuadrature
            if not isinstance(emitter, ColdFieldEmitter):
                raise ValueError("Independent product sampling is not implemented for this source law")
            emitter.quadrature = EmissionQuadrature(self.spatial_samples, self.direction_samples, self.energy_samples)
            emitter.ray_count = emitter.quadrature.total
            if surface is not None and surface.emission.spatial_sampling == "apex_stratified_v1" and self.spatial_samples < 9:
                raise ValueError("Full-cap stratification requires at least nine spatial samples")
        if self.axis in _FIELD_AXES:
            if surface is None:
                raise ValueError("Field-grid refinement requires an active grounded tip surface model")
            key = _FIELD_AXES[self.axis]
            replace(surface.field_numerics, **{key: getattr(surface.field_numerics, key)*2}).validate()
        if self.axis == "emission_samples" and getattr(emitter, "quadrature", None) is not None:
            raise ValueError("Select an independent product factor explicitly; total count cannot truncate a product quadrature")
        for _ in range(self.refinement_level):
            refine_state(state, self.axis)
        if self.axis in _FIELD_AXES:
            key = _FIELD_AXES[self.axis]
            numerics = state.electron_gun.emitter.surface_model.field_numerics
            replace(numerics, **{key: getattr(numerics, key)*2}).validate()
        count = int(emitter.ray_count)
        if count < 9 or count * (2 if self.axis in _PRODUCT_AXES or self.axis == "emission_samples" else 1) > self.maximum_rays:
            raise ValueError("The two-run comparison exceeds the selected per-run ray budget")
        if not np.isfinite(state.step_mm) or state.step_mm <= 0:
            raise ValueError("A finite positive column integration step is required")
        span = float(state.sample.upper_surface_z_mm - state.electron_gun.exit_plane_z_mm)
        minimum_step = state.step_mm * (.5 if self.axis in {"column_step", "joint_steps"} else 1)
        if span / minimum_step > 2_000_000:
            raise ValueError("The requested comparison exceeds the bounded column-step budget; no settings were changed")
        return state


def refine_state(state, axis):
    """One existing numerical refinement; never change physical source support."""
    if axis in {"gun_step", "joint_steps"}:
        state.electron_gun.trace_step_mm *= .5
    if axis in {"column_step", "joint_steps"}:
        state.step_mm *= .5
    emitter = state.electron_gun.emitter
    if axis in _PRODUCT_AXES:
        emitter.quadrature = replace(emitter.quadrature,
            **{axis: getattr(emitter.quadrature, axis)*2})
        emitter.ray_count = emitter.quadrature.total
    elif axis in _FIELD_AXES:
        key = _FIELD_AXES[axis]
        numerics = emitter.surface_model.field_numerics
        refined = replace(numerics, **{key: getattr(numerics, key)*2})
        refined.validate()
        emitter.surface_model = replace(emitter.surface_model, field_numerics=refined)
    elif axis == "emission_samples":
        emitter.ray_count *= 2


def _thresholds():
    from temsim.paths import OPERATING_MODE_CONFIG_ROOT
    path = Path(OPERATING_MODE_CONFIG_ROOT) / "illumination_targets.toml"
    content = input_io.read_bytes(path)
    targets = tomllib.loads(content.decode("utf-8"))
    relative = min(float(targets[key]["relative_tolerance"]) for key in ("micro_probe", "nano_probe"))
    if not np.isfinite(relative) or relative <= 0:
        raise ValueError("Invalid configured convergence tolerance")
    # Absolute floors describe numerical comparison resolution, not physical
    # acceptance targets. They are fixed before either calculation executes.
    return dict(relative=relative, absolute={"transmission": 1e-12, "diameter95_m": 1e-15,
        "alpha95_rad": 1e-12, "centroid_x_m": 1e-12, "centroid_y_m": 1e-12},
        minimum_transmitted_samples=16, minimum_effective_samples=16,
        focus_tolerance_nm=min(float(targets[key]["focus_tolerance_nm"]) for key in ("micro_probe", "nano_probe")),
        source=str(path), source_sha256=sha256(content).hexdigest(),
        interpretation="Numerical pair comparison only; N_eff and sample minima are necessary screens, never proof of convergence")


def _sampling_rule(state):
    emitter = state.electron_gun.emitter
    surface = getattr(emitter, "surface_model", None)
    quadrature = {}
    if surface is not None:
        for name in ("directions_per_position", "spatial_sampling", "spatial_stratum_allocation",
                     "angular_sampling", "angular_refinement_gain", "angular_refinement_width_sigma",
                     "angular_stratum_allocation"):
            quadrature[name] = getattr(surface.emission, name)
    return dict(emitter_type=type(emitter).__module__ + ":" + type(emitter).__name__,
        emitted_ray_budget=int(emitter.ray_count),
        sequence="Existing deterministic Halton emitter; exact rule bound to implementation identity",
        random_seed=None, seed_policy="No randomized emission replication or IID error estimate",
        independent_product=None if getattr(emitter, "quadrature", None) is None else asdict(emitter.quadrature),
        conditional_law="Surface energy given local direction preserves the specified joint emission law; legacy flat marginals retain their existing distribution",
        surface_quadrature=quadrature)


def compare_runs(before, after, thresholds):
    """Compare the same named planes; report the earliest evaluated divergence."""
    if len(before) != len(after) or any(a["plane_z_mm"] != b["plane_z_mm"] for a, b in zip(before, after)):
        raise ValueError("Refinement records must use identical physical planes")
    comparisons = []
    for left, right in zip(before, after):
        failures = []
        differences = {}
        supported = all(row["status"] == "AVAILABLE" and
            row["transmitted_samples"] >= thresholds["minimum_transmitted_samples"] and
            row["effective_samples"] >= thresholds["minimum_effective_samples"] for row in (left, right))
        for key, absolute in thresholds["absolute"].items():
            a, b = left.get(key), right.get(key)
            if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
                failures.append(key)
                continue
            delta, limit = abs(a-b), max(absolute, thresholds["relative"] * max(abs(a), abs(b)))
            differences[key] = dict(absolute_difference=delta, limit=limit)
            if delta > limit:
                failures.append(key)
        comparisons.append(dict(plane_z_mm=left["plane_z_mm"], differences=differences,
            failed_observables=failures, status="INSUFFICIENT_SUPPORT" if not supported else
            "DIVERGED" if failures else "STABLE_FOR_CHECKED_AXIS"))
    first = next((row["plane_z_mm"] for row in comparisons if row["status"] != "STABLE_FOR_CHECKED_AXIS"), None)
    return dict(planes=comparisons, first_unresolved_plane_mm=first,
                status="UNRESOLVED" if first is not None else "STABLE_FOR_CHECKED_AXIS")


@input_io.using_state_inputs
def run_convergence(request, *, cancelled=lambda: False, progress=lambda message: None):
    from temsim.optics.beam_path_audit import incident_checkpoints, optical_component_planes
    from temsim.physics.beam_current import effective_source_current_a

    def check_cancelled():
        if cancelled():
            raise SamplingCancelled("Sampling comparison cancelled; the previous complete result is preserved")

    check_cancelled()
    state = request.prepare()
    limits = _thresholds() if request.threshold_policy is None else thaw_json(request.threshold_policy)
    if request.threshold_policy is not None:
        from temsim.sampling_qualification import validate_thresholds
        validate_thresholds(limits)
    baseline = capture_instrument_snapshot(state)
    end = float(state.sample.upper_surface_z_mm)
    start = float(state.electron_gun.exit_plane_z_mm)
    planes = sorted({start, end, *(float(z) for _, z in optical_component_planes(state) if start < z < end)})
    if len(planes) > 256:
        raise ValueError("The incident audit exceeds the 256-plane bound")
    source_current = effective_source_current_a(state)
    from temsim.topology_evidence import topology_reference, topology_run, compare_topology
    reference = topology_reference(baseline)
    # Estimate stored coordinates, masks and integration workspace conservatively
    # before tracing. This is a bound for this audit, not all application memory.
    spacing_base = request.checkpoint_spacing_mm * (.5**request.refinement_level if request.axis == "checkpoint_spacing" else 1.)
    spacing = spacing_base*(.5 if request.axis == "checkpoint_spacing" else 1.)
    point_count = int(np.ceil((end-start)/spacing))+len(planes)+1 if request.check_topology else len(planes)
    rays = int(state.electron_gun.emitter.ray_count)*(2 if request.axis in _PRODUCT_AXES or request.axis == "emission_samples" else 1)
    if point_count > 8192 or point_count*rays*128 > request.maximum_checkpoint_bytes:
        raise ValueError("Audit checkpoints exceed the declared plane or memory budget")
    runs = []
    for variant in ("baseline", "refined"):
        check_cancelled()
        state = baseline.restore()
        if variant == "refined":
            refine_state(state, request.axis)
        snapshot = capture_instrument_snapshot(state)
        if effective_source_current_a(state) != source_current:
            raise ValueError("Numerical refinement changed the physical source current")
        state._tuning_cancelled = cancelled
        progress(f"{variant.capitalize()}: physical tip to specimen entrance ({request.axis})")
        started = perf_counter()
        spacing = spacing_base*(.5 if variant == "refined" and request.axis == "checkpoint_spacing" else 1.)
        audit_planes = sorted({*planes, *np.arange(start, end, spacing)}) if request.check_topology else planes
        try:
            gun, checkpoints, mask = incident_checkpoints(state, audit_planes, step_mm=state.step_mm)
        except Exception:
            check_cancelled()
            raise
        check_cancelled()
        rows = []
        # Scalar transport comparisons retain the same component planes even
        # when the separately reported root bracketing mesh is refined.
        for index, plane in enumerate(checkpoints.z_mm):
            if float(plane) not in planes:
                continue
            arrays = {key: getattr(checkpoints, key)[index] for key in ("x_m", "y_m", "tx_rad", "ty_rad")}
            arrays.update(weight=gun.exit_bundle.weight, alive=mask[index])
            rows.append(thaw_json(sampling_summary(arrays, plane_z_mm=plane, source_current_a=source_current)))
        proposals = []
        if request.check_topology:
            from temsim.optics.beam_path_audit import crossover_candidates
            proposals = crossover_candidates(checkpoints, mask, gun.exit_bundle.weight)
        # Release the dense path before exact root evaluations and the next run.
        del arrays, gun, checkpoints, mask
        topology = None
        if request.check_topology:
            if len(proposals) > 32:
                raise ValueError("More than 32 crossover proposals exceed this bounded comparison")
            from temsim.optics.beam_path_audit import refine_crossover_candidate
            roots, failures = [], []
            for index, proposal in enumerate(proposals):
                check_cancelled()
                progress(f"{variant.capitalize()}: executing crossover {index+1}/{len(proposals)} from the physical tip")
                try:
                    roots.append(refine_crossover_candidate(state, proposal, step_mm=state.step_mm))
                except (ValueError, RuntimeError) as exc:
                    check_cancelled()
                    failures.append(dict(proposal=proposal, reason=str(exc)))
            topology = topology_run(roots, optical_component_planes(state), surface_mm=end,
                focus_tolerance_nm=limits["focus_tolerance_nm"], reference=reference)
            topology.update(bracket_spacing_mm=spacing, checkpoint_count=len(audit_planes),
                execution_failures=failures)
            if failures:
                topology["status"] = "UNRESOLVED"
        runs.append(dict(kind=variant, snapshot_id=snapshot.digest,
            numerical_identity=json_digest(dict(instrument=snapshot.physical_digest,
                topology_requested=request.check_topology, bracket_spacing_mm=spacing)),
            sampling_rule=_sampling_rule(state),
            input_changes=snapshot_changes(baseline, snapshot), elapsed_s=perf_counter()-started, planes=rows,
            topology=topology))
        if request.threshold_policy is not None:
            # Exact captured graph, without duplicating pinned input-file bytes.
            # Parent snapshot owns those dependencies; this is not an exit source.
            runs[-1]["input_snapshot"] = dict(graph=thaw_json(snapshot.graph),
                external_inputs=[{k: v for k, v in row.items() if k != "content_hex"}
                                 for row in snapshot.external_inputs],
                implementation=snapshot.implementation, digest=snapshot.digest)
        # The receipt owns scalars only. Release the full path before the next
        # independent run; no downstream bundle is used as a new source.
        del state
    check_cancelled()
    comparison = compare_runs(runs[0]["planes"], runs[1]["planes"], limits)
    comparison["topology"] = compare_topology(runs[0]["topology"], runs[1]["topology"])
    if comparison["topology"]["status"] == "UNRESOLVED":
        comparison["status"] = "UNRESOLVED"
    report = dict(schema="incident-convergence-evidence-v1", checkpoint_id=request.checkpoint.digest,
        snapshot_id=request.checkpoint.snapshot.digest, implementation=baseline.implementation,
        created_at_utc=datetime.now(timezone.utc).isoformat(), axis=request.axis,
        refinement_level=request.refinement_level,
        maximum_rays=request.maximum_rays, thresholds=limits, runs=runs, comparison=comparison,
        topology_reference=reference, maximum_checkpoint_bytes=request.maximum_checkpoint_bytes,
        comparison_kind="JOINT_TRANSPORT_STEPS" if request.axis == "joint_steps" else "SINGLE_NUMERICAL_AXIS",
        prior_evidence_ids=[row["digest"] for row in request.prior_evidence] if request.axis == "joint_steps" else [],
        selection_to_baseline_changes=snapshot_changes(request.checkpoint.snapshot, baseline),
        physical_validation="NOT_ESTABLISHED", full_numerical_qualification="NOT_ESTABLISHED",
        sampling_rule=("Gun and column steps are refined together after separate identity-matched checks; prior unresolved findings remain unresolved evidence."
            if request.axis == "joint_steps" else "Only the selected factor is refined. Independent comparisons explicitly use a product/conditional-CDF baseline; legacy total-count refinement is joint."),
        scope="Classical gun and incident column at exact planes through specimen entrance; no sample/detector qualification",
        limitations=[*UNAVAILABLE_AXES.values(), "Topology is optional; fixed-sampling roots and one-axis agreement do not establish complete topology convergence",
                     "Empty sampled transmission is not proof of complete physical blockage"])
    report["digest"] = json_digest(report)
    return freeze_json(report)


def write_evidence(report, path):
    """Atomically export finite scalar JSON, with no numerical arrays."""
    import os
    from tempfile import NamedTemporaryFile
    path = Path(path)
    content = json.dumps(thaw_json(report), indent=2, allow_nan=False) + "\n"
    with NamedTemporaryFile(dir=path.parent, prefix=".sampling-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
