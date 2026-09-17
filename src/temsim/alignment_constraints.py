"""Optional joint constraints on executed tip-origin condenser transport.

Numerical bounds describe the requested search, not OEM hardware ratings.
Every trial has a detached complete snapshot; only registered lens controls
vary. No flux scaling or surrogate specimen-plane source is introduced.
"""
from dataclasses import asdict, dataclass
import math
from types import MappingProxyType

import numpy as np
from scipy.optimize import least_squares

from temsim.immutable_json import json_digest

METRICS = {
    "diameter95_um": "um", "alpha95_mrad": "mrad", "current_pa": "pA",
    "centroid_x_um": "um", "centroid_y_um": "um", "waist_offset_mm": "mm",
}


@dataclass(frozen=True)
class BeamConstraint:
    metric: str
    minimum: float | None = None
    maximum: float | None = None
    scale: float = 1.
    weight: float = 1.

    def __post_init__(self):
        if self.metric not in METRICS or (self.minimum is None and self.maximum is None):
            raise ValueError("Choose a supported beam observable and at least one bound")
        if any(not math.isfinite(float(v)) for v in (self.scale, self.weight, self.minimum, self.maximum) if v is not None):
            raise ValueError("Beam constraints require finite values")
        if self.scale <= 0 or self.weight <= 0:
            raise ValueError("Constraint scale and weight must be positive")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("Constraint minimum cannot exceed maximum")

    def residual(self, value):
        if value is None or not math.isfinite(value):
            return 1.e6
        violation = max(0., (self.minimum - value) if self.minimum is not None else 0.,
                        (value - self.maximum) if self.maximum is not None else 0.)
        return violation / self.scale * math.sqrt(self.weight)


@dataclass(frozen=True)
class ConstrainedAlignment:
    constraints: tuple[BeamConstraint, ...]
    bounds: object
    maximum_evaluations: int = 32
    minimum_effective_samples: float = 16.
    check_topology: bool = False

    def __post_init__(self):
        if type(self.maximum_evaluations) is not int or not 4 <= self.maximum_evaluations <= 256:
            raise ValueError("Joint alignment allows 4 to 256 trial evaluations")
        if not math.isfinite(self.minimum_effective_samples) or self.minimum_effective_samples < 2:
            raise ValueError("At least two effective samples are required; this does not establish convergence")
        if type(self.check_topology) is not bool:
            raise ValueError("Topology participation must be explicit")
        constraints = tuple(self.constraints)
        if any(not isinstance(c, BeamConstraint) for c in constraints):
            raise TypeError("Expected declared beam constraints")
        if len({c.metric for c in constraints}) != len(constraints):
            raise ValueError("A beam observable can have only one constraint interval")
        bounds = {str(k): tuple(float(v) for v in pair) for k, pair in self.bounds.items()}
        if not bounds or any(len(pair) != 2 or not all(math.isfinite(v) for v in pair)
                             or pair[0] < 0 or pair[0] >= pair[1] for pair in bounds.values()):
            raise ValueError("Declare finite nonnegative ordered lens search bounds")
        object.__setattr__(self, "bounds", MappingProxyType(bounds))
        object.__setattr__(self, "constraints", constraints)

    def to_dict(self):
        return dict(constraints=[asdict(c) for c in self.constraints], bounds=dict(self.bounds),
                    maximum_evaluations=self.maximum_evaluations,
                    minimum_effective_samples=self.minimum_effective_samples,
                    check_topology=self.check_topology)

    @property
    def digest(self):
        return json_digest(self.to_dict())


def constraint_assessment(options, metrics):
    return tuple(dict(metric=c.metric, unit=METRICS[c.metric], value=metrics.get(c.metric),
                      minimum=c.minimum, maximum=c.maximum, scale=c.scale, weight=c.weight,
                      residual=c.residual(metrics.get(c.metric)),
                      passed=c.residual(metrics.get(c.metric)) == 0.) for c in options.constraints)


def evaluate_incident(state, *, topology=False, spacing_mm=2., exact_roots=False):
    """Execute the same gun, fields and stops as the incident audit."""
    from temsim.optics.beam_path_audit import (incident_checkpoints, crossover_candidates,
        crossover_intervals, optical_component_planes, refine_crossover_candidate)
    from temsim.physics.beam_current import effective_source_current_a
    from temsim.physics.beam_statistics import transverse_beam_statistics
    from temsim.sampling_diagnostics import sampling_summary
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.topology_evidence import topology_reference, topology_run
    # The shared forward solver maintains operational readbacks and caches.
    # Own those mutations even when this observer is called directly by a UI.
    cancellation = getattr(state, "_tuning_cancelled", None)
    state = capture_instrument_snapshot(state).restore()
    if cancellation is not None:
        state._tuning_cancelled = cancellation
    start, end = float(state.electron_gun.exit_plane_z_mm), float(state.sample.upper_surface_z_mm)
    planes = sorted({end, *np.arange(start, end, spacing_mm)}) if topology else [end]
    if len(planes) > 8192 or len(planes) * state.electron_gun.emitter.ray_count * 128 > 256 * 1024**2:
        raise ValueError("Topology audit exceeds its 256 MiB / 8192-plane numerical budget")
    gun, checkpoints, mask = incident_checkpoints(state, planes, step_mm=state.step_mm)
    arrays = {name: getattr(checkpoints, name)[-1].copy() for name in ("x_m", "y_m", "tx_rad", "ty_rad")}
    arrays.update(weight=gun.exit_bundle.weight, alive=mask[-1].copy(),
                  energy_offset_ev=gun.exit_bundle.energy_offset_ev, gun_ray_id=gun.exit_bundle.ray_id)
    summary = sampling_summary(arrays, plane_z_mm=end, source_current_a=effective_source_current_a(state))
    summary = {**summary, "actual_backend": str(getattr(state, "active_backend", "not_recorded"))}
    metrics = dict(effective_samples=summary["effective_samples"] or 0.,
                   current_pa=summary["plane_current_a"] * 1.e12 if summary["plane_current_a"] is not None else None)
    if summary["status"] == "AVAILABLE":
        stats = transverse_beam_statistics(arrays["x_m"], arrays["y_m"], arrays["tx_rad"], arrays["ty_rad"],
                                           alive=arrays["alive"], weights=arrays["weight"])
        metrics.update(diameter95_um=stats.illumination_diameter_95_um, alpha95_mrad=stats.convergence_95_mrad,
                       alpha99_mrad=stats.convergence_99_mrad, centroid_x_um=stats.mean_x_m * 1.e6,
                       centroid_y_um=stats.mean_y_m * 1.e6, waist_offset_mm=stats.waist_offset_m * 1.e3,
                       curvature_per_m=stats.radial_wavefront_curvature_per_m)
        metrics.update(mean_tx_mrad=stats.mean_tx_rad * 1.e3, mean_ty_mrad=stats.mean_ty_rad * 1.e3)
    proposals = crossover_candidates(checkpoints, mask, gun.exit_bundle.weight) if topology else []
    del checkpoints, mask, gun
    topo = None
    if topology:
        reference = topology_reference(capture_instrument_snapshot(state))
        components = optical_component_planes(state)
        if reference["status"] != "TARGET_ONLY":
            raise ValueError("No scoped crossover reference for this captured assembly and mode")
        if len(proposals) > 32:
            raise ValueError("Topology audit exceeds the 32-root numerical budget")
        if exact_roots:
            roots = [refine_crossover_candidate(state, p, step_mm=state.step_mm) for p in proposals]
            topo = topology_run(roots, components, surface_mm=end, focus_tolerance_nm=1., reference=reference)
        else:
            # Search ranking only. Root proposals never pass final acceptance.
            from temsim.optics.beam_path_audit import partition_surface_crossovers
            intermediate, terminal = partition_surface_crossovers(proposals, end, tolerance_nm=1.)
            intervals = crossover_intervals([p["z_mm"] for p in intermediate], components)
            topo = dict(reference_match=(len(intermediate) == reference["reference"]["count"]
                         and json_digest(intervals) == json_digest(reference["reference"]["intervals"])),
                        status="PROPOSALS_ONLY")
    return metrics, arrays, summary, topo


def solve_constrained_candidate(request, *, cancelled):
    from temsim.alignment_transaction import AlignmentCandidate, AlignmentCancelled, _allowed_state
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.operating_modes import direct_alignment_by_key
    from temsim.optics.direct_alignment import DirectAlignmentResult
    from temsim.calculation_cache import calculation_signatures
    from temsim.working_point import WorkingPointCheckpoint
    definition = direct_alignment_by_key(request.key)
    options = request.options
    state = request.start_snapshot.restore()
    originals = {lens.key: lens.percent for lens in state.lenses if lens.key in definition.devices}
    keys = tuple(options.bounds)
    lower = np.array([options.bounds[k][0] for k in keys])
    upper = np.array([options.bounds[k][1] for k in keys])
    initial = np.clip([originals[k] for k in keys], lower, upper)
    observed = {}
    failures = []
    evaluations_allowed = max(1, options.maximum_evaluations - len(keys))
    target_metric = "alpha95_mrad" if request.key == "nanoprobe_convergence" else "diameter95_um"
    tolerance = float(definition.targets["maximum_relative_error"])

    def check():
        if cancelled():
            raise AlignmentCancelled("Joint alignment cancelled; previous working point retained")

    def residual(metrics, topo):
        value = metrics.get(target_metric)
        target_error = math.log(max(value, 1.e-30) / request.target) / tolerance if value is not None else 1.e6
        if request.key == "nanoprobe_convergence":
            physical = [max(0., abs(metrics.get("waist_offset_mm", 1.e6)) /
                           float(definition.targets["maximum_waist_offset_mm"]) - 1.)]
        else:
            physical = [max(0., abs(metrics.get("curvature_per_m", 1.e6)) /
                           float(definition.targets["maximum_curvature_per_m"]) - 1.),
                        max(0., metrics.get("alpha99_mrad", 1.e6) /
                           float(definition.targets["maximum_convergence_99_mrad"]) - 1.)]
        support = max(0., options.minimum_effective_samples - metrics.get("effective_samples", 0.))
        return np.array([target_error, *physical, support,
                         *(c.residual(metrics.get(c.metric)) for c in options.constraints),
                         *([0. if topo and topo["reference_match"] else 10.] if options.check_topology else [])])

    class BudgetReached(Exception):
        pass

    def trial(vector):
        check()
        key = tuple(float(v) for v in vector)
        if key in observed:
            return observed[key][0]
        if len(observed) >= evaluations_allowed:
            raise BudgetReached()
        strengths = {**originals, **dict(zip(keys, key, strict=True))}
        scratch = _allowed_state(request, strengths)
        scratch._tuning_cancelled = cancelled
        try:
            metrics, arrays, summary, topo = evaluate_incident(scratch, topology=options.check_topology)
            del arrays
            values = residual(metrics, topo)
        except (ValueError, RuntimeError) as exc:
            check()
            metrics, topo = {}, None
            values = residual(metrics, topo)
            failures.append(dict(controls=strengths, reason=str(exc)))
        observed[key] = (values, metrics, topo, strengths)
        return values

    # Bounded local search plus independent deterministic starting points.
    for seed in (initial, lower + .25 * (upper-lower), lower + .75 * (upper-lower)):
        try:
            least_squares(trial, seed, bounds=(lower, upper), diff_step=1.e-3,
                          max_nfev=options.maximum_evaluations, x_scale=np.maximum(upper-lower, 1.))
        except BudgetReached:
            break
    check()
    vector, best = min(observed.items(), key=lambda pair: float(np.linalg.norm(pair[1][0])))
    evaluations_allowed = options.maximum_evaluations
    # Independently observed local control authority, using signed observables
    # rather than inactive inequality penalties or optimizer regularisation.
    authority_metrics = {target_metric: max(request.target * tolerance, 1.e-12),
                         "waist_offset_mm": float(definition.targets.get("maximum_waist_offset_mm", .002))}
    authority_metrics.update({c.metric: c.scale for c in options.constraints})
    jacobian = np.zeros((len(authority_metrics), len(keys)))
    for column in range(len(keys)):
        shifted = np.array(vector)
        delta = .001 * (upper[column] - lower[column])
        shifted[column] += delta if shifted[column] + delta <= upper[column] else -delta
        trial(shifted)
        measured = observed[tuple(shifted)][1]
        for row, (metric, scale) in enumerate(authority_metrics.items()):
            if measured.get(metric) is not None and best[1].get(metric) is not None:
                jacobian[row, column] = ((measured[metric] - best[1][metric]) / scale /
                    ((shifted[column] - vector[column]) / (upper[column] - lower[column])))
    rank = int(np.linalg.matrix_rank(jacobian))
    condition = float(np.linalg.cond(jacobian))
    authority = rank == len(keys) and math.isfinite(condition) and condition <= 1.e8
    strengths = best[3]
    candidate = _allowed_state(request, strengths)
    candidate_snapshot = capture_instrument_snapshot(candidate)
    passes, refinements = [], []
    final_arrays = final_summary = final_state = None
    # Column and gun step checks are independent, and use the full tip chain.
    for axis in ("configured", "column_step", "gun_step"):
        check()
        refined = _allowed_state(request, strengths)
        if axis == "column_step":
            refined.step_mm *= .5
        elif axis == "gun_step":
            refined.electron_gun.trace_step_mm *= .5
        refined._tuning_cancelled = cancelled
        try:
            metrics, arrays, summary, topo = evaluate_incident(refined, topology=options.check_topology,
                spacing_mm=1. if axis == "column_step" else 2., exact_roots=True)
            values = residual(metrics, topo)
            passed = abs(values[0]) <= 1. and np.all(values[1:] == 0.) and metrics.get("current_pa", 0.) > 0
            refinements.append(dict(axis=axis, metrics=metrics, constraints=constraint_assessment(options, metrics),
                                    topology=topo, passed=bool(passed), snapshot_id=capture_instrument_snapshot(refined).digest,
                                    actual_backend=summary.get("actual_backend", "not_recorded")))
            passes.append(bool(passed))
            final_arrays, final_summary, final_state = arrays, summary, refined
        except (ValueError, RuntimeError) as exc:
            check()
            passes.append(False)
            refinements.append(dict(axis=axis, passed=False, failure=str(exc), metrics={}))
    check()
    baseline = refinements[0]["metrics"]
    spread = max((abs(r["metrics"].get(target_metric, 1.e30) - baseline.get(target_metric, 0.)) /
                  max(abs(baseline.get(target_metric, 0.)), 1.e-30) for r in refinements[1:]), default=1.e30)
    stable = spread <= float(definition.targets["maximum_numerical_spread"])
    metric_spreads = {}
    for metric, scale in authority_metrics.items():
        values = [row["metrics"].get(metric) for row in refinements]
        delta = (max(values) - min(values)) / max(scale, *(abs(v) for v in values)) if all(v is not None for v in values) else None
        metric_spreads[metric] = delta
        stable &= delta is not None and delta <= float(definition.targets["maximum_numerical_spread"])
    support = all(r["metrics"].get("effective_samples", 0.) >= options.minimum_effective_samples for r in refinements)
    valid = all(passes) and stable and authority
    final_metrics = refinements[-1]["metrics"]
    target_met = abs(residual(final_metrics, None)[0]) <= 1.
    failure = ("INSUFFICIENT_SAMPLED_TRANSMISSION" if not support else
               "UNAVAILABLE_CONTROL_AUTHORITY" if rank < len(keys) else "ILL_CONDITIONED_SOLVE" if not authority else
               "NUMERICAL_CHECK_FAILED" if not stable else "TARGET_MISSED" if not target_met else "CONSTRAINT_VIOLATED")
    checkpoint = None
    if final_state is not None:
        snapshot = capture_instrument_snapshot(final_state)
        checkpoint = WorkingPointCheckpoint(snapshot, final_arrays, final_summary["plane_z_mm"],
            calculation_signatures(final_state)["incident"],
            dict(source_representation="gun-derived-particles", source_current_a=final_summary["source_current_a"],
                 coordinate_precision="exact-integration-checkpoint", validation_status="NOT_RUN",
                 phase_status="NOT_COMPUTED", alignment_options=options.to_dict()), request.start_snapshot.digest)
    message = ("Joint constraints and independent gun/column step checks passed." if valid else
               f"{failure}: no acceptable candidate in this bounded search; no changes applied.")
    result = DirectAlignmentResult(key=request.key, success=valid, requested=request.target,
        achieved=final_metrics.get(target_metric, math.nan), unit=definition.unit,
        constraint_value=final_metrics.get("waist_offset_mm" if request.key == "nanoprobe_convergence" else "curvature_per_m", math.nan),
        constraint_unit="mm" if request.key == "nanoprobe_convergence" else "1/m",
        strengths=strengths, iterations=len(observed), validation_step_mm=candidate.step_mm / 2.,
        numerical_spread=spread, message=message, convergence_95_mrad=final_metrics.get("alpha95_mrad"),
        convergence_99_mrad=final_metrics.get("alpha99_mrad"), illumination_diameter_95_um=final_metrics.get("diameter95_um"))
    validation = dict(status="PASS" if valid else failure, options_id=options.digest,
        target=request.target, target_metric=target_metric, allowed_controls=list(keys),
        fixed_controls="All captured inputs except the listed lens excitations",
        bounds=dict(options.bounds), bounds_scope="Explicit numerical search bounds; no OEM hardware limit asserted",
        source_frozen=True, refinements=refinements, search_failures=failures, trial_count=len(observed),
        numerical_spread=spread, independent_reference="NOT_RUN",
        metric_step_spreads=metric_spreads,
        control_authority=dict(rank=rank, control_count=len(keys), condition=condition if math.isfinite(condition) else None,
            normalized_jacobian=jacobian.tolist(), observables=list(authority_metrics),
            scope="Local finite differences through executed tip-origin transport; not global attainability"),
        candidate_snapshot=candidate_snapshot.to_dict(),
        forward_snapshot_id=checkpoint.snapshot.digest if checkpoint else None,
        qualification="No sampling, field-mesh or physical-source qualification inferred")
    return AlignmentCandidate(request, result, checkpoint, validation, "READY_TO_APPLY" if valid else "FAILED")
