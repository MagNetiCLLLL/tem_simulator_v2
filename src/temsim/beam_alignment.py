"""Direct Alignment of actual paired-deflector controls at specimen entrance."""
from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.optimize import least_squares

from temsim.immutable_json import json_digest
from temsim.operating_modes import DirectAlignmentDefinition

KEY = "beam_centre_direction"
CONTROLS = ("upper_x_mrad", "upper_y_mrad", "lower_x_mrad", "lower_y_mrad")
OBSERVABLES = ("centroid_x_um", "centroid_y_um", "mean_tx_mrad", "mean_ty_mrad")
DEFINITION = DirectAlignmentDefinition(key=KEY, name="Beam centre and direction", family="beam",
    mode_key="nano_probe", unit="scaled residual", minimum=0., maximum=0., default_value=0.,
    devices=tuple("beam_deflector." + name for name in CONTROLS),
    observable="weighted_centroid_and_mean_direction_at_specimen_entrance",
    constraint="captured_source_stops_current_and_independent_numerical_checks",
    calibration_status="Measured local control authority required",
    calibration_reference="Existing paired physical kick planes and executed tip-origin population",
    targets={}, applies_to_modes=("nano_probe", "micro_probe"), state_parameters=())


@dataclass(frozen=True)
class BeamAlignmentOptions:
    targets: tuple[float, float, float, float] = (0., 0., 0., 0.)
    position_tolerance_um: float = .01
    angle_tolerance_mrad: float = .01
    maximum_kick_mrad: float = 10.
    maximum_evaluations: int = 32
    minimum_effective_samples: float = 16.
    minimum_current_pa: float = 0.

    def __post_init__(self):
        object.__setattr__(self, "targets", tuple(float(v) for v in self.targets))
        if len(self.targets) != 4 or any(not math.isfinite(v) for v in self.targets):
            raise ValueError("Specify finite X/Y centre and X/Y mean direction")
        if any(not math.isfinite(v) or v <= 0 for v in
               (self.position_tolerance_um, self.angle_tolerance_mrad, self.maximum_kick_mrad)):
            raise ValueError("Tolerances and numerical kick bounds must be positive")
        if type(self.maximum_evaluations) is not int or not 10 <= self.maximum_evaluations <= 256:
            raise ValueError("Beam alignment permits 10 to 256 search trials")
        if not math.isfinite(self.minimum_effective_samples) or self.minimum_effective_samples < 2:
            raise ValueError("At least two effective samples are required")
        if not math.isfinite(self.minimum_current_pa) or self.minimum_current_pa < 0:
            raise ValueError("Minimum specimen current must be finite and nonnegative")

    def to_dict(self):
        return asdict(self)

    @property
    def digest(self):
        return json_digest(self.to_dict())


def capability(state):
    if state.electron_gun.source_representation != "classical_particles":
        return False, "Beam alignment currently requires classical tip particles; coherent development remains paused"
    component = getattr(state, "beam_deflector", None)
    if component is None or not component.enabled:
        return False, "The physical Beam Shift/Tilt Deflector is absent or disabled"
    planes = [float(component.upper_z_mm), float(component.lower_z_mm)]
    if not state.electron_gun.exit_plane_z_mm < planes[0] < planes[1] < state.sample.upper_surface_z_mm:
        return False, "Two distinct kick planes must precede the specimen entrance"
    if state.vacuum_map.enabled:
        return False, "This incident observer does not yet support active vacuum scattering"
    return True, "Four physical kicks; four independent observations required. Local rank is measured before acceptance."


def allowed_state(request, controls):
    if request.registry_digest != json_digest(asdict(DEFINITION)) or set(controls) != set(DEFINITION.devices):
        raise ValueError("Beam alignment controls do not match the registered physical device")
    if not isinstance(request.options, BeamAlignmentOptions):
        raise TypeError("Expected captured beam-alignment targets and numerical limits")
    state = request.start_snapshot.restore()
    available, reason = capability(state)
    if not available:
        raise ValueError(reason)
    for key, value in controls.items():
        if not math.isfinite(value) or abs(value) > request.options.maximum_kick_mrad:
            raise ValueError("Deflector kick is outside the declared numerical search interval")
        setattr(state.beam_deflector, key.split(".")[1], float(value))
    return state


def solve_candidate(request, *, cancelled):
    from temsim.alignment_constraints import evaluate_incident
    from temsim.alignment_transaction import AlignmentCandidate, AlignmentCancelled
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.calculation_cache import calculation_signatures
    from temsim.optics.direct_alignment import DirectAlignmentResult
    from temsim.working_point import WorkingPointCheckpoint
    options = request.options
    state = request.start_snapshot.restore()
    scale = np.array([options.position_tolerance_um] * 2 + [options.angle_tolerance_mrad] * 2)
    targets = np.array(options.targets)
    bound = options.maximum_kick_mrad
    observed = {}
    failures = []

    def check():
        if cancelled():
            raise AlignmentCancelled("Beam alignment cancelled; previous working point retained")

    def controls(vector):
        return dict(zip(DEFINITION.devices, map(float, vector), strict=True))

    def measure(vector, axis="configured"):
        check()
        scratch = allowed_state(request, controls(vector))
        if axis == "column_step":
            scratch.step_mm *= .5
        elif axis == "gun_step":
            scratch.electron_gun.trace_step_mm *= .5
        scratch._tuning_cancelled = cancelled
        metrics, arrays, summary, _ = evaluate_incident(scratch)
        if any(metrics.get(key) is None for key in OBSERVABLES):
            raise ValueError("No finite transmitted population for centre and direction observations")
        return scratch, metrics, arrays, summary

    class BudgetReached(Exception):
        pass

    def residual(vector):
        check()
        key = tuple(float(v) for v in vector)
        if key not in observed:
            if len(observed) >= options.maximum_evaluations:
                raise BudgetReached()
            try:
                _, metrics, arrays, _ = measure(vector)
                del arrays
                values = (np.array([metrics[k] for k in OBSERVABLES]) - targets) / scale
                values = np.r_[values,
                    max(0., options.minimum_effective_samples - metrics["effective_samples"]),
                    max(0., options.minimum_current_pa - metrics.get("current_pa", 0.)) /
                    max(options.minimum_current_pa, 1.)]
            except (ValueError, RuntimeError) as exc:
                check()
                values, metrics = np.full(6, 1.e6), {}
                failures.append(dict(controls=controls(vector), reason=str(exc)))
            observed[key] = (values, metrics)
        return observed[key][0]

    initial = np.clip([getattr(state.beam_deflector, name) for name in CONTROLS], -bound, bound)
    try:
        least_squares(residual, initial, bounds=(-bound, bound), x_scale=np.ones(4) * bound,
                      max_nfev=options.maximum_evaluations)
    except BudgetReached:
        pass
    vector, best = min(observed.items(), key=lambda row: np.linalg.norm(row[1][0]))
    # Forward finite differences measure rank without optimizer penalties.
    jacobian = np.zeros((4, 4))
    for column in range(4):
        shifted = np.array(vector)
        delta = bound * 1.e-4
        shifted[column] += delta if shifted[column] + delta <= bound else -delta
        try:
            _, metrics, arrays, _ = measure(shifted)
            del arrays
            if best[1]:
                jacobian[:, column] = np.array([metrics[k]-best[1][k] for k in OBSERVABLES]) / scale / ((shifted[column]-vector[column])/bound)
        except (ValueError, RuntimeError):
            check()
    rank, condition = int(np.linalg.matrix_rank(jacobian)), float(np.linalg.cond(jacobian))
    authority = rank == 4 and math.isfinite(condition) and condition <= 1.e8
    refinements, checkpoint = [], None
    for axis in ("configured", "column_step", "gun_step"):
        try:
            scratch, metrics, arrays, summary = measure(vector, axis)
            values = (np.array([metrics[k] for k in OBSERVABLES])-targets) / scale
            supported = metrics["effective_samples"] >= options.minimum_effective_samples
            passed = supported and metrics["current_pa"] > 0 and metrics["current_pa"] >= options.minimum_current_pa and np.all(np.abs(values) <= 1.)
            snapshot = capture_instrument_snapshot(scratch)
            refinements.append(dict(axis=axis, metrics=metrics, residuals=values.tolist(), passed=bool(passed),
                                    actual_backend=summary.get("actual_backend"), snapshot_id=snapshot.digest))
            checkpoint = WorkingPointCheckpoint(snapshot, arrays, summary["plane_z_mm"],
                calculation_signatures(scratch)["incident"], dict(source_current_a=summary["source_current_a"],
                    source_representation="gun-derived-particles", coordinate_precision="exact-integration-checkpoint",
                    phase_status="NOT_COMPUTED", validation_status="NOT_RUN", alignment_options=options.to_dict()),
                request.start_snapshot.digest)
        except (ValueError, RuntimeError) as exc:
            check()
            refinements.append(dict(axis=axis, passed=False, metrics={}, failure=str(exc)))
    check()
    available = all("residuals" in row for row in refinements)
    spread = float(np.max(np.ptp([row["residuals"] for row in refinements], axis=0))) if available else 1.e30
    success = authority and spread <= .25 and all(row["passed"] for row in refinements)
    status = "PASS" if success else "UNAVAILABLE_CONTROL_AUTHORITY" if rank < 4 else "ILL_CONDITIONED_SOLVE" if not authority else "FORWARD_CHECK_FAILED"
    achieved = float(max(abs(v) for v in refinements[-1].get("residuals", [1.e30])))
    message = ("Beam centre and direction passed independent gun/column checks." if success else
               f"{status}: no acceptable candidate in this bounded search; no changes applied.")
    final_state = allowed_state(request, controls(vector))
    result = DirectAlignmentResult(key=KEY, success=success, requested=0., achieved=achieved,
        unit="scaled residual", constraint_value=spread, constraint_unit="tolerance fraction",
        strengths=controls(vector), iterations=len(observed), validation_step_mm=state.step_mm / 2.,
        numerical_spread=spread, message=message)
    validation = dict(status=status, options_id=options.digest, options=options.to_dict(),
        allowed_controls=list(DEFINITION.devices), fixed_controls="All other captured inputs",
        plane="Specimen entrance", observables=list(OBSERVABLES), units=["um", "um", "mrad", "mrad"],
        control_authority=dict(rank=rank, required_rank=4, condition=condition if math.isfinite(condition) else None,
                               normalized_jacobian=jacobian.tolist()),
        refinements=refinements, search_failures=failures, trial_count=len(observed),
        additional_forward_checks="Four authority perturbations and three independent numerical runs",
        bounds_scope="Explicit numerical search interval; no hardware rating asserted",
        independent_reference="NOT_RUN", qualification="No sampling, field-mesh or physical-source qualification inferred",
        candidate_snapshot=capture_instrument_snapshot(final_state).to_dict(),
        forward_snapshot_id=checkpoint.snapshot.digest if checkpoint else None)
    return AlignmentCandidate(request, result, checkpoint, validation, "READY_TO_APPLY" if success else "FAILED")
