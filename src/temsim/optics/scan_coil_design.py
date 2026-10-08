"""Read-only first-order design diagnostics for the two physical AC kicks.

All active column fields, including the mini condenser and objective, enter
the selected response model: historical ideal angular kicks, or derivatives
of the actual finite coil providers. Neither is a finite source-beam solution
or detector-stationarity certification. Production calibration is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np

from temsim import input_io
from temsim.optics.direct_alignment import canonical_source_basis
from temsim.physics.first_order import trace_transverse_transfers
from temsim.physics.scan_calibration import sample_reference_z_mm
from temsim.physics.finite_scan_response import finite_scan_responses


ScanAngleTarget = Literal["mechanical", "canonical"]
Matrix2 = tuple[tuple[float, float], tuple[float, float]]
MAX_PIVOT_SAMPLES = 8193
MAX_POSITION_CANDIDATES = 25


@dataclass(frozen=True, slots=True)
class ScanPivotResult:
    """Full two-axis scan displacement minimum, downstream of the lower coil.

    ``response_m_per_raster_factor`` maps dimensionless full-FOV X/Y factors
    to transverse position in metres. A small single matrix entry, determinant
    or singular value alone never establishes a common pivot. Both columns
    must vanish within ``relative_tolerance`` times their own sample-axis
    half-FOV. ``relative_residual`` is the larger of these two axis residuals;
    a very wide FOV axis cannot hide a missing pivot in the narrower axis.
    The finite grid and local refinement do not prove global absence of roots.
    """

    z_mm: float
    interval_z_mm: tuple[float, float]
    grid_step_mm: float
    grid_samples: int
    response_m_per_raster_factor: Matrix2
    singular_values_m: tuple[float, float]
    residual_m: float
    relative_residual: float
    relative_tolerance: float
    is_common_pivot: bool
    refined: bool
    refinement_success: bool

    @property
    def kind(self) -> str:
        return "common_pivot_candidate" if self.is_common_pivot else "closest_approach"


@dataclass(frozen=True, slots=True)
class ScanCoilDesignResult:
    """A suggested drive, never applied to the caller's instrument state.

    Kick matrices are physical foil angles in mrad per dimensionless raster
    factor, before any logical ``upper_coil_gain`` convention. Peak tuples are
    per physical X/Y axis over the whole FOV rectangle; total peaks reserve the
    currently configured *static* AC alignment kick. Wobble is not combined
    with the hypothetical raster. Limits are the configured model limits.
    """

    upper_z_mm: float
    lower_z_mm: float
    sample_z_mm: float
    scan_reference: str
    target: ScanAngleTarget
    field_of_view_nm: tuple[float, float]
    controllable: bool
    condition_number: float
    condition_length_m: float
    condition_limit: float
    upper_kick_matrix_mrad: Matrix2 | None
    lower_kick_matrix_mrad: Matrix2 | None
    lower_from_upper: Matrix2 | None
    upper_scan_peak_mrad: tuple[float, float] | None
    lower_scan_peak_mrad: tuple[float, float] | None
    upper_static_kick_mrad: tuple[float, float]
    lower_static_kick_mrad: tuple[float, float]
    upper_peak_mrad: tuple[float, float] | None
    lower_peak_mrad: tuple[float, float] | None
    kick_limit_mrad: float
    kick_limit_pass: bool
    position_residual_m: float | None
    position_relative_residual: float | None
    angle_residual_rad: float | None
    angle_relative_residual: float | None
    sample_position_matrix_m: Matrix2 | None
    sample_mechanical_angle_matrix_rad: Matrix2 | None
    sample_canonical_angle_matrix_rad: Matrix2 | None
    pivot: ScanPivotResult | None
    message: str
    geometry_qualified: bool = False
    scope: str = (
        "First-order ideal-kick design in the captured full column field; "
        "coil envelopes, installation clearance, finite-coil fields, finite beam "
        "aberrations and detector stationarity are not qualified")
    linearization_scope: str = (
        "Reference starts on axis at the candidate upper coil with AC drive "
        "neutralized; other captured affine drives remain active. Existing static "
        "AC bias is reserved for kick limits only; its off-axis feeddown is not qualified")
    response_model: str = "ideal_kick"

    @property
    def peak_kick_mrad(self) -> float:
        if self.upper_peak_mrad is None or self.lower_peak_mrad is None:
            return math.inf
        return max(*self.upper_peak_mrad, *self.lower_peak_mrad)


@dataclass(frozen=True, slots=True)
class ScanCoilPositionSearchResult:
    """Finite ranked candidates, without moving coils or changing currents."""

    upper_z_range_mm: tuple[float, float]
    coil_gap_mm: float
    target: ScanAngleTarget
    candidates: tuple[ScanCoilDesignResult, ...]

    @property
    def best(self) -> ScanCoilDesignResult | None:
        return next((item for item in self.candidates
                     if item.controllable and item.kick_limit_pass), None)


def _matrix(value) -> Matrix2:
    array = np.asarray(value, dtype=float)
    return tuple(tuple(float(x) for x in row) for row in array)


def _pair(value) -> tuple[float, float]:
    return tuple(float(x) for x in value)


def _positive(value, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _private_state(state):
    # State.to_dict/from_dict re-resolves TOML-owned geometry; deepcopy cannot
    # handle immutable map/assembly bindings. Preserve the exact current graph.
    from temsim.instrument_snapshot import decode_instrument, encode_instrument

    return decode_instrument(encode_instrument(state))


def _parameters(state, upper_z_mm, lower_z_mm, target, pivot_step_mm,
                maximum_step_mm, condition_limit, pivot_relative_tolerance):
    if target not in ("mechanical", "canonical"):
        raise ValueError("Scan angle target must be 'mechanical' or 'canonical'")
    ac = state.ac_deflector
    upper = float(ac.upper_z_mm if upper_z_mm is None else upper_z_mm)
    lower = float(ac.lower_z_mm if lower_z_mm is None else lower_z_mm)
    sample = float(sample_reference_z_mm(state))
    if not np.isfinite((upper, lower, sample)).all() or not 0 < upper < lower < sample:
        raise ValueError("AC coil planes must satisfy 0 < upper < lower < sample reference")
    step = _positive(pivot_step_mm, "Pivot sampling step (mm)")
    if maximum_step_mm is not None:
        maximum_step_mm = _positive(maximum_step_mm, "Maximum integration step (mm)")
    limit = _positive(condition_limit, "Condition-number limit")
    if limit < 1:
        raise ValueError("Condition-number limit must be at least one")
    tolerance = _positive(pivot_relative_tolerance, "Pivot relative tolerance")
    if tolerance >= 1:
        raise ValueError("Pivot relative tolerance must be less than one")
    count = int(math.ceil((sample - lower) / step)) + 1
    if count > MAX_PIVOT_SAMPLES:
        raise ValueError(f"Pivot sampling exceeds the {MAX_PIVOT_SAMPLES}-plane diagnostic limit")
    fov = np.array((ac.scan_field_of_view_x_nm, ac.scan_field_of_view_y_nm), dtype=float)
    if not np.isfinite(fov).all() or np.any(fov <= 0):
        raise ValueError("Scan FOV must have two finite positive extents")
    kick_limit = _positive(ac.maximum_kick_mrad, "Physical coil kick limit (mrad)")
    static = np.asarray(ac.coil_kicks_mrad(ac.kick_x_mrad, ac.kick_y_mrad), dtype=float)
    if static.shape != (2, 2) or not np.isfinite(static).all():
        raise ValueError("Static physical coil kicks must form a finite 2x2 array")
    return upper, lower, sample, step, maximum_step_mm, limit, tolerance, count, fov, kick_limit, static


@input_io.using_state_inputs
def evaluate_scan_coil_design(
    state, *, upper_z_mm: float | None = None, lower_z_mm: float | None = None,
    target: ScanAngleTarget = "mechanical", pivot_step_mm: float = 0.25,
    maximum_step_mm: float | None = None, condition_limit: float = 1.0e10,
    pivot_relative_tolerance: float = 1.0e-6,
    response_model: str = "ideal_kick",
) -> ScanCoilDesignResult:
    """Solve sample position and an explicit sample-angle constraint read-only.

    The sample centre/entrance follows ``state.ac_deflector.scan_reference``.
    Coil positions are diagnostic *candidate* centres and must stay upstream;
    this API does not authorize mechanical clearance or edit the installation.
    The target is mechanical telecentricity by default. Canonical telecentricity
    instead uses the same local vector-potential conversion as diffraction
    diagnostics. Neither target alone verifies a real detector's stationarity.
    ``response_model='finite_coil'`` uses the actual physical host's finite
    field length and registration, with static affine steering retained.
    The backward-compatible ``ideal_kick`` default keeps its original scope.

    Conditioning uses the dimensionless 4x4 map obtained by dividing position
    rows by the coil separation in metres. Invalid input raises ValueError;
    uncontrollable/ill-conditioned designs return ``controllable=False``.
    """
    if response_model not in ("ideal_kick", "finite_coil"):
        raise ValueError("Scan response model must be 'ideal_kick' or 'finite_coil'")
    params = _parameters(state, upper_z_mm, lower_z_mm, target, pivot_step_mm,
                         maximum_step_mm, condition_limit, pivot_relative_tolerance)
    return _evaluate(_private_state(state), target, params, response_model)


def _evaluate(state, target, params, response_model="ideal_kick"):
    (upper, lower, sample, step, maximum_step, condition_limit, pivot_tolerance,
     count, fov, kick_limit, static) = params
    # The candidate kicks replace AC's current drive. Retain every other active
    # field and affine steering term, not a peak-field surrogate or thin OL.
    grid = np.linspace(lower, sample, count)
    identity_kicks = np.eye(4)[:, 2:]
    if response_model == "finite_coil":
        finite = finite_scan_responses(state, state.ac_deflector, grid,
            upper_z_mm=upper, lower_z_mm=lower, maximum_step_mm=maximum_step)
        if max(support[1] for support in finite.coil_support_mm) >= sample:
            raise ValueError("Finite AC coil fields must end upstream of the sample reference")
        upper_response, lower_response = finite.upper[-1], finite.lower[-1]
        reference_position = finite.reference[-1, :2]
    else:
        state.ac_deflector.enabled = False
        transfers = trace_transverse_transfers(
            state, upper, grid, maximum_step_mm=maximum_step)
        lower_transfer = transfers[float(grid[0])].matrix
        sample_transfer = transfers[float(grid[-1])]
        reference_position = sample_transfer.position_offset_m
    length = (lower - upper) * 1.0e-3
    desired = np.diag(fov) * 0.5e-9
    basic = dict(
        upper_z_mm=upper, lower_z_mm=lower, sample_z_mm=sample,
        scan_reference=str(state.ac_deflector.scan_reference), target=target,
        field_of_view_nm=_pair(fov), condition_length_m=length,
        condition_limit=condition_limit, upper_static_kick_mrad=_pair(static[0]),
        lower_static_kick_mrad=_pair(static[1]), kick_limit_mrad=kick_limit,
        response_model=response_model,
    )
    if response_model == "finite_coil":
        basic.update(
            scope="Local finite-coil drive derivatives in the captured full column field; mechanical clearance, finite source beam, aberrations and detector stationarity are not qualified",
            linearization_scope="Reference starts on axis at the upstream registered coil-field boundary; static AC alignment and other captured affine drives are retained. AC/Descan raster and wobble are neutralized; other dynamic drives retain the captured simulation time")
    try:
        # The two impulses share one captured reference orbit. Starting a second
        # independent on-axis chief at the lower coil would miss off-axis feeddown.
        if response_model == "ideal_kick":
            lower_basis = np.linalg.solve(lower_transfer, identity_kicks)
            upper_response = sample_transfer.matrix @ identity_kicks
            lower_response = sample_transfer.matrix @ lower_basis
        g = canonical_source_basis(
            state, sample, position_xy_m=reference_position)[2:, :2]
        canonical_upper = upper_response[2:] - g @ upper_response[:2]
        canonical_lower = lower_response[2:] - g @ lower_response[:2]
        angle_upper, angle_lower = (
            (canonical_upper, canonical_lower) if target == "canonical"
            else (upper_response[2:], lower_response[2:]))
        response = np.block([[upper_response[:2] / length, lower_response[:2] / length],
                             [angle_upper, angle_lower]])
        if not np.isfinite(response).all():
            raise np.linalg.LinAlgError("Non-finite transverse response")
        condition = float(np.linalg.cond(response))
        if not np.isfinite(condition) or condition > condition_limit:
            return _uncontrollable(basic, condition, "Sample position/angle response is singular or ill-conditioned")
        kicks = np.linalg.solve(response, np.vstack((desired / length, np.zeros((2, 2)))))
    except np.linalg.LinAlgError as exc:
        return _uncontrollable(basic, math.inf, str(exc))
    upper_kick, lower_kick = kicks[:2], kicks[2:]
    realised = upper_response @ upper_kick + lower_response @ lower_kick
    canonical = realised[2:] - g @ realised[:2]
    angle = canonical if target == "canonical" else realised[2:]
    position_residual = float(np.linalg.norm(realised[:2] - desired))
    angle_residual = float(np.linalg.norm(angle))
    # Relative angle residual is normalized to the separately driven response,
    # not to the zero target, and does not hide cancellation behind a unit floor.
    angle_scale = max(float(np.linalg.norm(angle_upper @ upper_kick)),
                      float(np.linalg.norm(angle_lower @ lower_kick)), np.finfo(float).tiny)
    scan_peak = np.sum(np.abs(kicks), axis=1).reshape(2, 2) * 1.0e3
    peak = scan_peak + np.abs(static)
    upper_condition = float(np.linalg.cond(upper_kick))
    coupling = (np.linalg.solve(upper_kick.T, lower_kick.T).T
                if np.isfinite(upper_condition) and upper_condition <= condition_limit else None)
    if response_model == "finite_coil":
        paths = finite.upper[:, :2] @ upper_kick + finite.lower[:, :2] @ lower_kick

        def response_at(z):
            response = finite_scan_responses(state, state.ac_deflector, (z,),
                upper_z_mm=upper, lower_z_mm=lower, maximum_step_mm=maximum_step)
            ru, rl, _ = response.at(z)
            return ru[:2] @ upper_kick + rl[:2] @ lower_kick

        launch = None
    else:
        launch = identity_kicks @ upper_kick + lower_basis @ lower_kick
        paths = np.stack([transfers[float(z)].matrix[:2] @ launch for z in grid])
        response_at = None
    pivot = _pivot(state, upper, lower, sample, grid, paths, launch, desired,
                   maximum_step, pivot_tolerance, response_at=response_at)
    passed = bool(np.all(peak <= kick_limit))
    return ScanCoilDesignResult(
        **basic, controllable=True, condition_number=condition,
        upper_kick_matrix_mrad=_matrix(upper_kick * 1.0e3),
        lower_kick_matrix_mrad=_matrix(lower_kick * 1.0e3),
        lower_from_upper=None if coupling is None else _matrix(coupling),
        upper_scan_peak_mrad=_pair(scan_peak[0]), lower_scan_peak_mrad=_pair(scan_peak[1]),
        upper_peak_mrad=_pair(peak[0]), lower_peak_mrad=_pair(peak[1]),
        kick_limit_pass=passed, position_residual_m=position_residual,
        position_relative_residual=position_residual / float(np.linalg.norm(desired)),
        angle_residual_rad=angle_residual, angle_relative_residual=angle_residual / angle_scale,
        sample_position_matrix_m=_matrix(realised[:2]),
        sample_mechanical_angle_matrix_rad=_matrix(realised[2:]),
        sample_canonical_angle_matrix_rad=_matrix(canonical), pivot=pivot,
        message=("First-order constraints solved; configured physical kick limits pass"
                 if passed else "First-order constraints solved; configured physical kick limit exceeded"),
    )


def _uncontrollable(basic, condition, message):
    return ScanCoilDesignResult(
        **basic, controllable=False, condition_number=condition,
        upper_kick_matrix_mrad=None, lower_kick_matrix_mrad=None, lower_from_upper=None,
        upper_scan_peak_mrad=None, lower_scan_peak_mrad=None,
        upper_peak_mrad=None, lower_peak_mrad=None, kick_limit_pass=False,
        position_residual_m=None, position_relative_residual=None,
        angle_residual_rad=None, angle_relative_residual=None,
        sample_position_matrix_m=None, sample_mechanical_angle_matrix_rad=None,
        sample_canonical_angle_matrix_rad=None, pivot=None, message=message,
    )


def _pivot(state, upper, lower, sample, grid, paths, launch, desired,
           maximum_step, tolerance, response_at=None):
    half_extents = np.diag(desired)
    normalized_paths = paths / half_extents[None, None, :]
    norms = np.linalg.norm(normalized_paths, axis=(1, 2))
    index = int(np.argmin(norms))
    best_z, best_response = float(grid[index]), paths[index]
    refined = False
    refinement_success = True
    # Minimize all four entries together on each piecewise-linear segment.
    # This only selects a bracket; the reported candidate is traced afresh.
    delta = np.diff(normalized_paths, axis=0)
    denominator = np.sum(delta * delta, axis=(1, 2))
    fractions = np.zeros_like(denominator)
    usable = denominator > np.finfo(float).tiny
    fractions[usable] = np.clip(
        -np.sum(normalized_paths[:-1] * delta, axis=(1, 2))[usable] / denominator[usable], 0., 1.)
    interpolated = normalized_paths[:-1] + fractions[:, None, None] * delta
    segment = int(np.argmin(np.linalg.norm(interpolated, axis=(1, 2))))
    # A plateau (e.g. a pure drift translation) has no isolated pivot to refine.
    if usable[segment] and 0. < fractions[segment] < 1.:
        from scipy.optimize import minimize_scalar

        if response_at is None:
            def response_at(z):
                transfer = trace_transverse_transfers(
                    state, upper, (float(z),), maximum_step_mm=maximum_step)[float(z)]
                return transfer.matrix[:2] @ launch

        left = float(grid[segment])
        width = float(grid[segment + 1] - grid[segment])

        def squared_response_norm(offset):
            value = response_at(left + offset) / half_extents[None, :]
            return float(np.sum(value * value))

        # Use local offsets: scipy's relative stopping term at global Z~1600
        # mm otherwise limits location accuracy to tens of nanometres. The
        # squared full-matrix norm is smooth even at an actual common root.
        optimum = minimize_scalar(
            squared_response_norm, bounds=(0., width), method="bounded",
            options={"xatol": 1.0e-9, "maxiter": 32})
        refined = True
        refinement_success = bool(optimum.success)
        candidate_z = left + float(optimum.x)
        response = response_at(candidate_z)
        if np.linalg.norm(response / half_extents[None, :]) < np.linalg.norm(best_response / half_extents[None, :]):
            best_z, best_response = candidate_z, response
    residual = float(np.linalg.norm(best_response))
    relative = float(np.max(np.linalg.norm(best_response / half_extents[None, :], axis=0)))
    return ScanPivotResult(
        z_mm=best_z, interval_z_mm=(lower, sample),
        grid_step_mm=float(grid[1] - grid[0]), grid_samples=len(grid),
        response_m_per_raster_factor=_matrix(best_response),
        singular_values_m=_pair(np.linalg.svd(best_response, compute_uv=False)),
        residual_m=residual, relative_residual=relative, relative_tolerance=tolerance,
        is_common_pivot=bool(relative <= tolerance and refinement_success),
        refined=refined, refinement_success=refinement_success,
    )


@input_io.using_state_inputs
def search_scan_coil_positions(
    state, *, upper_z_range_mm: tuple[float, float], coil_gap_mm: float,
    candidate_count: int = 3, target: ScanAngleTarget = "mechanical",
    pivot_step_mm: float = 0.25, maximum_step_mm: float | None = None,
    condition_limit: float = 1.0e10, pivot_relative_tolerance: float = 1.0e-6,
    progress_callback=None, cancel_check=None,
    response_model: str = "ideal_kick",
) -> ScanCoilPositionSearchResult:
    """Rank a bounded, explicit position grid by total drive, then conditioning.

    ``coil_gap_mm`` is the centre-to-centre interaction-plane separation, not
    the inter-coil mechanical clearance. The complete supplied range must be
    ordered and upstream of the chosen sample reference; it is never clipped.
    Mechanical-envelope/clearance checks are the caller's separate obligation.
    Failed controllability/limit candidates are retained after passing ones.
    No recommendation is automatically installed and ``best`` may be None.
    """
    if response_model not in ("ideal_kick", "finite_coil"):
        raise ValueError("Scan response model must be 'ideal_kick' or 'finite_coil'")
    bounds = np.asarray(upper_z_range_mm, dtype=float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or not bounds[0] < bounds[1]:
        raise ValueError("Provide an ordered finite upper-coil Z range")
    if isinstance(candidate_count, bool) or not isinstance(candidate_count, (int, np.integer)):
        raise ValueError("Candidate count must be an integer")
    if not 2 <= candidate_count <= MAX_POSITION_CANDIDATES:
        raise ValueError(f"Candidate count must be between 2 and {MAX_POSITION_CANDIDATES}")
    gap = _positive(coil_gap_mm, "Coil centre separation (mm)")
    upper_values = np.linspace(*bounds, candidate_count)
    parameters = [_parameters(state, z, z + gap, target, pivot_step_mm,
                              maximum_step_mm, condition_limit, pivot_relative_tolerance)
                  for z in upper_values]
    if cancel_check is not None and cancel_check():
        raise InterruptedError("AC position design study cancelled")
    private = _private_state(state)
    candidates = []
    if progress_callback is not None:
        progress_callback(0, candidate_count)
    for index, params in enumerate(parameters):
        if cancel_check is not None and cancel_check():
            raise InterruptedError("AC position design study cancelled")
        candidates.append(_evaluate(private, target, params, response_model))
        if cancel_check is not None and cancel_check():
            raise InterruptedError("AC position design study cancelled")
        if progress_callback is not None:
            progress_callback(index + 1, candidate_count)
    candidates.sort(key=lambda item: (
        not item.controllable, not item.kick_limit_pass,
        item.peak_kick_mrad, item.condition_number, item.upper_z_mm))
    return ScanCoilPositionSearchResult(_pair(bounds), gap, target, tuple(candidates))
