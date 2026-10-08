"""Read-only, bounded CM/objective design studies in the installed column.

Microprobe uses positive and Nanoprobe negative CM excitation of *one* fixed
magnitude. Both retain the CM field. All other lens/source parameters remain
unchanged. The magnetic profiles are those admitted by the normal runtime;
an analytic profile translated here is not a new magnetostatic/OEM calibration.

Matrices use (x, y, theta_x, theta_y), metres/radians, and mechanical input
slopes. Both mechanical and canonical output coordinates are reported. The
optional covariance targets describe an explicitly supplied upstream beam;
they do not silently replace the instrument Tip or claim full-probe acceptance.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np

from temsim.optics.direct_alignment import canonical_source_basis
from temsim.physics.core import bz
from temsim.physics.first_order import trace_transverse_transfer
from temsim.physics.lens_field_provider import runtime_axial_magnetic_field_t


def _readonly(value):
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _copy_state(state):
    # Frozen resolved assemblies contain mapping proxies. Use the established
    # input graph round-trip rather than deepcopy or the legacy lossy JSON path.
    if hasattr(state, "to_dict"):
        from temsim.instrument_snapshot import decode_instrument, encode_instrument
        return decode_instrument(encode_instrument(state))
    return deepcopy(state)


@dataclass(frozen=True)
class CondenserObjectivePositionBounds:
    minimum_center_z_mm: float
    maximum_center_z_mm: float
    provenance: str

    def __post_init__(self):
        if (not math.isfinite(self.minimum_center_z_mm)
                or not math.isfinite(self.maximum_center_z_mm)
                or self.maximum_center_z_mm < self.minimum_center_z_mm):
            raise ValueError("CM centre bounds must be finite and ordered")
        if not self.provenance.strip():
            raise ValueError("CM position bounds require mechanical provenance")


@dataclass(frozen=True)
class CondenserObjectiveTargets:
    """Explicit acceptance limits; RMS quantities are radial, not per-axis."""

    nano_maximum_rms_radius_m: float
    micro_maximum_rms_angle_rad: float

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in (
                self.nano_maximum_rms_radius_m,
                self.micro_maximum_rms_angle_rad)):
            raise ValueError("CM design targets must be positive and finite")


@dataclass(frozen=True)
class CondenserObjectiveMode:
    name: str
    polarity: int
    mechanical_transfer: np.ndarray
    canonical_output_transfer: np.ndarray
    total_axial_field_t: np.ndarray
    total_field_zero_z_mm: tuple[float, ...]
    required_input_correlation_rad_per_m: np.ndarray | None
    specimen_covariance: np.ndarray | None
    rms_radius_m: float | None
    rms_mechanical_angle_rad: float | None


@dataclass(frozen=True)
class CondenserObjectiveDesign:
    center_z_mm: float
    excitation_magnitude_percent: float
    input_z_mm: float
    sample_z_mm: float
    field_z_mm: np.ndarray
    cm_objective_normalized_overlap: float
    polarity_focusing_integral_difference_t2_mm: float
    micro: CondenserObjectiveMode
    nano: CondenserObjectiveMode
    common_input_correlation_mismatch_rad_per_m: float | None
    objective_value: float | None
    targets_met: bool
    unmet: tuple[str, ...]
    qualification: str = (
        "First-order installed-column design only; axial envelope clearance only; "
        "not full 3D collision, finite-probe, diffraction or OEM qualification."
    )


@dataclass(frozen=True)
class CondenserObjectiveSearch:
    bounds: CondenserObjectivePositionBounds
    candidates: tuple[CondenserObjectiveDesign, ...]
    best: CondenserObjectiveDesign


def condenser_objective_position_bounds(state, *, clearance_mm=0.0):
    """Axially fit the CM body between AC's downstream face and upper pole.

This is deliberately conservative: overlapping named objective assemblies do
not imply that their solid pole pieces can interpenetrate. Radial clearances,
coils/yokes and edited geometries still require the mechanical assembly check.
"""
    clearance = float(clearance_mm)
    if not math.isfinite(clearance) or clearance < 0:
        raise ValueError("Mechanical clearance must be nonnegative and finite")
    cm = state.mini_condenser
    objective = state.objective_lens
    half_cm = float(cm.length_mm) / 2
    pole_start = (float(objective.upper_pole_piece_center_z_mm)
                  - float(objective.upper_pole_piece_axial_length_mm) / 2)
    return CondenserObjectivePositionBounds(
        float(state.ac_deflector.lower_surface_z_mm) + half_cm + clearance,
        pole_start - half_cm - clearance,
        "Installed AC downstream body face and upper-objective pole upstream face; "
        "CM rigid axial envelope, no assertion of full radial collision clearance.",
    )


def _field_zeros(z, values):
    # Ignore tiny numerical tails, which must not become artificial null planes.
    scale = max(float(np.max(np.abs(values))), 1e-30)
    indices = np.flatnonzero(values[:-1] * values[1:] < 0)
    return tuple(float(z[i] - values[i] * (z[i + 1] - z[i])
                       / (values[i + 1] - values[i])) for i in indices
                 if max(abs(values[i]), abs(values[i + 1])) > 1e-8 * scale)


def _covariance(value):
    if value is None:
        return None
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Input covariance must be finite 4x4 in metres/radians")
    if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=0.0):
        raise ValueError("Input covariance must be symmetric")
    # Scale out the coordinate units before testing positive semidefiniteness.
    diagonal = np.diag(matrix)
    if np.any(diagonal < 0):
        raise ValueError("Input covariance must be positive semidefinite")
    scales = np.sqrt(np.maximum(diagonal, 1e-300))
    normalized = matrix / np.outer(scales, scales)
    if np.min(np.linalg.eigvalsh(normalized)) < -1e-10:
        raise ValueError("Input covariance must be positive semidefinite")
    return matrix.copy()


def evaluate_condenser_objective_design(
    state, *, center_z_mm=None, excitation_magnitude_percent=None, input_z_mm=None,
    input_covariance=None, targets: CondenserObjectiveTargets | None = None,
    bounds: CondenserObjectivePositionBounds | None = None,
    field_step_mm=0.1, maximum_step_mm=0.2,
) -> CondenserObjectiveDesign:
    """Compare both polarities without mutating state or changing other fields.

    If no beam covariance/targets are supplied, the map is diagnostic only and
    cannot pass acceptance. The required input direction-position correlation is K in
    theta=K*r: Nano solves A+B*K=0; Micro solves C+D*K=0. A difference between
    these requirements measures the incompatibility of those two mechanical
    slope constraints. Inside a magnetic field it is not independently a phase
    wavefront claim; canonical matrices are supplied for the shared conversion.
    """
    limits = bounds or condenser_objective_position_bounds(state)
    if not state.mini_condenser.enabled:
        raise ValueError("CM design requires an installed, enabled Mini Condenser")
    from temsim.simulation_modes import uses_field_maps, mode_key
    descriptors = getattr(state, "lens_field_map_descriptors", {})
    if uses_field_maps(state) and (
            descriptors.get(state.mini_condenser.key)
            or mode_key(state) in {"linear_geometry", "nonlinear_material"}):
        raise ValueError("Moving CM design currently requires the analytic field model; "
                         "mapped fields need a geometry-matched magnetic re-solve")
    center = float(state.mini_condenser.mechanical_center_from_tip_mm
                   if center_z_mm is None else center_z_mm)
    amplitude = float(state.mini_condenser.percent
                      if excitation_magnitude_percent is None else excitation_magnitude_percent)
    entrance = float(min(state.ac_deflector.upper_surface_z_mm,
                         limits.minimum_center_z_mm - 8 * state.mini_condenser.a_mm)
                     if input_z_mm is None else input_z_mm)
    sample_z = float(state.sample.z_mm)
    if not math.isfinite(center) or not (
            limits.minimum_center_z_mm <= center <= limits.maximum_center_z_mm):
        raise ValueError("CM centre is outside the declared mechanical envelope")
    if not math.isfinite(amplitude) or not 0 < amplitude <= float(state.mini_condenser.max_percent):
        raise ValueError("CM fixed excitation magnitude must be positive and within range")
    if not math.isfinite(entrance) or not entrance < sample_z:
        raise ValueError("Input plane must precede the specimen")
    if any(not math.isfinite(float(v)) or v <= 0 for v in (field_step_mm, maximum_step_mm)):
        raise ValueError("Field/propagation steps must be positive and finite")
    cov = _covariance(input_covariance)
    if targets is not None and cov is None:
        raise ValueError("Design acceptance requires an explicit upstream covariance")
    # One common interval for both polarities and every candidate in a search.
    count = max(2, int(math.ceil((sample_z - entrance) / field_step_mm)) + 1)
    if count > 100001:
        raise ValueError("Design field grid exceeds 100001 points")
    z = np.linspace(entrance, sample_z, count)
    modes = []
    cm_field = objective_field = None
    for name, polarity in (("microprobe", 1), ("nanoprobe", -1)):
        candidate = _copy_state(state)
        cm = candidate.mini_condenser
        cm.mechanical_center_from_tip_mm = center
        cm.percent = amplitude
        cm.polarity = polarity
        cm.enabled = True  # Optical off retains signed excitation.
        matrix = trace_transverse_transfer(
            candidate, entrance, sample_z,
            maximum_step_mm=maximum_step_mm,
        ).matrix
        canonical = np.linalg.solve(canonical_source_basis(candidate, sample_z), matrix)
        field = np.asarray(bz(z, candidate), dtype=float)
        if cm_field is None:
            cm_field = runtime_axial_magnetic_field_t(candidate, cm.key, cm, z)[0]
            obj = candidate.objective_lens
            objective_field = runtime_axial_magnetic_field_t(candidate, obj.key, obj, z)[0]
        left, right = ((matrix[:2, 2:], matrix[:2, :2]) if polarity < 0
                       else (matrix[2:, 2:], matrix[2:, :2]))
        required = (-np.linalg.solve(left, right)
                    if np.linalg.cond(left) < 1e10 else None)
        propagated = matrix @ cov @ matrix.T if cov is not None else None
        modes.append(CondenserObjectiveMode(
            name, polarity, _readonly(matrix), _readonly(canonical), _readonly(field),
            _field_zeros(z, field), None if required is None else _readonly(required),
            None if propagated is None else _readonly(propagated),
            None if propagated is None else math.sqrt(max(0., float(np.trace(propagated[:2, :2])))),
            None if propagated is None else math.sqrt(max(0., float(np.trace(propagated[2:, 2:])))),
        ))
    micro, nano = modes
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    denominator = math.sqrt(float(integrate(cm_field**2, z) * integrate(objective_field**2, z)))
    overlap = float(integrate(cm_field * objective_field, z)) / denominator if denominator > 0 else 0.
    delta_power = float(integrate(micro.total_axial_field_t**2 - nano.total_axial_field_t**2, z))
    required_m = micro.required_input_correlation_rad_per_m
    required_n = nano.required_input_correlation_rad_per_m
    mismatch = (None if required_m is None or required_n is None
                else float(np.linalg.norm(required_m - required_n, ord=2)))
    unmet, objective_value = [], None
    if targets is None:
        unmet.append("No explicit incident beam covariance and acceptance limits supplied")
    else:
        nano_ratio = nano.rms_radius_m / targets.nano_maximum_rms_radius_m
        micro_ratio = micro.rms_mechanical_angle_rad / targets.micro_maximum_rms_angle_rad
        objective_value = max(nano_ratio, micro_ratio)
        if nano_ratio > 1:
            unmet.append("Nanoprobe specimen RMS radius exceeds the supplied limit")
        if micro_ratio > 1:
            unmet.append("Microprobe specimen mechanical angular RMS exceeds the supplied limit")
    return CondenserObjectiveDesign(
        center, amplitude, entrance, sample_z, _readonly(z), overlap, delta_power,
        micro, nano, mismatch, objective_value, not unmet, tuple(unmet),
    )


def solve_condenser_objective_design(
    state, *, excitation_magnitude_percent=None, input_z_mm=None, input_covariance=None,
    targets: CondenserObjectiveTargets | None = None, bounds=None, candidate_count=9,
    field_step_mm=0.1, maximum_step_mm=0.2,
    progress_callback=None, cancel_check=None,
) -> CondenserObjectiveSearch:
    """Bounded finite-grid search; no claim of a globally optimal geometry.

    Without explicit targets, rank the ideal focus/parallel input-correlation mismatch
    only. Such a result always remains unqualified for a particular physical beam.
    """
    limits = bounds or condenser_objective_position_bounds(state)
    if isinstance(candidate_count, bool) or int(candidate_count) != candidate_count or not 2 <= candidate_count <= 101:
        raise ValueError("Design search requires 2 to 101 position candidates")
    candidates = []
    for index, center in enumerate(np.linspace(limits.minimum_center_z_mm,
                                               limits.maximum_center_z_mm, int(candidate_count))):
        if cancel_check is not None and cancel_check():
            raise InterruptedError("CM/objective design search cancelled")
        candidates.append(evaluate_condenser_objective_design(
            state, center_z_mm=center, excitation_magnitude_percent=excitation_magnitude_percent,
            input_z_mm=input_z_mm, input_covariance=input_covariance, targets=targets,
            bounds=limits, field_step_mm=field_step_mm, maximum_step_mm=maximum_step_mm,
        ))
        if progress_callback is not None:
            progress_callback(index + 1, int(candidate_count))
    candidates = tuple(candidates)
    best = min(candidates, key=lambda candidate: (
        candidate.objective_value if candidate.objective_value is not None else
        candidate.common_input_correlation_mismatch_rad_per_m
        if candidate.common_input_correlation_mismatch_rad_per_m is not None else math.inf))
    return CondenserObjectiveSearch(limits, candidates, best)
