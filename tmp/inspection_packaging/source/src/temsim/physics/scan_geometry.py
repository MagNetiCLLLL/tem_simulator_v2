"""First-order AC scan and image-referenced descan geometry.

This module deliberately stops at beam-position geometry.  Specimen
scattering and detector image formation consume the same raster coordinates
later, without making the scan controls depend on TEM/STEM probe presets.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from temsim.physics.beam_observation import (
    transverse_kick_phase_space_response,
    transverse_kick_response,
    transverse_kick_response_path,
)
from temsim.physics.first_order import (
    trace_transverse_transfer,
    trace_transverse_transfers,
)


MAX_PREVIEW_PIXELS_PER_AXIS = 128
IMAGE_CONJUGACY_TOLERANCE_M_PER_RAD = 2.0e-5
DIFFRACTION_CONJUGACY_TOLERANCE = 1.0e-3
SHARED_RASTER_FIELDS = (
    "scan_frame_period_s",
    "scan_pixels_x",
    "scan_lines",
    "scan_pixel_size_nm",
)


@dataclass(frozen=True)
class DescanCalibrationResult:
    """One opposite-command descan solve at an image-reference station."""

    target_key: str
    target_name: str
    target_z_mm: float
    lower_from_upper: np.ndarray
    response_match_residual: float
    conjugacy_residual_m_per_rad: float
    plane_kind: str


@dataclass(frozen=True)
class ScanGeometryResult:
    """One decimated raster and its first-order downstream displacement."""

    times_s: np.ndarray
    sample_x_um: np.ndarray
    sample_y_um: np.ndarray
    plane_positions_um: dict[str, tuple[np.ndarray, np.ndarray]]
    plane_names: dict[str, str]
    requested_pixels_x: int
    requested_pixels_y: int
    ac_enabled: bool
    descan_enabled: bool
    ac_drift_pivot_z_mm: float | None
    descan_drift_pivot_z_mm: float | None
    ac_lower_from_upper: np.ndarray | None = None
    ac_angular_residual: float | None = None
    pixel_size_nm: float | None = None
    field_of_view_x_nm: float | None = None
    field_of_view_y_nm: float | None = None
    scan_command_matrix_mrad: np.ndarray | None = None
    scan_scale_residual: float | None = None
    plane_roles: dict[str, str] = field(default_factory=dict)
    plane_image_residuals_m_per_rad: dict[str, float] = field(
        default_factory=dict
    )
    plane_diffraction_residuals: dict[str, float] = field(
        default_factory=dict
    )
    descan_target_key: str | None = None
    descan_target_name: str | None = None
    descan_target_z_mm: float | None = None
    descan_lower_from_upper: np.ndarray | None = None
    descan_compensation_residual: float | None = None
    descan_target_conjugacy_residual_m_per_rad: float | None = None
    descan_target_plane_kind: str | None = None
    ac_distance_above_sample_mm: float | None = None
    descan_distance_below_sample_mm: float | None = None
    scan_pair_symmetry_error_mm: float | None = None


@dataclass(frozen=True)
class ScanRayPathResult:
    """Cached first-order scan displacement bases for ray-diagram playback."""

    responses_m_per_rad: dict[str, tuple[np.ndarray, np.ndarray]]
    baseline_ac_command_mrad: np.ndarray
    baseline_descan_command_mrad: np.ndarray
    frame_period_s: float
    pixels_x: int
    pixels_y: int


@dataclass(frozen=True)
class _CalculatedObservationPlane:
    """One state-dependent optical plane shown beside physical stations."""

    key: str
    name: str
    z_mm: float


def calibrate_ac_pure_shift(state):
    """Couple the AC foils so their first-order sample angle cancels."""

    component = state.ac_deflector
    sample_z_mm = float(state.sample.z_mm)
    _, upper_angle = transverse_kick_phase_space_response(
        state,
        float(component.upper_z_mm),
        sample_z_mm,
    )
    _, lower_angle = transverse_kick_phase_space_response(
        state,
        float(component.lower_z_mm),
        sample_z_mm,
    )
    try:
        condition = float(np.linalg.cond(lower_angle))
        if not np.isfinite(condition) or condition > 1.0e12:
            raise np.linalg.LinAlgError("lower AC angular response is singular")
        lower_from_upper = np.linalg.solve(lower_angle, -upper_angle)
    except np.linalg.LinAlgError:
        # A field-free equal-and-opposite pair remains the safest bounded
        # approximation if the current optical map cannot be inverted.
        lower_from_upper = -np.eye(2, dtype=float)
    residual_matrix = upper_angle + lower_angle @ lower_from_upper
    residual = float(
        np.linalg.norm(residual_matrix)
        / max(float(np.linalg.norm(upper_angle)), 1.0e-15)
    )
    component.set_pure_shift_coupling(lower_from_upper, residual)
    component.validate()
    return np.asarray(lower_from_upper, dtype=float), residual


def classify_sample_plane_transfer(transfer) -> tuple[str, float, float]:
    """Classify a plane from the full sample-to-plane first-order map.

    A sample-conjugate image plane has ``J_diff = 0``.  A diffraction plane
    has ``J_img = 0``.  A plane satisfying neither tolerance is explicitly
    reported as mixed contrast rather than inheriting the projector-mode
    label.
    """

    image_residual = float(
        np.linalg.norm(transfer.j_diff_m_per_rad, ord=2)
    )
    diffraction_residual = float(np.linalg.norm(transfer.j_img, ord=2))
    image_ok = image_residual <= IMAGE_CONJUGACY_TOLERANCE_M_PER_RAD
    diffraction_ok = (
        diffraction_residual <= DIFFRACTION_CONJUGACY_TOLERANCE
    )
    if image_ok and not diffraction_ok:
        kind = "image"
    elif diffraction_ok and not image_ok:
        kind = "diffraction"
    elif image_ok and diffraction_ok:
        kind = "degenerate"
    else:
        kind = "mixed"
    return kind, image_residual, diffraction_residual


def synchronize_scan_raster(source, target) -> None:
    """Copy the one physical raster clock/field definition between pairs."""

    for field_name in SHARED_RASTER_FIELDS:
        setattr(target, field_name, getattr(source, field_name))


def _descan_target(state) -> tuple[str, str, float]:
    """Resolve the physical image-reference station used by Descan.

    The Selected Area Aperture is the fixed first-image reference station in
    the column layout.  Its *current* transfer is still classified below: if
    the active lenses do not make it sample-conjugate, the GUI reports mixed
    contrast instead of silently calling it an image plane.
    """

    selected_area = getattr(state, "selected_area_aperture", None)
    if selected_area is not None:
        target = (
            str(selected_area.key),
            str(selected_area.name),
            float(selected_area.z_mm),
        )
    else:
        candidates = tuple(
            plane
            for plane in getattr(state, "recording_planes", ())
            if float(plane.z_mm) > float(state.descan_deflector.lower_z_mm)
        )
        if not candidates:
            raise ValueError(
                "Descan needs a downstream Selected Area Aperture or "
                "recording plane as its image-reference target."
            )
        plane = candidates[0]
        target = str(plane.key), str(plane.name), float(plane.z_mm)
    if target[2] <= float(state.descan_deflector.lower_z_mm):
        raise ValueError(
            "Descan image-reference target must follow both Descan foils."
        )
    return target


def _coil_kick_matrices(component):
    if hasattr(component, "coil_kick_matrices"):
        upper, lower = component.coil_kick_matrices()
    else:
        upper_gain = float(component.upper_coil_gain)
        lower_gain = float(component.lower_coil_gain)
        upper = ((upper_gain, 0.0), (0.0, upper_gain))
        lower = ((lower_gain, 0.0), (0.0, lower_gain))
    return np.asarray(upper, dtype=float), np.asarray(lower, dtype=float)


def paired_kick_response(state, component, observation_z_mm):
    """Map one pair command in radians to position at an observation plane."""

    observation = float(observation_z_mm)
    upper_response = transverse_kick_response(
        state,
        float(component.upper_z_mm),
        observation,
    )
    lower_response = transverse_kick_response(
        state,
        float(component.lower_z_mm),
        observation,
    )
    upper_map, lower_map = _coil_kick_matrices(component)
    return upper_response @ upper_map + lower_response @ lower_map


def paired_kick_response_grid(state, component, z_mm) -> np.ndarray:
    """Return pair-command position response at every requested axial Z.

    The output shape is ``(len(z_mm), 2, 2)`` in metres per radian.  Planes
    upstream of both physical foils have an exact zero response.
    """

    requested = np.asarray(z_mm, dtype=float)
    if requested.ndim != 1 or requested.size == 0:
        raise ValueError("Scan response Z coordinates must be a non-empty 1-D array.")
    if not np.all(np.isfinite(requested)):
        raise ValueError("Scan response Z coordinates must be finite.")
    response = np.zeros((requested.size, 2, 2), dtype=float)
    first_foil_z_mm = min(
        float(component.upper_z_mm),
        float(component.lower_z_mm),
    )
    downstream = requested >= first_foil_z_mm - 1.0e-9
    if not np.any(downstream):
        return response
    paths = _component_paths(
        state,
        component,
        float(np.max(requested[downstream])),
        save_z_mm=requested[downstream],
    )
    upper_map, lower_map = _coil_kick_matrices(component)
    upper_z, upper_response = paths["upper"]
    lower_z, lower_response = paths["lower"]
    for index in np.flatnonzero(downstream):
        observation = float(requested[index])
        response[index] = (
            _matrix_at(upper_z, upper_response, observation) @ upper_map
            + _matrix_at(lower_z, lower_response, observation) @ lower_map
        )
    return response


def calibrate_ac_scan_scale(state):
    """Map requested specimen pixel pitch to a two-axis AC coil command.

    Raster factors are dimensionless pixel-centre coordinates.  The desired
    specimen basis is axis-aligned and has half extents ``N * pitch / 2``;
    solving the active signed first-order response keeps FOV independent of
    lens rotation and anisotropic magnification.
    """

    component = state.ac_deflector
    # Lens excitation and sample Z both move the conjugate planes. Re-solve
    # instead of trusting a component-local flag from an earlier optical state.
    calibrate_ac_pure_shift(state)
    response_m_per_rad = paired_kick_response(
        state,
        component,
        float(state.sample.z_mm),
    )
    condition = float(np.linalg.cond(response_m_per_rad))
    if not np.isfinite(condition) or condition > 1.0e12:
        raise ValueError(
            "AC scan FOV cannot be calibrated because the sample-position "
            "response is singular or ill-conditioned."
        )
    pixel_size_m = float(component.scan_pixel_size_nm) * 1.0e-9
    desired_basis_m = np.diag((
        0.5 * int(component.scan_pixels_x) * pixel_size_m,
        0.5 * int(component.scan_lines) * pixel_size_m,
    ))
    command_matrix_rad = np.linalg.solve(
        response_m_per_rad,
        desired_basis_m,
    )
    realised_basis_m = response_m_per_rad @ command_matrix_rad
    residual = float(
        np.linalg.norm(realised_basis_m - desired_basis_m)
        / max(float(np.linalg.norm(desired_basis_m)), 1.0e-30)
    )
    old_matrix = component.scan_command_matrix_mrad
    old_residual = component.scan_scale_residual
    old_calibrated = bool(
        getattr(component, "_scan_scale_calibrated", False)
    )
    try:
        component.set_scan_command_matrix_mrad(
            command_matrix_rad * 1.0e3,
            residual,
        )
        component.validate()
    except Exception:
        object.__setattr__(
            component,
            "_scan_command_matrix_mrad",
            old_matrix,
        )
        object.__setattr__(component, "_scan_scale_residual", old_residual)
        object.__setattr__(
            component,
            "_scan_scale_calibrated",
            old_calibrated,
        )
        raise
    return np.asarray(command_matrix_rad * 1.0e3), residual


def calibrate_descan_image_plane(state) -> DescanCalibrationResult:
    """Match AC response at the image reference using opposite commands.

    Let ``q`` be the calibrated AC raster command. Descan is driven with
    ``-q``. Its lower-foil 2-D coupling is solved so the paired Descan
    position response equals the paired AC response at the Selected Area
    Aperture station. Therefore ``R_ac q + R_descan (-q) = 0`` to first order,
    including Larmor rotation and anisotropic active optics.
    """

    ac = state.ac_deflector
    descan = state.descan_deflector
    snapshot = dict(descan.__dict__)
    try:
        synchronize_scan_raster(ac, descan)
        command = np.asarray(ac.scan_command_matrix_mrad, dtype=float)
        descan.set_scan_command_matrix_mrad(
            -command,
            float(ac.scan_scale_residual),
        )
        target_key, target_name, target_z_mm = _descan_target(state)
        ac_response = paired_kick_response(state, ac, target_z_mm)
        upper_response = transverse_kick_response(
            state,
            float(descan.upper_z_mm),
            target_z_mm,
        )
        lower_response = transverse_kick_response(
            state,
            float(descan.lower_z_mm),
            target_z_mm,
        )
        lower_from_upper, response_residual = (
            _solve_descan_response_match(
                descan,
                ac_response,
                upper_response,
                lower_response,
            )
        )
        transfer = trace_transverse_transfer(
            state,
            float(state.sample.z_mm),
            target_z_mm,
        )
        plane_kind, image_residual, _ = classify_sample_plane_transfer(
            transfer
        )
        descan.set_image_plane_coupling(
            lower_from_upper,
            response_residual,
            target_z_mm=target_z_mm,
            target_key=target_key,
        )
        descan.validate()
    except Exception:
        for name, value in snapshot.items():
            object.__setattr__(descan, name, value)
        raise
    return DescanCalibrationResult(
        target_key=target_key,
        target_name=target_name,
        target_z_mm=target_z_mm,
        lower_from_upper=np.asarray(lower_from_upper, dtype=float),
        response_match_residual=response_residual,
        conjugacy_residual_m_per_rad=image_residual,
        plane_kind=plane_kind,
    )


def _solve_descan_response_match(
    descan,
    ac_response: np.ndarray,
    upper_response: np.ndarray,
    lower_response: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return the lower-foil map matching one AC response matrix."""

    gain = float(descan.upper_coil_gain)
    if abs(gain) <= 1.0e-12:
        raise ValueError(
            "Descan upper foil gain must be non-zero for image-plane "
            "compensation."
        )
    condition = float(np.linalg.cond(lower_response))
    if not np.isfinite(condition) or condition > 1.0e12:
        raise ValueError(
            "Descan image-plane coupling cannot be calibrated because the "
            "lower-foil response is singular or ill-conditioned."
        )
    lower_map = np.linalg.solve(
        lower_response,
        np.asarray(ac_response, dtype=float) - gain * upper_response,
    )
    lower_from_upper = lower_map / gain
    realised = gain * upper_response + lower_response @ lower_map
    response_residual = float(
        np.linalg.norm(realised - ac_response)
        / max(float(np.linalg.norm(ac_response)), 1.0e-30)
    )
    return np.asarray(lower_from_upper, dtype=float), response_residual


def calibrate_scan_system(state):
    """Calibrate specimen scan and, when active, opposite-command Descan."""

    command, scale_residual = calibrate_ac_scan_scale(state)
    descan = state.descan_deflector
    synchronize_scan_raster(state.ac_deflector, descan)
    descan_result = (
        calibrate_descan_image_plane(state)
        if bool(descan.enabled and descan.scan_enabled)
        else None
    )
    return command, scale_residual, descan_result


def _preview_indices(
    count: int,
    maximum_count: int | None,
) -> np.ndarray:
    count = int(count)
    displayed = (
        count if maximum_count is None else min(count, int(maximum_count))
    )
    if displayed == count:
        return np.arange(count, dtype=int)
    return np.unique(
        np.rint(np.linspace(0, count - 1, displayed)).astype(int)
    )


def raster_sample_grid(
    component,
    *,
    pixels_x: int | None = None,
    pixels_y: int | None = None,
    maximum_count: int | None = MAX_PREVIEW_PIXELS_PER_AXIS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pixel-centre factors and acquisition times for one raster."""

    pixels_x = int(
        component.scan_pixels_x if pixels_x is None else pixels_x
    )
    pixels_y = int(component.scan_lines if pixels_y is None else pixels_y)
    if pixels_x < 2 or pixels_y < 2:
        raise ValueError("A raster requires at least two pixels per axis.")
    columns = _preview_indices(pixels_x, maximum_count)
    rows = _preview_indices(pixels_y, maximum_count)
    column_phase = (columns.astype(float) + 0.5) / float(pixels_x)
    x_factors = 2.0 * column_phase - 1.0
    row_phase = (rows.astype(float) + 0.5) / float(pixels_y)
    y_factors = 2.0 * row_phase - 1.0
    factor_x, factor_y = np.meshgrid(x_factors, y_factors)
    times_s = (
        (rows[:, None].astype(float) + column_phase[None, :])
        / float(pixels_y)
        * float(component.scan_frame_period_s)
    )
    return factor_x, factor_y, times_s


def _scan_kicks_mrad(component, times_s: np.ndarray) -> np.ndarray:
    if not (
        bool(getattr(component, "enabled", False))
        and bool(getattr(component, "scan_enabled", False))
    ):
        return np.zeros((*times_s.shape, 2), dtype=float)
    flat = np.asarray([
        component.scan_kick_mrad(float(value))
        for value in times_s.ravel()
    ], dtype=float)
    return flat.reshape(*times_s.shape, 2)


def _matrix_at(
    z_mm: np.ndarray,
    response: np.ndarray,
    requested_z_mm: float,
) -> np.ndarray:
    requested = float(requested_z_mm)
    if requested < float(z_mm[0]) - 1.0e-9:
        return np.zeros((2, 2), dtype=float)
    return np.array([
        [
            np.interp(requested, z_mm, response[:, row, column])
            for column in range(2)
        ]
        for row in range(2)
    ], dtype=float)


def _component_paths(
    state,
    component,
    stop_z_mm: float,
    *,
    save_z_mm=(),
):
    return {
        "upper": transverse_kick_response_path(
            state,
            float(component.upper_z_mm),
            float(stop_z_mm),
            save_z_mm=save_z_mm,
        ),
        "lower": transverse_kick_response_path(
            state,
            float(component.lower_z_mm),
            float(stop_z_mm),
            save_z_mm=save_z_mm,
        ),
    }


def _pair_displacement_m(
    component,
    kicks_mrad: np.ndarray,
    paths,
    observation_z_mm: float,
) -> np.ndarray:
    if paths is None:
        return np.zeros_like(kicks_mrad, dtype=float)
    kicks_rad = np.asarray(kicks_mrad, dtype=float) * 1.0e-3
    response = _pair_response_from_paths(
        component,
        paths,
        observation_z_mm,
    )
    return np.einsum(
        "ij,...j->...i",
        response,
        kicks_rad,
    )


def _pair_response_from_paths(
    component,
    paths,
    observation_z_mm: float,
) -> np.ndarray:
    """Return one pair response using the exact displayed propagation grid."""

    upper_z, upper_response = paths["upper"]
    lower_z, lower_response = paths["lower"]
    upper_matrix = _matrix_at(
        upper_z, upper_response, observation_z_mm
    )
    lower_matrix = _matrix_at(
        lower_z, lower_response, observation_z_mm
    )
    upper_map, lower_map = _coil_kick_matrices(component)
    return upper_matrix @ upper_map + lower_matrix @ lower_map


def _drift_pivot_z_mm(component) -> float | None:
    """Return the field-free pivot implied by the signed coil gains."""

    if hasattr(component, "pure_shift_lower_ratio_matrix") or hasattr(
        component, "image_plane_lower_ratio_matrix"
    ):
        return None

    upper_gain = float(component.upper_coil_gain)
    lower_gain = float(component.lower_coil_gain)
    total = upper_gain + lower_gain
    if abs(total) <= 1.0e-12:
        return None
    return (
        upper_gain * float(component.upper_z_mm)
        + lower_gain * float(component.lower_z_mm)
    ) / total


def calculate_scan_geometry(state) -> ScanGeometryResult | None:
    """Calculate specimen scan and calibrated downstream descan geometry."""

    ac = state.ac_deflector
    descan = state.descan_deflector
    ac_enabled = bool(ac.enabled and ac.scan_enabled)
    descan_enabled = bool(descan.enabled and descan.scan_enabled)
    if not (ac_enabled or descan_enabled):
        return None
    scan_command_matrix, scan_scale_residual = calibrate_ac_scan_scale(state)
    ac_coupling = (
        np.asarray(ac.pure_shift_lower_ratio_matrix, dtype=float)
        if ac_enabled
        else None
    )
    ac_residual = ac.pure_shift_angular_residual if ac_enabled else None
    descan_calibration = (
        calibrate_descan_image_plane(state) if descan_enabled else None
    )

    driver = ac if ac_enabled else descan
    _, _, times_s = raster_sample_grid(driver)
    ac_kicks = _scan_kicks_mrad(ac, times_s)
    descan_kicks = _scan_kicks_mrad(descan, times_s)

    sample_z_mm = float(state.sample.z_mm)
    observation_planes = [
        plane for plane in state.recording_planes
        if float(plane.z_mm) > sample_z_mm
    ]
    objective_aperture = getattr(state, "objective_aperture", None)
    selected_area = getattr(state, "selected_area_aperture", None)
    existing_keys = {str(plane.key) for plane in observation_planes}
    calculated_suffixes = {}
    sync_objective = getattr(state, "sync_objective", None)
    if callable(sync_objective):
        sync_objective()
    calculated_planes = (
        (
            "objective_back_focal_plane",
            "Objective first diffraction plane",
            getattr(state, "objective_back_focal_plane_z_mm", None),
        ),
        (
            "objective_image_plane",
            "Objective first image plane",
            getattr(state, "objective_image_plane_z_mm", None),
        ),
    )
    for key, name, value in calculated_planes:
        if value is None or not np.isfinite(float(value)):
            continue
        plane_z_mm = float(value)
        if plane_z_mm <= sample_z_mm or key in existing_keys:
            continue
        observation_planes.append(
            _CalculatedObservationPlane(key, name, plane_z_mm)
        )
        existing_keys.add(key)
        calculated_suffixes[key] = " (current objective/sample state)"
    for reference in (objective_aperture, selected_area):
        if (
            reference is not None
            and str(reference.key) not in existing_keys
            and float(reference.z_mm) > sample_z_mm
        ):
            observation_planes.append(reference)
            existing_keys.add(str(reference.key))
    observation_planes.sort(key=lambda plane: float(plane.z_mm))
    stops = [float(state.sample.z_mm)]
    stops.extend(float(plane.z_mm) for plane in observation_planes)
    stop_z_mm = max(stops)
    ac_paths = (
        _component_paths(
            state,
            ac,
            stop_z_mm,
            save_z_mm=stops,
        )
        if ac_enabled or descan_calibration is not None
        else None
    )
    descan_paths = (
        _component_paths(
            state,
            descan,
            stop_z_mm,
            save_z_mm=stops,
        )
        if descan_enabled else None
    )
    if descan_calibration is not None:
        target_z_mm = float(descan_calibration.target_z_mm)
        ac_response = _pair_response_from_paths(
            ac,
            ac_paths,
            target_z_mm,
        )
        upper_z, upper_path = descan_paths["upper"]
        lower_z, lower_path = descan_paths["lower"]
        upper_response = _matrix_at(
            upper_z,
            upper_path,
            target_z_mm,
        )
        lower_response = _matrix_at(
            lower_z,
            lower_path,
            target_z_mm,
        )
        lower_from_upper, response_residual = (
            _solve_descan_response_match(
                descan,
                ac_response,
                upper_response,
                lower_response,
            )
        )
        snapshot = dict(descan.__dict__)
        try:
            descan.set_image_plane_coupling(
                lower_from_upper,
                response_residual,
                target_z_mm=target_z_mm,
                target_key=descan_calibration.target_key,
            )
            descan.validate()
        except Exception:
            for name, value in snapshot.items():
                object.__setattr__(descan, name, value)
            raise
        descan_calibration = DescanCalibrationResult(
            target_key=descan_calibration.target_key,
            target_name=descan_calibration.target_name,
            target_z_mm=target_z_mm,
            lower_from_upper=lower_from_upper,
            response_match_residual=response_residual,
            conjugacy_residual_m_per_rad=(
                descan_calibration.conjugacy_residual_m_per_rad
            ),
            plane_kind=descan_calibration.plane_kind,
        )

    sample_m = _pair_displacement_m(
        ac, ac_kicks, ac_paths, float(state.sample.z_mm)
    )
    plane_positions_um = {}
    plane_names = {}
    plane_roles = {}
    plane_image_residuals = {}
    plane_diffraction_residuals = {}
    plane_transfers = trace_transverse_transfers(
        state,
        sample_z_mm,
        (float(plane.z_mm) for plane in observation_planes),
    )
    for plane in observation_planes:
        observation_z_mm = float(plane.z_mm)
        combined_m = _pair_displacement_m(
            ac, ac_kicks, ac_paths, observation_z_mm
        ) + _pair_displacement_m(
            descan, descan_kicks, descan_paths, observation_z_mm
        )
        plane_positions_um[str(plane.key)] = (
            combined_m[..., 0] * 1.0e6,
            combined_m[..., 1] * 1.0e6,
        )
        plane_key = str(plane.key)
        reference_suffix = calculated_suffixes.get(plane_key, "") or (
            " (physical diffraction-reference station)"
            if plane is objective_aperture
            else " (physical image-reference station)"
            if plane is selected_area
            else ""
        )
        plane_names[plane_key] = f"{plane.name}{reference_suffix}"
        kind, image_residual, diffraction_residual = (
            classify_sample_plane_transfer(
                plane_transfers[observation_z_mm]
            )
        )
        plane_roles[plane_key] = kind
        plane_image_residuals[plane_key] = image_residual
        plane_diffraction_residuals[plane_key] = diffraction_residual

    return ScanGeometryResult(
        times_s=times_s,
        sample_x_um=sample_m[..., 0] * 1.0e6,
        sample_y_um=sample_m[..., 1] * 1.0e6,
        plane_positions_um=plane_positions_um,
        plane_names=plane_names,
        requested_pixels_x=int(driver.scan_pixels_x),
        requested_pixels_y=int(driver.scan_lines),
        ac_enabled=ac_enabled,
        descan_enabled=descan_enabled,
        ac_drift_pivot_z_mm=(
            _drift_pivot_z_mm(ac) if ac_enabled else None
        ),
        descan_drift_pivot_z_mm=(
            _drift_pivot_z_mm(descan) if descan_enabled else None
        ),
        ac_lower_from_upper=(ac_coupling if ac_enabled else None),
        ac_angular_residual=(ac_residual if ac_enabled else None),
        pixel_size_nm=float(driver.scan_pixel_size_nm),
        field_of_view_x_nm=(
            float(driver.scan_field_of_view_x_nm)
        ),
        field_of_view_y_nm=(
            float(driver.scan_field_of_view_y_nm)
        ),
        scan_command_matrix_mrad=scan_command_matrix,
        scan_scale_residual=scan_scale_residual,
        plane_roles=plane_roles,
        plane_image_residuals_m_per_rad=plane_image_residuals,
        plane_diffraction_residuals=plane_diffraction_residuals,
        descan_target_key=(
            descan_calibration.target_key
            if descan_calibration is not None
            else None
        ),
        descan_target_name=(
            descan_calibration.target_name
            if descan_calibration is not None
            else None
        ),
        descan_target_z_mm=(
            descan_calibration.target_z_mm
            if descan_calibration is not None
            else None
        ),
        descan_lower_from_upper=(
            descan_calibration.lower_from_upper
            if descan_calibration is not None
            else None
        ),
        descan_compensation_residual=(
            descan_calibration.response_match_residual
            if descan_calibration is not None
            else None
        ),
        descan_target_conjugacy_residual_m_per_rad=(
            descan_calibration.conjugacy_residual_m_per_rad
            if descan_calibration is not None
            else None
        ),
        descan_target_plane_kind=(
            descan_calibration.plane_kind
            if descan_calibration is not None
            else None
        ),
        ac_distance_above_sample_mm=(
            float(state.sample.z_mm) - float(ac.z_mm)
        ),
        descan_distance_below_sample_mm=(
            float(descan.z_mm) - float(state.sample.z_mm)
        ),
        scan_pair_symmetry_error_mm=abs(
            (float(state.sample.z_mm) - float(ac.z_mm))
            - (float(descan.z_mm) - float(state.sample.z_mm))
        ),
    )


def calculate_scan_ray_paths(state, simulation) -> ScanRayPathResult | None:
    """Precompute scan-response bases used by GUI-only frame playback."""

    ac = state.ac_deflector
    if not bool(ac.enabled and ac.scan_enabled):
        return None
    descan = state.descan_deflector
    descan_active = bool(descan.enabled and descan.scan_enabled)
    calibrate_ac_scan_scale(state)
    if descan_active:
        calibrate_descan_image_plane(state)
    baseline_time_s = float(getattr(state, "simulation_time_s", 0.0))
    zero_response_cache: dict[tuple[int, bytes], np.ndarray] = {}
    response_cache: dict[tuple[str, int, bytes], np.ndarray] = {}

    def response_for(component, z_values, *, active: bool) -> np.ndarray:
        z_values = np.asarray(z_values, dtype=float)
        z_key = (z_values.size, z_values.tobytes())
        if not active:
            return zero_response_cache.setdefault(
                z_key,
                np.zeros((z_values.size, 2, 2), dtype=float),
            )
        key = (str(component.key), *z_key)
        if key not in response_cache:
            response_cache[key] = paired_kick_response_grid(
                state,
                component,
                z_values,
            )
        return response_cache[key]

    branches = (simulation.incident, *simulation.branches.values())
    responses = {
        str(branch.name): (
            response_for(ac, branch.z, active=True),
            response_for(descan, branch.z, active=descan_active),
        )
        for branch in branches
    }
    return ScanRayPathResult(
        responses_m_per_rad=responses,
        baseline_ac_command_mrad=np.asarray(
            ac.scan_kick_mrad(baseline_time_s),
            dtype=float,
        ),
        baseline_descan_command_mrad=np.asarray(
            (
                descan.scan_kick_mrad(baseline_time_s)
                if descan_active
                else (0.0, 0.0)
            ),
            dtype=float,
        ),
        frame_period_s=float(ac.scan_frame_period_s),
        pixels_x=int(ac.scan_pixels_x),
        pixels_y=int(ac.scan_lines),
    )
