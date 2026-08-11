"""First-order AC scan and independent descan geometry.

This module deliberately stops at beam-position geometry.  Specimen
scattering and detector image formation consume the same raster coordinates
later, without making the scan controls depend on TEM/STEM probe presets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from temsim.physics.beam_observation import (
    transverse_kick_phase_space_response,
    transverse_kick_response,
    transverse_kick_response_path,
)


MAX_PREVIEW_PIXELS_PER_AXIS = 128


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
    y_factors = 2.0 * rows.astype(float) / float(pixels_y - 1) - 1.0
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


def _component_paths(state, component, stop_z_mm: float):
    return {
        "upper": transverse_kick_response_path(
            state, float(component.upper_z_mm), float(stop_z_mm)
        ),
        "lower": transverse_kick_response_path(
            state, float(component.lower_z_mm), float(stop_z_mm)
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
    upper_z, upper_response = paths["upper"]
    lower_z, lower_response = paths["lower"]
    upper_matrix = _matrix_at(
        upper_z, upper_response, observation_z_mm
    )
    lower_matrix = _matrix_at(
        lower_z, lower_response, observation_z_mm
    )
    upper_map, lower_map = _coil_kick_matrices(component)
    return np.einsum(
        "ij,...j->...i",
        upper_matrix @ upper_map + lower_matrix @ lower_map,
        kicks_rad,
    )


def _drift_pivot_z_mm(component) -> float | None:
    """Return the field-free pivot implied by the signed coil gains."""

    if hasattr(component, "pure_shift_lower_ratio_matrix"):
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
    """Calculate scan positions without specimen scattering or auto-descan."""

    ac = state.ac_deflector
    descan = state.descan_deflector
    ac_enabled = bool(ac.enabled and ac.scan_enabled)
    descan_enabled = bool(descan.enabled and descan.scan_enabled)
    if not (ac_enabled or descan_enabled):
        return None
    if ac_enabled:
        if bool(getattr(ac, "_pure_shift_calibrated", False)):
            ac_coupling = np.asarray(
                ac.pure_shift_lower_ratio_matrix,
                dtype=float,
            )
            ac_residual = ac.pure_shift_angular_residual
        else:
            ac_coupling, ac_residual = calibrate_ac_pure_shift(state)
    else:
        ac_coupling, ac_residual = None, None

    driver = ac if ac_enabled else descan
    _, _, times_s = raster_sample_grid(driver)
    ac_kicks = _scan_kicks_mrad(ac, times_s)
    descan_kicks = _scan_kicks_mrad(descan, times_s)

    observation_planes = [
        plane for plane in state.recording_planes
        if float(plane.z_mm) > min(
            float(ac.upper_z_mm), float(descan.upper_z_mm)
        )
    ]
    stops = [float(state.sample.z_mm)]
    stops.extend(float(plane.z_mm) for plane in observation_planes)
    stop_z_mm = max(stops)
    ac_paths = _component_paths(state, ac, stop_z_mm) if ac_enabled else None
    descan_paths = (
        _component_paths(state, descan, stop_z_mm)
        if descan_enabled else None
    )

    sample_m = _pair_displacement_m(
        ac, ac_kicks, ac_paths, float(state.sample.z_mm)
    )
    plane_positions_um = {}
    plane_names = {}
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
        plane_names[str(plane.key)] = str(plane.name)

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
    )
