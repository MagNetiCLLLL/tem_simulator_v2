"""Ray-density images at physical recording and observation planes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from temsim.component_keys import EELS_PLANE, VIRTUAL_OBSERVATION_PLANE
from temsim.detector.point_spread import (
    DetectorPointSpread,
    apply_point_spread,
)
from temsim.physics.beam_observation import observation_slices


@dataclass(frozen=True)
class PlaneRayImage:
    key: str
    name: str
    intensity: np.ndarray
    extent: tuple[float, float, float, float]
    unit: str
    centroid_x: float
    centroid_y: float
    rms_radius: float
    represented_weight: float


@dataclass(frozen=True)
class DetectorResponseImage:
    """Ideal detector hits and their forward PSF response on one plane."""

    key: str
    name: str
    ideal_intensity: np.ndarray
    response_intensity: np.ndarray
    intensity: np.ndarray
    extent: tuple[float, float, float, float]
    unit: str
    accepted_weight: float
    response_weight: float
    retained_fraction: float
    point_spread: DetectorPointSpread


def _histogram_image(key, name, positions, weights, pixels, unit="mm"):
    n = max(32, int(pixels))
    if not positions:
        return PlaneRayImage(
            str(key), name, np.zeros((n, n)), (-1.0, 1.0, -1.0, 1.0),
            unit, float("nan"), float("nan"), float("nan"), 0.0,
        )
    positions = np.vstack(positions)
    weights = np.concatenate(weights)
    if float(weights.sum()) <= 0.0:
        weights = np.ones(positions.shape[0], dtype=float)
    total = float(weights.sum())
    centroid = np.sum(positions * weights[:, None], axis=0) / total
    radial = np.hypot(positions[:, 0], positions[:, 1])
    half_span = max(float(np.percentile(radial, 99.5)) * 1.15, 1.0e-9)
    edges = np.linspace(-half_span, half_span, n + 1)
    intensity, _, _ = np.histogram2d(
        positions[:, 1], positions[:, 0], bins=(edges, edges), weights=weights
    )
    peak = float(intensity.max())
    if peak > 0.0:
        intensity = intensity / peak
    rms_radius = np.sqrt(
        np.sum(weights * np.sum((positions - centroid) ** 2, axis=1)) / total
    )
    return PlaneRayImage(
        str(key), name, intensity,
        (-half_span, half_span, -half_span, half_span), unit,
        float(centroid[0]), float(centroid[1]), float(rms_radius), total,
    )


def _column_plane_image(
    simulation, state, key, name, z_mm, pixels
):
    ignored = tuple(item.key for item in state.recording_planes)
    slices = observation_slices(
        simulation,
        state.sample.z_mm,
        float(z_mm),
        ignored_stop_keys=ignored,
    )
    positions = []
    weights = []
    for item in slices:
        finite = np.isfinite(item.x_m) & np.isfinite(item.y_m)
        if not np.any(finite):
            continue
        positions.append(
            np.column_stack((item.x_m[finite], item.y_m[finite])) * 1.0e3
        )
        weights.append(np.maximum(np.asarray(item.weight)[finite], 0.0))
    return _histogram_image(
        key, name, positions, weights, pixels
    )


def _column_plane_samples(simulation, state, z_mm):
    ignored = tuple(item.key for item in state.recording_planes)
    slices = observation_slices(
        simulation,
        state.sample.z_mm,
        float(z_mm),
        ignored_stop_keys=ignored,
    )
    positions = []
    weights = []
    for item in slices:
        finite = np.isfinite(item.x_m) & np.isfinite(item.y_m)
        if not np.any(finite):
            continue
        positions.append(
            np.column_stack((item.x_m[finite], item.y_m[finite])) * 1.0e3
        )
        weights.append(np.maximum(np.asarray(item.weight)[finite], 0.0))
    if not positions:
        return np.empty((0, 2), dtype=float), np.empty(0, dtype=float)
    positions = np.vstack(positions)
    weights = np.concatenate(weights)
    if float(weights.sum()) <= 0.0:
        weights = np.ones(positions.shape[0], dtype=float)
    return positions, weights


def detector_response_image(simulation, state, key, *, pixels=192):
    """Render accepted detector hits followed by the detector-plane PSF.

    Recording planes are sampled virtually so inspecting a retracted or
    downstream detector does not change the ray trace.  Its active-area mask
    is nevertheless applied before and after the PSF, so response spreading
    into a central hole or beyond the sensitive outer edge is explicitly lost.
    """

    key = str(key)
    plane = next(
        (item for item in state.recording_planes if item.key == key), None
    )
    if plane is None:
        raise ValueError(f"Unknown physical detector plane: {key}")
    point_spread = DetectorPointSpread.from_component(plane)
    positions, weights = _column_plane_samples(
        simulation, state, plane.z_mm
    )
    if positions.size:
        accepted = np.asarray(
            plane.hit_mask(positions[:, 0], positions[:, 1]), dtype=bool
        )
        positions = positions[accepted]
        weights = weights[accepted]

    n = max(64, int(pixels))
    active_half_span = 0.5 * float(plane.outer_width_mm)
    if positions.size:
        coordinate_span = float(np.max(np.abs(positions)))
        spread_support = 4.0 * max(
            point_spread.sigma_x_mm,
            point_spread.sigma_y_mm,
        )
        minimum_span = max(
            spread_support,
            active_half_span * 16.0 / n,
            1.0e-9,
        )
        half_span = min(
            active_half_span,
            max(coordinate_span + spread_support, minimum_span),
        )
    else:
        half_span = active_half_span
    edges = np.linspace(-half_span, half_span, n + 1)
    ideal, _, _ = np.histogram2d(
        positions[:, 1] if positions.size else np.empty(0),
        positions[:, 0] if positions.size else np.empty(0),
        bins=(edges, edges),
        weights=weights if positions.size else None,
    )
    pixel_size = 2.0 * half_span / n
    response = apply_point_spread(
        ideal,
        point_spread,
        pixel_size_x_mm=pixel_size,
        pixel_size_y_mm=pixel_size,
    )
    centres = 0.5 * (edges[:-1] + edges[1:])
    xx, yy = np.meshgrid(centres, centres)
    active = np.asarray(plane.hit_mask(xx, yy), dtype=bool)
    response = np.where(active, response, 0.0)
    accepted_weight = float(ideal.sum())
    response_weight = float(response.sum())
    retained_fraction = (
        response_weight / accepted_weight
        if accepted_weight > 0.0 else float("nan")
    )
    peak = float(response.max())
    display = response / peak if peak > 0.0 else response.copy()
    return DetectorResponseImage(
        key=key,
        name=str(plane.name),
        ideal_intensity=ideal,
        response_intensity=response,
        intensity=display,
        extent=(-half_span, half_span, -half_span, half_span),
        unit="mm",
        accepted_weight=accepted_weight,
        response_weight=response_weight,
        retained_fraction=float(retained_fraction),
        point_spread=point_spread,
    )


def _eels_plane_image(state, pixels):
    result = getattr(state, "energy_filter_result", None)
    if result is None:
        return _histogram_image(EELS_PLANE, "EELS", [], [], pixels)
    positions = []
    # The first stored path is the unweighted reference trajectory.  The
    # remaining paths correspond one-to-one with stop_keys.
    for path_x, path_y, stop_key in zip(
        result.paths_u_mm[1:], result.paths_y_mm[1:], result.stop_keys
    ):
        if stop_key != EELS_PLANE or len(path_x) == 0 or len(path_y) == 0:
            continue
        positions.append(np.asarray([[path_x[-1], path_y[-1]]], dtype=float))
    weights = [np.ones(len(item), dtype=float) for item in positions]
    return _histogram_image(EELS_PLANE, "EELS", positions, weights, pixels)


def plane_ray_image(simulation, state, key, *, pixels=96):
    """Render the rays that reach a selected physical plane.

    Column recording devices are treated as virtual observation planes so
    choosing one display does not insert it or stop rays before another view.
    Aperture and column-wall losses are still respected.
    """
    key = str(key)
    if key == EELS_PLANE:
        return _eels_plane_image(state, pixels)
    if key == VIRTUAL_OBSERVATION_PLANE:
        z_mm = float(state.virtual_observation_z_mm)
        return _column_plane_image(
            simulation,
            state,
            key,
            f"Virtual Z = {z_mm:g} mm",
            z_mm,
            pixels,
        )
    plane = next(
        (item for item in state.recording_planes if item.key == key), None
    )
    if plane is None:
        raise ValueError(f"Unknown physical observation plane: {key}")
    return _column_plane_image(
        simulation, state, plane.key, plane.name, plane.z_mm, pixels
    )
