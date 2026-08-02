"""Ray-density images at physical recording and observation planes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from temsim.component_keys import EELS_PLANE, VIRTUAL_OBSERVATION_PLANE
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
