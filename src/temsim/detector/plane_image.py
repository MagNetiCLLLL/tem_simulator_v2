"""Ray-density images at physical recording and observation planes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from temsim.detector.point_spread import (
    DetectorPointSpread,
    apply_point_spread,
)
from temsim.physics.beam_observation import observation_slices


@dataclass(frozen=True)
class DetectorResponseImage:
    """Weighted virtual optical-reference hits, not acquired detector counts.

    ``intensity`` alone is peak-normalized for display. The two other images
    and their sums preserve the supplied weights, including an all-zero beam.
    """

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
        item_weights = np.asarray(item.weight, dtype=float)
        if (item_weights.shape != np.shape(item.x_m)
                or not np.all(np.isfinite(item_weights)) or np.any(item_weights < 0.0)):
            raise ValueError("Detector weights must be finite, non-negative and match the rays.")
        finite = np.isfinite(item.x_m) & np.isfinite(item.y_m)
        if not np.any(finite):
            continue
        positions.append(
            np.column_stack((item.x_m[finite], item.y_m[finite])) * 1.0e3
        )
        weights.append(item_weights[finite])
    if not positions:
        return np.empty((0, 2), dtype=float), np.empty(0, dtype=float)
    positions = np.vstack(positions)
    weights = np.concatenate(weights)
    return positions, weights


def detector_response_image(simulation, state, key, *, pixels=192):
    """Render accepted detector hits followed by the detector-plane PSF.

    This is a virtual optical-reference response, not physical collected current
    or a readout channel. Recording planes are sampled virtually so inspecting a retracted or
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
    centre = np.array([getattr(plane, "centre_offset_x_mm", 0.0),
                       getattr(plane, "centre_offset_y_mm", 0.0)], dtype=float)
    if not np.isfinite(active_half_span) or active_half_span <= 0 or not np.all(np.isfinite(centre)):
        raise ValueError("Detector width must be finite and positive and its centre finite.")
    if positions.size:
        coordinate_span = float(np.max(np.abs(positions - centre)))
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
    x_edges, y_edges = edges + centre[0], edges + centre[1]
    ideal, _, _ = np.histogram2d(
        positions[:, 1] if positions.size else np.empty(0),
        positions[:, 0] if positions.size else np.empty(0),
        bins=(y_edges, x_edges),
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
    xx, yy = np.meshgrid(centres + centre[0], centres + centre[1])
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
        extent=(float(x_edges[0]), float(x_edges[-1]), float(y_edges[0]), float(y_edges[-1])),
        unit="mm",
        accepted_weight=accepted_weight,
        response_weight=response_weight,
        retained_fraction=float(retained_fraction),
        point_spread=point_spread,
    )
