"""Rotation-invariant sample-plane statistics for a weighted ray bundle."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True, slots=True)
class TransverseBeamStatistics:
    surviving_rays: int
    surviving_fraction: float
    mean_x_m: float
    mean_y_m: float
    mean_tx_rad: float
    mean_ty_rad: float
    convergence_rms_rad: float
    convergence_95_rad: float
    convergence_99_rad: float
    convergence_edge_rad: float
    radius_rms_m: float
    radius_95_m: float
    radius_99_m: float
    radial_position_angle_covariance_m_rad: float
    radial_wavefront_curvature_per_m: float
    waist_offset_m: float
    twofold_moment: float = 0.0
    threefold_moment: float = 0.0

    @property
    def illumination_diameter_95_um(self) -> float:
        return 2.0 * self.radius_95_m * 1.0e6

    @property
    def convergence_95_mrad(self) -> float:
        return self.convergence_95_rad * 1.0e3

    @property
    def convergence_99_mrad(self) -> float:
        return self.convergence_99_rad * 1.0e3


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, fraction: float
) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, fraction, side="left"))
    return float(values[order[min(index, values.size - 1)]])


def transverse_beam_statistics(
    x_m,
    y_m,
    tx_rad,
    ty_rad,
    *,
    alive=None,
    weights=None,
) -> TransverseBeamStatistics:
    """Measure one ray bundle relative to its current-weighted chief ray.

    Coordinates are metres and slopes are radians in the laboratory X-Y
    frame.  Radial norms, the dot-product curvature and the waist offset are
    invariant to a common Larmor rotation.  The 95% and 99% values are current
    containment quantiles, not maximum aperture-edge angles.
    """

    x = np.asarray(x_m, dtype=float)
    y = np.asarray(y_m, dtype=float)
    tx = np.asarray(tx_rad, dtype=float)
    ty = np.asarray(ty_rad, dtype=float)
    if not (x.shape == y.shape == tx.shape == ty.shape) or x.ndim != 1:
        raise ValueError("Beam statistics require four equal one-dimensional arrays")
    mask = np.ones(x.size, dtype=bool) if alive is None else np.asarray(
        alive, dtype=bool
    ).copy()
    if mask.shape != x.shape:
        raise ValueError("Beam-statistics alive mask has the wrong shape")
    mask &= np.isfinite(x) & np.isfinite(y) & np.isfinite(tx) & np.isfinite(ty)
    if not np.any(mask):
        raise ValueError("No finite surviving rays are available")

    if weights is None:
        raw_weights = np.ones(x.size, dtype=float)
    else:
        raw_weights = np.asarray(weights, dtype=float)
        if raw_weights.shape != x.shape:
            raise ValueError("Beam-statistics weights have the wrong shape")
        if not np.all(np.isfinite(raw_weights)):
            raise ValueError("Beam-statistics weights must be finite")
        if np.any(raw_weights < 0.0):
            raise ValueError("Beam-statistics weights must be non-negative")
    source_weight = float(np.sum(raw_weights))
    if not math.isfinite(source_weight) or source_weight <= 0.0:
        raise ValueError("Beam rays must have positive total weight")
    selected_weights = raw_weights[mask].copy()
    surviving_weight = float(np.sum(selected_weights))
    if not math.isfinite(surviving_weight) or surviving_weight <= 0.0:
        raise ValueError("Surviving rays must have positive total weight")
    selected_weights /= surviving_weight

    selected_x = x[mask]
    selected_y = y[mask]
    selected_tx = tx[mask]
    selected_ty = ty[mask]

    def weighted_mean(values: np.ndarray) -> float:
        return float(np.sum(selected_weights * values))

    mean_x = weighted_mean(selected_x)
    mean_y = weighted_mean(selected_y)
    directions = np.stack(
        (selected_tx, selected_ty, np.ones_like(selected_tx)), axis=1
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    chief_direction = np.sum(selected_weights[:, None] * directions, axis=0)
    chief_direction /= np.linalg.norm(chief_direction)
    mean_tx = float(chief_direction[0] / chief_direction[2])
    mean_ty = float(chief_direction[1] / chief_direction[2])

    dot = np.clip(directions @ chief_direction, -1.0, 1.0)
    cross_norm = np.linalg.norm(
        np.cross(directions, chief_direction[None, :]), axis=1
    )
    convergence = np.arctan2(cross_norm, dot)
    convergence_rms = math.sqrt(
        float(np.sum(selected_weights * convergence**2))
    )

    centred_x = selected_x - mean_x
    centred_y = selected_y - mean_y
    centred_tx = selected_tx - mean_tx
    centred_ty = selected_ty - mean_ty
    radius = np.hypot(centred_x, centred_y)
    radius_squared = float(np.sum(selected_weights * radius**2))
    angle_squared = float(
        np.sum(selected_weights * (centred_tx**2 + centred_ty**2))
    )
    covariance = float(
        np.sum(
            selected_weights
            * (centred_x * centred_tx + centred_y * centred_ty)
        )
    )
    curvature = covariance / max(radius_squared, 1.0e-30)
    # Linear free-space extrapolation: d<r^2>/dz = 2<r.theta>.
    waist_offset = -covariance / max(angle_squared, 1.0e-30)

    # Centered complex moments detect different angular symmetries. In
    # particular, an equilateral three-lobed spot has an isotropic covariance
    # (zero twofold moment), while its threefold moment can approach one.
    # Normalize the coordinates first to avoid underflow for very small probes.
    # These bounded shape measures are not fitted A1/A2 aberration coefficients.
    positive_current = selected_weights > 0.0
    shape_weights = selected_weights[positive_current]
    radial_scale = float(np.max(radius[positive_current]))
    if radial_scale > 0.0:
        normalized = (
            centred_x[positive_current] + 1j * centred_y[positive_current]
        ) / radial_scale
        normalized_radius = radius[positive_current] / radial_scale

        def shape_moment(order: int) -> float:
            denominator = float(np.sum(shape_weights * normalized_radius**order))
            if denominator <= 0.0:
                return 0.0
            return float(np.clip(
                abs(np.sum(shape_weights * normalized**order)) / denominator,
                0.0, 1.0,
            ))

        twofold_moment = shape_moment(2)
        threefold_moment = shape_moment(3)
    else:
        twofold_moment = threefold_moment = 0.0

    return TransverseBeamStatistics(
        surviving_rays=int(np.count_nonzero(mask)),
        surviving_fraction=surviving_weight / source_weight,
        mean_x_m=mean_x,
        mean_y_m=mean_y,
        mean_tx_rad=mean_tx,
        mean_ty_rad=mean_ty,
        convergence_rms_rad=convergence_rms,
        convergence_95_rad=_weighted_quantile(
            convergence, selected_weights, 0.95
        ),
        convergence_99_rad=_weighted_quantile(
            convergence, selected_weights, 0.99
        ),
        convergence_edge_rad=float(np.max(convergence)),
        radius_rms_m=math.sqrt(radius_squared),
        radius_95_m=_weighted_quantile(radius, selected_weights, 0.95),
        radius_99_m=_weighted_quantile(radius, selected_weights, 0.99),
        radial_position_angle_covariance_m_rad=covariance,
        radial_wavefront_curvature_per_m=curvature,
        waist_offset_m=waist_offset,
        twofold_moment=twofold_moment,
        threefold_moment=threefold_moment,
    )


def branch_sample_statistics(branch) -> TransverseBeamStatistics:
    """Measure the last plane of one simulation branch."""

    return transverse_beam_statistics(
        branch.x[-1],
        branch.y[-1],
        branch.tx[-1],
        branch.ty[-1],
        alive=branch.alive,
        weights=branch.ray_weight,
    )
