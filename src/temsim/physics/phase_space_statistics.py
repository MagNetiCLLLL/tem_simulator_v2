"""Common current-weighted population covariance; no current renormalisation.

Coordinates: (x [m], theta_x [rad], y [m], theta_y [rad]). Emittances are
geometric projected RMS m rad, not relativistically normalised emittances.
"""
import numpy as np


def weighted_phase_space_statistics(arrays):
    values = np.column_stack([arrays[k] for k in ("x_m", "tx_rad", "y_m", "ty_rad")])
    weight = np.asarray(arrays["weight"], dtype=float)
    alive = np.asarray(arrays["alive"], dtype=bool)
    if values.ndim != 2 or weight.shape != (len(values),) or alive.shape != weight.shape:
        raise ValueError("Phase-space arrays must describe one shared population")
    if not np.all(np.isfinite(weight)) or np.any(weight < 0):
        raise ValueError("Particle weights must be finite and nonnegative")
    mask = alive & (weight > 0)
    if not np.any(mask) or not np.all(np.isfinite(values[mask])):
        raise ValueError("No finite positive-weight transmitted phase-space population")
    selected = weight[mask]
    with np.errstate(over="ignore"):
        transmitted_weight = float(selected.sum())
    if not np.isfinite(transmitted_weight):
        raise ValueError("Transmitted weight exceeds finite numerical support")
    p = selected / transmitted_weight
    data = values[mask]
    mean = np.sum(data*p[:, None], axis=0)
    centered = data-mean
    cov = (centered*p[:, None]).T @ centered
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(cov)):
        raise ValueError("Phase-space moments exceed finite numerical support")
    effective = float(1./np.sum(p*p))
    emittances = [float(np.sqrt(max(0., np.linalg.det(cov[np.ix_(ids, ids)]))))
                 for ids in ((0, 1), (2, 3))]
    return dict(mean=mean, covariance=cov, effective_samples=effective,
                transmitted_weight=transmitted_weight, rms_emittance_m_rad=tuple(emittances),
                coordinate_order=("x_m", "tx_rad", "y_m", "ty_rad"),
                phase_status="NOT_COMPUTED", convention="current-weighted population moments")
