"""Beam phase-space slices and cached AC-kick response at an observation plane."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from temsim.physics.core import propagate


@dataclass(frozen=True)
class BeamObservationSlice:
    name: str
    colour: tuple
    z_mm: float
    x_m: np.ndarray
    y_m: np.ndarray
    tx_rad: np.ndarray
    ty_rad: np.ndarray
    weight: np.ndarray


def _interpolate_rows(branch, z_mm):
    z = np.asarray(branch.z, dtype=float)
    requested = float(z_mm)
    if requested < z[0] - 1.0e-9 or requested > z[-1] + 1.0e-9:
        raise ValueError(
            f"Observation plane {requested:g} mm is outside branch "
            f"range {z[0]:g}-{z[-1]:g} mm."
        )
    upper = int(np.searchsorted(z, requested, side="left"))
    if upper == 0:
        lower = upper = 0
        fraction = 0.0
    elif upper >= len(z):
        lower = upper = len(z) - 1
        fraction = 0.0
    elif abs(z[upper] - requested) <= 1.0e-9:
        lower = upper
        fraction = 0.0
    else:
        lower = upper - 1
        fraction = (requested - z[lower]) / (z[upper] - z[lower])

    def sample(values):
        if lower == upper:
            return np.asarray(values[lower], dtype=float)
        return (
            (1.0 - fraction) * np.asarray(values[lower], dtype=float)
            + fraction * np.asarray(values[upper], dtype=float)
        )

    return tuple(sample(values) for values in (
        branch.x,
        branch.y,
        branch.tx,
        branch.ty,
    ))


def _survives_at_plane(branch, z_mm, ignored_stop_keys=()):
    blocked_z = np.asarray(branch.blocked_z, dtype=float)
    survives = (
        np.asarray(branch.alive, dtype=bool)
        | np.isnan(blocked_z)
        | (blocked_z > float(z_mm) + 1.0e-9)
    )
    if ignored_stop_keys:
        survives |= np.isin(
            np.asarray(branch.blocked_key, dtype=object),
            tuple(str(key) for key in ignored_stop_keys),
        )
    return survives


def observation_slices(
    simulation,
    sample_z_mm,
    z_mm,
    *,
    ignored_stop_keys=(),
    branch_names=None,
):
    """Return weighted transverse phase-space at one requested TEM plane."""
    requested = float(z_mm)
    if requested <= float(sample_z_mm) + 1.0e-9:
        branches = (simulation.incident,)
    else:
        branches = tuple(simulation.branches.values())

    selected_names = None if branch_names is None else set(branch_names)
    result = []
    for branch in branches:
        if selected_names is not None and branch.name not in selected_names:
            continue
        x_m, y_m, tx_rad, ty_rad = _interpolate_rows(branch, requested)
        keep = _survives_at_plane(
            branch, requested, ignored_stop_keys=ignored_stop_keys
        )
        ray_weight = getattr(branch, "ray_weight", None)
        if ray_weight is None:
            weights = np.ones_like(x_m)
        else:
            weights = np.asarray(ray_weight, dtype=float)
        weights = weights * float(getattr(branch, "weight", 1.0))
        result.append(BeamObservationSlice(
            name=branch.name,
            colour=branch.colour,
            z_mm=requested,
            x_m=x_m[keep],
            y_m=y_m[keep],
            tx_rad=tx_rad[keep],
            ty_rad=ty_rad[keep],
            weight=weights[keep],
        ))
    return tuple(result)


def transverse_kick_response_path(state, start_z_mm, observation_z_mm):
    """Return position-response matrices along a downstream ray path."""
    start = float(start_z_mm)
    stop = float(observation_z_mm)
    if stop <= start:
        return (
            np.array([start], dtype=float),
            np.zeros((1, 2, 2), dtype=float),
        )
    probe_rad = 1.0e-6
    zeros = np.zeros(2, dtype=float)
    z, x, _, y, _ = propagate(
        state,
        start,
        stop,
        zeros,
        np.array([probe_rad, 0.0]),
        zeros,
        np.array([0.0, probe_rad]),
    )
    response = np.empty((len(z), 2, 2), dtype=float)
    response[:, 0, :] = x
    response[:, 1, :] = y
    return np.asarray(z, dtype=float), response / probe_rad


def transverse_kick_response(state, start_z_mm, observation_z_mm):
    """Return a 2x2 position response matrix in metres per radian."""
    _, response = transverse_kick_response_path(
        state,
        start_z_mm,
        observation_z_mm,
    )
    return response[-1]


def transverse_kick_phase_space_response(
    state,
    start_z_mm,
    observation_z_mm,
):
    """Return position and angle response matrices for a transverse kick.

    The two columns correspond to unit laboratory X and Y angular kicks.  The
    position matrix is in metres per radian and the angle matrix is in radians
    per radian.  Keeping the signed 2x2 angle map is important for a scan pair:
    magnetic lenses can rotate the lower-coil correction relative to the
    upper-coil kick, so a scalar equal-and-opposite drive is not generally a
    pure translation at the specimen.
    """

    start = float(start_z_mm)
    stop = float(observation_z_mm)
    if stop <= start:
        zeros = np.zeros((2, 2), dtype=float)
        return zeros, zeros.copy()
    probe_rad = 1.0e-6
    zeros = np.zeros(2, dtype=float)
    _, x, tx, y, ty = propagate(
        state,
        start,
        stop,
        zeros,
        np.array([probe_rad, 0.0]),
        zeros,
        np.array([0.0, probe_rad]),
    )
    position = np.array(
        (
            (x[-1, 0], x[-1, 1]),
            (y[-1, 0], y[-1, 1]),
        ),
        dtype=float,
    ) / probe_rad
    angle = np.array(
        (
            (tx[-1, 0], tx[-1, 1]),
            (ty[-1, 0], ty[-1, 1]),
        ),
        dtype=float,
    ) / probe_rad
    return position, angle


def transverse_scan_outline(
    response_per_rad,
    amplitude_x_mrad,
    amplitude_y_mrad,
    center=(0.0, 0.0),
):
    """Map angular half-ranges to a closed scan-field parallelogram."""

    response = np.asarray(response_per_rad, dtype=float)
    if response.shape != (2, 2):
        raise ValueError("Scan response must be a 2x2 matrix.")
    center = np.asarray(center, dtype=float)
    if center.shape != (2,):
        raise ValueError("Scan-field center must contain X and Y.")
    ax_rad = abs(float(amplitude_x_mrad)) * 1.0e-3
    ay_rad = abs(float(amplitude_y_mrad)) * 1.0e-3
    corners_rad = np.array([
        [-ax_rad, -ay_rad],
        [ax_rad, -ay_rad],
        [ax_rad, ay_rad],
        [-ax_rad, ay_rad],
        [-ax_rad, -ay_rad],
    ])
    return center + corners_rad @ response.T
