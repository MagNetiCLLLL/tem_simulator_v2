"""Executed classical particle energies; never nominal-voltage substitutes."""
from __future__ import annotations

import numpy as np


def validate_kinetic_energy_array(value, shape, label="Particle", *, allow_unknown=False, required=None):
    if value is None:
        return False
    energy = np.asarray(value)
    if (energy.shape != tuple(shape) or energy.dtype != np.dtype(np.float64)
            or np.any(np.isinf(energy)) or np.any(energy <= 0.)
            or (not allow_unknown and np.any(np.isnan(energy)))):
        raise ValueError(f"{label} kinetic energies must be positive float64 values matching geometry")
    if required is not None:
        mask = np.asarray(required, dtype=bool)
        if mask.shape != energy.shape or np.any(~np.isfinite(energy[mask])):
            raise ValueError(f"{label} surviving particles require finite executed kinetic energies")
    return True


def energy_survival_mask(branch, z_mm):
    """Only pre-stop and currently alive particles need a usable energy."""
    z = np.asarray(z_mm, dtype=float)[:, None]
    blocked = np.asarray(branch.blocked_z, dtype=float)[None, :]
    alive = np.asarray(branch.alive, dtype=bool)[None, :]
    return (np.isnan(blocked) & alive) | (z < blocked-1e-9)


def sample_kinetic_energy(branch, z_mm):
    """Read/interpolate retained energy; None means unavailable historical data.

    Exact stored planes retain their values. This is a readout interpolation,
    never the state used to resume an integration between checkpoints.
    """
    values = getattr(branch, "kinetic_energy_ev", None)
    if values is None:
        return None
    z = np.asarray(branch.z, dtype=float)
    energy = np.asarray(values)
    validate_kinetic_energy_array(energy, np.shape(branch.x), allow_unknown=True)
    target = float(z_mm)
    if not np.isfinite(target) or not len(z) or target < z[0] or target > z[-1]:
        return np.full(energy.shape[1], np.nan)
    upper = int(np.searchsorted(z, target, side="left"))
    if z[upper] == target:
        return energy[upper].copy()
    lower = upper-1
    fraction = (target-z[lower])/(z[upper]-z[lower])
    return energy[lower]*(1.-fraction)+energy[upper]*fraction
