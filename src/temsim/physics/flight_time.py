"""Read executed classical clocks; never reconstruct time from drawing paths.

Seconds in the laboratory frame, relative to the common tip emission event.
Sparse saved clocks can be interpolated between adjacent planes; this is a
display interpolation, not quantum phase or a new particle source.
"""
from __future__ import annotations

import numpy as np

FLIGHT_TIME_SCHEMA = "tip-origin-lab-clock-v1"


def sample_flight_time(branch, z_mm):
    """Per-path arrival at a covered plane; NaN for unknown/non-arriving paths.

    Only two float64 clock rows are read. In particular, this does not copy a
    full history, divide geometric chord lengths by an assumed velocity, or
    associate a single time with all descendants of one source particle.
    """
    positions = np.asarray(getattr(branch, "x", ()))
    count = positions.shape[1] if positions.ndim == 2 else 0
    missing = np.full(count, np.nan, dtype=np.float64)
    raw = getattr(branch, "flight_time_s", None)
    if raw is None:
        return missing
    try:
        z = np.asarray(branch.z, dtype=np.float64)
        times = np.asarray(raw)
        plane = float(z_mm)
        if (z.ndim != 1 or not z.size or not np.isfinite(plane)
                or not np.all(np.isfinite(z)) or np.any(np.diff(z) <= 0)
                or times.shape != (z.size, count) or times.dtype.kind != "f"
                or plane < z[0] or plane > z[-1]):
            return missing
        right = int(np.searchsorted(z, plane, side="left"))
        if right == 0 or z[right] == plane:
            values = np.asarray(times[right], dtype=np.float64).copy()
        else:
            left = right - 1
            before = np.asarray(times[left], dtype=np.float64)
            after = np.asarray(times[right], dtype=np.float64)
            fraction = (plane - z[left]) / (z[right] - z[left])
            values = before + fraction * (after - before)
            # First crossings of a turning gun trajectory need not have
            # monotonically increasing times with increasing Z.
        blocked = np.asarray(branch.blocked_z, dtype=np.float64)
        if blocked.shape != (count,):
            return missing
        valid = ((np.isnan(blocked) | (blocked >= plane - 1e-9))
                 & np.isfinite(values) & (values >= 0))
        return np.where(valid, values, np.nan)
    except (AttributeError, TypeError, ValueError, IndexError, OverflowError):
        return missing
