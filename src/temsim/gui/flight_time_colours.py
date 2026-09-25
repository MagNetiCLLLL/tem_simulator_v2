"""A shared time colour scale from executed, physically covered histories.

Times are laboratory seconds since tip emission. This display-only reduction
does not reconstruct time from positions, extend transport or alter archives.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from temsim.gui.beam_display_source import downstream_display_branches


def _maximum(values, valid=None):
    values = np.asarray(values)
    known = np.isfinite(values) & (values >= 0)
    if valid is not None:
        known &= valid
    return float(np.max(values, where=known, initial=0.))


def _branch_maximum(branch):
    raw = getattr(branch, "flight_time_s", None)
    if raw is None:
        return 0.
    times = np.asarray(raw)
    z = np.asarray(branch.z)
    stops = np.asarray(branch.blocked_z)
    if times.shape != np.shape(branch.x) or times.ndim != 2 or stops.shape != (times.shape[1],):
        return 0.
    maximum = 0.
    # Bound temporary masks; do not copy/promote the full ray history.
    for start in range(0, len(z), 64):
        end = start + 64
        covered = np.isnan(stops)[None, :] | (z[start:end, None] <= stops[None, :])
        covered &= np.isfinite(branch.x[start:end]) & np.isfinite(branch.y[start:end])
        maximum = max(maximum, _maximum(times[start:end], covered))
    # A stop between saved rows is a physical endpoint, not the next row.
    columns = np.flatnonzero(np.isfinite(stops) & (stops > z[0]) & (stops < z[-1]))
    if columns.size:
        right = np.searchsorted(z, stops[columns], side="left")
        left = right - 1
        fraction = (stops[columns] - z[left]) / (z[right] - z[left])
        values = times[left, columns] + fraction * (times[right, columns] - times[left, columns])
        maximum = max(maximum, _maximum(values))
    return maximum


@dataclass(frozen=True, slots=True)
class FlightTimeColourScale:
    """One common 0..maximum_s scale, independent of plane and display subset."""
    maximum_s: float

    @classmethod
    def from_result(cls, result):
        simulation = getattr(result, "simulation", None)
        incident = getattr(simulation, "incident", None)
        branches, _ = downstream_display_branches(result)
        maximum = max((_branch_maximum(b) for b in (incident, *branches) if b is not None), default=0.)
        gun = getattr(simulation, "gun_trace", None)
        history = getattr(gun, "equal_time_history", None)
        if history is not None:
            active = np.asarray(history.alive) & ~np.asarray(history.completed)
            rows = np.any(active, axis=1)
            maximum = max(maximum, _maximum(history.time_s, rows))
        output = getattr(result, "energy_filter", None)
        for plane in getattr(output, "timed_planes", ()):
            if getattr(plane, "time_reference", None) == "simultaneous_tip_emission":
                maximum = max(maximum, _maximum(plane.time_s, np.asarray(plane.reached)))
        return cls(maximum)

    @property
    def label(self):
        unit, factor = (("fs", 1e15) if self.maximum_s < 1e-12 else
                        ("ps", 1e12) if self.maximum_s < 1e-9 else ("ns", 1e9))
        return f"0–{self.maximum_s * factor:.6g} {unit} since emission"
