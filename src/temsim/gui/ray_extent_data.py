"""Read completed axial coverage without deriving it from displayed ray tails.

Published calculations and archive loaders own checkpoint validation. This
small presentation reader neither hashes large arrays nor admits restart data.
All returned coordinates use the main Ray Diagram's axial Z in millimetres;
energy-filter local path coordinates are deliberately excluded.
"""
from __future__ import annotations

import math


def _finite(value):
    if isinstance(value, (bool, str, bytes)):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def completed_ray_extent(result) -> dict:
    """Return only executed endpoint metadata belonging to the displayed result.

    The current requested plane is intentionally not an input. Material exit
    checkpoints can reside separately from the incident section checkpoint, so
    its last incident Z is not a valid cap for the complete result's endpoint.
    """
    empty = dict(start_z_mm=None, completed_z_mm=None, resumable_z_mm=None)
    simulation = getattr(result, "simulation", None)
    checkpoint = getattr(simulation, "section_checkpoint", None)
    if checkpoint is None:
        return empty
    metrics = getattr(simulation, "metrics", None) or {}
    completed = _finite(metrics.get("section_target_z_mm"))
    if completed is None:
        return empty
    trace = getattr(checkpoint, "gun_trace", None)
    if trace is None:
        trace = getattr(simulation, "gun_trace", None)
    z = getattr(trace, "z_mm", ())
    try:
        start = _finite(z[0])
    except (IndexError, TypeError, KeyError):
        start = None
    if start is not None and completed < start:
        return empty
    resumable = _finite(metrics.get("section_resumable_through_z_mm"))
    if resumable is not None and (resumable > completed
                                  or (start is not None and resumable < start)):
        resumable = None
    return dict(start_z_mm=start, completed_z_mm=completed, resumable_z_mm=resumable)
