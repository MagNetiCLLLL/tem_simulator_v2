"""Bounded display-only reduction of one connected chronological polyline.

Coordinates are logical screen pixels. Returned indices refer to the original
vertices; this helper neither interpolates new positions nor changes physical
histories. Callers must split clipped or disconnected runs before using it.
"""

from __future__ import annotations

import math
import operator

import numpy as np


def simplify_screen_vertices(points_xy, tolerance_px=.1, maximum_work=None) -> np.ndarray:
    """Retain chronological vertices within a finite-segment screen error bound.

    Iterative Ramer--Douglas--Peucker reduction preserves endpoints, global X/Y
    extent representatives, and both edges of an X turning plateau. The default
    budget is 64 times the vertex count, counted as point-distance evaluations.
    Exhausting that budget returns every original index, so a difficult path
    cannot cause unbounded quadratic GUI work or silently exceed the tolerance.
    Finite inputs whose differences overflow also retain the complete polyline.
    """
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("points_xy must contain finite screen coordinates with shape (N, 2)")
    if isinstance(tolerance_px, (bool, np.bool_)):
        raise ValueError("tolerance_px must be finite and positive")
    tolerance = float(tolerance_px)
    if not math.isfinite(tolerance) or tolerance <= 0.:
        raise ValueError("tolerance_px must be finite and positive")
    count = len(points)
    if maximum_work is None:
        budget = 64 * count
    else:
        try:
            budget = operator.index(maximum_work)
        except TypeError as error:
            raise ValueError("maximum_work must be a non-negative integer or None") from error
        if isinstance(maximum_work, (bool, np.bool_)) or budget < 0:
            raise ValueError("maximum_work must be a non-negative integer or None")
    all_indices = np.arange(count, dtype=np.intp)
    if count <= 2:
        return all_indices

    retained = np.zeros(count, dtype=bool)
    retained[[0, -1]] = True
    # First and last occurrences preserve extremum plateau edges without making
    # a constant coordinate retain every sample of an otherwise straight line.
    for axis in range(2):
        values = points[:, axis]
        for extremum in (values.min(), values.max()):
            matches = np.flatnonzero(values == extremum)
            retained[matches[[0, -1]]] = True
    direction = ((points[1:, 0] > points[:-1, 0]).astype(np.int8)
                 - (points[1:, 0] < points[:-1, 0]).astype(np.int8))
    moving = np.flatnonzero(direction)
    if len(moving) > 1:
        reversals = direction[moving[:-1]] != direction[moving[1:]]
        retained[moving[:-1][reversals] + 1] = True
        retained[moving[1:][reversals]] = True

    landmarks = np.flatnonzero(retained)
    pending = list(zip(landmarks[:-1], landmarks[1:]))
    work = 0
    while pending:
        first, last = pending.pop()
        interior_count = last - first - 1
        if interior_count <= 0:
            continue
        if work + interior_count > budget:
            return all_indices
        work += interior_count
        with np.errstate(over="ignore", invalid="ignore"):
            delta = points[last] - points[first]
            relative = points[first + 1:last] - points[first]
        length = math.hypot(*delta)
        if not math.isfinite(length) or not np.isfinite(relative).all():
            return all_indices
        if length == 0.:
            with np.errstate(over="ignore", invalid="ignore"):
                distance = np.hypot(relative[:, 0], relative[:, 1])
        else:
            unit = delta / length
            with np.errstate(over="ignore", invalid="ignore"):
                along = np.clip(relative[:, 0] * unit[0] + relative[:, 1] * unit[1], 0., length)
                offset = relative - along[:, None] * unit
                distance = np.hypot(offset[:, 0], offset[:, 1])
        if not np.isfinite(distance).all():
            return all_indices
        split_offset = int(np.argmax(distance))
        if distance[split_offset] > tolerance:
            split = first + 1 + split_offset
            retained[split] = True
            pending.extend(((first, split), (split, last)))
    return np.flatnonzero(retained)


__all__ = ["simplify_screen_vertices"]
