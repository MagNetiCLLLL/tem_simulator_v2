"""Stable numerical cell selection, independent of physical field evaluation."""
from __future__ import annotations

import numpy as np
from temsim.physics.discrete_gradient import register_jitable


@register_jitable(inline="always")
def step_cell_index(axis, value):
    """Resolve a roundoff-sized face tie consistently for step scheduling.

    Only the cell-width budget uses this convention. Particle coordinates,
    interpolation, potential differences and hardware intersections are never
    snapped. The caller still limits motion with the neighbouring cell width.
    """
    index = min(max(np.searchsorted(axis, value, side="right")-1, 0), len(axis)-2)
    if index > 0 and abs(value-axis[index]) <= 8.*abs(np.spacing(axis[index])):
        return index-1
    return index
