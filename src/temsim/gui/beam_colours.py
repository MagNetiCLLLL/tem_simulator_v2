"""Shared display colours for immutable physical tip emission quantities."""
from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6.QtGui import QColor


EMISSION_COLOUR_MODES = frozenset({"source", "emission_direction", "emission_angle"})


def emission_colour(mode: str, value: float, ray_id: int = 0) -> QColor:
    """Match source, selected-plane and ray colours; angles are radians."""
    if mode not in EMISSION_COLOUR_MODES:
        raise ValueError(f"Unknown emission colour quantity: {mode}")
    if ray_id < 0 or not math.isfinite(value):
        return QColor("#94a3b8")
    if mode == "emission_angle":
        return pg.colormap.get("viridis").mapToQColor(float(np.clip(value / (math.pi / 2), 0., 1.)))
    return QColor.fromHsvF(float(value % (2 * math.pi)) / (2 * math.pi), .88, 1.)
