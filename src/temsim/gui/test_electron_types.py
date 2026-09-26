"""Detached display records for independently calculated virtual electrons."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtGui import QColor


@dataclass(frozen=True, eq=False)
class ElectronPath:
    key: str
    label: str
    colour: str
    positions_m: np.ndarray
    time_s: np.ndarray | None = None
    selected: bool = False
    state: str = "current"

    def __post_init__(self):
        if self.state not in {"current", "previous", "in_progress"}:
            raise ValueError("Electron path state must be current, previous or in_progress")
        for name in ("key", "label"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Electron path {name} must be a non-empty string")
        if not isinstance(self.colour, str):
            raise ValueError("Electron path colour must be a visible colour string")
        colour = QColor(self.colour)
        if not colour.isValid() or colour.alpha() == 0:
            raise ValueError("Electron path colour must be a visible colour string")
        if not isinstance(self.selected, (bool, np.bool_)):
            raise ValueError("Electron path selected must be Boolean")
        positions = np.array(self.positions_m, dtype=np.float64, order="C", copy=True)
        if positions.shape == (0,):
            positions = positions.reshape(0, 3)
        if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
            raise ValueError("positions_m must contain finite XYZ positions with shape (N, 3)")
        times = None if self.time_s is None else np.array(self.time_s, dtype=np.float64, copy=True)
        if times is not None:
            if (times.shape != (len(positions),) or not np.isfinite(times).all()
                    or np.any(times < 0.) or np.any(np.diff(times) < 0.)):
                raise ValueError("time_s must contain finite non-negative times in chronological order with shape (N,)")
            times.setflags(write=False)
        positions.setflags(write=False)
        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "time_s", times)
        object.__setattr__(self, "selected", bool(self.selected))


__all__ = ["ElectronPath"]
