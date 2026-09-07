"""Shared conversion from dimensionless ray weights to physical current."""

from __future__ import annotations

import math

import numpy as np


DEFAULT_COLUMN_CURRENT_LIMIT_PERCENT = 100.0


def column_current_limit_percent(state) -> float:
    """Return the stored column-current ceiling as an emitted-current %."""

    value = float(
        getattr(
            state,
            "column_current_limit_percent",
            DEFAULT_COLUMN_CURRENT_LIMIT_PERCENT,
        )
    )
    if not math.isfinite(value) or not 0.0 <= value <= 100.0:
        raise ValueError(
            "Column current limit must be finite and between 0 and 100%."
        )
    return value


def effective_source_current_a(state) -> float:
    """Physical current represented by a complete normalized ray bundle."""

    emitted_current_a = max(
        float(state.electron_gun.emitted_current_a), 0.0
    )
    return emitted_current_a * column_current_limit_percent(state) / 100.0


def effective_source_current_pa(state) -> float:
    return effective_source_current_a(state) * 1.0e12


def sample_illumination_absent(simulation, state=None) -> bool:
    """Whether the traced specimen input contains no positive incident current.

    The optical result, rather than the requested blanker switch position,
    decides this: a zero or insufficient deflection voltage may still transmit.
    Incomplete diagnostic objects without a ray mask do not assert beam loss.
    """

    branch = getattr(simulation, "incident", None)
    raw_alive = getattr(branch, "alive", None)
    if raw_alive is None:
        return False
    alive = np.asarray(raw_alive, dtype=bool)
    if alive.ndim != 1:
        raise ValueError("Incident ray mask must be one-dimensional")
    if not np.any(alive):
        return True
    raw_weights = getattr(branch, "ray_weight", None)
    if raw_weights is not None:
        weights = np.asarray(raw_weights, dtype=float)
        if weights.shape != alive.shape:
            raise ValueError("Incident weights must match the ray mask")
        if np.any(~np.isfinite(weights[alive])) or np.any(weights[alive] < 0.0):
            raise ValueError("Incident weights must be finite and non-negative")
        if not np.any(weights[alive] > 0.0):
            return True
    if state is not None and hasattr(getattr(state, "electron_gun", None), "emitted_current_a"):
        return effective_source_current_a(state) <= 0.0
    return False
