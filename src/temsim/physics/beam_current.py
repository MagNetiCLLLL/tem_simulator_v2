"""Shared conversion from dimensionless ray weights to physical current."""

from __future__ import annotations

import math


DEFAULT_COLUMN_CURRENT_LIMIT_PERCENT = 100.0


def column_current_limit_percent(state) -> float:
    """Return the ideal Spot-size current ceiling as an emitted-current %."""

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
