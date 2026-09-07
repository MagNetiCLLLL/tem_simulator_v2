"""Shared headroom policy for calibrated magnetic-lens operating points."""

from __future__ import annotations

import math


DEFAULT_OPERATING_MIN_PERCENT = 30.0
DEFAULT_OPERATING_MAX_PERCENT = 70.0
DEFAULT_OPERATING_TARGET_PERCENT = 60.0
SATURATION_PERCENT = 100.0


def is_saturated_excitation(
    percent: float,
    maximum_percent: float = SATURATION_PERCENT,
) -> bool:
    """Return whether an operating point has reached its configured limit."""

    return float(percent) >= float(maximum_percent) - 1.0e-9


def rebase_peak_field(
    maximum_peak_field_t: float,
    solved_percent: float,
    target_percent: float = DEFAULT_OPERATING_TARGET_PERCENT,
) -> tuple[float, float]:
    """Move an operating point into the headroom window without changing Bz."""

    maximum_peak_field_t = float(maximum_peak_field_t)
    solved_percent = float(solved_percent)
    target_percent = float(target_percent)
    if not math.isfinite(maximum_peak_field_t) or maximum_peak_field_t < 0.0:
        raise ValueError("Maximum peak field must be finite and non-negative.")
    if not math.isfinite(solved_percent) or solved_percent < 0.0:
        raise ValueError("Solved excitation must be finite and non-negative.")
    if not (
        DEFAULT_OPERATING_MIN_PERCENT
        <= target_percent
        <= DEFAULT_OPERATING_MAX_PERCENT
    ):
        raise ValueError(
            "Rebased excitation must lie in the 30-70% operating window."
        )
    return (
        maximum_peak_field_t * solved_percent / target_percent,
        target_percent,
    )
