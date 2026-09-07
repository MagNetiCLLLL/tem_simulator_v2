"""Finite lateral sample-envelope geometry shared by all specimen models."""

from __future__ import annotations

import math

import numpy as np


SAMPLE_ENVELOPE_SHAPES = ("disk", "rectangle")


def canonical_sample_envelope_shape(value) -> str:
    """Return one supported shape key or raise on ambiguous geometry."""

    shape = str(value).strip().lower()
    if shape not in SAMPLE_ENVELOPE_SHAPES:
        raise ValueError(
            "Sample envelope shape must be 'disk' or 'rectangle'."
        )
    return shape


def sample_envelope_shape(sample) -> str:
    return canonical_sample_envelope_shape(
        getattr(sample, "envelope_shape", "rectangle")
    )


def envelope_contains_xy(
    shape: str,
    x_nm,
    y_nm,
    *,
    centre_xy_nm: tuple[float, float],
    size_xy_nm: tuple[float, float],
):
    """Return a scalar/array mask for the finite lateral envelope.

    ``disk`` is represented by equal X/Y diameters in normal use.  The
    normalized ellipse expression keeps partially edited or legacy states
    well-defined until the GUI synchronizes the two diameters.
    """

    kind = canonical_sample_envelope_shape(shape)
    centre_x, centre_y = (float(value) for value in centre_xy_nm)
    size_x, size_y = (float(value) for value in size_xy_nm)
    if not all(
        math.isfinite(value) and value > 0.0 for value in (size_x, size_y)
    ):
        raise ValueError("Sample X/Y envelope size must be finite and positive.")
    x = np.asarray(x_nm, dtype=float)
    y = np.asarray(y_nm, dtype=float)
    dx = x - centre_x
    dy = y - centre_y
    if kind == "disk":
        result = (dx / (0.5 * size_x)) ** 2 + (
            dy / (0.5 * size_y)
        ) ** 2 <= 1.0
    else:
        result = (np.abs(dx) <= 0.5 * size_x) & (
            np.abs(dy) <= 0.5 * size_y
        )
    if result.ndim == 0:
        return bool(result)
    return result


def sample_envelope_contains_xy(sample, x_nm, y_nm):
    return envelope_contains_xy(
        sample_envelope_shape(sample),
        x_nm,
        y_nm,
        centre_xy_nm=(
            float(getattr(sample, "centre_x_nm", 0.0)),
            float(getattr(sample, "centre_y_nm", 0.0)),
        ),
        size_xy_nm=(
            float(getattr(sample, "size_x_nm", 0.0)),
            float(getattr(sample, "size_y_nm", 0.0)),
        ),
    )


def envelope_intersects_bounds(
    shape: str,
    bounds_nm: tuple[float, float, float, float],
    *,
    centre_xy_nm: tuple[float, float],
    size_xy_nm: tuple[float, float],
) -> bool:
    """Return whether an axis-aligned rectangle intersects the envelope."""

    kind = canonical_sample_envelope_shape(shape)
    x0, x1, y0, y1 = (float(value) for value in bounds_nm)
    centre_x, centre_y = (float(value) for value in centre_xy_nm)
    size_x, size_y = (float(value) for value in size_xy_nm)
    if x1 < x0 or y1 < y0:
        raise ValueError("Sample ROI bounds must be ordered.")
    if kind == "rectangle":
        return not (
            x1 < centre_x - 0.5 * size_x
            or x0 > centre_x + 0.5 * size_x
            or y1 < centre_y - 0.5 * size_y
            or y0 > centre_y + 0.5 * size_y
        )
    nearest_x = min(max(centre_x, x0), x1)
    nearest_y = min(max(centre_y, y0), y1)
    return bool(
        envelope_contains_xy(
            kind,
            nearest_x,
            nearest_y,
            centre_xy_nm=(centre_x, centre_y),
            size_xy_nm=(size_x, size_y),
        )
    )


def envelope_contains_bounds(
    shape: str,
    bounds_nm: tuple[float, float, float, float],
    *,
    centre_xy_nm: tuple[float, float],
    size_xy_nm: tuple[float, float],
) -> bool:
    """Return whether all of an axis-aligned rectangle is inside the envelope."""

    x0, x1, y0, y1 = (float(value) for value in bounds_nm)
    corners_x = np.asarray((x0, x1, x1, x0), dtype=float)
    corners_y = np.asarray((y0, y0, y1, y1), dtype=float)
    return bool(
        np.all(
            envelope_contains_xy(
                shape,
                corners_x,
                corners_y,
                centre_xy_nm=centre_xy_nm,
                size_xy_nm=size_xy_nm,
            )
        )
    )
