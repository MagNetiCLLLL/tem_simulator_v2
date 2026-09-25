"""Bounded-palette scalar colours for already executed ray polylines.

This is display processing only. Coordinates and scalar values are aligned
one-dimensional arrays, with non-finite coordinates separating paths. Each
input segment is split at colour-bin boundaries before geometric screen
simplification; a geometrically straight ray can therefore retain its colour
gradient. The output has at most one disconnected polyline per palette bin.
"""

from __future__ import annotations

from functools import lru_cache
import operator

import numpy as np
import pyqtgraph as pg


SCALAR_COLOUR_BINS = 128
UNKNOWN_SCALAR_BIN = -1
UNKNOWN_SCALAR_RGB = (148, 163, 184)


def _parameters(maximum: float, bins: int) -> tuple[float, int]:
    if isinstance(bins, (bool, np.bool_)):
        raise ValueError("Scalar palette size must be an integer from 2 to 256")
    try:
        bins = operator.index(bins)
    except TypeError as exc:
        raise ValueError("Scalar palette size must be an integer from 2 to 256") from exc
    if not 2 <= bins <= 256:
        raise ValueError("Scalar palette size must be an integer from 2 to 256")
    maximum = float(maximum)
    if not np.isfinite(maximum) or maximum < 0.0:
        raise ValueError("Scalar colour maximum must be finite and non-negative")
    return maximum, bins


def scalar_colour_indices(values, maximum: float, *, bins: int = SCALAR_COLOUR_BINS) -> np.ndarray:
    """Map non-negative scalars to a shared, saturated linear colour scale.

    Unknown, negative and non-finite values map to ``UNKNOWN_SCALAR_BIN``.
    A zero maximum maps every known value to the first bin without division.
    The normalization maximum must describe the complete chosen population,
    never just the rays selected for display.
    """
    maximum, bins = _parameters(maximum, bins)
    values = np.asarray(values, dtype=np.float64)
    result = np.full(values.shape, UNKNOWN_SCALAR_BIN, dtype=np.int16)
    known = np.isfinite(values) & (values >= 0.0)
    if maximum == 0.0:
        result[known] = 0
    else:
        with np.errstate(over="ignore"):
            normalized = np.clip(values[known] / maximum, 0.0, 1.0)
        result[known] = np.minimum((normalized * bins).astype(np.int16), bins - 1)
    return result


@lru_cache(maxsize=8)
def scalar_colour_palette(*, bins: int = SCALAR_COLOUR_BINS) -> tuple[tuple[int, int, int], ...]:
    """The same viridis RGB palette for ray segments and transverse points."""
    _, bins = _parameters(0.0, bins)
    lookup = pg.colormap.get("viridis").getLookupTable(nPts=bins, alpha=False)
    return tuple(tuple(map(int, row)) for row in lookup)


def scalar_rgb(index: int, *, bins: int = SCALAR_COLOUR_BINS) -> tuple[int, int, int]:
    """Read one palette entry, or the neutral colour for an unknown scalar."""
    palette = scalar_colour_palette(bins=bins)
    if index == UNKNOWN_SCALAR_BIN:
        return UNKNOWN_SCALAR_RGB
    if not 0 <= index < len(palette):
        raise ValueError("Scalar colour index is outside the palette")
    return palette[index]


def scalar_colour_groups(x, y, values, maximum: float, *, bins: int = SCALAR_COLOUR_BINS):
    """Return ``{bin: (x, y)}`` disconnected polylines for ``RayCurveItem``.

    Splitting uses linear interpolation along each supplied segment, including
    endpoints already clipped to a physical stop. It neither extrapolates nor
    derives the scalar from displayed distance. A segment with either scalar
    endpoint missing/negative is grey; coordinate gaps are never connected.
    Consecutive segments of one colour remain one finite run, allowing RDP
    simplification. Grouping uses one stable integer sort, not one full-history
    scan per colour. Inputs are never modified.
    """
    maximum, bins = _parameters(maximum, bins)
    x, y, values = (np.asarray(a, dtype=np.float64) for a in (x, y, values))
    if x.ndim != 1 or x.shape != y.shape or x.shape != values.shape:
        raise ValueError("Ray coordinates and scalars must be aligned one-dimensional arrays")
    finite = np.isfinite(x) & np.isfinite(y)
    edges = np.flatnonzero(finite[:-1] & finite[1:])
    if not edges.size:
        return {}
    first, last = values[edges], values[edges + 1]
    known = np.isfinite(first) & np.isfinite(last) & (first >= 0.0) & (last >= 0.0)
    low_boundary = np.ones(edges.size, dtype=np.int16)
    high_boundary = np.zeros(edges.size, dtype=np.int16)
    if maximum > 0.0 and np.any(known):
        with np.errstate(over="ignore"):
            lo = np.clip(np.minimum(first[known], last[known]) / maximum, 0.0, 1.0)
            hi = np.clip(np.maximum(first[known], last[known]) / maximum, 0.0, 1.0)
        # Strictly internal boundaries only: no zero-length endpoint pieces.
        low_boundary[known] = np.maximum(1, np.floor(lo * bins).astype(np.int16) + 1)
        high_boundary[known] = np.minimum(bins - 1, np.ceil(hi * bins).astype(np.int16) - 1)
    crossings = np.maximum(0, high_boundary - low_boundary + 1)
    pieces = crossings.astype(np.int64) + 1
    source = np.repeat(np.arange(edges.size), pieces)
    offsets = np.arange(source.size) - np.repeat(np.cumsum(pieces) - pieces, pieces)
    expanded_edges = edges[source]
    start_fraction = np.zeros(source.size)
    end_fraction = np.ones(source.size)
    ascending = last[source] >= first[source]

    def crossing_fractions(mask, ordinal):
        selected = source[mask]
        boundary = np.where(ascending[mask], low_boundary[selected] + ordinal,
                            high_boundary[selected] - ordinal)
        # Scale first to avoid overflowing a difference between large scalars.
        scale = np.maximum(first[selected], last[selected])
        a, b = first[selected] / scale, last[selected] / scale
        return ((boundary / bins) * (maximum / scale) - a) / (b - a)

    interior_start = offsets > 0
    if np.any(interior_start):
        start_fraction[interior_start] = crossing_fractions(interior_start, offsets[interior_start] - 1)
    interior_end = offsets < crossings[source]
    if np.any(interior_end):
        end_fraction[interior_end] = crossing_fractions(interior_end, offsets[interior_end])
    middle = (start_fraction + end_fraction) * 0.5
    scalar = np.full(source.size, np.nan)
    valid = known[source]
    scalar[valid] = first[source[valid]] * (1.0 - middle[valid]) + last[source[valid]] * middle[valid]
    colours = scalar_colour_indices(scalar, maximum, bins=bins)

    def interpolate(coordinates, fraction):
        return coordinates[expanded_edges] * (1.0 - fraction) + coordinates[expanded_edges + 1] * fraction

    start_x, start_y = interpolate(x, start_fraction), interpolate(y, start_fraction)
    end_x, end_y = interpolate(x, end_fraction), interpolate(y, end_fraction)
    new_run = np.r_[True, (colours[1:] != colours[:-1]) | (np.diff(expanded_edges) > 1)]
    starts = np.flatnonzero(new_run)
    ends = np.r_[starts[1:] - 1, source.size - 1]
    end_positions = np.arange(source.size) + 2 * np.cumsum(new_run) - 1
    output_size = source.size + 2 * len(starts)
    output_x, output_y = np.full(output_size, np.nan), np.full(output_size, np.nan)
    for output, a, b in ((output_x, start_x, end_x), (output_y, start_y, end_y)):
        output[end_positions] = b
        output[end_positions[starts] - 1] = a[starts]
    run_sizes = ends - starts + 3  # Initial vertex, terminal vertices, separator.
    vertex_colours = np.repeat(colours[starts], run_sizes)
    order = np.argsort(vertex_colours, kind="stable")
    output_x, output_y, vertex_colours = output_x[order], output_y[order], vertex_colours[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(vertex_colours)) + 1, output_size]
    return {int(vertex_colours[a]): (output_x[a:b], output_y[a:b])
            for a, b in zip(boundaries[:-1], boundaries[1:])}
