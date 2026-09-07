"""Hard-edge clipping against the position-dependent TEM vacuum bore."""

from __future__ import annotations

import math

import numpy as np


COLUMN_WALL_KEY = "column_wall"


def _first_wall_intersection(z_mm, x_mm, y_mm, radius_mm):
    """Return the first piecewise-linear contact with a circular bore."""

    radial_squared = x_mm * x_mm + y_mm * y_mm
    radius_squared = radius_mm * radius_mm
    contact = radial_squared >= radius_squared
    if not np.any(contact):
        return None

    upper = int(np.argmax(contact))
    if upper == 0:
        return float(z_mm[0])

    lower = upper - 1
    p0 = np.array((x_mm[lower], y_mm[lower]), dtype=float)
    delta = np.array(
        (x_mm[upper] - x_mm[lower], y_mm[upper] - y_mm[lower]),
        dtype=float,
    )
    a = float(np.dot(delta, delta))
    b = 2.0 * float(np.dot(p0, delta))
    c = float(np.dot(p0, p0)) - radius_squared
    if a <= np.finfo(float).eps:
        fraction = 1.0
    else:
        discriminant = max(b * b - 4.0 * a * c, 0.0)
        root = math.sqrt(discriminant)
        candidates = (
            (-b - root) / (2.0 * a),
            (-b + root) / (2.0 * a),
        )
        valid = [value for value in candidates if 0.0 <= value <= 1.0]
        fraction = min(valid) if valid else 1.0
    return float(
        z_mm[lower] + fraction * (z_mm[upper] - z_mm[lower])
    )


def _vacuum_segments(state, z):
    assembly = getattr(state, "_resolved_assembly", None)
    segments = getattr(assembly, "vacuum_bore_segments", ())
    if segments:
        return tuple(segments)
    # Compatibility for standalone physics callers. Application calculations
    # always install the TOML-resolved profile above.
    diameter = float(getattr(state, "column_inner_diameter_mm", 20.0))
    if not math.isfinite(diameter) or diameter <= 0.0:
        raise ValueError("Vacuum inner diameter must be finite and positive")
    return (
        type("VacuumSegment", (), {
            "start_z_mm": float(z[0]),
            "end_z_mm": float(z[-1]),
            "inner_diameter_mm": diameter,
        })(),
    )


def _expanded_profile_axis(z, segments):
    start = float(z[0])
    end = float(z[-1])
    boundaries = [
        value
        for segment in segments
        for value in (float(segment.start_z_mm), float(segment.end_z_mm))
        if start < value < end
    ]
    axis = np.unique(np.concatenate((z, np.asarray(boundaries, dtype=float))))
    midpoints = 0.5 * (axis[:-1] + axis[1:])
    # A missing wall segment means that no mechanical wall is present there.
    # Ray propagation and magnetic-field evaluation are deliberately independent
    # of the mechanical envelope; this routine only contributes stop candidates
    # where a TOML-owned vacuum wall actually exists.
    interval_radius = np.full(midpoints.shape, np.inf, dtype=float)
    for segment in segments:
        radius = 0.5 * float(segment.inner_diameter_mm)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("Vacuum inner diameter must be finite and positive")
        active = (
            (midpoints >= float(segment.start_z_mm) - 1.0e-12)
            & (midpoints <= float(segment.end_z_mm) + 1.0e-12)
        )
        interval_radius[active] = radius
    node_radius = np.empty(axis.size, dtype=float)
    node_radius[0] = interval_radius[0]
    node_radius[-1] = interval_radius[-1]
    if axis.size > 2:
        # A diameter transition has a radial shoulder. Its passable radius is
        # the narrower of the two connected vacuum cylinders.
        node_radius[1:-1] = np.minimum(
            interval_radius[:-1], interval_radius[1:]
        )
    return axis, interval_radius, node_radius


def _first_profile_intersection(axis, x_mm, y_mm, interval_radius, node_radius):
    contact = x_mm * x_mm + y_mm * y_mm >= node_radius * node_radius
    if not np.any(contact):
        return None
    upper = int(np.argmax(contact))
    if upper == 0:
        return float(axis[0])
    previous_radius = float(interval_radius[upper - 1])
    if node_radius[upper] < previous_radius - 1.0e-12:
        return float(axis[upper])
    return _first_wall_intersection(
        axis[upper - 1:upper + 1],
        x_mm[upper - 1:upper + 1],
        y_mm[upper - 1:upper + 1],
        previous_radius,
    )


def clip_column_wall(
    state,
    z,
    x,
    y,
    alive=None,
    blocked_z=None,
    blocked_key=None,
):
    """Stop every live ray at its first contact with the TOML vacuum bore.

    The solver stores transverse coordinates in metres and axial coordinates
    in millimetres.  Contacts are interpolated between saved trajectory
    samples, so the reported stop position does not jump with history step.
    """

    z = np.asarray(z, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if z.ndim != 1 or x.shape != y.shape or x.ndim != 2:
        raise ValueError("Column-wall trajectories must be Z by ray arrays")
    if x.shape[0] != z.size:
        raise ValueError("Column-wall Z and trajectory lengths must match")

    ray_count = x.shape[1]
    alive = (
        np.ones(ray_count, dtype=bool)
        if alive is None
        else np.asarray(alive, dtype=bool).copy()
    )
    blocked_z = (
        np.full(ray_count, np.nan)
        if blocked_z is None
        else np.asarray(blocked_z, dtype=float).copy()
    )
    blocked_key = (
        [""] * ray_count if blocked_key is None else list(blocked_key)
    )
    segments = _vacuum_segments(state, z)
    axis, interval_radius, node_radius = _expanded_profile_axis(z, segments)
    x_mm = x * 1.0e3
    y_mm = y * 1.0e3
    for ray in range(ray_count):
        expanded_x = np.interp(axis, z, x_mm[:, ray])
        expanded_y = np.interp(axis, z, y_mm[:, ray])
        hit_z = _first_profile_intersection(
            axis, expanded_x, expanded_y, interval_radius, node_radius,
        )
        if hit_z is None:
            continue
        existing_z = float(blocked_z[ray])
        if math.isfinite(existing_z) and existing_z <= hit_z + 1.0e-9:
            continue
        alive[ray] = False
        blocked_z[ray] = hit_z
        blocked_key[ray] = COLUMN_WALL_KEY
    return alive, blocked_z, blocked_key
