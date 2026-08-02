"""Segment-local, forward-only aperture clipping.

The explicit seven-argument-compatible signature supports both first-segment use:
    clip_segment(state, z, x, y)
and continuation use:
    clip_segment(state, z, x, y, alive, blocked_z, blocked_key)
"""
import numpy as np

def clip_segment(
    state,
    z,
    x,
    y,
    alive=None,
    blocked_z=None,
    blocked_key=None,
):
    z = np.asarray(z, dtype=float)
    x = np.asarray(x)
    y = np.asarray(y)

    if z.ndim != 1 or x.ndim != 2 or y.ndim != 2:
        raise ValueError("Aperture clipping expects z[steps], x[steps,rays], y[steps,rays]")
    if x.shape != y.shape or x.shape[0] != z.size:
        raise ValueError("Aperture clipping array shapes are inconsistent")

    ray_count = x.shape[1]
    alive = (
        np.ones(ray_count, dtype=bool)
        if alive is None
        else np.asarray(alive, dtype=bool).copy()
    )
    blocked_z = (
        np.full(ray_count, np.nan, dtype=float)
        if blocked_z is None
        else np.asarray(blocked_z, dtype=float).copy()
    )
    blocked_key = (
        [""] * ray_count
        if blocked_key is None
        else list(blocked_key)
    )

    if alive.size != ray_count or blocked_z.size != ray_count or len(blocked_key) != ray_count:
        raise ValueError("Aperture clipping state has the wrong ray count")

    z_min = float(min(z[0], z[-1]))
    z_max = float(max(z[0], z[-1]))

    active_apertures = [
        aperture
        for aperture in state.apertures
        if aperture.enabled
        and bool(getattr(aperture, "installed", True))
        and z_min <= float(aperture.z_mm) <= z_max
    ]

    for aperture in sorted(active_apertures, key=lambda item: item.z_mm):
        index = int(np.argmin(np.abs(z - float(aperture.z_mm))))
        x_mm = x[index] * 1.0e3
        y_mm = y[index] * 1.0e3
        if hasattr(aperture, "transmission_mask"):
            passes = np.asarray(
                aperture.transmission_mask(x_mm, y_mm),
                dtype=bool,
            )
        else:
            radius_mm = max(0.0, float(aperture.radius_mm))
            radial_distance_mm = np.hypot(
                x_mm - float(aperture.offset_x_mm),
                y_mm - float(aperture.offset_y_mm),
            )
            passes = radial_distance_mm <= radius_mm
        newly_blocked = alive & ~passes

        blocked_z[newly_blocked] = float(aperture.z_mm)
        for ray_index in np.flatnonzero(newly_blocked):
            blocked_key[int(ray_index)] = aperture.key

        # Once false, a ray remains false for every downstream aperture.
        alive &= passes

    return alive, blocked_z, blocked_key
