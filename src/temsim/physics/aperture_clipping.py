"""Segment-local, forward-only aperture clipping.

The explicit seven-argument-compatible signature supports both first-segment use:
    clip_segment(state, z, x, y)
and continuation use:
    clip_segment(state, z, x, y, alive, blocked_z, blocked_key)
"""
import numpy as np


def posed_aperture_registration(state, aperture):
    from temsim.lens_pose import lens_pose_registration
    registration = lens_pose_registration(state, aperture.key)
    if (any(value != 0. for value in registration.origin_global_m)
            or registration.rotation_local_to_global != ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))):
        return registration
    return None


def _transmits(aperture, x_mm, y_mm):
    if float(getattr(aperture, "radius_mm", 1.)) <= 0.:
        return np.zeros(np.shape(x_mm), dtype=bool)
    if hasattr(aperture, "transmission_mask"):
        return np.asarray(aperture.transmission_mask(x_mm, y_mm), dtype=bool)
    return np.hypot(x_mm-float(aperture.offset_x_mm),
                    y_mm-float(aperture.offset_y_mm)) <= float(aperture.radius_mm)


def clip_posed_apertures(state, apertures, z, x, y, alive, blocked_z, blocked_key):
    """Intersect tilted aperture planes, testing their local openings exactly."""
    for aperture in apertures:
        if not aperture.enabled or not bool(getattr(aperture, "installed", True)):
            continue
        registration = posed_aperture_registration(state, aperture)
        if registration is None:
            continue
        rotation, offset = registration.rotation_array, registration.origin_array_m*1e3
        for index in range(len(z)-1):
            start = np.column_stack((x[index]*1e3, y[index]*1e3, np.full(len(alive), z[index])))
            end = np.column_stack((x[index+1]*1e3, y[index+1]*1e3, np.full(len(alive), z[index+1])))
            local_start, local_end = (start-offset) @ rotation, (end-offset) @ rotation
            delta = local_end-local_start
            moving = np.abs(delta[:, 2]) > 1e-14
            fraction = np.divide(float(aperture.z_mm)-local_start[:, 2], delta[:, 2],
                                 out=np.zeros(len(alive)), where=moving)
            crossing = moving & (fraction >= 0.) & (fraction <= 1.)
            on_plane = ~moving & (np.abs(local_start[:, 2]-float(aperture.z_mm)) <= 1e-12)
            crossing |= on_plane
            crossing &= np.all(np.isfinite(local_start), axis=1) & np.all(np.isfinite(local_end), axis=1)
            local = local_start + fraction[:, None]*delta
            hit_z = z[index] + fraction*(z[index+1]-z[index])
            reaches = alive | (np.isfinite(blocked_z) & (blocked_z > hit_z+1e-9))
            hit = crossing & reaches & ~_transmits(aperture, local[:, 0], local[:, 1])
            alive[hit] = False
            blocked_z[hit] = hit_z[hit]
            for ray_index in np.flatnonzero(hit):
                blocked_key[int(ray_index)] = aperture.key
    return alive, blocked_z, blocked_key


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

    apertures = list(state.apertures)
    nanopulser = getattr(state, "nanopulser", None)
    if nanopulser is not None and bool(nanopulser.installed):
        # The stop remains physically inserted in the open state. Switching
        # the field off restores its on-axis transmission, not its removal.
        nanopulser.validate()
        apertures.append(nanopulser.aperture)
    active_apertures = [
        aperture
        for aperture in apertures
        if aperture.enabled
        and bool(getattr(aperture, "installed", True))
        and posed_aperture_registration(state, aperture) is None
        and z_min <= float(aperture.z_mm) <= z_max
    ]

    for aperture in sorted(active_apertures, key=lambda item: item.z_mm):
        index = int(np.argmin(np.abs(z - float(aperture.z_mm))))
        x_mm = x[index] * 1.0e3
        y_mm = y[index] * 1.0e3
        if float(getattr(aperture, "radius_mm", 1.0)) <= 0.0:
            passes = np.zeros(x_mm.shape, dtype=bool)
        elif hasattr(aperture, "transmission_mask"):
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

    return clip_posed_apertures(state, apertures, z, x, y, alive, blocked_z, blocked_key)
