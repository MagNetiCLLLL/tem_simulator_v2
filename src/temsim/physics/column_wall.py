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


def _partition_vacuum_segments(state, segments):
    """Reuse bore placement for immutable captured geometry across CPU steps."""
    from temsim.column.module_assembly import ResolvedAssembly
    assembly = getattr(state, "_resolved_assembly", None)
    can_cache = isinstance(assembly, ResolvedAssembly) and isinstance(segments, tuple)
    if can_cache:
        cached = getattr(state, "_lens_pose_bore_cache", None)
        if cached is not None and cached[0] is assembly and cached[1] is segments:
            return cached[2]
    result = _partition_vacuum_segments_uncached(state, segments)
    if can_cache:
        state._lens_pose_bore_cache = (assembly, segments, result)
    return result


def _partition_vacuum_segments_uncached(state, segments):
    """Keep every independent bore when a lens leaves the coaxial baseline.

    The normal resolver reduces overlaps to the narrowest nominal cylinder.
    After a pose change the omitted tube/wider bore can become the first stop,
    so recover those constraints from the captured assembly before partitioning.
    Unposed callers retain the original reduced-profile fast path.
    """
    from temsim.lens_pose import lens_pose_registration, physical_pose_values, supports_physical_lens_pose
    assembly = getattr(state, "_resolved_assembly", None)
    parts = getattr(assembly, "parts", ())
    pose_rows = {part.key: part.data for module in getattr(assembly, "modules", ())
                 for part in module.parts}
    pose_rows.update({part.key: part.data for part in parts})
    if not any(any(value != 0. for value in physical_pose_values(row).values())
               for row in pose_rows.values() if supports_physical_lens_pose(row)):
        return tuple(segments), ()
    stationary, placed, registrations = [], [], {}
    def registration_for(key):
        if key not in registrations:
            registrations[key] = lens_pose_registration(state, key)
        return registrations[key]
    def is_placed(key):
        registration = registration_for(key)
        return (any(value != 0. for value in registration.origin_global_m)
                or registration.rotation_local_to_global != ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)))

    from temsim.component_representation import non_material_role
    bore_parts = [part for part in parts
                  if "vacuum_inner_diameter_mm" in part.data
                  and not part.data.get("axial_vacuum_context_only", False)
                  and not non_material_role(part.data)]
    any_placed = (any(is_placed(getattr(segment, "key", "")) for segment in segments)
                  or any(is_placed(part.key) for part in bore_parts))
    if any_placed:
        segments = _independent_bore_constraints(assembly, segments, bore_parts)
    for segment in segments:
        key = getattr(segment, "key", "")
        registration = registration_for(key)
        if is_placed(key):
            placed.append((segment, registration))
        else:
            stationary.append(segment)
    return tuple(stationary), tuple(placed)


def _independent_bore_constraints(assembly, reduced_segments, bore_parts):
    """Recover unreduced part bores and explicit fixed tubes without live I/O."""
    from temsim.column.module_assembly import VacuumBoreSegment, _module_continuous_vacuum_tube
    from temsim.component_keys import ENERGY_FILTER_ENTRANCE_APERTURE
    parts = getattr(assembly, "parts", ())
    segments = list(reduced_segments)
    bounds = {}
    for module in getattr(assembly, "modules", ()):
        module_parts = [part for part in parts if part.module_key == module.key]
        if not module_parts or not hasattr(module, "geometry"):
            continue
        reference = module_parts[0]
        origin = float(reference.center_z_mm)-float(reference.data["local_center_z_mm"])
        lower, upper = origin+module.entrance_z_mm, origin+module.exit_z_mm
        filter_entrance = next((part for part in module_parts if part.key == ENERGY_FILTER_ENTRANCE_APERTURE), None)
        if filter_entrance is not None:
            upper = float(filter_entrance.center_z_mm)
        bounds[module.key] = lower, upper
        tube = _module_continuous_vacuum_tube(module, origin)
        if tube is not None:
            start, end, diameter, _outer = tube
            start, end = max(lower, start), min(upper, end)
            if end > start:
                segments.append(VacuumBoreSegment("@vacuum_tube:c1_c2_to_upper_objective",
                    "C1/C2 to upper objective continuous vacuum tube", start, end, diameter))
    for part in bore_parts:
        lower, upper = bounds.get(part.module_key, (-np.inf, np.inf))
        start, end = max(float(part.start_z_mm), lower), min(float(part.end_z_mm), upper)
        if end > start:
            segments.append(VacuumBoreSegment(part.key, getattr(part, "name", part.key),
                start, end, float(part.data["vacuum_inner_diameter_mm"])))
    unique = {}
    for segment in segments:
        identity = (getattr(segment, "key", ""), segment.start_z_mm,
                    segment.end_z_mm, segment.inner_diameter_mm)
        unique.setdefault(identity, segment)
    return tuple(unique.values())


def _posed_bore_stop_fractions(start_mm, end_mm, segment, registration):
    """First contact with a finite bore, measured along each straight segment.

    Transform into the unposed bore frame, intersect its axial slab, then solve
    the circular wall/entry shoulder exactly.  Fractions are unchanged by the
    rigid transform, including when the local axial direction reverses.
    """
    radius = .5 * float(segment.inner_diameter_mm)
    if not math.isfinite(radius) or radius <= 0.:
        raise ValueError("Vacuum inner diameter must be finite and positive")
    rotation = registration.rotation_array
    offset = registration.origin_array_m * 1e3
    start = (np.asarray(start_mm) - offset) @ rotation
    end = (np.asarray(end_mm) - offset) @ rotation
    delta = end - start
    dz = delta[:, 2]
    moving = np.abs(dz) > 1e-14
    low_z, high_z = float(segment.start_z_mm), float(segment.end_z_mm)
    first = np.divide(low_z-start[:, 2], dz, out=np.zeros(len(start)), where=moving)
    last = np.divide(high_z-start[:, 2], dz, out=np.ones(len(start)), where=moving)
    low = np.maximum(0., np.minimum(first, last))
    high = np.minimum(1., np.maximum(first, last))
    active = ((high >= low) & np.all(np.isfinite(start), axis=1)
              & np.all(np.isfinite(end), axis=1))
    active &= moving | ((start[:, 2] >= low_z) & (start[:, 2] <= high_z))
    p = start[:, :2] + low[:, None] * delta[:, :2]
    q = start[:, :2] + high[:, None] * delta[:, :2]
    initial_squared = np.einsum("ij,ij->i", p, p)
    final_squared = np.einsum("ij,ij->i", q, q)
    outside = active & (initial_squared >= radius*radius)
    stops = np.full(len(start), np.nan)
    stops[outside] = low[outside]
    crossing = active & ~outside & (final_squared >= radius*radius)
    if np.any(crossing):
        d = q-p
        aa = np.einsum("ij,ij->i", d, d)
        bb = 2.*np.einsum("ij,ij->i", p, d)
        cc = initial_squared-radius*radius
        disc = np.maximum(bb*bb-4.*aa*cc, 0.)
        fraction = np.divide(-bb+np.sqrt(disc), 2.*aa,
                             out=np.ones(len(start)), where=aa > 0.)
        fraction = np.clip(fraction, 0., 1.)
        stops[crossing] = (low + fraction*(high-low))[crossing]
    return stops


def _posed_wall_stop_z(z, x_mm, y_mm, placed):
    """Batched ray contacts; used by saved trajectories and medium transport."""
    stops = np.full(x_mm.shape[1], np.nan)
    for index in range(len(z)-1):
        starts = np.column_stack((x_mm[index], y_mm[index], np.full(x_mm.shape[1], z[index])))
        ends = np.column_stack((x_mm[index+1], y_mm[index+1], np.full(x_mm.shape[1], z[index+1])))
        for segment, registration in placed:
            fractions = _posed_bore_stop_fractions(starts, ends, segment, registration)
            hit = z[index] + fractions*(z[index+1]-z[index])
            update = np.isfinite(hit) & (~np.isfinite(stops) | (hit < stops))
            stops[update] = hit[update]
    return stops


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
        interval_radius[active] = np.minimum(interval_radius[active], radius)
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
    segments, placed = _partition_vacuum_segments(state, _vacuum_segments(state, z))
    axis, interval_radius, node_radius = _expanded_profile_axis(z, segments)
    x_mm = x * 1.0e3
    y_mm = y * 1.0e3
    placed_stops = _posed_wall_stop_z(z, x_mm, y_mm, placed) if placed else np.full(ray_count, np.nan)
    for ray in range(ray_count):
        expanded_x = np.interp(axis, z, x_mm[:, ray])
        expanded_y = np.interp(axis, z, y_mm[:, ray])
        hit_z = _first_profile_intersection(
            axis, expanded_x, expanded_y, interval_radius, node_radius,
        )
        if math.isfinite(placed_stops[ray]):
            hit_z = (float(placed_stops[ray]) if hit_z is None
                     else min(hit_z, float(placed_stops[ray])))
        if hit_z is None:
            continue
        existing_z = float(blocked_z[ray])
        if math.isfinite(existing_z) and existing_z <= hit_z + 1.0e-9:
            continue
        alive[ray] = False
        blocked_z[ray] = hit_z
        blocked_key[ray] = COLUMN_WALL_KEY
    return alive, blocked_z, blocked_key
