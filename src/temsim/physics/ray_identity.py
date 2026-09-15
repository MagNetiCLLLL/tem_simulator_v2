"""Display-only source lineage for weighted electron trajectories.

Source-position colours label emitted-position azimuth, not velocity. Separate
emission-angle modes use the saved original direction and local normal. A loss
channel may share an ancestor with other weighted representatives. Neither
identity nor azimuth participates in propagation or detector integration.
"""

from __future__ import annotations

import numpy as np


def _frozen(values, dtype):
    array = np.ascontiguousarray(values, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=dtype).reshape(array.shape)


def emission_reference(bundle, surface_model=None):
    """Freeze actual pre-field launch data; SI positions and unit directions.

    The spherical tip centre is (0, 0, -R). Surface normals are not velocities.
    Historical planar particle sources retain their own launch distribution.
    This is display lineage, not an independently configurable source.
    """
    if surface_model is not None:
        positions = np.asarray(bundle.surface_position_m)
        directions = np.asarray(bundle.surface_direction)
        radius = surface_model.geometry.apex_radius_nm * 1e-9
        normals = (positions + [0., 0., radius])/radius
    else:
        positions = np.column_stack((bundle.x_m, bundle.y_m, np.zeros_like(bundle.x_m)))
        directions = np.column_stack((bundle.tx_rad, bundle.ty_rad, np.ones_like(bundle.x_m)))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        normals = np.tile([0., 0., 1.], (len(positions), 1))
    return {"ray_id": _frozen(bundle.ray_id, np.int64),
            "position_m": _frozen(positions, np.float64),
            "direction": _frozen(directions, np.float64),
            "normal": _frozen(normals, np.float64)}


def emission_colour_values(simulation, source_ids, mode):
    """Original direction azimuth / angle to local normal, mapped by ancestry.

    Angles are radians. Missing launch records stay NaN (neutral colour), even
    when a later cached slope is available. Backward launches retain their full
    direction; slopes cannot determine their hemisphere.
    """
    ids = np.asarray(source_ids)
    output = np.full(ids.shape, np.nan)
    reference = getattr(getattr(simulation, "gun_trace", None), "emission_reference", None)
    if reference is None:
        return output
    try:
        original_ids = np.asarray(reference["ray_id"])
        direction = np.asarray(reference["direction"], dtype=float)
        normal = np.asarray(reference["normal"], dtype=float)
        if (original_ids.ndim != 1 or original_ids.dtype.kind not in "iu"
                or np.any(original_ids < 0) or np.unique(original_ids).size != original_ids.size
                or direction.shape != (original_ids.size, 3) or normal.shape != direction.shape):
            return output
        if mode == "emission_direction":
            values = np.mod(np.arctan2(direction[:, 1], direction[:, 0]), 2*np.pi)
            values[np.hypot(direction[:, 0], direction[:, 1]) <= 1e-15] = np.nan
        elif mode == "emission_angle":
            # atan2 is well-conditioned for nearly normal emission.
            values = np.arctan2(np.linalg.norm(np.cross(direction, normal), axis=1),
                                np.einsum("ij,ij->i", direction, normal))
        else:
            raise ValueError("Unknown emission colour quantity")
        values[~np.all(np.isfinite(direction) & np.isfinite(normal), axis=1)] = np.nan
        order = np.argsort(original_ids)
        indices = np.searchsorted(original_ids[order], ids)
        if original_ids.size:
            valid = (ids >= 0) & (indices < original_ids.size)
            rows = np.minimum(indices, original_ids.size-1)
            valid &= original_ids[order[rows]] == ids
            output[valid] = values[order[rows[valid]]]
    except (KeyError, TypeError, IndexError):
        return output
    return output


def _count(branch) -> int:
    if branch is None:
        return 0
    x = np.asarray(getattr(branch, "x", ()))
    if x.ndim == 2:
        return x.shape[1]
    return np.asarray(getattr(branch, "alive", ())).size


def _unknown(count):
    return (_frozen(np.full(count, -1), np.int64),
            _frozen(np.full(count, np.nan), np.float64))


def _stored(branch, count):
    ids = getattr(branch, "source_ray_id", None)
    angles = getattr(branch, "source_azimuth_rad", None)
    if ids is None or angles is None:
        return None
    try:
        raw_ids = np.asarray(ids)
        angles = np.asarray(angles, dtype=float)
        if (raw_ids.shape != (count,) or angles.shape != (count,)
                or raw_ids.dtype.kind not in "iu" or np.any(raw_ids < -1)
                or np.any(raw_ids > np.iinfo(np.int64).max)
                or np.any(np.isinf(angles))):
            return None
        return _frozen(raw_ids, np.int64), _frozen(angles, np.float64)
    except (ValueError, TypeError, OverflowError):
        return None


def source_identity(incident, gun_trace=None):
    """Return immutable IDs and fixed source-position azimuths (radians).

    Legacy incident caches can recover this metadata from their original gun
    histories. Clipping, selected Z, and the current surviving centroid are
    deliberately excluded from the reference definition.
    """
    count = _count(incident)
    existing = _stored(incident, count)
    if existing is not None:
        known_ids = existing[0][existing[0] >= 0]
        # Repeated ancestors are valid downstream, but never identify two
        # different source samples. Recover corrupt legacy source metadata.
        if np.unique(known_ids).size == known_ids.size:
            return existing
    ids = np.arange(count, dtype=np.int64)
    gun_ids = getattr(getattr(gun_trace, "exit_bundle", None), "ray_id", None)
    if gun_ids is not None:
        values = np.asarray(gun_ids)
        if (values.shape == (count,) and values.dtype.kind in "iu"
                and np.all(values >= 0) and np.all(values <= np.iinfo(np.int64).max)
                and np.unique(values).size == count):
            ids = values
    angles = np.full(count, np.nan)
    x = np.asarray(getattr(incident, "x", ()), dtype=float)
    y = np.asarray(getattr(incident, "y", ()), dtype=float)
    reference = getattr(gun_trace, "emission_reference", None)
    launch_positions = False
    if reference is not None:
        # Preserve the real launch position, not the first common-Z resample
        # (a curved emitter has no single launch plane).
        positions = np.asarray(reference.get("position_m", ()), dtype=float)
        if positions.shape == (count, 3) and np.array_equal(reference.get("ray_id"), ids):
            x, y = positions[None, :, 0], positions[None, :, 1]
            launch_positions = True
    if x.ndim == 2 and y.shape == x.shape and x.shape[0]:
        finite = np.isfinite(x[0]) & np.isfinite(y[0])
        if np.any(finite):
            dx = x[0] - (0. if launch_positions else np.mean(x[0, finite]))
            dy = y[0] - (0. if launch_positions else np.mean(y[0, finite]))
            defined = finite & (np.hypot(dx, dy) > 1.0e-15)
            angles[defined] = np.mod(np.arctan2(dy[defined], dx[defined]), 2*np.pi)
    return _frozen(ids, np.int64), _frozen(angles, np.float64)


def select_identity(ids, angles, indices):
    """Select parent metadata by incident column, never by a compact row ID."""
    values = np.asarray(indices)
    count = values.size
    selected_ids = np.full(count, -1, dtype=np.int64)
    selected_angles = np.full(count, np.nan)
    ids, angles = np.asarray(ids), np.asarray(angles)
    if (values.ndim == 1 and values.dtype.kind in "iu" and ids.ndim == 1
            and angles.shape == ids.shape):
        valid = (values >= 0) & (values < ids.size)
        selected_ids[valid] = ids[values[valid]]
        selected_angles[valid] = angles[values[valid]]
    return _frozen(selected_ids, np.int64), _frozen(selected_angles, np.float64)


def branch_identity(branch, simulation=None):
    """Resolve lineage without guessing the identity of compact legacy rays.

    Only legacy ordinary simulation branches with full-width, matching boundary
    positions may inherit column identity. Independently sampled/scattered
    histories without explicit lineage stay neutral/unknown.
    """
    count = _count(branch)
    incident = getattr(simulation, "incident", None)
    if branch is incident or (incident is None and getattr(branch, "name", "") == "incident"):
        return source_identity(branch, getattr(simulation, "gun_trace", None))
    existing = _stored(branch, count)
    if existing is not None:
        return existing
    if incident is None:
        return _unknown(count)
    ids, angles = source_identity(incident, getattr(simulation, "gun_trace", None))
    source_indices = getattr(branch, "source_ray_index", None)
    if source_indices is not None and np.asarray(source_indices).shape == (count,):
        return select_identity(ids, angles, source_indices)
    try:
        ordinary_branch = any(branch is item for item in getattr(simulation, "branches", {}).values())
        if (ordinary_branch and count == ids.size
                and np.array_equal(np.asarray(branch.x)[0], np.asarray(incident.x)[-1])
                and np.array_equal(np.asarray(branch.y)[0], np.asarray(incident.y)[-1])):
            return ids, angles
    except (AttributeError, IndexError, TypeError):
        pass
    return _unknown(count)
