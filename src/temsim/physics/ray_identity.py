"""Display-only source lineage for weighted electron trajectories.

Colours label the azimuth of an emitted position, not its velocity. A loss
channel may share an ancestor with other weighted representatives. Neither
identity nor azimuth participates in propagation or detector integration.
"""

from __future__ import annotations

import numpy as np


def _frozen(values, dtype):
    array = np.ascontiguousarray(values, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=dtype).reshape(array.shape)


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
    if x.ndim == 2 and y.shape == x.shape and x.shape[0]:
        finite = np.isfinite(x[0]) & np.isfinite(y[0])
        if np.any(finite):
            dx = x[0] - np.mean(x[0, finite])
            dy = y[0] - np.mean(y[0, finite])
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
