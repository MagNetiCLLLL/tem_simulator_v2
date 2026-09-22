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

    The tip apex is (0, 0, 0); its sphere centre is (0, 0, -R).
    Surface normals are not velocities. In the historical angle-only model they are prescribed emission
    axes; positions stay in the launch plane. Historical sources retain their
    own launch distribution.
    This is display lineage, not an independently configurable source.
    """
    if surface_model is not None:
        positions = np.asarray(bundle.surface_position_m)
        directions = np.asarray(bundle.surface_direction)
        radius = surface_model.geometry.apex_radius_nm * 1e-9
        normals = (positions + [0., 0., radius])/radius
    elif hasattr(bundle, "surface_normal"):
        positions = np.asarray(bundle.surface_position_m)
        directions = np.asarray(bundle.surface_direction)
        normals = np.asarray(bundle.surface_normal)
    else:
        positions = np.column_stack((bundle.x_m, bundle.y_m, np.zeros_like(bundle.x_m)))
        directions = np.column_stack((bundle.tx_rad, bundle.ty_rad, np.ones_like(bundle.x_m)))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        normals = np.tile([0., 0., 1.], (len(positions), 1))
    return {"ray_id": _frozen(bundle.ray_id, np.int64),
            "position_m": _frozen(positions, np.float64),
            "direction": _frozen(directions, np.float64),
            "normal": _frozen(normals, np.float64)}


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
        angles = np.where(raw_ids >= 0, angles, np.nan)
        return _frozen(raw_ids, np.int64), _frozen(angles, np.float64)
    except (ValueError, TypeError, OverflowError):
        return None


def source_identity(incident):
    """Return immutable IDs and fixed source-position azimuths (radians).

    Read explicit current metadata only. Missing or duplicate incident IDs
    remain unknown; a row number or downstream position is not an identity.
    """
    count = _count(incident)
    existing = _stored(incident, count)
    if existing is not None:
        known_ids = existing[0][existing[0] >= 0]
        # Repeated ancestors are valid downstream, but never identify two
        # different source samples.
        if np.unique(known_ids).size == known_ids.size:
            return existing
    return _unknown(count)


def emitted_source_identity(gun_trace):
    """Attach lineage at a completed gun-to-column calculation boundary.

    IDs come from the executed exit bundle and positions from its recorded
    emission events. The lookup permits reordered or sparse IDs but never
    infers identity from coincident trajectories or a surviving centroid.
    """
    try:
        ids = np.asarray(gun_trace.exit_bundle.ray_id)
        reference = gun_trace.emission_reference
        original_ids = np.asarray(reference["ray_id"])
        positions = np.asarray(reference["position_m"], dtype=float)
        for values in (ids, original_ids):
            if (values.ndim != 1 or values.dtype.kind not in "iu"
                    or np.any(values < 0) or np.any(values > np.iinfo(np.int64).max)
                    or np.unique(values).size != values.size):
                raise ValueError
        if (positions.shape != (original_ids.size, 3)
                or not np.all(np.isfinite(positions))):
            raise ValueError
        order = np.argsort(original_ids)
        rows = np.searchsorted(original_ids[order], ids)
        if np.any(rows >= original_ids.size) or not np.array_equal(original_ids[order[rows]], ids):
            raise ValueError
        positions = positions[order[rows]]
    except (AttributeError, KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError("Executed gun source identity requires matching recorded emission IDs and positions") from exc
    angles = np.full(ids.size, np.nan)
    defined = np.hypot(positions[:, 0], positions[:, 1]) > 1.0e-15
    angles[defined] = np.mod(np.arctan2(positions[defined, 1], positions[defined, 0]), 2*np.pi)
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
    """Read explicit lineage, or map explicit parent indices to incident IDs."""
    count = _count(branch)
    incident = getattr(simulation, "incident", None)
    if branch is incident or (incident is None and getattr(branch, "name", "") == "incident"):
        return source_identity(branch)
    existing = _stored(branch, count)
    if existing is not None:
        return existing
    if incident is None:
        return _unknown(count)
    ids, angles = source_identity(incident)
    source_indices = getattr(branch, "source_ray_index", None)
    if source_indices is not None and np.asarray(source_indices).shape == (count,):
        return select_identity(ids, angles, source_indices)
    return _unknown(count)
