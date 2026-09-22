"""Immutable, Qt-free lookup of recorded tip emission for display only.

Build once when publishing a completed result. Reading a colour or changing a
display subset never emits electrons, retraces rays, or sorts the source again.
Missing historical launch records stay unavailable; downstream positions and
slopes are not substitutes for the actual tip boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import operator

import numpy as np


def _frozen(values, dtype):
    values = np.asarray(values, dtype=dtype)
    return np.frombuffer(values.tobytes(order="C"), dtype=values.dtype).reshape(values.shape)


def _vectors(reference, key, count, diagnostics):
    """Retain independent recorded quantities even in a partial old record."""
    try:
        values = np.asarray(reference[key], dtype=np.float64)
        if values.shape != (count, 3):
            raise ValueError
    except (KeyError, TypeError, ValueError, OverflowError):
        diagnostics.append(f"{key} is unavailable or has an invalid shape")
        return np.full((count, 3), np.nan), np.zeros(count, dtype=bool)
    valid = np.all(np.isfinite(values), axis=1)
    values = values.copy()
    values[~valid] = np.nan
    if not np.all(valid):
        diagnostics.append(f"{key} contains non-finite rows")
    return values, valid


def _unit_vectors(values, valid, key, diagnostics):
    # Scaling first avoids overflow when inspecting malformed historical data.
    scale = np.max(np.abs(values), axis=1)
    nonzero = valid & (scale > 0.0)
    if np.any(valid & ~nonzero):
        diagnostics.append(f"{key} contains zero vectors")
    result = np.full(values.shape, np.nan)
    if np.any(nonzero):
        scaled = values[nonzero] / scale[nonzero, None]
        result[nonzero] = scaled / np.linalg.norm(scaled, axis=1)[:, None]
    return result, nonzero


@dataclass(frozen=True, slots=True)
class EmissionSourceData:
    """Original emission rows and a precomputed source-ID lookup.

    Positions are metres and angles are radians. Source position azimuth is
    measured about the physical source axis, not a fitted surviving centroid.
    Direction azimuth is about +Z; polar angle is to the local surface normal.
    NaN denotes an unavailable or undefined quantity (including axial azimuth).
    """

    source_ids: np.ndarray
    position_m: np.ndarray
    source_azimuth_rad: np.ndarray
    direction_azimuth_rad: np.ndarray
    angle_to_normal_rad: np.ndarray
    status: str
    _sorted_ids: np.ndarray
    _sorted_rows: np.ndarray

    @classmethod
    def _empty(cls, status):
        empty = _frozen([], np.float64)
        ids = _frozen([], np.int64)
        return cls(ids, _frozen(np.empty((0, 3)), np.float64), empty,
                   empty, empty, str(status), ids, ids)

    @classmethod
    def from_simulation(cls, simulation):
        """Copy actual launch metadata without consulting later beam planes."""
        reference = getattr(getattr(simulation, "gun_trace", None),
                            "emission_reference", None)
        if reference is None:
            return cls._empty("Source emission record unavailable")
        if not isinstance(reference, Mapping):
            return cls._empty("Invalid source emission record")
        try:
            ids = np.asarray(reference["ray_id"])
            if (ids.ndim != 1 or ids.dtype.kind not in "iu"
                    or np.any(ids < 0) or np.any(ids > np.iinfo(np.int64).max)):
                raise ValueError
            ids = ids.astype(np.int64, copy=True)
            order = np.argsort(ids, kind="stable")
            sorted_ids = ids[order]
            if np.any(sorted_ids[1:] == sorted_ids[:-1]):
                raise ValueError
        except (KeyError, TypeError, ValueError, OverflowError):
            return cls._empty("Invalid source IDs: expected unique non-negative integers")
        count = ids.size
        if not count:
            return cls._empty("No recorded emitted source samples")

        diagnostics = []
        positions, position_valid = _vectors(reference, "position_m", count, diagnostics)
        directions, direction_valid = _vectors(reference, "direction", count, diagnostics)
        normals, normal_valid = _vectors(reference, "normal", count, diagnostics)
        directions, direction_valid = _unit_vectors(
            directions, direction_valid, "direction", diagnostics)
        normals, normal_valid = _unit_vectors(normals, normal_valid, "normal", diagnostics)

        source_azimuth = np.full(count, np.nan)
        radial = np.hypot(positions[:, 0], positions[:, 1])
        off_axis = position_valid & (radial > 1.0e-15)
        source_azimuth[off_axis] = np.mod(np.arctan2(
            positions[off_axis, 1], positions[off_axis, 0]), 2.0*np.pi)

        direction_azimuth = np.full(count, np.nan)
        off_axis = direction_valid & (np.hypot(directions[:, 0], directions[:, 1]) > 1.0e-15)
        direction_azimuth[off_axis] = np.mod(np.arctan2(
            directions[off_axis, 1], directions[off_axis, 0]), 2.0*np.pi)

        polar = np.full(count, np.nan)
        valid = direction_valid & normal_valid
        # atan2 remains accurate near normal emission and retains backwards
        # launch hemispheres; slopes alone cannot identify those hemispheres.
        polar[valid] = np.arctan2(
            np.linalg.norm(np.cross(directions[valid], normals[valid]), axis=1),
            np.einsum("ij,ij->i", directions[valid], normals[valid]))
        status = "Ready" if not diagnostics else "Partial source record: " + "; ".join(diagnostics)
        return cls(_frozen(ids, np.int64), _frozen(positions, np.float64),
                   _frozen(source_azimuth, np.float64), _frozen(direction_azimuth, np.float64),
                   _frozen(polar, np.float64), status,
                   _frozen(sorted_ids, np.int64), _frozen(order, np.int64))

    def indices_for(self, ids):
        """Return original record rows, or -1, in the query's original shape."""
        queries = np.asarray(ids)
        output = np.full(queries.size, -1, dtype=np.int64)
        if queries.dtype.kind in "iu" and self._sorted_ids.size:
            flat = queries.reshape(-1)
            valid = (flat >= 0) & (flat <= np.iinfo(np.int64).max)
            query_rows = np.flatnonzero(valid)
            wanted = flat[valid].astype(np.int64, copy=False)
            found = np.searchsorted(self._sorted_ids, wanted)
            present = found < self._sorted_ids.size
            candidates = np.minimum(found, self._sorted_ids.size - 1)
            present &= self._sorted_ids[candidates] == wanted
            output[query_rows[present]] = self._sorted_rows[found[present]]
        return _frozen(output.reshape(queries.shape), np.int64)

    def values(self, ids, mode):
        """Read source position/direction colours for any descendant IDs."""
        quantities = {"source": self.source_azimuth_rad,
                      "emission_direction": self.direction_azimuth_rad,
                      "emission_angle": self.angle_to_normal_rad}
        try:
            quantity = quantities[mode]
        except KeyError as exc:
            raise ValueError(f"Unknown emission colour quantity: {mode}") from exc
        rows = self.indices_for(ids)
        values = np.full(rows.shape, np.nan)
        valid = rows >= 0
        values[valid] = quantity[rows[valid]]
        return _frozen(values, np.float64)

    def display_indices(self, limit):
        """Select a fixed subset of all launch rows, independent of survival."""
        try:
            if isinstance(limit, (bool, np.bool_)):
                raise TypeError
            count = operator.index(limit)
        except TypeError as exc:
            raise ValueError("Source display limit must be a non-negative integer") from exc
        if count < 0:
            raise ValueError("Source display limit must be a non-negative integer")
        count = min(count, self.source_ids.size)
        return _frozen(np.linspace(0, self.source_ids.size - 1, count, dtype=np.int64), np.int64)
