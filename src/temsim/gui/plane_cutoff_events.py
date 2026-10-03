"""Bounded, display-only interception markers from completed ray histories.

X/Y are laboratory metres and every marker retains its own interception Z in
millimetres. These are recorded path representatives, not electron counts,
current fractions or an effective acceptance mask at the observation plane.
No propagation, clipping or source modification is performed here.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np

from temsim.gui.beam_display_source import downstream_display_branches


_PLANE_TOLERANCE_MM = 1.0e-9
_NUMERICAL_STOPS = frozenset({
    "projected_field_domain", "field_domain", "field_validity_boundary",
    "domain_exit", "initial_outside_domain", "step_limit", "step_underflow",
})


def _frozen(values, dtype=None):
    array = np.ascontiguousarray(values, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True, slots=True)
class PlaneCutoffGroup:
    """One saved stop identity; counts refer only to retained path columns."""

    blocked_key: str
    provenance: str
    stop_kind: str
    x_m: np.ndarray
    y_m: np.ndarray
    stop_z_mm: np.ndarray
    source_ray_id: np.ndarray
    column_index: np.ndarray
    event_id: np.ndarray
    observed_count: int

    @property
    def displayed_count(self) -> int:
        return int(self.x_m.size)


@dataclass(frozen=True, slots=True)
class PlaneCutoffEvents:
    z_mm: float
    groups: tuple[PlaneCutoffGroup, ...]
    total_recorded_count: int
    displayed_count: int
    diagnostics: tuple[str, ...] = ()


def _budget(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= value <= 16384:
        raise ValueError("Interception display budget must be an integer from 1 to 16384")
    return int(value)


def _hardware_keys(result):
    keys = {"column_wall", "feg_tip_reabsorbed"}
    for record in getattr(result, "aperture_stops", ()):
        if isinstance(record, Mapping) and record.get("key") is not None:
            keys.add(str(record["key"]))
    state = getattr(result, "state_snapshot", None)
    gun = getattr(state, "electron_gun", None)
    for row in (*getattr(state, "recording_planes", ()),
                *getattr(gun, "bore_components", ())):
        if getattr(row, "key", None) is not None:
            keys.add(str(row.key))
    return keys


def _stop_kind(key, hardware_keys):
    if key in _NUMERICAL_STOPS or key.startswith("unsupported_field:"):
        return "numerical"
    if key.startswith(("medium_removal:", "medium_backscatter:")) or key.endswith("_backstream"):
        return "non_hardware"
    return "hardware" if key in hardware_keys else "recorded_stop"


def _branch_events(branch, provenance, branch_index, hardware_keys, diagnostics):
    label = f"{provenance} {getattr(branch, 'name', branch_index)}"
    try:
        z = np.asarray(getattr(branch, "z", ()), dtype=float)
        x = np.asarray(getattr(branch, "x", ()))
        y = np.asarray(getattr(branch, "y", ()))
        stops = np.asarray(getattr(branch, "blocked_z", ()), dtype=float)
        keys = np.asarray(getattr(branch, "blocked_key", ()), dtype=object)
        if (z.ndim != 1 or not z.size or np.any(~np.isfinite(z))
                or np.any(np.diff(z) <= 0.) or x.ndim != 2
                or x.shape != y.shape or x.shape[0] != z.size
                or x.dtype.kind not in "fiu" or y.dtype.kind not in "fiu"
                or stops.shape != (x.shape[1],) or keys.shape != stops.shape):
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        diagnostics.append(f"{label}: invalid or unavailable cached interception history; markers omitted.")
        return []
    if np.any(np.isinf(stops)):
        diagnostics.append(f"{label}: non-finite interception Z omitted.")
    indices = np.flatnonzero(np.isfinite(stops))
    # Zero-weight numerical support probes are not emitted path representatives.
    weights = getattr(branch, "ray_weight", None)
    if weights is not None:
        try:
            weights = np.asarray(weights, dtype=float)
            if weights.shape == stops.shape:
                indices = indices[weights[indices] != 0.]
        except (TypeError, ValueError, OverflowError):
            pass
    ids = np.asarray(getattr(branch, "source_ray_id", ()))
    ids_valid = (ids.shape == stops.shape and ids.dtype.kind in "iu"
                 and not np.any(ids < -1) and not np.any(ids > np.iinfo(np.int64).max))
    if not ids_valid and indices.size:
        diagnostics.append(f"{label}: source lineage unavailable; path columns remain distinct.")
    output = []
    missing = outside = invalid = 0
    for index in indices:
        key = keys[index]
        if not isinstance(key, str) or not key.strip():
            missing += 1
            continue
        stop = float(stops[index])
        if not z[0] <= stop <= z[-1]:
            outside += 1
            continue
        right = int(np.searchsorted(z, stop, side="left"))
        if z[right] == stop:
            px, py = float(x[right, index]), float(y[right, index])
        else:
            left = right - 1
            fraction = (stop - z[left]) / (z[right] - z[left])
            px = float(x[left, index]) + fraction * (float(x[right, index]) - float(x[left, index]))
            py = float(y[left, index]) + fraction * (float(y[right, index]) - float(y[left, index]))
        if not math.isfinite(px) or not math.isfinite(py):
            invalid += 1
            continue
        source_id = int(ids[index]) if ids_valid else -1
        output.append((key, provenance, _stop_kind(key, hardware_keys), px, py, stop,
                       source_id, int(index), f"{provenance}:{branch_index}:{index}"))
    for count, reason in ((missing, "missing stop keys"),
                          (outside, "interception Z outside retained history; no extrapolation"),
                          (invalid, "non-finite position at interception")):
        if count:
            diagnostics.append(f"{label}: {count} markers omitted ({reason}).")
    return output


def _recorded_events(result):
    """Read lightweight actual-stop records once; do not copy ray histories."""
    simulation = getattr(result, "simulation", None)
    incident = getattr(simulation, "incident", None)
    if incident is None:
        return (), ("Cached incident paths are unavailable.",)
    diagnostics = []
    hardware_keys = _hardware_keys(result)
    events = _branch_events(incident, "Incident", 0, hardware_keys, diagnostics)
    # Branches may inherit an incident stop at their initial boundary. Preserve
    # weighted downstream siblings, but never count an incoming interception twice.
    incident_stops = {(event[6], event[0], event[5]) for event in events if event[6] >= 0}
    try:
        branches, provenance = downstream_display_branches(result)
    except (AttributeError, TypeError, ValueError):
        branches, provenance = (), "Unavailable"
        diagnostics.append("Cached downstream display history is unavailable.")
    for ordinal, branch in enumerate(branches, 1):
        if branch is incident:
            continue
        values = _branch_events(branch, provenance, ordinal, hardware_keys, diagnostics)
        for event in values:
            if event[6] >= 0 and (event[6], event[0], event[5]) in incident_stops:
                continue
            events.append(event)
    return tuple(events), tuple(dict.fromkeys(diagnostics))


def _group_markers(displayed, counts):
    """Freeze only the bounded visible marker columns, retaining group counts."""
    grouped = {}
    for event in displayed:
        grouped.setdefault(event[:3], []).append(event)
    groups = []
    for (key, provenance, kind), values in grouped.items():
        groups.append(PlaneCutoffGroup(
            key, provenance, kind, _frozen([row[3] for row in values], float),
            _frozen([row[4] for row in values], float),
            _frozen([row[5] for row in values], float),
            _frozen([row[6] for row in values], np.int64),
            _frozen([row[7] for row in values], np.int64),
            _frozen([row[8] for row in values]), counts[(key, provenance, kind)],
        ))
    return tuple(groups)


def sample_plane_cutoff_events(result, z_mm, *, max_events=2048) -> PlaneCutoffEvents:
    """Project already encountered stops upstream from Z, at their actual X/Y.

    Linear interpolation between retained history rows matches the ray diagram's
    endpoint display contract; it is not a newly executed physical intersection.
    Detailed specimen-exit and optical-reference histories are never combined.
    Sampling is deterministic and bounded across all branches. Missing keys or
    unbracketed/non-finite stop positions are explicitly omitted.
    """
    budget = _budget(max_events)
    try:
        selected_z = float(z_mm)
    except (TypeError, ValueError, OverflowError):
        selected_z = math.nan
    if not math.isfinite(selected_z):
        return PlaneCutoffEvents(selected_z, (), 0, 0, ("A finite observation Z is required.",))
    events, recorded_diagnostics = _recorded_events(result)
    diagnostics = list(recorded_diagnostics)
    encountered = [event for event in events if event[5] <= selected_z + _PLANE_TOLERANCE_MM]
    counts = {}
    for event in encountered:
        group_key = event[:3]
        counts[group_key] = counts.get(group_key, 0) + 1
    # Select from the fixed complete history, then filter by observation Z. This
    # keeps displayed path identities stable while the observation plane moves.
    if len(events) > budget:
        selected = np.linspace(0, len(events) - 1, budget, dtype=int)
        displayed = [events[index] for index in selected
                     if events[index][5] <= selected_z + _PLANE_TOLERANCE_MM]
        diagnostics.append(f"Interception display sampled to at most {budget} recorded path representatives.")
    else:
        displayed = encountered
    return PlaneCutoffEvents(selected_z, _group_markers(displayed, counts), len(encountered), len(displayed),
                             tuple(dict.fromkeys(diagnostics)))


class PlaneCutoffEventsCache:
    """Small per-view cache; clear when publishing a new or reused result.

    Retains lightweight actual-stop records once, four bounded plane marker sets,
    and one completed result reference. Moving Z only queries sorted stop planes
    and filters the fixed global display subset; it never reinterpolates paths.
    Completed input arrays are treated as immutable, like the ray display cache.
    """

    def __init__(self, *, max_events=2048):
        self.max_events = _budget(max_events)
        self._result = None
        self._entries = OrderedDict()
        self._events = None
        self._diagnostics = ()
        self._group_stop_z = {}
        self._display_events = ()
        self._selection_budget = None

    def clear(self):
        self._result = None
        self._entries.clear()
        self._events = None
        self._diagnostics = ()
        self._group_stop_z.clear()
        self._display_events = ()
        self._selection_budget = None

    def _prepare(self):
        if self._events is None:
            self._events, self._diagnostics = _recorded_events(self._result)
            stops = {}
            for event in self._events:
                stops.setdefault(event[:3], []).append(event[5])
            self._group_stop_z = {
                key: _frozen(np.sort(values), float) for key, values in stops.items()
            }
        budget = _budget(self.max_events)
        if self._selection_budget != budget:
            self._entries.clear()
            if len(self._events) > budget:
                selected = np.linspace(0, len(self._events) - 1, budget, dtype=int)
                self._display_events = tuple(self._events[index] for index in selected)
            else:
                self._display_events = self._events
            self._selection_budget = budget

    def sample(self, result, z_mm):
        if self._result is not result:
            self.clear()
            self._result = result
        z = float(z_mm)
        if not math.isfinite(z):
            return PlaneCutoffEvents(z, (), 0, 0, ("A finite observation Z is required.",))
        self._prepare()
        value = self._entries.get(z)
        if value is None:
            boundary = z + _PLANE_TOLERANCE_MM
            counts = {key: int(np.searchsorted(positions, boundary, side="right"))
                      for key, positions in self._group_stop_z.items()}
            displayed = [event for event in self._display_events if event[5] <= boundary]
            diagnostics = self._diagnostics
            if len(self._events) > self._selection_budget:
                diagnostics += (f"Interception display sampled to at most {self._selection_budget} recorded path representatives.",)
            value = PlaneCutoffEvents(z, _group_markers(displayed, counts), sum(counts.values()),
                                      len(displayed), diagnostics)
            self._entries[z] = value
            while len(self._entries) > 4:
                self._entries.popitem(last=False)
        else:
            self._entries.move_to_end(z)
        return value
