"""Read-only path evidence for conjugate candidates; never transports a ray."""

from collections import OrderedDict
from collections.abc import Mapping
import math

import numpy as np

from temsim.gui.beam_display_source import downstream_display_branches
from temsim.gui.plane_cutoff_events import PlaneCutoffEventsCache


def _branch_counts(branch, z_mm):
    """Count retained representatives at two neighbouring rows, never extrapolate."""
    name = str(getattr(branch, "name", "branch"))
    try:
        z = np.asarray(branch.z, dtype=float)
        x, y = np.asarray(branch.x), np.asarray(branch.y)
        stop = np.asarray(branch.blocked_z, dtype=float)
        if (z.ndim != 1 or not z.size or not np.all(np.isfinite(z))
                or np.any(np.diff(z) <= 0.) or x.ndim != 2 or y.shape != x.shape
                or x.shape[0] != z.size or x.dtype.kind not in "fiu" or y.dtype.kind not in "fiu"
                or stop.shape != (x.shape[1],)):
            raise ValueError("invalid cached history")
        if not z[0] <= z_mm <= z[-1]:
            return f"{name}: unavailable at this Z (outside retained history; no extrapolation)."
        weight = float(getattr(branch, "weight", 1.))
        if not math.isfinite(weight) or weight < 0.:
            raise ValueError("invalid branch weight")
        raw_weights = getattr(branch, "ray_weight", None)
        if raw_weights is None:
            eligible = np.full(x.shape[1], weight > 0., dtype=bool)
        else:
            weights = np.asarray(raw_weights, dtype=float)
            if weights.shape != stop.shape or np.any(~np.isfinite(weights)) or np.any(weights < 0.):
                raise ValueError("invalid path weights")
            eligible = (weights > 0.) & (weight > 0.)
        right = int(np.searchsorted(z, z_mm, side="left"))
        left = right if z[right] == z_mm else right - 1
        finite = (np.isfinite(x[left]) & np.isfinite(y[left])
                  & np.isfinite(x[right]) & np.isfinite(y[right]))
        before = eligible & np.isfinite(stop) & (stop < z_mm - 1.e-9)
        reached = eligible & finite & ~before & ~np.isinf(stop)
        unknown = eligible & ~before & ~reached
        return (
            f"{name}: {np.count_nonzero(reached)} reach this Z in retained history; "
            f"{np.count_nonzero(before)} have a recorded stop before Z; "
            f"{np.count_nonzero(unknown)} unavailable; "
            f"{np.count_nonzero(~eligible)} zero-weight support paths excluded."
        )
    except (AttributeError, TypeError, ValueError, OverflowError):
        return f"{name}: path reach unavailable (missing or invalid retained history/weights)."


class RecordedConjugateContext:
    """Small per-publication exact-Z cache, queried only for selected table rows."""

    CACHE_LIMIT = 64

    def __init__(self):
        self._result = None
        self._cache = OrderedDict()
        self._cutoffs = PlaneCutoffEventsCache()

    def set_result(self, result):
        # Explicit republication invalidates even a reused result object.
        self._result = result
        self._cache.clear()
        self._cutoffs.clear()

    def at(self, z_mm):
        z_mm = float(z_mm)
        if z_mm not in self._cache:
            self._cache[z_mm] = self._read(z_mm)
            while len(self._cache) > self.CACHE_LIMIT:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(z_mm)
        return self._cache[z_mm]

    def _read(self, z_mm):
        result = self._result
        lines = ["Recorded path representatives — not electron counts or transmission percentages."]
        if result is None or not math.isfinite(z_mm):
            return "\n".join((*lines, "Path reach unavailable: no captured history."))
        simulation = getattr(result, "simulation", None)
        incident = getattr(simulation, "incident", None)
        state = getattr(result, "state_snapshot", None)
        try:
            sample_z = float(state.sample.z_mm)
        except (AttributeError, TypeError, ValueError, OverflowError):
            try:
                sample_z = float(incident.z[-1])
            except (AttributeError, IndexError, TypeError, ValueError, OverflowError):
                sample_z = math.nan
        if not math.isfinite(sample_z):
            lines.append("Path reach unavailable: the captured incident/downstream boundary is missing.")
        elif z_mm <= sample_z:
            lines.append("Source: Incident paths.")
            lines.append(_branch_counts(incident, z_mm))
        else:
            try:
                branches, provenance = downstream_display_branches(result)
            except (AttributeError, TypeError, ValueError):
                branches, provenance = (), "Unavailable"
            lines.append(f"Source: {provenance}.")
            if not branches and provenance == "Specimen exit":
                lines.append("0 forward path representatives: the provenance-checked specimen exit is empty; no optical-reference fallback.")
            elif not branches:
                lines.append("Path reach unavailable: no retained downstream histories.")
            else:
                lines.extend(_branch_counts(branch, z_mm) for branch in branches)

        try:
            events = self._cutoffs.sample(result, z_mm)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return "\n".join((*lines, "Recorded stop details unavailable: invalid captured interception history."))
        names = {}
        for row in getattr(result, "aperture_stops", ()):
            if isinstance(row, Mapping) and row.get("key") is not None:
                names[str(row["key"])] = str(row.get("name", row["key"]))
        for row in (*getattr(state, "recording_planes", ()),
                    *getattr(getattr(result, "assembly", None), "parts", ())):
            if getattr(row, "key", None) is not None:
                names[str(row.key)] = str(getattr(row, "name", row.key))
        if events.groups:
            lines.append("Recorded stops at or upstream of this Z (actual stop positions):")
            for group in events.groups:
                values = np.unique(group.stop_z_mm)
                positions = ", ".join(f"{z:.9g}" for z in values[:6])
                if values.size > 6:
                    positions += f", … ({values.size} retained positions)"
                name = names.get(group.blocked_key, group.blocked_key)
                identity = name if name == group.blocked_key else f"{name} [{group.blocked_key}]"
                lines.append(
                    f"{identity} | {group.stop_kind} | {group.provenance} | Z {positions} mm | "
                    f"{group.observed_count} recorded path representatives."
                )
        else:
            lines.append("No located upstream stop markers in the retained history; this is not proof of clear passage.")
        lines.extend(events.diagnostics)
        return "\n".join(lines)
