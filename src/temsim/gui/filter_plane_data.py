"""Read physical filter crossings; global Z is not an outgoing filter axis."""
from dataclasses import replace
import math

import numpy as np

from temsim.gui.beam_plane_data import BeamPlaneData, _frozen
from temsim.gui.beam_display_source import downstream_display_branches
from temsim.gui.beam_tracking_modes import branch_interaction_style
from temsim.component_keys import ENERGY_FILTER_INTERNAL_KEYS, ENERGY_FILTER_ENTRANCE_APERTURE


def _recorded_interactions(result, output, ids):
    """Join saved filter ancestry to its actual upstream branch metadata.

    A repeated source ID can have descendants in several interaction channels;
    only the recorded branch and within-branch path index identify that history.
    Missing ancestry stays unclassified, without looking at the outgoing angle.
    """
    unknown = ("filter", "Energy-filter path (interaction unavailable)", (148, 163, 184), "o")
    styles = [unknown] * len(ids)
    names = getattr(output, "source_branch", ())
    indices = np.asarray(getattr(output, "source_path_index", None))
    if (len(names) != len(ids) or indices.shape != ids.shape or indices.dtype.kind not in "iu"):
        return styles
    provenance = getattr(output, "entrance_provenance", "")
    if provenance == "validated_specimen_exit":
        branches, label = downstream_display_branches(result)
        if label != "Specimen exit":
            return styles
        lookup = {str(branch.name): branch for branch in branches}
    elif provenance == "optical_column_reference":
        simulation = getattr(result, "simulation", None)
        lookup = dict(getattr(simulation, "branches", {}))
        lookup["incident"] = getattr(simulation, "incident", None)
    else:
        return styles
    for row, (name, path, ancestor) in enumerate(zip(names, indices, ids)):
        branch = lookup.get(name)
        branch_ids = np.asarray(getattr(branch, "source_ray_id", None))
        if (branch_ids.ndim != 1 or branch_ids.dtype.kind not in "iu"
                or not 0 <= path < len(branch_ids) or ancestor < 0
                or branch_ids[path] != ancestor):
            continue
        styles[row] = branch_interaction_style(branch)
    return styles


def sample_filter_plane(result, z_mm, component_key):
    """Return None before the filter, otherwise its saved named-plane data.

    A filter's bent trajectory must never inherit the straight column's
    numerical continuation downstream of its entrance. Arbitrary global Z
    cannot identify a plane in the outgoing frame.
    """
    state = getattr(result, "state_snapshot", None)
    config = getattr(state, "energy_filter", None)
    if config is None or not bool(getattr(config, "enabled", False)):
        return None
    output = getattr(result, "energy_filter", None)
    planes = getattr(output, "timed_planes", ()) or ()
    plane = next((p for p in planes if p.key == component_key), None)
    entrance = float(config.entrance_z_mm)
    named_keys = set(ENERGY_FILTER_INTERNAL_KEYS) - {ENERGY_FILTER_ENTRANCE_APERTURE}
    if plane is None and component_key not in named_keys and z_mm <= entrance:
        return None
    empty = _frozen([], dtype=float)
    empty_ids = _frozen([], dtype=np.int64)
    empty_labels = _frozen([], dtype="U1")
    reason = ("Select a saved energy-slit, filter-output or spectrometer detector plane. "
              "A global Z slice cannot sample the bent filter path.")
    data = BeamPlaneData(float(z_mm), empty, empty, empty, empty, empty_ids, empty,
        empty, empty_labels, empty_labels, _frozen(np.empty((0, 3), dtype=np.uint8)),
        empty_labels, empty_ids, 0, "Energy filter", "Unavailable", None, False,
        diagnostics=(reason,), flight_time_s=empty, coordinate_frame="sector_exit_local")
    if plane is None:
        return data
    if getattr(plane, "coordinate_frame", None) != "sector_exit_local":
        return replace(data, diagnostics=("Filter coordinate reference is unavailable.",))
    try:
        arrays = [np.asarray(getattr(plane, name), dtype=float)
                  for name in ("x_m", "y_m", "tx_rad", "ty_rad")]
        n = arrays[0].size
        reached = np.asarray(plane.reached)
        if any(a.shape != (n,) for a in (*arrays, reached)) or reached.dtype.kind != "b":
            raise ValueError("Filter crossing arrays are not aligned.")
        diagnostics = [
            "Executed physical crossing in the sector-exit dispersive/non-dispersive frame. "
            "The filter's existing representative-ray sampling controls resolution; "
            "this is incident flux before the selected detector response."]
        raw_ids = np.asarray(getattr(output, "source_ray_id", None))
        if (raw_ids.shape == (n,) and raw_ids.dtype.kind in "iu" and np.all(raw_ids >= -1)
                and np.all(raw_ids <= np.iinfo(np.int64).max)):
            ids = raw_ids.astype(np.int64)
        else:
            ids = np.full(n, -1, dtype=np.int64)
            diagnostics.append("Recorded filter source identities are unavailable.")
        fractions = np.asarray(getattr(output, "source_fraction", None), dtype=float)
        weights_valid = bool(fractions.shape == (n,) and np.all(np.isfinite(fractions))
                             and np.all(fractions >= 0)
                             and math.fsum(map(float, fractions)) <= 1. + 1e-10)
        if not weights_valid:
            fractions = np.full(n, np.nan)
            diagnostics.append("Filter source probabilities are unavailable; geometry only.")
        finite = np.logical_and.reduce([np.isfinite(a) for a in arrays])
        weighted = ~np.isfinite(fractions) | (fractions > 0)
        keep = reached & weighted & finite
        if np.any(reached & weighted & ~finite):
            weights_valid = False
            diagnostics.append("Non-finite reaching coordinates or slopes were omitted.")
        # Geometry and population are independent of the optional tip-origin
        # clock. Incomplete timing must not remove a physical arrival.
        clock = np.full(n, np.nan)
        try:
            raw_clock = np.asarray(getattr(plane, "time_s", None), dtype=float)
        except (TypeError, ValueError):
            raw_clock = np.empty(0)
        if (getattr(plane, "time_reference", None) == "simultaneous_tip_emission"
                and raw_clock.shape == (n,)):
            known = np.isfinite(raw_clock) & (raw_clock >= 0)
            clock[known] = raw_clock[known]
        else:
            diagnostics.append("Tip-origin filter arrival times are unavailable.")
        metrics = getattr(getattr(result, "simulation", None), "metrics", {})
        current = metrics.get("effective_source_current_pa")
        current = (float(current) if current is not None and not isinstance(current, bool)
                   and math.isfinite(float(current)) and float(current) >= 0 else None)
        count = int(np.count_nonzero(keep))
        upstream = getattr(output, "entrance_provenance", "")
        provenance = ("Specimen exit → energy filter" if upstream == "validated_specimen_exit"
                      else "Optical reference → energy filter" if upstream == "optical_column_reference"
                      else "Energy filter") + " (representative paths)"
        styles = _recorded_interactions(result, output, ids)
        keys = np.asarray([row[0] for row in styles], dtype=str)
        labels = np.asarray([row[1] for row in styles], dtype=str)
        rgb = np.asarray([row[2] for row in styles], dtype=np.uint8).reshape(n, 3)
        symbols = np.asarray([row[3] for row in styles], dtype=str)
        return BeamPlaneData(float(z_mm), *(_frozen(a[keep]) for a in arrays[:4]),
            _frozen(ids[keep]), _frozen(np.full(count, np.nan)), _frozen(fractions[keep]),
            _frozen(keys[keep]), _frozen(labels[keep]), _frozen(rgb[keep]),
            _frozen(symbols[keep]), _frozen(np.flatnonzero(keep), dtype=np.int64), n,
            provenance, "Ready" if count else "No reaching rays",
            current, weights_valid,
            tuple(diagnostics),
            flight_time_s=_frozen(clock[keep]), coordinate_frame="sector_exit_local")
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        return replace(data, diagnostics=(f"Saved filter crossing is incomplete: {exc}",))
