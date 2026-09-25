"""Gradient rendering of the captured ray clocks, without transport work."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt

from temsim.gui.ray_curve_item import RayCurveItem
from temsim.gui.ray_scalar_colours import scalar_colour_groups, scalar_rgb


@dataclass(frozen=True, slots=True)
class TimeColourPayload:
    colour_bin: int


def path_times(workspace, branch, index, z):
    """Read clocks at the same clipped vertices used by the ray drawing.

    Column samples use saved-plane interpolation. Curved-tip launch prefixes
    use their original temporal order, including repeated/turning Z positions.
    A missing clock remains NaN, even when geometry is available.
    """
    values = np.full(z.shape, np.nan)
    raw = getattr(branch, "flight_time_s", None)
    if raw is not None and np.shape(raw) == np.shape(branch.x):
        clock = np.asarray(raw)[:, index]
        values = np.interp(z, branch.z, clock, left=np.nan, right=np.nan)
    simulation = getattr(workspace._last_result, "simulation", None)
    history = getattr(getattr(simulation, "gun_trace", None), "equal_time_history", None)
    if (branch is getattr(simulation, "incident", None) and history is not None
            and history.z_mm.shape[1] == branch.x.shape[1]
            and np.any(history.z_mm[0] < branch.z[0])):
        end = 0
        while (end < len(history.z_mm) and history.z_mm[end, index] < branch.z[0]
               and not history.completed[end, index]):
            end += 1
        hz = history.z_mm[:end + 1, index]
        crossings = np.flatnonzero(hz >= branch.z[0])
        end = int(crossings[0]) if crossings.size else len(hz)
        hz = hz[:end]
        valid = (np.isfinite(hz) & np.isfinite(history.x_m[:end, index])
                 & np.isfinite(history.y_m[:end, index]))
        if np.isfinite(branch.blocked_z[index]):
            valid &= hz <= branch.blocked_z[index]
        prefix_z = hz[valid]
        if len(prefix_z) <= len(z) and np.array_equal(prefix_z, z[:len(prefix_z)]):
            active = np.asarray(history.alive)[:end, index] & ~np.asarray(history.completed)[:end, index]
            # A retained inactive row can already be frozen at a boundary or
            # overshoot its event. Its global snapshot time is not an arrival
            # clock. Keep geometry readable, with timing explicitly unknown.
            values[:len(prefix_z)] = np.where(active, np.asarray(history.time_s)[:end], np.nan)[valid]
    values[~np.isfinite(values) | (values < 0) | ~np.isfinite(z)] = np.nan
    return values


class RayFlightTimeColours:
    """Bounded grouped polylines; pan/zoom reuse RayCurveItem detail caches."""
    def __init__(self, workspace):
        self.workspace = workspace
        self.invalidate()

    def invalidate(self):
        self.bundles = ()
        self._signature = None
        self._groups = {}

    def groups(self):
        w = self.workspace
        scale = w.transverse_beam.analysis.tof.colour_scale(w._last_result)
        signature = (tuple(id(b) for b in self.bundles), w._projection_angle_deg,
                     scale.maximum_s)
        if signature == self._signature:
            return self._groups
        pieces = {}
        for branch in self.bundles:
            weights = getattr(branch, "ray_weight", None)
            for index in w._display_ray_indices(branch):
                if weights is not None and weights[index] == 0:
                    continue  # Handled separately with the diagnostic-probe style.
                z, transverse = w._display_bundle_lines(branch, np.array([index]))
                times = path_times(w, branch, index, z)
                for key, arrays in scalar_colour_groups(z, transverse, times, scale.maximum_s).items():
                    pieces.setdefault(key, []).append(arrays)
        self._groups = {key: tuple(np.concatenate([a[axis] for a in rows]) for axis in (0, 1))
                        for key, rows in pieces.items()}
        self._signature = signature
        return self._groups

    def lines(self, payload):
        return self.groups().get(payload.colour_bin, (np.array([]), np.array([])))

    def sync(self, bundles):
        w = self.workspace
        self.bundles = tuple(bundles)
        self._signature = None
        w._convergence_colour_reference_mrad = 0.
        for item in w._ray_legend_items.values():
            w.plot.removeItem(item)
        w._ray_legend_items.clear()
        w._ray_bundle_records = []
        active = set()
        for colour_bin, (z, transverse) in self.groups().items():
            key = ("tof", colour_bin)
            active.add(key)
            item = w._ray_items_by_group.get(key)
            if item is None:
                item = RayCurveItem([], [], pen=pg.mkPen(scalar_rgb(colour_bin), width=1.35))
                w.plot.addItem(item)
                w._ray_items_by_group[key] = item
            item.setData(z, transverse, connect="finite")
            item.setToolTip("Cumulative flight time since tip emission, on the same colour scale as "
                            "the selected-plane plots. Grey: unavailable clock. Cached trajectories only.")
            w._ray_bundle_records.append((item, TimeColourPayload(colour_bin)))
        probes = []
        for branch in bundles:
            weights = getattr(branch, "ray_weight", None)
            if weights is not None:
                indices = w._display_ray_indices(branch)
                indices = indices[np.asarray(weights)[indices] == 0]
                if indices.size:
                    probes.append((branch, indices))
        if probes:
            key = ("support", (248, 250, 252))
            active.add(key)
            item = w._ray_items_by_group.get(key)
            if item is None:
                item = RayCurveItem([], [], pen=pg.mkPen(key[1], width=1.35, style=Qt.PenStyle.DashLine))
                w.plot.addItem(item)
                w._ray_items_by_group[key] = item
            item.setData(*w._ray_record_lines(probes), connect="finite")
            item.setToolTip("Zero-current tip diagnostic probes; not predicted beam current.")
            w._ray_bundle_records.append((item, tuple(probes)))
        for key in tuple(w._ray_items_by_group):
            if key not in active:
                w.plot.removeItem(w._ray_items_by_group.pop(key))
