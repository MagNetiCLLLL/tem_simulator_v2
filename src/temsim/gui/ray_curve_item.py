"""Screen-resolution rendering of already-computed, disconnected ray paths.

Only the Qt curve is simplified. Original coordinates, colours, ray selection,
stops and scientific results remain unchanged. Each finite path keeps its own
endpoints and never connects across a NaN/Inf gap. Coordinates use the parent
plot's units (Ray Diagram: axial mm and transverse mm).
"""

from __future__ import annotations

from collections import OrderedDict
import math

import numpy as np
import pyqtgraph as pg

from temsim.gui.ray_curve_simplification import compiled_rdp


def finite_runs(x: np.ndarray, y: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Half-open slices of uninterrupted finite coordinates."""
    finite = np.isfinite(x) & np.isfinite(y)
    edges = np.flatnonzero(np.diff(np.r_[False, finite, False]))
    return tuple(zip(edges[::2].tolist(), edges[1::2].tolist()))


def screen_path_indices(
    x: np.ndarray, y: np.ndarray, x_per_pixel: float, y_per_pixel: float,
    tolerance_pixels: float = 0.35,
) -> np.ndarray:
    """RDP subset bounded by point-to-segment distance in screen coordinates.

    Scale is quantised conservatively by the caller, so this distance bound
    also holds at the actual viewport scale. Short paths keep every point.
    The input is one finite path, and both original endpoints are retained.
    """
    count = len(x)
    if count <= 32:
        return np.arange(count)
    # Shift first to avoid cancellation from a large common coordinate origin.
    xx, yy = (x - x[0]) / x_per_pixel, (y - y[0]) / y_per_pixel
    if not (np.all(np.isfinite(xx)) and np.all(np.isfinite(yy))):
        return np.arange(count)
    # A GUI display acceleration only: one compiled loop, with an independent
    # NumPy fallback if native compilation is unavailable on this installation.
    global compiled_rdp
    if compiled_rdp is not None:
        try:
            return compiled_rdp(xx, yy, tolerance_pixels**2)
        except Exception:
            compiled_rdp = None
    return _numpy_path_indices(xx, yy, tolerance_pixels**2)


def _numpy_path_indices(xx, yy, limit_squared):
    count = len(xx)
    keep = np.zeros(count, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, count - 1)]
    while stack:
        first, last = stack.pop()
        if last - first <= 1:
            continue
        dx, dy = xx[last] - xx[first], yy[last] - yy[first]
        px = xx[first + 1:last] - xx[first]
        py = yy[first + 1:last] - yy[first]
        length_squared = dx*dx + dy*dy
        if length_squared > 0.0:
            fraction = np.clip((px*dx + py*dy) / length_squared, 0.0, 1.0)
            distance_squared = (px - fraction*dx)**2 + (py - fraction*dy)**2
        else:
            distance_squared = px*px + py*py
        index = int(np.argmax(distance_squared))
        if distance_squared[index] > limit_squared:
            split = first + 1 + index
            keep[split] = True
            stack.extend(((first, split), (split, last)))
    return np.flatnonzero(keep)


class RayCurveItem(pg.PlotDataItem):
    """A plain ray polyline with bounded, view-dependent rendering detail.

    Supports the Ray Diagram's line-only styles, not symbols/transforms/fills.
    Public data access and auto-range bounds refer to the full input paths.
    Up to eight scale levels / 4 MiB per colour item are retained; publishing
    new coordinates clears them. Original data is outside this optional cache.
    """

    TOLERANCE_PIXELS = 0.35
    MAX_LEVELS = 8
    MAX_CACHE_BYTES = 4 * 1024**2

    def __init__(self, *args, **kwargs):
        self._ray_levels = OrderedDict()
        self._ray_level_bytes = 0
        self._ray_runs = ()
        self._ray_bounds = ((None, None), (None, None))
        self._ray_render_signature = None
        self._ray_render_source = None
        self._ray_rendered_points = 0
        kwargs.update(connect="finite", dynamicRangeLimit=None,
                      clipToView=False, autoDownsample=False)
        super().__init__(*args, **kwargs)

    def getData(self):
        """Return unchanged display-source samples, not the raster subset."""
        return self.getOriginalDataset()

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        # In particular, Fit must not fit only the currently clipped viewport.
        return self._ray_bounds[ax]

    def updateItems(self, styleUpdate=True):
        x, y = self.getOriginalDataset()
        # setData supplies new array views even when callers reuse an object.
        source = self._ray_render_source
        if source is None or source[0] is not x or source[1] is not y:
            self._ray_render_source = (x, y)
            self._ray_levels.clear()
            self._ray_level_bytes = 0
            self._ray_render_signature = None
            self._ray_runs = () if x is None else finite_runs(x, y)
            finite = None if x is None else np.isfinite(x) & np.isfinite(y)
            self._ray_bounds = tuple(
                (float(np.min(a[finite])), float(np.max(a[finite])))
                if finite is not None and finite.any() else (None, None)
                for a in (x, y)
            )
        # These are the only styles used by ray polylines. No physical data is
        # rewritten, and no calculation or diagnostic is invoked here.
        self.curve.setPen(self.opts["pen"])
        self.curve.setShadowPen(self.opts["shadowPen"])
        self.curve.opts["antialias"] = self.opts["antialias"]
        self.scatter.hide()
        self._refresh_ray_curve()

    def viewRangeChanged(self, vb=None, ranges=None, changed=None):
        self._refresh_ray_curve()

    def viewTransformChanged(self):
        super().viewTransformChanged()
        # Includes widget resizing without a change to the physical ranges.
        if hasattr(self, "curve"):
            self._refresh_ray_curve()

    def setExportMode(self, export, opts=None):
        super().setExportMode(export, opts)
        self._ray_render_signature = None
        self._refresh_ray_curve()

    def _refresh_ray_curve(self):
        x, y = self.getOriginalDataset()
        if x is None or y is None or not self._ray_runs:
            self.curve.setData([], [])
            self.curve.hide()
            self._ray_rendered_points = 0
            return
        view = self.getViewBox()
        if view is None or self._exportOpts is not False:
            self.curve.setData(x, y, connect="finite")
            self.curve.show()
            self._ray_rendered_points = len(x)
            self._ray_render_signature = None
            return
        (left, right), (bottom, top) = view.viewRange()
        scales = ((right-left) / max(view.width(), 1.0),
                  (top-bottom) / max(view.height(), 1.0))
        if not all(math.isfinite(s) and s > 0 for s in scales):
            return
        # Using the smaller dyadic scale makes the error bound conservative.
        level = tuple(math.floor(math.log2(s)) for s in scales)
        paths = self._ray_levels.get(level)
        if paths is None:
            paths = []
            for start, end in self._ray_runs:
                xx, yy = x[start:end], y[start:end]
                selected = screen_path_indices(
                    xx, yy, 2.0**level[0], 2.0**level[1], self.TOLERANCE_PIXELS,
                )
                xx, yy = xx[selected], yy[selected]
                paths.append((xx, yy, bool(np.all(np.diff(xx) >= 0.0))))
            size = sum(xx.nbytes + yy.nbytes for xx, yy, _ in paths)
            if size <= self.MAX_CACHE_BYTES:
                self._ray_levels[level] = paths
                self._ray_level_bytes += size
                while (len(self._ray_levels) > self.MAX_LEVELS
                       or self._ray_level_bytes > self.MAX_CACHE_BYTES):
                    _, removed = self._ray_levels.popitem(last=False)
                    self._ray_level_bytes -= sum(xx.nbytes + yy.nbytes for xx, yy, _ in removed)
        else:
            self._ray_levels.move_to_end(level)
        slices = []
        for xx, yy, monotonic in paths:
            if monotonic:
                if xx[-1] < left or xx[0] > right:
                    slices.append((0, 0))
                    continue
                # Retain one adjacent vertex at each edge to preserve crossing
                # segments even when no stored sample lies inside the view.
                first = max(0, int(np.searchsorted(xx, left, side="left")) - 1)
                last = min(len(xx), int(np.searchsorted(xx, right, side="right")) + 1)
                slices.append((first, last))
            else:
                # Turning/launch histories must never be sorted or clipped as
                # monotonic rays. Their complete simplified path is retained.
                slices.append((0, len(xx)))
        signature = (level, tuple(slices))
        if signature == self._ray_render_signature:
            return
        self._ray_render_signature = signature
        pieces_x, pieces_y = [], []
        for (xx, yy, _), (first, last) in zip(paths, slices):
            if first < last:
                pieces_x.extend((xx[first:last], np.array([np.nan])))
                pieces_y.extend((yy[first:last], np.array([np.nan])))
        render_x = np.concatenate(pieces_x) if pieces_x else np.array([])
        render_y = np.concatenate(pieces_y) if pieces_y else np.array([])
        self._ray_rendered_points = len(render_x)
        self.curve.setData(render_x, render_y, connect="finite")
        self.curve.setVisible(bool(len(render_x)))

    def rendering_info(self):
        x, _ = self.getOriginalDataset()
        return {"source_points": 0 if x is None else len(x),
                "rendered_points": self._ray_rendered_points,
                "cached_levels": len(self._ray_levels),
                "cached_bytes": self._ray_level_bytes,
                "compiled_simplification": compiled_rdp is not None,
                "tolerance_pixels": self.TOLERANCE_PIXELS}
