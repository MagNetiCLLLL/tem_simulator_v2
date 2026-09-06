"""Deterministic CPU display-data benchmark; no ray tracing or GPU painting."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from types import SimpleNamespace

import numpy as np
from PySide6.QtWidgets import QApplication

from temsim.gui.visualization import VisualizationWorkspace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()
    app = QApplication.instance() or QApplication([])
    workspace = VisualizationWorkspace()
    rng = np.random.default_rng(412)
    rows, rays = 2048, 512
    z = np.linspace(0.0, 3000.0, rows)
    tx = rng.normal(0.0, 0.003, (rows, rays))
    ty = rng.normal(0.0, 0.003, (rows, rays))
    branch = SimpleNamespace(
        name="incident", z=z,
        x=np.cumsum(tx, axis=0) * 1e-6,
        y=np.cumsum(ty, axis=0) * 1e-6,
        tx=tx, ty=ty,
        blocked_z=np.where(np.arange(rays) % 7 == 0, 1530.5, np.nan),
    )
    bounds = [[100.0, 2800.0], [-1.0, 1.0]]
    # Data preparation only: keep Qt painting/event scheduling out of timings.
    workspace.plot = SimpleNamespace(getViewBox=lambda: SimpleNamespace(viewRange=lambda: bounds))
    workspace._last_result = SimpleNamespace(simulation=SimpleNamespace(incident=branch, branches={}))
    angles = (0.0, 17.3, 41.7, 89.0, 117.1, 203.0, 359.9)
    rotation_times, pan_times = [], []
    for repeat in range(args.repeats + 1):
        start = time.perf_counter()
        for angle in angles:
            workspace._projection_angle_deg = angle
            workspace._display_bundle_lines(branch)
        duration = time.perf_counter() - start
        if repeat:
            rotation_times.append(duration / len(angles))
        start = time.perf_counter()
        for left in np.linspace(0.0, 500.0, 16):
            bounds[0] = [float(left), float(left) + 2000.0]
            workspace._maximum_visible_projection_slope()
        duration = time.perf_counter() - start
        if repeat:
            pan_times.append(duration / 16)
    output = {
        "scope": "CPU display data only; synthetic 2048 planes x 512 rays; 48 drawn, 256 slope samples",
        "rotation_ms_median": 1000.0 * statistics.median(rotation_times),
        "pan_slope_ms_median": 1000.0 * statistics.median(pan_times),
    }
    if hasattr(workspace, "ray_display_cache_info"):
        output["cache"] = workspace.ray_display_cache_info()
    print(json.dumps(output, indent=2))
    workspace.close()
    app.processEvents()


if __name__ == "__main__":
    main()
