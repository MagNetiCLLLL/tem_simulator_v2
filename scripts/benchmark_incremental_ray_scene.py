"""Synthetic Ray Diagram publication benchmark; no physical propagation.

Run with the project Python. Timings separate synchronous Qt/data updates from
event dispatch and an explicit offscreen raster capture. They are not desktop
frame rates, GPU timings, or evidence of electron-optical accuracy.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
from tempfile import TemporaryDirectory
from time import perf_counter, process_time
from types import SimpleNamespace
from typing import Callable

# Set this before importing Qt; an explicitly configured platform is retained.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
from PySide6 import __version__ as pyside_version
from PySide6.QtCore import QCoreApplication, QEvent, QSettings, qVersion
from PySide6.QtWidgets import QApplication

from temsim.gui.visualization import VisualizationWorkspace
from temsim.physics.simulation import Branch, Simulation


class SyntheticAssembly(SimpleNamespace):
    def part(self, key: str):
        for part in self.parts:
            if part.key == key:
                return part
        raise KeyError(key)


def synthetic_assembly(component_count: int) -> SyntheticAssembly:
    """Explicit display-only parts; dimensions are not an instrument design."""
    parts = []
    for index, z_mm in enumerate(np.linspace(80.0, 2920.0, component_count - 2)):
        kind = "aperture" if index % 9 == 0 else "lens" if index % 3 == 0 else "unit"
        parts.append(SimpleNamespace(
            key=f"synthetic_{kind}_{index}", name=f"Synthetic {kind} {index}",
            start_z_mm=float(z_mm - 5), center_z_mm=float(z_mm), end_z_mm=float(z_mm + 5),
            length_mm=10.0, data={},
        ))
    parts.extend((
        SimpleNamespace(key="sample", name="Synthetic sample reference", start_z_mm=1599.0,
                        center_z_mm=1600.0, end_z_mm=1601.0, length_mm=2.0, data={}),
        SimpleNamespace(key="camera", name="Synthetic camera", start_z_mm=2989.0,
                        center_z_mm=2990.0, end_z_mm=2991.0, length_mm=2.0,
                        data={"mechanical_profile": "camera_sensor_plane", "outer_width_mm": 1.0}),
    ))
    return SyntheticAssembly(
        parts=tuple(sorted(parts, key=lambda part: part.center_z_mm)),
        selected_module_paths=("synthetic-display-benchmark",),
        vacuum_bore_segments=(SimpleNamespace(
            key="synthetic_vacuum", start_z_mm=0.0, end_z_mm=3000.0, inner_diameter_mm=1.0,
        ),),
    )


def synthetic_result(assembly, planes: int, rays: int, variant: int):
    """Prescribed float64 histories: Z in mm, X/Y in m, slopes in rad.

    The three bundles exercise colour groups, stop markers and XY inspection.
    They have no claimed material, optical solution, or quantitative signal.
    """
    rng = np.random.default_rng(412)
    directions = rng.normal(size=(2, rays))

    def bundle(name, start, end, colour, kind, weight, offset):
        z_mm = np.linspace(start, end, planes)
        fraction = np.linspace(0.0, 1.0, planes)[:, None]
        envelope = 0.15 + np.abs(fraction - 0.55)
        x_m = 1e-4 * (directions[0][None, :] * envelope * (1 + 0.06 * variant)
                        + offset * fraction + 0.15 * np.sin(9 * fraction))
        y_m = 1e-4 * (directions[1][None, :] * envelope * (1 - 0.03 * variant)
                        - offset * fraction + 0.13 * np.cos(8 * fraction))
        tx, ty = (np.gradient(values, z_mm * 1e-3, axis=0) for values in (x_m, y_m))
        stopped = (np.arange(rays) % 11 == 0) if name != "incident" else np.zeros(rays, bool)
        blocked_z = np.where(stopped, start + 0.76 * (end - start), np.nan)
        return Branch(
            name, colour, z_mm, x_m, y_m, tx, ty, ~stopped, blocked_z,
            np.where(stopped, "synthetic_aperture", "").tolist(), weight,
            np.zeros(rays), np.full(rays, 1.0 / rays), interaction_kind=kind,
        )

    incident = bundle("incident", 0.0, 1600.0, (0.6, 0.8, 1.0), "incident", 1.0, 0.0)
    primary = bundle("000", 1600.0, 3000.0, (0.4, 1.0, 0.5), "transmitted", 0.8, 0.0)
    scattered = bundle("synthetic_scattered", 1600.0, 3000.0, (1.0, 0.6, 0.3), "elastic", 0.2, 0.45)
    state = SimpleNamespace(
        sample=SimpleNamespace(z_mm=1600.0, inserted=False, specimen_mode="virtual", diffraction_enabled=False),
        electron_gun=SimpleNamespace(emitted_current_a=1e-10),
        recording_planes=(SimpleNamespace(
            key="camera", z_mm=2990.0, outer_width_mm=1.0, inner_diameter_mm=0.0,
            centre_offset_x_mm=0.0, centre_offset_y_mm=0.0, inserted=True,
            readout_enabled=True, colour="#facc15",
        ),),
    )
    return SimpleNamespace(
        model_signature=f"synthetic-display-{variant}", state_snapshot=state, assembly=assembly,
        simulation=Simulation(incident, {"000": primary, "synthetic_scattered": scattered},
                              {"optical_tuning": True, "tuning_quality": "Preview",
                               "sample_scattering_model": "synthetic display fixture"}),
        lens_crossovers=({"z_mm": 880.0 + variant, "name": "Synthetic waist", "rms_radius_mm": 0.02},),
        aperture_stops=tuple({
            "key": part.key, "z_mm": part.center_z_mm, "enabled": True, "installed": True,
            "diameter_mm": 0.6, "offset_x_mm": 0.0, "offset_y_mm": 0.0,
        } for part in assembly.parts if "aperture" in part.key),
    )


def measure(
    app: QApplication, view: VisualizationWorkspace, operation: Callable[[], None],
    *, include_scene_timing: bool = False,
) -> dict[str, float]:
    start_wall, start_cpu = perf_counter(), process_time()
    operation()
    update_cpu_ms = 1000 * (process_time() - start_cpu)
    update_ms = 1000 * (perf_counter() - start_wall)
    start = perf_counter()
    # Two bounded dispatch passes, not a sleep or a claim to drain future timers.
    app.processEvents()
    app.processEvents()
    event_ms = 1000 * (perf_counter() - start)
    start = perf_counter()
    pixmap = view.ray_page.grab()
    if pixmap.isNull():
        raise RuntimeError("Offscreen Ray Diagram raster capture was empty")
    sample = {
        "update_wall_ms": update_ms, "update_process_cpu_ms": update_cpu_ms,
        "event_dispatch_ms": event_ms, "raster_capture_ms": 1000 * (perf_counter() - start),
        "update_and_events_ms": update_ms + event_ms,
    }
    if include_scene_timing:
        scene_ms = scene_info(view).get("last_update_ms")
        if scene_ms is not None:
            sample["ray_scene_update_ms"] = float(scene_ms)
    return sample


def summaries(samples: list[dict[str, float]]) -> dict[str, object]:
    return {
        "samples": len(samples),
        **{key: {"median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
                 "max": float(np.max(values))}
           for key in samples[0] for values in ([sample[key] for sample in samples],)},
    }


def scene_info(view: VisualizationWorkspace) -> dict[str, object]:
    getter = getattr(view, "ray_scene_info", None)
    return dict(getter()) if callable(getter) else {"available": False}


def make_workspace(app: QApplication, args) -> VisualizationWorkspace:
    view = VisualizationWorkspace()
    view.resize(args.width, args.height)
    view.transverse_beam_toggle.setChecked(args.transverse)
    view.magnetic_field_toggle.setChecked(False)
    view.auto_zoom.setChecked(False)
    view.show()
    view.show_ray_diagram()
    app.processEvents()
    return view


def close_workspace(app: QApplication, view: VisualizationWorkspace) -> None:
    view.close()
    app.processEvents()
    view.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--planes", type=int, default=2048)
    parser.add_argument("--rays", type=int, default=512)
    parser.add_argument("--components", type=int, default=80)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--target-ms", type=float, default=50.0)
    parser.add_argument("--transverse", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.repeats < 2 or args.warmups < 0 or min(args.planes, args.rays, args.components) < 3:
        parser.error("Use at least two repeats, nonnegative warmups, and at least three planes/rays/components")
    if not (640 <= args.width <= 3840 and 480 <= args.height <= 2160):
        parser.error("Window dimensions must be 640–3840 by 480–2160 pixels")
    if not np.isfinite(args.target_ms) or args.target_ms <= 0:
        parser.error("Target milliseconds must be finite and positive")
    retained_array_bytes = 2 * 3 * 4 * args.planes * args.rays * np.dtype(float).itemsize
    if retained_array_bytes > 1024**3 or args.components > 500 or args.repeats > 100 or args.warmups > 20:
        parser.error("Keep synthetic history storage below 1 GiB, components <=500, repeats <=100 and warmups <=20")
    app = QApplication.instance() or QApplication([])
    app.setOrganizationName("TemsimSyntheticBenchmark")
    app.setApplicationName("IncrementalRayScene")
    # Prevent a benchmark from reading or changing the user's saved layouts.
    with TemporaryDirectory(prefix="temsim-ray-benchmark-") as settings_dir:
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
        QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, settings_dir)
        assembly = synthetic_assembly(args.components)
        results = [synthetic_result(assembly, args.planes, args.rays, variant) for variant in (0, 1)]
        initial = []
        cold_first = None
        for index in range(args.repeats + args.warmups):
            view = make_workspace(app, args)
            try:
                sample = measure(app, view, lambda: view.display_result(results[0], "Preview"), include_scene_timing=True)
                if cold_first is None:
                    cold_first = sample
                if index >= args.warmups:
                    initial.append(sample)
            finally:
                close_workspace(app, view)
        view = make_workspace(app, args)
        try:
            view.display_result(results[0], "Preview")
            app.processEvents()
            samples = {"same_geometry_publication": [], "rotation": [], "selected_z": []}
            counters = {"before_updates": scene_info(view)}
            for name in samples:
                for index in range(args.repeats + args.warmups):
                    if name == "same_geometry_publication":
                        result = results[(index + 1) % 2]
                        operation = lambda result=result: view.display_result(result, "Preview")
                    elif name == "rotation":
                        angle = float((index * 37.1 + 17.3) % 360)
                        operation = lambda angle=angle: view._set_projection_angle(angle)
                    else:
                        z_mm = 200.0 + float((index * 179) % 2500)
                        operation = lambda z_mm=z_mm: view.jump_to_ray_position(z_mm, activate_tab=False)
                    sample = measure(app, view, operation, include_scene_timing=name == "same_geometry_publication")
                    if index >= args.warmups:
                        samples[name].append(sample)
                counters[f"after_{name}"] = scene_info(view)
            cases = {"fresh_scene_first_publication": summaries(initial),
                     **{name: summaries(rows) for name, rows in samples.items()}}
            output = {
                "schema": "temsim-incremental-ray-scene-benchmark-v1",
                "scope": "Synthetic prescribed rays; real Qt scene; no ray/field/specimen solves; no desktop or GPU FPS claim",
                "timing_notes": "Fixture generation and workspace construction are excluded. Update is synchronous wall/whole-process CPU time. Events are two dispatch passes; raster is an additional forced QWidget grab, not additive display latency.",
                "environment": {"platform": platform.platform(), "python": platform.python_version(),
                                "machine": platform.machine(), "processor": platform.processor(),
                                "numpy": np.__version__, "pyqtgraph": pg.__version__,
                                "pyside": pyside_version, "qt": qVersion(), "qt_platform": app.platformName()},
                "run_options": {"measured_repeats": args.repeats, "warmups": args.warmups, "seed": 412},
                "fixture": {"planes_per_bundle": args.planes, "rays_per_bundle": args.rays, "bundles": 3,
                            "components": len(assembly.parts), "window_size_px": [args.width, args.height],
                            "device_pixel_ratio": float(view.devicePixelRatioF()),
                            "transverse_visible": args.transverse, "magnetic_field_visible": False,
                            "two_result_history_array_bytes": retained_array_bytes,
                            "display_ray_limit_per_bundle": view.MAX_DISPLAY_RAYS},
                "first_process_publication": cold_first,
                "cases": cases, "scene_counters": counters,
                "display_data_cache": view.ray_display_cache_info(),
                "target": {"p95_update_and_events_ms": args.target_ms,
                           "rotation_synchronous_update_passed": cases["rotation"]["update_wall_ms"]["p95"] <= args.target_ms,
                           "selected_z_synchronous_update_passed": cases["selected_z"]["update_wall_ms"]["p95"] <= args.target_ms,
                           "rotation_passed": cases["rotation"]["update_and_events_ms"]["p95"] <= args.target_ms,
                           "selected_z_passed": cases["selected_z"]["update_and_events_ms"]["p95"] <= args.target_ms},
            }
            print(json.dumps(output, indent=2, allow_nan=False))
        finally:
            close_workspace(app, view)


if __name__ == "__main__":
    main()
