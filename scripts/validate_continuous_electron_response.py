"""Measure held-slider updates through the real E+B process and Qt canvas.

This is a diagnostic UI/transport benchmark, not full-instrument qualification.
Generated reports and screenshots stay in the requested local output directory.
"""
from __future__ import annotations

import argparse
from hashlib import blake2b
import json
from pathlib import Path
import time

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QDockWidget, QMainWindow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from temsim.app import create_application
    from temsim.cpu_resources import numerical_job
    from temsim.gui.magnetic_field_canvas import MagneticFieldCanvas
    from temsim.gui.magnetic_test_electron import TestElectronController
    from temsim.magnetic_field_lines import build_field_lines
    from temsim.magnetic_field_scene import prepare_magnetic_scene
    from temsim.optics.column import default_state

    app = create_application([])
    font_path = Path("C:/Windows/Fonts/segoeui.ttf")
    if font_path.is_file():
        QFontDatabase.addApplicationFont(str(font_path))
        app.setFont(QFont("Segoe UI", 9))
    window = QMainWindow()
    window.resize(1500, 940)
    canvas = MagneticFieldCanvas(window)
    window.setCentralWidget(canvas)
    controller = TestElectronController(window)
    dock = QDockWidget("Virtual electrons", window)
    dock.setWidget(controller.panel)
    window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
    window.resizeDocks([dock], [510], Qt.Orientation.Horizontal)
    controller.paths_changed.connect(canvas.set_electron_paths)
    controller.background_changed.connect(canvas.set_field_lines_visible)
    canvas.set_electron_mode(True)
    window.show()

    def pump(seconds=.02):
        end = time.perf_counter()+seconds
        while time.perf_counter() < end:
            app.processEvents()
            time.sleep(.001)

    def wait_result(timeout=120.):
        started = time.perf_counter()
        while controller.current_trajectory is None or controller._worker is not None:
            pump(.005)
            assert controller._scene_error is None, controller._scene_error
            assert controller.selected_record is None or controller.selected_record.error is None, controller.status_text
            assert time.perf_counter()-started < timeout, controller.status_text
        assert controller.selected_record.trajectory_settings == controller.settings()
        return time.perf_counter()-started

    report = {"scope": "Real default captured E+B diagnostic transport, process backend, native Qt dock and rendered canvas; offscreen display."}
    try:
        with numerical_job(1):
            state = default_state()
            magnetic = prepare_magnetic_scene(state, z_limits_mm=(0., 3026.4))
            geometry = build_field_lines(magnetic, reference_t=.15, max_lines=640, max_steps=64)
        canvas.set_geometry(geometry.segments_m, geometry.strengths_t,
                            reference_t=geometry.reference_t, bounds_m=geometry.bounds_m,
                            direction_segments_m=geometry.direction_segments_m)
        canvas.set_axial_range_mm(0., 3026.4)
        canvas.set_view_range_mm((0., 3026.4), (-.05, .05))
        controller.set_captured_scene(state, magnetic, (0., 3026.4))
        controller.set_active(True)
        report["initial_prepare_and_trace_s"] = wait_result()
        process = controller._execution_backend._process
        token = controller._scene.token
        report["pid"] = process.pid
        execution = []
        original_trace = controller._execution_backend.trace

        def measured_trace(scene, settings, **kwargs):
            started = time.perf_counter()
            entry = {"started": started, "polar_deg": settings.polar_angle_deg,
                     "x_m": settings.position_m[0], "prefixes": []}
            execution.append(entry)
            callback = kwargs.get("progress")
            if callback is not None:
                def progress(result):
                    entry["prefixes"].append(time.perf_counter()-started)
                    callback(result)
                kwargs["progress"] = progress
            try:
                result = original_trace(scene, settings, **kwargs)
                entry["reason"] = result.reason
                return result
            except Exception as error:
                entry["error"] = type(error).__name__
                raise
            finally:
                entry["seconds"] = time.perf_counter()-started

        controller._execution_backend.trace = measured_trace

        # Execute a fresh full path after warming process and compiled kernels.
        controller.azimuth.setValue(1.)
        report["warm_full_path_update_s"] = wait_result()
        controller.x.setValue(-.002650)
        controller.polar.setValue(45.)
        report["warm_user_offaxis_update_s"] = wait_result()

        slider = controller.sliders["x"]
        frames, timer_times = [], []
        drag_start = [None]
        frame_image = [None]

        def collect(paths):
            if not slider.isSliderDown():
                return
            values = [(path.key, path.state, blake2b(path.positions_m.tobytes(), digest_size=8).hexdigest(),
                       float(path.positions_m[-1, 2]), len(path.positions_m)) for path in paths]
            frames.append((time.perf_counter()-drag_start[0], values))
            if any(key.endswith("/progress") for key, *_ in values) and frame_image[0] is None:
                frame_image[0] = window.grab()

        controller.paths_changed.connect(collect)
        timer = QTimer()
        timer.setTimerType(Qt.TimerType.PreciseTimer)
        timer.setInterval(10)
        timer.timeout.connect(lambda: timer_times.append(time.perf_counter()))
        timer.start()
        drag_start[0] = time.perf_counter()
        slider.setSliderDown(True)
        # Move X monotonically around the user's -2.65 nm offset while holding
        # a 45 mrad polar angle. Unique values separate new integration from
        # cache hits and arrive faster than the live dispatch cadence.
        baseline = slider.value()
        for tick in range(80):
            slider.setValue(baseline + tick)
            pump(.015)
        held_seconds = time.perf_counter()-drag_start[0]
        slider.setSliderDown(False)
        report["after_release_final_s"] = wait_result()
        timer.stop()
        paths = controller.visible_paths()
        assert paths and all(path.state == "current" for path in paths)
        assert controller._scene.token == token and controller._execution_backend.process_id == process.pid
        assert frames and all(frame for _, frame in frames), "Plot became empty during editing"
        geometry_ids = {digest for _, values in frames for _, _, digest, _, _ in values}
        progress_ids = {digest for _, values in frames for key, _, digest, _, _ in values if key.endswith("/progress")}
        full_ids = {digest for _, values in frames for key, _, digest, _, _ in values if not key.endswith("/progress")}
        (args.output/"frames-check.json").write_text(json.dumps({**report, "frames": frames}, indent=2), encoding="utf-8")
        assert len(geometry_ids) > 2, "No continuously changing integrated geometry during held drag"
        gaps = np.diff(timer_times)*1000.
        report.update(
            held_slider_seconds=held_seconds, held_input_events=80,
            displayed_geometry_count=len(geometry_ids), progress_geometry_count=len(progress_ids),
            completed_path_updates_while_held=len(full_ids)-1,
            empty_frames=0, process_and_scene_reused=True,
            final_settings=repr(controller.settings()), final_reason=controller.current_trajectory.reason,
            final_end_z_mm=float(controller.current_trajectory.positions_m[-1, 2]*1000.),
            polar_text=controller.polar.text(), polar_degrees=controller.polar_degrees.text(),
            timer_gap_p95_ms=float(np.percentile(gaps, 95)), timer_gap_max_ms=float(gaps.max()),
            execution=execution, frames=frames)
        pump(.05)
        window.grab().save(str(args.output/"completed.png"))
        if frame_image[0] is not None:
            frame_image[0].save(str(args.output/"during-drag.png"))
    finally:
        controller.shutdown()
        window.close()
        pump(.1)
    report["process_exited"] = process.poll() is not None
    assert report["process_exited"]
    (args.output/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"frames", "execution"}}, indent=2))


if __name__ == "__main__":
    main()
