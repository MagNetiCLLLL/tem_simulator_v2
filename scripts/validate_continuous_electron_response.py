"""B06: bounded real Qt slider / isolated E+B transport / canvas response.

Scripted QSlider signals exercise the production controller, process and painter.
Offscreen paint completion is not monitor presentation or human input latency.
Reports and screenshots are local diagnostics, not physical qualification.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, fields
from datetime import datetime, timezone
from importlib import metadata
import json
import hashlib
import math
import os
from pathlib import Path
import platform
from threading import Event, Thread
import time
import traceback
import sys

MIN_LATENCY_SAMPLES = 20


def trajectory_array_inventory(result):
    """Exact terminal state inventory, computed after the timed paint endpoint."""
    import numpy as np
    inventory = {}
    for field in fields(result):
        value = getattr(result, field.name)
        if not isinstance(value, np.ndarray):
            continue
        if value.dtype.hasobject:
            raise TypeError("Object arrays cannot provide a reproducible physical-state hash")
        contiguous = np.ascontiguousarray(value)
        inventory[field.name] = {
            "shape": list(value.shape), "dtype": value.dtype.str, "nbytes": value.nbytes,
            "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
        }
    return inventory


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(_clean(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def latency_summary(samples_s, *, minimum=MIN_LATENCY_SAMPLES):
    """Publish p95 only after enough completed input/paint pairs were observed."""
    import numpy as np
    values = np.asarray(samples_s, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0.):
        raise ValueError("Latency samples must be finite non-negative durations")
    enough = len(values) >= minimum
    return {"status": "MEASURED" if enough else "INSUFFICIENT_SAMPLES",
            "samples": len(values), "minimum_samples": minimum,
            "p50_ms": float(np.median(values) * 1000.) if len(values) else None,
            "p95_ms": float(np.percentile(values, 95) * 1000.) if enough else None,
            "max_ms": float(np.max(values) * 1000.) if len(values) else None,
            "percentile_method": "linear", "raw_seconds": values.tolist()}


class Probes:
    """Benchmark-only inclusive spans using the existing scalar JobEvents schema."""

    def __init__(self):
        from temsim.job_events import JobEvents
        self.events = JobEvents(maximum=100000)
        self.enabled = True
        self.phase = "setup"
        self.counts = Counter()
        self.spans = []

    @contextmanager
    def stage(self, name):
        if not self.enabled:
            yield
            return
        started, phase = time.perf_counter(), self.phase
        self.events.record("stage_entry", stage=name, phase=phase, backend="Qt", timestamp=started)
        outcome = "returned"
        try:
            yield
        except BaseException:
            outcome = "raised"
            raise
        finally:
            ended = time.perf_counter()
            self.events.record("stage_exit", stage=name, phase=phase, backend="Qt",
                               outcome=outcome, timestamp=ended)
            self.spans.append({"stage": name, "phase": phase, "seconds": ended - started,
                               "outcome": outcome, "accounting": "inclusive; not additive"})


def benchmark_canvas_class():
    """Subclass because replacing an instance Qt virtual is unreliable."""
    from temsim.gui.magnetic_field_canvas import MagneticFieldCanvas

    class BenchmarkCanvas(MagneticFieldCanvas):
        def __init__(self, probes, parent=None, *, profile_paint=False):
            self.probes = probes
            self.profile_paint = profile_paint
            self.display_tokens = ()
            self.paints = []
            self.after_paint = None
            super().__init__(parent)

        def set_electron_paths(self, paths):
            with self.probes.stage("result_display_including_reproject"):
                super().set_electron_paths(paths)

        def _rebuild_projection(self):
            with self.probes.stage("field_reproject_including_electron_reproject"):
                super()._rebuild_projection()

        def _rebuild_electron_projection(self):
            previous = dict(getattr(self, "_electron_projections", {}))
            with self.probes.stage("electron_reproject"):
                super()._rebuild_electron_projection()
            if self.probes.enabled:
                for key, projection in self._electron_projections.items():
                    self.probes.counts["projection_cache_hits" if previous.get(key) is projection
                                       else "projection_builds"] += 1

        def _field_layer(self, transform):
            previous = self._field_layer_pixmap
            with self.probes.stage("field_raster_cache_lookup_or_build"):
                value = super()._field_layer(transform)
            if self.probes.enabled:
                self.probes.counts["field_raster_cache_hits" if previous is value
                                   else "field_raster_builds"] += 1
            return value

        def _electron_display_path(self, key, transform):
            # The pre-optimization production canvas does not call this method.
            cached = getattr(self, "_electron_display_paths", {}).get(key)
            previous = None if cached is None else cached[2]
            with self.probes.stage("electron_display_path_lookup_or_build"):
                value = super()._electron_display_path(key, transform)
            if self.probes.enabled:
                self.probes.counts["display_path_cache_hits" if previous is value
                                   else "display_path_builds"] += 1
            return value

        def paintEvent(self, event):
            tokens = self.display_tokens
            started = time.perf_counter()
            with self.probes.stage("canvas_paint_including_field_raster"):
                super().paintEvent(event)
            row = {"started": started, "ended": time.perf_counter(),
                   "tokens": tokens, "phase": self.probes.phase,
                   "path_count": self.electron_path_count,
                   "projection_revision": self._projection_revision}
            self.paints.append(row)
            if self.after_paint is not None:
                self.after_paint(row)

        def _paint_axes(self, painter):
            if not self.profile_paint:
                return super()._paint_axes(painter)
            with self.probes.stage("canvas_axes"):
                return super()._paint_axes(painter)

        def _paint_labels(self, painter, transform):
            if not self.profile_paint:
                return super()._paint_labels(painter, transform)
            with self.probes.stage("canvas_labels"):
                return super()._paint_labels(painter, transform)

        def _paint_electron_markers(self, painter, transform):
            if not self.profile_paint:
                return super()._paint_electron_markers(painter, transform)
            with self.probes.stage("canvas_electron_markers"):
                return super()._paint_electron_markers(painter, transform)

        def _paint_footer(self, painter):
            if not self.profile_paint:
                return super()._paint_footer(painter)
            with self.probes.stage("canvas_footer"):
                return super()._paint_footer(painter)

    return BenchmarkCanvas


def displayed_tokens(controller, paths):
    """Bind geometry to executed settings, never the latest editor alone."""
    tokens = []
    records = {record.key: record for record in controller.records}
    for path in paths:
        progress = path.key.endswith("/progress")
        key = path.key.removesuffix("/progress") if progress else path.key
        record = records.get(key)
        if record is None:
            continue
        result = record.progress_trajectory if progress else record.trajectory
        settings = (record.progress_key[1] if progress and record.progress_key is not None
                    else record.trajectory_settings)
        generation = (record.progress_key[0] if progress and record.progress_key is not None
                      else record.trajectory_generation)
        tokens.append((key, generation, settings, path.state,
                       0 if result is None else result.steps, id(result)))
    return tuple(tokens)


def matching_paint(paints, *, started, key, generation, settings):
    return next((row for row in paints if row["ended"] >= started and
                 any(token[:4] == (key, generation, settings, "current")
                     for token in row["tokens"])), None)


def instrument_controller(controller, probes):
    original_reuse, original_store = controller._reuse_result, controller._store_result

    def store(record, result, **kwargs):
        with probes.stage("result_acceptance_store"):
            value = original_store(record, result, **kwargs)
        probes.counts["result_store_calls"] += 1
        actual = record.settings if kwargs.get("settings") is None else kwargs["settings"]
        probes.counts["current_result_stores" if actual == record.settings else "older_result_stores"] += 1
        return value

    def reuse(record):
        with probes.stage("exact_result_reuse_lookup_including_store"):
            value = original_reuse(record)
        if value:
            probes.counts["exact_result_cache_hits"] += 1
        return value

    controller._store_result, controller._reuse_result = store, reuse


class ResidentMemory:
    """Sample this benchmark and descendants of its owned backend process only."""

    def __init__(self, backend):
        self.backend = backend
        self.done = Event()
        self.thread = Thread(target=self._sample, daemon=True)
        self._owner_handle = None
        self._owner_identity = None
        self._known_processes = {}
        self._next_discovery = 0.
        self._seen = set()
        self.result = {"interval_ms": 20, "samples": 0, "parent_peak_bytes": 0,
                       "owned_child_peak_bytes": 0, "simultaneous_total_peak_bytes": 0,
                       "owned_pids": [], "errors": [],
                       "tree_discovery_interval_ms": 1000, "tree_discoveries": 0,
                       "owner_changes": 0, "retired_process_identities": 0,
                       "ownership": "Discover descendants of the directly owned backend process at most once per second, or immediately when its process handle changes. Retain identity-bound psutil Process objects and check liveness/PID reuse before and after each RSS read.",
                       "limitation": "Sampled RSS can miss short peaks and children that start/exit between one-second discoveries; shared pages may be counted twice. RSS and identity checks are not an atomic OS snapshot."}

    def _sample_once(self, psutil, parent, now):
        """Cheap RSS sampling; Windows-wide ancestry discovery is rate limited."""
        own, children = parent.memory_info().rss, 0
        handle = self.backend._process
        if handle is not self._owner_handle:
            self._owner_handle = handle
            self._owner_identity = None
            self._known_processes.clear()
            self._next_discovery = now
            self.result["owner_changes"] += 1
        if handle is None or handle.poll() is not None:
            self._known_processes.clear()
        else:
            if now >= self._next_discovery:
                self._next_discovery = now + 1.
                self.result["tree_discoveries"] += 1
                self._known_processes.clear()
                try:
                    root = psutil.Process(handle.pid)
                    # The backend is launched directly by this benchmark process.
                    identity = (root.pid, root.create_time())
                    if (root.is_running() and root.ppid() == parent.pid
                            and self._owner_identity in (None, identity)):
                        self._owner_identity = identity
                        self._known_processes = {
                            child.pid: child for child in (root, *root.children(recursive=True))}
                except psutil.NoSuchProcess:
                    pass
            for pid, child in tuple(self._known_processes.items()):
                try:
                    if not child.is_running():
                        raise psutil.NoSuchProcess(pid)
                    rss = child.memory_info().rss
                    if not child.is_running():
                        raise psutil.NoSuchProcess(pid)
                except psutil.NoSuchProcess:
                    self._known_processes.pop(pid, None)
                    self.result["retired_process_identities"] += 1
                    if pid == handle.pid:
                        self._known_processes.clear()
                        break
                    continue
                children += rss
                self._seen.add(pid)
        self.result["samples"] += 1
        self.result["parent_peak_bytes"] = max(self.result["parent_peak_bytes"], own)
        self.result["owned_child_peak_bytes"] = max(self.result["owned_child_peak_bytes"], children)
        self.result["simultaneous_total_peak_bytes"] = max(self.result["simultaneous_total_peak_bytes"], own + children)

    def _sample(self):
        import psutil
        parent = psutil.Process()
        while not self.done.is_set():
            try:
                self._sample_once(psutil, parent, time.perf_counter())
            except (psutil.Error, OSError) as error:
                if len(self.result["errors"]) < 5:
                    self.result["errors"].append(str(error))
            self.done.wait(.02)
        self.result["owned_pids"] = sorted(self._seen)

    def start(self):
        self.thread.start()

    def stop(self):
        self.done.set()
        self.thread.join(2.)
        self.result["sampler_stopped"] = not self.thread.is_alive()


def probe_overhead(canvas, probes, *, repeats=3, calls=100):
    """Paired cached reprojection, no transport; does not qualify all probes."""
    rows = []
    for repeat in range(repeats):
        row = {"repeat": repeat + 1, "calls": calls}
        for enabled in ((False, True) if repeat % 2 == 0 else (True, False)):
            probes.enabled = enabled
            started = time.perf_counter()
            for _ in range(calls):
                canvas._rebuild_electron_projection()
            row["enabled_s" if enabled else "disabled_s"] = time.perf_counter() - started
        row["extra_seconds_per_call"] = (row["enabled_s"] - row["disabled_s"]) / calls
        rows.append(row)
    probes.enabled = True
    return {"scope": "Paired cached electron reprojection including benchmark instrumentation; no numerical job.",
            "limitations": "Noise can make differences negative. Not whole-pipeline or child-probe overhead.",
            "repeats": rows}


def worker(args):
    # Imports and state construction are inside the externally bounded child.
    import numpy as np
    from PySide6.QtCore import Qt, QTimer, qVersion
    from PySide6.QtWidgets import QDockWidget, QMainWindow
    from temsim.app import create_application
    from temsim.cpu_resources import numerical_job, available_cpu_count, numerical_thread_budget
    from temsim.gui.magnetic_test_electron import TestElectronController
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.magnetic_field_lines import build_field_lines
    from temsim.magnetic_field_scene import prepare_magnetic_scene
    from temsim.optics.column import default_state
    from temsim.test_electron_execution import ElectronExecutionBackend
    from validate_classical_scope import source_hashes, git_metadata

    output, root = args.output / "report.json", Path(__file__).resolve().parents[1]
    report = {"schema": "continuous-electron-response-v2", "case": "B06", "status": "RUNNING",
              "utc": datetime.now(timezone.utc).isoformat(), "runs": [], "backend_requests": [],
              "scope": "Default captured classical E+B diagnostic, real controller, isolated process and software Qt canvas.",
              "timing_semantics": {
                  "latency": "Before scripted QSlider.setValue to first completed canvas paint of those exact accepted settings.",
                  "paint": "Qt paintEvent returned; not monitor scan-out, v-sync or human input.",
                  "nested": "All stage spans are inclusive. IPC request wall includes child transport; store can be inside cache lookup; paint includes field raster. Do not add nested spans.",
                  "cold": "Fresh process first use; filesystem, field disk and Numba disk caches are not purged.",
                  "timer": "10 ms event-loop callback gaps, distinct from input-to-paint latency.",
                  "input": "Programmatic actual QSlider valueChanged/sliderPressed/sliderReleased; no native mouse events.",
              }, "fine_probes": getattr(args, "profile_paint", False),
              "primary_latency_evidence": not getattr(args, "profile_paint", False),
              "paint_profile": {"enabled": getattr(args, "profile_paint", False),
                  "method": "Explicit perf_counter spans in benchmark-only canvas subclass: axes, labels, electron markers, footer; field raster already observed.",
                  "scope": "Executed on the GUI thread; durations include scheduling delays. Remaining paint time is not a direct path-draw measurement.",
                  "limitation": "Fine instrumentation adds overhead. Use a separate run without --profile-paint for primary latency comparisons."},
              "trajectory_array_inventory_semantics": "Every terminal trajectory ndarray: exact dtype, shape and SHA256 of C-order bytes; no rounding or display downsampling. Computed after the measured paint endpoint."}
    _save(output, report)  # Clear old PASS even if provenance reading fails.
    before = None
    controller = window = monitor = backend = canvas = None
    probes, processes = Probes(), []
    deadline = time.perf_counter() + max(1., args.budget_seconds - 8.)
    try:
        before = source_hashes(root)
        report.update(source_hashes=before, git=git_metadata(root))
        app = create_application([])
        report["environment"] = {
            "platform": platform.platform(), "python": sys.version, "executable": sys.executable,
            "qt": qVersion(), "qt_platform": app.platformName(), "logical_cpus": os.cpu_count(),
            "available_cpus": available_cpu_count(), "numerical_budget": numerical_thread_budget(),
            "libraries": {name: metadata.version(name) for name in ("numpy", "scipy", "numba", "PySide6", "psutil")},
        }
        window = QMainWindow()
        window.resize(1500, 940)
        canvas = benchmark_canvas_class()(probes, window,
                                         profile_paint=getattr(args, "profile_paint", False))
        window.setCentralWidget(canvas)
        controller = TestElectronController(window)
        controller._execution_backend.close()
        backend = ElectronExecutionBackend(measure_performance=True)
        controller._execution_backend = backend
        controller.destroyed.connect(backend.close)
        instrument_controller(controller, probes)
        dock = QDockWidget("Virtual electrons", window)
        dock.setWidget(controller.panel)
        window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        window.resizeDocks([dock], [510], Qt.Orientation.Horizontal)

        def display(paths):
            canvas.display_tokens = displayed_tokens(controller, paths)
            canvas.set_electron_paths(paths)

        controller.paths_changed.connect(display)
        controller.background_changed.connect(canvas.set_field_lines_visible)
        canvas.set_electron_mode(True)
        window.show()
        monitor = ResidentMemory(backend)
        monitor.start()

        def measured_backend(name, function):
            def call(*values, **kwargs):
                started = time.perf_counter()
                entry = {"kind": name, "phase": probes.phase, "started": started,
                         "request_metadata": dict(kwargs.get("request_metadata") or {})}
                report["backend_requests"].append(entry)
                probes.counts[name + "_calls"] += 1
                try:
                    return function(*values, **kwargs)
                except BaseException as error:
                    entry["error"] = type(error).__name__ + ": " + str(error)
                    raise
                finally:
                    entry["wrapper_wall_s"] = time.perf_counter() - started
                    entry["performance"] = deepcopy(backend.last_performance)
                    process = backend._process
                    if process is not None and process not in processes:
                        processes.append(process)
            return call

        backend.start = measured_backend("start", backend.start)
        backend.prepare = measured_backend("prepare", backend.prepare)
        backend.trace = measured_backend("trace", backend.trace)

        def check():
            if time.perf_counter() >= deadline:
                raise TimeoutError("B06 total worker deadline reached")
            if controller._scene_error or (controller.selected_record and controller.selected_record.error):
                raise RuntimeError(controller._scene_error or controller.selected_record.error)

        def pump(seconds=.005):
            until = time.perf_counter() + seconds
            while time.perf_counter() < until:
                app.processEvents()
                check()
                time.sleep(.001)

        def wait_current(started):
            while True:
                pump()
                record = controller.selected_record
                if (record is not None and controller.current_trajectory is not None
                        and controller._worker is None and controller._scene_worker is None
                        and record.trajectory_settings == record.settings):
                    painted = matching_paint(canvas.paints, started=started, key=record.key,
                        generation=controller._generation, settings=record.settings)
                    if painted is not None:
                        return painted

        def result_details():
            result = controller.current_trajectory
            return {"settings": asdict(controller.settings()), "reason": result.reason,
                    "completed": result.completed, "steps": result.steps,
                    "end_z_mm": float(result.positions_m[-1, 2] * 1000.),
                    "energy_invariant_error_ev": result.energy_invariant_error_ev,
                    "physical_field_identity": result.physical_field_identity,
                    "numerical_field_identity": result.numerical_field_identity,
                    "execution_identity": result.execution_identity,
                    "trajectory_arrays": trajectory_array_inventory(result)}

        def settled(kind, control, value):
            probes.phase = kind
            old, counts = controller.settings(), probes.counts.copy()
            row = {"kind": kind, "control_value": value, "status": "RUNNING"}
            report["runs"].append(row)
            started = time.perf_counter()  # Includes setter's synchronous cache/display work.
            control.setValue(value)
            if controller.settings() == old:
                raise AssertionError("Benchmark input did not change settings")
            painted = wait_current(started)
            row.update(status="COMPLETE", input_to_paint_s=painted["ended"] - started,
                       counts=dict(probes.counts - counts), result=result_details())
            _save(output, report)
            return row

        probes.phase = "field_geometry_setup"
        with numerical_job(1) as receipt:
            state = default_state()
            report["input_identity"] = capture_instrument_snapshot(state).digest
            report["cpu"] = receipt.to_dict()
            magnetic = prepare_magnetic_scene(state, z_limits_mm=(0., 3026.4))
            geometry = build_field_lines(magnetic, reference_t=.15, max_lines=640, max_steps=64)
        canvas.set_geometry(geometry.segments_m, geometry.strengths_t,
            reference_t=geometry.reference_t, bounds_m=geometry.bounds_m,
            direction_segments_m=geometry.direction_segments_m)
        canvas.set_axial_range_mm(0., 3026.4)
        canvas.set_view_range_mm((0., 3026.4), (-.05, .05))
        probes.phase = "cold_prepare_and_trace"
        started = time.perf_counter()
        report["worker"] = backend.start()
        controller.set_captured_scene(state, magnetic, (0., 3026.4))
        controller.set_active(True)
        painted = wait_current(started)
        report["initial_prepare_and_trace_to_paint_s"] = painted["ended"] - started
        report["initial_result"] = result_details()
        report["field_identity"] = {name: getattr(controller._scene, name) for name in
                                    ("physical_identity", "numerical_identity", "transport_identity")}
        report["canvas"] = {"width": canvas.width(), "height": canvas.height(), "device_pixel_ratio": canvas.devicePixelRatioF()}
        original_pid, original_token = backend.process_id, controller._scene.token
        settled("setup_polar", controller.polar, .5)
        slider, baseline = controller.sliders["x"], controller.sliders["x"].value()
        for tick in range(1, args.warm_repeats + 1):
            row = settled("warm_edit", slider, baseline + tick)
            if row["counts"].get("trace_calls", 0) != 1 or row["counts"].get("exact_result_cache_hits", 0):
                raise AssertionError("Warm edit must execute one new trajectory")
        last = slider.value()
        for index in range(3):
            row = settled("exact_result_reuse", slider, last - 1 if index % 2 == 0 else last)
            if row["counts"].get("exact_result_cache_hits", 0) != 1 or row["counts"].get("trace_calls", 0):
                raise AssertionError("Exact GUI reuse must be observed without a transport request")
        for tick in range(args.settled_samples):
            row = settled("settled_slider", slider, last + tick + 1)
            if row["counts"].get("trace_calls", 0) != 1:
                raise AssertionError("Settled input must reach the real process transport")
        samples = [row["input_to_paint_s"] for row in report["runs"] if row["kind"] == "settled_slider"]
        report["settled_latency"] = latency_summary(samples)
        report["repeated_latencies"] = {kind: {
            "samples": len(values), "median_s": float(np.median(values)), "raw_seconds": values}
            for kind in ("warm_edit", "exact_result_reuse")
            for values in ([row["input_to_paint_s"] for row in report["runs"] if row["kind"] == kind],)}
        if report["settled_latency"]["status"] != "MEASURED":
            raise AssertionError("Insufficient completed input/paint samples")

        settled("held_setup_x", controller.x, -.002650)
        settled("held_setup_polar", controller.polar, 45.)
        probes.phase = "held_slider"
        frames, inputs, timer_times = [], [], []
        baseline, counts_before = slider.value(), probes.counts.copy()
        held_started = time.perf_counter()

        def held_paint(row):
            if slider.isSliderDown():
                frames.append(row)

        canvas.after_paint = held_paint
        timer = QTimer()
        timer.setTimerType(Qt.TimerType.PreciseTimer)
        timer.setInterval(10)
        timer.timeout.connect(lambda: timer_times.append(time.perf_counter()))
        timer.start()
        slider.setSliderDown(True)
        for tick in range(1, args.held_events + 1):
            event_started, old = time.perf_counter(), controller.settings()
            slider.setValue(baseline + tick)
            if old != controller.settings():
                inputs.append((event_started, controller.settings()))
            pump(.015)
        held_seconds = time.perf_counter() - held_started
        release_started = time.perf_counter()
        slider.setSliderDown(False)
        final_paint = wait_current(release_started)
        timer.stop()
        canvas.after_paint = None
        identities = {token[5] for frame in frames for token in frame["tokens"]}
        delivered = sum(any(frame["ended"] >= stamp and any(token[2] == settings and
                        token[3] in {"current", "in_progress"} for token in frame["tokens"])
                        for frame in frames) for stamp, settings in inputs)
        report["held_slider"] = {"seconds": held_seconds, "changed_inputs": len(inputs),
            "painted_frames": len(frames), "distinct_executed_results_or_prefixes": len(identities),
            "inputs_with_matching_painted_geometry": delivered,
            "inputs_coalesced_or_not_painted_while_held": len(inputs) - delivered,
            "empty_frames": sum(not frame["path_count"] for frame in frames),
            "counts": dict(probes.counts - counts_before),
            "timer_gaps": latency_summary(np.diff(timer_times)),
            "after_release_to_final_paint_s": final_paint["ended"] - release_started,
            "result": result_details()}
        if not frames or report["held_slider"]["empty_frames"]:
            raise AssertionError("Canvas displayed an empty frame during editing")
        if len(identities) < 3:
            raise AssertionError("No repeated integrated geometry painted while held")
        probes.phase = "projection_only"
        before_counts, result = probes.counts.copy(), controller.current_trajectory
        projection_rows = []
        for angle in (15., 45., 75.):
            started = time.perf_counter()
            canvas.set_projection_angle(angle)
            revision = canvas._projection_revision
            while not any(row["ended"] >= started and row["projection_revision"] == revision for row in canvas.paints):
                pump()
            projection_rows.append({"angle_deg": angle, "wall_s": time.perf_counter() - started})
        if controller.current_trajectory is not result or probes.counts["trace_calls"] != before_counts["trace_calls"]:
            raise AssertionError("Projection-only edits unexpectedly changed transport")
        report["projection_only"] = {"runs": projection_rows, "counts": dict(probes.counts - before_counts)}
        probes.phase = "probe_overhead"
        report["probe_overhead"] = probe_overhead(canvas, probes)
        probes.phase = "screenshots_excluded_from_timing"
        pump(.02)
        window.grab().save(str(args.output / "completed.png"))
        slider.setSliderDown(True)
        window.grab().save(str(args.output / "held-state.png"))
        slider.setSliderDown(False)
        report["process_and_scene_reused"] = backend.process_id == original_pid and controller._scene.token == original_token
        if not report["process_and_scene_reused"]:
            raise AssertionError("Process or field scene unexpectedly replaced")
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="TIMEOUT" if isinstance(error, TimeoutError) else "FAILED",
                      error=type(error).__name__ + ": " + str(error), traceback=traceback.format_exc())
    finally:
        probes.phase = "cleanup"
        try:
            if backend is not None and backend._process is not None and backend._process not in processes:
                processes.append(backend._process)
            if controller is not None:
                controller.shutdown()
            elif backend is not None:
                backend.close()
            report["owned_processes"] = [{"pid": process.pid, "exit_code": process.poll()} for process in processes]
            report["process_exited"] = all(process.poll() is not None for process in processes)
            if not report["process_exited"]:
                report.update(status="FAILED", cleanup_error="Owned numerical process still alive")
        except BaseException as error:
            report.update(status="FAILED", cleanup_error=type(error).__name__ + ": " + str(error))
        if monitor is not None:
            monitor.stop()
            report["memory"] = monitor.result
        if window is not None:
            window.close()
        report["counts"], report["gui_stages"] = dict(probes.counts), probes.spans
        report["gui_events"] = probes.events.snapshot()
        observed, unavailable = Counter(), 0
        for request in report["backend_requests"]:
            performance = request.get("performance") or {}
            if performance.get("child_measurement_received"):
                observed.update(performance["counts"])
            else:
                unavailable += 1
        report["observed_backend_counts"] = dict(observed)
        report["requests_without_child_measurement"] = unavailable
        report["backend_count_semantics"] = "Only received child observations contribute; failed/unobserved requests are unknown, not zero. GUI cache hits are separate."
        try:
            report["source_unchanged"] = before is not None and before == source_hashes(root)
        except BaseException as error:
            report["source_unchanged"], report["source_error"] = False, str(error)
        if not report["source_unchanged"]:
            report["status"] = "FAILED"
        _save(output, report)
    return 0 if report["status"] == "PASS" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=float, default=240.)
    parser.add_argument("--settled-samples", type=int, default=24)
    parser.add_argument("--warm-repeats", type=int, default=3)
    parser.add_argument("--held-events", type=int, default=80)
    parser.add_argument("--profile-paint", action="store_true",
                        help="Time canvas paint substages; this diagnostic run is not primary latency evidence")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not math.isfinite(args.budget_seconds) or args.budget_seconds <= 10:
        parser.error("--budget-seconds must be finite and greater than 10")
    if args.settled_samples < MIN_LATENCY_SAMPLES or args.warm_repeats < 3 or args.held_events < 20:
        parser.error("Require >=20 settled samples, >=3 warm repeats and >=20 held events")
    if args.settled_samples + args.warm_repeats >= 800 or args.held_events > 500:
        parser.error("Requested samples exceed the bounded physical slider range")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        return worker(args)
    from temsim.validation_process import run_bounded
    report_path = args.output / "report.json"
    _save(report_path, {"schema": "continuous-electron-response-v2", "case": "B06", "status": "STARTING"})
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--output", str(args.output),
               "--budget-seconds", str(args.budget_seconds), "--settled-samples", str(args.settled_samples),
               "--warm-repeats", str(args.warm_repeats), "--held-events", str(args.held_events)]
    if args.profile_paint:
        command.append("--profile-paint")
    environment = os.environ.copy()
    environment.setdefault("QT_QPA_PLATFORM", "offscreen")
    with (args.output / "stdout.log").open("w", encoding="utf-8") as stdout, (args.output / "stderr.log").open("w", encoding="utf-8") as stderr:
        try:
            result = run_bounded(command, timeout=args.budget_seconds, env=environment,
                                 stdout=stdout, stderr=stderr, cwd=Path(__file__).resolve().parents[1])
        except BaseException as error:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report.update(status="FAILED", supervisor_error=type(error).__name__ + ": " + str(error))
            _save(report_path, report)
            raise
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["supervisor_exit_code"] = result.returncode
    if result.returncode == 124:
        report.update(status="TIMEOUT", owned_tree_termination_confirmed=True)
    elif result.returncode or report.get("status") != "PASS":
        report["status"] = "FAILED"
    _save(report_path, report)
    print(json.dumps({"status": report["status"], "report": str(report_path),
                      "settled_latency": report.get("settled_latency")}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
