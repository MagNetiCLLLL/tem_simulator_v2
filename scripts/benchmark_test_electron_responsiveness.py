"""Measure Qt timer latency during real captured-field preparation/transport.

The numerical settings and model are unchanged; this measures GUI scheduling,
not physical qualification. Run with the project interpreter and PYTHONPATH=src.
Generated reports/arrays belong under tmp and are not committed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
from threading import Event, Thread
import time
import traceback

import numpy as np
from PySide6.QtCore import QEventLoop, QObject, QRunnable, QThreadPool, QTimer, Qt, Signal
from PySide6.QtWidgets import QApplication


class _Signals(QObject):
    done = Signal(object, object)


class _Work(QRunnable):
    def __init__(self, function):
        super().__init__()
        self.function = function
        self.signals = _Signals()

    def run(self):
        result, error = None, None
        try:
            result = self.function()
        except Exception as exc:
            error = exc
        self.signals.done.emit(result, error)


def measure(function):
    loop = QEventLoop()
    timer = QTimer()
    timer.setTimerType(Qt.TimerType.PreciseTimer)
    timer.setInterval(10)
    times = [time.perf_counter()]
    output = []
    timer.timeout.connect(lambda: times.append(time.perf_counter()))

    def finished(result, error):
        output.extend((result, error))
        loop.quit()

    worker = _Work(function)
    worker.signals.done.connect(finished)
    timer.start()
    QThreadPool.globalInstance().start(worker)
    loop.exec()
    timer.stop()
    times.append(time.perf_counter())
    gaps = np.diff(times)*1000.
    enough = len(gaps) >= 20
    if output[1] is not None:
        raise output[1]
    return output[0], {
        "seconds": times[-1]-times[0], "timer_interval_ms": 10.,
        "callbacks": len(gaps)-1,
        "gap_sample_status": "MEASURED" if enough else "INSUFFICIENT_SAMPLES",
        "gap_sample_scope": "Event-loop heartbeat including initial/final partial interval; not input-to-paint latency",
        "gap_p50_ms": float(np.percentile(gaps, 50)),
        "gap_p95_ms": float(np.percentile(gaps, 95)) if enough else None,
        "gap_p99_ms": float(np.percentile(gaps, 99)) if len(gaps) >= 100 else None,
        "gap_max_ms": float(gaps.max()),
        "gaps_over_50_ms": int(np.count_nonzero(gaps > 50.)),
        "gaps_over_100_ms": int(np.count_nonzero(gaps > 100.)),
    }


CASES = ("B01", "B02", "B03", "B04")
ARRAYS = ("positions_m", "directions", "momentum_kg_m_per_s", "time_s",
          "path_length_m", "kinetic_energy_ev", "electrostatic_potential_v", "speed_m_per_s")


def arrays(result):
    return {name: getattr(result, name) for name in ARRAYS}


def result_receipt(result):
    from temsim.particle_benchmark import array_inventory
    return {"reason": result.reason, "completed": result.completed, "steps": result.steps,
            "energy_invariant_error_ev": result.energy_invariant_error_ev,
            "energy_invariant_relative_error": result.energy_invariant_relative_error,
            "endpoint_m": result.positions_m[-1].tolist(), "tof_s": float(result.time_s[-1]),
            "path_m": float(result.path_length_m[-1]), "arrays": array_inventory(arrays(result)),
            "physical_field_identity": result.physical_field_identity,
            "numerical_field_identity": result.numerical_field_identity,
            "execution_identity": result.execution_identity}


def reference_comparison(first, second):
    """Declared local diagnostic tolerances, not instrument qualification."""
    tolerances = {"positions_m": (2e-7, 2e-12), "momentum_kg_m_per_s": (2e-7, 1e-29),
                  "time_s": (2e-7, 1e-17), "path_length_m": (2e-7, 2e-12),
                  "directions": (2e-7, 2e-9), "kinetic_energy_ev": (2e-7, 1e-5),
                  "electrostatic_potential_v": (2e-7, 1e-5), "speed_m_per_s": (2e-7, 1e-3)}
    checks = {}
    for name, (rtol, atol) in tolerances.items():
        left, right = getattr(first, name), getattr(second, name)
        shape = left.shape == right.shape
        checks[name] = {"rtol": rtol, "atol": atol, "same_shape": shape,
                       "close": bool(shape and np.allclose(left, right, rtol=rtol, atol=atol)),
                       "max_abs_difference": float(np.max(np.abs(left-right))) if shape else None}
    return {"accepted": (first.reason == second.reason and first.completed == second.completed
                          and all(row["close"] for row in checks.values())),
            "same_reason": first.reason == second.reason, "arrays": checks}


class MemorySample:
    """Sample only this process and its directly owned numerical child."""
    def __init__(self, backend):
        self.backend, self.done, self.peak, self.samples = backend, Event(), 0, 0
        self.thread = Thread(target=self.run, daemon=True)

    def run(self):
        import psutil
        while not self.done.is_set():
            total = psutil.Process().memory_info().rss
            child = self.backend._process
            if child is not None and child.poll() is None:
                try:
                    total += psutil.Process(child.pid).memory_info().rss
                except psutil.Error:
                    pass
            self.peak = max(self.peak, total)
            self.samples += 1
            self.done.wait(.025)

    def close(self):
        self.done.set()
        self.thread.join(timeout=2.)
        return {"sampled_peak_rss_bytes": self.peak, "samples": self.samples,
                "interval_s": .025, "scope": "Benchmark parent plus owned electron child; sampled, not OS high-water mark"}


def exact_gui_lookup(scene, settings, result, repeats):
    """Use the existing controller cache; never implement a benchmark-only cache."""
    from temsim.gui.magnetic_test_electron import TestElectronController, ElectronRecord
    from PySide6.QtWidgets import QMainWindow
    owner = QMainWindow()
    controller = TestElectronController(owner)
    controller._scene = scene
    record = ElectronRecord("benchmark", "Benchmark electron", "#ffd166", settings)
    controller._records[record.key] = record
    controller._store_result(record, result)
    timings = []
    try:
        for _ in range(repeats):
            record.trajectory = None
            start = time.perf_counter()
            hit = controller._reuse_result(record)
            timings.append(time.perf_counter()-start)
            if not hit or record.trajectory is not result:
                raise AssertionError("Exact controller lookup did not reuse the executed result")
        if controller._execution_backend._process is not None:
            raise AssertionError("Exact reuse unexpectedly started a numerical worker")
    finally:
        controller.shutdown()
        owner.close()
    return {"seconds": timings, "median_seconds": median(timings), "cache_hits": repeats,
            "trajectory_executions": 0, "field_builds": 0, "same_result_object": True,
            "scope": "Actual inactive GUI controller exact lookup, no paint or IPC; see B06 for end-to-end reuse"}


def case_inputs(case):
    from temsim.optics.column import default_state
    state = default_state()
    if case == "B03":
        state.electron_gun.emitter.curvature_nm_inv = .01
    if case == "B04":
        state.monochromator_installed = True
    gun = state.electron_gun
    if case == "B02":
        gun.dpa_aperture.enabled = True
        gun.dpa_aperture.offset_x_mm = 2.*gun.dpa_aperture.radius_mm
    return state, float(gun.exit_plane_z_mm)


def run_case(args):
    from temsim.cpu_resources import numerical_job
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.magnetic_field_scene import prepare_magnetic_scene
    from temsim.magnetic_test_particle import TestElectronSettings
    from temsim.test_electron_execution import ElectronExecutionBackend, ElectronExecutionPolicy
    from temsim.particle_benchmark import environment_receipt, require_exact_arrays
    from validate_classical_scope import source_hashes, git_metadata
    root = Path(__file__).resolve().parents[1]
    app = QApplication.instance() or QApplication([])
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output/"report.json"
    if destination.exists():
        raise ValueError("Use a new report directory")
    report = {"schema": "diagnostic-electron-performance-v1", "case": args.case,
              "status": "RUNNING", "started_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Bounded physical tip-origin classical diagnostic; no specimen/detector signals",
              "source_before": source_hashes(root), "git": git_metadata(root),
              "environment": environment_receipt(), "runs": [],
              "accounting": "Qt request wall includes IPC, acceptance and nested child stages; do not add them",
              "cold_definition": "Fresh child process; existing field/compiler disk caches retained and counted",
              "render_or_reproject": "NOT_RUN in B01-B04; B06 measures actual paint"}
    backend = ElectronExecutionBackend(measure_performance=True,
        policy=ElectronExecutionPolicy(prepare_timeout_s=args.budget_seconds,
                                       trace_timeout_s=args.budget_seconds))
    memory = MemorySample(backend)
    memory.thread.start()

    def save():
        pending = destination.with_suffix(".pending")
        pending.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        pending.replace(destination)

    def timed(label, function):
        value, timing = measure(function)
        row = {"label": label, "qt_wall": timing, "performance": backend.last_performance}
        report["runs"].append(row)
        save()
        return value, row

    try:
        state, stop_mm = case_inputs(args.case)
        report["inputs"] = capture_instrument_snapshot(state).to_dict()
        report["observation_plane_z_mm"] = stop_mm
        report["source"] = {"ray_id": 0, "weight": 1., "stochastic_sampling": False,
                            "definition": "Single virtual diagnostic electron launched at the represented tip"}
        with numerical_job(1) as cpu:
            start = time.perf_counter()
            magnetic = prepare_magnetic_scene(state, z_limits_mm=(0., stop_mm))
            report["magnetic_prepare_seconds"] = time.perf_counter()-start
            report["cpu"] = cpu.to_dict()
        report["worker"], _ = timed("process_start", backend.start)
        scene, _ = timed("cold_field_prepare_or_load", lambda: backend.prepare(
            state, magnetic, z_limits_mm=(0., stop_mm)))
        report["scene"] = asdict(scene)
        for index in range(args.repeats):
            scene, _ = timed(f"warm_field_prepare_or_load_{index+1}", lambda: backend.prepare(
                state, magnetic, z_limits_mm=(0., stop_mm)))
        settings = TestElectronSettings(kinetic_energy_ev=scene.initial_energy_ev,
            position_m=scene.initial_position_m, max_path_length_m=stop_mm*.001,
            step_m=.001, max_steps=20000, polar_angle_deg=1. if args.case == "B02" else 0.,
            azimuth_angle_deg=45. if args.case == "B02" else 0.)
        report["settings"] = asdict(settings)
        reference = None
        warm_seconds, plain_seconds, probe_seconds, probe_pairs = [], [], [], []
        for index in range(args.repeats+1):
            result, row = timed("cold_compiler_transport" if index == 0 else f"warm_transport_{index}",
                                lambda: backend.trace(scene, settings))
            row["result"] = result_receipt(result)
            if reference is None:
                reference = result
                np.savez_compressed(args.output/"executed-trajectory.npz", **arrays(result))
            else:
                row["exact_repeat"] = require_exact_arrays(arrays(reference), arrays(result), "Warm diagnostic transport")
                warm_seconds.append(row["qt_wall"]["seconds"])
            if not result.completed:
                raise AssertionError(f"Diagnostic path incomplete: {result.reason}")
            save()
        report["case_condition"] = {"actual_reason": reference.reason,
            "accepted": args.case != "B02" or reference.reason.startswith("aperture:")}
        if not report["case_condition"]["accepted"]:
            raise AssertionError("B02 did not reach an actual aperture stop")
        for index in range(args.repeats):
            pair = {}
            for enabled in ((True, False) if index % 2 == 0 else (False, True)):
                backend.measure_performance = enabled
                result, row = timed(f"probe_{'enabled' if enabled else 'disabled'}_{index+1}",
                                    lambda: backend.trace(scene, settings))
                row["exact_repeat"] = require_exact_arrays(arrays(reference), arrays(result), "Probe-paired transport")
                pair["enabled_s" if enabled else "disabled_s"] = row["qt_wall"]["seconds"]
                (probe_seconds if enabled else plain_seconds).append(row["qt_wall"]["seconds"])
                save()
            pair["added_seconds"] = pair["enabled_s"]-pair["disabled_s"]
            probe_pairs.append(pair)
        backend.measure_performance = True
        result, row = timed("reference_integrator", lambda: backend.trace(scene, settings, use_compiled=False))
        row["result"] = result_receipt(result)
        report["reference_comparison"] = reference_comparison(reference, result)
        if not report["reference_comparison"]["accepted"]:
            raise AssertionError("Compiled/reference diagnostic comparison exceeded declared tolerances")
        report["exact_result_reuse"] = exact_gui_lookup(scene, settings, reference, args.repeats)
        report["timing_summary"] = {"warm_median_seconds": median(warm_seconds),
            "probe_disabled_median_seconds": median(plain_seconds),
            "probe_enabled_median_seconds": median(probe_seconds),
            "probe_added_median_seconds": median(pair["added_seconds"] for pair in probe_pairs),
            "probe_ratio": median(probe_seconds)/median(plain_seconds), "probe_pairs": probe_pairs,
            "probe_scope": "Alternating paired requests with same arrays/settings/worker and warmed compiled state; scheduling noise included"}
        report["status"] = "MEASURED"
    except Exception:
        report.update(status="FAILED", traceback=traceback.format_exc())
        raise
    finally:
        try:
            backend.close()
        finally:
            report["memory"] = memory.close()
            report["source_after"] = source_hashes(root)
            report["source_unchanged"] = report["source_before"] == report["source_after"]
            if not report["source_unchanged"]:
                report["status"] = "INVALID_SOURCE_CHANGED"
            report["finished_utc"] = datetime.now(timezone.utc).isoformat()
            save()
            app.processEvents()
    print(json.dumps({"case": args.case, "status": report["status"], "report": str(destination)}), flush=True)
    return 0 if report["status"] == "MEASURED" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, default="B01")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--budget-seconds", type=float, default=240.)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 3 or not np.isfinite(args.budget_seconds) or args.budget_seconds <= 0:
        parser.error("At least three repeats and a finite positive wall budget are required")
    if args.worker:
        return run_case(args)
    from temsim.validation_process import run_bounded
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("Choose an empty output folder; prior evidence is never overwritten")
    env = dict(os.environ, QT_QPA_PLATFORM=os.environ.get("QT_QPA_PLATFORM", "offscreen"))
    for key in ("TEMSIM_CPU_THREADS", "NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "1"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]/"src")
    started = time.perf_counter()
    with (output/"worker.log").open("w", encoding="utf-8") as log:
        completed = run_bounded([sys.executable, str(Path(__file__).resolve()), "--worker",
            "--case", args.case, "--output", str(output), "--repeats", str(args.repeats),
            "--budget-seconds", str(args.budget_seconds)], timeout=args.budget_seconds,
            env=env, stdout=log, stderr=subprocess.STDOUT)
    receipt = {"case": args.case, "exit_code": completed.returncode,
               "total_wall_seconds": time.perf_counter()-started, "budget_seconds": args.budget_seconds,
               "status": "FINISHED" if completed.returncode == 0 else "INCOMPLETE"}
    (output/"supervisor.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt), flush=True)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
