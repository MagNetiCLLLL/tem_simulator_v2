"""Measure Qt timer latency during real captured-field preparation/transport.

The numerical settings and model are unchanged; this measures GUI scheduling,
not physical qualification. Run with the project interpreter and PYTHONPATH=src.
Generated reports/arrays belong under tmp and are not committed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

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
    if output[1] is not None:
        raise output[1]
    return output[0], {
        "seconds": times[-1]-times[0], "timer_interval_ms": 10.,
        "callbacks": len(gaps)-1,
        "gap_p50_ms": float(np.percentile(gaps, 50)),
        "gap_p95_ms": float(np.percentile(gaps, 95)),
        "gap_p99_ms": float(np.percentile(gaps, 99)),
        "gap_max_ms": float(gaps.max()),
        "gaps_over_50_ms": int(np.count_nonzero(gaps > 50.)),
        "gaps_over_100_ms": int(np.count_nonzero(gaps > 100.)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("thread", "process"), default="thread")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from temsim.cpu_resources import initialize_cpu_resources, numerical_job
    initialize_cpu_resources()
    from temsim.optics.column import default_state
    from temsim.magnetic_field_scene import prepare_magnetic_scene
    from temsim.test_electron_scene import prepare_test_electron_scene
    from temsim.magnetic_test_particle import TestElectronSettings, trace_test_electron
    app = QApplication.instance() or QApplication([])
    args.output.mkdir(parents=True, exist_ok=True)
    with numerical_job(1) as receipt:
        state = default_state()
        magnetic = prepare_magnetic_scene(state, z_limits_mm=(0., 3026.4))
    report = {"scope": "Real default captured E+B fields; 10 ms Qt timer, no rendering", "backend": args.backend,
              "cpu": receipt.to_dict(), "cases": {}}
    _, report["idle"] = measure(lambda: time.sleep(.5))
    if args.backend == "process":
        from temsim.test_electron_execution import ElectronExecutionBackend
        backend = ElectronExecutionBackend()
        worker_info, report["process_startup"] = measure(backend.start)
        report["worker"] = worker_info

        def prepare():
            return backend.prepare(state, magnetic, z_limits_mm=(0., 3026.4))

        def trace(scene, settings):
            return backend.trace(scene, settings)
    else:
        def prepare():
            with numerical_job(1):
                return prepare_test_electron_scene(state, magnetic, z_limits_mm=(0., 3026.4))

        def trace(scene, settings):
            with numerical_job(1):
                return trace_test_electron(scene, settings)

    scene, report["scene_preparation"] = measure(prepare)
    print(json.dumps({"scene_preparation": report["scene_preparation"]}), flush=True)
    for label, path, angle in (("default_full", 3.0264, 0.), ("offaxis_550mm", .55, 5.)):
        settings = TestElectronSettings(kinetic_energy_ev=scene.initial_energy_ev,
            position_m=scene.initial_position_m, max_path_length_m=path, step_m=.001,
            max_steps=20000, polar_angle_deg=angle, azimuth_angle_deg=45. if angle else 0.)
        result, timing = measure(lambda: trace(scene, settings))
        timing.update(settings=asdict(settings), reason=result.reason, steps=result.steps,
                      energy_invariant_error_ev=result.energy_invariant_error_ev)
        report["cases"][label] = timing
        np.savez_compressed(args.output / f"{label}.npz", positions_m=result.positions_m,
            momentum_kg_m_per_s=result.momentum_kg_m_per_s, time_s=result.time_s,
            path_length_m=result.path_length_m, kinetic_energy_ev=result.kinetic_energy_ev,
            electrostatic_potential_v=result.electrostatic_potential_v)
        print(json.dumps({label: timing}), flush=True)
    if args.backend == "process":
        backend.close()
    (args.output/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    app.processEvents()


if __name__ == "__main__":
    main()
