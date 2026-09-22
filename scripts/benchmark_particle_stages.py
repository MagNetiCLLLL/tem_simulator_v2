"""Bounded production classical-particle pipeline timings; never enables waves.

Run in a separate process using the project interpreter. The supervisor stops
only its own worker at the budget. Receipts contain scalar metadata, not caches.
Nested timers wrap real functions without replacing their physical calculation.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from threading import Event, Thread
from time import perf_counter
from unittest.mock import patch


def _clean(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _save(path, receipt):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(_clean(receipt), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def _array_digest(simulation):
    import numpy as np
    digest = hashlib.sha256()
    for branch in (simulation.incident, *simulation.branches.values()):
        for name in ("z", "x", "y", "tx", "ty", "alive", "flight_time_s"):
            value = np.asarray(getattr(branch, name))
            digest.update(str((name, value.dtype.str, value.shape)).encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def worker(options):
    import numpy as np
    import psutil
    from numba import get_num_threads
    from temsim.cpu_resources import numerical_job, numerical_thread_budget, initialize_numerical_thread
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.optics.column import default_state
    from temsim.optics.electron_gun import source, tracing
    import temsim.physics.simulation as simulation
    import temsim.simulation_pipeline as pipeline

    options.threads = numerical_thread_budget(options.threads)
    initialize_numerical_thread(options.threads)
    state = default_state()
    catalog = AssemblyCatalog()
    selection = catalog.default_selection()
    if options.energy_filter != "default":
        installed = options.energy_filter == "on"
        recording = next(option.name for option in catalog.recording_systems
                         if bool(option.properties.get("energy_filter")) == installed)
        selection = replace(selection, recording=recording)
    catalog.apply(state, selection)
    state.electron_gun.emitter.ray_count = options.rays
    if options.step_mm is not None:
        state.step_mm = options.step_mm
    if options.history_step_mm is not None:
        state.history_step_mm = options.history_step_mm
    state.acceleration_backend = "CPU"
    assert not state.sample.wave_enabled and not state.sample.stem_wave_enabled
    assert not state.ac_deflector.scan_enabled
    assert state.electron_gun.emitter.coherence is None
    assert state.electron_gun.emitter.surface_model is None
    snapshot = capture_instrument_snapshot(state)
    receipt = {
        "schema": "particle-stage-benchmark-v1", "status": "RUNNING",
        "host": {"platform": platform.platform(), "python": platform.python_version(),
                 "processor": platform.processor(), "logical_cpus": os.cpu_count(),
                 "physical_ram_bytes": psutil.virtual_memory().total,
                 "numba_threads": get_num_threads(), "numerical_thread_budget": options.threads,
                 "library_threads": 1},
        "input_identity": snapshot.digest, "implementation_identity": snapshot.implementation,
        "settings": {"ray_count": options.rays, "column_step_mm": state.step_mm,
                     "history_step_mm": state.history_step_mm, "backend": "CPU",
                     "acceleration_enabled": state.acceleration_enabled,
                     "beam_voltage_kv": state.beam_voltage_kv,
                     "gun_trace_step_mm": state.electron_gun.trace_step_mm,
                     "gun_history_step_mm": state.electron_gun.history_step_mm,
                     "sample_mode": state.sample.specimen_mode,
                     "sample_reference": state.sample.reference_sample_key,
                     "sample_thickness_nm": state.sample.thickness_nm,
                     "sample_inserted": state.sample.inserted,
                     "eds_enabled": state.sample.eds_enabled,
                     "eds_overlap_sampling_points": state.sample.eds_overlap_sampling_points,
                     "energy_filter_installed": state.energy_filter_installed,
                     "energy_filter_enabled": state.energy_filter.enabled,
                     "vacuum_enabled": state.vacuum_map.enabled,
                     "tem_wave_enabled": False, "stem_wave_enabled": False,
                     "scan_enabled": False, "source": "existing classical flat-tip emission"},
        "runs": [],
        "timing_semantics": {
            "cold": "First pipeline call in a fresh process; OS/filesystem/Numba disk caches were not purged.",
            "warm_recompute": "Same inputs without previous CalculationResult; legitimate internal gun/material caches remain.",
            "reuse": "Same inputs with the first completed CalculationResult as the reuse seed.",
            "nested": "Nested function timers overlap the pipeline stage timers; do not add them together.",
            "wall": "calculate call only, excluding imports, state construction, snapshot restore and GUI rendering.",
            "column_cache": "When the complete column product is reused, this record belongs to its original execution; inspect reused_products and this call's timers.",
        },
        "limits": ["Fixed declared numerical settings; no convergence qualification. Incomplete bounded calls are not full runtime measurements.",
                   "Selected stationary TEM assembly and requested EDS/filter products; no STEM raster.",
                   "No production function or user setting is replaced; wrappers add small unquantified timer overhead.",
                   "RSS is sampled every 20 ms and can miss brief peaks.",
                   "Not a cold OS, clean JIT compilation, GPU or coherent-wave benchmark."],
    }
    _save(options.output, receipt)
    deadline = perf_counter() + options.budget_seconds - 10.0
    process = psutil.Process()

    def measure(label, previous=None, edit=None):
        current = snapshot.restore()
        if edit is not None:
            lens = next(item for item in current.lenses if item.key == edit)
            lens.percent += .1
        current._tuning_cancelled = lambda: perf_counter() >= deadline
        nested = {}
        current_stage = ["not started"]
        def wrap(name, function):
            def measured(*args, **kwargs):
                start = perf_counter()
                try:
                    return function(*args, **kwargs)
                finally:
                    row = nested.setdefault(name, {"seconds": 0., "calls": 0})
                    row["seconds"] += perf_counter() - start
                    row["calls"] += 1
            return measured
        def progress(completed, total, text):
            current_stage[0] = text
            if perf_counter() >= deadline:
                raise TimeoutError("Bounded benchmark deadline reached")
        peak = [process.memory_info().rss]
        done = Event()
        start = perf_counter()
        def memory():
            last_receipt = start
            while not done.wait(.02):
                peak.append(process.memory_info().rss)
                if perf_counter() - last_receipt > 5.:
                    receipt["active_run"] = {"kind": label, "stage": current_stage[0],
                        "wall_s": perf_counter()-start, "rss_bytes": peak[-1],
                        "nested": {name: dict(row) for name, row in list(nested.items())}}
                    _save(options.output, receipt)
                    last_receipt = perf_counter()
        monitor = Thread(target=memory, daemon=True)
        monitor.start()
        result = None
        failure = None
        try:
            with ExitStack() as stack:
                for module, name, label_name in (
                    (source, "trace_source_to_exit", "gun_including_cache"),
                    (tracing, "_analytic_step", "analytic_step_including_retries"),
                    (simulation, "execute_propagation_plan", "incident_column_integrator"),
                    (simulation, "propagate", "post_reference_column_integrator"),
                    (pipeline, "run", "column_simulation_including_gun_and_diagnostics"),
                    (pipeline, "run_specimen_interactions", "specimen_interactions"),
                    (pipeline, "build_geometric_specimen_exit", "specimen_exit_downstream"),
                    (pipeline, "simulate_energy_filter", "energy_filter"),
                    (pipeline, "calculation_signatures", "calculation_signatures"),
                ):
                    stack.enter_context(patch.object(module, name, wrap(label_name, getattr(module, name))))
                result = pipeline.calculate(current, existing_result=previous, progress_callback=progress)
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
        finally:
            wall = perf_counter() - start
            done.set()
            monitor.join()
        row = {"kind": label, "wall_s": wall, "nested": nested,
               "last_stage": current_stage[0],
               "rss_before_bytes": peak[0], "sampled_peak_rss_bytes": max(peak),
               "rss_after_bytes": process.memory_info().rss}
        if edit is not None:
            row["changed_control"] = f"lenses[{edit}].percent += 0.1"
        if failure is not None:
            row.update(status="FAILED_OR_CANCELLED", error=failure)
        else:
            assert result.wave_imaging is None
            gun = result.simulation.gun_trace
            row.update(status="COMPLETE", performance=result.performance,
                calculated_products=sorted(result.calculated_products),
                reused_products=sorted(result.reused_products),
                column_cache=result.simulation.metrics.get("column_segment_cache"),
                gun_history_shape=list(gun.x_m.shape),
                incident_history_shape=list(result.simulation.incident.x.shape),
                gun_survivors=int(np.count_nonzero(gun.exit_bundle.alive)),
                specimen_survivors=int(np.count_nonzero(result.simulation.incident.alive)),
                column_array_digest=_array_digest(result.simulation),
                filter_summary={name: getattr(result.energy_filter, name, None) for name in (
                    "entrance_ray_count", "eels_ray_count", "slit_transmitted_fraction",
                    "eels_transmitted_fraction", "entrance_provenance")})
        receipt["runs"].append(row)
        receipt.pop("active_run", None)
        _save(options.output, receipt)
        print(json.dumps({"kind": label, "wall_s": wall, "status": row["status"],
                          "stages": None if result is None else result.performance["stages"]}), flush=True)
        if failure is not None:
            raise RuntimeError(failure)
        return result

    with numerical_job(options.threads):
        try:
            first = measure("cold")
            if not options.cold_only:
                second = measure("warm_recompute")
                assert _array_digest(first.simulation) == _array_digest(second.simulation)
                for index in range(options.reuse_repeats):
                    reused = measure(f"reuse_{index+1}", first)
                    assert _array_digest(first.simulation) == _array_digest(reused.simulation)
                measure("projector_edit", first, "projector_lens_2")
                measure("condenser_edit", first, "condenser_lens_2")
            receipt["status"] = "COMPLETE"
            receipt["same_input_column_arrays_exact"] = None if options.cold_only else True
        except Exception as error:
            receipt.update(status="FAILED_OR_CANCELLED", error=f"{type(error).__name__}: {error}")
    receipt["source_identity_after"] = capture_instrument_snapshot(state).implementation
    receipt["source_unchanged_during_measurement"] = receipt["source_identity_after"] == snapshot.implementation
    _save(options.output, receipt)
    return 0 if receipt["status"] == "COMPLETE" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rays", type=int, default=49)
    parser.add_argument("--step-mm", type=float, default=None,
                        help="Omit to use the selected assembly/state default")
    parser.add_argument("--history-step-mm", type=float, default=None)
    parser.add_argument("--energy-filter", choices=("default", "on", "off"), default="default")
    parser.add_argument("--cold-only", action="store_true")
    parser.add_argument("--threads", type=int, default=None,
                        help="Optional lower thread cap; default is half the available logical CPUs, BLAS stays serial")
    parser.add_argument("--reuse-repeats", type=int, default=3)
    parser.add_argument("--budget-seconds", type=float, default=180.)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    options = parser.parse_args()
    if (options.rays < 9 or (options.step_mm is not None and options.step_mm <= 0)
            or (options.history_step_mm is not None and options.history_step_mm <= 0)
            or (options.threads is not None and options.threads < 1) or options.budget_seconds < 30):
        parser.error("At least 9 rays, positive step/threads and a 30 second budget are required")
    if options.worker:
        return worker(options)
    if options.output.exists():
        parser.error("Output already exists; choose a new receipt path")
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"])
    try:
        return process.wait(timeout=options.budget_seconds)
    except subprocess.TimeoutExpired:
        # A Windows venv launcher can own a separate Python child. Terminate
        # only this benchmark's recorded process tree, including that child.
        import psutil
        try:
            descendants = psutil.Process(process.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            descendants = []
        for child in reversed(descendants):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        process.kill()
        process.wait()
        psutil.wait_procs(descendants, timeout=5)
        receipt = json.loads(options.output.read_text(encoding="utf-8")) if options.output.exists() else {}
        receipt.update(status="SUPERVISOR_TIMEOUT", budget_seconds=options.budget_seconds)
        _save(options.output, receipt)
        print("Only the bounded benchmark worker was stopped at its budget.", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
