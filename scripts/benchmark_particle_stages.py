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
import math
import os
from pathlib import Path
import platform
from statistics import median
import sys
from threading import Event, Thread, Timer
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
        for name in ("z", "x", "y", "tx", "ty", "alive", "flight_time_s",
                     "blocked_z", "blocked_key", "energy_offset_ev", "ray_weight",
                     "source_ray_id", "source_azimuth_rad"):
            value = np.asarray(getattr(branch, name))
            digest.update(str((name, value.dtype.str, value.shape)).encode())
            digest.update(value.tobytes())
    from temsim.particle_benchmark import gun_arrays
    for name, value in gun_arrays(simulation.gun_trace).items():
        value = np.ascontiguousarray(value)
        digest.update(str((name, value.dtype.str, value.shape)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def worker(options):
    if options.scope != "full":
        return scoped_worker(options)
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
    from temsim.particle_benchmark import array_inventory, environment_receipt, source_arrays
    artifacts = options.output.with_suffix(".artifacts")
    artifacts.mkdir(parents=True, exist_ok=False)
    _save(artifacts / "inputs.json", snapshot.to_dict())
    fixed_source = source_arrays(state.electron_gun, options.rays)
    np.savez(artifacts / "tip-source.npz", **fixed_source)
    receipt = {
        "schema": "particle-stage-benchmark-v1", "status": "RUNNING",
        "host": {"platform": platform.platform(), "python": platform.python_version(),
                 "processor": platform.processor(), "logical_cpus": os.cpu_count(),
                 "physical_ram_bytes": psutil.virtual_memory().total,
                 "numba_threads": get_num_threads(), "numerical_thread_budget": options.threads,
                 "library_threads": 1},
        "input_identity": snapshot.digest, "implementation_identity": snapshot.implementation,
        "environment": environment_receipt(),
        "input_snapshot": str((artifacts / "inputs.json").resolve()),
        "source_array_file": str((artifacts / "tip-source.npz").resolve()),
        "source_arrays": array_inventory(fixed_source),
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
    cancelled = Event()
    cancellation_timer = Timer(max(.01, deadline-perf_counter()), cancelled.set)
    cancellation_timer.daemon = True
    cancellation_timer.start()
    process = psutil.Process()

    def measure(label, previous=None, edit=None):
        current = snapshot.restore()
        if edit is not None:
            lens = next(item for item in current.lenses if item.key == edit)
            lens.percent += .1
        current._tuning_cancelled = cancelled.is_set
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
                for index in range(options.warm_repeats):
                    second = measure(f"warm_recompute_{index+1}")
                    assert _array_digest(first.simulation) == _array_digest(second.simulation)
                for index in range(options.reuse_repeats):
                    reused = measure(f"reuse_{index+1}", first)
                    assert _array_digest(first.simulation) == _array_digest(reused.simulation)
                measure("projector_edit", first, "projector_lens_2")
                measure("condenser_edit", first, "condenser_lens_2")
            receipt["status"] = "COMPLETE"
            receipt["same_input_column_arrays_exact"] = None if options.cold_only else True
        except Exception as error:
            receipt.update(status="NOT_COMPLETE", error=f"{type(error).__name__}: {error}")
        finally:
            cancellation_timer.cancel()
    receipt["implementation_identity_after"] = capture_instrument_snapshot(state).implementation
    receipt["implementation_unchanged_during_measurement"] = receipt["implementation_identity_after"] == snapshot.implementation
    receipt["medians_s"] = {prefix: median(row["wall_s"] for row in receipt["runs"]
        if row["kind"].startswith(prefix) and row["status"] == "COMPLETE")
        for prefix in ("warm_recompute_", "reuse_")
        if any(row["kind"].startswith(prefix) and row["status"] == "COMPLETE" for row in receipt["runs"])}
    if not receipt["implementation_unchanged_during_measurement"]:
        receipt.update(status="NOT_COMPLETE", error="Implementation changed during measurement")
    _save(options.output, receipt)
    return 0 if receipt["status"] == "COMPLETE" else 1


def scoped_worker(options):
    """B05/B07 use the same process budget and production entry points."""
    worker_started = perf_counter()
    from temsim.cpu_resources import initialize_cpu_resources
    initialize_cpu_resources()
    import numpy as np
    import psutil
    from numba import get_num_threads
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.cpu_resources import numerical_job, numerical_thread_budget
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.job_events import JobEvents, traced_job, job_stage
    from temsim.optics.column import default_state
    from temsim.optics.electron_gun import source, tracing
    from temsim.particle_benchmark import (
        CONTINUATION_TOLERANCES, OperationCounts, array_inventory, compare_section_terminals, continuation_planes,
        counter_overhead_probe, environment_receipt, gun_arrays, gun_result_receipt,
        require_exact_arrays, require_exact_prefix, section_result_receipt,
        section_terminal_arrays, source_arrays,
    )
    from temsim.particle_section_io import (
        _result_records, _same_executed_record, load_section_result, save_section_result,
    )
    from temsim.simulation_pipeline import calculate_particle_section
    from temsim.column.state_layout import apply_physical_layout_to_state
    from temsim.vacuum import bind_gun_environment

    artifacts = options.output.with_suffix(".artifacts")
    artifacts.mkdir(parents=True, exist_ok=False)
    state = default_state()
    catalog = AssemblyCatalog()
    selection = catalog.default_selection()
    if options.energy_filter != "default":
        installed = options.energy_filter == "on"
        selection = replace(selection, recording=next(item.name for item in catalog.recording_systems
            if bool(item.properties.get("energy_filter")) == installed))
    catalog.apply(state, selection)
    state.electron_gun.emitter.ray_count = options.rays
    state.acceleration_backend = "CPU"
    if options.step_mm is not None:
        state.step_mm = options.step_mm
    if options.history_step_mm is not None:
        state.history_step_mm = options.history_step_mm
    assert not state.sample.wave_enabled and not state.sample.stem_wave_enabled
    assert not state.ac_deflector.scan_enabled and not state.descan_deflector.scan_enabled
    assert state.electron_gun.emitter.surface_model is None
    assert state.electron_gun.emitter.coherence is None
    apply_physical_layout_to_state(state)
    bind_gun_environment(state)
    snapshot = capture_instrument_snapshot(state)
    _save(artifacts / "inputs.json", snapshot.to_dict())
    fixed_source = source_arrays(state.electron_gun, options.rays)
    np.savez(artifacts / "tip-source.npz", **fixed_source)
    cancelled = Event()
    state._tuning_cancelled = cancelled.is_set
    timer = Timer(max(.01, options.budget_seconds-10.-(perf_counter()-worker_started)), cancelled.set)
    timer.daemon = True
    timer.start()
    process, events, counts = psutil.Process(), JobEvents(), OperationCounts()
    receipt = {"schema": "particle-stage-benchmark-v2", "case": "B05" if options.scope == "gun" else "B07",
        "status": "RUNNING", "scope": options.scope, "environment": environment_receipt(),
        "worker_import_and_input_preparation_s": perf_counter()-worker_started,
        "input_identity": snapshot.digest, "implementation_identity": snapshot.implementation,
        "input_snapshot": str((artifacts / "inputs.json").resolve()),
        "source_array_file": str((artifacts / "tip-source.npz").resolve()),
        "source_arrays": array_inventory(fixed_source),
        "source_sampling": {"method": "deterministic existing tip emission; Halton dimensions (2,3,5,7,11,13) when no product quadrature",
            "random_seed": None, "random_seed_note": "Unrandomised emitter sequence; no invented RNG seed",
            "vacuum_seed": getattr(state.electron_gun, "_vacuum_seed", 914),
            "vacuum_enabled": bool(state.vacuum_map.enabled)},
        "settings": {"rays": options.rays, "warm_repeats": options.warm_repeats,
            "reuse_repeats": options.reuse_repeats, "cold_only": options.cold_only,
            "budget_seconds": options.budget_seconds,
            "gun_trace_step_mm": state.electron_gun.trace_step_mm,
            "gun_history_step_mm": state.electron_gun.history_step_mm,
            "column_step_mm": state.step_mm, "column_history_step_mm": state.history_step_mm},
        "timing_semantics": {
            "cold": "First requested production transport in a fresh worker; OS, field disk and Numba disk caches are not purged.",
            "warm_transport": "Uncached production gun execution with the same tip arrays and warm field; includes emission, physical stops and history assembly.",
            "exact_reuse": "Production dependency-checked reuse; execution counts establish whether transport actually ran.",
            "compiler_warmup": "First transport includes first-use compiler/cache-loading overhead; compiler time alone is not isolated.",
            "acceptance": "Array identities, errors and prefix checks are timed after each operation, separately from its execution.",
            "counts": "Wrappers observe outer calls only. No step or field-provider method is wrapped; requested column intervals are not adaptive gun steps.",
            "nested": "Stage and enclosing operation times overlap and must not be summed.",
        },
        "limits": ["Performance and exact-reuse evidence, not full-column numerical convergence or microscope qualification.",
                   "RSS sampled every 20 ms; brief peaks can be missed.",
                   "Field report cache_hit describes its original construction/load; current counters identify actual work.",
                   "No trajectory simplification, physics bypass, coherent source or downstream source is introduced."],
        "counter_overhead_probe": counter_overhead_probe(), "runs": []}
    _save(options.output, receipt)

    def measure(label, phase, operation, inspect=None):
        if cancelled.is_set():
            raise TimeoutError("Particle benchmark budget reached before starting another operation")
        before, call_index = counts.snapshot(), len(counts.column_calls)
        rss = process.memory_info().rss
        peak, done = [rss], Event()
        enclosing_started = perf_counter()
        receipt["active_run"] = {"kind": label, "phase": phase, "wall_s": 0.}
        _save(options.output, receipt)
        def sample_memory():
            last = enclosing_started
            while not done.wait(.02):
                peak[0] = max(peak[0], process.memory_info().rss)
                if perf_counter()-last >= 5.:
                    receipt["active_run"].update(enclosing_wall_s=perf_counter()-enclosing_started, sampled_peak_rss_bytes=peak[0])
                    _save(options.output, receipt)
                    last = perf_counter()
        monitor = Thread(target=sample_memory, daemon=True)
        monitor.start()
        value, error = None, None
        started = perf_counter()
        try:
            with traced_job(events, {"kind": label}), job_stage(phase, backend="CPU"):
                value = operation()
        except Exception as caught:
            error = caught
        finally:
            duration = perf_counter()-started
            done.set()
            monitor.join()
        row = {"kind": label, "phase": phase, "wall_s": duration,
            "enclosing_wall_s": perf_counter()-enclosing_started,
            "counts": counts.since(before), "column_calls": counts.column_calls[call_index:],
            "rss_before_bytes": rss, "sampled_peak_rss_bytes": peak[0],
            "rss_after_bytes": process.memory_info().rss, "status": "COMPLETE" if error is None else "NOT_COMPLETE"}
        acceptance_started = perf_counter()
        if error is None:
            try:
                row["result"] = inspect(value, row) if inspect is not None else {}
                row["accepted"] = True
            except Exception as caught:
                error = caught
        row["acceptance_s"] = perf_counter()-acceptance_started
        if error is not None:
            row.update(status="NOT_COMPLETE", accepted=False, error=f"{type(error).__name__}: {error}")
        receipt["runs"].append(row)
        receipt.pop("active_run", None)
        receipt["events"] = events.snapshot()
        _save(options.output, receipt)
        print(json.dumps({"kind": label, "wall_s": duration, "status": row["status"], "counts": row["counts"]}), flush=True)
        if error is not None:
            raise error
        return value

    def check_source(gun):
        require_exact_arrays(fixed_source, source_arrays(gun, options.rays), "Original tip emission")

    def inspect_gun(value, row, expected=None, execution_count=1):
        check_source(state.electron_gun)
        if row["counts"].get("gun_transport_calls", 0) != execution_count:
            raise AssertionError("Gun timing did not execute the declared transport/cache path")
        detail = gun_result_receipt(value)
        require_exact_arrays({"ray_id": fixed_source["ray_id"], "weight": fixed_source["weight"]},
            {"ray_id": value.exit_bundle.ray_id, "weight": value.exit_bundle.weight}, "Gun population identity")
        if expected is not None:
            detail["comparison"] = require_exact_arrays(gun_arrays(expected), gun_arrays(value), "Repeated gun result")
        return detail

    def fresh_state():
        current = snapshot.restore()
        current._tuning_cancelled = cancelled.is_set
        return current

    def inspect_section(value, row, previous=None, expected=None, zero_intervals=False):
        check_source(value.state_snapshot.electron_gun)
        detail = section_result_receipt(value)
        if previous is not None:
            if row["counts"].get("gun_request_calls", 0) or row["counts"].get("gun_transport_calls", 0):
                raise AssertionError("Continuation recalculated the executed gun")
            if not value.simulation.metrics["section_gun_reused"] or not value.simulation.metrics["section_reused_prefix"]:
                raise AssertionError("Continuation did not retain its checked upstream state")
            detail["prefix"] = require_exact_prefix(previous, value)
            if not row["column_calls"]:
                raise AssertionError("No production column call was observed")
            endpoint = previous.simulation.section_checkpoint.segments[-1].checkpoints.z_mm[-1]
            if row["column_calls"][-1]["start_z_mm"] != endpoint:
                raise AssertionError("Production integrator did not restart at the saved exact plane")
        if expected is not None:
            detail["comparison"] = require_exact_arrays(section_terminal_arrays(expected),
                section_terminal_arrays(value), "Same executed continuation")
        if zero_intervals and row["counts"].get("requested_column_intervals", 0) != 0:
            raise AssertionError("Exact final-plane reuse advanced new column intervals")
        return detail

    try:
        with numerical_job(options.threads) as cpu, counts.observe():
            receipt["cpu_resources"] = cpu.to_dict()
            receipt["actual_numba_threads"] = get_num_threads()
            if options.scope == "gun":
                field = measure("field_prepare", "field_prepare_or_load", lambda: state.electron_gun.electric_field,
                    lambda value, row: {"field_report": getattr(getattr(value, "base_field", value), "report", None),
                                       "request": getattr(getattr(value, "base_field", value), "request", None)})
                first = measure("cold_transport", "cold_transport", lambda: source.trace_source_to_exit(state), inspect_gun)
                if not options.cold_only:
                    for index in range(options.warm_repeats):
                        measure(f"warm_transport_{index+1}", "warm_transport",
                            lambda: tracing.trace_feg_to_exit(state.electron_gun, cancelled=cancelled.is_set),
                            lambda value, row: inspect_gun(value, row, first))
                    for index in range(options.reuse_repeats):
                        measure(f"exact_reuse_{index+1}", "exact_reuse", lambda: source.trace_source_to_exit(state),
                            lambda value, row: inspect_gun(value, row, first, 0))
                receipt["same_field_object_after"] = state.electron_gun.electric_field is field
            else:
                if state.vacuum_map.enabled:
                    raise ValueError("This exact-continuation benchmark requires its declared default vacuum-scattering-off state")
                first_z, second_z = continuation_planes(state, options.section_span_mm)
                receipt["observation_planes_mm"] = [first_z, second_z]
                receipt["direct_continuation_tolerances"] = CONTINUATION_TOLERANCES
                receipt["limits"].append("B07 stops before the specimen; it does not benchmark material, EDS, detector or energy-filter continuation.")
                initial = measure("cold_prefix", "cold_transport",
                    lambda: calculate_particle_section(fresh_state(), first_z, ()), inspect_section)

                def save_and_load(label):
                    path = artifacts / f"{label}.temresult"
                    temperature = "cold" if label == "cold" else "warm"
                    package = measure(label + "_save", temperature + "_save", lambda: save_section_result(initial, path),
                        lambda value, row: {"path": str(path.resolve()), "package_digest": value.digest,
                                           "archive_bytes": path.stat().st_size, "unique_arrays": len(value.arrays)})
                    def inspect_load(value, row):
                        if not _same_executed_record(_result_records(initial), _result_records(value)):
                            raise AssertionError("Lossless package changed an executed result field")
                        check_source(value.state_snapshot.electron_gun)
                        value.state_snapshot._tuning_cancelled = cancelled.is_set
                        return {"archive_info": value.section_archive_info,
                                "all_retained_records_exact": True,
                                "prefix": require_exact_prefix(initial, value)}
                    return measure(label + "_load", temperature + "_load",
                        lambda: load_section_result(path, expected_package_digest=package.digest), inspect_load)

                loaded = save_and_load("cold")
                baseline = measure("cold_continuation", "cold_continuation",
                    lambda: calculate_particle_section(loaded.state_snapshot, second_z, (), existing_result=loaded),
                    lambda value, row: inspect_section(value, row, loaded))

                def inspect_direct(value, row):
                    detail = inspect_section(value, row)
                    if value.simulation.metrics["section_reused_prefix"]:
                        raise AssertionError("The direct reference unexpectedly resumed a column prefix")
                    if not row["column_calls"] or any(call["start_index"] != 0 for call in row["column_calls"]):
                        raise AssertionError("The direct reference did not execute from the gun exit")
                    # The ordinary gun cache remains legitimate upstream reuse.
                    # Counts distinguish this from executing the gun again;
                    # no state is injected and no production cache is deleted.
                    row["direct_reference_gun_transport_calls"] = row["counts"].get("gun_transport_calls", 0)
                    row["direct_reference_gun_cache_hits"] = row["counts"].get("gun_result_cache_hits", 0)
                    comparison = compare_section_terminals(baseline, value)
                    row["direct_reference_comparison"] = comparison
                    if not comparison["accepted"]:
                        raise AssertionError("Direct-column/continued endpoint exceeded the declared restart consistency guard")
                    detail["comparison"] = comparison
                    return detail

                measure("direct_column_reference", "direct_column_reference",
                    lambda: calculate_particle_section(fresh_state(), second_z, ()), inspect_direct)
                if not options.cold_only:
                    for index in range(options.warm_repeats):
                        loaded = save_and_load(f"warm_{index+1}")
                        measure(f"warm_continuation_{index+1}", "warm_continuation",
                            lambda: calculate_particle_section(loaded.state_snapshot, second_z, (), existing_result=loaded),
                            lambda value, row: inspect_section(value, row, loaded, baseline))
                    for index in range(options.reuse_repeats):
                        measure(f"exact_reuse_{index+1}", "exact_reuse",
                            lambda: calculate_particle_section(fresh_state(), second_z, (), existing_result=baseline),
                            lambda value, row: inspect_section(value, row, baseline, baseline, True))
            check_source(state.electron_gun)
            receipt["status"] = "COMPLETE"
    except Exception as error:
        receipt.update(status="NOT_COMPLETE", error=f"{type(error).__name__}: {error}")
    finally:
        timer.cancel()
    receipt["totals"] = counts.snapshot()
    receipt["medians_s"] = {phase: median(row["wall_s"] for row in receipt["runs"]
        if row["phase"] == phase and row["status"] == "COMPLETE")
        for phase in {row["phase"] for row in receipt["runs"] if row["status"] == "COMPLETE"}}
    receipt["implementation_identity_after"] = capture_instrument_snapshot(state).implementation
    receipt["implementation_unchanged_during_measurement"] = receipt["implementation_identity_after"] == snapshot.implementation
    if not receipt["implementation_unchanged_during_measurement"]:
        receipt.update(status="NOT_COMPLETE", error="Implementation changed during measurement")
    _save(options.output, receipt)
    return 0 if receipt["status"] == "COMPLETE" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=("gun", "continuation", "full"), default="gun",
                        help="B05 tip-to-gun-exit, B07 lossless save/load/resume, or the explicit full existing pipeline")
    parser.add_argument("--rays", type=int, default=49)
    parser.add_argument("--step-mm", type=float, default=None,
                        help="Omit to use the selected assembly/state default")
    parser.add_argument("--history-step-mm", type=float, default=None)
    parser.add_argument("--energy-filter", choices=("default", "on", "off"), default="default")
    parser.add_argument("--cold-only", action="store_true")
    parser.add_argument("--warm-repeats", type=int, default=3)
    parser.add_argument("--section-span-mm", type=float, default=10., help="B07 first and second cutoffs are one/two spans after the gun exit, clipped before the specimen")
    parser.add_argument("--threads", type=int, default=None,
                        help="Optional lower thread cap; default is half the available logical CPUs, BLAS stays serial")
    parser.add_argument("--reuse-repeats", type=int, default=3)
    parser.add_argument("--budget-seconds", type=float, default=180.)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    options = parser.parse_args()
    if (options.rays < 9 or (options.step_mm is not None and (not math.isfinite(options.step_mm) or options.step_mm <= 0))
            or (options.history_step_mm is not None and (not math.isfinite(options.history_step_mm) or options.history_step_mm <= 0))
            or (options.threads is not None and options.threads < 1)
            or not math.isfinite(options.budget_seconds) or options.budget_seconds < 30
            or options.warm_repeats < 3 or options.reuse_repeats < 3
            or not math.isfinite(options.section_span_mm) or options.section_span_mm <= 0):
        parser.error("At least 9 rays, 3 warm/reuse repeats, positive steps/threads/span and a 30 second budget are required")
    if options.output.exists() or options.output.with_suffix(".artifacts").exists():
        parser.error("Output or artifact directory already exists; choose a new receipt path")
    if options.worker:
        return worker(options)
    from temsim.validation_process import run_bounded, ValidationCleanupError
    started = perf_counter()
    supervisor_error = None
    try:
        result = run_bounded([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"],
                             timeout=options.budget_seconds)
        returncode = result.returncode
    except ValidationCleanupError as error:
        returncode, supervisor_error = 125, str(error)
    receipt = json.loads(options.output.read_text(encoding="utf-8")) if options.output.exists() else {}
    receipt["supervisor"] = {"wall_s": perf_counter()-started, "returncode": returncode,
        "budget_seconds": options.budget_seconds,
        "scope": "Includes worker launch, imports, preparation, transport, acceptance and report writes"}
    if returncode != 0 or receipt.get("status") != "COMPLETE":
        receipt.update(status="NOT_COMPLETE", supervisor_reason=supervisor_error or (
            "Owned worker timed out; owned-tree termination confirmed" if returncode == 124 else
            "Worker failed or did not finish the requested benchmark scope"))
    _save(options.output, receipt)
    return returncode if returncode != 0 else (0 if receipt["status"] == "COMPLETE" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
