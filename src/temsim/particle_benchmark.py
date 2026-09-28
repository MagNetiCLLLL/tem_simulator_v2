"""Receipts for the existing bounded particle benchmark, without model changes.

Array digests describe executed data; they never create a source or authorize
checkpoint reuse. Production dependency checks remain responsible for reuse.
"""
from collections import Counter
from contextlib import ExitStack, contextmanager
import hashlib
from importlib.metadata import PackageNotFoundError, version
import os
import platform
from statistics import median
from time import perf_counter
from unittest.mock import patch

import numpy as np


SOURCE_FIELDS = ("x_m", "y_m", "tx_rad", "ty_rad", "energy_offset_ev", "weight", "ray_id")
CHECKPOINT_FIELDS = ("z_mm", "x_m", "y_m", "tx_rad", "ty_rad", "flight_time_s")
# Declared before measuring. These are local restart-consistency guards, not
# convergence tolerances for the instrument. All ancestry/stops/energies and
# the selected axial plane remain exact. The position absolute tolerance is
# no larger than the existing particle-section restart regression (1e-14 m).
CONTINUATION_TOLERANCES = {
    "x_m": (1e-12, 1e-14), "y_m": (1e-12, 1e-14),
    "tx_rad": (1e-12, 1e-14), "ty_rad": (1e-12, 1e-14),
    "flight_time_s": (1e-12, 1e-18),
}


def array_receipt(values):
    array = np.ascontiguousarray(values)
    if array.dtype.hasobject:
        raise TypeError("Benchmark identities require plain arrays")
    digest = hashlib.sha256()
    digest.update(str((array.dtype.str, array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return {"sha256": digest.hexdigest(), "dtype": array.dtype.str,
            "shape": list(array.shape), "nbytes": int(array.nbytes)}


def array_inventory(arrays):
    return {name: array_receipt(value) for name, value in arrays.items()}


def source_arrays(gun, count):
    emitted = gun.emit(count)
    arrays = {name: np.asarray(getattr(emitted, name)) for name in SOURCE_FIELDS}
    from temsim.physics.ray_identity import emission_reference
    reference = emission_reference(emitted, getattr(gun.emitter, "surface_model", None))
    arrays.update({"launch_" + name: value for name, value in reference.items()})
    arrays["launch_energy_ev"] = np.asarray(getattr(emitted, "surface_energy_ev",
        float(gun.emitter.emission_energy_ev) + emitted.energy_offset_ev))
    return arrays


def gun_arrays(result):
    arrays = {name: getattr(result, name) for name in
              ("z_mm", "x_m", "y_m", "tx_rad", "ty_rad", "flight_time_s", "blocked_z_mm")}
    arrays["blocked_key"] = np.asarray(result.blocked_key, dtype=str)
    arrays.update({"exit_" + name: getattr(result.exit_bundle, name) for name in
                   (*SOURCE_FIELDS, "alive", "flight_time_s")})
    arrays.update({"launch_" + name: value for name, value in result.emission_reference.items()})
    return arrays


def compare_arrays(expected, actual):
    """Report exact identity and finite deltas, without inventing tolerances."""
    if set(expected) != set(actual):
        raise ValueError("Compared benchmark arrays have different fields")
    fields = {}
    for name, first in expected.items():
        left, right = np.asarray(first), np.asarray(actual[name])
        same_layout = left.shape == right.shape and left.dtype == right.dtype
        exact = same_layout and array_receipt(left) == array_receipt(right)
        delta = None
        if same_layout and left.dtype.kind in "fciub":
            finite = np.isfinite(left) & np.isfinite(right)
            if np.any(finite):
                if left.dtype.kind in "iub":
                    delta = max(abs(int(a)-int(b)) for a, b in zip(left[finite].flat, right[finite].flat))
                else:
                    delta = float(np.max(np.abs(left[finite].astype(np.complex128)
                                               - right[finite].astype(np.complex128))))
        fields[name] = {"exact": bool(exact), "same_shape_dtype": bool(same_layout),
                        "maximum_finite_absolute_difference": delta}
    return {"exact": all(row["exact"] for row in fields.values()), "fields": fields}


def require_exact_arrays(expected, actual, label):
    report = compare_arrays(expected, actual)
    if not report["exact"]:
        failed = [name for name, row in report["fields"].items() if not row["exact"]]
        raise AssertionError(f"{label} changed exact arrays: {', '.join(failed)}")
    return report


def gun_result_receipt(result):
    alive = np.asarray(result.exit_bundle.alive, dtype=bool)
    counts = Counter(key for key, survives in zip(result.blocked_key, alive, strict=True) if not survives)
    if "" in counts:
        counts["unlabelled_stop"] = counts.pop("")
    counts["gun_exit"] = int(np.count_nonzero(alive))
    return {"arrays": array_inventory(gun_arrays(result)), "particles": len(alive),
            "array_identity_scope": "Retained axial X/Y/slopes/TOF, exit phase space/IDs/weights/energy/alive, stop coordinates/labels and original launch reference. Equal-time histories and additional products are outside this digest.",
            "actual_endpoint_z_mm": float(result.z_mm[-1]),
            "stop_counts": dict(counts),
            "transmitted_weight": float(np.sum(result.exit_bundle.weight[alive])),
            "electrostatic_model_report": result.electrostatic_model_report,
            "integration_steps": None,
            "integration_steps_note": "The production GunTraceResult does not expose accepted/rejected step counts; retained history rows are not step counts.",
            "plane_arrivals": [{"key": plane.key, "z_mm": plane.z_mm,
                "reached": int(np.count_nonzero(plane.reached)),
                "transmitted": int(np.count_nonzero(plane.transmitted))} for plane in result.plane_arrivals]}


def section_terminal_arrays(result):
    segment = result.simulation.section_checkpoint.segments[-1]
    arrays = {name: np.asarray(getattr(segment.checkpoints, name))[-1] for name in CHECKPOINT_FIELDS}
    arrays.update({name: getattr(segment.branch, name) for name in
                  ("source_ray_id", "ray_weight", "energy_offset_ev", "alive", "blocked_z")})
    arrays["blocked_key"] = np.asarray(segment.branch.blocked_key, dtype=str)
    return arrays


def section_result_receipt(result):
    simulation = result.simulation
    checkpoint = simulation.section_checkpoint
    branch = checkpoint.segments[-1].branch
    stops = Counter(key for key, survives in zip(branch.blocked_key, branch.alive, strict=True) if not survives)
    if "" in stops:
        stops["unlabelled_stop"] = stops.pop("")
    stops["requested_plane"] = int(np.count_nonzero(branch.alive))
    return {"actual_endpoint_z_mm": float(branch.z[-1]),
            "terminal_arrays": array_inventory(section_terminal_arrays(result)),
            "gun_dependency_signature": checkpoint.gun_dependency_signature,
            "segment_dependencies": {segment.name: segment.initial_dependency_signature for segment in checkpoint.segments},
            "stop_counts": dict(stops), "population": len(branch.alive),
            "transmitted_weight": float(np.sum(branch.ray_weight[branch.alive])),
            "metrics": {name: value for name, value in simulation.metrics.items()
                        if name.startswith("section_") or name == "column_segment_cache"},
            "performance": result.performance,
            "calculated_products": sorted(result.calculated_products),
            "reused_products": sorted(result.reused_products)}


def compare_section_terminals(expected, actual):
    first, second = section_terminal_arrays(expected), section_terminal_arrays(actual)
    report = compare_arrays(first, second)
    for name, row in report["fields"].items():
        rtol, atol = CONTINUATION_TOLERANCES.get(name, (0., 0.))
        row.update(rtol=rtol, atol=atol, accepted=row["exact"])
        if name in CONTINUATION_TOLERANCES and row["same_shape_dtype"]:
            row["accepted"] = bool(np.allclose(first[name], second[name], rtol=rtol, atol=atol, equal_nan=True))
    report["accepted"] = all(row["accepted"] for row in report["fields"].values())
    report["scope"] = "Declared local checkpoint restart consistency; full-chain convergence is not established"
    return report


def require_exact_prefix(previous, continued):
    old, new = previous.simulation.section_checkpoint, continued.simulation.section_checkpoint
    if old.gun_dependency_signature != new.gun_dependency_signature:
        raise AssertionError("Continuation changed the executed gun dependency identity")
    require_exact_arrays(gun_arrays(old.gun_trace), gun_arrays(new.gun_trace), "Executed gun prefix")
    rows = []
    for segment in old.segments:
        after = next(item for item in new.segments if item.name == segment.name)
        z = np.asarray(after.checkpoints.z_mm)
        indices = np.searchsorted(z, segment.checkpoints.z_mm)
        if np.any(indices >= len(z)) or not np.array_equal(z[indices], segment.checkpoints.z_mm):
            raise AssertionError("Continuation omitted an executed checkpoint plane")
        before_values = {name: getattr(segment.checkpoints, name) for name in CHECKPOINT_FIELDS}
        after_values = {name: np.asarray(getattr(after.checkpoints, name))[indices] for name in CHECKPOINT_FIELDS}
        require_exact_arrays(before_values, after_values, "Retained column prefix")
        for name in ("source_ray_id", "ray_weight", "energy_offset_ev"):
            require_exact_arrays({name: getattr(segment.branch, name)},
                                 {name: getattr(after.branch, name)}, "Retained source")
        rows.append({"name": segment.name, "checkpoint_planes": len(indices),
                     "through_z_mm": float(segment.checkpoints.z_mm[-1]), "exact": True})
    return {"exact": True, "segments": rows}


def continuation_planes(state, span_mm=10.):
    from temsim.physics.particle_sections import section_limits
    lower, limit = section_limits(state)
    upper = min(float(state.sample.z_mm), limit)
    if not np.isfinite(span_mm) or span_mm <= 0 or not upper > lower:
        raise ValueError("A positive pre-specimen continuation interval is required")
    span = min(float(span_mm), (upper-lower)/3.)
    first, second = lower+span, lower+2*span
    if not lower < first < second < upper:
        raise ValueError("Cannot resolve distinct pre-specimen checkpoint planes")
    return first, second


def environment_receipt():
    from temsim.cpu_resources import available_cpu_count
    versions = {}
    for package in ("numpy", "scipy", "numba", "llvmlite", "psutil", "PySide6", "threadpoolctl"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    keys = ("TEMSIM_CPU_THREADS", "NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OMP_THREAD_LIMIT",
            "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_CACHE_DIR")
    return {"python": platform.python_version(), "platform": platform.platform(),
            "available_logical_cpus": available_cpu_count(), "versions": versions,
            "resource_environment": {key: os.environ.get(key) for key in keys}}


class OperationCounts:
    """Outer-call counters only; no per-step timing or modified field methods."""
    def __init__(self):
        self.values = Counter()
        self.column_calls = []

    def snapshot(self):
        return dict(self.values)

    def since(self, before):
        return {key: self.values[key]-before.get(key, 0) for key in sorted(self.values)}

    def wrap(self, name, function):
        def observed(*args, **kwargs):
            self.values[name] += 1
            started = perf_counter()
            before_traces = self.values["gun_transport_calls"]
            try:
                value = function(*args, **kwargs)
            except BaseException:
                self.values[name + "_failed"] += 1
                raise
            if name == "gun_request_calls" and self.values["gun_transport_calls"] == before_traces:
                self.values["gun_result_cache_hits"] += 1
            if name == "field_build_or_disk_load_calls":
                self.values["field_disk_cache_hits" if value.report.get("cache_hit") else "field_builds"] += 1
            if name == "column_transport_calls":
                plan = args[1]
                index = int(kwargs.get("start_index", 0))
                intervals = max(0, len(plan.z_mm)-1-index)
                self.values["requested_column_intervals"] += intervals
                self.values["column_zero_interval_calls"] += int(intervals == 0)
                self.column_calls.append({"start_index": index, "start_z_mm": float(plan.z_mm[index]),
                    "stop_z_mm": float(plan.z_mm[-1]), "requested_intervals": intervals,
                    "seconds": perf_counter()-started})
            return value
        return observed

    @contextmanager
    def observe(self):
        from temsim.optics.electron_gun import field_emission, source, tracing
        from temsim.physics import closed_gun_field, continuous_gun_field, particle_sections
        targets = ((source, "trace_source_to_exit", "gun_request_calls"),
                   (field_emission, "trace_feg_to_exit", "gun_transport_calls"),
                   (tracing, "trace_feg_to_exit", "gun_transport_calls"),
                   (closed_gun_field, "_build_request_field", "field_build_or_disk_load_calls"),
                   (continuous_gun_field, "build_continuous_gun_field", "field_build_or_disk_load_calls"),
                   (particle_sections, "execute_propagation_plan", "column_transport_calls"))
        with ExitStack() as stack:
            for module, name, label in targets:
                stack.enter_context(patch.object(module, name, self.wrap(label, getattr(module, name))))
            yield self


def counter_overhead_probe(repeats=2000):
    """One cheap timer/counter probe; no claim about kernel instrumentation."""
    counts = OperationCounts()
    empty = lambda: None
    wrapped = counts.wrap("probe", empty)
    rows = []
    for _ in range(3):
        start = perf_counter()
        for _ in range(repeats):
            empty()
        plain = perf_counter()-start
        start = perf_counter()
        for _ in range(repeats):
            wrapped()
        timed = perf_counter()-start
        rows.append({"plain_s": plain, "wrapped_s": timed})
    return {"calls_per_repeat": repeats, "repeats": rows,
            "median_added_seconds_per_call": median((r["wrapped_s"]-r["plain_s"])/repeats for r in rows),
            "scope": "Scalar outer-call counter probe; excludes production hashing, memory sampling and output acceptance."}
