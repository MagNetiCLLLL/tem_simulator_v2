"""Opt-in, bounded observations of existing diagnostic execution boundaries.

No field or trajectory is replaced. Timings nest: compiler materialization,
for example, is part of the first transport call. They must not be summed with
the request wall time. Fine-grained particle-step instrumentation is absent.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager, ExitStack
from functools import wraps
import math
import os
from threading import RLock
from time import perf_counter

from temsim.job_events import JobEvents, traced_job, job_stage


MAX_EVENTS = 2048
MAX_STAGES = 64
COUNTS = ("trajectory_executions", "field_solver_calls", "field_disk_load_hits",
          "field_disk_load_attempts", "field_accessor_calls", "field_memory_cache_hits", "compiler_calls",
          "compiler_cache_hits", "compiler_cache_misses", "compiler_new_signatures",
          "compiled_field_admissions", "compiled_transport_returns", "compiler_fallbacks",
          "exact_result_cache_hits", "serialized_bytes", "received_bytes", "progress_messages")
ACCOUNTING = ("Nested observations; do not sum stages or parent/child intervals with request_wall_s. "
              "Durations use perf_counter; JobEvents monotonic timestamps are ordering evidence only.")


class ExecutionPerformance:
    """Small request-local recorder backed by the existing JobEvents contract."""

    def __init__(self, sequence, request_kind, *, scope):
        self.sequence, self.request_kind, self.scope = sequence, request_kind, scope
        self.events = JobEvents(maximum=MAX_EVENTS)
        self.counts = Counter({key: 0 for key in COUNTS})
        self.stages = {}
        self._lock = RLock()
        self.started = perf_counter()
        self._compiler_depth = 0

    def increment(self, name, count=1):
        with self._lock:
            self.counts[name] += int(count)

    @contextmanager
    def stage(self, name):
        started = perf_counter()
        try:
            with traced_job(self.events, {"sequence": self.sequence, "scope": self.scope,
                                         "pid": os.getpid(), "request_kind": self.request_kind}):
                with job_stage(name, backend="isolated_cpu"):
                    yield
        finally:
            with self._lock:
                values = self.stages.setdefault(name, [0., 0])
                values[0] += max(0., perf_counter() - started)
                values[1] += 1

    def snapshot(self):
        with self._lock:
            counts = dict(self.counts)
            if counts["compiled_transport_returns"]:
                actual = "compiled_blocks"
            elif counts["trajectory_executions"]:
                actual = ("reference_with_compiled_fields"
                          if counts["compiled_field_admissions"] and not counts["compiler_fallbacks"]
                          else "reference")
            else:
                actual = "not_executed"
            return {"events": self.events.snapshot(), "counts": counts,
                    "stages": [{"stage": key, "scope": self.scope, "seconds": value[0],
                                "calls": value[1], "nested": True}
                               for key, value in self.stages.items()],
                    "actual_backend": actual,
                    "field_solver_coverage": "AxisymmetricCutField construction (flat/continuous gun FEM)",
                    "accounting": ACCOUNTING}

    @contextmanager
    def _patch(self, owner, name, wrapper):
        original = getattr(owner, name)
        setattr(owner, name, wrapper(original))
        try:
            yield
        finally:
            setattr(owner, name, original)

    def _counted(self, stage, counter, *, hit_counter=None):
        def decorate(original):
            @wraps(original)
            def observed(*args, **kwargs):
                self.increment(counter)
                with self.stage(stage):
                    result = original(*args, **kwargs)
                if hit_counter is not None and result is not None:
                    self.increment(hit_counter)
                return result
            return observed
        return decorate

    def _compile_wrapper(self, original):
        # Numba Dispatcher.compile first checks existing overloads, then loads
        # its disk cache, then compiles. Its native hit/miss Counters distinguish
        # those operations. Only outer compile intervals are accumulated; inner
        # signature calls are still counted, without double-counting wall time.
        @wraps(original)
        def observed(dispatcher, signature):
            self.increment("compiler_calls")
            before = (sum(dispatcher._cache_hits.values()), sum(dispatcher._cache_misses.values()),
                      len(dispatcher.overloads))
            outer = self._compiler_depth == 0
            self._compiler_depth += 1
            try:
                if outer:
                    with self.stage("compiler_warmup"):
                        return original(dispatcher, signature)
                return original(dispatcher, signature)
            finally:
                self._compiler_depth -= 1
                after = (sum(dispatcher._cache_hits.values()), sum(dispatcher._cache_misses.values()),
                         len(dispatcher.overloads))
                for key, old, new in zip(("compiler_cache_hits", "compiler_cache_misses", "compiler_new_signatures"), before, after):
                    self.increment(key, max(0, new-old))
        return observed

    def _accessor_wrapper(self, original):
        @wraps(original)
        def observed(*args, **kwargs):
            self.increment("field_accessor_calls")
            before = (self.counts["field_solver_calls"], self.counts["field_disk_load_hits"])
            with self.stage("field_access"):
                result = original(*args, **kwargs)
            after = (self.counts["field_solver_calls"], self.counts["field_disk_load_hits"])
            if result is not None and before == after:
                self.increment("field_memory_cache_hits")
            return result
        return observed

    @contextmanager
    def observe(self):
        """Temporarily observe only the serial child's preparation/trace call."""
        with ExitStack() as patches:
            if self.request_kind == "prepare":
                from temsim.physics import closed_gun_field as closed
                from temsim.physics import continuous_gun_field as continuous
                from temsim.physics.axisymmetric_cut_field import AxisymmetricCutField
                patches.enter_context(self._patch(AxisymmetricCutField, "__init__",
                    self._counted("field_solve", "field_solver_calls")))
                for owner, name in ((closed, "closed_field"), (continuous, "continuous_field")):
                    patches.enter_context(self._patch(owner, name, self._accessor_wrapper))
                for owner, name in ((closed, "load_cached_field"), (continuous, "_load_cached")):
                    patches.enter_context(self._patch(owner, name,
                        self._counted("field_disk_load", "field_disk_load_attempts",
                                      hit_counter="field_disk_load_hits")))
            if self.request_kind in ("prepare", "trace"):
                try:
                    from numba.core.dispatcher import Dispatcher
                except ImportError:
                    pass
                else:
                    patches.enter_context(self._patch(Dispatcher, "compile", self._compile_wrapper))
            if self.request_kind == "trace":
                from temsim import magnetic_test_particle as particle
                from temsim import test_electron_compiled as compiled
                patches.enter_context(self._patch(compiled, "prepare_compiled_fields",
                    self._counted("compiled_field_preparation", "compiled_field_preparation_calls",
                                  hit_counter="compiled_field_admissions")))
                def completed(original):
                    @wraps(original)
                    def observe(*args, **kwargs):
                        result = original(*args, **kwargs)
                        self.increment("compiled_transport_returns")
                        return result
                    return observe
                patches.enter_context(self._patch(particle, "_trace_compiled_electromagnetic_electron", completed))
                def fallback(original):
                    @wraps(original)
                    def observe(*args, **kwargs):
                        result = original(*args, **kwargs)
                        if result:
                            self.increment("compiler_fallbacks")
                        return result
                    return observe
                patches.enter_context(self._patch(particle._Sampler, "compilation_fallback", fallback))
            yield


def validate_performance_payload(value):
    """Admit bounded scalar evidence only; never arbitrary model/array graphs."""
    if not isinstance(value, dict) or set(value) != {
            "events", "counts", "stages", "actual_backend", "field_solver_coverage", "accounting"}:
        raise ValueError("Malformed diagnostic performance payload")
    for key, maximum in (("events", MAX_EVENTS), ("stages", MAX_STAGES)):
        rows = value[key]
        if not isinstance(rows, list) or len(rows) > maximum:
            raise ValueError("Diagnostic performance row limit exceeded")
        for row in rows:
            if not isinstance(row, dict) or len(row) > 16:
                raise ValueError("Malformed diagnostic performance row")
            for name, item in row.items():
                if (not isinstance(name, str) or len(name) > 128
                        or type(item) not in (str, int, float, bool, type(None))
                        or isinstance(item, str) and len(item) > 1024
                        or isinstance(item, float) and not math.isfinite(item)):
                    raise ValueError("Diagnostic performance evidence must be finite bounded scalars")
            if key == "stages" and (set(row) != {"stage", "scope", "seconds", "calls", "nested"}
                    or type(row["seconds"]) not in (float, int) or row["seconds"] < 0
                    or type(row["calls"]) is not int or row["calls"] < 0):
                raise ValueError("Malformed diagnostic performance stage")
    counts = value["counts"]
    if not isinstance(counts, dict) or len(counts) > 64:
        raise ValueError("Malformed diagnostic performance counters")
    if any(not isinstance(key, str) or len(key) > 128 or type(count) is not int or not 0 <= count <= 2**63-1
           for key, count in counts.items()):
        raise ValueError("Invalid diagnostic performance counter")
    if value["actual_backend"] not in ("not_executed", "reference", "reference_with_compiled_fields", "compiled_blocks"):
        raise ValueError("Unknown measured diagnostic backend")
    if any(not isinstance(value[key], str) or len(value[key]) > 1024 for key in ("field_solver_coverage", "accounting")):
        raise ValueError("Malformed diagnostic performance description")
    return value
