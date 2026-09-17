"""Scalar execution receipts and inexpensive capability preflight, not physics."""
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from time import monotonic

from temsim.job_events import job_event
from temsim.physics.compute_backend import BACKEND_REQUIRE_GPU, GPUExecutionError, normalise_backend

_RECEIPTS = ContextVar("backend_execution_receipts", default=None)


@contextmanager
def backend_receipts(rows):
    token = _RECEIPTS.set(rows)
    try:
        yield rows
    finally:
        _RECEIPTS.reset(token)


def record_backend(stage, requested, actual, *, outcome="returned", reason="", **details):
    row = dict(stage=str(stage), requested=str(requested), actual=str(actual),
               outcome=outcome, reason=str(reason), monotonic_s=monotonic(), **details)
    job_event("backend_stage", **row)
    rows = _RECEIPTS.get()
    if rows is not None:
        rows.append(row)


def capture_worker_backends(function):
    @wraps(function)
    def wrapped(worker, *args, **kwargs):
        # Bounded call receipts, not arrays. Repeated scan/diagnostic calls may
        # evict old entries; the result explicitly states this limit.
        worker.backend_evidence = deque(maxlen=256)
        with backend_receipts(worker.backend_evidence):
            return function(worker, *args, **kwargs)
    return wrapped


def cpu_call(stage):
    """For CPU implementations only; internal exact cache reuse is permitted."""
    def decorate(function):
        @wraps(function)
        def wrapped(state, *args, **kwargs):
            outcome = "raised"
            try:
                result = function(state, *args, **kwargs)
                outcome = "returned"
                return result
            finally:
                record_backend(stage, getattr(state, "acceleration_backend", "Auto"), "CPU",
                    outcome=outcome, reason="CPU implementation; may reuse exact upstream data")
        return wrapped
    return decorate


def classical_backend_preflight(state):
    """No field construction, device probing, source launch or state mutation."""
    from temsim.simulation_modes import uses_field_maps
    requested = getattr(state, "acceleration_backend", "Auto")
    keys = set(getattr(state, "lens_field_map_descriptors", {}) or {})
    keys.update(getattr(state, "_lens_field_map_bindings", {}) or {})
    mapped = uses_field_maps(state) and any(
        lens.key in keys and bool(getattr(lens, "enabled", True)) for lens in getattr(state, "lenses", ()))
    vacuum = bool(getattr(getattr(state, "vacuum_map", None), "enabled", False))
    reason = "Vector-field transport is CPU-only" if mapped else "Residual-medium transport is CPU-only" if vacuum else ""
    rows = [dict(stage="gun", requested=requested, capability="CPU", reason="Physical emitter-to-exit transport; exact cache reuse allowed"),
            dict(stage="column", requested=requested, capability="CPU" if reason else "CPU / Numba CPU / CUDA GPU",
                 reason=reason or "Backend selected at execution; CUDA does not accelerate every stage")]
    if bool(getattr(getattr(state, "energy_filter", None), "enabled", False)):
        rows.append(dict(stage="energy_filter", requested=requested, capability="CPU", reason="Existing Boris/NumPy filter implementation"))
    rows.append(dict(stage="specimen", requested=requested, capability="CPU particle interactions",
                     reason="Coherent imaging remains separately source-gated; no wave qualification"))
    if normalise_backend(requested) == BACKEND_REQUIRE_GPU and reason:
        raise GPUExecutionError("unsupported_stage", reason + "; Require GPU cannot execute this column on CPU")
    return tuple(rows)
