"""Optional double-precision device execution for the tip-origin wave chain.

The device is an execution choice, never another source or numerical model.
Public checkpoints remain host arrays. A GPU job owns its allocation pool and
executes one mode at a time; only unavailable hardware or exhausted memory may
fall back, and a failed operator must be rerun from its unchanged input.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import RLock

import numpy as np

from temsim.physics.compute_backend import (
    GPUExecutionError, WAVE_BACKEND_CUPY, choose_wave_backend, cupy_module,
    gpu_retry_reason, normalise_backend as _normalise_backend,
)


_ACTIVE = ContextVar("coherent_wave_device", default=None)
_DEVICE_LOCK = RLock()
_RETIRED_POOLS = []
PRECISION = "complex128 / float64"


def normalise_backend(value):
    from temsim.physics.compute_backend import BACKEND_CHOICES
    aliases = {"cpu": "CPU", "auto": "Auto", "gpu": "CUDA GPU", "cuda": "CUDA GPU",
               "cupy": "CUDA GPU", "numpy": "CPU", "numba": "Numba CPU",
               "prefer_gpu": "Prefer GPU", "require_gpu": "Require GPU",
               "numpy cpu": "CPU", "cupy cuda": "CUDA GPU"}
    aliases.update({name.lower(): name for name in BACKEND_CHOICES})
    name = str(value).strip().lower()
    if name not in aliases:
        raise ValueError(f"Unknown coherent compute policy: {value!r}")
    return _normalise_backend(aliases[name])


def array_module(value):
    if hasattr(value, "__cuda_array_interface__"):
        import cupy
        return cupy
    return np


def to_host(value):
    xp = array_module(value)
    return np.asarray(value) if xp is np else xp.asnumpy(value)


def max_abs(value):
    return float(array_module(value).max(abs(value))) if value.size else 0.


def release_device_memory():
    """Drain completed pools after caller-owned GPU temporaries have expired.

    CuPy allocations reference their pool weakly. In particular, a split
    block cannot be recovered if its pool dies before all its arrays. Retain
    pools with live arrays and release only their unused blocks; live device
    outputs stay valid. Call again at the public job boundary.
    """
    with _DEVICE_LOCK:
        retained = []
        for cp, device_id, pool in _RETIRED_POOLS:
            with cp.cuda.Device(device_id):
                pool.free_all_blocks()
                if pool.used_bytes():
                    retained.append((cp, device_id, pool))
        _RETIRED_POOLS[:] = retained


def _retire_pool(cp, pool):
    pool.free_all_blocks()
    if pool.used_bytes():
        _RETIRED_POOLS.append((cp, int(cp.cuda.runtime.getDevice()), pool))


@dataclass
class WaveDevice:
    xp: object = np
    backend: str = "numpy"
    fallback_reason: str | None = None
    maximum_working_bytes: int = 0
    pool: object = None

    def evidence(self):
        return {"compute_backend": self.backend, "numeric_precision": PRECISION,
                "fallback_reason": self.fallback_reason}


def check_device_memory(required_bytes):
    """Check total estimated live device work, including FFT scratch space."""
    active = _ACTIVE.get()
    if active is None or active.backend != "cupy":
        return
    required = int(required_bytes)
    free, _ = active.xp.cuda.runtime.memGetInfo()
    # Owned allocations are already subtracted from CUDA's free count. The
    # estimate describes the whole operation, including those live arrays.
    owned = 0 if active.pool is None else active.pool.total_bytes()
    available = max(0, int(free) + int(owned) - 256 * 1024**2)
    allowed = min(active.maximum_working_bytes, available)
    if required > allowed:
        raise GPUExecutionError("out_of_memory",
            f"Coherent GPU work needs {required} bytes; device working limit "
            f"{active.maximum_working_bytes}, currently available {available} bytes")


@contextmanager
def device_scope(requested="cpu", *, maximum_working_bytes=24*1024**3,
                 work_items=0, required_bytes=0, acceleration_enabled=True):
    """Select and admit a bounded operator. Errors inside it are not hidden.

    The caller owns retry from its original host input if execution (rather
    than admission) exhausts memory. Require GPU is never weakened.
    """
    active = _ACTIVE.get()
    if active is not None and active.backend == "cupy":
        check_device_memory(required_bytes)
        yield active
        return
    requested = normalise_backend(requested)
    release_device_memory()
    selected, reason = choose_wave_backend(requested, acceleration_enabled=acceleration_enabled,
                                           work_items=int(work_items))
    if selected != WAVE_BACKEND_CUPY:
        yield WaveDevice(fallback_reason=reason)
        return
    with _DEVICE_LOCK:
        session = None
        token = None
        plan_cache = None
        plan_limits = None
        try:
            cp = cupy_module()
            pool = cp.cuda.MemoryPool()
            pool.set_limit(size=int(maximum_working_bytes))
            session = WaveDevice(cp, "cupy", None, int(maximum_working_bytes), pool)
            token = _ACTIVE.set(session)
            check_device_memory(required_bytes)
            # cuFFT plans own work buffers beyond an array's lifetime. Keep
            # only this stage's bounded plans; otherwise old private pools
            # remain alive across successive shapes and escape the budget.
            plan_cache = cp.fft.config.get_plan_cache()
            plan_limits = (plan_cache.get_size(), plan_cache.get_memsize())
            plan_cache.clear()
            plan_cache.set_size(4)
            plan_cache.set_memsize(min(2*1024**3, int(maximum_working_bytes)//8))
        except Exception as error:
            if token is not None:
                _ACTIVE.reset(token)
            if plan_cache is not None and plan_limits is not None:
                plan_cache.clear()
                plan_cache.set_size(plan_limits[0])
                plan_cache.set_memsize(plan_limits[1])
            if session is not None and session.pool is not None:
                _retire_pool(session.xp, session.pool)
            reason = gpu_retry_reason(error, requested, stage="wave_device_admission")
            yield WaveDevice(fallback_reason=reason)
            return
        try:
            with cp.cuda.using_allocator(pool.malloc):
                yield session
                cp.cuda.get_current_stream().synchronize()
        finally:
            _ACTIVE.reset(token)
            if plan_cache is not None:
                plan_cache.clear()
                plan_cache.set_size(plan_limits[0])
                plan_cache.set_memsize(plan_limits[1])
            _retire_pool(cp, pool)
