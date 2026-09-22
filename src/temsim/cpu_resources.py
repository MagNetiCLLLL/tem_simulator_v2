"""Process-wide numerical CPU budget; this is not a CPU-utilisation target.

One numerical GUI job owns the budget at a time. Numba may use that budget;
BLAS stays serial so BLAS calls inside parallel kernels cannot multiply it.
The GUI/event thread and unrelated external processes are outside this budget.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
import os
import sys
from threading import RLock


_NUMERICAL_JOB_LOCK = RLock()
_STARTUP_LIBRARY_LIMIT = None
_INITIALIZED = False
_ACTIVE_BUDGET = ContextVar("temsim_cpu_thread_budget", default=None)


def available_cpu_count() -> int:
    """Logical CPUs available to this process, respecting OS affinity."""
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    if os.name == "nt":
        try:
            import ctypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.GetCurrentProcess.restype = ctypes.c_void_p
            kernel.GetProcessAffinityMask.argtypes = (ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t))
            process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
            if kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(),
                    ctypes.byref(process_mask), ctypes.byref(system_mask)):
                return max(1, int(process_mask.value).bit_count())
        except (AttributeError, OSError):
            pass
    count = getattr(os, "process_cpu_count", os.cpu_count)()
    return max(1, int(count or 1))


def _positive_limit(value):
    try:
        # OpenMP permits comma-separated limits for nested levels. Nested
        # parallelism is disabled here, but keep its first-level lower cap.
        parsed = int(value.split(",", 1)[0] if isinstance(value, str) else value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def numerical_thread_budget(requested: int | None = None) -> int:
    limits = [max(1, available_cpu_count() // 2)]
    for key in ("TEMSIM_CPU_THREADS", "NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OMP_THREAD_LIMIT"):
        limit = _positive_limit(os.environ.get(key))
        if limit is not None:
            limits.append(limit)
    active = _ACTIVE_BUDGET.get()
    if active is not None:
        limits.append(active)
    if requested is not None:
        if type(requested) is not int or requested < 1:
            raise ValueError("Numerical CPU thread limit must be a positive integer")
        limits.append(requested)
    return min(limits)


def initialize_cpu_resources() -> int:
    """Set safe defaults before numerical imports, preserving lower limits."""
    global _INITIALIZED, _STARTUP_LIBRARY_LIMIT
    from temsim.numba_cache import configure_numba_cache
    configure_numba_cache()
    budget = numerical_thread_budget()
    if _INITIALIZED:
        return budget
    # Changing NUMBA_NUM_THREADS after Numba imports can make its config reload
    # reject an already-created pool. Late callers use the per-thread mask.
    if "numba" not in sys.modules:
        os.environ["NUMBA_NUM_THREADS"] = str(budget)
    for key in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT"):
        os.environ[key] = str(min(budget, _positive_limit(os.environ.get(key)) or budget))
    for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
    os.environ["OMP_NESTED"] = "FALSE"
    from threadpoolctl import threadpool_limits
    _STARTUP_LIBRARY_LIMIT = threadpool_limits(limits=1, user_api="blas")
    _INITIALIZED = True
    return budget


def initialize_numerical_thread(requested: int | None = None) -> int:
    """Apply Numba's thread-local mask in the thread that executes kernels."""
    budget = numerical_thread_budget(requested)
    try:
        import numba
    except ImportError:
        return budget
    actual = min(budget, int(numba.get_num_threads()))
    numba.set_num_threads(actual)
    return actual


@dataclass(frozen=True)
class CpuResourceReceipt:
    available_cpus: int
    numerical_thread_budget: int
    numba_threads: int
    blas_threads: int = 1
    numerical_jobs: int = 1

    def to_dict(self):
        return asdict(self)


class NumericalJobCancelled(InterruptedError):
    pass


@contextmanager
def numerical_job(requested: int | None = None, *, cancelled=None):
    """Serialize numerical jobs across coordinators; nested use is safe."""
    while not _NUMERICAL_JOB_LOCK.acquire(timeout=.05):
        if cancelled is not None and cancelled():
            raise NumericalJobCancelled("Numerical request cancelled before CPU admission")
    previous = None
    token = None
    try:
        if cancelled is not None and cancelled():
            raise NumericalJobCancelled("Numerical request cancelled before CPU admission")
        budget = numerical_thread_budget(requested)
        token = _ACTIVE_BUDGET.set(budget)
        try:
            import numba
            previous = int(numba.get_num_threads())
        except ImportError:
            numba = None
        from threadpoolctl import threadpool_limits
        # BLAS limits are process global; the common lock prevents overlapping
        # GUI contexts from restoring a larger pool while another job is live.
        with threadpool_limits(limits=1, user_api="blas"), threadpool_limits(limits=budget, user_api="openmp"):
            actual = initialize_numerical_thread(budget)
            yield CpuResourceReceipt(available_cpu_count(), budget, actual)
    finally:
        try:
            if previous is not None:
                numba.set_num_threads(previous)
        finally:
            if token is not None:
                _ACTIVE_BUDGET.reset(token)
            _NUMERICAL_JOB_LOCK.release()
