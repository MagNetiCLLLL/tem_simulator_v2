"""Process-local CUDA RK4 storage; no device handles enter scientific records.

The existing kernel, float64 arithmetic/checkpoints and requested history are
unchanged. Host results always own fresh storage. The lock covers execution and
downloads, so a second consumer cannot overwrite a buffer still being read.
"""
from collections import OrderedDict, deque
from contextvars import ContextVar
from contextlib import contextmanager
from hashlib import sha256
from threading import RLock
from time import perf_counter
import weakref

import numpy as np

PLAN_INDICES = (0, 1, 2, 3, 4, 6, 7, 8, 9, 14, 15, 16, 17)
PARTICLE_INDICES = (5, 10, 11, 12, 13)
SCHEMA = b"column-rk4-f64-checkpoint-v1"
_DEVICE_BUDGET = ContextVar("column_device_budget", default=8 * 1024**3)
_LAST_RECEIPT = ContextVar("column_device_receipt", default=None)


@contextmanager
def device_budget(byte_count):
    token = _DEVICE_BUDGET.set(int(byte_count))
    try:
        yield
    finally:
        _DEVICE_BUDGET.reset(token)


def plan_identity(inputs):
    digest = sha256(SCHEMA)
    for index in PLAN_INDICES:
        value = np.ascontiguousarray(inputs[index])
        digest.update(str((index, value.shape, value.dtype.str)).encode())
        digest.update(memoryview(value).cast("B"))
    digest.update(str(tuple(inputs[i].dtype.str for i in PARTICLE_INDICES)).encode())
    return digest.hexdigest()


def last_device_receipt():
    value = _LAST_RECEIPT.get()
    return dict(value) if value is not None else None


def _allocation_token(context, array):
    """Verify Numba's allocation owner, including resets of the same context.

    Custom memory managers without an inspectable owner registry are supported
    without residency. Pointer addresses alone cannot identify a reset context.
    """
    try:
        memory = array.gpu_data._mem
        pointer = memory.device_pointer
        key = getattr(pointer, "value", pointer)
        registry = context.memory_manager.allocations
        owner = registry.get(key)
        if owner is not None and owner == memory:
            # OwnedPointer._mem can itself be a weak proxy. Keeping the owner
            # strongly alive would interfere with Numba's context reset.
            return (registry, key, weakref.ref(owner))
    except (AttributeError, ReferenceError, TypeError):
        pass
    return None


class RayDeviceCache:
    def __init__(self, maximum_bytes=512 * 1024**2):
        self.maximum_bytes = int(maximum_bytes)
        self.lock = RLock()
        self.entry = None
        self.retained_bytes = 0

    def clear(self):
        with self.lock:
            self.entry = None
            self.retained_bytes = 0

    def execute(self, cuda, kernel, inputs):
        started = perf_counter()
        if len(inputs) != 18 or any(np.asarray(a).ndim != 1 for a in inputs):
            raise ValueError("RK4 device inputs must be eighteen one-dimensional arrays")
        key = plan_identity(inputs)
        rays, saved, checkpoints = inputs[10].size, inputs[16].size, inputs[17].size
        if any(inputs[i].size != rays for i in PARTICLE_INDICES):
            raise ValueError("RK4 particle arrays must have identical populations")
        required = sum(a.nbytes for a in inputs) + rays * (saved * 16 + checkpoints * 32)
        if required > _DEVICE_BUDGET.get():
            from temsim.physics.compute_backend import GPUExecutionError
            raise GPUExecutionError("out_of_memory", "Column CUDA buffers exceed the shared device reservation")
        with self.lock:
            context = cuda.current_context()
            entry = self.entry
            hit = False
            if entry is not None:
                token = entry["token"]
                valid = token is not None and token[2]() is not None and token[0].get(token[1]) is token[2]()
                hit = (entry["context"] is context and valid and entry["key"] == key
                       and entry["rays"] >= rays and entry["bytes"] <= _DEVICE_BUDGET.get())
            receipt = dict(plan_identity=key, plan_reused=hit, buffers_reused=hit,
                           required_bytes=required, ray_count=rays, step_count=inputs[9].size)
            try:
                if not hit:
                    self.clear()
                    # Release obsolete objects before replacement allocation.
                    entry = None
                    before = perf_counter()
                    constants = {i: cuda.to_device(inputs[i]) for i in PLAN_INDICES}
                    receipt["plan_upload_s"] = perf_counter() - before
                    before = perf_counter()
                    particles = {i: cuda.device_array(inputs[i].shape, dtype=inputs[i].dtype)
                                 for i in PARTICLE_INDICES}
                    outputs = [cuda.device_array((n, rays), dtype=dtype)
                               for n, dtype in ((saved, np.float32), (checkpoints, np.float64))
                               for _ in range(4)]
                    receipt["allocation_s"] = perf_counter() - before
                    token = _allocation_token(context, particles[10])
                    entry = dict(context=context, key=key, rays=rays, constants=constants,
                                 particles=particles, outputs=outputs, token=token, bytes=required)
                else:
                    receipt.update(plan_upload_s=0., allocation_s=0.)
                before = perf_counter()
                particle_views = {i: a[:rays] for i, a in entry["particles"].items()}
                for i, array in particle_views.items():
                    array.copy_to_device(inputs[i])
                receipt["particle_upload_s"] = perf_counter() - before
                device_inputs = [entry["constants"].get(i, particle_views.get(i)) for i in range(18)]
                outputs = [array[:, :rays] for array in entry["outputs"]]
                before = perf_counter()
                if rays:
                    kernel[(rays + 127) // 128, 128](*device_inputs, *outputs)
                receipt["kernel_launch_s"] = perf_counter() - before
                before = perf_counter()
                cuda.synchronize()
                receipt["synchronize_s"] = perf_counter() - before
                before = perf_counter()
                results = tuple(array.copy_to_host() for array in outputs)
                receipt["download_s"] = perf_counter() - before
                if entry["token"] is not None and entry["bytes"] <= self.maximum_bytes:
                    self.entry = entry
                    self.retained_bytes = entry["bytes"]
                receipt["retained_bytes"] = self.retained_bytes
                receipt["total_s"] = perf_counter() - started
                receipt["timing_convention"] = "host wall clock; synchronization includes outstanding device work"
                _LAST_RECEIPT.set(receipt)
                return results
            except Exception:
                self.clear()
                _LAST_RECEIPT.set(None)
                raise


DEVICE_CACHE = RayDeviceCache()


class MeasuredStageCosts:
    """Bounded observations of requested executions, without speculative runs."""
    def __init__(self):
        self.lock = RLock()
        self.rows = OrderedDict()

    def record(self, key, backend, seconds):
        if not np.isfinite(seconds) or seconds < 0:
            return
        with self.lock:
            row = self.rows.setdefault(key, {})
            row.setdefault(backend, deque(maxlen=5)).append(float(seconds))
            self.rows.move_to_end(key)
            while len(self.rows) > 128:
                self.rows.popitem(last=False)

    def choose(self, key, eligible, fallback):
        with self.lock:
            row = self.rows.get(key, {})
            if fallback not in row or len(row[fallback]) < 2:
                return fallback, None
            costs = {name: float(np.median(row[name])) for name in eligible
                     if name in row and len(row[name]) >= 2}
            best = min(costs, key=costs.get) if costs else fallback
            if best != fallback and costs[best] < .9 * costs[fallback]:
                return best, "Auto: measured median column execution including transfers (at least two matching observations)"
            return fallback, None


STAGE_COSTS = MeasuredStageCosts()


def measured_workload(inputs):
    key = plan_identity(inputs)
    entry = DEVICE_CACHE.entry
    resident = bool(entry is not None and entry["key"] == key and entry["token"] is not None
                    and entry["token"][2]() is not None
                    and entry["token"][0].get(entry["token"][1]) is entry["token"][2]())
    return (key, inputs[10].size, inputs[9].size, resident)
