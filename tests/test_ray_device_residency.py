"""CPU ownership fixtures, explicitly not hardware CUDA qualification."""
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from temsim.physics.ray_device_cache import RayDeviceCache, MeasuredStageCosts, device_budget, last_device_receipt
from temsim.physics.ray_integrator import vectorised_rk4
from temsim.physics.compute_backend import GPUExecutionError


class DeviceArray:
    def __init__(self, array, owner):
        self.array = array
        self.gpu_data = SimpleNamespace(_mem=owner)

    def __getitem__(self, item):
        return DeviceArray(self.array[item], self.gpu_data._mem)

    def copy_to_device(self, value):
        self.array[...] = value

    def copy_to_host(self):
        return self.array.copy()


class FakeCUDA:
    def __init__(self):
        self.context = SimpleNamespace(memory_manager=SimpleNamespace(allocations={}))
        self.uploads = 0
        self.allocations = 0

    def current_context(self):
        return self.context

    def device_array(self, shape, dtype):
        self.allocations += 1
        class Owner:
            pass
        owner = Owner()
        owner.device_pointer = self.allocations
        self.context.memory_manager.allocations[self.allocations] = owner
        return DeviceArray(np.empty(shape, dtype), weakref.proxy(owner))

    def to_device(self, value):
        self.uploads += 1
        result = self.device_array(value.shape, value.dtype)
        result.copy_to_device(value)
        return result

    def synchronize(self):
        pass


class ReferenceKernel:
    def __getitem__(self, launch):
        def run(*arrays):
            result = vectorised_rk4(*(a.array for a in arrays[:19]))
            for target, value in zip(arrays[19:], result, strict=True):
                target.array[...] = value
        return run


def inputs(rays=7):
    step = np.array([.0001, .0002, .0001])
    coefficients = [np.linspace(.01, .03, 7) for _ in range(5)]
    return (*coefficients, np.full(rays, .5), *(np.zeros(4) for _ in range(3)), step,
            np.linspace(-.003, .001, rays), np.full(rays, .001),
            np.linspace(.002, -.001, rays), np.full(rays, -.0003),
            np.zeros(4), np.zeros(4), np.array([0, 3]), np.array([0, 1, 2, 3]))


def test_unchanged_plan_has_no_reupload_or_allocation_and_outputs_are_owned():
    cuda, cache = FakeCUDA(), RayDeviceCache()
    original = inputs()
    first = cache.execute(cuda, ReferenceKernel(), original)
    assert len(first) == 8
    counts = cuda.uploads, cuda.allocations
    changed_particles = list(inputs(4))
    changed_particles[10] *= 2
    second = cache.execute(cuda, ReferenceKernel(), tuple(changed_particles))
    assert (cuda.uploads, cuda.allocations) == counts
    assert last_device_receipt()["plan_reused"]
    for actual, expected in zip(first, vectorised_rk4(*original), strict=True):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(second, vectorised_rk4(*changed_particles), strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert all(a.dtype == np.float64 for a in second[4:])
    assert all(not np.shares_memory(a, b) for a, b in zip(first, second, strict=True))


@pytest.mark.parametrize("change", ["coefficient", "checkpoint", "context", "reset", "population"])
def test_incompatible_inputs_or_context_rebuild(change):
    cuda, cache = FakeCUDA(), RayDeviceCache()
    values = list(inputs())
    cache.execute(cuda, ReferenceKernel(), values)
    before = cuda.uploads
    if change == "coefficient":
        values[0][0] += .001
    elif change == "checkpoint":
        values[17] = np.array([0, 2, 3])
    elif change == "context":
        cuda.context = SimpleNamespace(memory_manager=SimpleNamespace(allocations={}))
    elif change == "reset":
        cuda.context.memory_manager.allocations.clear()
    else:
        values = inputs(11)
    cache.execute(cuda, ReferenceKernel(), values)
    assert cuda.uploads > before
    assert not last_device_receipt()["plan_reused"]


def test_memory_admission_and_failure_leave_no_partial_cache():
    cuda, cache = FakeCUDA(), RayDeviceCache(maximum_bytes=1)
    cache.execute(cuda, ReferenceKernel(), inputs())
    assert cache.entry is None
    assert cache.retained_bytes == 0
    before = cuda.allocations
    with device_budget(1), pytest.raises(GPUExecutionError, match="reservation"):
        cache.execute(cuda, ReferenceKernel(), inputs())
    assert cuda.allocations == before
    cache.maximum_bytes = 1024**2
    cache.execute(cuda, ReferenceKernel(), inputs())
    class FailedKernel:
        def __getitem__(self, key):
            def run(*args):
                raise ValueError("Invalid physics fixture")
            return run
    with pytest.raises(ValueError, match="Invalid physics"):
        cache.execute(cuda, FailedKernel(), inputs())
    assert cache.entry is None


def test_measured_auto_needs_matching_multiple_timings_and_available_backend():
    costs = MeasuredStageCosts()
    costs.record("a", "CPU", 2.)
    costs.record("a", "CUDA GPU", .5)
    assert costs.choose("a", ("CPU", "CUDA GPU"), "CPU") == ("CPU", None)
    costs.record("a", "CPU", 1.8)
    costs.record("a", "CUDA GPU", .6)
    assert costs.choose("a", ("CPU", "CUDA GPU"), "CPU")[0] == "CUDA GPU"
    assert costs.choose("a", ("CPU",), "CPU") == ("CPU", None)
    assert costs.choose("different plan", ("CPU", "CUDA GPU"), "CPU") == ("CPU", None)
