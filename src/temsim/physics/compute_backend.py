"""Optional compute-backend discovery and selection.

The simulator must remain importable on machines without Numba or a CUDA
driver.  This module therefore performs only guarded capability checks and
returns canonical backend names consumed by the ray propagator and GUI.
"""

from __future__ import annotations

from dataclasses import dataclass


BACKEND_AUTO = "Auto"
BACKEND_CPU = "CPU"
BACKEND_NUMBA = "Numba CPU"
BACKEND_CUDA = "CUDA GPU"
BACKEND_PREFER_GPU = "Prefer GPU"
BACKEND_REQUIRE_GPU = "Require GPU"
BACKEND_CHOICES = (
    BACKEND_AUTO,
    BACKEND_CPU,
    BACKEND_NUMBA,
    BACKEND_CUDA,
    BACKEND_PREFER_GPU,
    BACKEND_REQUIRE_GPU,
)
WAVE_BACKEND_NUMPY = "NumPy CPU"
WAVE_BACKEND_CUPY = "CuPy CUDA"
AUTO_CUDA_MIN_RAYS = 2_048
AUTO_NUMBA_MIN_RAYS = 256
# One work item represents one complex grid point propagated through one
# specimen slice.  Below this scale PCIe transfers and CUDA-plan setup tend to
# cost more than the FFT work in an interactive preview.
AUTO_CUPY_MIN_WORK_ITEMS = 1_000_000

class GPUExecutionError(RuntimeError):
    def __init__(self, category, detail):
        self.category = category
        super().__init__(f"GPU {category}: {detail}")


def gpu_failure_category(error, *, stage="execution"):
    """Conservative CUDA taxonomy; an exception class alone is not recovery.

    Numeric identities follow NVIDIA's Driver/Runtime API error enumerations.
    Code 701 can mean a bad launch signature, not merely an exhausted device.
    Compiler, invalid-context and illegal-access errors must not become CPU
    successes. Driver-loading absence is only recognised during discovery.
    """
    if isinstance(error, GPUExecutionError):
        return error.category
    module = type(error).__module__
    name = type(error).__name__
    numba_driver = module.startswith("numba.cuda.cudadrv.")
    cupy_driver = module.startswith(("cupy.", "cupy_backends."))
    if numba_driver and name == "CudaSupportError" and stage == "discovery":
        return "unavailable"
    if (numba_driver or cupy_driver) and name in {
        "NVRTCError", "CompileException", "NvvmError", "NvrtcError", "NvvmSupportError", "NvrtcSupportError",
    }:
        return "compilation_failure"
    code = None
    if numba_driver and name == "CudaAPIError":
        code = getattr(error, "code", None)
    elif cupy_driver and name in {"CUDARuntimeError", "CUDADriverError"}:
        code = getattr(error, "status", None)
    if code is not None:
        if code == 2:
            return "out_of_memory"
        if code == 100 or (stage == "discovery" and code in {34, 35, 46}):
            return "unavailable"
        if code in {201, 700, 702, 709, 710, 714, 715, 716, 717, 718, 719}:
            return "context_corrupted"
        if code in {200, 209, 218, 222, 300}:
            return "compilation_failure"
        if code in {1, 101, 701}:
            return "invalid_configuration"
    if module == "cupy.cuda.memory" and name == "OutOfMemoryError":
        return "out_of_memory"
    return None


def gpu_failure_evidence(error, *, stage="execution", attempted_backend=BACKEND_CUDA):
    import traceback
    return dict(category=gpu_failure_category(error, stage=stage) or "unknown",
                exception_type=f"{type(error).__module__}.{type(error).__name__}",
                message=str(error), traceback="".join(traceback.format_exception(error)),
                attempted_backend=attempted_backend, failure_stage=stage)


def gpu_retry_reason(error, policy, *, stage="execution", attempted_backend=BACKEND_CUDA):
    from temsim.job_events import job_event
    evidence = gpu_failure_evidence(error, stage=stage, attempted_backend=attempted_backend)
    job_event("backend_failure", **evidence)
    category = evidence["category"]
    if category == "context_corrupted":
        from temsim.physics.ray_device_cache import DEVICE_CACHE
        DEVICE_CACHE.quarantine(str(error))
    if category not in {"unavailable", "out_of_memory"}:
        raise error
    if normalise_backend(policy) == BACKEND_REQUIRE_GPU:
        raise GPUExecutionError(category, str(error)) from error
    return f"{category}: {type(error).__name__}: {error}"


@dataclass(frozen=True)
class BackendCapability:
    available: bool
    detail: str


def numba_cpu_capability() -> BackendCapability:
    try:
        import numba  # noqa: F401
    except Exception as exc:
        return BackendCapability(False, f"Numba unavailable: {exc}")
    return BackendCapability(True, "Numba parallel CPU kernels available")


def cuda_capability() -> BackendCapability:
    from temsim.physics.ray_device_cache import DEVICE_CACHE
    DEVICE_CACHE.assert_usable()
    try:
        from numba import cuda

        if not cuda.is_available():
            # Numba combines driver presence AND NVVM availability here. A
            # broken compiler is not confirmed absence of a CUDA device.
            from numba.cuda.cudadrv.driver import driver
            if driver.is_available:
                raise GPUExecutionError("toolchain_unavailable", "CUDA driver found, but Numba's CUDA compiler is unavailable")
            return BackendCapability(False, "No usable CUDA device or driver")
        device = cuda.get_current_device()
        name = device.name
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        free_bytes, total_bytes = cuda.current_context().get_memory_info()
        detail = (
            f"{name}; {free_bytes / 1024**3:.1f} GiB free / "
            f"{total_bytes / 1024**3:.1f} GiB"
        )
        return BackendCapability(True, detail)
    except Exception as exc:
        if gpu_failure_category(exc, stage="discovery") == "context_corrupted":
            DEVICE_CACHE.quarantine(str(exc))
        if (isinstance(exc, ModuleNotFoundError) and exc.name in {"numba", "numba.cuda"}
                or gpu_failure_category(exc, stage="discovery") in {"unavailable", "out_of_memory"}):
            return BackendCapability(False, f"CUDA unavailable: {exc}")
        raise


def cupy_capability() -> BackendCapability:
    """Report whether the optional CuPy FFT backend is usable.

    CuPy is deliberately imported only inside this function so the simulator
    remains importable on CPU-only systems and in environments where the
    optional wheel has not been installed.
    """

    from temsim.physics.ray_device_cache import DEVICE_CACHE
    DEVICE_CACHE.assert_usable()
    try:
        import cupy as cp

        device_count = int(cp.cuda.runtime.getDeviceCount())
        if device_count < 1:
            return BackendCapability(False, "CuPy found no CUDA device")
        device_id = int(cp.cuda.runtime.getDevice())
        properties = cp.cuda.runtime.getDeviceProperties(device_id)
        name = properties.get("name", f"CUDA device {device_id}")
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        detail = (
            f"CuPy {cp.__version__}; {name}; "
            f"{free_bytes / 1024**3:.1f} GiB free / "
            f"{total_bytes / 1024**3:.1f} GiB"
        )
        return BackendCapability(True, detail)
    except Exception as exc:
        if gpu_failure_category(exc, stage="discovery") == "context_corrupted":
            DEVICE_CACHE.quarantine(str(exc))
        if (isinstance(exc, ModuleNotFoundError) and exc.name == "cupy"
                or gpu_failure_category(exc, stage="discovery") in {"unavailable", "out_of_memory"}):
            return BackendCapability(False, f"CuPy CUDA unavailable: {exc}")
        raise


def capability_detail_for_display(probe):
    """UI inspection may report a failed probe, never use it for fallback."""
    try:
        return probe().detail
    except Exception as error:
        return f"Capability check failed ({type(error).__name__}): {error}"


def cupy_module():
    """Return the optional CuPy module or raise a descriptive error."""

    capability = cupy_capability()
    if not capability.available:
        raise GPUExecutionError("unavailable", capability.detail)
    import cupy as cp

    return cp


def normalise_backend(value: object) -> str:
    requested = str(value or BACKEND_AUTO).strip()
    aliases = {
        "gpu": BACKEND_CUDA,
        "cuda": BACKEND_CUDA,
        "cupy": BACKEND_CUDA,
        WAVE_BACKEND_CUPY.lower(): BACKEND_CUDA,
        "numba": BACKEND_NUMBA,
        "numpy": BACKEND_CPU,
        "prefer_gpu": BACKEND_PREFER_GPU,
        "require_gpu": BACKEND_REQUIRE_GPU,
        "prefer gpu": BACKEND_PREFER_GPU,
        "require gpu": BACKEND_REQUIRE_GPU,
        WAVE_BACKEND_NUMPY.lower(): BACKEND_CPU,
    }
    requested = aliases.get(requested.lower(), requested)
    return requested if requested in BACKEND_CHOICES else BACKEND_AUTO


def validate_backend_selection(value: object) -> str:
    """New UI/API selections are explicit; historical loading stays tolerant."""
    if value not in BACKEND_CHOICES:
        raise ValueError(f"Select a compute policy from: {', '.join(BACKEND_CHOICES)}")
    return str(value)


def choose_wave_backend(
    requested: object,
    *,
    acceleration_enabled: bool,
    work_items: int,
) -> tuple[str, str | None]:
    """Choose the FFT/multislice backend actually used by wave optics.

    Numba's CPU selection maps to the NumPy reference because the wave solver
    uses vectorised FFTs rather than ray-wise kernels.  Explicit CUDA requests
    attempt CuPy even for small grids; Auto uses it only when the estimated
    grid-point-by-slice work is large enough to amortise setup and transfers.
    """

    policy = normalise_backend(requested).lower().replace(" ", "_")
    if policy in {"prefer_gpu", "require_gpu"}:
        if not acceleration_enabled:
            if policy == "require_gpu":
                raise GPUExecutionError("unavailable", "Acceleration is disabled for the requested wave stage")
            return WAVE_BACKEND_NUMPY, "GPU preference not used: acceleration is disabled"
        status = cupy_capability()
        if status.available:
            return WAVE_BACKEND_CUPY, None
        if policy == "require_gpu":
            raise GPUExecutionError("unavailable", status.detail)
        return WAVE_BACKEND_NUMPY, "unavailable: " + status.detail
    choice = normalise_backend(requested)
    if not acceleration_enabled or choice in (BACKEND_CPU, BACKEND_NUMBA):
        return WAVE_BACKEND_NUMPY, None

    if choice == BACKEND_CUDA:
        status = cupy_capability()
        if status.available:
            return WAVE_BACKEND_CUPY, None
        return WAVE_BACKEND_NUMPY, status.detail

    if int(work_items) < AUTO_CUPY_MIN_WORK_ITEMS:
        return WAVE_BACKEND_NUMPY, None
    status = cupy_capability()
    if status.available:
        return WAVE_BACKEND_CUPY, None
    return WAVE_BACKEND_NUMPY, status.detail


def choose_ray_backend(
    requested: object,
    *,
    acceleration_enabled: bool,
    ray_count: int,
) -> tuple[str, str | None]:
    """Choose a ray backend and return ``(backend, fallback_reason)``."""

    choice = normalise_backend(requested)
    if choice in {BACKEND_PREFER_GPU, BACKEND_REQUIRE_GPU}:
        if not acceleration_enabled:
            if choice == BACKEND_REQUIRE_GPU:
                raise GPUExecutionError("unavailable", "Acceleration is disabled for the requested column-ray stage")
            return BACKEND_CPU, "GPU preference not used: acceleration is disabled"
        status = cuda_capability()
        if status.available:
            return BACKEND_CUDA, None
        if choice == BACKEND_REQUIRE_GPU:
            raise GPUExecutionError("unavailable", status.detail)
        fallback = BACKEND_NUMBA if numba_cpu_capability().available else BACKEND_CPU
        return fallback, "unavailable: " + status.detail
    if not acceleration_enabled or choice == BACKEND_CPU:
        return BACKEND_CPU, None

    if choice == BACKEND_CUDA:
        cuda_status = cuda_capability()
        if cuda_status.available:
            return BACKEND_CUDA, None
        numba_status = numba_cpu_capability()
        fallback = BACKEND_NUMBA if numba_status.available else BACKEND_CPU
        return fallback, cuda_status.detail

    if choice == BACKEND_NUMBA:
        numba_status = numba_cpu_capability()
        if numba_status.available:
            return BACKEND_NUMBA, None
        return BACKEND_CPU, numba_status.detail

    # Auto avoids accelerator launch/JIT overhead for tiny GUI previews.
    if int(ray_count) < AUTO_NUMBA_MIN_RAYS:
        return BACKEND_CPU, None
    if int(ray_count) >= AUTO_CUDA_MIN_RAYS:
        cuda_status = cuda_capability()
        if cuda_status.available:
            return BACKEND_CUDA, None
    numba_status = numba_cpu_capability()
    if int(ray_count) >= AUTO_NUMBA_MIN_RAYS and numba_status.available:
        return BACKEND_NUMBA, None
    return BACKEND_CPU, None
