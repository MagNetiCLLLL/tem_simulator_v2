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
BACKEND_CHOICES = (
    BACKEND_AUTO,
    BACKEND_CPU,
    BACKEND_NUMBA,
    BACKEND_CUDA,
)
WAVE_BACKEND_NUMPY = "NumPy CPU"
WAVE_BACKEND_CUPY = "CuPy CUDA"
AUTO_CUDA_MIN_RAYS = 2_048
AUTO_NUMBA_MIN_RAYS = 256
# One work item represents one complex grid point propagated through one
# specimen slice.  Below this scale PCIe transfers and CUDA-plan setup tend to
# cost more than the FFT work in an interactive preview.
AUTO_CUPY_MIN_WORK_ITEMS = 1_000_000


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
    try:
        from numba import cuda

        if not cuda.is_available():
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
        return BackendCapability(False, f"CUDA unavailable: {exc}")


def cupy_capability() -> BackendCapability:
    """Report whether the optional CuPy FFT backend is usable.

    CuPy is deliberately imported only inside this function so the simulator
    remains importable on CPU-only systems and in environments where the
    optional wheel has not been installed.
    """

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
        return BackendCapability(False, f"CuPy CUDA unavailable: {exc}")


def cupy_module():
    """Return the optional CuPy module or raise a descriptive error."""

    capability = cupy_capability()
    if not capability.available:
        raise RuntimeError(capability.detail)
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
        WAVE_BACKEND_NUMPY.lower(): BACKEND_CPU,
    }
    requested = aliases.get(requested.lower(), requested)
    return requested if requested in BACKEND_CHOICES else BACKEND_AUTO


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
