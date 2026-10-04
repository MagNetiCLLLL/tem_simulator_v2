import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.physics import compute_backend
from temsim.physics.core import propagate


def test_cpu_backend_is_always_explicitly_selectable():
    backend, reason = compute_backend.choose_ray_backend(
        "CPU", acceleration_enabled=True, ray_count=10_000
    )
    assert backend == "CPU"
    assert reason is None


def test_disabled_acceleration_overrides_auto():
    backend, reason = compute_backend.choose_ray_backend(
        "Auto", acceleration_enabled=False, ray_count=10_000
    )
    assert backend == "CPU"
    assert reason is None


def test_explicit_cuda_falls_back_safely(monkeypatch):
    monkeypatch.setattr(
        compute_backend,
        "cuda_capability",
        lambda: compute_backend.BackendCapability(False, "test unavailable"),
    )
    monkeypatch.setattr(
        compute_backend,
        "numba_cpu_capability",
        lambda: compute_backend.BackendCapability(True, "test available"),
    )
    backend, reason = compute_backend.choose_ray_backend(
        "CUDA GPU", acceleration_enabled=True, ray_count=10_000
    )
    assert backend == "Numba CPU"
    assert reason == "test unavailable"


@pytest.mark.parametrize("ray_count", [1, 49, 1000, 3000])
def test_auto_uses_available_cuda_for_every_ray_population(monkeypatch, ray_count):
    monkeypatch.setattr(
        compute_backend,
        "cuda_capability",
        lambda: compute_backend.BackendCapability(True, "test GPU"),
    )
    monkeypatch.setattr(
        compute_backend,
        "numba_cpu_capability",
        lambda: pytest.fail("An available GPU must not be replaced by CPU"),
    )
    backend, reason = compute_backend.choose_ray_backend(
        "Auto", acceleration_enabled=True, ray_count=ray_count
    )
    assert backend == "CUDA GPU"
    assert reason is None


def test_explicit_cuda_wave_request_falls_back_to_numpy(monkeypatch):
    monkeypatch.setattr(
        compute_backend,
        "cupy_capability",
        lambda: compute_backend.BackendCapability(False, "test CuPy unavailable"),
    )
    backend, reason = compute_backend.choose_wave_backend(
        "CUDA GPU", acceleration_enabled=True, work_items=10_000_000
    )

    assert backend == "NumPy CPU"
    assert reason == "test CuPy unavailable"


def test_auto_uses_available_gpu_for_small_wave_fft(monkeypatch):
    monkeypatch.setattr(
        compute_backend,
        "cupy_capability",
        lambda: compute_backend.BackendCapability(True, "test CuPy GPU"),
    )
    backend, reason = compute_backend.choose_wave_backend(
        "Auto", acceleration_enabled=True, work_items=32 * 32 * 10
    )

    assert backend == "CuPy CUDA"
    assert reason is None


@pytest.mark.parametrize("ray_count,expected", [(49, "CPU"), (3000, "Numba CPU")])
def test_auto_without_gpu_uses_available_cpu_backend(monkeypatch, ray_count, expected):
    monkeypatch.setattr(compute_backend, "cuda_capability",
                        lambda: compute_backend.BackendCapability(False, "no GPU"))
    monkeypatch.setattr(compute_backend, "numba_cpu_capability",
                        lambda: compute_backend.BackendCapability(True, "CPU"))
    assert compute_backend.choose_ray_backend(
        "Auto", acceleration_enabled=True, ray_count=ray_count) == (expected, "no GPU")


def test_auto_without_cupy_preserves_wave_cpu_execution(monkeypatch):
    monkeypatch.setattr(compute_backend, "cupy_capability",
                        lambda: compute_backend.BackendCapability(False, "no CuPy"))
    assert compute_backend.choose_wave_backend(
        "Auto", acceleration_enabled=True, work_items=32*32) == ("NumPy CPU", "no CuPy")


def test_auto_cuda_ray_trace_matches_cpu_with_energy_spread(monkeypatch):
    if not compute_backend.cuda_capability().available:
        pytest.skip("CUDA device unavailable")
    state = default_state()
    count = 3000
    azimuth = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    x = 50.0e-6 * np.cos(azimuth)
    y = 50.0e-6 * np.sin(azimuth)
    tx = 1.0e-3 * np.sin(2.0 * azimuth)
    ty = 1.0e-3 * np.cos(2.0 * azimuth)
    energy_offset_ev = np.linspace(-0.5, 0.5, count)
    z0 = state.electron_gun.exit_plane_z_mm
    z1 = 900.3
    state.step_mm = 1.0
    state.history_step_mm = 5.0

    state.acceleration_enabled = False
    state.acceleration_backend = "CPU"
    state._active_backends_used = set()
    cpu_result = propagate(
        state, z0, z1, x, tx, y, ty,
        energy_offset_ev=energy_offset_ev,
    )
    state.acceleration_enabled = True
    state.acceleration_backend = "Auto"
    from temsim.physics.ray_device_cache import STAGE_COSTS
    monkeypatch.setattr(STAGE_COSTS, "choose",
                        lambda *_: pytest.fail("CPU timing history cannot override available GPU"))
    state._active_backends_used = set()
    gpu_result = propagate(
        state, z0, z1, x, tx, y, ty,
        energy_offset_ev=energy_offset_ev,
    )

    assert state.active_backend == "CUDA GPU"
    assert cpu_result[0][-1] == z1
    assert gpu_result[0][-1] == z1
    for cpu_values, gpu_values in zip(cpu_result, gpu_result):
        assert gpu_values == pytest.approx(cpu_values, rel=2.0e-6, abs=1.0e-9)
