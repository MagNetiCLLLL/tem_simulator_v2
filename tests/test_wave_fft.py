import numpy as np
import pytest

from temsim.physics import compute_backend
from temsim.physics import wave_fft
from temsim.physics.wave_fft import (
    form_tem_image,
    stem_diffraction_intensity,
)


def _test_wave(size=48):
    axis = (np.arange(size) - size // 2) * 0.25
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    wave = np.exp(
        -((xx / 2.0) ** 2 + (yy / 1.7) ** 2)
        + 1j * (0.2 * xx - 0.15 * yy)
    )
    wave /= np.linalg.norm(wave)
    return wave, xx, yy


def test_cupy_tem_image_fft_matches_numpy_reference():
    if not compute_backend.cupy_capability().available:
        pytest.skip("CuPy CUDA backend unavailable")
    wave, xx, yy = _test_wave()
    transfer = np.exp(-1j * 0.02 * (xx**2 + yy**2))
    cpu_image, cpu_diffraction, _ = form_tem_image(
        wave, transfer, compute_backend="NumPy CPU"
    )
    gpu_image, gpu_diffraction, diagnostics = form_tem_image(
        wave, transfer, compute_backend="CuPy CUDA"
    )

    assert diagnostics.compute_backend == "CuPy CUDA"
    assert gpu_image == pytest.approx(cpu_image, rel=2.0e-5, abs=2.0e-7)
    assert gpu_diffraction == pytest.approx(
        cpu_diffraction, rel=2.0e-5, abs=2.0e-7
    )


def test_cupy_stem_detector_fft_matches_numpy_reference():
    if not compute_backend.cupy_capability().available:
        pytest.skip("CuPy CUDA backend unavailable")
    wave, _, _ = _test_wave()
    batch = np.stack((wave, wave * 1j, -wave), axis=0)
    cpu, _ = stem_diffraction_intensity(
        batch, compute_backend="NumPy CPU"
    )
    gpu, diagnostics = stem_diffraction_intensity(
        batch, compute_backend="CuPy CUDA"
    )

    assert diagnostics.compute_backend == "CuPy CUDA"
    assert gpu == pytest.approx(cpu, rel=2.0e-5, abs=2.0e-7)


def test_cupy_fft_failure_falls_back_without_losing_the_result(monkeypatch):
    monkeypatch.setattr(
        wave_fft,
        "cupy_module",
        lambda: (_ for _ in ()).throw(RuntimeError("synthetic FFT failure")),
    )
    wave, xx, yy = _test_wave(32)
    transfer = np.exp(-1j * 0.02 * (xx**2 + yy**2))
    reference_image, reference_diffraction, _ = form_tem_image(
        wave, transfer, compute_backend="NumPy CPU"
    )
    image, diffraction, diagnostics = form_tem_image(
        wave, transfer, compute_backend="CuPy CUDA"
    )

    assert image == pytest.approx(reference_image)
    assert diffraction == pytest.approx(reference_diffraction)
    assert diagnostics.compute_backend == "NumPy CPU"
    assert "synthetic FFT failure" in diagnostics.fallback_reason
