"""Optional CuPy acceleration for image-formation and detector FFTs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from temsim.physics.compute_backend import (
    WAVE_BACKEND_CUPY,
    WAVE_BACKEND_NUMPY,
    cupy_module,
)


@dataclass(frozen=True)
class WaveFftDiagnostics:
    compute_backend: str
    numeric_precision: str
    fallback_reason: str | None


def _combined_reason(*reasons: str | None) -> str | None:
    values = [str(value) for value in reasons if value]
    return "; ".join(dict.fromkeys(values)) or None


def _release_cupy_pools(cp) -> None:
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def form_tem_image(
    exit_wave: np.ndarray,
    transfer: np.ndarray,
    *,
    compute_backend: str = WAVE_BACKEND_NUMPY,
    fallback_reason: str | None = None,
) -> tuple[np.ndarray, np.ndarray, WaveFftDiagnostics]:
    """Return raw image and shifted diffraction intensities on the host."""

    if str(compute_backend) == WAVE_BACKEND_CUPY:
        cp = None
        try:
            cp = cupy_module()
            device_wave = cp.asarray(exit_wave, dtype=cp.complex64)
            device_transfer = cp.asarray(transfer, dtype=cp.complex64)
            spectrum = cp.fft.fftshift(cp.fft.fft2(device_wave))
            image_wave = cp.fft.ifft2(
                cp.fft.ifftshift(spectrum * device_transfer)
            )
            raw_image = cp.asnumpy(cp.abs(image_wave) ** 2)
            raw_diffraction = cp.asnumpy(cp.abs(spectrum) ** 2)
            return raw_image, raw_diffraction, WaveFftDiagnostics(
                compute_backend=WAVE_BACKEND_CUPY,
                numeric_precision="complex64 / float32",
                fallback_reason=fallback_reason,
            )
        except Exception as exc:
            fallback_reason = _combined_reason(
                fallback_reason,
                f"CuPy CUDA FFT failed: {type(exc).__name__}: {exc}",
            )
            if cp is not None:
                _release_cupy_pools(cp)

    host_wave = np.asarray(exit_wave, dtype=np.complex128)
    host_transfer = np.asarray(transfer, dtype=np.complex128)
    spectrum = np.fft.fftshift(np.fft.fft2(host_wave))
    image_wave = np.fft.ifft2(np.fft.ifftshift(spectrum * host_transfer))
    return (
        np.abs(image_wave) ** 2,
        np.abs(spectrum) ** 2,
        WaveFftDiagnostics(
            compute_backend=WAVE_BACKEND_NUMPY,
            numeric_precision="complex128 / float64",
            fallback_reason=fallback_reason,
        ),
    )


def stem_diffraction_intensity(
    exit_wave: np.ndarray,
    *,
    compute_backend: str = WAVE_BACKEND_NUMPY,
    fallback_reason: str | None = None,
) -> tuple[np.ndarray, WaveFftDiagnostics]:
    """Return normalised, shifted STEM diffraction intensities on the host."""

    if str(compute_backend) == WAVE_BACKEND_CUPY:
        cp = None
        try:
            cp = cupy_module()
            device_wave = cp.asarray(exit_wave, dtype=cp.complex64)
            diffraction = cp.abs(
                cp.fft.fftshift(
                    cp.fft.fft2(device_wave, axes=(-2, -1)),
                    axes=(-2, -1),
                )
            ) ** 2
            diffraction /= cp.maximum(
                cp.sum(diffraction, axis=(-2, -1), keepdims=True),
                cp.float32(1.0e-30),
            )
            return cp.asnumpy(diffraction), WaveFftDiagnostics(
                compute_backend=WAVE_BACKEND_CUPY,
                numeric_precision="complex64 / float32",
                fallback_reason=fallback_reason,
            )
        except Exception as exc:
            fallback_reason = _combined_reason(
                fallback_reason,
                f"CuPy CUDA FFT failed: {type(exc).__name__}: {exc}",
            )
            if cp is not None:
                _release_cupy_pools(cp)

    host_wave = np.asarray(exit_wave, dtype=np.complex128)
    diffraction = np.abs(
        np.fft.fftshift(
            np.fft.fft2(host_wave, axes=(-2, -1)),
            axes=(-2, -1),
        )
    ) ** 2
    diffraction /= np.maximum(
        np.sum(diffraction, axis=(-2, -1), keepdims=True), 1.0e-30
    )
    return diffraction, WaveFftDiagnostics(
        compute_backend=WAVE_BACKEND_NUMPY,
        numeric_precision="complex128 / float64",
        fallback_reason=fallback_reason,
    )
