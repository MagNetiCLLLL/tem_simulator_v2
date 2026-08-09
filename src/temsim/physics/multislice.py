"""CPU reference multislice propagation for elastic electron scattering.

Coordinates use a right-handed specimen frame with electrons travelling along
+Z.  Real-space sampling and slice thicknesses are in angstrom, reciprocal
coordinates are in inverse angstrom, and each slice potential is already
projected through that slice in volt-angstrom.

NumPy's unshifted FFT convention is used internally::

    F(q) = sum_x psi(x) exp(-2 pi i q.x)

The symmetric split operator applies half a Fresnel propagation before and
after each phase grating.  This is a scalar, paraxial, elastic model; magnetic
vector potentials, spin, phonons and inelastic scattering are not included.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from temsim.physics.compute_backend import (
    WAVE_BACKEND_CUPY,
    WAVE_BACKEND_NUMPY,
    cupy_module,
)

PixelSizeAngstrom = float | tuple[float, float]


@dataclass(frozen=True)
class MultisliceDiagnostics:
    model: str
    slice_count: int
    total_thickness_angstrom: float
    slice_thickness_angstrom: float
    bandwidth_fraction: float
    maximum_isotropic_angle_mrad: float
    maximum_phase_per_slice_rad: float
    initial_integrated_intensity: float
    final_integrated_intensity: float
    maximum_relative_intensity_change: float
    compute_backend: str
    numeric_precision: str
    fallback_reason: str | None
    pixel_size_y_angstrom: float
    pixel_size_x_angstrom: float


def _pixel_spacing_yx(pixel_size_angstrom) -> tuple[float, float]:
    values = np.asarray(pixel_size_angstrom, dtype=float)
    if values.ndim == 0:
        spacing_y = spacing_x = float(values)
    elif values.shape == (2,):
        spacing_y, spacing_x = (float(value) for value in values)
    else:
        raise ValueError("Pixel size must be a scalar or a (Y, X) pair.")
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (spacing_y, spacing_x)
    ):
        raise ValueError("Pixel sizes must be finite and positive.")
    return spacing_y, spacing_x


def _validate_wave_and_sampling(
    incident_wave,
    pixel_size_angstrom: PixelSizeAngstrom,
    *,
    xp,
    complex_dtype,
):
    wave = xp.asarray(incident_wave, dtype=complex_dtype)
    if wave.ndim < 2 or min(wave.shape[-2:]) < 2:
        raise ValueError("Incident wave must have two non-trivial spatial axes.")
    if not bool(xp.all(xp.isfinite(wave)).item()):
        raise ValueError("Incident wave contains NaN or infinity.")
    _pixel_spacing_yx(pixel_size_angstrom)
    return wave.copy()


def _intensity_per_wave(wave, *, xp):
    return xp.sum(
        xp.abs(wave) ** 2,
        axis=(-2, -1),
        dtype=xp.float64,
    )


def _frequency_grid(
    shape: tuple[int, int], pixel_size_angstrom: PixelSizeAngstrom, *, xp, real_dtype
):
    ny, nx = shape
    spacing_y, spacing_x = _pixel_spacing_yx(pixel_size_angstrom)
    qx = xp.fft.fftfreq(nx, d=spacing_x).astype(real_dtype)
    qy = xp.fft.fftfreq(ny, d=spacing_y).astype(real_dtype)
    grid_qx, grid_qy = xp.meshgrid(qx, qy, indexing="xy")
    return grid_qx, grid_qy, grid_qx**2 + grid_qy**2


def _bandwidth_mask(
    shape: tuple[int, int],
    pixel_size_angstrom: PixelSizeAngstrom,
    bandwidth_fraction: float,
    *,
    xp,
    real_dtype,
):
    fraction = float(bandwidth_fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("Bandwidth fraction must be in (0, 1].")
    spacing_y, spacing_x = _pixel_spacing_yx(pixel_size_angstrom)
    nyquist = min(0.5 / spacing_y, 0.5 / spacing_x)
    if fraction >= 1.0:
        return xp.ones(shape, dtype=bool), nyquist
    _, _, frequency_squared = _frequency_grid(
        shape,
        pixel_size_angstrom,
        xp=xp,
        real_dtype=real_dtype,
    )
    cutoff = fraction * nyquist
    return frequency_squared <= cutoff**2, cutoff


def _propagator(
    frequency_squared,
    wavelength_angstrom: float,
    distance_angstrom: float,
    bandwidth_mask,
    *,
    xp,
    complex_dtype,
):
    propagator = xp.exp(
        -1j
        * math.pi
        * float(wavelength_angstrom)
        * float(distance_angstrom)
        * frequency_squared
    )
    return bandwidth_mask * propagator.astype(complex_dtype, copy=False)


def _propagate(wave, propagator, *, xp):
    return xp.fft.ifft2(
        xp.fft.fft2(wave, axes=(-2, -1)) * propagator,
        axes=(-2, -1),
    )


def _combined_reason(*reasons: str | None) -> str | None:
    values = [str(value) for value in reasons if value]
    return "; ".join(dict.fromkeys(values)) or None


def _propagate_multislice_backend(
    incident_wave,
    projected_potential_v_angstrom,
    *,
    pixel_size_angstrom: PixelSizeAngstrom,
    wavelength_angstrom: float,
    interaction_constant_rad_per_v_angstrom: float,
    total_thickness_angstrom: float | None,
    target_slice_thickness_angstrom: float,
    slice_thicknesses_angstrom: np.ndarray | None,
    bandwidth_fraction: float,
    xp,
    real_dtype,
    complex_dtype,
    compute_backend: str,
    numeric_precision: str,
    fallback_reason: str | None,
):
    wave = _validate_wave_and_sampling(
        incident_wave,
        pixel_size_angstrom,
        xp=xp,
        complex_dtype=complex_dtype,
    )
    potential = xp.asarray(projected_potential_v_angstrom, dtype=real_dtype)
    if potential.ndim not in (2, 3):
        raise ValueError("Projected potential must have shape (Y, X) or (Z, Y, X).")
    if potential.shape[-2:] != wave.shape[-2:]:
        raise ValueError("Wave and projected-potential spatial shapes must match.")
    if not bool(xp.all(xp.isfinite(potential)).item()):
        raise ValueError("Projected potential contains NaN or infinity.")

    wavelength = float(wavelength_angstrom)
    sigma = float(interaction_constant_rad_per_v_angstrom)
    if not math.isfinite(wavelength) or wavelength <= 0.0:
        raise ValueError("Electron wavelength must be finite and positive.")
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("Interaction constant must be finite and positive.")

    if potential.ndim == 2:
        if total_thickness_angstrom is None:
            raise ValueError(
                "Total thickness is required for a two-dimensional total "
                "projected potential."
            )
        total_thickness = float(total_thickness_angstrom)
        target_slice = float(target_slice_thickness_angstrom)
        if not math.isfinite(total_thickness) or total_thickness < 0.0:
            raise ValueError("Total specimen thickness cannot be negative.")
        if not math.isfinite(target_slice) or target_slice <= 0.0:
            raise ValueError("Target slice thickness must be positive.")
        if total_thickness == 0.0:
            slice_count = 0
            thicknesses = np.empty(0, dtype=np.float64)
        else:
            slice_count = max(1, int(math.ceil(total_thickness / target_slice)))
            thicknesses = np.full(
                slice_count, total_thickness / slice_count, dtype=np.float64
            )
        slice_potential = potential / max(slice_count, 1)

        def potential_at(_index: int):
            return slice_potential

        model = "continuous_column_multislice"
    else:
        slice_count = potential.shape[0]
        if slice_count == 0:
            raise ValueError("A three-dimensional potential needs at least one slice.")
        if slice_thicknesses_angstrom is None:
            target_slice = float(target_slice_thickness_angstrom)
            if not math.isfinite(target_slice) or target_slice <= 0.0:
                raise ValueError("Slice thickness must be positive.")
            thicknesses = np.full(slice_count, target_slice, dtype=np.float64)
        else:
            thicknesses = np.asarray(
                slice_thicknesses_angstrom, dtype=np.float64
            )
            if thicknesses.shape != (slice_count,):
                raise ValueError("Slice thickness array must match the Z dimension.")
        if not np.all(np.isfinite(thicknesses)) or np.any(thicknesses <= 0.0):
            raise ValueError("Every physical slice thickness must be positive.")
        total_thickness = float(np.sum(thicknesses))

        def potential_at(index: int):
            return potential[index]

        model = "atomic_slice_multislice"

    initial_intensity = _intensity_per_wave(wave, xp=xp)
    if bool(xp.any(initial_intensity <= 0.0).item()):
        raise ValueError("Every incident wave must contain positive intensity.")
    maximum_relative_change_device = xp.asarray(0.0, dtype=xp.float64)
    maximum_phase = 0.0
    bandwidth_mask, cutoff = _bandwidth_mask(
        wave.shape[-2:],
        pixel_size_angstrom,
        bandwidth_fraction,
        xp=xp,
        real_dtype=real_dtype,
    )
    _, _, frequency_squared = _frequency_grid(
        wave.shape[-2:],
        pixel_size_angstrom,
        xp=xp,
        real_dtype=real_dtype,
    )

    if slice_count:
        phase_potential = (
            potential_at(0) if potential.ndim == 2 else potential
        )
        maximum_phase = float(
            (sigma * xp.max(xp.abs(phase_potential))).item()
        )
        uniform_thickness = bool(
            np.allclose(thicknesses, thicknesses[0], rtol=0.0, atol=1.0e-15)
        )
        half = _propagator(
            frequency_squared,
            wavelength,
            0.5 * thicknesses[0],
            bandwidth_mask,
            xp=xp,
            complex_dtype=complex_dtype,
        )
        full = (
            _propagator(
                frequency_squared,
                wavelength,
                thicknesses[0],
                bandwidth_mask,
                xp=xp,
                complex_dtype=complex_dtype,
            )
            if uniform_thickness and slice_count > 1
            else None
        )
        wave = _propagate(wave, half, xp=xp)
        for index in range(slice_count):
            phase = sigma * potential_at(index)
            transmission = xp.exp(1j * phase).astype(
                complex_dtype, copy=False
            )
            wave *= transmission
            if uniform_thickness:
                next_propagator = full if index + 1 < slice_count else half
            else:
                if index + 1 < slice_count:
                    distance = 0.5 * (
                        thicknesses[index] + thicknesses[index + 1]
                    )
                else:
                    distance = 0.5 * thicknesses[index]
                next_propagator = _propagator(
                    frequency_squared,
                    wavelength,
                    distance,
                    bandwidth_mask,
                    xp=xp,
                    complex_dtype=complex_dtype,
                )
            wave = _propagate(
                wave,
                next_propagator,
                xp=xp,
            )
            current_intensity = _intensity_per_wave(wave, xp=xp)
            relative_change = xp.max(
                xp.abs(current_intensity - initial_intensity)
                / initial_intensity
            )
            maximum_relative_change_device = xp.maximum(
                maximum_relative_change_device,
                relative_change,
            )

    final_intensity = _intensity_per_wave(wave, xp=xp)
    maximum_relative_change = float(maximum_relative_change_device.item())
    maximum_angle_mrad = math.asin(min(wavelength * cutoff, 1.0)) * 1.0e3
    diagnostics = MultisliceDiagnostics(
        model=model,
        slice_count=slice_count,
        total_thickness_angstrom=float(total_thickness),
        slice_thickness_angstrom=(
            float(np.max(thicknesses)) if slice_count else 0.0
        ),
        bandwidth_fraction=float(bandwidth_fraction),
        maximum_isotropic_angle_mrad=maximum_angle_mrad,
        maximum_phase_per_slice_rad=maximum_phase,
        initial_integrated_intensity=float(xp.sum(initial_intensity).item()),
        final_integrated_intensity=float(xp.sum(final_intensity).item()),
        maximum_relative_intensity_change=maximum_relative_change,
        compute_backend=compute_backend,
        numeric_precision=numeric_precision,
        fallback_reason=fallback_reason,
        pixel_size_y_angstrom=_pixel_spacing_yx(pixel_size_angstrom)[0],
        pixel_size_x_angstrom=_pixel_spacing_yx(pixel_size_angstrom)[1],
    )
    return wave, diagnostics


def propagate_multislice_cupy_device(
    incident_wave,
    projected_potential_v_angstrom,
    *,
    pixel_size_angstrom: PixelSizeAngstrom,
    wavelength_angstrom: float,
    interaction_constant_rad_per_v_angstrom: float,
    total_thickness_angstrom: float | None = None,
    target_slice_thickness_angstrom: float = 2.0,
    slice_thicknesses_angstrom: np.ndarray | None = None,
    bandwidth_fraction: float = 2.0 / 3.0,
    fallback_reason: str | None = None,
    cupy=None,
):
    """Propagate CuPy arrays without copying the exit wave to the host.

    This is the device-resident building block for compound CUDA pipelines.
    It deliberately does not catch CUDA errors: the caller must treat the
    complete compound observable as atomic and recompute it with the NumPy
    reference after any failure. Arrays have the same ``(..., Y, X)`` and
    ``(Z, Y, X)`` conventions as :func:`propagate_multislice`.
    """

    cp = cupy if cupy is not None else cupy_module()
    return _propagate_multislice_backend(
        incident_wave,
        projected_potential_v_angstrom,
        pixel_size_angstrom=pixel_size_angstrom,
        wavelength_angstrom=wavelength_angstrom,
        interaction_constant_rad_per_v_angstrom=(
            interaction_constant_rad_per_v_angstrom
        ),
        total_thickness_angstrom=total_thickness_angstrom,
        target_slice_thickness_angstrom=target_slice_thickness_angstrom,
        slice_thicknesses_angstrom=slice_thicknesses_angstrom,
        bandwidth_fraction=bandwidth_fraction,
        xp=cp,
        real_dtype=cp.float32,
        complex_dtype=cp.complex64,
        compute_backend=WAVE_BACKEND_CUPY,
        numeric_precision="complex64 / float32",
        fallback_reason=fallback_reason,
    )


def propagate_multislice(
    incident_wave: np.ndarray,
    projected_potential_v_angstrom: np.ndarray,
    *,
    pixel_size_angstrom: PixelSizeAngstrom,
    wavelength_angstrom: float,
    interaction_constant_rad_per_v_angstrom: float,
    total_thickness_angstrom: float | None = None,
    target_slice_thickness_angstrom: float = 2.0,
    slice_thicknesses_angstrom: np.ndarray | None = None,
    bandwidth_fraction: float = 2.0 / 3.0,
    compute_backend: str = WAVE_BACKEND_NUMPY,
    fallback_reason: str | None = None,
) -> tuple[np.ndarray, MultisliceDiagnostics]:
    """Propagate one wave or a leading batch of waves through a specimen.

    ``incident_wave`` has shape ``(..., ny, nx)``. A two-dimensional
    ``projected_potential_v_angstrom`` is interpreted as the total projected
    potential of a continuous-column specimen and is divided uniformly across
    slices. A three-dimensional potential has shape ``(slices, ny, nx)`` and
    contains the already-integrated potential of each physical slice.

    The NumPy CPU path is the complex128 scientific reference.  The optional
    CuPy CUDA path uses complex64, returns a NumPy host array, and falls back
    to the reference path if importing CuPy, allocating device memory, or an
    FFT operation fails.
    """

    common = {
        "pixel_size_angstrom": pixel_size_angstrom,
        "wavelength_angstrom": wavelength_angstrom,
        "interaction_constant_rad_per_v_angstrom": (
            interaction_constant_rad_per_v_angstrom
        ),
        "total_thickness_angstrom": total_thickness_angstrom,
        "target_slice_thickness_angstrom": target_slice_thickness_angstrom,
        "slice_thicknesses_angstrom": slice_thicknesses_angstrom,
        "bandwidth_fraction": bandwidth_fraction,
    }
    if str(compute_backend) == WAVE_BACKEND_CUPY:
        cp = None
        try:
            cp = cupy_module()
            result, diagnostics = _propagate_multislice_backend(
                incident_wave,
                projected_potential_v_angstrom,
                **common,
                xp=cp,
                real_dtype=cp.float32,
                complex_dtype=cp.complex64,
                compute_backend=WAVE_BACKEND_CUPY,
                numeric_precision="complex64 / float32",
                fallback_reason=fallback_reason,
            )
            return cp.asnumpy(result), diagnostics
        except Exception as exc:
            runtime_reason = f"CuPy CUDA failed: {type(exc).__name__}: {exc}"
            fallback_reason = _combined_reason(
                fallback_reason, runtime_reason
            )
            if cp is not None:
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                except Exception:
                    pass

    result, diagnostics = _propagate_multislice_backend(
        incident_wave,
        projected_potential_v_angstrom,
        **common,
        xp=np,
        real_dtype=np.float64,
        complex_dtype=np.complex128,
        compute_backend=WAVE_BACKEND_NUMPY,
        numeric_precision="complex128 / float64",
        fallback_reason=fallback_reason,
    )
    return np.asarray(result), diagnostics
