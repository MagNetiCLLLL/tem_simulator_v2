"""Device-resident CUDA pipeline for angle-resolved STEM observables.

The caller supplies a shifted reciprocal-space probe aperture, periodic
specimen potentials and non-overlapping detector masks.  Probe formation,
multislice propagation, diffraction FFTs, frozen-phonon intensity averaging
and detector integration remain on one CuPy device.  Only the final detector
fractions and uncollected fraction are copied to the host.

Real-space arrays use ``(..., Y, X)``; explicit potentials use
``(Z, Y, X)``.  Lengths are angstrom, reciprocal coordinates are inverse
angstrom and each potential slice is in volt-angstrom.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time

import numpy as np

from temsim.physics.compute_backend import (
    WAVE_BACKEND_CUPY,
    cupy_module,
)
from temsim.physics.cuda_multislice_plan import CuPyMultislicePlan
from temsim.physics.multislice import (
    MultisliceDiagnostics,
    PixelSizeAngstrom,
)
from temsim.physics.wave_fft import WaveFftDiagnostics


@dataclass(frozen=True)
class ResidentStemCudaResult:
    fractions_flat: dict[str, np.ndarray]
    uncollected_flat: np.ndarray
    detector_relative_standard_error: dict[str, float]
    multislice_diagnostics: MultisliceDiagnostics | None
    fft_diagnostics: WaveFftDiagnostics
    metrics: dict


def release_cupy_memory_pools() -> None:
    """Release unused CuPy blocks after a failed compound calculation."""

    try:
        cp = cupy_module()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def _periodic_wrap(values: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    frequency_step = abs(float(frequencies[1] - frequencies[0]))
    if not math.isfinite(frequency_step) or frequency_step <= 0.0:
        raise ValueError("Reciprocal-space sampling must be finite and positive.")
    period = 1.0 / frequency_step
    return np.remainder(values + 0.5 * period, period) - 0.5 * period


def _validate_inputs(
    base_spectrum,
    frequencies_x,
    frequencies_y,
    scan_x_angstrom,
    scan_y_angstrom,
    potential_configurations_v_angstrom,
    detector_masks,
    batch_size,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple,
    dict,
]:
    spectrum = np.asarray(base_spectrum)
    frequencies_x = np.asarray(frequencies_x, dtype=np.float64)
    frequencies_y = np.asarray(frequencies_y, dtype=np.float64)
    scan_x = np.asarray(scan_x_angstrom, dtype=np.float64)
    scan_y = np.asarray(scan_y_angstrom, dtype=np.float64)
    configurations = tuple(potential_configurations_v_angstrom)
    masks = {
        str(key): np.asarray(mask, dtype=bool)
        for key, mask in detector_masks.items()
    }
    if spectrum.ndim != 2 or min(spectrum.shape) < 2:
        raise ValueError("The STEM probe spectrum must have shape (Y, X).")
    if spectrum.shape != (frequencies_y.size, frequencies_x.size):
        raise ValueError("Probe spectrum and reciprocal axes do not match.")
    if not np.all(np.isfinite(spectrum)):
        raise ValueError("The STEM probe spectrum contains NaN or infinity.")
    if scan_x.ndim != 1 or scan_x.shape != scan_y.shape or scan_x.size == 0:
        raise ValueError("Flattened STEM scan coordinates must be non-empty 1-D arrays.")
    if not np.all(np.isfinite(scan_x)) or not np.all(np.isfinite(scan_y)):
        raise ValueError("STEM scan coordinates contain NaN or infinity.")
    if not configurations:
        raise ValueError("At least one specimen-potential configuration is required.")
    if int(batch_size) < 1:
        raise ValueError("CUDA STEM batch size must be positive.")
    for configuration in configurations:
        if np.shape(configuration)[-2:] != spectrum.shape:
            raise ValueError("Potential and probe spatial shapes must match.")
    if not masks:
        raise ValueError("At least one detector mask is required.")
    for key, mask in masks.items():
        if mask.shape != spectrum.shape:
            raise ValueError(f"Detector mask {key!r} has the wrong shape.")
    return (
        spectrum,
        frequencies_x,
        frequencies_y,
        scan_x,
        scan_y,
        configurations,
        masks,
    )


def run_resident_stem_cuda(
    *,
    base_spectrum: np.ndarray,
    frequencies_x: np.ndarray,
    frequencies_y: np.ndarray,
    scan_x_angstrom: np.ndarray,
    scan_y_angstrom: np.ndarray,
    potential_configurations_v_angstrom: tuple[np.ndarray, ...],
    detector_masks: dict[str, np.ndarray],
    multislice_enabled: bool,
    pixel_size_angstrom: PixelSizeAngstrom,
    wavelength_angstrom: float,
    interaction_constant_rad_per_v_angstrom: float,
    total_thickness_angstrom: float,
    target_slice_thickness_angstrom: float,
    slice_thicknesses_angstrom: np.ndarray | None,
    bandwidth_fraction: float,
    batch_size: int = 8,
    fallback_reason: str | None = None,
) -> ResidentStemCudaResult:
    """Calculate all STEM detector fractions with one bulk host transfer.

    CUDA failures intentionally propagate to the caller.  The caller must
    discard every partial result and rerun the complete observable through the
    NumPy reference path.
    """

    (
        spectrum,
        frequencies_x,
        frequencies_y,
        scan_x,
        scan_y,
        configurations,
        masks,
    ) = _validate_inputs(
        base_spectrum,
        frequencies_x,
        frequencies_y,
        scan_x_angstrom,
        scan_y_angstrom,
        potential_configurations_v_angstrom,
        detector_masks,
        batch_size,
    )
    cp = cupy_module()
    started = time.perf_counter()
    detector_keys = tuple(masks)
    configuration_count = len(configurations)
    scan_count = scan_x.size

    device_spectrum = cp.asarray(spectrum, dtype=cp.complex64)
    device_frequency_x = cp.asarray(frequencies_x, dtype=cp.float32)
    device_frequency_y = cp.asarray(frequencies_y, dtype=cp.float32)
    device_fx, device_fy = cp.meshgrid(
        device_frequency_x,
        device_frequency_y,
        indexing="xy",
    )
    wrapped_x = _periodic_wrap(scan_x, frequencies_x)
    wrapped_y = _periodic_wrap(scan_y, frequencies_y)
    device_scan_x = cp.asarray(wrapped_x, dtype=cp.float32)
    device_scan_y = cp.asarray(wrapped_y, dtype=cp.float32)
    device_potentials = tuple(
        cp.asarray(configuration, dtype=cp.float32)
        for configuration in configurations
    )
    device_masks = {
        key: cp.asarray(masks[key], dtype=cp.bool_)
        for key in detector_keys
    }
    detector_sums = {
        key: cp.zeros(scan_count, dtype=cp.float64)
        for key in detector_keys
    }
    detector_sums_squared = {
        key: cp.zeros(scan_count, dtype=cp.float64)
        for key in detector_keys
    }
    multislice_plan = None
    maximum_phases = ()
    potential_phase_scan_count = 0
    phase_gratings = None
    if multislice_enabled:
        explicit_slices = device_potentials[0].ndim == 3
        multislice_plan = CuPyMultislicePlan.build(
            device_potentials[0],
            pixel_size_angstrom=pixel_size_angstrom,
            wavelength_angstrom=wavelength_angstrom,
            interaction_constant_rad_per_v_angstrom=(
                interaction_constant_rad_per_v_angstrom
            ),
            total_thickness_angstrom=(
                None if explicit_slices else total_thickness_angstrom
            ),
            target_slice_thickness_angstrom=(
                target_slice_thickness_angstrom
            ),
            slice_thicknesses_angstrom=(
                slice_thicknesses_angstrom if explicit_slices else None
            ),
            bandwidth_fraction=bandwidth_fraction,
            cupy=cp,
        )
        device_potentials = tuple(
            multislice_plan.validate_potential(potential)
            for potential in device_potentials
        )
        maximum_phases = tuple(
            multislice_plan.maximum_phase_per_slice_rad(
                potential,
                potential_is_validated=True,
            )
            for potential in device_potentials
        )
        potential_phase_scan_count = configuration_count
    else:
        if any(potential.ndim != 2 for potential in device_potentials):
            raise ValueError(
                "Every projected phase-object potential must be two-dimensional."
            )
        phase_gratings = tuple(
            cp.exp(
                1j
                * float(interaction_constant_rad_per_v_angstrom)
                * potential
            ).astype(cp.complex64, copy=False)
            for potential in device_potentials
        )

    first_diagnostics = None
    maximum_phase = 0.0
    maximum_intensity_change = 0.0
    diagnostic_count = 0
    effective_batch_size = min(int(batch_size), scan_count)
    batch_count = int(math.ceil(scan_count / effective_batch_size))
    for start in range(0, scan_count, effective_batch_size):
        stop = min(start + effective_batch_size, scan_count)
        x0 = device_scan_x[start:stop, None, None]
        y0 = device_scan_y[start:stop, None, None]
        shift = cp.exp(
            cp.asarray(-2j * math.pi, dtype=cp.complex64)
            * (
                device_fx[None, :, :] * x0
                + device_fy[None, :, :] * y0
            )
        )
        shifted_spectrum = device_spectrum[None, :, :] * shift
        probe = cp.fft.ifft2(
            cp.fft.ifftshift(shifted_spectrum, axes=(-2, -1)),
            axes=(-2, -1),
        )
        probe_norm = cp.sqrt(
            cp.maximum(
                cp.sum(
                    cp.abs(probe) ** 2,
                    axis=(-2, -1),
                    dtype=cp.float64,
                ),
                cp.float64(1.0e-30),
            )
        )
        normalised_probe = (
            probe / probe_norm[:, None, None]
        ).astype(cp.complex64, copy=False)

        for configuration_index, device_potential in enumerate(
            device_potentials
        ):
            if multislice_enabled:
                exit_wave, diagnostics = multislice_plan.propagate(
                    normalised_probe,
                    device_potential,
                    maximum_phase_per_slice_rad=(
                        maximum_phases[configuration_index]
                    ),
                    fallback_reason=fallback_reason,
                    potential_is_validated=True,
                )
                if first_diagnostics is None:
                    first_diagnostics = diagnostics
                maximum_phase = max(
                    maximum_phase,
                    diagnostics.maximum_phase_per_slice_rad,
                )
                maximum_intensity_change = max(
                    maximum_intensity_change,
                    diagnostics.maximum_relative_intensity_change,
                )
                diagnostic_count += 1
            else:
                exit_wave = (
                    normalised_probe
                    * phase_gratings[configuration_index]
                )

            diffraction = cp.abs(
                cp.fft.fftshift(
                    cp.fft.fft2(exit_wave, axes=(-2, -1)),
                    axes=(-2, -1),
                )
            ) ** 2
            diffraction /= cp.maximum(
                cp.sum(
                    diffraction,
                    axis=(-2, -1),
                    keepdims=True,
                ),
                cp.float32(1.0e-30),
            )
            for key in detector_keys:
                values = cp.sum(
                    diffraction[:, device_masks[key]],
                    axis=1,
                    dtype=cp.float64,
                )
                values = cp.clip(values, 0.0, 1.0)
                detector_sums[key][start:stop] += values
                detector_sums_squared[key][start:stop] += values**2

    if first_diagnostics is not None:
        multislice_diagnostics = replace(
            first_diagnostics,
            maximum_phase_per_slice_rad=maximum_phase,
            maximum_relative_intensity_change=maximum_intensity_change,
        )
    else:
        multislice_diagnostics = None

    detector_means = {
        key: detector_sums[key] / configuration_count
        for key in detector_keys
    }
    collected = cp.zeros(scan_count, dtype=cp.float64)
    for values in detector_means.values():
        collected += values
    device_uncollected = cp.maximum(1.0 - collected, 0.0)

    relative_standard_error = {}
    for key in detector_keys:
        if configuration_count > 1:
            centred_sum = cp.maximum(
                detector_sums_squared[key]
                - detector_sums[key] ** 2 / configuration_count,
                0.0,
            )
            standard_error_squared = (
                centred_sum
                / (configuration_count - 1)
                / configuration_count
            )
            relative_standard_error[key] = math.sqrt(
                float(cp.sum(standard_error_squared).item())
                / max(
                    float(cp.sum(detector_means[key] ** 2).item()),
                    1.0e-30,
                )
            )
        else:
            relative_standard_error[key] = 0.0

    device_output = cp.stack(
        tuple(detector_means[key] for key in detector_keys)
        + (device_uncollected,),
        axis=0,
    )
    host_output = cp.asnumpy(device_output)
    elapsed_s = time.perf_counter() - started
    fractions = {
        key: np.asarray(host_output[index], dtype=np.float64)
        for index, key in enumerate(detector_keys)
    }
    uncollected = np.asarray(host_output[-1], dtype=np.float64)
    resident_potential_bytes = int(
        sum(array.nbytes for array in device_potentials)
    )
    fixed_device_bytes = int(
        device_spectrum.nbytes
        + device_frequency_x.nbytes
        + device_frequency_y.nbytes
        + device_fx.nbytes
        + device_fy.nbytes
        + device_scan_x.nbytes
        + device_scan_y.nbytes
        + sum(mask.nbytes for mask in device_masks.values())
        + sum(array.nbytes for array in detector_sums.values())
        + sum(array.nbytes for array in detector_sums_squared.values())
        + (
            sum(array.nbytes for array in phase_gratings)
            if phase_gratings is not None
            else 0
        )
        + (
            multislice_plan.cached_device_bytes
            if multislice_plan is not None
            else 0
        )
    )
    return ResidentStemCudaResult(
        fractions_flat=fractions,
        uncollected_flat=uncollected,
        detector_relative_standard_error=relative_standard_error,
        multislice_diagnostics=multislice_diagnostics,
        fft_diagnostics=WaveFftDiagnostics(
            compute_backend=WAVE_BACKEND_CUPY,
            numeric_precision="complex64 / float32",
            fallback_reason=fallback_reason,
        ),
        metrics={
            "cuda_resident_pipeline": True,
            "cuda_pipeline_elapsed_s": elapsed_s,
            "cuda_probe_batch_size": effective_batch_size,
            "cuda_probe_batch_count": batch_count,
            "cuda_configuration_count": configuration_count,
            "cuda_potential_upload_count": configuration_count,
            "cuda_resident_potential_bytes": resident_potential_bytes,
            "cuda_fixed_device_bytes": fixed_device_bytes,
            "cuda_bulk_host_transfer_count": 1,
            "cuda_bulk_host_transfer_bytes": int(host_output.nbytes),
            "cuda_multislice_diagnostic_count": diagnostic_count,
            "cuda_multislice_plan_reused": multislice_plan is not None,
            "cuda_multislice_plan_build_count": (
                1 if multislice_plan is not None else 0
            ),
            "cuda_multislice_plan_use_count": (
                multislice_plan.use_count
                if multislice_plan is not None
                else 0
            ),
            "cuda_multislice_plan_cached_bytes": (
                multislice_plan.cached_device_bytes
                if multislice_plan is not None
                else 0
            ),
            "cuda_multislice_cached_propagator_count": (
                multislice_plan.cached_propagator_count
                if multislice_plan is not None
                else 0
            ),
            "cuda_multislice_plan_build_elapsed_s": (
                multislice_plan.build_elapsed_s
                if multislice_plan is not None
                else 0.0
            ),
            "cuda_potential_phase_scan_count": potential_phase_scan_count,
        },
    )
