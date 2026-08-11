"""Angle-resolved STEM wave imaging.

This is the wave-optical bridge between the specimen exit wave and the
existing BF/DF/HAADF detector geometry.  It supports symmetric multislice,
atomistic finite-projection potential slices, frozen-phonon intensity
ensembles, and a fast projected phase object.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from temsim.physics.compute_backend import (
    WAVE_BACKEND_CUPY,
    WAVE_BACKEND_NUMPY,
    choose_wave_backend,
)
from temsim.physics.core import electron
from temsim.physics.multislice import propagate_multislice
from temsim.physics.stem_cuda_pipeline import (
    release_cupy_memory_pools,
    run_resident_stem_cuda,
)
from temsim.physics.wave_fft import stem_diffraction_intensity
from temsim.physics.wave_imaging import (
    _objective_defocus_angstrom,
    _weighted_ray_statistics,
    interaction_constant_rad_per_v_angstrom,
    prepare_specimen_potentials,
)
from temsim.specimen.presets import (
    default_specimen_preset_key,
    load_specimen_preset,
)


@dataclass(frozen=True)
class AngularDetector:
    key: str
    inner_mrad: float
    outer_mrad: float

    def validate(self):
        if self.inner_mrad < 0.0:
            raise ValueError(f"{self.key}: inner collection angle cannot be negative.")
        if self.outer_mrad <= self.inner_mrad:
            raise ValueError(
                f"{self.key}: outer collection angle must exceed the inner angle."
            )
        return self


@dataclass(frozen=True)
class AngleResolvedStemResult:
    scan_x_um: np.ndarray
    scan_y_um: np.ndarray
    fractions: dict[str, np.ndarray]
    detector_ranges_mrad: dict[str, tuple[float, float]]
    maximum_isotropic_angle_mrad: float
    uncollected_fraction: np.ndarray
    metrics: dict


def integrate_angular_intensity(
    diffraction_intensity,
    scattering_angle_mrad,
    detectors,
    *,
    valid_mask=None,
):
    """Integrate sequential detector bands without double counting overlap."""
    intensity = np.asarray(diffraction_intensity, dtype=float)
    angles = np.asarray(scattering_angle_mrad, dtype=float)
    if intensity.shape != angles.shape:
        raise ValueError("Diffraction intensity and angle grids must match.")
    if valid_mask is None:
        valid = np.ones_like(intensity, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != intensity.shape:
            raise ValueError("Reciprocal-space validity mask has the wrong shape.")
    intensity = np.maximum(intensity, 0.0)
    total = float(intensity.sum())
    if total <= 0.0:
        raise ValueError("Diffraction intensity must contain positive weight.")
    intensity = intensity / total
    available = np.ones_like(intensity, dtype=bool)
    fractions = {}
    masks = {}
    for detector in detectors:
        detector.validate()
        angular_band = (
            (angles >= float(detector.inner_mrad))
            & (angles <= float(detector.outer_mrad))
        )
        mask = available & valid & angular_band
        masks[detector.key] = mask
        fractions[detector.key] = float(intensity[mask].sum())
        available[mask] = False
    return fractions, masks, float(intensity[available].sum())


def _wave_grid(state):
    preset_key = (
        str(state.sample.specimen_preset_key).strip()
        or default_specimen_preset_key()
    )
    preset = load_specimen_preset(preset_key)
    return preset, prepare_specimen_potentials(state, preset)


def _probe_spectrum(
    state,
    ray_stats,
    frequencies_x,
    frequencies_y,
    wavelength_angstrom,
):
    fx, fy = np.meshgrid(frequencies_x, frequencies_y, indexing="xy")
    frequency_squared = fx * fx + fy * fy
    frequency_step = max(
        abs(float(frequencies_x[1] - frequencies_x[0])),
        abs(float(frequencies_y[1] - frequencies_y[0])),
    )
    convergence_rad = max(
        float(ray_stats["convergence_semiangle_rad"]),
        frequency_step * wavelength_angstrom,
    )
    centre_fx = math.sin(float(ray_stats["mean_tx_rad"])) / wavelength_angstrom
    centre_fy = math.sin(float(ray_stats["mean_ty_rad"])) / wavelength_angstrom
    aperture_radius = math.sin(convergence_rad) / wavelength_angstrom
    aperture = (
        (fx - centre_fx) ** 2 + (fy - centre_fy) ** 2
        <= aperture_radius**2
    )
    if not np.any(aperture):
        nearest = np.unravel_index(
            np.argmin((fx - centre_fx) ** 2 + (fy - centre_fy) ** 2),
            fx.shape,
        )
        aperture[nearest] = True

    defocus_angstrom, _ = _objective_defocus_angstrom(state)
    if not math.isfinite(defocus_angstrom):
        defocus_angstrom = 0.0
    cs_angstrom = float(getattr(state.objective_lens, "cs_mm", 0.0) or 0.0) * 1.0e7
    chi = math.pi * wavelength_angstrom * defocus_angstrom * frequency_squared
    chi += (
        0.5
        * math.pi
        * cs_angstrom
        * wavelength_angstrom**3
        * frequency_squared**2
    )
    return aperture.astype(complex) * np.exp(-1j * chi)


def simulate_angle_resolved_stem(
    state,
    simulation,
    detectors,
    scan_x_um,
    scan_y_um,
    *,
    baseline_scan_offset_um=(0.0, 0.0),
    detector_center_shifts_mrad=None,
):
    """Form STEM images by integrating detector-angle bands."""
    detectors = tuple(detector.validate() for detector in detectors)
    if not detectors:
        raise ValueError("At least one angular STEM detector is required.")
    scan_x_um = np.asarray(scan_x_um, dtype=float)
    scan_y_um = np.asarray(scan_y_um, dtype=float)
    if scan_x_um.shape != scan_y_um.shape or scan_x_um.ndim != 2:
        raise ValueError("STEM scan coordinates must be matching 2-D arrays.")
    if scan_x_um.size == 0:
        raise ValueError("STEM scan coordinate arrays cannot be empty.")

    preset, prepared = _wave_grid(state)
    x_axis = prepared.x_angstrom
    y_axis = prepared.y_angstrom
    potential = prepared.mean_projected_potential_v_angstrom
    nx = x_axis.size
    ny = y_axis.size
    spacing_x = float(x_axis[1] - x_axis[0])
    spacing_y = float(y_axis[1] - y_axis[0])
    frequencies_x = np.fft.fftshift(np.fft.fftfreq(nx, d=spacing_x))
    frequencies_y = np.fft.fftshift(np.fft.fftfreq(ny, d=spacing_y))
    fx, fy = np.meshgrid(frequencies_x, frequencies_y, indexing="xy")
    radial_frequency = np.hypot(fx, fy)
    _, _, wavelength_nm = electron(state)
    wavelength_angstrom = wavelength_nm * 10.0
    scattering_angle_mrad = np.arcsin(
        np.clip(wavelength_angstrom * radial_frequency, 0.0, 1.0)
    ) * 1.0e3
    maximum_isotropic_frequency = min(
        abs(float(frequencies_x[0])),
        abs(float(frequencies_x[-1])),
        abs(float(frequencies_y[0])),
        abs(float(frequencies_y[-1])),
    )
    maximum_isotropic_angle_mrad = math.asin(
        min(wavelength_angstrom * maximum_isotropic_frequency, 1.0)
    ) * 1.0e3
    multislice_enabled = bool(
        getattr(state.sample, "wave_multislice_enabled", True)
    )
    total_thickness_angstrom = max(
        float(state.sample.thickness_nm), 0.0
    ) * 10.0
    target_slice_angstrom = float(
        getattr(state.sample, "wave_slice_thickness_angstrom", 2.0)
    )
    if prepared.slice_thicknesses_angstrom is not None:
        estimated_slices = int(prepared.slice_thicknesses_angstrom.size)
    else:
        estimated_slices = (
            max(
                1,
                int(
                    math.ceil(
                        total_thickness_angstrom / target_slice_angstrom
                    )
                ),
            )
            if (
                multislice_enabled
                and total_thickness_angstrom > 0.0
                and math.isfinite(target_slice_angstrom)
                and target_slice_angstrom > 0.0
            )
            else 1
        )
    configuration_count = len(
        prepared.potential_configurations_v_angstrom
    )
    wave_backend, wave_fallback_reason = choose_wave_backend(
        getattr(state, "acceleration_backend", "Auto"),
        acceleration_enabled=bool(
            getattr(state, "acceleration_enabled", True)
        ),
        work_items=(
            nx
            * ny
            * estimated_slices
            * scan_x_um.size
            * configuration_count
        ),
    )
    if multislice_enabled:
        multislice_frequency = (
            float(getattr(state.sample, "wave_bandwidth_fraction", 2.0 / 3.0))
            * min(0.5 / spacing_x, 0.5 / spacing_y)
        )
        maximum_isotropic_angle_mrad = min(
            maximum_isotropic_angle_mrad,
            math.asin(
                min(wavelength_angstrom * multislice_frequency, 1.0)
            ) * 1.0e3,
        )
    valid_reciprocal = (
        scattering_angle_mrad <= maximum_isotropic_angle_mrad
    )
    angle_x_mrad = np.arcsin(
        np.clip(wavelength_angstrom * fx, -1.0, 1.0)
    ) * 1.0e3
    angle_y_mrad = np.arcsin(
        np.clip(wavelength_angstrom * fy, -1.0, 1.0)
    ) * 1.0e3

    ray_stats = _weighted_ray_statistics(simulation.incident)
    base_spectrum = _probe_spectrum(
        state,
        ray_stats,
        frequencies_x,
        frequencies_y,
        wavelength_angstrom,
    )
    sigma = interaction_constant_rad_per_v_angstrom(state.beam_voltage_kv)
    phase_grating = (
        None if multislice_enabled else np.exp(1j * sigma * potential)
    )
    diagnostic_records = []
    fft_records = []
    fft_backend = wave_backend
    fft_fallback_reason = wave_fallback_reason
    fractions = {
        detector.key: np.zeros(scan_x_um.shape, dtype=float)
        for detector in detectors
    }
    uncollected = np.zeros(scan_x_um.shape, dtype=float)
    origin_x_um = float(ray_stats["mean_x_m"]) * 1.0e6 - float(
        baseline_scan_offset_um[0]
    )
    origin_y_um = float(ray_stats["mean_y_m"]) * 1.0e6 - float(
        baseline_scan_offset_um[1]
    )
    _, detector_masks, _ = integrate_angular_intensity(
        np.ones_like(scattering_angle_mrad),
        scattering_angle_mrad,
        detectors,
        valid_mask=valid_reciprocal,
    )
    flat_detector_centers = None
    if detector_center_shifts_mrad:
        flat_detector_centers = {}
        for detector in detectors:
            values = detector_center_shifts_mrad.get(detector.key)
            if values is None:
                center_x = np.zeros(scan_x_um.shape, dtype=float)
                center_y = np.zeros(scan_y_um.shape, dtype=float)
            else:
                center_x = np.asarray(values[0], dtype=float)
                center_y = np.asarray(values[1], dtype=float)
                if (
                    center_x.shape != scan_x_um.shape
                    or center_y.shape != scan_y_um.shape
                ):
                    raise ValueError(
                        f"{detector.key}: detector-centre shift must match "
                        "the STEM raster shape."
                    )
                if not (
                    np.all(np.isfinite(center_x))
                    and np.all(np.isfinite(center_y))
                ):
                    raise ValueError(
                        f"{detector.key}: detector-centre shift must be finite."
                    )
            flat_detector_centers[detector.key] = (
                center_x.ravel(),
                center_y.ravel(),
            )
    flat_x_angstrom = (origin_x_um + scan_x_um.ravel()) * 1.0e4
    flat_y_angstrom = (origin_y_um + scan_y_um.ravel()) * 1.0e4
    flat_fractions = {
        key: values.ravel() for key, values in fractions.items()
    }
    flat_uncollected = uncollected.ravel()
    detector_sem_numerator = {detector.key: 0.0 for detector in detectors}
    detector_sem_denominator = {detector.key: 0.0 for detector in detectors}
    batch_size = min(8, flat_x_angstrom.size)
    resident_cuda_result = None
    resident_pipeline_metrics = {
        "cuda_resident_pipeline": False,
        "cuda_pipeline_fallback_reason": None,
    }
    if wave_backend == WAVE_BACKEND_CUPY and flat_detector_centers is None:
        try:
            resident_cuda_result = run_resident_stem_cuda(
                base_spectrum=base_spectrum,
                frequencies_x=frequencies_x,
                frequencies_y=frequencies_y,
                scan_x_angstrom=flat_x_angstrom,
                scan_y_angstrom=flat_y_angstrom,
                potential_configurations_v_angstrom=(
                    prepared.potential_configurations_v_angstrom
                ),
                detector_masks=detector_masks,
                multislice_enabled=multislice_enabled,
                pixel_size_angstrom=(spacing_y, spacing_x),
                wavelength_angstrom=wavelength_angstrom,
                interaction_constant_rad_per_v_angstrom=sigma,
                total_thickness_angstrom=total_thickness_angstrom,
                target_slice_thickness_angstrom=target_slice_angstrom,
                slice_thicknesses_angstrom=(
                    prepared.slice_thicknesses_angstrom
                ),
                bandwidth_fraction=float(
                    getattr(
                        state.sample,
                        "wave_bandwidth_fraction",
                        2.0 / 3.0,
                    )
                ),
                batch_size=batch_size,
                fallback_reason=wave_fallback_reason,
            )
        except Exception as exc:
            cuda_failure = (
                "Resident CuPy STEM pipeline failed: "
                f"{type(exc).__name__}: {exc}"
            )
            wave_fallback_reason = "; ".join(
                dict.fromkeys(
                    reason
                    for reason in (wave_fallback_reason, cuda_failure)
                    if reason
                )
            )
            fft_fallback_reason = wave_fallback_reason
            wave_backend = WAVE_BACKEND_NUMPY
            fft_backend = WAVE_BACKEND_NUMPY
            resident_pipeline_metrics[
                "cuda_pipeline_fallback_reason"
            ] = cuda_failure
            release_cupy_memory_pools()
        else:
            for key, values in resident_cuda_result.fractions_flat.items():
                flat_fractions[key][:] = values
            flat_uncollected[:] = resident_cuda_result.uncollected_flat
            if resident_cuda_result.multislice_diagnostics is not None:
                diagnostic_records.append(
                    asdict(
                        resident_cuda_result.multislice_diagnostics
                    )
                )
            fft_records.append(resident_cuda_result.fft_diagnostics)
            resident_pipeline_metrics.update(resident_cuda_result.metrics)

    batch_starts = (
        ()
        if resident_cuda_result is not None
        else range(0, flat_x_angstrom.size, batch_size)
    )
    for start in batch_starts:
        stop = min(start + batch_size, flat_x_angstrom.size)
        x0 = flat_x_angstrom[start:stop, None, None]
        y0 = flat_y_angstrom[start:stop, None, None]
        shifted_spectrum = base_spectrum[None, :, :] * np.exp(
            -2j * math.pi * (fx[None, :, :] * x0 + fy[None, :, :] * y0)
        )
        probe = np.fft.ifft2(
            np.fft.ifftshift(shifted_spectrum, axes=(-2, -1)),
            axes=(-2, -1),
        )
        probe_norm = np.sqrt(
            np.maximum(np.sum(np.abs(probe) ** 2, axis=(-2, -1)), 1.0e-30)
        )
        normalised_probe = probe / probe_norm[:, None, None]
        configuration_values = {
            detector.key: [] for detector in detectors
        }
        batch_detector_masks = detector_masks
        if flat_detector_centers is not None:
            available = np.ones(
                (stop - start, *valid_reciprocal.shape),
                dtype=bool,
            )
            batch_detector_masks = {}
            for detector in detectors:
                center_x, center_y = flat_detector_centers[detector.key]
                shifted_angle = np.hypot(
                    angle_x_mrad[None, :, :]
                    - center_x[start:stop, None, None],
                    angle_y_mrad[None, :, :]
                    - center_y[start:stop, None, None],
                )
                angular_band = (
                    shifted_angle >= float(detector.inner_mrad)
                ) & (
                    shifted_angle <= float(detector.outer_mrad)
                )
                mask = (
                    available
                    & valid_reciprocal[None, :, :]
                    & angular_band
                )
                batch_detector_masks[detector.key] = mask
                available &= ~mask
        for configuration in prepared.potential_configurations_v_angstrom:
            if multislice_enabled:
                explicit_slices = configuration.ndim == 3
                exit_wave, diagnostics = propagate_multislice(
                    normalised_probe,
                    configuration,
                    pixel_size_angstrom=(spacing_y, spacing_x),
                    wavelength_angstrom=wavelength_angstrom,
                    interaction_constant_rad_per_v_angstrom=sigma,
                    total_thickness_angstrom=(
                        None if explicit_slices else total_thickness_angstrom
                    ),
                    target_slice_thickness_angstrom=target_slice_angstrom,
                    slice_thicknesses_angstrom=(
                        prepared.slice_thicknesses_angstrom
                        if explicit_slices else None
                    ),
                    bandwidth_fraction=float(
                        getattr(
                            state.sample,
                            "wave_bandwidth_fraction",
                            2.0 / 3.0,
                        )
                    ),
                    compute_backend=wave_backend,
                    fallback_reason=wave_fallback_reason,
                )
                diagnostic_records.append(asdict(diagnostics))
                if diagnostics.compute_backend != wave_backend:
                    # The failed CUDA job was already recomputed on the CPU.
                    # Keep later configurations and batches on that fallback.
                    wave_backend = diagnostics.compute_backend
                    wave_fallback_reason = diagnostics.fallback_reason
                    fft_backend = diagnostics.compute_backend
                    fft_fallback_reason = diagnostics.fallback_reason
            else:
                exit_wave = normalised_probe * phase_grating

            diffraction, fft_diagnostics = stem_diffraction_intensity(
                exit_wave,
                compute_backend=fft_backend,
                fallback_reason=fft_fallback_reason,
            )
            fft_records.append(fft_diagnostics)
            if fft_diagnostics.compute_backend != fft_backend:
                fft_backend = fft_diagnostics.compute_backend
                fft_fallback_reason = fft_diagnostics.fallback_reason
            for detector in detectors:
                if flat_detector_centers is None:
                    values = np.sum(
                        diffraction[:, batch_detector_masks[detector.key]],
                        axis=1,
                    )
                else:
                    values = np.sum(
                        diffraction
                        * batch_detector_masks[detector.key],
                        axis=(-2, -1),
                    )
                configuration_values[detector.key].append(
                    np.clip(values, 0.0, 1.0)
                )

        collected = np.zeros(stop - start, dtype=float)
        for detector in detectors:
            samples = np.stack(
                configuration_values[detector.key], axis=0
            )
            values = np.mean(samples, axis=0)
            flat_fractions[detector.key][start:stop] = values
            collected += values
            if configuration_count > 1:
                standard_error = (
                    np.std(samples, axis=0, ddof=1)
                    / math.sqrt(configuration_count)
                )
                detector_sem_numerator[detector.key] += float(
                    np.sum(standard_error**2)
                )
                detector_sem_denominator[detector.key] += float(
                    np.sum(values**2)
                )
        flat_uncollected[start:stop] = np.maximum(1.0 - collected, 0.0)

    detector_ranges = {
        detector.key: (detector.inner_mrad, detector.outer_mrad)
        for detector in detectors
    }
    truncated = tuple(
        detector.key
        for detector in detectors
        if detector.outer_mrad > maximum_isotropic_angle_mrad
    )
    scan_span_x_angstrom = float(np.ptp(scan_x_um)) * 1.0e4
    scan_span_y_angstrom = float(np.ptp(scan_y_um)) * 1.0e4
    field_of_view_x_angstrom = spacing_x * nx
    field_of_view_y_angstrom = spacing_y * ny
    if diagnostic_records:
        specimen_metrics = dict(diagnostic_records[0])
        for key in (
            "maximum_phase_per_slice_rad",
            "maximum_relative_intensity_change",
        ):
            specimen_metrics[key] = max(
                float(record[key]) for record in diagnostic_records
            )
        backends = {
            str(record["compute_backend"]) for record in diagnostic_records
        }
        precisions = {
            str(record["numeric_precision"]) for record in diagnostic_records
        }
        fallback_reasons = tuple(
            dict.fromkeys(
                str(record["fallback_reason"])
                for record in diagnostic_records
                if record.get("fallback_reason")
            )
        )
        specimen_metrics["compute_backend"] = (
            backends.pop()
            if len(backends) == 1
            else "Mixed (NumPy CPU + CuPy CUDA)"
        )
        specimen_metrics["numeric_precision"] = (
            precisions.pop() if len(precisions) == 1 else "mixed"
        )
        specimen_metrics["fallback_reason"] = (
            "; ".join(fallback_reasons) or None
        )
        if prepared.metrics["atomistic_applied"]:
            specimen_metrics["model"] = (
                "atomistic_frozen_phonon_multislice"
                if prepared.metrics["frozen_phonon_applied"]
                else "atomistic_static_multislice"
            )
    else:
        specimen_metrics = {
            "model": "projected_phase_object",
            "slice_count": 1 if state.sample.thickness_nm > 0.0 else 0,
            "total_thickness_angstrom": max(
                float(state.sample.thickness_nm), 0.0
            ) * 10.0,
            "slice_thickness_angstrom": max(
                float(state.sample.thickness_nm), 0.0
            ) * 10.0,
            "bandwidth_fraction": 1.0,
            "maximum_isotropic_angle_mrad": maximum_isotropic_angle_mrad,
            "maximum_phase_per_slice_rad": float(
                np.max(np.abs(sigma * potential))
            ),
            "maximum_relative_intensity_change": 0.0,
            "compute_backend": (
                resident_cuda_result.fft_diagnostics.compute_backend
                if resident_cuda_result is not None
                else WAVE_BACKEND_NUMPY
            ),
            "numeric_precision": (
                resident_cuda_result.fft_diagnostics.numeric_precision
                if resident_cuda_result is not None
                else "complex128 / float64"
            ),
            "fallback_reason": (
                resident_cuda_result.fft_diagnostics.fallback_reason
                if resident_cuda_result is not None
                else wave_fallback_reason
            ),
            "pixel_size_y_angstrom": spacing_y,
            "pixel_size_x_angstrom": spacing_x,
        }
    specimen_metrics.update(prepared.metrics)

    fft_backends = {record.compute_backend for record in fft_records}
    fft_precisions = {record.numeric_precision for record in fft_records}
    fft_reasons = tuple(
        dict.fromkeys(
            str(record.fallback_reason)
            for record in fft_records
            if record.fallback_reason
        )
    )
    fft_compute_backend = (
        fft_backends.pop()
        if len(fft_backends) == 1
        else "Mixed (NumPy CPU + CuPy CUDA)"
    )
    fft_numeric_precision = (
        fft_precisions.pop() if len(fft_precisions) == 1 else "mixed"
    )
    fft_fallback_reason = "; ".join(fft_reasons) or None
    actual_backends = {
        str(specimen_metrics["compute_backend"]),
        fft_compute_backend,
    }
    wave_compute_backend = (
        actual_backends.pop()
        if len(actual_backends) == 1
        else "Mixed (NumPy CPU + CuPy CUDA)"
    )
    if resident_cuda_result is not None:
        detector_relative_standard_error = (
            resident_cuda_result.detector_relative_standard_error
        )
    else:
        detector_relative_standard_error = {
            detector.key: (
                math.sqrt(
                    detector_sem_numerator[detector.key]
                    / max(
                        detector_sem_denominator[detector.key],
                        1.0e-30,
                    )
                )
                if configuration_count > 1
                else 0.0
            )
            for detector in detectors
        }
    return AngleResolvedStemResult(
        scan_x_um=scan_x_um,
        scan_y_um=scan_y_um,
        fractions=fractions,
        detector_ranges_mrad=detector_ranges,
        maximum_isotropic_angle_mrad=maximum_isotropic_angle_mrad,
        uncollected_fraction=uncollected,
        metrics={
            "model": (
                "multislice_angle_resolved"
                if multislice_enabled else "thin_phase_angle_resolved"
            ),
            **{
                f"specimen_{key}": value
                for key, value in specimen_metrics.items()
            },
            "preset_key": preset.key,
            "wavelength_angstrom": wavelength_angstrom,
            "grid_pixels": max(nx, ny),
            "grid_pixels_x": nx,
            "grid_pixels_y": ny,
            "pixel_size_angstrom": max(spacing_x, spacing_y),
            "pixel_size_x_angstrom": spacing_x,
            "pixel_size_y_angstrom": spacing_y,
            "field_of_view_angstrom": max(
                field_of_view_x_angstrom,
                field_of_view_y_angstrom,
            ),
            "field_of_view_x_angstrom": field_of_view_x_angstrom,
            "field_of_view_y_angstrom": field_of_view_y_angstrom,
            "scan_span_x_angstrom": scan_span_x_angstrom,
            "scan_span_y_angstrom": scan_span_y_angstrom,
            "scan_exceeds_periodic_field_of_view": bool(
                scan_span_x_angstrom > field_of_view_x_angstrom
                or scan_span_y_angstrom > field_of_view_y_angstrom
            ),
            "maximum_isotropic_angle_mrad": maximum_isotropic_angle_mrad,
            "truncated_detector_keys": truncated,
            "wave_sampling_truncates_illumination": bool(
                ray_stats["convergence_semiangle_rad"] * 1.0e3
                > maximum_isotropic_angle_mrad
            ),
            "wave_intensity_conservation_within_0_1_percent": bool(
                float(
                    specimen_metrics["maximum_relative_intensity_change"]
                ) <= 1.0e-3
            ),
            "multislice_enabled": multislice_enabled,
            "rutherford_tail_enabled": False,
            "descan_detector_shift_applied": bool(
                flat_detector_centers is not None
            ),
            "displayed_intensity_average": (
                "incoherent frozen-phonon intensity mean"
                if configuration_count > 1
                else "single configuration"
            ),
            "detector_configuration_relative_standard_error": (
                detector_relative_standard_error
            ),
            "fft_compute_backend": fft_compute_backend,
            "fft_numeric_precision": fft_numeric_precision,
            "fft_fallback_reason": fft_fallback_reason,
            "wave_compute_backend": wave_compute_backend,
            **resident_pipeline_metrics,
        },
    )
