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

from temsim.optics.aberrations import (
    aberration_phase_rad,
    active_effective_aberrations,
)

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
    effective_sample_thickness_nm,
    interaction_constant_rad_per_v_angstrom,
    prepare_specimen_potentials,
)
from temsim.specimen.presets import (
    default_specimen_preset_key,
    load_specimen_preset,
)
from temsim.specimen.geometry import build_sample_geometry_snapshot


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
class PhysicalAngularDetector:
    """A real detector plane viewed through the signed 2-D angular transfer."""

    key: str
    detector: object
    sample_to_detector_m_per_rad: np.ndarray
    inner_mrad: float
    outer_mrad: float

    def validate(self):
        transfer = np.asarray(self.sample_to_detector_m_per_rad, dtype=float)
        if transfer.shape != (2, 2) or not np.all(np.isfinite(transfer)):
            raise ValueError(
                f"{self.key}: sample-to-detector transfer must be finite and 2 by 2."
            )
        if abs(float(np.linalg.det(transfer))) <= 1.0e-18:
            raise ValueError(f"{self.key}: sample-to-detector transfer is singular.")
        if not hasattr(self.detector, "hit_mask"):
            raise ValueError(f"{self.key}: physical detector has no hit-mask geometry.")
        if self.inner_mrad < 0.0 or self.outer_mrad <= self.inner_mrad:
            raise ValueError(f"{self.key}: invalid diagnostic collection-angle range.")
        return self

    def acceptance_mask(self, angle_x_mrad, angle_y_mrad):
        angle_x = np.asarray(angle_x_mrad, dtype=float) * 1.0e-3
        angle_y = np.asarray(angle_y_mrad, dtype=float) * 1.0e-3
        if angle_x.shape != angle_y.shape:
            raise ValueError(f"{self.key}: angular coordinate grids must match.")
        transfer = np.asarray(self.sample_to_detector_m_per_rad, dtype=float)
        x_mm = 1.0e3 * (
            transfer[0, 0] * angle_x + transfer[0, 1] * angle_y
        )
        y_mm = 1.0e3 * (
            transfer[1, 0] * angle_x + transfer[1, 1] * angle_y
        )
        return np.asarray(self.detector.hit_mask(x_mm, y_mm), dtype=bool)


@dataclass(frozen=True)
class AngleResolvedStemResult:
    scan_x_um: np.ndarray
    scan_y_um: np.ndarray
    fractions: dict[str, np.ndarray]
    detector_ranges_mrad: dict[str, tuple[float, float]]
    maximum_isotropic_angle_mrad: float
    uncollected_fraction: np.ndarray
    truncated_fraction: np.ndarray | None
    metrics: dict


def _detector_mask(detector, angles, angle_x_mrad=None, angle_y_mrad=None):
    if isinstance(detector, PhysicalAngularDetector):
        if angle_x_mrad is None or angle_y_mrad is None:
            raise ValueError(
                f"{detector.key}: physical detector integration needs signed angle grids."
            )
        return detector.acceptance_mask(angle_x_mrad, angle_y_mrad)
    return (
        (angles >= float(detector.inner_mrad))
        & (angles <= float(detector.outer_mrad))
    )


def integrate_angular_intensity(
    diffraction_intensity,
    scattering_angle_mrad,
    detectors,
    *,
    valid_mask=None,
    angle_x_mrad=None,
    angle_y_mrad=None,
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
        angular_band = _detector_mask(
            detector,
            angles,
            angle_x_mrad=angle_x_mrad,
            angle_y_mrad=angle_y_mrad,
        )
        mask = available & valid & angular_band
        masks[detector.key] = mask
        fractions[detector.key] = float(intensity[mask].sum())
        available[mask] = False
    return fractions, masks, float(intensity[available].sum())


def _wave_grid(state, simulation, scan_x_um, scan_y_um):
    sample_inserted = bool(getattr(state.sample, "inserted", True))
    atomic_interaction = (
        sample_inserted
        and str(getattr(state.sample, "specimen_mode", "atomic")).lower()
        == "atomic"
    )
    preset_key = (
        (
            str(state.sample.specimen_preset_key).strip()
            or default_specimen_preset_key()
        )
        if atomic_interaction
        else "vacuum"
    )
    preset = load_specimen_preset(preset_key)
    ray_stats = _weighted_ray_statistics(simulation.incident)
    probe_radius_99_nm = float(ray_stats["radius_99_m"]) * 1.0e9
    padding_factor = float(
        getattr(state.sample, "wave_probe_padding_factor", 3.0)
    )
    if not math.isfinite(padding_factor) or padding_factor < 0.0:
        raise ValueError("Wave probe-padding factor must be finite and non-negative.")
    padding_nm = padding_factor * probe_radius_99_nm
    geometry = build_sample_geometry_snapshot(
        state.sample,
        scan_x_um=scan_x_um,
        scan_y_um=scan_y_um,
        probe_padding_nm=padding_nm,
        load_atoms=False,
    )
    if geometry.calculation_roi_bounds_nm is None:
        raise RuntimeError("STEM calculation ROI was not constructed.")
    x0_nm, x1_nm, y0_nm, y1_nm = geometry.calculation_roi_bounds_nm
    span_x_nm = x1_nm - x0_nm
    span_y_nm = y1_nm - y0_nm
    configured_fov = float(
        getattr(state.sample, "wave_field_of_view_angstrom", 0.0)
    )
    derived_fov_angstrom = max(span_x_nm, span_y_nm, 1.0e-6) * 10.0
    requested_fov_angstrom = max(
        derived_fov_angstrom,
        configured_fov if configured_fov > 0.0 else 0.0,
        float(preset.field_of_view_angstrom),
    )
    roi_centre_nm = (
        0.5 * (x0_nm + x1_nm),
        0.5 * (y0_nm + y1_nm),
    )
    prepared = prepare_specimen_potentials(
        state,
        preset,
        field_of_view_angstrom_override=requested_fov_angstrom,
        calculation_roi_centre_nm=roi_centre_nm,
        calculation_roi_bounds_nm=geometry.calculation_roi_bounds_nm,
    )
    prepared.metrics["sample_geometry_snapshot"] = {
        "mode": geometry.mode,
        "inserted": geometry.inserted,
        "centre_nm": geometry.centre_nm,
        "size_nm": geometry.size_nm,
        "orientation_quaternion_wxyz": (
            geometry.orientation_quaternion_wxyz
        ),
        "scan_fov_bounds_nm": geometry.scan_fov_bounds_nm,
        "calculation_roi_bounds_nm": (
            geometry.calculation_roi_bounds_nm
        ),
    }
    return preset, prepared


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

    coefficients = active_effective_aberrations(state, "probe")
    chi = aberration_phase_rad(
        fx,
        fy,
        wavelength_angstrom,
        coefficients,
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

    preset, prepared = _wave_grid(
        state,
        simulation,
        scan_x_um,
        scan_y_um,
    )
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
    total_thickness_nm = effective_sample_thickness_nm(state)
    total_thickness_angstrom = total_thickness_nm * 10.0
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
    truncated_fraction = np.zeros(scan_x_um.shape, dtype=float)
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
        angle_x_mrad=angle_x_mrad,
        angle_y_mrad=angle_y_mrad,
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
    roi_centre_nm = prepared.metrics.get(
        "calculation_roi_centre_nm",
        (0.0, 0.0),
    )
    flat_x_angstrom = (
        origin_x_um
        + scan_x_um.ravel()
        - float(roi_centre_nm[0]) * 1.0e-3
    ) * 1.0e4
    flat_y_angstrom = (
        origin_y_um
        + scan_y_um.ravel()
        - float(roi_centre_nm[1]) * 1.0e-3
    ) * 1.0e4
    flat_fractions = {
        key: values.ravel() for key, values in fractions.items()
    }
    flat_uncollected = uncollected.ravel()
    flat_truncated = truncated_fraction.ravel()
    detector_sem_numerator = {detector.key: 0.0 for detector in detectors}
    detector_sem_denominator = {detector.key: 0.0 for detector in detectors}
    batch_size = min(8, flat_x_angstrom.size)
    resident_cuda_result = None
    resident_pipeline_metrics = {
        "cuda_resident_pipeline": False,
        "cuda_pipeline_fallback_reason": None,
    }
    if wave_backend == WAVE_BACKEND_CUPY and flat_detector_centers is not None:
        dynamic_reason = (
            "Pixel-dependent physical detector acceptance uses the complete "
            "NumPy reference observable; partial CPU/GPU frames are not mixed."
        )
        wave_backend = WAVE_BACKEND_NUMPY
        fft_backend = WAVE_BACKEND_NUMPY
        wave_fallback_reason = "; ".join(
            reason
            for reason in (wave_fallback_reason, dynamic_reason)
            if reason
        )
        fft_fallback_reason = wave_fallback_reason
        resident_pipeline_metrics["cuda_pipeline_fallback_reason"] = (
            dynamic_reason
        )
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
        configuration_truncated = []
        batch_detector_masks = detector_masks
        if flat_detector_centers is not None:
            available = np.ones(
                (stop - start, *valid_reciprocal.shape),
                dtype=bool,
            )
            batch_detector_masks = {}
            for detector in detectors:
                center_x, center_y = flat_detector_centers[detector.key]
                shifted_x = (
                    angle_x_mrad[None, :, :]
                    - center_x[start:stop, None, None]
                )
                shifted_y = (
                    angle_y_mrad[None, :, :]
                    - center_y[start:stop, None, None]
                )
                shifted_angle = np.hypot(shifted_x, shifted_y)
                angular_band = _detector_mask(
                    detector,
                    shifted_angle,
                    angle_x_mrad=shifted_x,
                    angle_y_mrad=shifted_y,
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
            configuration_truncated.append(
                np.sum(
                    diffraction[:, ~valid_reciprocal],
                    axis=1,
                )
            )
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
        flat_truncated[start:stop] = np.mean(
            np.stack(configuration_truncated, axis=0),
            axis=0,
        )

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
            "slice_count": 1 if total_thickness_nm > 0.0 else 0,
            "total_thickness_angstrom": total_thickness_angstrom,
            "slice_thickness_angstrom": total_thickness_angstrom,
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
    specimen_metrics["sample_inserted"] = bool(
        getattr(state.sample, "inserted", True)
    )
    specimen_metrics["sample_interaction_applied"] = bool(
        getattr(state.sample, "inserted", True)
        and str(getattr(state.sample, "specimen_mode", "atomic")).lower()
        == "atomic"
    )

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
        truncated_fraction=(
            None if resident_cuda_result is not None else truncated_fraction
        ),
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
            "angular_coverage_complete": not bool(truncated),
            "truncated_fraction_available": bool(
                resident_cuda_result is None
            ),
            "mean_truncated_fraction": (
                None
                if resident_cuda_result is not None
                else float(np.mean(truncated_fraction))
            ),
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
