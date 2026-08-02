"""Angle-resolved thin-phase STEM imaging.

This is the wave-optical bridge between the specimen exit wave and the
existing BF/DF/HAADF detector geometry.  It deliberately keeps specimen
propagation as one phase grating for now; multislice and frozen-phonon
propagation can replace that boundary without changing detector integration.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from temsim.physics.core import electron
from temsim.physics.wave_imaging import (
    _objective_defocus_angstrom,
    _weighted_ray_statistics,
    interaction_constant_rad_per_v_angstrom,
    projected_potential,
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
    pixels_override = int(getattr(state.sample, "wave_grid_pixels", 0))
    fov_override = float(
        getattr(state.sample, "wave_field_of_view_angstrom", 0.0)
    )
    x_axis, y_axis, potential = projected_potential(
        preset,
        state.sample.thickness_nm,
        pixels=pixels_override if pixels_override > 0 else None,
        field_of_view_angstrom=(
            fov_override if fov_override > 0.0 else None
        ),
    )
    return preset, x_axis, y_axis, potential


def _probe_spectrum(state, ray_stats, frequencies, wavelength_angstrom):
    fx, fy = np.meshgrid(frequencies, frequencies, indexing="xy")
    frequency_squared = fx * fx + fy * fy
    frequency_step = abs(float(frequencies[1] - frequencies[0]))
    convergence_rad = max(
        float(ray_stats["convergence_rms_rad"]),
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
):
    """Form thin-phase STEM images by integrating detector-angle bands."""
    detectors = tuple(detector.validate() for detector in detectors)
    if not detectors:
        raise ValueError("At least one angular STEM detector is required.")
    scan_x_um = np.asarray(scan_x_um, dtype=float)
    scan_y_um = np.asarray(scan_y_um, dtype=float)
    if scan_x_um.shape != scan_y_um.shape or scan_x_um.ndim != 2:
        raise ValueError("STEM scan coordinates must be matching 2-D arrays.")
    if scan_x_um.size == 0:
        raise ValueError("STEM scan coordinate arrays cannot be empty.")

    preset, x_axis, _, potential = _wave_grid(state)
    n = x_axis.size
    spacing = float(x_axis[1] - x_axis[0])
    frequencies = np.fft.fftshift(np.fft.fftfreq(n, d=spacing))
    fx, fy = np.meshgrid(frequencies, frequencies, indexing="xy")
    radial_frequency = np.hypot(fx, fy)
    _, _, wavelength_nm = electron(state)
    wavelength_angstrom = wavelength_nm * 10.0
    scattering_angle_mrad = np.arcsin(
        np.clip(wavelength_angstrom * radial_frequency, 0.0, 1.0)
    ) * 1.0e3
    maximum_isotropic_frequency = min(
        abs(float(frequencies[0])),
        abs(float(frequencies[-1])),
    )
    maximum_isotropic_angle_mrad = math.asin(
        min(wavelength_angstrom * maximum_isotropic_frequency, 1.0)
    ) * 1.0e3
    valid_reciprocal = (
        scattering_angle_mrad <= maximum_isotropic_angle_mrad
    )

    ray_stats = _weighted_ray_statistics(simulation.incident)
    base_spectrum = _probe_spectrum(
        state, ray_stats, frequencies, wavelength_angstrom
    )
    sigma = interaction_constant_rad_per_v_angstrom(state.beam_voltage_kv)
    phase_grating = np.exp(1j * sigma * potential)
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
    flat_x_angstrom = (origin_x_um + scan_x_um.ravel()) * 1.0e4
    flat_y_angstrom = (origin_y_um + scan_y_um.ravel()) * 1.0e4
    flat_fractions = {
        key: values.ravel() for key, values in fractions.items()
    }
    flat_uncollected = uncollected.ravel()
    batch_size = min(8, flat_x_angstrom.size)
    for start in range(0, flat_x_angstrom.size, batch_size):
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
        exit_wave = probe / probe_norm[:, None, None] * phase_grating
        diffraction = np.abs(
            np.fft.fftshift(
                np.fft.fft2(exit_wave, axes=(-2, -1)),
                axes=(-2, -1),
            )
        ) ** 2
        diffraction /= np.maximum(
            np.sum(diffraction, axis=(-2, -1), keepdims=True), 1.0e-30
        )
        collected = np.zeros(stop - start, dtype=float)
        for detector in detectors:
            values = np.sum(
                diffraction[:, detector_masks[detector.key]], axis=1
            )
            flat_fractions[detector.key][start:stop] = values
            collected += values
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
    field_of_view_angstrom = spacing * n
    return AngleResolvedStemResult(
        scan_x_um=scan_x_um,
        scan_y_um=scan_y_um,
        fractions=fractions,
        detector_ranges_mrad=detector_ranges,
        maximum_isotropic_angle_mrad=maximum_isotropic_angle_mrad,
        uncollected_fraction=uncollected,
        metrics={
            "model": "thin_phase_angle_resolved",
            "preset_key": preset.key,
            "wavelength_angstrom": wavelength_angstrom,
            "grid_pixels": n,
            "pixel_size_angstrom": spacing,
            "field_of_view_angstrom": field_of_view_angstrom,
            "scan_span_x_angstrom": scan_span_x_angstrom,
            "scan_span_y_angstrom": scan_span_y_angstrom,
            "scan_exceeds_periodic_field_of_view": bool(
                max(scan_span_x_angstrom, scan_span_y_angstrom)
                > field_of_view_angstrom
            ),
            "maximum_isotropic_angle_mrad": maximum_isotropic_angle_mrad,
            "truncated_detector_keys": truncated,
            "multislice_enabled": False,
            "rutherford_tail_enabled": False,
        },
    )
