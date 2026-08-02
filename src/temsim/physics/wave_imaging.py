"""Local coherent wave optics from the specimen to the Objective image.

The full microscope remains a ray model.  This module converts the surviving
ray bundle at the specimen into an incident complex wave, applies a TOML-owned
projected specimen potential, and transfers the exit wave through an Objective
CTF.  It is intentionally compact and is a foundation for later multislice
potentials, not a first-principles scattering package.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from temsim.physics.core import C, E, H, M, electron
from temsim.specimen.presets import (
    SpecimenPreset,
    default_specimen_preset_key,
    load_specimen_preset,
)


@dataclass(frozen=True)
class WaveImagingResult:
    preset_key: str
    preset_name: str
    x_angstrom: np.ndarray
    y_angstrom: np.ndarray
    projected_potential_v_angstrom: np.ndarray
    exit_wave: np.ndarray
    diffraction_intensity: np.ndarray
    image_intensity: np.ndarray
    spatial_frequency_inv_angstrom: np.ndarray
    metrics: dict


def _normalise_image(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    low, high = np.percentile(values, (0.5, 99.5))
    if high <= low + 1.0e-30:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def projected_potential(
    preset: SpecimenPreset,
    thickness_nm: float,
    *,
    pixels: int | None = None,
    field_of_view_angstrom: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate the periodic projected potential defined by one TOML preset."""
    n = int(pixels or preset.pixels)
    fov = float(field_of_view_angstrom or preset.field_of_view_angstrom)
    if n < 32 or fov <= 0.0:
        raise ValueError("Wave grid must contain at least 32 pixels and a positive FOV.")
    spacing = fov / n
    axis = (np.arange(n, dtype=float) - n // 2) * spacing
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    potential = np.zeros((n, n), dtype=float)
    thickness_scale = max(float(thickness_nm), 0.0) / preset.reference_thickness_nm
    ax = preset.unit_cell_x_angstrom
    ay = preset.unit_cell_y_angstrom
    for column in preset.columns:
        x0 = column.x_fraction * ax
        y0 = column.y_fraction * ay
        dx = np.mod(xx - x0 + 0.5 * ax, ax) - 0.5 * ax
        dy = np.mod(yy - y0 + 0.5 * ay, ay) - 0.5 * ay
        gaussian = np.exp(-0.5 * (dx * dx + dy * dy) / column.sigma_angstrom**2)
        potential += (
            thickness_scale
            * column.occupancy
            * column.projected_potential_v_angstrom
            * gaussian
        )
    return axis, axis.copy(), potential


def interaction_constant_rad_per_v_angstrom(voltage_kv: float) -> float:
    """Relativistic phase-object interaction constant in rad/(V Angstrom)."""
    kinetic_j = E * float(voltage_kv) * 1000.0
    gamma = 1.0 + kinetic_j / (M * C * C)
    momentum = math.sqrt(kinetic_j * kinetic_j + 2.0 * kinetic_j * M * C * C) / C
    wavelength_m = H / momentum
    return 2.0 * math.pi * gamma * M * E * wavelength_m / (H * H) * 1.0e-10


def _weighted_ray_statistics(incident) -> dict:
    alive = np.asarray(incident.alive, dtype=bool)
    finite = (
        np.isfinite(incident.x[-1])
        & np.isfinite(incident.y[-1])
        & np.isfinite(incident.tx[-1])
        & np.isfinite(incident.ty[-1])
    )
    mask = alive & finite
    if not np.any(mask):
        raise ValueError("No surviving rays reach the specimen for wave imaging.")
    if incident.ray_weight is None:
        weights = np.ones(np.count_nonzero(mask), dtype=float)
    else:
        weights = np.asarray(incident.ray_weight, dtype=float)[mask]
    weights = np.maximum(weights, 0.0)
    if float(np.sum(weights)) <= 0.0:
        weights = np.ones_like(weights)
    weights /= np.sum(weights)

    def mean(values):
        return float(np.sum(np.asarray(values, dtype=float)[mask] * weights))

    mean_tx = mean(incident.tx[-1])
    mean_ty = mean(incident.ty[-1])
    delta_tx = np.asarray(incident.tx[-1], dtype=float)[mask] - mean_tx
    delta_ty = np.asarray(incident.ty[-1], dtype=float)[mask] - mean_ty
    convergence = math.sqrt(float(np.sum(weights * (delta_tx**2 + delta_ty**2))))
    return {
        "mean_x_m": mean(incident.x[-1]),
        "mean_y_m": mean(incident.y[-1]),
        "mean_tx_rad": mean_tx,
        "mean_ty_rad": mean_ty,
        "convergence_rms_rad": convergence,
        "surviving_rays": int(np.count_nonzero(mask)),
    }


def _incident_wave(
    state,
    ray_stats: dict,
    frequencies: np.ndarray,
    wavelength_angstrom: float,
) -> np.ndarray:
    n = frequencies.size
    fx, fy = np.meshgrid(frequencies, frequencies, indexing="xy")
    tilt_fx = ray_stats["mean_tx_rad"] / wavelength_angstrom
    tilt_fy = ray_stats["mean_ty_rad"] / wavelength_angstrom
    if str(getattr(state, "illumination_mode", "TEM")).upper() != "STEM":
        spacing = 1.0 / (n * (frequencies[1] - frequencies[0]))
        axis = (np.arange(n, dtype=float) - n // 2) * spacing
        xx, yy = np.meshgrid(axis, axis, indexing="xy")
        return np.exp(2j * math.pi * (tilt_fx * xx + tilt_fy * yy))

    alpha = max(
        ray_stats["convergence_rms_rad"],
        abs(frequencies[1] - frequencies[0]) * wavelength_angstrom,
    )
    radius = alpha / wavelength_angstrom
    aperture = ((fx - tilt_fx) ** 2 + (fy - tilt_fy) ** 2) <= radius**2
    if not np.any(aperture):
        nearest = np.unravel_index(
            np.argmin((fx - tilt_fx) ** 2 + (fy - tilt_fy) ** 2), fx.shape
        )
        aperture[nearest] = True
    fov = 1.0 / abs(frequencies[1] - frequencies[0])
    x0 = math.remainder(ray_stats["mean_x_m"] * 1.0e10, fov)
    y0 = math.remainder(ray_stats["mean_y_m"] * 1.0e10, fov)
    spectrum = aperture.astype(complex) * np.exp(-2j * math.pi * (fx * x0 + fy * y0))
    wave = np.fft.ifft2(np.fft.ifftshift(spectrum))
    return wave / math.sqrt(max(float(np.mean(np.abs(wave) ** 2)), 1.0e-30))


def _objective_aperture_rad(state) -> float:
    aperture = state.objective_aperture
    if not bool(getattr(aperture, "enabled", True)):
        return math.inf
    distance_mm = abs(float(aperture.z_mm) - float(state.sample.z_mm))
    if distance_mm <= 1.0e-12:
        return math.inf
    return max(float(aperture.radius_mm), 0.0) / distance_mm


def _objective_defocus_angstrom(state) -> tuple[float, float]:
    objective = state.objective_lens
    current_focal_mm = float(
        objective.focal_length_for_voltage_mm(state.beam_voltage_kv)
    )
    nominal_focal_mm = float(objective.nominal_focal_length_mm)
    user_defocus_nm = float(getattr(state.sample, "wave_defocus_nm", 0.0))
    # A changed Objective excitation changes its focal length.  Referencing the
    # phase plate to the TOML nominal calibration makes this visible directly.
    if not math.isfinite(current_focal_mm):
        return math.inf, current_focal_mm
    defocus_angstrom = user_defocus_nm * 10.0 + (current_focal_mm - nominal_focal_mm) * 1.0e7
    return defocus_angstrom, current_focal_mm


def simulate_wave_image(state, simulation) -> WaveImagingResult:
    preset_key = (
        str(state.sample.specimen_preset_key).strip()
        or default_specimen_preset_key()
    )
    preset = load_specimen_preset(preset_key)
    pixels_override = int(getattr(state.sample, "wave_grid_pixels", 0))
    fov_override = float(getattr(state.sample, "wave_field_of_view_angstrom", 0.0))
    x_axis, y_axis, potential = projected_potential(
        preset,
        state.sample.thickness_nm,
        pixels=pixels_override if pixels_override > 0 else None,
        field_of_view_angstrom=fov_override if fov_override > 0.0 else None,
    )
    n = x_axis.size
    spacing = float(x_axis[1] - x_axis[0])
    frequencies = np.fft.fftshift(np.fft.fftfreq(n, d=spacing))
    fx, fy = np.meshgrid(frequencies, frequencies, indexing="xy")
    frequency_squared = fx * fx + fy * fy
    _, _, wavelength_nm = electron(state)
    wavelength_angstrom = wavelength_nm * 10.0
    ray_stats = _weighted_ray_statistics(simulation.incident)
    incident_wave = _incident_wave(state, ray_stats, frequencies, wavelength_angstrom)
    sigma = interaction_constant_rad_per_v_angstrom(state.beam_voltage_kv)
    exit_wave = incident_wave * np.exp(1j * sigma * potential)

    defocus_angstrom, focal_mm = _objective_defocus_angstrom(state)
    cs_mm = float(getattr(state.objective_lens, "cs_mm", 0.0) or 0.0)
    cs_angstrom = cs_mm * 1.0e7
    if math.isfinite(defocus_angstrom):
        chi = math.pi * wavelength_angstrom * defocus_angstrom * frequency_squared
        chi += 0.5 * math.pi * cs_angstrom * wavelength_angstrom**3 * frequency_squared**2
    else:
        # With the Objective disabled there is no focused Objective image;
        # return the aperture-limited exit-wave intensity without NaNs.
        chi = np.zeros_like(frequency_squared)
    aperture_rad = _objective_aperture_rad(state)
    aperture_mask = np.ones_like(frequency_squared, dtype=bool)
    if math.isfinite(aperture_rad):
        aperture_mask = frequency_squared <= (aperture_rad / wavelength_angstrom) ** 2
    transfer = aperture_mask * np.exp(-1j * chi)
    exit_spectrum = np.fft.fftshift(np.fft.fft2(exit_wave))
    image_wave = np.fft.ifft2(np.fft.ifftshift(exit_spectrum * transfer))
    raw_image = np.abs(image_wave) ** 2
    raw_diffraction = np.abs(exit_spectrum) ** 2
    diffraction = np.log1p(raw_diffraction / max(float(raw_diffraction.max()), 1.0e-30) * 1.0e4)

    metrics = {
        **ray_stats,
        "wavelength_angstrom": wavelength_angstrom,
        "interaction_constant_rad_per_v_angstrom": sigma,
        "objective_focal_length_mm": focal_mm,
        "objective_defocus_nm": defocus_angstrom / 10.0,
        "objective_focused": math.isfinite(defocus_angstrom),
        "objective_cs_mm": cs_mm,
        "objective_aperture_mrad": aperture_rad * 1.0e3,
        "pixel_size_angstrom": spacing,
        "field_of_view_angstrom": spacing * n,
    }
    return WaveImagingResult(
        preset_key=preset.key,
        preset_name=preset.name,
        x_angstrom=x_axis,
        y_angstrom=y_axis,
        projected_potential_v_angstrom=potential,
        exit_wave=exit_wave,
        diffraction_intensity=(
            diffraction / max(float(diffraction.max()), 1.0e-30)
        ),
        image_intensity=_normalise_image(raw_image),
        spatial_frequency_inv_angstrom=frequencies,
        metrics=metrics,
    )
