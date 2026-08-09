"""Local coherent wave optics from the specimen to the Objective image.

The full microscope remains a ray model.  This module converts the surviving
ray bundle at the specimen into an incident complex wave, propagates through a
TOML-owned specimen definition, and transfers the exit wave through an
Objective CTF.  The specimen step can use finite-slice atomistic IAM
potentials, frozen phonons, continuous projected columns, or a fast projected
phase object.  It is not a bonded-charge or first-principles potential model.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict
import math

import numpy as np

from temsim.physics.core import C, E, H, M, electron
from temsim.physics.compute_backend import (
    WAVE_BACKEND_NUMPY,
    choose_wave_backend,
)
from temsim.physics.multislice import propagate_multislice
from temsim.physics.beam_statistics import branch_sample_statistics
from temsim.physics.wave_fft import form_tem_image
from temsim.specimen.atomistic import (
    AtomisticBackendUnavailable,
    build_atomistic_potential_ensemble,
)
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


@dataclass(frozen=True)
class PreparedSpecimen:
    x_angstrom: np.ndarray
    y_angstrom: np.ndarray
    potential_configurations_v_angstrom: tuple[np.ndarray, ...]
    mean_projected_potential_v_angstrom: np.ndarray
    slice_thicknesses_angstrom: np.ndarray | None
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


def prepare_specimen_potentials(
    state,
    preset: SpecimenPreset,
) -> PreparedSpecimen:
    """Build the selected qualitative or atomistic specimen representation."""

    pixels_override = int(getattr(state.sample, "wave_grid_pixels", 0))
    fov_override = float(
        getattr(state.sample, "wave_field_of_view_angstrom", 0.0)
    )
    pixels = pixels_override if pixels_override > 0 else preset.pixels
    requested_fov = (
        fov_override if fov_override > 0.0 else preset.field_of_view_angstrom
    )
    total_thickness = max(float(state.sample.thickness_nm), 0.0) * 10.0
    target_slice = float(
        getattr(state.sample, "wave_slice_thickness_angstrom", 2.0)
    )
    multislice_enabled = bool(
        getattr(state.sample, "wave_multislice_enabled", True)
    )
    atomistic_requested = bool(
        getattr(state.sample, "wave_atomistic_enabled", True)
    )
    frozen_requested = bool(
        getattr(state.sample, "wave_frozen_phonon_enabled", False)
    )
    atomistic_fallback_reason = None

    if atomistic_requested and multislice_enabled and total_thickness > 0.0:
        try:
            ensemble = build_atomistic_potential_ensemble(
                preset,
                thickness_angstrom=total_thickness,
                field_of_view_angstrom=requested_fov,
                pixels=pixels,
                target_slice_thickness_angstrom=target_slice,
                frozen_phonon_enabled=frozen_requested,
                frozen_phonon_configurations=int(
                    getattr(
                        state.sample,
                        "wave_frozen_phonon_configurations",
                        4,
                    )
                ),
                thermal_sigma_override_angstrom=float(
                    getattr(
                        state.sample,
                        "wave_frozen_phonon_sigma_angstrom",
                        0.0,
                    )
                ),
                thermal_seed=int(
                    getattr(state.sample, "wave_frozen_phonon_seed", 100)
                ),
            )
        except AtomisticBackendUnavailable as exc:
            atomistic_fallback_reason = str(exc)
        else:
            spacing_x, spacing_y = ensemble.sampling_angstrom_xy
            ny, nx = ensemble.grid_shape_yx
            x_axis = (np.arange(nx, dtype=float) - nx // 2) * spacing_x
            y_axis = (np.arange(ny, dtype=float) - ny // 2) * spacing_y
            return PreparedSpecimen(
                x_angstrom=x_axis,
                y_angstrom=y_axis,
                potential_configurations_v_angstrom=(
                    ensemble.configurations_v_angstrom
                ),
                mean_projected_potential_v_angstrom=(
                    ensemble.mean_projected_potential_v_angstrom
                ),
                slice_thicknesses_angstrom=(
                    ensemble.slice_thicknesses_angstrom
                ),
                metrics={
                    "potential_model": "atomistic_lobato_iam",
                    "atomistic_requested": True,
                    "atomistic_applied": True,
                    "atomistic_fallback_reason": None,
                    "atom_count": ensemble.atom_count,
                    "configuration_count": ensemble.configuration_count,
                    "frozen_phonon_requested": frozen_requested,
                    "frozen_phonon_applied": frozen_requested,
                    "thermal_sigma_angstrom": (
                        ensemble.thermal_sigma_angstrom
                    ),
                    "thermal_seed": ensemble.thermal_seed,
                    "thermal_model": ensemble.thermal_model,
                    "thermal_sigma_reference": (
                        ensemble.thermal_sigma_reference
                    ),
                    "parametrization": ensemble.parametrization,
                    "projection": ensemble.projection,
                    "potential_builder_backend": ensemble.builder_backend,
                    "potential_storage_bytes": (
                        ensemble.potential_storage_bytes
                    ),
                    "lateral_cell_commensurate": (
                        ensemble.lateral_cell_commensurate
                    ),
                    "requested_lateral_mismatch_angstrom": (
                        ensemble.lateral_mismatch_angstrom
                    ),
                    "realised_lateral_extent_angstrom": (
                        ensemble.extent_angstrom_xy
                    ),
                    "requested_field_of_view_angstrom": requested_fov,
                    "requested_thickness_mismatch_angstrom": (
                        ensemble.thickness_mismatch_angstrom
                    ),
                    "requested_thickness_angstrom": total_thickness,
                    "intensity_ensemble_average": frozen_requested,
                    "correlated_phonons": False,
                    "bonding_charge_included": False,
                },
            )

    if atomistic_requested and atomistic_fallback_reason is None:
        if not multislice_enabled:
            atomistic_fallback_reason = (
                "The 3-D atomistic potential requires multislice; using the "
                "analytic projected-column preview model."
            )
        elif total_thickness <= 0.0:
            atomistic_fallback_reason = (
                "A zero-thickness specimen has no 3-D atomistic potential; "
                "using the analytic projected-column representation."
            )

    x_axis, y_axis, potential = projected_potential(
        preset,
        state.sample.thickness_nm,
        pixels=pixels,
        field_of_view_angstrom=requested_fov,
    )
    return PreparedSpecimen(
        x_angstrom=x_axis,
        y_angstrom=y_axis,
        potential_configurations_v_angstrom=(potential,),
        mean_projected_potential_v_angstrom=potential,
        slice_thicknesses_angstrom=None,
        metrics={
            "potential_model": "analytic_projected_columns",
            "atomistic_requested": atomistic_requested,
            "atomistic_applied": False,
            "atomistic_fallback_reason": atomistic_fallback_reason,
            "atom_count": 0,
            "configuration_count": 1,
            "frozen_phonon_requested": frozen_requested,
            "frozen_phonon_applied": False,
            "thermal_sigma_angstrom": 0.0,
            "thermal_seed": int(
                getattr(state.sample, "wave_frozen_phonon_seed", 100)
            ),
            "thermal_model": "not applied",
            "thermal_sigma_reference": "not applicable",
            "parametrization": "qualitative TOML Gaussian columns",
            "projection": "total 2-D projection",
            "potential_builder_backend": "NumPy",
            "potential_storage_bytes": int(potential.nbytes),
            "lateral_cell_commensurate": True,
            "requested_lateral_mismatch_angstrom": (0.0, 0.0),
            "realised_lateral_extent_angstrom": (
                requested_fov,
                requested_fov,
            ),
            "requested_field_of_view_angstrom": requested_fov,
            "requested_thickness_mismatch_angstrom": 0.0,
            "requested_thickness_angstrom": total_thickness,
            "intensity_ensemble_average": False,
            "correlated_phonons": False,
            "bonding_charge_included": False,
        },
    )


def interaction_constant_rad_per_v_angstrom(voltage_kv: float) -> float:
    """Relativistic phase-object interaction constant in rad/(V Angstrom)."""
    kinetic_j = E * float(voltage_kv) * 1000.0
    gamma = 1.0 + kinetic_j / (M * C * C)
    momentum = math.sqrt(kinetic_j * kinetic_j + 2.0 * kinetic_j * M * C * C) / C
    wavelength_m = H / momentum
    return 2.0 * math.pi * gamma * M * E * wavelength_m / (H * H) * 1.0e-10


def _weighted_ray_statistics(incident) -> dict:
    statistics = branch_sample_statistics(incident)
    return {
        "mean_x_m": statistics.mean_x_m,
        "mean_y_m": statistics.mean_y_m,
        "mean_tx_rad": statistics.mean_tx_rad,
        "mean_ty_rad": statistics.mean_ty_rad,
        "convergence_rms_rad": statistics.convergence_rms_rad,
        "convergence_95_rad": statistics.convergence_95_rad,
        "convergence_99_rad": statistics.convergence_99_rad,
        "convergence_edge_rad": statistics.convergence_edge_rad,
        "convergence_semiangle_rad": statistics.convergence_99_rad,
        "surviving_rays": statistics.surviving_rays,
    }


def _incident_wave(
    state,
    ray_stats: dict,
    frequencies_x: np.ndarray,
    frequencies_y: np.ndarray,
    wavelength_angstrom: float,
) -> np.ndarray:
    nx = frequencies_x.size
    ny = frequencies_y.size
    fx, fy = np.meshgrid(frequencies_x, frequencies_y, indexing="xy")
    tilt_fx = ray_stats["mean_tx_rad"] / wavelength_angstrom
    tilt_fy = ray_stats["mean_ty_rad"] / wavelength_angstrom
    if str(getattr(state, "illumination_mode", "TEM")).upper() != "STEM":
        spacing_x = 1.0 / (
            nx * abs(frequencies_x[1] - frequencies_x[0])
        )
        spacing_y = 1.0 / (
            ny * abs(frequencies_y[1] - frequencies_y[0])
        )
        axis_x = (np.arange(nx, dtype=float) - nx // 2) * spacing_x
        axis_y = (np.arange(ny, dtype=float) - ny // 2) * spacing_y
        xx, yy = np.meshgrid(axis_x, axis_y, indexing="xy")
        return np.exp(2j * math.pi * (tilt_fx * xx + tilt_fy * yy))

    frequency_step = max(
        abs(frequencies_x[1] - frequencies_x[0]),
        abs(frequencies_y[1] - frequencies_y[0]),
    )
    alpha = max(
        ray_stats["convergence_semiangle_rad"],
        frequency_step * wavelength_angstrom,
    )
    radius = alpha / wavelength_angstrom
    aperture = ((fx - tilt_fx) ** 2 + (fy - tilt_fy) ** 2) <= radius**2
    if not np.any(aperture):
        nearest = np.unravel_index(
            np.argmin((fx - tilt_fx) ** 2 + (fy - tilt_fy) ** 2), fx.shape
        )
        aperture[nearest] = True
    fov_x = 1.0 / abs(frequencies_x[1] - frequencies_x[0])
    fov_y = 1.0 / abs(frequencies_y[1] - frequencies_y[0])
    x0 = math.remainder(ray_stats["mean_x_m"] * 1.0e10, fov_x)
    y0 = math.remainder(ray_stats["mean_y_m"] * 1.0e10, fov_y)
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
    prepared = prepare_specimen_potentials(state, preset)
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
    frequency_squared = fx * fx + fy * fy
    _, _, wavelength_nm = electron(state)
    wavelength_angstrom = wavelength_nm * 10.0
    ray_stats = _weighted_ray_statistics(simulation.incident)
    incident_wave = _incident_wave(
        state,
        ray_stats,
        frequencies_x,
        frequencies_y,
        wavelength_angstrom,
    )
    sigma = interaction_constant_rad_per_v_angstrom(state.beam_voltage_kv)
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
        work_items=nx * ny * estimated_slices * configuration_count,
    )
    exit_waves = []
    if multislice_enabled:
        diagnostic_records = []
        for configuration in prepared.potential_configurations_v_angstrom:
            explicit_slices = configuration.ndim == 3
            exit_wave, multislice_diagnostics = propagate_multislice(
                incident_wave,
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
            exit_waves.append(exit_wave)
            diagnostic_records.append(asdict(multislice_diagnostics))
            if multislice_diagnostics.compute_backend != wave_backend:
                wave_backend = multislice_diagnostics.compute_backend
                wave_fallback_reason = multislice_diagnostics.fallback_reason

        specimen_metrics = dict(diagnostic_records[0])
        for key in (
            "maximum_phase_per_slice_rad",
            "maximum_relative_intensity_change",
        ):
            specimen_metrics[key] = max(
                float(record[key]) for record in diagnostic_records
            )
        for key in (
            "initial_integrated_intensity",
            "final_integrated_intensity",
        ):
            specimen_metrics[key] = float(
                np.mean([float(record[key]) for record in diagnostic_records])
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
        exit_wave = incident_wave * np.exp(1j * sigma * potential)
        exit_waves.append(exit_wave)
        isotropic_nyquist = min(
            0.5 / spacing_x,
            0.5 / spacing_y,
        )
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
            "maximum_isotropic_angle_mrad": math.asin(
                min(wavelength_angstrom * isotropic_nyquist, 1.0)
            ) * 1.0e3,
            "maximum_phase_per_slice_rad": float(
                np.max(np.abs(sigma * potential))
            ),
            "initial_integrated_intensity": float(
                np.sum(np.abs(incident_wave) ** 2)
            ),
            "final_integrated_intensity": float(
                np.sum(np.abs(exit_wave) ** 2)
            ),
            "maximum_relative_intensity_change": 0.0,
            "compute_backend": WAVE_BACKEND_NUMPY,
            "numeric_precision": "complex128 / float64",
            "fallback_reason": None,
            "pixel_size_y_angstrom": spacing_y,
            "pixel_size_x_angstrom": spacing_x,
        }

    specimen_metrics.update(prepared.metrics)

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
    specimen_backend = str(specimen_metrics["compute_backend"])
    fft_backend = (
        specimen_backend
        if specimen_backend in {WAVE_BACKEND_NUMPY, "CuPy CUDA"}
        else WAVE_BACKEND_NUMPY
    )
    fft_fallback_seed = (
        specimen_metrics.get("fallback_reason") or wave_fallback_reason
    )
    raw_image = np.zeros((ny, nx), dtype=np.float64)
    image_m2 = np.zeros((ny, nx), dtype=np.float64)
    raw_diffraction = np.zeros((ny, nx), dtype=np.float64)
    coherent_exit_wave = np.zeros((ny, nx), dtype=np.complex128)
    fft_records = []
    for configuration_index, exit_configuration in enumerate(
        exit_waves, start=1
    ):
        image_configuration, diffraction_configuration, fft_diagnostics = (
            form_tem_image(
                exit_configuration,
                transfer,
                compute_backend=fft_backend,
                fallback_reason=fft_fallback_seed,
            )
        )
        image_delta = image_configuration - raw_image
        raw_image += image_delta / configuration_index
        image_m2 += image_delta * (image_configuration - raw_image)
        raw_diffraction += diffraction_configuration
        coherent_exit_wave += exit_configuration
        fft_records.append(fft_diagnostics)
        if fft_diagnostics.compute_backend != fft_backend:
            fft_backend = fft_diagnostics.compute_backend
            fft_fallback_seed = fft_diagnostics.fallback_reason
    raw_diffraction /= len(exit_waves)
    exit_wave = coherent_exit_wave / len(exit_waves)
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
    if len(exit_waves) > 1:
        image_standard_error = np.sqrt(
            image_m2 / (len(exit_waves) - 1) / len(exit_waves)
        )
        image_relative_standard_error = math.sqrt(
            float(np.mean(image_standard_error**2))
            / max(float(np.mean(raw_image**2)), 1.0e-30)
        )
    else:
        image_relative_standard_error = 0.0
    diffraction = np.log1p(raw_diffraction / max(float(raw_diffraction.max()), 1.0e-30) * 1.0e4)

    actual_backends = {
        str(specimen_metrics["compute_backend"]),
        fft_compute_backend,
    }
    wave_compute_backend = (
        actual_backends.pop()
        if len(actual_backends) == 1
        else "Mixed (NumPy CPU + CuPy CUDA)"
    )

    metrics = {
        **ray_stats,
        **{
            f"specimen_{key}": value
            for key, value in specimen_metrics.items()
        },
        "wavelength_angstrom": wavelength_angstrom,
        "interaction_constant_rad_per_v_angstrom": sigma,
        "objective_focal_length_mm": focal_mm,
        "objective_defocus_nm": defocus_angstrom / 10.0,
        "objective_focused": math.isfinite(defocus_angstrom),
        "objective_cs_mm": cs_mm,
        "objective_aperture_mrad": aperture_rad * 1.0e3,
        "pixel_size_angstrom": max(spacing_x, spacing_y),
        "pixel_size_x_angstrom": spacing_x,
        "pixel_size_y_angstrom": spacing_y,
        "field_of_view_angstrom": max(spacing_x * nx, spacing_y * ny),
        "field_of_view_x_angstrom": spacing_x * nx,
        "field_of_view_y_angstrom": spacing_y * ny,
        "fft_compute_backend": fft_compute_backend,
        "fft_numeric_precision": fft_numeric_precision,
        "fft_fallback_reason": fft_fallback_reason,
        "wave_compute_backend": wave_compute_backend,
        "exit_wave_representation": "coherent ensemble mean",
        "displayed_intensity_average": (
            "incoherent frozen-phonon intensity mean"
            if len(exit_waves) > 1
            else "single configuration"
        ),
        "image_configuration_relative_standard_error": (
            image_relative_standard_error
        ),
        "wave_sampling_truncates_illumination": bool(
            ray_stats["convergence_semiangle_rad"] * 1.0e3
            > float(specimen_metrics["maximum_isotropic_angle_mrad"])
        ),
        "wave_intensity_conservation_within_0_1_percent": bool(
            float(specimen_metrics["maximum_relative_intensity_change"])
            <= 1.0e-3
        ),
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
        spatial_frequency_inv_angstrom=frequencies_x,
        metrics=metrics,
    )
