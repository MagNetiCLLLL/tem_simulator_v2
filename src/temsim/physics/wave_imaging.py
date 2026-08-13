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
from pathlib import Path

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
from temsim.specimen.geometry import (
    quaternion_to_matrix,
    sample_orientation_quaternion,
)


@dataclass(frozen=True)
class WaveImagingResult:
    preset_key: str
    preset_name: str
    x_angstrom: np.ndarray
    y_angstrom: np.ndarray
    projected_potential_v_angstrom: np.ndarray
    exit_wave: np.ndarray
    linear_diffraction_probability: np.ndarray
    diffraction_intensity: np.ndarray
    image_intensity: np.ndarray
    spatial_frequency_inv_angstrom: np.ndarray
    spatial_frequency_y_inv_angstrom: np.ndarray
    metrics: dict


@dataclass(frozen=True)
class PreparedSpecimen:
    x_angstrom: np.ndarray
    y_angstrom: np.ndarray
    potential_configurations_v_angstrom: tuple[np.ndarray, ...]
    mean_projected_potential_v_angstrom: np.ndarray
    slice_thicknesses_angstrom: np.ndarray | None
    metrics: dict


# Conservative host allocations retained or created while forming one TEM
# image.  This covers real-valued coordinate/frequency/phase grids and the
# complex incident, exit, transfer and FFT work arrays.  Stored atomistic
# potentials and frozen-phonon exit waves are estimated separately below.
_TEM_WAVE_WORKING_BYTES_PER_PIXEL = 224
_ATOMISTIC_POTENTIAL_BYTES_PER_VOXEL = np.dtype(np.float32).itemsize
_COMPLEX_EXIT_WAVE_BYTES_PER_PIXEL = np.dtype(np.complex128).itemsize


def tem_wave_imaging_enabled(state) -> bool:
    """Return whether this state requests the local TEM wave observable.

    The separate STEM wave path owns raster detector images.  Virtual samples
    have explicit ray/detector interaction channels and do not define a
    specimen potential for this TEM image-forming calculation.
    """

    return bool(
        getattr(state.sample, "wave_enabled", False)
        and str(getattr(state, "illumination_mode", "TEM")).upper() == "TEM"
        and str(getattr(state.sample, "specimen_mode", "atomic")).lower()
        == "atomic"
    )


def estimate_tem_wave_memory_bytes(state) -> int:
    """Estimate incremental peak host memory for optional TEM wave imaging.

    The estimate deliberately does not import or construct an atomistic
    backend.  It uses the requested grid, thickness, slice target and
    frozen-phonon count so the application-wide memory guard can reject an
    unsafe calculation before ray tracing begins.
    """

    if not tem_wave_imaging_enabled(state):
        return 0

    sample = state.sample
    preset_key = (
        (
            str(getattr(sample, "specimen_preset_key", "")).strip()
            or default_specimen_preset_key()
        )
        if bool(getattr(sample, "inserted", True))
        else "vacuum"
    )
    preset = load_specimen_preset(preset_key)
    pixels_override = int(getattr(sample, "wave_grid_pixels", 0))
    pixels = pixels_override if pixels_override > 0 else int(preset.pixels)
    if pixels < 32:
        raise ValueError("Wave grid must contain at least 32 pixels.")

    grid_points = pixels * pixels
    working_bytes = grid_points * _TEM_WAVE_WORKING_BYTES_PER_PIXEL
    projected_potential_bytes = grid_points * np.dtype(np.float64).itemsize

    thickness_angstrom = effective_sample_thickness_nm(state) * 10.0
    multislice_enabled = bool(
        getattr(sample, "wave_multislice_enabled", True)
    )
    atomistic_requested = bool(
        getattr(sample, "wave_atomistic_enabled", True)
    )
    atomistic_source_available = bool(
        str(getattr(sample, "cif_path", "")).strip()
        or preset.atomistic is not None
    )
    atomistic_applies = bool(
        multislice_enabled
        and atomistic_requested
        and atomistic_source_available
        and thickness_angstrom > 0.0
    )

    configuration_count = 1
    potential_bytes = projected_potential_bytes
    if atomistic_applies:
        target_slice = float(
            getattr(sample, "wave_slice_thickness_angstrom", 2.0)
        )
        if not math.isfinite(target_slice) or target_slice <= 0.0:
            raise ValueError("Wave slice thickness must be finite and positive.")
        slice_count = max(1, int(math.ceil(thickness_angstrom / target_slice)))
        if bool(getattr(sample, "wave_frozen_phonon_enabled", False)):
            configuration_count = int(
                getattr(sample, "wave_frozen_phonon_configurations", 4)
            )
            if not 1 <= configuration_count <= 64:
                raise ValueError(
                    "Frozen-phonon configurations must be between 1 and 64."
                )

        # A commensurate periodic cell can be slightly larger than the
        # requested square FOV.  Reserve 25% extra grid points, plus one extra
        # copy for atomistic-potential construction/transposition.
        atomistic_grid_points = int(math.ceil(grid_points * 1.25))
        stored_potential_bytes = (
            configuration_count
            * slice_count
            * atomistic_grid_points
            * _ATOMISTIC_POTENTIAL_BYTES_PER_VOXEL
        )
        potential_bytes = (
            2 * stored_potential_bytes + projected_potential_bytes
        )

    retained_exit_waves = (
        configuration_count
        * grid_points
        * _COMPLEX_EXIT_WAVE_BYTES_PER_PIXEL
    )
    return int(working_bytes + potential_bytes + retained_exit_waves)


def effective_sample_thickness_nm(state) -> float:
    """Return interacting specimen thickness, preserving the reference plane."""

    if (
        not bool(getattr(state.sample, "inserted", True))
        or str(getattr(state.sample, "specimen_mode", "atomic")).lower()
        != "atomic"
    ):
        return 0.0
    return max(float(state.sample.thickness_nm), 0.0)


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
    *,
    field_of_view_angstrom_override: float | None = None,
    calculation_roi_centre_nm=(0.0, 0.0),
    calculation_roi_bounds_nm=None,
) -> PreparedSpecimen:
    """Build the selected qualitative or atomistic specimen representation."""

    pixels_override = int(getattr(state.sample, "wave_grid_pixels", 0))
    fov_override = float(
        getattr(state.sample, "wave_field_of_view_angstrom", 0.0)
    )
    pixels = pixels_override if pixels_override > 0 else preset.pixels
    requested_fov = (
        float(field_of_view_angstrom_override)
        if field_of_view_angstrom_override is not None
        else fov_override
        if fov_override > 0.0
        else preset.field_of_view_angstrom
    )
    if not math.isfinite(requested_fov) or requested_fov <= 0.0:
        raise ValueError("Wave calculation FOV must be finite and positive.")
    thickness_nm = effective_sample_thickness_nm(state)
    total_thickness = thickness_nm * 10.0
    target_slice = float(
        getattr(state.sample, "wave_slice_thickness_angstrom", 2.0)
    )
    multislice_enabled = bool(
        getattr(state.sample, "wave_multislice_enabled", True)
    )
    atomistic_requested = bool(
        getattr(state.sample, "wave_atomistic_enabled", True)
    )
    configured_cif_path = str(
        getattr(state.sample, "cif_path", "")
    ).strip()
    # A parked holder, virtual specimen, or zero-thickness specimen is an
    # interaction-free reference plane.  Dormant CIF settings must therefore
    # neither load a file nor make a vacuum calculation fail validation.
    cif_path = configured_cif_path if thickness_nm > 0.0 else ""
    rotation_deg_xyz = (
        float(getattr(state.sample, "specimen_rotation_x_deg", 0.0)),
        float(getattr(state.sample, "specimen_rotation_y_deg", 0.0)),
        float(getattr(state.sample, "specimen_rotation_z_deg", 0.0)),
    )
    orientation_quaternion = sample_orientation_quaternion(state.sample)
    orientation_matrix = quaternion_to_matrix(orientation_quaternion)
    roi_centre_nm = tuple(float(value) for value in calculation_roi_centre_nm)
    if len(roi_centre_nm) != 2 or not all(
        math.isfinite(value) for value in roi_centre_nm
    ):
        raise ValueError("Calculation ROI centre must contain two finite values.")
    frozen_requested = bool(
        getattr(state.sample, "wave_frozen_phonon_enabled", False)
    )
    atomistic_fallback_reason = None
    if cif_path and not atomistic_requested:
        raise ValueError(
            "A custom CIF requires the Atomistic IAM potential option."
        )
    if cif_path and not multislice_enabled:
        raise ValueError(
            "A custom CIF requires multislice specimen propagation."
        )

    if calculation_roi_bounds_nm is not None and total_thickness > 0.0:
        roi_x0, roi_x1, roi_y0, roi_y1 = (
            float(value) for value in calculation_roi_bounds_nm
        )
        sample_x0 = float(getattr(state.sample, "centre_x_nm", 0.0)) - 0.5 * float(
            getattr(state.sample, "size_x_nm", 0.0)
        )
        sample_x1 = float(getattr(state.sample, "centre_x_nm", 0.0)) + 0.5 * float(
            getattr(state.sample, "size_x_nm", 0.0)
        )
        sample_y0 = float(getattr(state.sample, "centre_y_nm", 0.0)) - 0.5 * float(
            getattr(state.sample, "size_y_nm", 0.0)
        )
        sample_y1 = float(getattr(state.sample, "centre_y_nm", 0.0)) + 0.5 * float(
            getattr(state.sample, "size_y_nm", 0.0)
        )
        overlaps = not (
            roi_x1 < sample_x0
            or roi_x0 > sample_x1
            or roi_y1 < sample_y0
            or roi_y0 > sample_y1
        )
        if not overlaps:
            spacing = requested_fov / pixels
            axis = (np.arange(pixels, dtype=float) - pixels // 2) * spacing
            vacuum = np.zeros((pixels, pixels), dtype=float)
            return PreparedSpecimen(
                x_angstrom=axis,
                y_angstrom=axis.copy(),
                potential_configurations_v_angstrom=(vacuum,),
                mean_projected_potential_v_angstrom=vacuum,
                slice_thicknesses_angstrom=None,
                metrics={
                    "potential_model": "finite_sample_vacuum_outside",
                    "atomistic_requested": atomistic_requested,
                    "atomistic_applied": False,
                    "atomistic_fallback_reason": None,
                    "atom_count": 0,
                    "configuration_count": 1,
                    "frozen_phonon_requested": frozen_requested,
                    "frozen_phonon_applied": False,
                    "calculation_roi_centre_nm": roi_centre_nm,
                    "calculation_roi_bounds_nm": tuple(
                        float(value) for value in calculation_roi_bounds_nm
                    ),
                    "finite_specimen_size_nm": (
                        float(getattr(state.sample, "size_x_nm", 0.0)),
                        float(getattr(state.sample, "size_y_nm", 0.0)),
                        thickness_nm,
                    ),
                    "specimen_orientation_quaternion_wxyz": (
                        orientation_quaternion
                    ),
                    "requested_field_of_view_angstrom": requested_fov,
                    "requested_thickness_angstrom": total_thickness,
                    "maximum_relative_intensity_change": 0.0,
                    "bonding_charge_included": False,
                },
            )

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
                cif_path=cif_path,
                rotation_deg_xyz=rotation_deg_xyz,
                rotation_matrix=orientation_matrix,
                specimen_size_xy_angstrom=(
                    float(getattr(state.sample, "size_x_nm", 0.0)) * 10.0,
                    float(getattr(state.sample, "size_y_nm", 0.0)) * 10.0,
                ),
                specimen_centre_xy_angstrom=(
                    float(getattr(state.sample, "centre_x_nm", 0.0)) * 10.0,
                    float(getattr(state.sample, "centre_y_nm", 0.0)) * 10.0,
                ),
                calculation_roi_centre_xy_angstrom=(
                    roi_centre_nm[0] * 10.0,
                    roi_centre_nm[1] * 10.0,
                ),
                thermal_sigma_by_element_angstrom=dict(
                    getattr(
                        state.sample,
                        "wave_frozen_phonon_sigma_by_element_angstrom",
                        {},
                    )
                    or {}
                ),
            )
        except AtomisticBackendUnavailable as exc:
            if cif_path:
                raise
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
                    "atomistic_source_kind": ensemble.source_kind,
                    "atomistic_source_path": ensemble.source_path,
                    "specimen_rotation_deg_xyz": (
                        ensemble.rotation_deg_xyz
                    ),
                    "specimen_orientation_quaternion_wxyz": (
                        orientation_quaternion
                    ),
                    "calculation_roi_centre_nm": roi_centre_nm,
                    "finite_specimen_size_nm": (
                        float(getattr(state.sample, "size_x_nm", 0.0)),
                        float(getattr(state.sample, "size_y_nm", 0.0)),
                        thickness_nm,
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
        thickness_nm,
        pixels=pixels,
        field_of_view_angstrom=requested_fov,
    )
    if calculation_roi_bounds_nm is not None and thickness_nm > 0.0:
        lab_x_nm = x_axis * 0.1 + roi_centre_nm[0]
        lab_y_nm = y_axis * 0.1 + roi_centre_nm[1]
        inside_x = np.abs(
            lab_x_nm - float(getattr(state.sample, "centre_x_nm", 0.0))
        ) <= 0.5 * float(getattr(state.sample, "size_x_nm", 0.0))
        inside_y = np.abs(
            lab_y_nm - float(getattr(state.sample, "centre_y_nm", 0.0))
        ) <= 0.5 * float(getattr(state.sample, "size_y_nm", 0.0))
        potential = potential * (inside_y[:, None] & inside_x[None, :])
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
            "calculation_roi_centre_nm": roi_centre_nm,
            "finite_specimen_size_nm": (
                float(getattr(state.sample, "size_x_nm", 0.0)),
                float(getattr(state.sample, "size_y_nm", 0.0)),
                thickness_nm,
            ),
            "specimen_orientation_quaternion_wxyz": orientation_quaternion,
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
        "radius_rms_m": statistics.radius_rms_m,
        "radius_99_m": statistics.radius_99_m,
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
            "slice_count": 1 if total_thickness_nm > 0.0 else 0,
            "total_thickness_angstrom": total_thickness_angstrom,
            "slice_thickness_angstrom": total_thickness_angstrom,
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
    specimen_metrics["sample_inserted"] = sample_inserted
    specimen_metrics["sample_interaction_applied"] = atomic_interaction

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
    linear_diffraction = raw_diffraction / max(
        float(np.sum(raw_diffraction)), 1.0e-30
    )
    incident_spectrum = np.fft.fftshift(np.fft.fft2(incident_wave))
    incident_diffraction = np.abs(incident_spectrum) ** 2
    incident_diffraction /= max(
        float(np.sum(incident_diffraction)), 1.0e-30
    )
    incident_cone_rad = max(
        float(ray_stats["convergence_semiangle_rad"]), 0.0
    )
    incident_cone_mask = (
        frequency_squared
        <= (incident_cone_rad / wavelength_angstrom) ** 2
        + np.finfo(float).eps
    )
    exit_outside_cone = float(
        np.sum(linear_diffraction[~incident_cone_mask])
    )
    incident_outside_cone = float(
        np.sum(incident_diffraction[~incident_cone_mask])
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

    real_interactions = getattr(simulation, "real_interactions", None)
    zero_loss_probability = 1.0
    absorbed_probability = 0.0
    mean_inelastic_events = 0.0
    if real_interactions is not None:
        absorbed_probability = float(
            real_interactions.absorbed_probability
        )
        mean_inelastic_events = float(
            real_interactions.mean_inelastic_events
        )
        zero_loss_probability = next(
            (
                float(channel.probability)
                for channel in real_interactions.channels
                if channel.key == "real_zero_loss"
            ),
            0.0,
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
        "image_display_scaling": "0.5-99.5 percentile clipped to [0, 1]",
        "diffraction_display_scaling": "log1p contrast, normalised to [0, 1]",
        "image_formation_scope": "specimen to objective CTF",
        "exit_wave_representation": "coherent ensemble mean",
        "displayed_intensity_average": (
            "incoherent frozen-phonon intensity mean"
            if len(exit_waves) > 1
            else "single configuration"
        ),
        "wave_energy_loss_scope": (
            "conditional zero-loss coherent elastic image; inelastic event "
            "probabilities are transported separately as ray populations"
        ),
        "zero_loss_probability_per_sample_incident": zero_loss_probability,
        "sample_absorbed_probability_per_sample_incident": (
            absorbed_probability
        ),
        "mean_inelastic_events_per_sample_incident": mean_inelastic_events,
        "elastic_wave_observable": (
            "conditional zero-loss exit-wave intensity outside the incident "
            "99%-current convergence cone; coherent/non-exclusive"
        ),
        "elastic_incident_cone_mrad": incident_cone_rad * 1.0e3,
        "elastic_exit_intensity_outside_incident_cone_fraction": (
            exit_outside_cone
        ),
        "elastic_incident_baseline_outside_cone_fraction": (
            incident_outside_cone
        ),
        "elastic_outside_cone_redistribution_delta": (
            exit_outside_cone - incident_outside_cone
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
    custom_cif_path = specimen_metrics.get("atomistic_source_path")
    display_key = (
        f"cif:{Path(custom_cif_path).name}"
        if custom_cif_path
        else preset.key
    )
    display_name = (
        f"Custom CIF: {Path(custom_cif_path).name}"
        if custom_cif_path
        else preset.name
    )
    return WaveImagingResult(
        preset_key=display_key,
        preset_name=display_name,
        x_angstrom=x_axis,
        y_angstrom=y_axis,
        projected_potential_v_angstrom=potential,
        exit_wave=exit_wave,
        linear_diffraction_probability=linear_diffraction,
        diffraction_intensity=(
            diffraction / max(float(diffraction.max()), 1.0e-30)
        ),
        image_intensity=_normalise_image(raw_image),
        spatial_frequency_inv_angstrom=frequencies_x,
        spatial_frequency_y_inv_angstrom=frequencies_y,
        metrics=metrics,
    )
