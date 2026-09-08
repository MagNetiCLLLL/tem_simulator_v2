"""Gun-conditioned TEM wave propagation to a physical recording plane.

The complete source-to-specimen ray bundle defines the incident coherent mode,
including clipping, focusing, affine steering, Larmor rotation and enabled
stigmators/correctors.  The specimen uses the selected multislice or phase
object model.  Its exit wave is then transferred through the Objective pupil
and higher-order image aberrations and through the complete post-specimen
projector Jacobian to the active FluScreen/Camera and detector PSF.

This is a non-OEM paraxial/multislice model, not a bonded-charge,
first-principles-potential or full Maxwell field solution.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from temsim.physics.core import C, E, H, M, electron
from temsim.physics.compute_backend import (
    WAVE_BACKEND_NUMPY,
    choose_wave_backend,
)
from temsim.physics.multislice import propagate_multislice
from temsim.physics.beam_statistics import branch_sample_statistics
from temsim.physics.wave_fft import apply_coherent_transfer
from temsim.physics.wave_sampling import plan_wave_sampling
from temsim.physics.prepared_specimen_cache import (
    cached_prepared_specimen,
    content_identity,
    exact_identity,
)
from temsim.physics.camera_wave import project_wave_to_recording_plane
from temsim.physics.recording_stop import active_tem_recording_plane
from temsim.optics.aberrations import (
    aberration_phase_rad,
    active_effective_aberrations,
)
from temsim.specimen.atomistic import (
    AtomisticBackendUnavailable,
    MAX_ATOMISTIC_POTENTIAL_BYTES,
    build_atomistic_potential_ensemble,
)
from temsim.specimen.presets import (
    SpecimenPreset,
    load_specimen_preset,
)
from temsim.specimen.geometry import (
    quaternion_to_matrix,
)
from temsim.specimen.scene import SpecimenScene
from temsim.specimen.reference_catalog import reference_thermal_sigma, reference_thermal_source


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
    camera_electron_optical_intensity: np.ndarray
    camera_x_mm: np.ndarray
    camera_y_mm: np.ndarray
    spatial_frequency_inv_angstrom: np.ndarray
    spatial_frequency_y_inv_angstrom: np.ndarray
    metrics: dict
    projector_checkpoint: "ProjectorWaveCheckpoint | None" = None


@dataclass(frozen=True)
class ProjectorWaveCheckpoint:
    """Reusable Objective-side waves for downstream D/I/P reprojection.

    Each entry is one frozen-phonon configuration after the Objective pupil
    and residual image-aberration phase.  Keeping the configurations separate
    is required: detector intensities are averaged incoherently, so projecting
    only their coherent mean would change the physical observable.
    """

    objective_wave_configurations: tuple[np.ndarray, ...]
    x_angstrom: np.ndarray
    y_angstrom: np.ndarray
    wavelength_angstrom: float
    convergence_semiangle_rad: float
    # Separate configurations BEFORE the angular Objective pupil. This permits
    # widening a readout aperture without inventing previously deleted phase.
    unapertured_wave_configurations: tuple[np.ndarray, ...] = ()
    objective_aperture_rad: float | None = None


def _readonly_array(values, *, dtype=None) -> np.ndarray:
    result = np.array(values, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _project_objective_configurations(state, checkpoint):
    """Project cached Objective-side waves through the current D/I/P state."""

    raw_image = None
    raw_electron_optical_image = None
    image_m2 = None
    camera_projection = None
    for configuration_index, objective_wave in enumerate(
        checkpoint.objective_wave_configurations, start=1
    ):
        camera_projection = project_wave_to_recording_plane(
            state,
            objective_wave,
            checkpoint.x_angstrom,
            checkpoint.y_angstrom,
            checkpoint.wavelength_angstrom,
            convergence_semiangle_rad=(
                checkpoint.convergence_semiangle_rad
            ),
        )
        image_configuration = np.asarray(
            camera_projection.intensity, dtype=np.float64
        )
        electron_optical_configuration = np.asarray(
            camera_projection.electron_optical_intensity,
            dtype=np.float64,
        )
        if raw_image is None:
            raw_image = np.zeros_like(image_configuration)
            raw_electron_optical_image = np.zeros_like(
                electron_optical_configuration
            )
            image_m2 = np.zeros_like(image_configuration)
        image_delta = image_configuration - raw_image
        raw_image += image_delta / configuration_index
        image_m2 += image_delta * (image_configuration - raw_image)
        raw_electron_optical_image += (
            electron_optical_configuration - raw_electron_optical_image
        ) / configuration_index

    if camera_projection is None:
        raise ValueError("Projector checkpoint contains no wave configurations.")
    return (
        raw_image,
        raw_electron_optical_image,
        image_m2,
        camera_projection,
    )


def _detector_probability(image, x_mm, y_mm) -> float:
    """Integrate one detector-plane density using its physical pixel area."""

    x_values = np.asarray(x_mm, dtype=float)
    y_values = np.asarray(y_mm, dtype=float)
    if x_values.size < 2 or y_values.size < 2:
        return 0.0
    pixel_area_m2 = (
        float(x_values[1] - x_values[0])
        * float(y_values[1] - y_values[0])
        * 1.0e-6
    )
    return float(np.sum(np.asarray(image, dtype=float)) * pixel_area_m2)


def reproject_wave_image(state, result: WaveImagingResult) -> WaveImagingResult:
    """Reuse specimen/Objective work and recompute only the recording plane."""

    checkpoint = result.projector_checkpoint
    if checkpoint is None:
        raise ValueError(
            "The cached TEM result has no Objective-side projector checkpoint."
        )
    aperture_rad = _objective_aperture_rad(state)
    if checkpoint.objective_aperture_rad != aperture_rad:
        if not checkpoint.unapertured_wave_configurations:
            # Legacy checkpoints cannot restore a wider pupil. An unchanged
            # legacy aperture remains compatible with ordinary D/I/P replay.
            previous = float(result.metrics.get("objective_aperture_mrad", math.inf)) * 1.0e-3
            if not math.isclose(previous, aperture_rad, rel_tol=1.0e-12, abs_tol=1.0e-15):
                raise ValueError("Recalculate TEM once to retain the pre-aperture wave configurations")
        else:
            dx = float(checkpoint.x_angstrom[1] - checkpoint.x_angstrom[0])
            dy = float(checkpoint.y_angstrom[1] - checkpoint.y_angstrom[0])
            fx, fy = np.meshgrid(np.fft.fftfreq(len(checkpoint.x_angstrom), dx),
                                 np.fft.fftfreq(len(checkpoint.y_angstrom), dy))
            mask = (fx**2 + fy**2 <= (aperture_rad / checkpoint.wavelength_angstrom)**2)
            configurations = tuple(_readonly_array(np.fft.ifft2(np.fft.fft2(w) * mask))
                                   for w in checkpoint.unapertured_wave_configurations)
            checkpoint = replace(checkpoint, objective_wave_configurations=configurations,
                                 objective_aperture_rad=aperture_rad)
    (
        raw_image,
        raw_electron_optical_image,
        image_m2,
        camera_projection,
    ) = _project_objective_configurations(state, checkpoint)
    configuration_count = len(checkpoint.objective_wave_configurations)
    if configuration_count > 1:
        image_standard_error = np.sqrt(
            image_m2 / (configuration_count - 1) / configuration_count
        )
        relative_standard_error = math.sqrt(
            float(np.mean(image_standard_error**2))
            / max(float(np.mean(raw_image**2)), 1.0e-30)
        )
    else:
        relative_standard_error = 0.0
    metrics = dict(result.metrics)
    metrics.update(camera_projection.metrics)
    metrics.update({
        "camera_collected_zero_loss_relative_intensity": (
            _detector_probability(
                raw_image,
                camera_projection.x_mm,
                camera_projection.y_mm,
            )
        ),
        "camera_collected_intensity_aggregation": (
            "incoherent_configuration_mean"
        ),
        "image_configuration_relative_standard_error": (
            relative_standard_error
        ),
        "projector_checkpoint_reused": True,
        "objective_aperture_mrad": aperture_rad * 1.0e3,
        "projector_checkpoint_configuration_count": configuration_count,
    })
    return replace(
        result,
        image_intensity=_normalise_image(raw_image),
        camera_electron_optical_intensity=raw_electron_optical_image,
        camera_x_mm=camera_projection.x_mm,
        camera_y_mm=camera_projection.y_mm,
        metrics=metrics,
        projector_checkpoint=checkpoint,
    )


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

    The separate STEM wave path owns raster detector images. Real mode obtains
    its potential only from an imported CIF/MCIF; Virtual mode obtains it only
    from the selected TOML reference specimen.
    """

    scene = SpecimenScene.from_state(state)
    try:
        active_tem_recording_plane(state)
        recording_available = True
    except ValueError:
        recording_available = False
    return bool(
        getattr(state.sample, "wave_enabled", False)
        and str(getattr(state, "illumination_mode", "TEM")).upper() == "TEM"
        and scene.structure_available
        and recording_available
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
    scene = SpecimenScene.from_state(state)
    preset = load_specimen_preset(scene.wave_template_key)
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
        scene.cif_path
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
        3 * configuration_count  # specimen exit, post-pupil, and pre-pupil configurations
        * grid_points
        * _COMPLEX_EXIT_WAVE_BYTES_PER_PIXEL
    )
    return int(working_bytes + potential_bytes + retained_exit_waves)


def effective_sample_thickness_nm(state) -> float:
    """Return interacting specimen thickness, preserving the reference plane."""

    return SpecimenScene.from_state(state).interacting_thickness_nm


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
    origin_angstrom_xy=(0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate the periodic projected potential defined by one TOML preset."""
    n = int(pixels or preset.pixels)
    fov = float(field_of_view_angstrom or preset.field_of_view_angstrom)
    if n < 32 or fov <= 0.0:
        raise ValueError("Wave grid must contain at least 32 pixels and a positive FOV.")
    spacing = fov / n
    axis = (np.arange(n, dtype=float) - n // 2) * spacing
    origin_x, origin_y = (float(value) for value in origin_angstrom_xy)
    if not (math.isfinite(origin_x) and math.isfinite(origin_y)):
        raise ValueError("Projected-potential origin must be finite.")
    xx, yy = np.meshgrid(axis + origin_x, axis + origin_y, indexing="xy")
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


def _prepared_specimen_identity(
    state, preset, *, field_of_view_angstrom_override,
    calculation_roi_centre_nm, calculation_roi_bounds_nm,
):
    """Potential dependencies only; incident beam/lens/detector state is absent."""
    from temsim.specimen import atomistic

    scene = SpecimenScene.from_state(state)
    sample = state.sample
    defaults = {
        "wave_grid_pixels": 0,
        "wave_field_of_view_angstrom": 0.0,
        "wave_slice_thickness_angstrom": 2.0,
        "wave_multislice_enabled": True,
        "wave_atomistic_enabled": True,
        "wave_frozen_phonon_enabled": False,
        "wave_frozen_phonon_configurations": 4,
        "wave_frozen_phonon_sigma_angstrom": 0.0,
        "wave_frozen_phonon_seed": 100,
        "wave_frozen_phonon_sigma_by_element_angstrom": {},
        "specimen_rotation_x_deg": 0.0,
        "specimen_rotation_y_deg": 0.0,
        "specimen_rotation_z_deg": 0.0,
    }
    parameters = {key: getattr(sample, key, default) for key, default in defaults.items()}
    parameters["effective_thermal_sigma_angstrom"] = reference_thermal_sigma(sample)
    parameters["effective_thermal_source"] = reference_thermal_source(sample)
    backend = {"numpy": np.__version__}
    if (parameters["wave_atomistic_enabled"] and parameters["wave_multislice_enabled"]
            and scene.interacting_thickness_nm > 0.0):
        capability = atomistic.atomistic_capability()
        backend["atomistic"] = asdict(capability)
        if capability.available:
            import abtem

            backend["precision"] = abtem.config.get("precision")
    return exact_identity({
        # Hash the parsed preset actually passed to the builder, not a possibly
        # different file hidden behind the preset loader's own cache.
        "preset": asdict(preset),
        "scene": {
            key: getattr(scene, key) for key in (
                "inserted", "mode", "source_kind", "source_key",
                "structure_available", "cif_path", "preset_key", "wave_template_key",
                "envelope_shape", "centre_xy_nm", "size_xy_nm", "thickness_nm",
                "orientation_quaternion_wxyz",
            )
        },
        "sample_parameters": parameters,
        "active_cif_content": content_identity(
            scene.cif_path if scene.interacting_thickness_nm > 0.0 else ""
        ),
        "field_of_view_angstrom_override": field_of_view_angstrom_override,
        "calculation_roi_centre_nm": tuple(calculation_roi_centre_nm),
        # None means the original TEM mask policy, not a supplied finite ROI.
        "calculation_roi_bounds_nm": (
            tuple(calculation_roi_bounds_nm) if calculation_roi_bounds_nm is not None else None
        ),
        "backend": backend,
        "potential_limit": MAX_ATOMISTIC_POTENTIAL_BYTES,
        "implementation": tuple(id(function) for function in (
            _prepare_specimen_potentials_uncached, build_atomistic_potential_ensemble,
            projected_potential, plan_wave_sampling, atomistic.build_equilibrium_atoms,
            atomistic._build_one_potential,
        )),
    })


def prepare_specimen_potentials(
    state,
    preset: SpecimenPreset,
    *,
    field_of_view_angstrom_override: float | None = None,
    calculation_roi_centre_nm=(0.0, 0.0),
    calculation_roi_bounds_nm=None,
) -> PreparedSpecimen:
    """Reuse exact specimen potentials independently of wave propagation."""
    started = perf_counter()
    arguments = {
        "field_of_view_angstrom_override": field_of_view_angstrom_override,
        "calculation_roi_centre_nm": tuple(calculation_roi_centre_nm),
        "calculation_roi_bounds_nm": (
            tuple(calculation_roi_bounds_nm) if calculation_roi_bounds_nm is not None else None
        ),
    }
    try:
        key = _prepared_specimen_identity(state, preset, **arguments)
    except (TypeError, ValueError, OverflowError):
        # Invalid/non-serializable inputs still pass through the original
        # scientific validator, which supplies its actionable domain error.
        key = None
    prepared, hit, build_seconds, retained_bytes = cached_prepared_specimen(
        key,
        lambda: _prepare_specimen_potentials_uncached(state, preset, **arguments),
        # A file/settings change while an expensive build runs must never
        # install its result under the earlier content identity.
        still_valid=lambda: key is not None and _prepared_specimen_identity(state, preset, **arguments) == key,
    )
    prepared.metrics.update({
        "preparation_seconds": perf_counter() - started,
        "preparation_build_seconds": build_seconds,
        "prepared_specimen_cache_hit": hit,
        "prepared_specimen_cache_key": key,
        "prepared_specimen_cache_retained_bytes": retained_bytes,
    })
    return prepared


def _prepare_specimen_potentials_uncached(
    state,
    preset: SpecimenPreset,
    *,
    field_of_view_angstrom_override: float | None = None,
    calculation_roi_centre_nm=(0.0, 0.0),
    calculation_roi_bounds_nm=None,
) -> PreparedSpecimen:
    """Build the selected qualitative or atomistic specimen representation."""

    scene = SpecimenScene.from_state(state)
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
    thickness_nm = scene.interacting_thickness_nm
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
    configured_cif_path = scene.cif_path
    # A parked holder, missing active structure, or zero-thickness specimen is
    # an interaction-free reference plane. Dormant CIF settings must neither
    # load a file nor make a Virtual-reference calculation fail validation.
    cif_path = configured_cif_path if thickness_nm > 0.0 else ""
    rotation_deg_xyz = (
        float(getattr(state.sample, "specimen_rotation_x_deg", 0.0)),
        float(getattr(state.sample, "specimen_rotation_y_deg", 0.0)),
        float(getattr(state.sample, "specimen_rotation_z_deg", 0.0)),
    )
    orientation_quaternion = scene.orientation_quaternion_wxyz
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

    half_fov_nm = requested_fov * 0.05
    wave_bounds_nm = (
        roi_centre_nm[0] - half_fov_nm, roi_centre_nm[0] + half_fov_nm,
        roi_centre_nm[1] - half_fov_nm, roi_centre_nm[1] + half_fov_nm,
    )
    overlaps = scene.sample_intersects_bounds(wave_bounds_nm)
    # FOV/grid defines spatial sampling. A larger beam (for example after an
    # Objective focus change) must enlarge the grid, not smear the same CIF
    # onto progressively coarser pixels. Plan before generating any atoms.
    sampling_plan = plan_wave_sampling(
        reference_fov_angstrom=(
            fov_override if fov_override > 0.0 else preset.field_of_view_angstrom
        ),
        reference_pixels=pixels,
        requested_fov_angstrom=requested_fov,
        thickness_angstrom=total_thickness,
        target_slice_thickness_angstrom=target_slice,
        configuration_count=(
            int(getattr(state.sample, "wave_frozen_phonon_configurations", 4))
            if frozen_requested else 1
        ),
        atomistic=bool(
            atomistic_requested and multislice_enabled
            and scene.matter_thickness_nm > 0.0 and overlaps
        ),
        max_potential_bytes=MAX_ATOMISTIC_POTENTIAL_BYTES,
    )
    pixels = sampling_plan.pixels
    material_bounds_nm = None
    if overlaps and scene.matter_thickness_nm > 0.0:
        cx, cy = scene.centre_xy_nm
        sx, sy = scene.size_xy_nm
        material_bounds_nm = (
            max(wave_bounds_nm[0], cx - sx / 2),
            min(wave_bounds_nm[1], cx + sx / 2),
            max(wave_bounds_nm[2], cy - sy / 2),
            min(wave_bounds_nm[3], cy + sy / 2),
        )
    domain_metrics = {
        "wave_sampling_plan": asdict(sampling_plan),
        "wave_window_bounds_nm": wave_bounds_nm,
        # This is a bounding box. The physical disk mask further clips atoms.
        "atom_generation_bounds_nm": material_bounds_nm,
    }

    if calculation_roi_bounds_nm is not None and total_thickness > 0.0:
        # Test the actual square FFT window, not the narrower scan rectangle:
        # padding/configured FOV may still illuminate matter outside that ROI.
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
                    **domain_metrics,
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
                        *scene.size_xy_nm,
                        thickness_nm,
                    ),
                    "finite_specimen_shape": scene.envelope_shape,
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
                thermal_sigma_override_angstrom=reference_thermal_sigma(state.sample),
                thermal_sigma_source=reference_thermal_source(state.sample),
                thermal_seed=int(
                    getattr(state.sample, "wave_frozen_phonon_seed", 100)
                ),
                cif_path=cif_path,
                rotation_deg_xyz=rotation_deg_xyz,
                rotation_matrix=orientation_matrix,
                specimen_size_xy_angstrom=(
                    scene.size_xy_nm[0] * 10.0,
                    scene.size_xy_nm[1] * 10.0,
                ),
                specimen_envelope_shape=scene.envelope_shape,
                specimen_centre_xy_angstrom=(
                    scene.centre_xy_nm[0] * 10.0,
                    scene.centre_xy_nm[1] * 10.0,
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
            configurations = ensemble.configurations_v_angstrom
            mean_projected = ensemble.mean_projected_potential_v_angstrom
            if calculation_roi_bounds_nm is not None:
                lab_x_nm = x_axis * 0.1 + roi_centre_nm[0]
                lab_y_nm = y_axis * 0.1 + roi_centre_nm[1]
                finite_mask = scene.sample_contains_xy(
                    lab_x_nm[None, :],
                    lab_y_nm[:, None],
                )
                configurations = tuple(
                    np.asarray(configuration) * finite_mask[None, :, :]
                    for configuration in configurations
                )
                mean_projected = np.asarray(mean_projected) * finite_mask
            return PreparedSpecimen(
                x_angstrom=x_axis,
                y_angstrom=y_axis,
                potential_configurations_v_angstrom=(
                    configurations
                ),
                mean_projected_potential_v_angstrom=(
                    mean_projected
                ),
                slice_thicknesses_angstrom=(
                    ensemble.slice_thicknesses_angstrom
                ),
                metrics={
                    **domain_metrics,
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
                        *scene.size_xy_nm,
                        thickness_nm,
                    ),
                    "finite_specimen_shape": scene.envelope_shape,
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
        origin_angstrom_xy=(
            (roi_centre_nm[0] - scene.centre_xy_nm[0]) * 10.0,
            (roi_centre_nm[1] - scene.centre_xy_nm[1]) * 10.0,
        ),
    )
    if calculation_roi_bounds_nm is not None and thickness_nm > 0.0:
        lab_x_nm = x_axis * 0.1 + roi_centre_nm[0]
        lab_y_nm = y_axis * 0.1 + roi_centre_nm[1]
        finite_mask = scene.sample_contains_xy(
            lab_x_nm[None, :],
            lab_y_nm[:, None],
        )
        potential = potential * finite_mask
    return PreparedSpecimen(
        x_angstrom=x_axis,
        y_angstrom=y_axis,
        potential_configurations_v_angstrom=(potential,),
        mean_projected_potential_v_angstrom=potential,
        slice_thicknesses_angstrom=None,
        metrics={
            **domain_metrics,
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
                *scene.size_xy_nm,
                thickness_nm,
            ),
            "finite_specimen_shape": scene.envelope_shape,
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
    alive = np.asarray(incident.alive, dtype=bool)
    weights = np.asarray(incident.ray_weight, dtype=float)[alive]
    weights /= max(float(weights.sum()), 1.0e-30)
    phase_space = np.stack(
        (
            np.asarray(incident.x[-1], dtype=float)[alive],
            np.asarray(incident.y[-1], dtype=float)[alive],
            np.asarray(incident.tx[-1], dtype=float)[alive],
            np.asarray(incident.ty[-1], dtype=float)[alive],
        ),
        axis=1,
    )
    centre = np.sum(weights[:, None] * phase_space, axis=0)
    centred = phase_space - centre
    covariance = (centred * weights[:, None]).T @ centred
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
        "radial_position_angle_covariance_m_rad": (
            statistics.radial_position_angle_covariance_m_rad
        ),
        "radial_wavefront_curvature_per_m": (
            statistics.radial_wavefront_curvature_per_m
        ),
        "waist_offset_m": statistics.waist_offset_m,
        "surviving_rays": statistics.surviving_rays,
        "phase_space_covariance": covariance.tolist(),
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
        xx_m = xx * 1.0e-10
        yy_m = yy * 1.0e-10
        delta_x = xx_m - float(ray_stats["mean_x_m"])
        delta_y = yy_m - float(ray_stats["mean_y_m"])
        covariance = np.asarray(
            ray_stats["phase_space_covariance"], dtype=float
        )
        position_covariance = covariance[:2, :2]
        eigenvalues = np.linalg.eigvalsh(position_covariance)
        if float(eigenvalues[-1]) <= 1.0e-30:
            amplitude = np.ones_like(xx_m)
            curvature = np.zeros((2, 2), dtype=float)
        else:
            floor = max(
                (max(abs(spacing_x), abs(spacing_y)) * 1.0e-10) ** 2,
                float(eigenvalues[-1]) * 1.0e-12,
            )
            regularised = position_covariance + np.eye(2) * floor
            inverse_position = np.linalg.inv(regularised)
            radius_squared = (
                inverse_position[0, 0] * delta_x**2
                + 2.0 * inverse_position[0, 1] * delta_x * delta_y
                + inverse_position[1, 1] * delta_y**2
            )
            amplitude = np.exp(-0.25 * np.maximum(radius_squared, 0.0))
            curvature = covariance[2:, :2] @ inverse_position
            curvature = 0.5 * (curvature + curvature.T)
        phase_path_m = (
            float(ray_stats["mean_tx_rad"]) * delta_x
            + float(ray_stats["mean_ty_rad"]) * delta_y
            + 0.5
            * (
                curvature[0, 0] * delta_x**2
                + 2.0 * curvature[0, 1] * delta_x * delta_y
                + curvature[1, 1] * delta_y**2
            )
        )
        phase = 2.0 * math.pi * phase_path_m / (
            wavelength_angstrom * 1.0e-10
        )
        wave = amplitude * np.exp(1j * phase)
        return wave / math.sqrt(
            max(float(np.mean(np.abs(wave) ** 2)), 1.0e-30)
        )

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
    wave = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(spectrum)))
    return wave / math.sqrt(max(float(np.mean(np.abs(wave) ** 2)), 1.0e-30))


def _objective_aperture_rad(state) -> float:
    aperture = state.objective_aperture
    if not bool(getattr(aperture, "enabled", True)):
        return math.inf
    distance_mm = abs(float(aperture.z_mm) - float(state.sample.z_mm))
    if distance_mm <= 1.0e-12:
        return math.inf
    return max(float(aperture.radius_mm), 0.0) / distance_mm


def simulate_wave_image(state, simulation) -> WaveImagingResult:
    scene = SpecimenScene.from_state(state)
    specimen_interaction = bool(
        not scene.is_vacuum and scene.structure_available
    )
    preset_key = scene.wave_template_key
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
    specimen_metrics["sample_inserted"] = scene.inserted
    specimen_metrics["sample_interaction_applied"] = specimen_interaction

    objective = state.objective_lens
    focal_mm = float(
        objective.focal_length_for_voltage_mm(state.beam_voltage_kv)
    )
    effective_aberrations = active_effective_aberrations(state, "image")
    configured_image_defocus_mm = float(
        (getattr(state, "image_aberrations", {}) or {}).get(
            "c1_mm",
            float(getattr(state.sample, "wave_defocus_nm", 0.0)) * 1.0e-6,
        )
    )
    # Objective/D/I/P1/P2 first-order focus is already present in the complete
    # sample-to-camera transfer.  Retain only an explicit specimen-referenced
    # image-defocus request in the residual pupil phase to avoid double count.
    effective_aberrations = replace(
        effective_aberrations, c1_mm=configured_image_defocus_mm
    )
    cs_mm = effective_aberrations.c3_mm
    if math.isfinite(focal_mm):
        chi = aberration_phase_rad(
            fx,
            fy,
            wavelength_angstrom,
            effective_aberrations,
        )
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
    raw_diffraction = np.zeros((ny, nx), dtype=np.float64)
    coherent_exit_wave = np.zeros((ny, nx), dtype=np.complex128)
    objective_wave_configurations = []
    unapertured_wave_configurations = []
    fft_records = []
    for exit_configuration in exit_waves:
        # Retain the residual aberration phase, but not the removable pupil.
        unapertured, _, _ = apply_coherent_transfer(
            exit_configuration, np.exp(-1j * chi),
            compute_backend=fft_backend, fallback_reason=fft_fallback_seed)
        unapertured_wave_configurations.append(_readonly_array(unapertured))
        objective_image_wave, diffraction_configuration, fft_diagnostics = (
            apply_coherent_transfer(
                exit_configuration,
                transfer,
                compute_backend=fft_backend,
                fallback_reason=fft_fallback_seed,
            )
        )
        objective_wave_configurations.append(
            _readonly_array(objective_image_wave)
        )
        raw_diffraction += diffraction_configuration
        coherent_exit_wave += exit_configuration
        fft_records.append(fft_diagnostics)
        if fft_diagnostics.compute_backend != fft_backend:
            fft_backend = fft_diagnostics.compute_backend
            fft_fallback_seed = fft_diagnostics.fallback_reason
    projector_checkpoint = ProjectorWaveCheckpoint(
        unapertured_wave_configurations=tuple(unapertured_wave_configurations),
        objective_aperture_rad=float(aperture_rad),
        objective_wave_configurations=tuple(objective_wave_configurations),
        x_angstrom=_readonly_array(x_axis, dtype=np.float64),
        y_angstrom=_readonly_array(y_axis, dtype=np.float64),
        wavelength_angstrom=float(wavelength_angstrom),
        convergence_semiangle_rad=float(
            ray_stats["convergence_semiangle_rad"]
        ),
    )
    (
        raw_image,
        raw_electron_optical_image,
        image_m2,
        camera_projection,
    ) = _project_objective_configurations(state, projector_checkpoint)
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
        "objective_defocus_nm": configured_image_defocus_mm * 1.0e6,
        "objective_focused": math.isfinite(focal_mm),
        "objective_cs_mm": cs_mm,
        "effective_aberration_correction_state": (
            effective_aberrations.correction_state
        ),
        "effective_aberrations": {
            name: value
            for name, value in effective_aberrations.__dict__.items()
        },
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
        "image_formation_scope": (
            "electron gun through complete illumination column to specimen; "
            "multislice interaction; Objective pupil and complete projector "
            "field transfer to the active recording plane with terminal mask "
            "and detector PSF; intermediate post-sample aperture wave masks "
            "are not yet applied"
        ),
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
        "projector_checkpoint_reused": False,
        "projector_checkpoint_configuration_count": len(
            projector_checkpoint.objective_wave_configurations
        ),
        "projector_checkpoint_scope": (
            "specimen and Objective pupil/aberration output retained before "
            "the Diffraction/Intermediate/P1/P2 recording-plane transfer"
        ),
        "wave_sampling_truncates_illumination": bool(
            ray_stats["convergence_semiangle_rad"] * 1.0e3
            > float(specimen_metrics["maximum_isotropic_angle_mrad"])
        ),
        "wave_intensity_conservation_within_0_1_percent": bool(
            float(specimen_metrics["maximum_relative_intensity_change"])
            <= 1.0e-3
        ),
        **camera_projection.metrics,
    }
    metrics["camera_collected_zero_loss_relative_intensity"] = (
        _detector_probability(
            raw_image,
            camera_projection.x_mm,
            camera_projection.y_mm,
        )
    )
    metrics["camera_collected_intensity_aggregation"] = (
        "incoherent_configuration_mean"
    )
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
        camera_electron_optical_intensity=raw_electron_optical_image,
        camera_x_mm=camera_projection.x_mm,
        camera_y_mm=camera_projection.y_mm,
        spatial_frequency_inv_angstrom=frequencies_x,
        spatial_frequency_y_inv_angstrom=frequencies_y,
        metrics=metrics,
        projector_checkpoint=projector_checkpoint,
    )
