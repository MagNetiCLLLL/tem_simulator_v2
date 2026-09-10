"""Paraxial wave transfer from the specimen exit surface to the camera.

The ray integrator is the authority for the ordered post-specimen column.  Its
signed laboratory-frame 4x4 Jacobian is used here as a two-dimensional linear
canonical transform (LCT), so Objective, Diffraction, Intermediate, P1, P2,
    round-lens rotation and every enabled quadrupole/stigmator all affect the
    recording-plane wave.  Deflectors are affine rather than Jacobian terms and
    are traced separately as a recording-plane displacement.

This is exact for the simulator's first-order paraxial Hamiltonian model.  It
does not claim an OEM field calibration or replace a full Maxwell/Schrodinger
field solution.  Higher-order image aberrations are applied to the specimen
exit spectrum before the LCT by :mod:`temsim.optics.aberrations`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from temsim.detector.point_spread import (
    DetectorPointSpread,
    apply_point_spread,
)
from temsim.physics.core import propagate
from temsim.physics.first_order import (
    detector_frame_from_component,
    linear_map_properties,
)
from temsim.physics.recording_stop import active_tem_recording_plane


@dataclass(frozen=True, slots=True)
class CameraWaveProjection:
    """Camera-plane intensity and inspectable transfer diagnostics.

    ``x_mm`` and ``y_mm`` are detector readout coordinates at pixel centres.
    ``intensity`` is the forward detector response after the configured PSF;
    it is not display-normalised.
    """

    x_mm: np.ndarray
    y_mm: np.ndarray
    intensity: np.ndarray
    electron_optical_intensity: np.ndarray
    transfer_matrix: np.ndarray
    affine_offset_mm: tuple[float, float]
    method: str
    metrics: dict


def _post_sample_kick_events(state, target_z_mm: float):
    """Collect the same enabled affine events used by ``simulation.run``."""

    sample_z_mm = float(state.sample.z_mm)
    target_z_mm = float(target_z_mm)
    events = []
    for collection_name in ("deflectors", "corrector_elements"):
        for component in getattr(state, collection_name, ()):
            if not bool(getattr(component, "enabled", False)):
                continue
            if not hasattr(component, "kick_events"):
                continue
            try:
                component_events = component.kick_events(
                    time_s=float(getattr(state, "simulation_time_s", 0.0))
                )
            except TypeError:
                component_events = component.kick_events()
            for event in component_events:
                z_mm, kick_x_rad, kick_y_rad = (
                    float(value) for value in event
                )
                if sample_z_mm <= z_mm <= target_z_mm:
                    events.append((z_mm, kick_x_rad, kick_y_rad))
    return tuple(sorted(events, key=lambda event: event[0]))


def _camera_affine_offset_m(state, camera) -> np.ndarray:
    """Trace the zero ray to retain post-specimen deflector displacement."""

    target_z_mm = float(camera.z_mm)
    zero = np.zeros(1, dtype=float)
    result = propagate(
        state,
        float(state.sample.z_mm),
        target_z_mm,
        zero,
        zero,
        zero,
        zero,
        _post_sample_kick_events(state, target_z_mm),
        include_spherical_aberration=False,
        include_hexapole=False,
        save_z_mm=(target_z_mm,),
    )
    return np.asarray((result[1][-1, 0], result[3][-1, 0]), dtype=float)


def _normalise_wave(wave: np.ndarray, dx_m: float, dy_m: float) -> np.ndarray:
    norm = float(np.sum(np.abs(wave) ** 2) * dx_m * dy_m)
    if not math.isfinite(norm):
        raise ValueError("Specimen exit wave has non-finite intensity.")
    if norm == 0:
        return np.zeros_like(wave, dtype=np.complex128)
    return np.asarray(wave, dtype=np.complex128) / math.sqrt(norm)


def _deposit_mapped_probability(
    probability: np.ndarray,
    mapped_x_m: np.ndarray,
    mapped_y_m: np.ndarray,
    camera_x_m: np.ndarray,
    camera_y_m: np.ndarray,
) -> np.ndarray:
    """Conservatively integrate mapped probability into detector pixels."""

    dx_camera = float(camera_x_m[1] - camera_x_m[0])
    dy_camera = float(camera_y_m[1] - camera_y_m[0])
    ix = (mapped_x_m - float(camera_x_m[0])) / dx_camera
    iy = (mapped_y_m - float(camera_y_m[0])) / dy_camera
    ix0 = np.floor(ix).astype(np.int64)
    iy0 = np.floor(iy).astype(np.int64)
    fx = ix - ix0
    fy = iy - iy0
    inside_sensor = (
        (mapped_x_m >= float(camera_x_m[0]) - 0.5 * dx_camera)
        & (mapped_x_m < float(camera_x_m[-1]) + 0.5 * dx_camera)
        & (mapped_y_m >= float(camera_y_m[0]) - 0.5 * dy_camera)
        & (mapped_y_m < float(camera_y_m[-1]) + 0.5 * dy_camera)
    )
    targets = []
    normalisation = np.zeros_like(probability, dtype=float)
    for offset_x, weight_x in ((0, 1.0 - fx), (1, fx)):
        for offset_y, weight_y in ((0, 1.0 - fy), (1, fy)):
            target_x = ix0 + offset_x
            target_y = iy0 + offset_y
            weight = weight_x * weight_y
            valid = (
                inside_sensor
                & (target_x >= 0)
                & (target_x < camera_x_m.size)
                & (target_y >= 0)
                & (target_y < camera_y_m.size)
            )
            normalisation[valid] += weight[valid]
            targets.append((target_x, target_y, weight, valid))
    deposited = np.zeros((camera_y_m.size, camera_x_m.size), dtype=float)
    for target_x, target_y, weight, valid in targets:
        np.add.at(
            deposited,
            (target_y[valid], target_x[valid]),
            (
                probability[valid]
                * weight[valid]
                / normalisation[valid]
            ),
        )
    return deposited / (dx_camera * dy_camera)


def _geometric_image_intensity(
    wave: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    camera_x_m: np.ndarray,
    camera_y_m: np.ndarray,
    a_block: np.ndarray,
    offset_m: np.ndarray,
) -> np.ndarray:
    """Conservatively integrate a conjugate image into Camera pixels.

    Point-sampling a highly binned detector can miss an image that is much
    smaller than one output pixel.  Bilinear charge deposition instead models
    pixel integration and preserves the finite-sensor collected intensity.
    """

    xx, yy = np.meshgrid(x_m, y_m, indexing="xy")
    source = np.stack((xx, yy), axis=0)
    mapped = np.einsum("ij,jyx->iyx", a_block, source)
    mapped[0] += offset_m[0]
    mapped[1] += offset_m[1]
    dx_source = float(x_m[1] - x_m[0])
    dy_source = float(y_m[1] - y_m[0])
    probability = np.abs(wave) ** 2 * dx_source * dy_source
    return _deposit_mapped_probability(
        probability,
        mapped[0],
        mapped[1],
        camera_x_m,
        camera_y_m,
    )


def _collins_lct_intensity(
    wave: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    camera_x_m: np.ndarray,
    camera_y_m: np.ndarray,
    a_block: np.ndarray,
    b_block_m_per_rad: np.ndarray,
    offset_m: np.ndarray,
    wavelength_m: float,
) -> np.ndarray:
    """Conservatively bin a Collins transform into detector pixels.

    The FFT is evaluated on its natural reciprocal grid.  Each sample carries
    its Parseval probability and is mapped by ``u = offset + wavelength B f``
    before bilinear detector-pixel integration.  This avoids losing a narrow
    diffraction pattern between coarse physical pixel centres.
    """

    determinant = float(np.linalg.det(b_block_m_per_rad))
    singular_values = np.linalg.svd(b_block_m_per_rad, compute_uv=False)
    if (
        abs(determinant) <= 1.0e-30
        or float(singular_values[-1])
        <= max(float(singular_values[0]) * 1.0e-10, 1.0e-15)
    ):
        raise ValueError(
            "Camera transfer has a rank-deficient B block outside the "
            "sampled image-conjugate limit."
        )

    inverse_b = np.linalg.inv(b_block_m_per_rad)
    quadratic = inverse_b @ a_block
    # Symplectic first-order propagation makes B^-1 A symmetric.  Numerical
    # integration leaves a small antisymmetric residual that carries no
    # scalar quadratic phase, so remove only that round-off component.
    quadratic = 0.5 * (quadratic + quadratic.T)
    xx, yy = np.meshgrid(x_m, y_m, indexing="xy")
    source_coordinates = np.stack((xx, yy), axis=0)
    source_phase = np.einsum(
        "iyx,ij,jyx->yx", source_coordinates, quadratic, source_coordinates
    )
    prechirped = wave * np.exp(1j * math.pi * source_phase / wavelength_m)

    dx_m = float(x_m[1] - x_m[0])
    dy_m = float(y_m[1] - y_m[0])
    spectrum = (
        np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(prechirped)))
        * dx_m
        * dy_m
    )
    frequency_x = np.fft.fftshift(np.fft.fftfreq(x_m.size, d=dx_m))
    frequency_y = np.fft.fftshift(np.fft.fftfreq(y_m.size, d=dy_m))
    dfx_m1 = float(frequency_x[1] - frequency_x[0])
    dfy_m1 = float(frequency_y[1] - frequency_y[0])
    frequency_xx, frequency_yy = np.meshgrid(
        frequency_x, frequency_y, indexing="xy"
    )
    frequency_coordinates = np.stack(
        (frequency_xx, frequency_yy), axis=0
    )
    mapped = wavelength_m * np.einsum(
        "ij,jyx->iyx", b_block_m_per_rad, frequency_coordinates
    )
    mapped[0] += offset_m[0]
    mapped[1] += offset_m[1]
    probability = np.abs(spectrum) ** 2 * dfx_m1 * dfy_m1
    return _deposit_mapped_probability(
        probability,
        mapped[0],
        mapped[1],
        camera_x_m,
        camera_y_m,
    )


def _project_wave_to_plane(
    state,
    exit_wave,
    x_angstrom,
    y_angstrom,
    wavelength_angstrom: float,
    *,
    recording_plane,
    convergence_semiangle_rad: float = 0.0,
    input_convention: str = "legacy_unit_shape",
    excluded_aperture_keys: tuple[str, ...] = (),
) -> CameraWaveProjection:
    """Propagate a specimen wave through the full projector to one stop."""

    recording_plane = recording_plane.validate()
    if not bool(recording_plane.inserted):
        raise ValueError("The selected TEM recording plane is retracted.")
    x_m = np.asarray(x_angstrom, dtype=float) * 1.0e-10
    y_m = np.asarray(y_angstrom, dtype=float) * 1.0e-10
    wave = np.asarray(exit_wave, dtype=np.complex128)
    if wave.shape != (y_m.size, x_m.size):
        raise ValueError("Exit-wave shape must match specimen X/Y axes.")
    if x_m.size < 2 or y_m.size < 2:
        raise ValueError("Camera wave propagation requires at least two samples per axis.")
    dx_m = float(x_m[1] - x_m[0])
    dy_m = float(y_m[1] - y_m[0])
    if dx_m <= 0.0 or dy_m <= 0.0:
        raise ValueError("Specimen wave axes must be strictly increasing.")
    wavelength_m = float(wavelength_angstrom) * 1.0e-10
    if not math.isfinite(wavelength_m) or wavelength_m <= 0.0:
        raise ValueError("Electron wavelength must be finite and positive.")
    if input_convention == "legacy_unit_shape":
        # Compatibility for standalone callers with arbitrary shape units.
        # Production TEM passes an explicitly weighted SI density instead.
        wave = _normalise_wave(wave, dx_m, dy_m)
    elif input_convention != "weighted_density_per_m":
        raise ValueError(f"Unknown wave input convention: {input_convention}")
    if not np.all(np.isfinite(wave)):
        raise ValueError("Specimen exit wave contains non-finite amplitudes")
    input_probability = float(np.sum(np.abs(wave)**2) * dx_m * dy_m)

    projector_mode = str(getattr(state, "projector_mode", "image")).lower()
    # The specimen lies inside the Objective field, so the Fourier coordinate
    # of every exit wave is canonical momentum in both image and diffraction
    # modes.  The mode selects the intended conjugacy, not a different phase-
    # space basis.  ``diffraction_transfer`` is the established stable
    # specimen-canonical transfer used by the projector solver.
    from temsim.optics.direct_alignment import diffraction_transfer

    transfer = diffraction_transfer(
        state,
        float(recording_plane.z_mm),
    )
    transfer_basis = "specimen_canonical_momentum"
    detector_frame = detector_frame_from_component(recording_plane)
    rotation = detector_frame.column_to_detector
    a_block = rotation @ np.asarray(transfer.j_img, dtype=float)
    b_block = rotation @ np.asarray(
        transfer.j_diff_m_per_rad, dtype=float
    )
    offset_m = rotation @ _camera_affine_offset_m(state, recording_plane)

    # Use the specimen calculation grid as explicit on-chip binning.  Setting
    # the wave grid to camera.pixels produces native detector sampling; smaller
    # wave grids calculate an integer/non-integer binned camera image without
    # pretending to add specimen information by interpolation.
    hardware_pixels_value = getattr(recording_plane, "pixels", None)
    hardware_pixels = (
        int(hardware_pixels_value)
        if hardware_pixels_value is not None
        else None
    )
    pixels_x = (
        min(hardware_pixels, int(x_m.size))
        if hardware_pixels is not None
        else int(x_m.size)
    )
    pixels_y = (
        min(hardware_pixels, int(y_m.size))
        if hardware_pixels is not None
        else int(y_m.size)
    )
    width_mm = float(
        getattr(
            recording_plane,
            "width_mm",
            getattr(recording_plane, "outer_width_mm"),
        )
    )
    width_m = width_mm * 1.0e-3
    pixel_x_m = width_m / pixels_x
    pixel_y_m = width_m / pixels_y
    camera_x_m = (
        np.arange(pixels_x, dtype=float) - 0.5 * (pixels_x - 1)
    ) * pixel_x_m
    camera_y_m = (
        np.arange(pixels_y, dtype=float) - 0.5 * (pixels_y - 1)
    ) * pixel_y_m

    # The exit wave can carry specimen spatial frequencies far beyond the
    # illumination cone.  Use the represented Nyquist band when deciding
    # whether a nearly image-conjugate B block is negligible.  Diffraction
    # mode must always retain B and the wave phase; a geometric A*x mapping
    # cannot form a diffraction pattern.
    represented_angle_rad = wavelength_m * math.hypot(
        0.5 / dx_m,
        0.5 / dy_m,
    )
    blur_angle_rad = max(
        float(convergence_semiangle_rad), represented_angle_rad
    )
    blur_m = float(np.linalg.norm(b_block, ord=2)) * blur_angle_rad
    camera_pixel_m = max(pixel_x_m, pixel_y_m)
    from temsim.physics.multiplane_wave import intermediate_apertures, project_through_apertures
    aperture_rows = ()
    active_apertures = intermediate_apertures(state, float(recording_plane.z_mm), excluded_keys=excluded_aperture_keys)
    if active_apertures or input_convention == "weighted_density_per_m":
        final_wave, aperture_rows = project_through_apertures(
            state, wave, x_m, y_m, wavelength_m, float(recording_plane.z_mm),
            excluded_keys=excluded_aperture_keys,
        )
        coordinates = np.einsum("ij,jyx->iyx", rotation, final_wave.coordinates_m())
        electron_optical_intensity = _deposit_mapped_probability(
            np.abs(final_wave.amplitude) ** 2, coordinates[0], coordinates[1], camera_x_m, camera_y_m,
        )
        method = ("multiplane_coherent_lct_with_apertures" if active_apertures
                  else "multiplane_coherent_lct")
    elif (
        projector_mode != "diffraction"
        and blur_m <= 0.25 * camera_pixel_m
    ):
        electron_optical_intensity = _geometric_image_intensity(
            wave,
            x_m,
            y_m,
            camera_x_m,
            camera_y_m,
            a_block,
            offset_m,
        )
        method = "image_conjugate_linear_canonical_transform"
    else:
        electron_optical_intensity = _collins_lct_intensity(
            wave,
            x_m,
            y_m,
            camera_x_m,
            camera_y_m,
            a_block,
            b_block,
            offset_m,
            wavelength_m,
        )
        method = "collins_fft_linear_canonical_transform"

    hit_mask = np.asarray(
        recording_plane.hit_mask(
            camera_x_m[None, :] * 1.0e3,
            camera_y_m[:, None] * 1.0e3,
        ),
        dtype=bool,
    )
    electron_optical_intensity = np.where(
        hit_mask, electron_optical_intensity, 0.0
    )
    point_spread = DetectorPointSpread.from_component(recording_plane)
    detector_intensity = apply_point_spread(
        electron_optical_intensity,
        point_spread,
        pixel_size_x_mm=pixel_x_m * 1.0e3,
        pixel_size_y_mm=pixel_y_m * 1.0e3,
    )
    detector_intensity = np.where(hit_mask, detector_intensity, 0.0)
    camera_integral = float(
        np.sum(detector_intensity) * pixel_x_m * pixel_y_m
    )
    properties = linear_map_properties(a_block)
    from temsim.physics.wave_flux import FluxLedger
    ledger = FluxLedger(branch_id="camera")
    remaining = input_probability
    for row in aperture_rows:
        ledger.record("physical_aperture:" + row["key"], row["incoming_probability"],
                      row["transmitted_probability"], "physical_stop",
                      physical_element_id=row["physical_element_id"],
                      parameters={k: row[k] for k in ("z_mm", "radius_mm", "offset_x_mm", "offset_y_mm", "transfer_matrix")})
        remaining = row["transmitted_probability"]
    before_psf = float(np.sum(electron_optical_intensity) * pixel_x_m * pixel_y_m)
    ledger.record("recording_plane:" + str(recording_plane.key), remaining, before_psf,
                  "missed_detector", physical_element_id="detector:" + str(recording_plane.key),
                  parameters={"z_mm": float(recording_plane.z_mm), "width_mm": width_mm,
                              "transfer_matrix": transfer.matrix.tolist()})
    ledger.record("detector_response", before_psf, camera_integral, "detector_response_crop",
                  parameters={"psf_model": point_spread.model, "sigma_x_mm": point_spread.sigma_x_mm,
                              "sigma_y_mm": point_spread.sigma_y_mm})
    metrics = {
        "camera_input_convention": input_convention,
        "camera_input_probability": input_probability,
        "camera_pre_psf_probability": before_psf,
        "camera_flux_ledger": ledger.weighted_rows(1.),
        "recording_plane_key": str(recording_plane.key),
        "recording_plane_name": str(recording_plane.name),
        "recording_plane_z_mm": float(recording_plane.z_mm),
        "recording_plane_width_mm": width_mm,
        "recording_plane_hardware_pixels": hardware_pixels,
        "recording_plane_sampling_model": (
            "native_pixel_binning"
            if hardware_pixels is not None
            else "continuous_screen_numerical_grid"
        ),
        # Legacy Camera-prefixed aliases remain for saved diagnostics and GUI
        # consumers while the target may now be the Fluorescent Screen.
        "camera_key": str(recording_plane.key),
        "camera_z_mm": float(recording_plane.z_mm),
        "camera_width_mm": width_mm,
        "camera_hardware_pixels": hardware_pixels,
        "camera_calculation_pixels_xy": (pixels_x, pixels_y),
        "camera_pixel_size_mm_xy": (
            pixel_x_m * 1.0e3,
            pixel_y_m * 1.0e3,
        ),
        "camera_binning_xy": (
            (
                float(hardware_pixels) / pixels_x,
                float(hardware_pixels) / pixels_y,
            )
            if hardware_pixels is not None
            else (None, None)
        ),
        "camera_affine_offset_mm_xy": tuple(offset_m * 1.0e3),
        "camera_collected_zero_loss_relative_intensity": camera_integral,
        "camera_point_spread_model": point_spread.model,
        "camera_point_spread_sigma_mm_xy": (
            point_spread.sigma_x_mm,
            point_spread.sigma_y_mm,
        ),
        "camera_point_spread_status": point_spread.status,
        "camera_detector_orientation_status": detector_frame.status,
        "projector_transfer_matrix": transfer.matrix.tolist(),
        "projector_transfer_input_basis": transfer_basis,
        "projector_image_map": a_block.tolist(),
        "projector_diffraction_map_m_per_rad": b_block.tolist(),
        "projector_magnification": properties.isotropic_scale,
        "projector_anisotropy_ratio": properties.anisotropy_ratio,
        "projector_orientation_deg": properties.orientation_deg,
        "projector_mirrored": properties.mirrored,
        "projector_conjugacy_blur_estimate_mm": blur_m * 1.0e3,
        "projector_exit_wave_bandlimit_mrad": represented_angle_rad * 1.0e3,
        "camera_wave_propagation_method": method,
        "camera_projector_mode": projector_mode,
        "recording_plane_observable": (
            "diffraction_pattern"
            if projector_mode == "diffraction"
            else "image"
        ),
        "intermediate_post_sample_masks_applied": bool(aperture_rows),
        "intermediate_aperture_transmissions": aperture_rows,
        "camera_wave_model_scope": (
            "specimen exit to the first inserted TEM recording stop through "
            "the enabled post-specimen paraxial fields; terminal detector "
            "mask and PSF applied; enabled intermediate aperture masks use "
            "coherent split-plane LCTs; unresolved sampling fails explicitly"
        ),
    }
    return CameraWaveProjection(
        x_mm=camera_x_m * 1.0e3,
        y_mm=camera_y_m * 1.0e3,
        intensity=np.asarray(detector_intensity, dtype=float),
        electron_optical_intensity=np.asarray(
            electron_optical_intensity, dtype=float
        ),
        transfer_matrix=transfer.matrix,
        affine_offset_mm=tuple(offset_m * 1.0e3),
        method=method,
        metrics=metrics,
    )


def project_wave_to_recording_plane(
    state,
    exit_wave,
    x_angstrom,
    y_angstrom,
    wavelength_angstrom: float,
    *,
    convergence_semiangle_rad: float = 0.0,
    input_convention: str = "legacy_unit_shape",
    excluded_aperture_keys: tuple[str, ...] = (),
) -> CameraWaveProjection:
    """Project to the first inserted Fluorescent Screen or Camera."""

    return _project_wave_to_plane(
        state,
        exit_wave,
        x_angstrom,
        y_angstrom,
        wavelength_angstrom,
        recording_plane=active_tem_recording_plane(state),
        convergence_semiangle_rad=convergence_semiangle_rad,
        input_convention=input_convention,
        excluded_aperture_keys=excluded_aperture_keys,
    )


def project_wave_to_camera(
    state,
    exit_wave,
    x_angstrom,
    y_angstrom,
    wavelength_angstrom: float,
    *,
    convergence_semiangle_rad: float = 0.0,
    input_convention: str = "legacy_unit_shape",
    excluded_aperture_keys: tuple[str, ...] = (),
) -> CameraWaveProjection:
    """Backward-compatible explicit Camera projection entry point."""

    camera = state.camera
    return _project_wave_to_plane(
        state,
        exit_wave,
        x_angstrom,
        y_angstrom,
        wavelength_angstrom,
        recording_plane=camera,
        convergence_semiangle_rad=convergence_semiangle_rad,
        input_convention=input_convention,
        excluded_aperture_keys=excluded_aperture_keys,
    )
