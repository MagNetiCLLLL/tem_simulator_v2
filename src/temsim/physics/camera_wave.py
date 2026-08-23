"""Paraxial wave transfer from the specimen exit surface to the camera.

The ray integrator is the authority for the ordered post-specimen column.  Its
signed laboratory-frame 4x4 Jacobian is used here as a two-dimensional linear
canonical transform (LCT), so Objective, Diffraction, Intermediate, P1, P2,
round-lens rotation and every enabled quadrupole/stigmator all affect the
camera wave.  Deflectors are affine rather than Jacobian terms and are traced
separately as a camera-plane displacement.

This is exact for the simulator's first-order paraxial Hamiltonian model.  It
does not claim an OEM field calibration or replace a full Maxwell/Schrodinger
field solution.  Higher-order image aberrations are applied to the specimen
exit spectrum before the LCT by :mod:`temsim.optics.aberrations`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import map_coordinates

from temsim.detector.point_spread import (
    DetectorPointSpread,
    apply_point_spread,
)
from temsim.physics.core import propagate
from temsim.physics.first_order import (
    detector_frame_from_component,
    linear_map_properties,
    trace_transverse_transfer,
)


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
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("Specimen exit wave has no finite positive intensity.")
    return np.asarray(wave, dtype=np.complex128) / math.sqrt(norm)


def _sample_complex_grid(
    values: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    query_x: np.ndarray,
    query_y: np.ndarray,
) -> np.ndarray:
    """Bilinearly sample one complex regular grid with zero exterior."""

    dx = float(x_axis[1] - x_axis[0])
    dy = float(y_axis[1] - y_axis[0])
    indices = np.vstack((
        ((query_y - float(y_axis[0])) / dy).ravel(),
        ((query_x - float(x_axis[0])) / dx).ravel(),
    ))
    real = map_coordinates(
        np.asarray(values.real, dtype=float),
        indices,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    imaginary = map_coordinates(
        np.asarray(values.imag, dtype=float),
        indices,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return (real + 1j * imaginary).reshape(query_x.shape)


def _geometric_image_wave(
    wave: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    camera_x_m: np.ndarray,
    camera_y_m: np.ndarray,
    a_block: np.ndarray,
    offset_m: np.ndarray,
) -> np.ndarray:
    """Evaluate the B=0 LCT (coherent magnified image) on camera pixels."""

    determinant = float(np.linalg.det(a_block))
    if abs(determinant) <= 1.0e-30:
        raise ValueError("Camera image A block is singular.")
    inverse_a = np.linalg.inv(a_block)
    uu, vv = np.meshgrid(camera_x_m, camera_y_m, indexing="xy")
    centred = np.stack((uu - offset_m[0], vv - offset_m[1]), axis=0)
    source = np.einsum("ij,jyx->iyx", inverse_a, centred)
    sampled = _sample_complex_grid(
        wave, x_m, y_m, source[0], source[1]
    )
    return sampled / math.sqrt(abs(determinant))


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
    dx_camera = float(camera_x_m[1] - camera_x_m[0])
    dy_camera = float(camera_y_m[1] - camera_y_m[0])
    ix = (mapped[0] - float(camera_x_m[0])) / dx_camera
    iy = (mapped[1] - float(camera_y_m[0])) / dy_camera
    ix0 = np.floor(ix).astype(np.int64)
    iy0 = np.floor(iy).astype(np.int64)
    fx = ix - ix0
    fy = iy - iy0
    probability = np.abs(wave) ** 2 * dx_source * dy_source
    deposited = np.zeros((camera_y_m.size, camera_x_m.size), dtype=float)
    for offset_x, weight_x in ((0, 1.0 - fx), (1, fx)):
        for offset_y, weight_y in ((0, 1.0 - fy), (1, fy)):
            target_x = ix0 + offset_x
            target_y = iy0 + offset_y
            valid = (
                (target_x >= 0)
                & (target_x < camera_x_m.size)
                & (target_y >= 0)
                & (target_y < camera_y_m.size)
            )
            np.add.at(
                deposited,
                (target_y[valid], target_x[valid]),
                (probability * weight_x * weight_y)[valid],
            )
    return deposited / (dx_camera * dy_camera)


def _collins_lct_wave(
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
    """Evaluate the two-dimensional Collins diffraction integral by FFT."""

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

    uu, vv = np.meshgrid(camera_x_m, camera_y_m, indexing="xy")
    centred = np.stack((uu - offset_m[0], vv - offset_m[1]), axis=0)
    query_frequency = np.einsum(
        "ij,jyx->iyx", inverse_b / wavelength_m, centred
    )
    sampled_spectrum = _sample_complex_grid(
        spectrum,
        frequency_x,
        frequency_y,
        query_frequency[0],
        query_frequency[1],
    )
    # The omitted output-only Collins chirp has unit magnitude and cannot
    # affect this terminal intensity detector.
    return sampled_spectrum / (wavelength_m * math.sqrt(abs(determinant)))


def project_wave_to_camera(
    state,
    exit_wave,
    x_angstrom,
    y_angstrom,
    wavelength_angstrom: float,
    *,
    convergence_semiangle_rad: float = 0.0,
) -> CameraWaveProjection:
    """Propagate a specimen exit wave through the full projector to Camera."""

    camera = state.camera.validate()
    if not bool(camera.inserted):
        raise ValueError("Camera must be inserted to calculate a camera image.")
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
    wave = _normalise_wave(wave, dx_m, dy_m)

    transfer = trace_transverse_transfer(
        state, float(state.sample.z_mm), float(camera.z_mm)
    )
    detector_frame = detector_frame_from_component(camera)
    rotation = detector_frame.column_to_detector
    a_block = rotation @ np.asarray(transfer.j_img, dtype=float)
    b_block = rotation @ np.asarray(
        transfer.j_diff_m_per_rad, dtype=float
    )
    offset_m = rotation @ _camera_affine_offset_m(state, camera)

    # Use the specimen calculation grid as explicit on-chip binning.  Setting
    # the wave grid to camera.pixels produces native detector sampling; smaller
    # wave grids calculate an integer/non-integer binned camera image without
    # pretending to add specimen information by interpolation.
    pixels_x = min(int(camera.pixels), int(x_m.size))
    pixels_y = min(int(camera.pixels), int(y_m.size))
    width_m = float(camera.width_mm) * 1.0e-3
    pixel_x_m = width_m / pixels_x
    pixel_y_m = width_m / pixels_y
    camera_x_m = (
        np.arange(pixels_x, dtype=float) - 0.5 * (pixels_x - 1)
    ) * pixel_x_m
    camera_y_m = (
        np.arange(pixels_y, dtype=float) - 0.5 * (pixels_y - 1)
    ) * pixel_y_m

    ray_angles = max(
        float(convergence_semiangle_rad),
        1.0e-6,
    )
    blur_m = float(np.linalg.norm(b_block, ord=2)) * ray_angles
    camera_pixel_m = max(pixel_x_m, pixel_y_m)
    if blur_m <= 0.25 * camera_pixel_m:
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
        camera_wave = _collins_lct_wave(
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
        electron_optical_intensity = np.abs(camera_wave) ** 2
        method = "collins_fft_linear_canonical_transform"

    point_spread = DetectorPointSpread.from_component(camera)
    detector_intensity = apply_point_spread(
        electron_optical_intensity,
        point_spread,
        pixel_size_x_mm=pixel_x_m * 1.0e3,
        pixel_size_y_mm=pixel_y_m * 1.0e3,
    )
    camera_integral = float(
        np.sum(detector_intensity) * pixel_x_m * pixel_y_m
    )
    properties = linear_map_properties(a_block)
    metrics = {
        "camera_key": str(camera.key),
        "camera_z_mm": float(camera.z_mm),
        "camera_width_mm": float(camera.width_mm),
        "camera_hardware_pixels": int(camera.pixels),
        "camera_calculation_pixels_xy": (pixels_x, pixels_y),
        "camera_pixel_size_mm_xy": (
            pixel_x_m * 1.0e3,
            pixel_y_m * 1.0e3,
        ),
        "camera_binning_xy": (
            float(camera.pixels) / pixels_x,
            float(camera.pixels) / pixels_y,
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
        "projector_image_map": a_block.tolist(),
        "projector_diffraction_map_m_per_rad": b_block.tolist(),
        "projector_magnification": properties.isotropic_scale,
        "projector_anisotropy_ratio": properties.anisotropy_ratio,
        "projector_orientation_deg": properties.orientation_deg,
        "projector_mirrored": properties.mirrored,
        "projector_conjugacy_blur_estimate_mm": blur_m * 1.0e3,
        "camera_wave_propagation_method": method,
        "camera_projector_mode": str(getattr(state, "projector_mode", "")),
        "camera_wave_model_scope": (
            "specimen exit to physical Camera through the complete enabled "
            "post-specimen paraxial column; detector PSF forward applied"
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
