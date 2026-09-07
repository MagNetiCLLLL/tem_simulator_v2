"""Coherent sampled LCTs and aperture masks on physical intermediate planes.

An affine two-dimensional lattice retains rotation and shear without resampling
the wave onto a fictitious axis-aligned grid. Amplitudes are probability per
pixel. FFTs are unitary; clipping never renormalises transmitted electrons.
"""

from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np


@dataclass(frozen=True)
class PlaneWave:
    amplitude: np.ndarray
    basis_m: np.ndarray
    origin_m: np.ndarray
    curvature_m1: np.ndarray | None = None
    tilt_rad: np.ndarray | None = None

    def coordinates_m(self):
        ny, nx = self.amplitude.shape
        yy, xx = np.meshgrid(np.arange(ny) - ny // 2, np.arange(nx) - nx // 2, indexing="ij")
        return self.origin_m[:, None, None] + np.einsum("ij,jyx->iyx", self.basis_m, np.stack((xx, yy)))

    @property
    def probability(self):
        return float(np.sum(np.abs(self.amplitude) ** 2))


def propagate_plane_wave(wave, matrix, translation, wavelength_m):
    """Apply one canonical affine map, retaining complex phase and flux."""
    matrix, shift = np.asarray(matrix, float), np.asarray(translation, float)
    a, b, c, d = matrix[:2, :2], matrix[:2, 2:], matrix[2:, :2], matrix[2:, 2:]
    xy = wave.coordinates_m()
    local_input = xy - wave.origin_m[:, None, None]
    curvature = np.zeros((2, 2)) if wave.curvature_m1 is None else wave.curvature_m1
    tilt = np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
    ny, nx = wave.amplitude.shape
    # A truly image-conjugate map has B=0. Merely small detector blur is not
    # sufficient to discard propagation phase at an intermediate aperture.
    theta = wavelength_m * np.linalg.norm(np.linalg.inv(wave.basis_m), ord=2)
    extent = np.linalg.norm(wave.basis_m, ord=2) * min(nx, ny)
    effective_a = a + b @ curvature
    if np.linalg.cond(effective_a) < 1e8:
        inverse_a = np.linalg.inv(effective_a)
        drift = inverse_a @ b
        if np.linalg.norm(drift, ord=2) * theta < .25 * extent:
            # M L(Q) = L(Q_out) S(A_eff) D(A_eff^-1 B). This
            # scaled angular-spectrum form resolves successive far-field
            # drifts without forcing another undersampled quadratic FFT.
            fy, fx = np.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nx), indexing="ij")
            frequency = np.einsum("ij,jyx->iyx", np.linalg.inv(wave.basis_m).T, np.stack((fx, fy)))
            phase = -np.pi*wavelength_m*np.einsum("iyx,ij,jyx->yx", frequency, drift, frequency)
            amplitude = (wave.amplitude if np.all(drift == 0)
                         else np.fft.ifft2(np.fft.fft2(wave.amplitude) * np.exp(1j*phase)))
            return PlaneWave(amplitude, effective_a @ wave.basis_m, a @ wave.origin_m + b @ tilt + shift[:2],
                             (c + d @ curvature) @ inverse_a, c @ wave.origin_m + d @ tilt + shift[2:])
    if np.linalg.norm(b, ord=2) * theta < 1e-7 * max(np.linalg.norm(a, ord=2) * extent, 1e-30):
        if abs(np.linalg.det(a)) < 1e-15:
            raise ValueError("Singular image-plane wave map")
        inverse_a = np.linalg.inv(a)
        return PlaneWave(wave.amplitude, a @ wave.basis_m, a @ wave.origin_m + shift[:2],
                         inverse_a.T @ curvature @ inverse_a + c @ inverse_a,
                         c @ wave.origin_m + d @ tilt + shift[2:])
    if np.linalg.cond(b) > 1e10:
        raise ValueError("Rank-deficient mixed-conjugacy wave map; increase plane separation or use a resolved grid")
    inverse = np.linalg.inv(b)
    chirp_matrix = inverse @ a + curvature
    action = .5 * np.einsum("iyx,ij,jyx->yx", local_input, chirp_matrix, local_input)
    phase = 2 * np.pi * action / wavelength_m
    # Undersampled chirps would generate false diffraction features. Reject
    # instead of presenting a aliased image as a high-accuracy result.
    occupied = np.abs(wave.amplitude) > np.max(np.abs(wave.amplitude)) * 1e-5
    for axis in (0, 1):
        adjacent = np.take(occupied, range(occupied.shape[axis] - 1), axis=axis) & np.take(occupied, range(1, occupied.shape[axis]), axis=axis)
        if np.any(np.abs(np.diff(phase, axis=axis))[adjacent] > np.pi):
            raise ValueError("Intermediate-plane phase is undersampled; refine the wave grid")
    amplitude = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(wave.amplitude * np.exp(1j * phase)), norm="ortho"))
    basis = wavelength_m * b @ np.linalg.inv(wave.basis_m).T @ np.diag((1 / nx, 1 / ny))
    # Keep the quadratic carrier analytic. Sampling it into wrapped phase and
    # later multiplying an opposite chirp would alias a perfectly valid field.
    return PlaneWave(amplitude, basis, a @ wave.origin_m + b @ tilt + shift[:2],
                     d @ inverse, c @ wave.origin_m + d @ tilt + shift[2:])


def _canonical_map_and_offset(state, z_mm):
    from temsim.optics.aberrations import _aberration_cache_signature
    from temsim.optics.direct_alignment import diffraction_transfer
    from temsim.physics.core import fields, electron, propagate
    from temsim.physics.camera_wave import _post_sample_kick_events
    events = _post_sample_kick_events(state, z_mm)
    signature = _aberration_cache_signature(state, "image")
    key = (float(z_mm), events)
    cached = getattr(state, "_multiplane_transfer_cache", None)
    if cached is None or cached[0] != signature:
        cached = (signature, {})
        state._multiplane_transfer_cache = cached
    if key in cached[1]:
        return cached[1][key]
    transfer = diffraction_transfer(state, z_mm)
    q, momentum, _ = electron(state)
    g = q * float(fields(np.array((z_mm,)), state)[0][0]) / (2 * momentum)
    # Canonical p = mechanical slope + q A / p0, A=(-By/2,Bx/2,0).
    gauge = np.eye(4)
    gauge[2:, :2] = np.array(((0, -g), (g, 0)))
    if events:
        zero = np.zeros(1)
        trace = propagate(state, float(state.sample.z_mm), z_mm, zero, zero, zero, zero,
                          events, include_spherical_aberration=False,
                          include_hexapole=False, save_z_mm=(z_mm,))
        offset = gauge @ np.array((trace[1][-1, 0], trace[3][-1, 0], trace[2][-1, 0], trace[4][-1, 0]))
    else:
        offset = gauge @ np.array((*transfer.position_offset_m, *transfer.angle_offset_rad))
    result = gauge @ transfer.matrix, offset
    for value in result:
        value.setflags(write=False)
    cached[1][key] = result
    return result


def intermediate_apertures(state, stop_z_mm):
    from temsim.physics.record_plane import _aperture_stop
    return tuple(sorted((_aperture_stop(state, aperture) for aperture in getattr(state, "apertures", ())
                         if bool(getattr(aperture, "enabled", True))
                         and bool(getattr(aperture, "installed", True))
                         and bool(getattr(aperture, "inserted", True))
                         and float(state.sample.z_mm) < float(aperture.z_mm) < stop_z_mm), key=lambda item: item.z_mm))


def project_through_apertures(state, wave, x_m, y_m, wavelength_m, target_z_mm):
    dx, dy = float(x_m[1] - x_m[0]), float(y_m[1] - y_m[0])
    current = PlaneWave(np.asarray(wave) * np.sqrt(dx * dy), np.diag((dx, dy)),
                        np.array((x_m[len(x_m)//2], y_m[len(y_m)//2])))
    previous, previous_offset = np.eye(4), np.zeros(4)
    rows = []
    for plane in (*intermediate_apertures(state, target_z_mm), SimpleNamespace(z_mm=target_z_mm)):
        matrix, offset = _canonical_map_and_offset(state, float(plane.z_mm))
        segment = matrix @ np.linalg.inv(previous)
        current = propagate_plane_wave(current, segment, offset - segment @ previous_offset, wavelength_m)
        if hasattr(plane, "key"):
            xy = current.coordinates_m()
            # Aperture APIs consume metres, unlike detector readout masks.
            mask = (plane.transmission_mask(xy[0], xy[1]) if plane.radius_mm > 0
                    else np.zeros(current.amplitude.shape, bool))
            before = current.probability
            current = PlaneWave(np.where(mask, current.amplitude, 0j), current.basis_m, current.origin_m,
                                current.curvature_m1, current.tilt_rad)
            rows.append({"key": plane.key, "z_mm": float(plane.z_mm), "incoming_probability": before,
                         "transmitted_probability": current.probability,
                         "aperture_diameter_pixels": 2*plane.radius_mm*1e-3 / np.linalg.norm(current.basis_m, ord=2)})
        previous, previous_offset = matrix, offset
    return current, tuple(rows)
