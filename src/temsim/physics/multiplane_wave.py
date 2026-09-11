"""Coherent sampled LCTs and aperture masks on physical intermediate planes.

An affine two-dimensional lattice retains rotation and shear without resampling
the wave onto a fictitious axis-aligned grid. Amplitudes are probability per
pixel. FFTs are unitary; clipping never renormalises transmitted electrons.
"""

from dataclasses import dataclass, replace
from types import SimpleNamespace
import math
import numpy as np


@dataclass(frozen=True)
class PlaneWave:
    amplitude: np.ndarray
    basis_m: np.ndarray
    origin_m: np.ndarray
    curvature_m1: np.ndarray | None = None
    tilt_rad: np.ndarray | None = None

    def __post_init__(self):
        amplitude = np.asarray(self.amplitude)
        if amplitude.ndim != 2 or min(amplitude.shape) < 2 or not np.all(np.isfinite(amplitude)):
            raise ValueError("Plane wave must be a finite two-dimensional sampled amplitude")
        basis, origin = np.asarray(self.basis_m), np.asarray(self.origin_m)
        if (basis.shape != (2, 2) or origin.shape != (2,)
                or not np.all(np.isfinite(basis)) or not np.all(np.isfinite(origin))
                or abs(np.linalg.det(basis)) == 0):
            raise ValueError("Plane wave needs a finite non-singular SI lattice and origin")
        for value, shape in ((self.curvature_m1, (2, 2)), (self.tilt_rad, (2,))):
            if value is not None and (np.shape(value) != shape or not np.all(np.isfinite(value))):
                raise ValueError("Plane-wave phase carriers have invalid shape or values")
        if self.curvature_m1 is not None:
            q = np.asarray(self.curvature_m1)
            # Symmetry is a matrix-norm condition. Entrywise relative tests
            # incorrectly reject near-zero off-diagonals after a rotated LCT
            # (e.g. 3e-10 roundoff beside a 3e3 m^-1 diagonal). Preserve the
            # values; this admits rounding error, not a repaired map.
            if np.linalg.norm(q-q.T, ord=2) > 1e-10*max(1., np.linalg.norm(q, ord=2)):
                raise ValueError("A scalar quadratic phase requires symmetric curvature")
        for name in ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad"):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value, dtype=complex if name == "amplitude" else float)
                frozen = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
                object.__setattr__(self, name, frozen)

    def full_amplitude(self, wavelength_m):
        """Full cell amplitude, with phase-carrier sampling checked explicitly."""
        from temsim.physics.canonical_phase import expanded_phase_amplitude
        return expanded_phase_amplitude(self, wavelength_m)

    def coordinates_m(self):
        ny, nx = self.amplitude.shape
        yy, xx = np.meshgrid(np.arange(ny) - ny // 2, np.arange(nx) - nx // 2, indexing="ij")
        return self.origin_m[:, None, None] + np.einsum("ij,jyx->iyx", self.basis_m, np.stack((xx, yy)))

    @property
    def probability(self):
        return float(np.sum(np.abs(self.amplitude) ** 2))

    def canonical_covariance(self, wavelength_m):
        """Symmetrised (x,y,p_x/p0,p_y/p0) covariance of the full field.

        Spectral envelope derivatives retain the analytic phase carrier;
        sampling it into wrapped phase is unnecessary for this observable.
        """
        norm = self.probability
        if norm <= 0:
            raise ValueError("An empty wave has no phase-space covariance")
        ny, nx = self.amplitude.shape
        fy, fx = np.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nx), indexing="ij")
        frequency = np.einsum("ij,jyx->iyx", np.linalg.inv(self.basis_m).T, np.stack((fx, fy)))
        xy = self.coordinates_m() - self.origin_m[:, None, None]
        curvature = np.zeros((2, 2)) if self.curvature_m1 is None else self.curvature_m1
        tilt = np.zeros(2) if self.tilt_rad is None else self.tilt_rad
        momentum = (wavelength_m*np.fft.ifft2(frequency*np.fft.fft2(self.amplitude), axes=(-2, -1))
                    + (np.einsum("ij,jyx->iyx", curvature, xy)+tilt[:, None, None])*self.amplitude)
        vectors = np.concatenate((xy*self.amplitude, momentum)).reshape(4, -1)
        mean = np.real(vectors@self.amplitude.conj().ravel())/norm
        return np.real(vectors.conj()@vectors.T)/norm - np.outer(mean, mean)


def propagate_plane_wave(wave, matrix, translation, wavelength_m, *,
                         affine_action_m=0., reference_phase_rad=None,
                         reference_length_m=1.):
    """Propagate the complete complex field, retaining affine/lift phases.

    A bare matrix uses its principal reference-Gaussian lift. A physical path
    must supply the continuous lift and scalar action from CanonicalPath;
    identical endpoint ray matrices alone do not identify the wave operator.
    """
    from temsim.physics.wave_flux import check_lossless_norm
    from temsim.physics.canonical_phase import validate_canonical_map
    from temsim.physics.canonical_action import (
        affine_centre_action, principal_reference_phase, validate_reference_phase,
    )
    validate_canonical_map(matrix)
    if (np.shape(matrix) != (4, 4) or np.shape(translation) != (4,)
            or not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(translation))):
        raise ValueError("Canonical wave map and translation must be finite 4-D arrays")
    if (not np.all(np.isfinite(wave.amplitude)) or not np.all(np.isfinite(wave.basis_m))
            or abs(np.linalg.det(wave.basis_m)) == 0
            or not np.isfinite(wavelength_m) or wavelength_m <= 0):
        raise ValueError("Invalid wave, sampling lattice or wavelength")
    if reference_phase_rad is None:
        reference_phase_rad = principal_reference_phase(matrix, reference_length_m)
    validate_reference_phase(matrix, reference_length_m, reference_phase_rad)
    result, chart_phase = _propagate_plane_wave(wave, matrix, translation, wavelength_m,
                                               reference_length_m)
    tilt = np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
    action = affine_centre_action(matrix, translation, wave.origin_m, tilt, affine_action_m)
    phase = 2*np.pi*action/wavelength_m + reference_phase_rad-chart_phase
    result = replace(result, amplitude=result.amplitude*np.exp(1j*phase))
    check_lossless_norm(wave.probability, result.probability, context="Canonical plane propagation")
    return result


def _propagate_plane_wave(wave, matrix, translation, wavelength_m, reference_length_m):
    """Apply one canonical affine map, retaining complex phase and flux."""
    from temsim.physics.canonical_action import drift_gaussian_phase
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
        conservative_drift = np.linalg.norm(drift, ord=2) * theta < .25 * extent
        amplitude = _sampled_angular_spectrum(wave, drift, wavelength_m, conservative=conservative_drift)
        if amplitude is not None:
            # M L(Q) = L(Q_out) S(A_eff) D(A_eff^-1 B). This
            # scaled angular-spectrum form resolves successive far-field
            # drifts without forcing another undersampled quadratic FFT.
            chart_phase = drift_gaussian_phase(drift, 1j*np.eye(2)/reference_length_m-curvature)
            return (PlaneWave(amplitude, effective_a @ wave.basis_m, a @ wave.origin_m + b @ tilt + shift[:2],
                              (c + d @ curvature) @ inverse_a, c @ wave.origin_m + d @ tilt + shift[2:]),
                    chart_phase)
    if np.linalg.norm(b, ord=2) * theta < 1e-7 * max(np.linalg.norm(a, ord=2) * extent, 1e-30):
        if abs(np.linalg.det(a)) < 1e-15:
            raise ValueError("Singular image-plane wave map")
        inverse_a = np.linalg.inv(a)
        return (PlaneWave(wave.amplitude, a @ wave.basis_m, a @ wave.origin_m + shift[:2],
                          inverse_a.T @ curvature @ inverse_a + c @ inverse_a,
                          c @ wave.origin_m + d @ tilt + shift[2:]), 0.)
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
    basis = wavelength_m * b @ np.linalg.inv(wave.basis_m).T @ np.diag((1 / nx, 1 / ny))
    chirped = wave.amplitude * np.exp(1j * phase)
    # A natural Fresnel FFT may put a narrow far-field beam into only one or
    # two output pixels. Evaluate the SAME Collins integral on a finer affine
    # output lattice with a chirp-z transform. Covariance chooses only the
    # numerical sampling; it never replaces the propagated complex field.
    if wave.probability > 0:
        output_covariance = matrix[:2]@wave.canonical_covariance(wavelength_m)@matrix[:2].T
        inverse_basis = np.linalg.inv(basis)
        index_covariance = inverse_basis@output_covariance@inverse_basis.T
        scale = min(1., max(18*math.sqrt(max(0., index_covariance[0, 0]))/nx,
                            18*math.sqrt(max(0., index_covariance[1, 1]))/ny, 1e-12))
    else:
        scale = 1.
    while True:
        if scale == 1.:
            amplitude = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(chirped), norm="ortho"))
        else:
            from scipy.signal import czt
            amplitude = chirped
            for axis, size in ((0, ny), (1, nx)):
                amplitude = czt(amplitude, m=size, w=np.exp(-2j*np.pi*scale/size),
                                a=np.exp(-2j*np.pi*scale*(size//2)/size), axis=axis)
                correction = np.exp(2j*np.pi*(size//2)*scale*(np.arange(size)-size//2)/size)
                amplitude *= correction[:, None] if axis == 0 else correction[None, :]
            amplitude *= scale/math.sqrt(nx*ny)
        norm = float(np.sum(abs(amplitude)**2))
        if math.isclose(norm, wave.probability, rel_tol=1e-10, abs_tol=1e-14) or scale == 1.:
            break
        scale = min(1., scale*1.5)
    basis = basis*scale
    # Keep the quadratic carrier analytic. Sampling it into wrapped phase and
    # later multiplying an opposite chirp would alias a perfectly valid field.
    # The unprefactored Fourier integral has this centre phase on a reference
    # Gaussian. Comparing analytic chart factors (not fitting the computed
    # wave) fixes the missing Collins prefactor and the physical path branch.
    precision = np.eye(2)/reference_length_m-1j*inverse@a
    chart_phase = -.5*float(np.sum(np.angle(np.linalg.eigvals(precision))))
    return (PlaneWave(amplitude, basis, a @ wave.origin_m + b @ tilt + shift[:2],
                      d @ inverse, c @ wave.origin_m + d @ tilt + shift[2:]), chart_phase)


def _sampled_angular_spectrum(wave, drift, wavelength_m, *, conservative):
    """Resolve a spectral drift without treating empty Nyquist bins as rays.

    Outside the existing conservative full-band domain, both the occupied
    spectral phase and the output boundary mass must pass. Failure selects
    Fresnel quadrature instead; it never changes or repairs the input field.
    """
    if np.all(drift == 0):
        return wave.amplitude
    ny, nx = wave.amplitude.shape
    fy, fx = np.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nx), indexing="ij")
    frequency = np.einsum("ij,jyx->iyx", np.linalg.inv(wave.basis_m).T, np.stack((fx, fy)))
    phase = -np.pi*wavelength_m*np.einsum("iyx,ij,jyx->yx", frequency, drift, frequency)
    spectrum = np.fft.fft2(wave.amplitude)
    if not conservative:
        occupied = np.fft.fftshift(abs(spectrum)**2 > np.max(abs(spectrum)**2)*1e-14)
        ordered_phase = np.fft.fftshift(phase)
        for axis in (0, 1):
            low = np.take(occupied, np.arange(occupied.shape[axis]-1), axis=axis)
            high = np.take(occupied, np.arange(1, occupied.shape[axis]), axis=axis)
            if np.any(abs(np.diff(ordered_phase, axis=axis))[low | high] >= np.pi):
                return None
    amplitude = np.fft.ifft2(spectrum*np.exp(1j*phase))
    if not conservative:
        probability = abs(amplitude)**2
        boundary = np.ones(amplitude.shape, bool)
        boundary[2:-2, 2:-2] = False
        if probability[boundary].sum() > max(1e-30, probability.sum()*1e-12):
            return None
    return amplitude


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


def intermediate_apertures(state, stop_z_mm, *, excluded_keys=()):
    from temsim.physics.record_plane import _aperture_stop
    planes = tuple(sorted((_aperture_stop(state, aperture) for aperture in getattr(state, "apertures", ())
                         if bool(getattr(aperture, "enabled", True))
                         and bool(getattr(aperture, "installed", True))
                         and bool(getattr(aperture, "inserted", True))
                         and str(aperture.key) not in excluded_keys
                         and float(state.sample.z_mm) < float(aperture.z_mm) < stop_z_mm), key=lambda item: item.z_mm))
    if len({plane.key for plane in planes}) != len(planes):
        raise ValueError("Duplicate physical aperture identity in propagation branch")
    return planes


def project_through_apertures(state, wave, x_m, y_m, wavelength_m, target_z_mm, *, excluded_keys=()):
    dx, dy = float(x_m[1] - x_m[0]), float(y_m[1] - y_m[0])
    current = PlaneWave(np.asarray(wave) * np.sqrt(dx * dy), np.diag((dx, dy)),
                        np.array((x_m[len(x_m)//2], y_m[len(y_m)//2])))
    previous, previous_offset = np.eye(4), np.zeros(4)
    rows = []
    for plane in (*intermediate_apertures(state, target_z_mm, excluded_keys=excluded_keys), SimpleNamespace(z_mm=target_z_mm)):
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
            rows.append({"key": plane.key, "physical_element_id": "aperture:" + plane.key,
                         "strategy": "physical_plane", "z_mm": float(plane.z_mm), "incoming_probability": before,
                         "radius_mm": plane.radius_mm, "offset_x_mm": plane.offset_x_mm,
                         "offset_y_mm": plane.offset_y_mm, "transfer_matrix": matrix.tolist(),
                         "transmitted_probability": current.probability,
                         "aperture_diameter_pixels": 2*plane.radius_mm*1e-3 / np.linalg.norm(current.basis_m, ord=2)})
        previous, previous_offset = matrix, offset
    return current, tuple(rows)
