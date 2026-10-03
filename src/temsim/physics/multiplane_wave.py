"""Coherent sampled LCTs and aperture masks on physical intermediate planes.

An affine two-dimensional lattice retains rotation and shear without resampling
the wave onto a fictitious axis-aligned grid. Amplitudes are probability per
pixel. FFTs are unitary; clipping never renormalises transmitted electrons.
"""

from dataclasses import dataclass, replace
from types import SimpleNamespace
import math
import numpy as np


class _AngularSpectrumDomainUnavailable(ValueError):
    """Only an optional domain chart's disabled/exceeded budget, not physics."""

    def __init__(self, message, diagnostic):
        super().__init__(message)
        self.diagnostic = dict(diagnostic)


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


def reselect_phase_carrier(wave, wavelength_m, *, grid_numerics,
                          retained_bytes=0, cancelled=lambda: False):
    """Exactly change phase gauge, retaining every non-Gaussian wave sample.

    The symmetric carrier minimizes centred residual momentum variance. This
    chooses a numerical representation only: amplitudes are never replaced by
    a Gaussian, fitted, filtered or renormalised. The compensating envelope
    chirp is resolved BEFORE applying it, with the usual refinement budget.
    """
    from scipy.linalg import solve_sylvester
    from temsim.physics.wave_grid import WaveSamplingError, WaveGridBudgetError, apply_resolved_operator
    from temsim.physics.wave_flux import check_lossless_norm
    if cancelled():
        raise InterruptedError("Phase-carrier selection cancelled")
    grid_numerics.check(wave.amplitude.shape, retained_bytes=retained_bytes)
    if not np.isfinite(wavelength_m) or wavelength_m <= 0:
        raise ValueError("Phase carrier needs a positive finite wavelength")
    if wave.probability == 0:
        return wave, []
    boundary = _input_boundary_probability(wave.amplitude)
    if boundary > max(1e-30, wave.probability*1e-12):
        # This gauge is optional numerical preconditioning, not a physical
        # operation. Its Fourier-derivative estimate needs a resolved domain.
        # Keep the entire original field and its existing chart when that
        # prerequisite is absent; physical propagation still runs normally.
        return wave, [{"method": "phase-carrier optimization not applied: unresolved input boundary",
            "from_shape": wave.amplitude.shape, "to_shape": wave.amplitude.shape,
            "input_boundary_probability": boundary, "probability": wave.probability}]
    covariance = wave.canonical_covariance(wavelength_m)
    xx, xp = covariance[:2, :2], covariance[:2, 2:]
    if np.linalg.eigvalsh(xx).min() <= 0:
        raise ValueError("Phase-carrier selection needs a positive spatial covariance")
    selected = solve_sylvester(xx, xx, xp+xp.T)
    selected = (selected+selected.T)/2
    old = np.zeros((2, 2)) if wave.curvature_m1 is None else wave.curvature_m1
    delta = selected-old
    if np.linalg.norm(delta) <= 1e-10*max(1., np.linalg.norm(old)):
        return wave, []
    selected_rows = []
    def rephase(current):
        if cancelled():
            raise InterruptedError("Phase-carrier selection cancelled")
        local = current.coordinates_m()-current.origin_m[:, None, None]
        compensation = -np.pi/wavelength_m*np.einsum("iyx,ij,jyx->yx", local, delta, local)
        occupied = abs(current.amplitude) > np.max(abs(current.amplitude))*1e-8
        spectrum = abs(np.fft.fft2(current.amplitude, norm="ortho"))**2
        total, largest = float(spectrum.sum()), 0.
        for axis in (0, 1):
            count = current.amplitude.shape[axis]
            low = np.take(occupied, np.arange(count-1), axis=axis)
            high = np.take(occupied, np.arange(1, count), axis=axis)
            added = float(abs(np.diff(compensation, axis=axis))[low & high].max(initial=0.))
            frequencies = abs(2*np.pi*np.fft.fftfreq(count))
            marginal = spectrum.sum(axis=1-axis)
            order = np.argsort(frequencies)
            tails = np.cumsum(marginal[order][::-1])[::-1]
            support = float(frequencies[order][tails > total*1e-12].max(initial=0.))
            largest = max(largest, support+added)
        if largest >= .8*np.pi:
            raise WaveSamplingError(f"Phase-carrier envelope plus compensating chirp is undersampled "
                f"({largest:.6g} rad per cell); refine BEFORE exact gauge rephasing", largest/(.8*np.pi))
        result = replace(current, amplitude=current.amplitude*np.exp(1j*compensation), curvature_m1=selected)
        check_lossless_norm(current.probability, result.probability, context="Exact phase-carrier gauge")
        selected_rows.append({"method": "exact envelope chirp compensation; full complex field retained",
            "from_curvature_m1": old.tolist(), "to_curvature_m1": selected.tolist(),
            "from_shape": current.amplitude.shape, "to_shape": result.amplitude.shape,
            "envelope_chirp_bandwidth_rad": largest, "shape": result.amplitude.shape,
            "probability": result.probability})
        return result
    try:
        result, rows = apply_resolved_operator(wave, rephase, numerics=grid_numerics,
            retained_bytes=retained_bytes, cancelled=cancelled)
    except WaveGridBudgetError as error:
        # This is only an optional change of representation. Keep the entire
        # ORIGINAL field when its compensating chirp cannot fit the budget;
        # the physical propagator must still pass its own sampling checks.
        return wave, [{"method": "phase-carrier optimization not applied: refinement budget",
            "from_shape": wave.amplitude.shape, "to_shape": wave.amplitude.shape,
            "reason": str(error), "probability": wave.probability}]
    rows.extend(selected_rows)
    return result, rows


def propagate_plane_wave(wave, matrix, translation, wavelength_m, *,
                         affine_action_m=0., reference_phase_rad=None,
                         reference_length_m=1., grid_numerics=None,
                         retained_bytes=0, cancelled=lambda: False,
                         sampling_records=None):
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
    if cancelled():
        raise InterruptedError("Canonical plane propagation cancelled")
    if grid_numerics is not None:
        grid_numerics.check(wave.amplitude.shape, retained_bytes=retained_bytes)
    completed_records = [] if sampling_records is not None else None
    result, chart_phase = _propagate_plane_wave(wave, matrix, translation, wavelength_m,
        reference_length_m, grid_numerics=grid_numerics, retained_bytes=retained_bytes,
        cancelled=cancelled, sampling_records=completed_records)
    tilt = np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
    action = affine_centre_action(matrix, translation, wave.origin_m, tilt, affine_action_m)
    phase = 2*np.pi*action/wavelength_m + reference_phase_rad-chart_phase
    result = replace(result, amplitude=result.amplitude*np.exp(1j*phase))
    check_lossless_norm(wave.probability, result.probability, context="Canonical plane propagation")
    if sampling_records is not None:
        sampling_records.extend(completed_records)
    return result


def _propagate_plane_wave(wave, matrix, translation, wavelength_m, reference_length_m, *,
                         grid_numerics=None, retained_bytes=0,
                         cancelled=lambda: False, sampling_records=None):
    """Apply one canonical affine map, retaining complex phase and flux."""
    from temsim.physics.canonical_action import drift_gaussian_phase
    matrix, shift = np.asarray(matrix, float), np.asarray(translation, float)
    a, b, c, d = matrix[:2, :2], matrix[:2, 2:], matrix[2:, :2], matrix[2:, 2:]
    curvature = np.zeros((2, 2)) if wave.curvature_m1 is None else wave.curvature_m1
    tilt = np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
    ny, nx = wave.amplitude.shape
    # A truly image-conjugate map has B=0. Merely small detector blur is not
    # sufficient to discard propagation phase at an intermediate aperture.
    theta = wavelength_m * np.linalg.norm(np.linalg.inv(wave.basis_m), ord=2)
    extent = np.linalg.norm(wave.basis_m, ord=2) * min(nx, ny)
    effective_a = a + b @ curvature
    unavailable_domain = None
    if np.linalg.cond(effective_a) < 1e8:
        inverse_a = np.linalg.inv(effective_a)
        drift = inverse_a @ b
        conservative_drift = np.linalg.norm(drift, ord=2) * theta < .25 * extent
        diagnostic = {}
        amplitude = _sampled_angular_spectrum(wave, drift, wavelength_m,
            conservative=conservative_drift, diagnostic=diagnostic)
        if amplitude is None and diagnostic.get("failure") == "output_boundary" and grid_numerics is not None:
            try:
                wave, amplitude = _extend_angular_spectrum_domain(wave, drift, wavelength_m,
                    diagnostic, numerics=grid_numerics, retained_bytes=retained_bytes,
                    cancelled=cancelled, sampling_records=sampling_records)
            except _AngularSpectrumDomainUnavailable as error:
                # This is a numerical chart choice. Keep the unmodified input
                # and try the original Collins operator within the SAME budget.
                # Cancellation, memory exhaustion, invalid settings and an
                # unresolved physical operator are deliberately not caught.
                unavailable_domain = error
        if amplitude is not None:
            # M L(Q) = L(Q_out) S(A_eff) D(A_eff^-1 B). This
            # scaled angular-spectrum form resolves successive far-field
            # drifts without forcing another undersampled quadratic FFT.
            chart_phase = drift_gaussian_phase(drift, 1j*np.eye(2)/reference_length_m-curvature)
            return (PlaneWave(amplitude, effective_a @ wave.basis_m, a @ wave.origin_m + b @ tilt + shift[:2],
                              (c + d @ curvature) @ inverse_a, c @ wave.origin_m + d @ tilt + shift[2:]),
                    chart_phase)
    try:
        result, chart_phase = _propagate_fresnel_chart(wave, matrix, shift, wavelength_m,
            reference_length_m, grid_numerics=grid_numerics, retained_bytes=retained_bytes,
            cancelled=cancelled, sampling_records=sampling_records)
    except ValueError as error:
        if unavailable_domain is None:
            raise
        from temsim.physics.wave_grid import WaveSamplingError
        message = f"{unavailable_domain}; Collins alternate failed: {error}"
        if isinstance(error, WaveSamplingError):
            raise WaveSamplingError(message, required_scale=error.required_scale) from error
        raise ValueError(message) from error
    if unavailable_domain is not None and sampling_records is not None:
        sampling_records.append({"method": "resolved Collins alternate after optional angular-spectrum domain unavailable",
            "reason": str(unavailable_domain), "from_shape": wave.amplitude.shape,
            "to_shape": result.amplitude.shape, "angular_spectrum": unavailable_domain.diagnostic,
            "probability": result.probability})
    return result, chart_phase


def _propagate_fresnel_chart(wave, matrix, shift, wavelength_m, reference_length_m, *,
                            grid_numerics=None, retained_bytes=0,
                            cancelled=lambda: False, sampling_records=None):
    """Original image/Collins chart, retaining its sampling and budget gates."""
    a, b, c, d = matrix[:2, :2], matrix[:2, 2:], matrix[2:, :2], matrix[2:, 2:]
    local_input = wave.coordinates_m()-wave.origin_m[:, None, None]
    curvature = np.zeros((2, 2)) if wave.curvature_m1 is None else wave.curvature_m1
    tilt = np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
    ny, nx = wave.amplitude.shape
    theta = wavelength_m*np.linalg.norm(np.linalg.inv(wave.basis_m), ord=2)
    extent = np.linalg.norm(wave.basis_m, ord=2)*min(nx, ny)
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
    largest_increment = 0.
    for axis in (0, 1):
        adjacent = np.take(occupied, range(occupied.shape[axis] - 1), axis=axis) & np.take(occupied, range(1, occupied.shape[axis]), axis=axis)
        largest_increment = max(largest_increment,
            float(np.abs(np.diff(phase, axis=axis))[adjacent].max(initial=0.)))
    if largest_increment > np.pi:
        from temsim.physics.wave_grid import WaveSamplingError
        raise WaveSamplingError(
            f"Intermediate-plane phase is undersampled ({largest_increment:.6g} rad per cell); refine the wave grid",
            largest_increment/(.8*np.pi))
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
        rms_pixels = np.sqrt(np.maximum(0., np.diag(index_covariance)))
        scale = min(1., max(18*rms_pixels[0]/nx, 18*rms_pixels[1]/ny, 1e-12))
    else:
        scale = 1.
        rms_pixels = np.full(2, np.inf)
    failed_czt_norm = None
    while True:
        if cancelled():
            raise InterruptedError("Collins plane propagation cancelled")
        if scale == 1.:
            # A cropped CZT can miss tiny, genuine tails or discretisation
            # sidelobes. Falling back to a full-period FFT preserves them,
            # but its natural output cells may be too coarse near a focus.
            # Zero extension of the SAME spatial quadrature samples increases
            # output resolution without changing the output frequency range.
            # This is deliberately distinct from refining the input Fourier
            # interpolant: that operation does not improve this output chart.
            factor = 1
            if grid_numerics is not None and min(rms_pixels) < 6.:
                if not grid_numerics.automatic_refinement:
                    raise ValueError("Collins output is undersampled; automatic output refinement is disabled")
                if min(rms_pixels) <= 0 or not np.all(np.isfinite(rms_pixels)):
                    raise ValueError("Collins output requires an unbounded output refinement")
                factor = 2**max(1, math.ceil(math.log2(6./min(rms_pixels))))
                boundary = _input_boundary_probability(wave.amplitude)
                if boundary > max(1e-30, wave.probability*1e-12):
                    raise ValueError("Collins output refinement needs a resolved input domain: "
                        f"boundary probability={boundary:.6g}; the input field was not zero-extended")
                shape = (ny*factor, nx*factor)
                try:
                    required = grid_numerics.check(shape,
                        retained_bytes=retained_bytes+wave.amplitude.nbytes+chirped.nbytes)
                except ValueError as error:
                    # This is NOT a WaveSamplingError: retrying an input-grid
                    # refinement would leave the natural output cell unchanged.
                    raise ValueError(f"Collins output refinement budget exceeded: {error}") from error
                if cancelled():
                    raise InterruptedError("Collins output refinement cancelled")
                padded = np.zeros(shape, dtype=np.complex128)
                first_y, first_x = shape[0]//2-ny//2, shape[1]//2-nx//2
                padded[first_y:first_y+ny, first_x:first_x+nx] = chirped
                amplitude = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded), norm="ortho"))
                basis = basis/factor
                if sampling_records is not None:
                    sampling_records.append({"from_shape": (ny, nx), "to_shape": shape,
                        "reason": "natural Collins output RMS below six cells; full output range retained",
                        "natural_output_rms_pixels_xy": rms_pixels.tolist(),
                        "czt_last_probability": failed_czt_norm,
                        "input_boundary_probability": boundary,
                        "estimated_working_bytes": required,
                        "probability": float(np.sum(abs(amplitude)**2)),
                        "method": "zero-padded spatial Collins quadrature; complete output frequency period"})
            else:
                amplitude = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(chirped), norm="ortho"))
        else:
            from scipy.signal import zoom_fft
            amplitude = chirped
            for axis, size in ((0, ny), (1, nx)):
                # Unit-circle chirps must be evaluated as exp(i*phase), not
                # powers of a rounded complex w: w**(k*k/2) amplifies its
                # tiny radial error and spuriously changes electron flux.
                # ZoomFFT evaluates exactly these SAME frequency points.
                frequencies = (-scale*(size//2)/size, scale*(size-size//2)/size)
                amplitude = zoom_fft(amplitude, frequencies, m=size, fs=1., endpoint=False, axis=axis)
                correction = np.exp(2j*np.pi*(size//2)*scale*(np.arange(size)-size//2)/size)
                amplitude *= correction[:, None] if axis == 0 else correction[None, :]
            amplitude *= scale/math.sqrt(nx*ny)
        norm = float(np.sum(abs(amplitude)**2))
        if math.isclose(norm, wave.probability, rel_tol=1e-10, abs_tol=1e-14) or scale == 1.:
            break
        failed_czt_norm = norm
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


def _input_boundary_probability(amplitude):
    """Probability in all four two-cell edge strips, counted exactly once."""
    ny, nx = amplitude.shape
    if min(ny, nx) <= 4:
        return float(np.sum(abs(amplitude)**2))
    return float(np.sum(abs(amplitude[:2])**2)+np.sum(abs(amplitude[-2:])**2)
        +np.sum(abs(amplitude[2:-2, :2])**2)+np.sum(abs(amplitude[2:-2, -2:])**2))


def _extend_angular_spectrum_domain(wave, drift, wavelength_m, diagnostic, *, numerics,
                                   retained_bytes=0, cancelled=lambda: False, sampling_records=None):
    """Keep all executed finite quadrature cells while expanding their domain.

    This is spatial zero extension at the SAME spacing and phase carriers,
    not Fourier interpolation, a crop, a new source or a claim that unknown
    infinite-domain tails vanish. Its admitted output retains the entire
    expanded field. Material/domain convergence remains a separate check.
    """
    from temsim.physics.wave_grid import WaveGridBudgetError
    if cancelled():
        raise InterruptedError("Angular-spectrum domain extension cancelled")
    initial_diagnostic = dict(diagnostic)
    if not numerics.automatic_refinement:
        raise _AngularSpectrumDomainUnavailable(
            "Angular-spectrum physical-domain extension is required; automatic refinement is disabled", initial_diagnostic)
    initial = wave
    ny, nx = initial.amplitude.shape
    original_boundary = _input_boundary_probability(initial.amplitude)
    inverse_basis = np.linalg.inv(initial.basis_m)
    # Maximum group displacement of EVERY represented Fourier bin, in
    # lattice-index x/y units. Padding leaves a larger margin than this
    # displacement around all retained input cells, not just the bright core.
    group_cells = .5*wavelength_m*np.sum(abs(inverse_basis@drift@inverse_basis.T), axis=1)
    factor = 2
    while True:
        if cancelled():
            raise InterruptedError("Angular-spectrum domain extension cancelled")
        shape = (ny*factor, nx*factor)
        try:
            required = numerics.check(shape, retained_bytes=retained_bytes+initial.amplitude.nbytes+wave.amplitude.nbytes)
        except WaveGridBudgetError as error:
            # Fourier input refinement keeps the physical window and cannot
            # resolve this failure. Do not expose it as a sampling retry.
            raise _AngularSpectrumDomainUnavailable(
                f"Angular-spectrum domain extension budget exceeded: {error}",
                {**initial_diagnostic, "requested_shape": shape,
                 "input_boundary_probability": original_boundary}) from error
        margin = np.array(((shape[1]-nx)/2, (shape[0]-ny)/2))
        if np.any(group_cells >= margin):
            factor *= 2
            continue
        padded = np.zeros(shape, dtype=np.complex128)
        start_y, start_x = shape[0]//2-ny//2, shape[1]//2-nx//2
        padded[start_y:start_y+ny, start_x:start_x+nx] = initial.amplitude
        wave = replace(initial, amplitude=padded)
        if cancelled():
            raise InterruptedError("Angular-spectrum domain extension cancelled")
        attempt = {}
        amplitude = _sampled_angular_spectrum(wave, drift, wavelength_m,
            conservative=False, full_band=True, diagnostic=attempt)
        if amplitude is None and attempt.get("failure") != "output_boundary":
            raise ValueError("Angular-spectrum expanded finite domain has unresolved spectral propagation; the optical operator was not applied")
        if amplitude is not None:
            if sampling_records is not None:
                sampling_records.append({"method": "spatial zero extension of every executed complex quadrature cell; full output domain retained",
                    "reason": "sampled spectral propagation exceeded the finite output boundary budget",
                    "from_shape": (ny, nx), "to_shape": shape,
                    "input_boundary_probability": original_boundary,
                    "previous_output_boundary_probability": diagnostic["output_boundary_probability"],
                    "output_boundary_probability": attempt["output_boundary_probability"],
                    "spectral_phase_increment_rad": attempt["spectral_phase_increment_rad"],
                    "full_spectral_band_checked": True,
                    "maximum_group_displacement_cells_xy": group_cells.tolist(),
                    "domain_margin_cells_xy": margin.tolist(),
                    "input_period_vectors_m": (initial.basis_m@np.diag((nx, ny))).tolist(),
                    "output_period_vectors_m": (wave.basis_m@np.diag((shape[1], shape[0]))).tolist(),
                    "basis_m": wave.basis_m.tolist(), "estimated_working_bytes": required,
                    "probability": float(np.sum(abs(amplitude)**2)),
                    "continuum_tail_scope": "All executed finite cells retained; unknown exterior continuum tails are not established"})
            return wave, amplitude
        diagnostic = attempt
        factor *= 2


def _sampled_angular_spectrum(wave, drift, wavelength_m, *, conservative, full_band=False, diagnostic=None):
    """Resolve a spectral drift without treating empty Nyquist bins as rays.

    Outside the existing conservative full-band domain, both the occupied
    spectral phase and the output boundary mass must pass. Failure selects
    Fresnel quadrature instead; it never changes or repairs the input field.
    """
    if diagnostic is None:
        diagnostic = {}
    if np.all(drift == 0):
        return wave.amplitude
    ny, nx = wave.amplitude.shape
    fy, fx = np.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nx), indexing="ij")
    frequency = np.einsum("ij,jyx->iyx", np.linalg.inv(wave.basis_m).T, np.stack((fx, fy)))
    phase = -np.pi*wavelength_m*np.einsum("iyx,ij,jyx->yx", frequency, drift, frequency)
    spectrum = np.fft.fft2(wave.amplitude)
    if not conservative:
        # Cumulative marginal probability bounds include collectively
        # significant weak tails, rather than discarding each small bin by
        # a threshold relative to the brightest spectral pixel. This is a
        # guard only: every complex frequency remains in the FFT operator.
        if full_band:
            occupied = np.ones(wave.amplitude.shape, bool)
        else:
            power, support = abs(spectrum)**2, []
            total = float(power.sum())
            for axis, count in ((0, ny), (1, nx)):
                frequency_axis = abs(2*np.pi*np.fft.fftfreq(count))
                marginal = power.sum(axis=1-axis)
                order = np.argsort(frequency_axis)
                tail = np.cumsum(marginal[order][::-1])[::-1]
                support.append(float(frequency_axis[order][tail > total*1e-12].max(initial=0.)))
            occupied = np.fft.fftshift((abs(2*np.pi*fy) <= support[0]) & (abs(2*np.pi*fx) <= support[1]))
        ordered_phase = np.fft.fftshift(phase)
        largest = 0.
        for axis in (0, 1):
            low = np.take(occupied, np.arange(occupied.shape[axis]-1), axis=axis)
            high = np.take(occupied, np.arange(1, occupied.shape[axis]), axis=axis)
            largest = max(largest, float(abs(np.diff(ordered_phase, axis=axis))[low | high].max(initial=0.)))
        diagnostic.update(spectral_phase_increment_rad=largest, spectral_tail_probability_per_axis=1e-12)
        if largest >= np.pi:
            diagnostic["failure"] = "spectral_sampling"
            return None
    amplitude = np.fft.ifft2(spectrum*np.exp(1j*phase))
    if not conservative:
        probability = float(np.sum(abs(amplitude)**2))
        boundary = _input_boundary_probability(amplitude)
        diagnostic.update(output_boundary_probability=boundary)
        if boundary > max(1e-30, probability*1e-12):
            diagnostic["failure"] = "output_boundary"
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
