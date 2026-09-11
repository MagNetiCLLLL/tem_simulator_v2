"""Full complex phase checks, isolated from any production illumination input."""
from dataclasses import replace

import numpy as np
import pytest

from temsim.physics.canonical_action import CanonicalPath, validate_reference_phase
from temsim.physics.multiplane_wave import PlaneWave, propagate_plane_wave


def gaussian():
    sigma, step, n = 5e-9, 5e-10, 512
    x = (np.arange(n)-n//2)*step
    amplitude = np.exp(-(x[:, None]**2+x[None, :]**2)/(4*sigma**2)).astype(complex)
    amplitude /= np.linalg.norm(amplitude)
    return PlaneWave(amplitude, np.eye(2)*step, np.array((2e-9, -3e-9)),
                     tilt_rad=np.array((2e-4, -1e-4)))


def drift(distance):
    m = np.eye(4)
    m[:2, 2:] = np.eye(2)*distance
    return m


@pytest.mark.parametrize("distance", [1e-6, 2e-4, 1e-3, -1e-3])
def test_tilted_off_axis_gaussian_drift_matches_complex_analytic_solution(distance):
    initial, lam, sigma = gaussian(), 2e-12, 5e-9
    actual = propagate_plane_wave(initial, drift(distance), np.zeros(4), lam)
    local = actual.coordinates_m()-actual.origin_m[:, None, None]
    precision = 1j*lam/(4*np.pi*sigma**2)
    factor = 1+distance*precision
    centre = initial.amplitude.shape[0]//2
    field = (initial.amplitude[centre, centre]/factor
             * np.exp(1j*np.pi/lam*(precision/factor)*np.sum(local**2, axis=0))
             * np.exp(2j*np.pi/lam*np.einsum("i,iyx->yx", initial.tilt_rad, local))
             * np.exp(1j*np.pi/lam*distance*np.dot(initial.tilt_rad, initial.tilt_rad)))
    field *= np.sqrt(abs(np.linalg.det(actual.basis_m)/np.linalg.det(initial.basis_m)))
    # Complex residual, with no fitted phase/normalisation. Analytic Gaussian
    # The 512-cell window avoids the 256-cell propagated boundary tail and
    # resolves the far-field tilt carrier. No physical source value or error
    # tolerance is changed to admit an undersampled field.
    assert np.linalg.norm(actual.full_amplitude(lam)-field) < 2e-8


def test_off_axis_lens_and_deflector_keep_the_physical_scalar_phase():
    w, lam = gaussian(), 2e-12
    lens = np.eye(4)
    lens[2:, :2] = np.diag((-500., -200.))
    kick = np.array((0., 0., 1e-4, -2e-4))
    result = propagate_plane_wave(w, lens, kick, lam)
    xy = w.coordinates_m()
    action = .5*np.einsum("iyx,ij,jyx->yx", xy, lens[2:, :2], xy)
    action += np.einsum("i,iyx->yx", kick[2:], xy)
    expected = w.full_amplitude(lam)*np.exp(2j*np.pi*action/lam)
    np.testing.assert_allclose(result.full_amplitude(lam), expected, atol=1e-13, rtol=1e-11)


def test_affine_path_split_whole_inverse_preserve_complex_field_and_weight():
    initial, lam = gaussian(), 2e-12
    lens = np.eye(4)
    lens[2:, :2] = -np.eye(2)*500
    first_m, second_m = lens@drift(1e-6), drift(2e-6)
    first_d, second_d = np.array((0., 0., 1e-4, -2e-4)), np.array((1e-10, 0., -1e-4, 0.))
    first = propagate_plane_wave(initial, first_m, first_d, lam)
    split = propagate_plane_wave(first, second_m, second_d, lam)
    path = CanonicalPath(1.)
    path.append(first_m, first_d)
    path.append(second_m, second_d)
    assert path.action_m != 0
    whole = propagate_plane_wave(initial, path.matrix, path.offset, lam, **path.phase_kwargs())
    np.testing.assert_allclose(split.full_amplitude(lam), whole.full_amplitude(lam), rtol=1e-9, atol=1e-10)
    inverse = CanonicalPath(1.)
    for m, d in ((second_m, second_d), (first_m, first_d)):
        inv = np.linalg.inv(m)
        inverse.append(inv, -inv@d)
    back = propagate_plane_wave(whole, inverse.matrix, inverse.offset, lam, **inverse.phase_kwargs())
    np.testing.assert_allclose(back.full_amplitude(lam), initial.full_amplitude(lam), rtol=1e-9, atol=1e-10)
    assert back.probability == pytest.approx(initial.probability, abs=1e-12)


def oscillator(angle, length):
    """One transverse harmonic oscillator, the other coordinate unchanged."""
    m = np.eye(4)
    m[0, 0] = m[2, 2] = np.cos(angle)
    m[0, 2], m[2, 0] = length*np.sin(angle), -np.sin(angle)/length
    return m


@pytest.mark.parametrize("turns", [1, 2, -1])
def test_endpoint_identity_does_not_erase_metaplectic_winding(turns):
    initial, lam, length = gaussian(), 2e-12, 1e-4
    path = CanonicalPath(length)
    step = oscillator(turns*2*np.pi/256, length)
    for _ in range(256):
        path.append(step)
    np.testing.assert_allclose(path.matrix, np.eye(4), rtol=0, atol=2e-9)
    assert path.reference_phase_rad == pytest.approx(-turns*np.pi, abs=1e-12)
    out = propagate_plane_wave(initial, path.matrix, path.offset, lam, **path.phase_kwargs())
    np.testing.assert_allclose(out.full_amplitude(lam), (-1.)**turns*initial.full_amplitude(lam),
                               rtol=1e-9, atol=1e-10)


def test_phase_path_rejects_unresolved_or_inconsistent_phase():
    path = CanonicalPath(1e-4)
    with pytest.raises(ValueError, match="undersampled"):
        path.append(oscillator(2., 1e-4))
    np.testing.assert_array_equal(path.matrix, np.eye(4))
    with pytest.raises(ValueError, match="inconsistent"):
        validate_reference_phase(np.eye(4), 1., .1)
    with pytest.raises(ValueError, match="finite"):
        path.append(np.eye(4), action_m=float("nan"))


def test_rejected_lift_validation_is_atomic(monkeypatch):
    from temsim.physics import canonical_action as action
    path = CanonicalPath(1.)
    path.append(drift(1e-6), np.array((0., 0., 1e-4, 0.)))
    before = (path.matrix.copy(), path.offset.copy(), path.action_m, path.reference_phase_rad)
    def reject(*args):
        raise ValueError("inconsistent endpoint fixture")
    monkeypatch.setattr(action, "validate_reference_phase", reject)
    with pytest.raises(ValueError, match="inconsistent endpoint"):
        path.append(drift(2e-6), np.array((0., 0., 0., 2e-4)))
    np.testing.assert_array_equal(path.matrix, before[0])
    np.testing.assert_array_equal(path.offset, before[1])
    assert path.action_m == before[2] and path.reference_phase_rad == before[3]


def test_non_gaussian_carried_wave_matches_direct_fft_without_phase_fitting():
    initial, lam, distance = gaussian(), 2e-12, 1e-5
    xy = initial.coordinates_m()-initial.origin_m[:, None, None]
    shaped = initial.amplitude*(1+.2*xy[0]/5e-9+.1j*xy[1]/5e-9)
    initial = replace(initial, amplitude=shaped)
    actual = propagate_plane_wave(initial, drift(distance), np.zeros(4), lam)
    fy, fx = np.meshgrid(np.fft.fftfreq(shaped.shape[0], d=5e-10),
                         np.fft.fftfreq(shaped.shape[1], d=5e-10), indexing="ij")
    # Direct full-field FFT on the original lattice; evaluate its Fourier
    # series at the translated output lattice. No envelope reconstruction.
    spectrum = np.fft.fft2(initial.full_amplitude(lam))
    free_phase = np.exp(-1j*np.pi*lam*distance*(fx*fx+fy*fy))
    delta = actual.origin_m-initial.origin_m
    translation = np.exp(2j*np.pi*(fx*delta[0]+fy*delta[1]))
    expected = np.fft.ifft2(spectrum*free_phase*translation)
    np.testing.assert_allclose(actual.full_amplitude(lam), expected, rtol=1e-10, atol=1e-12)


def test_auxiliary_reference_length_does_not_change_the_physical_wave():
    initial, lam = gaussian(), 2e-12
    results = []
    for length in (1e-4, 1e-3, 1.):
        path = CanonicalPath(length)
        for _ in range(256):
            path.append(oscillator(2*np.pi/256, 1e-3))
        results.append(propagate_plane_wave(initial, path.matrix, path.offset, lam,
                                           **path.phase_kwargs()).full_amplitude(lam))
    for result in results[1:]:
        np.testing.assert_allclose(result, results[0], rtol=1e-9, atol=1e-11)
