"""Full complex phase checks, isolated from any production illumination input."""
from dataclasses import replace

import numpy as np
import pytest

from temsim.physics.canonical_action import CanonicalPath, validate_reference_phase
from temsim.physics.multiplane_wave import PlaneWave, propagate_plane_wave, reselect_phase_carrier, _sampled_angular_spectrum
from temsim.physics.wave_grid import WaveGridNumerics, WaveSamplingError, WaveGridBudgetError, refine_plane_wave


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


def _collins_tail_fixture(*, angle=0., sigma_cells=3.5):
    """Analytic Gaussian plus a small coherent near-Nyquist angular tail.

    The tail is physical represented input. A narrow cropped CZT omits part
    of it even though covariance is dominated by the central Gaussian.
    """
    n, step, wavelength, distance = 64, 1e-9, 2e-12, .001
    axis = np.arange(n)-n//2
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    sigma, carrier, tail = sigma_cells*step, .47, 1e-4*np.exp(.3j)
    gaussian = np.exp(-(xx*xx+yy*yy)/(4*sigma_cells**2))
    amplitude = gaussian*(1+tail*np.exp(2j*np.pi*carrier*xx))
    amplitude /= np.linalg.norm(amplitude)
    rotation = np.array(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))
    wave = PlaneWave(amplitude, rotation*step, np.zeros(2))
    matrix = np.block([[np.zeros((2, 2)), np.eye(2)*distance],
                       [-np.eye(2)/distance, np.zeros((2, 2))]])
    return wave, matrix, wavelength, distance, sigma, carrier, tail, rotation


@pytest.mark.parametrize("angle", [0., .37])
def test_full_period_collins_output_refinement_preserves_gaussian_and_coherent_tail_phase(angle):
    wave, matrix, lam, distance, sigma, carrier, tail, rotation = _collins_tail_fixture(angle=angle)
    rows = []
    output = propagate_plane_wave(wave, matrix, np.zeros(4), lam,
        grid_numerics=WaveGridNumerics(maximum_pixels=512), sampling_records=rows)
    assert output.amplitude.shape == (512, 512)
    assert len(rows) == 1 and rows[0]["method"].startswith("zero-padded spatial Collins")
    assert rows[0]["czt_last_probability"] is not None
    assert rows[0]["input_boundary_probability"] < 1e-12*wave.probability
    natural_basis = lam*matrix[:2, 2:]@np.linalg.inv(wave.basis_m).T/64
    # The full physical Fourier period is unchanged, including the tail.
    np.testing.assert_allclose(output.basis_m*512, natural_basis*64, rtol=1e-14, atol=1e-20)
    assert output.probability == pytest.approx(wave.probability, rel=2e-14)
    xy = np.einsum("ij,jyx->iyx", rotation.T, output.coordinates_m())
    frequency = xy/(lam*distance)
    step = 1e-9
    # Poisson summation of the independent analytic Gaussian accounts for
    # EVERY represented spectral image, including the near-Nyquist tail that
    # wraps into the opposite edge of this complete output period. No fitted
    # phase, flux correction or normalization is applied to the output.
    expected = np.zeros(output.amplitude.shape, dtype=complex)
    for x_image in (-1, 0, 1):
        for y_image in (-1, 0, 1):
            fx, fy = frequency[0]-x_image/step, frequency[1]-y_image/step
            expected += np.exp(-4*np.pi**2*sigma*sigma*(fx*fx+fy*fy))
            expected += tail*np.exp(-4*np.pi**2*sigma*sigma*((fx-carrier/step)**2+fy*fy))
    input_coefficient = wave.amplitude[32, 32]/(1+tail)
    area_ratio = abs(np.linalg.det(output.basis_m)/np.linalg.det(wave.basis_m))
    expected *= -1j*input_coefficient*4*np.pi*sigma*sigma/(lam*distance)*np.sqrt(area_ratio)
    # The input is a finite 64-cell quadrature of this infinite Gaussian.
    # Parseval gives a rigorous complex-amplitude error bound from the known
    # omitted input cells; tiny probability tails are not identically zero.
    # This checks the analytic field within its actual finite-domain error,
    # while the following exact finite sum tests every represented cell.
    extended_axis = np.arange(-128, 129)
    ey, ex = np.meshgrid(extended_axis, extended_axis, indexing="ij")
    extended_input = (input_coefficient*np.exp(-(ex*ex+ey*ey)/(4*(sigma/step)**2))
                      *(1+tail*np.exp(2j*np.pi*carrier*ex)))
    outside = (ex < -32) | (ex >= 32) | (ey < -32) | (ey >= 32)
    omitted_norm = np.linalg.norm(extended_input[outside])
    assert omitted_norm < 1e-8  # independently bounded finite-window error
    assert np.linalg.norm(output.full_amplitude(lam)-expected) <= omitted_norm+1e-13
    # Independent finite Collins quadrature also verifies the complete
    # complex cell amplitude for non-Gaussian tails, not just its covariance.
    index = np.arange(512)-256
    input_index = np.arange(64)-32
    fourier = np.exp(-2j*np.pi*np.outer(index, input_index)/512)
    discrete = -1j*(fourier@wave.amplitude@fourier.T)/512
    np.testing.assert_allclose(output.full_amplitude(lam), discrete, rtol=1e-10, atol=1e-13)


@pytest.mark.parametrize("numerics, message", [
    (WaveGridNumerics(maximum_pixels=128), "Collins output refinement budget exceeded"),
    (WaveGridNumerics(maximum_pixels=512, maximum_working_bytes=2*1024**2), "Collins output refinement budget exceeded"),
    (WaveGridNumerics(automatic_refinement=False, maximum_pixels=512), "automatic output refinement is disabled"),
])
def test_collins_output_budget_failure_is_not_an_input_sampling_retry(numerics, message):
    wave, matrix, lam, *_ = _collins_tail_fixture()
    original = wave.amplitude.copy()
    records = []
    with pytest.raises(ValueError, match=message) as caught:
        propagate_plane_wave(wave, matrix, np.zeros(4), lam,
            grid_numerics=numerics, sampling_records=records)
    assert not isinstance(caught.value, WaveSamplingError)
    np.testing.assert_array_equal(wave.amplitude, original)
    assert records == []


def test_collins_zero_extension_rejects_input_boundary_mass_instead_of_truncating_it():
    wave, matrix, lam, *_ = _collins_tail_fixture(sigma_cells=12.)
    with pytest.raises(ValueError, match="resolved input domain"):
        propagate_plane_wave(wave, matrix, np.zeros(4), lam,
            grid_numerics=WaveGridNumerics(maximum_pixels=2048))


def test_collins_cancellation_is_checked_before_fft_or_output_allocation():
    wave, matrix, lam, *_ = _collins_tail_fixture()
    checks = []
    def cancel_during_czt():
        checks.append(True)
        return len(checks) >= 3
    rows = []
    with pytest.raises(InterruptedError, match="Collins plane propagation cancelled"):
        propagate_plane_wave(wave, matrix, np.zeros(4), lam,
            grid_numerics=WaveGridNumerics(maximum_pixels=512),
            cancelled=cancel_during_czt, sampling_records=rows)
    assert len(checks) == 3 and rows == []


def _sheared_non_gaussian_carrier():
    n, lam = 128, 2e-12
    rotation = np.array(((np.cos(.31), -np.sin(.31)), (np.sin(.31), np.cos(.31))))
    basis = rotation@np.array(((.5e-9, .1e-9), (0., .6e-9)))
    zero = PlaneWave(np.zeros((n, n), complex), basis, np.array((1e-9, -2e-9)))
    local = zero.coordinates_m()-zero.origin_m[:, None, None]
    yy, xx = np.meshgrid(np.arange(n)-n//2, np.arange(n)-n//2, indexing="ij")
    amplitude = np.exp(-(local[0]**2/(3e-9)**2+local[1]**2/(4e-9)**2)/4)
    amplitude = amplitude*(1+.17*local[0]/3e-9+.06j*local[1]/4e-9+1e-4*np.exp(2j*np.pi*.35*xx))
    envelope_q = np.array(((7500., -1300.), (-1300., 4500.)))
    amplitude = amplitude*np.exp(1j*np.pi/lam*np.einsum("iyx,ij,jyx->yx", local, envelope_q, local))
    amplitude /= np.linalg.norm(amplitude)
    return replace(zero, amplitude=amplitude, curvature_m1=np.array(((-2200., 300.), (300., 1800.))),
        tilt_rad=np.array((3e-5, -2e-5))), lam


def test_exact_carrier_gauge_preserves_off_axis_sheared_non_gaussian_full_field_and_inverse():
    initial, lam = _sheared_non_gaussian_carrier()
    original = initial.amplitude.copy()
    output, rows = reselect_phase_carrier(initial, lam, grid_numerics=WaveGridNumerics(maximum_pixels=1024))
    baseline = refine_plane_wave(initial, output.amplitude.shape, numerics=WaveGridNumerics(maximum_pixels=1024))
    assert rows and rows[-1]["method"].startswith("exact envelope chirp")
    np.testing.assert_array_equal(initial.amplitude, original)
    np.testing.assert_array_equal(output.origin_m, initial.origin_m)
    np.testing.assert_array_equal(output.tilt_rad, initial.tilt_rad)
    np.testing.assert_array_equal(output.basis_m, baseline.basis_m)
    np.testing.assert_allclose(output.full_amplitude(lam), baseline.full_amplitude(lam), rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(abs(output.amplitude), abs(baseline.amplitude), rtol=1e-13, atol=1e-15)
    assert output.probability == pytest.approx(initial.probability, rel=1e-13)
    local = output.coordinates_m()-output.origin_m[:, None, None]
    delta = output.curvature_m1-initial.curvature_m1
    inverse = output.amplitude*np.exp(1j*np.pi/lam*np.einsum("iyx,ij,jyx->yx", local, delta, local))
    np.testing.assert_allclose(inverse, baseline.amplitude, rtol=1e-11, atol=1e-13)
    # The independent optimality condition applies on a noncommuting/sheared
    # covariance; sym(PX X^-1) would not solve this equation in general.
    covariance = initial.canonical_covariance(lam)
    x, s = covariance[:2, :2], covariance[:2, 2:]
    np.testing.assert_allclose(output.curvature_m1@x+x@output.curvature_m1, s+s.T,
                              rtol=1e-10, atol=1e-24)


@pytest.mark.parametrize("automatic", [True, False])
def test_exact_carrier_gauge_budget_refusal_leaves_complete_input_unchanged(automatic):
    initial, lam = _sheared_non_gaussian_carrier()
    original = tuple(getattr(initial, name).copy() for name in
                     ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad"))
    numerics = WaveGridNumerics(automatic_refinement=automatic, maximum_pixels=128)
    if automatic:
        # A covariance-selected carrier is optional preconditioning. Refusing
        # its extra grid must leave the already represented physical field
        # available to the separately checked propagation operator.
        result, rows = reselect_phase_carrier(initial, lam, grid_numerics=numerics)
        assert result is initial
        assert len(rows) == 1
        assert rows[0]["method"] == "phase-carrier optimization not applied: refinement budget"
        assert rows[0]["from_shape"] == rows[0]["to_shape"] == (128, 128)
        assert "refinement budget exceeded" in rows[0]["reason"]
        assert rows[0]["probability"] == initial.probability
    else:
        # Explicitly disabling refinement still reports the typed unresolved
        # chirp. This request is not a declared memory/size-budget failure.
        with pytest.raises(WaveSamplingError, match="compensating chirp"):
            reselect_phase_carrier(initial, lam, grid_numerics=numerics)
    for name, expected in zip(("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad"), original):
        np.testing.assert_array_equal(getattr(initial, name), expected)


def test_optional_carrier_fallback_does_not_admit_an_unresolved_physical_multipole():
    from temsim.physics.multipole_wave import apply_multipole_phase
    from temsim.physics.wave_grid import apply_resolved_operator
    initial, lam = _sheared_non_gaussian_carrier()
    numerics = WaveGridNumerics(maximum_pixels=128)
    retained, rows = reselect_phase_carrier(initial, lam, grid_numerics=numerics)
    assert retained is initial and "not applied" in rows[0]["method"]
    operator = lambda wave: apply_multipole_phase(wave, lam, normal_m2=1e17)
    with pytest.raises(WaveSamplingError, match="Nonlinear column phase is undersampled"):
        operator(retained)
    with pytest.raises(WaveGridBudgetError, match="refinement budget exceeded"):
        apply_resolved_operator(retained, operator, numerics=numerics)
    assert retained is initial


def test_optional_carrier_memory_budget_keeps_original_grid_and_full_phase_state():
    initial, lam = _sheared_non_gaussian_carrier()
    numerics = WaveGridNumerics(maximum_pixels=1024, maximum_working_bytes=8*1024**2)
    # The admitted input fits; only the optional compensating chirp's larger
    # grid exceeds this memory budget. Nothing lowers the physical budget.
    assert numerics.check(initial.amplitude.shape) < numerics.maximum_working_bytes
    result, rows = reselect_phase_carrier(initial, lam, grid_numerics=numerics)
    assert result is initial
    assert rows[0]["from_shape"] == rows[0]["to_shape"] == (128, 128)
    assert "maximum_working_bytes=8388608" in rows[0]["reason"]
    assert "not applied" in rows[0]["method"]


@pytest.mark.parametrize("failure", [ValueError("invalid operator"), MemoryError("physical memory unavailable"),
                                    InterruptedError("gauge cancelled"),
                                    WaveSamplingError("unresolved numerical operator", required_scale=2.)])
def test_optional_carrier_budget_fallback_does_not_swallow_other_operator_failures(monkeypatch, failure):
    import temsim.physics.wave_grid as grid
    initial, lam = _sheared_non_gaussian_carrier()
    original = initial.amplitude.copy()
    def fail(*args, **kwargs):
        raise failure
    # The exception-control boundary is isolated here; the preceding cases
    # exercise actual chirp calculation and actual physical-operator refusal.
    monkeypatch.setattr(grid, "apply_resolved_operator", fail)
    with pytest.raises(type(failure)) as caught:
        reselect_phase_carrier(initial, lam, grid_numerics=WaveGridNumerics(maximum_pixels=128))
    assert caught.value is failure
    np.testing.assert_array_equal(initial.amplitude, original)


def test_physical_propagation_still_honours_cancellation_after_optional_carrier_fallback():
    initial, lam = _sheared_non_gaussian_carrier()
    numerics = WaveGridNumerics(maximum_pixels=128)
    retained, _ = reselect_phase_carrier(initial, lam, grid_numerics=numerics)
    assert retained is initial
    records = [{"previous_completed_stage": True}]
    with pytest.raises(InterruptedError, match="Canonical plane propagation cancelled"):
        propagate_plane_wave(retained, drift(1e-6), np.zeros(4), lam,
                             grid_numerics=numerics, cancelled=lambda: True, sampling_records=records)
    assert records == [{"previous_completed_stage": True}]


def test_exact_carrier_gauge_never_resets_physical_oscillator_winding():
    n, lam, length, sigma = 128, 2e-12, 1e-4, 3e-9
    axis = (np.arange(n)-n//2)*1e-9
    amplitude = np.exp(-(axis[:, None]**2+axis[None, :]**2)/(4*sigma*sigma)).astype(complex)
    amplitude /= np.linalg.norm(amplitude)
    initial = PlaneWave(amplitude, np.eye(2)*1e-9, np.zeros(2))
    current, accumulated = initial, CanonicalPath(length)
    step = oscillator(2*np.pi/256, length)
    for _ in range(256):
        current, _ = reselect_phase_carrier(current, lam, grid_numerics=WaveGridNumerics(maximum_pixels=1024))
        increment = CanonicalPath(length)
        increment.append(step)
        accumulated.append(step)
        current = propagate_plane_wave(current, step, np.zeros(4), lam,
            **increment.phase_kwargs(), grid_numerics=WaveGridNumerics(maximum_pixels=1024))
    assert accumulated.reference_phase_rad == pytest.approx(-np.pi, abs=1e-12)
    # Gauge-dependent scaled-AS charts can end on a different numerical
    # lattice. Independently evaluate the exact physical -1 loop field at
    # every actual output coordinate, including its probability-cell area.
    local = current.coordinates_m()-current.origin_m[:, None, None]
    expected = -initial.amplitude[n//2, n//2]*np.exp(-np.sum(local**2, axis=0)/(4*sigma*sigma))
    expected *= np.sqrt(abs(np.linalg.det(current.basis_m)/np.linalg.det(initial.basis_m)))
    np.testing.assert_allclose(current.full_amplitude(lam), expected, rtol=1e-6, atol=2e-9)


def test_optional_carrier_with_boundary_mass_keeps_the_original_complex_field():
    n, sigma = 64, 12.
    axis = np.arange(n)-n//2
    wave = PlaneWave(np.exp(-(axis[:, None]**2+axis[None, :]**2)/(4*sigma*sigma)),
                     np.eye(2)*1e-9, np.zeros(2), curvature_m1=np.eye(2)*1000.)
    result, rows = reselect_phase_carrier(wave, 2e-12,
        grid_numerics=WaveGridNumerics(maximum_pixels=64))
    assert result is wave and rows[0]["method"].startswith("phase-carrier optimization not applied")
    np.testing.assert_array_equal(result.amplitude, wave.amplitude)
    np.testing.assert_array_equal(result.curvature_m1, wave.curvature_m1)
    np.testing.assert_array_equal(result.basis_m, wave.basis_m)
    np.testing.assert_array_equal(result.origin_m, wave.origin_m)
    assert result.tilt_rad is wave.tilt_rad
    # This finite Gaussian window has a significant boundary cusp. Its
    # independently calculated Nyquist-bin mass alone exceeds the declared
    # spectral-tail budget. Retaining the original representation is valid;
    # materializing an unqualified full field is a separate operation.
    spectrum = abs(np.fft.fft2(wave.amplitude, norm='ortho'))**2
    assert spectrum[n//2, :].sum() > wave.probability*1e-12
    for preserved in (wave, result):
        with pytest.raises(WaveSamplingError, match='undersampled'):
            preserved.full_amplitude(2e-12)


def test_carrier_cancel_and_budget_are_checked_even_for_empty_or_noop_fields(monkeypatch):
    empty = PlaneWave(np.zeros((64, 64), complex), np.eye(2)*1e-9, np.zeros(2))
    def should_not_allocate(*args):
        raise AssertionError("Covariance was allocated before admission")
    monkeypatch.setattr(PlaneWave, "canonical_covariance", should_not_allocate)
    with pytest.raises(InterruptedError, match="selection cancelled"):
        reselect_phase_carrier(empty, 2e-12, grid_numerics=WaveGridNumerics(), cancelled=lambda: True)
    with pytest.raises(ValueError, match="refinement budget exceeded"):
        reselect_phase_carrier(empty, 2e-12,
            grid_numerics=WaveGridNumerics(maximum_working_bytes=1))


@pytest.mark.parametrize("real_factor", [-1e-8, 0., 1e-8])
def test_real_isotropic_carrier_caustic_keeps_finite_complex_gaussian(real_factor):
    n, step, sigma, lam, distance = 128, .5e-9, 3e-9, 2e-12, 1e-4
    axis = (np.arange(n)-n//2)*step
    amplitude = np.exp(-(axis[:, None]**2+axis[None, :]**2)/(4*sigma*sigma)).astype(complex)
    amplitude /= np.linalg.norm(amplitude)
    curvature = (real_factor-1)/distance
    initial = PlaneWave(amplitude, np.eye(2)*step, np.zeros(2), curvature_m1=np.eye(2)*curvature)
    actual = propagate_plane_wave(initial, drift(distance), np.zeros(4), lam,
        grid_numerics=WaveGridNumerics(maximum_pixels=1024))
    complex_q = curvature+1j*lam/(4*np.pi*sigma*sigma)
    factor = 1+distance*complex_q
    local = actual.coordinates_m()-actual.origin_m[:, None, None]
    expected = initial.amplitude[n//2, n//2]/factor*np.exp(
        1j*np.pi/lam*complex_q/factor*np.sum(local**2, axis=0))
    expected *= np.sqrt(abs(np.linalg.det(actual.basis_m)/np.linalg.det(initial.basis_m)))
    np.testing.assert_allclose(actual.full_amplitude(lam), expected, rtol=1e-9, atol=1e-12)
    assert actual.probability == pytest.approx(initial.probability, abs=1e-12)


def _short_drift_domain_fixture():
    n, step, sigma, lam, distance, q = 64, 1e-9, 3.5e-9, 2e-12, 8e-5, 1000.
    rotation = np.array(((np.cos(.37), -np.sin(.37)), (np.sin(.37), np.cos(.37))))
    zero = PlaneWave(np.zeros((n, n), complex), rotation*step, np.array((2e-9, -3e-9)))
    local = zero.coordinates_m()-zero.origin_m[:, None, None]
    packet_shift = rotation@np.array((3e-9, -2e-9))
    extra_tilt = rotation@np.array((lam*.03/step, 0.))
    tail = 1e-4*np.exp(.3j)
    amplitude = np.exp(-np.sum(local**2, axis=0)/(4*sigma*sigma))
    amplitude = amplitude+tail*np.exp(-np.sum((local-packet_shift[:, None, None])**2, axis=0)/(4*sigma*sigma)) \
        *np.exp(2j*np.pi/lam*np.einsum('i,iyx->yx', extra_tilt, local))
    amplitude /= np.linalg.norm(amplitude)
    wave = replace(zero, amplitude=amplitude, curvature_m1=np.eye(2)*q,
                   tilt_rad=np.array((2e-5, -1e-5)))
    return wave, lam, distance, sigma, q, tail, packet_shift, extra_tilt


def test_short_drift_domain_extension_retains_full_finite_complex_field_and_analytic_packets():
    wave, lam, distance, sigma, q, tail, shift, extra_tilt = _short_drift_domain_fixture()
    rows = []
    output = propagate_plane_wave(wave, drift(distance), np.zeros(4), lam,
        grid_numerics=WaveGridNumerics(maximum_pixels=1024), sampling_records=rows)
    assert len(rows) == 1 and rows[0]['method'].startswith('spatial zero extension')
    row = rows[0]
    assert row['previous_output_boundary_probability'] > wave.probability*1e-12
    assert row['output_boundary_probability'] < wave.probability*1e-12
    assert row['full_spectral_band_checked'] and row['spectral_phase_increment_rad'] < np.pi
    assert np.all(np.array(row['domain_margin_cells_xy']) > row['maximum_group_displacement_cells_xy'])
    np.testing.assert_array_equal(row['basis_m'], wave.basis_m)
    assert output.probability == pytest.approx(wave.probability, abs=1e-12)
    # Independently evaluate the finite periodic Collins/Fresnel kernel from
    # its exact Fourier eigenvalues using direct matrices, not the runtime
    # FFT. All retained finite input cells and all expanded output cells enter
    # this complex quadrature; no Gaussian fit or output crop is involved.
    n, m = wave.amplitude.shape[0], output.amplitude.shape[0]
    padded = np.zeros((m, m), complex)
    start = m//2-n//2
    padded[start:start+n, start:start+n] = wave.amplitude
    indices = np.arange(m)
    fourier = np.exp(-2j*np.pi*np.outer(indices, indices)/m)/np.sqrt(m)
    spectral_phase = np.exp(-1j*np.pi*lam*distance/(1+distance*q)*(np.fft.fftfreq(m)/1e-9)**2)
    kernel = fourier.conj().T@(spectral_phase[:, None]*fourier)
    expected = kernel@padded@kernel.T
    local = output.coordinates_m()-output.origin_m[:, None, None]
    expected *= np.exp(1j*np.pi/lam*(q/(1+distance*q)*np.sum(local**2, axis=0)
        +2*np.einsum('i,iyx->yx', wave.tilt_rad, local)+distance*np.dot(wave.tilt_rad, wave.tilt_rad)))
    np.testing.assert_allclose(output.full_amplitude(lam), expected, rtol=1e-9, atol=1e-12)
    # A second independent reference is the analytic sum of two coherent
    # Gaussians. Its only finite-input discrepancy is independently bounded
    # by the norm of omitted cells of that KNOWN analytic input, not by the
    # small output edge strip or a relaxed runtime probability tolerance.
    coefficient = wave.amplitude[n//2, n//2]/(1+tail*np.exp(-np.dot(shift, shift)/(4*sigma*sigma)))
    complex_q = q+1j*lam/(4*np.pi*sigma*sigma)
    factor = 1+distance*complex_q
    analytic = np.zeros(output.amplitude.shape, complex)
    for centre, angular, weight in ((np.zeros(2), np.zeros(2), 1.), (shift, extra_tilt, tail)):
        tilt = wave.tilt_rad+q*centre+angular
        packet_centre = wave.origin_m+centre+distance*tilt
        delta = output.coordinates_m()-packet_centre[:, None, None]
        constant = q*np.dot(centre, centre)/2+np.dot(wave.tilt_rad+angular, centre)+distance*np.dot(tilt, tilt)/2
        analytic += coefficient*weight/factor*np.exp(1j*np.pi/lam*complex_q/factor*np.sum(delta**2, axis=0)
            +2j*np.pi/lam*(np.einsum('i,iyx->yx', tilt, delta)+constant))
    analytic *= np.sqrt(abs(np.linalg.det(output.basis_m)/np.linalg.det(wave.basis_m)))
    expanded = replace(wave, amplitude=np.zeros((512, 512), complex))
    input_local = expanded.coordinates_m()-expanded.origin_m[:, None, None]
    known_input = coefficient*(np.exp(-np.sum(input_local**2, axis=0)/(4*sigma*sigma))
        +tail*np.exp(-np.sum((input_local-shift[:, None, None])**2, axis=0)/(4*sigma*sigma))
        *np.exp(2j*np.pi/lam*np.einsum('i,iyx->yx', extra_tilt, input_local)))
    outside = np.ones((512, 512), bool)
    outside[256-n//2:256+n//2, 256-n//2:256+n//2] = False
    omitted = np.linalg.norm(known_input[outside])
    assert omitted < 1e-7
    assert np.linalg.norm(output.full_amplitude(lam)-analytic) <= omitted+1e-11
    # Further physical-domain extension retains the full result and tests
    # periodic-image contamination at the actual physical coordinates.
    twice = np.zeros((2*m, 2*m), complex)
    start = m-n//2
    twice[start:start+n, start:start+n] = wave.amplitude
    larger = propagate_plane_wave(replace(wave, amplitude=twice), drift(distance), np.zeros(4), lam,
        grid_numerics=WaveGridNumerics(maximum_pixels=2048))
    centre = larger.amplitude.shape[0]//2
    np.testing.assert_allclose(output.full_amplitude(lam),
        larger.full_amplitude(lam)[centre-m//2:centre+m//2, centre-m//2:centre+m//2], rtol=1e-9, atol=1e-12)
    assert larger.probability == pytest.approx(wave.probability, abs=1e-12)


@pytest.mark.parametrize('numerics,message', [
    (WaveGridNumerics(maximum_pixels=128), 'domain extension budget exceeded'),
    (WaveGridNumerics(maximum_pixels=1024, maximum_working_bytes=2*1024**2), 'domain extension budget exceeded'),
    (WaveGridNumerics(maximum_pixels=1024, automatic_refinement=False), 'automatic refinement is disabled'),
])
def test_spatial_domain_budget_failure_never_retries_fourier_input_refinement(numerics, message):
    from temsim.physics.multiplane_wave import _extend_angular_spectrum_domain
    wave, lam, distance, *_ = _short_drift_domain_fixture()
    original = wave.amplitude.copy()
    rows = []
    # The domain chart must still reject its SAME budget. The public optical
    # operator may now select a separately resolved Collins representation;
    # that behavior has independent full-complex references below.
    spectral_drift = np.eye(2)*distance/(1+distance*wave.curvature_m1[0, 0])
    diagnostic = {}
    assert _sampled_angular_spectrum(wave, spectral_drift, lam,
        conservative=False, diagnostic=diagnostic) is None
    assert diagnostic['failure'] == 'output_boundary'
    with pytest.raises(ValueError, match=message) as caught:
        _extend_angular_spectrum_domain(wave, spectral_drift, lam, diagnostic,
            numerics=numerics, sampling_records=rows)
    assert not isinstance(caught.value, WaveSamplingError)
    np.testing.assert_array_equal(wave.amplitude, original)
    assert rows == []


def test_spatial_domain_extension_cancels_before_padding_or_fft():
    wave, lam, distance, *_ = _short_drift_domain_fixture()
    checks = []
    def cancelled():
        checks.append(True)
        return len(checks) >= 2
    rows = []
    with pytest.raises(InterruptedError, match='domain extension cancelled'):
        propagate_plane_wave(wave, drift(distance), np.zeros(4), lam,
            grid_numerics=WaveGridNumerics(maximum_pixels=1024), cancelled=cancelled, sampling_records=rows)
    assert len(checks) == 2 and rows == []


def test_angular_spectrum_guard_keeps_collectively_significant_weak_frequency_tails():
    n, lam = 64, 2e-12
    spectrum = np.zeros((n, n), complex)
    spectrum[0, 0] = 1.
    spectrum[:, 20:32] = 8e-8*np.exp(.3j)
    tail_probability = np.sum(abs(spectrum[:, 20:32])**2)
    assert tail_probability > 1e-12 and np.max(abs(spectrum[:, 20:32])**2) < 1e-14
    wave = PlaneWave(np.fft.ifft2(spectrum, norm='ortho'), np.eye(2)*1e-9, np.zeros(2))
    diagnostic = {}
    assert _sampled_angular_spectrum(wave, np.eye(2)*6e-5, lam,
        conservative=False, diagnostic=diagnostic) is None
    assert diagnostic['failure'] == 'spectral_sampling'


def test_full_phase_expansion_keeps_resolved_nodal_sign_changes():
    # A low-frequency cosine has a legitimate pi sign change across its
    # zeros. Adding a small resolved tilt must not interpret that branch as
    # a local bandwidth exceeding Nyquist.
    n, step, sigma_cells, lam = 128, 1e-9, 4., 2e-12
    axis = np.arange(n)-n//2
    yy, xx = np.meshgrid(axis, axis, indexing='ij')
    carrier_increment, frequency = .1, np.pi/16
    raw = np.exp(-(xx*xx+yy*yy)/(4*sigma_cells*sigma_cells))*np.cos(frequency*(xx+.35))
    coefficient = 1/np.linalg.norm(raw)
    wave = PlaneWave((coefficient*raw).astype(complex), np.eye(2)*step, np.zeros(2),
                     tilt_rad=np.array((carrier_increment*lam/(2*np.pi*step), 0.)))
    expected = coefficient*raw*np.exp(1j*carrier_increment*xx)
    # Independent Fourier-series evaluation on a twice-finer physical
    # lattice matches the KNOWN analytic cosine/Gaussian/tilt field. This
    # compares the complete complex field without fitting phase or norm.
    fine_axis = (np.arange(2*n)-n)/2
    fine_y, fine_x = np.meshgrid(fine_axis, fine_axis, indexing='ij')
    analytic_fine = coefficient/2*np.exp(-(fine_x*fine_x+fine_y*fine_y)/(4*sigma_cells*sigma_cells)) \
        *np.cos(frequency*(fine_x+.35))*np.exp(1j*carrier_increment*fine_x)
    evaluation = np.exp(2j*np.pi*np.outer(np.arange(2*n)/2, np.fft.fftfreq(n)))/np.sqrt(2*n)
    interpolated = evaluation@np.fft.fft2(expected, norm='ortho')@evaluation.T
    assert np.linalg.norm(interpolated-analytic_fine) < 2e-12
    fine_spectrum = abs(np.fft.fft2(analytic_fine, norm='ortho'))**2
    frequency_cells = abs(2*np.fft.fftfreq(2*n))
    outside = (frequency_cells[:, None] >= .5) | (frequency_cells[None, :] >= .5)
    assert fine_spectrum[outside].sum() < 1e-20
    actual = wave.full_amplitude(lam)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
    assert np.sum(abs(actual)**2) == pytest.approx(wave.probability, abs=1e-12)


@pytest.mark.parametrize('kind', ['joint', 'curvature'])
def test_full_phase_expansion_rejects_real_alias_with_typed_refinement(kind):
    n, step, lam = 128, 1e-9, 2e-12
    if kind == 'joint':
        # Each known plane-wave frequency is individually below Nyquist;
        # together they reach it exactly. The guard must share the budget.
        envelope_increment = 2*np.pi*40/n
        carrier_increment = 2*np.pi*24/n
        amplitude = np.broadcast_to(np.exp(1j*envelope_increment*np.arange(n)), (n, n))/n
        wave = PlaneWave(amplitude, np.eye(2)*step, np.zeros(2),
            tilt_rad=np.array((carrier_increment*lam/(2*np.pi*step), 0.)))
    else:
        axis = np.arange(n)-n//2
        amplitude = np.exp(-(axis[:, None]**2+axis[None, :]**2)/64).astype(complex)
        amplitude /= np.linalg.norm(amplitude)
        wave = PlaneWave(amplitude, np.eye(2)*step, np.zeros(2), curvature_m1=np.eye(2)*1e5)
    original = wave.amplitude.copy()
    with pytest.raises(WaveSamplingError, match='undersampled') as caught:
        wave.full_amplitude(lam)
    assert caught.value.required_scale > 1.
    np.testing.assert_array_equal(wave.amplitude, original)


@pytest.mark.parametrize('automatic', [True, False])
def test_optional_domain_budget_can_use_resolved_full_complex_collins_alternate(automatic):
    n, step, sigma, lam, distance, q = 64, 1e-9, 3.5e-9, 2e-12, 8e-5, 1000.
    axis = (np.arange(n)-n//2)*step
    raw = np.exp(-(axis[:, None]**2+axis[None, :]**2)/(4*sigma*sigma)).astype(complex)
    coefficient = 1/np.linalg.norm(raw)
    initial = PlaneWave(coefficient*raw, np.eye(2)*step, np.zeros(2), curvature_m1=np.eye(2)*q)
    lens = np.eye(4)
    lens[2:, :2] = -np.eye(2)/distance
    matrix = lens@drift(distance)
    # All represented spectral bins need >74 cells of physical-domain
    # margin; a 128-cell padded domain provides only 32. This optional AS
    # chart therefore cannot fit the budget, while Collins is resolvable.
    group_cells = lam*distance/(1+distance*q)/(2*step*step)
    assert group_cells > (128-n)/2
    as_diagnostic = {}
    assert _sampled_angular_spectrum(initial, np.eye(2)*distance/(1+distance*q), lam,
        conservative=False, diagnostic=as_diagnostic) is None
    assert as_diagnostic['failure'] == 'output_boundary'
    assert as_diagnostic['spectral_phase_increment_rad'] < np.pi
    records = []
    result = propagate_plane_wave(initial, matrix, np.zeros(4), lam,
        grid_numerics=WaveGridNumerics(maximum_pixels=128, automatic_refinement=automatic), sampling_records=records)
    assert max(result.amplitude.shape) <= 128
    assert any('Collins alternate' in row['method'] for row in records)
    assert result.probability == pytest.approx(initial.probability, abs=1e-12)
    output_axis = result.coordinates_m()[0, result.amplitude.shape[0]//2]
    # Independent finite Collins quadrature retains every complex INPUT
    # cell and compares every returned OUTPUT cell, including all tails.
    chirp = np.exp(1j*np.pi/lam*(1/distance+q)*(axis[:, None]**2+axis[None, :]**2))
    kernel = np.exp(-2j*np.pi/(lam*distance)*np.outer(output_axis, axis))
    output_step = np.sqrt(abs(np.linalg.det(result.basis_m)))
    finite_expected = -1j*step*output_step/(lam*distance)*kernel@(initial.amplitude*chirp)@kernel.T
    np.testing.assert_allclose(result.full_amplitude(lam), finite_expected, rtol=1e-9, atol=1e-12)
    # A second independent reference is the continuous known Gaussian,
    # including the actual downstream lens and the Fresnel scalar phase.
    complex_q = q+1j*lam/(4*np.pi*sigma*sigma)
    factor = 1+distance*complex_q
    q_out = (-1/distance)/factor
    local = result.coordinates_m()-result.origin_m[:, None, None]
    analytic = coefficient/factor*np.exp(1j*np.pi/lam*q_out*np.sum(local**2, axis=0))*output_step/step
    # Known omitted input Gaussian mass is computed independently on a
    # larger exact analytic grid; no fitted phase or normalization appears.
    outer_axis = (np.arange(512)-256)*step
    known = coefficient*np.exp(-(outer_axis[:, None]**2+outer_axis[None, :]**2)/(4*sigma*sigma))
    outside = np.ones((512, 512), bool)
    outside[256-n//2:256+n//2, 256-n//2:256+n//2] = False
    omitted_norm = np.linalg.norm(known[outside])
    assert omitted_norm < 1e-9
    assert np.linalg.norm(result.full_amplitude(lam)-analytic) <= omitted_norm+1e-11


@pytest.mark.parametrize('failure', [ValueError('field implementation failure'),
                                    MemoryError('allocation failure'), InterruptedError('cancelled')])
def test_optional_domain_alternate_never_swallows_cancellation_or_unrelated_errors(monkeypatch, failure):
    from temsim.physics import multiplane_wave as module
    wave, lam, distance, *_ = _short_drift_domain_fixture()
    def stop(*args, **kwargs):
        raise failure
    monkeypatch.setattr(module, '_extend_angular_spectrum_domain', stop)
    monkeypatch.setattr(module, '_propagate_fresnel_chart', lambda *a, **k: pytest.fail('unrelated failure was swallowed'))
    records = [{'previous_operator': 'completed'}]
    with pytest.raises(type(failure)) as caught:
        propagate_plane_wave(wave, drift(distance), np.zeros(4), lam,
            grid_numerics=WaveGridNumerics(maximum_pixels=1024), sampling_records=records)
    assert caught.value is failure and records == [{'previous_operator': 'completed'}]


def test_both_chart_refusals_keep_diagnostics_typed_refinement_and_atomic_records(monkeypatch):
    from temsim.physics import multiplane_wave as module
    wave, lam, distance, *_ = _short_drift_domain_fixture()
    def unresolved_collins(*args, **kwargs):
        kwargs['sampling_records'].append({'unaccepted_partial_attempt': True})
        raise WaveSamplingError('independently unresolved Collins chirp', required_scale=3.25)
    monkeypatch.setattr(module, '_propagate_fresnel_chart', unresolved_collins)
    original = wave.amplitude.copy()
    records = [{'previous_operator': 'completed'}]
    with pytest.raises(WaveSamplingError, match='domain extension budget exceeded.*Collins alternate failed') as caught:
        propagate_plane_wave(wave, drift(distance), np.zeros(4), lam,
            grid_numerics=WaveGridNumerics(maximum_pixels=64), sampling_records=records)
    assert caught.value.required_scale == 3.25
    assert isinstance(caught.value.__cause__, WaveSamplingError)
    assert records == [{'previous_operator': 'completed'}]
    np.testing.assert_array_equal(wave.amplitude, original)


def test_only_declared_grid_limits_raise_the_budget_subclass():
    with pytest.raises(WaveGridBudgetError, match='budget exceeded'):
        WaveGridNumerics(maximum_pixels=32).check((64, 64))
    with pytest.raises(ValueError, match='Maximum wave grid pixels') as caught:
        WaveGridNumerics(maximum_pixels=1).check((64, 64))
    assert not isinstance(caught.value, WaveGridBudgetError)
