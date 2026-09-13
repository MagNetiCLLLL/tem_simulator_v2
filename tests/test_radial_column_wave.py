"""Analytic phase/current checks for the round-column reduction."""
import numpy as np
import pytest

from temsim.physics.radial_column_wave import RadialWave, propagate_radial, radial_cs, refine_radial
from temsim.physics.canonical_action import CanonicalPath


def gaussian():
    # Both reciprocal endpoints must be negligible, independently of the
    # phase/current test. FFTLog is periodic in log radius.
    radius = np.geomspace(1e-10, 100., 32768, endpoint=False)
    return RadialWave(radius, np.exp(-radius**2/2)/np.sqrt(np.pi))


@pytest.mark.parametrize("distance", (.2, -.2, 2.))
def test_complex_free_gaussian_has_absolute_phase_and_conserved_current(distance):
    wave = gaussian()
    m = np.eye(4); m[:2, 2:] = np.eye(2)*distance
    path = CanonicalPath(1.)
    path.append(m)
    result = propagate_radial(wave, m, 2*np.pi, reference_phase_rad=path.reference_phase_rad,
                              reference_length_m=1.)
    exact = np.exp(-result.radius_m**2/(2*(1+1j*distance)))/(np.sqrt(np.pi)*(1+1j*distance))
    # Global weighted error; no fitted phase or output normalisation.
    full = result.amplitude*np.exp(.5j*result.curvature_m1*result.radius_m**2)
    error = np.sqrt(2*np.pi*result.dln*np.sum(abs(result.radius_m*(full-exact))**2))
    assert error < 1e-6
    assert result.probability == pytest.approx(wave.probability, rel=1e-12)


def test_spherical_phase_is_kept_not_fitted_into_quadratic_curvature():
    wave = gaussian()
    result = radial_cs(wave, 2*np.pi, .1)
    np.testing.assert_allclose(result.amplitude, wave.amplitude*np.exp(-.025j*wave.radius_m**4))
    assert result.curvature_m1 == wave.curvature_m1
    assert result.probability == pytest.approx(wave.probability, rel=1e-12)


def test_log_grid_refinement_preserves_complex_wave_not_just_current():
    wave = gaussian()
    finer = refine_radial(wave, 1.5)
    exact = np.exp(-finer.radius_m**2/2)/np.sqrt(np.pi)
    assert finer.probability == pytest.approx(wave.probability, rel=1e-12)
    weighted = np.sqrt(2*np.pi*finer.dln*np.sum(abs(finer.radius_m*(finer.amplitude-exact))**2))
    assert weighted < 1e-8


def test_refinement_replays_hard_stops_from_executed_field_and_replaces_loss_ledger():
    from temsim.physics.radial_column_wave import RadialExecution, _radial_stop
    from temsim.physics.wave_grid import WaveSamplingError
    loaded = []
    def load(samples):
        loaded.append(samples)
        r = np.geomspace(1e-8, 10., samples, endpoint=False)
        return RadialWave(r, np.exp(-r*r/2)/np.sqrt(np.pi))
    execution = RadialExecution(load, 64, 4096)
    execution.apply(lambda w: _radial_stop(w, 1.), metadata={"component": "aperture"})
    old_loss = execution.ledger[0].copy()
    def fine_operator(wave):
        if len(wave.radius_m) < 512:
            raise WaveSamplingError("Independent resolution check", 512/len(wave.radius_m))
        return wave
    result = execution.apply(fine_operator)
    assert len(loaded) == 2
    assert execution.refinements[-1]["kind"] == "upstream_replay"
    # No Fourier ringing past the physical hard stop; no normalization.
    assert np.all(result.amplitude[result.radius_m > 1.] == 0)
    assert len(execution.ledger) == 1
    assert execution.ledger[0]["outgoing"] == result.probability
    assert execution.ledger[0]["outgoing"] != old_loss["outgoing"]
    exact = np.where(result.radius_m <= 1., np.exp(-result.radius_m**2/2)/np.sqrt(np.pi), 0j)
    np.testing.assert_array_equal(result.amplitude, exact)


def test_replay_cannot_silently_exceed_its_declared_budget():
    from temsim.physics.radial_column_wave import RadialExecution
    from temsim.physics.wave_grid import WaveSamplingError
    execution = RadialExecution(lambda samples: gaussian(), 64, 128)
    def unresolved(wave):
        raise WaveSamplingError("Unresolved physical phase", 100.)
    with pytest.raises(ValueError, match="above its 128 budget"):
        execution.apply(unresolved)
    assert not execution.operations


def test_quartic_field_hankel_transform_against_direct_bessel_integral():
    from scipy.special import j0
    from temsim.physics.radial_column_wave import _hankel_grid, _complex_fht
    radius = np.geomspace(1e-10, 100., 524288, endpoint=False)
    wave = RadialWave(radius, np.exp(-radius**2/2-5j*radius**4)/np.sqrt(np.pi))
    offset, frequency = _hankel_grid(wave)
    transformed = _complex_fht(radius*wave.amplitude, wave.dln, offset)/frequency
    requested = np.array([.1, 1., 3., 10., 30., 100.])
    nodes, weights = np.polynomial.legendre.leggauss(8)
    edges = np.linspace(0., 80., 80001)
    half = np.diff(edges)/2
    u = ((edges[:-1]+edges[1:])[:, None]/2+half[:, None]*nodes).ravel()
    integrand = np.exp(-u/2-5j*u*u)/(2*np.sqrt(np.pi))*(half[:, None]*weights).ravel()
    # Independent quadrature in u=r^2. No fitted global phase or normalisation.
    exact = np.array([np.dot(integrand, j0(k*np.sqrt(u))) for k in requested])
    actual = np.interp(np.log(requested), np.log(frequency), transformed)
    np.testing.assert_allclose(actual, exact, atol=2e-8, rtol=0.)
