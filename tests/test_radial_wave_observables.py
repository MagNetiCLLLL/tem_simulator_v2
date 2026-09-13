"""Independent derivative and flux checks; not a full-source certificate."""
import numpy as np
import pytest
from scipy.special import eval_laguerre

from temsim.physics.radial_wave_observables import radial_mode_observables, incoherent_radial_observables
from temsim.physics.quartic_radial_phase import quartic_operators


def parameters():
    return dict(width_nm=1.7, curvature_per_nm=.12, reference_k_per_nm=3.,
        width_log_rate_per_nm=.15, curvature_prime_per_nm2=.2,
        quartic_phase_per_nm4=.006, quartic_prime_per_nm5=.015)


@pytest.mark.parametrize("count", [1, 4, 16])
def test_physical_derivative_includes_exact_outside_basis_chart_terms(count):
    p = parameters()
    rng = np.random.default_rng(871+count)
    a = rng.normal(size=count)+1j*rng.normal(size=count)
    d = rng.normal(size=count)+1j*rng.normal(size=count)
    r = np.linspace(0., 8., 151)
    result = radial_mode_observables(r, a, d, **p)
    _, connection, _ = quartic_operators(count, p["width_nm"], p["curvature_per_nm"],
        p["reference_k_per_nm"], p["width_log_rate_per_nm"], p["curvature_prime_per_nm2"],
        p["quartic_phase_per_nm4"], p["quartic_prime_per_nm5"])
    slope = d-1j*connection@a
    def physical_field(z):
        width = p["width_nm"]*np.exp(p["width_log_rate_per_nm"]*z)
        curvature = p["curvature_per_nm"]+p["curvature_prime_per_nm2"]*z
        quartic = p["quartic_phase_per_nm4"]+p["quartic_prime_per_nm5"]*z
        x = (r/width)**2
        basis = np.column_stack([eval_laguerre(n, x) for n in range(count)])
        return basis@(a+z*slope)*np.exp(-x/2+1j*(1.5*curvature*r*r+quartic*r**4))/(np.sqrt(np.pi)*width)
    h = 2e-6
    exact = (physical_field(-2*h)-8*physical_field(-h)+8*physical_field(h)-physical_field(2*h))/(12*h)
    np.testing.assert_allclose(result.normal_derivative_per_nm, exact, atol=3e-9, rtol=3e-9)
    np.testing.assert_allclose(result.amplitude, physical_field(0), atol=1e-13)


def test_radial_integral_matches_full_covariant_flux_without_local_clipping():
    p = parameters()
    a = np.array([1.+.2j, -.1+.5j, .3-.4j])
    d = np.array([.2+3j, 1.-.3j, -.2-.5j])
    x, w = np.polynomial.legendre.leggauss(160)
    u, weights = (x+1)*40, w*40*np.pi*p["width_nm"]**2
    result = radial_mode_observables(p["width_nm"]*np.sqrt(u), a, d, **p)
    assert np.any(result.axial_flux_fraction_per_nm2 < 0)  # Not clipped to particle counts.
    assert weights@result.axial_flux_fraction_per_nm2 == pytest.approx(np.vdot(a, d).imag, abs=2e-12)
    assert result.integrated_axial_flux_fraction == np.vdot(a, d).imag
    assert not result.amplitude.flags.writeable


def test_standing_wave_has_intensity_but_no_net_flux_and_mixture_has_no_phase():
    p = {**parameters(), "width_log_rate_per_nm": 0., "curvature_prime_per_nm2": 0.,
         "quartic_prime_per_nm5": 0.}
    radius = np.linspace(0., 10., 200)
    standing = radial_mode_observables(radius, [2.], [0.], **p)
    assert standing.scalar_intensity.max() > 0
    np.testing.assert_array_equal(standing.axial_flux_fraction_per_nm2, 0)
    forward = radial_mode_observables(radius, [1.], [3j], **p)
    opposite_phase = radial_mode_observables(radius, [-1.], [-3j], **p)
    readout = incoherent_radial_observables([forward, opposite_phase], [.2, .3], reference_current_a=1e-9)
    np.testing.assert_allclose(readout["scalar_intensity"], .5*forward.scalar_intensity)
    assert readout["integrated_axial_current_a"] == pytest.approx(1.5e-9)
    assert "amplitude" not in readout and "phase_rad" not in readout
    assert "No aggregate phase" in readout["phase_scope"]


def test_exact_zero_wave_has_undefined_phase_and_zero_current():
    result = radial_mode_observables([0., 1.], [0.], [0.], **parameters())
    assert np.all(np.isnan(result.phase_rad))
    assert result.integrated_axial_flux_fraction == 0


@pytest.mark.parametrize("radius,a,d,extra", [
    ([0., 0.], [1.], [1j], {}), ([-1., 1.], [1.], [1j], {}),
    ([0., 1.], [1.], [1j, 0j], {}), ([0., 1.], [np.nan], [1j], {}),
    ([0., 1.], [1.], [1j], {"width_nm": 0}),
    ([0., 1.], [1.], [1j], {"quartic_prime_per_nm5": np.inf})])
def test_invalid_observable_inputs_are_not_repaired(radius, a, d, extra):
    with pytest.raises(ValueError):
        radial_mode_observables(radius, a, d, **{**parameters(), **extra})


def test_stored_boundary_reader_does_not_guess_missing_rates_or_interpolate_across_stops():
    from temsim.physics.radial_wave_observables import observe_executed_boundary
    history = {"z_nm": np.array([1., 1.]), "width_curvature": np.array([[2., 0.], [2., 0.]]),
        "quartic_phase_per_nm4": np.zeros(2), "chart_derivatives": np.zeros((2, 3)),
        "coefficients": np.array([[1.], [.5]]), "covariant_derivatives": np.array([[2j], [1j]])}
    before = observe_executed_boundary(history, 0, [0., 1.], reference_k_per_nm=2.)
    after = observe_executed_boundary(history, 1, [0., 1.], reference_k_per_nm=2.)
    assert before.integrated_axial_flux_fraction == 2.
    assert after.integrated_axial_flux_fraction == .5
    with pytest.raises(ValueError, match="index"):
        observe_executed_boundary(history, .5, [0., 1.], reference_k_per_nm=2.)
    del history["chart_derivatives"]
    with pytest.raises(ValueError, match="historical"):
        observe_executed_boundary(history, 0, [0., 1.], reference_k_per_nm=2.)
