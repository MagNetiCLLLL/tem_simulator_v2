"""Projected current fixtures, not physical-tip/full-image acceptance."""
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import roots_hermite

from temsim.physics.radial_current_projection import (
    hermite_root_weights, radial_current_projection, project_executed_boundaries)
from temsim.physics.radial_wave_observables import radial_mode_observables


def chart(**values):
    return dict(width_nm=2., curvature_per_nm=.3, reference_k_per_nm=5.,
        width_log_rate_per_nm=.2, curvature_prime_per_nm2=.4,
        quartic_phase_per_nm4=.02, quartic_prime_per_nm5=.03) | values


@pytest.mark.parametrize("order", [1, 5, 33, 129, 513])
def test_root_weights_preserve_tail_and_gaussian_integral(order):
    nodes, root = hermite_root_weights(order)
    expected_nodes, weights = roots_hermite(order)
    assert np.array_equal(nodes, expected_nodes)
    valid = weights > 1e-290
    np.testing.assert_allclose(root[valid]**2, weights[valid], rtol=3e-12, atol=0)
    assert np.sum(root**2) == pytest.approx(math.sqrt(math.pi), rel=3e-14)
    assert np.all(root > 0)
    if order == 513:
        assert np.any(weights == 0)


def test_gaussian_projection_absolute_flux_and_backflow_without_rescaling():
    x = np.linspace(-12, 12, 501)
    for factor in (2., -3., 0.):
        p = radial_current_projection(x, [1.], [1j*factor], **chart(
            width_log_rate_per_nm=0., curvature_prime_per_nm2=0., quartic_prime_per_nm5=0.))
        np.testing.assert_allclose(p.axial_flux_fraction_per_nm,
            factor*np.exp(-(x/2)**2)/(2*np.sqrt(np.pi)), rtol=4e-14, atol=1e-16)
        assert p.integrated_axial_flux_fraction == factor
        assert np.trapezoid(p.axial_flux_fraction_per_nm, x) == pytest.approx(factor, abs=1e-13)


def test_moving_chart_projection_matches_independent_real_space_quadrature():
    a = np.array([.7+.2j, -.3j, .1-.04j])
    d = np.array([.1+1.2j, .4+.03j, -.08+.06j])
    x = np.array([-3., -.5, 0., .5, 3.])
    p = radial_current_projection(x, a, d, **chart())
    expected = []
    for coordinate in x:
        def integrand(y):
            return radial_mode_observables([np.hypot(coordinate, y)], a, d,
                                          **chart()).axial_flux_fraction_per_nm2[0]
        expected.append(2*quad(integrand, 0, np.inf, epsabs=1e-11, epsrel=1e-11)[0])
    np.testing.assert_allclose(p.axial_flux_fraction_per_nm, expected, rtol=1e-11, atol=1e-12)


def test_high_order_and_chunked_projection_preserve_all_mode_current():
    rng = np.random.default_rng(917)
    a = rng.normal(size=64)+1j*rng.normal(size=64)
    a /= np.linalg.norm(a)
    d = (1.2+.7j)*a
    x = np.linspace(-40, 40, 1601)
    parameters = chart(width_log_rate_per_nm=0., curvature_prime_per_nm2=0., quartic_prime_per_nm5=0.)
    p = radial_current_projection(x, a, d, **parameters)
    q = radial_current_projection(x, a, d, maximum_working_bytes=256*129*2+32*len(x), **parameters)
    np.testing.assert_array_equal(p.axial_flux_fraction_per_nm, q.axial_flux_fraction_per_nm)
    assert np.trapezoid(p.axial_flux_fraction_per_nm, x) == pytest.approx(.7, rel=2e-12)
    crop = radial_current_projection([-1., 0., 1.], a, d, **parameters)
    assert crop.integrated_axial_flux_fraction == p.integrated_axial_flux_fraction
    assert crop.axial_flux_fraction_per_nm[1] == p.axial_flux_fraction_per_nm[800]


def test_invalid_projection_is_explicit():
    with pytest.raises(ValueError, match="increasing"):
        radial_current_projection([1., 0.], [1.], [1j], **chart())
    with pytest.raises(MemoryError):
        radial_current_projection([0.], [1.], [1j], maximum_working_bytes=1, **chart())


def history():
    return dict(z_nm=np.array([0., 10., 10., 20.]),
        coefficients=np.array([[1.], [1.], [.5], [.5]], complex),
        covariant_derivatives=np.array([[1j], [1j], [.5j], [.5j]]),
        width_curvature=np.tile([2., 0.], (4, 1)), quartic_phase_per_nm4=np.zeros(4),
        chart_derivatives=np.zeros((4, 3)))


def test_executed_stop_sides_and_original_absolute_weights_survive_display_sampling():
    state = history()
    saved = {key: value.copy() for key, value in state.items()}
    result = project_executed_boundaries(state, (0, 1, 2, 3), [-2., 0., 2.],
        reference_k_per_nm=5., reference_current_a=1e-7, source_mixture_weight=.2)
    np.testing.assert_array_equal(result.z_nm, [0., 10., 10., 20.])
    np.testing.assert_allclose(result.total_axial_current_a, [2e-8, 2e-8, .5e-8, .5e-8], rtol=1e-15)
    np.testing.assert_array_equal(result.axial_current_a_per_nm[2], result.axial_current_a_per_nm[1]*.25)
    sparse = project_executed_boundaries(state, (0, 3), [-2., 0., 2.],
        reference_k_per_nm=5., reference_current_a=1e-7, source_mixture_weight=.2)
    np.testing.assert_array_equal(sparse.axial_current_a_per_nm, result.axial_current_a_per_nm[[0, 3]])
    for key in saved:
        np.testing.assert_array_equal(saved[key], state[key])


def test_cancel_missing_historical_rates_and_budget_do_not_delete_cache():
    options = dict(reference_k_per_nm=5., reference_current_a=1e-7, source_mixture_weight=.2)
    with pytest.raises(InterruptedError):
        project_executed_boundaries(history(), (0,), [0.], cancelled=lambda: True, **options)
    state = history()
    del state["chart_derivatives"]
    with pytest.raises(ValueError, match="no exact chart"):
        project_executed_boundaries(state, (0,), [0.], **options)
    with pytest.raises(MemoryError):
        project_executed_boundaries(history(), (0,), [0.], maximum_working_bytes=1, **options)
