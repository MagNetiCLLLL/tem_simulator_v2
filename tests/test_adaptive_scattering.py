"""Independent ODE reference and strict adaptive failure/resource checks."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import solve

from temsim.physics.adaptive_scattering import adaptive_cf4, AxialRefinement, scattering_distance
from temsim.physics.magnus_scattering import cf4_slab, GAUSS_FRACTIONS
from temsim.physics.scattering_load import outgoing_load, hermitian_slab


def sample(f):
    q = np.array([[.8*np.sin(19*f), .2*np.cos(7*f)], [.2*np.cos(7*f), .4*f]])
    g = np.array([[.7*np.cos(13*f), .15j*np.sin(11*f)], [-.15j*np.sin(11*f), -.2*f]])
    return q, g


def test_adaptive_complex_field_repairs_unresolved_constant_step():
    def derivative(z, flat):
        q, g = sample(z)
        return (np.block([[-1j*g, np.eye(2)], [-4*np.eye(2)-q, -1j*g]])@flat.reshape(4, 4)).ravel()
    result = solve_ivp(derivative, (0., 1.), np.eye(4, dtype=complex).ravel(),
        method="DOP853", rtol=2e-13, atol=2e-14)
    assert result.success
    left = solve(result.y[:, -1].reshape(4, 4), np.vstack((np.eye(2), 2j*np.eye(2))))
    expected = solve(left[:2], np.array([1., .2j]))
    coarse, _ = cf4_slab(*(sample(f) for f in GAUSS_FRACTIONS), 1., 2.)
    coarse_load = outgoing_load([coarse], 4*np.eye(2), 2.)
    coarse_field, _ = coarse_load.propagate(np.array([1., .2j]))
    assert np.linalg.norm(coarse_field[-1]-expected) > .05
    errors = []
    for tolerance in (1e-4, 1e-8):
        op, record = adaptive_cf4(sample, 1., 2., AxialRefinement(tolerance=tolerance))
        load = outgoing_load([op], 4*np.eye(2), 2.)
        field, derivative = load.propagate(np.array([1., .2j]))
        errors.append(np.linalg.norm(field[-1]-expected))
        np.testing.assert_allclose(np.imag(np.sum(field.conj()*derivative, axis=1)),
                                   np.vdot(field[0], derivative[0]).imag, rtol=1e-11)
        assert record["summed_local_indicator"] <= tolerance
        assert record["accepted_substeps"] > 2
    assert errors[-1] < 1e-7
    assert errors[-1] < errors[0]/10


def test_constant_carrier_is_not_mistaken_for_transverse_error():
    residual = np.diag([1e-12, -1e-12])
    constant = lambda f: (residual, np.zeros((2, 2)))
    result, record = adaptive_cf4(constant, 6e14, 3000., AxialRefinement(tolerance=1e-9))
    expected, _ = hermitian_slab(residual, np.zeros((2, 2)), 6e14, 3000., carrier_k=3000.)
    assert scattering_distance(result, expected) < 1e-12
    assert record["accepted_substeps"] == 2


def test_refinement_budget_and_cancellation_do_not_return_unresolved_output():
    with pytest.raises(ValueError, match="exhausted"):
        adaptive_cf4(sample, 1., 2., AxialRefinement(tolerance=1e-8, maximum_evaluations=3))
    with pytest.raises(ValueError, match="depth"):
        adaptive_cf4(sample, 1., 2., AxialRefinement(tolerance=1e-12, maximum_depth=1))
    with pytest.raises(InterruptedError):
        adaptive_cf4(sample, 1., 2., cancelled=lambda: True)


@pytest.mark.parametrize("tolerance", (0., -1., float("nan"), True))
def test_invalid_refinement_settings(tolerance):
    with pytest.raises(ValueError):
        AxialRefinement(tolerance=tolerance).validate()
