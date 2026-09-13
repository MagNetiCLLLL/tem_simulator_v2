"""Independent complex ODE and reversibility checks; not gun acceptance."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import solve

from temsim.physics.magnus_sixth import quadrature, cf6_slab
from temsim.physics.scattering_load import compose, outgoing_load, hermitian_slab


def operators(z):
    return (np.array([[.9*np.sin(2*z), .3+.4j*z], [.3-.4j*z, -.6*z*z]]),
            np.array([[.2*z, .4j*np.cos(z)], [-.4j*np.cos(z), -.3*z]]))


def reference():
    def derivative(z, flat):
        h, g = operators(z)
        return (np.block([[-1j*g, np.eye(2)], [-4*np.eye(2)-h, -1j*g]])@flat.reshape(4, 4)).ravel()
    result = solve_ivp(derivative, (0., 2.), np.eye(4, dtype=complex).ravel(),
                       method="DOP853", rtol=3e-14, atol=3e-15)
    assert result.success
    return result.y[:, -1].reshape(4, 4)


def full(blocks):
    return np.block([[blocks[0], blocks[1]], [blocks[2], blocks[3]]])


def test_sixth_order_convergence_with_complex_backward_coupling():
    left = solve(reference(), np.vstack((np.eye(2), 2j*np.eye(2))))
    initial = np.array([1., .3j])
    expected = solve(left[:2], initial)
    admittance = left[2:]@np.linalg.inv(left[:2])
    errors = []
    for count in (4, 8, 16):
        ops = [cf6_slab([operators((i+f)*2/count) for f in quadrature()[0]], 2/count, 2.)[0]
               for i in range(count)]
        for op in ops:
            np.testing.assert_allclose(full(op).conj().T@full(op), np.eye(4), atol=2e-12)
        load = outgoing_load(ops, 4*np.eye(2), 2.)
        fields, derivatives = load.propagate(initial)
        currents = np.imag(np.sum(fields.conj()*derivatives, axis=1))
        np.testing.assert_allclose(currents, currents[0], rtol=3e-12)
        errors.append(max(np.linalg.norm(fields[-1]-expected), np.linalg.norm(load.input_admittance-admittance)))
    assert errors[-1] < 2e-8
    assert all(a/b > 50 for a, b in zip(errors, errors[1:])), errors


@pytest.mark.parametrize("connection", [0., .3])
@pytest.mark.parametrize("residual", [1., -4., -10.])
def test_constant_propagating_grazing_and_evanescent_channels(connection, residual):
    sample = (np.diag([residual, .5]), np.array([[0., connection], [connection, 0.]]))
    for width in (.03, -.03):
        actual, _ = cf6_slab([sample]*4, width, 2.)
        expected, _ = hermitian_slab(*sample, width, 2., carrier_k=2.)
        np.testing.assert_allclose(full(actual), full(expected), atol=2e-11)


def test_time_reversal_preserves_all_ports_and_absolute_phase():
    forward = cf6_slab([operators(f) for f in quadrature()[0]], 1., 2.)[0]
    backward = cf6_slab([operators(1-f) for f in quadrature()[0]], -1., 2.)[0]
    expected = np.block([[np.zeros((2, 2)), np.eye(2)], [np.eye(2), np.zeros((2, 2))]])
    np.testing.assert_allclose(full(compose(forward, backward)), expected, atol=2e-12)


def test_negative_evanescent_substep_matches_independent_matrix_exponential():
    from scipy.linalg import expm
    from temsim.physics.covariant_boundary import _slab
    q = np.array([[-3., .2j], [-.2j, 2.]])
    g = np.array([[.3, .2], [.2, -.7]])
    eye = np.eye(2)
    transform = np.block([[eye, eye], [2j*eye, -2j*eye]])
    generator = np.block([[-1j*g, eye], [-q, -1j*g]])
    expected = solve(transform, expm(-.8*generator)@transform)
    a, b, c, d = expected[:2, :2], expected[:2, 2:], expected[2:, :2], expected[2:, 2:]
    refl = -solve(d, c)
    expected_s = np.block([[refl, solve(d, eye)], [a+b@refl, b@solve(d, eye)]])
    actual, record = _slab(q, g, -.8, 2., lambda: False)
    np.testing.assert_allclose(full(actual), expected_s, atol=2e-12)
    assert 0 <= record["seed_log_growth_bound"] <= .5


def test_selection_cancellation_and_signed_weights():
    nodes, weights = quadrature()
    np.testing.assert_allclose(weights.sum(), 1., atol=1e-15)
    assert np.all((nodes > 0) & (nodes < 1)) and weights.sum(axis=1).min() < 0
    with pytest.raises(InterruptedError):
        cf6_slab([operators(0.)]*4, 1., 2., cancelled=lambda: True)


def test_local_diagnostic_uses_same_complex_ports_without_a_phase_fit():
    from scripts.audit_current_gun_integrators import compare_interval
    rows = compare_interval(lambda _: operators(0.), 1., 2., np.array([1., .2j]),
                            np.array([.1j, .3]), divisions=(1, 4))
    assert len(rows) == 4
    assert max(row["occupied_difference_from_finest_cf6"] for row in rows) < 1e-12
