"""Independent first-order ODE and Airy checks; not source/image acceptance."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import solve
from scipy.special import airy

from temsim.physics.magnus_scattering import GAUSS_FRACTIONS, cf4_slab
from temsim.physics.scattering_load import outgoing_load
from temsim.physics.liouville_wave import coordinate_terms, PhysicalCoordinateLoad, derivative_jump


def _operators(z):
    residual = np.array([[.2*np.sin(z), .3+.1*z], [.3+.1*z, -.4*z*z]])
    connection = np.array([[.1*z, .2j*np.cos(z)], [-.2j*np.cos(z), -.2*z]])
    return residual, connection


def _ode_transfer():
    def derivative(z, flat):
        h, g = _operators(z)
        generator = np.block([[-1j*g, np.eye(2)], [-4*np.eye(2)-h, -1j*g]])
        return (generator@flat.reshape(4, 4)).ravel()
    result = solve_ivp(derivative, (0., 1.), np.eye(4, dtype=complex).ravel(),
        method="DOP853", rtol=2e-13, atol=2e-14)
    assert result.success
    return result.y[:, -1].reshape(4, 4)


def test_cf4_complex_field_converges_against_independent_covariant_ode():
    transfer = _ode_transfer()
    boundary = np.vstack((np.eye(2), 2j*np.eye(2)))
    left = solve(transfer, boundary)
    expected_y = left[2:]@np.linalg.inv(left[:2])
    initial = np.array([1., .2j])
    expected_exit = solve(left[:2], initial)
    errors = []
    for intervals in (8, 16, 32):
        ops = []
        for j in range(intervals):
            first, second = (_operators((j+f)/intervals) for f in GAUSS_FRACTIONS)
            op, _ = cf4_slab(first, second, 1/intervals, 2.)
            full = np.block([[op[0], op[1]], [op[2], op[3]]])
            np.testing.assert_allclose(full.conj().T@full, np.eye(4), atol=3e-13)
            ops.append(op)
        load = outgoing_load(ops, 4*np.eye(2), 2.)
        field, derivative = load.propagate(initial)
        current = np.imag(np.sum(field.conj()*derivative, axis=1))
        np.testing.assert_allclose(current, current[0], rtol=1e-12)
        errors.append(max(np.linalg.norm(load.input_admittance-expected_y),
                          np.linalg.norm(field[-1]-expected_exit)))
    assert errors[-1] < 2e-8
    assert all(a/b > 14 for a, b in zip(errors, errors[1:]))


def test_cf4_action_nodes_and_jump_match_airy_ramp():
    # The ramp is linear in q=k^2, so z(s) has an analytic inverse.
    kref, q0, q1, length = 2., 4., 25., 10.
    slope = (q1-q0)/length
    def fundamental(q):
        a, ap, b, bp = airy(-q/slope**(2/3))
        return np.array([[a, b], [-slope**(1/3)*ap, -slope**(1/3)*bp]])
    transfer = fundamental(q1)@np.linalg.inv(fundamental(q0))
    expected_left = solve(transfer, np.array([1., 5j]))
    expected = np.array([expected_left[1]/expected_left[0], 1/expected_left[0]])
    errors = []
    for intervals in (32, 64, 128):
        ops, alphas, logs = [], [], []
        alpha, log, _ = coordinate_terms(q0, slope, 0., kref)
        alphas.append(alpha); logs.append(log)
        for a, b in zip(np.linspace(q0, q1, intervals+1)[:-1], np.linspace(q0, q1, intervals+1)[1:]):
            ds = 2*(b**1.5-a**1.5)/(3*slope*kref)
            samples = []
            for f in GAUSS_FRACTIONS:
                q = (a**1.5+f*(b**1.5-a**1.5))**(2/3)
                alpha, _, correction = coordinate_terms(q, slope, 0., kref)
                samples.append((np.array([[correction/alpha**2]]), np.zeros((1, 1))))
            op, _ = cf4_slab(*samples, ds, kref)
            ops.append(op)
            alpha, log, _ = coordinate_terms(b, slope, 0., kref)
            alphas.append(alpha); logs.append(log)
        ops.append(derivative_jump(logs[-1]/alphas[-1], kref, 1))
        alphas.append(alphas[-1]); logs.append(0.)
        load = PhysicalCoordinateLoad(outgoing_load(ops, np.array([[kref*kref]]), kref),
            np.asarray(alphas), np.asarray(logs))
        field, _ = load.propagate(np.array([1.+0j]))
        errors.append(np.max(abs(np.array([load.input_admittance[0, 0], field[-1, 0]])-expected)))
    assert errors[-1] < 1e-7
    assert all(a/b > 12 for a, b in zip(errors, errors[1:]))


def test_cf4_honours_cancellation_and_explicit_numerical_selection():
    from temsim.physics.radial_gun_wave import RadialGunNumerics
    with pytest.raises(InterruptedError):
        cf4_slab(_operators(0.), _operators(1.), 1., 2., cancelled=lambda: True)
    assert RadialGunNumerics().axial_integrator == "midpoint"
    assert RadialGunNumerics(axial_integrator="cf4").validate()
    with pytest.raises(ValueError, match="integrator"):
        RadialGunNumerics(axial_integrator="unvalidated").validate()
