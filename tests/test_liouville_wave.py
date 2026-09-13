"""Absolute physical-field checks against independent Airy ramp solutions."""
import numpy as np
from scipy.special import airy
from temsim.physics.liouville_wave import coordinate_terms, derivative_jump, PhysicalCoordinateLoad
from temsim.physics.scattering_load import hermitian_slab, outgoing_load


def ramp_solution(intervals):
    segments = ((4., 25., 10.), (25., 36., 15.))
    kref = 2.
    operators, alphas, logs = [], [], []
    transfer = np.eye(2)
    for q0, q1, length in segments:
        slope = (q1-q0)/length
        def fundamental(q):
            a, ap, b, bp = airy(-q/slope**(2/3))
            return np.array([[a, b], [-slope**(1/3)*ap, -slope**(1/3)*bp]])
        transfer = fundamental(q1)@np.linalg.solve(fundamental(q0), transfer)
        edges = np.linspace(q0, q1, intervals+1)
        alpha, log, _ = coordinate_terms(q0, slope, 0., kref)
        if logs:
            operators.append(derivative_jump(float((logs[-1]-log)/alpha), kref, 1))
        alphas.append(float(alpha)); logs.append(float(log))
        for a, b in zip(edges, edges[1:]):
            mid = (a+b)/2
            alpha, _, correction = coordinate_terms(mid, slope, 0., kref)
            ds = 2*(b**1.5-a**1.5)/(3*slope*kref)
            op, _ = hermitian_slab(np.array([[correction/alpha**2]]), np.zeros((1, 1)),
                                   ds, kref, carrier_k=kref)
            operators.append(op)
            alpha, log, _ = coordinate_terms(b, slope, 0., kref)
            alphas.append(float(alpha)); logs.append(float(log))
    operators.append(derivative_jump(logs[-1]/alphas[-1], kref, 1))
    alphas.append(alphas[-1]); logs.append(0.)
    load = PhysicalCoordinateLoad(outgoing_load(operators, np.array([[kref*kref]]), kref),
                                  np.asarray(alphas), np.asarray(logs))
    field, derivative = load.propagate(np.array([1.+0j]))
    incident = np.linalg.solve(transfer, np.array([1., 6j]))
    expected_y, expected_exit = incident[1]/incident[0], 1/incident[0]
    current = np.imag((field.conj()*derivative).sum(axis=1))
    np.testing.assert_allclose(current, current[0], rtol=1e-11, atol=1e-12)
    return max(abs(load.input_admittance[0, 0]-expected_y), abs(field[-1, 0]-expected_exit))


def test_action_coordinate_retains_reflection_and_derivative_knots():
    errors = [ramp_solution(n) for n in (256, 512, 1024)]
    assert errors[-1] < 2e-6
    assert all(a/b > 3.8 for a, b in zip(errors, errors[1:]))


def test_derivative_jump_is_lossless_and_not_transparent():
    blocks = derivative_jump(.7, 2., 3)
    matrix = np.block([[blocks[0], blocks[1]], [blocks[2], blocks[3]]])
    np.testing.assert_allclose(matrix.conj().T@matrix, np.eye(6), atol=1e-14)
    assert np.linalg.norm(blocks[0]) > .1
