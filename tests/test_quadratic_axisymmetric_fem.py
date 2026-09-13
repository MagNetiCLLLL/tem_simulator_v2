"""Complex-field convergence, not just a discrete conservation identity."""
import numpy as np
import pytest

from temsim.physics.quadratic_axisymmetric_fem import mesh, triangle_basis, volume
from temsim.physics.surface_wave import solve_driven_wave, KINETIC_NM2_PER_EV


def test_partition_and_linear_coordinates():
    for u, v in ((.1, .2), (.2, .7), (0., 0.)):
        n, dn = triangle_basis(u, v)
        assert n.sum() == pytest.approx(1)
        np.testing.assert_allclose(dn.sum(axis=0), 0., atol=1e-14)
        nodes = np.array(((0., 0.), (1., 0.), (0., 1.), (.5, 0.), (.5, .5), (0., .5)))
        np.testing.assert_allclose(n@nodes, (u, v), atol=1e-14)
        np.testing.assert_allclose(nodes.T@dn, np.eye(2), atol=1e-14)


def test_quadratic_complex_wave_converges_without_phase_fit():
    errors = []
    for count in (65, 129, 257):
        radial = (count-1)//4+1
        p, t, edges, z = mesh(np.linspace(0, 1., radial), np.zeros(radial), 3., count)
        edges["side"] = np.empty((0, 3), int)
        drive = np.zeros(len(p), complex)
        drive[np.arange(radial)*count] = 1.
        psi, flux = solve_driven_wave(p, t, edges, np.zeros(len(p)), .3, drive)
        # Input flux is k*pi*r_max^2 for an amplitude-one reservoir.
        k = np.sqrt(.3*KINETIC_NM2_PER_EV)
        exact = np.exp(1j*k*z)/np.sqrt(k*np.pi)
        errors.append(np.max(abs(psi.reshape(z.shape)-exact)))
        assert flux["balance_error"] < 1e-10
    assert errors[-1] < 1e-5
    # P2 has third-order field accuracy; phase-dispersion superconvergence
    # on a 1-D lattice is not a promise for this triangular 2-D mesh.
    assert errors[0]/errors[1] > 6
    assert errors[1]/errors[2] > 6


def test_curved_element_laplace_constant_and_volume():
    r = np.linspace(0, 4., 9)
    bottom = -.1*r*r
    p, t, _, _ = mesh(r, bottom, 3., 9)
    k, m, v = volume(p, t, np.ones(len(p))*7.)
    np.testing.assert_allclose(k@np.ones(len(p)), 0., atol=1e-11)
    np.testing.assert_allclose(v.toarray(), 7*m.toarray(), atol=1e-13)
    exact = np.pi*4**2*3 + .05*np.pi*4**4
    assert np.ones(len(p))@m@np.ones(len(p)) == pytest.approx(exact, rel=1e-12)


def test_even_node_counts_are_not_silently_reinterpreted():
    with pytest.raises(ValueError, match="odd"):
        mesh(np.linspace(0, 1., 8), np.zeros(8), 3., 9)
