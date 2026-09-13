"""Independent operator comparisons, not physical source acceptance."""
import numpy as np
import pytest

from temsim.physics.scattering_load import covariant_carrier_slab
from temsim.physics.covariant_boundary import _slab


@pytest.mark.parametrize("size", [2, 8, 24])
@pytest.mark.parametrize("width", [.01, .7, -.13])
@pytest.mark.parametrize("coordinates", ["graph", "riccati"])
def test_graph_preserves_complete_constant_operator(size, width, coordinates):
    rng = np.random.default_rng(816)
    a = rng.normal(size=(size, size))+1j*rng.normal(size=(size, size))
    residual = (a+a.conj().T)/size
    a = rng.normal(size=(size, size))+1j*rng.normal(size=(size, size))
    connection = .2*(a+a.conj().T)/size
    actual, record = covariant_carrier_slab(residual, connection, width, 1.3, 2., current_coordinates=coordinates)
    expected, _ = _slab(4*np.eye(size)+residual, connection, width, 1.3, lambda: False)
    np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-12)
    assert record["slab_unitarity_residual"] < 1e-11
    assert record["current_graph_invariant_phase_defect"] < 1e-11


@pytest.mark.parametrize("coordinates", ["graph", "riccati"])
def test_graph_coordinates_preserve_tiny_transverse_phase_without_subtracting_large_carrier(coordinates):
    from scipy.linalg import expm
    carrier, width = 3000., 6e14
    residual = np.array([[1., .2], [.2, -1.]])*1e-12
    connection = np.array([[.3, .15j], [-.15j, -.2]])*1e-16
    blocks, _ = covariant_carrier_slab(residual, connection, width, carrier, carrier, current_coordinates=coordinates)
    expected = np.exp(1j*carrier*width)*expm(1j*width*(residual/(2*carrier)-connection))
    np.testing.assert_allclose(blocks[2], expected, atol=1e-11, rtol=1e-11)


def test_graph_solves_invariance_instead_of_only_whitening_a_wrong_subspace():
    from temsim.physics.carrier_subspaces import graph_current_subspaces
    basis = np.array([[1., 0.], [.1, 1.]])
    spaces, record = graph_current_subspaces(basis, np.array([1., -1.]), np.zeros((1, 1)),
                                            np.zeros((1, 1)), 2., .5)
    np.testing.assert_allclose(spaces[0], [[1.], [0.]], atol=1e-14)
    assert record["current_graph_newton_iterations"] > 0


def test_nonpropagating_graph_is_not_admitted_by_orthogonalisation():
    from temsim.physics.carrier_subspaces import graph_current_subspaces
    with pytest.raises(ValueError):
        graph_current_subspaces(np.eye(2), np.array([1., -1.]), np.array([[-2.]]),
                                np.zeros((1, 1)), 2., .5)
