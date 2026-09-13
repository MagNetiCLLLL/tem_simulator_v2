"""Independent finite-basis references, not a complete tip-to-image test."""
import numpy as np
import pytest
from scipy.linalg import expm

from temsim.physics.galerkin_potential import potential_action


def fourier_basis(shape, quadrature_shape):
    # Explicit exponentials, no FFT implementation shared with the operator.
    coordinates = [np.arange(m)-m//2 for m in quadrature_shape]
    frequencies = [np.arange(n)-n//2 for n in shape]
    yy, xx = np.meshgrid(*coordinates, indexing="ij")
    ky, kx = np.meshgrid(*frequencies, indexing="ij")
    return np.exp(2j*np.pi*(yy.ravel()[:, None]*ky.ravel()/quadrature_shape[0]
                          + xx.ravel()[:, None]*kx.ravel()/quadrature_shape[1])) / np.sqrt(np.prod(quadrature_shape))


@pytest.mark.parametrize("shape,quadrature", [((4, 6), (8, 12)), ((3, 5), (7, 11))])
def test_matches_independent_dense_hermitian_exponential(shape, quadrature):
    rng = np.random.default_rng(9301)
    a = rng.normal(size=shape)+1j*rng.normal(size=shape)
    a /= np.linalg.norm(a)
    phase = rng.uniform(-3., 5., quadrature)
    embedding = fourier_basis(shape, quadrature)
    h = embedding.conj().T @ (phase.ravel()[:, None]*embedding)
    np.testing.assert_allclose(h, h.conj().T, atol=2e-15)
    basis = fourier_basis(shape, shape)
    reference = basis @ expm(1j*h) @ basis.conj().T @ a.ravel()
    result, record = potential_action(a, phase)
    np.testing.assert_allclose(result.ravel(), reference, rtol=2e-13, atol=2e-14)
    assert abs(np.linalg.norm(result)-1) < 1e-13
    assert record["exponential_error_bound"] < 1e-13
    assert not record["renormalised"]
    restored, _ = potential_action(result, -phase)
    np.testing.assert_allclose(restored, a, atol=3e-14)


def test_constant_phase_keeps_global_phase_and_all_frequency_bins():
    a = np.random.default_rng(30).normal(size=(7, 8)).astype(complex)
    result, record = potential_action(a, np.full((14, 16), 71.2))
    np.testing.assert_allclose(result, a*np.exp(71.2j), atol=3e-15)
    assert record["applications"] == 0


def test_unresolved_scattering_does_not_wrap_into_negative_frequency():
    # Start in kx=+3 on N=8; V=cos(x) couples it to +2 and +4.
    # +4 is outside the basis, NOT its circular alias -4. With a small phase,
    # a seven-hop path to -4 is negligible, whereas circular multiplication
    # invents a first-order amplitude there.
    n, m, strength = 8, 16, .01
    basis = fourier_basis((n, n), (n, n))
    coefficients = np.zeros((n, n), complex)
    coefficients[n//2, -1] = 1.
    a = (basis @ coefficients.ravel()).reshape((n, n))
    fine_x = 2*np.pi*(np.arange(m)-m//2)/m
    phase = np.broadcast_to(strength*np.cos(fine_x), (m, m))
    result, _ = potential_action(a, phase)
    result_k = (basis.conj().T @ result.ravel()).reshape((n, n))
    circular_k = (basis.conj().T @ (a*np.exp(1j*phase[::2, ::2])).ravel()).reshape((n, n))
    assert abs(result_k[n//2, 0]) < 1e-14
    assert abs(circular_k[n//2, 0]) > .0049
    assert abs(result_k[n//2, -2]) > .0049


def test_periodic_analytic_solution_converges_with_basis():
    errors = []
    for n in (8, 16, 32):
        x = 2*np.pi*(np.arange(n)-n//2)/n
        xf = 2*np.pi*(np.arange(2*n)-n)/(2*n)
        a = np.ones((n, n), complex)/n
        phase = np.broadcast_to(1.5*np.cos(xf), (2*n, 2*n))
        result, _ = potential_action(a, phase)
        exact = a*np.exp(1.5j*np.cos(x))[None, :]
        errors.append(np.linalg.norm(result-exact))
    assert errors[1] < errors[0]*.01
    assert errors[2] < errors[1]*.01
    assert errors[2] < 1e-13


def test_deterministic_no_random_state_consumed():
    a = np.ones((4, 4), complex)/4
    phase = np.arange(64).reshape(8, 8)*.1
    before = np.random.get_state()
    first, _ = potential_action(a, phase)
    second, _ = potential_action(a, phase)
    np.testing.assert_array_equal(first, second)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])


def test_invalid_quadrature_cancel_and_budgets_fail_before_execution():
    a = np.ones((4, 4), complex)/4
    with pytest.raises(ValueError, match="twice"):
        potential_action(a, np.ones((4, 4)))
    with pytest.raises(ValueError, match="Real potential"):
        potential_action(a, np.ones((8, 8), complex))
    with pytest.raises(MemoryError):
        potential_action(a, np.ones((8, 8)), maximum_working_bytes=1)
    with pytest.raises(InterruptedError):
        potential_action(a, np.ones((8, 8)), cancelled=lambda: True)
    with pytest.raises(ValueError, match="application budget"):
        potential_action(a, np.arange(64).reshape(8, 8), maximum_applications=1)
