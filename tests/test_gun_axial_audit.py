"""Local observers retain complex two-way phase and do not certify a source."""
import numpy as np
import pytest

from scripts.audit_gun_axial_steps import refinement_comparison, occupied_difference


def test_constant_step_has_no_fictitious_refinement_error():
    q = np.diag([-.1, -.2])
    g = np.array([[0., .07j], [-.07j, .02]])
    rows, _ = refinement_comparison(lambda fraction: (q, g), 2., 3.)
    assert max(row["difference_from_cf4_sixteen"] for row in rows) < 1e-12


def test_variable_step_converges_to_independent_direct_ode():
    # Exact transfer from an independent adaptive real-z IVP. This is only a
    # tiny mathematical fixture, not a replacement beam/production solver.
    from scipy.integrate import solve_ivp
    from temsim.physics.adaptive_scattering import scattering_distance
    from temsim.physics.covariant_boundary import _solve
    k, length = 2., 1.3
    def sample(fraction):
        return np.array([[-.2-.3*fraction**2]]), np.array([[.15*fraction]])
    chart = np.array([[1, 1], [1j, -1j]])
    inverse = np.linalg.inv(chart)
    def derivative(z, flat):
        q, g = sample(z/length)
        generator = inverse@np.block([[-1j*g, np.array([[k]])],
                                      [-(k*k*np.eye(1)+q)/k, -1j*g]])@chart
        return (generator@flat.reshape(2, 2)).ravel()
    solution = solve_ivp(derivative, (0., length), np.eye(2, dtype=complex).ravel(),
                         method="DOP853", atol=1e-13, rtol=1e-13)
    assert solution.success
    transfer = solution.y[:, -1].reshape(2, 2)
    a, b, c, d = (transfer[0:1, 0:1], transfer[0:1, 1:2], transfer[1:2, 0:1], transfer[1:2, 1:2])
    reflection = -_solve(d, c)
    exact = reflection, _solve(d, np.eye(1)), a+b@reflection, b@_solve(d, np.eye(1))
    _, outputs = refinement_comparison(sample, length, k)
    errors = [scattering_distance(outputs["cf4", divisions], exact) for divisions in (1, 2, 4, 8, 16)]
    assert all(a > b*8 for a, b in zip(errors[:-1], errors[1:]))
    assert errors[-1] < 1e-7
    assert scattering_distance(outputs["midpoint", 1], exact) > errors[-1]*1000


def test_occupied_error_keeps_reflection_and_complex_phase():
    one, zero = np.eye(1), np.zeros((1, 1))
    first = zero, one, one, zero
    phased = zero, 1j*one, 1j*one, zero
    assert occupied_difference(first, phased, np.array([1., 0.])) == pytest.approx(np.sqrt(2))
    reflected = one, zero, zero, one
    assert occupied_difference(first, reflected, np.array([0., 1.])) == pytest.approx(np.sqrt(2))
    with pytest.raises(ValueError, match="finite nonzero"):
        occupied_difference(first, phased, np.zeros(2))
