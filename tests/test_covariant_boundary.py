"""Gauge and current checks of the internal kernel, not full-gun acceptance."""
import numpy as np
import pytest
from scipy.linalg import expm
from scipy.integrate import solve_ivp

from temsim.physics.coupled_low_energy import propagate_coupled_boundary
from temsim.physics.covariant_boundary import propagate_covariant_boundary as propagate


def test_zero_connection_matches_independent_electric_solver():
    q = np.array([[[1., .4j], [-.4j, -.2]], [[.6, .2], [.2, 1.1]]])*1e18
    end = np.eye(2)*1e18
    left = np.array([1., .3j])
    expected = propagate_coupled_boundary(q, [1e-9, 2e-9], end, left)
    actual = propagate(q, np.zeros_like(q), [1e-9, 2e-9], end, left)
    np.testing.assert_allclose(actual.boundary_amplitude, expected.boundary_amplitude, atol=4e-14)
    np.testing.assert_allclose(actual.boundary_covariant_derivative_per_m,
                               expected.boundary_derivative_per_m, atol=8e-5)
    assert not actual.boundary_amplitude.flags.writeable


def test_pure_longitudinal_gauge_changes_phase_not_flux():
    q = np.diag([1., -.2])*1e18
    g = np.eye(2)*3e8
    left = np.array([1., .5j])
    widths = np.array([1., 2.])*1e-9
    without = propagate([q, q], [g*0, g*0], widths, q, left)
    with_gauge = propagate([q, q], [g, g], widths, q, left)
    phase = np.exp(-1j*3e8*np.r_[0., np.cumsum(widths)])
    np.testing.assert_allclose(with_gauge.boundary_amplitude,
        without.boundary_amplitude*phase[:, None], atol=2e-14)
    np.testing.assert_allclose(with_gauge.current_per_boundary_density_m_per_s,
        without.current_per_boundary_density_m_per_s, rtol=2e-14)


def test_noncommuting_connection_matches_independent_first_order_exponential():
    length = 1e-9
    q = np.array([[1., .2j], [-.2j, -.1]])/length**2
    g = np.array([[.2, .1], [.1, -.4]])/length
    end = np.diag([1., .7])/length**2
    assert np.linalg.norm(q@g-g@q) > 0
    generator = np.block([[-1j*g*length, np.eye(2)], [-q*length**2, -1j*g*length]])
    transfer = expm(generator)
    load = 1j*np.diag(np.sqrt(np.diag(end)))*length
    left = np.array([1., .2j])
    derivative = np.linalg.solve(transfer[2:, 2:]-load@transfer[:2, 2:],
                                (load@transfer[:2, :2]-transfer[2:, :2])@left)
    expected = transfer@np.r_[left, derivative]
    actual = propagate([q], [g], [length], end, left)
    np.testing.assert_allclose(actual.boundary_amplitude[-1], expected[:2], atol=3e-14)
    np.testing.assert_allclose(actual.boundary_covariant_derivative_per_m[-1]*length,
                               expected[2:], atol=4e-14)
    assert actual.record["current_balance_residual"] < 1e-13


def test_basis_rotation_covariance_preserves_amplitude_and_current():
    q = np.array([[1., .3], [.3, .7]])*1e18
    g = np.array([[.1, .2j], [-.2j, -.3]])*1e9
    unitary = expm(1j*np.array([[.2, .6j], [-.6j, -.4]]))
    left = np.array([1., .1j])
    original = propagate([q], [g], [2e-9], np.eye(2)*1e18, left)
    rotated = propagate([unitary@q@unitary.conj().T], [unitary@g@unitary.conj().T],
                         [2e-9], np.eye(2)*1e18, unitary@left)
    np.testing.assert_allclose(rotated.boundary_amplitude, original.boundary_amplitude@unitary.T, atol=5e-14)
    np.testing.assert_allclose(rotated.current_per_boundary_density_m_per_s,
                               original.current_per_boundary_density_m_per_s, rtol=3e-14)


def test_long_evanescent_slab_does_not_form_an_overflowing_transfer_matrix():
    q, left = -np.eye(2)*1e18, np.ones(2)
    result = propagate([q], [np.eye(2)*1e8], [5e-7], np.eye(2)*1e18, left)
    assert np.all(np.isfinite(result.boundary_amplitude))
    assert np.all(abs(result.boundary_amplitude[-1]) > 0)
    assert np.max(abs(result.boundary_amplitude[-1])) < 1e-210
    assert result.record["exit_log_current_per_boundary_density_m_per_s"] < -900
    assert result.record["exit_current_representation"] == "log_below_normal_float_range"
    with pytest.raises(ValueError, match="underflow"):
        propagate([q], [np.eye(2)*1e8], [1e-5], np.eye(2)*1e18, left)


def test_budgets_cancel_and_nonhermitian_connection_fail_closed(monkeypatch):
    q = np.eye(2)*1e18
    g = np.zeros_like(q)
    with pytest.raises(ValueError, match="Hermitian"):
        propagate([q], [g+np.array([[0., 1e8], [0., 0.]])], [1e-9], q, np.ones(2))
    with pytest.raises(InterruptedError):
        propagate([q], [g], [1e-9], q, np.ones(2), cancelled=lambda: True)
    def forbidden(*args, **kwargs):
        pytest.fail("Budget check must precede matrix exponentiation")
    monkeypatch.setattr("temsim.physics.covariant_boundary.expm", forbidden)
    with pytest.raises(MemoryError):
        propagate([q], [g], [1e-9], q, np.ones(2), maximum_working_bytes=1)


def test_smooth_covariant_field_converges_against_independent_ode(record_property):
    length = 1e-9
    def q(t):
        return np.array([[1.+.3*t, .2j*t], [-.2j*t, .5+.2*t]])
    def g(t):
        return np.array([[.1*np.sin(t), .2*t], [.2*t, -.3*t]])
    values, vectors = np.linalg.eigh(q(1.))
    load = (vectors*(1j*np.sqrt(values)))@vectors.conj().T
    def rhs(t, flat):
        field, derivative = np.split(flat.reshape(4, 2), 2)
        return np.r_[derivative-1j*g(t)@field, -q(t)@field-1j*g(t)@derivative].ravel()
    ode = solve_ivp(rhs, (1., 0.), np.r_[np.eye(2), load].ravel(),
                    method="DOP853", rtol=1e-12, atol=1e-13)
    assert ode.success
    target = np.linalg.solve(ode.y[:, -1].reshape(4, 2)[:2], np.array([1., .2j]))
    errors = []
    for n in (16, 32, 64, 128):
        midpoints = (np.arange(n)+.5)/n
        result = propagate([q(t)/length**2 for t in midpoints],
            [g(t)/length for t in midpoints], np.full(n, length/n), q(1.)/length**2, [1., .2j])
        errors.append(float(np.linalg.norm(result.boundary_amplitude[-1]-target)))
    record_property("covariant_axial_errors_16_32_64_128", str(errors))
    assert all(b < a/3.5 for a, b in zip(errors, errors[1:]))
    assert errors[-1] < 1e-5
