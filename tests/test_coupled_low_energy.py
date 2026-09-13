"""Two-way operator references, not full tip-to-gun or image acceptance."""
import numpy as np
import pytest
from scipy.constants import e, hbar, m_e
from scipy.linalg import expm
from scipy.integrate import solve_ivp

from temsim.physics.coupled_low_energy import propagate_coupled_boundary as solve
from temsim.physics.low_energy_boundary import propagate_low_energy_boundary


def test_exact_uniform_mixed_propagating_grazing_and_evanescent_modes():
    q = np.diag([1.e18, 0., -.49e18])
    left = np.array([.6+.1j, -.3j, .8j])
    widths = [1e-9, 2e-9]
    result = solve([q, q], widths, q, left)
    z = np.r_[0., np.cumsum(widths)]
    k = np.sqrt(np.diag(q).astype(complex))
    expected = left*np.exp(1j*z[:, None]*k)
    np.testing.assert_allclose(result.boundary_amplitude, expected, atol=2e-14)
    np.testing.assert_allclose(result.boundary_derivative_per_m, 1j*k*expected, atol=2e-5)
    np.testing.assert_allclose(result.current_per_boundary_density_m_per_s,
        np.sum(abs(left)**2*k.real)*hbar/m_e, atol=1e-9)
    assert result.boundary_amplitude[-1, -1] != 0
    assert not result.boundary_amplitude.flags.writeable
    assert result.record["gun_source_admission"] == "NOT_A_GUN_CHECKPOINT"


def test_noncommuting_coupled_layers_match_independent_full_transfer_exponential():
    # Independent first-order [f, L*f'] transfer reference is well conditioned
    # for these short layers. It is not the production scattering recursion.
    scale = 1e-9
    q1 = np.array([[1.3, .4j], [-.4j, -.2]])/scale**2
    q2 = np.array([[2., .7], [.7, 1.]])/scale**2
    q3 = np.array([[1.1, -.1j], [.1j, .9]])/scale**2
    assert np.linalg.norm(q1@q2-q2@q1) > 0
    widths = np.array([.8, .6, .5])*scale
    left = np.array([.7+.2j, .3-.1j])
    end = np.diag([1.5, .7])/scale**2
    propagators = []
    for q, d in zip([q1, q2, q3], widths):
        generator = np.block([[np.zeros((2, 2)), np.eye(2)], [-q*scale**2, np.zeros((2, 2))]])
        propagators.append(expm(generator*d/scale))
    p = propagators[2]@propagators[1]@propagators[0]
    load = 1j*np.diag(np.sqrt(np.diag(end)))*scale
    derivative = np.linalg.solve(p[2:, 2:]-load@p[:2, 2:], (load@p[:2, :2]-p[2:, :2])@left)
    state = np.r_[left, derivative]
    reference = [state]
    for operator in propagators:
        state = operator@state
        reference.append(state)
    reference = np.array(reference)
    result = solve([q1, q2, q3], widths, end, left)
    np.testing.assert_allclose(result.boundary_amplitude, reference[:, :2], atol=3e-14)
    np.testing.assert_allclose(result.boundary_derivative_per_m*scale, reference[:, 2:], atol=3e-14)
    assert result.record["current_balance_residual"] < 1e-13
    # The artificial reference scale cannot affect the physical answer.
    for kappa in (3e8, 4e9):
        other = solve([q1, q2, q3], widths, end, left, reference_wave_number_per_m=kappa)
        np.testing.assert_allclose(other.boundary_amplitude, result.boundary_amplitude, atol=3e-14)


def test_diagonal_limit_matches_old_axial_solver_with_evanescent_coupling():
    energies, widths = [.01, .3], [1e-9, 2e-9]
    transverse = np.sqrt(2*m_e*e*.01)/hbar*np.array([.2, 1.5])
    q = [np.diag(2*m_e*e*energy/hbar**2-transverse**2) for energy in energies]
    end = np.diag(2*m_e*e*.3/hbar**2-transverse**2)
    for channel in range(2):
        left = np.eye(2)[channel]
        a = solve(q, widths, end, left)
        b = propagate_low_energy_boundary(energies, transverse, widths, .3)
        np.testing.assert_allclose(a.boundary_amplitude[:, channel], b.boundary_amplitude[:, channel], atol=2e-14)
        np.testing.assert_allclose(a.current_per_boundary_density_m_per_s,
                                  b.current_per_boundary_density_m_per_s[:, channel], atol=2e-9)
        assert a.current_per_boundary_density_m_per_s[0] > 0


def test_smooth_transverse_coupling_converges_against_independent_adaptive_ode(record_property):
    length = 2e-9
    def q(t):
        return np.array([[1.2+t, .3*np.cos(2*t)], [.3*np.cos(2*t), -.15+1.1*t]])
    values, vectors = np.linalg.eigh(q(1.))
    load = (vectors*(1j*np.sqrt(values)))@vectors.T
    terminal = np.r_[np.eye(2), load].astype(complex)
    def rhs(t, f):
        fields = f.reshape(4, 2)
        return np.r_[fields[2:], -q(t)@fields[:2]].ravel()
    reference = solve_ivp(rhs, (1., 0.), terminal.ravel(), method="DOP853",
                          rtol=1e-12, atol=1e-13)
    assert reference.success
    at_left = reference.y[:, -1].reshape(4, 2)
    left = np.array([1., .2j])
    target = np.linalg.solve(at_left[:2], left)
    errors = []
    for n in (16, 32, 64, 128):
        result = solve([q(t)/length**2 for t in (np.arange(n)+.5)/n],
                       np.full(n, length/n), q(1.)/length**2, left)
        errors.append(float(np.linalg.norm(result.boundary_amplitude[-1]-target)))
        assert result.record["current_balance_residual"] < 1e-12
    record_property("axial_errors_16_32_64_128", str(errors))
    assert all(b < a/3.5 for a, b in zip(errors, errors[1:]))
    assert errors[-1] < 1e-5


def test_no_coupling_is_silently_dropped_and_zero_field_stays_zero():
    q = np.array([[1., .6], [.6, 1.4]])*1e18
    left = np.array([1., 0.])
    result = solve([q], [2e-9], np.eye(2)*1e18, left)
    assert abs(result.boundary_amplitude[-1, 1]) > .1
    zero = solve([q], [2e-9], np.eye(2)*1e18, np.zeros(2))
    assert not np.any(zero.boundary_amplitude)
    assert not np.any(zero.current_per_boundary_density_m_per_s)


def test_budgets_cancellation_and_nonhermitian_fields_fail_closed(monkeypatch):
    q, left = np.eye(2)*1e18, np.ones(2)
    with pytest.raises(ValueError, match="Hermitian"):
        solve([q+np.array([[0., 1e16], [0., 0.]])], [1e-9], q, left)
    with pytest.raises(ValueError, match="real metres"):
        solve([q], [-1e-9], q, left)
    with pytest.raises(ValueError, match="numerical budget"):
        solve([q], [1e-9], q, left, maximum_channels=1)
    with pytest.raises(InterruptedError):
        solve([q], [1e-9], q, left, cancelled=lambda: True)
    def forbidden(*args, **kwargs):
        pytest.fail("Budget must be checked before eigendecomposition")
    monkeypatch.setattr("temsim.physics.coupled_low_energy.eigh", forbidden)
    with pytest.raises(MemoryError):
        solve([q], [1e-9], q, left, maximum_working_bytes=1)


def test_unrepresentable_coupled_transmission_is_not_silently_zero_filled():
    with pytest.raises(ValueError, match="log-scaled"):
        solve([-np.eye(2)*1e18], [1e-5], np.eye(2)*1e18, np.ones(2))


def test_boundary_problem_is_detached_from_live_input_arrays():
    q = np.array([[[1., .3], [.3, .7]]])*1e18
    widths, end, left = np.array([1e-9]), np.eye(2)*1e18, np.ones(2, complex)
    expected = solve(q, widths, end, left)
    calls = 0
    def edit_after_capture():
        nonlocal calls
        calls += 1
        if calls == 2:
            q[:] *= 2
            widths[:] *= 3
            end[:] *= 4
            left[:] *= 5
        return False
    actual = solve(q, widths, end, left, cancelled=edit_after_capture)
    np.testing.assert_array_equal(actual.boundary_amplitude, expected.boundary_amplitude)
    np.testing.assert_array_equal(actual.boundary_derivative_per_m, expected.boundary_derivative_per_m)
