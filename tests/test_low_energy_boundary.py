"""Independent boundary-operator checks, NOT full gun or image acceptance."""
import numpy as np
import pytest
from scipy.constants import e, hbar, m_e
from scipy.integrate import solve_ivp

from temsim.physics.low_energy_boundary import propagate_low_energy_boundary as solve


def k(energy):
    return np.sqrt(2*m_e*energy*e)/hbar


def test_uniform_medium_exact_large_angle_and_evanescent_branches():
    energy, length = .3, 2e-9
    transverse = k(energy)*np.array([0., .3, .95, 1., 1.2, 2.])
    result = solve([energy], transverse, [length], energy)
    kz = np.sqrt((k(energy)**2-transverse**2).astype(complex))
    expected = np.exp(1j*kz*length)
    np.testing.assert_allclose(result.boundary_amplitude[-1], expected, atol=2e-14)
    np.testing.assert_allclose(result.boundary_derivative_per_m[-1], 1j*kz*expected, atol=1e-5)
    np.testing.assert_allclose(result.current_per_boundary_density_m_per_s[-1], hbar/m_e*kz.real,
                               atol=1e-9)
    assert result.boundary_amplitude[-1, -1] != 0  # not a clipped evanescent field
    assert result.current_per_boundary_density_m_per_s[-1, -1] == 0
    assert not result.boundary_amplitude.flags.writeable


def test_accelerating_step_has_reflection_and_conserves_current():
    energies = np.array([.3, 4.])
    widths = np.array([1e-9, 2e-9])
    result = solve(energies, np.array([0., .7*k(.3)]), widths, 4.)
    for j, transverse in enumerate([0., .7*k(.3)]):
        k1, k2 = np.sqrt(k(energies)**2-transverse**2)
        reflection = (k1-k2)/(k1+k2)
        transmission = 2*k1/(k1+k2)
        expected = transmission*np.exp(1j*(k1*widths[0]+k2*widths[1]))/(1+reflection*np.exp(2j*k1*widths[0]))
        assert result.boundary_amplitude[-1, j] == pytest.approx(expected, abs=2e-14)
    np.testing.assert_allclose(result.current_per_boundary_density_m_per_s,
        np.broadcast_to(result.current_per_boundary_density_m_per_s[0], (3, 2)), rtol=2e-14)


def test_evanescent_entrance_can_couple_to_propagating_exit_without_clipping():
    result = solve([.01, .3], np.array([k(.01)*1.5]), [1e-9, 2e-9], .3)
    assert result.evanescent_layers[:, 0].tolist() == [True, False]
    current = result.current_per_boundary_density_m_per_s[:, 0]
    assert current[0] > 0  # boundary current solved, not assigned zero from k_left
    np.testing.assert_allclose(current, current[0], rtol=2e-14)
    assert abs(result.boundary_amplitude[-1, 0]) > 0


def test_thick_forbidden_layer_does_not_overflow_or_restore_flux():
    result = solve([.01], np.array([k(.01)*2]), [1e-4], .3)
    assert result.log_transfer[-1, 0].real < -1000
    assert result.boundary_amplitude[-1, 0] == 0  # numerical underflow, log retained
    assert abs(result.current_per_boundary_density_m_per_s[0, 0]) < 1e-9


def test_finite_barrier_matches_analytic_tunnelling_probability():
    transverse = k(.1)
    k_forward, kappa = np.sqrt(k(.3)**2-transverse**2), np.sqrt(transverse**2-k(.01)**2)
    width = 1e-9
    result = solve([.3, .01], np.array([transverse]), [2e-9, width], .3)
    # Convert TOTAL prescribed entrance field and derivative to incoming and
    # reflected amplitudes. A raw |exit/entrance|^2 is not transmission.
    incident = (1+result.boundary_derivative_per_m[0, 0]/(1j*k_forward))/2
    reflected = 1-incident
    transmission = abs(result.boundary_amplitude[-1, 0]/incident)**2
    analytical = 1/(1+(k_forward*k_forward+kappa*kappa)**2/
                     (4*k_forward*k_forward*kappa*kappa)*np.sinh(kappa*width)**2)
    assert transmission == pytest.approx(analytical, rel=2e-14)
    assert transmission+abs(reflected/incident)**2 == pytest.approx(1., abs=2e-14)


def test_linear_accelerating_potential_converges_against_independent_ode():
    # Dimensionless z=t*length keeps the independent adaptive ODE well scaled.
    length, transverse = 3e-9, .8*k(.3)
    exit_k = np.sqrt(k(1.2)**2-transverse**2)
    def rhs(t, state):
        q2 = (k(.3+.9*t)**2-transverse**2)*length**2
        return [state[1], -q2*state[0]]
    reference = solve_ivp(rhs, (1., 0.), [1.+0j, 1j*exit_k*length],
                          method="DOP853", rtol=1e-12, atol=1e-13).y[:, -1]
    target = 1/reference[0]
    errors = []
    for n in (64, 128, 256):
        energies = .3+.9*(np.arange(n)+.5)/n
        result = solve(energies, np.array([transverse]), np.full(n, length/n), 1.2)
        errors.append(abs(result.boundary_amplitude[-1, 0]-target))
        current = result.current_per_boundary_density_m_per_s[:, 0]
        np.testing.assert_allclose(current, current[0], rtol=1e-12)
    assert errors[1] < errors[0]/3.5
    assert errors[2] < errors[1]/3.5
    assert errors[-1] < 2e-5


@pytest.mark.parametrize("arguments", [([], [0.], [], .3), ([.3], [0.], [0.], .3),
    ([300000.], [0.], [1e-9], .3), ([.3], [np.nan], [1e-9], .3)])
def test_invalid_physical_inputs_rejected(arguments):
    with pytest.raises(ValueError):
        solve(*arguments)


def test_cancellation_and_memory_limits():
    with pytest.raises(InterruptedError):
        solve([.3], [0.], [1e-9], .3, cancelled=lambda: True)
    with pytest.raises(MemoryError):
        solve([.3], [0.], [1e-9], .3, maximum_working_bytes=1)
