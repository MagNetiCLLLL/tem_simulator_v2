"""Independent analytic, energy and step-convergence checks for a static pusher."""
import numpy as np
import pytest
from scipy.constants import m_e, c, e
from scipy.integrate import solve_ivp

from temsim.physics.relativistic_lorentz import RelativisticPhaseSpace, momentum_from_kinetic_energy_ev
from temsim.physics.static_energy_lorentz import static_energy_step


class Fields:
    def __init__(self, voltage_gradient=1e4, curvature=0., magnetic=(0., 0., 0.)):
        self.gradient, self.curvature, self.magnetic = voltage_gradient, curvature, magnetic

    def potential_v_at_global_positions(self, p):
        return self.gradient*p[..., 2]+self.curvature*np.sum(p*p, axis=-1)

    def field_at_global_positions_v_per_m(self, p):
        field = -2*self.curvature*p
        field[..., 2] -= self.gradient
        return field

    def field_at_global_positions_t(self, p):
        return np.broadcast_to(self.magnetic, p.shape)


def energy(p):
    s = np.sum((p/(m_e*c))**2, axis=-1)
    return (m_e*c*c/e)*s/(np.sqrt(1+s)+1)


def test_uniform_extraction_impulse_and_work():
    fields = Fields()
    p0 = momentum_from_kinetic_energy_ev(np.array([.2]), np.array([[0., 0., 1.]]))
    start = RelativisticPhaseSpace(np.zeros((1, 3)), p0)
    dt = 1e-13
    end = static_energy_step(start, dt, fields, fields)
    np.testing.assert_allclose(end.momentum_kg_m_per_s-p0, [[0, 0, e*dt*fields.gradient]], rtol=1e-9, atol=1e-35)
    np.testing.assert_allclose(energy(end.momentum_kg_m_per_s)-fields.potential_v_at_global_positions(end.position_m), [.2], atol=1e-10)


def test_magnetic_field_does_no_work():
    fields = Fields(0., magnetic=(0, 0, 1))
    start = RelativisticPhaseSpace(np.zeros((1, 3)), momentum_from_kinetic_energy_ev([1000], [[1., 0, .1]]))
    result = static_energy_step(start, 1e-13, fields, fields)
    np.testing.assert_allclose(energy(result.momentum_kg_m_per_s), [1000], atol=1e-8)
    assert result.momentum_kg_m_per_s[0, 1] > 0


def test_smooth_combined_field_converges_to_independent_dop853():
    field = Fields(2e5, 2e7, (0, 0, .1))
    start = RelativisticPhaseSpace(np.array([[1e-5, 2e-5, 0.]]), momentum_from_kinetic_energy_ev([1000.], [[.1, .1, 1.]]))
    duration = 2e-12
    # Dimensionless momentum/position improve the independent IVP scaling.
    xs, ps = 1e-4, m_e*c
    initial = np.r_[start.position_m[0]/xs, start.momentum_kg_m_per_s[0]/ps]
    def rhs(_t, y):
        x, p = y[:3]*xs, y[3:]*ps
        velocity = p/(m_e*np.sqrt(1+np.sum((p/ps)**2)))
        force = -e*(field.field_at_global_positions_v_per_m(x)+np.cross(velocity, field.magnetic))
        return np.r_[velocity/xs, force/ps]
    reference = solve_ivp(rhs, (0, duration), initial, method="DOP853", rtol=1e-12, atol=1e-14).y[:, -1]
    errors = []
    h0 = energy(start.momentum_kg_m_per_s)-field.potential_v_at_global_positions(start.position_m)
    for count in (8, 16, 32):
        result = start
        for _ in range(count):
            result = static_energy_step(result, duration/count, field, field)
        errors.append(np.linalg.norm(np.r_[result.position_m[0]/xs, result.momentum_kg_m_per_s[0]/ps]-reference))
        np.testing.assert_allclose(energy(result.momentum_kg_m_per_s)-field.potential_v_at_global_positions(result.position_m), h0, atol=1e-7)
    assert errors[-1] < 1e-6
    assert errors[0]/errors[1] > 3.5 and errors[1]/errors[2] > 3.5


def test_failure_is_explicit():
    f = Fields()
    p = RelativisticPhaseSpace(np.zeros((1, 3)), momentum_from_kinetic_energy_ev([.2], [[0, 0, 1]]))
    with pytest.raises(ValueError):
        static_energy_step(p, 0, f, f)
    with pytest.raises(ValueError, match="did not converge"):
        static_energy_step(p, 1e-13, f, f, maximum_iterations=0)
