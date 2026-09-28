"""Common batched/scalar update and independent work/impulse checks."""
import numpy as np
import pytest

from temsim.magnetic_test_particle import _Sampler, _discrete_gradient_step
from temsim.physics.relativistic_lorentz import (
    RelativisticPhaseSpace, momentum_from_kinetic_energy_ev,
    kinetic_energy_ev_from_momentum, ELECTRON_MASS_KG as M,
    SPEED_OF_LIGHT_M_PER_S as C, ELEMENTARY_CHARGE_C as E,
)
from temsim.physics.static_energy_lorentz import static_energy_step


class Fields:
    bounds_m = np.array(((-1., -1., -1.), (1., 1., 1.)))
    diagnostic_bounds_m = bounds_m
    def __init__(self, gradient, curvature, magnetic):
        self.gradient, self.curvature, self.magnetic = gradient, curvature, np.array(magnetic)

    def potential_v_at_global_positions(self, p):
        return self.gradient*p[..., 2]+self.curvature*np.sum(p*p, axis=-1)

    def field_at_global_positions_v_per_m(self, p):
        result = -2.*self.curvature*p.copy()
        result[..., 2] -= self.gradient
        return result

    def field_at_global_positions_t(self, p):
        return np.broadcast_to(self.magnetic, p.shape)

    def diagnostic_fields_at_global_position(self, p):
        return (self.magnetic, self.field_at_global_positions_v_per_m(p),
                self.potential_v_at_global_positions(p))


@pytest.mark.parametrize("gradient,curvature,magnetic", [
    (2e5, 0., (0., 0., 0.)), (0., 0., (0., 0., .2)),
    (1e5, 1e7, (.01, -.02, .2)),
])
def test_batched_gun_and_scalar_diagnostic_use_identical_step(gradient, curvature, magnetic):
    field = Fields(gradient, curvature, magnetic)
    x = np.array([[1e-5, -2e-5, 3e-5], [-1e-5, 1e-5, 2e-5]])
    p = momentum_from_kinetic_energy_ev(np.array([1000., 1700.]), np.array([[.1, -.2, 1.], [-.2, .1, 1.]]))
    phase = RelativisticPhaseSpace(x, p)
    duration = 1e-14
    batch = static_energy_step(phase, duration, field, field)
    sampler = _Sampler(field)
    for i in range(len(x)):
        scalar = _discrete_gradient_step(sampler, x[i], p[i]/(M*C), sampler.fields(x[i]), duration, lambda: False)
        np.testing.assert_allclose(scalar[0], batch.position_m[i], rtol=1e-14, atol=1e-21)
        np.testing.assert_allclose(scalar[1]*(M*C), batch.momentum_kg_m_per_s[i], rtol=1e-13, atol=1e-35)
    initial_invariant = kinetic_energy_ev_from_momentum(p)-field.potential_v_at_global_positions(x)
    final_invariant = kinetic_energy_ev_from_momentum(batch.momentum_kg_m_per_s)-field.potential_v_at_global_positions(batch.position_m)
    np.testing.assert_allclose(final_invariant, initial_invariant, rtol=0., atol=1e-8)


def test_axial_turning_midpoint_does_not_cancel_electric_force():
    field = Fields(-1e4, 0., (0., 0., 0.))
    momentum = momentum_from_kinetic_energy_ev(np.array([.3]), np.array([[0., 0., 1.]]))
    phase = RelativisticPhaseSpace(np.zeros((1, 3)), momentum)
    duration = 2.*momentum[0, 2]/(E*1e4)
    actual = static_energy_step(phase, duration, field, field)
    np.testing.assert_allclose(actual.momentum_kg_m_per_s, -momentum, rtol=1e-12, atol=1e-35)
    np.testing.assert_allclose(actual.position_m, 0., atol=1e-15)


def test_shared_helpers_remain_callable_without_optional_numba():
    """The reference backend must work when the accelerator is not installed."""
    import os
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent('''
        import builtins
        original = builtins.__import__
        def without_numba(name, *args, **kwargs):
            if name == "numba" or name.startswith("numba."):
                raise ImportError("Numba deliberately unavailable in this test")
            return original(name, *args, **kwargs)
        builtins.__import__ = without_numba
        import numpy as np
        from temsim.physics.discrete_gradient import discrete_gradient_update, norm3
        from temsim.test_electron_sampling import step_cell_index
        from temsim.physics.electrostatic_column_transport import _momentum_speed, _grid_electric
        assert norm3(np.array([3., 0., 4.])) == 5.
        assert step_cell_index(np.array([0., .1, .2]), .1) == 0
        p, speed = _momentum_speed(300000.)
        assert p > 0. and 0. < speed < 299792458.
        data = (np.array([0., 1.]), np.array([0., 1.]), np.array([[0., 10.], [0., 10.]]))
        np.testing.assert_allclose(_grid_electric(0., 0., .5, data), (5., 0., 0., -10.))
        u = discrete_gradient_update(.01, .02, .1, 2.02, 0., 0., 1e-4,
                                     0., 0., 0., 0., 0., 0., 0., 0., 1e-12)
        np.testing.assert_allclose(u, (.01, .02, .1), rtol=1e-14)
    ''')
    environment = dict(os.environ, NUMBA_NUM_THREADS="1", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-c", script], env=environment,
                            capture_output=True, text=True, timeout=30.)
    assert result.returncode == 0, result.stdout+result.stderr
