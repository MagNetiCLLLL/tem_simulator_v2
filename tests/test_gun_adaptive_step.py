from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.electron_gun.field_emission import FieldEmissionGun
from temsim.optics.electron_gun.tracing import _analytic_step
from temsim.physics.relativistic_lorentz import (
    RelativisticPhaseSpace, momentum_from_kinetic_energy_ev,
    kinetic_energy_ev_from_momentum, velocity_from_momentum_m_per_s,
)


def test_equal_time_step_accounts_for_trailing_electrons_in_a_field():
    gun = SimpleNamespace(field_supports_mm=((1., 2.), (4., 5.)),
        trace_step_mm=.1, drift_step_mm=2., monochromator_installed=False)
    assert FieldEmissionGun.integration_step_mm_at(gun, [1.5, 3.]) == .1
    assert FieldEmissionGun.integration_step_mm_at(gun, .95) == pytest.approx(.05)
    assert FieldEmissionGun.integration_step_mm_at(gun, 6.) == 2.


def test_monochromator_resolution_applies_to_any_active_electron():
    gun = SimpleNamespace(field_supports_mm=((1., 2.),),
        trace_step_mm=.1, drift_step_mm=2., monochromator_installed=True,
        monochromator=SimpleNamespace(trace_step_mm=.01,
            wien=SimpleNamespace(field_support_mm=(1., 2.))))
    assert FieldEmissionGun.integration_step_mm_at(gun, [1.5, 3.]) == .01


class UniformElectric:
    def potential_v_at_global_positions(self, positions):
        return positions[..., 2] * 1e5

    def field_at_global_positions_v_per_m(self, positions):
        result = np.zeros_like(positions)
        result[..., 2] = -1e5
        return result


class ZeroMagnetic:
    def field_at_global_positions_t(self, positions):
        return np.zeros_like(positions)


def test_adaptive_boris_rejects_excessive_low_energy_step_and_retains_invariant():
    electric = UniformElectric()
    gun = SimpleNamespace(electric_field=electric)
    p = momentum_from_kinetic_energy_ev([.3, .3], [[.1, 0., 1.], [0., .1, 1.]])
    phase = RelativisticPhaseSpace(np.zeros((2, 3)), p)
    out, used = _analytic_step(gun, phase, 1e-9, np.array([True, False]),
        ZeroMagnetic(), electric, np.array([.3]))
    assert 0 < used < 1e-9
    assert np.array_equal(out.position_m[1], phase.position_m[1])
    assert np.array_equal(out.momentum_kg_m_per_s[1], p[1])
    energy = kinetic_energy_ev_from_momentum(out.momentum_kg_m_per_s[0])
    assert energy-electric.potential_v_at_global_positions(out.position_m[0]) == pytest.approx(.3, abs=1e-9)
    assert out.position_m[0, 2] > 0
    assert velocity_from_momentum_m_per_s(out.momentum_kg_m_per_s[0])[2] > 0
