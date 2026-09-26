"""Execution and persistence contracts for the production electrode field."""
import numpy as np
import pytest

from temsim.optics.electron_gun.field_emission import FieldEmissionGun
from temsim.physics.closed_gun_field import ClosedGunField
from temsim.physics.relativistic_lorentz import RelativisticPhaseSpace, momentum_from_kinetic_energy_ev
from temsim.physics.static_energy_lorentz import static_energy_step


def uniform_field():
    r = np.array([0., .01, .02])
    z = np.array([0., .1, .2])
    return ClosedGunField({'high_tension_v': 2000.}, r, z,
                          np.broadcast_to(10000.*z, (3, 3)))


def test_compiled_closed_step_matches_same_scalar_potential_reference():
    gun = FieldEmissionGun()
    field = uniform_field()
    phase = RelativisticPhaseSpace(np.array([[.0001, 0., .04], [0., .0002, .08]]),
        momentum_from_kinetic_energy_ev([400., 800.], [[.01, 0., 1.], [0., -.02, 1.]]))
    field.compiled_particle_steps = False
    expected = static_energy_step(phase, 1e-13, gun.magnetic_field, field)
    field.compiled_particle_steps = True
    actual = static_energy_step(phase, 1e-13, gun.magnetic_field, field)
    np.testing.assert_allclose(actual.position_m, expected.position_m, atol=1e-21, rtol=1e-14)
    np.testing.assert_allclose(actual.momentum_kg_m_per_s, expected.momentum_kg_m_per_s,
                               atol=1e-36, rtol=1e-13)


def test_compiled_field_keeps_cathode_interior_and_rejects_downstream_extrapolation():
    from temsim.physics.grounded_particle_step import _electric, _closed_field_data
    field = uniform_field()
    points = np.array([[0., 0., -.001], [.001, 0., .045], [0., 0., .2]])
    expected = field.interpolate(points)
    actual = _electric(points, _closed_field_data(field), 2000., True, True)
    for got, wanted in zip(actual, expected):
        np.testing.assert_allclose(got, wanted, atol=1e-11, rtol=1e-14)
    with pytest.raises(ValueError, match='outside'):
        _electric(np.array([[0., 0., .201]]), _closed_field_data(field), 2000., True, True)


def test_input_snapshot_excludes_solved_arrays_but_keeps_electrical_inputs():
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.optics.column import default_state
    state = default_state()
    before = capture_instrument_snapshot(state)
    state.electron_gun._closed_gun_field = uniform_field()
    state.electron_gun._continuous_gun_field = object()
    after = capture_instrument_snapshot(state)
    assert after.physical_digest == before.physical_digest
    restored = after.restore()
    assert not hasattr(restored.electron_gun, '_closed_gun_field')
    assert restored.electron_gun._grounded_outlet_liner_segments == state.electron_gun._grounded_outlet_liner_segments
    state.electron_gun._gun_field_cells_per_bore *= 2
    assert capture_instrument_snapshot(state).physical_digest != before.physical_digest
