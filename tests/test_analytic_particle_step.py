"""Compiled analytic tracing retains the existing physical step and budgets."""

import numpy as np
import pytest

from temsim.optics.electron_gun.field_emission import FieldEmissionGun
from temsim.optics.electron_gun.tracing import _analytic_step
from temsim.physics import analytic_particle_step as compiled
from temsim.physics.relativistic_lorentz import (
    RelativisticPhaseSpace, momentum_from_kinetic_energy_ev,
)


@pytest.fixture
def gun():
    # Mathematical regression fixture for the historical analytic kernel.
    # Production cold FEGs now use the coupled electrode field.
    from temsim.optics.electron_gun.electrostatic import FegElectrostaticField

    class AnalyticGunFixture(FieldEmissionGun):
        @property
        def uses_geometry_electric_field(self):
            return False

        @property
        def base_electric_field(self):
            return FegElectrostaticField(self.emitter, self.extractor,
                                         self.electrostatic_lens, self.accelerator)

    value = AnalyticGunFixture()
    value.emitter.surface_model = None
    return value


def _phase(gun, z_mm, *, count=None):
    z = np.asarray(z_mm, dtype=float)
    if count is not None:
        z = np.full(count, float(z))
    indices = np.arange(len(z))
    x = 1e-5*np.cos(indices)
    y = 2e-5*np.sin(indices)
    points = np.column_stack((x, y, z*1e-3))
    invariant = np.full(len(z), gun.emitter.emission_energy_ev)
    energy = invariant+gun.electric_field.potential_v_at_global_positions(points)
    directions = np.column_stack((.004*np.cos(indices), .003*np.sin(indices), np.ones(len(z))))
    return RelativisticPhaseSpace(points, momentum_from_kinetic_energy_ev(energy, directions)), invariant


def _reference(gun, phase, dt, active, magnetic, electric, invariant):
    electric.compiled_particle_steps = False
    try:
        return _analytic_step(gun, phase, dt, active, magnetic, electric, invariant)
    finally:
        electric.compiled_particle_steps = True


@pytest.mark.skipif(compiled._compiled_step is None, reason="Numba optional")
@pytest.mark.parametrize("configuration", ["zero", "strong", "blanked", "disabled"])
def test_compiled_step_matches_reference_at_fields_and_preserves_stopped_rays(gun, configuration):
    if configuration != "zero":
        gun.extractor.voltage_kv = 8.
        gun.electrostatic_lens.voltage_kv = 3.1
        gun.deflector.upper_field_x_mt, gun.deflector.upper_field_y_mt = .8, -1.2
        gun.deflector.lower_field_x_mt, gun.deflector.lower_field_y_mt = -.3, .5
        gun.deflector.field_center_offset_mm = .17
        gun.stigmator.gradient_t_per_m = 43.
        gun.stigmator.rotation_deg = 37.
        gun.stigmator.field_center_offset_mm = -.23
    gun.deflector.beam_blanked = configuration == "blanked"
    if configuration in {"blanked", "disabled"}:
        gun.deflector.enabled = False
    if configuration == "disabled":
        gun.stigmator.enabled = False
    # Includes extractor, both lens edges, accelerator, both coils, quadrupole,
    # and a field-free drift. Inactive coordinates are retained bit for bit.
    phase, invariant = _phase(gun, [1., 5., 13., 14., 22., 41., 77., 402., 418., 433., 550.])
    active = np.ones(len(invariant), dtype=bool)
    active[[1, 4]] = False
    magnetic, electric = gun.magnetic_field, gun.electric_field
    reference, reference_dt = _reference(gun, phase, 1e-12, active, magnetic, electric, invariant[active])
    actual, actual_dt = compiled.try_analytic_step(gun, phase, 1e-12, active, magnetic, electric, invariant[active])
    assert actual_dt == reference_dt
    assert actual_dt < 1e-12  # Same adaptive rejection rather than looser budgets.
    np.testing.assert_allclose(actual.position_m, reference.position_m, rtol=2e-13, atol=1e-20)
    np.testing.assert_allclose(actual.momentum_kg_m_per_s, reference.momentum_kg_m_per_s, rtol=2e-11, atol=1e-35)
    np.testing.assert_array_equal(actual.position_m[~active], phase.position_m[~active])
    np.testing.assert_array_equal(actual.momentum_kg_m_per_s[~active], phase.momentum_kg_m_per_s[~active])
    assert actual.time_s == reference.time_s


@pytest.mark.skipif(compiled._compiled_step is None, reason="Numba optional")
def test_field_free_motion_and_repeated_parameter_changes_are_not_cached(gun):
    phase, invariant = _phase(gun, [402., 418., 433., 550.])
    active = np.ones(4, dtype=bool)
    magnetic, electric = gun.magnetic_field, gun.electric_field
    previous = None
    for setting in (0., .8, -.8):
        gun.deflector.upper_field_x_mt = setting
        gun.stigmator.gradient_t_per_m = 10.*setting
        reference, reference_dt = _reference(gun, phase, 1e-14, active, magnetic, electric, invariant)
        actual, actual_dt = compiled.try_analytic_step(gun, phase, 1e-14, active, magnetic, electric, invariant)
        assert actual_dt == reference_dt == 1e-14
        np.testing.assert_allclose(actual.position_m, reference.position_m, rtol=2e-13, atol=1e-20)
        np.testing.assert_allclose(actual.momentum_kg_m_per_s, reference.momentum_kg_m_per_s, rtol=2e-11, atol=1e-35)
        if previous is not None:
            assert np.any(actual.momentum_kg_m_per_s != previous)
        previous = actual.momentum_kg_m_per_s


@pytest.mark.skipif(compiled._compiled_step is None, reason="Numba optional")
def test_repeated_steps_near_axis_keep_trajectory_energy_and_step_selection(gun):
    from temsim.physics.relativistic_lorentz import kinetic_energy_ev_from_momentum
    points = np.array([[0., 0., .013], [1e-9, -2e-9, .013], [1e-5, 2e-5, .013]])
    electric, magnetic = gun.electric_field, gun.magnetic_field
    invariant = np.full(3, gun.emitter.emission_energy_ev)
    phase = RelativisticPhaseSpace(points, momentum_from_kinetic_energy_ev(
        invariant+electric.potential_v_at_global_positions(points), np.tile([0., 0., 1.], (3, 1))))
    expected = phase.copy()
    active = np.ones(3, bool)
    for _ in range(40):
        expected, reference_dt = _reference(gun, expected, 1e-13, active, magnetic, electric, invariant)
        phase, actual_dt = compiled.try_analytic_step(gun, phase, 1e-13, active, magnetic, electric, invariant)
        assert actual_dt == reference_dt
    np.testing.assert_allclose(phase.position_m, expected.position_m, rtol=3e-12, atol=1e-18)
    np.testing.assert_allclose(phase.momentum_kg_m_per_s, expected.momentum_kg_m_per_s, rtol=3e-10, atol=1e-34)
    np.testing.assert_array_equal(phase.position_m[0, :2], [0., 0.])
    np.testing.assert_array_equal(phase.momentum_kg_m_per_s[0, :2], [0., 0.])
    np.testing.assert_allclose(kinetic_energy_ev_from_momentum(phase.momentum_kg_m_per_s)
                               - electric.potential_v_at_global_positions(phase.position_m),
                               invariant, rtol=0., atol=1e-8)


@pytest.mark.parametrize("provider,method", [
    ("electric", "field_at_global_positions_v_per_m"),
    ("electric", "potential_v_at_global_positions"),
    ("magnetic", "field_at_global_positions_t"),
    ("extractor", "axial_potential_v_and_derivatives_per_mm"),
    ("deflector", "field_at_global_positions_t"),
    ("stigmator", "field_at_global_positions_t"),
])
def test_instance_field_overrides_use_reference(gun, provider, method):
    phase, invariant = _phase(gun, [402.])
    electric, magnetic = gun.electric_field, gun.magnetic_field
    obj = {"electric": electric, "magnetic": magnetic}.get(provider, getattr(gun, provider, None))
    setattr(obj, method, lambda _positions: None)
    assert compiled.try_analytic_step(gun, phase, 1e-14, np.ones(1, bool), magnetic, electric, invariant) is None


def test_class_override_optional_dependency_and_explicit_disable_use_reference(gun, monkeypatch):
    phase, invariant = _phase(gun, [402.])
    electric, magnetic = gun.electric_field, gun.magnetic_field
    args = (gun, phase, 1e-14, np.ones(1, bool), magnetic, electric, invariant)
    electric.compiled_particle_steps = False
    assert compiled.try_analytic_step(*args) is None
    electric.compiled_particle_steps = True
    magnetic.compiled_particle_steps = False
    assert compiled.try_analytic_step(*args) is None
    magnetic.compiled_particle_steps = True
    gun.compiled_particle_steps = False
    assert compiled.try_analytic_step(*args) is None
    gun.compiled_particle_steps = True
    with monkeypatch.context() as patch:
        patch.setattr(type(magnetic), "field_at_global_positions_t", lambda self, x: np.zeros_like(x))
        assert compiled.try_analytic_step(*args) is None
    with monkeypatch.context() as patch:
        patch.setattr(compiled, "_compiled_step", None)
        assert compiled.try_analytic_step(*args) is None
    class AddedPhysics(type(electric)):
        pass
    electric.__class__ = AddedPhysics
    assert compiled.try_analytic_step(*args) is None


def test_different_energy_projection_field_uses_reference(gun):
    phase, invariant = _phase(gun, [402.])
    other = FieldEmissionGun()
    other.emitter.surface_model = None
    assert compiled.try_analytic_step(gun, phase, 1e-14, np.ones(1, bool),
                                      gun.magnetic_field, other.electric_field, invariant) is None


@pytest.mark.skipif(compiled._compiled_step is None, reason="Numba optional")
def test_parallel_attempt_matches_serial_and_reference_with_global_rejection(gun):
    from numba import get_num_threads, set_num_threads
    gun.deflector.upper_field_x_mt = .7
    gun.stigmator.gradient_t_per_m, gun.stigmator.rotation_deg = 17., 41.
    z = np.resize([1., 13., 41., 402., 433., 550.], 1536)
    phase, invariant = _phase(gun, z)
    active = np.ones(len(z), bool)
    active[::11] = False
    magnetic, electric = gun.magnetic_field, gun.electric_field
    args = (gun, phase, 1e-12, active, magnetic, electric, invariant[active])
    reference, reference_dt = _reference(*args)
    previous = get_num_threads()
    try:
        for threads in (1, min(previous, 4)):
            set_num_threads(threads)
            actual, dt = compiled.try_analytic_step(*args)
            assert dt == reference_dt
            assert get_num_threads() == threads
            np.testing.assert_allclose(actual.position_m, reference.position_m, rtol=2e-13, atol=1e-20)
            np.testing.assert_allclose(actual.momentum_kg_m_per_s, reference.momentum_kg_m_per_s, rtol=2e-11, atol=1e-35)
            np.testing.assert_array_equal(actual.position_m[~active], phase.position_m[~active])
    finally:
        set_num_threads(previous)


@pytest.mark.skipif(compiled._compiled_step is None, reason="Numba optional")
@pytest.mark.parametrize("failure", ["zero_momentum", "nonfinite_field"])
def test_parallel_attempt_never_swallows_any_particle_failure(gun, failure):
    from numba import get_num_threads, set_num_threads
    phase, invariant = _phase(gun, 550., count=1100)
    if failure == "zero_momentum":
        phase.momentum_kg_m_per_s[-1] = 0.
    else:
        phase.position_m[-1, 0] = 1e300
    active = np.ones(1100, bool)
    magnetic, electric = gun.magnetic_field, gun.electric_field
    args = (gun, phase, 1e-14, active, magnetic, electric, invariant)
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(ValueError):
        _reference(*args)
    previous = get_num_threads()
    try:
        set_num_threads(min(previous, 4))
        with pytest.raises(ValueError):
            compiled.try_analytic_step(*args)
    finally:
        set_num_threads(previous)


@pytest.mark.skipif(compiled._compiled_step is None, reason="Numba optional")
@pytest.mark.parametrize("count", [11, 1300])
def test_fused_current_and_midpoint_impulse_cap_matches_numpy(gun, count):
    from numba import get_num_threads, set_num_threads
    from temsim.physics.relativistic_lorentz import lorentz_derivative, velocity_from_momentum_m_per_s
    gun.deflector.upper_field_y_mt = .7
    gun.stigmator.gradient_t_per_m, gun.stigmator.rotation_deg = 19., 23.
    phase, _ = _phase(gun, np.resize([.01, 1., 13., 41., 402., 433., 550.], count))
    active = np.ones(count, bool)
    active[::13] = False
    magnetic, electric = gun.magnetic_field, gun.electric_field
    position, momentum = phase.position_m[active], phase.momentum_kg_m_per_s[active]
    velocity = velocity_from_momentum_m_per_s(momentum)
    previous = get_num_threads()
    try:
        set_num_threads(min(4, previous))
        for dt in (1e-14, 4e-10):
            for impulse in (.01, .05):
                _, force = lorentz_derivative(position, momentum, magnetic, electric_field=electric)
                _, mid_force = lorentz_derivative(position+.5*dt*velocity, momentum, magnetic, electric_field=electric)
                magnitude = np.maximum(np.linalg.norm(force, axis=1), np.linalg.norm(mid_force, axis=1))
                nonzero = magnitude > 0.
                expected = min(dt, float(np.min(impulse*np.linalg.norm(momentum, axis=1)[nonzero]/magnitude[nonzero])))
                actual = compiled.try_analytic_time_step(phase, dt, active, magnetic, electric, impulse)
                assert actual == pytest.approx(expected, rel=2e-13, abs=0.)
                assert get_num_threads() == min(4, previous)
    finally:
        set_num_threads(previous)


def test_impulse_cap_falls_back_for_custom_or_disabled_provider(gun):
    phase, _ = _phase(gun, [1., 14., 433.])
    electric, magnetic = gun.electric_field, gun.magnetic_field
    args = (phase, 1e-12, np.ones(3, bool), magnetic, electric, .05)
    electric.compiled_particle_steps = False
    assert compiled.try_analytic_time_step(*args) is None
    electric.compiled_particle_steps = True
    magnetic.field_at_global_positions_t = lambda x: np.zeros_like(x)
    assert compiled.try_analytic_time_step(*args) is None


@pytest.mark.skipif(compiled._compiled_step is None, reason="Numba optional")
def test_zero_force_and_inactive_particles_do_not_limit_time_step(gun):
    phase, _ = _phase(gun, [550., 550.])
    phase.position_m[1, 0] = 1e300  # Must not query an inactive ray's field.
    magnetic, electric = gun.magnetic_field, gun.electric_field
    for active in (np.array([True, False]), np.zeros(2, bool)):
        assert compiled.try_analytic_time_step(phase, 1e-12, active, magnetic, electric, .05) == 1e-12
