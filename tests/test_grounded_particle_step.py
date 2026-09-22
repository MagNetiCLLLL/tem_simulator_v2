"""Compiled gun integration must retain fields, stops, work and cancellation."""
import numpy as np
import pytest
from temsim.optics.electron_gun.field_emission import FieldEmissionGun
from temsim.physics.relativistic_lorentz import RelativisticPhaseSpace, momentum_from_kinetic_energy_ev
from temsim.physics.static_energy_lorentz import static_energy_step


@pytest.mark.parametrize('blanked', [False, True])
def test_compiled_magnetic_fields_match_driven_and_blanked_coils(blanked):
    from temsim.physics.grounded_particle_step import _magnetic
    gun = FieldEmissionGun()
    d, s = gun.deflector, gun.stigmator
    d.upper_field_x_mt, d.upper_field_y_mt = .31, -.21
    d.lower_field_x_mt, d.lower_field_y_mt = -.17, .11
    d.beam_blanked = blanked
    d.enabled = not blanked  # The independent blanking drive still acts.
    s.gradient_t_per_m, s.rotation_deg = 12.3, 31.
    upper = (0., d.blanking_field_y_mt) if blanked else (.31, -.21)
    lower = (0., 0.) if blanked else (-.17, .11)
    parameters = np.array([d.upper_center_from_tip_mm, .5*d.coil_length_mm, d.soft_edge_mm,
        upper[0]*1e-3, upper[1]*1e-3, d.lower_center_from_tip_mm, .5*d.coil_length_mm,
        d.soft_edge_mm, lower[0]*1e-3, lower[1]*1e-3, s.optical_reference_from_tip_mm,
        .5*s.effective_length_mm, s.soft_edge_mm, s.gradient_t_per_m, np.deg2rad(s.rotation_deg)])
    points = np.column_stack((np.full(601, 1e-4), np.full(601, -2e-4), np.linspace(.3, .45, 601)))
    np.testing.assert_allclose(_magnetic(points, parameters),
        gun.magnetic_field.field_at_global_positions_t(points), rtol=1e-13, atol=1e-17)


def test_compiled_step_matches_reference_at_emitter_electrodes_and_alignment():
    from temsim.optics.electron_gun.tip_surface import load_tip_surface_reference
    gun = FieldEmissionGun()
    gun.emitter.surface_model = load_tip_surface_reference()
    field = gun.electric_field
    gun.deflector.upper_field_y_mt = .7
    gun.stigmator.gradient_t_per_m = 2.
    b = gun.emit(9)
    surface, _ = field.surface_mesh_positions(b.surface_position_m)
    try:
        for points, dt in ((surface, 1e-20),
                           (np.array([[1e-4, 2e-4, .02], [2e-4, -1e-4, .35],
                                      [1e-4, -2e-4, gun.exit_plane_z_mm*1e-3+.001]]), 1e-13)):
            energy = field.potential_rise_v_at_global_positions(points)+.3
            phase = RelativisticPhaseSpace(points, momentum_from_kinetic_energy_ev(
                energy, np.tile([.01, .02, 1.], (len(points), 1))))
            field.compiled_particle_steps = False
            reference = static_energy_step(phase, dt, gun.magnetic_field, field)
            field.compiled_particle_steps = True
            actual = static_energy_step(phase, dt, gun.magnetic_field, field)
            np.testing.assert_allclose(actual.position_m, reference.position_m, rtol=1e-13, atol=1e-22)
            np.testing.assert_allclose(actual.momentum_kg_m_per_s, reference.momentum_kg_m_per_s, rtol=1e-11, atol=1e-35)
    finally:
        field.compiled_particle_steps = True


def test_custom_field_override_uses_reference_instead_of_bypassing_physics():
    from temsim.physics.grounded_particle_step import try_step
    gun = FieldEmissionGun()
    field, magnetic = gun.electric_field, gun.magnetic_field
    phase = RelativisticPhaseSpace(np.array([[1e-4, 0., .35]]),
        momentum_from_kinetic_energy_ev(np.array([300000.]), np.array([[0., 0., 1.]])))
    calls = []
    def custom_field(points):
        calls.append(True)
        return np.tile([.01, 0., 0.], (len(points), 1))
    magnetic.field_at_global_positions_t = custom_field
    assert try_step(phase, 1e-15, magnetic, field, 1e-11, 64) is None
    actual = static_energy_step(phase, 1e-15, magnetic, field)
    assert calls and actual.momentum_kg_m_per_s[0, 1] < 0.


def test_cancelled_trace_never_populates_cache_and_checks_inside_gun():
    gun = FieldEmissionGun()
    calls = []
    def cancelled():
        calls.append(True)
        return len(calls) >= 6
    with pytest.raises(RuntimeError, match='Superseded'):
        gun.trace_to_exit(11, cancelled=cancelled)
    assert len(calls) == 6
    assert gun._trace_cache is None and gun._trace_cache_key is None


def test_explicit_and_default_ray_budget_reuse_executed_trace():
    gun = FieldEmissionGun()
    gun.emitter.surface_model = None
    gun.emitter.ray_count = 17
    assert gun.trace_to_exit(17) is gun.trace_to_exit()
    assert gun._cache_key(17) != gun._cache_key(19)
