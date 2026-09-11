"""Actual installed-gun tests, plus independent analytic Hamiltonian limits.

These qualify the stated quadratic gun operator, not the missing full chain.
No retired exit emission, source-policy override or synthetic exit pupil is used.
"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import c, e, m_e

from temsim.optics.electron_gun.field_emission import FieldEmissionGun
from temsim.optics.electron_gun.tip_coherence import TipCoherence, TipWaveNumerics, wavelength_m
from temsim.physics.tip_gun_wave import (
    GunWaveNumerics, _energy_transport, _field_coefficients, _momentum_velocity, build_tip_gun_checkpoint,
)

SOURCE = TipWaveNumerics(energy_samples=1)
NUMERICS = GunWaveNumerics(field_step_mm=.05, bore_step_mm=2.)


@pytest.fixture
def gun():
    gun = FieldEmissionGun()
    gun.emitter.coherence = TipCoherence()
    gun.emitter.energy_spread_fwhm_ev = 0.
    return gun


def test_uniform_acceleration_has_correct_variable_momentum_drift_and_time():
    length = .01
    energy, gain = 100., 2900.
    field = SimpleNamespace(potential_v_at_global_positions=lambda p: gain*p[:, 2]/length)
    gun = SimpleNamespace(electric_field=field)
    p0, _ = _momentum_velocity(energy)
    p1, _ = _momentum_velocity(energy+gain)
    mass_ev = m_e*c*c/e
    exact_b = float(p0*c*length/(e*gain)*(np.arccosh(1+(energy+gain)/mass_ev)-np.arccosh(1+energy/mass_ev)))
    exact_t = float((p1-p0)*length/(e*gain))
    errors = []
    for count in (128, 256):
        z = np.linspace(0., length*1e3, count+1)
        phi = gain*(z[1:]+z[:-1])*.5/(length*1e3)
        vector, tensor = np.zeros((count, 2)), np.zeros((count, 2, 2))
        _, row = _energy_transport(gun, z, {float(z[-1])}, (phi, vector, tensor, vector, tensor), energy, lambda: False)
        matrix = np.array(row["canonical_map"])
        assert matrix[0, 0] == 1 and matrix[2, 2] == 1
        assert row["exit_axial_energy_ev"] == energy+gain
        errors.append(abs(matrix[0, 2]-exact_b))
        assert row["reference_flight_time_s"] == pytest.approx(exact_t, rel=1e-4)
    assert 3.8 < errors[0]/errors[1] < 4.2
    assert errors[1]/exact_b < 1e-4


def test_shared_axial_field_gun_operator_converges_to_exact_solenoid():
    energy, length, magnetic = 10000., .001, .001
    momentum, velocity = _momentum_velocity(energy)
    g = -e*magnetic/(2*momentum)
    angle = g*length
    rotation = np.array(((np.cos(angle), np.sin(angle)), (-np.sin(angle), np.cos(angle))))
    exact = np.block([[np.cos(angle)*rotation, np.sin(angle)/g*rotation],
                      [-g*np.sin(angle)*rotation, np.cos(angle)*rotation]])
    gun = SimpleNamespace(electric_field=SimpleNamespace(potential_v_at_global_positions=lambda p: np.zeros(len(p))))
    errors = []
    for count in (16, 32):
        z = np.linspace(0., length*1e3, count+1)
        zeros = np.zeros(count); v = np.zeros((count, 2)); t = np.zeros((count, 2, 2))
        _, row = _energy_transport(gun, z, {float(z[-1])}, (zeros, v, t, v, t), energy,
                                  lambda: False, np.full(count, magnetic))
        errors.append(np.linalg.norm(np.array(row["canonical_map"])-exact))
        assert row["reference_flight_time_s"] == pytest.approx(length/velocity)
        assert row["reference_longitudinal_action_j_s"] == pytest.approx(float(momentum*length), rel=1e-14, abs=0)
    assert errors[0]/errors[1] == pytest.approx(4., rel=.01)


def test_constant_transverse_force_retains_weyl_phase_with_second_order_convergence():
    length, energy, force = .001, 100., .3
    p0, _ = _momentum_velocity(energy)
    gun = SimpleNamespace(electric_field=SimpleNamespace(potential_v_at_global_positions=lambda p: np.zeros(len(p))))
    actions = []
    for count in (16, 32):
        z = np.linspace(0., length*1e3, count+1)
        zeros = np.zeros(count)
        vector, tensor = np.zeros((count, 2)), np.zeros((count, 2, 2))
        magnetic = vector.copy()
        magnetic[:, 0] = force*p0/e
        _, row = _energy_transport(gun, z, {float(z[-1])}, (zeros, vector, tensor, magnetic, tensor), energy, lambda: False)
        offset = row["canonical_offset"]
        assert offset[0] == pytest.approx(force*length**2/2, rel=1e-12)
        assert offset[2] == pytest.approx(force*length, rel=1e-12)
        actions.append(row["weyl_action_m"])
    exact = force**2*length**3/12
    assert (actions[0]-exact)/(actions[1]-exact) == pytest.approx(4., rel=1e-8)


def test_actual_gun_transports_tip_energy_phase_and_reuses_only_executed_inputs(gun):
    result = build_tip_gun_checkpoint(gun, source_numerics=SOURCE, numerics=NUMERICS)
    assert result.plane_z_mm == gun.exit_plane_z_mm
    assert result.record["tip_emission"]["plane_z_mm"] == 0
    assert result.record["energy_transport"][0]["launch_energy_ev"] == .3
    assert result.beam.modes[0].energy_kev == pytest.approx(300.)
    assert result.beam.total_weight == pytest.approx(1., abs=1e-9)
    assert abs(result.record["energy_transport"][0]["metaplectic_reference_phase_rad"]) > 1
    losses = result.record["mode_records"][0]["losses"]
    assert {row["component"] for row in losses} >= {gun.dpa_aperture.key, gun.c1_aperture.key}
    assert build_tip_gun_checkpoint(gun, source_numerics=SOURCE, numerics=NUMERICS) is result
    gun.c1_aperture.radius_mm = 0
    blocked = build_tip_gun_checkpoint(gun, source_numerics=SOURCE, numerics=NUMERICS)
    assert blocked is not result and blocked.digest != result.digest
    assert blocked.transmitted_current_a == 0.
    assert blocked.beam.total_weight == 0.


def test_gun_wave_converges_and_agrees_with_boris_in_the_paraxial_limit(gun):
    # A larger explicitly selected TIP gives small diffraction angles. This
    # is an independent near-axis validation case, not a changed default.
    gun.emitter.virtual_source_fwhm_nm = 50.
    gun.emitter.ray_count = 1024
    gun.trace_step_mm = .01
    source = replace(SOURCE, grid_pixels=256)
    coarse = build_tip_gun_checkpoint(gun, source_numerics=source, numerics=NUMERICS, use_cache=False)
    fine = build_tip_gun_checkpoint(gun, source_numerics=source,
        numerics=replace(NUMERICS, field_step_mm=.025), use_cache=False)
    finer = build_tip_gun_checkpoint(gun, source_numerics=source,
        numerics=replace(NUMERICS, field_step_mm=.0125), use_cache=False)
    matrices = [np.array(r.record["energy_transport"][0]["canonical_map"]) for r in (coarse, fine, finer)]
    assert np.linalg.norm(matrices[0]-matrices[1]) > 3*np.linalg.norm(matrices[1]-matrices[2])
    rays = gun.trace_to_exit()
    assert np.all(rays.exit_bundle.alive)
    mode = finer.beam.modes[0]
    covariance = mode.plane.canonical_covariance(float(wavelength_m(mode.energy_kev*1e3)))
    np.testing.assert_allclose(np.sqrt(np.diag(covariance)[:2]),
                               (np.std(rays.exit_bundle.x_m), np.std(rays.exit_bundle.y_m)), rtol=.02)
    assert finer.record["energy_transport"][0]["reference_flight_time_s"] == pytest.approx(
        float(np.mean(next(p for p in rays.plane_arrivals if p.z_mm == gun.exit_plane_z_mm).time_s)), rel=.003)


def test_accelerator_and_focusing_are_inputs_not_exit_fit_parameters(gun):
    first = build_tip_gun_checkpoint(gun, source_numerics=SOURCE, numerics=NUMERICS)
    gun.accelerator.high_tension_kv = 200.
    slower = build_tip_gun_checkpoint(gun, source_numerics=SOURCE, numerics=NUMERICS)
    assert slower.beam.modes[0].energy_kev == pytest.approx(200.)
    assert slower.record["energy_transport"][0]["reference_flight_time_s"] > first.record["energy_transport"][0]["reference_flight_time_s"]
    gun.electrostatic_lens.voltage_kv *= 1.01
    focused = build_tip_gun_checkpoint(gun, source_numerics=SOURCE, numerics=NUMERICS)
    assert len({r.digest for r in (first, slower, focused)}) == 3
    assert not np.array_equal(focused.beam.modes[0].plane.curvature_m1, slower.beam.modes[0].plane.curvature_m1)
    assert focused.record["tip_emission_id"] == first.record["tip_emission_id"]


def test_installed_wien_and_c1_slit_are_executed(gun):
    gun.install_monochromator()
    gun.c1_aperture.select_slit_mode(True)
    gun.monochromator.slit.gap_um = 100.
    result = build_tip_gun_checkpoint(gun, source_numerics=SOURCE, numerics=NUMERICS)
    assert gun.monochromator.wien.key in result.record["physical_components"]
    losses = result.record["mode_records"][0]["losses"]
    assert any(row["kind"] == "hard_edge_two_blade_stop" for row in losses)
    assert result.beam.total_weight < .99


def test_no_cached_or_caller_exit_can_bypass_tip_configuration_or_cancellation(gun):
    with pytest.raises(InterruptedError):
        build_tip_gun_checkpoint(gun, cancelled=lambda: True)
    gun.source_representation = "effective_gaussian_schell"
    with pytest.raises(ValueError, match="Custom exit"):
        build_tip_gun_checkpoint(gun)
    gun.source_representation = "classical_particles"
    gun.emitter.coherence = None
    with pytest.raises(ValueError, match="explicitly"):
        build_tip_gun_checkpoint(gun)


def test_imported_wien_provider_cannot_be_silently_omitted(gun):
    gun.install_monochromator()
    gun.monochromator._field_provider_override = object()
    with pytest.raises(ValueError, match="Imported Wien"):
        build_tip_gun_checkpoint(gun, source_numerics=SOURCE, numerics=NUMERICS)


def test_actual_rotated_gun_stigmator_and_deflector_field_coefficients(gun):
    gun.stigmator.gradient_t_per_m = .3
    gun.stigmator.rotation_deg = 30.
    z = np.array((gun.stigmator.optical_reference_from_tip_mm,))
    *_, magnetic_hessian = _field_coefficients(gun, z)
    angle = np.deg2rad(60.)
    expected = .3*np.array(((np.cos(angle), np.sin(angle)), (np.sin(angle), -np.cos(angle))))
    np.testing.assert_allclose(magnetic_hessian[0], expected, rtol=1e-12, atol=1e-14)
    gun.deflector.upper_field_x_mt = .02
    gun.deflector.upper_field_y_mt = -.03
    z = np.array((gun.deflector.upper_center_from_tip_mm+gun.deflector.field_center_offset_mm,))
    _, _, _, force, _ = _field_coefficients(gun, z)
    np.testing.assert_allclose(force[0], (-.03e-3, -.02e-3), rtol=1e-12, atol=1e-14)


def test_wave_memory_budget_is_checked_before_allocation(gun):
    with pytest.raises(ValueError, match="maximum_checkpoint_bytes"):
        build_tip_gun_checkpoint(gun, source_numerics=replace(SOURCE, grid_pixels=4096),
                                 numerics=replace(NUMERICS, maximum_checkpoint_bytes=1024))
