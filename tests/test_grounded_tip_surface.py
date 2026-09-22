"""Reference surface/grounded-field regressions; not coherent image acceptance."""
from dataclasses import replace

import numpy as np
import pytest

from temsim.optics.electron_gun.field_emission import FieldEmissionGun, field_emission_gun_from_dict
from temsim.optics.electron_gun.tip_surface import load_tip_surface_reference, emit_surface, TipSurfaceModel
from temsim.physics.grounded_tip_field import solve_axisymmetric_laplace


@pytest.fixture
def gun():
    gun = FieldEmissionGun()
    gun.emitter.surface_model = load_tip_surface_reference()
    return gun


def test_reference_and_exact_cap_cone_join():
    model = load_tip_surface_reference()
    assert TipSurfaceModel.from_dict(model.to_dict()) == model
    g = model.geometry
    assert g.apex_radius_nm == 100
    radius = g.apex_radius_nm*1e-9
    angle = np.deg2rad(g.cone_half_angle_deg)
    join = -radius*(1-np.sin(angle))
    assert g.radius_m(0) == 0
    assert g.radius_m(join) == pytest.approx(radius*np.cos(angle))
    h = radius*1e-5
    derivative = (g.radius_m(join+h)-g.radius_m(join-h))/(2*h)
    assert derivative == pytest.approx(-np.tan(angle), abs=1e-5)
    assert g.radius_m(1e-9) == 0


def test_surface_distribution_energy_direction_and_flux_are_consistent(gun):
    model = gun.emitter.surface_model
    position, direction, energy, weights = emit_surface(model, 65536)
    radius = model.geometry.apex_radius_nm*1e-9
    normal = (position+np.array([0, 0, radius]))/radius
    np.testing.assert_allclose(np.linalg.norm(normal, axis=1), 1, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(direction, axis=1), 1, atol=1e-14)
    assert np.all(np.sum(normal*direction, axis=1) > 0)
    assert np.all(energy > 0)
    assert weights.sum() == pytest.approx(1)
    assert np.mean(energy) == pytest.approx(model.emission.mean_energy_ev, rel=.002)
    assert np.std(energy) == pytest.approx(model.emission.energy_sigma_ev, rel=.004)
    assert gun.emitted_current_a == pytest.approx(model.emission.current_na*1e-9)
    gun.emitter.emission_energy_ev = -100  # inactive historical input
    gun.emitter.energy_spread_fwhm_ev = 50
    emitted = gun.emit(9)
    assert np.all(emitted.surface_energy_ev > 0)
    assert emitted.surface_position_m.shape == (9, 3)


@pytest.mark.parametrize("change", [{"normal_mean_energy_ev": 0}, {"current_na": float("nan")},
                                     {"cap_half_angle_deg": 89}, {"tangential_mean_energy_ev": -1}])
def test_invalid_surface_inputs_fail_before_trace(gun, change):
    m = gun.emitter.surface_model
    with pytest.raises(ValueError):
        replace(m, emission=replace(m.emission, **change)).validate()


def test_cylindrical_laplace_matches_linear_voltage_on_graded_grid():
    r = np.r_[0, np.geomspace(1e-4, 1, 24)]
    z = np.r_[-.1, 0, np.geomspace(1e-5, 1, 53)]
    fixed = np.zeros((r.size, z.size), bool)
    fixed[:, [0, -1]] = True
    values = np.zeros(fixed.shape)
    values[:, 0], values[:, -1] = -4000, 0
    phi, residual = solve_axisymmetric_laplace(r, z, fixed, values)
    expected = -4000*(z[-1]-z)/(z[-1]-z[0])
    np.testing.assert_allclose(phi, np.broadcast_to(expected, phi.shape), atol=1e-6)
    assert residual < 1e-9


def test_grounded_electrodes_and_tip_near_field(gun):
    field = gun.electric_field
    ext = gun.extractor
    p = np.array([[0., 0., 0.], [ext.mechanical_clear_bore_diameter_mm*1e-3, 0., ext.mechanical_center_from_tip_mm*1e-3],
                  [0., 0., gun.exit_plane_z_mm*1e-3]])
    np.testing.assert_allclose(field.potential_v_at_global_positions(p), [-300000, -296000, 0], atol=1e-6)
    field_z = field.field_at_global_positions_v_per_m(np.array([[0, 0, 1e-9], [0, 0, 1e-6]]))[:, 2]
    assert np.all(field_z < 0)  # force on a negative electron is downstream
    assert field.rise.min() >= -1e-6 and field.rise.max() <= 300000+1e-6
    assert field.report["linear_residual"] < 1e-9


def test_potential_and_field_are_consistent_at_off_axis_point(gun):
    f = gun.electric_field
    p = np.array([[7.13e-7, 3.27e-7, .00123]])
    field = f.field_at_global_positions_v_per_m(p)[0]
    derivative = []
    for k in range(3):
        dx = np.zeros_like(p)
        dx[0, k] = 1e-10
        derivative.append(-(f.potential_rise_v_at_global_positions(p+dx)-f.potential_rise_v_at_global_positions(p-dx))[0]/2e-10)
    np.testing.assert_allclose(field, derivative, rtol=2e-4, atol=.1)


def test_axisymmetric_potential_is_regular_not_a_radial_cusp():
    from temsim.physics.grounded_tip_field import GroundedTipField
    # Independent interpolation fixture. The r^2 profile must be represented
    # exactly and E_r must tend continuously to zero at the optical axis.
    field = GroundedTipField.__new__(GroundedTipField)
    field.r = np.array([0., .5, 1.])
    field.z = np.array([0., 1., 2.])
    field.rise = field.r[:, None]**2+field.z[None, :]
    field.high_tension_v = 10.
    x = np.array([-1e-10, 0., 1e-10, .123, .6])
    points = np.column_stack((x, np.zeros_like(x), np.full_like(x, .4)))
    potential = field.potential_rise_v_at_global_positions(points)
    electric = field.field_at_global_positions_v_per_m(points)
    np.testing.assert_allclose(potential, x*x+.4, atol=1e-15)
    np.testing.assert_allclose(electric[:, 0], -2*x, atol=1e-15)
    np.testing.assert_allclose(electric[:, 2], -1., atol=1e-15)


def test_voltage_reference_is_independent_of_source_energy_and_cache_is_segmented(gun):
    first = gun.electric_field
    m = gun.emitter.surface_model
    gun.emitter.surface_model = replace(m, emission=replace(m.emission, normal_mean_energy_ev=1))
    assert gun.electric_field is first
    gun.extractor.voltage_kv += .1
    assert gun.electric_field is not first
    gun.extractor.voltage_kv = 4.0  # exact original input, not floating subtraction
    assert gun.electric_field is first
    gun.extractor.mechanical_center_from_tip_mm += .2
    assert gun.electric_field is not first
    gun.emitter.surface_model = replace(m, geometry=replace(m.geometry, apex_radius_nm=120))
    assert gun.electric_field.geometry.apex_radius_nm == 120


def test_old_window_and_potential_scale_do_not_control_new_field(gun):
    f = gun.electric_field
    gun.extractor.transition_start_mm = .123
    gun.extractor.transition_end_mm = .456
    gun.electrostatic_lens.potential_scale = 2.0
    assert gun.electric_field is f
    gun.extractor.transition_end_mm = -5
    gun.electrostatic_lens.potential_scale = -1
    gun.electrostatic_lens.soft_edge_mm = 0
    gun.accelerator.stages[0].soft_edge_mm = 0
    gun.validate()  # inactive historical ramps do not constrain physical metal
    assert gun.field_supports_mm == ((-1., gun.exit_plane_z_mm),)
    gun.emitter.surface_model = None
    with pytest.raises(ValueError, match="field transition"):
        gun.validate()


def test_discretised_tip_absorbs_returning_electrons_without_moving_launch(gun):
    emitted = gun.emit(9)
    field = gun.electric_field
    position, error = field.surface_mesh_positions(emitted.surface_position_m)
    assert error < .1*gun.emitter.surface_model.geometry.apex_radius_nm*1e-9
    assert not np.any(field.tip_material_mask(position))
    position[:, 2] -= 1e-9
    assert np.all(field.tip_material_mask(position))


def test_final_anode_cannot_have_an_independent_nonzero_potential(gun):
    gun.accelerator.stages[-1].voltage_fraction = .999999
    with pytest.raises(ValueError, match="fixed at ground"):
        gun.electric_field


def test_curved_surface_audit_is_not_a_plane_quantum_admission(gun):
    from temsim.physics.source_admission import launch_emittance_audit
    report = launch_emittance_audit(gun, count=32)
    assert report["reference_plane"] == "curved tip surface; Cartesian projection"
    assert report["necessary_quantum_covariance_condition"] == "NOT_APPLICABLE_CURVED_CLASSICAL_SURFACE"


def test_analytical_eels_cannot_reuse_an_inactive_historical_width(gun):
    from types import SimpleNamespace
    from temsim.detector.eels_forward import source_energy_fwhm_ev
    state = SimpleNamespace(electron_gun=gun)
    for simulation in (None, SimpleNamespace(gun_trace=SimpleNamespace(output_energy_fwhm_ev=.3))):
        with pytest.raises(ValueError, match="historical source FWHM is not substituted"):
            source_energy_fwhm_ev(state, simulation)


def test_medium_tuning_uses_the_same_surface_source_not_old_gaussian_probes(gun):
    gun.emitter._tuning_boundary_probes = 33
    gun.emitter.virtual_source_fwhm_nm = 10000  # not an active source input
    bundle = gun.emit(193)
    expected = emit_surface(gun.emitter.surface_model, 193)
    np.testing.assert_array_equal(bundle.surface_position_m, expected[0])
    np.testing.assert_array_equal(bundle.surface_direction, expected[1])
    assert np.all(bundle.weight > 0)  # all are physical surface samples


def test_source_model_round_trip_does_not_migrate_legacy(gun):
    copy = field_emission_gun_from_dict(gun.to_dict())
    assert copy.emitter.surface_model == gun.emitter.surface_model
    assert copy.to_dict()["integrator"]["method"] == "static_discrete_gradient"
    old = FieldEmissionGun()
    old.emitter.surface_model = None  # explicitly construct a historical source
    old.emitter.energy_spread_fwhm_ev = .5
    restored = field_emission_gun_from_dict(old.to_dict())
    assert restored.emitter.surface_model is None
    assert restored.emitter.energy_spread_fwhm_ev == .5


def test_profile_and_snapshot_preserve_surface_identity(gun, tmp_path):
    from temsim.optics.column import default_state
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.instrument_snapshot import encode_instrument, decode_instrument
    from temsim.profile_io import save_profile, read_profile, apply_profile_values
    from temsim.immutable_json import json_digest
    from temsim.runtime_parameters import runtime_targets, editable_parameters
    state = default_state()
    state.electron_gun.emitter.surface_model = gun.emitter.surface_model
    graph = encode_instrument(state)
    assert json_digest(encode_instrument(decode_instrument(graph))) == json_digest(graph)
    path = tmp_path/"grounded.toml"
    save_profile(path, state, AssemblyCatalog().default_selection())
    _, values = read_profile(path)
    other = default_state()
    assert apply_profile_values(other, values) is None
    assert other.electron_gun.emitter.surface_model == gun.emitter.surface_model
    targets = runtime_targets(other)
    assert [p.name for p in editable_parameters(targets["feg_tip"])] == ["ray_count"]
    assert "transition_start_mm" not in {p.name for p in editable_parameters(targets["feg_extractor"])}
    legacy = default_state()
    legacy.electron_gun.emitter.surface_model = None
    save_profile(path, legacy, AssemblyCatalog().default_selection())
    _, values = read_profile(path)
    apply_profile_values(other, values)
    assert other.electron_gun.emitter.surface_model is None


def test_surface_editor_has_one_parameter_set_and_derived_quantities(gun, qtbot):
    from temsim.gui.gun_source_dialog import GunSourceDialog
    dialog = GunSourceDialog(gun)
    qtbot.addWidget(dialog)
    assert dialog.surface_enabled.isChecked()
    assert dialog.analytic_tip_panel.isHidden()
    assert set(dialog.surface_inputs) == {"current_na", "cap_half_angle_deg", "normal_mean_energy_ev", "tangential_mean_energy_ev",
                                         "flux_electrons_per_nm2_s", "maximum_angle_deg", "kinetic_mean_ev", "kinetic_sigma_ev"}
    assert "Final anode 0 V" in dialog.surface_voltage.text()
    dialog.surface_inputs["normal_mean_energy_ev"].setText("0.6")
    assert "0.7 eV" in dialog.surface_derived.text()
    dialog.accept()
    assert dialog.value()["surface_model"].emission.normal_mean_energy_ev == .6
    assert gun.emitter.surface_model.emission.normal_mean_energy_ev == .2


def test_coherent_wave_request_cannot_substitute_a_legacy_source(gun):
    from temsim.optics.electron_gun.tip_coherence import generate_tip_emission
    with pytest.raises(ValueError, match="not coherent amplitudes"):
        generate_tip_emission(gun)


def test_full_reference_gun_executes_surface_to_exit_and_caches(gun):
    result = gun.trace_to_exit(9)
    assert np.any(result.exit_bundle.alive)
    assert result.equal_time_history.time_s[-1] > 0
    exit_arrival = result.plane_arrivals[-1]
    assert np.max(exit_arrival.time_s[exit_arrival.reached]) <= result.equal_time_history.time_s[-1]
    assert result.surface_model_report["maximum_exit_energy_error_ev"] < 1e-3
    assert result.c1_transmitted_current_a <= gun.emitted_current_a*(1+1e-12)
    assert gun.trace_to_exit(9) is result
    assert set(arrival.key for arrival in result.plane_arrivals) >= {gun.dpa_aperture.key}
    # The physical emission energy is not subtracted from the accelerator gain.
    assert np.all(result.exit_bundle.energy_offset_ev > 0)
