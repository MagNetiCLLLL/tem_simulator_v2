"""Vector-display contracts; synthetic maps are not microscope validation."""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.magnetic_field_scene import prepare_magnetic_scene, sample_magnetic_diagnostic
from temsim.optics.column import default_state
from temsim.physics.core import fields
from temsim.physics.lens_field_provider import (
    CoordinateRegistration, FieldMapProvenance, GeometryAwareAnalyticFieldProvider,
    LensGeometryBinding, MagneticFieldMap, MappedLensFieldProvider,
)


@dataclass
class Native:
    key: str = "a"
    percent: float = 100.0
    polarity: int = 1
    enabled: bool = True
    z_mm: float = 25.0
    a_mm: float = 2.0
    bore_diameter_mm: float = 4.0

    @property
    def gaussian(self):
        return (SimpleNamespace(sigma=1.0, offset=0.0),)

    def field_support_mm(self):
        return self.z_mm - 14.0, self.z_mm + 14.0

    def magnetic_field_t(self, z_mm):
        return (self.enabled * self.percent * .01 * self.polarity
                * np.exp(-.5 * ((np.asarray(z_mm) - self.z_mm) / self.a_mm) ** 2))


def analytic(native):
    return GeometryAwareAnalyticFieldProvider(native.key, native, binding(native.key), "synthetic")


def binding(key):
    return LensGeometryBinding(key, "synthetic", "synthetic", "{}")


def scene_for(monkeypatch, providers, **kwargs):
    lenses = [provider.native_provider for provider in providers.values()]
    state = SimpleNamespace(lenses=lenses, _field_provider_diagnostics={})
    monkeypatch.setattr("temsim.magnetic_field_scene._resolved_provider", lambda _state, lens: providers[lens.key])
    return prepare_magnetic_scene(state, **kwargs)


def mapped(native, *, kind="axisymmetric_rz", registration=None, status="measured_or_fem_geometry_bound", field=0.4):
    if kind == "axisymmetric_rz":
        axes = (np.linspace(0, .002, 3), np.linspace(.02, .03, 3))
        components = (np.zeros((3, 3)), np.full((3, 3), field))
    else:
        axes = (np.linspace(-.002, .002, 3), np.linspace(-.003, .003, 3), np.linspace(.02, .03, 3))
        components = (np.zeros((3, 3, 3)), np.zeros((3, 3, 3)), np.full((3, 3, 3), field))
    field_map = MagneticFieldMap(kind, axes, components, registration or CoordinateRegistration(),
                                "synthetic", 100., 1, FieldMapProvenance("fem", "synthetic", "0" * 64, "Test only"))
    return MappedLensFieldProvider(native.key, field_map, native, binding(native.key), model_status=status)


def test_real_default_axis_equals_existing_round_lens_bz():
    state = default_state()
    scene = prepare_magnetic_scene(state, z_limits_mm=(0., 3100.))
    z_mm = np.linspace(0., 3100., 119)
    points = np.column_stack((np.zeros((len(z_mm), 2)), z_mm * 1e-3))
    actual = scene.field_at_global_positions_t(points)
    np.testing.assert_allclose(actual[:, 2], fields(z_mm, state)[0], rtol=2e-13, atol=1e-15)
    np.testing.assert_array_equal(actual[:, :2], 0.)
    assert len(scene.seed_regions_m) == len(scene.source_keys) > 1
    assert scene.bounds_m.shape == (2, 3) and not scene.bounds_m.flags.writeable


def test_analytic_radial_direction_domain_and_snapshot_are_preserved(monkeypatch):
    native = Native()
    provider = analytic(native)
    scene = scene_for(monkeypatch, {"a": provider})
    r = .5 * scene.transverse_radius_m
    points = np.array(((r, 0., .024), (-r, 0., .024), (0., r, .026), (r * 4, 0., .025), (0., 0., 1.)))
    actual = scene.field_at_global_positions_t(points)
    assert actual[0, 0] < 0 and actual[1, 0] > 0 and actual[2, 1] > 0
    np.testing.assert_allclose(actual[:3], provider.field_at_global_positions_t(points[:3]))
    np.testing.assert_array_equal(actual[3:], 0.)
    assert scene.contains(points).tolist() == [True, True, True, False, False]
    native.percent = 0.
    np.testing.assert_array_equal(scene.field_at_global_positions_t(points), actual)
    assert scene.transverse_radius_m == pytest.approx(.0002)


def test_equal_opposite_sources_cancel_without_spurious_network_strength(monkeypatch):
    scene = scene_for(monkeypatch, {"a": analytic(Native("a")), "b": analytic(Native("b", polarity=-1))})
    points = np.array(((0., 0., .025), (.0001, 0., .024)))
    assert scene.contains(points).all()
    np.testing.assert_array_equal(scene.field_at_global_positions_t(points), 0.)


def test_overlapping_analytic_domains_stop_before_a_contribution_would_disappear(monkeypatch):
    narrow, broad = analytic(Native("a", a_mm=1.)), analytic(Native("b", a_mm=2.))
    scene = scene_for(monkeypatch, {"a": narrow, "b": broad})
    points = np.array(((.00005, 0., .024), (.00015, 0., .024)))
    assert scene.contains(points).tolist() == [True, False]
    values = scene.field_at_global_positions_t(points)
    expected = narrow.field_at_global_positions_t(points[:1]) + broad.field_at_global_positions_t(points[:1])
    np.testing.assert_allclose(values[:1], expected)
    # This zero is an inadmissible display position, never a partial B field.
    np.testing.assert_array_equal(values[1], 0.)


def test_registered_map_uses_actual_vectors_and_finite_cylinder(monkeypatch):
    registration = CoordinateRegistration((.005, -.002, .1), ((0., 0., 1.), (0., 1., 0.), (-1., 0., 0.)))
    native = Native()
    provider = mapped(native, registration=registration)
    scene = scene_for(monkeypatch, {"a": provider})
    local = np.array(((.001, 0., .025), (.0019, .0019, .025), (0., 0., .031)))
    points = local @ registration.rotation_array.T + registration.origin_array_m
    assert scene.contains(points).tolist() == [True, False, False]
    values = scene.field_at_global_positions_t(points)
    np.testing.assert_allclose(values[0], (.4, 0., 0.), atol=1e-15)
    np.testing.assert_array_equal(values[1:], 0.)
    native.percent = 50.
    np.testing.assert_array_equal(scene.field_at_global_positions_t(points), values)


def test_cartesian_map_region_and_axial_display_clip(monkeypatch):
    scene = scene_for(monkeypatch, {"a": mapped(Native(), kind="cartesian_xyz")}, z_limits_mm=(23., 27.))
    np.testing.assert_allclose(scene.bounds_m, ((-.002, -.003, .023), (.002, .003, .027)))
    points = np.array(((0., 0., .024), (0., .0031, .024), (0., 0., .029)))
    assert scene.contains(points).tolist() == [True, False, False]


def test_joint_members_do_not_double_count_the_combined_field(monkeypatch):
    owner = mapped(Native("a"), status="joint_nonlinear_field")
    member = mapped(Native("b"), status="included_in_joint_nonlinear_field", field=0.)
    providers = {"a": owner, "b": member}
    state = SimpleNamespace(lenses=[owner.native_provider, member.native_provider],
                            _field_provider_diagnostics={"b": {"joint_field_owner": "a"}})
    monkeypatch.setattr("temsim.magnetic_field_scene._resolved_provider", lambda _state, lens: providers[lens.key])
    all_fields = prepare_magnetic_scene(state)
    assert all_fields.source_keys == ("a",)
    point = np.array(((0., 0., .025),))
    np.testing.assert_allclose(all_fields.field_at_global_positions_t(point), owner.field_at_global_positions_t(point))
    assert any("joint" in note for note in all_fields.notes)


def test_generated_field_without_current_capture_does_not_launch_solver(monkeypatch):
    state = default_state()
    state.simulation_mode = "custom"
    lens = state.lenses[0]
    state.lens_field_map_descriptors = {lens.key: {"solver": "axisymmetric_linear_fem"}}
    monkeypatch.setattr("temsim.physics.axisymmetric_magnetostatics.solve_geometry_field_map",
                        lambda *_args, **_kwargs: pytest.fail("3D display must not start a FEM solve"))
    with pytest.raises(ValueError, match="captured magnetic field cache is missing"):
        prepare_magnetic_scene(state)


def test_disabled_and_zero_excitation_fields_are_honest_empty_or_zero(monkeypatch):
    empty = scene_for(monkeypatch, {"a": analytic(Native(enabled=False))})
    assert empty.source_keys == empty.seed_regions_m == ()
    assert not empty.contains(np.zeros((1, 3)))[0]
    zero = scene_for(monkeypatch, {"a": analytic(Native(percent=0.))})
    point = np.array(((0., 0., .025),))
    assert zero.contains(point)[0]
    np.testing.assert_array_equal(zero.field_at_global_positions_t(point), 0.)


@pytest.mark.parametrize("limits", ((1., 1.), (2., 1.), (0., np.inf), (0.,)))
def test_invalid_limits_are_rejected(limits):
    with pytest.raises(ValueError, match="limits"):
        prepare_magnetic_scene(SimpleNamespace(lenses=()), z_limits_mm=limits)


def test_invalid_vector_positions_are_rejected(monkeypatch):
    scene = scene_for(monkeypatch, {"a": analytic(Native())})
    for values in (np.zeros(3), np.zeros((2, 2)), np.array(((np.nan, 0., 0.),))):
        with pytest.raises(ValueError, match="XYZ"):
            scene.field_at_global_positions_t(values)


def bare_state(**kwargs):
    values = dict(lenses=(), stigmators=(), corrector_elements=(), deflectors=(),
                  beam_voltage_kv=300., simulation_mode="analytical", simulation_time_s=.237,
                  sample=SimpleNamespace(z_mm=0.))
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_stigmator_off_axis_effect_matches_lorentz_transport_and_has_ring_readout():
    from temsim.optics.model import Stigmator
    from temsim.specimen.vector_field_transport import SpecimenFieldTransport
    component = Stigmator("Stigmator", "s", 25., strength_x_percent=23., strength_y_percent=-37.)
    state = bare_state(stigmators=(component,))
    scene = prepare_magnetic_scene(state)
    points = np.array(((1e-5, 2e-5, .025), (-2e-5, 1e-5, .026), (0., 0., .025)))
    values = scene.field_at_global_positions_t(points)
    np.testing.assert_allclose(values, SpecimenFieldTransport(state).field_at_global_positions_t(points), rtol=2e-15)
    np.testing.assert_array_equal(values[-1], 0.)
    assert np.linalg.norm(values[0]) > 0.
    profile = sample_magnetic_diagnostic(scene, np.array((25., 26.)))
    assert profile.valid.all() and np.all(profile.transverse_rms_t > 0.)
    assert not profile.valid.flags.writeable
    np.testing.assert_allclose(profile.transverse_rms_t,
                               profile.transverse_gradient_rms_t_per_m*profile.probe_radius_m, rtol=1e-13)
    assert scene.source_categories == ("stigmator",)
    component.strength_x_percent = component.strength_y_percent = 0.
    state.beam_voltage_kv = 80.
    np.testing.assert_array_equal(scene.field_at_global_positions_t(points), values)


@pytest.mark.parametrize("mode", ("analytical", "ideal"))
def test_corrector_multipoles_match_shared_lorentz_field_and_mode(mode):
    from temsim.optics.quadrupole import QuadrupoleComponent
    from temsim.optics.hexapole import HexapoleComponent
    from temsim.specimen.vector_field_transport import SpecimenFieldTransport
    geometry = dict(z_mm=25., effective_length_mm=4., enabled=True, colour="white",
                    mechanical_center_from_tip_mm=25., mechanical_length_mm=8.,
                    mechanical_outer_diameter_mm=20., mechanical_clear_bore_diameter_mm=4.,
                    optical_reference_from_tip_mm=25.)
    quad = QuadrupoleComponent("Quadrupole", "q", strength_m2=80., maximum_strength_m2=100., **geometry)
    hexa = HexapoleComponent("Hexapole", "h", strength_m3=3e7, maximum_strength_m3=4e7,
                             orientation_rad=.23, **geometry)
    state = bare_state(corrector_elements=(quad, hexa), simulation_mode=mode)
    scene = prepare_magnetic_scene(state)
    points = np.array(((1e-5, 2e-5, .025), (-2e-5, 1e-5, .026), (0., 0., .025)))
    expected = SpecimenFieldTransport(state).field_at_global_positions_t(points)
    np.testing.assert_allclose(scene.field_at_global_positions_t(points), expected, rtol=2e-15)
    assert scene.source_categories == ("corrector", "corrector")
    if mode == "ideal":
        without_hex = prepare_magnetic_scene(bare_state(corrector_elements=(quad,), simulation_mode=mode))
        np.testing.assert_array_equal(scene.field_at_global_positions_t(points), without_hex.field_at_global_positions_t(points))


def test_deflector_signed_integrals_use_effective_thickness_and_snapshot_energy():
    from temsim.optics.model import DeflectorPair
    from temsim.physics.core import electron
    component = DeflectorPair("Deflector", "d", 25., 35., 2., -3., -4., 5., thickness_mm=4.)
    state = bare_state(deflectors=(component,))
    scene = prepare_magnetic_scene(state)
    charge, momentum, _ = electron(state)
    for index, (center, dx, dy) in enumerate(((.025, .002, -.003), (.035, -.004, .005))):
        region = scene.source_regions[index]
        z = np.linspace(region.bounds_m[0, 2], region.bounds_m[1, 2], 129)
        points = np.column_stack((np.zeros((len(z), 2)), z))
        values = scene.field_at_global_positions_t(points)
        integral = np.trapezoid(values, z, axis=0)
        np.testing.assert_allclose((-charge*integral[1]/momentum, charge*integral[0]/momentum), (dx, dy), rtol=1e-14)
        assert region.category == "deflector" and region.label == "Deflector"
    assert any("display-only equivalent" in note for note in scene.notes)
    points = np.array(((0., 0., .0229), (0., 0., .0271), (0., 0., .030)))
    np.testing.assert_array_equal(scene.field_at_global_positions_t(points), 0.)
    before = scene.field_at_global_positions_t(np.array(((0., 0., .025),)))
    component.upper_x_mrad = 0.
    state.beam_voltage_kv = 80.
    np.testing.assert_array_equal(scene.field_at_global_positions_t(np.array(((0., 0., .025),))), before)


def test_actual_scan_deflector_captures_live_time_without_modifying_scan():
    from temsim.optics.ac_deflector import AC_DEFLECTOR_DEFINITION
    from temsim.physics.core import electron
    component = AC_DEFLECTOR_DEFINITION.create_component()
    component.scan_enabled = True
    component.scan_amplitude_x_mrad = .73
    component.scan_amplitude_y_mrad = .51
    state = bare_state(deflectors=(component,))
    events = component.kick_events(time_s=state.simulation_time_s)
    scene = prepare_magnetic_scene(state)
    charge, momentum, _ = electron(state)
    for z_mm, dx, dy in events:
        field = scene.field_at_global_positions_t(np.array(((0., 0., z_mm*1e-3),)))[0]
        length = component.effective_thickness_mm*1e-3
        np.testing.assert_allclose((-charge*field[1]*length/momentum, charge*field[0]*length/momentum), (dx, dy))
    assert component.scan_enabled and events == component.kick_events(time_s=state.simulation_time_s)
    changed = prepare_magnetic_scene(bare_state(deflectors=(component,), simulation_time_s=.639))
    points = np.array(((0., 0., events[0][0]*1e-3),))
    assert not np.array_equal(scene.field_at_global_positions_t(points), changed.field_at_global_positions_t(points))


def test_gun_fields_reuse_actual_finite_providers_and_blanking():
    from temsim.optics.electron_gun.field_emission import FieldEmissionGun
    gun = FieldEmissionGun()
    gun.deflector.upper_field_y_mt = .5
    gun.stigmator.gradient_t_per_m = 3.
    state = bare_state(electron_gun=gun)
    scene = prepare_magnetic_scene(state)
    points = np.array(((1e-5, 2e-5, gun.deflector.upper_center_from_tip_mm*1e-3),
                       (2e-5, 1e-5, gun.stigmator.optical_reference_from_tip_mm*1e-3)))
    np.testing.assert_allclose(scene.field_at_global_positions_t(points), gun.magnetic_field.field_at_global_positions_t(points))
    gun.deflector.enabled = False
    gun.deflector.beam_blanked = True
    blanked = prepare_magnetic_scene(state)
    np.testing.assert_allclose(blanked.field_at_global_positions_t(points), gun.magnetic_field.field_at_global_positions_t(points))
    assert blanked.field_at_global_positions_t(points)[0, 1] != scene.field_at_global_positions_t(points)[0, 1]


def test_total_field_sums_lens_stigmator_and_deflector_before_sampling(monkeypatch):
    from temsim.optics.model import Stigmator, DeflectorPair
    lens = Native()
    provider = analytic(lens)
    state = bare_state(lenses=(lens,), stigmators=(Stigmator("Stigmator", "s", 25., strength_x_percent=23.),),
                       deflectors=(DeflectorPair("Deflector", "d", 25., 35., upper_x_mrad=2.),))
    monkeypatch.setattr("temsim.magnetic_field_scene._resolved_provider", lambda *_: provider)
    combined = prepare_magnetic_scene(state)
    points = np.array(((1e-5, 2e-5, .025), (-2e-5, 1e-5, .026)))
    extras = prepare_magnetic_scene(bare_state(stigmators=state.stigmators, deflectors=state.deflectors))
    np.testing.assert_allclose(combined.field_at_global_positions_t(points),
                               provider.field_at_global_positions_t(points)+extras.field_at_global_positions_t(points))
    profile = sample_magnetic_diagnostic(combined, np.array((25.,)), probe_radius_m=.001)
    assert profile.valid[0] and profile.probe_radius_m[0] < .001


def test_exactly_unpowered_controls_do_not_truncate_an_overlapping_lens(monkeypatch):
    from temsim.optics.model import Stigmator, DeflectorPair
    lens = Native()
    provider = analytic(lens)
    stigmator = Stigmator("Stigmator", "s", 25., length_mm=.1)
    deflector = DeflectorPair("Deflector", "d", 25., 35., thickness_mm=.1)
    state = bare_state(lenses=(lens,), stigmators=(stigmator,), deflectors=(deflector,))
    monkeypatch.setattr("temsim.magnetic_field_scene._resolved_provider", lambda *_: provider)
    points = np.array(((1e-4, 0., .025), (0., -1e-4, .025)))
    zero_controls = prepare_magnetic_scene(state)
    assert zero_controls.contains(points).all()
    np.testing.assert_allclose(zero_controls.field_at_global_positions_t(points), provider.field_at_global_positions_t(points))
    stigmator.strength_x_percent = 1.
    powered = prepare_magnetic_scene(state)
    assert not powered.contains(points).any()
    assert zero_controls.contains(points).all()


def test_installed_velocity_selector_uses_only_its_existing_magnetic_provider(monkeypatch):
    from temsim.optics.electron_gun.field_emission import FieldEmissionGun
    gun = FieldEmissionGun()
    gun.install_monochromator()
    monkeypatch.setattr(type(gun), "base_electric_field", property(lambda _self: pytest.fail("Scene must not solve electrostatics")))
    scene = prepare_magnetic_scene(bare_state(electron_gun=gun))
    mono = gun.monochromator.wien
    point = np.array(((1e-5, 2e-5, mono.optical_reference_from_tip_mm*1e-3),))
    np.testing.assert_allclose(scene.field_at_global_positions_t(point), gun.magnetic_field.field_at_global_positions_t(point))
    assert "crossed_field" in scene.source_categories
    assert any("electric contribution is not" in note for note in scene.notes)
