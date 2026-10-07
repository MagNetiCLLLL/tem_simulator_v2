"""Compiled contacts preserve reference hardware geometry and chronological stops."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.test_electron_intercepts import prepare_compiled_intercepts, compiled_intercept
from temsim.test_electron_scene import TestElectronScene, _Bore
from temsim.optics.electron_gun.aperture import GunAperture
from temsim.optics.condenser_aperture import ContinuousApertureComponent


def fixture_scene(*, bores=(), apertures=(), stops=()):
    bounds = np.array(((-1., -1., -.01), (1., 1., 2.)))
    return TestElectronScene(SimpleNamespace(), SimpleNamespace(), SimpleNamespace(),
                             bounds, bounds, (0., 0., 0.), .3, 2., (),
                             tuple(apertures), tuple(bores), (), _flat_cathode=True,
                             _unsupported_stops=tuple(stops))


def circle(*, radius=.2, x=.1, y=-.15, z=5., key="circle"):
    return SimpleNamespace(key=key, z_mm=z, radius_mm=radius, offset_x_mm=x, offset_y_mm=y)


def compiled_result(scene, start, end, packed=None):
    prepared = packed or prepare_compiled_intercepts(scene)
    assert prepared is not None
    data, reasons = prepared
    fraction, index = compiled_intercept(np.asarray(start, float), np.asarray(end, float), data)
    return None if index < 0 else (fraction, reasons[index])


def assert_matches(scene, start, end, packed=None):
    expected = scene.diagnostic_segment_stop(start, end)
    actual = compiled_result(scene, start, end, packed)
    if expected is None:
        assert actual is None
    else:
        assert actual is not None
        assert actual[1] == expected[1]
        assert actual[0] == pytest.approx(expected[0], abs=2e-14, rel=2e-13)


@pytest.mark.parametrize("start,end", [
    ((0., 0., 0.), (.002, 0., .01)),
    ((.002, 0., .01), (0., 0., 0.)),
    ((0., 0., .0055), (.002, 0., .0055)),
    ((.002, 0., 0.), (.002, 0., .01)),
    ((0., 0., .0055), (.001, 0., .0055)),
    ((.001, 0., .0055), (0., 0., .0055)),
    ((0., .001, .0055), (.002, .001, .0055)),
    ((-.002, .001, .0055), (.002, .001, .0055)),
    ((.001, 0., .0055), (.001, 0., .0055)),
    ((0., 0., .006), (0., 0., .006)),
])
@pytest.mark.parametrize("lower,upper,outer", [(.005, .006, np.inf), (.005, .006, .0011), (.0055, .0055, .0011)])
def test_bore_forward_backward_transverse_shell_tangent_and_terminal_contacts(start, end, lower, upper, outer):
    scene = fixture_scene(bores=(_Bore("wall", lower, upper, .001, outer),))
    assert_matches(scene, start, end)


@pytest.mark.parametrize("start,end", [
    ((0., 0., 0.), (0., 0., -.001)),
    ((0., 0., .001), (0., 0., -.001)),
    ((0., 0., -.001), (0., 0., .001)),
    ((0., 0., .001), (0., 0., 0.)),
    ((0., 0., 0.), (0., 0., .001)),
    ((0., 0., 0.), (0., 0., 0.)),
])
def test_flat_cathode_return_uses_original_strict_negative_z(start, end):
    assert_matches(fixture_scene(), start, end)


@pytest.mark.parametrize("start,end", [
    ((0., 0., .001), (.002, 0., .008)),
    ((.002, 0., .008), (0., 0., .001)),
    ((.0001, -.00015, .005), (.001, 0., .005)),
    ((.001, 0., .005), (.0001, -.00015, .005)),
    ((.0001, -.00015, .005), (.0002, -.00015, .005)),
    ((.0001, -.00015, .006), (.001, 0., .006)),
    ((.0001, -.00015, .005), (.0001, -.00015, .005)),
])
@pytest.mark.parametrize("radius", [.2, 0., -1.])
def test_offset_aperture_forward_backward_and_in_plane_bisection(start, end, radius):
    scene = fixture_scene(apertures=(circle(radius=radius),))
    assert_matches(scene, start, end)


def test_known_mask_zero_radius_semantics_are_not_replaced_by_generic_circle():
    gun = GunAperture("gun", "gun", 5., 1., 10., 2., .1, 0., 1.)
    ordinary = ContinuousApertureComponent("ordinary", "ordinary", 5., 0., 0., 0., True,
                                         "white", 1., 10., 2., .1, 1.)
    start, end = (0., 0., .004), (0., 0., .006)
    gun_scene, ordinary_scene = fixture_scene(apertures=(gun,)), fixture_scene(apertures=(ordinary,))
    assert compiled_result(gun_scene, start, end) == (.5, "aperture:gun")
    assert compiled_result(ordinary_scene, start, end) is None
    assert_matches(gun_scene, start, end)
    assert_matches(ordinary_scene, start, end)
    gun.enabled = False
    assert compiled_result(fixture_scene(apertures=(gun,)), (.001, 0., .004), (.001, 0., .006)) is None


def test_same_fraction_preserves_reference_priority_and_backward_unsupported_stop_is_ignored():
    scene = fixture_scene(bores=(_Bore("wall", .005, .006, .001),),
                          apertures=(circle(z=5., radius=0., x=0., y=0.),),
                          stops=((.005, "unsupported:first"), (.005, "unsupported:second")))
    assert compiled_result(scene, (.002, 0., 0.), (.002, 0., .01)) == (.5, "unsupported:first")
    assert_matches(scene, (.002, 0., .01), (.002, 0., 0.))


def test_random_chords_match_complete_reference_in_both_directions():
    scene = fixture_scene(bores=(_Bore("wall", .001, .009, .001),
                                 _Bore("shell", .005, .0050001, .0008, .0009),
                                 _Bore("plane", .007, .007, .0007, .0008)),
                          apertures=(circle(), circle(z=8., radius=.4, x=-.1, y=.2, key="later")),
                          stops=((.009, "unsupported:downstream"),))
    packed = prepare_compiled_intercepts(scene)
    rng = np.random.default_rng(7163)
    for _ in range(1000):
        endpoints = rng.uniform((-.002, -.002, -.001), (.002, .002, .012), size=(2, 3))
        assert_matches(scene, *endpoints, packed)
        assert_matches(scene, *endpoints[::-1], packed)


def test_capture_owns_immutable_geometry_and_does_not_retain_live_aperture_values():
    aperture = circle()
    scene = fixture_scene(apertures=(aperture,))
    prepared = prepare_compiled_intercepts(scene)
    assert not prepared[0][0].flags.writeable
    start, end = (.0001, -.00015, .004), (.0001, -.00015, .006)
    aperture.radius_mm = 0.
    assert compiled_result(scene, start, end, prepared) is None
    assert compiled_result(scene, start, end) is not None


def test_unsupported_masks_curved_tip_subclasses_and_unavailable_compiler_keep_reference(monkeypatch):
    custom = circle()
    custom.transmission_mask = lambda x, y: x > 0.
    assert prepare_compiled_intercepts(fixture_scene(apertures=(custom,))) is None
    gun = GunAperture("gun", "gun", 5., 1., 10., 2., .1, .2, 1.)
    gun.bind_slit_profile(SimpleNamespace(inserted=True, gap_um=50., centre_offset_um=0.))
    gun.select_slit_mode(True)
    assert prepare_compiled_intercepts(fixture_scene(apertures=(gun,))) is None
    assert prepare_compiled_intercepts(replace(fixture_scene(), _flat_cathode=False)) is None
    assert prepare_compiled_intercepts(SimpleNamespace(_flat_cathode=True)) is None
    changed = fixture_scene()
    object.__setattr__(changed, "diagnostic_segment_stop", lambda start, end: (0., "custom boundary"))
    assert prepare_compiled_intercepts(changed) is None
    assert prepare_compiled_intercepts(fixture_scene(apertures=(SimpleNamespace(),))) is None
    monkeypatch.setattr("temsim.test_electron_intercepts.njit", None)
    assert prepare_compiled_intercepts(fixture_scene()) is None


def test_default_captured_hardware_accepts_compilation_and_agrees_at_all_bore_faces():
    from temsim.cpu_resources import numerical_job
    from temsim.optics.column import default_state
    from temsim.magnetic_field_scene import prepare_magnetic_scene
    from temsim.test_electron_scene import prepare_test_electron_scene
    with numerical_job(1):
        state = default_state()
        scene = prepare_test_electron_scene(state, prepare_magnetic_scene(state))
        packed = prepare_compiled_intercepts(scene)
        assert packed is not None
        for bore in scene._bores:
            for face in (bore.lower_m, bore.upper_m):
                for radius in (bore.inner_m*.99, bore.inner_m*1.01):
                    assert_matches(scene, (radius, 0., face-1e-6), (radius, 0., face+1e-6), packed)
                    assert_matches(scene, (radius, 0., face+1e-6), (radius, 0., face-1e-6), packed)


def test_real_posed_objective_contacts_match_particles_without_nominal_liner_ghosts():
    from temsim.optics.column import default_state
    from temsim.physics.column_wall import clip_column_wall
    from temsim.test_electron_scene import _physical_bores
    state = default_state()
    original = state._resolved_assembly
    # The field request contains nominal copies of these same mechanical
    # liners. Keep an unrelated joint to prove it is not discarded with them.
    liner_rows = [dict(key=row.key, start_m=row.start_z_mm*1e-3,
        stop_m=row.end_z_mm*1e-3, inner_m=row.inner_diameter_mm*.5e-3)
        for row in original.vacuum_liner_segments]
    liner_rows.append(dict(key="independent_joint", start_m=2.8, stop_m=2.81, inner_m=.001))
    base = SimpleNamespace(request={"grounded_liner": liner_rows})
    for sign in (-1., 1.):
        state._resolved_assembly = replace(original, parts=tuple(
            replace(part, data={**part.data, "offset_x_mm": sign*.5})
            if part.key == "objective_lens" else part for part in original.parts))
        captured = fixture_scene(bores=_physical_bores(state, SimpleNamespace(), base))
        assert prepare_compiled_intercepts(captured) is None
        assert any(bore.key == "independent_joint" for bore in captured._bores)
        for x_mm, expected_alive in ((-sign*2.5, False), (sign*3.1, True)):
            x = np.full((2, 1), x_mm*1e-3)
            alive, stops, _ = clip_column_wall(state, [1620., 1621.], x, np.zeros_like(x))
            hit = captured.diagnostic_segment_stop((x[0, 0], 0., 1.620), (x[1, 0], 0., 1.621))
            assert bool(alive[0]) == expected_alive
            assert (hit is None) == expected_alive
            if hit is not None:
                assert hit[1] == "hardware:column_wall"
                assert 1620.+hit[0] == pytest.approx(stops[0])


def test_posed_bore_keeps_fixed_tube_and_checks_tilted_slab_outside_nominal_z():
    from temsim.physics.column_wall import clip_column_wall
    from temsim.test_electron_scene import _physical_bores
    from temsim.lens_pose import rotation_matrix_mrad
    from temsim.physics.lens_field_provider import CoordinateRegistration
    row = dict(key="lens", mechanical_profile="magnetic_lens_assembly",
               local_center_z_mm=10., offset_x_mm=.5)
    segments = (SimpleNamespace(key="lens", start_z_mm=0., end_z_mm=20., inner_diameter_mm=2.),
                SimpleNamespace(key="fixed", start_z_mm=0., end_z_mm=20., inner_diameter_mm=1.5))
    part = SimpleNamespace(key="lens", module_key="module", center_z_mm=10., data=row)
    state = SimpleNamespace(_resolved_assembly=SimpleNamespace(parts=(part,), vacuum_bore_segments=segments))
    scene = fixture_scene(bores=_physical_bores(state, SimpleNamespace(), SimpleNamespace()))
    x = np.full((2, 1), .001)
    alive, _, _ = clip_column_wall(state, [5., 6.], x, np.zeros_like(x))
    assert not alive[0]
    assert scene.diagnostic_segment_stop((.001, 0., .005), (.001, 0., .006)) == (0., "hardware:column_wall")
    rotation = rotation_matrix_mrad((0., 100., 0.))
    pivot = np.array((0., 0., .01))
    registration = CoordinateRegistration(tuple(pivot-rotation@pivot), tuple(map(tuple, rotation)))
    tilted = fixture_scene(bores=(_Bore("column_wall", 0., .02, .00025, registration=registration),))
    hit = tilted.diagnostic_segment_stop((0., 0., .01), (0., 0., .02))
    assert hit[0] == pytest.approx(.00025/np.sin(.1)/.01)
    # This global-Z slab lies beyond the nominal end, but is inside the
    # tilted local axial slab and outside its radial opening.
    assert tilted.diagnostic_segment_stop((-.002, 0., .0201), (-.002, 0., .0202)) == (0., "hardware:column_wall")
