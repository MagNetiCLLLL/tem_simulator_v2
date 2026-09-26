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
