"""Full-field scene contracts with small scalar-potential fixtures."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.magnetic_field_scene import MagneticSceneField, _Source
from temsim.physics.planar_gun_field import PlanarGunField
from temsim.physics.closed_gun_field import ClosedGunField
from temsim.optics.electron_gun.monochromator import CombinedElectricField
from temsim.test_electron_scene import prepare_test_electron_scene, _Bore, _bore_intercept, _prepare_electric_provider


def magnetic_scene(*sources):
    bounds = np.array(((-.001, -.001, .005), (.001, .001, .01)))
    return MagneticSceneField(bounds, .001, tuple(s.key for s in sources), (),
                              tuple(s.bounds_m for s in sources), sources, bounds)


def fixture_scene(monkeypatch, *, base=None, magnetic=None, apertures=(), bores=(), extra_state=None):
    if base is None:
        r, z = np.array([0., .002]), np.array([0., .001, .01])
        base = PlanarGunField({}, r, z, np.broadcast_to(z*1e6, (2, 3)))
    gun = SimpleNamespace(emitter=SimpleNamespace(emission_energy_ev=.3, surface_model=None,
                                                  mechanical_center_from_tip_mm=-.5),
                          bore_components=(), dpa_aperture=None, c1_aperture=None)
    state = SimpleNamespace(apertures=apertures, _resolved_assembly=SimpleNamespace(vacuum_bore_segments=bores),
                            energy_filter_installed=False)
    if extra_state:
        vars(state).update(extra_state)
    monkeypatch.setattr("temsim.test_electron_scene._prepare_electric_provider", lambda state, stop: (gun, base, ("Fixture field",)))
    return prepare_test_electron_scene(state, magnetic or magnetic_scene(), z_limits_mm=(5., 10.))


def test_tip_defaults_and_complete_electric_field_are_not_downstream_source(monkeypatch):
    scene = fixture_scene(monkeypatch)
    assert scene.initial_position_m == (0., 0., 0.)
    assert scene.initial_energy_ev == .3
    assert scene.default_path_length_m == .01
    assert scene.has_electric_field
    # Ray Diagram's current window begins after the gun; tip domain is retained.
    assert scene.diagnostic_bounds_m[0, 2] == 0.
    b, e, phi = scene.diagnostic_fields_at_global_position((0., 0., .0002))
    np.testing.assert_array_equal(b, 0.)
    np.testing.assert_allclose(e, (0., 0., -1e6))
    assert phi == pytest.approx(200.)
    assert not scene.diagnostic_bounds_m.flags.writeable


def test_unknown_electric_field_domain_remains_unknown(monkeypatch):
    scene = fixture_scene(monkeypatch)
    assert scene.diagnostic_fields_at_global_position((0., 0., .01001)) is None
    assert scene.diagnostic_fields_at_global_position((.003, 0., .001)) is None
    assert scene.diagnostic_fields_at_global_position((0., 0., -1e-8)) is None


def test_magnetic_gap_is_zero_but_active_missing_support_is_not_extrapolated(monkeypatch):
    class Magnetic:
        def field_at_global_positions_t(self, positions):
            return np.broadcast_to((0., .1, 0.), positions.shape)
    bounds = np.array(((-.0001, -.0001, .005), (.0001, .0001, .006)))
    source = _Source("test_magnet", Magnetic(), bounds, .0001)
    scene = fixture_scene(monkeypatch, magnetic=magnetic_scene(source))
    assert scene.diagnostic_fields_at_global_position((.001, 0., .001)) is not None
    assert scene.diagnostic_fields_at_global_position((.001, 0., .0055)) is None
    b, _, _ = scene.diagnostic_fields_at_global_position((0., 0., .0055))
    np.testing.assert_allclose(b, (0., .1, 0.))


def test_optional_wien_electric_and_potential_added_once_without_duplicate_b(monkeypatch):
    base = PlanarGunField({}, [0., .002], [0., .01], [[0., 1e4], [0., 1e4]])
    class Wien:
        def field_at_global_positions_v_per_m(self, points):
            return np.broadcast_to((1e3, 0., 0.), points.shape)
        def potential_v_at_global_positions(self, points):
            return -1e3*points[:, 0]
    scene = fixture_scene(monkeypatch, base=CombinedElectricField(base, Wien()))
    b, electric, potential = scene.diagnostic_fields_at_global_position((.0002, 0., .001))
    np.testing.assert_array_equal(b, 0.)
    np.testing.assert_allclose(electric, (1e3, 0., -1e6))
    assert potential == pytest.approx(999.8)


@pytest.mark.parametrize("start,end,expected", [
    ((0., 0., 0.), (.002, 0., .01), .5),
    ((.002, 0., .01), (0., 0., 0.), .4),
    ((0., 0., .0055), (.002, 0., .0055), .5),
    ((.002, 0., 0.), (.002, 0., .01), .5),
])
def test_first_bore_contact_handles_forward_backward_and_transverse(start, end, expected):
    bore = _Bore("wall", .005, .006, .001)
    actual = _bore_intercept(np.array(start), np.array(end), bore)
    assert actual == pytest.approx(expected)


def test_thin_annular_electrode_cannot_be_skipped_with_both_ends_in_vacuum():
    bore = _Bore("electrode", .005, .005001, .001, .002)
    assert _bore_intercept(np.array((.0015, 0., 0.)), np.array((.0015, 0., .01)), bore) == pytest.approx(.5)


@pytest.mark.parametrize("backward", [False, True])
def test_aperture_uses_frozen_original_transmission_mechanism(monkeypatch, backward):
    aperture = SimpleNamespace(key="ap", enabled=True, z_mm=5., radius_mm=.2,
                               offset_x_mm=0., offset_y_mm=0.)
    scene = fixture_scene(monkeypatch, apertures=(aperture,))
    aperture.radius_mm = 2.
    start, end = np.array((.001, 0., 0.)), np.array((.001, 0., .01))
    hit = scene.diagnostic_segment_stop(end, start) if backward else scene.diagnostic_segment_stop(start, end)
    assert hit == (.5, "aperture:ap")
    assert scene.diagnostic_segment_stop((0., 0., 0.), (0., 0., .01)) is None


def test_retracted_or_absent_apertures_do_not_stop_particle(monkeypatch):
    apertures = (SimpleNamespace(key="retracted", enabled=False, z_mm=5., radius_mm=0.),
                 SimpleNamespace(key="absent", enabled=True, installed=False, z_mm=6., radius_mm=0.))
    scene = fixture_scene(monkeypatch, apertures=apertures)
    assert scene.diagnostic_segment_stop((0., 0., 0.), (0., 0., .01)) is None


def test_sample_and_detector_do_not_become_hardware_stops(monkeypatch):
    state = {"sample": SimpleNamespace(z_mm=4., inserted=True),
             "detectors": [SimpleNamespace(z_mm=7., inserted=True)]}
    scene = fixture_scene(monkeypatch, extra_state=state)
    assert scene.diagnostic_segment_stop((0., 0., 0.), (0., 0., .01)) is None


def test_flat_tip_return_is_a_hardware_event_and_forward_tip_launch_is_allowed(monkeypatch):
    base = ClosedGunField({}, [0., .002], [0., .01], [[0., 1e4], [0., 1e4]])
    scene = fixture_scene(monkeypatch, base=base)
    assert scene.diagnostic_segment_stop((0., 0., 0.), (0., 0., .001)) is None
    assert scene.diagnostic_segment_stop((0., 0., .001), (0., 0., -.001)) == (.5, "tip_return")
    assert scene.diagnostic_fields_at_global_position((0., 0., -1e-8)) is not None


def test_local_mesh_step_does_not_apply_tip_cells_to_downstream_axis(monkeypatch):
    r = np.array([0., 1e-9, .002])
    z = np.array([0., 1e-9, 1e-6, .001, .01])
    base = PlanarGunField({}, r, z, np.broadcast_to(z*1e6, (3, len(z))))
    scene = fixture_scene(monkeypatch, base=base)
    at_tip = scene.diagnostic_spatial_step_m((0, 0, 0), (0, 0, 1), .001)
    downstream = scene.diagnostic_spatial_step_m((0, 0, .005), (0, 0, 1), .001)
    assert at_tip == pytest.approx(.5e-9)
    assert downstream == .001
    sideways = scene.diagnostic_spatial_step_m((0, 0, .005), (1, 0, 0), .001)
    assert sideways == pytest.approx(.5e-9)


def test_installed_filter_stops_at_entrance_with_explicit_scope(monkeypatch):
    aperture = SimpleNamespace(key="energy_filter_entrance_aperture", z_mm=8., enabled=True,
                               installed=True, radius_mm=1., offset_x_mm=0., offset_y_mm=0.)
    scene = fixture_scene(monkeypatch, apertures=(aperture,), extra_state={"energy_filter_installed": True})
    assert scene.diagnostic_bounds_m[1, 2] == .008
    assert any("stops at its entrance" in note for note in scene.notes)


def test_physical_grounded_liner_retained_without_resolved_assembly(monkeypatch):
    request = {"grounded_liner": [{"key": "captured_liner", "start_m": .005,
                                  "stop_m": .009, "inner_m": .001, "outer_m": .002}]}
    base = PlanarGunField(request, [0., .003], [0., .01], [[0., 1e4], [0., 1e4]])
    scene = fixture_scene(monkeypatch, base=base)
    assert scene.diagnostic_segment_stop((.002, 0., 0.), (.002, 0., .01)) == (.5, "hardware:captured_liner")


def test_transverse_motion_in_aperture_plane_hits_opening_edge(monkeypatch):
    aperture = SimpleNamespace(key="ap", enabled=True, z_mm=5., radius_mm=.2, offset_x_mm=0., offset_y_mm=0.)
    scene = fixture_scene(monkeypatch, apertures=(aperture,))
    fraction, key = scene.diagnostic_segment_stop((0., 0., .005), (.001, 0., .005))
    assert fraction == pytest.approx(.2, abs=1e-13)
    assert key == "aperture:ap"


def test_zero_thickness_shell_plane_is_not_skipped():
    bore = _Bore("joint", .005, .005, .001, .002)
    assert _bore_intercept(np.array((.0015, 0., 0.)), np.array((.0015, 0., .01)), bore) == .5


def test_local_electric_step_crosses_cell_face_without_zeno_pinning(monkeypatch):
    scene = fixture_scene(monkeypatch)
    step = scene.diagnostic_spatial_step_m((0., 0., .001-1e-14), (0., 0., 1.), .001)
    assert step == pytest.approx(.0005)


def test_electric_preparation_uses_instrument_capture_without_cutoff_remeshing(monkeypatch):
    captured = SimpleNamespace(gun_snapshot=object(), provider=object(), notes=("fixed field",),
                              bounds_m=np.array(((-.01, -.01, 0.), (.01, .01, 3.2))))
    calls = []
    def capture(state):
        calls.append(state)
        return captured
    monkeypatch.setattr("temsim.physics.instrument_electric.capture_instrument_electric_field", capture)
    state = object()
    for stop in (.1, 1., 3.):
        clone, provider, notes = _prepare_electric_provider(state, stop)
        assert clone is captured.gun_snapshot and provider is captured.provider
        assert notes == captured.notes
    assert calls == [state]*3
    with pytest.raises(ValueError, match="outside the fixed instrument"):
        _prepare_electric_provider(state, 3.3)


def test_c1_slit_opening_owns_transmission_even_when_stored_circular_radius_is_zero(monkeypatch):
    from temsim.optics.electron_gun.aperture import create_c1_aperture
    from temsim.optics.electron_gun.monochromator import MonochromatorSlit
    aperture = create_c1_aperture()
    aperture.radius_mm = 0.
    aperture.bind_slit_profile(MonochromatorSlit(gap_um=10.)).select_slit_mode(True)
    scene = fixture_scene(monkeypatch, apertures=(aperture,))
    assert scene._aperture_passes(scene._apertures[0], np.array([0., 0.]))
    assert not scene._aperture_passes(scene._apertures[0], np.array([6e-6, 0.]))


def test_fine_adjacent_electric_cell_only_limits_motion_near_its_face(monkeypatch):
    r = np.array([0., .001, .00100001, .003])
    z = np.array([0., .005, .00500001, .01])
    base = PlanarGunField({}, r, z, np.broadcast_to(z*1e6, (len(r), len(z))))
    scene = fixture_scene(monkeypatch, base=base)
    # A 10 nm cell ahead must not force 5 nm steps across the preceding 5 mm.
    assert scene.diagnostic_spatial_step_m((0., 0., .002), (0., 0., 1.), .001) == .001
    near = scene.diagnostic_spatial_step_m((0., 0., .005-1e-8), (0., 0., 1.), .001)
    assert near == pytest.approx(1.5e-8)
    assert scene.diagnostic_spatial_step_m((.0002, 0., .002), (1., 0., 0.), .0001) == .0001
    near_radial = scene.diagnostic_spatial_step_m((.001-1e-8, 0., .002), (1., 0., 0.), .001)
    assert near_radial == pytest.approx(1.5e-8)


@pytest.mark.parametrize('direction', (-1., 1.))
def test_step_cell_face_ulp_ties_are_stable_and_preserve_thin_cell_resolution(monkeypatch, direction):
    from temsim.test_electron_sampling import step_cell_index
    r = np.array([0., .001, .003])
    z = np.array([0., .005, .00500001, .01])
    base = PlanarGunField({}, r, z, np.broadcast_to(z*1e6, (len(r), len(z))))
    scene = fixture_scene(monkeypatch, base=base)
    face = z[2]
    positions = [np.nextafter(face, -np.inf), face, np.nextafter(face, np.inf)]
    assert [step_cell_index(z, value) for value in positions] == [1, 1, 1]
    steps = [scene.diagnostic_spatial_step_m((0., 0., value), (0., 0., direction), .001)
             for value in positions]
    np.testing.assert_array_equal(steps, np.full(3, .5*(z[2]-z[1])))
    # Away from floating-point uncertainty, the actual narrow cell is retained.
    assert step_cell_index(z, .5*(z[1]+z[2])) == 1
    assert step_cell_index(z, face+1e-10) == 2


def test_active_electrostatic_blanker_stops_before_unknown_continuous_field(monkeypatch):
    from temsim.optics.nanopulser import NanoPulser
    pulser = NanoPulser(installed=True, blanked=True, z_mm=6., stop_z_mm=9.,
                       plate_length_mm=2., mechanical_center_from_tip_mm=6., mechanical_length_mm=2.)
    scene = fixture_scene(monkeypatch, extra_state={"nanopulser": pulser})
    assert scene.diagnostic_bounds_m[1, 2] == .005
    assert scene.diagnostic_fields_at_global_position((0., 0., .004)) is not None
    assert scene.diagnostic_fields_at_global_position((0., 0., .006)) is None
    assert scene.diagnostic_segment_stop((0., 0., .004), (0., 0., .006)) == (.5, "unsupported_field:electrostatic_blanker")
    assert any("continuous electric field is unavailable" in note for note in scene.notes)
    assert pulser.installed and pulser.blanked


def test_open_blanker_keeps_physical_stop_without_requiring_unknown_active_field(monkeypatch):
    from temsim.optics.nanopulser import NanoPulser
    pulser = NanoPulser(installed=True, blanked=False, z_mm=6., stop_z_mm=9.,
                       plate_length_mm=2., mechanical_center_from_tip_mm=6., mechanical_length_mm=2.)
    scene = fixture_scene(monkeypatch, extra_state={"nanopulser": pulser})
    assert scene.diagnostic_bounds_m[1, 2] == .01
    fraction, reason = scene.diagnostic_segment_stop((.0002, 0., .008), (.0002, 0., .01))
    assert fraction == pytest.approx(.5)
    assert reason == "aperture:nanopulser_aperture"


def test_child_aperture_captures_parent_pose_for_forward_and_backward_contacts(monkeypatch):
    from temsim.physics.aperture_clipping import clip_segment
    from temsim.test_electron_intercepts import prepare_compiled_intercepts
    parent_data = dict(key="lens", mechanical_profile="magnetic_lens_assembly",
                       local_center_z_mm=5., offset_x_mm=.2, rotation_y_mrad=100.)
    parts = (SimpleNamespace(key="lens", module_key="test", center_z_mm=5., data=parent_data),
             SimpleNamespace(key="ap", module_key="test", center_z_mm=6.,
                 data=dict(key="ap", parent_key="lens", local_center_z_mm=6.)))
    assembly = SimpleNamespace(parts=parts, vacuum_bore_segments=())
    aperture = SimpleNamespace(key="ap", enabled=True, installed=True, z_mm=6.,
                               radius_mm=.05, offset_x_mm=0., offset_y_mm=0.)
    state = SimpleNamespace(_resolved_assembly=assembly, apertures=[aperture])
    scene = fixture_scene(monkeypatch, apertures=(aperture,), extra_state={"_resolved_assembly": assembly})
    assert prepare_compiled_intercepts(scene) is None
    x = np.zeros((2, 1))
    alive, stops, _ = clip_segment(state, [4., 8.], x, x)
    assert not alive[0]
    start, end = np.array((0., 0., .004)), np.array((0., 0., .008))
    forward = scene.diagnostic_segment_stop(start, end)
    backward = scene.diagnostic_segment_stop(end, start)
    assert forward[1] == backward[1] == "aperture:ap"
    assert 4.+forward[0]*4. == pytest.approx(stops[0])
    assert backward[0] == pytest.approx(1.-forward[0])
    parent_data["offset_x_mm"] = -.4
    assert scene.diagnostic_segment_stop(start, end) == forward
