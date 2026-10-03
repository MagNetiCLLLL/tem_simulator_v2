"""Independent physical dimensions and capture rules for plane outlines."""
from dataclasses import FrozenInstanceError
import math
from types import SimpleNamespace

import pytest

from temsim.gui.plane_hardware_geometry import (
    hardware_geometry_snapshot,
    plane_hardware_outlines,
    projected_hardware_outlines,
)


def _result(*, walls=(), apertures=(), detectors=(), gun_bores=()):
    return SimpleNamespace(
        assembly=SimpleNamespace(vacuum_bore_segments=walls),
        aperture_stops=apertures,
        state_snapshot=SimpleNamespace(recording_planes=detectors,
                                       electron_gun=SimpleNamespace(bore_components=gun_bores)),
    )


def _wall(start, stop, diameter, key="column_wall"):
    return SimpleNamespace(key=key, name=key, start_z_mm=start,
                           end_z_mm=stop, inner_diameter_mm=diameter)


def _aperture(**changes):
    values = dict(key="aperture", name="Circular aperture", z_mm=10., enabled=True,
                  installed=True, shape="circular", diameter_mm=.04,
                  offset_x_mm=.01, offset_y_mm=-.02)
    values.update(changes)
    return values


def _detector(key="detector", geometry="annulus", **changes):
    values = dict(key=key, name=key, z_mm=20., inserted=True,
                  readout_enabled=True, geometry=geometry, outer_width_mm=4.,
                  inner_diameter_mm=2., centre_offset_x_mm=.3,
                  centre_offset_y_mm=-.4)
    values.update(changes)
    return SimpleNamespace(**values)


def test_column_bore_uses_local_section_and_narrower_transition_face():
    result = _result(walls=(_wall(0., 10., 4., "upper"), _wall(10., 20., 2., "lower")))
    upper, = plane_hardware_outlines(result, 5., circle_segments=16)
    lower, = plane_hardware_outlines(result, 15., circle_segments=16)
    transition, = plane_hardware_outlines(result, 10., circle_segments=16)
    assert upper.polylines_mm[0][0] == (2., 0.)
    assert lower.polylines_mm[0][0] == transition.polylines_mm[0][0] == (1., 0.)
    assert transition.key == "lower"
    assert transition.role == "wall"
    assert plane_hardware_outlines(result, 25.) == ()


def test_wall_gap_is_not_filled_with_a_default_bore():
    result = _result(walls=(_wall(0., 10., 4.), _wall(12., 20., 2.)))
    assert plane_hardware_outlines(result, 11.) == ()
    assert plane_hardware_outlines(SimpleNamespace(), 5.) == ()


def test_shifted_aperture_is_an_exact_optical_plane_opening_in_lab_mm():
    record = _aperture()
    result = _result(apertures=(record,))
    outline, = plane_hardware_outlines(result, 10., circle_segments=16)
    assert outline.role == "opening"
    assert outline.kind == "aperture"
    assert outline.polylines_mm[0][0] == pytest.approx((.03, -.02))
    assert outline.polylines_mm[0][4] == pytest.approx((.01, 0.))
    assert outline.polylines_mm[0][-1] == outline.polylines_mm[0][0]
    assert plane_hardware_outlines(result, 10. + 5.e-10)
    assert plane_hardware_outlines(result, 10. + 2.e-9) == ()
    # A long aperture carrier is not an active opening at surrounding Z.
    assert plane_hardware_outlines(result, 9.9) == ()
    with pytest.raises(FrozenInstanceError):
        outline.role = "absorbing"
    with pytest.raises(TypeError):
        outline.polylines_mm[0][0] = (0., 0.)


@pytest.mark.parametrize("field", ["enabled", "installed"])
def test_inactive_aperture_is_not_drawn(field):
    record = _aperture()
    record[field] = False
    assert plane_hardware_outlines(_result(apertures=(record,)), 10.) == ()


def test_closed_aperture_retains_its_position_and_explicit_blocking_status():
    record = _aperture()
    record["diameter_mm"] = 0.
    outline, = plane_hardware_outlines(_result(apertures=(record,)), 10.)
    assert set(outline.polylines_mm[0]) == {(.01, -.02)}
    assert "all rays are blocked" in outline.description


def _slit(**changes):
    values = dict(key="slit", name="Two-blade slit", z_mm=10., enabled=True,
                  installed=True, shape="two_blade_slit", bore_diameter_mm=2.,
                  slit_inserted=True, slit_gap_mm=.4, slit_centre_x_mm=.1)
    values.update(changes)
    return values


def test_slit_blades_are_clipped_to_circular_bore_not_infinite_lines():
    outline, = plane_hardware_outlines(_result(apertures=(_slit(),)), 10., circle_segments=16)
    bore, left, right = outline.polylines_mm
    assert bore[0] == (1., 0.)
    assert left[0] == pytest.approx((-.1, -math.sqrt(.99)))
    assert left[1] == pytest.approx((-.1, math.sqrt(.99)))
    assert right[0] == pytest.approx((.3, -math.sqrt(.91)))
    assert right[1] == pytest.approx((.3, math.sqrt(.91)))
    for path in (left, right):
        for x, y in path:
            assert math.hypot(x, y) == pytest.approx(1.)


def test_retracted_slit_blades_still_have_the_circular_bore():
    outline, = plane_hardware_outlines(_result(apertures=(_slit(slit_inserted=False),)), 10.)
    assert len(outline.polylines_mm) == 1
    assert "blades retracted" in outline.description


def test_closed_and_outside_slits_are_explicit_without_invented_edges():
    closed, = plane_hardware_outlines(_result(apertures=(_slit(slit_gap_mm=0.),)), 10.)
    assert len(closed.polylines_mm) == 2
    assert "No transmitting opening" in closed.description
    outside, = plane_hardware_outlines(_result(apertures=(_slit(slit_centre_x_mm=3.),)), 10.)
    assert len(outside.polylines_mm) == 1
    assert "No transmitting opening" in outside.description


def test_detector_annulus_has_hole_offset_and_absorbing_band_with_readout_off():
    detector = _detector(readout_enabled=False)
    outline, = plane_hardware_outlines(_result(detectors=(detector,)), 20., circle_segments=16)
    assert outline.role == "absorbing"
    assert outline.kind == "detector"
    outer, inner = outline.polylines_mm
    assert outer[0] == pytest.approx((2.3, -.4))
    assert inner[0] == pytest.approx((1.3, -.4))
    assert "central hole transmits" in outline.description
    assert "physical absorption is still active" in outline.description


def test_disk_absorbs_its_interior_and_retracted_detector_is_excluded():
    disk = _detector(geometry="disk", inner_diameter_mm=0.)
    outline, = plane_hardware_outlines(_result(detectors=(disk,)), 20.)
    assert len(outline.polylines_mm) == 1
    assert "interior absorbs" in outline.description
    disk.inserted = False
    assert plane_hardware_outlines(_result(detectors=(disk,)), 20.) == ()


def test_zero_inner_radius_annulus_has_no_fictitious_transmitting_hole():
    outline, = plane_hardware_outlines(_result(detectors=(_detector(inner_diameter_mm=0.),)), 20.)
    assert len(outline.polylines_mm) == 1
    assert "full interior absorbing" in outline.description


def test_camera_square_stop_does_not_inherit_readout_axis_rotation_or_offsets():
    camera = _detector(geometry="square", detector_axis_rotation_deg=37.,
                       detector_flip_x=True, detector_flip_y=True)
    outline, = plane_hardware_outlines(_result(detectors=(camera,)), 20.)
    assert outline.polylines_mm == (((-2., -2.), (2., -2.), (2., 2.),
                                    (-2., 2.), (-2., -2.)),)
    assert plane_hardware_outlines(_result(detectors=(camera,)), 20. + 2.e-9) == ()


def test_multiple_detector_channels_at_same_z_keep_individual_boundaries():
    detectors = (_detector("haadf"), _detector("df", inner_diameter_mm=.5),
                 _detector("bf", "disk", outer_width_mm=.4, inner_diameter_mm=0.))
    outlines = plane_hardware_outlines(_result(detectors=detectors), 20.)
    assert [row.key for row in outlines] == ["haadf", "df", "bf"]


def test_draws_captured_inputs_without_consulting_an_unrelated_live_state():
    result = _result(apertures=(_aperture(),))
    result.live_state = SimpleNamespace(apertures=(_aperture(diameter_mm=20.),))
    outline, = plane_hardware_outlines(result, 10., circle_segments=16)
    assert outline.polylines_mm[0][0] == pytest.approx((.03, -.02))
    assert result.aperture_stops[0]["diameter_mm"] == .04


def test_finite_gun_bore_remains_when_field_is_disabled():
    body = SimpleNamespace(key="gun_lens", name="Gun lens body", installed=True,
                           enabled=False, mechanical_center_from_tip_mm=5.,
                           mechanical_length_mm=2., mechanical_clear_bore_diameter_mm=.8)
    result = _result(gun_bores=(body,))
    outline, = plane_hardware_outlines(result, 5., circle_segments=16)
    assert outline.kind == "gun_bore"
    assert outline.polylines_mm[0][0] == (.4, 0.)
    assert plane_hardware_outlines(result, 4.)
    assert plane_hardware_outlines(result, 6.)
    assert plane_hardware_outlines(result, 6.01) == ()


def test_filter_exit_frame_cannot_receive_global_column_boundaries():
    result = _result(walls=(_wall(0., 30., 2.),), detectors=(_detector(),))
    assert plane_hardware_outlines(result, 20., coordinate_frame="sector_exit_local") == ()


@pytest.mark.parametrize("z", [True, "10", None, math.nan, math.inf])
def test_selected_z_must_be_a_finite_numerical_position(z):
    with pytest.raises(ValueError):
        plane_hardware_outlines(_result(), z)


@pytest.mark.parametrize("sampling", [True, 0, 15, 1025, 16.5])
def test_circle_sampling_is_bounded_for_interactive_rendering(sampling):
    with pytest.raises(ValueError):
        plane_hardware_outlines(_result(), 10., circle_segments=sampling)


@pytest.mark.parametrize("record", [
    _slit(bore_diameter_mm=0.), _slit(slit_gap_mm=-1.),
    {**_aperture(), "diameter_mm": math.nan},
    {**_aperture(), "offset_x_mm": math.inf},
    {**_aperture(), "shape": "unknown"},
])
def test_invalid_captured_aperture_geometry_is_explicit(record):
    with pytest.raises(ValueError):
        plane_hardware_outlines(_result(apertures=(record,)), 10.)


def test_invalid_detector_annulus_and_wall_are_not_replaced_with_defaults():
    with pytest.raises(ValueError):
        plane_hardware_outlines(_result(detectors=(_detector(inner_diameter_mm=5.),)), 20.)
    with pytest.raises(ValueError):
        plane_hardware_outlines(_result(walls=(_wall(0., 20., -2.),)), 10.)


def test_upstream_projection_retains_actual_hardware_planes_and_excludes_downstream():
    result = _result(
        apertures=(_aperture(), _aperture(key="downstream_aperture", z_mm=30.)),
        detectors=(_detector("haadf", z_mm=15.), _detector("df", z_mm=20.),
                   _detector("bf", "disk", z_mm=25.)),
    )
    projected = projected_hardware_outlines(result, 22., circle_segments=16)
    assert [(row.key, row.z_mm) for row in projected] == [
        ("aperture", 10.), ("haadf", 15.), ("df", 20.)]
    assert all("Axial projection" in row.description for row in projected)
    assert all("not an effective cutoff" in row.description for row in projected)
    assert plane_hardware_outlines(result, 22.) == ()
    assert {row.key for row in projected_hardware_outlines(result, 25.)} == {
        "aperture", "haadf", "df", "bf"}


def test_projected_masks_reuse_local_geometry_and_keep_lab_offsets():
    result = _result(apertures=(_aperture(), _slit(key="slit")),
                     detectors=(_detector(), _detector("camera", "square")))
    projected = projected_hardware_outlines(result, 40., circle_segments=16)
    for row in projected:
        local = next(item for item in plane_hardware_outlines(result, row.z_mm, circle_segments=16)
                     if item.key == row.key)
        assert row.polylines_mm == local.polylines_mm
        assert row.role == local.role
    aperture = next(row for row in projected if row.key == "aperture")
    assert aperture.polylines_mm[0][0] == pytest.approx((.03, -.02))
    annulus = next(row for row in projected if row.key == "detector")
    assert annulus.polylines_mm[0][0] == pytest.approx((2.3, -.4))


def test_projected_masks_follow_insertion_and_readout_is_not_physical_removal():
    result = _result(
        apertures=(_aperture(enabled=False), _aperture(key="missing", installed=False),
                   _slit(key="slit", slit_inserted=False)),
        detectors=(_detector("retracted", inserted=False),
                   _detector("readout_off", readout_enabled=False)),
    )
    projected = projected_hardware_outlines(result, 30.)
    assert [row.key for row in projected] == ["slit", "readout_off"]
    assert len(projected[0].polylines_mm) == 1
    assert "physical absorption is still active" in projected[1].description


def test_projected_column_walls_deduplicate_one_body_without_filling_axial_gaps():
    walls = (_wall(0., 10., 4., "body"), _wall(12., 20., 4., "body"),
             _wall(30., 40., 2., "later"))
    result = _result(walls=walls)
    projected, = projected_hardware_outlines(result, 25., circle_segments=16)
    assert projected.key == "body"
    assert projected.z_mm == 0.
    assert projected.polylines_mm[0][0] == (2., 0.)
    assert "0\u201310; 12\u201320" in projected.description
    assert "Column vacuum wall" in projected.name
    assert plane_hardware_outlines(result, 11.) == ()


def test_projected_column_uses_functional_name_but_preserves_raw_identity():
    wall = _wall(0., 30., 20., "module_id")
    wall.name = "project_and_recording_system:noenergyfilter vacuum drift"
    outline, = projected_hardware_outlines(_result(walls=(wall,)), 20.)
    assert outline.name == "Column vacuum wall"
    assert wall.name in outline.description
    assert "key module_id" in outline.description


def test_projected_gun_body_keeps_upstream_extent_even_after_leaving_body():
    body = SimpleNamespace(key="gun_lens", name="Gun lens body", installed=True,
                           enabled=False, mechanical_center_from_tip_mm=5.,
                           mechanical_length_mm=2., mechanical_clear_bore_diameter_mm=.8)
    result = _result(gun_bores=(body,))
    assert projected_hardware_outlines(result, 3.) == ()
    outline, = projected_hardware_outlines(result, 8., circle_segments=16)
    assert outline.z_mm == 4.
    assert outline.polylines_mm[0][0] == (.4, 0.)
    assert "4\u20136 mm" in outline.description
    assert plane_hardware_outlines(result, 8.) == ()


def test_projected_geometry_requires_straight_column_frame():
    result = _result(detectors=(_detector(),))
    assert projected_hardware_outlines(result, 30., coordinate_frame="sector_exit_local") == ()


def test_live_geometry_snapshot_is_detached_and_allocates_no_execution_state(monkeypatch):
    from temsim.optics.column import default_state
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim import simulation_pipeline

    state = default_state()
    original_digest = capture_instrument_snapshot(state).digest
    monkeypatch.setattr(simulation_pipeline, "calculate",
                        lambda *_args, **_kwargs: pytest.fail("Display geometry ran transport"))
    monkeypatch.setattr(simulation_pipeline, "run",
                        lambda *_args, **_kwargs: pytest.fail("Display geometry ran transport"))
    snapshot = hardware_geometry_snapshot(state)
    assert snapshot.assembly is state._resolved_assembly
    assert not hasattr(snapshot, "simulation")
    assert not hasattr(snapshot.state_snapshot, "lenses")
    before = projected_hardware_outlines(snapshot, 2822.04)
    assert capture_instrument_snapshot(state).digest == original_digest
    detector = state.dark_field_detector
    detector.inserted = False
    detector.centre_offset_x_mm = .8
    aperture = state.condenser_aperture_2
    aperture.diameter_mm = .5
    assert projected_hardware_outlines(snapshot, 2822.04) == before
    fresh = hardware_geometry_snapshot(state)
    assert "df" not in {row.key for row in projected_hardware_outlines(fresh, 2822.04)
                        if row.kind == "detector"}
    assert capture_instrument_snapshot(state).digest != original_digest
    projected_hardware_outlines(hardware_geometry_snapshot(state), 2822.04)


def test_actual_default_projection_at_2822_contains_upstream_haadf_and_df_only():
    from temsim.optics.column import default_state

    snapshot = hardware_geometry_snapshot(default_state())
    projected = projected_hardware_outlines(snapshot, 2822.04)
    detectors = {row.key: row for row in projected if row.kind == "detector"}
    # Independent optical surfaces from the current recording-system manifest:
    # HAADF 2594.15; screen 2714.15; DF 2804.15; BF 2874.15; camera 2986.65 mm.
    assert set(detectors) == {"haadf", "flu_screen", "df"}
    assert detectors["haadf"].z_mm == pytest.approx(2594.15)
    assert detectors["df"].z_mm == pytest.approx(2804.15)
    assert detectors["haadf"].polylines_mm[0][0] == (11., 0.)
    assert detectors["haadf"].polylines_mm[1][0] == (2., 0.)
    assert detectors["df"].polylines_mm[0][0] == (7., 0.)
    assert detectors["df"].polylines_mm[1][0] == (1., 0.)
    local = plane_hardware_outlines(snapshot, 2822.04)
    assert all(row.kind == "column" for row in local)
    assert len(projected) < len(snapshot.assembly.vacuum_bore_segments)
