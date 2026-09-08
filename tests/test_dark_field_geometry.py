"""Small independent ray fixtures; no source tracing or wave calculation."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from temsim.physics.dark_field_geometry import propose_dark_field_geometry
from temsim.physics.first_order import TransverseTransfer
from temsim.physics.record_plane import (
    PlaneStop,
    RecordPlanePlan,
    project_sample_phase_space,
    route_record_planes,
)


def _plan(*, matrix=None, magnification=None, affine=(0, 0), centre=(0, 0), outer=14):
    plane = PlaneStop(
        "df", "DF", 100, "detector", "annulus", inner_diameter_mm=2,
        outer_width_mm=outer, offset_x_mm=centre[0], offset_y_mm=centre[1],
        readout_enabled=True,
    )
    transfer = TransverseTransfer(
        0, 100, np.eye(2) if magnification is None else magnification,
        np.eye(2) if matrix is None else matrix, np.zeros((2, 2)), np.eye(2),
        position_offset_m=affine,
    )
    return RecordPlanePlan(0, (plane,), (transfer,), "a" * 64, "b" * 64)


def _apply(plan, proposal):
    assert proposal.supported, proposal.detail
    return replace(plan, planes=tuple(
        replace(p, inner_diameter_mm=proposal.inner_diameter_mm,
                outer_width_mm=proposal.outer_width_mm) if p.key == "df" else p
        for p in plan.planes
    ))


def _angles(radii_mrad, chief=(0, 0)):
    phi = np.arange(181) * (2 * np.pi / 181)
    unit = np.stack((np.cos(phi), np.sin(phi)), axis=-1)
    return (np.asarray(radii_mrad)[:, None, None] * unit[None, :, :] + chief).reshape(-1, 2) * 1e-3


def test_camera_length_changes_overlap_and_proposed_dimensions():
    positions = np.zeros((1, 2))
    angles = _angles([0, 5, 15, 24.8])
    long = _plan(matrix=np.eye(2), outer=14)
    short = _plan(matrix=0.03 * np.eye(2), outer=14)
    assert route_record_planes(long, positions, angles).interactions[0].signal_weight > 0
    assert route_record_planes(short, positions, angles).interactions[0].signal_weight == 0
    long_proposal = propose_dark_field_geometry(long, positions, 25, target_inner_mrad=30, target_outer_mrad=50)
    short_proposal = propose_dark_field_geometry(short, positions, 25, target_inner_mrad=30, target_outer_mrad=50)
    assert long_proposal.camera_length_min_m == pytest.approx(1)
    assert short_proposal.camera_length_max_m == pytest.approx(0.03)
    assert long_proposal.inner_diameter_mm == pytest.approx(short_proposal.inner_diameter_mm / 0.03)
    assert long_proposal.outer_width_mm == pytest.approx(short_proposal.outer_width_mm / 0.03)
    for plan, proposal in ((long, long_proposal), (short, short_proposal)):
        routed = route_record_planes(_apply(plan, proposal), positions, _angles([0, 5, 15, 25]))
        assert routed.interactions[0].signal_weight == 0
        assert proposal.direct_disk_clearance_mm > 0


def test_anisotropic_signed_map_static_timed_scan_and_chief_offsets_exclude_disk():
    positions = np.array([[[-3e-5, -2e-5], [3e-5, -2e-5]], [[-3e-5, 2e-5], [3e-5, 2e-5]]])
    timed = np.array([[[1e-4, -2e-4], [-1e-4, 0]], [[0, 3e-4], [2e-4, 0]]])
    plan = replace(
        _plan(matrix=np.array([[0.9, -0.08], [0.12, -1.0]]),
              magnification=np.array([[8, -3], [4, 11]]),
              affine=(2e-4, -1e-4), centre=(0.3, -0.4)),
        scan_times_s=np.arange(4).reshape(2, 2), scan_position_offsets_m=(timed,),
        time_dependent_deflection=True,
    )
    chief = (0.7, -0.5)
    proposal = propose_dark_field_geometry(
        plan, positions, 25, target_inner_mrad=30, target_outer_mrad=50,
        probe_center_mrad=chief, chamber_inner_diameter_mm=110,
    )
    sized = _apply(plan, proposal)
    angles = _angles([0, 5, 15, 25, 29.9, 36, 40, 50.1, 65], chief)
    flat_positions = positions.reshape(-1, 2)
    result = route_record_planes(sized, flat_positions[:, None, :], angles[None, :, :])
    hits = result.interactions[0].signal_mask.reshape(4, 9, 181)
    assert not hits[:, :5].any()
    assert hits[:, 5:7].any()
    assert not hits[:, 7:].any()
    projected_chief = project_sample_phase_space(plan.transfers[0], flat_positions, np.asarray(chief) * 1e-3).position_m
    displaced = projected_chief + timed.reshape(-1, 2) - np.array((0.3, -0.4)) * 1e-3
    expected_maximum = np.linalg.norm(displaced, axis=-1).max() * 1e3
    assert proposal.maximum_affine_shift_mm == pytest.approx(expected_maximum)
    assert proposal.angular_inner_mrad == pytest.approx(30)
    assert proposal.angular_outer_mrad == pytest.approx(50)
    assert result.balance_error == 0
    # Public hit mask also excludes direct disk without any upstream stop.
    coords = result.interactions[0].projected_position_m.reshape(4, 9, 181, 2)
    assert not sized.planes[0].hit_mask(coords[:, :4, :, 0], coords[:, :4, :, 1]).any()
    assert plan.planes[0].inner_diameter_mm == 2
    with pytest.raises(FrozenInstanceError):
        proposal.inner_diameter_mm = 1


def test_haadf_shadow_does_not_reject_wider_df_or_restore_shadowed_signal():
    plan = _plan()
    haadf = PlaneStop("haadf", "HAADF", 50, "detector", "annulus",
                      inner_diameter_mm=120, outer_width_mm=660, readout_enabled=True)
    upstream = replace(plan.transfers[0], target_z_mm=50)
    plan = replace(plan, planes=(haadf,) + plan.planes, transfers=(upstream,) + plan.transfers)
    proposal = propose_dark_field_geometry(plan, [[0, 0]], 25, target_inner_mrad=30,
                                          target_outer_mrad=100, chamber_inner_diameter_mm=220)
    assert proposal.supported
    assert proposal.target_outer_mrad == 100
    result = route_record_planes(_apply(plan, proposal), [[0, 0]], _angles([0, 25, 40, 80]))
    first, second = result.interactions
    haadf_hits = first.signal_mask.reshape(4, 181)
    df_hits = second.signal_mask.reshape(4, 181)
    assert not df_hits[:2].any()
    assert df_hits[2].all()
    assert not df_hits[3].any()
    assert haadf_hits[3].all()
    assert first.signal_weight + second.signal_weight + result.surviving_weight == result.initial_weight


def test_default_keeps_sufficient_existing_outer_diameter():
    plan = _plan(matrix=np.eye(2) * 0.06)
    proposal = propose_dark_field_geometry(plan, [[0, 0]], 24.8, chamber_inner_diameter_mm=20)
    assert proposal.inner_diameter_mm == pytest.approx(2 * 0.06 * 29.8)
    assert proposal.outer_width_mm == pytest.approx(14)
    assert proposal.target_outer_mrad == pytest.approx(7 / 0.06)
    assert "retained" in proposal.detail


def test_default_enlarges_outer_only_when_new_inner_hole_needs_it():
    proposal = propose_dark_field_geometry(_plan(), [[0, 0]], 24.8, chamber_inner_diameter_mm=180)
    assert proposal.inner_diameter_mm == pytest.approx(59.6)
    assert proposal.outer_width_mm == pytest.approx(99.6)
    assert proposal.target_outer_mrad == pytest.approx(49.8)


def test_explicit_outer_is_respected_even_when_existing_diameter_is_wider():
    proposal = propose_dark_field_geometry(_plan(outer=160), [[0, 0]], 25,
                                          target_outer_mrad=50)
    assert proposal.outer_width_mm == pytest.approx(100)


@pytest.mark.parametrize("changes, expected", [
    ({"target_inner_mrad": 29}, "margin"),
    ({"target_outer_mrad": 29}, "exceed"),
    ({"chamber_inner_diameter_mm": 90}, "chamber"),
])
def test_infeasible_request_is_explicit_and_not_silently_clipped(changes, expected):
    proposal = propose_dark_field_geometry(_plan(), [[0, 0]], 25, **changes)
    assert not proposal.supported
    assert expected in proposal.detail
    assert proposal.inner_diameter_mm is None


def test_chamber_bound_includes_off_axis_detector_body():
    plan = _plan(centre=(4, 0))
    # Angular outer 54 mrad gives Rout=50 mm after D=4 mm.  Its off-axis
    # body extends to 54 mm, so a 100 mm chamber is too small.
    proposal = propose_dark_field_geometry(plan, [[0, 0]], 25, target_outer_mrad=54,
                                          chamber_inner_diameter_mm=100)
    assert not proposal.supported
    assert "108 mm" in proposal.detail


@pytest.mark.parametrize("matrix", [np.diag([1., 0.]), np.diag([1., 0.1])])
def test_singular_or_excessively_anisotropic_map_is_not_treated_as_scalar_camera(matrix):
    proposal = propose_dark_field_geometry(_plan(matrix=matrix), [[0, 0]], 25,
                                          target_outer_mrad=50)
    assert not proposal.supported
    assert "singular" in proposal.detail or "anisotropy" in proposal.detail


def test_timed_plan_requires_complete_matching_raster():
    plan = replace(_plan(), time_dependent_deflection=True)
    assert not propose_dark_field_geometry(plan, [[0, 0]], 25).supported
    plan = replace(plan, scan_times_s=np.zeros((2, 2)))
    assert not propose_dark_field_geometry(plan, [[0, 0]], 25).supported
    assert propose_dark_field_geometry(plan, np.zeros((4, 2)), 25).supported


@pytest.mark.parametrize("kwargs", [
    {"scan_positions_m": []}, {"scan_positions_m": [[np.nan, 0]]},
    {"probe_semiangle_mrad": -1}, {"probe_semiangle_mrad": np.inf},
    {"probe_center_mrad": [0]}, {"target_outer_mrad": np.nan},
    {"chamber_inner_diameter_mm": 0},
])
def test_invalid_input_cannot_produce_a_geometry_proposal(kwargs):
    arguments = {"scan_positions_m": [[0, 0]], "probe_semiangle_mrad": 25} | kwargs
    with pytest.raises(ValueError):
        propose_dark_field_geometry(_plan(), **arguments)


def test_missing_or_nonannular_detector_is_explicitly_unsupported():
    plan = _plan()
    assert not propose_dark_field_geometry(plan, [[0, 0]], 25, detector_key="missing").supported
    plan = replace(plan, planes=(replace(plan.planes[0], geometry="disk"),))
    assert not propose_dark_field_geometry(plan, [[0, 0]], 25).supported
