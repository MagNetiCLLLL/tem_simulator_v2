from types import SimpleNamespace
import math

import numpy as np
import pytest

from temsim.physics.stem_sampling import (
    detector_angular_bounds, detector_sampling_report, frame_sampling_report,
)
from temsim.physics.stem_wave_imaging import AngularDetector


def report(bounds=None, **kwargs):
    params = dict(maximum_angle_mrad=42., wavelength_angstrom=.019687,
                  requested_fov_angstrom=40., requested_grid_pixels=256,
                  bandwidth_fraction=2/3, probe_semiangle_mrad=25.,
                  potential_storage_bytes=1024**2)
    params.update(kwargs)
    return detector_sampling_report(
        bounds or {"bf": (0., 10.4), "df": (16., 112.), "haadf": (60., 331.)}, **params)


def test_three_detector_coverage_and_realistic_grid_proposal():
    result = report()
    assert [result["detectors"][k]["status"] for k in ("bf", "df", "haadf")] == ["full", "partial", "outside"]
    assert result["detectors"]["df"]["overlaps_illumination_disk"]
    assert not result["detectors"]["haadf"]["overlaps_illumination_disk"]
    assert not result["coverage_complete"]
    pixels = result["recommended_grid_pixels"]
    assert 2000 < pixels < 2100
    # Verify the remedy with the sampling formula, not a hard-coded pixel count.
    supported = 1000 * (2/3) * .019687 * pixels / (2 * 40)
    assert supported > 331
    assert result["grid_area_factor"] == (pixels / 256)**2
    assert result["estimated_potential_bytes"] > 60 * 1024**2


def test_illumination_failure_is_not_a_valid_bf_image():
    result = report({"bf": (0., 10.)}, probe_semiangle_mrad=60.)
    assert result["detectors"]["bf"]["status"] == "full"
    assert not result["illumination_covered"]
    assert not result["coverage_complete"]


def test_boundary_conservative_and_unknown_transfer_are_explicit():
    result = report({"bf": (0., math.sin(.042)*1000), "image": (0., math.inf)})
    assert result["detectors"]["bf"]["status"] == "full"
    assert result["detectors"]["image"]["status"] == "unknown"
    assert result["recommended_grid_pixels"] is None


def test_record_plane_bounds_include_anisotropy_raster_and_affine_offsets_once():
    plane = SimpleNamespace(key="df", kind="detector", geometry="annulus",
                            inner_diameter_mm=1., outer_width_mm=2.,
                            offset_x_mm=1., offset_y_mm=0.)
    transfer = SimpleNamespace(j_diff_m_per_rad=np.diag([.01, .02]),
                               j_img=np.eye(2), position_offset_m=np.array([.0005, 0.]))
    plan = SimpleNamespace(planes=[plane], transfers=[transfer])
    result = detector_angular_bounds(
        [AngularDetector("df", 30., 70.)], record_plane_plan=plan,
        positions_m=np.array([[0., 0.], [-.001, 0.]]),
        detector_center_shifts_mrad={"df": (np.array([999.]), np.array([0.]))},
    )
    assert result["df"] == pytest.approx((0., 250.))
    assert report(result)["detectors"]["df"]["status"] == "partial"


def test_square_detector_requires_corners_not_just_half_width():
    physical = SimpleNamespace(geometry="square", inner_diameter_mm=0., outer_width_mm=2.)
    detector = SimpleNamespace(key="bf", detector=physical, sample_to_detector_m_per_rad=np.eye(2))
    bounds = detector_angular_bounds([detector], positions_m=np.zeros((1, 2)))
    assert bounds["bf"] == pytest.approx((0., math.sqrt(2)))


def test_legacy_wave_results_are_unchecked_not_implicitly_full():
    assert frame_sampling_report({"model": "multislice_angle_resolved"})["legacy_unchecked"]
    assert frame_sampling_report({"model": "geometric_detector_interception"}) is None


@pytest.mark.parametrize("kwargs", [dict(maximum_angle_mrad=float("nan")),
                                    dict(wavelength_angstrom=0.), dict(bandwidth_fraction=2.),
                                    dict(probe_semiangle_mrad=10000.)])
def test_invalid_sampling_rejected(kwargs):
    with pytest.raises(ValueError):
        report(**kwargs)


def test_grid_change_reuses_unaffected_products():
    from temsim.optics.column import default_state
    from temsim.calculation_cache import calculation_signatures
    state = default_state()
    before = calculation_signatures(state)
    state.sample.wave_grid_pixels = 2048
    after = calculation_signatures(state)
    for key in ("column", "incident", "elastic", "eds", "sample_region"):
        assert after[key] == before[key], key
    for key in ("stem", "wave", "fourdstem_cube"):
        assert after[key] != before[key], key


def test_coordinate_schema_does_not_invalidate_incident_or_eds(monkeypatch):
    from temsim.optics.column import default_state
    from temsim import calculation_cache
    state = default_state()
    before = calculation_cache.calculation_signatures(state)
    monkeypatch.setattr(calculation_cache, "_WAVE_COORDINATE_SCHEMA", "old-test-schema")
    after = calculation_cache.calculation_signatures(state)
    for key in ("column", "incident", "elastic", "eds", "sample_region"):
        assert before[key] == after[key]
    for key in ("stem", "wave", "wave_source", "fourdstem_cube"):
        assert before[key] != after[key]


@pytest.mark.parametrize("camera_length, chief_mrad, centre_mm, expected_overlap", [
    (1., 0., 0., False),
    (1., 7., 0., True),
    (.25, 7., 0., False),
    (3., 7., 0., False),
    (1., 7., 7., False),
    (-1., 7., -14., True),
])
def test_direct_disk_overlap_matches_physical_routing_for_tilt_camera_and_offset(
        camera_length, chief_mrad, centre_mm, expected_overlap):
    from temsim.physics.first_order import TransverseTransfer
    from temsim.physics.record_plane import PlaneStop, RecordPlanePlan, route_record_planes

    plane = PlaneStop("df", "DF", 100., "detector", "annulus",
                      inner_diameter_mm=8., outer_width_mm=20.,
                      offset_x_mm=centre_mm, readout_enabled=True)
    transfer = TransverseTransfer(0., 100., np.zeros((2, 2)),
                                 np.eye(2) * camera_length, np.zeros((2, 2)), np.eye(2))
    plan = RecordPlanePlan(0., (plane,), (transfer,), "a" * 64, "b" * 64)
    positions = np.zeros((1, 2))
    detectors = [SimpleNamespace(key="df")]
    axis_bounds = detector_angular_bounds(detectors, positions_m=positions, record_plane_plan=plan)
    chief_bounds = detector_angular_bounds(detectors, positions_m=positions, record_plane_plan=plan,
                                          angular_origin_mrad=(chief_mrad, 0.))
    result = report(axis_bounds, probe_semiangle_mrad=2.,
                    probe_center_mrad=(chief_mrad, 0.), illumination_bounds_mrad=chief_bounds)
    phi = np.arange(180) * 2 * np.pi / 180
    disk = (np.linspace(0., 2., 21)[:, None, None]
            * np.stack((np.cos(phi), np.sin(phi)), axis=-1)).reshape(-1, 2)
    angles = (disk + [chief_mrad, 0.]) * 1e-3
    routed = route_record_planes(plan, positions, angles)
    assert bool(np.any(routed.interactions[0].signal_mask)) is expected_overlap
    assert result["detectors"]["df"]["overlaps_illumination_disk"] is expected_overlap
    # Reciprocal coverage must retain the absolute angular acceptance.
    assert result["detectors"]["df"]["required_inner_mrad"] == axis_bounds["df"][0]


def test_tilted_pupil_outside_fft_requires_more_sampling_even_with_covered_detector():
    result = report({"bf": (0., 10.)}, probe_semiangle_mrad=5., probe_center_mrad=(40., 0.),
                    illumination_bounds_mrad={"bf": (30., 50.)})
    assert result["detectors"]["bf"]["status"] == "full"
    assert not result["illumination_covered"]
    assert not result["coverage_complete"]
    assert result["illumination_extent_mrad"] > 45.
    assert result["recommended_grid_pixels"] > 256


def test_probe_relative_bounds_include_timed_raster_offsets_only_once():
    plane = SimpleNamespace(key="df", kind="detector", geometry="annulus",
                            inner_diameter_mm=8., outer_width_mm=20.,
                            offset_x_mm=0., offset_y_mm=0.)
    transfer = SimpleNamespace(j_diff_m_per_rad=np.eye(2), j_img=np.eye(2),
                               position_offset_m=np.zeros(2))
    positions = np.array([[0., 0.], [-.002, 0.]])
    plan = SimpleNamespace(planes=[plane], transfers=[transfer],
                           scan_position_offsets_m=(np.array([[0., 0.], [.009, 0.]]),))
    bounds = detector_angular_bounds(
        [SimpleNamespace(key="df")], positions_m=positions, record_plane_plan=plan,
        angular_origin_mrad=(-7., 0.),
        detector_center_shifts_mrad={"df": (np.array([999., 999.]), np.zeros(2))})
    # The first scan position accepts the chief; the second centres the bore on it.
    assert bounds["df"] == pytest.approx((0., 17.))


def test_legacy_shifted_angular_detector_can_miss_a_tilted_disk_entirely():
    bounds = detector_angular_bounds(
        [AngularDetector("df", 4., 10.)], positions_m=np.zeros((1, 2)),
        detector_center_shifts_mrad={"df": (np.array([-7.]), np.array([0.]))},
        angular_origin_mrad=(7., 0.))
    assert bounds["df"] == pytest.approx((4., 24.))


def test_nonzero_probe_chief_requires_matching_overlap_coordinate_frame():
    with pytest.raises(ValueError, match="probe chief"):
        report(probe_center_mrad=(7., 0.))
