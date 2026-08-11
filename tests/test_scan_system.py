from types import SimpleNamespace

import numpy as np
import pytest

from temsim.detector.stem_signal import DetectorSignal, StemScanResult
from temsim.detector import stem_signal
from temsim.gui.scan_panel import ScanControlView
from temsim.gui.visualization import VisualizationWorkspace
from temsim.optics.ac_deflector import create_ac_deflector
from temsim.optics.column import default_state
from temsim.optics.descan_deflector import create_descan_deflector
from temsim.physics import scan_geometry
from temsim.physics.scan_geometry import (
    ScanGeometryResult,
    ScanRayPathResult,
)


def test_ac_lower_foil_is_derived_from_one_pure_shift_pair_gain():
    component = create_ac_deflector()
    component.wobble_enabled = False
    component.set_pure_shift_coupling(
        ((-1.25, 0.1), (-0.1, -1.25)),
        1.0e-12,
    )
    component.upper_coil_gain = 0.4
    component.kick_x_mrad = 2.0
    component.kick_y_mrad = -3.0

    assert component.lower_coil_gain == pytest.approx(-0.5)
    upper, lower = component.coil_kicks_mrad(2.0, -3.0)
    assert upper == pytest.approx((0.8, -1.2))
    assert lower == pytest.approx((-1.12, 1.42))
    events = component.kick_events()
    assert events[0][1:] == pytest.approx((0.8e-3, -1.2e-3))
    assert events[1][1:] == pytest.approx((-1.12e-3, 1.42e-3))


def test_ac_pure_shift_calibration_cancels_sample_angle(monkeypatch):
    component = create_ac_deflector()
    component.wobble_enabled = False
    component.scan_enabled = True
    upper_angle = np.array(((2.0, 0.4), (-0.4, 2.0)))
    lower_angle = np.array(((1.5, -0.2), (0.2, 1.5)))

    def phase_response(_state, start_z_mm, _observation_z_mm):
        angle = (
            upper_angle
            if float(start_z_mm) == pytest.approx(component.upper_z_mm)
            else lower_angle
        )
        return np.zeros((2, 2)), angle

    monkeypatch.setattr(
        scan_geometry,
        "transverse_kick_phase_space_response",
        phase_response,
    )
    state = SimpleNamespace(
        ac_deflector=component,
        sample=SimpleNamespace(z_mm=component.lower_z_mm + 10.0),
    )

    coupling, residual = scan_geometry.calibrate_ac_pure_shift(state)

    assert upper_angle + lower_angle @ coupling == pytest.approx(
        np.zeros((2, 2)),
        abs=1.0e-12,
    )
    assert residual < 1.0e-12


def test_paired_response_uses_both_planes_and_cross_axis_coupling(monkeypatch):
    component = SimpleNamespace(
        upper_z_mm=1.0,
        lower_z_mm=2.0,
        coil_kick_matrices=lambda: (
            ((0.5, 0.0), (0.0, 0.5)),
            ((-0.6, 0.2), (-0.2, -0.6)),
        ),
    )
    upper_response = np.array(((2.0, 0.0), (0.0, 3.0)))
    lower_response = np.array(((5.0, 1.0), (-1.0, 4.0)))

    monkeypatch.setattr(
        scan_geometry,
        "transverse_kick_response",
        lambda _state, start, _stop: (
            upper_response if float(start) == 1.0 else lower_response
        ),
    )

    actual = scan_geometry.paired_kick_response(None, component, 3.0)
    expected = (
        upper_response @ np.asarray(component.coil_kick_matrices()[0])
        + lower_response @ np.asarray(component.coil_kick_matrices()[1])
    )
    assert actual == pytest.approx(expected)


def test_descan_uses_opposite_ac_command_and_cancels_at_image_reference(
    monkeypatch,
):
    ac = create_ac_deflector()
    ac.wobble_enabled = False
    ac.scan_enabled = True
    ac.z_mm = 10.0
    ac.set_pure_shift_coupling(
        ((-1.1, 0.2), (-0.2, -1.1)),
        0.0,
    )
    ac.set_scan_command_matrix_mrad(
        ((0.4, -0.1), (0.2, 0.5)),
        0.0,
    )
    descan = create_descan_deflector()
    object.__setattr__(descan, "z_mm", 30.0)
    object.__setattr__(descan, "optical_reference_z_mm", 30.0)
    object.__setattr__(descan, "mechanical_center_below_sample_mm", 20.0)
    descan.scan_enabled = True
    responses = {
        float(ac.upper_z_mm): np.array(((2.0, 0.3), (-0.3, 2.0))),
        float(ac.lower_z_mm): np.array(((1.5, -0.2), (0.2, 1.5))),
        float(descan.upper_z_mm): np.array(((1.2, 0.1), (-0.1, 1.2))),
        float(descan.lower_z_mm): np.array(((0.8, -0.05), (0.05, 0.8))),
    }
    monkeypatch.setattr(
        scan_geometry,
        "transverse_kick_response",
        lambda _state, start, _stop: responses[float(start)],
    )
    monkeypatch.setattr(
        scan_geometry,
        "trace_transverse_transfer",
        lambda *_args: SimpleNamespace(
            j_img=np.eye(2),
            j_diff_m_per_rad=np.zeros((2, 2)),
        ),
    )
    state = SimpleNamespace(
        ac_deflector=ac,
        descan_deflector=descan,
        sample=SimpleNamespace(z_mm=10.0),
        selected_area_aperture=SimpleNamespace(
            key="selected_area_aperture",
            name="Selected Area Aperture",
            z_mm=50.0,
        ),
    )

    result = scan_geometry.calibrate_descan_image_plane(state)
    ac_response = scan_geometry.paired_kick_response(state, ac, 50.0)
    descan_response = scan_geometry.paired_kick_response(
        state,
        descan,
        50.0,
    )

    assert np.asarray(descan.scan_command_matrix_mrad) == pytest.approx(
        -np.asarray(ac.scan_command_matrix_mrad)
    )
    assert descan_response == pytest.approx(ac_response, abs=1.0e-12)
    assert result.response_match_residual < 1.0e-12
    assert result.plane_kind == "image"


def test_default_column_scan_descan_symmetry_and_optical_cancellation():
    state = default_state()
    state.ac_deflector.wobble_enabled = False
    state.ac_deflector.scan_enabled = True
    state.descan_deflector.scan_enabled = True

    command, _, calibration = scan_geometry.calibrate_scan_system(state)
    ac_response = scan_geometry.paired_kick_response(
        state,
        state.ac_deflector,
        calibration.target_z_mm,
    )
    descan_response = scan_geometry.paired_kick_response(
        state,
        state.descan_deflector,
        calibration.target_z_mm,
    )
    combined = (
        ac_response @ np.asarray(command)
        + descan_response
        @ np.asarray(state.descan_deflector.scan_command_matrix_mrad)
    )

    assert state.sample.z_mm - state.ac_deflector.z_mm == pytest.approx(
        state.descan_deflector.z_mm - state.sample.z_mm
    )
    assert np.asarray(
        state.descan_deflector.scan_command_matrix_mrad
    ) == pytest.approx(-np.asarray(command))
    assert ac_response == pytest.approx(descan_response, rel=1.0e-10)
    assert np.linalg.norm(combined) < 1.0e-15

    geometry = scan_geometry.calculate_scan_geometry(state)
    target_x_um, target_y_um = geometry.plane_positions_um[
        calibration.target_key
    ]
    assert np.ptp(geometry.sample_x_um) == pytest.approx(
        31.0e-3,
        rel=1.0e-5,
    )
    assert np.ptp(geometry.sample_y_um) == pytest.approx(
        31.0e-3,
        rel=1.0e-5,
    )
    assert np.ptp(target_x_um) < 1.0e-6
    assert np.ptp(target_y_um) < 1.0e-6
    assert geometry.plane_roles["objective_image_plane"] == "image"
    assert "current objective/sample state" in geometry.plane_names[
        "objective_image_plane"
    ]
    assert "physical diffraction-reference station" in (
        geometry.plane_names["objective_aperture"]
    )
    assert "physical image-reference station" in (
        geometry.plane_names["selected_area_aperture"]
    )


@pytest.mark.parametrize(
    ("j_img", "j_diff", "expected"),
    (
        (np.eye(2), np.zeros((2, 2)), "image"),
        (np.zeros((2, 2)), np.eye(2), "diffraction"),
        (np.eye(2), np.eye(2), "mixed"),
    ),
)
def test_plane_classification_distinguishes_image_diffraction_and_mixed(
    j_img,
    j_diff,
    expected,
):
    kind, _, _ = scan_geometry.classify_sample_plane_transfer(
        SimpleNamespace(j_img=j_img, j_diff_m_per_rad=j_diff)
    )

    assert kind == expected


def test_pixel_pitch_calibrates_axis_aligned_fov_through_active_optics(
    monkeypatch,
):
    component = create_ac_deflector()
    component.wobble_enabled = False
    component.scan_enabled = True
    component.scan_pixels_x = 4
    component.scan_lines = 2
    component.scan_pixel_size_nm = 2.0
    component.set_pure_shift_coupling(-np.eye(2), 0.0)
    response_m_per_rad = np.diag((2.0, 4.0))
    monkeypatch.setattr(
        scan_geometry,
        "paired_kick_response",
        lambda *_args, **_kwargs: response_m_per_rad,
    )
    state = SimpleNamespace(
        ac_deflector=component,
        sample=SimpleNamespace(z_mm=100.0),
    )

    command_mrad, residual = scan_geometry.calibrate_ac_scan_scale(state)
    factors_x, factors_y, times_s = scan_geometry.raster_sample_grid(
        component,
        maximum_count=None,
    )
    commands_mrad = np.asarray([
        component.scan_kick_mrad(value)
        for value in times_s.ravel()
    ]).reshape(*times_s.shape, 2)
    positions_m = np.einsum(
        "ij,...j->...i",
        response_m_per_rad,
        commands_mrad * 1.0e-3,
    )

    assert command_mrad == pytest.approx(
        np.diag((2.0e-6, 0.5e-6)),
        rel=1.0e-12,
        abs=1.0e-18,
    )
    assert residual < 1.0e-15
    assert factors_x[0] == pytest.approx((-0.75, -0.25, 0.25, 0.75))
    assert factors_y[:, 0] == pytest.approx((-0.5, 0.5))
    assert np.diff(positions_m[0, :, 0]) == pytest.approx(
        np.full(3, 2.0e-9),
        rel=1.0e-12,
        abs=1.0e-18,
    )
    assert np.diff(positions_m[:, 0, 1]) == pytest.approx(
        (2.0e-9,),
        rel=1.0e-12,
        abs=1.0e-18,
    )
    assert component.scan_field_of_view_x_nm == pytest.approx(8.0)
    assert component.scan_field_of_view_y_nm == pytest.approx(4.0)


@pytest.mark.parametrize(
    "component_factory",
    (create_ac_deflector, create_descan_deflector),
)
@pytest.mark.parametrize("pixel_size_nm", (0.0009, 1.0e6 + 1.0))
def test_scan_pixel_size_rejects_values_outside_pm_to_mm_range(
    component_factory,
    pixel_size_nm,
):
    component = component_factory()
    component.scan_pixel_size_nm = pixel_size_nm

    with pytest.raises(ValueError, match="pixel size"):
        component.validate()


def test_scan_view_exposes_pixel_pitch_and_derived_fov(qtbot):
    component = create_ac_deflector()
    descan = create_descan_deflector()
    state = SimpleNamespace(
        ac_deflector=component,
        descan_deflector=descan,
        sample=SimpleNamespace(stem_wave_enabled=False),
    )
    view = ScanControlView()
    qtbot.addWidget(view)

    view.set_state(state)

    assert set(view.ac_controls) == set(view.descan_controls)
    assert "scan_pixel_size_nm" in view.ac_controls
    assert "scan_amplitude_x_mrad" not in view.ac_controls
    assert "scan_amplitude_x_mrad" not in view.descan_controls
    assert not view.ac_controls["lower_coil_gain"].isEnabled()
    assert not view.descan_controls["lower_coil_gain"].isEnabled()
    assert view.ac_fov_x.text() == "32 nm"
    assert view.ac_fov_y.text() == "32 nm"
    assert view.descan_fov_x.text() == "32 nm"
    assert view.descan_fov_y.text() == "32 nm"
    assert view.result_tabs.tabText(0) == "Geometry"
    assert view.result_tabs.tabText(1) == "Images"
    for image_view in view.detector_image_views.values():
        view_box = image_view.getViewBox()
        assert view_box.state["aspectLocked"] == pytest.approx(1.0)
        assert view_box.state["mouseEnabled"] == [True, True]


def test_detector_position_and_size_define_collection_angle(monkeypatch):
    response_m_per_rad = np.diag((2.0, 1.0))
    monkeypatch.setattr(
        stem_signal,
        "transverse_kick_response",
        lambda _state, sample_z, detector_z: (
            response_m_per_rad
            if (sample_z, detector_z) == (100.0, 300.0)
            else np.zeros((2, 2))
        ),
    )
    state = SimpleNamespace(sample=SimpleNamespace(z_mm=100.0))
    detector = SimpleNamespace(
        z_mm=300.0,
        inner_diameter_mm=2.0,
        outer_width_mm=10.0,
    )

    angle = stem_signal.collection_angle(state, detector)

    assert angle.inner_range_mrad == pytest.approx((0.5, 1.0))
    assert angle.outer_range_mrad == pytest.approx((2.5, 5.0))
    assert angle.inner_mrad == pytest.approx(np.sqrt(0.5))
    assert angle.outer_mrad == pytest.approx(np.sqrt(12.5))
    assert angle.anisotropic is True


def test_scan_view_replays_one_cached_detector_frame_until_stopped(qtbot):
    view = ScanControlView()
    qtbot.addWidget(view)
    playback_times = []
    view.playback_time_changed.connect(playback_times.append)
    view._state = SimpleNamespace(
        ac_deflector=SimpleNamespace(
            enabled=True,
            scan_enabled=True,
            scan_frame_period_s=0.2,
        )
    )
    images = {
        key: np.arange(12, dtype=float).reshape(3, 4) / 12.0
        for key in ("haadf", "df", "bf")
    }
    signals = {
        key: DetectorSignal(key, key.upper(), 0.25, 1.0, 2.0, 3.0, None)
        for key in images
    }
    frame = StemScanResult(
        scan_x_um=np.zeros((3, 4)),
        scan_y_um=np.zeros((3, 4)),
        fractions=images,
        detector_signals=signals,
        metrics={"model": "test", "scan_frame_period_s": 0.2},
    )
    geometry = ScanGeometryResult(
        times_s=np.zeros((3, 4)),
        sample_x_um=np.zeros((3, 4)),
        sample_y_um=np.zeros((3, 4)),
        plane_positions_um={},
        plane_names={},
        requested_pixels_x=4,
        requested_pixels_y=3,
        ac_enabled=True,
        descan_enabled=False,
        ac_drift_pivot_z_mm=None,
        descan_drift_pivot_z_mm=None,
        ac_lower_from_upper=-np.eye(2),
        ac_angular_residual=0.0,
    )

    view.display_result(geometry, frame)

    assert view._playback_timer.isActive()
    assert playback_times
    assert "Scanning continuously" in view.detector_playback_summary.text()
    view._state.ac_deflector.scan_enabled = False
    view._set_playback_active(False)
    assert not view._playback_timer.isActive()
    assert "last frame retained" in view.detector_playback_summary.text()


def test_stem_images_use_physical_pixel_edges_and_explain_geometry_preview(
    qtbot,
    tmp_path,
):
    from ase import Atoms
    from ase.io import write

    cif_path = tmp_path / "LaTi2O6.cif"
    write(
        cif_path,
        Atoms(
            "LaTi",
            positions=((0.0, 0.0, 0.0), (1.8, 0.0, 0.0)),
            cell=(4.0, 4.0, 4.0),
            pbc=True,
        ),
    )
    view = ScanControlView()
    qtbot.addWidget(view)
    view._state = SimpleNamespace(
        sample=SimpleNamespace(
            cif_path=str(cif_path),
            size_x_nm=2.0,
            size_y_nm=2.0,
        ),
        stem_detectors=(),
    )
    scan_x = np.tile(
        np.asarray((-1.5, -0.5, 0.5, 1.5)) * 1.0e-3,
        (3, 1),
    )
    scan_y = np.tile(
        (np.asarray((-1.0, 0.0, 1.0)) * 1.0e-3)[:, None],
        (1, 4),
    )
    images = {
        key: np.arange(12, dtype=float).reshape(3, 4)
        for key in ("haadf", "df", "bf")
    }
    frame = StemScanResult(
        scan_x_um=scan_x,
        scan_y_um=scan_y,
        fractions=images,
        detector_signals={},
        metrics={
            "model": "geometric_detector_interception",
            "scan_pixel_size_nm": 1.0,
            "scan_field_of_view_x_nm": 4.0,
            "scan_field_of_view_y_nm": 3.0,
        },
    )

    view._set_stem_frame(frame)

    image_item = view.detector_image_items["bf"]
    rectangle = image_item.mapRectToParent(image_item.boundingRect())
    assert rectangle.left() == pytest.approx(-2.0e-3)
    assert rectangle.right() == pytest.approx(2.0e-3)
    assert rectangle.top() == pytest.approx(-1.5e-3)
    assert rectangle.bottom() == pytest.approx(1.5e-3)
    assert "not a specimen STEM image" in view.image_model_notice.text()
    assert "LaTi2O6.cif" in view.image_model_notice.text()
    assert "atomic columns are undersampled" in view.image_model_notice.text()
    assert "outside the finite sample" in view.image_model_notice.text()


def test_ray_playback_reprojects_cached_scan_offset_without_retracing(qtbot):
    workspace = VisualizationWorkspace()
    qtbot.addWidget(workspace)
    branch = SimpleNamespace(
        name="incident",
        z=np.asarray((0.0, 1.0)),
        x=np.zeros((2, 1)),
        y=np.zeros((2, 1)),
        blocked_z=np.asarray((np.nan,)),
    )
    ac = SimpleNamespace(
        scan_kick_mrad=lambda time_s: (float(time_s), 0.0),
    )
    descan = SimpleNamespace(
        enabled=False,
        scan_enabled=False,
        scan_kick_mrad=lambda _time_s: (0.0, 0.0),
    )
    state = SimpleNamespace(ac_deflector=ac, descan_deflector=descan)
    response = np.zeros((2, 2, 2), dtype=float)
    response[:, 0, 0] = 1.0
    paths = ScanRayPathResult(
        responses_m_per_rad={
            "incident": (response, np.zeros_like(response))
        },
        baseline_ac_command_mrad=np.zeros(2),
        baseline_descan_command_mrad=np.zeros(2),
        frame_period_s=1.0,
        pixels_x=4,
        pixels_y=2,
    )
    workspace._last_result = SimpleNamespace(state_snapshot=state)
    workspace._scan_ray_paths = paths
    item = workspace.plot.plot()
    workspace._ray_bundle_records = [(item, branch)]

    workspace._scan_playback_time_changed(0.5)
    _, projected_x_mm = item.getData()
    assert projected_x_mm[np.isfinite(projected_x_mm)] == pytest.approx(
        (0.5, 0.5)
    )

    workspace._projection_angle_deg = 90.0
    workspace._redraw_projection_items()
    _, projected_y_mm = item.getData()
    assert projected_y_mm[np.isfinite(projected_y_mm)] == pytest.approx(
        (0.0, 0.0),
        abs=1.0e-12,
    )
