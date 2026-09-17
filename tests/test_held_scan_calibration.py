"""Drive persistence and actual first-order optics, without synthetic imaging."""
import json
import numpy as np
import pytest
from temsim.optics.column import default_state
from temsim.physics import scan_geometry as scan
from temsim.physics.scan_calibration import restore_held


@pytest.fixture
def state():
    state = default_state()
    state.ac_deflector.wobble_enabled = False
    state.ac_deflector.scan_enabled = True
    state.descan_deflector.scan_enabled = True
    state.ac_deflector.scan_pixels_x = state.ac_deflector.scan_lines = 4
    state.ac_deflector.scan_pixel_size_nm = .02
    state.descan_deflector.descan_target_key = "flu_screen"
    return state


@pytest.mark.parametrize("control", ["pivot", "diffraction_focus"])
def test_held_scan_survives_optics_change_without_a_new_solve(state, monkeypatch, control):
    scan.calibrate_scan_system(state, force=True, hold=True)
    saved = state.ac_deflector.calibration_record_json
    matrix = np.asarray(state.descan_deflector.image_plane_lower_ratio_matrix)
    baseline = scan.calculate_scan_geometry(state)
    assert baseline.descan_target_key == "flu_screen"
    assert baseline.descan_compensation_residual < 1e-8
    if control == "pivot":
        state.ac_deflector.pivot_offset_x = .1
    else:
        lens = next(l for l in state.lenses if l.key == "diffraction_lens")
        lens.percent *= 1.01
    monkeypatch.setattr(scan, "_solve_descan_response_match",
                        lambda *_args: pytest.fail("Held scan must not solve new coil ratios"))
    command, _, result = scan.calibrate_scan_system(state)
    changed = scan.calculate_scan_geometry(state)
    if control == "pivot":
        assert changed.ac_angular_residual > baseline.ac_angular_residual + .01
    assert result.response_match_residual > 1e-7
    assert state.ac_deflector.calibration_record_json == saved
    np.testing.assert_array_equal(state.descan_deflector.image_plane_lower_ratio_matrix, matrix)
    np.testing.assert_allclose(state.descan_deflector.scan_command_matrix_mrad, -command)
    before = np.asarray(baseline.plane_positions_um["flu_screen"])
    after = np.asarray(changed.plane_positions_um["flu_screen"])
    assert np.linalg.norm(after-before) > 1e-8


def test_snapshot_and_state_payload_keep_calibration_and_clock(state):
    from temsim.instrument_snapshot import capture_instrument_snapshot
    scan.calibrate_scan_system(state, force=True, hold=True)
    state.ac_deflector.scan_frame_period_s = 2.
    record = state.ac_deflector.calibration_record_json
    assert json.loads(record)["calibrated_snapshot_id"]
    for restored in (type(state).from_dict(state.to_dict()), capture_instrument_snapshot(state).restore()):
        scan.calibrate_scan_system(restored)
        assert restored.ac_deflector.calibration_record_json == record
        assert restored.ac_deflector.calibration_mode == "held"
        assert restored.descan_deflector.scan_frame_period_s == 2.
        for t in (.01, .7, 1.2):
            np.testing.assert_allclose(restored.ac_deflector.scan_kick_mrad(t),
                                       -np.asarray(restored.descan_deflector.scan_kick_mrad(t)))


def test_held_fov_rescales_command_without_changing_ratios(state):
    scan.calibrate_scan_system(state, force=True, hold=True)
    before = np.asarray(state.ac_deflector.scan_command_matrix_mrad)
    record = state.ac_deflector.calibration_record_json
    state.ac_deflector.scan_pixel_size_nm *= 2
    restore_held(state)
    np.testing.assert_allclose(state.ac_deflector.scan_command_matrix_mrad, 2*before)
    assert state.ac_deflector.calibration_record_json == record


def test_failed_explicit_calibration_rolls_back_both_drives(state, monkeypatch):
    scan.calibrate_scan_system(state, force=True, hold=True)
    before = [dict(c.__dict__) for c in (state.ac_deflector, state.descan_deflector)]
    monkeypatch.setattr(scan, "_solve_descan_response_match",
                        lambda *_args: (_ for _ in ()).throw(ValueError("singular fixture")))
    with pytest.raises(ValueError, match="singular fixture"):
        scan.calibrate_scan_system(state, force=True, hold=True)
    for c, original in zip((state.ac_deflector, state.descan_deflector), before):
        assert c.__dict__ == original


def test_no_silent_hold_without_record_and_no_virtual_target(state):
    state.ac_deflector.calibration_mode = "held"
    with pytest.raises(ValueError, match="No held"):
        scan.calibrate_scan_system(state)
    state.descan_deflector.descan_target_key = "arbitrary_virtual_plane"
    with pytest.raises(ValueError, match="installed physical"):
        scan._descan_target(state)


def test_specimen_entrance_reference_is_explicit(state):
    from temsim.physics.scan_calibration import sample_reference_z_mm
    assert sample_reference_z_mm(state) == state.sample.z_mm
    state.ac_deflector.scan_reference = "sample_entrance"
    assert sample_reference_z_mm(state) == state.sample.upper_surface_z_mm


def test_automatic_calibration_reports_user_pivot_residual(state):
    state.ac_deflector.pivot_offset_x = .1
    _, residual = scan.calibrate_ac_pure_shift(state)
    assert residual > 0.
    assert residual == pytest.approx(scan._ac_angle_residual(state))
    assert state.ac_deflector.pivot_offset_x == .1


def test_filter_target_cannot_bypass_transported_filter_response(state):
    from types import SimpleNamespace
    state.energy_filter_installed = True
    state.energy_filter = SimpleNamespace(entrance_z_mm=2000.)
    state.descan_deflector.descan_target_key = "flu_screen"
    with pytest.raises(ValueError, match="transported response"):
        scan._descan_target(state)


def test_ui_has_explicit_calibration_and_target_controls(qtbot, state):
    from temsim.gui.scan_panel import ScanControlView
    view = ScanControlView()
    qtbot.addWidget(view)
    view.set_state(state)
    assert view.descan_target.currentData() == "flu_screen"
    assert view.scan_calibration_mode.currentData() == "automatic"
    view._calibrate_and_hold()
    assert state.ac_deflector.calibration_mode == "held"
    assert view.scan_calibration_mode.currentData() == "held"
    assert "Held drive" in view.calibration_status.text()
