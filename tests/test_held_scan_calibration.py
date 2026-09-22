"""Drive persistence and actual first-order optics, without synthetic imaging."""
import json
import numpy as np
import pytest
from temsim.optics.column import default_state
from temsim.physics import scan_geometry as scan
from temsim.physics.scan_calibration import restore_held
from temsim.component_keys import SELECTED_AREA_APERTURE


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


def test_scan_observer_never_supplies_an_unfiltered_downstream_response(state):
    from types import SimpleNamespace
    # A relocated physical station must obey the same boundary as a target.
    # This is an observer-boundary test, not a filter transport calculation.
    state.energy_filter_installed = True
    boundary = max(p.z_mm for p in state.recording_planes)
    state.energy_filter = SimpleNamespace(entrance_z_mm=boundary)
    state.descan_deflector.descan_target_key = SELECTED_AREA_APERTURE
    with pytest.raises(ValueError, match="transported response"):
        scan.paired_kick_response(state, state.ac_deflector, boundary)
    with pytest.raises(ValueError, match="transported response"):
        scan.paired_kick_response_grid(state, state.ac_deflector, [boundary-1., boundary])
    result = scan.calculate_scan_geometry(state)
    blocked = {p.key for p in state.recording_planes if p.z_mm >= boundary}
    assert blocked
    assert not blocked.intersection(result.plane_positions_um)
    assert blocked.issubset(result.unavailable_planes)
    assert result.sample_x_um.size
    simulation = SimpleNamespace(incident=SimpleNamespace(name="incident", z=np.array([100., boundary-1., boundary, boundary+1.])), branches={})
    path = scan.calculate_scan_ray_paths(state, simulation)
    for response in path.responses_m_per_rad["incident"]:
        assert np.all(np.isfinite(response[:2]))
        assert np.all(np.isnan(response[2:]))


def test_ui_has_explicit_calibration_and_target_controls(qtbot, state):
    from temsim.gui.scan_panel import ScanControlView
    view = ScanControlView()
    qtbot.addWidget(view)
    view.set_state(state)
    assert view.descan_target.currentData() == "flu_screen"
    assert view.scan_calibration_mode.currentData() == "automatic"
    assert view.scan_calibration_mode.currentText() == "Automatic recalibration"
    assert view.scan_reference.currentText() == "Specimen centre"
    assert "Z " in view.descan_target.currentText()
    assert "mm" in view.descan_target.currentText()
    assert view.descan_target.findData(SELECTED_AREA_APERTURE) >= 0
    assert view.descan_target.findData("legacy_image_reference") == -1
    view._calibrate_and_hold()
    assert state.ac_deflector.calibration_mode == "held"
    assert view.scan_calibration_mode.currentData() == "held"
    assert "Held drive" in view.calibration_status.text()


def test_default_descan_target_is_a_component_not_an_inferred_reference():
    current = default_state()
    assert current.descan_deflector.descan_target_key == SELECTED_AREA_APERTURE
    key, _, z_mm = scan._descan_target(current)
    assert key == SELECTED_AREA_APERTURE
    assert z_mm == current.selected_area_aperture.z_mm


def test_scan_runtime_controls_do_not_offer_inactive_diagonal_amplitude_seeds(state):
    from temsim.runtime_parameters import RuntimeTarget, editable_parameters
    for component in (state.ac_deflector, state.descan_deflector):
        target = RuntimeTarget(component.key, component.name, component)
        fields = {item.name for item in editable_parameters(target)}
        assert "scan_pixel_size_nm" in fields
        assert "scan_amplitude_x_mrad" not in fields
        assert "scan_amplitude_y_mrad" not in fields


@pytest.mark.parametrize("key", ["legacy_image_reference", "unknown_station", ""])
def test_unavailable_target_never_substitutes_another_station(state, key):
    state.descan_deflector.descan_target_key = key
    with pytest.raises(ValueError, match="installed physical"):
        scan._descan_target(state)
    assert state.descan_deflector.descan_target_key == key


def test_missing_selected_aperture_does_not_choose_first_recording_plane():
    from types import SimpleNamespace
    current = default_state()
    partial = SimpleNamespace(
        descan_deflector=current.descan_deflector,
        selected_area_aperture=None,
        recording_planes=current.recording_planes,
    )
    with pytest.raises(ValueError, match="selected_area_aperture"):
        scan._descan_target(partial)


def test_target_choices_and_resolution_share_the_same_component_contract(qtbot, state):
    from temsim.gui.scan_panel import ScanControlView
    view = ScanControlView()
    qtbot.addWidget(view)
    view.set_state(state)
    for index in range(view.descan_target.count()):
        key = view.descan_target.itemData(index)
        state.descan_deflector.descan_target_key = key
        actual_key, name, z_mm = scan._descan_target(state)
        assert actual_key == key
        assert name in view.descan_target.itemText(index)
        assert f"{z_mm:.6f} mm" in view.descan_target.itemText(index)


@pytest.mark.parametrize("kind", ["scan", "descan"])
def test_paired_deflector_loader_rejects_obsolete_or_missing_component_key(kind):
    from temsim.optics.ac_deflector import create_ac_deflector, ac_deflector_from_dict
    from temsim.optics.descan_deflector import create_descan_deflector, descan_deflector_from_dict
    create, restore = ((create_ac_deflector, ac_deflector_from_dict) if kind == "scan"
                       else (create_descan_deflector, descan_deflector_from_dict))
    record = create().to_dict()
    assert restore(record).key == record["key"]
    for invalid in ("single_plane_deflector", None):
        record["key"] = invalid
        with pytest.raises(ValueError, match="requires component key"):
            restore(record)


def test_scan_loader_does_not_silently_disable_an_active_drive():
    from temsim.optics.ac_deflector import create_ac_deflector, ac_deflector_from_dict
    record = create_ac_deflector().to_dict()
    record.update(wobble_enabled=True, scan_enabled=True)
    with pytest.raises(ValueError, match="(?i)wobble"):
        ac_deflector_from_dict(record)


def test_descan_loader_rejects_implicit_target():
    from temsim.optics.descan_deflector import create_descan_deflector, descan_deflector_from_dict
    record = create_descan_deflector().to_dict()
    record["descan_target_key"] = "legacy_image_reference"
    with pytest.raises(ValueError, match="Implicit descan"):
        descan_deflector_from_dict(record)


def test_cutoff_before_descan_does_not_certify_unexecuted_calibration(state, monkeypatch):
    # Both drives are enabled, but the accepted path ends at the specimen.
    monkeypatch.setattr(scan, "calibrate_descan_image_plane",
                        lambda *_: pytest.fail("Descan is beyond the selected cutoff"))
    _, _, descan_result = scan.calibrate_scan_system(
        state, force=True, hold=True, observation_stop_z_mm=state.sample.z_mm)
    assert descan_result is None
    assert json.loads(state.ac_deflector.calibration_record_json)["descan_calibrated"] is False
    with pytest.raises(ValueError, match="Descan was not calibrated"):
        restore_held(state)


@pytest.mark.parametrize("invalid", [None, "", "legacy_image_reference"])
def test_held_descan_record_requires_an_explicit_physical_target(state, invalid):
    from temsim.physics.scan_calibration import validate_record
    scan.calibrate_scan_system(state, force=True, hold=True)
    record = json.loads(state.ac_deflector.calibration_record_json)
    record["target_key"] = invalid
    with pytest.raises(ValueError, match="target key"):
        validate_record(json.dumps(record))


def test_derived_lower_gain_cannot_override_primary_gain_on_current_roundtrips(state, tmp_path):
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.profile_io import save_profile, read_profile, apply_profile_values
    from temsim.runtime_parameters import RuntimeTarget, editable_parameters
    scan.calibrate_scan_system(state, force=True, hold=True)
    ratio = np.asarray(state.ac_deflector.pure_shift_lower_ratio_matrix)
    assert np.linalg.norm(ratio + np.eye(2)) > 1e-4
    pairs = (state.ac_deflector, state.descan_deflector)
    for pair in pairs:
        fields = {p.name for p in editable_parameters(RuntimeTarget(pair.key, pair.name, pair))}
        assert "upper_coil_gain" in fields and "lower_coil_gain" not in fields
        assert "lower_coil_gain" not in pair.to_dict()
    payload = state.to_dict()
    for record in payload["corrector_elements"]:
        if record["key"] in {pair.key for pair in pairs}:
            assert "lower_coil_gain" not in record
    restored = type(state).from_dict(payload)
    path = tmp_path / "current-scan.toml"
    catalog = AssemblyCatalog()
    selection = catalog.selection_for_resolved(state._resolved_assembly)
    save_profile(path, state, selection)
    selection, values = read_profile(path)
    for pair in pairs:
        assert "lower_coil_gain" not in values[pair.key]
    profile_state = default_state()
    catalog.apply(profile_state, selection)
    apply_profile_values(profile_state, values)
    for target in (restored, profile_state):
        assert target.ac_deflector.upper_coil_gain == state.ac_deflector.upper_coil_gain
        assert target.descan_deflector.upper_coil_gain == state.descan_deflector.upper_coil_gain
        scan.calibrate_scan_system(target)
        np.testing.assert_allclose(target.ac_deflector.pure_shift_lower_ratio_matrix, ratio)
        np.testing.assert_allclose(target.descan_deflector.image_plane_lower_ratio_matrix,
                                   state.descan_deflector.image_plane_lower_ratio_matrix)


@pytest.mark.parametrize("kind", ["scan", "descan"])
def test_derived_lower_gain_is_rejected_as_an_operating_input(kind):
    from temsim.optics.ac_deflector import create_ac_deflector, ac_deflector_from_dict
    from temsim.optics.descan_deflector import create_descan_deflector, descan_deflector_from_dict
    create, restore = ((create_ac_deflector, ac_deflector_from_dict) if kind == "scan"
                       else (create_descan_deflector, descan_deflector_from_dict))
    record = create().to_dict()
    record["lower_coil_gain"] = -1.
    with pytest.raises(ValueError, match="derived readback"):
        restore(record)
