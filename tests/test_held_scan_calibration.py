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
    from temsim.optics.energy_filter import EnergyFilterSystem
    # A relocated physical station must obey the same boundary as a target.
    # This is an observer-boundary test, not a filter transport calculation.
    state.energy_filter_installed = True
    boundary = max(p.z_mm for p in state.recording_planes)
    # The finite observer preserves an exact captured model graph. Keep this
    # boundary-only filter fixture serializable without supplying fake fields.
    state.energy_filter = EnergyFilterSystem(enabled=True)
    state.energy_filter.entrance_z_mm = boundary
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
    assert view.calibrate_descan_button.text() == "Calibrate Descan (keep AC)"
    assert "mechanical-angle" in view.calibrate_hold_button.toolTip()
    view._calibrate_and_hold()
    assert state.ac_deflector.calibration_mode == "held"
    assert view.scan_calibration_mode.currentData() == "held"
    assert "Held drive" in view.calibration_status.text()


def _scan_label_result(**captured):
    """A retained geometry record; no propagation or beam qualification."""
    from types import SimpleNamespace
    result = scan.ScanGeometryResult(
        times_s=np.zeros((2, 2)), sample_x_um=np.array([[0., 1.], [0., 1.]]),
        sample_y_um=np.array([[0., 0.], [2., 2.]]),
        plane_positions_um={}, plane_names={}, requested_pixels_x=2,
        requested_pixels_y=2, ac_enabled=True, descan_enabled=False,
        ac_drift_pivot_z_mm=None, descan_drift_pivot_z_mm=None,
        ac_lower_from_upper=-np.eye(2), ac_angular_residual=.125)
    values = dict(vars(result))
    values.update(captured)
    return SimpleNamespace(**values)


def test_scan_summary_uses_captured_held_reference_and_mechanical_span(qtbot):
    from temsim.gui.scan_panel import ScanControlView
    from PySide6.QtWidgets import QLabel
    view = ScanControlView()
    qtbot.addWidget(view)
    live = default_state()
    live.ac_deflector.calibration_mode = "automatic"
    live.ac_deflector.scan_reference = "sample_centre"
    view.set_state(live)
    view.display_result(_scan_label_result(calibration_mode="held",
        sample_reference_name="Specimen entrance", sample_reference_z_mm=1599.1999975,
        sample_mechanical_angle_span_mrad=(.12, .34)))
    text = view.summary.text()
    assert "held (constraint provenance unspecified)" in text
    assert "automatic (mechanical-angle target)" not in text
    assert "Specimen entrance at Z 1599.1999975 mm" in text
    assert "sample-centre span" not in text and "pure-shift" not in text
    assert "canonical" not in text
    assert "mechanical-angle response residual (dimensionless) 0.125" in text
    assert "first-order scan-induced mechanical slope span (peak-to-peak): X 0.12 mrad, Y 0.34 mrad" in text
    help_text = view.summary.toolTip()
    assert "not the absolute beam angle" in help_text and "nonzero magnetic field" in help_text
    scope = view.findChild(QLabel, "scanScopeNotice").toolTip()
    assert "Automatic AC calibration targets" in scope
    assert "Held calibration preserves saved drives" in scope
    assert "AC is coupled for zero first-order angle" not in scope


def test_scan_summary_marks_automatic_target_and_reports_actual_zero_span(qtbot):
    from temsim.gui.scan_panel import ScanControlView
    view = ScanControlView()
    qtbot.addWidget(view)
    view.display_result(_scan_label_result(calibration_mode="automatic",
        sample_reference_name="Specimen centre", sample_reference_z_mm=1600.,
        sample_mechanical_angle_span_mrad=(0., 0.), ac_angular_residual=0.))
    text = view.summary.text()
    assert "automatic (mechanical-angle target)" in text
    assert "Specimen centre at Z 1600 mm" in text
    assert "X 0 mrad, Y 0 mrad" in text
    assert "mechanical-angle response residual (dimensionless) 0" in text
    assert "pure-shift" not in text


@pytest.mark.parametrize("missing", (None, float("nan")))
def test_scan_summary_does_not_turn_missing_residuals_into_zero(qtbot, missing):
    from temsim.gui.scan_panel import ScanControlView
    view = ScanControlView()
    qtbot.addWidget(view)
    legacy = _scan_label_result(ac_angular_residual=missing,
        descan_target_z_mm=2800., descan_target_name="Fluorescent Screen",
        descan_lower_from_upper=-np.eye(2), descan_target_plane_kind="image",
        descan_compensation_residual=missing,
        descan_target_conjugacy_residual_m_per_rad=missing)
    # Old results have no captured calibration/reference/span fields.
    for name in ("calibration_mode", "sample_reference_name", "sample_reference_z_mm",
                 "sample_mechanical_angle_span_mrad"):
        vars(legacy).pop(name, None)
    view.display_result(legacy)
    text = view.summary.text()
    assert "captured calibration: unavailable" in text
    assert "Specimen reference unavailable at Z unavailable" in text
    assert "(peak-to-peak): unavailable" in text
    assert "mechanical-angle response residual (dimensionless) unavailable" in text
    assert "response residual unavailable (dimensionless)" in text
    assert "image conjugacy ||J_diff|| unavailable" in text
    assert "residual 0" not in text and "nan" not in text


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


def _hold_custom_ac(state, *, descan_calibrated=False):
    from temsim.physics.scan_calibration import capture_record
    ac, ds = state.ac_deflector, state.descan_deflector
    ac.set_pure_shift_coupling(((-1.31, .06), (-.02, -1.21)), .08)
    ac.set_scan_command_matrix_mrad(((.0002, -.0001), (.00012, .00018)))
    ds.set_image_plane_coupling(((-.8, .03), (-.01, -.9)),
        target_key=SELECTED_AREA_APERTURE, target_z_mm=state.selected_area_aperture.z_mm)
    capture_record(state, descan_calibrated=descan_calibrated)
    ac.calibration_mode = "held"


def _mock_downstream_response(monkeypatch):
    def phase(_state, component, _z):
        blocks = (((.4, .03), (-.02, .6)), ((.2, -.01), (.04, .3)))
        if component.key == "descan_deflector":
            blocks = (((.3, -.02), (.05, .2)), ((.5, .06), (-.03, .4)))
        return tuple(np.vstack((block, np.zeros((2, 2)))) for block in blocks)
    monkeypatch.setattr(scan, "_coil_phase_responses", phase)
    monkeypatch.setattr(scan, "_sample_plane_classification", lambda *_: ("image", 0., 1.))
    monkeypatch.setattr(scan, "calibrate_ac_pure_shift",
                        lambda *_args, **_kw: pytest.fail("Descan-only must never solve AC"))


def test_partial_held_restore_accepts_uncalibrated_descan_and_never_changes_it(state):
    _hold_custom_ac(state)
    before = dict(state.descan_deflector.__dict__)
    expected = np.asarray(state.ac_deflector.scan_command_matrix_mrad)
    state.ac_deflector.scan_pixel_size_nm *= 2
    restore_held(state, restore_descan=False)
    np.testing.assert_allclose(state.ac_deflector.scan_command_matrix_mrad, expected*2)
    assert state.descan_deflector.__dict__ == before
    with pytest.raises(ValueError, match="Descan was not calibrated"):
        restore_held(state)


def test_descan_only_preserves_custom_ac_and_physical_settings(state, monkeypatch):
    _hold_custom_ac(state)
    _mock_downstream_response(monkeypatch)
    ac, ds = state.ac_deflector, state.descan_deflector
    state.projector_mode = "image"
    state.fluorescent_screen.inserted = True
    state.image_diffraction_deflector.upper_x_mrad = .15
    before = state.to_dict()
    command = np.asarray(ac.scan_command_matrix_mrad)
    ratio = ac.pure_shift_lower_ratio_matrix
    result = scan.recalibrate_descan_only(state)
    assert result.target_key == "flu_screen"
    assert result.response_match_residual < 1e-14
    assert ac.calibration_mode == "held"
    np.testing.assert_array_equal(ac.scan_command_matrix_mrad, command)
    assert ac.pure_shift_lower_ratio_matrix == ratio
    np.testing.assert_allclose(ds.scan_command_matrix_mrad, -command)
    for payload in (before, after := state.to_dict()):
        payload["corrector_elements"] = [c for c in payload["corrector_elements"]
                                         if c["key"] not in (ac.key, ds.key)]
    assert before == after
    record = json.loads(ac.calibration_record_json)
    assert record["descan_calibrated"] is True and record["target_key"] == "flu_screen"
    np.testing.assert_array_equal(record["ac_ratio"], ratio)
    expected = ds.image_plane_lower_ratio_matrix
    ds.set_image_plane_coupling(np.eye(2))
    restore_held(state)
    assert ds.image_plane_lower_ratio_matrix == expected


@pytest.mark.parametrize("failure", ["solve", "capture"])
def test_descan_only_failure_restores_both_components_and_record(state, monkeypatch, failure):
    _hold_custom_ac(state)
    _mock_downstream_response(monkeypatch)
    state.ac_deflector.scan_pixel_size_nm *= 2
    before = [dict(c.__dict__) for c in (state.ac_deflector, state.descan_deflector)]
    name = "_solve_descan_response_match" if failure == "solve" else "capture_record"
    def fail(*_args, **_kw):
        raise ValueError("Rejected new downstream calibration")
    monkeypatch.setattr(scan, name, fail)
    with pytest.raises(ValueError, match="Rejected new downstream"):
        scan.recalibrate_descan_only(state)
    for component, saved in zip((state.ac_deflector, state.descan_deflector), before):
        assert component.__dict__ == saved


@pytest.mark.parametrize("mode", ["automatic", "descan_off", "host_off"])
def test_descan_only_never_adopts_or_enables_controls(state, mode):
    _hold_custom_ac(state)
    if mode == "automatic":
        state.ac_deflector.calibration_mode = "automatic"
    elif mode == "descan_off":
        state.descan_deflector.scan_enabled = False
    else:
        state.image_diffraction_deflector.enabled = False
    before = state.to_dict()
    with pytest.raises(ValueError, match="held AC|Enable AC"):
        scan.recalibrate_descan_only(state)
    assert state.to_dict() == before


def test_ui_descan_only_button_uses_selected_target_and_keeps_ac(qtbot, state, monkeypatch):
    from temsim.gui.scan_panel import ScanControlView
    _hold_custom_ac(state)
    _mock_downstream_response(monkeypatch)
    ratio = state.ac_deflector.pure_shift_lower_ratio_matrix
    view = ScanControlView()
    qtbot.addWidget(view)
    view.set_state(state)
    changed, errors = [], []
    view.parameters_changed.connect(changed.append)
    view.error.connect(errors.append)
    view.calibrate_descan_button.click()
    assert not errors
    assert changed == ["scan_calibration.descan_captured"]
    assert view.descan_target.currentData() == "flu_screen"
    assert state.descan_deflector.image_plane_target_key == "flu_screen"
    assert state.ac_deflector.pure_shift_lower_ratio_matrix == ratio
    assert view.scan_calibration_mode.currentData() == "held"


@pytest.mark.parametrize("record_kind", ["uncalibrated", "old_host"])
def test_ui_can_enable_descan_then_explicitly_recalibrate_held_ac(qtbot, state, monkeypatch, record_kind):
    from temsim.gui.scan_panel import ScanControlView
    _hold_custom_ac(state, descan_calibrated=record_kind == "old_host")
    if record_kind == "old_host":
        record = json.loads(state.ac_deflector.calibration_record_json)
        record["descan_physical_host_key"] = "previous_physical_host"
        state.ac_deflector.calibration_record_json = json.dumps(record)
    state.descan_deflector.scan_enabled = False
    _mock_downstream_response(monkeypatch)
    ratio = state.ac_deflector.pure_shift_lower_ratio_matrix
    original_record = state.ac_deflector.calibration_record_json
    view = ScanControlView()
    qtbot.addWidget(view)
    view.set_state(state)
    errors = []
    view.error.connect(errors.append)
    view.descan_controls["scan_enabled"].setChecked(True)
    assert not errors
    assert state.descan_deflector.scan_enabled
    assert state.ac_deflector.calibration_record_json == original_record
    # Editing the switch does not certify a usable drive or run a new solve.
    with pytest.raises(ValueError, match="not calibrated|different physical"):
        restore_held(state)
    view.calibrate_descan_button.click()
    assert not errors
    assert state.ac_deflector.pure_shift_lower_ratio_matrix == ratio
    assert state.descan_deflector.image_plane_target_key == "flu_screen"
    restore_held(state)


@pytest.mark.parametrize("axis", ["x", "y"])
def test_descan_only_rejects_logical_static_bias_without_altering_it(state, axis):
    _hold_custom_ac(state)
    setattr(state.descan_deflector, f"kick_{axis}_mrad", .003)
    before = state.to_dict()
    with pytest.raises(ValueError, match="logical Descan static bias"):
        scan.recalibrate_descan_only(state)
    assert state.to_dict() == before


def test_descan_only_profile_load_consumes_new_finite_screen_calibration(state, tmp_path):
    """Independent coil execution after a real profile load, not a self-measure.

    The downstream lens is changed to exercise retuning. Its current plane may
    be mixed; this checks position compensation, not image conjugacy or focus.
    """
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.column.state_layout import apply_physical_layout_to_state
    from temsim.optics.direct_alignment import canonical_source_basis
    from temsim.physics import core
    from temsim.physics.finite_scan_response import finite_scan_responses
    from temsim.physics.instrument_magnetic import active_column_events, column_dipole_fields
    from temsim.physics.scan_calibration import capture_record
    from temsim.profile_io import save_profile, read_profile, apply_profile_values

    state.acceleration_backend, state.acceleration_enabled = "Numba CPU", True
    state.step_mm = .05
    ac = state.ac_deflector
    z = state.sample.z_mm
    response = finite_scan_responses(state, ac, [z])
    upper, lower, _ = response.at(z)
    g = canonical_source_basis(state, z)[2:, :2]
    response_matrix = np.block([[upper[:2], lower[:2]],
                               [upper[2:]-g@upper[:2], lower[2:]-g@lower[:2]]])
    desired = np.diag((ac.scan_field_of_view_x_nm, ac.scan_field_of_view_y_nm))*.5e-9
    kicks = np.linalg.solve(response_matrix, np.vstack((desired, np.zeros((2, 2)))))
    ac.set_pure_shift_coupling(np.linalg.solve(kicks[:2].T, kicks[2:].T).T)
    ac.set_scan_command_matrix_mrad(kicks[:2]*1e3/ac.upper_coil_gain)
    capture_record(state, descan_calibrated=False)
    ac.calibration_mode = "held"
    original_ac = ac.pure_shift_lower_ratio_matrix, ac.scan_command_matrix_mrad
    state.projector_mode = "image"
    next(l for l in state.lenses if l.key == "diffraction_lens").percent *= .97
    state.fluorescent_screen.inserted = True
    scan.recalibrate_descan_only(state)
    assert (ac.pure_shift_lower_ratio_matrix, ac.scan_command_matrix_mrad) == original_ac

    catalog = AssemblyCatalog()
    path = tmp_path / "new-image-descan.toml"
    save_profile(path, state, catalog.selection_for_resolved(state._resolved_assembly))
    selection, values = read_profile(path)
    loaded = default_state()
    catalog.apply(loaded, selection)
    apply_profile_values(loaded, values)
    apply_physical_layout_to_state(loaded, preserve_operating_parameters=True)
    scan.calibrate_scan_system(loaded, observation_stop_z_mm=loaded.fluorescent_screen.z_mm)
    assert loaded.ac_deflector.calibration_mode == "held"
    assert loaded.ac_deflector.calibration_record_json == ac.calibration_record_json
    assert loaded.descan_deflector.image_plane_target_key == "flu_screen"
    assert (loaded.ac_deflector.pure_shift_lower_ratio_matrix,
            loaded.ac_deflector.scan_command_matrix_mrad) == original_ac
    source = min(c.field_support_mm[0] for c in column_dipole_fields(loaded)
                 if c.key.startswith("ac_deflector:"))
    stop = loaded.fluorescent_screen.z_mm
    def position():
        zeros = np.zeros(1)
        result = core.propagate(loaded, source, stop, zeros, zeros, zeros, zeros,
            events=active_column_events(loaded), include_spherical_aberration=False,
            include_hexapole=True, maximum_step_mm=.05,
            checkpoint_z_mm=[stop], return_checkpoints=True)[-1]
        return np.array([result.x_m[-1, 0], result.y_m[-1, 0]])
    loaded.ac_deflector.scan_enabled = loaded.descan_deflector.scan_enabled = False
    baseline = position()
    loaded.ac_deflector.scan_enabled = loaded.descan_deflector.scan_enabled = True
    for time_s in (.03, .4):
        loaded.simulation_time_s = time_s
        assert np.linalg.norm(position()-baseline) < 3e-12
