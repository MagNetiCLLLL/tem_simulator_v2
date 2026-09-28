"""Real Qt session transitions with labelled synthetic transport fixtures."""
from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtCore import Qt

from temsim.electron_diagnostic_session import (
    DiagnosticDependency, DiagnosticDisplayState, DiagnosticElectronRecord,
    DiagnosticSession, save_diagnostic_session,
)
from test_magnetic_test_electron_gui import (
    controller, UniformScene, install_trace, wait_for_result, synthetic_trajectory,
)


def calculated(qtbot, controller, monkeypatch):
    calls = install_trace(monkeypatch)
    controller.set_scene(UniformScene())
    controller.set_active(True)
    wait_for_result(qtbot, controller)
    return calls


def test_save_load_history_never_automatically_executes_or_enters_cache(qtbot, controller, monkeypatch, tmp_path):
    calls = calculated(qtbot, controller, monkeypatch)
    original = controller.current_trajectory
    record = controller.selected_record
    settings, key = record.settings, record.key
    path = tmp_path / "diagnostic.temdiag"
    controller.session_actions.save(path)
    loaded = controller.session_actions.load(path)
    qtbot.wait(230)
    assert len(calls) == 1
    assert controller.current_trajectory is None
    assert controller._cache == {}
    assert controller.selected_record.historical
    assert controller.settings() == settings
    assert controller.selected_record.key == key
    assert loaded.records[0].trajectory.completed
    np.testing.assert_array_equal(controller.visible_paths()[0].positions_m, original.positions_m)
    assert controller.visible_paths()[0].state == "previous"
    assert "Historical" in controller.status_text
    assert not controller.energy.isEnabled()
    qtbot.mouseClick(controller.session_actions.recalculate_button, Qt.MouseButton.LeftButton)
    wait_for_result(qtbot, controller)
    assert len(calls) == 2
    assert not controller.selected_record.historical


def test_loaded_history_survives_absent_and_replaced_fields(qtbot, controller, monkeypatch, tmp_path):
    calculated(qtbot, controller, monkeypatch)
    path = tmp_path / "history.temdiag"
    controller.session_actions.save(path)
    controller.invalidate()
    controller.session_actions.load(path)
    result = controller.selected_record.trajectory
    assert controller.visible_paths()
    assert not controller.history_fields_match()
    assert not controller.session_actions.recalculate_button.isEnabled()
    controller.set_scene(UniformScene(field=(0., .2, 0.)))
    qtbot.wait(200)
    assert controller.selected_record.trajectory is result
    assert controller.current_trajectory is None
    assert controller.session_actions.recalculate_button.isEnabled()


def test_corrupt_load_preserves_live_records_and_pending_ownership(qtbot, controller, monkeypatch, tmp_path):
    calculated(qtbot, controller, monkeypatch)
    records, generation, scene = controller.records, controller._generation, controller._scene
    path = tmp_path / "corrupt.temdiag"
    path.write_bytes(b"not a session")
    with pytest.raises((ValueError, OSError)):
        controller.session_actions.load(path)
    assert controller.records == records
    assert controller._generation == generation
    assert controller._scene is scene
    assert controller.current_trajectory is records[0].trajectory


def test_save_during_new_edit_keeps_executed_settings_separate(qtbot, controller, monkeypatch, tmp_path):
    calculated(qtbot, controller, monkeypatch)
    controller.set_active(False)
    original = controller.selected_record.trajectory_settings
    controller.x.setValue(controller.x.value()+.001)
    path = tmp_path / "previous.temdiag"
    controller.session_actions.save(path)
    loaded = controller.session_actions.load(path)
    stored = loaded.records[0]
    assert stored.state == "previous"
    assert stored.settings != original
    assert stored.trajectory_settings == original


def test_uncomputed_and_prefix_survive_restart_with_independent_ids(controller, tmp_path):
    controller.set_scene(UniformScene())
    settings = controller.settings()
    prefix = replace(synthetic_trajectory(settings), completed=False, reason="in_progress")
    session = DiagnosticSession((
        DiagnosticElectronRecord("electron-24", "Uncomputed", "#ffd166", settings),
        DiagnosticElectronRecord("electron-30", "Prefix", "#48cae4", settings,
                                 trajectory=prefix, trajectory_settings=settings, state="incomplete"),
    ), display=DiagnosticDisplayState(overlay=True, selected_key="electron-30", show_background=False))
    path = tmp_path / "prefix.temdiag"
    save_diagnostic_session(session, path)
    controller.invalidate()
    controller.session_actions.load(path)
    assert len(controller.records) == 2
    assert len(controller.visible_paths()) == 1
    assert controller.current_trajectory is None
    assert "incomplete" in controller._row_status(controller.selected_record)
    assert not controller.background.isChecked()
    controller.set_scene(UniformScene())
    controller.add_electron()
    assert controller.selected_record.key == "electron-31"


def test_historical_dependency_retained_when_other_record_recalculated(qtbot, controller, monkeypatch, tmp_path):
    calculated(qtbot, controller, monkeypatch)
    settings = controller.settings()
    old = DiagnosticDependency(physical_identity="old", numerical_identity="old-n", transport_identity="old-t")
    stored = DiagnosticElectronRecord("old", "Old fields", "#ffd166", settings,
        trajectory=synthetic_trajectory(settings), trajectory_settings=settings, state="previous", dependency=old)
    path = tmp_path / "mixed.temdiag"
    save_diagnostic_session(DiagnosticSession((stored,)), path)
    controller.session_actions.load(path)
    controller.add_electron()
    wait_for_result(qtbot, controller)
    snapshot = controller.session_actions.snapshot()
    assert snapshot.records[0].dependency == old
    assert snapshot.records[0].state == "previous"
    assert snapshot.records[1].state == "completed"


def test_session_restores_view_through_shared_projection_signal(qtbot, tmp_path):
    from temsim.gui.magnetic_field_3d import MagneticField3DPage
    from temsim.magnetic_test_particle import TestElectronSettings
    page = MagneticField3DPage()
    qtbot.addWidget(page)
    page.set_electron_mode(True)
    observed = []
    page.session_projection_requested.connect(observed.append)
    display = DiagnosticDisplayState(projection_degrees=75.4, axial_limits_mm=(10., 90.),
                                     transverse_limits_mm=(-.2, .3), selected_key="one")
    path = tmp_path / "view.temdiag"
    save_diagnostic_session(DiagnosticSession((DiagnosticElectronRecord(
        "one", "One", "#ffd166", TestElectronSettings()),), display=display), path)
    page.electron.session_actions.load(path)
    assert observed == [75.4]
    assert page.canvas.projection_angle_deg == 75.4
    assert page.canvas.view_range_mm() == ((10., 90.), (-.2, .3))
    assert not page.canvas._field_lines_visible
    page.electron.shutdown()


def test_history_navigation_retry_and_view_failure_do_not_modify_saved_state(controller, tmp_path):
    controller.set_scene(UniformScene())
    record = DiagnosticElectronRecord("failed", "Failed", "#ffd166", controller.settings(),
                                      state="failed", error="saved failure")
    path = tmp_path / "failed.temdiag"
    save_diagnostic_session(DiagnosticSession((record,)), path)
    controller.session_actions.load(path)
    historical = controller.selected_record
    controller.set_axial_range_mm(3., 4.)
    assert not controller.start_at_view.isEnabled()
    assert not controller.retry_button.isEnabled()
    assert not controller.retry_failed_execution()
    assert historical.error == "saved failure"
    assert not controller.current_trajectory
    generation = controller._generation
    def fail_view(_display):
        raise ValueError("presentation unavailable")
    controller.session_actions.restore_view = fail_view
    with pytest.raises(ValueError, match="presentation unavailable"):
        controller.session_actions.load(path)
    assert controller.selected_record is historical
    assert controller._generation == generation


def test_mismatched_history_load_does_not_request_field_geometry(qtbot, monkeypatch, tmp_path):
    from temsim.gui.magnetic_field_3d import MagneticField3DPage
    from temsim.magnetic_test_particle import TestElectronSettings
    page = MagneticField3DPage()
    qtbot.addWidget(page)
    page._active = True
    page._state = object()
    page.set_electron_mode(True)
    path = tmp_path / "history.temdiag"
    save_diagnostic_session(DiagnosticSession((DiagnosticElectronRecord(
        "one", "One", "#ffd166", TestElectronSettings()),)), path)
    page.electron.session_actions.load(path)
    page._request_geometry()
    assert page._worker is None
    assert not page.canvas._field_lines_visible
    page.set_active(False)
    page.electron.shutdown()


@pytest.mark.parametrize("missing", ["physical_field_identity", "numerical_field_identity"])
def test_dependency_cannot_fill_unknown_trajectory_field_identity(controller, tmp_path, missing):
    scene = UniformScene()
    scene.physical_identity = "captured-physical"
    scene.numerical_identity = "captured-numerical"
    scene.transport_identity = "captured-transport"
    controller.set_scene(scene)
    settings = controller.settings()
    dependency = DiagnosticDependency(physical_identity=scene.physical_identity,
        numerical_identity=scene.numerical_identity, transport_identity=scene.transport_identity)
    trajectory = replace(synthetic_trajectory(settings), physical_field_identity=scene.physical_identity,
                         numerical_field_identity=scene.numerical_identity)
    record = DiagnosticElectronRecord("saved", "Saved", "#ffd166", settings,
        trajectory=trajectory, trajectory_settings=settings, state="previous", dependency=dependency)
    path = tmp_path / "history.temdiag"
    save_diagnostic_session(DiagnosticSession((record,)), path)
    controller.session_actions.load(path)
    assert controller.history_fields_match()
    unknown = replace(record, trajectory=replace(trajectory, **{missing: None}))
    save_diagnostic_session(DiagnosticSession((unknown,)), path)
    controller.session_actions.load(path)
    assert not controller.history_fields_match()
    assert "unavailable" in controller.status_text
    assert getattr(controller.selected_record.trajectory, missing) is None
    assert controller.current_trajectory is None


def test_historical_duplicate_without_current_fields_preserves_data_only(controller, qtbot, tmp_path):
    from temsim.magnetic_test_particle import TestElectronSettings
    settings = TestElectronSettings()
    dependency = DiagnosticDependency(result_reference="request:historical-main-result")
    stored = DiagnosticElectronRecord("electron-24", "Saved", "#ffd166", settings,
        trajectory=synthetic_trajectory(settings), trajectory_settings=settings,
        state="previous", dependency=dependency)
    path = tmp_path / "history.temdiag"
    save_diagnostic_session(DiagnosticSession((stored,)), path)
    controller.session_actions.load(path)
    original = controller.selected_record
    assert controller._scene is None
    assert controller.duplicate_button.isEnabled()
    qtbot.mouseClick(controller.duplicate_button, Qt.MouseButton.LeftButton)
    duplicate = controller.selected_record
    assert duplicate is not original and duplicate.key == "electron-25"
    assert duplicate.historical and duplicate.settings == original.settings
    assert duplicate.history_dependency == original.history_dependency
    assert duplicate.trajectory is original.trajectory
    assert duplicate.trajectory_settings == original.trajectory_settings
    assert len(controller.records) == 2 and controller._cache == {}
    assert controller._worker is None and controller._scene_worker is None
    assert controller.current_trajectory is None
    assert len(controller.visible_paths()) == 1


def test_saved_session_restores_after_controller_close_and_recreation(controller, qtbot, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QMainWindow
    from temsim.gui.magnetic_test_electron import TestElectronController
    calls = calculated(qtbot, controller, monkeypatch)
    original = controller.selected_record
    path = tmp_path / "restart.temdiag"
    controller.session_actions.save(path)
    controller.shutdown()
    owner = QMainWindow()
    qtbot.addWidget(owner)
    restored = TestElectronController(owner)
    try:
        restored.session_actions.load(path)
        restored.set_active(True)
        qtbot.wait(180)
        assert len(calls) == 1
        assert restored.selected_record.key == original.key
        assert restored.settings() == original.settings
        np.testing.assert_array_equal(restored.visible_paths()[0].positions_m, original.trajectory.positions_m)
        assert restored.current_trajectory is None and restored._cache == {}
        assert restored._scene is None and restored._worker is None and restored._scene_worker is None
    finally:
        restored.shutdown()
