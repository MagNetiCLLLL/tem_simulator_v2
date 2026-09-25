"""Result-file UI restores executed data without submitting a new calculation.

The shared 49-ray fixture executes once; file-open, startup and rollback tests
then forbid transport and calculation dispatch. They exercise real Qt views.
"""
import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QSettings

from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.particle_section_io import section_archive_summary
from temsim.result_library import ResultLibrary
from test_particle_section_io import executed_section


@pytest.fixture
def gui_environment(qtbot, monkeypatch, tmp_path):
    from temsim.gui import main_window, interactive_calculation, calculation_controller
    settings = QSettings(str(tmp_path / "workspace.ini"), QSettings.IniFormat)
    monkeypatch.setattr(main_window, "QSettings", lambda: settings)
    monkeypatch.setattr(interactive_calculation, "QSettings", lambda: settings)
    monkeypatch.setattr(calculation_controller, "default_artifact_cache_root", lambda: tmp_path / "artifacts")
    monkeypatch.setattr(main_window.MainWindow, "INITIAL_PREVIEW_DELAY_MS", 60000)
    errors = []
    monkeypatch.setattr(main_window.MainWindow, "_show_error", lambda self, text: errors.append(text))
    windows = []

    def make():
        window = main_window.MainWindow()
        window.preview_timer.stop()
        qtbot.addWidget(window)
        windows.append(window)
        return window

    yield make, errors, tmp_path
    for window in windows:
        window.preview_timer.stop()
        window.calculations.invalidate_pending(include_explicit=True)
        window.calculations.pool.waitForDone(3000)
        window.close()
        window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


@pytest.fixture
def window(gui_environment, monkeypatch):
    make, errors, _ = gui_environment
    window = make()
    def forbidden(*args, **kwargs):
        pytest.fail("Opening a result submitted a new calculation")
    monkeypatch.setattr(window.calculations, "submit_background", forbidden)
    monkeypatch.setattr(window.calculations, "archive_completed_section", forbidden)
    return window


def loaded_info(result, path, token="load-1"):
    return dict(section_archive_summary(result, path=path, saved_at_utc="2026-09-22T12:00:00+00:00"),
                operation_token=token, status="loaded", _package_digest="a" * 64,
                compressed_size_bytes=1048576, unpacked_size_bytes=2097152)


def begin_load(window, monkeypatch, path, token="load-1"):
    def load(selected):
        window.calculations.section_archive_changed.emit(dict(
            status="loading", path=selected, identity=token, operation_token=token))
        return token
    monkeypatch.setattr(window.calculations, "load_section_archive", load)
    window.result_files.load(path)
    assert window.result_files.loading
    assert not window.workspace.interactive_calculation.isEnabled()
    assert not window.working_points.isEnabled()


def deliver(window, result, path, token="load-1"):
    window._particle_section_loaded(result, loaded_info(result, path, token))


def test_open_restores_real_views_independent_settings_and_no_compute(window, executed_section, monkeypatch, gui_environment):
    _, errors, root = gui_environment
    from temsim.optics.electron_gun import source
    monkeypatch.setattr(source, "trace_source_to_exit", lambda *a, **k: pytest.fail("Gun transport on load"))
    original_digest = capture_instrument_snapshot(executed_section.state_snapshot).digest
    begin_load(window, monkeypatch, root / "opened.temresult")
    deliver(window, executed_section, root / "opened.temresult")
    assert not errors
    assert not window.result_files.loading
    assert window.workspace._last_result is executed_section
    assert window.workspace._preview_result is executed_section
    assert window.workspace._last_quality == "Preview"
    assert window.workspace._ray_bundle_records
    assert window.workspace.interactive_calculation._section_result is executed_section
    assert window.workspace.interactive_calculation.segment_request() == {
        "target_z_mm": executed_section.simulation.metrics["section_target_z_mm"],
        "component_keys": ("objective_lens",)}
    assert "no recalculation" in window.status_label.text()
    assert "1.00 MiB" in window.status_label.text()
    assert "Unpacked data" in window.workspace.interactive_calculation.section_archive_status.text()
    assert capture_instrument_snapshot(window.state).digest == original_digest
    assert window.state is not executed_section.state_snapshot
    assert window.state.electron_gun is not executed_section.state_snapshot.electron_gun
    assert window.state._resolved_assembly is not executed_section.state_snapshot._resolved_assembly
    assert window.workspace.interactive_calculation.source_state is not window.state
    saved_coefficients = executed_section.state_snapshot.energy_filter.multipoles[0].calibration.normal_trim_coefficients
    editable_coefficients = window.state.energy_filter.multipoles[0].calibration.normal_trim_coefficients
    assert not np.shares_memory(saved_coefficients, editable_coefficients)
    editable_coefficients[0] += .001
    window.state.lenses[0].percent += .25
    assert capture_instrument_snapshot(executed_section.state_snapshot).digest == original_digest
    assert not window.preview_timer.isActive()
    window._calculation_finished("Preview")
    window._interactive_operation_finished()
    window.preview_timer.timeout.emit()
    assert not window.preview_timer.isActive()
    assert window.result_files.hold_automatic_preview


def test_new_open_supersedes_old_callbacks_and_failure_retains_result(window, executed_section, monkeypatch, gui_environment):
    _, errors, root = gui_environment
    begin_load(window, monkeypatch, root / "one.temresult", "first")
    begin_load(window, monkeypatch, root / "two.temresult", "second")
    deliver(window, executed_section, root / "one.temresult", "first")
    assert window.workspace._last_result is None
    window._section_archive_status_changed(dict(status="failed", operation="load",
        operation_token="first", identity="first", path=str(root / "one.temresult"), error="Old file"))
    assert window.result_files.loading and not errors
    deliver(window, executed_section, root / "two.temresult", "second")
    assert not errors
    old_state = window.state
    begin_load(window, monkeypatch, root / "broken.temresult", "third")
    window._section_archive_status_changed(dict(status="failed", operation="load",
        operation_token="third", identity="third", path=str(root / "broken.temresult"), error="Damaged file"))
    assert "Damaged file" in errors[-1]
    assert window.state is old_state
    assert window.workspace._last_result is executed_section
    assert window.result_files.hold_automatic_preview
    assert not window.preview_timer.isActive()
    assert window.instrument_editor.isEnabled()


@pytest.mark.parametrize("has_old_result", [False, True])
def test_late_publication_failure_restores_every_display_and_cutoff(window, executed_section, monkeypatch, gui_environment, has_old_result):
    _, errors, root = gui_environment
    if has_old_result:
        begin_load(window, monkeypatch, root / "old.temresult", "old")
        deliver(window, executed_section, root / "old.temresult", "old")
        assert not errors
    page = window.workspace.interactive_calculation
    page.section_z.setValue(page.section_z.value() - 2.)
    old_request = page.segment_request()
    old_section = page._section_result
    window.tuning_quality.setCurrentIndex(window.tuning_quality.findData("Medium"))
    window.preview_timer.stop()
    before = capture_instrument_snapshot(window.state).digest
    old_state = window.state
    old_signals = page.particle_signal_status.text()
    old_vacuum = window.workspace.vacuum_map.result_text.text()
    old_summary = window.workspace.result_readout.label.text()
    old_timing = page.calculation_timing.text.toPlainText()
    original = window.workspace.jump_to_ray_position
    def late_failure(*args, **kwargs):
        original(*args, **kwargs)
        raise ValueError("Controlled late presentation failure")
    monkeypatch.setattr(window.workspace, "jump_to_ray_position", late_failure)
    begin_load(window, monkeypatch, root / "new.temresult", "new")
    deliver(window, executed_section, root / "new.temresult", "new")
    assert "Controlled late presentation failure" in errors[-1]
    assert window.state is old_state
    assert capture_instrument_snapshot(window.state).digest == before
    assert window.tuning_quality.currentData() == "Medium"
    assert page.segment_request() == old_request
    assert page._section_result is old_section
    assert page.particle_signal_status.text() == old_signals
    assert page.calculation_timing.text.toPlainText() == old_timing
    assert window.workspace.vacuum_map.result_text.text() == old_vacuum
    assert window.workspace.result_readout.label.text() == old_summary
    assert window.workspace._last_result is (executed_section if has_old_result else None)
    if not has_old_result:
        assert not window.workspace._ray_bundle_records
        assert not window.workspace._pending_ray_panels
        assert window.workspace._preview_result is None
        assert window.workspace.transverse_beam._result is None
        assert window.workspace.eds_page._result is None
        assert not window.workspace.result_readout._records
    assert not window.preview_timer.isActive()


def test_export_uses_historical_display_and_remembers_only_success(window, executed_section, monkeypatch, gui_environment):
    _, errors, root = gui_environment
    begin_load(window, monkeypatch, root / "view.temresult")
    deliver(window, executed_section, root / "view.temresult")
    window.state.lenses[0].percent += .25
    window.result_files.refresh_actions()
    assert window.export_result_action.isEnabled()
    assert window.open_result_action.shortcut().toString() == "Ctrl+O"
    assert window.export_result_action.shortcut().toString() == "Ctrl+S"
    assert window.open_profile_action.shortcut().isEmpty()
    requests = []
    path = window.result_files.library.allocate_path()
    def archive(result, *, path):
        requests.append(result)
        window._section_archive_status_changed(dict(section_archive_summary(result, path=path),
            operation_token="save", status="saving"))
    monkeypatch.setattr(window.calculations, "archive_completed_section", archive)
    window.result_files.save(path, name="Classical result")
    assert requests == [executed_section]
    assert window.result_files.library.entries() == ()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"Successful file event routing fixture; codec validated separately.")
    info = dict(loaded_info(executed_section, path, "save"), status="saved")
    window._section_archive_status_changed(dict(info, operation_token="stale"))
    assert window.result_files.library.entries() == ()
    window._section_archive_status_changed(info)
    assert not errors
    entry, = window.result_files.library.entries()
    assert entry["path"] == str(path.resolve())
    assert entry["name"] == "Classical result"
    before = capture_instrument_snapshot(window.state).digest
    window.result_files.set_startup(entry["id"])
    assert window.result_files.library.startup_entry()["id"] == entry["id"]
    assert capture_instrument_snapshot(window.state).digest == before
    assert not window.preview_timer.isActive()
    other = root / "mismatched.temresult"
    window.result_files.save(other, name="Wrong identity")
    window._section_archive_status_changed(dict(loaded_info(executed_section, other, "save"),
        status="saved", identity="wrong"))
    assert "different calculation identity" in errors[-1]
    assert not window.result_files._saving
    assert window.export_result_action.isEnabled()
    assert len(window.result_files.library.entries()) == 1


@pytest.mark.parametrize("failure", [None, "missing", "corrupt_index"])
def test_startup_saved_choice_never_runs_preset_gun_or_preview(gui_environment, executed_section, monkeypatch, qtbot, failure):
    from temsim.gui import main_window, calculation_controller
    from temsim.optics.electron_gun import source
    from temsim.physics import simulation, particle_sections
    make, errors, root = gui_environment
    library = ResultLibrary(root / "saved_results")
    path = library.allocate_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"Startup selection fixture; no decoding in this dispatch test.")
    info = loaded_info(executed_section, path, "startup")
    entry = library.remember("Saved classic state", info)
    library.set_startup(entry["id"])
    if failure == "corrupt_index":
        library.index_path.write_text("broken", encoding="utf-8")
    elif failure == "missing":
        path.unlink()
    def forbidden(*args, **kwargs):
        pytest.fail("Startup result selection must not execute physical calculations")
    monkeypatch.setattr(main_window.MainWindow, "_apply_state_operating_modes", forbidden)
    monkeypatch.setattr(calculation_controller.CalculationController, "submit_background", forbidden)
    monkeypatch.setattr(source, "trace_source_to_exit", forbidden)
    monkeypatch.setattr(simulation, "run", forbidden)
    monkeypatch.setattr(particle_sections, "run_particle_section", forbidden)
    requests = []
    def load(self, selected):
        requests.append(selected)
        if failure == "missing":
            raise FileNotFoundError("Saved default is missing")
        return "startup"
    monkeypatch.setattr(calculation_controller.CalculationController, "load_section_archive", load)
    window = make()
    qtbot.waitUntil(lambda: bool(requests or errors), timeout=3000)
    assert not window.preview_timer.isActive()
    if failure is None:
        window._particle_section_loaded(executed_section, info)
        assert not errors
        assert window.workspace._last_result is executed_section
    else:
        assert errors
        assert window.workspace._last_result is None
    assert window.result_files.hold_automatic_preview
    window.preview_timer.timeout.emit()
    assert not window.preview_timer.isActive()


def test_library_identity_mismatch_does_not_replace_live_state(window, executed_section, monkeypatch, gui_environment):
    _, errors, root = gui_environment
    before = window.state
    begin_load(window, monkeypatch, root / "changed.temresult")
    window.result_files._expected_entry = {"result_identity": "different", "package_digest": "a" * 64}
    deliver(window, executed_section, root / "changed.temresult")
    assert "replaced" in errors[-1]
    assert window.state is before
    assert window.workspace._last_result is None


def test_live_availability_updates_never_read_or_rebuild_saved_result_menus(window, monkeypatch):
    files = window.result_files
    saved_actions = tuple(files.saved_menu.actions())
    startup_actions = tuple(files.startup_menu.actions())
    def forbidden(*args, **kwargs):
        pytest.fail("A live frame or progress update must not read the result library")
    monkeypatch.setattr(files.library, "entries", forbidden)
    monkeypatch.setattr(files.library, "startup_entry", forbidden)
    for _ in range(50):
        files.refresh_actions()
        window._set_progress_active("calculation", True)
        window._set_progress_active("calculation", False)
    assert tuple(files.saved_menu.actions()) == saved_actions
    assert tuple(files.startup_menu.actions()) == startup_actions
    assert window.open_result_action.isEnabled()
    assert not window.export_result_action.isEnabled()
    files.loading = True
    files.refresh_actions()
    assert not files.named_action.isEnabled()
    assert not files.startup_menu.isEnabled()
    files.loading = False


def test_opening_file_menu_refreshes_library_contents(window, monkeypatch):
    files = window.result_files
    calls = []
    entry = dict(id="entry", name="An existing saved result", path="known-result.temresult",
                 quality="High accuracy", target_z_mm=1234.)
    monkeypatch.setattr(files.library, "entries", lambda: (calls.append("entries") or (entry,)))
    monkeypatch.setattr(files.library, "startup_entry", lambda: (calls.append("startup") or entry))
    # Keep the QAction wrappers alive while retrieving the real File QMenu.
    actions = window.menuBar().actions()
    file_menu = actions[0].menu()
    file_menu.aboutToShow.emit()
    assert calls == ["entries", "startup"]
    assert files.saved_menu.actions()[0].text().startswith(entry["name"])
    assert files.startup_menu.actions()[1].isChecked()


def test_real_compressed_file_background_load_restores_gui_without_transport(window, executed_section, monkeypatch, gui_environment, qtbot):
    from temsim.particle_section_io import save_section_result
    from temsim.optics.electron_gun import source
    from temsim.physics import simulation, particle_sections
    _, errors, root = gui_environment
    path = root / "real.temresult"
    save_section_result(executed_section, path)
    def forbidden(*args, **kwargs):
        pytest.fail("Restoring an executed result must not run physical transport")
    monkeypatch.setattr(source, "trace_source_to_exit", forbidden)
    monkeypatch.setattr(simulation, "run", forbidden)
    monkeypatch.setattr(particle_sections, "run_particle_section", forbidden)
    window.result_files.load(path)
    qtbot.waitUntil(lambda: not window.result_files.loading, timeout=30000)
    assert window.calculations.section_file_pool.waitForDone(3000)
    assert not errors
    loaded = window.workspace._last_result
    assert loaded is not None and loaded is not executed_section
    assert loaded.loaded_section_only
    np.testing.assert_array_equal(loaded.simulation.incident.x, executed_section.simulation.incident.x)
    assert loaded.simulation.section_checkpoint is not None
    assert window.workspace.interactive_calculation._section_result is loaded
    assert window.workspace.interactive_calculation._section_archive_current
    assert window.workspace._ray_bundle_records
    assert str(path) in window.workspace.interactive_calculation.section_archive_status.text()
    assert not window.preview_timer.isActive()
