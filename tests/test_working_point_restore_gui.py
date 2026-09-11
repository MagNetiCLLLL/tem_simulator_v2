"""Complete-state restore keeps selectors truthful without editing that state."""
from dataclasses import replace

import pytest
from PySide6.QtCore import QSettings

from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.working_point import WorkingPointCheckpoint


@pytest.fixture
def window(qtbot, monkeypatch, tmp_path):
    from temsim.gui import main_window, interactive_calculation, calculation_controller

    settings = QSettings(str(tmp_path / "workspace.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(main_window, "QSettings", lambda: settings)
    monkeypatch.setattr(interactive_calculation, "QSettings", lambda: settings)
    monkeypatch.setattr(calculation_controller, "default_artifact_cache_root", lambda: tmp_path / "artifacts")
    monkeypatch.setattr(main_window.MainWindow, "INITIAL_PREVIEW_DELAY_MS", 60000)
    result = main_window.MainWindow()
    result.preview_timer.stop()
    qtbot.addWidget(result)
    yield result
    result.preview_timer.stop()
    result.calculations.invalidate_pending()
    result.calculations.pool.waitForDone(3000)
    # Dispose while settings/cache patches still apply. A merely hidden window
    # can retain timers and show a modal error in a later event-loop test.
    result.close()
    result.deleteLater()
    from PySide6.QtCore import QCoreApplication, QEvent
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def checkpoint(window):
    return WorkingPointCheckpoint(capture_instrument_snapshot(window.state), {},
                                  window.state.sample.z_mm, "gui-restore-fixture", {})


def change_assembly_and_backend(window):
    selection = replace(window.selection, column="C3", gun="FEG + Mono")
    window.assembly_panel.set_selection(selection)
    window.load_assembly(selection)
    window.preview_timer.stop()
    window.compute_backend.setCurrentIndex(window.compute_backend.findData("CPU"))
    window.preview_timer.stop()


def assert_restored(window, saved):
    assert capture_instrument_snapshot(window.state).digest == saved.snapshot.digest
    assert window.assembly_panel.gun.currentText() == window.selection.gun
    assert window.assembly_panel.column.currentText() == window.selection.column
    assert window.assembly_panel.beam_blanker.currentText() == window.selection.beam_blanker
    assert window.compute_backend.currentData() == window.state.acceleration_backend
    assert window._active_working_checkpoint is saved
    assert not window.preview_timer.isActive()


@pytest.mark.parametrize("fork", [False, True])
def test_restore_synchronizes_assembly_and_backend_without_signals(window, monkeypatch, fork):
    saved = checkpoint(window)
    change_assembly_and_backend(window)
    edits = []
    for combo in (window.assembly_panel.gun, window.assembly_panel.column,
                  window.assembly_panel.beam_blanker, window.compute_backend):
        combo.currentIndexChanged.connect(lambda *_: edits.append(True))
    errors = []
    monkeypatch.setattr(window, "_show_error", errors.append)
    window._restore_working_point(saved, fork)
    assert not errors
    assert not edits
    assert_restored(window, saved)
    assert window._working_point_parent == (saved.digest if fork else None)


def test_alignment_undo_synchronizes_selectors(window, monkeypatch):
    from test_alignment_transactions import candidate_for

    saved = checkpoint(window)
    candidate = candidate_for(window.state, revision=window._physical_revision)
    updated = window._alignment_commits.apply(window.state, candidate,
        revision=window._physical_revision, previous_checkpoint=saved)
    window._install_working_point(updated, candidate.checkpoint)
    window._physical_revision += 1
    change_assembly_and_backend(window)
    errors = []
    monkeypatch.setattr(window, "_show_error", errors.append)
    window._undo_alignment()
    assert not errors
    assert_restored(window, saved)
    with pytest.raises(ValueError, match="No applied Direct Alignment"):
        window._alignment_commits.peek_undo()


def test_failed_restore_rolls_selectors_and_state_back(window, monkeypatch):
    saved = checkpoint(window)
    change_assembly_and_backend(window)
    before = capture_instrument_snapshot(window.state)
    before_selection = window.selection
    original = window._sync_working_point_selectors

    def fail_after_selector_refresh():
        original()
        if window.selection != before_selection:
            raise ValueError("Controlled view failure after selector refresh")

    monkeypatch.setattr(window, "_sync_working_point_selectors", fail_after_selector_refresh)
    errors = []
    monkeypatch.setattr(window, "_show_error", errors.append)
    window._restore_working_point(saved)
    assert errors and "Controlled view failure" in errors[0]
    assert capture_instrument_snapshot(window.state).digest == before.digest
    assert window.selection == before_selection
    assert window.assembly_panel.gun.currentText() == before_selection.gun
    assert window.assembly_panel.column.currentText() == before_selection.column
    assert window.compute_backend.currentData() == window.state.acceleration_backend


def test_historical_exit_source_checkpoint_cannot_replace_live_state(window, monkeypatch):
    from temsim.instrument_snapshot import decode_instrument, encode_instrument
    before = capture_instrument_snapshot(window.state)
    historical = decode_instrument(encode_instrument(window.state))
    historical.electron_gun.source_representation = "effective_gaussian_schell"
    saved = WorkingPointCheckpoint(capture_instrument_snapshot(historical), {},
                                  historical.sample.z_mm, "historical-exit-fixture", {})
    errors = []
    monkeypatch.setattr(window, "_show_error", errors.append)
    window._restore_working_point(saved)
    assert errors and "Custom exit sources are not permitted" in errors[0]
    assert capture_instrument_snapshot(window.state).digest == before.digest
