"""Main-window routing and interaction during detached request preparation."""
from PySide6.QtCore import QSettings
import pytest


@pytest.fixture
def window(qtbot, monkeypatch, tmp_path):
    from temsim.gui import main_window as shell, interactive_calculation as gui, calculation_controller as controller
    settings = QSettings(str(tmp_path / "workspace.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(shell, "QSettings", lambda: settings)
    monkeypatch.setattr(gui, "QSettings", lambda: settings)
    monkeypatch.setattr(controller, "default_artifact_cache_root", lambda: tmp_path / "artifacts")
    monkeypatch.setattr(shell.MainWindow, "INITIAL_PREVIEW_DELAY_MS", 60000)
    instance = shell.MainWindow()
    instance.preview_timer.stop()
    qtbot.addWidget(instance)
    yield instance
    instance.preview_timer.stop()
    instance.calculations.invalidate_pending()
    instance.calculations.pool.waitForDone(3000)


@pytest.mark.parametrize("quality", ["Preview", "Medium"])
def test_toolbar_preview_uses_background_preparation(window, monkeypatch, quality):
    from temsim.physics.optical_tuning import TUNING_PROFILES
    assert window.state.electron_gun.emitter.surface_model is None
    assert window.state.electron_gun.emitter.coherence is None
    assert window.state.electron_gun.source_representation == "classical_particles"
    window.tuning_quality.setCurrentIndex(window.tuning_quality.findData(quality))
    window.preview_timer.stop()
    calls = []
    monkeypatch.setattr(window.calculations, "submit_background", lambda *args: calls.append(args))
    monkeypatch.setattr(window.calculations, "submit", lambda *args: pytest.fail("Foreground request preparation"))
    window.run_preview()
    profile = TUNING_PROFILES[quality]
    assert calls == [(window.state, quality, profile.rays, profile.step_mm)]
    snapshot = window.calculations._calculation_snapshot(*calls[0])
    assert snapshot.electron_gun.emitter.surface_model is None
    snapshot.electron_gun.validate()


def test_tip_entry_points_share_physical_editor_and_invalidate_ray_display(window, monkeypatch):
    from temsim.gui.gun_source_dialog import GunSourceDialog
    from PySide6.QtWidgets import QPushButton
    previous = object()
    window.workspace._last_result = previous
    original = window.state.electron_gun._cache_key(49)
    page = window.workspace.physical_layout.model_editor
    applied = []
    matches = []
    monkeypatch.setattr(window, "apply_direct_alignment", lambda *args: matches.append(args))

    def apply(dialog):
        assert dialog.parentWidget() is page
        assert window.workspace.tabs.currentWidget() is window.workspace.physical_layout
        assert page._selected_key == "feg_tip" and page._mesh_records
        assert all(edit.isReadOnly() for edit in dialog.geometry_inputs.values())
        dialog.surface_enabled.setChecked(not dialog.surface_enabled.isChecked())
        dialog.accept()
        assert dialog.result() == dialog.DialogCode.Accepted, dialog.error.text()
        applied.append(dialog.value())
        return dialog.result()

    monkeypatch.setattr(GunSourceDialog, "exec", apply)
    window.workspace.model_inspector.tip_editor_requested.emit()
    window.preview_timer.stop()
    page._flush_runtime_refresh()
    assert window.state.electron_gun.emitter.surface_model is not None
    assert window.state.electron_gun._cache_key(49) != original
    assert window.workspace._last_result is previous
    assert "awaiting recalculation" in window.workspace.ray_source_status.text()
    assert "Curved tip" in window.workspace.ray_source_status.text()
    assert any("emitting_cap" in mesh["surfaces"] for mesh in page._mesh_records)
    assert not window.parameter_panel.quick_box.isHidden()
    button = next(button for button in window.parameter_panel.quick_box.findChildren(QPushButton)
                  if button.text() == "Open tip in Physical Layout…")
    button.click()
    window.preview_timer.stop()
    page._flush_runtime_refresh()
    assert len(applied) == 2
    assert matches == [("column_transport", .01)]
    assert window.state.electron_gun.emitter.surface_model is None
    assert "Flat tip" in window.workspace.ray_source_status.text()
    emitting = next(mesh for mesh in page._mesh_records if mesh["region"] == "emitting_cap")
    assert (emitting["vertices"][:, 2] == 0).all()
    assert window.workspace._last_result is previous


def test_tip_navigation_preserves_unsaved_geometry(window, monkeypatch):
    from temsim.gui.gun_source_dialog import GunSourceDialog
    window._reveal_physical_model("feg_tip", 0.)
    page = window.workspace.physical_layout.model_editor
    page._open_pending_part()
    page.session.set_dimension(("parts", "feg_tip", "tip_radius_nm"), 120.)
    before = window.state.electron_gun.to_dict()
    monkeypatch.setattr(GunSourceDialog, "exec", lambda _: pytest.fail("Cannot replace a geometry draft"))
    window._open_tip_editor()
    assert page.session.dirty
    assert page.session.part("feg_tip")["tip_radius_nm"] == 120.
    assert window.state.electron_gun.to_dict() == before
    assert "Save or revert" in page.status.text()


def test_capture_error_does_not_leave_live_queue_running(window, monkeypatch):
    def fail_capture(*_args):
        raise ValueError("Controlled invalid capture")
    errors = []
    monkeypatch.setattr(window.calculations, "submit_background", fail_capture)
    monkeypatch.setattr(window, "_show_error", errors.append)
    old_plot = object()
    window.workspace._last_result = old_plot
    window._interactive_preview_pending = True
    window.run_preview()
    assert errors == ["Controlled invalid capture"]
    assert window._interactive_preview_generation is None
    assert not window._interactive_preview_pending
    assert not window.preview_timer.isActive()
    assert window.workspace._last_result is old_plot


@pytest.mark.parametrize("coherent", [False, True])
def test_rejected_surface_image_keeps_applied_source_and_previous_result(window, qtbot, monkeypatch, coherent):
    from temsim.gui.gun_source_dialog import GunSourceDialog
    from temsim.instrument_snapshot import capture_instrument_snapshot

    def apply(dialog):
        dialog.surface_enabled.setChecked(True)
        dialog.surface_coherent.setChecked(coherent)
        dialog.match_transport.setChecked(False)  # Exercise image admission without a concurrent alignment.
        dialog.accept()
        assert dialog.result() == dialog.DialogCode.Accepted, dialog.error.text()
        return dialog.result()

    monkeypatch.setattr(GunSourceDialog, "exec", apply)
    window.workspace.model_inspector._edit_gun_source()
    window.preview_timer.stop()
    window.state.ac_deflector.enabled = True
    window.state.ac_deflector.scan_enabled = True
    window.state.sample.stem_wave_enabled = True
    before = capture_instrument_snapshot(window.state).digest
    previous = object()
    window.workspace._last_result = previous
    errors = []
    monkeypatch.setattr(window, "_show_error", errors.append)
    monkeypatch.setattr(window.calculations.pool, "start", lambda *_: pytest.fail("Unsupported imaging must not start a worker"))
    window.run_high_accuracy()
    assert len(errors) == 1 and "Wave imaging" in errors[0]
    assert "Source selection and previous results are unchanged" in errors[0]
    assert capture_instrument_snapshot(window.state).digest == before
    assert window.workspace._last_result is previous
    reopened = GunSourceDialog(window.state.electron_gun, instrument_state=window.state)
    qtbot.addWidget(reopened)
    assert reopened.surface_enabled.isChecked()
    assert reopened.surface_coherent.isChecked() == coherent


def test_live_edits_during_preparation_keep_one_frame_and_latest_pending(window, qtbot, monkeypatch):
    from threading import Event
    from types import SimpleNamespace
    from PySide6.QtCore import QTimer
    from temsim.gui.calculation_controller import CalculationWorker
    from temsim.gui.calculation_request import CapturedCalculationRequest
    from temsim.interactive_calculation import CalculationRange, available_controls

    entered, release = Event(), Event()
    errors = []
    monkeypatch.setattr(window, "_show_error", errors.append)
    prepare = CapturedCalculationRequest.prepare
    preparations = []

    def held_prepare(request, cancel_event):
        preparations.append(request)
        if len(preparations) == 1:
            entered.set()
            if not release.wait(30):
                raise RuntimeError("Test preparation release timed out")
        return prepare(request, cancel_event)

    monkeypatch.setattr(CapturedCalculationRequest, "prepare", held_prepare)
    start = window.calculations.pool.start
    solver_jobs = []

    def dispatch(worker):
        if isinstance(worker, CalculationWorker):
            solver_jobs.append(worker)
        else:
            start(worker)

    monkeypatch.setattr(window.calculations.pool, "start", dispatch)
    old_high = SimpleNamespace(model_signature="retained complete result")
    window.workspace._high_accuracy_result = old_high
    window.workspace._high_accuracy_current = False
    control = next(c for c in available_controls(window.state) if c.key == "objective_lens" and c.field == "percent")
    axis = CalculationRange(control, 0., 100.)
    try:
        window._apply_interactive_tuning(((axis, 67.5),))
        window.preview_timer.stop()
        window.run_preview()
        qtbot.waitUntil(entered.is_set, timeout=5000)
        generation = window.calculations.generation
        cancel_event = window.calculations._cancel_event
        assert window._interactive_preview_in_flight()
        assert "calculation" in window._progress_owners
        ticks = []
        QTimer.singleShot(0, lambda: ticks.append(True))
        qtbot.waitUntil(lambda: bool(ticks))
        for value in (67.6, 67.7, 68.25):
            window._apply_interactive_tuning(((axis, value),))
        assert window.calculations.generation == generation
        assert not cancel_event.is_set()
        assert window._interactive_preview_pending
        assert solver_jobs == []
        release.set()
        qtbot.waitUntil(lambda: len(solver_jobs) == 1 or bool(errors), timeout=20000)
        assert not errors
        first = solver_jobs[0]
        assert first.state.objective_lens.percent == pytest.approx(67.5)
        assert window.state.objective_lens.percent == pytest.approx(68.25)
        assert window._interactive_preview_in_flight(), "Preparation must not finish the full request"
        window.calculations._accept_finished(first.generation, first.quality)
        qtbot.waitUntil(lambda: len(solver_jobs) == 2 or bool(errors), timeout=20000)
        assert not errors
        second = solver_jobs[1]
        assert second.state.objective_lens.percent == pytest.approx(68.25)
        assert len(preparations) == 2
        window.calculations._accept_finished(second.generation, second.quality)
        assert not window._interactive_preview_pending
        assert "calculation" not in window._progress_owners
        assert window.workspace._high_accuracy_result is old_high
    finally:
        release.set()
        window.calculations.invalidate_pending()
