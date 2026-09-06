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
    window.tuning_quality.setCurrentIndex(window.tuning_quality.findData(quality))
    window.preview_timer.stop()
    calls = []
    monkeypatch.setattr(window.calculations, "submit_background", lambda *args: calls.append(args))
    monkeypatch.setattr(window.calculations, "submit", lambda *args: pytest.fail("Foreground request preparation"))
    window.run_preview()
    profile = TUNING_PROFILES[quality]
    assert calls == [(window.state, quality, profile.rays, profile.step_mm)]


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


def test_live_edits_during_preparation_keep_one_frame_and_latest_pending(window, qtbot, monkeypatch):
    from threading import Event
    from types import SimpleNamespace
    from PySide6.QtCore import QTimer
    from temsim.gui.calculation_controller import CalculationWorker
    from temsim.gui.calculation_request import CapturedCalculationRequest
    from temsim.interactive_calculation import CalculationRange, available_controls

    entered, release = Event(), Event()
    prepare = CapturedCalculationRequest.prepare
    preparations = []

    def held_prepare(request, cancel_event):
        preparations.append(request)
        if len(preparations) == 1:
            entered.set()
            if not release.wait(5):
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
        qtbot.waitUntil(lambda: len(solver_jobs) == 1, timeout=10000)
        first = solver_jobs[0]
        assert first.state.objective_lens.percent == pytest.approx(67.5)
        assert window.state.objective_lens.percent == pytest.approx(68.25)
        assert window._interactive_preview_in_flight(), "Preparation must not finish the full request"
        window.calculations._accept_finished(first.generation, first.quality)
        qtbot.waitUntil(lambda: len(solver_jobs) == 2, timeout=10000)
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
