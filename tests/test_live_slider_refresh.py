"""Continuous drag scheduling with Qt events and controlled worker completion."""
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QSlider

from temsim.component_keys import OBJECTIVE_LENS
from temsim.gui.interactive_calculation import InteractiveCalculationPage
from temsim.gui.main_window import MainWindow
from temsim.optics.column import default_state


@pytest.fixture
def controlled_solver_requests(monkeypatch):
    """Keep queue unit tests at the solver boundary; preparation has its own tests.

    The real sustained-drag test below uses the production background path.
    """
    def install(window):
        monkeypatch.setattr(window.calculations, "submit_background", window.calculations.submit)
    return install


def _configure(page, state):
    page.set_source(state)
    index = next(i for i in range(page.choice.count())
                 if page.choice.itemData(i).key == OBJECTIVE_LENS
                 and page.choice.itemData(i).field == "percent")
    page.choice.setCurrentIndex(index)
    page._add_range()
    page.ranges.cellWidget(0, 1).setText("67")
    page.ranges.cellWidget(0, 2).setText("70")
    page.start_live_tuning()
    page.timer.stop()
    assert page._live_mode
    return page.live_table.findChild(QSlider)


def test_held_slider_emits_updates_before_mouse_release(qtbot):
    page = InteractiveCalculationPage()
    qtbot.addWidget(page)
    slider = _configure(page, default_state())
    updates = []
    page.tuning_changed.connect(lambda values: updates.append((slider.isSliderDown(), values)))
    slider.setSliderDown(True)
    for position in range(1000, 7000, 150):
        slider.setSliderPosition(position)
        qtbot.wait(15)
    assert len(updates) >= 2, "Continuous motion must not restart the refresh deadline"
    assert all(held for held, _ in updates)
    slider.setSliderPosition(8123)
    slider.setSliderDown(False)
    assert updates[-1][1][0][1] == pytest.approx(67 + 3 * 8123 / 10000)
    assert not page.timer.isActive()
    page.shutdown()


def test_live_edits_do_not_cancel_the_frame_in_flight(qtbot, monkeypatch, controlled_solver_requests):
    window = MainWindow()
    qtbot.addWidget(window)
    window.preview_timer.stop()
    controlled_solver_requests(window)
    workers = []
    monkeypatch.setattr(window.calculations.pool, "start", workers.append)
    page = window.workspace.interactive_calculation
    _configure(page, window.state)
    axis = page.live_plan.ranges[0]
    window._apply_interactive_tuning(((axis, 67.5),))
    window.preview_timer.stop()
    window.run_preview()
    first = workers[0]
    for value in (67.6, 67.7, 68.0, 68.2):
        window._apply_interactive_tuning(((axis, value),))
    assert not first.cancel_event.is_set(), "New drag values must allow the current frame to finish"
    assert len(workers) == 1
    assert first.state.objective_lens.percent == pytest.approx(67.5)
    assert window.state.objective_lens.percent == pytest.approx(68.2)
    window.calculations._accept_finished(first.generation, first.quality)
    qtbot.waitUntil(lambda: len(workers) == 2)
    assert workers[1].state.objective_lens.percent == pytest.approx(68.2)
    window.calculations._accept_finished(workers[1].generation, workers[1].quality)
    qtbot.wait(100)
    assert len(workers) == 2
    page.shutdown()


@pytest.mark.parametrize("next_action", ["edit", "high", "error"])
def test_new_model_edits_high_accuracy_and_errors_exit_live_queue_safely(
        qtbot, monkeypatch, next_action, controlled_solver_requests):
    window = MainWindow()
    qtbot.addWidget(window)
    window.preview_timer.stop()
    controlled_solver_requests(window)
    workers = []
    monkeypatch.setattr(window.calculations.pool, "start", workers.append)
    monkeypatch.setattr(window, "_show_error", lambda *_: None)
    page = window.workspace.interactive_calculation
    _configure(page, window.state)
    axis = page.live_plan.ranges[0]
    window._apply_interactive_tuning(((axis, 67.5),))
    window.preview_timer.stop()
    window.run_preview()
    first = workers[0]
    window._apply_interactive_tuning(((axis, 68.25),))
    if next_action == "edit":
        window.schedule_preview("sample_changed")
        window.preview_timer.stop()
        assert first.cancel_event.is_set()
        assert not window._interactive_preview_pending
        window.calculations._accept_finished(first.generation, first.quality)
        qtbot.wait(100)
        assert len(workers) == 1
    elif next_action == "high":
        window.run_high_accuracy()
        assert first.cancel_event.is_set()
        assert len(workers) == 2 and workers[1].quality == "High accuracy"
        assert workers[1].state.objective_lens.percent == pytest.approx(68.25)
        assert not window._interactive_preview_pending
        window.calculations._accept_finished(first.generation, first.quality)
        window.calculations._accept_finished(workers[1].generation, workers[1].quality)
        qtbot.wait(100)
        assert len(workers) == 2
    else:
        window.calculations._accept_error(first.generation, first.quality, "Controlled failure")
        window.calculations._accept_finished(first.generation, first.quality)
        qtbot.waitUntil(lambda: len(workers) == 2)
        assert workers[1].state.objective_lens.percent == pytest.approx(68.25)
        window.calculations._accept_finished(workers[1].generation, workers[1].quality)
    page.shutdown()


def test_intermediate_frame_is_labelled_and_never_writes_back_lens_values(qtbot, monkeypatch, controlled_solver_requests):
    window = MainWindow()
    qtbot.addWidget(window)
    window.preview_timer.stop()
    controlled_solver_requests(window)
    workers = []
    monkeypatch.setattr(window.calculations.pool, "start", workers.append)
    page = window.workspace.interactive_calculation
    _configure(page, window.state)
    axis = page.live_plan.ranges[0]
    window._apply_interactive_tuning(((axis, 67.5),))
    window.preview_timer.stop()
    window.run_preview()
    window._apply_interactive_tuning(((axis, 68.25),))
    rendered = []
    monkeypatch.setattr(window.workspace, "display_result", lambda r, q: rendered.append((r, q)))
    monkeypatch.setattr(page, "display_tuning_status", lambda *a: None)
    monkeypatch.setattr(window.workspace.model_inspector, "display_result",
                        lambda *a: pytest.fail("Intermediate frame must not replace current diagnostics"))
    old_high = SimpleNamespace(model_signature="retained")
    window.workspace._high_accuracy_result = old_high
    frame = SimpleNamespace(state_snapshot=workers[0].state)
    window.calculations._accept_result(workers[0].generation, "Preview", frame, 0.1)
    assert rendered == [(frame, "Preview")]
    assert "last completed frame" in page.live_status.text()
    assert "Updating" in window.workspace.heading.text()
    assert window.state.objective_lens.percent == pytest.approx(68.25)
    assert window.workspace._high_accuracy_result is old_high
    window.calculations.invalidate_pending()
    page.shutdown()


def test_real_ray_frames_arrive_during_sustained_slider_motion(qtbot, monkeypatch):
    """Use the real small-bundle solver; no specimen or high-accuracy solve."""
    window = MainWindow()
    qtbot.addWidget(window)
    window.preview_timer.stop()
    errors = []
    monkeypatch.setattr(window, "_show_error", errors.append)
    window.state.acceleration_enabled = False
    window.state.acceleration_backend = "CPU"
    page = window.workspace.interactive_calculation
    slider = _configure(page, window.state)
    frames = []
    window.calculations.result_ready.connect(
        lambda quality, result, duration: frames.append(
            (slider.isSliderDown(), result.state_snapshot.objective_lens.percent,
             result.simulation.incident.x.copy())))
    old_high = SimpleNamespace(model_signature="retained high frame")
    window.workspace._high_accuracy_result = old_high
    window.workspace._high_accuracy_current = False
    position = [1500]
    motion = QTimer(window)

    def move():
        position[0] = 1000 + (position[0] + 170) % 7000
        slider.setSliderPosition(position[0])

    motion.timeout.connect(move)
    slider.setSliderDown(True)
    motion.start(15)
    try:
        qtbot.waitUntil(lambda: len(frames) >= 2 or bool(errors), timeout=30000)
        assert not errors
        assert all(held for held, _, _ in frames)
        assert len({round(value, 8) for _, value, _ in frames}) >= 2
        assert not np.array_equal(frames[0][2], frames[1][2], equal_nan=True)
        assert window.workspace._high_accuracy_result is old_high
        motion.stop()
        slider.setSliderPosition(9123)
        slider.setSliderDown(False)
        expected = 67 + 3 * 9123 / 10000
        qtbot.waitUntil(lambda: abs(frames[-1][1] - expected) < 1e-8 or bool(errors), timeout=15000)
        assert not errors
        assert window.state.objective_lens.percent == pytest.approx(expected)
        assert window.workspace._last_result.state_snapshot.objective_lens.percent == pytest.approx(expected)
        assert window.workspace.plot.listDataItems()
        assert not hasattr(page, "tuning_plot")
        assert window.workspace._high_accuracy_result is old_high
        assert not window.calculations.completed_high_accuracy_results()
    finally:
        motion.stop()
        window.preview_timer.stop()
        window.calculations.invalidate_pending()
        window.calculations.pool.waitForDone(3000)
        page.shutdown()
