"""Ray Diagram consolidation and native dock lifecycle; no physics solves."""
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QDockWidget, QSlider

from temsim.gui import interactive_calculation as gui
from temsim.gui import main_window as shell
from temsim.component_keys import OBJECTIVE_LENS


@pytest.fixture
def windows(qtbot, monkeypatch, tmp_path):
    settings = QSettings(str(tmp_path / "workspace.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(shell, "QSettings", lambda: settings)
    monkeypatch.setattr(gui, "QSettings", lambda: settings)
    monkeypatch.setattr(shell.MainWindow, "INITIAL_PREVIEW_DELAY_MS", 60000)

    def make():
        window = shell.MainWindow()
        window.preview_timer.stop()
        qtbot.addWidget(window)
        monkeypatch.setattr(window.calculations.pool, "start", lambda *_: pytest.fail("Unexpected calculation"))
        window.show()
        qtbot.wait(20)
        return window

    return make, settings


def _configure(window):
    window._capture_interactive_settings()
    page = window.workspace.interactive_calculation
    page.choice.setCurrentIndex(next(i for i in range(page.choice.count())
                                    if page.choice.itemData(i).key == OBJECTIVE_LENS
                                    and page.choice.itemData(i).field == "percent"))
    page._add_range()
    page.ranges.cellWidget(0, 1).setText("67")
    page.ranges.cellWidget(0, 2).setText("70")
    page.start_live_tuning()
    page.timer.stop()
    assert page._live_mode, page.status.text()
    return page


def test_live_tuning_is_a_toggleable_dock_and_ray_diagram_has_no_duplicate_plot(windows, qtbot):
    make, _ = windows
    window = make()
    workspace = window.workspace
    page = workspace.interactive_calculation
    dock = window.findChild(QDockWidget, "liveTuningDock")
    assert dock is window.live_tuning_dock and dock.widget() is page
    assert dock.features() & QDockWidget.DockWidgetFeature.DockWidgetFloatable
    assert dock.isHidden()
    assert not window._live_tuning_layout_initialized
    assert workspace.tabs.indexOf(page) == -1
    assert "Interactive Calculation" not in [workspace.tabs.tabText(i) for i in range(workspace.tabs.count())]
    assert workspace.ray_result_tabs.tabText(0) == "Rays"
    assert workspace.ray_result_tabs.tabText(1) == "Cached signals"
    assert workspace.ray_page.isAncestorOf(page.readout_panel)
    assert not page.isAncestorOf(page.readout_panel)
    assert not hasattr(page, "tuning_plot")
    assert page.readout_panel.isAncestorOf(page.signal_table)
    assert not hasattr(page, "views")
    assert not hasattr(page, "tem_plot") and not hasattr(page, "stem_plot")
    menu_actions = window.menuBar().actions()
    view_action = next(action for action in menu_actions if action.text() == "View")
    view = view_action.menu()
    action = dock.toggleViewAction()
    assert action in view.actions()
    assert workspace.live_tuning_toggle.defaultAction() is action
    workspace.tabs.setCurrentWidget(workspace.eds_page)
    action.trigger()
    qtbot.waitUntil(page.isVisible)
    assert window._live_tuning_layout_initialized
    assert dock in window.tabifiedDockWidgets(window.instrument_dock)
    assert workspace.tabs.currentWidget() is workspace.ray_page
    assert workspace.ray_result_tabs.currentWidget() is workspace.ray_workspace_splitter
    assert action.isChecked()
    qtbot.mouseClick(workspace.live_tuning_toggle, Qt.MouseButton.LeftButton)
    assert dock.isHidden() and not action.isChecked()
    assert not page.timer.isActive() and not window.preview_timer.isActive()


def test_dock_close_and_reopen_retains_controls_readout_and_cache(windows, qtbot, monkeypatch):
    make, _ = windows
    window = make()
    dock = window.live_tuning_dock
    dock.toggleViewAction().trigger()
    monkeypatch.setattr(window, "resizeDocks", lambda *_: pytest.fail("Do not reset the user's dock width"))
    page = _configure(window)
    slider = page.live_table.findChild(QSlider)
    value = next(iter(page.live_widgets.values())).value()
    bank, readout, high = object(), object(), object()
    page.controller.bank = bank
    page._readout = readout
    window.workspace._high_accuracy_result = high
    page.result_status.setText("Cached result retained")
    window.workspace.plot.setRange(xRange=(1590, 1610), yRange=(-0.4, 0.4), padding=0)
    before = window.workspace.plot.viewRange()
    for _ in range(2):
        dock.close()
        assert dock.isHidden()
        dock.toggleViewAction().trigger()
        qtbot.waitUntil(page.isVisible)
        assert page.live_table.findChild(QSlider) is slider
        assert next(iter(page.live_widgets.values())).value() == value
        assert page.controller.bank is bank and page._readout is readout
        assert window.workspace._high_accuracy_result is high
        assert page.result_status.text() == "Cached result retained"
        assert window.workspace.plot.viewRange() == before
        assert not page.timer.isActive() and not window.preview_timer.isActive()
    page.controller.bank = None
    page._readout = None


def test_shared_ray_status_cannot_replace_cached_signal_readout(windows, qtbot, monkeypatch):
    make, _ = windows
    window = make()
    page = _configure(window)
    old_readout = object()
    page._readout = old_readout
    page.result_status.setText("Cached detector result")
    page.notes.setText("Cached model details")
    frame = SimpleNamespace(simulation=SimpleNamespace(metrics={"tuning_quality": "Medium"}))
    rendered = []
    monkeypatch.setattr(window.workspace, "display_result", lambda r, q: rendered.append((r, q)))
    monkeypatch.setattr(window, "_interactive_preview_in_flight", lambda: True)
    window._interactive_preview_pending = True
    window._calculation_ready("Medium", frame, 0.1)
    assert rendered == [(frame, "Medium")]
    assert "last completed frame" in page.live_status.text()
    assert page._readout is old_readout
    assert page.result_status.text() == "Cached detector result"
    assert page.notes.text() == "Cached model details"
    # Inspecting cached results and returning to rays only changes visibility.
    window.workspace.ray_result_tabs.setCurrentWidget(page.readout_panel)
    assert window.workspace.ray_result_tabs.currentWidget() is page.readout_panel
    window.workspace.show_ray_diagram()
    assert window.workspace.ray_result_tabs.currentWidget() is window.workspace.ray_workspace_splitter
    assert page._readout is old_readout


def test_live_tuning_dock_float_visibility_and_size_restore(windows, qtbot):
    make, settings = windows
    window = make()
    dock = window.live_tuning_dock
    dock.toggleViewAction().trigger()
    dock.setFloating(True)
    # Stay within the virtual screen where possible: Qt clamps off-screen
    # floating windows on restore (the same safeguard used on real monitors).
    dock.resize(max(dock.minimumSizeHint().width(), dock.screen().availableGeometry().width() - 100), 630)
    dock.move(30, 30)
    qtbot.wait(20)
    size = dock.size()
    window.close()
    assert not settings.value(window.SETTINGS_STATE).isEmpty()
    restored = make()
    qtbot.waitUntil(restored.live_tuning_dock.isVisible)
    assert restored.live_tuning_dock.isFloating()
    assert restored._live_tuning_layout_initialized
    assert restored.live_tuning_dock.size() == size
    assert restored.workspace.live_tuning_toggle.isChecked()
    restored.reset_workspace()
    assert not restored.live_tuning_dock.isFloating()
    assert restored.live_tuning_dock.isHidden()
    assert not restored.workspace.live_tuning_toggle.isChecked()
