"""Native virtual-electron dock integration, with no instrument calculation."""
from copy import deepcopy
from threading import Event

import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QDockWidget, QScrollArea, QSplitter

from temsim.gui import interactive_calculation as gui
from temsim.gui import main_window as shell
from tests.test_magnetic_test_electron_gui import (
    UniformScene, install_trace, synthetic_trajectory, wait_for_result,
)


@pytest.fixture
def windows(qtbot, monkeypatch, tmp_path):
    settings = QSettings(str(tmp_path / "electron-dock.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(shell, "QSettings", lambda: settings)
    monkeypatch.setattr(gui, "QSettings", lambda: settings)
    monkeypatch.setattr(shell.MainWindow, "INITIAL_PREVIEW_DELAY_MS", 60000)
    monkeypatch.setattr(shell.MainWindow, "_apply_state_operating_modes", lambda *_: object())
    created = []

    def make():
        window = shell.MainWindow()
        window.preview_timer.stop()
        qtbot.addWidget(window)
        monkeypatch.setattr(window.calculations.pool, "start",
                            lambda *_: pytest.fail("Dock changes must not launch instrument calculations"))
        window.show()
        qtbot.wait(20)
        created.append(window)
        return window

    yield make, settings
    for window in created:
        page = window.workspace.magnetic_field.field_lines
        page.set_active(False)
        qtbot.waitUntil(lambda: page._worker is None and page.electron._worker is None
                        and page.electron._scene_worker is None, timeout=10000)


def electron_controller(window):
    return window.workspace.magnetic_field.field_lines.electron


def show_electron_dock(window, qtbot):
    electron_controller(window).show_controls()
    qtbot.waitUntil(window.virtual_electrons_dock.isVisible)
    return window.virtual_electrons_dock


def test_electron_controls_are_a_native_tabified_dock_with_view_menu_and_toolbar_action(windows, qtbot):
    make, _settings = windows
    window = make()
    controller = electron_controller(window)
    dock = window.findChild(QDockWidget, "virtualElectronsDock")
    assert dock is window.virtual_electrons_dock and dock.widget() is controller.panel
    assert not hasattr(controller, "dialog")
    for feature in (QDockWidget.DockWidgetFeature.DockWidgetClosable,
                    QDockWidget.DockWidgetFeature.DockWidgetMovable,
                    QDockWidget.DockWidgetFeature.DockWidgetFloatable):
        assert dock.features() & feature
    assert dock.allowedAreas() == Qt.DockWidgetArea.AllDockWidgetAreas
    assert dock.isHidden()
    menu_actions = window.menuBar().actions()
    view_action = next(action for action in menu_actions if action.text() == "View")
    view_menu = view_action.menu()
    assert dock.toggleViewAction() in view_menu.actions()
    assert "virtualElectronControlsSplitter" in window.workspace_layouts.splitters
    window.workspace.tabs.setCurrentWidget(window.workspace.eds_page)
    dock.toggleViewAction().trigger()
    qtbot.waitUntil(controller.panel.isVisible)
    assert dock in window.tabifiedDockWidgets(window.instrument_dock)
    assert window.workspace.tabs.currentWidget() is window.workspace.ray_page
    assert window.workspace.magnetic_field_toggle.isChecked()
    assert window.workspace.magnetic_field.display_mode.currentData() == "electron"
    assert dock.toggleViewAction().isChecked()
    dock.close()
    assert dock.isHidden() and not dock.toggleViewAction().isChecked()
    controller.controls_button.click()
    qtbot.waitUntil(controller.panel.isVisible)
    assert dock.toggleViewAction().isChecked() and not window.preview_timer.isActive()


def test_dock_close_and_view_switch_retain_electron_records_results_and_user_width(windows, qtbot, monkeypatch):
    make, _settings = windows
    window = make()
    controller = electron_controller(window)
    calls = install_trace(monkeypatch)
    before_state = deepcopy(window.state.to_dict())
    dock = show_electron_dock(window, qtbot)
    controller.background.setChecked(False)
    scene = UniformScene(initial_energy_ev=.3, initial_position_m=(0., 0., 0.))
    controller.set_scene(scene)
    controller.set_active(True)
    wait_for_result(qtbot, controller)
    first_key = controller.selected_record.key
    controller.duplicate_button.click()
    controller.energy.setValue(.7)
    last_result = wait_for_result(qtbot, controller)
    definitions = tuple((r.key, r.settings, r.trajectory) for r in controller.records)
    cache = dict(controller._cache)
    monkeypatch.setattr(window, "resizeDocks", lambda *_: pytest.fail("Reopening must retain user dock width"))
    for _ in range(2):
        dock.close()
        assert dock.isHidden()
        show_electron_dock(window, qtbot)
        assert controller.current_trajectory is last_result
        assert all((r.key, r.settings) == (key, settings) and r.trajectory is result
                   for r, (key, settings, result) in zip(controller.records, definitions))
        assert len(controller._cache) == len(cache)
        assert all(controller._cache[key] is result for key, result in cache.items())
    window.workspace.magnetic_field.display_mode.setCurrentIndex(
        window.workspace.magnetic_field.display_mode.findData("3d"))
    assert dock.isVisible() and not controller._active
    controller.select_electron(first_key)
    assert controller.current_trajectory is definitions[0][2]
    show_electron_dock(window, qtbot)
    qtbot.wait(200)
    assert len(calls) == 2 and controller._scene is scene
    assert window.state.to_dict() == before_state
    assert not window.preview_timer.isActive()


def test_floating_dock_is_compact_and_advanced_controls_remain_scroll_reachable(windows, qtbot):
    make, _settings = windows
    window = make()
    controller = electron_controller(window)
    dock = show_electron_dock(window, qtbot)
    dock.setFloating(True)
    dock.resize(480, 640)
    qtbot.wait(50)
    assert dock.width() <= 480 and dock.height() <= 640
    assert controller.panel.minimumSizeHint().width() <= 480
    splitter = controller.panel.findChild(QSplitter, "virtualElectronControlsSplitter")
    assert splitter is not None and splitter.orientation() == Qt.Orientation.Vertical
    assert controller.electron_list.height() <= 230
    controller.advanced_toggle.setChecked(True)
    qtbot.wait(30)  # Let the scroll area measure the newly expanded controls.
    scroll = next(scroll for scroll in controller.panel.findChildren(QScrollArea)
                  if scroll.widget().isAncestorOf(controller.reset_to_tip))
    for control in (controller.position_tolerance, controller.reset_to_tip):
        scroll.ensureWidgetVisible(control)
        qtbot.wait(10)
        centre = control.mapTo(scroll.viewport(), control.rect().center())
        assert scroll.viewport().rect().contains(centre)
    assert not window.preview_timer.isActive()


def test_named_layouts_restore_electron_dock_and_do_not_change_physical_inputs(windows, qtbot):
    make, _settings = windows
    window = make()
    before_state = deepcopy(window.state.to_dict())
    manager = window.workspace_layouts
    saved_id = manager.save_as("Virtual electron comparison")
    dock = show_electron_dock(window, qtbot)
    dock.setFloating(True)
    dock.resize(480, 620)
    dock.move(30, 30)
    qtbot.wait(30)
    expected_size = dock.size()
    manager.save_current()
    manager.select("default")
    qtbot.wait(30)
    assert dock.isHidden() and not dock.isFloating()
    manager.select(saved_id)
    qtbot.waitUntil(dock.isVisible)
    assert dock.isFloating() and dock.size() == expected_size
    assert window.state.to_dict() == before_state
    assert not window.preview_timer.isActive()


def test_floating_dock_geometry_and_visibility_restore_after_restart_and_reset(windows, qtbot):
    make, settings = windows
    window = make()
    dock = show_electron_dock(window, qtbot)
    dock.setFloating(True)
    dock.resize(480, 620)
    dock.move(30, 30)
    qtbot.wait(30)
    expected_size = dock.size()
    window.close()
    assert not settings.value(window.SETTINGS_STATE).isEmpty()
    restored = make()
    restored_dock = restored.virtual_electrons_dock
    qtbot.waitUntil(restored_dock.isVisible)
    assert restored_dock.isFloating() and restored_dock.size() == expected_size
    assert restored_dock.toggleViewAction().isChecked()
    restored.reset_workspace()
    qtbot.wait(30)
    assert restored_dock.isHidden() and not restored_dock.isFloating()
    assert not restored_dock.toggleViewAction().isChecked()
    show_electron_dock(restored, qtbot)
    assert restored_dock in restored.tabifiedDockWidgets(restored.instrument_dock)


def test_main_window_cancels_virtual_electron_before_waiting_for_other_numerical_jobs(windows, qtbot, monkeypatch):
    make, _settings = windows
    window = make()
    controller = electron_controller(window)
    entered, release, cancellation_seen = Event(), Event(), Event()

    def trace(_scene, settings, *, cancelled, **_kwargs):
        entered.set()
        while not cancelled() and not release.wait(.005):
            pass
        if cancelled():
            cancellation_seen.set()
        return synthetic_trajectory(settings, reason="cancelled")

    monkeypatch.setattr("temsim.magnetic_test_particle.trace_test_electron", trace)
    controller.set_scene(UniformScene())
    controller.set_active(True)
    try:
        qtbot.waitUntil(entered.is_set, timeout=10000)
        worker = controller._worker
        waited = []

        def wait_for_done(*_args):
            waited.append(True)
            assert worker.cancelled.is_set() and not controller._active
            return True

        monkeypatch.setattr(window.calculations.pool, "waitForDone", wait_for_done)
        window.close()
        assert waited and not window.isVisible()
        qtbot.waitUntil(cancellation_seen.is_set, timeout=10000)
        qtbot.waitUntil(lambda: controller._worker is None, timeout=10000)
        assert controller.current_trajectory is None
    finally:
        release.set()
