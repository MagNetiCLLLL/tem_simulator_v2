"""Presentation layout persistence; no ray, image, spectrum or preset solves."""
from copy import deepcopy

import numpy as np
import pytest
from PySide6.QtCore import QSettings

from temsim.gui import interactive_calculation as gui
from temsim.gui import main_window as shell
from temsim.gui.workspace_layouts import WorkspaceLayouts


@pytest.fixture
def windows(qtbot, monkeypatch, tmp_path):
    settings = QSettings(str(tmp_path / "layouts.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(shell, "QSettings", lambda: settings)
    monkeypatch.setattr(gui, "QSettings", lambda: settings)
    monkeypatch.setattr(shell.MainWindow, "INITIAL_PREVIEW_DELAY_MS", 60000)

    def make():
        window = shell.MainWindow()
        window.preview_timer.stop()
        qtbot.addWidget(window)
        monkeypatch.setattr(window.calculations.pool, "start", lambda *_: pytest.fail("Layout must not calculate"))
        window.resize(2200, 1100)
        window.show()
        qtbot.wait(40)
        # Keep the test viewport fixed: the offscreen plugin's virtual monitor
        # is smaller than this application. Native floating geometry is covered
        # in test_live_tuning_dock; here we compare splitter proportions.
        monkeypatch.setattr(window, "restoreGeometry", lambda value: True)
        return window

    return make, settings


def _resize_splitter(window, name, sizes, qtbot):
    splitter = window.workspace_layouts.splitters[name]
    assert splitter.isVisible()
    splitter.setSizes(sizes)
    splitter.splitterMoved.emit(sizes[0], 1)
    qtbot.wait(20)
    return np.asarray(splitter.sizes()) / sum(splitter.sizes())


def _assert_ratio(splitter, expected):
    actual = np.asarray(splitter.sizes()) / sum(splitter.sizes())
    np.testing.assert_allclose(actual, expected, atol=0.015, rtol=0)


def test_named_layouts_restore_each_page_and_do_not_change_optics(windows, qtbot):
    make, settings = windows
    window = make()
    manager, workspace = window.workspace_layouts, window.workspace
    before_state = deepcopy(window.state.to_dict())
    old_high, old_bank = object(), object()
    workspace._high_accuracy_result = old_high
    workspace.interactive_calculation.controller.bank = old_bank
    workspace.tabs.setCurrentWidget(workspace.sample_page)
    qtbot.wait(30)
    default_ratio = _resize_splitter(window, "samplePageSplitter", [440, 1250], qtbot)
    saved_id = manager.save_as("Sample analysis")
    custom_ratio = _resize_splitter(window, "samplePageSplitter", [750, 940], qtbot)
    assert not np.allclose(default_ratio, custom_ratio)
    workspace.magnetic_field_toggle.setChecked(True)
    manager.save_current()
    manager.select("default")
    qtbot.wait(40)
    assert workspace.tabs.currentWidget() is workspace.sample_page
    _assert_ratio(manager.splitters["samplePageSplitter"], default_ratio)
    assert not workspace.magnetic_field_toggle.isChecked()
    manager.select(saved_id)
    qtbot.wait(40)
    _assert_ratio(manager.splitters["samplePageSplitter"], custom_ratio)
    assert workspace.magnetic_field_toggle.isChecked()
    assert workspace._high_accuracy_result is old_high
    assert workspace.interactive_calculation.controller.bank is old_bank
    assert window.state.to_dict() == before_state
    assert not window.preview_timer.isActive()
    assert settings.value(f"{manager.ROOT}/active") == saved_id
    workspace.interactive_calculation.controller.bank = None


def test_restart_restores_active_layout_and_hidden_splitters(windows, qtbot):
    make, settings = windows
    window = make()
    workspace, manager = window.workspace, window.workspace_layouts
    workspace.tabs.setCurrentWidget(workspace.sample_page)
    qtbot.wait(30)
    _resize_splitter(window, "samplePageSplitter", [620, 1050], qtbot)
    sample_width = manager.splitters["samplePageSplitter"].sizes()[0]
    workspace.tabs.setCurrentWidget(workspace.scanning_page)
    qtbot.wait(30)
    _resize_splitter(window, "scanningImageSplitter", [540, 1190], qtbot)
    scan_width = manager.splitters["scanningImageSplitter"].sizes()[0]
    saved_id = manager.save_as("Scanning")
    workspace.show_ray_diagram()
    window.live_tuning_dock.toggleViewAction().trigger()
    qtbot.wait(30)
    workspace.interactive_calculation.advanced_bank.setChecked(True)
    qtbot.wait(30)
    control_ratio = _resize_splitter(window, "interactiveCalculationSplitter", [610, 370], qtbot)
    manager.save_current()
    window.close()
    saved = settings.value(manager._key(saved_id, "data"))
    assert saved["splitters"]["samplePageSplitter"]
    restored = make()
    qtbot.wait(60)
    manager, workspace = restored.workspace_layouts, restored.workspace
    assert manager.active_id == saved_id
    assert not restored.live_tuning_dock.isHidden()
    assert workspace.interactive_calculation.advanced_bank.isChecked()
    _assert_ratio(manager.splitters["interactiveCalculationSplitter"], control_ratio)
    for page, name, expected in (
        (workspace.sample_page, "samplePageSplitter", sample_width),
        (workspace.scanning_page, "scanningImageSplitter", scan_width),
    ):
        workspace.tabs.setCurrentWidget(page)
        qtbot.wait(50)
        # These editors have zero stretch: their absolute width stays fixed
        # when a wider Live tuning dock leaves less space for the plot.
        assert manager.splitters[name].sizes()[0] == pytest.approx(expected, abs=2)


def test_ray_panel_combinations_keep_independent_sizes_across_restart(windows, qtbot):
    make, _ = windows
    window = make()
    manager, workspace = window.workspace_layouts, window.workspace
    vertical = workspace.ray_vertical_splitter
    plain = _resize_splitter(window, vertical.objectName(), [660, 120, 0], qtbot)
    workspace.magnetic_field_toggle.setChecked(True)
    qtbot.wait(30)
    magnetic = _resize_splitter(window, vertical.objectName(), [430, 110, 260], qtbot)
    workspace.transverse_beam_toggle.setChecked(False)
    qtbot.wait(30)
    wide_magnetic = _resize_splitter(window, vertical.objectName(), [300, 120, 420], qtbot)
    for magnetic_on, transverse_on, expected in ((True, True, magnetic), (False, True, plain), (True, False, wide_magnetic)):
        workspace.magnetic_field_toggle.setChecked(magnetic_on)
        workspace.transverse_beam_toggle.setChecked(transverse_on)
        qtbot.wait(50)
        _assert_ratio(vertical, expected)
    manager.save_current()
    window.close()
    restored = make()
    workspace = restored.workspace
    assert workspace.magnetic_field_toggle.isChecked()
    assert not workspace.transverse_beam_toggle.isChecked()
    qtbot.wait(40)
    _assert_ratio(workspace.ray_vertical_splitter, wide_magnetic)
    workspace.transverse_beam_toggle.setChecked(True)
    qtbot.wait(40)
    _assert_ratio(workspace.ray_vertical_splitter, magnetic)


def test_layout_changes_autosave_without_normal_exit(windows, qtbot):
    make, settings = windows
    window = make()
    manager = window.workspace_layouts
    manager.save_timer.setInterval(20)
    expected = _resize_splitter(window, "instrumentEditorSplitter", [300, 650], qtbot)
    qtbot.waitUntil(lambda: isinstance(settings.value(manager._key("default", "data")), dict))
    qtbot.wait(40)
    data = settings.value(manager._key("default", "data"))
    assert data["splitters"]["instrumentEditorSplitter"]["default"] == window.instrument_editor.saveState()
    manager.save_as("Second layout")
    _resize_splitter(window, "instrumentEditorSplitter", [650, 300], qtbot)
    manager.select("default")
    qtbot.wait(40)
    _assert_ratio(window.instrument_editor, expected)


def test_layout_names_and_corrupt_entries_do_not_overwrite_current_layout(windows):
    make, settings = windows
    window = make()
    manager = window.workspace_layouts
    manager.save_as("Teaching")
    original = manager.active_id
    for name in ("", "  ", "teaching", "x" * 65):
        with pytest.raises(ValueError):
            manager.save_as(name)
        assert manager.active_id == original
    settings.setValue(manager._key("broken", "name"), "Broken")
    settings.setValue(manager._key("broken", "data"), {"schema": 999})
    with pytest.raises(ValueError):
        manager.select("broken")
    assert manager.active_id == original
    assert manager.entries()[original] == "Teaching"


def test_menu_selection_restores_presentation_subtabs_without_requesting_work(windows, qtbot):
    make, _ = windows
    window = make()
    manager = window.workspace_layouts
    second = manager.save_as("Alignment")
    window.assembly_panel.component_pages.setCurrentIndex(2)
    window.parameter_panel.tabs.setCurrentIndex(1)
    manager.save_current()
    manager.refresh_menu()
    action = next(action for action in window.layouts_menu.actions() if action.text() == "Default")
    action.trigger()
    qtbot.wait(30)
    assert manager.active_id == "default"
    assert window.assembly_panel.component_pages.currentIndex() == 0
    assert window.parameter_panel.tabs.currentIndex() == 0
    action = next(action for action in window.layouts_menu.actions() if action.text() == "Alignment")
    action.trigger()
    qtbot.wait(30)
    assert manager.active_id == second
    assert window.assembly_panel.component_pages.currentIndex() == 2
    assert window.parameter_panel.tabs.currentIndex() == 1
    assert not window.preview_timer.isActive()


def test_component_selection_does_not_reset_instrument_splitter(windows, qtbot):
    make, _ = windows
    window = make()
    expected = _resize_splitter(window, "instrumentEditorSplitter", [600, 180], qtbot)
    window._select_component_from_workspace("objective_lens")
    qtbot.wait(30)
    _assert_ratio(window.instrument_editor, expected)


def test_reset_only_affects_selected_layout_and_registry_covers_workspace(windows, qtbot):
    make, _ = windows
    window = make()
    manager, workspace = window.workspace_layouts, window.workspace
    assert {"samplePageSplitter", "sampleInteractionContentSplitter", "scanningImageSplitter",
            "scanPlotSplitter", "energyFilterSplitter", "designExplorerSplitter",
            "designExplorerVerticalSplitter", "designSweepTablesSplitter", "magneticValidationSplitter",
            "instrumentEditorSplitter", "interactiveCalculationSplitter"} <= manager.splitters.keys()
    manager.save_current()
    custom = manager.save_as("Field inspection")
    workspace.magnetic_field_toggle.setChecked(True)
    manager.save_current()
    manager.select("default")
    qtbot.wait(30)
    window.reset_workspace()
    qtbot.wait(30)
    assert not workspace.magnetic_field_toggle.isChecked()
    manager.select(custom)
    qtbot.wait(30)
    assert workspace.magnetic_field_toggle.isChecked()
