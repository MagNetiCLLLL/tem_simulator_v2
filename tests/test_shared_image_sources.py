"""Main-window source routing, without propagating rays or calculating images."""
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QTabWidget


@pytest.fixture
def window(qtbot, monkeypatch, tmp_path):
    from temsim.gui import main_window as shell, interactive_calculation as gui
    from temsim.gui import calculation_controller as controller
    settings = QSettings(str(tmp_path / "workspace.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(shell, "QSettings", lambda: settings)
    monkeypatch.setattr(gui, "QSettings", lambda: settings)
    monkeypatch.setattr(controller, "default_artifact_cache_root", lambda: tmp_path / "artifacts")
    monkeypatch.setattr(shell.MainWindow, "INITIAL_PREVIEW_DELAY_MS", 60000)
    widget = shell.MainWindow()
    widget.preview_timer.stop()
    qtbot.addWidget(widget)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Result source selection must not request computation")
    monkeypatch.setattr(widget.calculations, "submit", forbidden)
    monkeypatch.setattr(widget.calculations, "submit_background", forbidden)
    monkeypatch.setattr(widget.workspace.interactive_calculation.controller, "read", forbidden)
    monkeypatch.setattr(widget.workspace.interactive_calculation.controller, "build", forbidden)
    monkeypatch.setattr(widget, "_show_error", forbidden)
    yield widget
    widget.preview_timer.stop()


def empty_readout():
    return SimpleNamespace(
        wave=None, stem=None, coordinates={"lens:objective:percent": 67.},
        detector_fractions={"camera": .25}, detector_current_pa={"camera": 5.},
        notes=("Bank point: no wave products.",), state_snapshot=None,
    )


def test_one_detector_table_and_two_shared_viewers(window):
    workspace = window.workspace
    page = workspace.interactive_calculation
    assert workspace.ray_result_tabs.tabText(1) == "Cached signals"
    assert not page.readout_panel.findChildren(QTabWidget)
    for name in ("tem_image", "stem_image", "tem_plot", "stem_plot", "stem_choice"):
        assert not hasattr(page, name)
    for viewer in (workspace.wave_imaging, workspace.scan_control):
        assert [viewer.image_source.itemText(i) for i in range(viewer.image_source.count())] == [
            "Current calculation", "Advanced bank",
        ]
        assert viewer.image_source.currentData() == "current"
    assert "cachedSignalTabs" not in window.workspace_layouts.tabs


def test_bank_signals_reach_viewers_without_mutation_or_calculation(window):
    workspace = window.workspace
    page = workspace.interactive_calculation
    before = window.state.to_dict()
    high = object()
    workspace._high_accuracy_result = high
    result = empty_readout()
    page._readout_ready(result)
    assert workspace.wave_imaging._bank_readout is result
    assert workspace.scan_control._bank_readout is result
    for viewer in (workspace.wave_imaging, workspace.scan_control):
        viewer.image_source.setCurrentIndex(viewer.image_source.findData("bank"))
    page.show_error("Controlled failed readout")
    assert workspace.wave_imaging._bank_readout is result
    assert workspace.scan_control._bank_readout is result
    assert "previous" in workspace.wave_imaging.image_source_status.text().lower()
    # Each viewer chooses its own presentation source, not a shared state mode.
    workspace.wave_imaging.image_source.setCurrentIndex(0)
    assert workspace.scan_control.image_source.currentData() == "bank"
    workspace.scan_control.image_source.setCurrentIndex(0)
    assert window.state.to_dict() == before
    assert workspace._high_accuracy_result is high
    assert not window.preview_timer.isActive()
    assert window.calculations.pool.activeThreadCount() == 0
    assert page.controller.pool.activeThreadCount() == 0


def test_old_cached_subtab_layout_is_ignored_without_losing_other_tabs(window):
    layouts = window.workspace_layouts
    data = layouts._snapshot()
    data["tabs"]["cachedSignalTabs"] = "STEM image"
    data["tabs"]["rayResultTabs"] = "Cached signals"
    layouts._apply(data)
    assert window.workspace.ray_result_tabs.currentIndex() == 1
    assert window.workspace.wave_imaging.image_source.currentData() == "current"
    assert window.workspace.scan_control.image_source.currentData() == "current"
    assert not window.preview_timer.isActive()
