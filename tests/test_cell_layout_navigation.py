"""Single applied geometry editor; viewing never launches a calculation."""
from copy import deepcopy

import pytest
from PySide6.QtCore import QSettings

from temsim.gui import main_window as shell
from temsim.gui.cell_environment_editor import CellGeometryDialog


@pytest.fixture
def window(qtbot, monkeypatch, tmp_path):
    settings = QSettings(str(tmp_path/'layout.ini'), QSettings.Format.IniFormat)
    monkeypatch.setattr(shell, 'QSettings', lambda: settings)
    monkeypatch.setattr(shell.MainWindow, 'INITIAL_PREVIEW_DELAY_MS', 60000)
    widget = shell.MainWindow()
    widget.preview_timer.stop()
    qtbot.addWidget(widget)
    monkeypatch.setattr(widget.calculations.pool, 'start', lambda *_: pytest.fail('Unexpected solve'))
    return widget


def test_applied_geometry_reaches_layout_map_and_invalidation(window, monkeypatch):
    sample = deepcopy(window.state.sample)
    edits = []
    original = window._runtime_parameter_changed

    def changed(key):
        edits.append(key)
        original(key)
        window.preview_timer.stop()

    monkeypatch.setattr(window, '_runtime_parameter_changed', changed)

    def enter(dialog):
        dialog.inserted.setChecked(True)
        dialog.fields['length_mm'].setValue(500)
        dialog.window_editors['upstream_window'].thickness.setValue(7)
        dialog.window_editors['downstream_window'].thickness.setValue(9)
        dialog._accept_medium()
        return dialog.result()

    monkeypatch.setattr(CellGeometryDialog, 'exec', enter)
    window.workspace.physical_layout.edit_cell.click()
    assert edits == ['vacuum_map']
    assert window.state.sample == sample
    assert not window.state.vacuum_map.enabled  # Insertion never opts into physics.
    layout, vacuum = window.workspace.physical_layout, window.workspace.vacuum_map
    assert len(layout.cell_overlay.context.layers) == 3
    assert window.workspace.tabs.currentWidget() is vacuum
    assert vacuum.current_key == 'specimen_cell'
    assert window.state.vacuum_map.cell.upstream_window.thickness_nm == 7
    z = window.state.sample.z_mm
    for key in ('specimen_cell', 'cell_window_upstream', 'cell_sample_reference'):
        window._navigate_physical_component(key, 'ray', z)
        assert window.workspace.tabs.currentWidget() is window.workspace.ray_page
        window._navigate_physical_component(key, 'vacuum', z)
        assert vacuum.current_key == 'specimen_cell'


def test_cancel_from_vacuum_map_keeps_all_applied_inputs(window, monkeypatch):
    before = window.state.vacuum_map.to_dict()
    monkeypatch.setattr(CellGeometryDialog, 'exec', lambda dialog: dialog.DialogCode.Rejected)
    window.workspace.vacuum_map.edit_geometry.click()
    assert window.workspace.tabs.currentWidget() is window.workspace.physical_layout
    assert window.state.vacuum_map.to_dict() == before
    assert not window.preview_timer.isActive()
