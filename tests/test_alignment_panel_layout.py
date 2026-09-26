"""Navigation size regressions; GUI-only, no electron or alignment calculation."""
import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QSplitter

from temsim.assembly_catalog import AssemblyCatalog
from temsim.gui.assembly_panel import AssemblyPanel
from temsim.gui.parameter_panel import ParameterPanel
from temsim.optics.column import default_state


def geometry(panel, splitter):
    return (panel.setup_scroll_area.geometry().getRect(),
            panel.component_pages.geometry().getRect(),
            tuple(splitter.sizes()), splitter.size().toTuple())


@pytest.mark.parametrize('size', [(745, 887), (420, 650)])
def test_navigation_switch_and_long_alignment_text_preserve_setup_space(qtbot, size):
    catalog = AssemblyCatalog()
    panel = AssemblyPanel(catalog, catalog.default_selection())
    panel.set_direct_alignment_state(default_state())
    splitter = QSplitter(Qt.Orientation.Vertical)
    splitter.setChildrenCollapsible(False)
    splitter.addWidget(panel)
    splitter.addWidget(ParameterPanel())
    qtbot.addWidget(splitter)
    splitter.resize(*size)
    splitter.setSizes([size[1]*2//3, size[1]//3])
    splitter.show()
    qtbot.wait(20)
    before = geometry(panel, splitter)
    alignment = panel.direct_alignment_panel
    target = alignment.controls['nanoprobe_convergence'].target
    target.setValue(26.)
    for index in (2, 0, 1, 3, 2):
        panel.component_pages.setCurrentIndex(index)
        qtbot.wait(15)
        assert geometry(panel, splitter) == before
    alignment.show_status_message('Detailed alignment result and limitations. '*100)
    alignment.mode_status.setText('Applied mode and current calibration details. '*40)
    alignment.validation_details.setPlainText('Constraint report\n'*100)
    alignment.set_busy('nanoprobe_convergence')
    qtbot.wait(20)
    assert geometry(panel, splitter) == before
    assert alignment.cancel_button.isVisible() and alignment.cancel_button.isEnabled()
    assert alignment.rect().contains(alignment.cancel_button.geometry())
    assert not alignment.scroll_area.isAncestorOf(alignment.cancel_button)
    assert alignment.scroll_area.isAncestorOf(alignment.result_status)
    assert alignment.scroll_area.verticalScrollBar().maximum() > 0
    bar = alignment.scroll_area.verticalScrollBar()
    bar.setValue(bar.maximum())
    qtbot.wait(15)
    point = alignment.validation_details.mapTo(alignment.scroll_area.viewport(),
                                             alignment.validation_details.rect().center())
    assert alignment.scroll_area.viewport().rect().contains(point)
    assert geometry(panel, splitter) == before
    with qtbot.waitSignal(alignment.cancellation_requested):
        alignment.cancel_button.click()
    alignment.set_busy(None)
    assert target.value() == 26.
    assert target.isEnabled()


def test_floating_instrument_dock_keeps_user_size_and_splitter_on_alignment_tab(qtbot, monkeypatch, tmp_path):
    from temsim.gui import main_window as module
    monkeypatch.setattr(module, 'QSettings', lambda: QSettings(
        str(tmp_path/'window.ini'), QSettings.Format.IniFormat))
    window = module.MainWindow()
    qtbot.addWidget(window)
    window.preview_timer.stop()
    monkeypatch.setattr(window.calculations, 'submit_background', lambda *_a, **_k: None)
    window.resize(1400, 900)
    window.show()
    dock = window.instrument_dock
    dock.setFloating(True)
    dock.resize(420, 650)
    dock.show()
    window.instrument_editor.setSizes([420, 220])
    panel = window.assembly_panel
    panel.component_pages.setCurrentIndex(panel.assembly_tree_index)
    qtbot.wait(30)
    original = geometry(panel, window.instrument_editor)
    dock_rect, window_rect = dock.geometry(), window.geometry()
    for index in (2, panel.assembly_tree_index, 2):
        panel.component_pages.setCurrentIndex(index)
        qtbot.wait(20)
        assert geometry(panel, window.instrument_editor) == original
        assert dock.geometry() == dock_rect
        assert window.geometry() == window_rect
