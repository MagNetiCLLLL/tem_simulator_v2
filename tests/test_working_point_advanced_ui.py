"""The embedded checkpoint browser keeps routine actions visible without executing physics."""
from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton, QScrollArea

from temsim.gui.working_point_panel import WorkingPointPanel
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state
from temsim.working_point import WorkingPointArchiveIndex, WorkingPointCheckpoint


@pytest.fixture
def point():
    state = default_state()
    return WorkingPointCheckpoint(capture_instrument_snapshot(state),
        {"fixture": np.arange(8.)}, state.sample.z_mm, "advanced-ui-fixture", {})


@pytest.fixture
def panel(qtbot, monkeypatch):
    from temsim.gui import sampling_panel
    from temsim.optics.electron_gun import source
    from temsim.physics import core, simulation

    def forbidden(*args, **kwargs):
        pytest.fail("Browsing or expanding advanced tools must not execute physics")

    monkeypatch.setattr(source, "trace_source_to_exit", forbidden)
    monkeypatch.setattr(core, "propagate", forbidden)
    monkeypatch.setattr(simulation, "run", forbidden)
    monkeypatch.setattr(sampling_panel, "run_convergence", forbidden)
    widget = WorkingPointPanel()
    qtbot.addWidget(widget)
    widget.resize(800, 650)
    widget.show()
    yield widget
    assert widget.shutdown()


def test_routine_actions_visible_and_advanced_closed_by_default(panel):
    assert not panel.advanced_toggle.isChecked()
    assert panel.advanced.isHidden()
    assert panel.points.isVisible() and panel.filter.isVisible()
    assert {button.objectName() for button in panel.findChildren(QPushButton)
            if button.isVisible()} == {"workingPointLoadData", "workingPointRestore"}
    assert ".temresult" in panel.restore_button.toolTip()
    assert "does not run a calculation" in panel.load_button.toolTip()


def test_expand_and_collapse_preserve_selection_without_jobs(panel, point, qtbot, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("The Advanced toggle must not queue a job")

    panel.add_checkpoint(point)
    monkeypatch.setattr(panel._archive_pool, "start", forbidden)
    monkeypatch.setattr(panel.sampling.pool, "start", forbidden)
    qtbot.mouseClick(panel.advanced_toggle, Qt.MouseButton.LeftButton)
    assert panel.advanced.isVisible()
    assert panel.advanced_toggle.arrowType() == Qt.ArrowType.DownArrow
    assert panel.sampling._checkpoint is point
    panel.detail_tabs.setCurrentIndex(1)
    assert panel.sampling.isVisible()
    qtbot.mouseClick(panel.advanced_toggle, Qt.MouseButton.LeftButton)
    assert panel.advanced.isHidden()
    assert panel.advanced_toggle.arrowType() == Qt.ArrowType.RightArrow
    assert panel.selected is point
    assert panel.tree.topLevelItemCount() == len(point.snapshot.graph["nodes"])


def test_hidden_details_do_not_block_filtering_or_restore(panel, point, qtbot):
    other = replace(point, stage_signature="second-ui-fixture")
    panel.add_checkpoint(point, label="First")
    panel.add_checkpoint(other, label="Second")
    panel.filter.setText("Second")
    assert sum(not panel.points.isRowHidden(row) for row in range(2)) == 1
    requested = []
    panel.restore_requested.connect(lambda checkpoint, fork: requested.append((checkpoint, fork)))
    qtbot.mouseClick(panel.restore_button, Qt.MouseButton.LeftButton)
    assert requested == [(other, False)]
    assert panel.advanced.isHidden()


def test_advanced_actions_keep_signals_and_comparison_available(panel, point, qtbot):
    panel.add_checkpoint(point)
    panel.current_snapshot = lambda: point.snapshot
    restored, illumination, undo = [], [], []
    panel.restore_requested.connect(lambda checkpoint, fork: restored.append((checkpoint, fork)))
    panel.illumination_requested.connect(illumination.append)
    panel.undo_requested.connect(lambda: undo.append(True))
    qtbot.mouseClick(panel.advanced_toggle, Qt.MouseButton.LeftButton)
    for name in ("workingPointPinA", "workingPointPinB", "workingPointComparePins",
                 "workingPointCompareCurrent", "workingPointFork", "workingPointApplyIllumination",
                 "workingPointUndo"):
        button = panel.findChild(QPushButton, name)
        assert button is not None and button.isVisible()
        qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    assert panel._pins == {"A": point, "B": point}
    assert restored == [(point, True)]
    assert illumination == [point]
    assert undo == [True]
    assert panel.values.rowCount() > 0


def test_retained_data_load_then_restore_work_with_advanced_closed(panel, point, tmp_path, qtbot):
    path = tmp_path / "checkpoint.temwp"
    point.write_package(path)
    panel.add_checkpoint(WorkingPointArchiveIndex.read(path))
    requested = []
    panel.restore_requested.connect(lambda checkpoint, fork: requested.append((checkpoint, fork)))
    qtbot.mouseClick(panel.restore_button, Qt.MouseButton.LeftButton)
    assert not requested
    assert "Load retained data" in panel.status.text()
    qtbot.mouseClick(panel.load_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: not panel._archive_loading, timeout=5000)
    assert isinstance(panel.selected, WorkingPointCheckpoint)
    np.testing.assert_array_equal(panel.selected.arrays["fixture"], point.arrays["fixture"])
    qtbot.mouseClick(panel.restore_button, Qt.MouseButton.LeftButton)
    assert requested == [(panel.selected, False)]
    assert panel.advanced.isHidden()


def test_expanded_tools_fit_dock_width_and_sampling_can_scroll(panel, qtbot):
    qtbot.mouseClick(panel.advanced_toggle, Qt.MouseButton.LeftButton)
    panel.layout().activate()
    assert panel.width() == 800
    assert panel.minimumSizeHint().width() < 800
    assert panel.advanced.horizontalScrollBar().maximum() == 0
    for button in panel.advanced.findChildren(QPushButton):
        if button.objectName().startswith("workingPoint"):
            top_left = button.mapTo(panel.advanced.viewport(), button.rect().topLeft())
            assert top_left.x() >= 0
            assert top_left.x() + button.width() <= panel.advanced.viewport().width()
    panel.detail_tabs.setCurrentIndex(1)
    scroll = panel.findChild(QScrollArea, "workingPointSamplingScroll")
    assert scroll.widget() is panel.sampling
    assert panel.width() == 800
    assert scroll.horizontalScrollBar().maximum() > 0
