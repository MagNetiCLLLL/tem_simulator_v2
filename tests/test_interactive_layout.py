"""Aligned dock controls and shared Ray Diagram; no signal calculations."""
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QPoint, QSettings, Qt
from PySide6.QtWidgets import QLabel, QLineEdit, QSlider, QSplitter, QWidget

from temsim.gui import interactive_calculation as gui
from temsim.component_keys import CONDENSER_LENS_1, OBJECTIVE_LENS
from temsim.optics.column import default_state


@pytest.fixture
def page_factory(qtbot, monkeypatch, tmp_path):
    settings = QSettings(str(tmp_path / "layout.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(gui, "QSettings", lambda: settings)
    pages = []

    def make():
        page = gui.InteractiveCalculationPage()
        pages.append(page)
        qtbot.addWidget(page)
        page.resize(1600, 820)
        page.show()
        qtbot.waitUntil(lambda: page.isVisible())
        return page

    yield make, settings
    for page in pages:
        page.shutdown()


def _start_lenses(page):
    page.set_source(default_state())
    indices = [i for i in range(page.choice.count())
               if page.choice.itemData(i).group == "lens"
               and page.choice.itemData(i).field == "percent"][:2]
    indices.append(next(i for i in range(page.choice.count())
                        if page.choice.itemData(i).key == OBJECTIVE_LENS
                        and page.choice.itemData(i).field == "percent"))
    for row, index in enumerate(indices):
        page.choice.setCurrentIndex(index)
        page._add_range()
        page.ranges.cellWidget(row, 1).setText("67")
        page.ranges.cellWidget(row, 2).setText("70")
    page.start_live_tuning()
    page.timer.stop()
    assert page._live_mode, page.status.text()


def test_lens_sliders_remain_beside_selection_without_duplicate_ray_plot(page_factory, qtbot):
    make, _ = page_factory
    page = make()
    _start_lenses(page)
    qtbot.wait(10)
    assert page.splitter.count() == 2
    left, middle = [page.splitter.widget(i) for i in range(2)]
    assert left.isAncestorOf(page.ranges)
    assert middle.isAncestorOf(page.live_table)
    assert not page.splitter.isAncestorOf(page.readout_panel)
    assert not hasattr(page, "tuning_plot")
    assert left.x() < middle.x()
    assert page.live_table.height() > 230
    for i, axis in enumerate(page.live_plan.ranges):
        control_row = page.live_table.control_row(axis.control.identity)
        label = page.live_table.item(control_row, 0)
        row = page.live_table.cellWidget(control_row, 1)
        slider = row.findChild(QSlider)
        assert label.text() == axis.control.label.removesuffix(" / Excitation")
        assert slider.isVisible() and slider.width() >= 90
        assert row.isAncestorOf(page.live_widgets[axis.control.identity])
        assert not label.flags() & Qt.ItemFlag.ItemIsEditable
    _assert_aligned(page)

    values = list(page.live_widgets.values())
    before = [w.value() for w in values]
    first_row = page.live_table.cellWidget(0, 1)
    first_row.findChild(QSlider).setValue(2500)
    page.timer.stop()
    assert values[0].value() == pytest.approx(67.75)
    assert [w.value() for w in values[1:]] == before[1:]
    with qtbot.waitSignal(page.tuning_changed) as output:
        page._read()
    assert output.args[0][0][1] == pytest.approx(67.75)
    assert page.controller.bank is None


def test_control_widths_restore_without_old_full_page_state(page_factory, qtbot):
    make, settings = page_factory
    old = QSplitter(Qt.Orientation.Horizontal)
    old.addWidget(QWidget())
    old.addWidget(QWidget())
    old.addWidget(QWidget())
    old.setSizes([520, 400, 680])
    old_state = old.saveState()
    settings.setValue("interactive_calculation/three_column_splitter", old_state)
    page = make()
    assert len(page.splitter.sizes()) == 2
    assert all(size > 0 for size in page.splitter.sizes())
    page.splitter.moveSplitter(470, 1)
    qtbot.wait(10)
    sizes = page.splitter.sizes()
    assert settings.value(gui._SPLITTER_SETTINGS_KEY) == page.splitter.saveState()
    restored = make()
    qtbot.wait(10)
    assert restored.splitter.sizes() == pytest.approx(sizes, abs=1)
    assert settings.value("interactive_calculation/three_column_splitter") == old_state


def test_many_control_rows_scroll_without_taking_readout_height(page_factory, qtbot):
    make, _ = page_factory
    page = make()
    readout_height = page.readout_panel.height()
    for index in range(50):
        control = SimpleNamespace(label=f"Lens {index} / Excitation", unit="%", identity=str(index))
        page._add_control_row(control, QLabel("68.5 %"))
    qtbot.waitUntil(lambda: page.live_table.verticalScrollBar().maximum() > 0)
    assert page.readout_panel.height() == readout_height
    assert page.live_table.rowCount() == 50


def test_cached_controls_use_same_middle_pane_and_recapture_clears_live_rows(page_factory, qtbot):
    make, _ = page_factory
    page = make()
    _start_lenses(page)
    bank = SimpleNamespace(plan=page.live_plan, points=(), retained_bytes=0)
    page._bank_ready(bank)
    page.timer.stop()
    assert page.live_heading.text() == "Cached controls"
    assert page.live_table.rowCount() == 3
    qtbot.waitUntil(lambda: not page.live_table.findChildren(QSlider))
    assert all(page.splitter.widget(1).isAncestorOf(w) for w in page.live_widgets.values())
    page.start_live_tuning()
    page.timer.stop()
    assert page.live_table.rowCount() == 3
    page.set_source(default_state())
    assert page.live_table.rowCount() == 0
    assert not page.live_widgets
    assert page.splitter.count() == 2


def _assert_aligned(page):
    left, right = page.ranges, page.live_table
    for row in range(left.rowCount()):
        control = left.item(row, 0).data(Qt.ItemDataRole.UserRole)
        paired = right.control_row(control.identity)
        left_y = left.viewport().mapToGlobal(QPoint(0, left.rowViewportPosition(row))).y()
        right_y = right.viewport().mapToGlobal(QPoint(0, right.rowViewportPosition(paired))).y()
        assert abs(left_y - right_y) <= 1
        assert left.rowHeight(row) == right.rowHeight(paired)


def _mixed_ranges(page, detector_bounds=("2790", "2810")):
    page.set_source(default_state())
    for row, (key, field, bounds) in enumerate((
        (CONDENSER_LENS_1, "percent", ("50", "53")),
        ("df", "z_mm", detector_bounds),
        (OBJECTIVE_LENS, "percent", ("65", "72")),
    )):
        index = next(i for i in range(page.choice.count())
                     if page.choice.itemData(i).key == key
                     and page.choice.itemData(i).field == field)
        page.choice.setCurrentIndex(index)
        page._add_range()
        for column, value in zip((1, 2), bounds):
            page.ranges.cellWidget(row, column).setText(value)


@pytest.mark.parametrize("detector_bounds", [("2790", "2810"), ("", ""), ("invalid", "2810")])
def test_detector_draft_does_not_hide_or_block_lens_sliders(
        page_factory, qtbot, monkeypatch, detector_bounds):
    make, _ = page_factory
    page = make()
    _mixed_ranges(page, detector_bounds)
    monkeypatch.setattr(page.controller, "build", lambda *_: pytest.fail("No automatic bank"))
    monkeypatch.setattr(page.controller, "read", lambda *_: pytest.fail("No live detector readout"))
    detector = page.ranges.item(1, 0).data(Qt.ItemDataRole.UserRole)
    page.start_live_tuning()
    page.timer.stop()
    assert page._live_mode, page.status.text()
    assert [axis.control.key for axis in page.live_plan.ranges] == [CONDENSER_LENS_1, OBJECTIVE_LENS]
    assert detector.identity not in page.live_widgets
    assert "Advanced bank" in page.status.text()
    qtbot.wait(20)
    _assert_aligned(page)
    for axis in page.live_plan.ranges:
        row = page.live_table.control_row(axis.control.identity)
        slider = page.live_table.cellWidget(row, 1).findChild(QSlider)
        assert slider.isVisible() and slider.width() >= 90
        slider.setValue(2500)
    page.timer.stop()
    row = page.live_table.control_row(detector.identity)
    cell = page.live_table.cellWidget(row, 1)
    assert cell.findChild(QSlider) is None
    assert cell.findChild(QLineEdit).isReadOnly()
    assert cell.findChild(QLineEdit).text() == f"{detector.current:.9g} {detector.unit}"
    hint = cell.findChild(QLabel, "detectorReferenceHint")
    assert hint.isVisible() and hint.text() == "Advanced bank"
    assert hint.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse
    with qtbot.waitSignal(page.tuning_changed) as output:
        page._read()
    assert [axis.control.group for axis, _ in output.args[0]] == ["lens", "lens"]
    assert page.ranges.cellWidget(1, 1).text() == detector_bounds[0]
    assert page.ranges.cellWidget(1, 2).text() == detector_bounds[1]
    assert next(d for d in page.source_state.recording_planes if d.key == "df").z_mm == detector.current
    assert page.controller.bank is None


def test_mixed_ranges_remain_available_to_advanced_bank(page_factory, qtbot):
    make, _ = page_factory
    page = make()
    _mixed_ranges(page)
    full_plan = page.plan()
    page.start_live_tuning()
    page.timer.stop()
    assert page._live_mode, page.status.text()
    assert page.plan() == full_plan
    assert len(full_plan.ranges) == 3
    assert full_plan.point_count == 25
    bank = SimpleNamespace(plan=full_plan, points=(), retained_bytes=0)
    page._bank_ready(bank)
    page.timer.stop()
    detector = full_plan.ranges[1].control
    assert detector.identity in page.live_widgets
    assert page.live_widgets[detector.identity].value() == pytest.approx(detector.current)
    qtbot.wait(20)
    _assert_aligned(page)
    assert not page.live_table.findChildren(QLabel, "detectorReferenceHint")


def test_detector_only_live_request_is_explained_without_starting_work(page_factory):
    make, _ = page_factory
    page = make()
    _mixed_ranges(page, ("", ""))
    for row in (2, 0):
        page.ranges.setCurrentCell(row, 0)
        page._remove_range()
    page.start_live_tuning()
    assert not page._live_mode and not page.timer.isActive() and not page.busy
    assert not page.live_widgets
    assert "Advanced bank" in page.status.text()
    assert "lens or aperture" in page.status.text()


@pytest.mark.parametrize("minimum", ["", "73"])
def test_mixed_ranges_still_validate_live_lens_endpoints(page_factory, minimum):
    make, _ = page_factory
    page = make()
    _mixed_ranges(page)
    page.ranges.cellWidget(2, 1).setText(minimum)
    page.start_live_tuning()
    assert not page._live_mode and not page.timer.isActive()
    assert not page.live_widgets
    if minimum:
        assert "minimum smaller than the maximum" in page.status.text()
    else:
        assert "Range 3: minimum and maximum are required" in page.status.text()


def test_reference_is_current_when_added_and_does_not_invent_bounds(page_factory):
    make, _ = page_factory
    page = make()
    state = default_state()
    page.set_source(state)
    index = next(i for i in range(page.choice.count())
                 if page.choice.itemData(i).key == OBJECTIVE_LENS
                 and page.choice.itemData(i).field == "percent")
    # The combo was populated before this edit. References must not use that
    # stale combo value, nor silently create a calculation interval.
    state.objective_lens.percent = 68.1234567
    page.choice.setCurrentIndex(index)
    changes = []
    page.tuning_changed.connect(changes.append)
    page._add_range()
    assert float(page.ranges.item(0, 5).text()) == pytest.approx(68.1234567)
    assert not page.ranges.item(0, 5).flags() & Qt.ItemFlag.ItemIsEditable
    assert page.ranges.cellWidget(0, 5).isReadOnly()
    assert page.ranges.horizontalHeader().visualIndex(5) == 1
    assert page.ranges.cellWidget(0, 1).text() == ""
    assert page.ranges.cellWidget(0, 2).text() == ""
    assert page.live_table.cellWidget(0, 1).text() == "68.1234567 %"
    with pytest.raises(ValueError, match="required"):
        page.plan()
    assert not changes and not page.timer.isActive() and not page.busy


def test_rows_align_after_resize_advanced_expansion_and_both_scroll_directions(page_factory, qtbot):
    from PySide6.QtWidgets import QGroupBox
    make, _ = page_factory
    page = make()
    page.set_source(default_state())
    for i in range(page.choice.count()):
        page.choice.setCurrentIndex(i)
        page._add_range()
    qtbot.waitUntil(lambda: page.ranges.verticalScrollBar().maximum() > 0)
    qtbot.wait(20)
    _assert_aligned(page)
    page.resize(1350, 700)
    page.splitter.moveSplitter(550, 1)
    advanced = next(w for w in page.findChildren(QGroupBox) if w.title() == "Advanced bank (optional)")
    advanced.setChecked(True)
    qtbot.wait(40)
    _assert_aligned(page)
    for table in (page.ranges, page.live_table):
        bar = table.verticalScrollBar()
        for value in (bar.maximum() // 2, bar.maximum(), 0):
            bar.setValue(value)
            qtbot.wait(10)
            _assert_aligned(page)
    page.ranges.setCurrentCell(2, 0)
    page._remove_range()
    qtbot.wait(20)
    _assert_aligned(page)


def test_edited_draft_keeps_bank_controls_and_matches_by_identity(page_factory, qtbot):
    make, _ = page_factory
    page = make()
    _start_lenses(page)
    axes = page.live_plan.ranges
    page._bank_ready(SimpleNamespace(plan=page.live_plan, points=(), retained_bytes=0))
    page.timer.stop()
    cached_widget = page.live_widgets[axes[0].control.identity]
    page.ranges.setCurrentCell(0, 0)
    page._remove_range()
    qtbot.wait(20)
    _assert_aligned(page)
    assert page.live_widgets[axes[0].control.identity] is cached_widget
    old_row = page.live_table.control_row(axes[0].control.identity)
    assert page.live_table.item(old_row, 0).text().startswith("Active only:")
    assert page.live_table.verticalHeader().visualIndex(old_row) == 2


def test_main_window_reference_reads_live_value_not_capture_snapshot(qtbot):
    from PySide6.QtWidgets import QApplication
    from temsim.gui.main_window import MainWindow
    window = MainWindow()
    qtbot.addWidget(window)
    window.preview_timer.stop()
    window._capture_interactive_settings()
    page = window.workspace.interactive_calculation
    old_value = page.source_state.objective_lens.percent
    window.state.objective_lens.percent = old_value + 0.012345
    page.choice.setCurrentIndex(next(i for i in range(page.choice.count())
                                    if page.choice.itemData(i).key == OBJECTIVE_LENS
                                    and page.choice.itemData(i).field == "percent"))
    page._add_range()
    value = page.ranges.cellWidget(0, 5)
    assert float(value.text()) == pytest.approx(window.state.objective_lens.percent)
    assert page.source_state.objective_lens.percent == old_value
    value.selectAll()
    value.copy()
    assert QApplication.clipboard().text() == value.text()
    assert not page.timer.isActive() and not window.preview_timer.isActive()
    page.shutdown()
