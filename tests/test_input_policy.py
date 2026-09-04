"""Exercise native wheel delivery, including Qt's parent-scroll propagation."""

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QScrollArea,
    QSpinBox,
    QStyle,
    QStyleOptionSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from temsim.gui.input_policy import (
    WheelSafeComboBox,
    WheelSafeDoubleSpinBox,
    WheelSafeSpinBox,
    install_numeric_input_policy,
)
from temsim.gui.parameter_panel import ParameterPanel
from temsim.optics.column import default_state
from temsim.runtime_parameters import runtime_targets


def _remove_policy(app):
    event_filter = getattr(app, "_numeric_wheel_filter", None)
    if event_filter is not None:
        app.removeEventFilter(event_filter)
        del app._numeric_wheel_filter


@pytest.fixture
def without_application_bootstrap(qapp):
    _remove_policy(qapp)
    yield qapp
    install_numeric_input_policy(qapp)


def _show(widget, qtbot):
    qtbot.addWidget(widget)
    widget.show()
    widget.activateWindow()
    QApplication.processEvents()
    assert QTest.qWaitForWindowExposed(widget)


def _wheel(widget, delta=-120):
    # Use the visible left edge, since a form's long numeric editor may extend
    # beyond its scroll viewport.  Deliver through QWindow like a real mouse;
    # QApplication.sendEvent(widget, ...) would bypass parent propagation.
    position = QPoint(8, widget.height() // 2)
    window = widget.window()
    destination = widget.mapTo(window, position)
    assert window.rect().contains(destination)
    hit = QApplication.widgetAt(widget.mapToGlobal(position))
    assert hit is widget or widget.isAncestorOf(hit)
    QTest.wheelEvent(window.windowHandle(), destination, QPoint(0, delta))
    QApplication.processEvents()


def _scroll_host(control, qtbot):
    host = QScrollArea()
    host.setWidgetResizable(True)
    content = QWidget()
    layout = QVBoxLayout(content)
    layout.addWidget(control)
    filler = QWidget()
    filler.setMinimumHeight(900)
    layout.addWidget(filler)
    host.setWidget(content)
    host.resize(420, 240)
    _show(host, qtbot)
    assert host.verticalScrollBar().maximum() > 0
    return host


@pytest.mark.parametrize("editor_type", [WheelSafeSpinBox, WheelSafeDoubleSpinBox, WheelSafeComboBox])
@pytest.mark.parametrize("focused", [False, True])
def test_inputs_reject_native_wheel_and_scroll_the_page(
    qtbot, without_application_bootstrap, editor_type, focused
):
    editor = editor_type()
    if isinstance(editor, QComboBox):
        editor.addItems(["10", "20", "30"])
        editor.setCurrentIndex(1)
        value = editor.currentIndex
    else:
        editor.setRange(0, 100)
        editor.setValue(50)
        value = editor.value
    host = _scroll_host(editor, qtbot)
    if focused:
        editor.setFocus()
    else:
        host.setFocus()
    initial = value()
    _wheel(editor)
    assert value() == initial
    assert host.verticalScrollBar().value() > 0


def test_standalone_parameter_panel_installs_policy_and_protects_dynamic_controls(
    qtbot, without_application_bootstrap
):
    app = without_application_bootstrap
    assert not hasattr(app, "_numeric_wheel_filter")
    state = default_state()
    targets = runtime_targets(state)
    panel = ParameterPanel()
    panel.set_context("C2", targets["condenser_lens_2"], None, (), None)
    panel.resize(790, 420)
    _show(panel, qtbot)
    assert hasattr(app, "_numeric_wheel_filter")
    excitation = panel.lens_excitation.value()
    panel.lens_excitation.setFocus()
    _wheel(panel.lens_excitation)
    assert panel.lens_excitation.value() == excitation
    assert targets["condenser_lens_2"].obj.percent == excitation
    assert panel.scroll_area.verticalScrollBar().value() > 0

    aperture = targets["condenser_aperture_2"]
    panel.set_context("Aperture", aperture, None, (), None)
    QApplication.processEvents()
    diameter = panel._quick_widgets["diameter_mm"]
    before = diameter.value()
    diameter.setFocus()
    _wheel(diameter)
    assert diameter.value() == before
    assert aperture.obj.diameter_mm == pytest.approx(before / 1000.0)


def test_native_delegate_created_after_policy_is_protected(qtbot, without_application_bootstrap):
    # Establish that the native-wheel helper really reaches a stepping handler.
    native = QSpinBox()
    native.setValue(50)
    native.resize(160, 35)
    _show(native, qtbot)
    native.setFocus()
    _wheel(native)
    assert native.value() == 49
    native.close()

    install_numeric_input_policy()
    table = QTableWidget(30, 1)
    for row in range(30):
        item = QTableWidgetItem()
        item.setData(Qt.ItemDataRole.EditRole, 50)
        table.setItem(row, 0, item)
    table.resize(340, 240)
    _show(table, qtbot)
    table.editItem(table.item(0, 0))
    QApplication.processEvents()
    editor = table.findChild(QSpinBox)
    assert editor is not None
    assert not isinstance(editor, WheelSafeSpinBox)
    editor.setFocus()
    _wheel(editor)
    assert editor.value() == 50
    assert table.verticalScrollBar().value() > 0


def test_typing_keyboard_and_arrow_buttons_still_edit(qtbot):
    editor = WheelSafeDoubleSpinBox()
    editor.setRange(0, 100)
    editor.setValue(30)
    editor.resize(200, 35)
    _show(editor, qtbot)
    editor.setFocus()
    QTest.keyClick(editor, Qt.Key.Key_Up)
    assert editor.value() == 31
    editor.lineEdit().selectAll()
    QTest.keyClicks(editor.lineEdit(), "42.50")
    QTest.keyClick(editor, Qt.Key.Key_Return)
    assert editor.value() == 42.5
    option = QStyleOptionSpinBox()
    editor.initStyleOption(option)
    up = editor.style().subControlRect(
        QStyle.ComplexControl.CC_SpinBox,
        option,
        QStyle.SubControl.SC_SpinBoxUp,
        editor,
    )
    QTest.mouseClick(editor, Qt.MouseButton.LeftButton, pos=up.center())
    assert editor.value() == 43.5


def test_open_combo_popup_can_scroll_without_changing_selected_value(qtbot):
    combo = WheelSafeComboBox()
    combo.setStyleSheet("QComboBox { combobox-popup: 0; }")
    combo.setMaxVisibleItems(6)
    combo.addItems([str(value) for value in range(100)])
    combo.resize(240, 35)
    _show(combo, qtbot)
    combo.showPopup()
    QApplication.processEvents()
    view = combo.view()
    assert view.isVisible()
    assert view.verticalScrollBar().maximum() > 0
    _wheel(view.viewport())
    assert view.verticalScrollBar().value() > 0
    assert combo.currentIndex() == 0
    combo.hidePopup()
    QTest.keyClick(combo, Qt.Key.Key_Down)
    assert combo.currentIndex() == 1


def test_plot_wheel_zoom_is_not_intercepted(qtbot):
    import pyqtgraph as pg

    plot = pg.PlotWidget()
    plot.resize(420, 300)
    plot.plot([0, 1], [0, 1])
    _show(plot, qtbot)
    before = plot.viewRange()
    viewport = plot.viewport()
    position = viewport.rect().center()
    QTest.wheelEvent(
        plot.windowHandle(), viewport.mapTo(plot, position), QPoint(0, 120)
    )
    QApplication.processEvents()
    assert plot.viewRange() != before
