"""Screen-space labels stay separated across hidden tabs and viewport changes."""
import pyqtgraph as pg
import pytest
from PySide6.QtWidgets import QTabWidget, QWidget

from temsim.gui.diagnostic_tabs import PhysicalLayoutView


def assert_separated(page):
    labels = [item.label for item in page._label_callouts.values() if item.label.isVisible()]
    assert len(labels) >= 8
    for index, first in enumerate(labels):
        for second in labels[index + 1:]:
            assert not first.sceneBoundingRect().intersects(second.sceneBoundingRect()), (
                first.toPlainText(), second.toPlainText(), first.sceneBoundingRect(), second.sceneBoundingRect())


@pytest.mark.parametrize("initially_hidden", [False, True])
def test_callouts_after_tab_activation_resize_and_zoom(qtbot, initially_hidden):
    tabs = QTabWidget()
    qtbot.addWidget(tabs)
    tabs.addTab(QWidget(), "Other")
    page = PhysicalLayoutView()
    tabs.addTab(page, "Physical")
    tabs.resize(1300, 900)
    if not initially_hidden:
        tabs.setCurrentWidget(page)
    tabs.show()
    qtbot.wait(20)
    page.plot.setRange(xRange=(-20., 3000.), yRange=(-150., 150.), padding=0)
    for index in range(24):
        label = pg.TextItem(f"Component {index} long name", anchor=(.5, .5))
        page.plot.addItem(label)
        page._register_label_callout(key=f"component:test{index}", label=label,
            anchor_z_mm=index*15., anchor_radius_mm=10., colour="#ffffff",
            priority=0, preferred_side=1, component_key=f"test{index}")
    page._layout_component_labels()
    tabs.setCurrentWidget(page)
    qtbot.wait(30)
    assert_separated(page)
    for width in (850, 1600):
        tabs.resize(width, 900)
        qtbot.wait(30)
        assert_separated(page)
    page.plot.setRange(xRange=(-20., 1000.), yRange=(-100., 100.), padding=0)
    qtbot.wait(30)
    assert_separated(page)
    tabs.setCurrentIndex(0)
    tabs.resize(1100, 700)
    page.plot.setRange(xRange=(-20., 3000.), yRange=(-150., 150.), padding=0)
    tabs.setCurrentWidget(page)
    qtbot.wait(30)
    assert_separated(page)


def test_actual_assembly_labels_are_pixel_sized_before_first_paint(qtbot):
    from types import SimpleNamespace
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.column.state_layout import apply_physical_layout_to_state
    from temsim.optics.column import default_state

    state = default_state()
    catalog = AssemblyCatalog()
    assembly = catalog.apply(state, catalog.default_selection())
    page = PhysicalLayoutView()
    qtbot.addWidget(page)
    page.resize(1350, 700)
    page.show()
    qtbot.wait(20)
    # Render the resolved assembly only: no ray/field calculation is needed to
    # detect the first-paint text transform error.
    page.display_result(SimpleNamespace(assembly=assembly, state_snapshot=state,
        layout=apply_physical_layout_to_state(state)))
    for callout in page._label_callouts.values():
        if callout.label.isVisible():
            scene = callout.label.sceneBoundingRect()
            text = callout.label.textItem.boundingRect()
            assert scene.width() == pytest.approx(text.width())
            assert scene.height() == pytest.approx(text.height())
    assert_separated(page)
    qtbot.wait(30)
    assert_separated(page)
