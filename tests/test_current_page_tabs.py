"""Hidden workspace pages cannot push a small current page below the screen."""

from PySide6.QtWidgets import QScrollArea, QWidget

from temsim.gui.current_page_tabs import CurrentPageHeightTabs


def test_hidden_tall_page_does_not_force_scrolling_and_switching_updates_bounds(qtbot):
    tabs = CurrentPageHeightTabs()
    short, tall = QWidget(), QWidget()
    short.setMinimumHeight(80)
    tall.setMinimumHeight(900)
    tabs.addTab(short, "Small page")
    tabs.addTab(tall, "Tall page")
    scroll = QScrollArea()
    qtbot.addWidget(scroll)
    scroll.setWidgetResizable(True)
    scroll.setWidget(tabs)
    scroll.resize(640, 400)
    scroll.show()
    qtbot.wait(20)
    assert scroll.verticalScrollBar().maximum() == 0
    tabs.setCurrentWidget(tall)
    qtbot.waitUntil(lambda: scroll.verticalScrollBar().maximum() > 0)
    assert tall.height() >= 900
    tabs.setCurrentWidget(short)
    qtbot.waitUntil(lambda: scroll.verticalScrollBar().maximum() == 0)


def test_hidden_wrapping_page_cannot_impose_height_for_width(qtbot):
    class WrappingPage(QWidget):
        def hasHeightForWidth(self):
            return True

        def heightForWidth(self, width):
            return 700 + width

    tabs = CurrentPageHeightTabs()
    qtbot.addWidget(tabs)
    short, wrapping = QWidget(), WrappingPage()
    tabs.addTab(short, "Small page")
    tabs.addTab(wrapping, "Wrapping page")
    assert not tabs.hasHeightForWidth()
    assert tabs.heightForWidth(300) < 100
    tabs.setCurrentWidget(wrapping)
    assert tabs.hasHeightForWidth()
    assert tabs.heightForWidth(300) >= 1000
    tabs.setCurrentWidget(short)
    assert not tabs.hasHeightForWidth()
    assert tabs.heightForWidth(300) < 100
