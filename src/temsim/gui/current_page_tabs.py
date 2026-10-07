"""Workspace tabs must not reserve vertical space for hidden pages."""

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QStyle, QStyleOptionTabWidgetFrame, QTabWidget


class CurrentPageHeightTabs(QTabWidget):
    """For horizontal tab bars, keep usual widths and use the visible height."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.currentChanged.connect(lambda _index: self.updateGeometry())

    def _with_tab_frame(self, height):
        option = QStyleOptionTabWidgetFrame()
        self.initStyleOption(option)
        tabs_height = self.tabBar().sizeHint().height() if not self.tabBar().isHidden() else 0
        return self.style().sizeFromContents(
            QStyle.ContentsType.CT_TabWidget, option,
            QSize(0, height + tabs_height), self,
        ).height()

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        page = self.currentWidget()
        if page is not None:
            hint.setHeight(self._with_tab_frame(max(
                page.minimumHeight(), page.minimumSizeHint().height(),
            )))
        return hint

    def hasHeightForWidth(self):
        page = self.currentWidget()
        return page is not None and page.hasHeightForWidth()

    def heightForWidth(self, width):
        page = self.currentWidget()
        if page is not None and page.hasHeightForWidth():
            return max(self.minimumSizeHint().height(), self._with_tab_frame(page.heightForWidth(width)))
        return self.minimumSizeHint().height()
