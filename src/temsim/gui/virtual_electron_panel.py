"""Compact, dockable layout for virtual-electron records and their editor."""
from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QSplitter, QVBoxLayout, QWidget


class VirtualElectronPanel(QWidget):
    """Stack controls in a narrow dock; use columns when floated wider."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("virtualElectronPanel")
        self.columns = QSplitter(Qt.Orientation.Vertical, self)
        self.columns.setObjectName("virtualElectronControlsSplitter")
        self.columns.setChildrenCollapsible(False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self.columns)
        self._list_panel = None

    def sizeHint(self):
        return QSize(460, 680)

    def minimumSizeHint(self):
        return QSize(300, 240)

    def install_columns(self, list_panel, editor):
        self._list_panel = list_panel
        self.columns.addWidget(list_panel)
        self.columns.addWidget(editor)
        self.columns.setStretchFactor(0, 0)
        self.columns.setStretchFactor(1, 1)
        self._adapt_layout(force=True)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._adapt_layout()

    def showEvent(self, event):
        super().showEvent(event)
        self._adapt_layout()

    def _adapt_layout(self, *, force=False):
        if self._list_panel is None:
            return
        narrow = self.width() < 720
        orientation = Qt.Orientation.Vertical if narrow else Qt.Orientation.Horizontal
        self._list_panel.setMaximumHeight(230 if narrow else 16777215)
        if force or orientation != self.columns.orientation():
            self.columns.setOrientation(orientation)
            self.columns.setSizes((200, max(200, self.height()-220)) if narrow else (330, max(350, self.width()-350)))
