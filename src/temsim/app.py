"""Qt application composition root."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

from temsim.gui.main_window import MainWindow


APPLICATION_STYLE = """
QMainWindow, QWidget {
    background: #111827;
    color: #e5e7eb;
}
QMenuBar, QMenu, QStatusBar, QToolBar {
    background: #172033;
    color: #e5e7eb;
}
QDockWidget::title {
    background: #1f2937;
    padding: 6px;
}
QTreeWidget, QTableWidget, QPlainTextEdit {
    background: #0f172a;
    alternate-background-color: #172033;
    border: 1px solid #334155;
}
QHeaderView::section {
    background: #1f2937;
    color: #e5e7eb;
    padding: 5px;
    border: 0;
}
QPushButton, QComboBox, QLineEdit {
    background: #1f2937;
    border: 1px solid #475569;
    border-radius: 3px;
    padding: 4px 7px;
}
QPushButton:hover {
    background: #334155;
}
QProgressBar {
    border: 1px solid #475569;
    border-radius: 3px;
    text-align: center;
}
QProgressBar::chunk {
    background: #2563eb;
}
"""


def create_application(argv: Sequence[str] | None = None) -> QApplication:
    """Return the process-wide application instance."""

    existing = QApplication.instance()
    if existing is not None:
        return existing

    QCoreApplication.setOrganizationName("TEM Simulator")
    QCoreApplication.setApplicationName("TEM Simulator v2")
    application = QApplication(list(argv) if argv is not None else sys.argv)
    application.setStyle("Fusion")
    application.setStyleSheet(APPLICATION_STYLE)
    return application


def run(argv: Sequence[str] | None = None) -> int:
    application = create_application(argv)
    window = MainWindow()
    window.show()
    return application.exec()
