"""Explicit page requests; this widget never executes numerical work."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QWidget


class PageCalculationBar(QWidget):
    requested = Signal()

    def __init__(self, label, object_name, *, button=None, note="", parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.label = str(label)
        self.button = button if button is not None else QPushButton(f"Calculate {label}")
        self.button.setText(f"Calculate {label}")
        self.button.setObjectName(object_name)
        self.button.setToolTip(
            "Request this page's calculation in the background. Valid upstream "
            "results can be reused. Changing a tab does not calculate anything."
        )
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setMinimumWidth(0)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.status.setStyleSheet("color: #94a3b8;")
        self.note = str(note)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.button)
        layout.addWidget(self.status, 1)
        self.button.clicked.connect(self.requested.emit)
        self.set_result_available(False)

    def set_result_available(self, available):
        self.status.setText(
            ("Completed result displayed." if available else f"Click Calculate {self.label} to update this page.")
            + (" " + self.note if self.note else "")
        )

    def mark_stale(self, *_args):
        self.status.setText(f"Inputs changed. Click Calculate {self.label} to refresh this page.")
