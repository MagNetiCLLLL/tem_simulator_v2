"""Read-only inspection of retired specimen-entrance source definitions."""
from copy import deepcopy
import json

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QPlainTextEdit, QVBoxLayout


class IlluminationDialog(QDialog):
    """Historical definitions must never become production source controls."""

    def __init__(self, config, energy_kev, parent=None):
        super().__init__(parent)
        self.config = deepcopy(config)
        self.setWindowTitle("Historical illumination (read-only)")
        self.resize(620, 440)
        layout = QVBoxLayout(self)
        notice = QLabel("Independent specimen-entrance sources are retired. Use the electron gun and column controls.")
        notice.setWordWrap(True)
        layout.addWidget(notice)
        self.editor = QPlainTextEdit()
        self.editor.setReadOnly(True)
        self.editor.setPlainText(json.dumps(self.config, indent=2))
        layout.addWidget(self.editor)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def value(self):
        return deepcopy(self.config)

    def accept(self):
        # Even a programmatic acceptance cannot publish a historical source.
        self.reject()
