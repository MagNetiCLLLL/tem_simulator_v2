"""Complete TEM assembly selector."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from temsim.assembly_catalog import AssemblySelection
from temsim.gui.instrument_tree import InstrumentTree


class AssemblyPanel(QWidget):
    selection_requested = Signal(object)

    def __init__(self, catalog, selection: AssemblySelection, parent=None) -> None:
        super().__init__(parent)
        self.catalog = catalog

        box = QGroupBox("Assembly modules")
        form = QFormLayout(box)
        self.gun = QComboBox()
        self.column = QComboBox()
        self.recording = QComboBox()
        self.gun.addItems([option.name for option in catalog.guns])
        self.column.addItems([option.name for option in catalog.columns])
        self.recording.addItems(
            [option.name for option in catalog.recording_systems]
        )
        self.set_selection(selection)
        form.addRow("Gun", self.gun)
        form.addRow("Column", self.column)
        form.addRow("Recording", self.recording)
        apply_button = QPushButton("Load assembly")
        apply_button.setObjectName("loadAssemblyButton")
        apply_button.clicked.connect(self._request_selection)
        form.addRow(apply_button)

        self.tree = InstrumentTree()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(box)
        layout.addWidget(self.tree, 1)

    def current_selection(self) -> AssemblySelection:
        return AssemblySelection(
            gun=self.gun.currentText(),
            column=self.column.currentText(),
            recording=self.recording.currentText(),
        )

    def set_selection(self, selection: AssemblySelection) -> None:
        self.gun.setCurrentText(selection.gun)
        self.column.setCurrentText(selection.column)
        self.recording.setCurrentText(selection.recording)

    def reload_catalog(self, catalog, selection: AssemblySelection) -> None:
        self.catalog = catalog
        for combo, options in (
            (self.gun, catalog.guns),
            (self.column, catalog.columns),
            (self.recording, catalog.recording_systems),
        ):
            combo.clear()
            combo.addItems([option.name for option in options])
        self.set_selection(selection)

    def _request_selection(self) -> None:
        self.selection_requested.emit(self.current_selection())
