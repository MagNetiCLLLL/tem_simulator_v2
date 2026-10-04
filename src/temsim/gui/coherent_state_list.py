"""Display controls for explicitly captured physical-tip initial states.

This widget owns names and relative display weights only. Source settings,
propagation sessions and result validation belong to its controller. No edit
here implicitly starts a calculation or changes the active instrument source.
"""

from dataclasses import dataclass

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDoubleSpinBox, QHeaderView, QHBoxLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)


@dataclass(frozen=True)
class TipStateEntry:
    """Display metadata; ``weight`` is a relative probability, not a count."""

    id: str
    name: str
    weight: float
    checked: bool


class TipStateList(QWidget):
    add_requested = Signal()
    replace_requested = Signal(str)
    remove_requested = Signal(str)
    restore_requested = Signal(str)
    display_changed = Signal()
    entries_edited = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._summaries: dict[str, str] = {}
        self._statuses: dict[str, str] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.mode = QComboBox()
        self.mode.setObjectName("coherentStateDisplayMode")
        self.mode.addItem("Current tip parameters", "current")
        self.mode.addItem("Selected state", "selected")
        self.mode.addItem("Overlay checked states", "overlay")
        self.mode.setToolTip(
            "Choose a display only. Independent states contribute intensities; "
            "their complex phases are never added together.")
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Display"))
        mode_row.addWidget(self.mode, 1)
        layout.addLayout(mode_row)

        self.table = QTableWidget(0, 4)
        self.table.setObjectName("coherentTipStateTable")
        self.table.setHorizontalHeaderLabels(("", "State", "Weight", "Status / tip"))
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed)
        self.table.setWordWrap(False)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(28)
        self.table.setMinimumHeight(100)
        self.table.setMaximumHeight(200)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(0, 25)
        self.table.setColumnWidth(1, 115)
        self.table.setColumnWidth(2, 100)
        layout.addWidget(self.table)

        self.add_button = QPushButton("Add current state")
        self.replace_button = QPushButton("Replace selected")
        self.restore_button = QPushButton("Apply selected state")
        self.remove_button = QPushButton("Remove")
        self.add_button.setToolTip(
            "Capture the already applied physical-tip source. Apply tip parameters first; "
            "unapplied drafts are not captured and no calculation starts.")
        self.replace_button.setToolTip(
            "Replace the selected capture with the already applied physical-tip source. "
            "No calculation starts.")
        self.restore_button.setToolTip(
            "Explicitly apply the selected captured source to the physical tip, "
            "for both particles and waves. This is not a downstream source override.")
        self.remove_button.setToolTip("Remove the selected state from this comparison list.")
        first_buttons = QHBoxLayout()
        first_buttons.addWidget(self.add_button)
        first_buttons.addWidget(self.replace_button)
        first_buttons.addStretch(1)
        second_buttons = QHBoxLayout()
        second_buttons.addWidget(self.restore_button)
        second_buttons.addWidget(self.remove_button)
        second_buttons.addStretch(1)
        layout.addLayout(first_buttons)
        layout.addLayout(second_buttons)

        self.help = QLabel(
            "Independent states add intensities. Weights are relative probabilities, "
            "not electron counts. Each row records initial emission parameters for the same Tip. "
            "Apply tip parameters before Add or Replace. Choose Selected state or "
            "Overlay checked states, then use Calculate beam above.")
        self.help.setTextFormat(Qt.TextFormat.PlainText)
        self.help.setWordWrap(True)
        self.help.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.help)

        self.add_button.clicked.connect(self.add_requested.emit)
        self.replace_button.clicked.connect(lambda: self._emit_selected(self.replace_requested))
        self.restore_button.clicked.connect(lambda: self._emit_selected(self.restore_requested))
        self.remove_button.clicked.connect(lambda: self._emit_selected(self.remove_requested))
        self.mode.currentIndexChanged.connect(lambda _index: self.display_changed.emit())
        self.table.itemChanged.connect(self._item_changed)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self._update_buttons()

    def _row(self, entry_id: str) -> int:
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).data(Qt.ItemDataRole.UserRole) == entry_id:
                return row
        raise KeyError(entry_id)

    def add_entry(self, entry_id: str, name: str, summary: str):
        """Add an initially checked state with unit relative weight; select it."""
        if not isinstance(entry_id, str) or not entry_id:
            raise ValueError("A captured tip state needs a non-empty string ID")
        if entry_id in self._summaries:
            raise ValueError(f"Duplicate captured tip state ID: {entry_id}")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("A captured tip state needs a name")
        self._summaries[entry_id] = str(summary)
        self._statuses[entry_id] = "Not calculated"
        with QSignalBlocker(self.table):
            row = self.table.rowCount()
            self.table.insertRow(row)
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                           | Qt.ItemFlag.ItemIsUserCheckable)
            check.setCheckState(Qt.CheckState.Checked)
            check.setData(Qt.ItemDataRole.UserRole, entry_id)
            self.table.setItem(row, 0, check)
            self.table.setItem(row, 1, QTableWidgetItem(name.strip()))
            weight = QDoubleSpinBox()
            weight.setObjectName("coherentStateRelativeWeight")
            weight.setDecimals(4)
            weight.setRange(0., 1e12)
            weight.setSingleStep(.1)
            weight.setValue(1.)
            weight.setKeyboardTracking(False)
            weight.setToolTip("Relative probability in the displayed mixture; zero omits this contribution.")
            weight.valueChanged.connect(lambda _value: self._metadata_changed())
            self.table.setCellWidget(row, 2, weight)
            info = QTableWidgetItem()
            info.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.table.setItem(row, 3, info)
            self._refresh_info(row, entry_id)
            self.table.selectRow(row)
        self._update_buttons()
        self._metadata_changed()

    def remove_entry(self, entry_id: str):
        """Remove a row without issuing another removal request."""
        row = self._row(entry_id)
        selected = self.selected_id()
        with QSignalBlocker(self.table):
            self.table.removeRow(row)
            del self._summaries[entry_id]
            del self._statuses[entry_id]
            if selected and selected != entry_id:
                self.table.selectRow(self._row(selected))
            elif self.table.rowCount():
                self.table.selectRow(min(row, self.table.rowCount() - 1))
        self._update_buttons()
        self._metadata_changed()

    def set_status(self, entry_id: str, text: str):
        row = self._row(entry_id)
        with QSignalBlocker(self.table):
            self._statuses[entry_id] = str(text)
            self._refresh_info(row, entry_id)

    def set_summary(self, entry_id: str, text: str):
        """Refresh an external capture's summary without changing selection."""
        row = self._row(entry_id)
        with QSignalBlocker(self.table):
            self._summaries[entry_id] = str(text)
            self._refresh_info(row, entry_id)

    def _refresh_info(self, row: int, entry_id: str):
        status, summary = self._statuses[entry_id], self._summaries[entry_id]
        item = self.table.item(row, 3)
        item.setText(f"{status} · {summary}" if summary else status)
        item.setToolTip(f"{status}\n{summary}".rstrip())

    def entries(self) -> tuple[TipStateEntry, ...]:
        return tuple(TipStateEntry(
            self.table.item(row, 0).data(Qt.ItemDataRole.UserRole),
            self.table.item(row, 1).text(),
            self.table.cellWidget(row, 2).value(),
            self.table.item(row, 0).checkState() == Qt.CheckState.Checked,
        ) for row in range(self.table.rowCount()))

    def selected_id(self) -> str | None:
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            return None
        return self.table.item(selected[0].row(), 0).data(Qt.ItemDataRole.UserRole)

    def display_mode(self) -> str:
        return self.mode.currentData()

    def _emit_selected(self, signal):
        entry_id = self.selected_id()
        if entry_id is not None:
            signal.emit(entry_id)

    def _update_buttons(self):
        selected = self.selected_id() is not None
        for button in (self.replace_button, self.restore_button, self.remove_button):
            button.setEnabled(selected)

    def _selection_changed(self):
        self._update_buttons()
        if self.display_mode() == "selected":
            self.display_changed.emit()

    def _item_changed(self, item):
        if item.column() in (0, 1):
            if item.column() == 1 and not item.text().strip():
                with QSignalBlocker(self.table):
                    item.setText("Unnamed state")
            self._metadata_changed()

    def _metadata_changed(self):
        self.entries_edited.emit()
        self.display_changed.emit()
