"""Review persistent axial placement in the Physical Layout workspace."""
from copy import deepcopy

from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFormLayout, QLabel,
                               QLineEdit, QTableWidget, QTableWidgetItem, QVBoxLayout)
from temsim.gui.input_policy import WheelSafeComboBox
from temsim.subassemblies import resolved_origins


class SubassemblyPlacementDialog(QDialog):
    def __init__(self, session, on_applied, parent=None):
        super().__init__(parent)
        self.session, self.on_applied = session, on_applied
        self.before = deepcopy(session.document)
        self.entries = {entry["key"]: entry for entry in self.before["subassemblies"]}
        self.candidate = None
        self.setWindowTitle("Subassemblies · axial placement")
        self.resize(820, 590)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)
        self.group = WheelSafeComboBox()
        for key, entry in self.entries.items():
            self.group.addItem(entry.get("name", key), key)
        form.addRow("Subassembly", self.group)
        self.storage = QLabel()
        self.storage.setWordWrap(True)
        form.addRow("Part definitions", self.storage)
        self.mode = WheelSafeComboBox()
        self.mode.addItem("Fixed module-local origin", "fixed")
        self.mode.addItem("Follow a component anchor", "anchor")
        form.addRow("Placement", self.mode)
        self.origin = QLineEdit()
        form.addRow("Origin Z (mm)", self.origin)
        self.reference = WheelSafeComboBox()
        for part in self.before["parts"]:
            self.reference.addItem(part.get("name", part["key"]), part["key"])
        form.addRow("Reference component", self.reference)
        self.point = WheelSafeComboBox()
        for name in ("start", "center", "end"):
            self.point.addItem(name.title(), name)
        form.addRow("Reference point", self.point)
        self.offset = QLineEdit("0")
        self.datum = QLineEdit("0")
        form.addRow("Offset from reference (mm)", self.offset)
        form.addRow("Local datum to align (mm)", self.datum)
        note = QLabel("All Z values below are local to the installed module. "
                      "Anchor + offset = subassembly origin + local datum. "
                      "Only declared dependent subassemblies follow. Filter path s stays unchanged. "
                      "Apply updates the draft; Save validates every compatible instrument and writes the files.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.preview = QTableWidget(0, 3)
        self.preview.setHorizontalHeaderLabels(("Affected component", "Center Z before (mm)", "Center Z after (mm)"))
        self.preview.horizontalHeader().setStretchLastSection(True)
        self.preview.setColumnWidth(0, 325)
        self.preview.setColumnWidth(1, 170)
        self.preview.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.preview, 1)
        self.error = QLabel()
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel)
        self.apply_button = self.buttons.button(QDialogButtonBox.StandardButton.Apply)
        self.apply_button.setText("Apply to draft")
        self.apply_button.clicked.connect(self.apply)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.group.currentIndexChanged.connect(self.load)
        for control in (self.mode, self.reference, self.point):
            control.currentIndexChanged.connect(self.refresh)
        for control in (self.origin, self.offset, self.datum):
            control.textChanged.connect(self.refresh)
        self.load()

    def load(self, *_):
        self._loading = True
        entry = self.entries[self.group.currentData()]
        placement = entry["placement"]
        self.storage.setText(entry["file"])
        self.mode.setCurrentIndex(self.mode.findData(placement["mode"]))
        self.origin.setText(format(resolved_origins(self.before)[entry["key"]], ".15g"))
        self.reference.setCurrentIndex(max(0, self.reference.findData(placement.get("reference_part"))))
        self.point.setCurrentIndex(max(0, self.point.findData(placement.get("reference_point", "center"))))
        self.offset.setText(str(placement.get("offset_mm", 0)))
        self.datum.setText(str(placement.get("local_datum_mm", 0)))
        self._loading = False
        self.refresh()

    def refresh(self, *_):
        if getattr(self, "_loading", False):
            return
        fixed = self.mode.currentData() == "fixed"
        self.origin.setEnabled(fixed)
        for widget in (self.reference, self.point, self.offset, self.datum):
            widget.setEnabled(not fixed)
        self.candidate = None
        self.preview.setRowCount(0)
        try:
            if self.session.document != self.before:
                raise ValueError("The draft changed; reopen placement to review it")
            placement = ({"mode": "fixed", "origin_z_mm": float(self.origin.text())} if fixed else
                         {"mode": "anchor", "reference_part": self.reference.currentData(),
                          "reference_point": self.point.currentData(), "offset_mm": float(self.offset.text()),
                          "local_datum_mm": float(self.datum.text())})
            candidate = deepcopy(self.session)
            candidate.place_subassembly(self.group.currentData(), placement)
            previous = {part["key"]: part for part in self.before["parts"]}
            changed = [part for part in candidate.document["parts"] if part != previous[part["key"]]]
            self.preview.setRowCount(len(changed))
            for index, part in enumerate(changed):
                values = (part.get("name", part["key"]), format(previous[part["key"]]["local_center_z_mm"], ".12g"),
                          format(part["local_center_z_mm"], ".12g"))
                for column, text in enumerate(values):
                    self.preview.setItem(index, column, QTableWidgetItem(text))
            self.candidate = placement
            self.error.setText(f"{len(changed)} components move. Resolved origin: "
                               f"{resolved_origins(candidate.document)[self.group.currentData()]:.12g} mm")
        except (ValueError, KeyError) as exc:
            self.error.setText(str(exc))
        self.apply_button.setEnabled(self.candidate is not None)

    def apply(self):
        try:
            if self.parent() is not None and getattr(self.parent(), "session", self.session) is not self.session:
                raise ValueError("The editor switched files; reopen placement for the current file")
            if self.session.document != self.before:
                raise ValueError("The draft changed while this dialog was open; reopen placement to review it")
            self.session.assert_source_current()
            if self.candidate is None:
                return
            self.session.place_subassembly(self.group.currentData(), self.candidate)
            self.on_applied()
            self.accept()
        except ValueError as exc:
            self.error.setText(str(exc))
