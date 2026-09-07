"""Read-only, searchable dimensions and evidence audit with source navigation."""

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from temsim.gui.input_policy import WheelSafeComboBox as QComboBox


class DimensionAuditDialog(QDialog):
    parameter_requested = Signal(str, str, object)

    def __init__(self, audit, parent=None, *, scope="Saved catalog; unsaved drafts excluded"):
        super().__init__(parent)
        self.audit = audit
        self.setWindowTitle("Dimension definitions and evidence audit")
        self.resize(1280, 700)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Find component, TOML field, source file or issue")
        self.filter = QComboBox()
        self.filter.addItem("All dimensions", "all")
        self.filter.addItem("Needs review", "review")
        self.filter.addItem("Unknown meaning", "unknown")
        self.filter.addItem("Unspecified evidence", "unspecified")
        self.filter.addItem("Mechanism / display envelope", "envelope")
        self.filter.addItem("CAD dimensions", "cad")
        self.export_button = QPushButton("Export report…")
        row = QHBoxLayout()
        row.addWidget(self.search, 1)
        row.addWidget(self.filter)
        row.addWidget(self.export_button)
        self.table = QTableWidget(0, 7)
        self.table.setObjectName("dimensionAuditTable")
        self.table.setHorizontalHeaderLabels(("Component", "Parameter", "Value", "Meaning", "Evidence", "File", "Review note"))
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.verticalHeader().hide()
        self.table.setWordWrap(False)
        for index, width in enumerate((150, 225, 85, 150, 145, 170)):
            self.table.horizontalHeader().setSectionResizeMode(index, QHeaderView.ResizeMode.Interactive)
            self.table.setColumnWidth(index, width)
        self.table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        self.detail = QLabel("Double-click a dimension to locate its component and parameter.")
        self.detail.setWordWrap(True)
        self.detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(scope))
        layout.addWidget(self.summary)
        layout.addLayout(row)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.detail)
        self._rows = self._make_rows()
        self.search.textChanged.connect(self._refresh)
        self.filter.currentIndexChanged.connect(self._refresh)
        self.table.cellDoubleClicked.connect(self._activate)
        self.table.currentCellChanged.connect(self._selection_changed)
        self.export_button.clicked.connect(self._export)
        self._refresh()

    def _make_rows(self):
        def parameter_name(path):
            path = tuple(path)
            relative = path[2:] if path and path[0] in {"parts", "runtime", "derived"} else path
            return ".".join(map(str, relative))

        issues = {}
        for item in self.audit.issues:
            issues.setdefault((item.source_file, item.part_key, tuple(item.path)), []).append(item.message)
        rows = []
        located = set()
        for record in self.audit.records:
            identity = (record.source_file, record.part_key, tuple(record.path))
            located.add(identity)
            meaning = record.semantics
            note = "\n".join(issues.get(identity, ()))
            value = f"{record.value:g} {meaning.unit}".strip()
            rows.append(dict(identity=identity, category=meaning.category, source=meaning.source_kind,
                             review=bool(note) or meaning.category == "unknown" or meaning.source_kind == "unspecified",
                             columns=(record.part_name or record.part_key, parameter_name(record.path), value,
                                      meaning.category_label, meaning.source_label, Path(record.source_file).name, note),
                             detail=f"Source: {record.source_file}\n{meaning.label}: {record.value} {meaning.unit}\n{meaning.description}\nEvidence: {meaning.source_note}"))
        for identity, messages in issues.items():
            if identity not in located:
                rows.append(dict(identity=identity, category="unknown", source="unspecified", review=True,
                                 columns=(identity[1], parameter_name(identity[2]), "—", "Needs definition", "Unspecified", Path(identity[0]).name, "\n".join(messages)),
                                 detail=f"Source: {identity[0]}\n" + "\n".join(messages)))
        return rows

    def _refresh(self):
        query = self.search.text().casefold().strip()
        choice = self.filter.currentData()
        rows = [row for row in self._rows if (not query or query in (" ".join(row["columns"]) + " " + str(row["identity"])).casefold())
                and (choice == "all" or choice == "review" and row["review"]
                     or choice == "unspecified" and row["source"] == choice
                     or row["category"] == choice)]
        self.table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            for column, value in enumerate(row["columns"]):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, row["identity"])
                item.setToolTip(row["detail"] + ("\n" + row["columns"][-1] if row["columns"][-1] else ""))
                self.table.setItem(index, column, item)
        self.summary.setText(f"{self.audit.module_count} modules · {self.audit.part_count} components · "
                             f"{len(self.audit.records)} dimensions · {len(self.audit.issues)} review items · {len(rows)} rows shown")

    def _selection_changed(self, row, *_):
        item = self.table.item(row, 0)
        if item:
            self.detail.setText(item.toolTip())

    def _activate(self, row, _column):
        source, key, path = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        self.parameter_requested.emit(source, key, path)

    def _export(self):
        filename, selected_filter = QFileDialog.getSaveFileName(self, "Export dimension audit", "dimension-audit.md",
                                                               "Markdown report (*.md);;JSON data (*.json)")
        if filename:
            path = Path(filename)
            if not path.suffix:
                path = path.with_suffix(".json" if "JSON" in selected_filter else ".md")
            try:
                path.write_text(self.audit.to_json() if path.suffix.lower() == ".json" else self.audit.to_markdown(), encoding="utf-8")
                self.detail.setText(f"Report exported: {path}")
            except OSError as exc:
                self.detail.setText(f"Report was not exported: {exc}")
