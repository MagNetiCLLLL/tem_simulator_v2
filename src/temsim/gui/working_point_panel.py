"""Read-only cache browser. Restore/fork are explicit main-window transactions."""
import json

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QListWidget,
    QTreeWidget, QTreeWidgetItem, QTableWidget, QTableWidgetItem, QSplitter,
    QLabel, QPushButton, QFileDialog, QAbstractItemView)

from temsim.immutable_json import thaw_json
from temsim.working_point import WorkingPointCheckpoint, OBSERVABLE_DEFINITIONS, snapshot_changes


class WorkingPointPanel(QWidget):
    restore_requested = Signal(object, bool)
    undo_requested = Signal()
    error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("workingPointPanel")
        self._points = []
        self.current_snapshot = None
        layout = QVBoxLayout(self)
        self.status = QLabel("Selected checkpoint | read-only | no working point selected")
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        bar = QHBoxLayout()
        for label, callback in (("Import...", self._import), ("Export...", self._export),
                                ("Compare with current", self._compare), ("Restore working point", lambda: self._restore(False)),
                                ("Continue from this point", lambda: self._restore(True)), ("Undo alignment", self.undo_requested.emit)):
            button = QPushButton(label)
            button.clicked.connect(callback)
            bar.addWidget(button)
        layout.addLayout(bar)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setObjectName("workingPointBrowserSplitter")
        self.points = QListWidget()
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Component", "Model"])
        self.values = QTableWidget(0, 2)
        self.values.setHorizontalHeaderLabels(["Captured parameter", "Value"])
        self.values.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.values.horizontalHeader().setStretchLastSection(True)
        self.observables = QTableWidget(0, 4)
        self.observables.setHorizontalHeaderLabels(["Observable", "Value", "Unit", "Status"])
        self.observables.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.observables.horizontalHeader().setStretchLastSection(True)
        right = QSplitter(Qt.Orientation.Vertical)
        right.setObjectName("workingPointDetailsSplitter")
        right.addWidget(self.values)
        right.addWidget(self.observables)
        for widget in (self.points, self.tree, right):
            self.splitter.addWidget(widget)
        self.splitter.setSizes([180, 250, 570])
        layout.addWidget(self.splitter)
        self.points.currentRowChanged.connect(self._select)
        self.tree.currentItemChanged.connect(self._component)

    @property
    def selected(self):
        index = self.points.currentRow()
        return self._points[index] if 0 <= index < len(self._points) else None

    def add_checkpoint(self, checkpoint, *, label="Completed"):
        for index, point in enumerate(self._points):
            if point.digest == checkpoint.digest:
                self.points.setCurrentRow(index)
                return
        self._points.append(checkpoint)
        self.points.addItem(f"{label} | {checkpoint.digest[:12]}")
        self.points.setCurrentRow(len(self._points)-1)

    def _select(self, index):
        self.tree.clear()
        cp = self.selected
        if cp is None:
            return
        self.status.setText(f"Selected checkpoint {cp.digest[:12]} | Z {cp.plane_z_mm:g} mm | "
                            f"{cp.metadata.get('source_representation', 'historical')} | read-only")
        # Include every graph node, including non-component configuration and
        # shared references, rather than filtering out disabled hardware.
        for index, node in enumerate(cp.snapshot.graph["nodes"]):
            attrs = node.get("attributes", {})
            name = str(attrs.get("name", attrs.get("key", f"Object {index}")))
            item = QTreeWidgetItem([name, node["type"].split(":")[-1]])
            item.setData(0, Qt.ItemDataRole.UserRole, thaw_json(attrs))
            self.tree.addTopLevelItem(item)
        self.observables.setRowCount(len(OBSERVABLE_DEFINITIONS))
        reader = cp.observables
        for row, key in enumerate(OBSERVABLE_DEFINITIONS):
            record = reader.get(key)
            for column, value in enumerate((key, record.value, record.unit, record.status)):
                item = QTableWidgetItem("—" if value is None else str(value))
                item.setToolTip(f"{record.definition_id}\n{record.reason}\n{record.plane_id}")
                self.observables.setItem(row, column, item)
        if self.tree.topLevelItemCount():
            self.tree.setCurrentItem(self.tree.topLevelItem(0))

    def _component(self, item, previous=None):
        if item is None:
            return
        self._show_values((key, json.dumps(value, ensure_ascii=True)) for key, value in
                          item.data(0, Qt.ItemDataRole.UserRole).items())

    def _show_values(self, rows):
        rows = list(rows)
        self.values.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setToolTip(str(value))
                self.values.setItem(row, column, item)

    def _compare(self):
        if self.selected is None or self.current_snapshot is None:
            return
        try:
            changes = snapshot_changes(self.selected.snapshot, self.current_snapshot())
            self._show_values((path, json.dumps({"captured": old, "current": new})) for path, old, new in changes)
            self.status.setText(f"Selected checkpoint → current | {len(changes)} exact differences | no changes applied")
        except Exception as exc:
            self.error.emit(str(exc))

    def _restore(self, fork):
        if self.selected is not None:
            self.restore_requested.emit(self.selected, fork)

    def _import(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import working point", "", "Working point (*.temwp)")
        if path:
            try:
                self.add_checkpoint(WorkingPointCheckpoint.read_package(path), label="Imported")
            except Exception as exc:
                self.error.emit(str(exc))

    def _export(self):
        if self.selected is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export working point", "working-point.temwp", "Working point (*.temwp)")
        if path:
            try:
                self.selected.write_package(path, overwrite=True)  # Native dialog confirms an existing target.
            except Exception as exc:
                self.error.emit(str(exc))
