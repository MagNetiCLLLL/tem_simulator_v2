"""Identity-matched control rows aligned with a separate range table."""
from PySide6.QtCore import QEvent, QPoint, QTimer, Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QHeaderView, QHBoxLayout, QLabel, QLineEdit,
    QTableWidget, QTableWidgetItem, QWidget,
)


class AlignedControlTable(QTableWidget):
    def __init__(self, source, parent):
        super().__init__(0, 2, parent)
        self.source = source
        self._rows = {}
        self._active = set()
        self._aligning = False
        self._scrolling = False
        self.setObjectName("interactiveControlTable")
        self.setHorizontalHeaderLabels(["Unit", "Excitation / value"])
        self.verticalHeader().hide()
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        self.setColumnWidth(0, 136)
        self.setWordWrap(False)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._layout_timer = QTimer(self)
        self._layout_timer.setSingleShot(True)
        self._layout_timer.timeout.connect(self._align)
        for table in (source, self):
            table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
            table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            table.verticalHeader().sectionResized.connect(self.schedule_alignment)
            table.horizontalHeader().sectionResized.connect(self.schedule_alignment)
            table.verticalScrollBar().rangeChanged.connect(self.schedule_alignment)
            table.installEventFilter(self)
            table.viewport().installEventFilter(self)
        parent.installEventFilter(self)
        source.verticalScrollBar().valueChanged.connect(lambda v: self._scroll(self, v))
        self.verticalScrollBar().valueChanged.connect(lambda v: self._scroll(source, v))
        self.schedule_alignment()

    def _scroll(self, target, value):
        if self._scrolling:
            return
        self._scrolling = True
        try:
            target.verticalScrollBar().setValue(value)
        finally:
            self._scrolling = False

    def eventFilter(self, watched, event):
        if event.type() in (QEvent.Type.Resize, QEvent.Type.Move, QEvent.Type.Show,
                            QEvent.Type.FontChange, QEvent.Type.StyleChange):
            self.schedule_alignment()
        return super().eventFilter(watched, event)

    def schedule_alignment(self, *_):
        if not self._aligning and not self._layout_timer.isActive():
            self._layout_timer.start(0)

    def clear_controls(self):
        for row in range(self.rowCount()):
            widget = self.cellWidget(row, 1)
            if widget is not None:
                widget.hide()
        self.setRowCount(0)
        self._rows.clear()
        self._active.clear()

    def _ensure_row(self, control):
        key = control.identity
        if key not in self._rows:
            row = self.rowCount()
            self.insertRow(row)
            item = QTableWidgetItem(control.label.removesuffix(" / Excitation"))
            item.setData(Qt.ItemDataRole.UserRole, control)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.setItem(row, 0, item)
            self._rows[key] = item
        return self._rows[key].row()

    def add_control(self, control, widget):
        row = self._ensure_row(control)
        self._active.add(control.identity)
        old = self.cellWidget(row, 1)
        if old is not None:
            # Qt deletes replaced index widgets on the next event-loop turn.
            # Hide them immediately so no obsolete reference overlays a label.
            old.hide()
        self.setCellWidget(row, 1, widget)
        self.sync_draft()

    def sync_draft(self):
        draft = [self.source.item(row, 0).data(Qt.ItemDataRole.UserRole)
                 for row in range(self.source.rowCount()) if self.source.item(row, 0)]
        keys = [control.identity for control in draft]
        for key in list(self._rows):
            if key not in keys and key not in self._active:
                self.removeRow(self._rows.pop(key).row())
        for control in draft:
            row = self._ensure_row(control)
            if control.identity not in self._active:
                cell = self.cellWidget(row, 1)
                if cell is None:
                    value = QLineEdit()
                    value.setReadOnly(True)
                    if control.group == "detector":
                        cell = QWidget()
                        layout = QHBoxLayout(cell)
                        layout.setContentsMargins(0, 0, 4, 0)
                        layout.addWidget(value, 1)
                        hint = QLabel("Advanced bank")
                        hint.setObjectName("detectorReferenceHint")
                        hint.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse
                                                     | Qt.TextInteractionFlag.TextSelectableByKeyboard)
                        layout.addWidget(hint)
                    else:
                        cell = value
                    self.setCellWidget(row, 1, cell)
                value = cell if isinstance(cell, QLineEdit) else cell.findChild(QLineEdit)
                value.setText(f"{control.current:.9g} {control.unit}")
                cell.setToolTip(
                    "Reference at insertion. Use Advanced bank for detector readout; live lens tuning stays available."
                    if control.group == "detector" else
                    "Reference at insertion. Enter a range and start live tuning to enable the slider.")
        # Draft edits must not destroy controls belonging to a completed bank
        # or an existing live plan. Unmatched controls are explicitly separate.
        order = keys + [key for key in self._rows if key not in keys]
        header = self.verticalHeader()
        for visual, key in enumerate(order):
            item = self._rows[key]
            current = header.visualIndex(item.row())
            if current != visual:
                header.moveSection(current, visual)
            control = item.data(Qt.ItemDataRole.UserRole)
            label = control.label.removesuffix(" / Excitation")
            item.setText(label if key in keys else f"Active only: {label}")
            item.setToolTip(f"{control.label} ({control.unit})" + (
                "" if key in keys else "\nNot in the draft ranges; belongs to the active live plan or completed bank."))
        self.schedule_alignment()

    def control_row(self, identity):
        return self._rows[identity].row()

    def _align(self):
        if self._aligning:
            return
        self._aligning = True
        try:
            parent = self.parentWidget()
            top = parent.mapFromGlobal(self.source.mapToGlobal(QPoint(0, 0))).y()
            self.setGeometry(9, top, max(100, parent.width() - 18), self.source.height())
            header_height = max(self.source.horizontalHeader().sizeHint().height(),
                                self.horizontalHeader().sizeHint().height())
            for table in (self.source, self):
                table.horizontalHeader().setFixedHeight(header_height)
            minimum_field_width = max((self.cellWidget(row, 1).minimumSizeHint().width() + 8
                                       for row in range(self.rowCount()) if self.cellWidget(row, 1)), default=160)
            self.setColumnWidth(1, max(minimum_field_width, self.viewport().width() - self.columnWidth(0)))
            needs_scroll = any(table.horizontalHeader().length() > table.viewport().width()
                               for table in (self.source, self))
            policy = (Qt.ScrollBarPolicy.ScrollBarAlwaysOn if needs_scroll
                      else Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            for table in (self.source, self):
                table.setHorizontalScrollBarPolicy(policy)
            for source_row in range(self.source.rowCount()):
                item = self.source.item(source_row, 0)
                if item is None:
                    continue
                control = item.data(Qt.ItemDataRole.UserRole)
                if control.identity not in self._rows:
                    continue
                row = self.control_row(control.identity)
                widget = self.cellWidget(row, 1)
                height = max(38, self.fontMetrics().height() + 14,
                             widget.minimumSizeHint().height() + 8 if widget else 0,
                             widget.sizeHint().height() + 8 if widget else 0)
                self.source.setRowHeight(source_row, height)
                self.setRowHeight(row, height)
            self._scroll(self, self.source.verticalScrollBar().value())
        finally:
            self._aligning = False
