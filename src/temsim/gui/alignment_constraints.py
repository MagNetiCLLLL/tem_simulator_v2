"""Request-owned optional constraints, never independent beam-state inputs."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)
from temsim.gui.input_policy import WheelSafeSpinBox
from temsim.alignment_constraints import BeamConstraint, ConstrainedAlignment, METRICS


class AlignmentConstraintsEditor(QWidget):
    def __init__(self, devices, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.enabled = QCheckBox("Use joint constraints at the specimen entrance")
        self.enabled.setObjectName("useJointAlignmentConstraints")
        layout.addWidget(self.enabled)
        self.body = QWidget()
        form = QFormLayout(self.body)
        notice = QLabel("Only checked lens excitations may change. All other inputs, including the physical tip, are fixed. Bounds below are numerical search intervals, not hardware ratings.")
        notice.setWordWrap(True)
        form.addRow(notice)
        self.controls = QTableWidget(len(devices), 3)
        self.controls.setHorizontalHeaderLabels(("Allowed control", "Lower (%)", "Upper (%)"))
        self.controls.horizontalHeader().setStretchLastSection(True)
        self.controls.setMaximumHeight(130)
        for row, key in enumerate(devices):
            item = QTableWidgetItem(key)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.controls.setItem(row, 0, item)
            self.controls.setItem(row, 1, QTableWidgetItem("0"))
            self.controls.setItem(row, 2, QTableWidgetItem("100"))
        form.addRow(self.controls)
        self.constraints = QTableWidget(0, 5)
        self.constraints.setHorizontalHeaderLabels(("Observable / unit", "Minimum", "Maximum", "Scale", "Weight"))
        self.constraints.horizontalHeader().setStretchLastSection(True)
        self.constraints.setMaximumHeight(180)
        form.addRow(self.constraints)
        add = QPushButton("Add constraint")
        remove = QPushButton("Remove selected constraint")
        add.clicked.connect(self.add_constraint)
        remove.clicked.connect(lambda: self.constraints.removeRow(self.constraints.currentRow()))
        form.addRow(add, remove)
        self.trials = WheelSafeSpinBox()
        self.trials.setRange(4, 256)
        self.trials.setValue(32)
        form.addRow("Maximum search trials", self.trials)
        self.support = WheelSafeSpinBox()
        self.support.setRange(2, 1000000)
        self.support.setValue(16)
        form.addRow("Minimum effective samples", self.support)
        self.topology = QCheckBox("Require the scoped crossover sequence")
        form.addRow(self.topology)
        details = QLabel("Each trial executes tip-to-entrance transport. Final checks independently halve gun and column steps. Sampling and field-mesh qualification remain separate. A low effective-sample threshold does not establish convergence.")
        details.setWordWrap(True)
        form.addRow(details)
        layout.addWidget(self.body)
        self.body.setVisible(False)
        self.enabled.toggled.connect(self.body.setVisible)

    def add_constraint(self, checked=False):
        row = self.constraints.rowCount()
        self.constraints.insertRow(row)
        choice = QComboBox()
        for key, unit in METRICS.items():
            choice.addItem(f"{key} ({unit})", key)
        self.constraints.setCellWidget(row, 0, choice)
        for column, value in enumerate(("", "", "1", "1"), 1):
            self.constraints.setItem(row, column, QTableWidgetItem(value))

    def options(self):
        if not self.enabled.isChecked():
            return None
        bounds = {self.controls.item(r, 0).text():
                  (float(self.controls.item(r, 1).text()), float(self.controls.item(r, 2).text()))
                  for r in range(self.controls.rowCount())
                  if self.controls.item(r, 0).checkState() == Qt.CheckState.Checked}
        constraints = []
        for row in range(self.constraints.rowCount()):
            values = [self.constraints.item(row, col).text().strip() for col in range(1, 5)]
            constraints.append(BeamConstraint(self.constraints.cellWidget(row, 0).currentData(),
                float(values[0]) if values[0] else None, float(values[1]) if values[1] else None,
                float(values[2]), float(values[3])))
        return ConstrainedAlignment(tuple(constraints), bounds, self.trials.value(), float(self.support.value()), self.topology.isChecked())
