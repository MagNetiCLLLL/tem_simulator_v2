"""Edit configured coefficients without evaluating a field fit on the UI thread."""

from copy import deepcopy
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLabel, QLineEdit, QTableWidget,
    QTableWidgetItem, QDialogButtonBox,
)
from temsim.optics.aberrations import EffectiveAberrationSet, SYSTEM_COEFFICIENT_ROWS
from temsim.optics.aberration_basis import WAVE_TERMS, UNIMPLEMENTED_TERMS
from temsim.optics.field_aberrations import fit_options


class AberrationSettingsDialog(QDialog):
    def __init__(self, options, *, system, state_step_mm, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{system.title()} aberration settings")
        self.resize(740, 650)
        self._options = deepcopy(options)
        self._step = state_step_mm
        self.result_options = None
        layout = QVBoxLayout(self)
        hint = QLabel("Check a row to override its default. Length coefficients are mm; azimuths are degrees in the laboratory X-Y frame. Field-derived mode retains these settings but uses fitted higher-order coefficients. C1 is an additional focus offset; transported focus is not added twice.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.table = QTableWidget(len(SYSTEM_COEFFICIENT_ROWS), 5)
        self.table.setHorizontalHeaderLabels(("Override", "Term", "Coefficient (mm)", "Azimuth (deg)", "Wave basis"))
        self.table.horizontalHeader().setStretchLastSection(True)
        for row, (name, field, angle_field) in enumerate(SYSTEM_COEFFICIENT_ROWS):
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            check.setCheckState(Qt.CheckState.Checked if field in options else Qt.CheckState.Unchecked)
            term = next((term for term in WAVE_TERMS if term.name == name), None)
            basis = (f"{term.polar_code}; theta^{term.power}; harmonic {term.harmonic}"
                     if term else "Cc × delta E / E")
            self.table.setItem(row, 0, check)
            for column, text in ((1, name), (2, str(options.get(field, 0.0))),
                                 (3, str(options.get(angle_field, 0.0)) if angle_field else "—"), (4, basis)):
                item = QTableWidgetItem(text)
                if column in (1, 4) or (column == 3 and angle_field is None):
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, column, item)
        for column, width in enumerate((72, 52, 125, 120)):
            self.table.setColumnWidth(column, width)
        layout.addWidget(self.table, 1)
        self.fit_angle = QLineEdit(str(options.get("fit_semiangle_mrad", 10)))
        self.fit_step = QLineEdit(str(options.get("fit_step_mm", .025)))
        self.energy_steps = QLineEdit(", ".join(str(v) for v in options.get("fit_energy_relative_steps", (.002, .001, .0005))))
        form = QFormLayout()
        form.addRow("Fit semi-angle (mrad)", self.fit_angle)
        form.addRow("Maximum ray step (mm)", self.fit_step)
        form.addRow("Cc relative energy steps", self.energy_steps)
        self.energy_steps.setToolTip("3–6 decreasing positive fractions, at fixed coil settings and geometry. 0.001 = 0.1%. No autofocus.")
        layout.addLayout(form)
        coverage = QLabel("Transfer: exp(-i chi), chi = 2 pi W / wavelength. A5 = C56 (six-fold astigmatism). Not implemented: " + ", ".join(UNIMPLEMENTED_TERMS) + ".")
        coverage.setWordWrap(True)
        layout.addWidget(coverage)
        self.error = QLabel()
        self.error.setWordWrap(True)
        self.error.setStyleSheet("color: #ef7777")
        layout.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def accept(self):
        try:
            options, coefficients = deepcopy(self._options), {}
            for row, (_, field, angle_field) in enumerate(SYSTEM_COEFFICIENT_ROWS):
                options.pop(field, None)
                if angle_field:
                    options.pop(angle_field, None)
                if self.table.item(row, 0).checkState() == Qt.CheckState.Checked:
                    coefficients[field] = float(self.table.item(row, 2).text())
                    if angle_field:
                        coefficients[angle_field] = float(self.table.item(row, 3).text())
            EffectiveAberrationSet("configured", "manual", **coefficients).validate()
            options.update(coefficients)
            options.update(fit_semiangle_mrad=float(self.fit_angle.text()), fit_step_mm=float(self.fit_step.text()),
                           fit_energy_relative_steps=[float(v.strip()) for v in self.energy_steps.text().split(",")])
            fit_options(options, self._step)
        except (ValueError, TypeError) as exc:
            self.error.setText(str(exc))
            return
        self.result_options = options
        super().accept()
