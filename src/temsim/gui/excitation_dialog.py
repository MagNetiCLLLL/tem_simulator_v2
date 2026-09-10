"""Sourced electrical calibration and explicit current edits."""

from PySide6.QtWidgets import QDialog, QVBoxLayout, QFormLayout, QLabel, QLineEdit, QCheckBox, QDialogButtonBox
from temsim.gui.input_policy import WheelSafeComboBox as QComboBox
from temsim.excitation_calibration import ExcitationCalibration, SOURCE_KINDS


class ExcitationDialog(QDialog):
    def __init__(self, calibration, lens, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Coil calibration / current")
        self.resize(620, 480)
        self.calibration = self.control = None
        self._lens = lens
        layout = QVBoxLayout(self)
        note = QLabel("Percent describes an electrical control, not a universal B-field scale. Enter winding turns only from a stated source. Saturated magnetic fields are solved at the full operating point.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.turns = QLineEdit("" if calibration.turns is None else str(calibration.turns))
        self.turns.setPlaceholderText("Unknown — current unavailable")
        self.ni = QLineEdit(str(calibration.ampere_turns_at_reference))
        self.reference = QLineEdit(str(calibration.reference_excitation_percent))
        self.kind = QComboBox()
        self.kind.addItems(SOURCE_KINDS)
        self.kind.setCurrentText(calibration.source_kind)
        self.source = QLineEdit(calibration.source)
        self.date = QLineEdit(calibration.source_date or "")
        self.date.setPlaceholderText("YYYY-MM-DD, or unavailable")
        self.uncertainty = QLineEdit("" if calibration.relative_uncertainty is None else str(calibration.relative_uncertainty))
        self.uncertainty.setPlaceholderText("Relative fraction, or unavailable")
        self.valid_range = QLineEdit("" if calibration.valid_current_range_A is None else ", ".join(map(str, calibration.valid_current_range_A)))
        self.valid_range.setPlaceholderText("Minimum, maximum |I| in A, or unavailable")
        form = QFormLayout()
        for label, widget in (("Known winding turns N", self.turns), ("Reference ampere-turns NI", self.ni),
                              ("Reference excitation (%)", self.reference), ("Source category", self.kind),
                              ("Source reference / file", self.source), ("Source date", self.date),
                              ("Relative uncertainty", self.uncertainty), ("Validity range |I| (A)", self.valid_range)):
            form.addRow(label, widget)
        self.apply_current = QCheckBox("Also change the operating current")
        self.current = QLineEdit()
        self.current.setEnabled(False)
        existing = calibration.at_control(lens.percent, getattr(lens, "polarity", 1), enabled=lens.enabled)["current_A"]
        if existing is not None:
            self.current.setText(str(existing))
        self.current.setPlaceholderText("Signed current in A; known turns required")
        self.apply_current.toggled.connect(self.current.setEnabled)
        form.addRow(self.apply_current)
        form.addRow("New current (A)", self.current)
        layout.addLayout(form)
        self.error = QLabel()
        self.error.setStyleSheet("color:#ef7777")
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def accept(self):
        try:
            calibration = ExcitationCalibration(float(self.ni.text()), float(self.reference.text()),
                turns=int(self.turns.text()) if self.turns.text().strip() else None,
                source_kind=self.kind.currentText(), source=self.source.text(), source_date=self.date.text().strip() or None,
                relative_uncertainty=float(self.uncertainty.text()) if self.uncertainty.text().strip() else None,
                valid_current_range_A=tuple(float(v) for v in self.valid_range.text().split(",")) if self.valid_range.text().strip() else None).validate()
            control = (calibration.control_for_current(float(self.current.text()), maximum_percent=self._lens.max_percent,
                       zero_polarity=getattr(self._lens, "polarity", 1)) if self.apply_current.isChecked() else None)
        except (ValueError, TypeError) as exc:
            self.error.setText(str(exc))
            return
        self.calibration, self.control = calibration, control
        super().accept()
