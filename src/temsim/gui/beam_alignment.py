"""Physical beam alignment targets and explicit unsupported observation tasks."""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QGroupBox, QFormLayout, QLabel, QPushButton
from temsim.gui.input_policy import WheelSafeDoubleSpinBox, WheelSafeSpinBox
from temsim.beam_alignment import BeamAlignmentOptions, KEY, capability


class BeamAlignmentEditor(QGroupBox):
    requested = Signal(str, float)

    def __init__(self, parent=None):
        super().__init__("Beam centre and direction", parent)
        self._available = False
        form = QFormLayout(self)
        text = QLabel("Solve the existing upper/lower X/Y deflector kicks together. Targets refer to the weighted beam at the specimen entrance; the tip, lenses, apertures and all other controls stay fixed.")
        text.setWordWrap(True)
        form.addRow(text)
        self.targets = []
        for label, suffix in (("Centre X", " um"), ("Centre Y", " um"), ("Mean direction X", " mrad"), ("Mean direction Y", " mrad")):
            editor = WheelSafeDoubleSpinBox()
            editor.setRange(-1.e6, 1.e6)
            editor.setDecimals(6)
            editor.setSuffix(suffix)
            editor.setKeyboardTracking(False)
            self.targets.append(editor)
            form.addRow(label, editor)
        self.position_tolerance = self._positive(.01)
        self.angle_tolerance = self._positive(.01)
        self.kick_bound = self._positive(10.)
        self.minimum_current = self._positive(0., allow_zero=True)
        self.minimum_samples = WheelSafeSpinBox()
        self.minimum_samples.setRange(2, 1000000)
        self.minimum_samples.setValue(16)
        self.trials = WheelSafeSpinBox()
        self.trials.setRange(10, 256)
        self.trials.setValue(32)
        form.addRow("Position tolerance (um)", self.position_tolerance)
        form.addRow("Direction tolerance (mrad)", self.angle_tolerance)
        form.addRow("Symmetric numerical kick bound (mrad)", self.kick_bound)
        form.addRow("Minimum current (pA)", self.minimum_current)
        form.addRow("Minimum effective samples", self.minimum_samples)
        form.addRow("Maximum search trials", self.trials)
        self.status = QLabel("Select an instrument")
        self.status.setWordWrap(True)
        form.addRow(self.status)
        self.apply_button = QPushButton("Solve and apply beam alignment")
        self.apply_button.setObjectName("applyBeamCentreDirection")
        self.apply_button.clicked.connect(lambda: self.requested.emit(KEY, 0.))
        form.addRow(self.apply_button)
        unavailable = QLabel("Two-axis condenser stigmator alignment: unavailable. The current field uses only (X strength - Y strength), giving one independent control direction.\nDynamic pivot / scan-descan matching: a time-dependent target and observation-plane contract must be specified before enabling an inverse task. Static centre/direction alignment does not certify those tasks.")
        unavailable.setWordWrap(True)
        form.addRow(unavailable)

    @staticmethod
    def _positive(value, allow_zero=False):
        editor = WheelSafeDoubleSpinBox()
        editor.setRange(0. if allow_zero else 1.e-6, 1.e6)
        editor.setDecimals(6)
        editor.setValue(value)
        editor.setKeyboardTracking(False)
        return editor

    def set_state(self, state):
        self._available, reason = capability(state)
        self.status.setText(reason)
        self.apply_button.setEnabled(self._available)

    def set_busy(self, busy):
        self.apply_button.setEnabled(self._available and not busy)

    def options(self):
        return BeamAlignmentOptions(tuple(editor.value() for editor in self.targets),
            self.position_tolerance.value(), self.angle_tolerance.value(), self.kick_bound.value(),
            self.trials.value(), float(self.minimum_samples.value()), self.minimum_current.value())
