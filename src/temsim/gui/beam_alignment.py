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
        unavailable = QLabel("Twofold beam-shape alignment has separate controls below. Scan / descan calibration and physical observation planes are in Scanning Image.")
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


class StigmatorAlignmentEditor(QGroupBox):
    requested = Signal(str, float)

    def __init__(self, parent=None):
        super().__init__("Condenser twofold beam shape", parent)
        self._available = False
        form = QFormLayout(self)
        self.tolerance = BeamAlignmentEditor._positive(.01)
        self.tolerance.setMaximum(.99)
        self.bound = BeamAlignmentEditor._positive(100.)
        self.diameter = BeamAlignmentEditor._positive(10.)
        self.current = BeamAlignmentEditor._positive(2., allow_zero=True)
        self.samples = WheelSafeSpinBox()
        self.samples.setRange(3, 1000000)
        self.samples.setValue(16)
        self.trials = WheelSafeSpinBox()
        self.trials.setRange(8, 256)
        self.trials.setValue(24)
        form.addRow("Shape tolerance (dimensionless)", self.tolerance)
        form.addRow("Numerical strength bound (+/- %)", self.bound)
        form.addRow("Maximum D95 (nm)", self.diameter)
        form.addRow("Minimum current (pA)", self.current)
        form.addRow("Minimum effective samples", self.samples)
        form.addRow("Maximum search trials", self.trials)
        self.status = QLabel()
        self.status.setWordWrap(True)
        from PySide6.QtCore import Qt
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self.status)
        self.apply_button = QPushButton("Solve and apply twofold shape")
        self.apply_button.setObjectName("applyCondenserTwofoldShape")
        self.apply_button.clicked.connect(lambda: self.requested.emit("condenser_twofold_shape", 0.))
        form.addRow(self.apply_button)

    def set_state(self, state):
        from temsim.stigmator_alignment import capability
        self._available, reason = capability(state)
        self.status.setText(reason)
        self.apply_button.setEnabled(self._available)

    def set_busy(self, busy):
        self.apply_button.setEnabled(self._available and not busy)

    def options(self):
        from temsim.stigmator_alignment import StigmatorAlignmentOptions
        return StigmatorAlignmentOptions(self.tolerance.value(), self.bound.value(),
            self.diameter.value(), self.trials.value(), float(self.samples.value()), self.current.value())
