"""Create or edit one parametric material-removal feature."""

from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QLabel, QVBoxLayout,
)

from temsim.gui.input_policy import WheelSafeComboBox


class PartFeatureDialog(QDialog):
    def __init__(self, feature, parent=None):
        super().__init__(parent)
        self._feature = dict(feature)
        self.setWindowTitle("Edit hole" if feature["kind"] == "hole" else "Edit slot / groove")
        layout = QVBoxLayout(self)
        hint = QLabel("Position is relative to the component centre, before its model transform. "
                      "Depth is centred on the chosen position; extend it through the body for a through hole.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        form = QFormLayout()
        self.axis = WheelSafeComboBox()
        for name in ("x", "y", "z"):
            self.axis.addItem(name.upper(), name)
        self.axis.setCurrentIndex(max(0, self.axis.findData(feature.get("axis", "z"))))
        form.addRow("Cut axis", self.axis)
        self.values = {}
        for index, name in enumerate(("X position", "Y position", "Z position")):
            spin = self._number(feature.get("center_mm", [0, 0, 0])[index])
            self.values[("center_mm", index)] = spin
            form.addRow(name + " (mm)", spin)
        fields = [("diameter_mm", "Diameter")] if feature["kind"] == "hole" else [
            ("width_mm", "Slot width"), ("length_mm", "Slot length")]
        fields.append(("depth_mm", "Cut depth"))
        for key, label in fields:
            spin = self._number(feature[key], positive=True)
            self.values[key] = spin
            form.addRow(label + " (mm)", spin)
        if feature["kind"] == "slot":
            self.values["rotation_deg"] = self._number(feature.get("rotation_deg", 0))
            form.addRow("Rotation around cut axis (°)", self.values["rotation_deg"])
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(400, 330)

    def _number(self, value, *, positive=False):
        spin = QDoubleSpinBox()
        spin.setDecimals(6)
        spin.setRange(0.000001 if positive else -1e6, 1e6)
        spin.setValue(float(value))
        spin.setKeyboardTracking(False)
        return spin

    def feature(self):
        result = dict(self._feature)
        result["axis"] = self.axis.currentData()
        result["center_mm"] = [self.values[("center_mm", index)].value() for index in range(3)]
        for key, spin in self.values.items():
            if isinstance(key, str):
                result[key] = spin.value()
        return result
