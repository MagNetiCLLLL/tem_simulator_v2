"""Editable, validated specimen-entrance pupil and independent source modes."""
import json
import math

from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QPlainTextEdit, QGroupBox,
)
from temsim.gui.input_policy import WheelSafeComboBox as QComboBox, WheelSafeDoubleSpinBox as QDoubleSpinBox, WheelSafeSpinBox as QSpinBox
from temsim.physics.illumination import (
    LEGACY_MODEL, EXPLICIT_MODEL, default_illumination_config,
    validate_illumination_config, gaussian_quadrature, source_nodes,
)


class IlluminationDialog(QDialog):
    def __init__(self, config, energy_kev, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Illumination pupil and source modes")
        self.resize(780, 760)
        self.energy_kev = energy_kev
        self.config = config
        layout = QVBoxLayout(self)
        self.model = QComboBox()
        self.model.addItem("Ray-conditioned reduced-order (existing model)", LEGACY_MODEL)
        self.model.addItem("Declared specimen-entrance pupil and incoherent modes", EXPLICIT_MODEL)
        self.model.setCurrentIndex(max(0, self.model.findData(config.get("model", LEGACY_MODEL))))
        layout.addWidget(self.model)
        scope = QLabel("Explicit pupil: TEM and elastic STEM. Positions are at the specimen plane (nm); directions are in mrad. "
                       "Modes add as intensities, with independent phonon averaging. Energy nodes update wavelength and column transport at fixed hardware settings. "
                       "Ray-based Rutherford tails and multi-mode bulk absorption / 4D-STEM capture require the existing model.")
        scope.setWordWrap(True)
        layout.addWidget(scope)
        generator = QGroupBox("Build an analytic pupil and Gaussian quadratures")
        form = QFormLayout(generator)
        self.shape = QComboBox()
        self.shape.addItems(["ellipse", "rectangle"])
        form.addRow("Shape", self.shape)
        self.controls = {}
        for label, keys, values, suffix, low, high in (
            ("Semi axes X / Y", ("ax", "ay"), (20., 20.), " mrad", .001, 199.),
            ("Pupil offset X / Y", ("ox", "oy"), (0., 0.), " mrad", -199., 199.),
            ("Rotation", ("rotation",), (0.,), "°", -180., 180.),
            ("Source position RMS X / Y", ("sx", "sy"), (0., 0.), " nm", 0., 1000.),
            ("Source angle RMS X / Y", ("tx", "ty"), (0., 0.), " mrad", 0., 100.),
            ("Energy RMS", ("energy",), (0.,), " eV", 0., 10000.),
        ):
            row = QHBoxLayout()
            for key, value in zip(keys, values):
                control = QDoubleSpinBox()
                control.setRange(low, high)
                control.setDecimals(6)
                control.setSuffix(suffix)
                control.setValue(value)
                self.controls[key] = control
                row.addWidget(control)
            form.addRow(label, row)
        self.orders = {}
        pupil = config.get("pupil", {})
        if pupil.get("shape", "ellipse") in {"ellipse", "rectangle"}:
            self.shape.setCurrentText(pupil.get("shape", "ellipse"))
            for keys, values in ((("ax", "ay"), pupil.get("semi_axes_mrad", (20., 20.))),
                                 (("ox", "oy"), pupil.get("offset_mrad", (0., 0.)))):
                for key, value in zip(keys, values):
                    self.controls[key].setValue(float(value))
            basis = pupil.get("basis", ((1., 0.), (0., 1.)))
            self.controls["rotation"].setValue(math.degrees(math.atan2(basis[1][0], basis[0][0])))
        row = QHBoxLayout()
        for key in ("position", "angle", "energy"):
            row.addWidget(QLabel(key))
            control = QSpinBox()
            control.setRange(2, 15)
            control.setValue(3)
            self.orders[key] = control
            row.addWidget(control)
        form.addRow("Nodes per nonzero axis", row)
        build = QPushButton("Replace configuration below with these values")
        build.clicked.connect(self.generate)
        form.addRow(build)
        layout.addWidget(generator)
        self.editor = QPlainTextEdit()
        self.editor.setObjectName("illuminationConfigJson")
        self.editor.setPlainText(json.dumps(config if config.get("model") == EXPLICIT_MODEL else default_illumination_config(), indent=2))
        layout.addWidget(QLabel("Configuration JSON — manual weighted modes, affine bases and sampled amplitude/intensity pupils are supported"))
        layout.addWidget(self.editor, 1)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.model.currentIndexChanged.connect(self._validate_preview)
        self.editor.textChanged.connect(self._validate_preview)
        self._validate_preview()

    def generate(self):
        try:
            c = {k: v.value() for k, v in self.controls.items()}
            angle = math.radians(c["rotation"])
            config = default_illumination_config()
            config["pupil"].update(shape=self.shape.currentText(), semi_axes_mrad=[c["ax"], c["ay"]],
                                   offset_mrad=[c["ox"], c["oy"]],
                                   basis=[[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
            config["positions_nm"] = gaussian_quadrature([c["sx"], c["sy"]], self.orders["position"].value())
            config["angles_mrad"] = gaussian_quadrature([c["tx"], c["ty"]], self.orders["angle"].value())
            config["energies_ev"] = gaussian_quadrature([c["energy"]], self.orders["energy"].value())
            config = validate_illumination_config(config, self.energy_kev)
            self.model.setCurrentIndex(1)
            self.editor.setPlainText(json.dumps(config, indent=2))
        except ValueError as exc:
            self.status.setText(str(exc))

    def value(self):
        config = {"model": LEGACY_MODEL} if self.model.currentData() == LEGACY_MODEL else json.loads(self.editor.toPlainText())
        if self.model.currentData() == EXPLICIT_MODEL and config.get("model") != EXPLICIT_MODEL:
            raise ValueError("Configuration model must match the selected explicit model")
        return validate_illumination_config(config, self.energy_kev)

    def _validate_preview(self):
        explicit = self.model.currentData() == EXPLICIT_MODEL
        self.editor.setEnabled(explicit)
        try:
            config = self.value()
            count = len(source_nodes(config))
            self.status.setText(f"{count} source mode(s) × frozen-phonon configurations. Source/energy convergence has not been assessed. "
                               "Unknown physical aperture edge stays unknown in the reduced-order model.")
            self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
        except (ValueError, TypeError) as exc:
            self.status.setText(str(exc))
            self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)

    def accept(self):
        try:
            self.config = self.value()
        except (ValueError, TypeError) as exc:
            self.status.setText(str(exc))
            return
        super().accept()
