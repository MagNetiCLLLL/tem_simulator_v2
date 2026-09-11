"""Explicit, reversible selection of a versioned gun-owned source model."""
from dataclasses import asdict

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFormLayout, QLabel,
                               QLineEdit, QVBoxLayout)
from temsim.gui.input_policy import WheelSafeComboBox
from temsim.optics.electron_gun.effective_source import (
    EffectiveGunSource, bind_effective_source, generate_gun_emission)


class GunSourceDialog(QDialog):
    """Draft-only UI. No gun mutation until the owning page applies value()."""
    fields = (
        ("reference_current_a", "Exit reference current (A)", float),
        ("source_fwhm_nm", "Exit source FWHM (nm)", float),
        ("incoherent_angle_rms_mrad", "Extra incoherent angular RMS (mrad)", float),
        ("energy_fwhm_ev", "Exit energy FWHM (eV)", float),
        ("energy_nodes", "Energy nodes", int),
        ("dispersion_nm_per_ev", "Exit X dispersion (nm/eV)", float),
        ("angular_dispersion_mrad_per_ev", "Exit X angular dispersion (mrad/eV)", float),
        ("mode_tail_tolerance", "Omitted mode weight limit", float),
        ("maximum_modes", "Mode budget", int),
        ("grid_pixels", "Source grid pixels", int),
    )

    def __init__(self, gun, parent=None):
        super().__init__(parent)
        self._gun = gun
        self._value = None
        self.setWindowTitle("Electron-gun source model")
        self.resize(620, 520)
        layout = QVBoxLayout(self)
        self.representation = WheelSafeComboBox()
        self.representation.addItem("Legacy classical particles", "classical_particles")
        self.representation.addItem("Effective exit Gaussian-Schell v1", "effective_gaussian_schell")
        self.representation.setCurrentIndex(max(0, self.representation.findData(getattr(gun, "source_representation", "classical_particles"))))
        layout.addWidget(self.representation)
        note = QLabel(f"Reference: installed gun exit at Z = {gun.exit_plane_z_mm:g} mm. Legacy parameters are retained.")
        note.setWordWrap(True)
        note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        note.setToolTip("The new phenomenological source is not a resolved quantum gun or measured calibration. Its coherent modes and particle samples describe the same exit distribution. Extra angular RMS adds to diffraction, not a specimen convergence target. Apply explicitly binds these inputs to the current gun; a subsequent gun change requires a new explicit binding.")
        layout.addWidget(note)
        form = QFormLayout()
        existing = getattr(gun, "effective_source", None)
        # A required current has no inferred calibration or default value.
        defaults = asdict(existing) if existing else asdict(EffectiveGunSource(0.))
        self.inputs = {}
        for key, label, _ in self.fields:
            edit = QLineEdit(str(defaults[key]) if existing or key != "reference_current_a" else "")
            self.inputs[key] = edit
            form.addRow(label, edit)
        self.inputs["reference_current_a"].setPlaceholderText("Required: current at the gun exit")
        layout.addLayout(form)
        self.error = QLabel()
        self.error.setWordWrap(True)
        self.error.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.representation.currentIndexChanged.connect(self._update_enabled)
        self._update_enabled()

    def _update_enabled(self):
        for edit in self.inputs.values():
            edit.setEnabled(self.representation.currentData() == "effective_gaussian_schell")

    def accept(self):
        representation = self.representation.currentData()
        parameters = getattr(self._gun, "effective_source", None)
        if representation == "effective_gaussian_schell":
            try:
                parameters = bind_effective_source(self._gun, EffectiveGunSource(**{
                    key: parse(self.inputs[key].text()) for key, _, parse in self.fields}))
                generate_gun_emission(self._gun, parameters)  # Binding/budget, no wave allocation.
            except (TypeError, ValueError) as error:
                self.error.setText(str(error))
                return
        self._value = representation, parameters
        super().accept()

    def value(self):
        if self._value is None:
            raise ValueError("No electron-gun source selection was applied")
        return self._value
