"""Draft edits to the physical FEG tip; no independent exit-source controls."""
from copy import copy
import math

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QScrollArea, QVBoxLayout, QWidget
from temsim.optics.electron_gun.tip_coherence import TipCoherence, tip_covariance


class GunSourceDialog(QDialog):
    coherence_fields = (
        ("incoherent_angle_rms_mrad", "Incoherent angular RMS per axis (mrad)"),
        ("curvature_x_m1", "Wavefront curvature xx (1/m)"),
        ("curvature_xy_m1", "Wavefront curvature xy (1/m)"),
        ("curvature_y_m1", "Wavefront curvature yy (1/m)"),
        ("offset_x_nm", "Tip emission centre x (nm)"),
        ("offset_y_nm", "Tip emission centre y (nm)"),
        ("tilt_x_mrad", "Mean transverse momentum px/p (mrad)"),
        ("tilt_y_mrad", "Mean transverse momentum py/p (mrad)"),
    )
    fields = (
        ("emission_current_na", "Tip emission current (nA)"),
        ("emission_energy_ev", "Tip launch mean kinetic energy (eV)"),
        ("minimum_kinetic_energy_ev", "Minimum launch kinetic energy (eV)"),
        ("virtual_source_fwhm_nm", "Tip launch spatial FWHM (nm)"),
        ("angular_rms_mrad", "Tip angular RMS (mrad)"),
        ("angular_cutoff_mrad", "Tip angular cutoff (mrad)"),
        ("energy_spread_fwhm_ev", "Tip energy spread FWHM (eV)"),
        ("young_decay_width_ev", "Young energy decay width (eV)"),
        ("boersch_sigma_ev", "Boersch energy sigma (eV)"),
        ("energy_half_range_ev", "Energy sampling half-range (eV)"),
    )

    def __init__(self, gun, parent=None):
        super().__init__(parent)
        if gun.type_key != "cold_feg":
            raise ValueError("This editor controls FEG tip emission only")
        self._gun = gun
        self._value = None
        self.setWindowTitle("FEG tip emission")
        self.resize(620, 520)
        layout = QVBoxLayout(self)
        note = QLabel(
            "All inputs describe emission at the FEG tip. Electrons then pass through "
            "the installed extractor, gun lens, accelerator, deflectors and apertures. "
            "Downstream beam states are calculated from those components and may be cached."
        )
        note.setWordWrap(True)
        note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(note)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        panel = QWidget()
        form = QFormLayout(panel)
        scroll.setWidget(panel)
        self.inputs = {}
        for key, label in self.fields:
            edit = QLineEdit(str(getattr(gun.emitter, key)))
            self.inputs[key] = edit
            form.addRow(label, edit)
        self.coherence_enabled = QCheckBox("Use Gaussian-Schell emission at the physical tip")
        self.coherence_enabled.setChecked(gun.emitter.coherence is not None)
        form.addRow(self.coherence_enabled)
        description = QLabel(
            "This explicitly selects an untruncated quantum tip distribution for both waves and ray diagnostics. "
            "Spatial FWHM, current and energy inputs above still apply. Classical angular RMS/cutoff apply only "
            "when this option is off. Total angular spread includes diffraction, incoherent spread and curvature. "
            "All physical apertures remain active. Full TEM/STEM wave imaging is still under development.")
        description.setWordWrap(True)
        form.addRow(description)
        parameters = gun.emitter.coherence or TipCoherence()
        self.coherence_inputs = {}
        for key, label in self.coherence_fields:
            edit = QLineEdit(str(getattr(parameters, key)))
            self.coherence_inputs[key] = edit
            form.addRow(label, edit)
        self.coherence_enabled.toggled.connect(self._sync_fields)
        self._sync_fields()
        layout.addWidget(scroll)
        self.error = QLabel()
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _sync_fields(self):
        enabled = self.coherence_enabled.isChecked()
        for edit in self.coherence_inputs.values():
            edit.setEnabled(enabled)
        for key in ("angular_rms_mrad", "angular_cutoff_mrad"):
            self.inputs[key].setEnabled(not enabled)

    def accept(self):
        candidate = copy(self._gun.emitter)
        try:
            values = {key: float(self.inputs[key].text()) for key, _ in self.fields}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError("Tip emission values must be finite")
            for key, value in values.items():
                setattr(candidate, key, value)
            candidate.coherence = (TipCoherence(**{key: float(edit.text()) for key, edit in self.coherence_inputs.items()})
                                   if self.coherence_enabled.isChecked() else None)
            candidate.validate()
            if candidate.coherence is not None:
                tip_covariance(candidate, candidate.emission_energy_ev)
            values["coherence"] = candidate.coherence
        except (TypeError, ValueError) as error:
            self.error.setText(str(error))
            return
        self._value = values
        super().accept()

    def value(self):
        if self._value is None:
            raise ValueError("No tip emission values were applied")
        return dict(self._value)
