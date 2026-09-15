"""Central calculation readouts; physical hardware remains independently owned."""
from copy import deepcopy
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QGroupBox,
    QCheckBox, QLabel, QDialogButtonBox)


class CalculateSetupDialog(QDialog):
    changed = Signal(str)

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.setWindowTitle("Calculate setup")
        self.setObjectName("calculateSetupDialog")
        self.resize(660, 590)
        layout = QVBoxLayout(self)
        note = QLabel("Choose outputs for the next High accuracy calculation. Preview traces particles only. Settings are saved with the operating profile.")
        note.setWordWrap(True)
        layout.addWidget(note)
        group = QGroupBox("Particle calculation")
        form = QFormLayout(group)
        self.controls = {}
        for key, label, obj, attr in (
            ("stem", "Generate STEM detector images (AC raster must be enabled)", state.sample, "stem_image_enabled"),
            ("eds", "Calculate EDS spectrum", state.sample, "eds_enabled"),
            ("vacuum", "Include vacuum / cell transport", state.vacuum_map, "enabled"),
            ("stem_poisson", "Generate seeded Poisson STEM counts", state.sample, "stem_poisson_enabled"),
            ("eds_poisson", "Generate seeded Poisson EDS counts", state.sample, "eds_poisson_enabled"),
        ):
            check = QCheckBox(label)
            check.setObjectName("calculateSetup_"+key)
            check.setChecked(bool(getattr(obj, attr)))
            if key == "vacuum":
                check.setToolTip("Off by default. Choose before the first Preview. Changing this option or active vacuum settings invalidates cached calculation results.")
            self.controls[key] = (check, attr)
            form.addRow(check)
        layout.addWidget(group)
        waves = QGroupBox("Coherent images — development paused")
        wave_form = QFormLayout(waves)
        for key, label, attr in (("tem", "Generate TEM wave image", "wave_enabled"),
                                  ("stem_wave", "Use coherent STEM wave imaging", "stem_wave_enabled"),
                                  ("fourdstem", "Record 4D-STEM diffraction cube", "stem_fourdstem_enabled")):
            check = QCheckBox(label)
            checked = bool(getattr(state.sample, attr))
            check.setChecked(checked)
            # An explicit historical request may be turned off, never silently
            # changed merely by opening the dialog or loading its profile.
            check.setEnabled(checked)
            check.toggled.connect(lambda on, control=check: control.setEnabled(on))
            self.controls[key] = (check, attr)
            wave_form.addRow(check)
        note = QLabel("The current FEG tip uses classical particle emission. New coherent TEM/STEM images are unavailable until the tip-to-column phase chain is qualified. Historical results remain viewable; loaded wave requests can be turned off here.")
        note.setWordWrap(True)
        note.setMinimumHeight(65)
        wave_form.addRow(note)
        layout.addWidget(waves)
        note = QLabel("Extraction, acceleration, lenses, apertures and detector absorption remain active. Disabling an image or spectrum does not retract its detector. Poisson counts change the readout statistics, not electron trajectories.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Close)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def apply(self):
        from temsim.vacuum import resolve_regions
        from copy import copy
        candidate = copy(self.state)
        candidate.vacuum_map = deepcopy(self.state.vacuum_map)
        candidate.vacuum_map.enabled = self.controls["vacuum"][0].isChecked()
        try:
            resolve_regions(candidate)
        except ValueError as exc:
            self.status.setText(f"Not applied: {exc}. Load a vacuum map in the Vacuum map tab.")
            return False
        self.state.vacuum_map = candidate.vacuum_map
        for key, (control, attr) in self.controls.items():
            if key != "vacuum":
                setattr(self.state.sample, attr, control.isChecked())
        self.changed.emit("calculate_setup")
        self.status.setText("Applied. Run High accuracy to generate the selected outputs.")
        return True
