"""Explicit detached geometry requests; selecting a candidate edits no state."""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QFormLayout, QComboBox, QLineEdit, QLabel, QPushButton, QScrollArea
from temsim.gui.alignment_constraints import AlignmentConstraintsEditor
from temsim.geometry_experiments import plan_geometry_sweep
from temsim.operating_modes import direct_alignment_by_key


class GeometryExperimentEditor(QScrollArea):
    requested = Signal(object, object, object)
    error = Signal(str)

    def __init__(self, recipe_for_slot, parent=None):
        super().__init__(parent)
        self.recipe_for_slot = recipe_for_slot
        self.setWidgetResizable(True)
        body = QWidget()
        self.setWidget(body)
        form = QFormLayout(body)
        info = QLabel("Candidates use the captured assembly and the same validated part editor. Live definitions are unchanged. Analytical fields remain analytical; a field map must match the candidate geometry. Failed dimensions, clearances and alignment remain visible in the results.")
        info.setWordWrap(True)
        form.addRow(info)
        self.slot = QComboBox()
        self.slot.addItems(("A", "B"))
        form.addRow("Captured inputs", self.slot)
        self.module = QComboBox()
        for title, key in (("Column", "column"), ("Gun", "gun"), ("Recording", "recording"), ("Beam blanker", "beam_blanker")):
            self.module.addItem(title, key)
        form.addRow("Selected module", self.module)
        self.part = QLineEdit("condenser_aperture_2")
        self.part.setToolTip("Existing component key in the selected module; use the Model Inspector to inspect part dimensions")
        form.addRow("Component key", self.part)
        self.dimension = QComboBox()
        self.dimension.setEditable(True)
        self.dimension.addItems(("vacuum_inner_diameter_mm", "length_mm", "mechanical_bore_diameter_mm", "tip_radius_nm", "tip_cone_half_angle_deg"))
        form.addRow("Existing dimension", self.dimension)
        self.values = QLineEdit("4.0, 5.0, 6.0")
        form.addRow("Values (dimension units)", self.values)
        self.mode = QComboBox()
        self.mode.addItems(("Fixed-control comparison", "Optimize selected controls for each candidate"))
        form.addRow("Operating controls", self.mode)
        self.optimization = QWidget()
        options_form = QFormLayout(self.optimization)
        self.target_key = QComboBox()
        self.target_key.addItem("Microprobe illumination diameter (µm)", "microprobe_illumination")
        self.target_key.addItem("Nanoprobe convergence (mrad)", "nanoprobe_convergence")
        options_form.addRow("Registered target", self.target_key)
        self.target = QLineEdit("1.0")
        options_form.addRow("Requested value", self.target)
        self.constraints = AlignmentConstraintsEditor(direct_alignment_by_key("microprobe_illumination").devices)
        self.constraints.enabled.setChecked(True)
        self.constraints.enabled.hide()
        options_form.addRow(self.constraints)
        self.optimization.hide()
        self.mode.currentIndexChanged.connect(lambda index: self.optimization.setVisible(index == 1))
        form.addRow(self.optimization)
        self.run_button = QPushButton("Run detached geometry experiment")
        self.run_button.clicked.connect(self.request)
        form.addRow(self.run_button)

    def request(self):
        try:
            recipe = self.recipe_for_slot(self.slot.currentText())
            optimization = None
            if self.mode.currentIndex() == 1:
                optimization = dict(key=self.target_key.currentData(), target=float(self.target.text()),
                    options=self.constraints.options().to_dict())
            sweep = plan_geometry_sweep(recipe, self.module.currentData(), self.part.text().strip(),
                self.dimension.currentText().strip(), [float(v.strip()) for v in self.values.text().split(",")],
                optimization=optimization)
        except (KeyError, TypeError, ValueError) as exc:
            self.error.emit(str(exc))
            return
        self.requested.emit(recipe, sweep, ())
