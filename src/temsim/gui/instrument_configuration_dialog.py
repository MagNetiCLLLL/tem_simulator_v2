"""Unit-based assembly draft with a linked, read-only physical review."""
from types import SimpleNamespace

from PySide6.QtCore import QSettings, QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QDialog, QHBoxLayout,
    QHeaderView, QLabel, QPushButton, QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget)

from temsim.gui.diagnostic_tabs import PhysicalLayoutView
from temsim.gui.input_policy import WheelSafeComboBox
from temsim.instrument_configuration import (InstrumentUnits,
    check_instrument_configuration, unit_for_component)


UNITS = (
    ("source", "Electron source"),
    ("monochromator", "Monochromator"),
    ("beam_blanker", "Electrostatic beam blanker"),
    ("c3_lens", "C3 lens"),
    ("probe_corrector", "Probe corrector"),
    ("image_corrector", "Image corrector"),
    ("energy_filter", "Energy filter"),
)


class InstrumentConfigurationDialog(QDialog):
    component_selected = Signal(str)

    def __init__(self, catalog, state_provider, on_assemble, parent=None, *, settings=None, layout_id="default"):
        super().__init__(parent)
        self.catalog, self.state_provider, self.on_assemble = catalog, state_provider, on_assemble
        self.settings = settings if settings is not None else QSettings()
        self.settings_key = f"instrument_configuration/v1/{layout_id}"
        self.checked = None
        self._updating = False
        self._assembly = None
        self.setObjectName("instrumentConfigurationDialog")
        self.setWindowTitle("Instrument configuration")
        self.resize(1240, 780)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        outer = QVBoxLayout(self)
        note = QLabel("Choose units from source to detector. Check the draft, then assemble.")
        note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        outer.addWidget(note)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(self.splitter, 1)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(len(UNITS), 2)
        self.table.setObjectName("instrumentUnitChoices")
        self.table.setHorizontalHeaderLabels(("Assembly unit", "Install / source"))
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.source = WheelSafeComboBox()
        self.source.addItem("Cold FEG", "cold_feg")
        self.source.addItem("Thermionic", "thermionic")
        self.options = {}
        for row, (key, label) in enumerate(UNITS):
            self.table.setItem(row, 0, QTableWidgetItem(label))
            self.table.setRowHeight(row, 44)
            if key == "source":
                control = self.source
                control.currentIndexChanged.connect(lambda *_: self._changed("source"))
            else:
                control = QCheckBox("Installed")
                self.options[key] = control
                control.toggled.connect(lambda _, k=key: self._changed(k))
            control.setObjectName(f"instrumentUnit_{key}")
            self.table.setCellWidget(row, 1, control)
        left_layout.addWidget(self.table, 1)
        fixed = QLabel("C1/C2, objective, sample stage and recording hardware remain part of the base assembly.")
        fixed.setWordWrap(True)
        fixed.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        left_layout.addWidget(fixed)
        self.unit_status = QLabel()
        self.unit_status.setWordWrap(True)
        self.unit_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        left_layout.addWidget(self.unit_status)
        self.splitter.addWidget(left)
        self.review = PhysicalLayoutView()
        self.review.setObjectName("configurationPhysicalReview")
        # Configuration chooses whole units, never edits a component definition.
        self.review.tabs.setTabVisible(1, False)
        self.review.edit_cell.hide()
        self.review.assembly_3d.edit_part.hide()
        self.review.plot.setToolTip("Select hardware to highlight its assembly unit. Geometry is read-only here.")
        self.review.component_selected.connect(self.select_component)
        self.review.component_activated.connect(lambda key, _: self.select_component(key))
        self.splitter.addWidget(self.review)
        self.splitter.setSizes([390, 850])
        self.preview_status = QLabel("Installed assembly · draft not checked")
        self.preview_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        outer.addWidget(self.preview_status)
        self.status = QLabel("Check validates assembly geometry and interfaces; existing operating values are retained where supported.")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        outer.addWidget(self.status)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.check_button = QPushButton("Check")
        self.assemble_button = QPushButton("Assemble")
        self.assemble_button.setEnabled(False)
        self.close_button = QPushButton("Close")
        for button in (self.check_button, self.assemble_button, self.close_button):
            buttons.addWidget(button)
        outer.addLayout(buttons)
        self.check_button.clicked.connect(self.check)
        self.assemble_button.clicked.connect(self.assemble)
        self.close_button.clicked.connect(self.reject)
        self.table.currentCellChanged.connect(lambda *_: self._select_unit())
        state = state_provider()
        units = InstrumentUnits.from_selection(catalog, catalog.selection_for_resolved(state._resolved_assembly))
        self._updating = True
        self.source.setCurrentIndex(self.source.findData(units.source))
        for key, control in self.options.items():
            control.setChecked(getattr(units, key))
        self._updating = False
        self._dependencies()
        self._show_preview(state)
        geometry = self.settings.value(self.settings_key + "/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        split = self.settings.value(self.settings_key + "/splitter")
        if split is not None:
            self.splitter.restoreState(split)

    def units(self):
        return InstrumentUnits(source=self.source.currentData(),
            **{key: control.isChecked() for key, control in self.options.items()})

    def _dependencies(self):
        required = {
            "monochromator": (self.source.currentData() == "cold_feg", "Requires Cold FEG"),
            "probe_corrector": (self.options["c3_lens"].isChecked(), "Requires C3 lens"),
            "image_corrector": (self.options["c3_lens"].isChecked(), "Requires C3 lens"),
        }
        for key, (enabled, reason) in required.items():
            control = self.options[key]
            with QSignalBlocker(control):
                if not enabled:
                    control.setChecked(False)
                control.setEnabled(enabled)
                control.setText("Installed" if enabled else reason)
                control.setToolTip(reason)

    def _changed(self, key):
        if self._updating:
            return
        self._dependencies()
        self.checked = None
        self.assemble_button.setEnabled(False)
        self.preview_status.setText("Previous assembly view · draft changed; Check to update")
        self.status.setText("Draft changed. Check before assembling.")
        self.table.setCurrentCell(dict((k, i) for i, (k, _) in enumerate(UNITS))[key], 0)

    def _show_preview(self, state):
        self._assembly = state._resolved_assembly
        self.review.display_result(SimpleNamespace(assembly=self._assembly,
            layout=state._resolved_optics_layout, state_snapshot=state))
        self.review.assembly_3d.set_assembly(self._assembly)
        self._select_unit()

    def _select_unit(self):
        row = self.table.currentRow()
        if self._assembly is None or row < 0:
            return
        unit, label = UNITS[row]
        members = [p for p in self._assembly.parts if unit_for_component(self._assembly, p.key) == unit]
        if not members:
            self.unit_status.setText(f"{label}: not in the displayed assembly")
            return
        axial = [p for p in members if not p.data.get("branch_path_only")]
        self.unit_status.setText(f"{label}: {len(members)} components" +
            (f" · Z {min(p.start_z_mm for p in axial):.6g}–{max(p.end_z_mm for p in axial):.6g} mm" if axial else ""))
        part = next((p for p in members if p.key in {"feg_tip", "thermionic_cathode", "condenser_lens_3", "energy_filter"}), members[0])
        self.review.reveal_component(part)
        self.component_selected.emit(part.key)

    def select_component(self, key):
        if self._assembly is None:
            return
        unit = unit_for_component(self._assembly, key)
        if unit is None:
            self.unit_status.setText("Base assembly component · always installed")
            return
        row = next(i for i, (name, _) in enumerate(UNITS) if name == unit)
        self.table.setCurrentCell(row, 0)

    def check(self):
        self.checked = None
        self.assemble_button.setEnabled(False)
        try:
            checked = check_instrument_configuration(self.state_provider(), self.catalog, self.units())
            self._show_preview(checked.candidate.restore())
            self.checked = checked
            self.preview_status.setText("Checked draft · current instrument unchanged")
            self.status.setText("Check passed. Assemble applies this configuration; lens preset calibration is separate.")
            self.assemble_button.setEnabled(True)
        except Exception as exc:
            self.status.setText(f"Check failed: {exc}")

    def assemble(self):
        if self.checked is None or self.checked.units != self.units():
            self.status.setText("Check this draft before assembling.")
            return
        try:
            self.on_assemble(self.checked)
        except Exception as exc:
            self.checked = None
            self.assemble_button.setEnabled(False)
            self.status.setText(f"Assembly not applied: {exc}")
            return
        self.accept()

    def done(self, result):
        self.settings.setValue(self.settings_key + "/geometry", self.saveGeometry())
        self.settings.setValue(self.settings_key + "/splitter", self.splitter.saveState())
        super().done(result)
