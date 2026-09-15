"""Beam-path vacuum map and finite insertable cell editor."""
from copy import copy, deepcopy
import json
from dataclasses import asdict

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QSplitter,
    QLabel, QPushButton, QCheckBox, QComboBox, QLineEdit, QDoubleSpinBox,
    QSpinBox, QScrollArea, QFileDialog)

from temsim.vacuum import VacuumMap, Medium, boundary_anchors, resolve_regions, module_axial_ranges, DEFAULT_PATH
from temsim.gui.vacuum_axial_view import VacuumAxialView
from temsim.gui.cell_environment_editor import WindowEditor, CellChamberView


class VacuumMapPage(QWidget):
    changed = Signal(str)
    CORE_KEYS = {"gun_tip", "gun_accelerator", "column", "specimen", "post_column", "projection"}

    def __init__(self):
        super().__init__()
        self.state = None
        self.current_key = ""
        self._filling = False
        outer = QVBoxLayout(self)
        outer.setContentsMargins(11, 9, 11, 9)
        toolbar = QHBoxLayout()
        self.enabled = QCheckBox("Include vacuum / cell transport in calculations")
        self.enabled.setToolTip("Off by default. Choose before the first Preview. Changing this option or active vacuum settings invalidates cached calculation results.")
        toolbar.addWidget(self.enabled)
        for title, method in (("Open map…", self.open_map), ("Save map as…", self.save_map),
                              ("Load normal-operation defaults", self.load_defaults)):
            button = QPushButton(title)
            button.clicked.connect(method)
            toolbar.addWidget(button)
        toolbar.addStretch()
        outer.addLayout(toolbar)
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.diagram = VacuumAxialView()
        splitter.addWidget(self.diagram)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        form_widget = QWidget()
        columns = QHBoxLayout(form_widget)
        geometry_widget, medium_widget = QWidget(), QWidget()
        columns.addWidget(geometry_widget, 1)
        columns.addWidget(medium_widget, 1)
        form = QFormLayout(geometry_widget)
        self.form_layout = form
        self.selection = QComboBox()
        toolbar.insertWidget(0, self.selection)
        self.selection.setMinimumWidth(180)
        self.selection.setMaximumWidth(280)
        self.name = QLineEdit()
        form.addRow("Name", self.name)
        self.bounds = QLabel()
        self.bounds.setWordWrap(True)
        form.addRow("Resolved position", self.bounds)
        self.modules_text = QLabel()
        self.modules_text.setWordWrap(True)
        form.addRow("Assembly modules", self.modules_text)
        self.position_mode = QComboBox()
        self.position_mode.addItem("Component / module anchors", "anchors")
        self.position_mode.addItem("Z positions", "positions")
        form.addRow("Define range using", self.position_mode)
        self.coordinate_frame = QComboBox()
        form.addRow("Z coordinates", self.coordinate_frame)
        self.start_z, self.end_z = self._spin(), self._spin()
        form.addRow("Start Z (mm)", self.start_z)
        form.addRow("End Z (mm)", self.end_z)
        self._coordinate_origin = 0.0
        self.start_anchor, self.end_anchor = QComboBox(), QComboBox()
        self.start_offset, self.end_offset = self._spin(), self._spin()
        for label, widget in (("Start boundary", self.start_anchor), ("Start offset (mm)", self.start_offset),
                              ("End boundary", self.end_anchor), ("End offset (mm)", self.end_offset)):
            form.addRow(label, widget)
        self.cell_inserted = QCheckBox("Insert cell around specimen")
        form.addRow(self.cell_inserted)
        self.cell_fields = {}
        for name, title in (("diameter_mm", "Cell aperture diameter (nm)"), ("length_mm", "Cell gap (nm)"),
                            ("offset_x_mm", "Cell centre X (nm)"), ("offset_y_mm", "Cell centre Y (nm)"),
                            ("offset_z_mm", "Cell Z offset from Sample (nm)")):
            widget = self._spin(0 if name in {"diameter_mm", "length_mm"} else -1e12, 1e12)
            widget.setDecimals(6)
            widget.setSingleStep(1)
            self.cell_fields[name] = widget
            form.addRow(title, widget)
        self.cell_fields["length_mm"].setToolTip("Distance between window inner faces. Window thickness is additional. Sample geometry remains defined in Sample.")
        self.windows = QWidget()
        window_layout = QHBoxLayout(self.windows)
        window_layout.setContentsMargins(0, 0, 0, 0)
        self.window_editors = {}
        for key, label in (("upstream_window", "Upstream window (−Z)"), ("downstream_window", "Downstream window (+Z)")):
            editor = WindowEditor(label)
            self.window_editors[key] = editor
            window_layout.addWidget(editor)
        form.addRow(self.windows)
        form = QFormLayout(medium_widget)
        self.phase = QComboBox()
        for label, value in (("Gas / residual gas", "gas"), ("Liquid — elastic approximation", "liquid"), ("Ideal vacuum", "vacuum")):
            self.phase.addItem(label, value)
        form.addRow("Medium", self.phase)
        self.formula = QLineEdit()
        self.formula.setPlaceholderText("N2, H2, He, Ar, H2O …")
        form.addRow("Chemical formula", self.formula)
        self.mixture = QLineEdit()
        self.mixture.setPlaceholderText('Optional mole fractions: {"Ar": 0.9, "H2": 0.1}')
        self.mixture.setToolTip("JSON chemical-formula/mole-fraction pairs; fractions must sum to 1. Empty uses the single formula above. Liquid additionally needs measured mass density.")
        form.addRow("Mixture (optional)", self.mixture)
        self.pressure = QLineEdit()
        self.pressure.setToolTip("Gas: total absolute pressure determines number density. Liquid: recorded pressure only; density is a separate measured input. No equation of state or window bulging.")
        form.addRow("Pressure (mbar)", self.pressure)
        self.temperature = self._spin(0.001, 100000)
        form.addRow("Temperature (K)", self.temperature)
        self.density = self._spin(.000001, 1e6)
        form.addRow("Liquid mass density (kg/m³)", self.density)
        self.removal = QLineEdit()
        self.removal.setToolTip("Independent removal cross section per molecule. Leave zero for elastic-only transport; do not enter total elastic scattering here.")
        form.addRow("Removal cross section (m²/molecule)", self.removal)
        self.removal_reference = QLineEdit()
        form.addRow("Removal model / measurement", self.removal_reference)
        self.seed = QLineEdit()
        self.seed.setToolTip("Integer from 0 to 4294967295.")
        form.addRow("Reproducible collision seed", self.seed)
        self.reference = QLabel()
        self.reference.setWordWrap(True)
        form.addRow("Pressure reference", self.reference)
        self.chamber = CellChamberView()
        form.addRow(self.chamber)
        buttons = QHBoxLayout()
        self.apply_button = QPushButton("Apply region / cell")
        self.split_button = QPushButton("Split region")
        self.remove_button = QPushButton("Merge with previous")
        for b in (self.apply_button, self.split_button, self.remove_button):
            buttons.addWidget(b)
        form.addRow(buttons)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.result_text = QLabel("No particle calculation for this map yet.")
        self.result_text.setWordWrap(True)
        form.addRow("Last particle calculation", self.result_text)
        self.editor = scroll
        scroll.setWidget(form_widget)
        scroll.hide()
        self.selection_hint = QLabel("Select a vacuum interval or choose a region above to display its parameters. Z follows Physical Layout.")
        self.selection_hint.setWordWrap(True)
        outer.addWidget(self.selection_hint)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setSizes([255, 550])
        splitter.setStretchFactor(1, 2)
        outer.addWidget(splitter, 1)
        note = QLabel("Cell windows and fluid: classical elastic transport. Sample geometry: edit in Sample. Details: hover over controls.")
        note.setToolTip("Pressure gives gas number density via P/(kT); liquids and windows use supplied mass density. Existing fields, apertures and detector stops remain active. No inelastic chemistry, window crystal diffraction, photon absorption, pressure bulging or quantitative cell multislice imaging.")
        note.setWordWrap(True)
        outer.addWidget(note)
        outer.addWidget(self.status)
        self.diagram.selected.connect(self.select_region)
        self.selection.currentIndexChanged.connect(self._selection_changed)
        self.phase.currentIndexChanged.connect(self._medium_controls)
        self.mixture.textChanged.connect(self._medium_controls)
        self.apply_button.clicked.connect(self.apply)
        self.split_button.clicked.connect(self.split_region)
        self.remove_button.clicked.connect(self.remove_region)
        self.enabled.clicked.connect(self._toggle_enabled)
        self.cell_inserted.clicked.connect(lambda: self.select_region("specimen_cell"))
        self.position_mode.currentIndexChanged.connect(self._position_controls)
        self.coordinate_frame.currentIndexChanged.connect(self._coordinate_changed)
        for label in self.findChildren(QLabel):
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

    def _refresh_diagram(self):
        self.diagram.sample_z_mm = float(self.state.sample.z_mm)
        self.chamber.set_state(self.state)
        try:
            rows = resolve_regions(self.state, include_disabled=True)
            self.diagram.set_regions(rows, module_axial_ranges(self.state), self.current_key)
        except ValueError as exc:
            self.diagram.set_regions((), module_axial_ranges(self.state), self.current_key)
            self.status.setText(f"Invalid vacuum map: {exc}")

    def showEvent(self, event):
        super().showEvent(event)
        if self.state is not None:
            # Sample edits in another tab update the reference geometry, without
            # discarding unapplied cell inputs or fitting the shared column view.
            self._refresh_diagram()

    def focus_component(self, part):
        """Keep selection in step with other tabs without fitting or moving Z."""
        self.diagram.component = part
        if self.state is None:
            return
        try:
            rows = resolve_regions(self.state, include_disabled=True)
        except ValueError:
            self._refresh_diagram()
            return
        matches = [r for r in rows if r.start_z_mm <= part.center_z_mm < r.end_z_mm
                   and (r.radius_mm is None or part.key == "sample")]
        self.select_region(matches[-1].key if matches else "")

    def _position_controls(self):
        cell = self.current_key == "specimen_cell"
        positions = self.position_mode.currentData() == "positions"
        for widget in (self.position_mode, self.modules_text):
            self.form_layout.setRowVisible(widget, not cell)
        for widget in (self.coordinate_frame, self.start_z, self.end_z):
            self.form_layout.setRowVisible(widget, not cell and positions)
        for widget in (self.start_anchor, self.end_anchor, self.start_offset, self.end_offset):
            self.form_layout.setRowVisible(widget, not cell and not positions)
        if not self._filling and positions and not cell and self.state is not None:
            anchors = boundary_anchors(self.state)
            origin = anchors.get(self.coordinate_frame.currentData(), 0.0)
            self.start_z.setValue(anchors.get(self.start_anchor.currentData(), 0.0)+self.start_offset.value()-origin)
            self.end_z.setValue(anchors.get(self.end_anchor.currentData(), 0.0)+self.end_offset.value()-origin)
        elif not self._filling and not cell and self.state is not None:
            anchors = boundary_anchors(self.state)
            self.start_offset.setValue(self.start_z.value()+self._coordinate_origin-anchors.get(self.start_anchor.currentData(), 0.0))
            self.end_offset.setValue(self.end_z.value()+self._coordinate_origin-anchors.get(self.end_anchor.currentData(), 0.0))

    def _coordinate_changed(self):
        if self._filling or self.state is None:
            return
        origin = boundary_anchors(self.state).get(self.coordinate_frame.currentData(), 0.0)
        self.start_z.setValue(self.start_z.value()+self._coordinate_origin-origin)
        self.end_z.setValue(self.end_z.value()+self._coordinate_origin-origin)
        self._coordinate_origin = origin

    def _toggle_enabled(self):
        if self.state is not None and not self._filling:
            config = deepcopy(self.state.vacuum_map)
            config.enabled = self.enabled.isChecked()
            try:
                self._install(config, "Vacuum calculation setting updated.")
            except ValueError as exc:
                self.enabled.setChecked(self.state.vacuum_map.enabled)
                self.status.setText(f"Not applied: {exc}")

    @staticmethod
    def _spin(low=-1e6, high=1e6):
        widget = QDoubleSpinBox()
        widget.setDecimals(9)
        widget.setRange(low, high)
        widget.setSingleStep(.001)
        return widget

    def set_state(self, state):
        self.state = state
        self._filling = True
        self.selection.clear()
        self.selection.addItem("Select vacuum region…", "")
        for r in state.vacuum_map.regions:
            self.selection.addItem(r.name, r.key)
        self.selection.addItem("Inserted specimen cell", "specimen_cell")
        try:
            for r in resolve_regions(state, include_cell=False, include_disabled=True):
                if r.end_medium is not None:
                    self.selection.addItem(f"Auto transition: {r.name}", r.key)
        except ValueError:
            pass  # Display validation errors in the map, retaining editable regions.
        self.enabled.setChecked(state.vacuum_map.enabled)
        self.seed.setText(str(state.vacuum_map.seed))
        self.cell_inserted.setChecked(state.vacuum_map.cell.inserted)
        for name, widget in self.cell_fields.items():
            widget.setValue(getattr(state.vacuum_map.cell, name)*1e6)
        for key, editor in self.window_editors.items():
            editor.set_window(getattr(state.vacuum_map.cell, key))
        anchors = boundary_anchors(state)
        self.coordinate_frame.clear()
        self.coordinate_frame.addItem("Global Z", "axis_origin")
        for module in module_axial_ranges(state):
            self.coordinate_frame.addItem(f"{module.key} · local Z", f"module:{module.key}.origin")
        for widget in (self.start_anchor, self.end_anchor):
            widget.clear()
            for key, z in anchors.items():
                widget.addItem(f"{key} · {z:g} mm", key)
        self._filling = False
        self._refresh_diagram()
        self.select_region(self.current_key)

    def select_region(self, key):
        if key and key.startswith("cell_window_"):
            key = "specimen_cell"
        self.current_key = key
        index = self.selection.findData(key)
        self.selection.blockSignals(True)
        self.selection.setCurrentIndex(max(index, 0))
        self.selection.blockSignals(False)
        self._selection_changed()

    def _selection_changed(self):
        if self._filling or self.state is None:
            return
        config = self.state.vacuum_map
        key = self.selection.currentData()
        self.current_key = key
        self._refresh_diagram()
        self.editor.setVisible(bool(key) and not key.startswith("transition:"))
        self.selection_hint.setVisible(not key or key.startswith("transition:"))
        if not key:
            self.selection_hint.setText("Select a vacuum interval or choose a region above to display its parameters. Z follows Physical Layout.")
            return
        if key.startswith("transition:"):
            row = next(r for r in self.diagram.rows if r.key == key)
            start_pressure = 0.0 if row.medium.phase == "vacuum" else row.medium.pressure_mbar
            end_pressure = 0.0 if row.end_medium.phase == "vacuum" else row.end_medium.pressure_mbar
            self.selection_hint.setText(f"{row.name} · Z {row.start_z_mm:g}–{row.end_z_mm:g} mm · "
                f"automatic linear pressure transition: {start_pressure:g} → {end_pressure:g} mbar. "
                "Edit either adjacent region to change this transition.")
            return
        self._filling = True
        cell = key == "specimen_cell"
        self.form_layout.setRowVisible(self.windows, cell)
        self.chamber.setVisible(cell or key == "specimen")
        for widget in self.cell_fields.values():
            self.form_layout.setRowVisible(widget, cell)
        r = next((r for r in config.regions if r.key == key), None)
        medium = config.cell.medium if cell else r.medium
        self.name.setText("Inserted specimen cell" if cell else r.name)
        self.name.setEnabled(not cell)
        for widget in (self.start_anchor, self.end_anchor, self.start_offset, self.end_offset):
            widget.setEnabled(not cell)
            self.form_layout.setRowVisible(widget, not cell)
        if r:
            self.start_anchor.setCurrentIndex(self.start_anchor.findData(r.start_anchor))
            self.end_anchor.setCurrentIndex(self.end_anchor.findData(r.end_anchor))
            self.start_offset.setValue(r.start_offset_mm)
            self.end_offset.setValue(r.end_offset_mm)
            anchors = boundary_anchors(self.state)
            simple = (r.start_anchor == r.end_anchor or r.key == "projection") and self.coordinate_frame.findData(r.end_anchor) >= 0
            self.position_mode.setCurrentIndex(1 if simple else 0)
            self.coordinate_frame.setCurrentIndex(self.coordinate_frame.findData(r.end_anchor) if simple else 0)
            self._coordinate_origin = anchors[self.coordinate_frame.currentData()]
            self.start_z.setValue(anchors.get(r.start_anchor, 0.0)+r.start_offset_mm-self._coordinate_origin)
            self.end_z.setValue(anchors.get(r.end_anchor, 0.0)+r.end_offset_mm-self._coordinate_origin)
            self.start_z.setEnabled(r.key != "projection")
            if r.key == "projection":
                self.start_anchor.setEnabled(False)
                self.start_offset.setEnabled(False)
        self.phase.setCurrentIndex(self.phase.findData(medium.phase))
        self.formula.setText(medium.formula)
        self.mixture.setText(json.dumps(medium.mixture_mole_fractions) if medium.mixture_mole_fractions else "")
        self.pressure.setText(f"{medium.pressure_mbar:.9g}")
        self.temperature.setValue(medium.temperature_k)
        self.density.setValue(medium.density_kg_m3)
        self.removal.setText(f"{medium.removal_cross_section_m2:.9g}")
        self.removal_reference.setText(medium.removal_reference)
        self.reference.setText(r.pressure_reference if r else "Finite cell; internal medium excludes the Sample envelope.")
        try:
            rows = resolve_regions(self.state, include_cell=False, include_disabled=True)
            resolved = next((v for v in rows if v.key == key), None)
            self.bounds.setText(f"Z {resolved.start_z_mm:g} to {resolved.end_z_mm:g} mm" if resolved else "Relative to specimen centre")
            self.modules_text.setText("\n".join(f"{m.key} · {m.source_file}"
                for m in module_axial_ranges(self.state) if resolved and
                m.start_z_mm < resolved.end_z_mm and m.end_z_mm > resolved.start_z_mm))
            if cell:
                centre = self.state.sample.z_mm+config.cell.offset_z_mm
                self.bounds.setText(f"Z {centre-config.cell.length_mm/2:.12g} to {centre+config.cell.length_mm/2:.12g} mm; diameter {config.cell.diameter_mm:g} mm")
        except ValueError as exc:
            rows = ()
            self.bounds.setText(str(exc))
        self.split_button.setEnabled(not cell)
        self.remove_button.setEnabled(not cell and key not in self.CORE_KEYS)
        self._medium_controls()
        self._position_controls()
        self._filling = False

    def _medium_controls(self):
        phase = self.phase.currentData()
        self.pressure.setEnabled(phase in {"gas", "liquid"})
        self.temperature.setEnabled(phase in {"gas", "liquid"})
        self.formula.setEnabled(phase != "vacuum" and not self.mixture.text().strip())
        self.mixture.setEnabled(phase != "vacuum")
        self.density.setEnabled(phase == "liquid")

    def _install(self, config, message):
        config.validate()
        candidate = copy(self.state)
        candidate.vacuum_map = config
        resolve_regions(candidate, include_disabled=True)
        self.state.vacuum_map = config
        self.set_state(self.state)
        self.status.setText(message)
        self.result_text.setText("Map changed. Previous particle results are out of date.")
        self.changed.emit("vacuum_map")

    def set_result(self, result):
        simulation = result.simulation
        reports = [getattr(simulation.gun_trace, "vacuum_report", None),
                   getattr(simulation.incident, "vacuum_report", None)]
        reports.extend(getattr(b, "vacuum_report", None) for b in simulation.branches.values())
        seen, lines = set(), []
        for report in reports:
            if not report or id(report) in seen:
                continue
            seen.add(id(report))
            for r in report["regions"]:
                lines.append(f"{r['name']}: mean path {r['mean_path_m']*1000:.5g} mm; "
                             f"uncollided {(100*r['mean_uncollided_fraction']):.9g}%; "
                             f"{r['elastic_events']} elastic events")
        self.result_text.setText("\n".join(lines) if lines else "Residual-medium transport was disabled for this calculation.")

    def apply(self):
        if self.state is None:
            return False
        if not self.current_key or self.current_key.startswith("transition:"):
            return True
        try:
            config = deepcopy(self.state.vacuum_map)
            config.enabled, config.seed = self.enabled.isChecked(), int(self.seed.text())
            config.cell.inserted = self.cell_inserted.isChecked()
            for name, widget in self.cell_fields.items():
                setattr(config.cell, name, widget.value()*1e-6)
            for key, editor in self.window_editors.items():
                setattr(config.cell, key, editor.window())
            medium = Medium(self.phase.currentData(), self.formula.text().strip(), float(self.pressure.text()),
                            self.temperature.value(), self.density.value(), float(self.removal.text()), self.removal_reference.text().strip(),
                            json.loads(self.mixture.text()) if self.mixture.text().strip() else {})
            if self.current_key == "specimen_cell":
                config.cell.medium = medium
            else:
                index = next(i for i, r in enumerate(config.regions) if r.key == self.current_key)
                r = config.regions[index]
                r.name, r.medium = self.name.text().strip(), medium
                r.start_anchor, r.start_offset_mm = self.start_anchor.currentData(), self.start_offset.value()
                r.end_anchor, r.end_offset_mm = self.end_anchor.currentData(), self.end_offset.value()
                if self.position_mode.currentData() == "positions":
                    r.end_anchor = self.coordinate_frame.currentData()
                    r.end_offset_mm = self.end_z.value()
                    if r.key != "projection":
                        r.start_anchor = r.end_anchor
                        r.start_offset_mm = self.start_z.value()
                r.pressure_reference = "User edited; previous reference: "+r.pressure_reference if asdict(medium) != asdict(self.state.vacuum_map.regions[index].medium) else r.pressure_reference
            self._install(config, "Applied. Previous calculations are stale; the next particle calculation uses this map.")
            return True
        except (ValueError, TypeError) as exc:
            self.status.setText(f"Not applied: {exc}")
            return False

    def split_region(self):
        if not self.apply():
            return
        config = deepcopy(self.state.vacuum_map)
        index = next(i for i, r in enumerate(config.regions) if r.key == self.current_key)
        r = config.regions[index]
        resolved = next(v for v in resolve_regions(self.state, include_disabled=True) if v.key == r.key)
        suffix = 1
        while any(v.key == f"region_{suffix}" for v in config.regions):
            suffix += 1
        new = deepcopy(r)
        new.key, new.name = f"region_{suffix}", f"{r.name} — second section"
        r.end_anchor = r.start_anchor
        r.end_offset_mm = r.start_offset_mm+(resolved.end_z_mm-resolved.start_z_mm)/2
        new.start_anchor, new.start_offset_mm = r.end_anchor, r.end_offset_mm
        config.regions.insert(index+1, new)
        self.current_key = new.key
        try:
            self._install(config, "Region split. Both sections have independent medium parameters.")
        except ValueError as exc:
            self.status.setText(f"Not split: {exc}")

    def remove_region(self):
        config = deepcopy(self.state.vacuum_map)
        index = next(i for i, r in enumerate(config.regions) if r.key == self.current_key)
        if index == 0 or self.current_key in self.CORE_KEYS:
            return
        removed = config.regions.pop(index)
        config.regions[index-1].end_anchor = removed.end_anchor
        config.regions[index-1].end_offset_mm = removed.end_offset_mm
        self.current_key = config.regions[index-1].key
        try:
            self._install(config, "Regions merged; previous region's medium retained.")
        except ValueError as exc:
            self.status.setText(f"Not merged: {exc}")

    def open_map(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open vacuum map", str(DEFAULT_PATH.parent), "TOML (*.toml)")
        if path:
            try:
                self._install(VacuumMap.load(path), f"Loaded {path}")
            except (ValueError, OSError) as exc:
                self.status.setText(f"Not loaded: {exc}")

    def save_map(self):
        if not self.apply():
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save vacuum map", str(DEFAULT_PATH), "TOML (*.toml)")
        if path:
            try:
                self.state.vacuum_map.save(path)
                self.status.setText(f"Saved {path}. Operating profiles also retain the complete map.")
            except OSError as exc:
                self.status.setText(f"Not saved: {exc}")

    def load_defaults(self):
        if self.state is not None:
            try:
                self._install(VacuumMap.load(), "Loaded normal-operation pressure defaults.")
            except (ValueError, OSError) as exc:
                self.status.setText(f"Defaults not loaded: {exc}")
