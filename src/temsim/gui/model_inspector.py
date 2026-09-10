"""Compact model evidence and explicit numerical-model controls."""

from copy import deepcopy

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QPushButton, QLineEdit, QTabWidget, QFileDialog, QPlainTextEdit, QCheckBox
from temsim.gui.input_policy import WheelSafeComboBox as QComboBox, WheelSafeSpinBox as QSpinBox


class ModelInspectorPage(QWidget):
    changed = Signal(str)
    error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = None
        self._completed_diagnostics = {}
        self._completed_mode = None
        self._completed_aberrations = {}
        self.lens = QComboBox()
        self.field_solver = QComboBox()
        self.field_solver.addItem("Linear geometry", "axisymmetric_linear_fem")
        self.field_solver.addItem("Nonlinear B-H", "axisymmetric_nonlinear_fem")
        self.bh_material = QComboBox()
        from temsim.magnetic_materials import lens_material_defaults, reference_materials
        self._material_defaults = lens_material_defaults()
        self._linear_material_reference = None
        for material in reference_materials():
            self.bh_material.addItem(material["label"], material)
        self.import_bh = QPushButton("Import B-H CSV...")
        self.import_bh.clicked.connect(self._import_bh)
        self.default_material = QPushButton("Use default material")
        self.default_material.setToolTip("Use the generic pure-iron reference in this draft. Coil inputs and saved per-material overrides are retained; apply with Use geometry field.")
        self.default_material.clicked.connect(self._use_default_material)
        self.material_hint = QLabel()
        self.material_hint.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.material_hint.setWordWrap(True)
        self.permeability = QLineEdit()
        self.permeability.setPlaceholderText("Relative permeability")
        self.permeability.setToolTip("Relative permeability (dimensionless). A constant linear approximation, not a saturation model; source details are below.")
        self.ampere_turns = QLineEdit()
        self.ampere_turns.setPlaceholderText("Ampere-turns at 100%")
        self.mesh = QSpinBox()
        self.mesh.setRange(12, 128)
        self.mesh.setValue(40)
        self.mesh.setPrefix("Radial nodes: ")
        self.padding = QLineEdit("2")
        self.padding.setMaximumWidth(90)
        self.padding.setToolTip("Finite boundary padding factor (1.5–10). Increase it and compare results to check domain convergence.")
        generate = QPushButton("Use geometry field")
        reset = QPushButton("Use analytic field")
        generate.clicked.connect(self._apply_field)
        reset.clicked.connect(self._reset_field)
        controls = QHBoxLayout()
        for widget in (self.lens, self.permeability, self.ampere_turns):
            controls.addWidget(widget)
        material_controls = QHBoxLayout()
        for widget in (self.field_solver, self.bh_material, self.import_bh, self.default_material):
            material_controls.addWidget(widget)
        self.field_solver.currentIndexChanged.connect(self._field_controls_changed)
        self.bh_material.currentIndexChanged.connect(self._field_controls_changed)
        self.permeability.textChanged.connect(self._field_controls_changed)
        self._field_controls_changed()
        numerical_controls = QHBoxLayout()
        for widget in (self.mesh, self.padding, generate, reset):
            numerical_controls.addWidget(widget)
        numerical_controls.addStretch(1)
        self.dimension_approximation = QCheckBox("Use authoritative axisymmetric dimensions; ignore CAD-only features")
        self.dimension_approximation.setToolTip("Explicit approximation for features/transforms that are not consumed by the magnetic solver. Without it, new field recipes reject unrepresented CAD geometry.")
        self.current_summary = QLabel("Current unavailable — configure reference ampere-turns")
        self.current_summary.setWordWrap(True)
        calibration_button = QPushButton("Coil calibration / current…")
        calibration_button.clicked.connect(self._edit_excitation)
        electrical_controls = QHBoxLayout()
        electrical_controls.addWidget(self.current_summary, 1)
        electrical_controls.addWidget(calibration_button)
        hint = QLabel("Axisymmetric field · explicit material and coil inputs · calculated on next run")
        hint.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        hint.setWordWrap(True)
        hint.setToolTip("Reference materials are not Titan assignments. Nonlinear fields solve all configured B-H channels jointly; use matching material/mesh settings. No hysteresis, thermal coupling or preset optimisation. Conflicting geometry, nonconvergence and out-of-range B-H results fail explicitly. Check mesh and boundary convergence separately.")
        self.aberration_system = QComboBox()
        self.aberration_system.addItem("Probe aberrations", "probe")
        self.aberration_system.addItem("Image aberrations", "image")
        self.aberration_mode = QComboBox()
        self.aberration_mode.addItem("Manual", "manual")
        self.aberration_mode.addItem("Field-derived", "field_derived")
        self.fit_angle = QLineEdit("10")
        self.fit_angle.setPlaceholderText("Fit semi-angle (mrad)")
        self.fit_angle.setToolTip("Finite pupil used by the ray-gradient fit, in mrad. C1 is already in first-order transport; Cc is fitted by an energy perturbation. Unmapped round lenses do not supply high-order field information.")
        apply_mode = QPushButton("Apply aberration mode")
        apply_mode.clicked.connect(self._apply_mode)
        modes = QHBoxLayout()
        for widget in (self.aberration_system, self.aberration_mode):
            modes.addWidget(widget)
        modes.addStretch(1)
        fit_controls = QHBoxLayout()
        for widget in (QLabel("Semi-angle (mrad)"), self.fit_angle, apply_mode):
            fit_controls.addWidget(widget)
        fit_controls.addStretch(1)
        edit_coefficients = QPushButton("Coefficients / fit settings…")
        edit_coefficients.clicked.connect(self._edit_aberrations)
        fit_controls.addWidget(edit_coefficients)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(("Model / check", "Status", "Evidence"))
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 260)
        self.table.setColumnWidth(1, 120)
        self.circuit_table = QTableWidget(0, 5)
        self.circuit_table.setHorizontalHeaderLabels(("Circuit", "Topology", "Channels", "Magnetic bodies", "Evidence"))
        self.circuit_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.circuit_table.setAlternatingRowColors(True)
        self.circuit_table.horizontalHeader().setStretchLastSection(True)
        self.circuit_table.setColumnWidth(0, 230)
        self.circuit_table.setColumnWidth(1, 230)
        tables = QTabWidget()
        tables.setObjectName("modelInspectorTabs")
        tables.addTab(self.table, "Model checks")
        tables.addTab(self.circuit_table, "Magnetic circuits")
        from temsim.gui.magnetic_validation import MagneticValidationPage
        self.validation_page = MagneticValidationPage()
        tables.addTab(self.validation_page, "Field validation")
        fit_evidence_page = QWidget()
        fit_evidence_layout = QVBoxLayout(fit_evidence_page)
        fit_note = QLabel("Last completed field fits, with their original inputs. Select Probe or Image above. New controls take effect on the next calculation.")
        fit_note.setWordWrap(True)
        fit_evidence_layout.addWidget(fit_note)
        self.fit_evidence = QPlainTextEdit()
        self.fit_evidence.setReadOnly(True)
        self.fit_evidence.setObjectName("aberrationFitEvidence")
        fit_evidence_layout.addWidget(self.fit_evidence)
        self.export_fit = QPushButton("Export fit evidence…")
        self.export_fit.setEnabled(False)
        self.export_fit.clicked.connect(self._export_fit_evidence)
        fit_evidence_layout.addWidget(self.export_fit)
        tables.addTab(fit_evidence_page, "Aberration evidence")
        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addLayout(material_controls)
        layout.addWidget(self.material_hint)
        layout.addLayout(numerical_controls)
        layout.addWidget(self.dimension_approximation)
        layout.addLayout(electrical_controls)
        layout.addWidget(hint)
        layout.addLayout(modes)
        layout.addLayout(fit_controls)
        layout.addWidget(tables, 1)
        self.lens.currentIndexChanged.connect(self._load_field)
        self.aberration_system.currentIndexChanged.connect(self._load_mode)

    def set_state(self, state):
        self._state = state
        selected = self.lens.currentData()
        self.lens.blockSignals(True)
        self.lens.clear()
        for lens in state.lenses:
            self.lens.addItem(lens.name, lens.key)
        self.lens.setCurrentIndex(max(self.lens.findData(selected), 0))
        self.lens.blockSignals(False)
        self._load_field()
        self._load_mode()
        self.refresh()

    def _load_field(self):
        if self._state is None:
            return
        row = self._state.lens_field_map_descriptors.get(self.lens.currentData(), {})
        reference = self._material_defaults["linear_material_reference"]
        self._linear_material_reference = deepcopy(row.get("linear_material_reference")) if "relative_permeability" in row else deepcopy(reference)
        self.permeability.setText(str(row.get("relative_permeability", reference["relative_permeability"])))
        self.ampere_turns.setText(str(row.get("ampere_turns", "")))
        self.ampere_turns.setReadOnly("excitation_calibration" in row)
        self.dimension_approximation.setChecked(row.get("geometry_policy") in {"authoritative_dimensions", "legacy_authoritative_dimensions"})
        try:
            from temsim.excitation_calibration import calibration_from_recipe
            lens = next(item for item in self._state.lenses if item.key == self.lens.currentData())
            operating = calibration_from_recipe(row).at_control(lens.percent, getattr(lens, "polarity", 1), enabled=lens.enabled)
            current = operating["current_A"]
            self.current_summary.setText(f"NI {operating['ampere_turns']:.6g} A-turn | " +
                ("Current unavailable: winding turns unknown" if current is None else f"I {current:.6g} A | {operating['current_status']}"))
        except (KeyError, ValueError, StopIteration):
            self.current_summary.setText("Current unavailable — configure reference ampere-turns")
        self.mesh.setValue(int(row.get("radial_nodes", 40)))
        self.padding.setText(str(row.get("padding_factor", 2)))
        self.field_solver.setCurrentIndex(max(0, self.field_solver.findData(row.get("solver", "axisymmetric_linear_fem"))))
        # A new lens must not inherit the previously selected lens's custom CSV.
        self._select_material(row.get("bh_material", self._material_defaults["bh_material"]))
        self._field_controls_changed()
        self.validation_page.set_context(self._state, self.lens.currentData())

    def _select_material(self, material):
        index = next((i for i in range(self.bh_material.count()) if self.bh_material.itemData(i) == material), -1)
        if index < 0:
            self.bh_material.addItem(material["label"], deepcopy(material))
            index = self.bh_material.count()-1
        self.bh_material.setCurrentIndex(index)

    def _use_default_material(self):
        self._select_material(self._material_defaults["bh_material"])
        self._linear_material_reference = deepcopy(self._material_defaults["linear_material_reference"])
        self.permeability.setText(str(self._linear_material_reference["relative_permeability"]))
        self._field_controls_changed()

    def _linear_reference(self):
        reference = self._linear_material_reference
        try:
            if reference and float(self.permeability.text()) == reference["relative_permeability"]:
                return reference
        except ValueError:
            pass
        return None

    def _field_controls_changed(self):
        nonlinear = self.field_solver.currentData() == "axisymmetric_nonlinear_fem"
        self.permeability.setEnabled(not nonlinear)
        self.bh_material.setEnabled(nonlinear)
        self.import_bh.setEnabled(nonlinear)
        material = self.bh_material.currentData() or {}
        self.bh_material.setToolTip(f"{material.get('scope', '')}\n{material.get('source_url', '')}\nDefault for magnetic regions; no automatic OEM material assignment.")
        reference = self._linear_reference()
        if nonlinear:
            self.material_hint.setText(f"Magnetic bodies: {material.get('label', 'Select material')} | B-H curve")
            self.material_hint.setToolTip(self.bh_material.toolTip())
        elif reference:
            self.material_hint.setText(f"Magnetic bodies: {reference['label']} | constant permeability")
            self.material_hint.setToolTip(f"{reference['scope']}\n{reference['source_url']}\nSource fields: Mu_x / Mu_y. Per-material overrides, if saved, take precedence.")
        else:
            self.material_hint.setText("Magnetic bodies: user-defined constant permeability")
            self.material_hint.setToolTip("Explicit linear permeability; no reference-material identity inferred from its value. Per-material overrides, if saved, take precedence.")

    def _import_bh(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import B-H reference", "", "B-H data (*.csv)")
        if not path:
            return
        try:
            from temsim.magnetic_materials import import_bh_csv
            material = import_bh_csv(path)
            self.bh_material.addItem(material["label"], material)
            self.bh_material.setCurrentIndex(self.bh_material.count()-1)
        except (OSError, ValueError, UnicodeError) as exc:
            self.error.emit(str(exc))

    def _load_mode(self):
        if self._state is None:
            return
        row = getattr(self._state, f"{self.aberration_system.currentData()}_aberrations")
        self.aberration_mode.setCurrentIndex(max(self.aberration_mode.findData(row.get("mode", "manual")), 0))
        self.fit_angle.setText(str(row.get("fit_semiangle_mrad", 10)))
        self._show_fit_evidence()

    def _show_fit_evidence(self):
        import json
        evidence = self._completed_aberrations.get(self.aberration_system.currentData())
        self.fit_evidence.setPlainText(json.dumps(evidence, indent=2) if evidence is not None else "No completed field fit. Choose Field-derived, then run a calculation.")
        self.export_fit.setEnabled(evidence is not None)

    def _export_fit_evidence(self):
        import json
        from pathlib import Path
        evidence = self._completed_aberrations.get(self.aberration_system.currentData())
        if evidence is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export completed aberration fit", "aberration-fit.json", "JSON (*.json)")
        if path:
            try:
                Path(path).write_text(json.dumps(evidence, indent=2), encoding="utf-8")
            except OSError as exc:
                self.error.emit(str(exc))

    def _apply_field(self):
        import math
        if self._state is None:
            return
        try:
            nonlinear = self.field_solver.currentData() == "axisymmetric_nonlinear_fem"
            if not self.ampere_turns.text().strip():
                raise ValueError("Enter ampere-turns at 100%; the material default does not define coil excitation")
            mur = 1.0 if nonlinear else float(self.permeability.text())
            turns, padding = float(self.ampere_turns.text()), float(self.padding.text())
            if not (math.isfinite(mur) and mur > 0 and math.isfinite(turns) and turns >= 0 and 1.5 <= padding <= 10):
                raise ValueError("Enter positive permeability, nonnegative ampere-turns and padding from 1.5 to 10")
            key = self.lens.currentData()
            from temsim.magnetic_circuits import circuit_inventory
            circuit = next((c for c in circuit_inventory(getattr(getattr(self._state, "_resolved_assembly", None), "parts", ())) if key in c.channels), None)
            if circuit and circuit.topology == "monolithic_saturated_insert" and not nonlinear:
                raise ValueError("This circuit requires nonlinear B-H magnetostatics; the linear geometry solver is not applicable")
            recipe = {
                "solver": self.field_solver.currentData(), "ampere_turns": turns,
                "radial_nodes": self.mesh.value(), "axial_nodes": self.mesh.value()*2, "padding_factor": padding,
                "geometry_policy": "authoritative_dimensions" if self.dimension_approximation.isChecked() else "require_full_geometry",
            }
            previous = self._state.lens_field_map_descriptors.get(key, {})
            if "excitation_calibration" in previous:
                recipe["excitation_calibration"] = deepcopy(previous["excitation_calibration"])
            if nonlinear:
                from temsim.magnetic_materials import validate_bh_material
                recipe["bh_material"] = validate_bh_material(self.bh_material.currentData())
                for name in ("material_bh_overrides", "relative_tolerance", "max_iterations"):
                    if name in previous:
                        recipe[name] = previous[name]
            else:
                recipe["relative_permeability"] = mur
                if self._linear_reference():
                    recipe["linear_material_reference"] = deepcopy(self._linear_reference())
                if "material_permeabilities" in previous:
                    recipe["material_permeabilities"] = previous["material_permeabilities"]
            from temsim.excitation_calibration import validate_excitation_recipe
            validate_excitation_recipe(recipe)
            from temsim.simulation_modes import promote_custom_mode
            promote_custom_mode(self._state)
            self._state.lens_field_map_descriptors[key] = recipe
            getattr(self._state, "_lens_field_map_bindings", {}).pop(key, None)
            self.changed.emit("lens_field_map_descriptors")
            self.refresh()
        except ValueError as exc:
            self.error.emit(str(exc))

    def _reset_field(self):
        if self._state is None:
            return
        key = self.lens.currentData()
        from temsim.simulation_modes import promote_custom_mode
        promote_custom_mode(self._state)
        self._state.lens_field_map_descriptors.pop(key, None)
        getattr(self._state, "_lens_field_map_bindings", {}).pop(key, None)
        self.changed.emit("lens_field_map_descriptors")
        self.refresh()

    def _apply_mode(self):
        import math
        if self._state is None:
            return
        try:
            angle = float(self.fit_angle.text())
            if not math.isfinite(angle) or not 0 < angle <= 100:
                raise ValueError("Fit semi-angle must be in (0, 100] mrad")
            system = self.aberration_system.currentData()
            from temsim.simulation_modes import promote_custom_mode
            promote_custom_mode(self._state)
            options = getattr(self._state, f"{system}_aberrations")
            options.update(mode=self.aberration_mode.currentData(), fit_semiangle_mrad=angle)
            self.changed.emit(f"{system}_aberrations")
            self.refresh()
        except ValueError as exc:
            self.error.emit(str(exc))

    def _edit_aberrations(self):
        if self._state is None:
            return
        from temsim.gui.aberration_dialog import AberrationSettingsDialog
        from temsim.simulation_modes import promote_custom_mode
        system = self.aberration_system.currentData()
        dialog = AberrationSettingsDialog(getattr(self._state, f"{system}_aberrations"),
                                         system=system, state_step_mm=self._state.step_mm, parent=self)
        if dialog.exec() == dialog.DialogCode.Accepted:
            promote_custom_mode(self._state)
            setattr(self._state, f"{system}_aberrations", dialog.result_options)
            getattr(self._state, "_effective_aberration_cache", {}).pop(system, None)
            self._load_mode()
            self.changed.emit(f"{system}_aberrations")
            self.refresh()

    def refresh(self):
        if self._state is None:
            return
        self.validation_page.set_context(self._state, self.lens.currentData())
        rows = []
        from temsim.simulation_modes import mode_key, MODE_BY_KEY
        selected_mode = mode_key(self._state)
        rows.append(("Column lens model", "Selected", MODE_BY_KEY[selected_mode].label, MODE_BY_KEY[selected_mode].detail))
        from temsim.magnetic_circuits import circuit_inventory, TOPOLOGIES
        circuits = circuit_inventory(getattr(getattr(self._state, "_resolved_assembly", None), "parts", ()))
        self.circuit_table.setRowCount(len(circuits))
        for row, circuit in enumerate(circuits):
            values = (circuit.key, TOPOLOGIES.get(circuit.topology, "Legacy / unspecified"),
                      str(len(circuit.channels)), str(len(circuit.body_keys)), circuit.evidence.replace("_", " ").title())
            detail = (f"Channels: {', '.join(circuit.channels)}\nBodies: {', '.join(circuit.body_keys) or 'None'}"
                      f"\nCoils: {', '.join(circuit.coil_keys) or 'None'}\n{circuit.source}"
                      "\nEdit physical topology and membership in the assembly TOML. This evidence label is not a solver validation.")
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(detail)
                self.circuit_table.setItem(row, column, item)
        for lens in self._state.lenses:
            descriptor = self._state.lens_field_map_descriptors.get(lens.key, {})
            if selected_mode in {"ideal", "analytical"}:
                evidence = "Ideal paraxial field" if selected_mode == "ideal" else "Analytic field profile"
                detail = "Saved field recipes are retained but inactive in this global model. Pole/material geometry does not solve the field boundary-value problem."
            elif descriptor.get("solver") == "axisymmetric_linear_fem":
                evidence = "Linear geometry field requested"
                detail = "Generated from current assembly geometry on the next run. Overlapping materials are rejected; no measured B-H curve or convergence evidence supplied."
            elif descriptor.get("solver") == "axisymmetric_nonlinear_fem":
                evidence = "Joint static B-H field requested"
                material = descriptor.get("bh_material", {})
                detail = f"{material.get('label', 'Missing material')}\n{material.get('source_url', '')}\nAll configured B-H channels share one total field; individual saturated fields are not additive. No hysteresis or thermal feedback."
            elif descriptor:
                evidence = "Registered field map requested"
                detail = "File identity and geometry binding are checked by the solver. A registered map alone does not establish mesh convergence or experimental calibration."
            else:
                evidence = "Analytic field profile"
                detail = "Provisional parameterised field. Pole shape and material do not solve the magnetostatic boundary-value problem."
            rows.append((lens.name, "Approximate", evidence, detail))
        rows.extend((
            ("Mesh / boundary convergence", "Per study", "Open Field validation", "Run a detached study for the selected circuit. A small algebraic residual does not establish discretisation or finite-boundary accuracy; per-study results are not whole-column or Cs/Cc validation."),
            ("Specimen transport", "Approximate", "Shared vector fields", "Relativistic magnetic flights; chord boundary tolerance is tied to the material geometry epsilon. Collision cross-section limitations remain separate."),
            ("TEM intermediate planes", "Approximate", "Coherent aperture propagation", "Affine-lattice LCT; masks remove probability without renormalisation. Undersampled chirps and mixed singular conjugacy fail explicitly."),
        ))
        for system in ("probe", "image"):
            mode = getattr(self._state, f"{system}_aberrations").get("mode", "manual")
            if selected_mode == "ideal":
                mode = "disabled; defocus retained"
            rows.append((f"{system.title()} aberrations", "Approximate", mode.replace("_", " ").title(), "Manual and field-derived coefficients are mutually exclusive. Field fits report finite-pupil residuals and unmapped round lenses; first-order focus is not applied twice."))
        diagnostics = self._completed_diagnostics if selected_mode == self._completed_mode else {}
        from temsim.physics.lens_field_provider import lens_geometry_binding, _fingerprint
        for key, diagnostic in diagnostics.items():
            if (diagnostic.get("geometry_fingerprint") != lens_geometry_binding(self._state, key).geometry_fingerprint
                    or diagnostic.get("descriptor_fingerprint") != _fingerprint(self._state.lens_field_map_descriptors.get(key))):
                continue
            if "operating_point" in diagnostic:
                from temsim.physics.lens_field_provider import _runtime_excitation
                from temsim.physics.nonlinear_circuits import nonlinear_state_fingerprint
                if tuple(diagnostic["operating_point"]) != tuple(_runtime_excitation(lens) for lens in self._state.lenses):
                    continue
                if diagnostic.get("joint_state_fingerprint") != nonlinear_state_fingerprint(self._state):
                    continue
                rows.append((f"{key}: B-H solve", "Converged", f"Joint field owner: {diagnostic['joint_field_owner']}", diagnostic.get("source_note", "")))
            passed = diagnostic.get("divergence_within_tolerance")
            if passed is not None:
                rows.append((f"{key}: divergence", "Validated" if passed else "Approximate", "Within tolerance" if passed else "Outside tolerance", "This validates only the discrete divergence check for this field, not the complete microscope or experimental calibration."))
        self.table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values[:3]):
                item = QTableWidgetItem(value)
                item.setToolTip(values[3])
                self.table.setItem(row, column, item)

    def display_result(self, result):
        from dataclasses import asdict
        snapshot = getattr(result, "state_snapshot", None)
        from temsim.simulation_modes import mode_key
        self._completed_mode = mode_key(snapshot)
        self._completed_diagnostics = dict(getattr(snapshot, "_field_provider_diagnostics", {}))
        self._completed_aberrations = {
            system: {"system": system, "signature": signature, "coefficients": asdict(values[1]),
                     "diagnostics": values[2], "beam_energy_kev": snapshot.beam_voltage_kv,
                     "options": deepcopy(getattr(snapshot, f"{system}_aberrations", {}))}
            for system, (signature, values) in getattr(snapshot, "_effective_aberration_cache", {}).items()
            if values[1].correction_state == "field-derived"
        }
        self._show_fit_evidence()
        self.refresh()

    def _edit_excitation(self):
        if self._state is None:
            return
        from temsim.excitation_calibration import calibration_from_recipe, recipe_with_calibration
        from temsim.gui.excitation_dialog import ExcitationDialog
        key = self.lens.currentData()
        recipe = self._state.lens_field_map_descriptors.get(key, {})
        try:
            calibration = calibration_from_recipe(recipe)
        except (KeyError, ValueError) as exc:
            self.error.emit("Configure reference ampere-turns with Use geometry field first. " + str(exc))
            return
        lens = next(item for item in self._state.lenses if item.key == key)
        dialog = ExcitationDialog(calibration, lens, self)
        if dialog.exec() == dialog.DialogCode.Accepted:
            from temsim.simulation_modes import promote_custom_mode
            from temsim.excitation_calibration import validate_excitation_recipe
            validate_excitation_recipe(recipe)
            promote_custom_mode(self._state)
            self._state.lens_field_map_descriptors[key] = recipe_with_calibration(recipe, dialog.calibration)
            if dialog.control is not None:
                lens.percent, lens.polarity = dialog.control
            self.changed.emit("lens_field_map_descriptors")
            self._load_field()
            self.refresh()
