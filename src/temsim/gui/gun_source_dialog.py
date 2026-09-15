"""Draft edits to the physical FEG tip; no independent exit-source controls."""
from copy import copy
from dataclasses import replace
import math
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QScrollArea, QVBoxLayout, QWidget, QPushButton, QSpinBox
from temsim.optics.electron_gun.tip_coherence import TipCoherence, tip_covariance
from temsim.optics.electron_gun.tip_surface import SurfaceCoherence, load_tip_surface_reference, reference_path_for_gun
from temsim.optics.electron_gun.tip_patch import patch_dimensions
from temsim.gui.tip_geometry_preview import TipGeometryPreview


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

    def __init__(self, gun, parent=None, *, instrument_state=None):
        super().__init__(parent)
        if gun.type_key != "cold_feg":
            raise ValueError("This editor controls FEG tip emission only")
        self._gun = gun
        self._instrument_state = instrument_state
        self._value = None
        self.setWindowTitle("FEG tip geometry and emission")
        self.resize(680, 760)
        layout = QVBoxLayout(self)
        note = QLabel(
            "Define emission at the tip. Extraction, acceleration and downstream optics are calculated. "
            "These are operating overrides. Edit the installed tip in Physical Layout / TOML to save assembly defaults."
        )
        note.setWordWrap(True)
        note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(note)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        self.particle_button = QPushButton("Use curved-tip particles")
        self.particle_button.setToolTip("Select classical emission from the editable curved surface. Apply is required; historical source settings are retained.")
        self.particle_button.clicked.connect(self._use_particles)
        panel_layout.addWidget(self.particle_button)
        self.surface_enabled = QCheckBox("Curved tip surface")
        self.surface_enabled.setChecked(gun.emitter.surface_model is not None)
        panel_layout.addWidget(self.surface_enabled)
        self.surface_panel = QWidget()
        surface_form = QFormLayout(self.surface_panel)
        self.surface_form = surface_form
        surface_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self._reference_path = reference_path_for_gun(gun)
        from temsim.paths import INSTRUMENT_CONFIG_ROOT
        self._assembly_tip_path = Path(getattr(gun.emitter, "_manifest_source_file",
            INSTRUMENT_CONFIG_ROOT / "gun" / ("FEG_Mono.toml" if gun.monochromator_installed else "FEG.toml")))
        if not self._assembly_tip_path.is_absolute():
            self._assembly_tip_path = Path(getattr(gun, "_manifest_catalog_root", INSTRUMENT_CONFIG_ROOT)) / self._assembly_tip_path
        from temsim.shared_tip import definition_path, raw_document
        installed = next(part for part in raw_document(self._assembly_tip_path)["parts"] if part["key"] == "feg_tip")
        self._assembly_tip_path = definition_path(self._assembly_tip_path, installed) or self._assembly_tip_path
        from temsim.optics.electron_gun.tip_assembly import model_from_part
        from temsim import module_manifest
        self._surface_draft = (gun.emitter.surface_model
            or model_from_part(module_manifest.part_data(self._assembly_tip_path, "feg_tip"))
            or load_tip_surface_reference(self._reference_path))
        reference = QLabel(str(self._assembly_tip_path) + " : feg_tip")
        reference.setWordWrap(True)
        reference.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        surface_form.addRow("Shared tip TOML", reference)
        self.surface_geometry = QLabel()
        self.surface_geometry.setWordWrap(True)
        self.surface_geometry.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        surface_form.addRow(self.surface_geometry)
        reload_reference = QPushButton("Reload installed tip TOML defaults")
        reload_reference.clicked.connect(self._reload_surface)
        surface_form.addRow(reload_reference)
        self.geometry_inputs = {}
        for key, label in (("apex_radius_nm", "Apex radius of curvature (nm)"),
                           ("cone_half_angle_deg", "Cone half-angle (deg)"),
                           ("shank_length_um", "Shank length (µm)")):
            edit = QLineEdit(str(getattr(self._surface_draft.geometry, key)))
            self.geometry_inputs[key] = edit
            surface_form.addRow(label, edit)
            edit.textChanged.connect(self._surface_summary)
        self.geometry_inputs["apex_radius_nm"].setToolTip("Physical spherical apex radius, not the emission-patch radius. Changes the surface positions, normals and extraction-field boundary.")
        self.geometry_inputs["cone_half_angle_deg"].setToolTip("Cone angle relative to the axis. The spherical surface joins the cone tangentially; the emitting patch must remain inside that join.")
        self.geometry_inputs["shank_length_um"].setToolTip("Physical metal length upstream of the apex; not an independently placed electron source.")
        self.geometry_preview = TipGeometryPreview()
        surface_form.addRow(self.geometry_preview)
        self.patch_summary = QLabel()
        self.patch_summary.setWordWrap(True)
        self.patch_summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        surface_form.addRow("Derived patch geometry", self.patch_summary)
        self.surface_inputs = {}
        self.flux_density_enabled = QCheckBox("Prescribe electrons per unit surface area per second")
        self.flux_density_enabled.setChecked(self._surface_draft.emission.flux_electrons_per_nm2_s is not None)
        surface_form.addRow(self.flux_density_enabled)
        self.energy_law = QComboBox()
        for label, key in (("Normal / tangential exponential", "normal_tangential_exponential"),
                           ("Positive gamma energy", "gamma"), ("Monoenergetic", "monoenergetic")):
            self.energy_law.addItem(label, key)
        self.energy_law.setCurrentIndex(self.energy_law.findData(self._surface_draft.emission.energy_distribution))
        surface_form.addRow("Particle energy distribution", self.energy_law)
        self.directions_per_position = QSpinBox()
        self.directions_per_position.setRange(1, 256)
        self.directions_per_position.setValue(self._surface_draft.emission.directions_per_position)
        self.directions_per_position.setToolTip(
            "Numerical samples per surface site, within the total ray budget. "
            "More directions means fewer sites. A remainder is distributed across sites; "
            "weights preserve equal-area flux. 1 retains historical sampling. "
            "Zero angular spread still produces parallel directions."
        )
        surface_form.addRow("Directions per position", self.directions_per_position)
        for key, label in (("current_na", "Prescribed surface current (nA)"),
                           ("flux_electrons_per_nm2_s", "Emitted electrons / (nm² s)"),
                           ("cap_half_angle_deg", "Emitting cap half-angle (deg)"),
                           ("maximum_angle_deg", "Maximum angle from local normal (deg)"),
                           ("normal_mean_energy_ev", "Mean normal kinetic energy (eV)"),
                           ("tangential_mean_energy_ev", "Mean tangential kinetic energy (eV)"),
                           ("kinetic_mean_ev", "Mean total kinetic energy (eV)"),
                           ("kinetic_sigma_ev", "Total kinetic energy RMS (eV)")):
            value = getattr(self._surface_draft.emission, key)
            if value is None:
                value = (self._surface_draft.current_na if key == "current_na" else
                         self._surface_draft.current_na / (1.602176634e-10 * self._surface_draft.emission.area_nm2(self._surface_draft.geometry)))
            edit = QLineEdit(str(value))
            self.surface_inputs[key] = edit
            surface_form.addRow(label, edit)
            edit.textChanged.connect(self._surface_summary)
        self.surface_inputs["cap_half_angle_deg"].setToolTip("Geometric polar half-angle measured from the sphere centre. This defines emitting area, not the angular spread of electrons about each local normal.")
        self.wave_options = QCheckBox("Show wave development (paused)")
        self.wave_options.setChecked(self._surface_draft.coherence is not None)
        surface_form.addRow(self.wave_options)
        self.surface_coherent = QCheckBox("Coherent surface reservoir (development, paused)")
        self.surface_coherent.setChecked(self._surface_draft.coherence is not None)
        surface_form.addRow(self.surface_coherent)
        self.quantum_inputs = {}
        quantum = self._surface_draft.coherence or SurfaceCoherence()
        for key, label in (("mean_energy_ev", "Mean total kinetic energy at tip (eV)"),
                           ("energy_rms_ev", "Energy RMS at tip (eV)"),
                           ("edge_phase_rad", "Surface edge phase relative to apex (rad)")):
            edit = QLineEdit(str(getattr(quantum, key)))
            self.quantum_inputs[key] = edit
            surface_form.addRow(label, edit)
            edit.textChanged.connect(self._surface_summary)
        self.surface_derived = QLabel()
        self.surface_derived.setWordWrap(True)
        self.surface_derived.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        surface_form.addRow("Derived — not independent", self.surface_derived)
        self.surface_derived.setToolTip(
            "Uniform area on the curved tip cap. Normal and tangential energies have exponential distributions; "
            "directions are relative to the local surface normal. Total energy, RMS energy width, angular spread "
            "and current density follow from these inputs. Brightness/virtual-source size are downstream results. "
            "The local angular limit conditions the exponential energy law. Gamma / monoenergetic laws use uniform solid-angle directions within that limit. "
            "This prescribed classical flux does not predict tunnelling current or coherent phase.")
        self._classical_surface_tooltip = self.surface_derived.toolTip()
        self.surface_voltage = QLabel()
        self.surface_voltage.setWordWrap(True)
        self.surface_voltage.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        surface_form.addRow("Electrical reference", self.surface_voltage)
        self.surface_status = QLabel()
        self.surface_status.setWordWrap(True)
        self.surface_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        surface_form.addRow(self.surface_status)
        self.near_field_button = QPushButton("Preview coherent tip near field...")
        self.near_field_button.setToolTip("Calculate this draft in a separate window without changing current instrument or image results. Classical ray sampling is unavailable for this coherent source.")
        self.near_field_button.clicked.connect(self._preview_surface)
        surface_form.addRow(self.near_field_button)
        panel_layout.addWidget(self.surface_panel)
        self.legacy_panel = QWidget()
        form = QFormLayout(self.legacy_panel)
        self.legacy_form = form
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.addRow(QLabel("Historical emission model — not converted to a surface source"))
        panel_layout.addWidget(self.legacy_panel)
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
        self.legacy_coherent_description = description
        form.addRow(description)
        parameters = gun.emitter.coherence or TipCoherence()
        self.coherence_inputs = {}
        for key, label in self.coherence_fields:
            edit = QLineEdit(str(getattr(parameters, key)))
            self.coherence_inputs[key] = edit
            form.addRow(label, edit)
        self.coherence_enabled.toggled.connect(self._sync_fields)
        self.surface_enabled.toggled.connect(self._sync_fields)
        self.surface_coherent.toggled.connect(self._sync_fields)
        self.wave_options.toggled.connect(self._sync_fields)
        self.flux_density_enabled.toggled.connect(self._sync_fields)
        self.energy_law.currentIndexChanged.connect(self._sync_fields)
        self._sync_fields()
        self._surface_summary()
        layout.addWidget(scroll)
        self.error = QLabel()
        self.error.setWordWrap(True)
        self.error.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _sync_fields(self):
        self.surface_panel.setVisible(self.surface_enabled.isChecked())
        self.legacy_panel.setVisible(not self.surface_enabled.isChecked())
        enabled = self.coherence_enabled.isChecked()
        for edit in self.coherence_inputs.values():
            edit.setEnabled(enabled)
            self.legacy_form.setRowVisible(edit, enabled)
        self.legacy_coherent_description.setVisible(enabled)
        self.coherence_enabled.setVisible(enabled)
        for key in ("angular_rms_mrad", "angular_cutoff_mrad"):
            self.inputs[key].setEnabled(not enabled)
        quantum = self.surface_coherent.isChecked()
        # Hiding development controls must not convert an archived source.
        if quantum and not self.wave_options.isChecked():
            self.wave_options.setChecked(True)
        self.surface_form.setRowVisible(self.surface_coherent, self.wave_options.isChecked())
        self.surface_status.setText(
            "Development near-field preview only. Ray Diagram and TEM/STEM imaging are unavailable."
            if quantum else "Classical rays available; no coherent phase for TEM/STEM wave imaging.")
        self.surface_status.setToolTip(
            "Apply saves this source selection even if a calculation is rejected. "
            "Coherent tip/gun boundary, basis and column convergence remain unqualified. "
            "The near-field preview does not change the applied source or previous images.")
        self.surface_form.labelForField(self.surface_inputs["current_na"]).setText(
            "Incoming reservoir current (nA)" if quantum else "Prescribed surface current (nA)")
        density = self.flux_density_enabled.isChecked()
        self.surface_form.setRowVisible(self.surface_inputs["current_na"], not density)
        self.surface_form.setRowVisible(self.surface_inputs["flux_electrons_per_nm2_s"], density)
        law = self.energy_law.currentData()
        self.surface_form.setRowVisible(self.energy_law, not quantum)
        self.surface_form.setRowVisible(self.directions_per_position, not quantum)
        self.surface_form.setRowVisible(self.surface_inputs["maximum_angle_deg"], not quantum)
        for key in ("normal_mean_energy_ev", "tangential_mean_energy_ev"):
            self.surface_form.setRowVisible(self.surface_inputs[key], not quantum and law == "normal_tangential_exponential")
        self.surface_form.setRowVisible(self.surface_inputs["kinetic_mean_ev"], not quantum and law != "normal_tangential_exponential")
        self.surface_form.setRowVisible(self.surface_inputs["kinetic_sigma_ev"], not quantum and law == "gamma")
        for edit in self.quantum_inputs.values():
            self.surface_form.setRowVisible(edit, quantum)
        self.near_field_button.setVisible(quantum)
        self.near_field_button.setEnabled(False)
        self.near_field_button.setToolTip("Wave propagation development is paused. Existing profiles and code are retained; this editor does not start a wave calculation.")
        self._surface_summary()

    def _use_particles(self):
        """Explicit draft conversion; never changes the live instrument itself."""
        self.surface_coherent.setChecked(False)
        self.coherence_enabled.setChecked(False)
        self.wave_options.setChecked(False)
        self.surface_enabled.setChecked(True)
        self._sync_fields()

    def _reload_surface(self):
        try:
            from temsim import module_manifest
            from temsim.optics.electron_gun.tip_assembly import model_from_part
            part = module_manifest.part_data(self._assembly_tip_path, "feg_tip")
            self._surface_draft = model_from_part(part) or load_tip_surface_reference(self._reference_path)
            for key, edit in self.geometry_inputs.items():
                edit.setText(str(getattr(self._surface_draft.geometry, key)))
            for key, edit in self.surface_inputs.items():
                value = getattr(self._surface_draft.emission, key)
                edit.setText(str(value if value is not None else 0.0))
            self.flux_density_enabled.setChecked(self._surface_draft.emission.flux_electrons_per_nm2_s is not None)
            self.energy_law.setCurrentIndex(self.energy_law.findData(self._surface_draft.emission.energy_distribution))
            self.directions_per_position.setValue(self._surface_draft.emission.directions_per_position)
            self.surface_coherent.setChecked(self._surface_draft.coherence is not None)
            quantum = self._surface_draft.coherence or SurfaceCoherence()
            for key, edit in self.quantum_inputs.items():
                edit.setText(str(getattr(quantum, key)))
            self._surface_summary()
        except (OSError, ValueError) as error:
            self.error.setText(str(error))

    def _surface_value(self):
        quantum = self.surface_coherent.isChecked()
        density = self.flux_density_enabled.isChecked()
        active = {"cap_half_angle_deg", "flux_electrons_per_nm2_s" if density else "current_na"}
        law = self.energy_law.currentData()
        if not quantum:
            active.add("maximum_angle_deg")
            active.update(("normal_mean_energy_ev", "tangential_mean_energy_ev") if law == "normal_tangential_exponential"
                          else ("kinetic_mean_ev", "kinetic_sigma_ev") if law == "gamma" else ("kinetic_mean_ev",))
        values = {key: float(self.surface_inputs[key].text()) for key in active}
        values["current_na" if density else "flux_electrons_per_nm2_s"] = None
        emission = replace(self._surface_draft.emission,
                           **values, energy_distribution=law,
                           directions_per_position=self.directions_per_position.value())
        coherence = SurfaceCoherence(**{key: float(edit.text()) for key, edit in self.quantum_inputs.items()}) if quantum else None
        geometry = replace(self._surface_draft.geometry,
                           **{key: float(edit.text()) for key, edit in self.geometry_inputs.items()})
        return replace(self._surface_draft, geometry=geometry, emission=emission, coherence=coherence).validate()

    def _surface_summary(self):
        if not hasattr(self, "surface_derived"):
            return
        try:
            value = self._surface_value()
            g, e = value.geometry, value.emission
            dimensions = patch_dimensions(g, e.cap_half_angle_deg)
            self.geometry_preview.set_model(value)
            self.surface_geometry.setText(f"{g.material} | spherical cap + tangent cone | reference, uncalibrated")
            self.patch_summary.setText(
                f"Diameter {dimensions['projected_diameter_nm']:.5g} nm | depth {dimensions['cap_depth_nm']:.5g} nm\n"
                f"Half-angle {dimensions['half_angle_rad']:.5g} rad | apex-to-edge arc {dimensions['apex_to_edge_arc_nm']:.5g} nm")
            self.patch_summary.setToolTip(
                f"Derived from radius R and cap half-angle θ; not independent inputs. Curvature = 1/R = {dimensions['apex_curvature_nm_inv']:.5g} nm⁻¹. "
                "Diameter = 2R sin θ; depth = R(1−cos θ); apex-to-edge arc = Rθ; area = 2πR²(1−cos θ). "
                "All angles are geometric. Normal/tangential energy components below define the particle direction distribution.")
            self.surface_derived.setToolTip(self._classical_surface_tooltip)
            self.surface_derived.setText(f"Mean energy {e.mean_energy_ev:g} eV | energy RMS {e.energy_sigma_ev:.4g} eV\n"
                f"Patch area {e.area_nm2(g):.4g} nm² | total current {value.current_na:.5g} nA\n"
                f"Maximum outgoing angle {e.maximum_angle_deg:g}° from the local normal")
            if value.coherence is not None:
                self.surface_derived.setText(f"Patch area {e.area_nm2(g):.4g} nm² | one spatial mode per energy\n"
                    "Angles follow the wave; not an independent angular spread. Different energies are incoherent.")
                self.surface_derived.setToolTip("Prescribed post-emission reservoir, not a tunnelling prediction. Cosine-squared surface amplitude, quadratic surface-arc phase, positive gamma total-energy law. Normal/tangential classical energy inputs are inactive. No single phase exists for the energy mixture.")
            ht, ext = self._gun.accelerator.high_tension_kv, self._gun.extractor.voltage_kv
            lens = self._gun.electrostatic_lens
            lens_ground_kv = lens.potential_rise_from_tip_v(ext,ht)/1000-ht
            self.surface_voltage.setText(f"Final anode 0 V | Tip {-ht:g} kV | Extractor {-ht+ext:g} kV\n"
                f"Gun lens {lens_ground_kv:g} kV | control relative to {lens.voltage_reference}")
            self.surface_voltage.setToolTip(
                "Displayed electrode potentials are relative to final-anode ground. "
                "Extraction is relative to the tip. Gun-lens voltage_reference is set on the lens: "
                "tip, extractor or ground. Changing this reference changes the physical field; "
                "a microscope's displayed lens voltage is not necessarily ground-referenced. "
                "Radial focusing and axial acceleration follow the same joint electrostatic potential.")
        except (TypeError, ValueError) as error:
            self.surface_derived.setText(str(error))
            self.patch_summary.setText("Invalid geometry or emission inputs")
            self.geometry_preview.set_model(None)

    def _preview_surface(self):
        try:
            from temsim.gui.surface_wave_dialog import SurfaceWaveDialog
            from temsim.instrument_snapshot import decode_instrument, encode_instrument
            # Installed assemblies contain immutable mapping proxies. Preserve
            # the complete live graph without deepcopy or manifest reloading.
            state = decode_instrument(encode_instrument(self._instrument_state))
            state.electron_gun.emitter.surface_model = self._surface_value()
            state.electron_gun.emitter.coherence = None
            state.electron_gun.source_representation = "classical_particles"
            dialog = SurfaceWaveDialog(state, self)
            dialog.exec()
        except (TypeError, ValueError, RuntimeError) as error:
            self.error.setText(str(error))

    def accept(self):
        candidate = copy(self._gun.emitter)
        try:
            if self.surface_enabled.isChecked():
                model = self._surface_value()
                candidate.coherence = None
                candidate.surface_model = model
                candidate.validate()
                self._value = {"coherence": None, "surface_model": model}
                super().accept()
                return
            candidate.surface_model = None
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
            values["surface_model"] = None
        except (TypeError, ValueError) as error:
            self.error.setText(str(error))
            return
        self._value = values
        super().accept()

    def value(self):
        if self._value is None:
            raise ValueError("No tip emission values were applied")
        return dict(self._value)
