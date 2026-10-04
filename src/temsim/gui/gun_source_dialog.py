"""Draft edits to the physical FEG tip; no independent exit-source controls."""
from copy import copy
from dataclasses import replace
import math
from pathlib import Path

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QScrollArea, QVBoxLayout, QWidget, QPushButton, QSpinBox
from temsim.optics.electron_gun.tip_coherence import TipCoherence, tip_covariance
from temsim.optics.electron_gun.tip_surface import SharedSurfaceCoherence, SurfaceCoherence, load_tip_surface_reference, reference_path_for_gun
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
        ("curvature_nm_inv", "Tip curvature (nm⁻¹; 0 = flat)"),
        ("emission_current_na", "Tip emission current (nA)"),
        ("emission_energy_ev", "Tip launch mean kinetic energy (eV)"),
        ("minimum_kinetic_energy_ev", "Minimum launch kinetic energy (eV)"),
        ("virtual_source_fwhm_nm", "Projected emission FWHM (nm)"),
        ("angular_rms_mrad", "Local angular RMS (mrad)"),
        ("angular_cutoff_mrad", "Local angular cutoff (mrad)"),
        ("energy_spread_fwhm_ev", "Tip energy width, RMS-equivalent FWHM (eV)"),
        ("young_decay_width_ev", "Young energy decay width (eV)"),
        ("boersch_sigma_ev", "Boersch energy sigma (eV)"),
        ("energy_half_range_ev", "Energy sampling half-range (eV)"),
    )

    def __init__(self, gun, parent=None, *, instrument_state=None):
        super().__init__(parent)
        if gun.type_key != "cold_feg":
            raise ValueError("This editor controls FEG tip emission only")
        self._gun = gun
        self._curvature_model = gun.emitter.curvature_model
        self._instrument_state = instrument_state
        self._value = None
        self.edit_dimensions_requested = False
        self.setWindowTitle("Physical Layout · Tip parameters")
        self.resize(680, 760)
        layout = QVBoxLayout(self)
        note = QLabel(
            "Tip emission settings. Curvature is adjustable here and in Live tuning; 0 is flat."
        )
        note.setWordWrap(True)
        note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(note)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        self.particle_button = QPushButton("Use continuous tip · start flat")
        self.particle_button.setToolTip("Explicitly select tip emission at curvature 0 without a phase model. Apply is required. Existing files are not converted.")
        self.particle_button.clicked.connect(self._use_particles)
        panel_layout.addWidget(self.particle_button)
        self.surface_enabled = QCheckBox("Curved metal tip with electrode fields")
        self.surface_enabled.setToolTip(
            "Switches emission AND the gun field/integrator, not curvature alone. "
            "Both models start at the tip and retain all gun components.")
        self.surface_enabled.setChecked(gun.emitter.surface_model is not None)
        panel_layout.addWidget(self.surface_enabled)
        self.model_change_summary = QLabel()
        self.model_change_summary.setWordWrap(True)
        self.model_change_summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        panel_layout.addWidget(self.model_change_summary)
        self.surface_panel = QWidget()
        surface_form = QFormLayout(self.surface_panel)
        self.surface_form = surface_form
        surface_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.match_transport = QCheckBox("Match C1 / C2 / C3 transport after applying")
        self.match_transport.setChecked(gun.emitter.surface_model is None and instrument_state is not None)
        self.match_transport.setToolTip(
            "Recalculate only the three condenser lens strengths for this source. "
            "Physical dimensions, gun voltages and apertures are retained. "
            "Transport recovery does not calibrate probe focus or images.")
        surface_form.addRow(self.match_transport)
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
        self._historical_reservoir = isinstance(self._surface_draft.coherence, SurfaceCoherence)
        reference = QLabel(str(self._assembly_tip_path) + " : feg_tip")
        reference.setWordWrap(True)
        reference.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        surface_form.addRow("Tip TOML", reference)
        self.surface_geometry = QLabel()
        self.surface_geometry.setWordWrap(True)
        self.surface_geometry.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        surface_form.addRow(self.surface_geometry)
        self.reload_reference = QPushButton("Reload installed tip TOML defaults")
        self.reload_reference.clicked.connect(self._reload_surface)
        surface_form.addRow(self.reload_reference)
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
        for edit in self.geometry_inputs.values():
            edit.setReadOnly(True)
        self.dimensions_button = QPushButton("Edit dimensions in Physical Layout…")
        self.dimensions_button.clicked.connect(self._open_dimensions)
        surface_form.addRow(self.dimensions_button)
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
        surface_form.addRow("Tip energy distribution", self.energy_law)
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
        self.wave_options = QCheckBox("Show phase and coherence parameters")
        self.wave_options.setChecked(self._surface_draft.coherence is not None)
        surface_form.addRow(self.wave_options)
        self.surface_coherent = QCheckBox("Constant-phase surface emission")
        self.surface_coherent.setToolTip(
            "Explicitly selects cosine-cap injected flux and a gamma or monoenergetic tip energy law. "
            "Surface phase is constant. Rays launch along local normals as a geometric-optics approximation; "
            "they do not reproduce wave diffraction or reflection.")
        self.surface_coherent.setChecked(self._surface_draft.coherence is not None)
        surface_form.addRow(self.surface_coherent)
        self.replace_reservoir_button = QPushButton("Replace historical reservoir with Tip parameters")
        self.replace_reservoir_button.setToolTip(
            "Explicit draft replacement. The saved reservoir mean and RMS become the single tip energy law; "
            "phase becomes constant and rays use local-normal geometric launch. Apply is required.")
        self.replace_reservoir_button.clicked.connect(self._replace_historical_reservoir)
        surface_form.addRow(self.replace_reservoir_button)
        self.quantum_inputs = {}
        quantum = self._surface_draft.coherence if self._historical_reservoir else SurfaceCoherence()
        for key, label in (("mean_energy_ev", "Historical reservoir mean energy (eV)"),
                           ("energy_rms_ev", "Historical reservoir energy RMS (eV)"),
                           ("edge_phase_rad", "Tip surface phase difference (rad; fixed 0)")):
            edit = QLineEdit(str(getattr(quantum, key)))
            edit.setReadOnly(True)
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
        self.near_field_button = QPushButton("Preview tip near field...")
        self.near_field_button.setToolTip("Calculate this draft in a separate window without changing applied Tip parameters or image results.")
        self.near_field_button.clicked.connect(self._preview_surface)
        surface_form.addRow(self.near_field_button)
        panel_layout.addWidget(self.surface_panel)
        self.analytic_tip_panel = QWidget()
        form = QFormLayout(self.analytic_tip_panel)
        self.analytic_tip_form = form
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.addRow(QLabel("Tip emission · projected size and local emission law"))
        panel_layout.addWidget(self.analytic_tip_panel)
        scroll.setWidget(panel)
        self.inputs = {}
        for key, label in self.fields:
            edit = QLineEdit(str(getattr(gun.emitter, key)))
            from temsim.parameter_registry import parameter_definition
            definition = parameter_definition("feg_tip", key)
            if definition is not None:
                edit.setToolTip(f"{definition.label} ({definition.unit or 'dimensionless'})\n{definition.description}")
            self.inputs[key] = edit
            form.addRow(label, edit)
        self.inputs["curvature_nm_inv"].setToolTip(
            "Curvature κ = 1/R. Start from 0 with small increments such as 1e-8 nm^-1. "
            "The tip centre remains at Z = 0; off-axis points bend upstream to negative Z. "
            "Directions follow the local surface normals. Historical models retain their saved geometry. "
            "Projected D95, total current and launch energies remain fixed. "
            "The analytic gun field is unchanged, not solved again for the new metal boundary.")
        self.inputs["curvature_nm_inv"].textChanged.connect(self._sync_fields)
        self.inputs["virtual_source_fwhm_nm"].textChanged.connect(self._sync_fields)
        self.continuous_preview = TipGeometryPreview()
        form.addRow(self.continuous_preview)
        self.coherence_enabled = QCheckBox("Gaussian-Schell emission")
        self.coherence_enabled.setChecked(gun.emitter.coherence is not None)
        form.addRow(self.coherence_enabled)
        self.tip_boundary = QComboBox()
        self.tip_boundary.addItem("Driven emission · non-paraxial near tip", "driven_gaussian_schell")
        self.tip_boundary.addItem("Forward emission · paraxial", "forward_gaussian_schell")
        self.tip_boundary.setCurrentIndex(self.tip_boundary.findData(
            gun.emitter.coherence.boundary_model if gun.emitter.coherence else "driven_gaussian_schell"))
        form.addRow("Tip boundary", self.tip_boundary)
        description = QLabel(
            "Defines the tip phase and partial coherence with an untruncated Gaussian-Schell distribution. "
            "Ray Diagram and wave calculations read these same Tip parameters. "
            "Spatial FWHM, current and energy inputs above still apply. Local angular RMS/cutoff apply only "
            "when this option is off. Driven emission uses geometric rays of the local phase directions; "
            "the wave retains diffraction and reflection. Forward emission uses its paraxial Wigner distribution. "
            "All physical apertures remain active. Full TEM/STEM wave imaging is still under development.")
        description.setWordWrap(True)
        self.coherence_description = description
        form.addRow(description)
        parameters = gun.emitter.coherence or TipCoherence()
        self.coherence_inputs = {}
        for key, label in self.coherence_fields:
            edit = QLineEdit(str(getattr(parameters, key)))
            self.coherence_inputs[key] = edit
            form.addRow(label, edit)
        self.coherence_enabled.toggled.connect(self._sync_fields)
        self.surface_enabled.toggled.connect(self._sync_fields)
        self.surface_coherent.toggled.connect(self._surface_coherence_changed)
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
        text = self.inputs['curvature_nm_inv'].text()
        scope = "Centre Z = 0 · edges bend to negative Z"
        self.model_change_summary.setText(
            "Curved metal tip: emission on the physical cap; electrode fields follow its geometry."
            if self.surface_enabled.isChecked() else
            f"Tip curvature {text} nm⁻¹ · Flat tip at 0 · {scope}")
        self.model_change_summary.setToolTip(
            "Tip inputs specify emission positions, local directions, energies and current. "
            "The flat tip uses one coupled electrode field for extraction, gun focusing and acceleration, "
            "including the grounded downstream liner. Gun-lens voltage is the electrode potential "
            "relative to its selected reference; an analytic focusing multiplier is not used. "
            "A deformed conductor requires its corresponding curved-tip field. "
            "Source and field parameters remain uncalibrated for real microscope performance.")
        self.surface_panel.setVisible(self.surface_enabled.isChecked())
        self.analytic_tip_panel.setVisible(not self.surface_enabled.isChecked())
        self.match_transport.setEnabled(self._instrument_state is not None
                                       and not self._historical_reservoir
                                       and not self.surface_coherent.isChecked())
        enabled = self.coherence_enabled.isChecked()
        for edit in self.coherence_inputs.values():
            edit.setEnabled(enabled)
            self.analytic_tip_form.setRowVisible(edit, enabled)
        self.coherence_description.setVisible(enabled)
        self.tip_boundary.setEnabled(enabled)
        self.analytic_tip_form.setRowVisible(self.tip_boundary, enabled)
        # The model selector must remain reachable before phase is defined.
        # Its parent selects the analytic geometry; only its fields are optional.
        self.coherence_enabled.setVisible(True)
        self.inputs["curvature_nm_inv"].setEnabled(not enabled)
        self.continuous_preview.setVisible(not enabled)
        try:
            preview = copy(self._gun.emitter)
            preview.surface_model = preview.coherence = None
            preview.curvature_nm_inv = float(text)
            preview.curvature_model = self._curvature_model
            preview.virtual_source_fwhm_nm = float(self.inputs["virtual_source_fwhm_nm"].text())
            preview.validate()
            self.continuous_preview.set_continuous_tip(preview)
        except (ValueError, TypeError):
            self.continuous_preview.set_model(None)
        for key in ("angular_rms_mrad", "angular_cutoff_mrad"):
            self.inputs[key].setEnabled(not enabled)
        quantum = self.surface_coherent.isChecked()
        # Hiding development controls must not convert an archived source.
        if quantum and not self.wave_options.isChecked():
            self.wave_options.setChecked(True)
        self.surface_form.setRowVisible(self.surface_coherent, self.wave_options.isChecked())
        historical = self._historical_reservoir
        self.surface_coherent.setText("Historical surface reservoir (read-only)" if historical else
                                      "Constant-phase surface emission")
        self.surface_coherent.setEnabled(not historical)
        self.surface_enabled.setEnabled(not historical)
        self.replace_reservoir_button.setVisible(historical)
        self.reload_reference.setEnabled(not historical)
        self.dimensions_button.setEnabled(not historical)
        self.flux_density_enabled.setEnabled(not historical)
        self.flux_density_enabled.setText("Prescribe mean injected flux over the emitting cap" if quantum else
                                         "Prescribe electrons per unit surface area per second")
        self.surface_form.labelForField(self.surface_inputs["flux_electrons_per_nm2_s"]).setText(
            "Mean injected electrons / (nm² s)" if quantum else "Emitted electrons / (nm² s)")
        self.surface_inputs["flux_electrons_per_nm2_s"].setToolTip(
            "Mean flux density over the complete cap area; multiplied by cap area to obtain injection-reference "
            "current. The cosine-cap profile redistributes this prescribed current over the cap; "
            "it is not a peak flux density or a prediction of escaping current." if quantum else
            "Prescribed uniform outgoing flux per unit surface area. Total current follows the emitting cap area.")
        self.directions_per_position.setEnabled(not historical)
        for edit in self.surface_inputs.values():
            edit.setReadOnly(historical)
        self.energy_law.setEnabled(not historical)
        self.energy_law.model().item(self.energy_law.findData("normal_tangential_exponential")).setEnabled(not quantum)
        if historical:
            status = ("Historical reservoir retained read-only. Ray Diagram is unavailable for this model. "
                      "Use the explicit replacement button to define Tip parameters.")
        elif quantum:
            status = ("Tip phase is defined: constant surface phase. Ray Diagram uses local-normal "
                      "geometric launch; diffraction and reflected flux require wave calculation. "
                      "Full TEM/STEM propagation remains under development.")
        else:
            status = "Tip phase is not defined. Ray Diagram is available; wave imaging requires a phase model."
        self.surface_status.setText(status)
        self.surface_status.setToolTip(
            "Apply saves these Tip parameters even if a calculation is rejected. "
            "Wave propagation through the tip/gun boundary, basis and column convergence remain unqualified. "
            "The near-field preview does not change applied Tip parameters or previous images.")
        self.surface_form.labelForField(self.surface_inputs["current_na"]).setText(
            "Tip injection-reference current (nA)" if quantum else "Prescribed surface current (nA)")
        density = self.flux_density_enabled.isChecked()
        self.surface_form.setRowVisible(self.surface_inputs["current_na"], not density)
        self.surface_form.setRowVisible(self.surface_inputs["flux_electrons_per_nm2_s"], density)
        law = self.energy_law.currentData()
        self.surface_form.setRowVisible(self.energy_law, not historical)
        self.surface_form.setRowVisible(self.directions_per_position, not quantum)
        self.surface_form.setRowVisible(self.surface_inputs["maximum_angle_deg"], not quantum)
        for key in ("normal_mean_energy_ev", "tangential_mean_energy_ev"):
            self.surface_form.setRowVisible(self.surface_inputs[key], not quantum and law == "normal_tangential_exponential")
        self.surface_form.setRowVisible(self.surface_inputs["kinetic_mean_ev"], not historical and law != "normal_tangential_exponential")
        self.surface_form.setRowVisible(self.surface_inputs["kinetic_sigma_ev"], not historical and law == "gamma")
        for key, edit in self.quantum_inputs.items():
            self.surface_form.setRowVisible(edit, quantum if key == "edge_phase_rad" else historical)
        self.surface_form.labelForField(self.quantum_inputs["edge_phase_rad"]).setText(
            "Historical surface edge phase (rad)" if historical else
            "Tip surface phase difference (rad; fixed 0)")
        self.near_field_button.setVisible(quantum)
        self.near_field_button.setEnabled(False)
        self.near_field_button.setToolTip("Calculation here remains paused. Calculate waves in the Coherent beam page using these Tip parameters. This editor does not start propagation or qualify a TEM/STEM image.")
        self._surface_summary()

    def _surface_coherence_changed(self, checked):
        if (checked and not self._historical_reservoir
                and (not self._surface_draft.shared_boundary
                     or self.energy_law.currentData() == "normal_tangential_exponential")):
            try:
                emission = self._surface_emission_value()
                self._select_shared_boundary(emission.mean_energy_ev, emission.energy_sigma_ev)
            except (TypeError, ValueError) as error:
                self.error.setText(str(error))
        self._sync_fields()

    def _select_shared_boundary(self, mean_ev, rms_ev):
        """An explicit draft model selection, with no live-state modification."""
        law = "gamma" if rms_ev > 0 else "monoenergetic"
        emission = replace(self._surface_draft.emission,
                           energy_distribution=law, kinetic_mean_ev=mean_ev, kinetic_sigma_ev=rms_ev,
                           flux_profile="cosine_cap", maximum_angle_deg=0.0,
                           spatial_sampling="uniform_area", spatial_stratum_allocation=(),
                           angular_sampling="uniform_cdf", angular_stratum_allocation=(),
                           angular_refinement_gain=0.0)
        self._surface_draft = replace(self._surface_draft, emission=emission,
                                      coherence=SharedSurfaceCoherence()).validate()
        self._historical_reservoir = False
        with QSignalBlocker(self.energy_law):
            self.energy_law.setCurrentIndex(self.energy_law.findData(law))
        self.surface_inputs["kinetic_mean_ev"].setText(str(mean_ev))
        self.surface_inputs["kinetic_sigma_ev"].setText(str(rms_ev))
        self.surface_inputs["maximum_angle_deg"].setText("0")
        self.quantum_inputs["edge_phase_rad"].setText("0")

    def _replace_historical_reservoir(self):
        if not self._historical_reservoir:
            return
        self._select_shared_boundary(self._surface_draft.mean_energy_ev,
                                     self._surface_draft.energy_sigma_ev)
        self.surface_coherent.setChecked(True)
        self._sync_fields()

    def _use_particles(self):
        """Explicit draft conversion; never changes the live instrument itself."""
        from temsim.optics.electron_gun.tip_curvature import MODEL
        self._curvature_model = MODEL
        self._historical_reservoir = False
        self._surface_draft = replace(self._surface_draft, coherence=None)
        self.surface_coherent.setChecked(False)
        self.coherence_enabled.setChecked(False)
        self.wave_options.setChecked(False)
        self.inputs["curvature_nm_inv"].setText("0")
        self.surface_enabled.setChecked(False)
        self._sync_fields()

    def _open_dimensions(self):
        self.edit_dimensions_requested = True
        self.reject()

    def _reload_surface(self):
        try:
            from temsim import module_manifest
            from temsim.optics.electron_gun.tip_assembly import model_from_part
            part = module_manifest.part_data(self._assembly_tip_path, "feg_tip")
            self._surface_draft = model_from_part(part) or load_tip_surface_reference(self._reference_path)
            self._historical_reservoir = isinstance(self._surface_draft.coherence, SurfaceCoherence)
            for key, edit in self.geometry_inputs.items():
                edit.setText(str(getattr(self._surface_draft.geometry, key)))
            for key, edit in self.surface_inputs.items():
                value = getattr(self._surface_draft.emission, key)
                edit.setText(str(value if value is not None else 0.0))
            self.flux_density_enabled.setChecked(self._surface_draft.emission.flux_electrons_per_nm2_s is not None)
            self.energy_law.setCurrentIndex(self.energy_law.findData(self._surface_draft.emission.energy_distribution))
            self.directions_per_position.setValue(self._surface_draft.emission.directions_per_position)
            with QSignalBlocker(self.surface_coherent):
                self.surface_coherent.setChecked(self._surface_draft.coherence is not None)
            quantum = self._surface_draft.coherence if self._historical_reservoir else SurfaceCoherence()
            for key, edit in self.quantum_inputs.items():
                edit.setText(str(getattr(quantum, key)))
            self._sync_fields()
        except (OSError, ValueError) as error:
            self.error.setText(str(error))

    def _surface_emission_value(self):
        density = self.flux_density_enabled.isChecked()
        active = {"cap_half_angle_deg", "flux_electrons_per_nm2_s" if density else "current_na"}
        law = self.energy_law.currentData()
        shared = self.surface_coherent.isChecked() and self._surface_draft.shared_boundary
        if not shared:
            active.add("maximum_angle_deg")
        active.update(("normal_mean_energy_ev", "tangential_mean_energy_ev") if law == "normal_tangential_exponential"
                      else ("kinetic_mean_ev", "kinetic_sigma_ev") if law == "gamma" else ("kinetic_mean_ev",))
        values = {key: float(self.surface_inputs[key].text()) for key in active}
        if shared:
            values["maximum_angle_deg"] = 0.0
        values["current_na" if density else "flux_electrons_per_nm2_s"] = None
        return replace(self._surface_draft.emission,
                       **values, energy_distribution=law,
                       directions_per_position=self.directions_per_position.value())

    def _surface_value(self):
        if self._historical_reservoir:
            return self._surface_draft.validate()
        quantum = self.surface_coherent.isChecked()
        emission = self._surface_emission_value()
        if quantum:
            emission = replace(emission, flux_profile="cosine_cap", maximum_angle_deg=0.0,
                               spatial_sampling="uniform_area", spatial_stratum_allocation=(),
                               angular_sampling="uniform_cdf", angular_stratum_allocation=(),
                               angular_refinement_gain=0.0)
        coherence = SharedSurfaceCoherence(edge_phase_rad=float(self.quantum_inputs["edge_phase_rad"].text())) if quantum else None
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
                "All angles are geometric. Constant-phase surface emission uses local-normal launch. "
                "Without a phase model, the tip uses its configured local angular law.")
            self.surface_derived.setToolTip(self._classical_surface_tooltip)
            self.surface_derived.setText(f"Mean energy {e.mean_energy_ev:g} eV | energy RMS {e.energy_sigma_ev:.4g} eV\n"
                f"Patch area {e.area_nm2(g):.4g} nm² | total current {value.current_na:.5g} nA\n"
                f"Maximum outgoing angle {e.maximum_angle_deg:g}° from the local normal")
            if value.shared_boundary:
                self.surface_derived.setText(
                    f"Tip mean energy {value.mean_energy_ev:g} eV | energy RMS {value.energy_sigma_ev:.4g} eV\n"
                    f"Patch area {e.area_nm2(g):.4g} nm² | injection-reference current {value.current_na:.5g} nA\n"
                    "Cosine-cap flux | constant phase 0 | rays launch along local normals")
                self.surface_derived.setToolTip(
                    "One prescribed physical cap boundary supplies the particle and wave models. "
                    "The emitted flux density is proportional to cos⁴(π surface-arc / (2 cap-edge arc)); "
                    "amplitude is its square root. One spatial mode per energy, with incoherent energy mixing. "
                    "Classical rays are a local-normal geometric-optics approximation, not an exact quantum "
                    "phase-space distribution. Wave reflection can reduce net escaping current. No aggregate "
                    "phase exists for the energy mixture; this model does not predict tunnelling current.")
            elif value.coherence is not None:
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
            state.electron_gun.emitter.curvature_nm_inv = 0.0
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
                candidate.curvature_nm_inv = 0.0
                candidate.surface_model = model
                candidate.validate()
                self._value = {"coherence": None, "surface_model": model, "curvature_nm_inv": 0.0}
                from temsim.optics.electron_gun.tip_edit import candidate_tip_edit
                candidate_tip_edit(self._gun, self._value)
                super().accept()
                return
            candidate.surface_model = None
            values = {key: float(self.inputs[key].text()) for key, _ in self.fields}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError("Tip emission values must be finite")
            for key, value in values.items():
                setattr(candidate, key, value)
            candidate.curvature_model = self._curvature_model
            values["curvature_model"] = candidate.curvature_model
            candidate.coherence = (TipCoherence(boundary_model=self.tip_boundary.currentData(),
                                   **{key: float(edit.text()) for key, edit in self.coherence_inputs.items()})
                                   if self.coherence_enabled.isChecked() else None)
            candidate.validate()
            if candidate.coherence is not None:
                tip_covariance(candidate, candidate.emission_energy_ev)
                from temsim.optics.electron_gun.tip_source_domain import validate_tip_particle_domain
                validate_tip_particle_domain(candidate)
            values["coherence"] = candidate.coherence
            values["surface_model"] = None
            from temsim.optics.electron_gun.tip_edit import candidate_tip_edit
            candidate_tip_edit(self._gun, values)
        except (TypeError, ValueError) as error:
            self._value = None
            self.error.setText(str(error))
            return
        self._value = values
        super().accept()

    def value(self):
        if self._value is None:
            raise ValueError("No tip emission values were applied")
        return dict(self._value)

    @property
    def match_transport_requested(self):
        return (self.surface_enabled.isChecked() and self.match_transport.isEnabled()
                and self.match_transport.isChecked() and not self.surface_coherent.isChecked())
