"""Operating, TOML and anchor parameters for the selected component."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from temsim.component_keys import ENERGY_FILTER_SLIT
from temsim.manifest_editor import format_toml_value, parse_toml_value
from temsim.runtime_parameters import (
    convert_runtime_value,
    editable_parameters,
    validate_runtime_assignment,
)


class ParameterPanel(QWidget):
    runtime_changed = Signal(str)
    energy_filter_match_requested = Signal()
    manifest_save_requested = Signal(object, object)
    error = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("parameterPanel")
        self.setMinimumWidth(380)
        self._runtime_target = None
        self._manifest_target = None
        self._manifest_fields = ()
        self._updating = False

        self.title = QLabel("Parameters")
        self.title.setObjectName("parameterTitle")
        font = self.title.font()
        font.setBold(True)
        font.setPointSize(font.pointSize() + 2)
        self.title.setFont(font)

        self.lens_box = QGroupBox("Live lens control")
        lens_form = QFormLayout(self.lens_box)
        self.lens_enabled = QCheckBox("Enabled")
        self.lens_excitation = QDoubleSpinBox()
        self.lens_excitation.setDecimals(5)
        self.lens_excitation.setRange(0.0, 1000.0)
        self.lens_excitation.setSuffix(" %")
        self.lens_cs = QDoubleSpinBox()
        self.lens_cs.setDecimals(6)
        self.lens_cs.setRange(-1.0e6, 1.0e6)
        self.lens_cs.setSuffix(" mm")
        self.lens_cs.setToolTip(
            "Signed third-order spherical aberration coefficient. "
            "Zero disables the calibrated ray-direction correction."
        )
        self.lens_field_direction = QComboBox()
        self.lens_field_direction.addItem("+Z", 1)
        self.lens_field_direction.addItem("-Z", -1)
        self.lens_field_direction.setToolTip(
            "Effective direction of the on-axis magnetic field in the "
            "right-handed simulation coordinate system. The selected "
            "assembly TOML supplies the default; changing this control is a "
            "runtime override and does not edit the manifest."
        )
        self.lens_diagnostics = QLabel(
            "Recalculate to update field and focal diagnostics."
        )
        self.lens_diagnostics.setWordWrap(True)
        self.lens_diagnostics.setStyleSheet(
            "color: #475569; font-weight: 600;"
        )
        lens_form.addRow(self.lens_enabled)
        lens_form.addRow("Excitation", self.lens_excitation)
        lens_form.addRow("Spherical aberration Cs", self.lens_cs)
        lens_form.addRow(
            "Effective axial field direction", self.lens_field_direction
        )
        lens_form.addRow(self.lens_diagnostics)
        self.lens_enabled.toggled.connect(self._lens_enabled_changed)
        self.lens_excitation.valueChanged.connect(self._lens_excitation_changed)
        self.lens_cs.valueChanged.connect(self._lens_cs_changed)
        self.lens_field_direction.currentIndexChanged.connect(
            self._lens_field_direction_changed
        )
        self.lens_box.hide()

        self.quick_box = QGroupBox("Device quick controls")
        self.quick_form = QFormLayout(self.quick_box)
        self._quick_widgets = {}
        self.quick_box.hide()

        self.energy_filter_box = QGroupBox(
            "Iliad reference acquisition controls"
        )
        energy_filter_form = QFormLayout(self.energy_filter_box)
        self.energy_filter_enabled = QCheckBox("Optical branch enabled")
        self.energy_filter_mode = QComboBox()
        self.energy_filter_mode.addItem("EELS / Zebra", "eels")
        self.energy_filter_mode.addItem("EFTEM / filtered image", "eftem")
        self.energy_filter_multi_eels = QCheckBox("Enable MultiEELS")
        self.energy_filter_regions = QSpinBox()
        self.energy_filter_regions.setRange(1, 5)
        self.energy_filter_selected_loss = QDoubleSpinBox()
        self.energy_filter_selected_loss.setRange(-4_000.0, 4_000.0)
        self.energy_filter_selected_loss.setDecimals(3)
        self.energy_filter_selected_loss.setSuffix(" eV")
        self.energy_filter_slit_width = QDoubleSpinBox()
        self.energy_filter_slit_width.setRange(0.0, 4_000.0)
        self.energy_filter_slit_width.setDecimals(3)
        self.energy_filter_slit_width.setSuffix(" eV")
        self.energy_filter_active_strip = QSpinBox()
        self.energy_filter_active_strip.setRange(1, 5)
        self.energy_filter_bias = QDoubleSpinBox()
        self.energy_filter_bias.setRange(-4_000.0, 4_000.0)
        self.energy_filter_bias.setDecimals(3)
        self.energy_filter_bias.setSuffix(" eV")
        self.energy_filter_alignment = QCheckBox("Use 2-D alignment area")
        self.energy_filter_shutter = QCheckBox("Fast shutter open")
        self.energy_filter_status = QLabel()
        self.energy_filter_status.setWordWrap(True)
        self.energy_filter_status.setStyleSheet(
            "color: #475569; font-weight: 600;"
        )
        energy_filter_form.addRow(self.energy_filter_enabled)
        energy_filter_form.addRow("Acquisition", self.energy_filter_mode)
        energy_filter_form.addRow(self.energy_filter_multi_eels)
        energy_filter_form.addRow(
            "Spectrum ranges", self.energy_filter_regions
        )
        energy_filter_form.addRow(
            "Selected loss", self.energy_filter_selected_loss
        )
        energy_filter_form.addRow(
            "EFTEM slit width", self.energy_filter_slit_width
        )
        energy_filter_form.addRow(
            "Zebra active strip", self.energy_filter_active_strip
        )
        energy_filter_form.addRow(
            "MultiEELS bias", self.energy_filter_bias
        )
        energy_filter_form.addRow(self.energy_filter_alignment)
        energy_filter_form.addRow(self.energy_filter_shutter)
        self.energy_filter_match = QPushButton(
            "Match prism + M01-M10 to current HT"
        )
        self.energy_filter_match.setToolTip(
            "Scale all magnetic fields by relativistic rigidity, then "
            "re-measure dispersion and place the physical slit"
        )
        energy_filter_form.addRow(self.energy_filter_match)
        energy_filter_form.addRow(self.energy_filter_status)
        for widget, signal in (
            (self.energy_filter_enabled, self.energy_filter_enabled.toggled),
            (self.energy_filter_mode, self.energy_filter_mode.currentIndexChanged),
            (self.energy_filter_multi_eels, self.energy_filter_multi_eels.toggled),
            (self.energy_filter_regions, self.energy_filter_regions.valueChanged),
            (self.energy_filter_selected_loss, self.energy_filter_selected_loss.valueChanged),
            (self.energy_filter_slit_width, self.energy_filter_slit_width.valueChanged),
            (self.energy_filter_active_strip, self.energy_filter_active_strip.valueChanged),
            (self.energy_filter_bias, self.energy_filter_bias.valueChanged),
            (self.energy_filter_alignment, self.energy_filter_alignment.toggled),
            (self.energy_filter_shutter, self.energy_filter_shutter.toggled),
        ):
            signal.connect(self._energy_filter_controls_changed)
        self.energy_filter_match.clicked.connect(
            self.energy_filter_match_requested.emit
        )
        self.energy_filter_box.hide()

        self.tabs = QTabWidget()
        self.runtime_table = self._table(("Operating parameter", "Value"))
        self.manifest_table = self._table(("TOML parameter", "Value"))
        self.anchor_table = self._table(("Anchor property", "Value"))
        self.anchor_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tabs.addTab(self.runtime_table, "Operating")

        manifest_page = QWidget()
        manifest_layout = QVBoxLayout(manifest_page)
        manifest_layout.setContentsMargins(0, 0, 0, 0)
        manifest_layout.addWidget(self.manifest_table, 1)
        self.save_manifest_button = QPushButton("Validate and save TOML")
        self.save_manifest_button.clicked.connect(self._save_manifest)
        manifest_layout.addWidget(self.save_manifest_button)
        self.tabs.addTab(manifest_page, "TOML")
        self.tabs.addTab(self.anchor_table, "Anchors")

        self.runtime_table.itemChanged.connect(self._runtime_item_changed)

        # The controls above the tables are populated dynamically.  In
        # particular, the Sample page is taller than the lower half of the
        # instrument splitter on many displays.  Keep their natural size in a
        # scrollable content widget instead of allowing that size hint to make
        # the splitter (and eventually the main window) larger than the
        # available screen.
        self.scroll_content = QWidget()
        self.scroll_content.setObjectName("parameterScrollContent")
        content_layout = QVBoxLayout(self.scroll_content)
        content_layout.setSizeConstraint(
            QLayout.SizeConstraint.SetMinAndMaxSize
        )
        content_layout.addWidget(self.title)
        content_layout.addWidget(self.lens_box)
        content_layout.addWidget(self.quick_box)
        content_layout.addWidget(self.energy_filter_box)
        content_layout.addWidget(self.tabs, 1)

        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("parameterScrollArea")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.scroll_area.setWidget(self.scroll_content)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.scroll_area)

    def set_lens_diagnostics(self, text: str) -> None:
        self.lens_diagnostics.setText(
            text or "Recalculate to update field and focal diagnostics."
        )

    @staticmethod
    def _table(headers) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setStretchLastSection(True)
        table.setAlternatingRowColors(True)
        table.verticalHeader().hide()
        return table

    def set_context(
        self,
        label: str,
        runtime_target,
        manifest_target,
        manifest_fields,
        anchor_record,
    ) -> None:
        self._updating = True
        try:
            self.title.setText(label)
            self._runtime_target = runtime_target
            self._manifest_target = manifest_target
            self._manifest_fields = tuple(manifest_fields)
            self._load_runtime()
            self._load_manifest()
            self._load_anchor(anchor_record)
            self._load_lens_controls()
            self._load_quick_controls()
            self._load_energy_filter_controls()
        finally:
            self._updating = False
        # Visibility and row-count changes alter the content height.  Activate
        # the layout now so the scrollbar is correct immediately; previously
        # a window-state change was often the first event that forced this
        # recalculation.
        self.scroll_content.layout().activate()
        self.scroll_content.updateGeometry()
        self.scroll_area.verticalScrollBar().setValue(0)

    def _clear_quick_controls(self) -> None:
        while self.quick_form.rowCount():
            self.quick_form.removeRow(0)
        self._quick_widgets = {}

    def _load_energy_filter_controls(self) -> None:
        target = self._runtime_target
        energy_filter = getattr(target, "obj", None)
        visible = (
            getattr(target, "key", None) == "energy_filter"
            and energy_filter is not None
        )
        self.energy_filter_box.setVisible(visible)
        if not visible:
            return
        self.energy_filter_enabled.setChecked(bool(energy_filter.enabled))
        mode_index = self.energy_filter_mode.findData(
            str(energy_filter.operating_mode).lower()
        )
        self.energy_filter_mode.setCurrentIndex(max(mode_index, 0))
        self.energy_filter_multi_eels.setChecked(
            bool(energy_filter.multi_eels_enabled)
        )
        self.energy_filter_regions.setValue(
            int(energy_filter.multi_eels_region_count)
        )
        self.energy_filter_selected_loss.setValue(
            float(energy_filter.selected_loss_ev)
        )
        self.energy_filter_slit_width.setValue(
            float(energy_filter.slit_width_ev)
        )
        self.energy_filter_active_strip.setValue(
            int(energy_filter.camera_deflector.active_strip)
        )
        self.energy_filter_bias.setValue(
            float(energy_filter.bias_tube.offset_ev)
        )
        self.energy_filter_alignment.setChecked(
            bool(energy_filter.zebra_detector.alignment_mode)
        )
        self.energy_filter_shutter.setChecked(
            bool(energy_filter.fast_shutter.open)
        )
        multi = bool(energy_filter.multi_eels_enabled)
        is_eels = str(energy_filter.operating_mode).lower() == "eels"
        self.energy_filter_regions.setEnabled(is_eels and multi)
        self.energy_filter_active_strip.setEnabled(is_eels and multi)
        self.energy_filter_bias.setEnabled(is_eels and multi)
        self.energy_filter_alignment.setEnabled(is_eels)
        self.energy_filter_slit_width.setEnabled(not is_eels)
        metrics = getattr(energy_filter, "_last_slit_metrics", None)
        metric_text = (
            f" Last solve: {metrics.summary()}"
            if metrics is not None
            else ""
        )
        self.energy_filter_status.setText(
            f"{energy_filter.calibration_status}; public topology: one large "
            "tapered prism and 10 multipoles (most dodecapoles; individual "
            "assignments and exact production order not public; M01-M10 are "
            "model indices). Adjustable non-OEM reference radius "
            f"{energy_filter.prism_radius_mm:g} mm. Zebra 5 x 2048: each "
            f"active strip {energy_filter.zebra_detector.spectral_width_mm:g} "
            f"x {energy_filter.zebra_detector.spectral_height_mm:g} mm; 2-D "
            "alignment area "
            f"{energy_filter.zebra_detector.alignment_width_mm:g} x "
            f"{energy_filter.zebra_detector.alignment_height_mm:g} mm."
            f"{metric_text}"
        )

    def _energy_filter_controls_changed(self, _value=None) -> None:
        if self._updating or self._runtime_target is None:
            return
        if self._runtime_target.key != "energy_filter":
            return
        energy_filter = self._runtime_target.obj
        try:
            energy_filter.enabled = self.energy_filter_enabled.isChecked()
            energy_filter.operating_mode = str(
                self.energy_filter_mode.currentData()
            )
            energy_filter.multi_eels_enabled = (
                self.energy_filter_multi_eels.isChecked()
            )
            energy_filter.multi_eels_region_count = (
                self.energy_filter_regions.value()
            )
            energy_filter.selected_loss_ev = (
                self.energy_filter_selected_loss.value()
            )
            energy_filter.slit_width_ev = (
                self.energy_filter_slit_width.value()
            )
            energy_filter.camera_deflector.active_strip = (
                self.energy_filter_active_strip.value()
            )
            energy_filter.bias_tube.offset_ev = (
                self.energy_filter_bias.value()
            )
            energy_filter.zebra_detector.alignment_mode = (
                self.energy_filter_alignment.isChecked()
            )
            energy_filter.fast_shutter.open = (
                self.energy_filter_shutter.isChecked()
            )
            from temsim.optics.energy_filter import (
                configure_energy_filter_operating_mode,
                configure_energy_slit_from_software,
            )
            configure_energy_filter_operating_mode(
                energy_filter, energy_filter.operating_mode
            )
            configure_energy_slit_from_software(energy_filter)
            energy_filter.bias_tube.validate()
            energy_filter.camera_deflector.validate()
            self._updating = True
            self._load_runtime()
            self._load_energy_filter_controls()
            self._updating = False
            self.runtime_changed.emit("energy_filter_acquisition")
        except Exception as exc:
            self._updating = True
            self._load_energy_filter_controls()
            self._updating = False
            self.error.emit(str(exc))

    @staticmethod
    def _quick_specs(target) -> tuple[tuple[str, str, float, str], ...]:
        obj = getattr(target, "obj", None)
        if obj is None:
            return ()
        if getattr(target, "key", None) == ENERGY_FILTER_SLIT:
            return (
                ("inserted", "Inserted", 1.0, ""),
                (
                    "requested_centre_loss_ev",
                    "Selected loss",
                    1.0,
                    " eV",
                ),
                (
                    "requested_width_ev",
                    "Energy window",
                    1.0,
                    " eV",
                ),
            )
        if hasattr(obj, "radius_mm"):
            return (
                ("enabled", "Inserted", 1.0, ""),
                ("radius_mm", "Opening radius", 1_000.0, " µm"),
                ("offset_x_mm", "X offset", 1_000.0, " µm"),
                ("offset_y_mm", "Y offset", 1_000.0, " µm"),
            )
        if hasattr(obj, "strength_x_percent"):
            return (
                ("enabled", "Enabled", 1.0, ""),
                ("strength_x_percent", "X strength", 1.0, " %"),
                ("strength_y_percent", "Y strength", 1.0, " %"),
            )
        if hasattr(obj, "upper_x_mrad"):
            return (
                ("enabled", "Enabled", 1.0, ""),
                ("upper_x_mrad", "Upper X", 1.0, " mrad"),
                ("upper_y_mrad", "Upper Y", 1.0, " mrad"),
                ("lower_x_mrad", "Lower X", 1.0, " mrad"),
                ("lower_y_mrad", "Lower Y", 1.0, " mrad"),
            )
        if hasattr(obj, "inserted"):
            specs = [("inserted", "Inserted", 1.0, "")]
            if hasattr(obj, "readout_enabled"):
                specs.append(("readout_enabled", "Readout", 1.0, ""))
            return tuple(specs)
        if hasattr(obj, "ray_count"):
            source_fields = (
                ("emission_current_na", "Emission current", " nA"),
                ("virtual_source_fwhm_nm", "Virtual source FWHM", " nm"),
                ("angular_cutoff_mrad", "Angular cutoff", " mrad"),
                ("energy_spread_fwhm_ev", "Energy spread FWHM", " eV"),
                (
                    "minimum_kinetic_energy_ev",
                    "Minimum kinetic energy",
                    " eV",
                ),
                ("energy_half_range_ev", "Energy half range", " eV"),
                ("tip_radius_nm", "Tip radius", " nm"),
                ("emitting_radius_um", "Emitting radius", " µm"),
                ("cathode_temperature_k", "Cathode temperature", " K"),
                ("work_function_ev", "Work function", " eV"),
                ("cathode_anode_gap_mm", "Cathode-anode gap", " mm"),
                ("extraction_field_scale", "Extraction-field scale", ""),
            )
            return tuple(
                (name, label, 1.0, suffix)
                for name, label, suffix in source_fields
                if hasattr(obj, name)
            )
        if getattr(target, "key", None) == "sample":
            return (
                ("diffraction_enabled", "Diffraction", 1.0, ""),
                (
                    "wave_enabled",
                    "TEM wave image (high accuracy only)",
                    1.0,
                    "",
                ),
                (
                    "wave_multislice_enabled",
                    "Multislice specimen propagation",
                    1.0,
                    "",
                ),
                (
                    "wave_atomistic_enabled",
                    "Atomistic Lobato IAM potential",
                    1.0,
                    "",
                ),
                (
                    "wave_frozen_phonon_enabled",
                    "Frozen-phonon intensity ensemble",
                    1.0,
                    "",
                ),
                (
                    "wave_frozen_phonon_configurations",
                    "Frozen-phonon configurations",
                    1.0,
                    "",
                ),
                (
                    "wave_frozen_phonon_sigma_angstrom",
                    "Thermal RMS sigma (0 = preset)",
                    1.0,
                    " Å",
                ),
                (
                    "wave_frozen_phonon_seed",
                    "Frozen-phonon random seed",
                    1.0,
                    "",
                ),
                (
                    "wave_slice_thickness_angstrom",
                    "Multislice target slice",
                    1.0,
                    " Å",
                ),
                ("thickness_nm", "Thickness", 1.0, " nm"),
                ("g_inv_nm", "g", 1.0, " 1/nm"),
                (
                    "excitation_error_inv_nm",
                    "Excitation error",
                    1.0,
                    " 1/nm",
                ),
                ("rocking_width_inv_nm", "Rocking width", 1.0, " 1/nm"),
                ("diffuse_broadening_mrad", "Diffuse broadening", 1.0, " mrad"),
                ("wave_defocus_nm", "Additional defocus", 1.0, " nm"),
            )
        return ()

    def _load_quick_controls(self) -> None:
        self._clear_quick_controls()
        obj = getattr(self._runtime_target, "obj", None)
        specs = self._quick_specs(self._runtime_target)
        for name, label, scale, suffix in specs:
            if not hasattr(obj, name):
                continue
            value = getattr(obj, name)
            if isinstance(value, bool):
                widget = QCheckBox()
                widget.setChecked(value)
                widget.toggled.connect(
                    lambda checked, field=name: self._quick_changed(
                        field, checked, 1.0
                    )
                )
            elif isinstance(value, int):
                widget = QSpinBox()
                if name == "wave_frozen_phonon_configurations":
                    widget.setRange(1, 64)
                elif name == "wave_frozen_phonon_seed":
                    widget.setRange(0, 2_147_483_647)
                else:
                    widget.setRange(-1_000_000_000, 1_000_000_000)
                widget.setKeyboardTracking(False)
                widget.setValue(int(value))
                widget.valueChanged.connect(
                    lambda changed, field=name: self._quick_changed(
                        field, changed, 1.0
                    )
                )
            else:
                widget = QDoubleSpinBox()
                widget.setDecimals(6)
                widget.setRange(-1.0e9, 1.0e9)
                if name in {
                    "radius_mm",
                    "thickness_nm",
                    "rocking_width_inv_nm",
                    "wave_slice_thickness_angstrom",
                    "wave_frozen_phonon_sigma_angstrom",
                    "requested_width_ev",
                }:
                    widget.setMinimum(0.0)
                if name == "wave_slice_thickness_angstrom":
                    widget.setMinimum(1.0e-6)
                if name.endswith("_mrad") and hasattr(obj, "maximum_kick_mrad"):
                    maximum = abs(float(obj.maximum_kick_mrad))
                    widget.setRange(-maximum, maximum)
                if name == "radius_mm" and hasattr(obj, "maximum_radius_mm"):
                    widget.setMaximum(float(obj.maximum_radius_mm) * scale)
                widget.setSuffix(suffix)
                widget.setKeyboardTracking(False)
                widget.setValue(float(value) * scale)
                widget.valueChanged.connect(
                    lambda changed, field=name, factor=scale: self._quick_changed(
                        field, changed, factor
                    )
                )
            widget.setObjectName(f"quick_{name}")
            tooltips = {
                "wave_atomistic_enabled": (
                    "Use finite Z-slice neutral-atom IAM potentials when the "
                    "selected preset has an atomistic definition."
                ),
                "wave_frozen_phonon_enabled": (
                    "Randomly displace atoms and average TEM/STEM "
                    "intensities over configurations."
                ),
                "wave_frozen_phonon_configurations": (
                    "More configurations reduce ensemble sampling error but "
                    "increase runtime approximately linearly."
                ),
                "wave_frozen_phonon_sigma_angstrom": (
                    "One-axis Gaussian RMS displacement; zero uses the "
                    "temperature-qualified specimen preset value."
                ),
                "wave_frozen_phonon_seed": (
                    "Seed for reproducible independent atomic displacements."
                ),
            }
            if name in tooltips:
                widget.setToolTip(tooltips[name])
            self.quick_form.addRow(label, widget)
            self._quick_widgets[name] = widget
        self._update_sample_wave_control_states()
        self.quick_box.setVisible(bool(self._quick_widgets))

    def _update_sample_wave_control_states(self) -> None:
        widgets = self._quick_widgets
        if "wave_multislice_enabled" not in widgets:
            return
        multislice = bool(
            widgets["wave_multislice_enabled"].isChecked()
        )
        atomistic = multislice and bool(
            widgets["wave_atomistic_enabled"].isChecked()
        )
        frozen = atomistic and bool(
            widgets["wave_frozen_phonon_enabled"].isChecked()
        )
        widgets["wave_atomistic_enabled"].setEnabled(multislice)
        widgets["wave_frozen_phonon_enabled"].setEnabled(atomistic)
        for name in (
            "wave_frozen_phonon_configurations",
            "wave_frozen_phonon_sigma_angstrom",
            "wave_frozen_phonon_seed",
        ):
            widgets[name].setEnabled(frozen)

    def _quick_changed(self, name: str, value, scale: float) -> None:
        if self._updating or self._runtime_target is None:
            return
        obj = self._runtime_target.obj
        old_value = getattr(obj, name)
        try:
            if isinstance(value, bool):
                converted = value
            elif isinstance(old_value, int):
                converted = int(value)
            else:
                converted = float(value) / scale
            converted = validate_runtime_assignment(
                self._runtime_target, name, converted
            )
            if (
                self._runtime_target.key == ENERGY_FILTER_SLIT
                and name in {
                    "requested_centre_loss_ev",
                    "requested_width_ev",
                }
            ):
                centre_loss_ev = (
                    float(converted)
                    if name == "requested_centre_loss_ev"
                    else float(obj.requested_centre_loss_ev)
                )
                width_ev = (
                    float(converted)
                    if name == "requested_width_ev"
                    else float(obj.requested_width_ev)
                )
                obj.configure_energy_window(centre_loss_ev, width_ev)
            else:
                setattr(obj, name, converted)
            self._update_sample_wave_control_states()
            self.runtime_changed.emit(name)
            self._updating = True
            self._load_runtime()
        except Exception as exc:
            self._updating = True
            widget = self._quick_widgets.get(name)
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(old_value))
            elif widget is not None:
                widget.setValue(float(old_value) * scale)
            self.error.emit(str(exc))
        finally:
            self._updating = False

    def _load_runtime(self) -> None:
        parameters = (
            editable_parameters(self._runtime_target)
            if self._runtime_target is not None else ()
        )
        self.runtime_table.setRowCount(len(parameters))
        for row, parameter in enumerate(parameters):
            name = QTableWidgetItem(parameter.name)
            name.setFlags(name.flags() & ~Qt.ItemFlag.ItemIsEditable)
            value = QTableWidgetItem(
                "none" if parameter.value is None else str(parameter.value).lower()
                if isinstance(parameter.value, bool) else str(parameter.value)
            )
            value.setData(Qt.ItemDataRole.UserRole, parameter.value)
            self.runtime_table.setItem(row, 0, name)
            self.runtime_table.setItem(row, 1, value)

    def _load_manifest(self) -> None:
        self.manifest_table.setRowCount(len(self._manifest_fields))
        for row, field in enumerate(self._manifest_fields):
            label = QTableWidgetItem(field.label)
            label.setFlags(label.flags() & ~Qt.ItemFlag.ItemIsEditable)
            value = QTableWidgetItem(format_toml_value(field.value))
            value.setData(Qt.ItemDataRole.UserRole, field.path)
            if not field.editable:
                value.setFlags(value.flags() & ~Qt.ItemFlag.ItemIsEditable)
                value.setForeground(Qt.GlobalColor.gray)
            self.manifest_table.setItem(row, 0, label)
            self.manifest_table.setItem(row, 1, value)
        self.save_manifest_button.setEnabled(self._manifest_target is not None)

    def _load_anchor(self, record) -> None:
        rows = []
        if record is not None:
            rows = [
                ("Module", record.module_key),
                ("Part key", record.part_key),
                ("Assembly anchor", record.anchor),
                ("Start Z", f"{record.start_z_mm:.9g} mm"),
                ("Centre Z", f"{record.center_z_mm:.9g} mm"),
                ("End Z", f"{record.end_z_mm:.9g} mm"),
                ("Optical references", ", ".join(
                    f"{value:.9g} mm" for value in record.optical_references_mm
                ) or "None"),
            ]
        self.anchor_table.setRowCount(len(rows))
        for row, (name, value) in enumerate(rows):
            self.anchor_table.setItem(row, 0, QTableWidgetItem(name))
            self.anchor_table.setItem(row, 1, QTableWidgetItem(value))

    def _load_lens_controls(self) -> None:
        obj = getattr(self._runtime_target, "obj", None)
        is_lens = obj is not None and hasattr(obj, "percent")
        self.lens_box.setVisible(is_lens)
        if not is_lens:
            return
        self.lens_enabled.setChecked(bool(getattr(obj, "enabled", True)))
        self.lens_excitation.setMaximum(float(getattr(obj, "max_percent", 1000.0)))
        self.lens_excitation.setValue(float(obj.percent))
        self.lens_cs.setValue(float(getattr(obj, "cs_mm", 0.0) or 0.0))
        polarity = int(getattr(obj, "polarity", 1))
        direction_index = self.lens_field_direction.findData(
            -1 if polarity < 0 else 1
        )
        self.lens_field_direction.setCurrentIndex(max(direction_index, 0))

    def _runtime_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating or item.column() != 1 or self._runtime_target is None:
            return
        name = self.runtime_table.item(item.row(), 0).text()
        old_value = item.data(Qt.ItemDataRole.UserRole)
        try:
            value = convert_runtime_value(old_value, item.text())
            value = validate_runtime_assignment(
                self._runtime_target, name, value
            )
            if (
                self._runtime_target.key == ENERGY_FILTER_SLIT
                and name in {
                    "requested_centre_loss_ev",
                    "requested_width_ev",
                }
            ):
                obj = self._runtime_target.obj
                centre_loss_ev = (
                    float(value)
                    if name == "requested_centre_loss_ev"
                    else float(obj.requested_centre_loss_ev)
                )
                width_ev = (
                    float(value)
                    if name == "requested_width_ev"
                    else float(obj.requested_width_ev)
                )
                obj.configure_energy_window(centre_loss_ev, width_ev)
                value = getattr(obj, name)
            else:
                setattr(self._runtime_target.obj, name, value)
            item.setData(Qt.ItemDataRole.UserRole, value)
            self._updating = True
            self._load_lens_controls()
            self._load_quick_controls()
            self._updating = False
            self.runtime_changed.emit(name)
        except Exception as exc:
            self._updating = True
            item.setText("none" if old_value is None else str(old_value))
            self._updating = False
            self.error.emit(str(exc))

    def _lens_enabled_changed(self, checked: bool) -> None:
        if self._updating or self._runtime_target is None:
            return
        self._runtime_target.obj.enabled = checked
        self._updating = True
        self._load_runtime()
        self._load_quick_controls()
        self._updating = False
        self.runtime_changed.emit("enabled")

    def _lens_excitation_changed(self, value: float) -> None:
        if self._updating or self._runtime_target is None:
            return
        self._runtime_target.obj.percent = value
        self._updating = True
        self._load_runtime()
        self._updating = False
        self.runtime_changed.emit("percent")

    def _lens_cs_changed(self, value: float) -> None:
        if self._updating or self._runtime_target is None:
            return
        self._runtime_target.obj.cs_mm = float(value)
        self._updating = True
        self._load_runtime()
        self._updating = False
        self.runtime_changed.emit("cs_mm")

    def _lens_field_direction_changed(self, index: int) -> None:
        if self._updating or self._runtime_target is None or index < 0:
            return
        polarity = int(self.lens_field_direction.itemData(index))
        self._runtime_target.obj.polarity = polarity
        self._updating = True
        self._load_runtime()
        self._updating = False
        self.runtime_changed.emit("polarity")

    def _save_manifest(self) -> None:
        if self._manifest_target is None:
            return
        try:
            updates = {}
            for row, field in enumerate(self._manifest_fields):
                if not field.editable:
                    continue
                item = self.manifest_table.item(row, 1)
                parsed = parse_toml_value(item.text())
                if parsed != field.value:
                    updates[field.path] = parsed
            self.manifest_save_requested.emit(self._manifest_target, updates)
        except Exception as exc:
            self.error.emit(str(exc))
