"""Operating, TOML and anchor parameters for the selected component."""

from __future__ import annotations

from collections.abc import Mapping

from temsim.gui.input_policy import (
    WheelSafeComboBox as QComboBox,
    WheelSafeDoubleSpinBox as QDoubleSpinBox,
    WheelSafeSpinBox as QSpinBox,
)

from PySide6.QtCore import QPersistentModelIndex, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QLayout,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from temsim.component_keys import ENERGY_FILTER_SLIT, FIXED_APERTURE_KEYS
from temsim.manifest_editor import (
    format_toml_value,
    parse_toml_value,
    resized_part_axial_coordinates,
)
from temsim.runtime_parameters import (
    convert_runtime_value,
    editable_parameters,
    validate_runtime_assignment,
)
from temsim.optics.aberrations import intrinsic_lens_aberration_profile
from temsim.part_geometry import geometry_from_part
from temsim.mechanical_profiles import MAGNETIC_LENS_MECHANICAL_PROFILES


class _ManifestDraftDelegate(QStyledItemDelegate):
    """Expose literal, not-yet-committed text to geometry-save transactions."""

    def __init__(self, parent):
        super().__init__(parent)
        self._editors = {}
        self.closeEditor.connect(lambda editor, _hint: self._editors.pop(id(editor), None))

    def createEditor(self, parent, option, index):
        editor = super().createEditor(parent, option, index)
        if isinstance(editor, QLineEdit):
            token = id(editor)
            self._editors[token] = (QPersistentModelIndex(index), editor)
            editor.destroyed.connect(lambda _obj=None, key=token: self._editors.pop(key, None))
        return editor

    def pending_texts(self):
        return {tuple(index.data(Qt.ItemDataRole.UserRole) or ()): editor.text()
                for index, editor in self._editors.values() if index.isValid()}

    def close_pending_editors(self):
        editors = tuple(editor for _index, editor in self._editors.values())
        self._editors.clear()
        for editor in editors:
            self.closeEditor.emit(editor, QStyledItemDelegate.EndEditHint.RevertModelCache)


class ParameterPanel(QWidget):
    runtime_changed = Signal(str)
    energy_filter_match_requested = Signal()
    manifest_save_requested = Signal(object, object)
    geometry_edit_requested = Signal(object)
    error = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("parameterPanel")
        self.setMinimumWidth(380)
        self._runtime_target = None
        self._manifest_target = None
        self._manifest_fields = ()
        self._updating = False
        self._simulation_mode = None
        self._simulation_descriptors = None
        self._semantic_by_key = {}

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
        self.lens_cs.setDecimals(9)
        self.lens_cs.setRange(-1.0e6, 1.0e6)
        self.lens_cs.setSuffix(" mm")
        self.lens_cs.setToolTip(
            "Signed third-order spherical aberration coefficient. "
            "Zero disables the calibrated ray-direction correction."
        )
        self.lens_cc = QDoubleSpinBox()
        self.lens_cc.setDecimals(9)
        self.lens_cc.setRange(0.0, 1.0e6)
        self.lens_cc.setSuffix(" mm")
        self.lens_cc.setToolTip(
            "First-order chromatic coefficient used with Δf = Cc ΔE/E0."
        )
        self.lens_aberration_model = QComboBox()
        self.lens_aberration_model.addItem(
            "Focal-length estimate (provisional)", "estimate"
        )
        self.lens_aberration_model.addItem(
            "Explicit component coefficients", "explicit"
        )
        self.lens_aberration_provenance = QLabel()
        self.lens_aberration_provenance.setWordWrap(True)
        self.lens_aberration_provenance.setStyleSheet(
            "color: #64748b; font-weight: 600;"
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
        lens_form.addRow("Chromatic aberration Cc", self.lens_cc)
        lens_form.addRow("Coefficient model", self.lens_aberration_model)
        lens_form.addRow(self.lens_aberration_provenance)
        lens_form.addRow(
            "Effective axial field direction", self.lens_field_direction
        )
        lens_form.addRow(self.lens_diagnostics)
        self.lens_enabled.toggled.connect(self._lens_enabled_changed)
        self.lens_excitation.valueChanged.connect(self._lens_excitation_changed)
        self.lens_cs.valueChanged.connect(self._lens_cs_changed)
        self.lens_cc.valueChanged.connect(self._lens_cc_changed)
        self.lens_aberration_model.currentIndexChanged.connect(
            self._lens_aberration_model_changed
        )
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
        self.energy_filter_shutter = QCheckBox("Zebra detector shutter open")
        self.energy_filter_shutter.setToolTip(
            "Shutter in the EELS detector branch. Controls detector exposure; "
            "sample illumination is controlled by the pre-specimen blankers."
        )
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

        self.geometry_box = QGroupBox("Saved mechanical dimensions")
        geometry_layout = QVBoxLayout(self.geometry_box)
        self.geometry_summary = QLabel()
        self.geometry_summary.setWordWrap(True)
        self.geometry_edit_button = QPushButton("Edit dimensions…")
        self.geometry_edit_button.setObjectName("editPartDimensionsButton")
        self.geometry_edit_button.clicked.connect(self._request_geometry_editor)
        geometry_layout.addWidget(self.geometry_summary)
        geometry_layout.addWidget(self.geometry_edit_button)
        self.geometry_box.hide()

        self.tabs = QTabWidget()
        self.runtime_table = self._table(("Operating parameter", "Value"))
        self.manifest_table = self._table(("TOML parameter", "Value"))
        self._manifest_delegate = _ManifestDraftDelegate(self.manifest_table)
        self.manifest_table.setItemDelegate(self._manifest_delegate)
        self.anchor_table = self._table(("Anchor property", "Value"))
        self.anchor_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tabs.addTab(self.runtime_table, "Operating")

        manifest_page = QWidget()
        manifest_layout = QVBoxLayout(manifest_page)
        manifest_layout.setContentsMargins(0, 0, 0, 0)
        self.manifest_draft_notice = QLabel()
        self.manifest_draft_notice.setWordWrap(True)
        self.manifest_draft_notice.hide()
        manifest_layout.addWidget(self.manifest_draft_notice)
        manifest_layout.addWidget(self.manifest_table, 1)
        self.save_manifest_button = QPushButton("Validate and save TOML")
        self.save_manifest_button.clicked.connect(self._save_manifest)
        manifest_layout.addWidget(self.save_manifest_button)
        self.tabs.addTab(manifest_page, "TOML")
        self.tabs.addTab(self.anchor_table, "Anchors")

        self.runtime_table.itemChanged.connect(self._runtime_item_changed)
        self.manifest_table.itemChanged.connect(self._manifest_item_changed)
        self.runtime_table.currentCellChanged.connect(self._refresh_parameter_details)
        self.manifest_table.currentCellChanged.connect(self._refresh_parameter_details)
        self.tabs.currentChanged.connect(self._refresh_parameter_details)
        self.parameter_details = QLabel()
        self.parameter_details.setObjectName("parameterMeaningDetails")
        self.parameter_details.setTextFormat(Qt.TextFormat.PlainText)
        self.parameter_details.setWordWrap(True)
        self.parameter_details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.parameter_details.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        # Device controls are populated dynamically. Keep their natural size
        # in a scrollable content widget so a long component editor does not
        # force the main window beyond the available screen.
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
        content_layout.addWidget(self.geometry_box)
        content_layout.addWidget(self.tabs, 1)
        content_layout.addWidget(self.parameter_details)

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
        self._refresh_parameter_details()

    def set_simulation_context(self, mode=None, descriptors=None, by_key=None):
        """Refresh explanations without rebuilding editors or touching drafts."""
        self._simulation_mode = mode
        self._simulation_descriptors = descriptors
        self._semantic_by_key = dict(by_key or {})
        self._refresh_parameter_details()

    def _semantic_part(self, path):
        key = path[1] if len(path) >= 3 and path[0] in {"parts", "runtime"} else None
        candidate = self._semantic_by_key.get(key)
        if candidate is not None:
            return candidate if isinstance(candidate, Mapping) else candidate.data
        part = {field.path[2]: field.value for field in self._manifest_fields
                if len(field.path) == 3 and field.path[:2] == ("parts", key)}
        if key is not None:
            part.setdefault("key", key)
        return part

    def _refresh_parameter_details(self, *_args):
        if not hasattr(self, "parameter_details"):
            return
        index = self.tabs.currentIndex()
        self.parameter_details.setVisible(index in {0, 1})
        if index not in {0, 1}:
            return
        table = self.runtime_table if index == 0 else self.manifest_table
        row = table.currentRow()
        meaning = None
        if row < 0 or table.item(row, 0) is None:
            self.parameter_details.setText("Select a parameter to see its meaning, source and simulation impact.")
            return
        if index == 0:
            if self._runtime_target is None:
                return
            path = ("runtime", self._runtime_target.key, table.item(row, 0).text())
        else:
            value = table.item(row, 1)
            if value is None:
                return
            path = tuple(value.data(Qt.ItemDataRole.UserRole) or ())
            meaning = next((getattr(field, "meaning", None) for field in self._manifest_fields
                            if tuple(field.path) == path), None)
        from temsim.parameter_semantics import describe_parameter
        from temsim.parameter_impact import describe_parameter_impact

        part = self._semantic_part(path)
        meaning = meaning or describe_parameter(part, path, by_key=self._semantic_by_key)
        impact = describe_parameter_impact(part, path, by_key=self._semantic_by_key,
            simulation_mode=self._simulation_mode, descriptors=self._simulation_descriptors)
        mode = getattr(self._simulation_mode, "value", self._simulation_mode)
        mode_text = str(mode) if mode is not None else "not connected"
        details = [meaning.label + (f" ({meaning.unit})" if meaning.unit else ""),
                   f"Category: {meaning.category_label}",
                   f"Source: {meaning.source_label}. {meaning.source_note}", meaning.description,
                   f"Simulation mode: {mode_text}. Impact: {impact.label}. {impact.detail}"]
        if impact.affected_results:
            details.append("Affected results: " + ", ".join(impact.affected_results))
        self.parameter_details.setText("\n".join(line for line in details if line))
        self.parameter_details.setToolTip(self.parameter_details.text())

    def refresh_runtime_values(self) -> None:
        """Refresh retained controls after a model-shelf switch, without edits."""
        previous = self._updating
        self._updating = True
        try:
            self._load_runtime()
            self._load_lens_controls()
            self._load_quick_controls()
        finally:
            self._updating = previous

    def refresh_live_values(self, changed_keys) -> None:
        """Update existing editors after a slider edit, without rebuilding them."""
        if self._runtime_target is None or self._runtime_target.key not in changed_keys:
            return
        previous = self._updating
        self._updating = True
        try:
            parameters = editable_parameters(self._runtime_target)
            names = [self.runtime_table.item(row, 0).text()
                     for row in range(self.runtime_table.rowCount())]
            if names != [parameter.name for parameter in parameters]:
                self._load_runtime()
            else:
                for row, parameter in enumerate(parameters):
                    item = self.runtime_table.item(row, 1)
                    value = parameter.value
                    text = ("none" if value is None else str(value).lower()
                            if isinstance(value, bool) else str(value))
                    if item.text() != text:
                        item.setText(text)
                        item.setData(Qt.ItemDataRole.UserRole, value)
            self._load_lens_controls()
            obj = self._runtime_target.obj
            for name, _label, scale, _suffix in self._quick_specs(self._runtime_target):
                widget = self._quick_widgets.get(name)
                if widget is None or not hasattr(obj, name):
                    continue
                value = getattr(obj, name)
                if isinstance(widget, QCheckBox):
                    widget.setChecked(bool(value))
                elif isinstance(widget, QComboBox):
                    widget.setCurrentIndex(max(0, widget.findData(str(value))))
                elif isinstance(widget, QLineEdit):
                    if widget.text() != str(value):
                        widget.setText(str(value))
                else:
                    widget.setValue(float(value) * scale)
        finally:
            self._updating = previous

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
        *, geometry_parent=None,
    ) -> None:
        self._updating = True
        try:
            self.title.setText(label)
            self._runtime_target = runtime_target
            self._manifest_target = manifest_target
            self._manifest_fields = tuple(manifest_fields)
            self._geometry_parent = geometry_parent
            self._load_runtime()
            self._load_manifest()
            self._load_geometry_controls()
            self._load_anchor(anchor_record)
            self._load_lens_controls()
            self._load_quick_controls()
            self._load_energy_filter_controls()
        finally:
            self._updating = False
        self._refresh_parameter_details()
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
        detail_text = (
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
        self.energy_filter_status.setText(
            f"{energy_filter.calibration_status} | prism radius "
            f"{energy_filter.prism_radius_mm:g} mm | Zebra 5 × 2048"
            + metric_text
        )
        self.energy_filter_status.setToolTip(detail_text)

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
        if getattr(target, "key", None) == "nanopulser_deflector":
            return (
                ("blanked", "NanoPulser blanked (static)", 1.0, ""),
                ("voltage_v", "Plate voltage difference", 1.0, " V"),
                ("azimuth_deg", "Deflection azimuth", 1.0, "°"),
            )
        if hasattr(obj, "beam_blanked") and hasattr(obj, "upper_field_x_mt"):
            return (("beam_blanked", "Blank beam", 1.0, ""),)
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
        if hasattr(obj, "diameter_mm"):
            insertion = (() if target.key in FIXED_APERTURE_KEYS else
                         (("enabled", "Inserted", 1.0, ""),))
            return insertion + (
                ("diameter_mm", "Opening diameter", 1_000.0, " µm"),
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
            if hasattr(obj, "centre_offset_x_mm"):
                specs.extend((
                    ("centre_offset_x_mm", "Detector centre X", 1_000.0, " µm"),
                    ("centre_offset_y_mm", "Detector centre Y", 1_000.0, " µm"),
                ))
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
        return ()

    def _load_quick_controls(self) -> None:
        self._clear_quick_controls()
        obj = getattr(self._runtime_target, "obj", None)
        fixed = (
            getattr(self._runtime_target, "key", None) in FIXED_APERTURE_KEYS
            or getattr(self._manifest_target, "part_key", None) in FIXED_APERTURE_KEYS
        )
        if fixed:
            status = QLabel("Always inserted")
            status.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
                | Qt.TextInteractionFlag.TextSelectableByKeyboard
            )
            status.setToolTip(
                "Non-retractable stop. Edit axial position in TOML. "
                + ("Adjust opening and X/Y using the controls below."
                   if obj is not None else "Edit opening size in TOML.")
            )
            self.quick_form.addRow("Insertion", status)
        specs = self._quick_specs(self._runtime_target)
        for name, label, scale, suffix in specs:
            if not hasattr(obj, name):
                continue
            value = getattr(obj, name)
            if isinstance(value, str):
                widget = QLineEdit()
                widget.setText(value)
                widget.editingFinished.connect(
                    lambda field=name, control=widget: self._quick_changed(
                        field,
                        control.text(),
                        1.0,
                    )
                )
            elif isinstance(value, bool):
                widget = QCheckBox()
                widget.setChecked(value)
                widget.toggled.connect(
                    lambda checked, field=name: self._quick_changed(
                        field, checked, 1.0
                    )
                )
            elif isinstance(value, int):
                widget = QSpinBox()
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
                    "diameter_mm",
                    "requested_width_ev",
                }:
                    widget.setMinimum(0.0)
                if name.endswith("_mrad") and hasattr(obj, "maximum_kick_mrad"):
                    maximum = abs(float(obj.maximum_kick_mrad))
                    widget.setRange(-maximum, maximum)
                if name == "diameter_mm" and hasattr(
                    obj, "maximum_diameter_mm"
                ):
                    widget.setMaximum(float(obj.maximum_diameter_mm) * scale)
                elif name == "diameter_mm" and hasattr(obj, "maximum_radius_mm"):
                    widget.setMaximum(2.0 * float(obj.maximum_radius_mm) * scale)
                widget.setSuffix(suffix)
                widget.setKeyboardTracking(False)
                widget.setValue(float(value) * scale)
                widget.valueChanged.connect(
                    lambda changed, field=name, factor=scale: self._quick_changed(
                        field, changed, factor
                    )
                )
            widget.setObjectName(f"quick_{name}")
            if name == "beam_blanked":
                widget.setToolTip(
                    "Blank the beam using these gun tilt coils. "
                    "Uncheck to restore the saved gun alignment."
                )
            self.quick_form.addRow(label, widget)
            self._quick_widgets[name] = widget
        self.quick_box.setVisible(bool(self._quick_widgets) or fixed)

    def _quick_changed(self, name: str, value, scale: float) -> None:
        if self._updating or self._runtime_target is None:
            return
        obj = self._runtime_target.obj
        old_value = getattr(obj, name)
        try:
            if isinstance(value, bool):
                converted = value
            elif isinstance(old_value, str):
                converted = str(value)
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
            self.runtime_changed.emit(name)
            self._updating = True
            self._load_runtime()
        except Exception as exc:
            self._updating = True
            widget = self._quick_widgets.get(name)
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(old_value))
            elif isinstance(widget, QComboBox):
                index = widget.findData(str(old_value))
                widget.setCurrentIndex(index if index >= 0 else 0)
            elif isinstance(widget, QLineEdit):
                widget.setText(str(old_value))
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

    def _load_geometry_controls(self) -> None:
        target = self._manifest_target
        part = {
            field.path[2]: field.value
            for field in self._manifest_fields
            if target is not None and len(field.path) == 3
            and field.path[:2] == ("parts", target.part_key)
        }
        visible = part.get("mechanical_profile") in MAGNETIC_LENS_MECHANICAL_PROFILES
        self.geometry_box.setVisible(visible)
        self.geometry_edit_button.setEnabled(False)
        if not visible:
            return
        try:
            geometry = geometry_from_part(part, parent=self._geometry_parent)
        except ValueError as exc:
            self.geometry_summary.setText(str(exc))
            return
        self.geometry_summary.setText(
            f"Length {geometry.length_mm:.9g} mm · ID {geometry.inner_diameter_mm:.9g} mm"
            f" · OD {geometry.outer_diameter_mm:.9g} mm\n"
            f"Radial thickness {geometry.thickness_mm:.9g} mm. "
            f"Beam passage {geometry.vacuum_inner_diameter_mm:.9g} mm."
        )
        self.geometry_edit_button.setEnabled(True)
        self.geometry_edit_button.setToolTip(
            "Edit the saved part's axisymmetric section using dimensions or drag handles. "
            "The centre stays fixed. Uncommitted TOML table edits are not imported."
        )

    def _request_geometry_editor(self) -> None:
        if self._manifest_target is not None and self.geometry_edit_button.isEnabled():
            self.geometry_edit_requested.emit(self._manifest_target)

    def manifest_draft_texts(self, target):
        """Preserve even invalid text without treating it as a geometry edit."""
        if self._manifest_target != target:
            return {}
        original = {field.path: format_toml_value(field.value) for field in self._manifest_fields}
        draft = {}
        pending = self._manifest_delegate.pending_texts()
        for row in range(self.manifest_table.rowCount()):
            item = self.manifest_table.item(row, 1)
            path = tuple(item.data(Qt.ItemDataRole.UserRole) or ())
            text = pending.get(path, item.text())
            if path in original and text != original[path]:
                draft[path] = text
        return draft

    def restore_manifest_draft_texts(self, target, draft):
        if self._manifest_target != target or not draft:
            return
        self._manifest_delegate.close_pending_editors()
        blocked = self.manifest_table.blockSignals(True)
        try:
            for row in range(self.manifest_table.rowCount()):
                item = self.manifest_table.item(row, 1)
                path = tuple(item.data(Qt.ItemDataRole.UserRole) or ())
                if path in draft:
                    item.setText(draft[path])
        finally:
            self.manifest_table.blockSignals(blocked)
        if not self.manifest_draft_texts(target):
            self.manifest_draft_notice.hide()
            return
        self.manifest_draft_notice.setText(
            "Your earlier TOML table edits are retained here and remain unsaved. "
            "Saved mechanical dimensions above shows the applied geometry."
        )
        self.manifest_draft_notice.show()

    def _load_manifest(self) -> None:
        self.manifest_draft_notice.hide()
        self._manifest_delegate.close_pending_editors()
        blocked = self.manifest_table.blockSignals(True)
        try:
            self.manifest_table.setRowCount(len(self._manifest_fields))
            for row, field in enumerate(self._manifest_fields):
                label = QTableWidgetItem(field.label)
                label.setFlags(label.flags() & ~Qt.ItemFlag.ItemIsEditable)
                value = QTableWidgetItem(format_toml_value(field.value))
                value.setData(Qt.ItemDataRole.UserRole, field.path)
                if not field.editable:
                    value.setFlags(value.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    value.setForeground(Qt.GlobalColor.gray)
                if len(field.path) == 3 and field.path[0] == "parts":
                    if field.path[2] == "length_mm":
                        tooltip = (
                            "Changing length keeps local_center_z_mm fixed and "
                            "updates local_start_z_mm and local_end_z_mm, preserving "
                            "the centre's relative position within the part. Gaps "
                            "to neighbouring parts may change. Use Validate and "
                            "save TOML to apply the edit."
                        )
                    elif field.path[2] in {"local_start_z_mm", "local_end_z_mm"}:
                        tooltip = (
                            "Updated automatically when this part's length_mm "
                            "changes, with local_center_z_mm held fixed. Gaps to "
                            "neighbouring parts may change. This coordinate can "
                            "also be edited directly."
                        )
                    elif field.path[2] == "vacuum_inner_diameter_mm":
                        tooltip = (
                            "Electron-beam passage diameter, not material thickness. "
                            "Use Edit dimensions or mechanical_inner_diameter_mm / "
                            "mechanical_outer_diameter_mm to change the coil or wall thickness."
                        )
                    elif field.path[2] in {
                        "mechanical_inner_diameter_mm", "mechanical_outer_diameter_mm"
                    }:
                        tooltip = (
                            "Material diameter. Radial thickness = (outer diameter - "
                            "inner diameter) / 2. Use Edit dimensions for a section preview."
                        )
                    else:
                        tooltip = ""
                    meaning = getattr(field, "meaning", None)
                    if meaning is not None:
                        interaction = tooltip if field.path[2] in {
                            "length_mm", "local_start_z_mm", "local_end_z_mm", "vacuum_inner_diameter_mm"
                        } else ""
                        if meaning.category == "physical" and field.path[2] in {
                            "mechanical_inner_diameter_mm", "mechanical_outer_diameter_mm"
                        }:
                            interaction = tooltip
                        tooltip = "\n".join(filter(None, (
                            "TOML key: " + field.label, meaning.label, "Category: " + meaning.category_label,
                            "Source: " + meaning.source_label + ". " + meaning.source_note,
                            meaning.description, interaction,
                        )))
                    label.setToolTip(tooltip)
                    value.setToolTip(tooltip)
                self.manifest_table.setItem(row, 0, label)
                self.manifest_table.setItem(row, 1, value)
        finally:
            self.manifest_table.blockSignals(blocked)
        self.save_manifest_button.setEnabled(self._manifest_target is not None)

    def _manifest_item_changed(self, item: QTableWidgetItem) -> None:
        if (
            self._updating
            or self._manifest_target is None
            or item.column() != 1
            or not item.flags() & Qt.ItemFlag.ItemIsEditable
        ):
            return
        path = tuple(item.data(Qt.ItemDataRole.UserRole) or ())
        if len(path) != 3 or path[0] != "parts" or path[2] != "length_mm":
            return
        coordinates = {}
        for row in range(self.manifest_table.rowCount()):
            candidate = self.manifest_table.item(row, 1)
            candidate_path = tuple(candidate.data(Qt.ItemDataRole.UserRole) or ())
            if (
                len(candidate_path) == 3
                and candidate_path[:2] == path[:2]
                and candidate_path[2] in {
                    "local_start_z_mm", "local_center_z_mm", "local_end_z_mm"
                }
            ):
                coordinates[candidate_path[2]] = candidate
        if len(coordinates) != 3:
            return
        try:
            part = {
                name: parse_toml_value(coordinate.text())
                for name, coordinate in coordinates.items()
            }
            updates = resized_part_axial_coordinates(part, parse_toml_value(item.text()))
            rendered = {name: format_toml_value(value) for name, value in updates.items()}
        except (TypeError, ValueError, OverflowError):
            # Keep invalid input visible for the existing Save validation path.
            return
        blocked = self.manifest_table.blockSignals(True)
        try:
            for name, text in rendered.items():
                coordinates[name].setText(text)
        finally:
            self.manifest_table.blockSignals(blocked)

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
        parent_state = getattr(self.parent(), "state", None)
        voltage_kv = float(getattr(parent_state, "beam_voltage_kv", 300.0))
        profile = intrinsic_lens_aberration_profile(obj, voltage_kv)
        explicit = (
            getattr(obj, "cs_mm", None) is not None
            or getattr(obj, "cc_mm", None) is not None
        )
        model_index = self.lens_aberration_model.findData(
            "explicit" if explicit else "estimate"
        )
        self.lens_aberration_model.setCurrentIndex(max(model_index, 0))
        coefficients_available = (
            profile.cs_mm is not None and profile.cc_mm is not None
        )
        self.lens_cs.setEnabled(explicit and coefficients_available)
        self.lens_cc.setEnabled(explicit and coefficients_available)
        self.lens_cs.setValue(float(profile.cs_mm or 0.0))
        self.lens_cc.setValue(float(profile.cc_mm or 0.0))
        provenance_detail = (
            f"{profile.status}: {profile.model}. {profile.source}."
        )
        self.lens_aberration_provenance.setText(
            f"{profile.status} | {profile.model}"
        )
        self.lens_aberration_provenance.setToolTip(provenance_detail)
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

    def _lens_cc_changed(self, value: float) -> None:
        if self._updating or self._runtime_target is None:
            return
        self._runtime_target.obj.cc_mm = float(value)
        self._updating = True
        self._load_runtime()
        self._load_lens_controls()
        self._updating = False
        self.runtime_changed.emit("cc_mm")

    def _lens_aberration_model_changed(self, index: int) -> None:
        if self._updating or self._runtime_target is None or index < 0:
            return
        obj = self._runtime_target.obj
        mode = str(self.lens_aberration_model.itemData(index))
        if mode == "estimate":
            obj.cs_mm = None
            obj.cc_mm = None
        else:
            parent_state = getattr(self.parent(), "state", None)
            voltage_kv = float(getattr(parent_state, "beam_voltage_kv", 300.0))
            profile = intrinsic_lens_aberration_profile(obj, voltage_kv)
            if profile.cs_mm is None or profile.cc_mm is None:
                self.error.emit(
                    "Cannot create explicit coefficients because this lens has no "
                    "resolvable focal-length scale."
                )
                self._updating = True
                self._load_lens_controls()
                self._updating = False
                return
            obj.cs_mm = float(profile.cs_mm)
            obj.cc_mm = float(profile.cc_mm)
        self._updating = True
        self._load_runtime()
        self._load_lens_controls()
        self._updating = False
        self.runtime_changed.emit("aberration_model")

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
