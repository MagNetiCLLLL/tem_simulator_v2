"""Main TEM Simulator v2 desktop window."""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace

from PySide6.QtCore import QByteArray, QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QDockWidget,
    QDoubleSpinBox,
    QComboBox,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QSpinBox,
    QToolBar,
    QWidget,
)

from temsim.assembly_catalog import AssemblyCatalog
from temsim.column.state_layout import (
    apply_physical_layout_to_state,
    layout_configuration_from_state,
)
from temsim.gui.assembly_panel import AssemblyPanel
from temsim.gui.calculation_controller import (
    CalculationController,
    estimate_calculation_memory_bytes,
    format_memory_size,
)
from temsim.gui.direct_alignment_controller import (
    DirectAlignmentController,
)
from temsim.gui.parameter_panel import ParameterPanel
from temsim.gui.visualization import VisualizationWorkspace
from temsim.manifest_editor import ManifestEditor, ManifestTarget
from temsim.optics.column import default_state
from temsim.operating_modes import (
    apply_operating_mode_pair,
    compatible_modes,
    direct_alignment_by_key,
)
from temsim.profile_io import apply_profile_values, read_profile, save_profile
from temsim.physics.compute_backend import (
    BACKEND_AUTO,
    BACKEND_CHOICES,
    BACKEND_CPU,
    BACKEND_CUDA,
    cupy_capability,
    cuda_capability,
)
from temsim.runtime_parameters import runtime_targets


class MainWindow(QMainWindow):
    SETTINGS_GEOMETRY = "main_window/geometry"
    SETTINGS_STATE = "main_window/state"
    PREVIEW_RAYS = 49
    PREVIEW_STEP_MM = 2.5
    INITIAL_PREVIEW_DELAY_MS = 50
    PREVIEW_DEBOUNCE_MS = 250

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("mainWindow")
        self.setWindowTitle("TEM Simulator v2")
        self.resize(1500, 920)
        self.setDockNestingEnabled(True)

        self.catalog = AssemblyCatalog()
        self.selection = self.catalog.default_selection()
        self.state = default_state()
        self.assembly = self.catalog.apply(self.state, self.selection)
        if self._apply_state_operating_modes(
            self.state, self.selection
        ) is None:
            raise ValueError(
                "The default assembly has no compatible operating-mode pair"
            )
        self.manifest_editor = ManifestEditor()
        catalog_audit = self.manifest_editor.validate_catalog()
        self._runtime_targets = {}
        self._anchors_by_key = {}
        self._selected_component_key = None

        self.workspace = VisualizationWorkspace(self)
        self.setCentralWidget(self.workspace)

        self.assembly_panel = AssemblyPanel(
            self.catalog, self.selection, self
        )
        self.parameter_panel = ParameterPanel(self)
        self.instrument_editor = QSplitter(Qt.Orientation.Vertical, self)
        self.instrument_editor.setObjectName("instrumentEditorSplitter")
        self.instrument_editor.setChildrenCollapsible(False)
        self.instrument_editor.addWidget(self.assembly_panel)
        self.instrument_editor.addWidget(self.parameter_panel)
        self.instrument_editor.setStretchFactor(0, 1)
        self.instrument_editor.setStretchFactor(1, 1)
        self.instrument_editor.setSizes([430, 430])
        self.log_output = QPlainTextEdit(self)
        self.log_output.setObjectName("calculationLog")
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumBlockCount(2_000)
        self.log_output.appendPlainText("TEM Simulator v2 started.")
        self.log_output.appendPlainText(
            f"TOML catalog validated: {catalog_audit.module_count} modules, "
            f"{catalog_audit.part_definition_count} part definitions, "
            f"{catalog_audit.assembly_count} assembly combinations."
        )

        self.instrument_dock = self._create_dock(
            "Instrument setup and parameters", "instrumentDock",
            self.instrument_editor,
            Qt.DockWidgetArea.LeftDockWidgetArea,
        )
        self.instrument_dock.setMinimumWidth(420)
        self.log_dock = self._create_dock(
            "Status and calculation log", "logDock", self.log_output,
            Qt.DockWidgetArea.BottomDockWidgetArea,
        )

        self.calculations = CalculationController(self)
        self.direct_alignments = DirectAlignmentController(self)
        self._direct_alignment_state_token: str | None = None
        self._progress_owners: set[str] = set()
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(self.PREVIEW_DEBOUNCE_MS)

        self.assembly_panel.selection_requested.connect(self.load_assembly)
        self.assembly_panel.operating_mode_requested.connect(
            self.apply_operating_modes
        )
        self.assembly_panel.direct_alignment_requested.connect(
            self.apply_direct_alignment
        )
        self.assembly_panel.component_selected.connect(
            self._select_tree_item
        )
        self.workspace.component_selected.connect(
            self._select_component_from_workspace
        )
        self.workspace.scan_parameters_changed.connect(
            self._runtime_parameter_changed
        )
        self.workspace.scan_error.connect(self._show_error)
        self.parameter_panel.runtime_changed.connect(
            self._runtime_parameter_changed
        )
        self.parameter_panel.energy_filter_match_requested.connect(
            self.match_energy_filter_to_ht
        )
        self.parameter_panel.manifest_save_requested.connect(
            self._save_manifest_updates
        )
        self.parameter_panel.error.connect(self._show_error)
        self.preview_timer.timeout.connect(self.run_preview)
        self.calculations.started.connect(self._calculation_started)
        self.calculations.result_ready.connect(self._calculation_ready)
        self.calculations.failed.connect(self._calculation_failed)
        self.calculations.finished.connect(self._calculation_finished)
        self.direct_alignments.started.connect(
            self._direct_alignment_started
        )
        self.direct_alignments.result_ready.connect(
            self._direct_alignment_ready
        )
        self.direct_alignments.failed.connect(
            self._direct_alignment_failed
        )
        self.direct_alignments.finished.connect(
            self._direct_alignment_finished
        )

        self._create_actions()
        self._create_toolbar()
        self._create_status_bar()
        self._refresh_assembly_views()
        self._restore_workspace()
        self.preview_timer.start(self.INITIAL_PREVIEW_DELAY_MS)

    def _create_dock(self, title, object_name, widget, area) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(object_name)
        dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def _create_actions(self) -> None:
        self.open_profile_action = QAction("Open operating profile...", self)
        self.open_profile_action.setShortcut("Ctrl+O")
        self.open_profile_action.triggered.connect(self.open_profile)
        self.save_profile_action = QAction("Save operating profile...", self)
        self.save_profile_action.setShortcut("Ctrl+S")
        self.save_profile_action.triggered.connect(self.save_profile)
        self.reload_toml_action = QAction(
            "Reload and validate TOML catalog", self
        )
        self.reload_toml_action.setShortcut("F5")
        self.reload_toml_action.triggered.connect(self.reload_toml_catalog)
        self.exit_action = QAction("Exit", self)
        self.exit_action.setShortcut("Ctrl+Q")
        self.exit_action.triggered.connect(self.close)
        self.reset_layout_action = QAction("Reset workspace layout", self)
        self.reset_layout_action.triggered.connect(self.reset_workspace)

        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(self.open_profile_action)
        file_menu.addAction(self.save_profile_action)
        file_menu.addAction(self.reload_toml_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)
        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction(self.instrument_dock.toggleViewAction())
        view_menu.addAction(self.log_dock.toggleViewAction())
        view_menu.addSeparator()
        view_menu.addAction(self.reset_layout_action)

    def _create_toolbar(self) -> None:
        toolbar = QToolBar("Calculation", self)
        toolbar.setObjectName("calculationToolbar")
        toolbar.setMovable(False)

        preview_button = QPushButton("Recalculate preview")
        preview_button.setObjectName("previewButton")
        preview_button.clicked.connect(self.run_preview)
        toolbar.addWidget(preview_button)
        toolbar.addSeparator()

        toolbar.addWidget(QLabel("High-accuracy rays"))
        self.high_rays = QSpinBox()
        self.high_rays.setObjectName("highAccuracyRayCount")
        self.high_rays.setRange(1_000, 1_000_000)
        self.high_rays.setSingleStep(1_000)
        # Tuned for the supported 32 GiB workstation profile. The controller
        # also enforces a conservative 24 GiB process budget for custom values.
        self.high_rays.setValue(15_000)
        toolbar.addWidget(self.high_rays)

        toolbar.addWidget(QLabel("Step (mm)"))
        self.high_step = QDoubleSpinBox()
        self.high_step.setObjectName("highAccuracyStep")
        self.high_step.setDecimals(4)
        self.high_step.setRange(0.01, 1.0)
        self.high_step.setValue(0.1)
        toolbar.addWidget(self.high_step)

        toolbar.addWidget(QLabel("Compute"))
        self.compute_backend = QComboBox()
        self.compute_backend.setObjectName("computeBackend")
        for backend in BACKEND_CHOICES:
            label = backend
            if backend == BACKEND_AUTO:
                label = "Auto (GPU / CPU)"
            self.compute_backend.addItem(label, backend)
        selected_backend = str(
            getattr(self.state, "acceleration_backend", BACKEND_AUTO)
        )
        selected_index = self.compute_backend.findData(selected_backend)
        self.compute_backend.setCurrentIndex(max(selected_index, 0))
        self.state.acceleration_backend = str(
            self.compute_backend.currentData() or BACKEND_AUTO
        )
        self.state.acceleration_enabled = (
            self.state.acceleration_backend != BACKEND_CPU
        )
        cuda_status = cuda_capability()
        cupy_status = cupy_capability()
        self.compute_backend.setToolTip(
            "Shared ray and wave-optics preference. Auto uses CUDA for "
            "sufficiently large ray bundles and CuPy for sufficiently large "
            "multislice/FFT workloads; small jobs remain on CPU. "
            f"Ray CUDA: {cuda_status.detail}. Wave CUDA: {cupy_status.detail}."
        )
        self.compute_backend.currentIndexChanged.connect(
            self._compute_backend_changed
        )
        toolbar.addWidget(self.compute_backend)

        high_button = QPushButton("Run high-accuracy once")
        high_button.setObjectName("highAccuracyButton")
        high_button.clicked.connect(self.run_high_accuracy)
        toolbar.addWidget(high_button)
        self.addToolBar(toolbar)

    def _create_status_bar(self) -> None:
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("statusLabel")
        self.progress = QProgressBar()
        self.progress.setObjectName("calculationProgress")
        self.progress.setRange(0, 0)
        self.progress.setMaximumWidth(220)
        self.progress.hide()
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(self.progress)

    def _set_progress_active(self, owner: str, active: bool) -> None:
        if active:
            self._progress_owners.add(str(owner))
        else:
            self._progress_owners.discard(str(owner))
        self.progress.setVisible(bool(self._progress_owners))

    def _refresh_assembly_views(self) -> None:
        self._runtime_targets = runtime_targets(self.state)
        # The persisted runtime key predates the explicit TOML part name.
        # Expose the same live object under the active assembly key so the
        # Objective Stigmator stays on the optical page with working controls.
        objective_stigmator = self._runtime_targets.get(
            "objective_stigmator"
        )
        if objective_stigmator is not None:
            self._runtime_targets.setdefault(
                "objective_stigmator", objective_stigmator
            )
        anchors = self.manifest_editor.anchor_records(self.assembly)
        self._anchors_by_key = {record.part_key: record for record in anchors}
        self.assembly_panel.load_assembly(
            self.assembly, self._runtime_targets
        )
        condenser_key, projector_key = self._state_operating_mode_keys(
            self.state
        )
        self.assembly_panel.load_operating_modes(
            self.selection, condenser_key, projector_key
        )
        self.assembly_panel.set_direct_alignment_state(self.state)
        self.workspace.scan_control.set_state(self.state)
        self.workspace.sample_page.set_state(self.state)
        self.log_output.appendPlainText(
            f"Assembly validated: {len(self.assembly.parts)} parts, "
            f"{len(anchors)} confirmed anchors."
        )

    @staticmethod
    def _state_operating_mode_keys(state) -> tuple[str, str]:
        condenser_key = (
            "micro_probe"
            if str(state.illumination_mode).upper() == "TEM"
            else "nano_probe"
        )
        projector_key = (
            "imaging"
            if str(state.projector_mode).lower() == "image"
            else "diffraction"
        )
        return condenser_key, projector_key

    def _apply_state_operating_modes(self, state, selection):
        """Apply the live mode labels only when the assembly supports them."""

        selection = self.catalog.normalise_selection(selection)
        condenser_key, projector_key = self._state_operating_mode_keys(state)
        available_condenser = {
            mode.key
            for mode in compatible_modes(
                "condenser", selection.column, selection.recording
            )
        }
        available_projector = {
            mode.key
            for mode in compatible_modes(
                "projector", selection.column, selection.recording
            )
        }
        if (
            condenser_key not in available_condenser
            or projector_key not in available_projector
        ):
            return None
        result = apply_operating_mode_pair(
            state,
            condenser_key,
            projector_key,
            column_name=selection.column,
            recording_name=selection.recording,
        )
        apply_physical_layout_to_state(state)
        return result

    def _select_tree_item(self, selection) -> None:
        self._selected_component_key = selection.key
        runtime_target = self._runtime_targets.get(selection.key)
        manifest_target = None
        fields = ()
        if selection.module_path is not None:
            manifest_target = ManifestTarget(
                module_path=selection.module_path,
                part_key=None if selection.is_module else selection.key,
            )
            try:
                fields = self.manifest_editor.fields(manifest_target)
            except Exception as exc:
                fields = ()
                self._show_error(str(exc))
        self.parameter_panel.set_context(
            selection.label,
            runtime_target,
            manifest_target,
            fields,
            self._anchors_by_key.get(selection.key),
        )
        if not selection.is_module:
            try:
                part = self.assembly.part(selection.key)
            except KeyError:
                pass
            else:
                self.workspace.focus_component(part)
        self.parameter_panel.set_lens_diagnostics(
            self.workspace.magnetic_field.diagnostic_text(selection.key)
        )

    def _select_component_from_workspace(self, key: str) -> None:
        """Open the left editor for a component clicked in a plot."""

        key = str(key)
        if key == "sample":
            self.workspace.show_sample_page()
            self.status_label.setText(
                "Sample parameters opened in the central Sample workspace"
            )
            return
        self.instrument_dock.show()
        self.instrument_dock.raise_()
        if not self.assembly_panel.select_key(key):
            self.status_label.setText(
                f"No editable component is registered for {key}"
            )
            return
        sizes = self.instrument_editor.sizes()
        if len(sizes) == 2 and sizes[1] < 320:
            total = max(sum(sizes), 640)
            parameter_size = min(420, total - 220)
            self.instrument_editor.setSizes([
                total - parameter_size,
                parameter_size,
            ])
        if key in self._runtime_targets:
            self.parameter_panel.tabs.setCurrentIndex(0)
            self.parameter_panel.runtime_table.setFocus(
                Qt.FocusReason.OtherFocusReason
            )
        else:
            self.parameter_panel.tabs.setCurrentIndex(1)
            self.parameter_panel.manifest_table.setFocus(
                Qt.FocusReason.OtherFocusReason
            )
        self.status_label.setText(
            f"Selected {self.parameter_panel.title.text()} from layout"
        )

    def load_assembly(self, selection) -> None:
        self._invalidate_direct_alignment()
        try:
            selection = self.catalog.normalise_selection(selection)
            candidate_state = type(self.state).from_dict(self.state.to_dict())
            candidate_assembly = self.catalog.apply(candidate_state, selection)
            self._apply_state_operating_modes(candidate_state, selection)
            self.selection = selection
            self.state = candidate_state
            self.assembly = candidate_assembly
            self._refresh_assembly_views()
            self.log_output.appendPlainText(
                f"Loaded assembly: {selection.gun} | {selection.column} | "
                f"{selection.recording}"
            )
            self.schedule_preview()
        except Exception as exc:
            self._show_error(f"Unable to load assembly: {exc}")

    def apply_operating_modes(
        self, condenser_key: str, projector_key: str
    ) -> None:
        self._invalidate_direct_alignment()
        try:
            result = apply_operating_mode_pair(
                self.state,
                condenser_key,
                projector_key,
                column_name=self.selection.column,
                recording_name=self.selection.recording,
            )
            # Lens strengths do not own geometry, but the calculated objective
            # image/BFP coordinates depend on excitation and must be refreshed.
            apply_physical_layout_to_state(self.state)
            self._refresh_assembly_views()
            self.assembly_panel.set_operating_mode_keys(
                condenser_key, projector_key
            )
            details = self.assembly_panel.operating_mode_status.text()
            self.assembly_panel.set_operating_mode_status(
                f"Applied: {result.summary}. {details}"
            )
            self.log_output.appendPlainText(
                f"Applied operating preset: {result.summary}. "
                f"Updated {len(result.changed_devices)} optical devices."
            )
            self.schedule_preview()
        except Exception as exc:
            self._show_error(f"Unable to apply operating preset: {exc}")

    def apply_direct_alignment(self, key: str, target: float) -> None:
        """Submit one transactional user-level coupled lens adjustment."""

        self.preview_timer.stop()
        self.calculations.invalidate_pending()
        self._set_progress_active("calculation", False)
        try:
            self._direct_alignment_state_token = repr(self.state.to_dict())
            self.direct_alignments.submit(self.state, key, target)
        except Exception as exc:
            self._direct_alignment_state_token = None
            self.assembly_panel.set_direct_alignment_busy(None)
            self._set_progress_active("direct_alignment", False)
            self.assembly_panel.set_direct_alignment_message(
                f"Direct Alignment could not start: {exc}", error=True
            )
            self._show_error(f"Unable to apply Direct Alignment: {exc}")

    def _direct_alignment_started(self, key: str, target: float) -> None:
        self.assembly_panel.set_direct_alignment_busy(key)
        self._set_progress_active("direct_alignment", True)
        self.status_label.setText(
            f"Direct Alignment solving {key}: {target:g}..."
        )

    def _direct_alignment_ready(
        self, key: str, result, duration: float
    ) -> None:
        definition = direct_alignment_by_key(key)
        expected_keys = set(definition.devices)
        result_keys = set(result.strengths)
        if result.key != key or result_keys != expected_keys:
            lenses = {lens.key: lens for lens in self.state.lenses}
            current = {
                lens_key: float(lenses[lens_key].percent)
                for lens_key in expected_keys
                if lens_key in lenses
            }
            result = replace(
                result,
                key=key,
                success=False,
                strengths=current,
                message=(
                    "The background result did not match the submitted "
                    "Direct Alignment key and exact coupled-device set; it "
                    "was rejected without changing any lens."
                ),
            )
        state_is_current = (
            self._direct_alignment_state_token is not None
            and repr(self.state.to_dict())
            == self._direct_alignment_state_token
        )
        if not state_is_current:
            current = {
                lens.key: float(lens.percent)
                for lens in self.state.lenses
                if lens.key in result.strengths
            }
            result = replace(
                result,
                success=False,
                strengths=current,
                message=(
                    "The microscope state changed while the background solve "
                    "was running; the stale result was discarded and no "
                    "lens value was changed."
                ),
            )

        if result.success:
            # Reject any ordinary calculation snapshot that may have been
            # submitted while the Direct Alignment worker was running.
            self.calculations.invalidate_pending()
            self._set_progress_active("calculation", False)
            lenses = {lens.key: lens for lens in self.state.lenses}
            updates = []
            try:
                for lens_key, value in result.strengths.items():
                    lens = lenses.get(lens_key)
                    if lens is None or not bool(
                        getattr(lens, "enabled", True)
                    ):
                        raise ValueError(
                            f"Coupled lens {lens_key!r} is no longer available"
                        )
                    numeric = float(value)
                    if not 0.0 <= numeric <= float(lens.max_percent):
                        raise ValueError(
                            f"Coupled lens {lens_key!r} result is outside limits"
                        )
                    updates.append((lens, numeric))
            except Exception as exc:
                current = {
                    lens_key: float(lenses[lens_key].percent)
                    for lens_key in result.strengths
                    if lens_key in lenses
                }
                result = replace(
                    result,
                    success=False,
                    strengths=current,
                    message=(
                        f"The background result could not be committed: {exc}. "
                        "No lens value was changed."
                    ),
                )

        if result.success:
            previous = [(lens, float(lens.percent)) for lens, _ in updates]
            previous_equivalent_image_lenses = bool(
                getattr(
                    self.state, "equivalent_image_lenses_enabled", False
                )
            )
            try:
                for lens, numeric in updates:
                    lens.percent = numeric
                if key == "image_magnification":
                    self.state.equivalent_image_lenses_enabled = True
                self._refresh_assembly_views()
            except Exception as exc:
                for lens, numeric in previous:
                    lens.percent = numeric
                self.state.equivalent_image_lenses_enabled = (
                    previous_equivalent_image_lenses
                )
                result = replace(
                    result,
                    success=False,
                    strengths={lens.key: numeric for lens, numeric in previous},
                    message=(
                        f"The coupled values passed optical validation but "
                        f"the live GUI refresh failed: {exc}. The exact "
                        "previous lens values were restored."
                    ),
                )

        if result.success:
            strengths = ", ".join(
                f"{lens_key}={value:.5g}%"
                for lens_key, value in result.strengths.items()
            )
            self.log_output.appendPlainText(
                f"Direct Alignment applied in {duration:.3f} s: "
                f"{result.message} Coupled values: {strengths}."
            )
            self.status_label.setText(
                f"Direct Alignment applied: {result.achieved:.6g} "
                f"{result.unit}"
            )
            self.schedule_preview()
        else:
            self.log_output.appendPlainText(
                f"Direct Alignment not applied after {duration:.3f} s: "
                f"{result.message}"
            )
            self.status_label.setText(
                "Direct Alignment not applied; live lens values are unchanged"
            )
        self.assembly_panel.show_direct_alignment_result(result)

    def _direct_alignment_failed(self, key: str, message: str) -> None:
        self.assembly_panel.set_direct_alignment_message(
            f"Direct Alignment {key} failed: {message}", error=True
        )
        self._show_error(
            f"Direct Alignment {key} failed without changing lenses: {message}"
        )

    def _direct_alignment_finished(self, _key: str) -> None:
        self._direct_alignment_state_token = None
        self.assembly_panel.set_direct_alignment_busy(None)
        self._set_progress_active("direct_alignment", False)

    def match_energy_filter_to_ht(self) -> None:
        self._invalidate_direct_alignment()
        try:
            from temsim.optics.energy_filter import (
                match_energy_filter_to_voltage,
            )
            match = match_energy_filter_to_voltage(self.state)
            self._refresh_assembly_views()
            self.assembly_panel.select_key("energy_filter")
            detail = (
                f", dispersion {match.slit_dispersion_um_per_ev:.6g} um/eV"
                if match.slit_dispersion_um_per_ev is not None
                else ""
            )
            if match.diagnostic_message:
                detail += f"; diagnostic: {match.diagnostic_message}"
            self.log_output.appendPlainText(
                f"Energy Filter matched to {match.target_voltage_kv:g} kV; "
                f"rigidity scale {match.rigidity_scale:.8g}{detail}."
            )
            self.schedule_preview()
        except Exception as exc:
            self._show_error(f"Unable to match Energy Filter: {exc}")

    def schedule_preview(self, _parameter: str = "") -> None:
        self.preview_timer.start(self.PREVIEW_DEBOUNCE_MS)

    def _runtime_parameter_changed(self, parameter: str = "") -> None:
        # A background coupled solution was calculated for the pre-edit state.
        # Its generation must not be allowed to overwrite a newer manual edit.
        self._invalidate_direct_alignment()
        self.schedule_preview(parameter)

    def _invalidate_direct_alignment(self) -> None:
        was_running = self._direct_alignment_state_token is not None
        self.direct_alignments.invalidate_pending()
        self._direct_alignment_state_token = None
        self.assembly_panel.set_direct_alignment_busy(None)
        if was_running:
            self._set_progress_active("direct_alignment", False)
            self.assembly_panel.set_direct_alignment_message(
                "Direct Alignment cancelled because the microscope state "
                "changed before the background solve completed."
            )

    def _compute_backend_changed(self, _index: int) -> None:
        self._invalidate_direct_alignment()
        backend = str(self.compute_backend.currentData() or BACKEND_AUTO)
        self.state.acceleration_backend = backend
        self.state.acceleration_enabled = backend != BACKEND_CPU
        if backend == BACKEND_CUDA:
            ray_status = cuda_capability()
            wave_status = cupy_capability()
            detail = (
                f" (ray: {ray_status.detail}; wave: {wave_status.detail})"
            )
        else:
            detail = ""
        self.status_label.setText(f"Compute backend: {backend}{detail}")
        self.log_output.appendPlainText(
            f"Compute backend requested: {backend}{detail}."
        )

    def run_preview(self) -> None:
        if self._direct_alignment_state_token is not None:
            self.status_label.setText(
                "Preview deferred until Direct Alignment finishes"
            )
            return
        self.calculations.submit(
            self.state,
            "Preview",
            self.PREVIEW_RAYS,
            self.PREVIEW_STEP_MM,
        )

    def run_high_accuracy(self) -> None:
        if self._direct_alignment_state_token is not None:
            self.status_label.setText(
                "High-accuracy calculation deferred until Direct Alignment "
                "finishes"
            )
            return
        self.preview_timer.stop()
        try:
            estimate = estimate_calculation_memory_bytes(
                self.state,
                "High accuracy",
                self.high_rays.value(),
                self.high_step.value(),
            )
            self.log_output.appendPlainText(
                "High accuracy estimated peak memory: "
                f"{format_memory_size(estimate)}."
            )
            self.calculations.submit(
                self.state,
                "High accuracy",
                self.high_rays.value(),
                self.high_step.value(),
            )
        except ValueError as exc:
            self._show_error(str(exc))

    def _calculation_started(self, quality: str) -> None:
        self._set_progress_active("calculation", True)
        self.status_label.setText(f"{quality} calculation running...")

    def _calculation_ready(self, quality: str, result, duration: float) -> None:
        self.workspace.display_result(result, quality)
        if self._selected_component_key is not None:
            self.parameter_panel.set_lens_diagnostics(
                self.workspace.magnetic_field.diagnostic_text(
                    self._selected_component_key
                )
            )
        metrics = result.simulation.metrics
        self.assembly_panel.update_direct_alignment_metrics(metrics)
        mode = metrics.get("mode", "unknown")
        backend = str(
            getattr(result.state_snapshot, "active_backend", "CPU")
        )
        wave_result = getattr(result, "wave_imaging", None)
        wave_backend = (
            str(wave_result.metrics.get("wave_compute_backend", "unknown"))
            if wave_result is not None
            else None
        )
        wave_status = f" | wave: {wave_backend}" if wave_backend else ""
        wave_log = f", wave backend={wave_backend}" if wave_backend else ""
        self.status_label.setText(
            f"{quality} completed in {duration:.3f} s | "
            f"mode: {mode} | rays: {backend}{wave_status}"
        )
        self.log_output.appendPlainText(
            f"{quality}: {duration:.3f} s, "
            f"{result.simulation.incident.x.shape[1]} rays, mode={mode}, "
            f"ray backend={backend}{wave_log}."
        )

    def _calculation_failed(self, quality: str, message: str) -> None:
        self._show_error(f"{quality} calculation failed: {message}")

    def _calculation_finished(self, _quality: str) -> None:
        self._set_progress_active("calculation", False)

    def _save_manifest_updates(self, target, updates) -> None:
        if not updates:
            self.status_label.setText("No TOML values changed")
            return
        self._invalidate_direct_alignment()
        try:
            configuration = layout_configuration_from_state(self.state)
            originals = self.manifest_editor.save(
                target, updates, configuration
            )
            try:
                candidate_state = type(self.state).from_dict(
                    self.state.to_dict()
                )
                candidate_assembly = self.catalog.apply(
                    candidate_state, self.selection
                )
                self._apply_state_operating_modes(
                    candidate_state, self.selection
                )
            except Exception:
                from temsim import module_manifest

                module_manifest.restore_manifest_texts(
                    originals, root=self.manifest_editor.root
                )
                raise
            self.state = candidate_state
            self.assembly = candidate_assembly
            self._refresh_assembly_views()
            self.status_label.setText(
                f"Saved and validated {Path(target.module_path).name}"
            )
            self.log_output.appendPlainText(
                f"TOML saved: {target.module_path}; all active anchors confirmed."
            )
            self.schedule_preview()
        except Exception as exc:
            self._show_error(f"TOML was not saved: {exc}")

    def save_profile(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save operating profile",
            "",
            "TOML profiles (*.toml)",
        )
        if not path:
            return
        try:
            save_profile(path, self.state, self.selection)
            self.status_label.setText(f"Saved profile: {Path(path).name}")
        except Exception as exc:
            self._show_error(f"Unable to save profile: {exc}")

    def open_profile(self) -> None:
        self._invalidate_direct_alignment()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open operating profile",
            "",
            "TOML profiles (*.toml)",
        )
        if not path:
            return
        try:
            selection, values = read_profile(path)
            selection = self.catalog.normalise_selection(selection)
            candidate_state = type(self.state).from_dict(self.state.to_dict())
            candidate_assembly = self.catalog.apply(
                candidate_state, selection
            )
            skipped = apply_profile_values(candidate_state, values)
            # Reassert the catalog-owned topology and TOML geometry after the
            # operating values have been applied. Only validated operating
            # fields survive this second assembly resolution.
            apply_physical_layout_to_state(
                candidate_state, preserve_operating_parameters=True
            )
            candidate_assembly = candidate_state._resolved_assembly
            self.selection = selection
            self.state = candidate_state
            self.assembly = candidate_assembly
            self.assembly_panel.set_selection(selection)
            self._refresh_assembly_views()
            self.log_output.appendPlainText(
                f"Loaded profile: {path}; skipped values: {len(skipped)}."
            )
            self.schedule_preview()
        except Exception as exc:
            self._show_error(f"Unable to open profile: {exc}")

    def reload_toml_catalog(self) -> None:
        self._invalidate_direct_alignment()
        try:
            catalog = AssemblyCatalog()
            audit = self.manifest_editor.validate_catalog()
            selection = catalog.normalise_selection(self.selection)
            candidate_state = type(self.state).from_dict(self.state.to_dict())
            assembly = catalog.apply(candidate_state, selection)
            self._apply_state_operating_modes(
                candidate_state, selection
            )
            self.catalog = catalog
            self.selection = selection
            self.state = candidate_state
            self.assembly = assembly
            self.assembly_panel.reload_catalog(catalog, selection)
            self._refresh_assembly_views()
            self.status_label.setText(
                f"TOML catalog valid: {audit.part_definition_count} "
                f"variant-scoped definitions, "
                f"{audit.logical_part_key_count} logical part keys, and "
                f"{audit.assembly_count} collision-free assemblies"
            )
            self.schedule_preview()
        except Exception as exc:
            self._show_error(f"TOML catalog reload failed: {exc}")

    def _show_error(self, message: str) -> None:
        self.status_label.setText(message)
        self.log_output.appendPlainText(f"ERROR: {message}")
        QMessageBox.critical(self, "TEM Simulator", message)

    def _restore_workspace(self) -> None:
        settings = QSettings()
        geometry = settings.value(self.SETTINGS_GEOMETRY, QByteArray())
        state = settings.value(self.SETTINGS_STATE, QByteArray())
        if isinstance(geometry, QByteArray) and not geometry.isEmpty():
            self.restoreGeometry(geometry)
        if isinstance(state, QByteArray) and not state.isEmpty():
            self.restoreState(state)

    def reset_workspace(self) -> None:
        for dock, area in (
            (self.instrument_dock, Qt.DockWidgetArea.LeftDockWidgetArea),
            (self.log_dock, Qt.DockWidgetArea.BottomDockWidgetArea),
        ):
            dock.setFloating(False)
            self.addDockWidget(area, dock)
            dock.show()
        self.instrument_editor.setSizes([430, 430])
        self.resize(1500, 920)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        settings = QSettings()
        settings.setValue(self.SETTINGS_GEOMETRY, self.saveGeometry())
        settings.setValue(self.SETTINGS_STATE, self.saveState())
        self.calculations.pool.clear()
        self.calculations.pool.waitForDone(3_000)
        super().closeEvent(event)
