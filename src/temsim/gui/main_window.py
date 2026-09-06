"""Main TEM Simulator v2 desktop window."""

from __future__ import annotations

from temsim.gui.input_policy import (
    install_numeric_input_policy,
    WheelSafeDoubleSpinBox as QDoubleSpinBox,
    WheelSafeComboBox as QComboBox,
    WheelSafeSpinBox as QSpinBox,
)

from pathlib import Path
from dataclasses import replace

from PySide6.QtCore import QByteArray, QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QToolBar,
    QWidget,
)

from temsim.assembly_catalog import AssemblyCatalog
from temsim.column.state_layout import (
    apply_physical_layout_to_state,
    layout_configuration_from_state,
)
from temsim.component_keys import ENERGY_FILTER_INTERNAL_KEYS
from temsim.calculation_cache import external_model_signature
from temsim.cache_preferences import load_cache_preferences
from temsim.design_explorer import (
    HighAccuracyRequest,
    capture_design_snapshot,
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
from temsim.gui.design_sweep_controller import DesignSweepController
from temsim.gui.operating_preset_controller import OperatingPresetController
from temsim.gui.parameter_panel import ParameterPanel
from temsim.gui.instrument_tree import TreeSelection
from temsim.gui.visualization import VisualizationWorkspace
from temsim.gui.workspace_layouts import WorkspaceLayouts
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
from temsim.simulation_modes import MODE_BY_KEY, mode_key, switch_mode, promote_custom_mode
from temsim.gui.simulation_menu import SimulationMenu


class MainWindow(QMainWindow):
    SETTINGS_GEOMETRY = "main_window/geometry"
    SETTINGS_STATE = "main_window/state"
    SETTINGS_LIVE_TUNING_LAYOUT = "main_window/live_tuning_layout_initialized"
    PREVIEW_RAYS = 49
    PREVIEW_STEP_MM = 1.0
    INITIAL_PREVIEW_DELAY_MS = 50
    PREVIEW_DEBOUNCE_MS = 250
    DESIGN_EXPLORER_DEBOUNCE_MS = 200

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        install_numeric_input_policy()
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
        switch_mode(self.state, "ideal")
        self.manifest_editor = ManifestEditor()
        catalog_audit = self.manifest_editor.validate_catalog()
        self._runtime_targets = {}
        self._anchors_by_key = {}
        self._selected_component_key = None
        self._selected_energy_filter_key = "energy_filter"

        self.workspace = VisualizationWorkspace(self)
        self.workspace.model_inspector.set_state(self.state)
        self.workspace.model_inspector.changed.connect(self._runtime_parameter_changed)
        self.workspace.model_inspector.error.connect(self._show_error)
        self.setCentralWidget(self.workspace)

        self.assembly_panel = AssemblyPanel(
            self.catalog, self.selection, self
        )
        self.parameter_panel = ParameterPanel(self)
        self.parameter_panel.tabs.setObjectName("instrumentParameterTabs")
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
        self.live_tuning_dock = self._create_dock(
            "Live tuning", "liveTuningDock", self.workspace.interactive_calculation,
            Qt.DockWidgetArea.LeftDockWidgetArea,
        )
        self.tabifyDockWidget(self.instrument_dock, self.live_tuning_dock)
        self.live_tuning_dock.hide()
        self.workspace.live_tuning_toggle.setDefaultAction(self.live_tuning_dock.toggleViewAction())
        self.live_tuning_dock.toggleViewAction().triggered.connect(self._live_tuning_visibility_requested)
        self.log_dock = self._create_dock(
            "Status and calculation log", "logDock", self.log_output,
            Qt.DockWidgetArea.BottomDockWidgetArea,
        )

        self.calculations = CalculationController(self)
        self._cache_settings = QSettings()
        self._cache_settings_dialog = None
        self._apply_cache_preferences(load_cache_preferences(self._cache_settings))
        self.design_sweeps = DesignSweepController(
            self,
            catalog=self.catalog,
            artifact_store=self.calculations.artifact_store,
        )
        self.direct_alignments = DirectAlignmentController(self)
        self.workspace.interactive_calculation.capture_requested.connect(self._capture_interactive_settings)
        self.workspace.interactive_calculation.build_requested.connect(self._build_interactive_cache)
        self.workspace.interactive_calculation.tuning_changed.connect(self._apply_interactive_tuning)
        self.workspace.interactive_calculation.current_state = lambda: self.state
        self.workspace.interactive_calculation.high_accuracy_requested.connect(self.run_high_accuracy)
        self.workspace.interactive_calculation.operation_allowed = self._interactive_work_allowed
        self._preview_deferred_for_interactive = False
        self.workspace.interactive_calculation.operation_finished.connect(self._interactive_operation_finished)
        self.operating_presets = OperatingPresetController(self)
        self._preset_state_token: str | None = None
        self._direct_alignment_state_token: str | None = None
        self._progress_owners: set[str] = set()
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(self.PREVIEW_DEBOUNCE_MS)
        self._interactive_preview_generation: int | None = None
        self._interactive_preview_pending = False
        self.design_explorer_timer = QTimer(self)
        self.design_explorer_timer.setSingleShot(True)
        self.design_explorer_timer.setInterval(
            self.DESIGN_EXPLORER_DEBOUNCE_MS
        )
        self._design_explorer_dirty = True
        self._preview_deferred_for_sweep = False

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
        self.assembly_panel.component_activated.connect(
            self._reveal_tree_item
        )
        self.workspace.component_selected.connect(
            self._select_component_from_workspace
        )
        self.workspace.magnetic_field.field_map_import_requested.connect(
            self._import_lens_field_map
        )
        self.workspace.magnetic_field.field_map_clear_requested.connect(
            self._clear_lens_field_map
        )
        self.workspace.scan_parameters_changed.connect(
            self._runtime_parameter_changed
        )
        self.workspace.scan_error.connect(self._show_error)
        self.workspace.design_explorer.capture_requested.connect(
            self._capture_design_snapshot
        )
        self.workspace.design_explorer.sweep_requested.connect(
            self._run_design_sweep
        )
        self.workspace.design_explorer.sweep_cancel_requested.connect(
            self.design_sweeps.cancel
        )
        self.workspace.design_explorer.sweep_error.connect(self._show_error)
        self.workspace.calculation_artifacts_changed.connect(
            self._calculation_artifacts_changed
        )
        self.workspace.tabs.currentChanged.connect(
            self._design_explorer_tab_changed
        )
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
        energy_filter_parameters = self.workspace.energy_filter_parameters
        energy_filter_parameters.runtime_changed.connect(
            self._runtime_parameter_changed
        )
        energy_filter_parameters.energy_filter_match_requested.connect(
            self.match_energy_filter_to_ht
        )
        energy_filter_parameters.manifest_save_requested.connect(
            self._save_manifest_updates
        )
        energy_filter_parameters.error.connect(
            self._show_error
        )
        self.preview_timer.timeout.connect(self.run_preview)
        self.design_explorer_timer.timeout.connect(
            self._refresh_design_explorer
        )
        self.calculations.started.connect(self._calculation_started)
        self.calculations.progress_changed.connect(
            self._calculation_progress
        )
        self.calculations.result_ready.connect(self._calculation_ready)
        self.calculations.failed.connect(self._calculation_failed)
        self.calculations.finished.connect(self._calculation_finished)
        self.design_sweeps.started.connect(
            self.workspace.design_explorer.set_sweep_running
        )
        self.design_sweeps.progress_changed.connect(
            self.workspace.design_explorer.set_sweep_progress
        )
        self.design_sweeps.result_ready.connect(self._design_sweep_ready)
        self.design_sweeps.failed.connect(self._design_sweep_failed)
        self.design_sweeps.finished.connect(self._design_sweep_finished)
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
        self.operating_presets.result_ready.connect(self._operating_preset_ready)
        self.operating_presets.failed.connect(self._operating_preset_failed)
        self.operating_presets.finished.connect(self._operating_preset_finished)

        self._create_actions()
        self._create_toolbar()
        self._create_status_bar()
        self._refresh_assembly_views()
        self.workspace_layouts = WorkspaceLayouts(self, QSettings(), self.layouts_menu)
        self._restore_workspace()
        self.workspace_layouts.restore_active()
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
        view_menu.addAction(self.live_tuning_dock.toggleViewAction())
        view_menu.addAction(self.log_dock.toggleViewAction())
        self.layouts_menu = view_menu.addMenu("Layouts")
        view_menu.addSeparator()
        view_menu.addAction(self.reset_layout_action)
        self.simulation_menu = SimulationMenu(self)
        self.menuBar().addMenu(self.simulation_menu)
        self.simulation_menu.mode_requested.connect(self.set_simulation_mode)
        self.simulation_menu.configure_requested.connect(
            lambda: self.workspace.tabs.setCurrentWidget(self.workspace.model_inspector)
        )
        self.simulation_menu.aboutToShow.connect(self._refresh_simulation_mode)
        self.simulation_menu.set_state(self.state)
        self.simulation_menu.addSeparator()
        self.cache_settings_action = self.simulation_menu.addAction("Performance and cache...")
        self.cache_settings_action.setObjectName("performanceCacheAction")
        self.cache_settings_action.triggered.connect(self._show_cache_settings)

    def _apply_cache_preferences(self, preferences) -> None:
        """Change retention only; never invalidate results or request a solve."""
        self.calculations.configure_cache(**preferences.controller_kwargs())
        self.workspace.set_ray_display_cache_limit_bytes(preferences.ray_display_cache_budget_bytes)

    def _show_cache_settings(self) -> None:
        from temsim.gui.cache_settings import CacheSettingsDialog

        if self._cache_settings_dialog is None:
            self._cache_settings_dialog = CacheSettingsDialog(
                self._cache_settings,
                lambda: {
                    "calculation": self.calculations.cache_statistics(),
                    "ray_display": self.workspace.ray_display_cache_info(),
                },
                parent=self,
            )
            self._cache_settings_dialog.preferencesChanged.connect(self._apply_cache_preferences)
        self._cache_settings_dialog.show()
        self._cache_settings_dialog.raise_()
        self._cache_settings_dialog.activateWindow()

    def _refresh_simulation_mode(self) -> None:
        if not hasattr(self, "simulation_menu"):
            return
        self.simulation_menu.set_state(self.state)
        if hasattr(self, "simulation_mode_label"):
            selected = MODE_BY_KEY[mode_key(self.state)]
            self.simulation_mode_label.setText(selected.label)
            self.simulation_mode_label.setToolTip(selected.detail)

    def set_simulation_mode(self, key: str) -> None:
        try:
            changed = switch_mode(self.state, key)
        except ValueError as exc:
            self._refresh_simulation_mode()
            self._show_error(str(exc))
            return
        self._refresh_simulation_mode()
        if not changed:
            return
        self._invalidate_direct_alignment()
        self.workspace.model_inspector.set_state(self.state)
        self.parameter_panel.refresh_runtime_values()
        selected = MODE_BY_KEY[mode_key(self.state)]
        self.log_output.appendPlainText(f"Column lens model: {selected.label}. No preset recalculation; other model caches retained.")
        self.status_label.setText(f"{selected.label} selected; previous plots remain until the next result")
        self.schedule_preview("simulation_mode")

    def _create_toolbar(self) -> None:
        toolbar = QToolBar("Calculation", self)
        toolbar.setObjectName("calculationToolbar")
        toolbar.setMovable(False)

        preview_button = QPushButton("Update rays")
        preview_button.setObjectName("previewButton")
        preview_button.clicked.connect(self.run_preview)
        toolbar.addWidget(preview_button)
        self.tuning_quality = QComboBox()
        self.tuning_quality.setObjectName("tuningQuality")
        self.tuning_quality.addItem("Preview: fast rays", "Preview")
        self.tuning_quality.addItem("Medium: sampled beam", "Medium")
        self.tuning_quality.setToolTip("Live lens edits update ray optics only. Medium adds internal samples and zero-current boundary probes. High accuracy is a separate, explicit calculation.")
        self.tuning_quality.currentIndexChanged.connect(self._tuning_quality_changed)
        toolbar.addWidget(self.tuning_quality)
        toolbar.addSeparator()

        toolbar.addWidget(QLabel("High-accuracy rays"))
        self.high_rays = QSpinBox()
        self.high_rays.setObjectName("highAccuracyRayCount")
        self.high_rays.setRange(1_000, 1_000_000)
        self.high_rays.setSingleStep(1_000)
        # Tuned for the supported 32 GiB workstation profile. The controller
        # also enforces a conservative 24 GiB process budget for custom values.
        self.high_rays.setValue(15_000)
        self.high_rays.valueChanged.connect(
            self._schedule_design_explorer_refresh
        )
        toolbar.addWidget(self.high_rays)

        toolbar.addWidget(QLabel("Step (mm)"))
        self.high_step = QDoubleSpinBox()
        self.high_step.setObjectName("highAccuracyStep")
        self.high_step.setDecimals(4)
        self.high_step.setRange(0.01, 1.0)
        self.high_step.setValue(0.1)
        self.high_step.valueChanged.connect(
            self._schedule_design_explorer_refresh
        )
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
        self.progress.setMinimumWidth(260)
        self.progress.setMaximumWidth(420)
        self.progress.setTextVisible(True)
        self.progress.hide()
        self.statusBar().addWidget(self.status_label, 1)
        self.simulation_mode_label = QLabel()
        self.simulation_mode_label.setObjectName("simulationModeLabel")
        self.simulation_mode_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.statusBar().addPermanentWidget(self.simulation_mode_label)
        self.statusBar().addPermanentWidget(self.progress)
        self._refresh_simulation_mode()

    def _set_progress_active(self, owner: str, active: bool) -> None:
        if active:
            self._progress_owners.add(str(owner))
        else:
            self._progress_owners.discard(str(owner))
        self.progress.setVisible(bool(self._progress_owners))

    def _refresh_assembly_views(self) -> None:
        self._refresh_simulation_mode()
        self.workspace.model_inspector.set_state(self.state)
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
        energy_filter_selections = self._energy_filter_selections()
        available_energy_filter_keys = {
            selection.key for selection in energy_filter_selections
        }
        selected_energy_filter_key = self._selected_energy_filter_key
        if selected_energy_filter_key not in available_energy_filter_keys:
            selected_energy_filter_key = (
                "energy_filter"
                if "energy_filter" in available_energy_filter_keys
                else next(iter(available_energy_filter_keys), None)
            )
        self.workspace.set_energy_filter_components(
            tuple(
                (selection.key, selection.label)
                for selection in energy_filter_selections
            ),
            selected_energy_filter_key,
        )
        if selected_energy_filter_key is not None:
            self._select_energy_filter_component(
                selected_energy_filter_key,
                activate_page=False,
                focus_editor=False,
            )
        else:
            self.workspace.energy_filter_parameters.set_context(
                "No Energy Filter in the active assembly",
                None,
                None,
                (),
                None,
            )
        self.workspace.scan_control.set_state(self.state)
        self.workspace.sample_page.set_state(self.state)
        self.workspace.eds_page.set_state(self.state)
        self.log_output.appendPlainText(
            f"Assembly validated: {len(self.assembly.parts)} parts, "
            f"{len(anchors)} confirmed anchors."
        )
        self._schedule_design_explorer_refresh()

    def _design_explorer_is_visible(self) -> bool:
        page = self.workspace.design_explorer
        return self.workspace.tabs.currentWidget() is page

    def _schedule_design_explorer_refresh(self, *_args) -> None:
        """Coalesce edits and avoid signature work while the page is hidden."""

        self._design_explorer_dirty = True
        if self._design_explorer_is_visible():
            self.design_explorer_timer.start()

    def _calculation_artifacts_changed(self, result) -> None:
        self.calculations.refresh_cached_result_metadata(result)
        self._schedule_design_explorer_refresh()

    def _design_explorer_tab_changed(self, _index: int) -> None:
        if self._design_explorer_is_visible() and self._design_explorer_dirty:
            self.design_explorer_timer.start(0)

    def _refresh_design_explorer(self, *_args) -> None:
        """Refresh cache reuse rows without starting or reprioritising work."""

        if not self._design_explorer_is_visible():
            self._design_explorer_dirty = True
            return
        calculations = getattr(self, "calculations", None)
        high_rays = getattr(self, "high_rays", None)
        high_step = getattr(self, "high_step", None)
        if calculations is None or high_rays is None or high_step is None:
            return
        try:
            plan = calculations.describe_high_accuracy_reuse(
                self.state,
                high_rays.value(),
                high_step.value(),
                completed_summary=(
                    self.workspace.high_accuracy_result_summary()
                ),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            page = self.workspace.design_explorer
            page.set_calculation_status("Dependency status unavailable")
            page.cache_summary.setToolTip(str(exc))
            return
        self.workspace.design_explorer.set_product_statuses(
            plan.product_statuses
        )
        self._design_explorer_dirty = False

    def _capture_design_snapshot(self, slot: str) -> None:
        """Capture current settings for A/B comparison without calculating."""

        request = HighAccuracyRequest(
            self.high_rays.value(), self.high_step.value()
        )
        try:
            plan = self.calculations.describe_high_accuracy_reuse(
                self.state,
                request.ray_count,
                request.step_mm,
                completed_summary=(
                    self.workspace.high_accuracy_result_summary()
                ),
            )
            snapshot = capture_design_snapshot(
                self.state,
                self.selection,
                slot=slot,
                request=request,
                request_signatures=plan.request_signatures,
            )
        except (AttributeError, OSError, TypeError, ValueError) as exc:
            self._show_error(f"Could not capture design {slot}: {exc}")
            return
        self.workspace.design_explorer.set_capture(snapshot)
        self.status_label.setText(f"Design {snapshot.slot} captured")

    def _live_tuning_visibility_requested(self, visible: bool) -> None:
        if visible:
            self.workspace.show_ray_diagram()
            self.live_tuning_dock.raise_()
            if not self._live_tuning_layout_initialized:
                # Give both control columns room on first use only. Later
                # toggles keep the user's dock size, including after restart.
                if not self.live_tuning_dock.isFloating():
                    self.resizeDocks([self.live_tuning_dock], [900], Qt.Orientation.Horizontal)
                self._live_tuning_layout_initialized = True

    def _capture_interactive_settings(self) -> None:
        page = self.workspace.interactive_calculation
        try:
            snapshot = CalculationController._calculation_snapshot(
                self.state, "High accuracy", self.high_rays.value(), self.high_step.value())
            page.set_source(snapshot, self.calculations.completed_high_accuracy_results())
        except (ValueError, RuntimeError, TypeError) as exc:
            page.show_error(str(exc))

    def _tuning_quality_changed(self, *_):
        quality = str(self.tuning_quality.currentData())
        self.workspace.interactive_calculation.set_tuning_quality(quality)
        self.schedule_preview("tuning_quality")

    def _apply_interactive_tuning(self, axes_and_values) -> None:
        from temsim.interactive_calculation import apply_live_tuning_values
        axes_and_values = tuple(axes_and_values)
        page = self.workspace.interactive_calculation
        if (page.busy or self.design_sweeps.running
                or self._direct_alignment_state_token is not None
                or self._preset_state_token is not None
                or self.calculations._running_high_generation is not None):
            page.show_error("Finish or cancel the active calculation before live tuning")
            return
        try:
            apply_live_tuning_values(self.state, axes_and_values)
        except (ValueError, AttributeError) as exc:
            page.show_error(str(exc))
            return
        # Existing left-side controls use the same live objects.
        self.parameter_panel.refresh_live_values({axis.control.key for axis, _ in axes_and_values})
        self._runtime_parameter_changed("interactive_tuning")

    def _build_interactive_cache(self) -> None:
        page = self.workspace.interactive_calculation
        if ("calculation" in self._progress_owners or self.design_sweeps.running
                or self._direct_alignment_state_token is not None or self._preset_state_token is not None):
            page.show_error("Finish the current calculation or alignment first")
            return
        self.preview_timer.stop()
        page.start_build()

    def _interactive_work_allowed(self) -> bool:
        return not ("calculation" in self._progress_owners or self.design_sweeps.running
                    or self._direct_alignment_state_token is not None or self._preset_state_token is not None)

    def _interactive_operation_finished(self) -> None:
        if self._preview_deferred_for_interactive:
            self._preview_deferred_for_interactive = False
            self.preview_timer.start(0)

    def _run_design_sweep(self, recipe, sweep, tolerance_rules=()) -> None:
        """Start detached High-accuracy points without editing live state."""

        if (
            self.workspace.interactive_calculation.busy
            or
            "calculation" in self._progress_owners
            or self._direct_alignment_state_token is not None
            or self._preset_state_token is not None
        ):
            self.workspace.design_explorer.set_sweep_error(
                "Finish the current calculation or alignment first"
            )
            return
        try:
            if (
                not str(recipe.external_model_signature)
                or external_model_signature(self.state)
                != str(recipe.external_model_signature)
            ):
                raise ValueError(
                    "Current CIF, selected assembly, or field map differs "
                    "from this design capture; capture A/B again"
                )
            self.preview_timer.stop()
            self.design_sweeps.seed_completed_results(
                self.calculations.completed_high_accuracy_results()
            )
            self.design_sweeps.submit(
                recipe,
                sweep,
                tolerance_rules=tolerance_rules,
            )
            self.status_label.setText(
                f"Design sweep running: {len(sweep.points)} points"
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            self.workspace.design_explorer.set_sweep_error(str(exc))

    def _design_sweep_ready(self, result, duration_s: float) -> None:
        self.workspace.design_explorer.set_sweep_result(result, duration_s)
        status = "cancelled" if result.cancelled else "completed"
        self.status_label.setText(
            f"Design sweep {status}: {result.completed_points} points"
        )
        self.log_output.appendPlainText(
            f"Design sweep {status}: {result.completed_points} points in "
            f"{duration_s:.3f} s; {len(result.sensitivities)} sensitivities."
        )

    def _design_sweep_failed(self, message: str) -> None:
        self.workspace.design_explorer.set_sweep_error(message)
        self.status_label.setText(f"Design sweep failed: {message}")
        self.log_output.appendPlainText(f"ERROR: Design sweep failed: {message}")

    def _design_sweep_finished(self) -> None:
        self.workspace.design_explorer.set_sweep_finished()
        if self._preview_deferred_for_sweep:
            self._preview_deferred_for_sweep = False
            self.preview_timer.start(0)

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

    def _reveal_tree_item(self, selection, source: str) -> None:
        """Navigate a double-clicked tree component to its visual view."""

        self._select_tree_item(selection)
        if selection.is_module:
            self.status_label.setText(
                f"No axial component is associated with {selection.label}"
            )
            return
        try:
            part = self.assembly.part(selection.key)
        except KeyError:
            self.status_label.setText(
                f"No axial component is associated with {selection.label}"
            )
            return
        preferred_view = "physical" if source == "mechanical" else "ray"
        view_name = self.workspace.reveal_component(
            part, preferred_view=preferred_view
        )
        self.status_label.setText(f"Showing {part.name} in {view_name}")

    def _energy_filter_selections(self) -> tuple[TreeSelection, ...]:
        """Return the devices owned by the central Energy Filter page."""

        module_paths = dict(self.assembly.selected_module_paths)
        module_types = {
            module.key: module.type for module in self.assembly.modules
        }
        selections = []
        for key in ("energy_filter", *ENERGY_FILTER_INTERNAL_KEYS):
            runtime_target = self._runtime_targets.get(key)
            try:
                part = self.assembly.part(key)
            except KeyError:
                part = None
            if runtime_target is None and part is None:
                continue
            if part is not None:
                label = part.name
                module_path = module_paths.get(
                    module_types.get(part.module_key, "")
                )
            else:
                label = runtime_target.label
                module_path = None
            selections.append(TreeSelection(
                key=key,
                label=label,
                module_path=module_path,
            ))
        return tuple(selections)

    def _select_energy_filter_component(
        self,
        key: str,
        *,
        activate_page: bool = True,
        focus_editor: bool = True,
    ) -> bool:
        """Open one Iliad/EELS device in the Energy Filter-local editor."""

        key = str(key)
        selection = next(
            (
                candidate for candidate in self._energy_filter_selections()
                if candidate.key == key
            ),
            None,
        )
        if selection is None:
            self.status_label.setText(
                f"No Energy Filter component is registered for {key}"
            )
            return False
        runtime_target = self._runtime_targets.get(key)
        manifest_target = None
        fields = ()
        if selection.module_path is not None:
            manifest_target = ManifestTarget(
                module_path=selection.module_path,
                part_key=selection.key,
            )
            try:
                fields = self.manifest_editor.fields(manifest_target)
            except Exception as exc:
                self._show_error(str(exc))
        panel = self.workspace.energy_filter_parameters
        panel.set_context(
            selection.label,
            runtime_target,
            manifest_target,
            fields,
            self._anchors_by_key.get(key),
        )
        self.workspace.select_energy_filter_component(key)
        self._selected_energy_filter_key = key
        if activate_page:
            self.workspace.show_energy_filter_page()
        if runtime_target is not None:
            panel.tabs.setCurrentIndex(0)
            if focus_editor:
                panel.runtime_table.setFocus(
                    Qt.FocusReason.OtherFocusReason
                )
        else:
            panel.tabs.setCurrentIndex(1)
            if focus_editor:
                panel.manifest_table.setFocus(
                    Qt.FocusReason.OtherFocusReason
                )
        if activate_page:
            self.status_label.setText(
                f"Selected {selection.label} in Energy Filter Parameters"
            )
        return True

    def _select_component_from_workspace(self, key: str) -> None:
        """Open the left editor for a component clicked in a plot."""

        key = str(key)
        if key == "sample":
            self.workspace.show_sample_page()
            self.status_label.setText(
                "Sample parameters opened in the central Sample workspace"
            )
            return
        if key == "energy_filter" or key in ENERGY_FILTER_INTERNAL_KEYS:
            self._select_energy_filter_component(key)
            return
        self.instrument_dock.show()
        self.instrument_dock.raise_()
        if not self.assembly_panel.select_key(key):
            self.status_label.setText(
                f"No editable component is registered for {key}"
            )
            return
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
            if selection.beam_blanker != "None":
                condenser_key, projector_key = self._state_operating_mode_keys(
                    self.state
                )
                self._start_operating_preset(
                    selection, condenser_key, projector_key, load_assembly=True
                )
                return
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
            if self.state.nanopulser.installed:
                self._start_operating_preset(
                    self.selection, condenser_key, projector_key
                )
                return
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

    def _start_operating_preset(
        self, selection, condenser_key, projector_key, *, load_assembly=False
    ) -> None:
        if self.workspace.interactive_calculation.busy:
            self.status_label.setText("Finish or cancel the Live tuning bank operation first")
            return
        self.preview_timer.stop()
        self.calculations.invalidate_pending()
        self._set_progress_active("calculation", False)
        self._preset_state_token = repr(self.state.to_dict())
        self.assembly_panel.set_direct_alignment_busy("condenser_preset")
        self.progress.setRange(0, 0)
        self.progress.setFormat("Calibrating condenser preset")
        self._set_progress_active("operating_preset", True)
        self.status_label.setText(
            "Solving condenser preset for the installed NanoPulser geometry…"
        )
        try:
            self.operating_presets.submit(
                self.state, self.catalog, selection, condenser_key, projector_key,
                load_assembly=load_assembly,
            )
        except Exception:
            self._operating_preset_finished()
            raise

    def _operating_preset_ready(self, state, selection, result, duration) -> None:
        if self._preset_state_token != repr(self.state.to_dict()):
            self.log_output.appendPlainText(
                "Discarded condenser preset after a newer microscope edit."
            )
            return
        previous = self.state, self.selection, self.assembly
        try:
            self.state = state
            self.selection = selection
            self.assembly = state._resolved_assembly
            self.assembly_panel.set_selection(selection)
            self._refresh_assembly_views()
        except Exception as exc:
            self.state, self.selection, self.assembly = previous
            self.assembly_panel.set_selection(self.selection)
            self._refresh_assembly_views()
            self._operating_preset_failed(str(exc))
            return
        detail = (
            result.summary if result is not None
            else "Assembly loaded; no compatible condenser preset"
        )
        self.assembly_panel.set_operating_mode_status(f"Applied: {detail}")
        self.status_label.setText(f"NanoPulser assembly/preset applied in {duration:.2f} s")
        self.log_output.appendPlainText(
            f"Loaded {selection.gun} | {selection.column} | {selection.beam_blanker}: "
            f"{detail} ({duration:.2f} s)."
        )
        self.schedule_preview()

    def _operating_preset_failed(self, message) -> None:
        self._show_error(f"Unable to apply NanoPulser assembly/preset: {message}")

    def _operating_preset_finished(self) -> None:
        self._preset_state_token = None
        self.assembly_panel.set_direct_alignment_busy(None)
        self._set_progress_active("operating_preset", False)

    def apply_direct_alignment(self, key: str, target: float) -> None:
        """Submit one transactional user-level coupled lens adjustment."""
        if self.workspace.interactive_calculation.busy:
            self.status_label.setText("Finish or cancel the Live tuning bank operation first")
            return

        self._invalidate_operating_preset()
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
        self.progress.setRange(0, 0)
        self.progress.setFormat("Direct Alignment")
        self._set_progress_active("direct_alignment", True)
        self.status_label.setText(
            f"Direct Alignment solving {key}: {target:g}..."
        )

    def _direct_alignment_ready(
        self, key: str, result, duration: float
    ) -> None:
        definition = direct_alignment_by_key(key)
        expected_keys = set(definition.devices)
        expected_state_parameters = set(definition.state_parameters)
        result_keys = set(result.strengths)
        result_state_updates = dict(result.state_updates or {})
        if (
            result.key != key
            or result_keys != expected_keys
            or set(result_state_updates) != expected_state_parameters
        ):
            lenses = {lens.key: lens for lens in self.state.lenses}
            current = {
                lens_key: float(lenses[lens_key].percent)
                for lens_key in expected_keys
                if lens_key in lenses
            }
            current_state_updates = {
                name: float(getattr(self.state, name))
                for name in expected_state_parameters
                if hasattr(self.state, name)
            }
            result = replace(
                result,
                key=key,
                success=False,
                strengths=current,
                state_updates=current_state_updates,
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
                state_updates={
                    name: float(getattr(self.state, name))
                    for name in expected_state_parameters
                    if hasattr(self.state, name)
                },
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
            state_updates = []
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
                for name, value in result_state_updates.items():
                    if name != "column_current_limit_percent":
                        raise ValueError(
                            f"Unsupported state update {name!r}"
                        )
                    numeric = float(value)
                    if not definition.minimum <= numeric <= definition.maximum:
                        raise ValueError(
                            f"State update {name!r} is outside limits"
                        )
                    state_updates.append((name, numeric))
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
                    state_updates={
                        name: float(getattr(self.state, name))
                        for name in expected_state_parameters
                        if hasattr(self.state, name)
                    },
                    message=(
                        f"The background result could not be committed: {exc}. "
                        "No lens value was changed."
                    ),
                )

        if result.success:
            previous = [(lens, float(lens.percent)) for lens, _ in updates]
            previous_state = [
                (name, float(getattr(self.state, name)))
                for name, _ in state_updates
            ]
            previous_equivalent_image_lenses = bool(
                getattr(
                    self.state, "equivalent_image_lenses_enabled", False
                )
            )
            try:
                for lens, numeric in updates:
                    lens.percent = numeric
                for name, numeric in state_updates:
                    setattr(self.state, name, numeric)
                if key == "image_magnification":
                    self.state.equivalent_image_lenses_enabled = True
                self._refresh_assembly_views()
            except Exception as exc:
                for lens, numeric in previous:
                    lens.percent = numeric
                for name, numeric in previous_state:
                    setattr(self.state, name, numeric)
                self.state.equivalent_image_lenses_enabled = (
                    previous_equivalent_image_lenses
                )
                result = replace(
                    result,
                    success=False,
                    strengths={lens.key: numeric for lens, numeric in previous},
                    state_updates={
                        name: numeric for name, numeric in previous_state
                    },
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
            state_values = ", ".join(
                f"{name}={value:.6g}"
                for name, value in result_state_updates.items()
            )
            applied_values = "; ".join(
                value for value in (strengths, state_values) if value
            )
            self.log_output.appendPlainText(
                f"Direct Alignment applied in {duration:.3f} s: "
                f"{result.message} Applied values: {applied_values}."
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
            self._select_energy_filter_component("energy_filter")
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

    def _native_lens_field_provider(self, lens_key: str):
        """Return the live provider whose geometry owns an imported map."""

        from temsim.component_keys import CONDENSER_LENS_KEYS

        key = str(lens_key)
        lens = next(
            (candidate for candidate in self.state.lenses
             if candidate.key == key),
            None,
        )
        if lens is None:
            raise ValueError(f"Unknown magnetic lens: {key}")
        if key in CONDENSER_LENS_KEYS:
            return self.state.condenser_system[key]
        return lens

    def _import_lens_field_map(
        self,
        lens_key: str,
        path: str,
        provenance_kind: str,
        reference_excitation_percent: float,
        reference_polarity: int,
    ) -> None:
        """Bind one explicitly sourced SI map to the current live geometry."""

        try:
            from temsim.physics.lens_field_provider import (
                bind_imported_lens_field_map,
                lens_geometry_binding,
                load_magnetic_field_map,
            )

            provider = self._native_lens_field_provider(lens_key)
            binding = lens_geometry_binding(
                self.state, lens_key, provider
            )
            field_map = load_magnetic_field_map(
                path,
                geometry_binding=binding,
                provenance_kind=provenance_kind,
                reference_excitation_percent=(
                    float(reference_excitation_percent)
                ),
                reference_polarity=int(reference_polarity),
                source_note=(
                    "GUI import explicitly bound by the user to the active "
                    "lens/assembly; coordinates and fields are explicit SI "
                    "m/T; NPZ registration is metadata-owned, while tidy CSV "
                    "coordinates use the documented global-column convention"
                ),
            )
            promote_custom_mode(self.state)
            bind_imported_lens_field_map(
                self.state,
                lens_key,
                field_map,
                native_provider=provider,
            )
            validation = field_map.validation
            status = (
                f"{provenance_kind.upper()} map bound to {lens_key} | "
                f"grid {validation.grid_shape} | relative divergence "
                f"{validation.relative_divergence:.4g}"
            )
            if not validation.divergence_within_tolerance:
                status += " (warning: above tolerance)"
            self.workspace.magnetic_field.set_field_map_operation_status(
                status,
                error=not validation.divergence_within_tolerance,
            )
            self.log_output.appendPlainText(status + ".")
            self._mark_operating_preset_stale()
            # Use the normal runtime invalidation path. This schedules only a
            # preview and never recalculates an operating preset.
            self._runtime_parameter_changed("lens_field_map")
        except Exception as exc:
            message = f"Unable to import magnetic field map: {exc}"
            self.workspace.magnetic_field.set_field_map_operation_status(
                message, error=True
            )
            self._show_error(message)

    def _clear_lens_field_map(self, lens_key: str) -> None:
        """Remove one imported map and expose the provisional fallback."""

        try:
            from temsim.physics.lens_field_provider import (
                clear_imported_lens_field_map,
            )

            promote_custom_mode(self.state)
            clear_imported_lens_field_map(self.state, lens_key)
            status = (
                f"{lens_key}: imported map cleared | provisional TOML "
                "Gaussian fallback"
            )
            self.workspace.magnetic_field.set_field_map_operation_status(
                status
            )
            self.log_output.appendPlainText(status + ".")
            self._mark_operating_preset_stale()
            self._runtime_parameter_changed("lens_field_map")
        except Exception as exc:
            message = f"Unable to clear magnetic field map: {exc}"
            self.workspace.magnetic_field.set_field_map_operation_status(
                message, error=True
            )
            self._show_error(message)

    def _interactive_preview_in_flight(self) -> bool:
        return (self._interactive_preview_generation is not None
                and self._interactive_preview_generation == self.calculations.generation)

    def schedule_preview(self, _parameter: str = "") -> None:
        if _parameter == "interactive_tuning":
            # Finish one ray frame while edits accumulate in live state. Keep
            # only a latest-value marker, not one job/snapshot per mouse event.
            # Cancelling every 50 ms would starve any slower Preview/Medium job.
            self.workspace.mark_high_accuracy_stale()
            self._schedule_design_explorer_refresh()
            self._interactive_preview_pending = True
            if not self._interactive_preview_in_flight():
                self.preview_timer.start(0)
            return
        # Invalidate immediately rather than waiting for the debounce timer.
        # A completed worker for the pre-edit state must never be committed to
        # the new live state.  Existing complete displays remain available as
        # explicitly stale results until their replacements succeed.
        self._interactive_preview_generation = None
        self._interactive_preview_pending = False
        self.calculations.invalidate_pending()
        self.workspace.mark_high_accuracy_stale()
        self._schedule_design_explorer_refresh()
        self._set_progress_active("calculation", False)
        self.preview_timer.start(self.PREVIEW_DEBOUNCE_MS)

    def _runtime_parameter_changed(self, parameter: str = "") -> None:
        # A background coupled solution was calculated for the pre-edit state.
        # Its generation must not be allowed to overwrite a newer manual edit.
        self._invalidate_direct_alignment()
        self._refresh_simulation_mode()
        self.schedule_preview(parameter)

    def _invalidate_direct_alignment(self) -> None:
        self._invalidate_operating_preset()
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

    def _invalidate_operating_preset(self) -> None:
        self.operating_presets.invalidate_pending()
        if self._preset_state_token is not None:
            self._operating_preset_finished()
            self.log_output.appendPlainText(
                "Condenser preset cancelled after a microscope state change."
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
        self._schedule_design_explorer_refresh()

    def run_preview(self) -> None:
        if self.workspace.interactive_calculation.busy:
            self._preview_deferred_for_interactive = True
            return
        if self.design_sweeps.running:
            self._preview_deferred_for_sweep = True
            self.status_label.setText(
                "Preview deferred until the detached design sweep finishes"
            )
            return
        if (
            self._direct_alignment_state_token is not None
            or self._preset_state_token is not None
        ):
            operation = (
                "Direct Alignment" if self._direct_alignment_state_token is not None
                else "the condenser preset"
            )
            self.status_label.setText(
                f"Preview deferred until {operation} finishes"
            )
            return
        from temsim.physics.optical_tuning import TUNING_PROFILES
        profile = TUNING_PROFILES[str(self.tuning_quality.currentData())]
        self.preview_timer.stop()
        if self._interactive_preview_in_flight():
            self._interactive_preview_pending = True
            return
        interactive = self._interactive_preview_pending
        self._interactive_preview_pending = False
        self._interactive_preview_generation = None
        try:
            self.calculations.submit_background(
                self.state,
                profile.quality,
                profile.rays,
                profile.step_mm,
            )
        except ValueError as exc:
            self._show_error(str(exc))
            return
        if interactive:
            self._interactive_preview_generation = self.calculations.generation

    def run_high_accuracy(self) -> None:
        if self.workspace.interactive_calculation.busy:
            self.status_label.setText("Finish or cancel the Live tuning bank operation first")
            return
        if self.design_sweeps.running:
            self.status_label.setText(
                "High-accuracy calculation deferred until the design sweep finishes"
            )
            return
        if (
            self._direct_alignment_state_token is not None
            or self._preset_state_token is not None
        ):
            operation = (
                "Direct Alignment" if self._direct_alignment_state_token is not None
                else "the condenser preset"
            )
            self.status_label.setText(
                f"High-accuracy calculation deferred until {operation} finishes"
            )
            return
        self.preview_timer.stop()
        self._interactive_preview_generation = None
        self._interactive_preview_pending = False
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
        if quality == "High accuracy":
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self.progress.setFormat("0% · Preparing")
            self.workspace.design_explorer.set_calculation_status(
                "High accuracy running"
            )
        else:
            self.progress.setRange(0, 0)
            self.progress.setFormat(quality)
        self._set_progress_active("calculation", True)
        self.status_label.setText(f"{quality} calculation running...")

    def _calculation_progress(
        self,
        quality: str,
        completed: int,
        total: int,
        stage: str,
    ) -> None:
        if quality != "High accuracy" or total <= 0:
            return
        bounded_completed = min(max(int(completed), 0), int(total))
        self.progress.setRange(0, int(total))
        self.progress.setValue(bounded_completed)
        percentage = 100.0 * bounded_completed / int(total)
        self.progress.setFormat(f"{percentage:.1f}% · {stage}")
        self.status_label.setText(f"{quality}: {stage}...")

    def _calculation_ready(self, quality: str, result, duration: float) -> None:
        self.workspace.display_result(result, quality)
        if quality in ("Preview", "Medium"):
            self.workspace.interactive_calculation.display_tuning_status(result)
            page = self.workspace.interactive_calculation
            newer_values_pending = (self._interactive_preview_pending
                                    or (page._live_mode and page.timer.isActive()))
            if self._interactive_preview_in_flight() and newer_values_pending:
                # This is a completed intermediate frame, not the latest lens
                # setting. Never write its snapshot back into the live controls.
                self.workspace.heading.setText(self.workspace.heading.text() + " | Updating")
                self.workspace.interactive_calculation.live_status.setText(
                    "Live tuning | last completed frame; updating to latest settings")
                self.status_label.setText(
                    "Live tuning: completed ray frame displayed; newer settings pending")
                return
        self.workspace.model_inspector.display_result(result)
        self._schedule_design_explorer_refresh()
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
        if bool(getattr(result, "cache_hit", False)):
            cache_status = " | complete cache hit"
            cache_log = ", complete cache hit"
        else:
            reused = sorted(getattr(result, "reused_products", ()))
            calculated = sorted(getattr(result, "calculated_products", ()))
            cache_status = (
                f" | reused {len(reused)}, calculated {len(calculated)}"
                if reused
                else ""
            )
            cache_log = (
                f", reused={','.join(reused)}, calculated={','.join(calculated)}"
                if reused
                else ""
            )
        segment_cache = metrics.get("column_segment_cache", {})
        if segment_cache.get("mode") == "checkpoint":
            resume_z = float(segment_cache.get("resume_z_mm", 0.0))
            cache_status += f" | column resumed at {resume_z:.1f} mm"
            cache_log += f", column resume z={resume_z:.6g} mm"
        elif segment_cache.get("mode") == "full_incident":
            cache_status += " | incident beam reused"
            cache_log += ", incident beam reused"
        self.status_label.setText(
            f"{quality} completed in {duration:.3f} s | "
            f"mode: {mode} | {MODE_BY_KEY[mode_key(result.state_snapshot)].label} | rays: {backend}{wave_status}{cache_status}"
        )
        self.log_output.appendPlainText(
            f"{quality}: {duration:.3f} s, "
            f"{result.simulation.incident.x.shape[1]} rays, mode={mode}, "
            f"ray backend={backend}{wave_log}{cache_log}."
        )

    def _calculation_failed(self, quality: str, message: str) -> None:
        self._show_error(f"{quality} calculation failed: {message}")
        self._schedule_design_explorer_refresh()

    def _calculation_finished(self, _quality: str) -> None:
        self._set_progress_active("calculation", False)
        self._schedule_design_explorer_refresh()
        self._interactive_preview_generation = None
        if self._interactive_preview_pending:
            self.preview_timer.start(0)

    def _mark_operating_preset_stale(self) -> None:
        message = (
            "Geometry changed. Lens strengths kept; apply the preset after "
            "assembly edits are complete."
        )
        self.assembly_panel.set_operating_mode_status(message)
        self.assembly_panel.operating_mode_status.setToolTip(message)

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
                    candidate_state,
                    self.selection,
                    preserve_operating_parameters=True,
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
            self._mark_operating_preset_stale()
            self.status_label.setText(
                f"Saved {Path(target.module_path).name}; lens strengths retained"
            )
            self.log_output.appendPlainText(
                f"TOML saved: {target.module_path}; geometry reloaded and "
                "lens strengths retained. Apply one preset after assembly "
                "editing is complete."
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
        if self.design_sweeps.running:
            self.workspace.design_explorer.set_sweep_error(
                "Cancel the design sweep before reloading TOML"
            )
            return
        self._invalidate_direct_alignment()
        self.workspace.model_inspector.refresh()
        try:
            catalog = AssemblyCatalog()
            audit = self.manifest_editor.validate_catalog()
            selection = catalog.normalise_selection(self.selection)
            candidate_state = type(self.state).from_dict(self.state.to_dict())
            assembly = catalog.apply(
                candidate_state,
                selection,
                preserve_operating_parameters=True,
            )
            self.catalog = catalog
            self.design_sweeps.set_catalog(catalog)
            self.selection = selection
            self.state = candidate_state
            self.assembly = assembly
            self.assembly_panel.reload_catalog(catalog, selection)
            self._refresh_assembly_views()
            self._mark_operating_preset_stale()
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
        self._live_tuning_layout_initialized = settings.value(self.SETTINGS_LIVE_TUNING_LAYOUT, False, type=bool)
        geometry = settings.value(self.SETTINGS_GEOMETRY, QByteArray())
        state = settings.value(self.SETTINGS_STATE, QByteArray())
        if isinstance(geometry, QByteArray) and not geometry.isEmpty():
            self.restoreGeometry(geometry)
        if isinstance(state, QByteArray) and not state.isEmpty():
            self.restoreState(state)

    def reset_workspace(self) -> None:
        if hasattr(self, "workspace_layouts"):
            self.workspace_layouts.reset_current()
            return
        for dock, area in (
            (self.instrument_dock, Qt.DockWidgetArea.LeftDockWidgetArea),
            (self.live_tuning_dock, Qt.DockWidgetArea.LeftDockWidgetArea),
            (self.log_dock, Qt.DockWidgetArea.BottomDockWidgetArea),
        ):
            dock.setFloating(False)
            self.addDockWidget(area, dock)
            dock.show()
        self.tabifyDockWidget(self.instrument_dock, self.live_tuning_dock)
        self.live_tuning_dock.hide()
        self._live_tuning_layout_initialized = False
        self.instrument_dock.raise_()
        self.instrument_editor.setSizes([430, 430])
        self.resize(1500, 920)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self.workspace_layouts.close()
        self.workspace.interactive_calculation.shutdown()
        self.workspace.model_inspector.validation_page.shutdown()
        settings = QSettings()
        settings.setValue(self.SETTINGS_GEOMETRY, self.saveGeometry())
        settings.setValue(self.SETTINGS_STATE, self.saveState())
        settings.setValue(self.SETTINGS_LIVE_TUNING_LAYOUT, self._live_tuning_layout_initialized)
        self.calculations.invalidate_pending()
        self.calculations.pool.waitForDone(3_000)
        self.design_sweeps.invalidate_pending()
        self.design_sweeps.pool.waitForDone(3_000)
        self.operating_presets.invalidate_pending()
        self.operating_presets.pool.waitForDone(3_000)
        super().closeEvent(event)
