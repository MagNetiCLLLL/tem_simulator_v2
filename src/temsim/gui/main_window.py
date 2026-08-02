"""Main TEM Simulator v2 desktop window."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QByteArray, QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
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
from temsim.gui.parameter_panel import ParameterPanel
from temsim.gui.visualization import VisualizationWorkspace
from temsim.manifest_editor import ManifestEditor, ManifestTarget
from temsim.optics.column import default_state
from temsim.profile_io import apply_profile_values, read_profile, save_profile
from temsim.runtime_parameters import runtime_targets


class MainWindow(QMainWindow):
    SETTINGS_GEOMETRY = "main_window/geometry"
    SETTINGS_STATE = "main_window/state"
    PREVIEW_RAYS = 49
    PREVIEW_STEP_MM = 2.5

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
            "Instrument", "instrumentDock", self.assembly_panel,
            Qt.DockWidgetArea.LeftDockWidgetArea,
        )
        self.parameter_dock = self._create_dock(
            "Parameters", "parameterDock", self.parameter_panel,
            Qt.DockWidgetArea.RightDockWidgetArea,
        )
        self.log_dock = self._create_dock(
            "Status and calculation log", "logDock", self.log_output,
            Qt.DockWidgetArea.BottomDockWidgetArea,
        )

        self.calculations = CalculationController(self)
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(250)

        self.assembly_panel.selection_requested.connect(self.load_assembly)
        self.assembly_panel.tree.component_selected.connect(
            self._select_tree_item
        )
        self.workspace.component_selected.connect(
            self.assembly_panel.tree.select_key
        )
        self.parameter_panel.runtime_changed.connect(self.schedule_preview)
        self.parameter_panel.manifest_save_requested.connect(
            self._save_manifest_updates
        )
        self.parameter_panel.error.connect(self._show_error)
        self.preview_timer.timeout.connect(self.run_preview)
        self.calculations.started.connect(self._calculation_started)
        self.calculations.result_ready.connect(self._calculation_ready)
        self.calculations.failed.connect(self._calculation_failed)
        self.calculations.finished.connect(self._calculation_finished)

        self._create_actions()
        self._create_toolbar()
        self._create_status_bar()
        self._refresh_assembly_views()
        self._restore_workspace()
        QTimer.singleShot(50, self.run_preview)

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
        view_menu.addAction(self.parameter_dock.toggleViewAction())
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

    def _refresh_assembly_views(self) -> None:
        self._runtime_targets = runtime_targets(self.state)
        anchors = self.manifest_editor.anchor_records(self.assembly)
        self._anchors_by_key = {record.part_key: record for record in anchors}
        self.assembly_panel.tree.load_assembly(
            self.assembly, self._runtime_targets
        )
        self.log_output.appendPlainText(
            f"Assembly validated: {len(self.assembly.parts)} parts, "
            f"{len(anchors)} confirmed anchors."
        )

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

    def load_assembly(self, selection) -> None:
        try:
            candidate_state = type(self.state).from_dict(self.state.to_dict())
            candidate_assembly = self.catalog.apply(candidate_state, selection)
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

    def schedule_preview(self, _parameter: str = "") -> None:
        self.preview_timer.start()

    def run_preview(self) -> None:
        self.calculations.submit(
            self.state,
            "Preview",
            self.PREVIEW_RAYS,
            self.PREVIEW_STEP_MM,
        )

    def run_high_accuracy(self) -> None:
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
        self.progress.show()
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
        mode = metrics.get("mode", "unknown")
        self.status_label.setText(
            f"{quality} completed in {duration:.3f} s | mode: {mode}"
        )
        self.log_output.appendPlainText(
            f"{quality}: {duration:.3f} s, "
            f"{result.simulation.incident.x.shape[1]} rays, mode={mode}."
        )

    def _calculation_failed(self, quality: str, message: str) -> None:
        self._show_error(f"{quality} calculation failed: {message}")

    def _calculation_finished(self, _quality: str) -> None:
        self.progress.hide()

    def _save_manifest_updates(self, target, updates) -> None:
        if not updates:
            self.status_label.setText("No TOML values changed")
            return
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
        try:
            catalog = AssemblyCatalog()
            audit = self.manifest_editor.validate_catalog()
            candidate_state = type(self.state).from_dict(self.state.to_dict())
            assembly = catalog.apply(candidate_state, self.selection)
            self.catalog = catalog
            self.state = candidate_state
            self.assembly = assembly
            self.assembly_panel.reload_catalog(catalog, self.selection)
            self._refresh_assembly_views()
            self.status_label.setText(
                f"TOML catalog valid: {audit.part_definition_count} part "
                f"definitions and {audit.assembly_count} assemblies"
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
            (self.parameter_dock, Qt.DockWidgetArea.RightDockWidgetArea),
            (self.log_dock, Qt.DockWidgetArea.BottomDockWidgetArea),
        ):
            dock.setFloating(False)
            self.addDockWidget(area, dock)
            dock.show()
        self.resize(1500, 920)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        settings = QSettings()
        settings.setValue(self.SETTINGS_GEOMETRY, self.saveGeometry())
        settings.setValue(self.SETTINGS_STATE, self.saveState())
        self.calculations.pool.clear()
        self.calculations.pool.waitForDone(3_000)
        super().closeEvent(event)
