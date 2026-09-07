"""Physical Layout's file-backed three-dimensional dimension workspace."""

from pathlib import Path
from copy import deepcopy

from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QBrush, QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QPushButton,
    QSizePolicy, QSplitter, QTabWidget, QTableWidget, QTableWidgetItem,
    QStyledItemDelegate, QTextBrowser, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from temsim.gui.input_policy import WheelSafeComboBox as QComboBox
from temsim.manifest_editor import format_toml_value
from temsim.part_model_document import PartModelDocument


class _DimensionDelegate(QStyledItemDelegate):
    def sizeHint(self, option, index):
        # Leave room around wrapped semantic labels and multi-line CAD fields.
        return super().sizeHint(option, index) + QSize(0, 6)


class PartModelEditorPage(QWidget):
    component_selected = Signal(str)
    saved = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        from temsim.gui.part_model_view import PartModelView

        self.setObjectName("partModelEditorPage")
        self.session = None
        self._project_root = None
        self._project_paths = ()
        self._project_save = None
        self._pending_part = None
        self._pending_reveal = False
        self._selected_key = None
        self._selected_region = "body"
        self._aperture_index = 0
        self._loading = False
        self._saving = False
        self._mesh_records = ()
        self._runtime_values = {}
        self._runtime_refresh_pending = False
        self._runtime_parameters_pending = False
        self._invalid_inputs = {}
        self._topology_selection = ()
        self._simulation_mode = None
        self._field_descriptors = {}
        self._assembly_parts = {}
        self._calculation_status = "not_calculated"
        self._calculation_detail = "No completed calculation in this window."
        self._audit_dialog = None
        self.open_button = QPushButton("Open TOML…")
        self.save_button = QPushButton("Save")
        self.save_copy_button = QPushButton("Save copy…")
        self.undo_button = QPushButton("Undo")
        self.redo_button = QPushButton("Redo")
        self.revert_button = QPushButton("Revert")
        self.audit_button = QPushButton("Dimension audit…")
        self.audit_button.setToolTip("Review dimension meanings, evidence and missing definitions across saved modules")
        self.source_label = QLabel("Open an instrument module or select a component in Physical Layout")
        self.source_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.modules = QComboBox()
        self.modules.setPlaceholderText("Active instrument modules")
        self.modules.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.modules.setMinimumContentsLength(12)
        self.modules.setToolTip("Source module for the current component. Components in one module share its TOML file.")
        self.load_module_button = QPushButton("Open active module")
        self.load_module_button.setEnabled(False)
        self.scope = QComboBox()
        for label, key in (("Selected part", "part"), ("Part + neighbours", "context"), ("Entire module", "module")):
            self.scope.addItem(label, key)
        self.scope.setCurrentIndex(0)
        self.fit_button = QPushButton("Fit")
        self.iso_button = QPushButton("Isometric")
        self.front_button = QPushButton("Axial view")
        self.section = QCheckBox("Section")
        self.aperture = QComboBox()
        self.aperture.setToolTip("Preview one of the existing apertures; no new holes are created")
        self.aperture.hide()
        self.view = PartModelView()
        self.tree = QTreeWidget()
        self.tree.setObjectName("partModelTree")
        self.tree.setHeaderLabels(("Components / regions",))
        self.tree.setMinimumWidth(0)
        self.tree.setIndentation(14)
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        self.parameter_tabs = QTabWidget()
        self.parameter_tabs.setObjectName("partModelParameterTabs")
        self.dimensions = QTableWidget(0, 4)
        self.dimensions.setObjectName("partModelDimensions")
        self.dimensions.setItemDelegate(_DimensionDelegate(self.dimensions))
        self.dimensions.setHorizontalHeaderLabels(("Dimension", "Value", "Unit", "Use"))
        self.dimensions.horizontalHeaderItem(3).setToolTip(
            "Participation in the active simulation model. Inactive dimensions still update the mechanical display; select a row for the exact effects.")
        self.dimensions.verticalHeader().hide()
        self.dimensions.verticalHeader().setMinimumSectionSize(30)
        self.dimensions.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.dimensions.setWordWrap(True)
        self.dimensions.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.dimensions.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.dimensions.horizontalHeader().sectionResized.connect(self.dimensions.resizeRowsToContents)
        self.dimensions.setColumnWidth(1, 88)
        self.dimensions.setColumnWidth(2, 32)
        self.dimensions.setColumnWidth(3, 65)
        self.dimensions.setAlternatingRowColors(True)
        self.parameter_tabs.addTab(self.dimensions, "Dimensions")

        materials_page = QWidget()
        material_layout = QVBoxLayout(materials_page)
        self.region = QComboBox()
        self.material = QComboBox()
        self.material.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.material.setMinimumContentsLength(12)
        from temsim.part_materials import material_catalog
        self._materials = material_catalog()
        for item in self._materials:
            self.material.addItem(item["label"], item["material_key"])
        self.material_hint = QLabel()
        self.material_hint.setWordWrap(True)
        self.material_current = QLabel()
        self.material_current.setWordWrap(True)
        self.assign_button = QPushButton("Assign to this region")
        for widget in (QLabel("Existing material region"), self.region, self.material_current,
                       QLabel("Material"), self.material, self.material_hint, self.assign_button):
            material_layout.addWidget(widget)
        material_layout.addStretch(1)
        self.parameter_tabs.addTab(materials_page, "Materials")
        self.parameters = QTreeWidget()
        self.parameters.setObjectName("partModelAllParameters")
        self.parameters.setHeaderLabels(("Related parameter", "Value"))
        self.parameters.setColumnWidth(0, 190)
        self.parameter_tabs.addTab(self.parameters, "All parameters")
        self._create_features_page()
        self.selection_label = QLabel("No component selected")
        self.selection_label.setWordWrap(True)
        self.calculation_label = QLabel("Simulation context not connected")
        self.calculation_label.setWordWrap(True)
        self.calculation_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.parameter_detail = QTextBrowser()
        self.parameter_detail.setObjectName("partModelParameterDetail")
        self.parameter_detail.setMinimumHeight(95)
        self.parameter_detail.setMaximumHeight(130)
        self.parameter_detail.setPlainText("Select a parameter to inspect its meaning, evidence and use in the active model.")
        self.model_note = QLabel()
        self.model_note.setWordWrap(True)
        self.model_note.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        side = QWidget()
        side.setMinimumWidth(0)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.addWidget(self.selection_label)
        side_layout.addWidget(self.calculation_label)
        side_layout.addWidget(self.parameter_tabs, 1)
        side_layout.addWidget(self.parameter_detail)
        side_layout.addWidget(self.model_note)
        canvas = QWidget()
        canvas_layout = QVBoxLayout(canvas)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.addWidget(self.view, 1)
        hint = QLabel("Drag: rotate freely · Right drag: pan · Wheel: zoom · Click: select a surface")
        hint.setWordWrap(True)
        canvas_layout.addWidget(hint)
        picking_row = QHBoxLayout()
        self.selection_mode = QComboBox()
        for label, mode in (("Select faces", "face"), ("Select edges", "edge"), ("Select components", "part")):
            self.selection_mode.addItem(label, mode)
        self.clear_topology_button = QPushButton("Clear selection")
        self.clear_topology_button.setToolTip("Clear selected faces and edges")
        self.topology_hint = QLabel("Select a face or edge to highlight its parameters. Ctrl-click adds to the selection.")
        self.topology_hint.setWordWrap(True)
        picking_row.addWidget(self.selection_mode)
        picking_row.addWidget(self.clear_topology_button)
        picking_row.addStretch(1)
        canvas_layout.addLayout(picking_row)
        canvas_layout.addWidget(self.topology_hint)
        self.splitter = QSplitter()
        self.splitter.setObjectName("partModelSplitter")
        self.splitter.addWidget(self.tree)
        self.splitter.addWidget(canvas)
        self.splitter.addWidget(side)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes([190, 660, 350])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        file_row = QHBoxLayout()
        for widget in (self.open_button, self.save_button, self.save_copy_button, self.undo_button,
                       self.redo_button, self.revert_button):
            file_row.addWidget(widget)
        file_row.addWidget(self.source_label, 1)
        file_row.addWidget(self.audit_button)
        layout.addLayout(file_row)
        # Source selection and camera controls must not form one nonwrapping
        # row that forces every central workspace tab wider than the window.
        module_row = QHBoxLayout()
        module_row.addWidget(self.modules, 1)
        module_row.addWidget(self.load_module_button)
        layout.addLayout(module_row)
        view_row = QHBoxLayout()
        for widget in (self.scope, self.fit_button,
                       self.iso_button, self.front_button, self.section):
            view_row.addWidget(widget)
        view_row.addWidget(self.aperture)
        layout.addLayout(view_row)
        layout.addWidget(self.splitter, 1)
        self.status = QLabel("Dimensions stay in the draft until Save. The axial centre stays fixed.")
        self.status.setWordWrap(True)
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout.addWidget(self.status)
        self.open_button.clicked.connect(self.open_file)
        self.save_button.clicked.connect(self.save)
        self.save_copy_button.clicked.connect(self.save_copy)
        self.undo_button.clicked.connect(self.undo)
        self.redo_button.clicked.connect(self.redo)
        self.revert_button.clicked.connect(self.revert)
        self.audit_button.clicked.connect(self.show_dimension_audit)
        self.load_module_button.clicked.connect(self._open_active_module)
        self.modules.activated.connect(self._open_active_module)
        self.scope.currentIndexChanged.connect(lambda: self._render())
        self.fit_button.clicked.connect(self.view.fit_all)
        self.iso_button.clicked.connect(self.view.set_isometric_view)
        self.front_button.clicked.connect(self.view.set_axial_view)
        self.section.toggled.connect(self.view.set_section_enabled)
        self.tree.currentItemChanged.connect(self._tree_selected)
        self.view.selection_changed.connect(self._surface_selected)
        self.view.topology_selection_changed.connect(self._topology_changed)
        self.selection_mode.currentIndexChanged.connect(
            lambda: self.view.set_selection_mode(self.selection_mode.currentData()))
        self.clear_topology_button.clicked.connect(self.view.clear_topology_selection)
        self.view.set_selection_mode(self.selection_mode.currentData())
        self.dimensions.itemChanged.connect(self._dimension_changed)
        self.dimensions.currentCellChanged.connect(self._parameter_selected)
        self.parameters.currentItemChanged.connect(self._all_parameter_selected)
        self.material.currentIndexChanged.connect(self._material_changed)
        self.region.currentIndexChanged.connect(self._region_changed)
        self.aperture.currentIndexChanged.connect(self._aperture_changed)
        self.assign_button.clicked.connect(self.assign_material)
        for sequence, slot in ((QKeySequence.StandardKey.Save, self.save),
                               (QKeySequence.StandardKey.Undo, self.undo),
                               (QKeySequence.StandardKey.Redo, self.redo)):
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(slot)
        self._material_changed()
        self._update_buttons()

    def set_project_context(self, root, assembly, save_callback, runtime_values=None):
        previous_mesh = self._runtime_mesh_dependencies()
        previous_parameters = self._selected_runtime_values()
        self._project_root = Path(root).resolve()
        self._project_save = save_callback
        self._runtime_values = deepcopy(runtime_values or {})
        paths = tuple(str(path) for _kind, path in assembly.selected_module_paths)
        self._project_paths = paths
        self.load_module_button.setEnabled(bool(paths))
        selected = self.modules.currentData()
        blocked = self.modules.blockSignals(True)
        self.modules.clear()
        for path in paths:
            self.modules.addItem(Path(path).stem, path)
        self.modules.setCurrentIndex(max(0, self.modules.findData(selected)))
        self.modules.blockSignals(blocked)
        source_reloaded = False
        # Another editor or catalog reload may have accepted newer source
        # dimensions. Refresh a clean document, retaining unresolved drafts.
        if (self.session is not None and not self._saving
                and not self.session.dirty and not self._invalid_inputs):
            try:
                self.session.assert_source_current()
            except (OSError, ValueError):
                try:
                    self.session.reload()
                    keys = {part["key"] for part in self.session.document["parts"]}
                    if self._selected_key not in keys:
                        self._selected_key = self.session.document["parts"][0]["key"]
                        self._selected_region = "body"
                    self._topology_selection = ()
                    self._load_tree()
                    self.select_part(self._selected_key, region=self._selected_region, emit=False)
                    source_reloaded = True
                    self._message("Source reloaded from saved dimensions; view and component selection retained.")
                    self._update_buttons()
                except Exception as exc:
                    self._message(f"Source could not be reloaded: {exc}", error=True)
        if self.session is not None and not source_reloaded:
            # A module can become inactive while its document stays open, or a
            # newly loaded profile can change the working opening. Neither
            # transition may leave the old instrument's aperture in the mesh.
            self._refresh_runtime_dependencies(previous_mesh, previous_parameters)
        self._sync_source_context()

    def _model_runtime_values(self):
        if self.session is None or self._project_root is None:
            return {}
        active_paths = {(self._project_root / path).resolve() for path in self._project_paths}
        return self._runtime_values if self.session.path in active_paths else {}

    def set_runtime_values(self, values):
        """Refresh operating apertures without replacing the mechanical draft."""
        previous_mesh = self._runtime_mesh_dependencies()
        previous_parameters = self._selected_runtime_values()
        self._runtime_values = deepcopy(values or {})
        self._refresh_runtime_dependencies(previous_mesh, previous_parameters)

    def _selected_runtime_values(self):
        if self.session is None or self._selected_key is None:
            return {}
        values = self._model_runtime_values()
        key = self._selected_key
        if key not in values:
            key = self.session.part(key).get("parent_key")
        return values.get(key, {})

    def _runtime_mesh_dependencies(self):
        """Only strip openings consume operating values in the CAD renderer."""
        if self.session is None or self._selected_key is None:
            return ()
        from temsim.part_model_apertures import _opening, is_strip_aperture

        parts = {part["key"]: part for part in self.session.document["parts"]}
        scope = self.scope.currentData()
        if scope == "module":
            keys = set(parts)
        else:
            keys = {self._selected_key}
            # Match part_model_from_document's descendant/shared-body closure.
            while True:
                more = {key for key, part in parts.items()
                        if part.get("parent_key") in keys
                        or keys.intersection(part.get("magnetic_lens_keys", ()))}
                if more <= keys:
                    break
                keys.update(more)
            if scope == "context":
                parent = parts[self._selected_key].get("parent_key")
                keys.update(key for key, part in parts.items()
                            if parent and part.get("parent_key") == parent)
        runtime = self._model_runtime_values()
        dependencies = []
        for key, part in parts.items():
            if (key not in keys or not is_strip_aperture(part)
                    or part.get("model_3d", {}).get("base", {}).get("kind", "existing") != "existing"):
                continue
            try:
                index = self._aperture_index if key == self._selected_key and scope != "module" else 0
                opening = _opening(part, runtime.get(key), index)
            except ValueError as exc:
                # Invalid input is handled by the normal preview error path.
                opening = str(exc)
            dependencies.append((key, opening))
        return tuple(dependencies)

    def _refresh_runtime_dependencies(self, previous_mesh, previous_parameters):
        self._runtime_refresh_pending |= previous_mesh != self._runtime_mesh_dependencies()
        self._runtime_parameters_pending |= previous_parameters != self._selected_runtime_values()
        if self.isVisible():
            self._flush_runtime_refresh()

    def _flush_runtime_refresh(self):
        if self.session is None or self._selected_key is None:
            return
        mesh_pending = self._runtime_refresh_pending
        parameters_pending = self._runtime_parameters_pending
        self._runtime_refresh_pending = self._runtime_parameters_pending = False
        if mesh_pending:
            self._load_parameters()
            self._render(preserve_view=True)
        elif parameters_pending:
            # Operating annotations can change without changing any CAD mesh.
            self._load_all_parameters()

    def set_simulation_context(self, mode=None, descriptors=None, by_key=None):
        self._simulation_mode = mode
        self._field_descriptors = descriptors or {}
        self._assembly_parts = by_key or {}
        if self.session is not None:
            self._refresh_parameter_annotations()
        self._refresh_calculation_status()

    def set_calculation_status(self, status, detail=""):
        if status not in {"not_calculated", "stale", "running", "current", "failed"}:
            raise ValueError(f"Unknown calculation status: {status}")
        self._calculation_status, self._calculation_detail = status, detail
        self._refresh_calculation_status()

    def _meaning_context(self):
        active = (self.session is not None and self._project_root is not None
                  and self.session.path in {(self._project_root / path).resolve() for path in self._project_paths})
        parts = dict(self._assembly_parts) if active else {}
        if self.session is not None:
            parts.update({part["key"]: part for part in self.session.document["parts"]})
        return parts, self._simulation_mode if active else None, self._field_descriptors if active else {}

    def _parameter_information(self, path, field=None):
        from temsim.parameter_semantics import describe_parameter
        from temsim.parameter_impact import describe_parameter_impact
        parts, mode, descriptors = self._meaning_context()
        part = (parts.get(path[1], {}) if len(path) > 1 and path[0] in {"parts", "runtime", "derived"}
                else self.session.document if self.session is not None else {})
        meaning = describe_parameter(part, tuple(path), by_key=parts)
        impact = describe_parameter_impact(part, tuple(path), by_key=parts,
                                           simulation_mode=mode, descriptors=descriptors)
        label = field.label if field is not None else meaning.label
        reason = field.reason if field is not None and field.reason else meaning.description
        text = (f"{label} · {meaning.category_label}\n"
                f"Evidence: {meaning.source_label}\nUse: {impact.label}\n\n"
                f"{reason}\nSource detail: {meaning.source_note}\nCalculation detail: {impact.detail}")
        return meaning, impact, text

    def _refresh_parameter_annotations(self):
        if self.session is None:
            return
        blocked = self.dimensions.blockSignals(True)
        try:
            for row in range(self.dimensions.rowCount()):
                value = self.dimensions.item(row, 1)
                if value is None:
                    continue
                field = value.data(Qt.ItemDataRole.UserRole + 2)
                path = value.data(Qt.ItemDataRole.UserRole)
                meaning, impact, text = self._parameter_information(path, field)
                short = {"active": "Active", "inactive": "Inactive", "unsupported": "Unsupported",
                         "configuration_required": "Setup", "unknown": "Check"}.get(impact.status, "Check")
                if meaning.category == "cad":
                    short = "CAD only"
                # A short status remains visible; the full evidence and routing
                # are available for every row and in the selection inspector.
                use = QTableWidgetItem(short)
                use.setFlags(use.flags() & ~Qt.ItemFlag.ItemIsEditable)
                use.setForeground(QColor({"active": "#7fcea5", "inactive": "#d1b36d",
                                          "unsupported": "#a3bdd8"}.get(impact.status, "#bdc6d1")))
                self.dimensions.setItem(row, 3, use)
                for column in range(self.dimensions.columnCount()):
                    item = self.dimensions.item(row, column)
                    if item is not None:
                        item.setToolTip(text)
                value.setData(Qt.ItemDataRole.UserRole + 3, meaning)
                use.setData(Qt.ItemDataRole.UserRole, impact)
        finally:
            self.dimensions.blockSignals(blocked)
        if self.dimensions.currentRow() < 0 and self.dimensions.rowCount():
            self.dimensions.setCurrentCell(0, 0)
        if self.parameter_tabs.currentWidget() is self.parameters and self.parameters.currentItem() is not None:
            self._all_parameter_selected(self.parameters.currentItem())
        else:
            self._parameter_selected(self.dimensions.currentRow())

    def _parameter_selected(self, row, *_):
        if self._loading:
            return
        item = self.dimensions.item(row, 1)
        if item is not None:
            _, _, text = self._parameter_information(item.data(Qt.ItemDataRole.UserRole),
                                                      item.data(Qt.ItemDataRole.UserRole + 2))
            self.parameter_detail.setPlainText(text)

    def _all_parameter_selected(self, item, _previous=None):
        if self._loading or item is None or self.session is None:
            return
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if path and len(path) >= 2:
            self.parameter_detail.setPlainText(self._parameter_information(path)[2])

    def _refresh_calculation_status(self):
        if not hasattr(self, "calculation_label"):
            return
        parts, mode, descriptors = self._meaning_context()
        from temsim.simulation_modes import MODE_BY_KEY
        mode_label = MODE_BY_KEY[mode].label if mode in MODE_BY_KEY else "Not linked to active simulation"
        status = {"not_calculated": "Not calculated", "stale": "Results out of date",
                  "running": "Calculating", "current": "Result current for active model",
                  "failed": "Calculation failed"}[self._calculation_status]
        lines = [mode_label]
        details = [self._calculation_detail]
        source_changed = False
        if self.session is not None and not self._saving:
            try:
                self.session.assert_source_current()
            except (OSError, ValueError) as exc:
                source_changed = True
                lines.append("Source changed or unavailable · displayed dimensions need reload")
                details.append(str(exc) + "\nResolve any draft, then reopen the source or reload the catalog.")
        if self.session is not None and self._selected_key in parts:
            from temsim.parameter_impact import component_impact_summary, describe_parameter_impact
            summary = component_impact_summary(parts[self._selected_key], by_key=parts,
                                               simulation_mode=mode, descriptors=descriptors)
            details.append(summary.detail)
            if summary.status == "configuration_required":
                lines.append("Field setup required · inspect parameter usage")
            ignored = summary.ignored_cad_parameters
            if ignored:
                lines.append(f"CAD changes excluded from physics ({len(ignored)} parameters)")
                details.append("Excluded from beam / field calculation:\n" + "\n".join(ignored))
            if self.session.dirty or self._invalid_inputs:
                lines.append("Unsaved draft · simulation still uses saved dimensions")
                affected = set()
                for path in self.session.updates():
                    impact = describe_parameter_impact(parts[path[1]], path, by_key=parts,
                                                       simulation_mode=mode, descriptors=descriptors)
                    affected.update(impact.affected_results)
                if affected:
                    details.append("After Save, review: " + ", ".join(sorted(affected)))
            elif mode is not None and not source_changed:
                lines.append(status)
        self.calculation_label.setText("\n".join(lines))
        self.calculation_label.setToolTip("\n\n".join(text for text in details if text))

    def show_dimension_audit(self):
        from temsim.dimension_audit import audit_catalog, audit_document
        from temsim.gui.dimension_audit import DimensionAuditDialog
        try:
            if self._project_root is not None:
                audit = audit_catalog(self._project_root)
                scope = "Saved catalog dimensions; unsaved drafts are excluded"
            elif self.session is not None:
                audit = audit_document(self.session.document, source=str(self.session.path))
                scope = "Current file draft"
            else:
                self._message("Open a module to audit its dimensions.")
                return
            if self._audit_dialog is not None:
                self._audit_dialog.close()
                self._audit_dialog.deleteLater()
            dialog = DimensionAuditDialog(audit, self, scope=scope)
            dialog.parameter_requested.connect(self._audit_parameter_requested)
            self._audit_dialog = dialog
            dialog.show()
            return dialog
        except Exception as exc:
            self._message(f"Dimension audit failed: {exc}", error=True)

    def _audit_parameter_requested(self, source, key, path):
        location = Path(source)
        if not location.is_absolute() and self._project_root is not None:
            location = self._project_root / location
        if self.open_path(location, selected_key=key):
            path = tuple(path)
            self._highlight_parameter_paths({path})
            located = False
            for row in range(self.dimensions.rowCount()):
                if tuple(self.dimensions.item(row, 1).data(Qt.ItemDataRole.UserRole)) == path:
                    self.parameter_tabs.setCurrentWidget(self.dimensions)
                    self.dimensions.setCurrentCell(row, 0)
                    located = True
                    break
            if not located:
                self.parameter_tabs.setCurrentWidget(self.parameters)
                def find(parent):
                    if tuple(parent.data(0, Qt.ItemDataRole.UserRole) or ()) == path:
                        return parent
                    for index in range(parent.childCount()):
                        match = find(parent.child(index))
                        if match is not None:
                            return match
                    return None
                item = None
                for index in range(self.parameters.topLevelItemCount()):
                    item = find(self.parameters.topLevelItem(index))
                    if item is not None:
                        break
                if item is not None:
                    self.parameters.setCurrentItem(item)
                    self.parameters.scrollToItem(item)
                    located = True
                else:
                    self.parameters.clearSelection()
                    message = ("Audit field " + ".".join(map(str, path))
                               + " is not defined in this source file. The component is open for review; no dimension was substituted.")
                    self._message(message, error=True)
                    self.parameter_detail.setPlainText(message)
            if self._audit_dialog is not None:
                self._audit_dialog.accept()
            if key:
                self._emit_project_selection()

    def _create_features_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)
        self.base_shape = QComboBox()
        self.base_shape.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.base_shape.setMinimumContentsLength(12)
        for label, key in (("Existing component geometry", "existing"),
                           ("Box", "box"), ("Elliptic cylinder", "elliptic_cylinder")):
            self.base_shape.addItem(label, key)
        self.apply_shape_button = QPushButton("Use this base shape")
        self.reset_model_button = QPushButton("Restore original shape")
        layout.addWidget(QLabel("Base shape"))
        layout.addWidget(self.base_shape)
        layout.addWidget(self.apply_shape_button)
        layout.addWidget(self.reset_model_button)
        self.feature_tree = QTreeWidget()
        self.feature_tree.setObjectName("partModelFeatures")
        self.feature_tree.setHeaderLabels(("Cut feature", "Type"))
        self.feature_tree.setRootIsDecorated(False)
        self.feature_tree.setMinimumWidth(0)
        self.feature_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.feature_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.feature_tree, 1)
        row = QHBoxLayout()
        self.add_hole_button = QPushButton("Add hole…")
        self.add_slot_button = QPushButton("Add slot / groove…")
        row.addWidget(self.add_hole_button)
        row.addWidget(self.add_slot_button)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.edit_feature_button = QPushButton("Edit feature…")
        self.remove_feature_button = QPushButton("Remove feature")
        row.addWidget(self.edit_feature_button)
        row.addWidget(self.remove_feature_button)
        layout.addLayout(row)
        note = QLabel("Change model dimensions and X/Y scale, offset or rotation in Dimensions. "
                      "A selected face or edge supplies the initial cut position. "
                      "Holes and slots remove material; all changes support Undo and Save.\n\n"
                      "This is the mechanical design model. Ray and field calculations still use "
                      "the existing beam-passage and axisymmetric parameters.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.parameter_tabs.addTab(page, "Features")
        self.apply_shape_button.clicked.connect(self._apply_base_shape)
        self.reset_model_button.clicked.connect(self._reset_model_shape)
        self.add_hole_button.clicked.connect(lambda: self._edit_feature("hole"))
        self.add_slot_button.clicked.connect(lambda: self._edit_feature("slot"))
        self.edit_feature_button.clicked.connect(lambda: self._edit_feature())
        self.remove_feature_button.clicked.connect(self._remove_feature)
        self.feature_tree.itemSelectionChanged.connect(self._feature_selected)
        self.feature_tree.itemDoubleClicked.connect(lambda *_: self._edit_feature())

    def _load_features(self):
        part = self.session.part(self._selected_key)
        model = part.get("model_3d", {})
        self.base_shape.setCurrentIndex(max(0, self.base_shape.findData(model.get("base", {}).get("kind", "existing"))))
        current = self.feature_tree.currentItem()
        selected = current.data(0, Qt.ItemDataRole.UserRole) if current else None
        blocked = self.feature_tree.blockSignals(True)
        self.feature_tree.clear()
        for feature in model.get("features", ()):
            node = QTreeWidgetItem((feature["id"], feature["kind"]))
            node.setData(0, Qt.ItemDataRole.UserRole, feature["id"])
            self.feature_tree.addTopLevelItem(node)
            if feature["id"] == selected:
                self.feature_tree.setCurrentItem(node)
        self.feature_tree.blockSignals(blocked)
        self.reset_model_button.setEnabled(bool(model))
        self.edit_feature_button.setEnabled(self.feature_tree.currentItem() is not None)
        self.remove_feature_button.setEnabled(self.feature_tree.currentItem() is not None)

    def _apply_base_shape(self):
        if self.session is None or self._selected_key is None:
            return
        from copy import deepcopy
        from temsim.part_model_features import default_model_3d
        part = self.session.part(self._selected_key)
        model = deepcopy(part.get("model_3d") or default_model_3d(part))
        kind = self.base_shape.currentData()
        base = {"kind": kind}
        if kind != "existing":
            diameter = float(part.get("mechanical_outer_diameter_mm", 10.0))
            base.update(width_mm=max(diameter, 0.01), height_mm=max(diameter, 0.01),
                        length_mm=max(float(part.get("length_mm", 10.0)), 0.01))
        model["base"] = base
        try:
            self.session.set_model_3d(self._selected_key, model)
            prefix = ("parts", self._selected_key, "model_3d", "base")
            self._invalid_inputs = {path: text for path, text in self._invalid_inputs.items() if path[:4] != prefix}
            self._draft_changed()
            self.parameter_tabs.setCurrentWidget(self.dimensions)
        except Exception as exc:
            self._message(str(exc), error=True)

    def _reset_model_shape(self):
        if self.session is not None:
            self.session.set_model_3d(self._selected_key, None)
            prefix = ("parts", self._selected_key, "model_3d")
            self._invalid_inputs = {path: text for path, text in self._invalid_inputs.items() if path[:3] != prefix}
            self._topology_selection = ()
            self._draft_changed()

    def _feature_defaults(self, kind):
        part = self.session.part(self._selected_key)
        features = part.get("model_3d", {}).get("features", ())
        count = 1
        while f"{kind}_{count}" in {item["id"] for item in features}:
            count += 1
        diameter = max(float(part.get("mechanical_outer_diameter_mm", 10.0)), 0.01)
        length = max(float(part.get("length_mm", 10.0)), 0.01)
        inner = float(part.get("mechanical_inner_diameter_mm", part.get("mechanical_bore_diameter_mm", 0.0)))
        center, axis, depth = [(inner + diameter) * 0.25, 0.0, 0.0], "z", length * 1.2
        hit = next((item for item in reversed(self._topology_selection) if item.get("key") == self._selected_key), None)
        if hit and hit.get("point") is not None:
            from temsim.part_model_features import feature_placement
            center, axis, depth = feature_placement(part, hit["point"], hit.get("normal", [0, 0, 1]))
        feature = dict(id=f"{kind}_{count}", kind=kind, axis=axis,
                       center_mm=list(center), depth_mm=float(depth))
        size = max(0.01, min(diameter * 0.08, max(diameter - inner, 0.1) * 0.25))
        if kind == "hole":
            feature["diameter_mm"] = size
        else:
            feature.update(width_mm=size, length_mm=size * 3, rotation_deg=0.0)
        return feature

    def _edit_feature(self, kind=None):
        if self.session is None or not self._selected_key:
            return
        from temsim.gui.part_feature_dialog import PartFeatureDialog
        if kind:
            feature = self._feature_defaults(kind)
        else:
            current = self.feature_tree.currentItem()
            if current is None:
                return
            feature = next(item for item in self.session.part(self._selected_key).get("model_3d", {}).get("features", ())
                           if item["id"] == current.data(0, Qt.ItemDataRole.UserRole))
        dialog = PartFeatureDialog(feature, self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        try:
            self.session.upsert_model_feature(self._selected_key, dialog.feature())
            index = next(index for index, item in enumerate(self.session.part(self._selected_key)["model_3d"]["features"])
                         if item["id"] == feature["id"])
            prefix = ("parts", self._selected_key, "model_3d", "features", index)
            self._invalid_inputs = {path: text for path, text in self._invalid_inputs.items() if path[:5] != prefix}
            self._draft_changed()
            for index in range(self.feature_tree.topLevelItemCount()):
                node = self.feature_tree.topLevelItem(index)
                if node.data(0, Qt.ItemDataRole.UserRole) == feature["id"]:
                    self.feature_tree.setCurrentItem(node)
                    break
        except Exception as exc:
            self._message(str(exc), error=True)

    def _remove_feature(self):
        current = self.feature_tree.currentItem()
        if self.session is not None and current is not None:
            feature_id = current.data(0, Qt.ItemDataRole.UserRole)
            index = next(index for index, item in enumerate(self.session.part(self._selected_key)["model_3d"]["features"])
                         if item["id"] == feature_id)
            self.session.remove_model_feature(self._selected_key, feature_id)
            prefix = ("parts", self._selected_key, "model_3d", "features")
            invalid = {}
            for path, text in self._invalid_inputs.items():
                if path[:4] == prefix:
                    if path[4] == index:
                        continue
                    if path[4] > index:
                        path = (*prefix, path[4] - 1, *path[5:])
                invalid[path] = text
            self._invalid_inputs = invalid
            self._topology_selection = ()
            self._draft_changed()

    def _feature_selected(self):
        if self._loading or self.session is None:
            return
        node = self.feature_tree.currentItem()
        self.edit_feature_button.setEnabled(node is not None)
        self.remove_feature_button.setEnabled(node is not None)
        if node is None:
            return
        features = self.session.part(self._selected_key).get("model_3d", {}).get("features", ())
        index = next((index for index, item in enumerate(features) if item["id"] == node.data(0, Qt.ItemDataRole.UserRole)), None)
        if index is not None:
            self._highlight_parameter_paths({("parts", self._selected_key, "model_3d", "features", index)})

    def _topology_changed(self, selection, *, activate_parameters=True):
        self._topology_selection = tuple(selection or ())
        labels = [item.get("label", str(item.get("id", ""))) for item in self._topology_selection]
        self.topology_hint.setText("Selected: " + "; ".join(labels) if labels else
                                   "Select a face or edge to highlight its parameters. Ctrl-click adds to the selection.")
        paths = {tuple(path) for item in self._topology_selection for path in item.get("parameter_paths", ())}
        self._highlight_parameter_paths(paths)
        if paths and activate_parameters:
            self.parameter_tabs.setCurrentWidget(self.dimensions)

    def _highlight_parameter_paths(self, paths):
        paths = {tuple(path) for path in paths}
        def matches(path):
            return path is not None and any(tuple(path[:len(prefix)]) == prefix for prefix in paths)
        def priority(path):
            if len(path) > 3 and path[2:4] == ["model_3d", "features"]:
                return 0
            if len(path) > 3 and path[2:4] == ["model_3d", "transform"]:
                return 2
            return 1
        first = None
        first_priority = 3
        blocked = self.dimensions.blockSignals(True)
        for row in range(self.dimensions.rowCount()):
            path = self.dimensions.item(row, 1).data(Qt.ItemDataRole.UserRole)
            match = matches(path)
            for column in range(self.dimensions.columnCount()):
                item = self.dimensions.item(row, column)
                item.setBackground(QBrush(QColor("#665125")) if match else QBrush())
                item.setData(Qt.ItemDataRole.UserRole + 1, bool(match))
            if match and priority(list(path)) < first_priority:
                first = self.dimensions.item(row, 0)
                first_priority = priority(list(path))
        self.dimensions.blockSignals(blocked)
        if first is not None:
            self.dimensions.scrollToItem(first, QTableWidget.ScrollHint.PositionAtTop)
            if paths:
                self.dimensions.setCurrentCell(first.row(), 0)
        tree_first = [None, 3]
        def visit(node):
            path = node.data(0, Qt.ItemDataRole.UserRole)
            match = matches(path)
            for column in range(2):
                node.setBackground(column, QBrush(QColor("#665125")) if match else QBrush())
            if match:
                if node.childCount() == 0 and priority(list(path)) < tree_first[1]:
                    tree_first[:] = [node, priority(list(path))]
                parent = node.parent()
                while parent is not None:
                    parent.setExpanded(True)
                    parent = parent.parent()
            for index in range(node.childCount()):
                visit(node.child(index))
        for index in range(self.parameters.topLevelItemCount()):
            visit(self.parameters.topLevelItem(index))
        if tree_first[0] is not None:
            self.parameters.scrollToItem(tree_first[0], QTreeWidget.ScrollHint.PositionAtTop)

    def focus_project_part(self, part):
        if self._project_root is None:
            return
        path = self._project_root / part.source_file
        if self.session is not None and self.session.path == path.resolve() and self._selected_key == part.key:
            self._pending_part = None
            self._pending_reveal = False
            return
        if self._pending_part != (path, part.key):
            self._pending_reveal = False
        self._pending_part = (path, part.key)
        if self.isVisible() and not self._saving:
            self._open_pending_part()

    def reveal_project_part(self, part):
        """Explicit 2D navigation also fits an already selected 3D component."""
        if self._project_root is None:
            return
        self._pending_part = (self._project_root / part.source_file, part.key)
        self._pending_reveal = True
        if self.isVisible() and not self._saving:
            self._open_pending_part()

    def showEvent(self, event):
        super().showEvent(event)
        if self._pending_part is not None:
            self._open_pending_part()
        elif self.session is None and self.modules.count():
            self._open_active_module()
        self._flush_runtime_refresh()

    def _open_pending_part(self):
        path, key = self._pending_part
        reveal = self._pending_reveal
        if self.session is not None and self.session.path == Path(path).resolve():
            if any(part["key"] == key for part in self.session.document["parts"]):
                self.select_part(key, emit=False)
            self._pending_part = None
        elif self.session is None or not (self.session.dirty or self._invalid_inputs):
            if self.open_path(path, selected_key=key):
                self._pending_part = None
        else:
            self._message("Save or Revert the current draft before switching source files.", error=True)
        if self._pending_part is None:
            self._pending_reveal = False
            if reveal and not self.view.fit_selection():
                self.view.fit_all()

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open instrument model", "", "Instrument TOML (*.toml)")
        if path:
            self.open_path(path)

    def _open_active_module(self):
        path = self.modules.currentData()
        if path and self._project_root is not None:
            self.open_path(self._project_root / path)
            self._sync_source_context()

    def _sync_source_context(self):
        """The selector and file label always describe the document on screen."""
        if self.session is None:
            return
        path = self.session.path
        relative = (path.relative_to(self._project_root).as_posix()
                    if self._project_root is not None and path.is_relative_to(self._project_root)
                    else str(path))
        blocked = self.modules.blockSignals(True)
        index = self.modules.findData(relative)
        if index < 0:
            self.modules.addItem(f"External: {path.name}", relative)
            index = self.modules.count() - 1
        self.modules.setCurrentIndex(index)
        self.modules.blockSignals(blocked)
        marker = "* " if self.session.dirty or self._invalid_inputs else ""
        self.source_label.setText(f"{marker}{path.name} · {self._selected_key or ''}")
        self.source_label.setToolTip(
            f"Source: {path}\nComponent: {self._selected_key or ''}\n"
            "Components belonging to this module share this TOML file."
        )

    def open_path(self, path, *, selected_key=None):
        if self.session is not None and (self.session.dirty or self._invalid_inputs):
            self._message("Save, Save copy or Revert the current draft before opening another file.", error=True)
            return False
        try:
            candidate = PartModelDocument(path)
        except Exception as exc:
            self._message(str(exc), error=True)
            return False
        self.session = candidate
        self._invalid_inputs = {}
        self._topology_selection = ()
        self._pending_part = None
        self._pending_reveal = False
        self._selected_key = selected_key or candidate.document["parts"][0]["key"]
        self._selected_region = "body"
        self._aperture_index = 0
        self._load_tree()
        self.select_part(self._selected_key, emit=False)
        self.view.fit_all()
        self._message("File opened. Edit existing dimensions or assign materials, then Save.")
        self._update_buttons()
        return True

    def _load_tree(self):
        self._loading = True
        try:
            self.tree.clear()
            nodes = {}
            for part in self.session.document["parts"]:
                item = QTreeWidgetItem((part.get("name", part["key"]),))
                item.setData(0, Qt.ItemDataRole.UserRole, part["key"])
                item.setToolTip(0, part["key"])
                nodes[part["key"]] = item
            for part in self.session.document["parts"]:
                item = nodes[part["key"]]
                parent = nodes.get(part.get("parent_key"))
                if parent is not None and parent is not item:
                    parent.addChild(item)
                else:
                    self.tree.addTopLevelItem(item)
            self._tree_nodes = nodes
        finally:
            self._loading = False

    def _tree_selected(self, current, _previous):
        if current is not None and not self._loading:
            self.select_part(current.data(0, Qt.ItemDataRole.UserRole))

    def _surface_selected(self, key, region):
        if key:
            self.select_part(key, region=region)
        else:
            self._sync_view_selection()

    def select_part(self, key, *, region="body", emit=True):
        if self.session is None:
            return
        try:
            part = self.session.part(key)
        except StopIteration:
            return
        preserve_view = self._selected_key == key or self.scope.currentData() == "module"
        if self._selected_key != key:
            self._aperture_index = 0
            self._topology_selection = ()
        self._selected_key, self._selected_region = key, region
        self._loading = True
        self.tree.setCurrentItem(self._tree_nodes[key])
        self.tree.scrollToItem(self._tree_nodes[key])
        self._loading = False
        self.selection_label.setText(part.get("name", key))
        self._load_parameters()
        self._render(preserve_view=preserve_view)
        self._sync_source_context()
        if emit:
            self._emit_project_selection()

    def _emit_project_selection(self):
        if self._project_root is not None and self.session.path.is_relative_to(self._project_root):
            if self.session.path.relative_to(self._project_root).as_posix() in self._project_paths:
                self.component_selected.emit(self._selected_key)

    def _sync_view_selection(self):
        """A selected assembly highlights its material children as one component."""
        keys = {self._selected_key}
        if self.session is not None:
            while True:
                children = {part["key"] for part in self.session.document["parts"]
                            if part.get("parent_key") in keys}
                if children <= keys:
                    break
                keys.update(children)
        self.view.set_selection(self._selected_key,
                                None if self._selected_region == "body" else self._selected_region,
                                related_keys=keys)

    def _load_parameters(self):
        if self.session is None or self._selected_key is None:
            return
        from temsim.part_model_3d import part_model_from_document, part_dimension_specs
        runtime_values = self._model_runtime_values()
        fields = part_dimension_specs(self.session.document, self._selected_key, runtime_values=runtime_values)
        base = self.session.part(self._selected_key).get("model_3d", {}).get("base", {})
        if base.get("kind", "existing") != "existing":
            # Explicit solids own their model dimensions. Original assembly
            # dimensions remain available in All parameters and the TOML panel.
            fields = tuple(field for field in fields if field.path[2] == "model_3d")
        try:
            model = part_model_from_document(self.session.document, self._selected_key,
                                             aperture_index=self._aperture_index, runtime_values=runtime_values)
            notes = model.notes
            regions = sorted({mesh.region for mesh in model.meshes if mesh.key == self._selected_key})
        except ValueError as exc:
            notes, regions = ("Preview needs valid dimensions: " + str(exc),), ()
        self._loading = True
        try:
            self.dimensions.setRowCount(len(fields))
            for row, field in enumerate(fields):
                dimension_name = field.path[2]
                text = field.label
                if dimension_name == "model_3d":
                    text = self._model_dimension_label(field.path, field.label)
                label = QTableWidgetItem(text)
                label.setFlags(label.flags() & ~Qt.ItemFlag.ItemIsEditable)
                value = QTableWidgetItem(self._invalid_inputs.get(tuple(field.path), format_toml_value(field.value)))
                value.setData(Qt.ItemDataRole.UserRole, field.path)
                value.setData(Qt.ItemDataRole.UserRole + 2, field)
                if not field.editable:
                    value.setFlags(value.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    value.setForeground(Qt.GlobalColor.gray)
                for item in (label, value):
                    item.setToolTip(field.reason or str(field.path))
                    if dimension_name == "vacuum_inner_diameter_mm" and not field.reason:
                        item.setToolTip(
                            "Electron-beam passage constraint, separate from the material hole. "
                            "Edit Material inner diameter or Existing bore diameter to resize the body hole. "
                            "The saved assembly must satisfy its shared vacuum constraints."
                        )
                units = QTableWidgetItem(field.unit)
                units.setFlags(units.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.dimensions.setItem(row, 0, label)
                self.dimensions.setItem(row, 1, value)
                self.dimensions.setItem(row, 2, units)
            self.model_note.setText("\n".join(notes[:2]))
            self.model_note.setToolTip("\n".join(notes))
            selected_part = self.session.part(self._selected_key)
            hole_field = next((name for name in ("aperture_hole_diameters_mm", "aperture_hole_diameters_um",
                "hole_diameters_mm", "hole_diameters_um") if name in selected_part), None)
            self.aperture.clear()
            if hole_field:
                for index, value in enumerate(selected_part[hole_field]):
                    self.aperture.addItem(f"Hole {index + 1}: {value:g} {hole_field.rsplit('_', 1)[-1]}", index)
                self.aperture.setCurrentIndex(min(self._aperture_index, self.aperture.count() - 1))
            self.aperture.setVisible(bool(hole_field))
            self.region.clear()
            self.region.addItem("Entire part (default)", "body")
            for name in regions:
                if name != "body":
                    self.region.addItem(name.title(), name)
            self.region.setCurrentIndex(max(0, self.region.findData(self._selected_region)))
            self._selected_region = self.region.currentData()
            self._load_all_parameters()
            self._load_features()
        finally:
            self._loading = False
        self._refresh_parameter_annotations()
        self._refresh_calculation_status()
        self._region_changed()
        self._highlight_parameter_paths({tuple(path) for item in self._topology_selection
                                         for path in item.get("parameter_paths", ())})

    def _model_dimension_label(self, path, fallback):
        if len(path) < 5:
            return fallback
        section, field = path[3:5]
        if section == "base":
            return {"width_mm": "Base width (X)", "height_mm": "Base height (Y)",
                    "length_mm": "Base length (Z)"}.get(field, fallback)
        if section == "transform" and len(path) == 6:
            axis = "XYZ"[path[5]]
            return axis + {"scale_xy": " scale", "offset_mm": " offset",
                           "rotation_deg": " rotation"}.get(field, " " + field)
        if section == "features" and len(path) >= 6:
            feature = self.session.part(path[1])["model_3d"]["features"][field]
            name = {"diameter_mm": "Diameter", "depth_mm": "Cut depth",
                    "width_mm": "Slot width", "length_mm": "Slot length",
                    "rotation_deg": "Slot rotation"}.get(path[5], path[5])
            if path[5] == "center_mm" and len(path) == 7:
                name = "XYZ"[path[6]] + " position"
            return feature["id"] + "\n" + name
        return fallback

    def _load_all_parameters(self):
        self._runtime_parameters_pending = False
        self.parameters.clear()
        part = self.session.part(self._selected_key)
        by_key = {item["key"]: item for item in self.session.document["parts"]}
        rows = [("Selected component", part, ("parts", self._selected_key))]
        if part.get("parent_key") in by_key:
            rows.append(("Parent assembly", by_key[part["parent_key"]], ("parts", part["parent_key"])))
        rows.extend(("Child: " + item.get("name", item["key"]), item, ("parts", item["key"]))
                    for item in by_key.values() if item.get("parent_key") == self._selected_key)
        rows.extend((name.title(), self.session.document[name], (name,)) for name in ("module", "geometry", "ports")
                    if name in self.session.document)
        runtime_values = self._model_runtime_values()
        if self._selected_key in runtime_values:
            rows.append(("Current operating values (read-only)", runtime_values[self._selected_key], ("runtime", self._selected_key)))
        elif part.get("parent_key") in runtime_values:
            rows.append(("Parent operating values (read-only)", runtime_values[part["parent_key"]], ("runtime", part["parent_key"])))
        for name, values, path in rows:
            node = QTreeWidgetItem((name, ""))
            self.parameters.addTopLevelItem(node)
            self._parameter_children(node, values, path)
        self.parameters.topLevelItem(0).setExpanded(True)

    @classmethod
    def _parameter_children(cls, parent, values, path=()):
        entries = values.items() if isinstance(values, dict) else enumerate(values)
        for key, value in entries:
            nested = isinstance(value, (dict, list, tuple))
            item = QTreeWidgetItem((str(key), "" if nested else str(value)))
            item.setData(0, Qt.ItemDataRole.UserRole, (*path, key))
            item.setToolTip(1, str(value))
            parent.addChild(item)
            if nested:
                cls._parameter_children(item, value, (*path, key))

    def _dimension_changed(self, item):
        if self._loading or item.column() != 1 or self.session is None:
            return
        path = tuple(item.data(Qt.ItemDataRole.UserRole))
        try:
            self.session.set_dimension(path, float(item.text()))
            self._invalid_inputs.pop(path, None)
            if len(path) == 4 and path[2] in {"aperture_hole_diameters_mm", "aperture_hole_diameters_um",
                                           "hole_diameters_mm", "hole_diameters_um"}:
                self._aperture_index = path[3]
            self._load_parameters()
            if self._render(preserve_view=True):
                self._message("Unapplied draft. Other dimensions can be edited before validating and saving.")
        except Exception as exc:
            self._invalid_inputs[path] = item.text()
            self._message(str(exc), error=True)
        self._update_buttons()

    def _material_changed(self):
        key = self.material.currentData()
        row = next((item for item in self._materials if item["material_key"] == key), {})
        self.material_hint.setText(row.get("scope", ""))

    def _aperture_changed(self):
        if not self._loading and self.aperture.currentData() is not None:
            self._aperture_index = self.aperture.currentData()
            self._render(preserve_view=True)

    def _region_changed(self):
        if self._loading or self.session is None or not self._selected_key:
            return
        from temsim.part_materials import material_for_region, material_application_scope
        self._selected_region = self.region.currentData() or "body"
        self._sync_view_selection()
        try:
            assignment = material_for_region(self.session.part(self._selected_key), self._selected_region)
            current_class = self.session.part(self._selected_key).get("material_class", "").lower()
            material_key = assignment["material_key"] if assignment else (
                "copper" if "copper" in current_class or "winding" in current_class else
                "nonmagnetic_stainless_steel" if "nonmagnetic" in current_class else "femm_pure_iron"
            )
            index = self.material.findData(material_key)
            if index >= 0:
                self.material.setCurrentIndex(index)
            self.material_current.setText("Assigned: " + assignment["label"] if assignment else
                "Current: " + self.session.part(self._selected_key).get("material_class", "Unspecified") + " (existing model default)")
            self.material_current.setText(self.material_current.text() + "\n\n" +
                material_application_scope(self.session.part(self._selected_key)))
        except ValueError as exc:
            self.material_current.setText(str(exc))

    def assign_material(self):
        if self.session is None or self._selected_key is None:
            return
        try:
            self.session.assign_material(self._selected_key, self.material.currentData(), self._selected_region)
            self._load_parameters()
            self._render(preserve_view=True)
            self._message("Material assignment staged. Save to apply it to the file.")
        except Exception as exc:
            self._message(str(exc), error=True)
        self._update_buttons()

    def _render(self, *, preserve_view=False):
        if self.session is None or self._selected_key is None:
            return
        self._runtime_refresh_pending = False
        from temsim.part_model_3d import part_model_from_document, module_model_from_document
        from temsim.part_materials import configured_region_colour
        runtime_values = self._model_runtime_values()
        try:
            if self.scope.currentData() == "module":
                model = module_model_from_document(self.session.document, angular_segments=16,
                                                   runtime_values=runtime_values)
                meshes = list(model.meshes)
            else:
                model = part_model_from_document(self.session.document, self._selected_key,
                                                 aperture_index=self._aperture_index, runtime_values=runtime_values)
                meshes = list(model.meshes)
                if self.scope.currentData() == "context":
                    parent = self.session.part(self._selected_key).get("parent_key")
                    for part in self.session.document["parts"]:
                        if parent and part.get("parent_key") == parent and part["key"] != self._selected_key:
                            meshes.extend(part_model_from_document(self.session.document, part["key"],
                                include_children=False, runtime_values=runtime_values).meshes)
            records, seen = [], set()
            for mesh in meshes:
                part = self.session.part(mesh.key)
                if part.get("mechanical_profile") == "magnetic_lens_assembly" and not part.get("model_3d") and len(meshes) > 1:
                    continue  # The parent envelope would hide its actual material children.
                # A part/region may consist of several distinct surface meshes.
                signature = (mesh.key, mesh.region, mesh.description)
                if signature in seen:
                    continue
                seen.add(signature)
                color = configured_region_colour(part, mesh.region, mesh.color)
                records.append(dict(vertices=mesh.vertices, faces=mesh.faces, key=mesh.key,
                                    region=mesh.region, color=color,
                                    face_groups=mesh.face_groups, surfaces=mesh.surfaces, edges=mesh.edges))
            self._mesh_records = tuple(records)
            selected_topology = self._topology_selection
            self.view.set_meshes(records, preserve_view=preserve_view)
            self._sync_view_selection()
            self.view.set_topology_selection(selected_topology)
            self._topology_changed(self.view.topology_selection, activate_parameters=False)
            return True
        except Exception as exc:
            self._mesh_records = ()
            self.view.set_meshes((), preserve_view=True)
            self._topology_changed(())
            self._message("Preview: " + str(exc), error=True)
            return False

    def save(self):
        if self.session is None:
            return False
        if self._invalid_inputs:
            self._message("Correct the invalid dimension entries before saving.", error=True)
            return False
        self._saving = True
        pending, reveal = self._pending_part, self._pending_reveal
        completed = False
        try:
            callback = None
            if self._project_root is not None and self.session.path.is_relative_to(self._project_root):
                relative = self.session.path.relative_to(self._project_root).as_posix()
                if relative in self._project_paths:
                    callback = self._project_save
                else:
                    raise ValueError("This file is outside the active module selection. Save a copy outside the instrument catalog.")
            self.session.save(project_save=callback)
            self._load_parameters()
            self._message("Saved and validated. Geometry and material assignments are up to date.")
            self.saved.emit(str(self.session.path))
            completed = True
            return True
        except Exception as exc:
            self._message("Not saved: " + str(exc), error=True)
            return False
        finally:
            self._saving = False
            self._update_buttons()
            if pending is not None:
                self._pending_part, self._pending_reveal = pending, reveal
            if completed:
                self._resume_pending_part()

    def save_copy(self):
        if self.session is None:
            return
        if self._invalid_inputs:
            self._message("Correct the invalid dimension entries before saving a copy.", error=True)
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save model copy", str(Path.cwd() / (self.session.path.stem + "-edited.toml")), "Instrument TOML (*.toml)")
        if not path:
            return
        try:
            destination = Path(path).resolve()
            if self._project_root is not None and destination.is_relative_to(self._project_root):
                raise ValueError("Save copies outside the instrument catalog; use Save for the active module.")
            self.session.save_copy(destination)
            self._load_parameters()
            self._render(preserve_view=True)
            self._message("Model copy saved. This file is independent of the active instrument.")
            self._update_buttons()
        except Exception as exc:
            self._message("Not saved: " + str(exc), error=True)

    def undo(self):
        if self.session is not None:
            if self._invalid_inputs:
                self._invalid_inputs.clear()
            else:
                self.session.undo()
            self._draft_changed()

    def redo(self):
        if self.session is not None:
            self.session.redo()
            self._draft_changed()

    def revert(self):
        if self.session is not None:
            self._invalid_inputs.clear()
            self.session.revert()
            self._draft_changed()
            self._resume_pending_part()

    def _resume_pending_part(self):
        if self.isVisible() and self._pending_part is not None:
            path, key = self._pending_part
            self._open_pending_part()
            if self._pending_part is None and self.session.path == Path(path).resolve() and self._selected_key == key:
                self._emit_project_selection()

    def _draft_changed(self):
        self._load_parameters()
        rendered = self._render(preserve_view=True)
        self._update_buttons()
        if rendered:
            self._message("Draft updated." if self.session.dirty else "Saved dimensions restored.")

    def _message(self, text, *, error=False):
        self.status.setText(text)
        self.status.setStyleSheet("color: #ffb4a9;" if error else "")

    def _update_buttons(self):
        loaded = self.session is not None
        self.save_button.setEnabled(loaded and self.session.dirty and not self._invalid_inputs)
        self.save_copy_button.setEnabled(loaded and not self._invalid_inputs)
        self.revert_button.setEnabled(loaded and (self.session.dirty or bool(self._invalid_inputs)))
        self.undo_button.setEnabled(loaded and (self.session.can_undo or bool(self._invalid_inputs)))
        self.redo_button.setEnabled(loaded and self.session.can_redo and not self._invalid_inputs)
        self.assign_button.setEnabled(loaded and self._selected_key is not None)
        self.apply_shape_button.setEnabled(loaded)
        self.base_shape.setEnabled(loaded)
        self.add_hole_button.setEnabled(loaded)
        self.add_slot_button.setEnabled(loaded)
        if loaded:
            self._sync_source_context()
        self._refresh_calculation_status()
