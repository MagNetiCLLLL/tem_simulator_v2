"""Read-only whole-column 3D view of the resolved, saved assembly.

This page owns presentation state only. Meshes are built lazily and retained
across tab changes; camera, visibility and selection never request a solve.
"""

from copy import deepcopy
from types import SimpleNamespace

from PySide6.QtCore import QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox, QHBoxLayout, QLabel, QPushButton, QSplitter, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget,
)

from temsim.assembly_model_3d import (
    assembly_model_fingerprint, assembly_model_from_assembly,
)
from temsim.gui.part_model_view import PartModelView


class AssemblyModelPage(QWidget):
    component_selected = Signal(str)
    edit_part_requested = Signal(str, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("assemblyModelPage")
        self._assembly = None
        self._runtime_values = {}
        self._pending = False
        self._fingerprint = None
        self._model = None
        self._parts = {}
        self._editable_keys = set()
        self._tree_items = {}
        self._hidden_keys = set()
        self._selected_key = None
        self._has_fitted = False
        self.mesh_builds = 0
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self._flush)

        self.fit_all = QPushButton("Fit assembly")
        self.fit_selected = QPushButton("Fit selected")
        self.column_view = QPushButton("Side view")
        self.isometric = QPushButton("Isometric")
        self.edit_part = QPushButton("Open in 3D Parts")
        self.edit_part.setEnabled(False)
        self.section = QCheckBox("Section")
        self.section.setToolTip("Hide the y > 0 half to expose internal surfaces from the default view. Display only.")
        self.envelopes = QCheckBox("Schematic envelopes")
        self.envelopes.setChecked(True)
        self.envelopes.setToolTip(
            "Show labelled envelopes for parts without a configured solid model. "
            "These are not verified OEM structures."
        )
        self.view = PartModelView()
        self.view.setObjectName("assemblyModelView")
        self.view.set_view_labels(title="3D assembly", empty_text="No visible assembly surfaces")
        self.view.set_column_view()
        self.tree = QTreeWidget()
        self.tree.setObjectName("assemblyModelTree")
        self.tree.setHeaderLabels(("Component / visibility",))
        self.tree.setMinimumWidth(140)
        self.tree.setIndentation(12)
        self.tree.setToolTip("Select a component; uncheck it to hide its surfaces. Display only.")
        self.selection_label = QLabel("Select a component")
        self.status = QLabel("Current saved assembly | no calculation required")
        for label in (self.selection_label, self.status):
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.status.setToolTip(
            "Global column coordinates in mm, equal scale in X/Y/Z. +Z is downstream. "
            "The default side view places the source above the detectors. "
            "Unsaved 3D Parts drafts are not included. Viewing does not change optics or cached signals."
        )
        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.addWidget(self.tree, 1)
        side_layout.addWidget(self.selection_label)
        side_layout.addWidget(self.edit_part)
        canvas = QWidget()
        canvas_layout = QVBoxLayout(canvas)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        actions = QHBoxLayout()
        for widget in (self.fit_all, self.fit_selected, self.column_view, self.isometric):
            actions.addWidget(widget)
        actions.addStretch(1)
        options = QHBoxLayout()
        options.addWidget(self.section)
        options.addWidget(self.envelopes)
        options.addStretch(1)
        canvas_layout.addLayout(actions)
        canvas_layout.addLayout(options)
        canvas_layout.addWidget(self.view, 1)
        self.splitter = QSplitter()
        self.splitter.setObjectName("assemblyModelSplitter")
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(side)
        self.splitter.addWidget(canvas)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([240, 1050])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self.splitter, 1)
        layout.addWidget(self.status)
        self.fit_all.clicked.connect(self.view.fit_all)
        self.fit_selected.clicked.connect(self.fit_current_selection)
        self.column_view.clicked.connect(self.view.set_column_view)
        self.isometric.clicked.connect(self.view.set_column_isometric_view)
        self.section.toggled.connect(lambda checked: self.view.set_section_enabled(checked, keep_positive_y=False))
        self.envelopes.toggled.connect(self._display_meshes)
        self.edit_part.clicked.connect(self._open_selected_part)
        self.tree.currentItemChanged.connect(self._tree_selected)
        self.tree.itemChanged.connect(self._visibility_changed)
        self.view.selection_changed.connect(self._surface_selected)

    def set_assembly(self, assembly, runtime_values=None):
        """Keep only the latest context while hidden or edits are arriving."""
        self._assembly = assembly
        self._runtime_values = deepcopy(runtime_values or {})
        self._pending = True
        if self.isVisible():
            self._refresh_timer.start(40)

    def set_runtime_values(self, runtime_values):
        self.set_assembly(self._assembly, runtime_values)

    def showEvent(self, event):
        super().showEvent(event)
        if self._pending:
            self._refresh_timer.start(0)

    def _flush(self):
        if not self._pending or not self.isVisible() or self._assembly is None:
            return
        self._pending = False
        try:
            fingerprint = assembly_model_fingerprint(self._assembly, self._runtime_values)
            if fingerprint == self._fingerprint:
                return
            model = assembly_model_from_assembly(
                self._assembly, runtime_values=self._runtime_values, angular_segments=16,
            )
        except Exception as exc:
            # Never label an old assembly as the new input after a build error.
            self._fingerprint = None
            self._model = None
            self._parts = {}
            self._editable_keys = set()
            self._tree_items = {}
            self.tree.clear()
            self.view.set_meshes((), preserve_view=True)
            self.status.setText(f"Assembly view unavailable: {exc}")
            self.selection_label.setText("Select a component")
            self.edit_part.setEnabled(False)
            return
        self._fingerprint, self._model = fingerprint, model
        self.mesh_builds += 1
        self._parts = {part.key: part for part in self._assembly.parts}
        self._editable_keys = set(self._parts)
        for liner in getattr(self._assembly, "vacuum_liner_segments", ()):
            self._parts.setdefault(liner.key, SimpleNamespace(
                key=liner.key, name=liner.name,
                center_z_mm=(liner.start_z_mm + liner.end_z_mm) * 0.5, data={},
            ))
        self._hidden_keys.intersection_update(self._parts)
        with QSignalBlocker(self.tree):
            self.tree.clear()
            self._tree_items = {}
            modeled = {mesh.key for mesh in model.meshes}
            approximate = {mesh.key for mesh in model.meshes if not mesh.is_exact}
            for part in sorted(self._parts.values(), key=lambda p: (p.center_z_mm, p.key)):
                item = QTreeWidgetItem((part.name,))
                item.setData(0, Qt.ItemDataRole.UserRole, part.key)
                if part.key in modeled:
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(0, Qt.CheckState.Unchecked if part.key in self._hidden_keys else Qt.CheckState.Checked)
                kind = "Schematic envelope" if part.key in approximate else "Configured geometry"
                if part.key not in modeled:
                    kind = "No solid surface (reference / assembly parent / unavailable)"
                if part.key not in self._editable_keys:
                    kind += "; module-defined liner (read-only)"
                item.setToolTip(0, f"{part.name}\nZ {part.center_z_mm:.6g} mm\n{kind}")
                self.tree.addTopLevelItem(item)
                self._tree_items[part.key] = item
        self._display_meshes()
        self.focus_component(self._selected_key)

    def _display_meshes(self, *_):
        if self._model is None:
            return
        meshes = tuple(mesh for mesh in self._model.meshes
                       if mesh.key not in self._hidden_keys
                       and (self.envelopes.isChecked() or mesh.is_exact))
        self.view.set_meshes(meshes, preserve_view=self._has_fitted)
        self._has_fitted |= bool(meshes)
        self._sync_selection()
        approximation_count = len({mesh.key for mesh in meshes if not mesh.is_exact})
        error_text = f" | {len(self._model.errors)} unavailable" if self._model.errors else ""
        self.status.setText(
            f"Saved assembly | {len({mesh.key for mesh in meshes})} visible parts | "
            f"{approximation_count} schematic | mm | +Z downstream{error_text}"
        )
        details = ["Configured geometry is not a claim of OEM CAD accuracy.",
                   "Unsaved 3D Parts drafts are excluded. Display changes do not alter calculations.",
                   *self._model.notes, *self._model.errors]
        self.status.setToolTip("\n".join(details))

    def _visibility_changed(self, item, _column):
        key = item.data(0, Qt.ItemDataRole.UserRole)
        if item.checkState(0) == Qt.CheckState.Checked:
            self._hidden_keys.discard(key)
        else:
            self._hidden_keys.add(key)
        self._display_meshes()

    def _related_keys(self, key):
        if not key:
            return set()
        keys = {key}
        while True:
            more = {part.key for part in self._parts.values()
                    if part.data.get("parent_key") in keys
                    or keys.intersection(part.data.get("magnetic_lens_keys", ()))}
            if more <= keys:
                return keys
            keys.update(more)

    def _sync_selection(self):
        self.view.set_selection(self._selected_key, related_keys=self._related_keys(self._selected_key))

    def focus_component(self, part):
        key = getattr(part, "key", part)
        self._selected_key = key
        with QSignalBlocker(self.tree):
            item = self._tree_items.get(key)
            self.tree.setCurrentItem(item)
            if item is not None:
                self.tree.scrollToItem(item)
        self._sync_selection()
        selected = self._parts.get(key)
        self.edit_part.setEnabled(key in self._editable_keys)
        self.selection_label.setText(
            f"{selected.name} | Z {selected.center_z_mm:.6g} mm"
            if selected is not None else "Select a component"
        )

    def _tree_selected(self, item, _previous):
        if item is not None:
            self._surface_selected(item.data(0, Qt.ItemDataRole.UserRole), "")

    def _surface_selected(self, key, _region):
        self.focus_component(key)
        if key in self._editable_keys:
            self.component_selected.emit(key)

    def _open_selected_part(self):
        part = self._parts.get(self._selected_key)
        if part is not None and part.key in self._editable_keys:
            self.edit_part_requested.emit(part.key, float(part.center_z_mm))

    def fit_current_selection(self):
        self._flush()
        fitted = self.view.fit_selection()
        if not fitted:
            self.selection_label.setText("No visible solid for this selection; enable its surfaces or open 3D Parts.")
        return fitted
