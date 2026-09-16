"""Complete assembly tree over legacy manifests, with stable selection IDs."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QMenu, QTreeWidget, QTreeWidgetItem

from temsim.assembly_structure import GROUPS, build_assembly_structure, stable_id
from temsim.gui.instrument_tree import TreeSelection
from temsim.assembly_navigation import section_by_component


IDENTITY_ROLE = Qt.ItemDataRole.UserRole + 1


class AssemblyStructureTree(QTreeWidget):
    component_selected = Signal(object)
    component_activated = Signal(object)
    navigation_requested = Signal(str, str, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("assemblyStructureTree")
        self.setHeaderLabels(["Whole instrument assembly"])
        self.setAlternatingRowColors(True)
        self.structure = None
        self._items_by_id = {}
        self.currentItemChanged.connect(lambda current, previous: self._emit_selection(current))
        self.itemDoubleClicked.connect(lambda item, column: self._emit_selection(item, activated=True))
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_menu)

    def _context_menu(self, position):
        item = self.itemAt(position)
        selection = item.data(0, Qt.ItemDataRole.UserRole) if item else None
        if selection is None:
            return
        self.setCurrentItem(item)
        row = self.structure.resolve(selection.key)
        menu = QMenu(self)
        menu.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        for label, destination in (("Open in 3D Parts", "parts"), ("Show in Ray Diagram", "ray"), ("Show in Vacuum map", "vacuum")):
            action = menu.addAction(label)
            action.triggered.connect(lambda _checked=False, dest=destination:
                self.navigation_requested.emit(row.key, dest, row.center_z_mm))
        menu.popup(self.viewport().mapToGlobal(position))

    def _emit_selection(self, item, activated=False):
        selection = item.data(0, Qt.ItemDataRole.UserRole) if item else None
        if selection is not None:
            (self.component_activated if activated else self.component_selected).emit(selection)

    def current_identity(self):
        item = self.currentItem()
        return item.data(0, IDENTITY_ROLE) if item else None

    def select_identity(self, instance_id, *, emit=True):
        item = self._items_by_id.get(instance_id)
        if item is None:
            return False
        blocked = self.signalsBlocked()
        self.blockSignals(blocked or not emit)
        try:
            self.setCurrentItem(item)
            parent = item.parent()
            while parent:
                parent.setExpanded(True)
                parent = parent.parent()
            self.scrollToItem(item)
        finally:
            self.blockSignals(blocked)
        return True

    def select_key(self, key, *, emit=True):
        if self.structure is None:
            return False
        try:
            identity = self.structure.resolve(key).instance_id
        except KeyError:
            return False
        return self.select_identity(identity, emit=emit)

    def select_first(self):
        if self.structure and self.structure.components:
            return self.select_identity(self.structure.components[0].instance_id)
        return False

    def load_assembly(self, assembly):
        structure = build_assembly_structure(assembly)
        sections = section_by_component(assembly)
        selected = self.current_identity()
        expanded = {key for key, item in self._items_by_id.items() if item.isExpanded()}
        initial = self.structure is None
        blocked = self.blockSignals(True)
        try:
            self.clear()
            self.structure = structure
            self._items_by_id = {}
            roots = {}
            for key, label in GROUPS:
                count = sum(row.group_key == key for row in structure.components)
                if not count:
                    continue
                item = QTreeWidgetItem([f"{label} ({count})"])
                identity = stable_id("group", key)
                item.setData(0, IDENTITY_ROLE, identity)
                item.setToolTip(0, "Navigation group; component positions and physical parents are unchanged.")
                self.addTopLevelItem(item)
                roots[key] = item
                self._items_by_id[identity] = item
            for row in structure.components:
                item = QTreeWidgetItem([row.name])
                item.setData(0, Qt.ItemDataRole.UserRole, TreeSelection(row.key, row.name, row.source_file))
                item.setData(0, IDENTITY_ROLE, row.instance_id)
                section = sections.get(row.key)
                storage = section.source_file if section else row.source_file
                tip = f"Z {row.start_z_mm:g} to {row.end_z_mm:g} mm\nStorage: {storage}\nComponent ID: {row.instance_id}"
                if section and section.kind == "subassembly":
                    tip += f"\nSubassembly: {section.name}"
                if row.definition_reference:
                    tip += "\nShared definition: " + row.definition_reference
                if row.path_coordinate:
                    tip += f"\nFilter path: {row.path_coordinate}"
                    if row.path_reference:
                        tip += " from " + row.path_reference
                    if row.path_center_mm is not None:
                        tip += f"; s = {row.path_center_mm} mm"
                item.setToolTip(0, tip)
                self._items_by_id[row.instance_id] = item
            # Keep every original mechanical parent, even across functional roles.
            # Display siblings spatially; source ordering and coordinates stay intact.
            for row in sorted(structure.components, key=lambda row: (row.center_z_mm, row.key)):
                parent = self._items_by_id[row.parent_id] if row.parent_id else roots[row.group_key]
                parent.addChild(self._items_by_id[row.instance_id])
            for identity, item in self._items_by_id.items():
                item.setExpanded(identity in expanded or (initial and item.parent() is None))
            if selected in self._items_by_id:
                self.setCurrentItem(self._items_by_id[selected])
            elif self.topLevelItemCount():
                # Opening an instrument selects its heading, not a physical
                # point upstream of emission (the tip body's centre). Startup
                # must leave the ray/beam-analysis observation plane alone.
                self.setCurrentItem(self.topLevelItem(0))
        finally:
            self.blockSignals(blocked)
