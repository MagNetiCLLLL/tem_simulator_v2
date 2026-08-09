"""Filtered, TOML-backed instrument navigation trees."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem

from temsim.component_keys import (
    ENERGY_FILTER_INTERNAL_KEYS,
    IMAGE_CORRECTOR_KEYS,
    PROBE_CORRECTOR_KEYS,
)


OPTICAL_FILTERS = (
    ("all", "All optical"),
    ("lens", "Lenses"),
    ("deflector", "Deflectors"),
    ("aperture", "Apertures"),
    ("stigmator", "Stigmators"),
    ("corrector", "Correctors"),
    ("energy_filter", "Energy Filter"),
    ("other", "Source / sample / detectors"),
)
OPTICAL_CATEGORY_LABELS = {
    "lens": "Lenses",
    "deflector": "Deflectors",
    "aperture": "Apertures",
    "stigmator": "Stigmators",
    "corrector": "Correctors",
    "energy_filter": "Energy Filter",
    "other": "Source / sample / detectors",
}
OPTICAL_CATEGORY_ORDER = tuple(OPTICAL_CATEGORY_LABELS)
CORRECTOR_KEYS = frozenset((*PROBE_CORRECTOR_KEYS, *IMAGE_CORRECTOR_KEYS))
GLOBAL_RUNTIME_KEYS = ("simulation", "electron_gun")


@dataclass(frozen=True, slots=True)
class TreeSelection:
    key: str
    label: str
    module_path: str | None
    is_module: bool = False


class InstrumentTree(QTreeWidget):
    component_selected = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("instrumentTree")
        self.setHeaderLabel("Active TEM assembly")
        self.setAlternatingRowColors(True)
        self.setMinimumWidth(285)
        self.currentItemChanged.connect(self._emit_current_item)

    @staticmethod
    def _selection_item(label: str, selection: TreeSelection):
        item = QTreeWidgetItem([label])
        if not selection.is_module:
            item.setToolTip(0, selection.key)
        item.setData(0, Qt.ItemDataRole.UserRole, selection)
        return item

    @staticmethod
    def _module_paths(assembly) -> dict[str, str]:
        selected_paths = dict(assembly.selected_module_paths)
        return {
            module.key: selected_paths[module.type]
            for module in assembly.modules
        }

    @staticmethod
    def _is_optical_part(part, runtime_targets) -> bool:
        return (
            part.key in runtime_targets
            and not bool(part.data.get("mechanical_only", False))
        )

    @staticmethod
    def optical_category(part, runtime_target=None) -> str:
        key = str(part.key)
        if key == "energy_filter" or key in ENERGY_FILTER_INTERNAL_KEYS:
            return "energy_filter"
        if key in CORRECTOR_KEYS:
            return "corrector"
        if "aperture" in key:
            return "aperture"
        if "stigmator" in key:
            return "stigmator"
        if "deflector" in key:
            return "deflector"
        obj = getattr(runtime_target, "obj", None)
        if "lens" in key or (obj is not None and hasattr(obj, "percent")):
            return "lens"
        return "other"

    def load_optical(
        self,
        assembly,
        runtime_targets=None,
        category: str = "all",
        select_first: bool = True,
    ) -> None:
        self.clear()
        self.setHeaderLabel("Optical calculation components")
        targets = runtime_targets or {}
        module_paths = self._module_paths(assembly)
        grouped = {name: [] for name in OPTICAL_CATEGORY_ORDER}
        for part in assembly.parts:
            if not self._is_optical_part(part, targets):
                continue
            part_category = self.optical_category(
                part, targets.get(part.key)
            )
            if category not in {"all", part_category}:
                continue
            grouped[part_category].append(part)

        # Curvilinear Energy Filter internals are runtime optical targets,
        # not fictitious axial column parts.  Show them in their own group
        # without flattening their branch geometry into the main assembly.
        energy_filter_keys = (
            "energy_filter",
            *ENERGY_FILTER_INTERNAL_KEYS,
        )
        parts_by_key = {part.key: part for part in assembly.parts}
        energy_filter_entries = []
        for key in energy_filter_keys:
            target = targets.get(key)
            part = parts_by_key.get(key)
            if target is None and part is None:
                continue
            label = part.name if part is not None else target.label
            module_path = (
                module_paths[part.module_key]
                if part is not None else None
            )
            energy_filter_entries.append((key, label, module_path))

        if category in {"all", "other"}:
            controls = [
                targets[key]
                for key in GLOBAL_RUNTIME_KEYS
                if key in targets
            ]
        else:
            controls = []
        if controls:
            root = QTreeWidgetItem([f"Global controls ({len(controls)})"])
            self.addTopLevelItem(root)
            for target in controls:
                root.addChild(self._selection_item(
                    target.label,
                    TreeSelection(target.key, target.label, None),
                ))

        for part_category in OPTICAL_CATEGORY_ORDER:
            if part_category == "energy_filter":
                if (
                    category in {"all", "energy_filter"}
                    and energy_filter_entries
                ):
                    root = QTreeWidgetItem([
                        f"Energy Filter ({len(energy_filter_entries)})"
                    ])
                    root.setData(
                        0,
                        Qt.ItemDataRole.UserRole + 1,
                        "energy_filter",
                    )
                    self.addTopLevelItem(root)
                    for key, label, module_path in energy_filter_entries:
                        root.addChild(self._selection_item(
                            label,
                            TreeSelection(
                                key, label, module_path
                            ),
                        ))
                continue
            parts = grouped[part_category]
            if not parts:
                continue
            label = OPTICAL_CATEGORY_LABELS[part_category]
            root = QTreeWidgetItem([f"{label} ({len(parts)})"])
            root.setData(0, Qt.ItemDataRole.UserRole + 1, part_category)
            self.addTopLevelItem(root)
            for part in parts:
                root.addChild(self._selection_item(
                    part.name,
                    TreeSelection(
                        part.key,
                        part.name,
                        module_paths[part.module_key],
                    ),
                ))

        self._finish_load(select_first)

    def load_mechanical(
        self, assembly, runtime_targets=None, select_first: bool = False
    ) -> None:
        self.clear()
        self.setHeaderLabel("Mechanical layout components")
        targets = runtime_targets or {}
        selected_paths = dict(assembly.selected_module_paths)
        module_paths = self._module_paths(assembly)

        manifest_root = QTreeWidgetItem(["Module manifests"])
        self.addTopLevelItem(manifest_root)
        for module in assembly.modules:
            module_path = selected_paths[module.type]
            manifest_root.addChild(self._selection_item(
                module.key,
                TreeSelection(
                    f"@module:{module.type}",
                    module.key,
                    module_path,
                    is_module=True,
                ),
            ))

        for module in assembly.modules:
            parts = [
                part
                for part in assembly.parts
                if part.module_key == module.key
                and part.key not in ENERGY_FILTER_INTERNAL_KEYS
                and not self._is_optical_part(part, targets)
            ]
            if not parts:
                continue
            root = QTreeWidgetItem([f"{module.key} ({len(parts)})"])
            self.addTopLevelItem(root)
            for part in parts:
                root.addChild(self._selection_item(
                    part.name,
                    TreeSelection(
                        part.key,
                        part.name,
                        module_paths[part.module_key],
                    ),
                ))

        self._finish_load(select_first)

    def _finish_load(self, select_first: bool) -> None:
        self.expandToDepth(0)
        if select_first:
            self.select_first()

    def select_first(self) -> bool:
        pending = [
            self.topLevelItem(index)
            for index in range(self.topLevelItemCount())
        ]
        while pending:
            item = pending.pop(0)
            selection = item.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(selection, TreeSelection):
                self.setCurrentItem(item)
                return True
            pending.extend(
                item.child(index) for index in range(item.childCount())
            )
        return False

    def current_key(self) -> str | None:
        current = self.currentItem()
        if current is None:
            return None
        selection = current.data(0, Qt.ItemDataRole.UserRole)
        return selection.key if isinstance(selection, TreeSelection) else None

    def _emit_current_item(self, current, _previous) -> None:
        if current is None:
            return
        selection = current.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(selection, TreeSelection):
            self.component_selected.emit(selection)

    def select_key(self, key: str) -> bool:
        """Select a component by canonical key for plot-to-tree linking."""

        pending = [
            self.topLevelItem(index)
            for index in range(self.topLevelItemCount())
        ]
        while pending:
            item = pending.pop(0)
            selection = item.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(selection, TreeSelection) and selection.key == key:
                self.setCurrentItem(item)
                self.scrollToItem(item)
                return True
            pending.extend(
                item.child(index) for index in range(item.childCount())
            )
        return False
