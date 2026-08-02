"""TOML-backed instrument navigation tree."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem


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

    def load_assembly(self, assembly, runtime_targets=None) -> None:
        self.clear()
        parts_by_module = {}
        for part in assembly.parts:
            parts_by_module.setdefault(part.module_key, []).append(part)
        selected_paths = dict(assembly.selected_module_paths)

        for module in assembly.modules:
            module_path = selected_paths[module.type]
            root = QTreeWidgetItem([module.key])
            root.setData(0, Qt.ItemDataRole.UserRole, TreeSelection(
                key=f"@module:{module.type}",
                label=module.key,
                module_path=module_path,
                is_module=True,
            ))
            self.addTopLevelItem(root)
            for part in parts_by_module.get(module.key, ()):
                child = QTreeWidgetItem([part.name])
                child.setToolTip(0, part.key)
                child.setData(0, Qt.ItemDataRole.UserRole, TreeSelection(
                    key=part.key,
                    label=part.name,
                    module_path=module_path,
                ))
                root.addChild(child)

        part_keys = {part.key for part in assembly.parts}
        extra_targets = [
            target for key, target in (runtime_targets or {}).items()
            if key not in part_keys
        ]
        if extra_targets:
            root = QTreeWidgetItem(["Additional operating controls"])
            self.addTopLevelItem(root)
            for target in sorted(extra_targets, key=lambda item: item.label):
                child = QTreeWidgetItem([target.label])
                child.setToolTip(0, target.key)
                child.setData(0, Qt.ItemDataRole.UserRole, TreeSelection(
                    key=target.key,
                    label=target.label,
                    module_path=None,
                ))
                root.addChild(child)

        self.expandToDepth(0)
        if self.topLevelItemCount():
            first_root = self.topLevelItem(0)
            self.setCurrentItem(
                first_root.child(0) if first_root.childCount() else first_root
            )

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
