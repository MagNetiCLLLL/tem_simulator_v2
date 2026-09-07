"""Named, presentation-only workspace layouts with per-ray-panel size variants."""
from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

from PySide6.QtCore import QByteArray, QEvent, QObject, QSignalBlocker, QTimer
from PySide6.QtGui import QActionGroup
from PySide6.QtWidgets import QDockWidget, QInputDialog, QSplitter, QTabWidget


class WorkspaceLayouts(QObject):
    ROOT = "workspace_layouts/v1"
    RAY_SPLITTERS = {"rayDiagramVerticalSplitter", "rayDiagramWorkspaceSplitter"}

    def __init__(self, window, settings, menu):
        super().__init__(window)
        self.window, self.settings, self.menu = window, settings, menu
        self.active_id = "default"
        self.restoring = True
        self.splitters = self._named_widgets(QSplitter)
        self.tabs = self._named_widgets(QTabWidget)
        workspace = window.workspace
        self.toggles = {
            "magnetic": workspace.magnetic_field_toggle,
            "transverse": workspace.transverse_beam_toggle,
            "advanced_bank": workspace.interactive_calculation.advanced_bank,
        }
        self.ray_variant = self._ray_variant()
        self.splitter_states = {
            name: {self._variant(name): widget.saveState()}
            for name, widget in self.splitters.items()
        }
        self.pending = set()
        self.restore_timer = QTimer(self)
        self.restore_timer.setSingleShot(True)
        self.restore_timer.timeout.connect(self._restore_pending)
        self.save_timer = QTimer(self)
        self.save_timer.setSingleShot(True)
        self.save_timer.setInterval(600)
        self.save_timer.timeout.connect(self.save_current)
        for name, splitter in self.splitters.items():
            splitter.splitterMoved.connect(lambda *_, n=name: self._splitter_moved(n))
            splitter.installEventFilter(self)
        for tabs in self.tabs.values():
            tabs.currentChanged.connect(self.schedule_save)
        for toggle in self.toggles.values():
            toggle.toggled.connect(self.schedule_save)
        window.installEventFilter(self)
        for dock in window.findChildren(QDockWidget):
            dock.installEventFilter(self)
            dock.visibilityChanged.connect(self.schedule_save)
            dock.topLevelChanged.connect(self.schedule_save)
            dock.dockLocationChanged.connect(self.schedule_save)
        workspace.ray_layout_changing.connect(self._before_ray_change)
        workspace.ray_layout_changed.connect(self._after_ray_change)
        self.defaults = self._snapshot()
        self.action_group = None
        self.menu.aboutToShow.connect(self.refresh_menu)

    def _named_widgets(self, widget_type):
        # Persist application-owned widgets by stable names, never by list
        # indices or ambiguous names belonging to plotting-library internals.
        widgets = self.window.findChildren(widget_type)
        names = [widget.objectName() for widget in widgets]
        return {widget.objectName(): widget for widget in widgets
                if widget.objectName() and names.count(widget.objectName()) == 1
                and widget.count() > 0}

    def _ray_variant(self):
        workspace = self.window.workspace
        return f"m{int(workspace.magnetic_field_toggle.isChecked())}t{int(workspace.transverse_beam_toggle.isChecked())}"

    def _variant(self, name):
        return self.ray_variant if name in self.RAY_SPLITTERS else "default"

    def _remember(self, name):
        widget = self.splitters[name]
        if widget.isVisible() and name not in self.pending:
            self.splitter_states.setdefault(name, {})[self._variant(name)] = widget.saveState()

    def _splitter_moved(self, name):
        if not self.restoring:
            # A real drag takes precedence over a pending restore.
            self.pending.discard(name)
            self._remember(name)
            self.schedule_save()

    def _before_ray_change(self):
        if not self.restoring:
            for name in self.RAY_SPLITTERS:
                if name in self.splitters:
                    self._remember(name)

    def _after_ray_change(self):
        if self.restoring:
            return
        self.ray_variant = self._ray_variant()
        self.pending.update(self.RAY_SPLITTERS & self.splitters.keys())
        self.restore_timer.start(0)
        self.schedule_save()

    def eventFilter(self, watched, event):
        if not self.restoring:
            if event.type() == QEvent.Type.Show and isinstance(watched, QSplitter):
                self.pending.add(watched.objectName())
                self.restore_timer.start(0)
            if event.type() in (QEvent.Type.Resize, QEvent.Type.Move, QEvent.Type.Show, QEvent.Type.Hide):
                self.schedule_save()
        return super().eventFilter(watched, event)

    def _restore_pending(self):
        previous = self.restoring
        self.restoring = True
        try:
            for name in tuple(self.pending):
                widget = self.splitters.get(name)
                if widget is None:
                    self.pending.discard(name)
                elif widget.isVisible():
                    state = self.splitter_states.get(name, {}).get(self._variant(name))
                    if isinstance(state, QByteArray):
                        widget.restoreState(state)
                    self.pending.discard(name)
        finally:
            self.restoring = previous

    def schedule_save(self, *_):
        if not self.restoring:
            self.save_timer.start()

    def _snapshot(self):
        for name in self.splitters:
            self._remember(name)
        return {
            "schema": 1,
            "geometry": self.window.saveGeometry(),
            "docks": self.window.saveState(),
            "splitters": deepcopy(self.splitter_states),
            "tabs": {name: tabs.tabText(tabs.currentIndex()) for name, tabs in self.tabs.items()},
            "toggles": {name: toggle.isChecked() for name, toggle in self.toggles.items()},
            "live_width_initialized": getattr(self.window, "_live_tuning_layout_initialized", False),
        }

    def _key(self, layout_id, field):
        return f"{self.ROOT}/layouts/{layout_id}/{field}"

    def entries(self):
        self.settings.beginGroup(f"{self.ROOT}/layouts")
        keys = self.settings.childGroups()
        self.settings.endGroup()
        return {key: str(self.settings.value(self._key(key, "name"), key)) for key in keys}

    def restore_active(self):
        if "default" not in self.entries():
            self.settings.setValue(self._key("default", "name"), "Default")
        selected = str(self.settings.value(f"{self.ROOT}/active", "default"))
        self.active_id = selected if selected in self.entries() else "default"
        saved = self.settings.value(self._key(self.active_id, "data"))
        if isinstance(saved, dict) and saved.get("schema") == 1:
            self._apply(saved)
        else:
            # Migration: keep the legacy main-window state and live splitter
            # on first use, instead of resetting existing user preferences.
            self.restoring = False
        self.refresh_menu()

    def _apply(self, data):
        self.save_timer.stop()
        self.restore_timer.stop()
        self.restoring = True
        try:
            for field, restore in (("geometry", self.window.restoreGeometry), ("docks", self.window.restoreState)):
                value = data.get(field)
                if isinstance(value, QByteArray) and not value.isEmpty():
                    restore(value)
            toggles = data.get("toggles", {})
            for name, checked in (toggles.items() if isinstance(toggles, dict) else ()):
                if name in self.toggles and isinstance(checked, bool):
                    self.toggles[name].setChecked(checked)
            self.ray_variant = self._ray_variant()
            raw = data.get("splitters", {})
            self.splitter_states = {
                name: {variant: state for variant, state in variants.items() if isinstance(state, QByteArray)}
                for name, variants in raw.items() if isinstance(variants, dict)
            } if isinstance(raw, dict) else {}
            self.window._live_tuning_layout_initialized = bool(data.get("live_width_initialized", True))
            # Restoring presentation tabs must not emit computation requests.
            tab_states = data.get("tabs", {})
            for name, title in (tab_states.items() if isinstance(tab_states, dict) else ()):
                if name == "physicalLayoutTabs" and isinstance(title, str):
                    title = {"2D section": "2D", "3D model editor": "3D Parts"}.get(title, title)
                tabs = self.tabs.get(name)
                if tabs is not None:
                    for index in range(tabs.count()):
                        if tabs.tabText(index) == title:
                            with QSignalBlocker(tabs):
                                tabs.setCurrentIndex(index)
                            break
            self.pending = set(self.splitters)
        finally:
            self.restoring = False
        # Hidden pages get their own saved sizes after being laid out on Show.
        self.restore_timer.start(0)

    def save_current(self):
        if self.restoring:
            return
        self.save_timer.stop()
        self._restore_pending()
        self.settings.setValue(self._key(self.active_id, "data"), self._snapshot())
        self.settings.setValue(f"{self.ROOT}/active", self.active_id)
        self.settings.sync()
        if self.settings.status() != self.settings.Status.NoError:
            self.window.status_label.setText("Could not save workspace layout settings")

    def select(self, layout_id):
        if layout_id == self.active_id:
            return
        if layout_id not in self.entries():
            raise ValueError("Unknown workspace layout")
        data = self.settings.value(self._key(layout_id, "data"))
        if not isinstance(data, dict) or data.get("schema") != 1:
            raise ValueError("This workspace layout is unavailable or incompatible")
        self.save_current()
        self.active_id = layout_id
        self._apply(data)
        self.settings.setValue(f"{self.ROOT}/active", layout_id)
        self.refresh_menu()

    def save_as(self, name):
        name = name.strip()
        if not name or len(name) > 64:
            raise ValueError("Enter a layout name with 1 to 64 characters")
        if name.casefold() in {value.casefold() for value in self.entries().values()}:
            raise ValueError("A layout already uses this name")
        self.save_current()
        layout_id = uuid4().hex
        self.settings.setValue(self._key(layout_id, "name"), name)
        self.active_id = layout_id
        self.save_current()
        self.refresh_menu()
        return layout_id

    def _ask_save_as(self):
        name, accepted = QInputDialog.getText(self.window, "Save layout as", "Layout name")
        if accepted:
            try:
                self.save_as(name)
            except ValueError as exc:
                self.window.status_label.setText(str(exc))

    def _select_from_menu(self, layout_id):
        try:
            self.select(layout_id)
        except ValueError as exc:
            self.window.status_label.setText(str(exc))

    def refresh_menu(self):
        self.menu.clear()
        if self.action_group is not None:
            self.action_group.deleteLater()
        self.action_group = QActionGroup(self)
        for key, name in sorted(self.entries().items(), key=lambda item: (item[0] != "default", item[1].casefold())):
            action = self.menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(key == self.active_id)
            self.action_group.addAction(action)
            action.triggered.connect(lambda checked=False, k=key: self._select_from_menu(k))
        self.menu.addSeparator()
        self.menu.addAction("Save current layout", self.save_current)
        self.menu.addAction("Save layout as...", self._ask_save_as)

    def reset_current(self):
        self._apply(deepcopy(self.defaults))
        self.schedule_save()

    def close(self):
        self.save_current()
        self.restoring = True
        self.save_timer.stop()
        self.restore_timer.stop()
