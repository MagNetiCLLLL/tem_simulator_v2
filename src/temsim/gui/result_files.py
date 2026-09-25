"""Result-file actions and transactional presentation of executed calculations.

Opening a result performs file IO and restores recorded inputs and observables.
It never submits transport, applies an optical preset, or replaces the source.
"""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from PySide6.QtCore import QObject, QSignalBlocker
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QFileDialog, QInputDialog, QLabel, QPushButton
from shiboken6 import isValid

from temsim.instrument_snapshot import decode_instrument, encode_instrument
from temsim.particle_section_io import section_archive_summary
from temsim.result_library import ResultLibrary


def format_data_size(value):
    """File-sized units; sub-GiB result packages must not display as zero."""
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024. or unit == "TiB":
            return f"{size:.0f} B" if unit == "B" else f"{size:.2f} {unit}"
        size /= 1024.


class ResultFiles(QObject):
    """Own file intent; the calculation controller owns bounded background IO."""

    def __init__(self, window):
        super().__init__(window)
        from temsim.gui import calculation_controller
        self.window = window
        self.library = None
        self.startup_error = None
        try:
            self.library = ResultLibrary(calculation_controller.default_artifact_cache_root().parent / "saved_results")
            self.startup_entry = self.library.startup_entry()
        except (OSError, ValueError) as exc:
            self.startup_entry = None
            self.startup_error = str(exc)
        self.hold_automatic_preview = self.startup_entry is not None or self.startup_error is not None
        self.loading = self.hold_automatic_preview
        self._loading_token = None
        self._installed_load_token = None
        self._loading_path = None
        self._expected_entry = None
        self._saving = {}
        self._disabled_widgets = []
        self._closed = False

    def install_actions(self, menu):
        window = self.window
        self.open_action = QAction("Open result…", window)
        self.open_action.setObjectName("openCalculatedResult")
        self.open_action.setShortcut("Ctrl+O")
        self.open_action.triggered.connect(self.open_dialog)
        self.export_action = QAction("Export result…", window)
        self.export_action.setObjectName("exportCalculatedResult")
        self.export_action.setShortcut("Ctrl+S")
        self.export_action.setToolTip(
            "Save the displayed result and its calculated settings, with lossless compression and exact continuation state.")
        self.export_action.triggered.connect(self.export_dialog)
        self.named_action = QAction("Save current as named result…", window)
        self.named_action.setObjectName("saveNamedResult")
        self.named_action.triggered.connect(self.save_named_dialog)
        for action in (self.open_action, self.export_action, self.named_action):
            menu.addAction(action)
        window.open_result_action = self.open_action
        window.export_result_action = self.export_action
        self.saved_menu = menu.addMenu("Open saved results")
        self.startup_menu = menu.addMenu("Startup result")
        self.startup_actions = QActionGroup(self)
        self.startup_actions.setExclusive(True)
        menu.aboutToShow.connect(self.refresh_menus)
        self.refresh_menus()

    def start(self):
        """Run after all controls exist, while initial calculations are held."""
        if self._closed:
            return
        if self.startup_error is not None:
            self.loading = False
            self._error("Saved result library could not be read: " + self.startup_error)
        elif self.startup_entry is not None:
            self.load(self.startup_entry["path"], entry=self.startup_entry)

    def _conflicting_work(self):
        w = self.window
        return bool(w.calculations.has_pending_requests or w.workspace.interactive_calculation.busy
                    or w.design_sweeps.running or w._direct_alignment_state_token is not None
                    or w._preset_state_token is not None)

    def refresh_actions(self, *_):
        """Update availability on live frames without file IO or menu rebuilds."""
        if not hasattr(self, "open_action") or self._closed:
            return
        conflict = self._conflicting_work()
        self.open_action.setEnabled(not conflict and not self._saving)
        result = self.window.workspace._last_result
        available = (result is not None and getattr(getattr(result, "simulation", None), "section_checkpoint", None)
                     is not None and getattr(result, "wave_imaging", None) is None)
        self.export_action.setEnabled(available and not conflict and not self.loading and not self._saving)
        self.named_action.setEnabled(self.export_action.isEnabled() and self.startup_error is None)
        self.saved_menu.setEnabled(not conflict and not self._saving)
        self.startup_menu.setEnabled(not conflict and not self.loading and not self._saving)

    def refresh_menus(self):
        """Read the library only for explicit menu use or a library change."""
        if not hasattr(self, "open_action") or self._closed:
            return
        self.refresh_actions()
        self.saved_menu.clear()
        self.startup_menu.clear()
        for action in self.startup_actions.actions():
            self.startup_actions.removeAction(action)
        try:
            if self.library is None:
                raise ValueError(self.startup_error or "Saved-result library is unavailable")
            entries = self.library.entries()
            selected = self.library.startup_entry()
        except (OSError, ValueError) as exc:
            self.startup_error = str(exc)
            self.named_action.setEnabled(False)
            for submenu in (self.saved_menu, self.startup_menu):
                submenu.addAction("Library unavailable — index was not changed").setEnabled(False)
            return
        none = self.startup_menu.addAction("None — start with standard settings")
        none.setCheckable(True)
        none.setChecked(selected is None)
        self.startup_actions.addAction(none)
        none.triggered.connect(lambda: self.set_startup(None))
        if not entries:
            self.saved_menu.addAction("No saved results yet").setEnabled(False)
        for entry in entries:
            label = f"{entry['name']} | {entry['quality']} | Z {entry['target_z_mm']:.9g} mm"
            action = self.saved_menu.addAction(label)
            action.setToolTip(entry["path"])
            action.triggered.connect(lambda _checked=False, item=entry: self.load(item["path"], entry=item))
            action = self.startup_menu.addAction(entry["name"])
            action.setCheckable(True)
            action.setChecked(selected is not None and selected["id"] == entry["id"])
            self.startup_actions.addAction(action)
            action.triggered.connect(lambda _checked=False, identity=entry["id"]: self.set_startup(identity))

    def set_startup(self, identity):
        try:
            self.library.set_startup(identity)
            self.refresh_menus()
            self.window.status_label.setText(
                "Startup result saved; current settings and calculations are unchanged."
                if identity is not None else "Startup result cleared; current settings are unchanged.")
        except (OSError, ValueError) as exc:
            self._error(str(exc))

    def open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self.window, "Open calculated result", "",
            "Calculated results (*.temresult *.temsection)")
        if path:
            self.load(path)

    def export_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self.window,
            "Export displayed result with its calculated settings", "result.temresult",
            "Calculated result (*.temresult)")
        if path:
            if not Path(path).suffix:
                path += ".temresult"
            self.save(path)

    def save_named_dialog(self):
        name, accepted = QInputDialog.getText(self.window, "Save named result", "Name")
        if accepted and name.strip():
            try:
                self.save(self.library.allocate_path(), name=name.strip())
            except (OSError, ValueError) as exc:
                self._error(str(exc))

    def save(self, path, *, name=None):
        if self.loading or self._saving or self._conflicting_work():
            self._error("Finish the current operation before exporting a result.")
            return
        result = self.window.workspace._last_result
        key = str(Path(path).resolve())
        try:
            summary = section_archive_summary(result)
            self._saving[key] = {"name": name, "identity": summary["identity"], "token": None}
            self.window.workspace.interactive_calculation.expect_section_archive(result)
            self.window.calculations.archive_completed_section(result, path=key)
            self.refresh_actions()
        except (AttributeError, OSError, ValueError, TypeError, RuntimeError) as exc:
            self._saving.pop(key, None)
            self.refresh_actions()
            self._error("Result was not exported: " + str(exc))

    def _stop_automatic_updates(self):
        w = self.window
        w.preview_timer.stop()
        w._interactive_preview_pending = False
        w._interactive_preview_generation = None
        w._preview_deferred_for_interactive = False
        w._preview_deferred_for_sweep = False
        w.workspace.interactive_calculation.pause_live_tuning()

    def _lock_inputs(self):
        if self._disabled_widgets:
            return
        w = self.window
        widgets = [w.instrument_editor, w.workspace, w.workspace.interactive_calculation,
                   w.tuning_quality, w.high_rays,
                   w.high_step, w.compute_backend, w.open_profile_action, w.reload_toml_action,
                   w.simulation_menu]
        widgets += [getattr(w, name, None) for name in (
            "_part_geometry_dialog", "_df_geometry_dialog", "_configuration_dialog", "_cache_settings_dialog")]
        widgets += [w.findChild(QPushButton, name) for name in
                    ("previewButton", "highAccuracyButton", "calculateSetupButton")]
        for widget in widgets:
            if widget is not None:
                self._disabled_widgets.append((widget, widget.isEnabled()))
                widget.setEnabled(False)

    def _unlock_inputs(self):
        for widget, enabled in self._disabled_widgets:
            if isValid(widget):
                widget.setEnabled(enabled)
        self._disabled_widgets.clear()

    def load(self, path, *, entry=None):
        if self._closed:
            return
        if self._saving or self._conflicting_work():
            self._error("Finish or cancel the current calculation before opening a result.")
            return
        self.loading = self.hold_automatic_preview = True
        self._loading_token = None
        self._installed_load_token = None
        self._loading_path = str(Path(path).resolve())
        self._expected_entry = entry
        self._stop_automatic_updates()
        self.window.calculations.invalidate_pending(include_explicit=True)
        self._lock_inputs()
        try:
            self._loading_token = self.window.calculations.load_section_archive(self._loading_path)
            self.refresh_actions()
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            self._load_failed(str(exc))

    def archive_status(self, info):
        """Match events to their requested file, never merely a result identity."""
        if self._closed:
            return
        status, token = info.get("status"), info.get("operation_token")
        if status == "loading" and info.get("path") == self._loading_path:
            self._loading_token = token
            self.window.status_label.setText("Opening saved result — no calculation is running.")
        if status == "failed" and info.get("operation") == "load":
            if self.loading and token == self._loading_token:
                self._load_failed(str(info.get("error", "Unknown file error")))
            return
        path = info.get("path")
        pending = self._saving.get(str(Path(path).resolve())) if path else None
        if pending is None:
            return
        if info.get("identity") != pending["identity"]:
            if status in {"saved", "failed"} and token == pending["token"] and token is not None:
                self._saving.pop(str(Path(path).resolve()))
                self.refresh_actions()
                self._error("Result export returned a different calculation identity; the saved-results list was not changed.")
            return
        if status == "saving":
            pending["token"] = token
            self.window.status_label.setText("Exporting displayed result and calculated settings — lossless compression.")
            return
        if token != pending["token"] or status not in {"saved", "failed"}:
            return
        self._saving.pop(str(Path(path).resolve()))
        if status == "failed":
            self._error("Result export failed: " + str(info.get("error", "Unknown file error")))
        else:
            try:
                if pending["name"] is not None:
                    self.library.remember(pending["name"], info)
                    self.refresh_menus()
                self.window.status_label.setText("Exported calculated settings and result. " + self._description(info))
            except (OSError, ValueError) as exc:
                self._error(f"Result file saved to {path}, but the saved-results list was not changed: {exc}")
        self.refresh_actions()

    def presentation_status(self, info):
        """A worker finishing IO is not proof that the GUI accepted its result."""
        if self._closed:
            return None
        status = info.get("status")
        token = info.get("operation_token")
        if status == "loaded":
            return (dict(info, restored=True) if token == self._installed_load_token
                    and token is not None else None)
        if status == "loading":
            return info if self.loading and info.get("path") == self._loading_path else None
        if status == "failed" and info.get("operation") == "load":
            return info if self.loading and token == self._loading_token else None
        return info

    @staticmethod
    def _description(info):
        text = (f"{info['quality']} | {int(info.get('particle_count', 0)):,} particles | "
                f"calculated Z {info['target_z_mm']:.9g} mm | "
                f"resumable Z {info['resumable_through_z_mm']:.9g} mm")
        if info.get("compressed_size_bytes") is not None:
            text += " | file " + format_data_size(info["compressed_size_bytes"])
        if info.get("unpacked_size_bytes") is not None:
            text += " | unpacked data " + format_data_size(info["unpacked_size_bytes"])
        return text

    def loaded(self, result, info):
        if self._closed or not self.loading or info.get("operation_token") != self._loading_token:
            return
        try:
            expected = self._expected_entry
            if expected is not None and (expected["result_identity"] != info.get("identity")
                    or expected["package_digest"] != info.get("_package_digest")):
                raise ValueError("The saved-result file was replaced; open the file explicitly to inspect it.")
            self._install_result(result, info)
        except (OSError, ValueError, TypeError, RuntimeError, AttributeError) as exc:
            self._load_failed(str(exc))
            return
        self.loading = False
        self._installed_load_token = info["operation_token"]
        self.window._invalidate_direct_alignment()
        self._stop_automatic_updates()
        self._unlock_inputs()
        self.refresh_actions()
        self.window.status_label.setText("Loaded result; no recalculation. " + self._description(info))
        self.window.log_output.appendPlainText(f"Loaded calculated result: {info['path']}")

    def _install_result(self, result, info):
        w = self.window
        state = decode_instrument(encode_instrument(result.state_snapshot))
        workspace = w.workspace
        page = workspace.interactive_calculation
        previous = {
            "state": w.state, "checkpoint": w._active_working_checkpoint,
            "parent": getattr(w, "_working_point_parent", None),
            "workspace": {key: getattr(workspace, key) for key in (
                "_last_result", "_last_quality", "_preview_result", "_high_accuracy_result",
                "_high_accuracy_current", "_sample_region_result", "_ray_extent_stale", "_selected_z_mm")},
            "rays": (w.high_rays.minimum(), w.high_rays.maximum(), w.high_rays.value()),
            "step": (w.high_step.decimals(), w.high_step.minimum(), w.high_step.maximum(), w.high_step.value()),
            "quality": w.tuning_quality.currentIndex(),
            "page": page.capture_result_presentation(),
            "labels": [(label, label.text(), label.toolTip(), label.styleSheet())
                       for label in w.findChildren(QLabel)],
            "readout": (dict(workspace.result_readout._records), dict(workspace.result_readout._stale)),
            "model_status": (workspace.physical_layout.model_editor._calculation_status,
                             workspace.physical_layout.model_editor._calculation_detail),
        }
        try:
            w._install_working_point(state, None)
            # Display the exact executed numerical controls without rounding
            # them back into the captured input state or emitting a new edit.
            with QSignalBlocker(w.high_rays), QSignalBlocker(w.high_step), QSignalBlocker(w.tuning_quality):
                rays = int(state.electron_gun.ray_count)
                step = float(state.step_mm)
                w.high_rays.setRange(min(w.high_rays.minimum(), rays), max(w.high_rays.maximum(), rays))
                w.high_rays.setValue(rays)
                w.high_step.setDecimals(max(4, -Decimal(str(step)).as_tuple().exponent))
                w.high_step.setRange(min(w.high_step.minimum(), step), max(w.high_step.maximum(), step))
                w.high_step.setValue(step)
                index = w.tuning_quality.findData(info["quality"])
                w.tuning_quality.setCurrentIndex(index if index >= 0 else w.tuning_quality.findData("Preview"))
            workspace.clear_result()
            workspace.display_result(result, info["quality"])
            page.restore_calculated_result(result, info)
            workspace.model_inspector.display_result(result)
            workspace.vacuum_map.set_result(result)
            workspace.physical_layout.model_editor.set_calculation_status("current", "Loaded calculated settings and completed result; no recalculation.")
            w.assembly_panel.update_direct_alignment_metrics(result.simulation.metrics)
            w.workspace.jump_to_ray_position(float(info["target_z_mm"]), activate_tab=False)
            w._schedule_design_explorer_refresh()
        except Exception:
            w._install_working_point(previous["state"], previous["checkpoint"])
            w._working_point_parent = previous["parent"]
            with QSignalBlocker(w.high_rays), QSignalBlocker(w.high_step), QSignalBlocker(w.tuning_quality):
                lo, hi, value = previous["rays"]
                w.high_rays.setRange(lo, hi)
                w.high_rays.setValue(value)
                decimals, lo, hi, value = previous["step"]
                w.high_step.setDecimals(decimals)
                w.high_step.setRange(lo, hi)
                w.high_step.setValue(value)
                w.tuning_quality.setCurrentIndex(previous["quality"])
            old = previous["workspace"]
            old_result, old_high = old["_last_result"], old["_high_accuracy_result"]
            workspace.clear_result()
            if old_high is not None and old_high is not old_result:
                workspace.display_result(old_high, "High accuracy")
            if old_result is not None:
                workspace.display_result(old_result, old["_last_quality"])
                workspace.model_inspector.display_result(old_result)
                workspace.vacuum_map.set_result(old_result)
                w.assembly_panel.update_direct_alignment_metrics(old_result.simulation.metrics)
            for key, value in old.items():
                setattr(workspace, key, value)
            page.restore_result_presentation(previous["page"])
            workspace.result_readout._records, workspace.result_readout._stale = previous["readout"]
            workspace.result_readout._refresh()
            workspace._refresh_ray_calculation_extent()
            workspace.physical_layout.model_editor.set_calculation_status(*previous["model_status"])
            for label, text, tooltip, style in previous["labels"]:
                if not isValid(label):  # Assembly controls are rebuilt from the restored state.
                    continue
                label.setText(text)
                label.setToolTip(tooltip)
                label.setStyleSheet(style)
            raise

    def _load_failed(self, message):
        self.loading = False
        self._stop_automatic_updates()
        self._unlock_inputs()
        self.refresh_actions()
        self._error("Result was not opened; previous settings and result are retained. " + message)

    def _error(self, message):
        self.window._show_error(message)

    def close(self):
        self._closed = True
        self.hold_automatic_preview = True
        self.window.preview_timer.stop()
