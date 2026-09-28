"""Transactional diagnostic-session actions, separate from calculation ownership.

Loading admits historical display records only. Field references describe an
executed dependency; they never configure the microscope or a particle source.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QPushButton

from temsim.diagnostic_electron_record import ElectronRecord
from temsim.electron_diagnostic_session import (
    DiagnosticDependency, DiagnosticDisplayState, DiagnosticElectronRecord,
    DiagnosticSession, load_diagnostic_session, save_diagnostic_session,
)


def scene_dependency(scene, *, result_reference=None):
    bounds = getattr(scene, "diagnostic_bounds_m", None)
    if bounds is None:
        bounds = getattr(scene, "bounds_m", None)
    return DiagnosticDependency(
        result_reference=(result_reference or ("captured-fields:" + str(scene.transport_identity)
                          if getattr(scene, "transport_identity", None) else None)),
        physical_identity=getattr(scene, "physical_identity", None),
        numerical_identity=getattr(scene, "numerical_identity", None),
        transport_identity=getattr(scene, "transport_identity", None),
        provider_notes=tuple(getattr(scene, "notes", ())),
        bounds_m=None if bounds is None else tuple(tuple(float(v) for v in row) for row in bounds),
    )


class ElectronSessionActions:
    """Own file presentation and atomic record replacement, never integration."""

    def __init__(self, controller, layout):
        self.controller = controller
        self.capture_view = lambda: DiagnosticDisplayState()
        self.restore_view = lambda display: None
        row = QHBoxLayout()
        self.save_button = QPushButton("Save session…")
        self.save_button.setObjectName("testElectronSaveSession")
        self.load_button = QPushButton("Open session…")
        self.load_button.setObjectName("testElectronLoadSession")
        self.recalculate_button = QPushButton("Recalculate in current fields")
        self.recalculate_button.setObjectName("testElectronRecalculateSession")
        self.recalculate_button.setToolTip(
            "Calculate the selected historical electron as a new execution in the current captured fields. "
            "Loading alone never recalculates or changes the microscope.")
        self.save_button.clicked.connect(self.choose_save)
        self.load_button.clicked.connect(self.choose_load)
        self.recalculate_button.clicked.connect(controller.recalculate_historical)
        row.addWidget(self.save_button)
        row.addWidget(self.load_button)
        layout.addLayout(row)
        layout.addWidget(self.recalculate_button)

    def refresh(self):
        c = self.controller
        record = c.selected_record
        self.save_button.setEnabled(bool(c.records) and not c._shutdown)
        self.load_button.setEnabled(not c._shutdown)
        self.recalculate_button.setEnabled(
            not c._shutdown and record is not None and record.historical
            and (c._scene is not None or c._scene_request is not None))

    def snapshot(self):
        c = self.controller
        dependency = scene_dependency(c._scene, result_reference=c._result_reference)
        records = []
        for record in c.records:
            result, settings = record.trajectory, record.trajectory_settings
            state = "uncomputed"
            if result is not None:
                state = ("completed" if result.completed else "incomplete")
                if settings != record.settings or record.historical:
                    state = "previous"
            elif record.progress_trajectory is not None and record.progress_key is not None:
                result, settings = record.progress_trajectory, record.progress_key[1]
                state = "incomplete" if settings == record.settings else "previous"
            elif record.error:
                state = "failed"
            records.append(DiagnosticElectronRecord(
                key=record.key, label=record.label, colour=record.colour,
                settings=record.settings, checked=record.checked, trajectory=result,
                trajectory_settings=settings, revision=record.revision, state=state,
                error=record.error, dependency=record.history_dependency or dependency))
        display = replace(self.capture_view(),
            overlay=c.display_mode.currentData() == "overlay", selected_key=c._selected_key,
            show_background=c.background.isChecked())
        return DiagnosticSession(tuple(records), dependency=dependency, display=display)

    def save(self, destination):
        # Snapshot on the GUI thread: no worker can splice new controls into an
        # older result during synchronous validation/atomic serialization.
        return save_diagnostic_session(self.snapshot(), destination)

    def load(self, source):
        session = load_diagnostic_session(source)
        c = self.controller
        # Complete validation and detached record construction before any state
        # change or cancellation. Corrupt archives leave the live session intact.
        records = OrderedDict()
        for stored in session.records:
            settings = stored.settings
            values = ((c.energy, settings.kinetic_energy_ev),
                      (c.x, settings.position_m[0]*1e6), (c.y, settings.position_m[1]*1e6),
                      (c.z, settings.position_m[2]*1e3),
                      (c.polar, settings.polar_angle_deg*3.141592653589793/180.*1000.),
                      (c.azimuth, settings.azimuth_angle_deg),
                      (c.length, settings.max_path_length_m*1000.), (c.step, settings.step_m*1000.),
                      (c.max_steps, settings.max_steps), (c.relative_tolerance, settings.relative_tolerance),
                      (c.position_tolerance, settings.position_tolerance_m*1e9))
            if any(not control.minimum() <= value <= control.maximum() for control, value in values):
                raise ValueError(f"{stored.label}: saved settings exceed this editor's supported range")
            records[stored.key] = ElectronRecord(
                key=stored.key, label=stored.label, colour=stored.colour, settings=settings,
                checked=stored.checked, trajectory=stored.trajectory, revision=stored.revision,
                attempted=True, error=stored.error, trajectory_settings=stored.trajectory_settings,
                historical=True, history_dependency=stored.dependency or session.dependency)
        # Display restoration is also preflighted before replacing records.
        # Roll back view changes on a presentation failure where possible.
        old_view = self.capture_view()
        try:
            self.restore_view(session.display)
        except Exception:
            try:
                self.restore_view(old_view)
            except Exception:
                pass
            raise
        c._generation += 1
        c._timer.stop()
        for worker in (c._worker, c._scene_worker):
            if worker is not None:
                worker.cancelled.set()
        c._cache.clear()
        c._editing_controls.clear()
        c._records = records
        c._created_initial_record = True
        c._selected_key = session.display.selected_key or next(iter(records), None)
        c._next_id = max([0]+[int(key[9:]) for key in records
                              if key.startswith("electron-") and key[9:].isdigit()])+1
        with QSignalBlocker(c.display_mode), QSignalBlocker(c.background):
            c.display_mode.setCurrentIndex(c.display_mode.findData("overlay" if session.display.overlay else "selected"))
            c.background.setChecked(session.display.show_background)
        c._load_editor()
        c._refresh()
        c.background_changed.emit(c.background.isChecked())
        return session

    def choose_save(self):
        c = self.controller
        path, _ = QFileDialog.getSaveFileName(c.panel, "Save virtual electron session", "electrons.temdiag",
                                             "Virtual electron diagnostic session (*.temdiag)")
        if path:
            try:
                self.save(path)
            except (OSError, ValueError, TypeError) as exc:
                c._status(f"Diagnostic session was not saved: {exc}")
            else:
                c._status(f"Diagnostic session saved: {Path(path)}")

    def choose_load(self):
        c = self.controller
        path, _ = QFileDialog.getOpenFileName(c.panel, "Open virtual electron session", "",
                                             "Virtual electron diagnostic session (*.temdiag)")
        if path:
            try:
                self.load(path)
            except (OSError, ValueError, TypeError) as exc:
                c._status(f"Diagnostic session was not loaded; existing records retained: {exc}")
