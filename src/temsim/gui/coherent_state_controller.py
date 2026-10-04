"""Named, explicitly captured tip states in one set of installed optics.

This is a comparison/readout owner, not a second source or propagation engine.
Every entry is a complete capture of an applied physical tip. Calculate rebases
those declared emission inputs onto current optics, then uses the normal wave
session. Raw complex results stay separate; only intensities are combined.
"""
from collections import OrderedDict
from dataclasses import asdict, dataclass, field, replace
from threading import Event
from uuid import uuid4

from PySide6.QtCore import QObject, QTimer, Slot

from temsim.gui.job_coordinator import CoordinatedPool
from temsim.immutable_json import json_digest
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.physics.coherent_inputs import source_settings_from_state, wave_input_summary
from temsim.physics.coherent_state_set import (
    combine_intensity_previews, rebase_tip_state,
)


@dataclass
class _Run:
    state: object
    request: object
    observer: object
    instrument_identity: str
    cache: OrderedDict = field(default_factory=OrderedDict)


class CoherentStateController(QObject):
    """One running job and one latest target, across all listed initial states."""

    def __init__(self, page, controls):
        super().__init__(page)
        self.page, self.controls = page, controls
        self.snapshots = {}
        self.runs = {}
        self.row_keys = {}
        self.failed = {}
        self.worker = None
        self._job_key = None
        self.generation = 0
        self.active = False
        self.admitted_ids = set()
        self.closed = False
        self.single_display = None
        self._display_mode = controls.display_mode()
        self.pool = CoordinatedPool(self)
        self.pool.setMaxThreadCount(1)
        self.pool.coordinator.register_retained(self, "retained_roots")
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._advance)
        controls.add_requested.connect(self.add_current)
        controls.replace_requested.connect(self.replace_current)
        controls.remove_requested.connect(self.remove)
        controls.restore_requested.connect(self.load_tip)
        controls.display_changed.connect(self.display_changed)

    def retained_roots(self):
        return self.snapshots, self.runs, self.single_display

    def viewing_states(self):
        return self.controls.display_mode() != "current"

    def _state(self):
        return self.page.state_provider() if self.page.state_provider else self.page._state

    def _message(self, text):
        self.page.status.setText(text)

    def _capture(self):
        # The existing guard rejects unapplied source drafts. Capturing a row
        # must not secretly apply them, even when no propagation is requested.
        state, _, _ = self.page.capture_calculation_inputs()
        return capture_instrument_snapshot(state)

    @staticmethod
    def _summary(snapshot):
        settings = source_settings_from_state(snapshot.restore())
        if settings.surface_mean_energy_ev is not None:
            return (f"Tip cap | E {settings.surface_mean_energy_ev:.6g} eV | "
                    f"RMS {settings.surface_energy_rms_ev:.6g} eV")
        return (f"E {settings.tip_mean_energy_ev:.6g} eV | width {settings.tip_energy_spread_fwhm_ev:.6g} eV FWHM | "
                f"X/Y {settings.tip_offset_x_nm:.5g}, {settings.tip_offset_y_nm:.5g} nm | "
                f"tilt {settings.tip_tilt_x_mrad:.5g}, {settings.tip_tilt_y_mrad:.5g} mrad")

    @Slot()
    def add_current(self):
        try:
            snapshot = self._capture()
            summary = self._summary(snapshot)
        except Exception as error:
            self._message(f"Tip state not added: {error}")
            return
        identity = uuid4().hex
        self.snapshots[identity] = snapshot
        self.controls.add_entry(identity, f"State {len(self.snapshots)}", summary)
        self._message("Applied tip state added. Edit and Apply tip parameters to define another state, then Add or Replace. Calculate beam uses current shared optics.")

    @Slot(str)
    def replace_current(self, identity):
        if identity not in self.snapshots:
            return
        try:
            snapshot = self._capture()
            summary = self._summary(snapshot)
        except Exception as error:
            self._message(f"Tip state not replaced: {error}")
            return
        self.cancel()
        self.snapshots[identity] = snapshot
        self.row_keys.pop(identity, None)
        self.controls.set_summary(identity, summary)
        self.controls.set_status(identity, "Inputs replaced; calculate")
        self._drop_unused_runs()
        self._message("Selected state replaced with the applied tip. Other saved emission states are unchanged.")

    @Slot(str)
    def remove(self, identity):
        if identity not in self.snapshots:
            return
        self.cancel()
        self.snapshots.pop(identity)
        self.row_keys.pop(identity, None)
        self.controls.remove_entry(identity)
        self._drop_unused_runs()

    @Slot(str)
    def load_tip(self, identity):
        """An explicit user action publishes through the existing shared editor."""
        if identity not in self.snapshots:
            return
        try:
            state = self._state()
            candidate = rebase_tip_state(self.snapshots[identity], state)
            settings = source_settings_from_state(candidate)
            if self.page.source_applier is not None:
                state = self.page.source_applier(settings)
            else:
                emitter = state.electron_gun.emitter
                emitter.__dict__.clear()
                emitter.__dict__.update(candidate.electron_gun.emitter.__dict__)
                state.electron_gun.source_representation = candidate.electron_gun.source_representation
                state.electron_gun._trace_cache = state.electron_gun._trace_cache_key = None
            self.page.set_state(state)
            self.page.source_applied.emit(state)
            self._message("Selected tip state applied to the shared instrument. Both particle rays and coherent waves now read it; no calculation started.")
        except Exception as error:
            self._message(f"Saved tip state not applied: {error}")

    def _entries(self):
        entries = self.controls.entries()
        if self.controls.display_mode() == "selected":
            return tuple(entry for entry in entries if entry.id == self.controls.selected_id())
        return tuple(entry for entry in entries if entry.checked and entry.weight > 0.)

    @Slot()
    def calculate(self):
        if self.closed:
            return
        if not self.viewing_states():
            self.controls.mode.setCurrentIndex(self.controls.mode.findData("overlay"))
        self.page._cancel_request()
        self.page.invalidate_pair("Saved emission states. No matching classical ensemble is claimed. Apply selected state, choose Current tip parameters, then enable Compare classical rays and Calculate beam for a same-state comparison.")
        self.cancel()
        entries = self._entries()
        if not entries:
            self._message("Choose a state, or check states with positive weights, before Calculate beam.")
            return
        try:
            state = self._state()
            request = self.page._make_request()
            request_key = json_digest(asdict(replace(request, observation_z_mm=0.)))
            prepared = {}
            for entry in entries:
                captured = rebase_tip_state(self.snapshots[entry.id], state)
                snapshot = capture_instrument_snapshot(captured)
                key = (snapshot.physical_digest, request_key)
                if key not in self.runs:
                    summary = wave_input_summary(captured, request)
                    if summary["status"] != "SOURCE_READY":
                        raise ValueError(f"{entry.name}: {summary.get('reason', 'Source is not ready')}")
                    if (summary.get("minimum_initial_wave_bytes") or 0) > request.wave_grid.maximum_working_bytes:
                        raise MemoryError(f"{entry.name}: complete source modes exceed the working budget")
                prepared[entry.id] = (key, captured, snapshot.digest)
            # Admit the entire selection before submitting any job. A rejected
            # member is never silently removed from the requested population.
            from temsim.gui.coherent_beam import TipWaveObservationSession
            for identity, (key, captured, instrument_identity) in prepared.items():
                if key not in self.runs:
                    run = _Run(captured, request, TipWaveObservationSession(captured, request), instrument_identity)
                    # A completed current-tip session is the same calculation
                    # when both physical and numerical identities match.
                    page = self.page
                    if page._captured is not None and page._request is not None and page._observation_session is not None:
                        single_key = (capture_instrument_snapshot(page._captured).physical_digest,
                                      json_digest(asdict(replace(page._request, observation_z_mm=0.))))
                        if key == single_key:
                            run = _Run(page._captured, request, page._observation_session,
                                       capture_instrument_snapshot(page._captured).digest,
                                       OrderedDict(page._cache))
                    self.runs[key] = run
                self.row_keys[identity] = key
                self.controls.set_status(identity, "Queued")
        except Exception as error:
            self._message(f"State calculation not started: {error}. Previous complete image retained.")
            return
        self.failed.clear()
        self.admitted_ids = {entry.id for entry in entries}
        self.active = True
        self._drop_unused_runs()
        self._advance()

    def _drop_unused_runs(self):
        used = set(self.row_keys.values())
        self.runs = {key: run for key, run in self.runs.items() if key in used}

    @Slot()
    def display_changed(self):
        if self.closed:
            return
        mode = self.controls.display_mode()
        previous_mode = self._display_mode
        self._display_mode = mode
        if mode == "current" and previous_mode == "current":
            # Names/weights/checkboxes cannot affect the current-tip readout or
            # a pending paired-particle calculation owned by MainWindow.
            return
        if not self.viewing_states():
            self.cancel()
            if self.single_display is not None:
                self.page._display(*self.single_display)
                if self.page._stale:
                    self._message("Previous current-tip result displayed; inputs changed. Calculate beam to update.")
                else:
                    # The shared Z may have moved while viewing the saved-state
                    # ensemble. Restore/query the current session at that Z,
                    # rather than promising an update with no queued request.
                    self.page._z_changed(self.page._target_z_mm)
            else:
                self._message("Current tip selected. Calculate beam to show it; the previous saved-state image is retained.")
            return
        self.page._cancel_request()
        self.page.invalidate_pair("Saved tip states: intensities are added, with separate phase references. No matched particle ensemble is claimed.")
        if not self._publish():
            self._message("Selected states are not all calculated at the requested Z. Click Calculate beam; the previous complete image is retained.")

    def set_z(self):
        if self._publish():
            return
        if self.active and self.worker is None and not self.timer.isActive():
            self.timer.start(self.page.QUERY_INTERVAL_MS)
        elif not self.active:
            self._message("Click Calculate beam to start at this Z. Previous complete image retained.")

    def _publish(self):
        entries = self._entries()
        z_mm = self.page._target_z_mm
        items = []
        for entry in entries:
            if entry.id not in self.admitted_ids:
                return False
            run = self.runs.get(self.row_keys.get(entry.id))
            if run is None or z_mm not in run.cache:
                return False
            result, preview = run.cache[z_mm]
            run.cache.move_to_end(z_mm)
            # Selected state is viewed at unit probability, even if its overlay
            # weight was set to zero. Weights matter only for an overlay.
            items.append((result, preview, entry.weight if self.controls.display_mode() == "overlay" else 1.))
        if not items:
            return False
        result, preview = combine_intensity_previews(items)
        self.page.result, self.page.preview = result, preview
        self.page._update_plane_positions()
        self.page._refresh_intensity()
        names = ", ".join(entry.name for entry in entries)
        self._message(f"Completed {len(entries)} tip state(s) at Z {z_mm:.9g} mm: {names}.")
        self.page.readout.setText(
            f"{len(entries)} independent tip states | Density / µm² per electron | "
            f"displayed probability {preview.probability:.8g} | relative weights normalized over shown states. "
            "Each state's complex fields and phase references are retained; there is no aggregate phase. "
            "Changing names, weights or visibility does not propagate waves.")
        return True

    @Slot()
    def _advance(self):
        if self.closed or not self.active or not self.viewing_states() or self.worker is not None:
            return
        if self._publish():
            return
        z_mm = self.page._target_z_mm
        for entry in self._entries():
            if entry.id not in self.admitted_ids:
                self._message("A selected state needs Calculate beam before Z browsing.")
                return
            key = self.row_keys.get(entry.id)
            run = self.runs.get(key)
            if run is None:
                self._message("A selected state needs Calculate beam before Z browsing.")
                return
            if z_mm in run.cache:
                continue
            if (key, z_mm) in self.failed:
                self._message(f"State {entry.name} failed at Z {z_mm:.9g} mm: {self.failed[key, z_mm]}. No partial overlay was published. Click Calculate beam to retry.")
                return
            from temsim.gui.coherent_beam import _WaveWorker
            self.generation += 1
            worker = _WaveWorker(self.generation, self.generation, run.state,
                replace(run.request, observation_z_mm=z_mm), Event(),
                (self.retained_roots(), self.page.result, self.page.preview), observer=run.observer)
            worker.signals.solved.connect(self._solved)
            worker.signals.failed.connect(self._failed)
            worker.signals.progress.connect(self._progress)
            worker.signals.finished.connect(self._finished)
            self.worker, self._job_key = worker, key
            self.controls.set_status(entry.id, f"Calculating Z {z_mm:.7g} mm")
            self.page.cancel_button.setEnabled(True)
            self.page._update_plane_positions()
            self.pool.start(worker)
            return

    @Slot(int, int, object, object)
    def _solved(self, generation, session, result, preview):
        if self.closed or generation != self.generation or session != generation or self.worker is None:
            return
        z_mm = float(result.checkpoint.plane_z_mm)
        if z_mm != self.worker.request.observation_z_mm:
            self._failed(generation, "Executed Z differs from the submitted plane")
            return
        run = self.runs[self._job_key]
        if result.instrument_digest != run.instrument_identity:
            self._failed(generation, "Executed source/optics identity differs from the saved state")
            return
        run.cache[z_mm] = (result, preview)
        run.cache.move_to_end(z_mm)
        for identity, key in self.row_keys.items():
            if key == self._job_key:
                self.controls.set_status(identity, f"Ready Z {z_mm:.7g} mm")
        self._trim_cache()
        self._publish()

    def _trim_cache(self):
        # One shared display-cache budget across the list, not one budget per
        # electron. The coordinator additionally inventories all raw sessions.
        budget = round(self.page.ram_cache_gib.value()*1024**3)
        def size():
            return sum(p.retained_bytes for run in self.runs.values() for _, p in run.cache.values())
        def count():
            return sum(len(run.cache) for run in self.runs.values())
        while count() > self.page.CACHE_LIMIT or size() > budget:
            evicted = False
            for run in self.runs.values():
                for z_mm in tuple(run.cache):
                    if z_mm != self.page._target_z_mm:
                        del run.cache[z_mm]
                        evicted = True
                        break
                if evicted:
                    break
            if not evicted:
                break  # Current complete population is accounted by admission.

    @Slot(int, str)
    def _failed(self, generation, message):
        if generation != self.generation or self.worker is None:
            return
        z_mm = self.worker.request.observation_z_mm
        self.failed[self._job_key, z_mm] = message
        for identity, key in self.row_keys.items():
            if key == self._job_key:
                self.controls.set_status(identity, f"Failed Z {z_mm:.7g} mm: {message}")
        self._message(f"Saved-state calculation failed at Z {z_mm:.9g} mm: {message}. Previous complete image retained; no partial overlay.")

    @Slot(int, str)
    def _progress(self, generation, message):
        if generation == self.generation and self.worker is not None:
            self._message(f"Saved tip states | calculating Z {self.worker.request.observation_z_mm:.9g} mm | {message}")

    @Slot(int)
    def _finished(self, generation):
        if generation != self.generation:
            return
        self.worker = None
        self.page._update_cancel_enabled()
        self.page._update_plane_positions()
        if self.active:
            self.timer.start(0)

    def cancel(self):
        self.active = False
        self.generation += 1
        self.timer.stop()
        if self.worker is not None:
            self.worker.event.set()
        self.pool.clear()
        self.worker = None
        self.page._update_cancel_enabled()

    def inputs_changed(self):
        self.cancel()
        # Rows are input records, not executed downstream sources. Cached runs
        # remain keyed by exact inputs but must be re-admitted by Calculate.
        self.admitted_ids.clear()
        self.single_display = None
        for entry in self.controls.entries():
            self.controls.set_status(entry.id, "Inputs changed; calculate")

    def shutdown(self, msecs):
        self.closed = True
        self.cancel()
        return self.pool.waitForDone(msecs)
