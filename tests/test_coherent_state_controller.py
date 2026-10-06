"""Real source ownership and Qt scheduling; expensive wave execution is replaced.

These tests exercise actual instrument snapshots and source admission. Synthetic
screen densities test only publication/cache control, not wave propagation or
the physical validity of a full microscope calculation.
"""
from dataclasses import replace
from threading import Event, Lock
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QThread, Qt
from pytestqt.exceptions import TimeoutError as QtTimeoutError

import temsim.gui.coherent_beam as module
from temsim.gui.coherent_beam import CoherentBeamPage, _Preview
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state
from temsim.physics.coherent_inputs import candidate_tip_emission, source_settings_from_state
from temsim.physics.coherent_state_set import EnsembleObservation


class _Execution:
    def __init__(self):
        self.calls = []
        self.sessions = []
        self.fail = None
        self.block = None
        self.release = Event()
        self.active = 0
        self.peak_active = 0
        self.lock = Lock()

    def solve(self, state, request, **kwargs):
        snapshot = capture_instrument_snapshot(state)
        energy = state.electron_gun.emitter.emission_energy_ev
        z_mm = request.observation_z_mm
        with self.lock:
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
            self.calls.append((energy, z_mm, snapshot,
                               QThread.currentThread() == self.page.thread()))
        try:
            if self.block is not None and self.block(energy, z_mm):
                # Deliberately finish even after cancellation. Worker/controller
                # acceptance, rather than the numerical fake, must reject it.
                assert self.release.wait(5.), "test did not release blocked solver"
            if self.fail is not None and self.fail(energy, z_mm):
                raise ValueError("intentional single-state numerical failure")
            return SimpleNamespace(checkpoint=SimpleNamespace(
                plane_z_mm=z_mm, reference_current_a=state.electron_gun.emitter.emitted_current_a,
                initial_energy_ev=energy), instrument_digest=snapshot.digest)
        finally:
            with self.lock:
                self.active -= 1


@pytest.fixture
def page(qtbot, monkeypatch):
    execution = _Execution()

    class ObservationSession:
        def __init__(self, state, request):
            self.state, self.request = state, request
            execution.sessions.append(self)

        def observe(self, z_mm, **kwargs):
            return module.simulate_tip_wave(self.state,
                replace(self.request, observation_z_mm=z_mm), **kwargs)

    def preview(checkpoint, **_kwargs):
        density = np.zeros((4, 4))
        density[0 if checkpoint.initial_energy_ev == 4. else 3, 1] = 1.
        return _Preview(density, np.array(((0., 4.), (0., 4.))),
                        1., 1, density.nbytes + 32, ((4, 4),))

    monkeypatch.setattr(module, "simulate_tip_wave", execution.solve)
    monkeypatch.setattr(module, "TipWaveObservationSession", ObservationSession)
    monkeypatch.setattr(module, "_intensity_preview", preview)
    view = CoherentBeamPage()
    execution.page = view
    view.execution = execution
    qtbot.addWidget(view)
    state = default_state()
    # Supported mathematical source fixture; deliberately not a startup FEG
    # default or a claim that realistic low-energy gun physics is qualified.
    settings = replace(source_settings_from_state(state), enabled=True,
        tip_fwhm_nm=100., tip_mean_energy_ev=4., tip_minimum_energy_ev=3.,
        tip_energy_spread_fwhm_ev=0., incoherent_angle_rms_mrad=0.,
        tip_curvature_x_m1=0., tip_curvature_xy_m1=0., tip_curvature_y_m1=0.,
        tip_offset_x_nm=0., tip_offset_y_nm=0., tip_tilt_x_mrad=0., tip_tilt_y_mrad=0.)
    state.electron_gun = candidate_tip_emission(state, settings)
    view.set_state(state)
    view.show()
    yield view
    execution.release.set()
    assert view.shutdown(5000)


def _add(page):
    old_count = len(page.state_list.entries())
    page.state_list.add_button.click()
    assert len(page.state_list.entries()) == old_count + 1, page.status.text()
    return page.state_list.entries()[-1].id


def _two_states(page):
    first = _add(page)
    page.tip_energy.setValue(5.)
    page.apply_source_button.click()
    assert page._state.electron_gun.emitter.emission_energy_ev == 5., page.status.text()
    return first, _add(page)


def _calculate(page, qtbot, count=2):
    if page.state_list.display_mode() == "current":
        page.state_list.mode.setCurrentIndex(page.state_list.mode.findData("overlay"))
    page.calculate_button.click()
    _wait_overlay(page, qtbot, page._target_z_mm, count)


def _wait_overlay(page, qtbot, z_mm, count=2):
    try:
        qtbot.waitUntil(lambda: isinstance(page.result, EnsembleObservation)
            and page.result.checkpoint.plane_z_mm == z_mm
            and len(page.result.members) == count
            and page.state_set.worker is None, timeout=10000)
    except QtTimeoutError as error:
        raise AssertionError(f"Overlay did not complete: {page.status.text()}; "
            f"failed states={page.state_set.failed}; calls={len(page.execution.calls)}") from error


def test_add_applied_states_captures_exact_sources_without_starting_jobs(page):
    original = capture_instrument_snapshot(page._state)
    first = _add(page)
    assert page.state_set.snapshots[first].digest == original.digest
    assert page.execution.calls == []
    page.tip_energy.setValue(5.)
    page.state_list.add_button.click()
    assert len(page.state_list.entries()) == 1
    assert "not added" in page.status.text().lower()
    assert "Apply tip parameters" in page.status.text()
    assert capture_instrument_snapshot(page._state).digest == original.digest
    page.apply_source_button.click()
    second = _add(page)
    assert page.state_set.snapshots[first].restore().electron_gun.emitter.emission_energy_ev == 4.
    assert page.state_set.snapshots[second].restore().electron_gun.emitter.emission_energy_ev == 5.
    assert page.execution.calls == []


def test_different_states_overlay_same_plane_in_shared_optics_without_live_mutation(page, qtbot):
    _two_states(page)
    before = capture_instrument_snapshot(page._state)
    _calculate(page, qtbot)
    assert [(call[0], call[1]) for call in page.execution.calls] == [(4., page._target_z_mm), (5., page._target_z_mm)]
    assert not any(call[3] for call in page.execution.calls)
    assert page.execution.peak_active == 1
    assert capture_instrument_snapshot(page._state).digest == before.digest
    np.testing.assert_allclose(page.preview.density[:, 1], (.5, 0., 0., .5))
    assert page.preview.probability == pytest.approx(1.)
    assert "independent tip states" in page.readout.text()
    assert page.result.members[0].checkpoint.initial_energy_ev == 4.
    assert page.result.members[1].checkpoint.initial_energy_ev == 5.
    assert page.result.relative_weights == (.5, .5)


def test_duplicates_and_repeat_calculate_reuse_one_executed_state(page, qtbot):
    _add(page)
    _add(page)
    _calculate(page, qtbot)
    assert len(page.execution.calls) == len(page.execution.sessions) == 1
    assert page.result.members[0] is page.result.members[1]
    assert len(page.state_set.runs) == 1
    page.calculate_button.click()
    _wait_overlay(page, qtbot, page._target_z_mm)
    assert len(page.execution.calls) == len(page.execution.sessions) == 1


def test_replace_one_applied_state_reuses_the_unchanged_member(page, qtbot):
    first, second = _two_states(page)
    _calculate(page, qtbot)
    unchanged_result = page.result.members[1]
    unchanged_snapshot = page.state_set.snapshots[second]
    page.state_list.table.selectRow(0)
    page.tip_energy.setValue(6.)
    page.apply_source_button.click()
    page.state_list.replace_button.click()
    assert page.state_list.entries()[0].id == first
    assert page.state_set.snapshots[first].restore().electron_gun.emitter.emission_energy_ev == 6.
    assert page.state_set.snapshots[second] is unchanged_snapshot
    assert len(page.execution.calls) == 2
    _calculate(page, qtbot)
    assert len(page.execution.calls) == 3
    assert page.execution.calls[-1][0] == 6.
    assert page.result.members[1] is unchanged_result
    assert page._state.electron_gun.emitter.emission_energy_ev == 6.


def test_unadmitted_old_optics_state_cannot_resume_or_publish_by_selection(page, qtbot):
    first, second = _two_states(page)
    _calculate(page, qtbot)
    original_second_run = page.state_set.row_keys[second]
    page._state.lenses[0].percent += 1.
    page.mark_inputs_stale()
    page.state_list.mode.setCurrentIndex(page.state_list.mode.findData("selected"))
    page.state_list.table.selectRow(0)
    _calculate(page, qtbot, count=1)
    assert page.state_set.admitted_ids == {first}
    assert len(page.execution.calls) == 3
    accepted_current_optics = page.result
    # B still has an old executed session, but only A has been admitted to the
    # current optical configuration. Browsing B must not run its old optics.
    assert page.state_set.row_keys[second] == original_second_run
    page.state_list.table.selectRow(1)
    assert page.result is accepted_current_optics
    page.set_selected_z(page._target_z_mm + 5.)
    qtbot.waitUntil(lambda: "needs Calculate beam" in page.status.text(), timeout=10000)
    assert len(page.execution.calls) == 3
    assert page.state_set.worker is None
    assert page.result is accepted_current_optics
    assert second not in page.state_set.admitted_ids


def test_weight_rename_hide_selection_and_scale_reuse_complete_results(page, qtbot):
    _two_states(page)
    _calculate(page, qtbot)
    page.state_list.table.cellWidget(0, 2).setValue(3.)
    assert page.result.relative_weights == (.75, .25)
    np.testing.assert_allclose(page.preview.density[:, 1], (.75, 0., 0., .25))
    page.state_list.table.item(0, 1).setText("First state renamed")
    assert "First state renamed" in page.status.text()
    page.state_list.table.item(1, 0).setCheckState(Qt.CheckState.Unchecked)
    assert len(page.result.members) == 1
    assert page.result.members[0].checkpoint.initial_energy_ev == 4.
    page.state_list.mode.setCurrentIndex(page.state_list.mode.findData("selected"))
    page.state_list.table.selectRow(1)
    assert len(page.result.members) == 1
    assert page.result.members[0].checkpoint.initial_energy_ev == 5.
    page.intensity_scale.setCurrentIndex(1)
    qtbot.wait(80)
    assert len(page.execution.calls) == 2


def test_saved_states_reexecute_new_optics_and_z_then_reuse_exact_plane(page, qtbot):
    _two_states(page)
    _calculate(page, qtbot)
    initial_z = page._target_z_mm
    page._state.lenses[0].percent += 1.
    expected_percent = page._state.lenses[0].percent
    page.mark_inputs_stale()
    _calculate(page, qtbot)
    assert len(page.execution.calls) == 4
    assert all(call[2].restore().lenses[0].percent == expected_percent for call in page.execution.calls[2:])
    page.set_selected_z(initial_z + 5.)
    _wait_overlay(page, qtbot, initial_z + 5.)
    assert len(page.execution.calls) == 6
    assert [call[0] for call in page.execution.calls[-2:]] == [4., 5.]
    assert page.execution.peak_active == 1
    page.set_selected_z(initial_z)
    _wait_overlay(page, qtbot, initial_z)
    assert len(page.execution.calls) == 6


@pytest.mark.parametrize("mode", ["overlay", "selected"])
@pytest.mark.parametrize("draft", [False, True])
def test_live_refresh_readmits_saved_states_or_yields_to_tip_draft(page, qtbot, mode, draft):
    """Keep actual snapshot/admission/publication; only propagation is fake."""
    from temsim.gui.live_beam_refresh import LiveBeamRefresh

    first, second = _two_states(page)
    page.state_list.mode.setCurrentIndex(page.state_list.mode.findData(mode))
    page.state_list.table.selectRow(1)
    expected_ids = {first, second} if mode == "overlay" else {second}
    _calculate(page, qtbot, count=len(expected_ids))
    previous = page.result
    old_count = len(page.execution.calls)
    old_keys = dict(page.state_set.row_keys)
    source_digests = {identity: snapshot.digest for identity, snapshot in page.state_set.snapshots.items()}
    refresh = LiveBeamRefresh(page, page)
    refresh.SETTLE_MS = 20
    try:
        page._state.lenses[0].percent += 1.
        expected_percent = page._state.lenses[0].percent
        live_digest = capture_instrument_snapshot(page._state).digest
        refresh.invalidate(lambda: page.set_state(page._state))
        assert refresh.pending
        assert page.state_set.admitted_ids == set() and not page.state_set.active
        if draft:
            page.tip_offset_x.setValue(.125)
        refresh.particle_ready()
        if draft:
            qtbot.wait(60)
            assert not refresh.pending
            assert len(page.execution.calls) == old_count
            assert page.result is previous and page.state_set.admitted_ids == set()
            assert page.state_set.worker is None and not page.state_set.active
            assert page.tip_offset_x.value() == .125
        else:
            qtbot.waitUntil(lambda: page.result is not previous and page.state_set.worker is None,
                            timeout=10000)
            assert page.state_set.admitted_ids == expected_ids
            assert len(page.result.members) == len(expected_ids)
            assert len(page.execution.calls) == old_count + len(expected_ids)
            assert all(page.state_set.row_keys[identity] != old_keys[identity] for identity in expected_ids)
            assert all(call[2].restore().lenses[0].percent == expected_percent
                       for call in page.execution.calls[old_count:])
            energies = [call[0] for call in page.execution.calls[old_count:]]
            assert energies == ([4., 5.] if mode == "overlay" else [5.])
            refresh.particle_ready()
            qtbot.wait(40)
            assert len(page.execution.calls) == old_count + len(expected_ids)
        assert capture_instrument_snapshot(page._state).digest == live_digest
        assert {identity: snapshot.digest for identity, snapshot in page.state_set.snapshots.items()} == source_digests
    finally:
        refresh.cancel()


def test_failed_member_retains_previous_complete_plane_and_no_partial_overlay(page, qtbot):
    _two_states(page)
    _calculate(page, qtbot)
    previous = page.result
    old_preview = page.preview
    next_z = page._target_z_mm + 10.
    page.execution.fail = lambda energy, z: energy == 5. and z == next_z
    page.set_selected_z(next_z)
    qtbot.waitUntil(lambda: bool(page.state_set.failed) and page.state_set.worker is None, timeout=10000)
    assert page.result is previous
    assert page.preview is old_preview
    assert "partial overlay" in page.status.text().lower()
    assert page.result.checkpoint.plane_z_mm != next_z
    assert len(page.execution.calls) == 4


@pytest.mark.parametrize("action", ["cancel", "remove"])
def test_cancel_or_remove_rejects_late_completion(page, qtbot, action):
    _two_states(page)
    _calculate(page, qtbot)
    previous = page.result
    next_z = page._target_z_mm + 20.
    page.execution.block = lambda _energy, z: z == next_z
    page.set_selected_z(next_z)
    qtbot.waitUntil(lambda: page.execution.active == 1, timeout=10000)
    stale_worker = page.state_set.worker
    stale_key = page.state_set._job_key
    if action == "cancel":
        page.cancel_button.click()
    else:
        page.state_set.remove(page.state_list.entries()[0].id)
    page.execution.release.set()
    qtbot.waitUntil(lambda: page.execution.active == 0 and page.state_set.pool.activeThreadCount() == 0,
                    timeout=10000)
    assert page.state_set.worker is None
    assert page.result is previous
    if stale_key in page.state_set.runs:
        assert next_z not in page.state_set.runs[stale_key].cache
    # Late signal from an obsolete generation cannot publish even if an
    # external executor completed after cancellation was acknowledged.
    late = SimpleNamespace(checkpoint=SimpleNamespace(plane_z_mm=next_z, reference_current_a=1.))
    page.state_set._solved(stale_worker.generation, stale_worker.session, late, page.preview)
    assert page.result is previous


def test_load_selected_publishes_source_for_both_paths_without_job(page):
    first, _ = _two_states(page)
    page.state_list.table.selectRow(0)
    page.state_list.restore_button.click()
    assert source_settings_from_state(page._state) == source_settings_from_state(
        page.state_set.snapshots[first].restore())
    assert page.tip_energy.value() == 4.
    assert page.execution.calls == []
    assert "no calculation started" in page.status.text().lower()
    # Explicitly loading the same applied source still discards editor drafts.
    page.tip_offset_x.setValue(.125)
    page.state_list.restore_button.click()
    assert page._settings() == source_settings_from_state(page._state)
    assert page.tip_offset_x.value() == 0.
    assert page.execution.calls == []


def test_current_tip_result_survives_overlay_and_switch_back(page, qtbot):
    _two_states(page)
    page.calculate_button.click()
    qtbot.waitUntil(lambda: page.result is not None and page._worker is None, timeout=10000)
    current_result, current_preview = page.result, page.preview
    assert not isinstance(current_result, EnsembleObservation)
    assert current_result.checkpoint.initial_energy_ev == 5.
    _calculate(page, qtbot)
    assert isinstance(page.result, EnsembleObservation)
    page.state_list.mode.setCurrentIndex(page.state_list.mode.findData("current"))
    assert page.result is current_result
    assert page.preview is current_preview
    assert page._worker is None
    # The current-tip execution is the same initial state as the second row.
    # It must be adopted into the comparison cache rather than propagated twice.
    assert len(page.execution.calls) == 2


def test_display_edits_keep_cancel_enabled_for_active_saved_state_worker(page, qtbot):
    _two_states(page)
    _calculate(page, qtbot)
    previous = page.result
    next_z = page._target_z_mm + 25.
    page.execution.block = lambda _energy, z: z == next_z
    page.set_selected_z(next_z)
    qtbot.waitUntil(lambda: page.execution.active == 1, timeout=10000)
    worker = page.state_set.worker
    assert worker is not None
    assert page.cancel_button.isEnabled()
    page.state_list.table.cellWidget(0, 2).setValue(2.)
    page.state_list.table.item(0, 1).setText("Renamed during calculation")
    page.state_list.table.item(1, 0).setCheckState(Qt.CheckState.Unchecked)
    assert page.state_set.worker is worker
    assert page.cancel_button.isEnabled(), "Display-only edits disabled Cancel for a live state worker"
    page.cancel_button.click()
    assert worker.event.is_set()
    page.execution.release.set()
    qtbot.waitUntil(lambda: page.execution.active == 0 and page.state_set.pool.activeThreadCount() == 0,
                    timeout=10000)
    assert page.state_set.worker is None
    assert page.result is previous
    assert len(page.execution.calls) == 3


def test_switch_back_to_current_tip_automatically_observes_latest_requested_z(page, qtbot):
    _two_states(page)
    initial_z = page._target_z_mm
    page.calculate_button.click()
    qtbot.waitUntil(lambda: page.result is not None and page._worker is None, timeout=10000)
    assert page.result.checkpoint.plane_z_mm == initial_z
    _calculate(page, qtbot)
    latest_z = initial_z + 30.
    page.set_selected_z(latest_z)
    _wait_overlay(page, qtbot, latest_z)
    assert page._target_z_mm == latest_z
    page.state_list.mode.setCurrentIndex(page.state_list.mode.findData("current"))
    try:
        qtbot.waitUntil(lambda: page.result is not None
            and not isinstance(page.result, EnsembleObservation)
            and page.result.checkpoint.plane_z_mm == latest_z
            and page._worker is None, timeout=10000)
    except QtTimeoutError as error:
        raise AssertionError(f"Current tip did not follow requested Z: {page.status.text()}; "
            f"displayed={page.result.checkpoint.plane_z_mm}, requested={latest_z}") from error
    assert page.result.checkpoint.initial_energy_ev == 5.
    assert page.state_set.worker is None
    assert page.execution.peak_active == 1


def test_current_tip_metadata_preserves_pending_pair_status_and_cancel(page):
    _add(page)
    assert page.state_list.display_mode() == "current"
    busy = {"paired": True}
    page.paired_busy_provider = lambda: busy["paired"]
    progress = "Paired particles: executing the captured tip-to-specimen transport"
    page.status.setText(progress)
    page.cancel_button.setEnabled(True)
    invalidated = []

    def cancel_pair():
        invalidated.append(True)
        busy["paired"] = False

    page.pair_invalidated.connect(cancel_pair)
    before = capture_instrument_snapshot(page._state)
    page.state_list.table.cellWidget(0, 2).setValue(3.)
    page.state_list.table.item(0, 1).setText("Named while paired particles run")
    page.state_list.table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    assert page.status.text() == progress
    assert page.cancel_button.isEnabled()
    assert invalidated == []
    assert page._worker is None and page.state_set.worker is None
    page._update_cancel_enabled()
    assert page.cancel_button.isEnabled()
    assert capture_instrument_snapshot(page._state).digest == before.digest
    assert page.execution.calls == []
    page.cancel_button.click()
    assert invalidated == [True]
    assert not page.cancel_button.isEnabled()
