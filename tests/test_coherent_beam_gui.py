"""GUI-only coherent-session controls; propagation is substituted.

Real Qt widgets, timers, coordinated workers, publication and cancellation are
exercised. These tests do not establish coherent gun/column physics support.
"""

from copy import deepcopy
from dataclasses import replace
import json
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QThread, QTimer, Qt

import temsim.gui.coherent_beam as module
from temsim.gui.coherent_beam import CoherentBeamPage, _Preview


def instrument(surface=None):
    return SimpleNamespace(electron_gun=SimpleNamespace(emitter=SimpleNamespace(
        surface_model=surface, coherence=None, virtual_source_fwhm_nm=5.,
        emission_energy_ev=.3, minimum_kinetic_energy_ev=.01,
        energy_spread_fwhm_ev=.2, emission_current_na=10.)),
        sample=SimpleNamespace(z_mm=100.))


def result(z_mm):
    return SimpleNamespace(checkpoint=SimpleNamespace(plane_z_mm=z_mm,
        reference_current_a=10e-9), instrument_digest="GUI-only substituted propagation")


def preview():
    return _Preview(np.ones((4, 4))/16, np.array(((-.5, .5), (-.5, .5))),
                    1., 1, 160, ((4, 4),))


class _GuiObservationSession:
    """Keep real GUI ownership while substituting only numerical execution."""

    def __init__(self, state, request):
        self.state = state
        self.request = request
        self.observed_z = []

    def observe(self, z_mm, **kwargs):
        self.observed_z.append(z_mm)
        return module.simulate_tip_wave(self.state,
            replace(self.request, observation_z_mm=z_mm), **kwargs)


@pytest.fixture
def panel(qtbot, monkeypatch):
    monkeypatch.setattr(module, "prepare_coherent_state", lambda state, settings:
        deepcopy(state) if settings.enabled else (_ for _ in ()).throw(
            ValueError("Select a coherent tip boundary before calculating a wave")))
    monkeypatch.setattr(module, "wave_input_summary", lambda *_args:
        {"status": "SOURCE_READY", "model": "GUI-only fixture", "mode_count": 1})
    monkeypatch.setattr(module, "simulate_tip_wave", lambda _state, request, **_kwargs:
                        result(request.observation_z_mm))
    monkeypatch.setattr(module, "TipWaveObservationSession", _GuiObservationSession)
    monkeypatch.setattr(module, "_intensity_preview", lambda *_args, **_kwargs: preview())
    view = CoherentBeamPage()
    qtbot.addWidget(view)
    view.set_state(instrument())
    view.show()
    yield view
    assert view.shutdown()


def calculate(panel, qtbot):
    panel.source_enabled.setChecked(True)
    qtbot.mouseClick(panel.calculate_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: panel.result is not None)
    qtbot.waitUntil(lambda: panel._worker is None)


def test_tab_open_and_z_selection_never_start_first_calculation(panel, qtbot, monkeypatch):
    calls = []
    monkeypatch.setattr(module, "simulate_tip_wave", lambda *_args, **_kwargs: calls.append(1))
    assert not panel.source_enabled.isChecked()
    assert panel.follow_ray.isChecked()
    panel.set_selected_z(101.12345678912345)
    panel.source_enabled.setChecked(True)
    qtbot.wait(230)
    assert calls == []
    assert not panel.timer.isActive()
    assert panel.result is None
    assert panel._target_z_mm == 101.12345678912345


def test_explicit_calculate_requires_tip_opt_in(panel, qtbot):
    qtbot.mouseClick(panel.calculate_button, Qt.MouseButton.LeftButton)
    assert "Select a coherent tip boundary" in panel.status.text()
    assert panel.result is None
    assert panel.pool.activeThreadCount() == 0
    assert panel.calculate_button.property("calculationAction") is True
    assert "#3ce878" in panel.calculate_button.styleSheet()
    calculate(panel, qtbot)
    assert panel.result.checkpoint.plane_z_mm == 100.
    assert "Completed exact plane" in panel.status.text()
    assert "not full TEM/STEM qualification" in panel.scope_label.text()
    assert "per tip electron" in panel.readout.text()
    assert "4 × 4 display bins; the executed wave grid is unchanged" in panel.readout.text()
    assert "Displayed wave grid: 4 × 4 cells" in panel.readout.text()


def test_worker_uses_captured_copy_and_runs_off_gui_thread(panel, qtbot, monkeypatch):
    calls = []
    original = panel._state

    def solve(state, request, **kwargs):
        calls.append((state is original, QThread.currentThread() == panel.thread(), request))
        assert callable(kwargs["cancelled"])
        return result(request.observation_z_mm)

    monkeypatch.setattr(module, "simulate_tip_wave", solve)
    calculate(panel, qtbot)
    assert calls[0][:2] == (False, False)
    assert original.electron_gun.emitter.coherence is None
    assert original.electron_gun.emitter.emission_energy_ev == .3
    assert original.electron_gun.emitter.virtual_source_fwhm_nm == 5.
    assert calls[0][2].stop == "plane"
    assert calls[0][2].inelastic.method == "trajectories"


def test_explicit_tip_controls_change_only_requested_boundary(panel):
    original = panel._state.electron_gun.emitter
    panel.tip_fwhm.setValue(50.)
    panel.tip_energy_spread.setValue(0.)
    panel.tip_curvature_x.setValue(12.)
    selected = panel._settings()
    assert selected.tip_fwhm_nm == 50.
    assert selected.tip_energy_spread_fwhm_ev == 0.
    assert selected.tip_curvature_x_m1 == 12.
    assert original.virtual_source_fwhm_nm == 5.
    assert original.energy_spread_fwhm_ev == .2
    assert panel.result is None and panel._worker is None


@pytest.mark.parametrize("entry", ["manual", "defaults"])
def test_fresh_gui_can_capture_every_gaussian_tip_input_without_changing_particles(panel, monkeypatch, entry):
    from dataclasses import asdict
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.optics.column import default_state
    from temsim.optics.electron_gun.tip_coherence import TipCoherence
    from temsim.physics.coherent_inputs import prepare_coherent_state, wave_input_summary

    state = default_state()
    before = capture_instrument_snapshot(state).digest
    assert state.electron_gun.emitter.coherence is None
    panel.set_state(state)
    # These are the explicitly prescribed tip-boundary values in the retained
    # Si example. This test checks GUI capture, not diffraction propagation.
    expected = TipCoherence(curvature_x_m1=624.3690658100246,
        curvature_xy_m1=-9.348776164576383e-12,
        curvature_y_m1=624.369065810008,
        offset_x_nm=-.06696090871824144, offset_y_nm=-.07266953387249751,
        tilt_x_mrad=-4.18060377176702e-5, tilt_y_mrad=-4.537013038384244e-5)
    for control, value in (
            (panel.angle_rms, expected.incoherent_angle_rms_mrad),
            (panel.tip_curvature_x, expected.curvature_x_m1),
            (panel.tip_curvature_xy, expected.curvature_xy_m1),
            (panel.tip_curvature_y, expected.curvature_y_m1),
            (panel.tip_offset_x, expected.offset_x_nm),
            (panel.tip_offset_y, expected.offset_y_nm),
            (panel.tip_tilt_x, expected.tilt_x_mrad),
            (panel.tip_tilt_y, expected.tilt_y_mrad)):
        assert control.isEnabled() and not control.isHidden()
        # Exercise Qt's text parser, as when the user pastes a decimal value;
        # setValue alone would not establish a reachable manual-entry path.
        if entry == "manual":
            control.lineEdit().setText(f"{value:.{control.decimals()}f}")
            control.interpretText()
        assert control.value() == pytest.approx(value, rel=2e-15, abs=5e-16)
    for control, value in ((panel.tip_fwhm, 28390.10000542304),
            (panel.tip_energy, 30.), (panel.tip_minimum_energy, .01),
            (panel.tip_energy_spread, 0.)):
        if entry == "manual":
            control.lineEdit().setText(f"{value:.{control.decimals()}f}")
            control.interpretText()
        assert control.value() == pytest.approx(value, rel=2e-15, abs=5e-16)
    if entry == "manual":
        panel.energy_samples.setValue(1)
    assert panel.energy_samples.value() == 1
    assert not panel.source_enabled.isChecked()
    panel.source_enabled.setChecked(True)
    monkeypatch.setattr(module, "prepare_coherent_state", prepare_coherent_state)
    monkeypatch.setattr(module, "wave_input_summary", wave_input_summary)
    queued = []
    monkeypatch.setattr(panel.pool, "start", queued.append)
    panel.calculate()
    assert len(queued) == 1  # Job is captured, but never executed in this test.
    assert "Source ready" in panel.status.text()
    captured = panel._captured.electron_gun.emitter
    for field, value in asdict(expected).items():
        assert getattr(captured.coherence, field) == pytest.approx(value, rel=2e-15, abs=5e-16)
    assert captured.virtual_source_fwhm_nm == pytest.approx(28390.10000542304, rel=2e-15)
    assert captured.emission_energy_ev == 30.
    assert captured.minimum_kinetic_energy_ev == .01
    assert captured.energy_spread_fwhm_ev == 0.
    assert state.electron_gun.emitter.coherence is None
    assert capture_instrument_snapshot(state).digest == before
    assert "not the metal" in panel.source_assumption.text()
    assert "Idealised diffraction example defaults" in panel.source_info.text()
    assert "not the metal apex radius" in panel.tip_fwhm.toolTip()
    panel.cancel()
    for control in (panel.tip_offset_x, panel.tip_offset_y, panel.tip_curvature_xy):
        panel._stale = False
        panel._session_ready = True
        control.setValue(control.value()+1e-6)
        assert panel._stale and not panel._session_ready
    assert capture_instrument_snapshot(state).digest == before


def test_existing_gaussian_centre_and_cross_curvature_initialize_and_reset(panel):
    from temsim.optics.electron_gun.tip_coherence import TipCoherence

    state = instrument()
    state.electron_gun.emitter.coherence = TipCoherence(
        offset_x_nm=1.25, offset_y_nm=-2.5, curvature_xy_m1=17.125,
        incoherent_angle_rms_mrad=.125)
    panel.set_state(state)
    assert panel.tip_offset_x.value() == 1.25
    assert panel.tip_offset_y.value() == -2.5
    assert panel.tip_curvature_xy.value() == 17.125
    assert panel.angle_rms.value() == .125
    assert panel.tip_fwhm.value() == 5.
    assert panel.tip_energy.value() == .3
    assert panel.tip_energy_spread.value() == .2
    assert panel.energy_samples.value() == 9
    panel.set_state(instrument())
    assert panel.tip_offset_x.value() == pytest.approx(-.06696090871824144)
    assert panel.tip_offset_y.value() == pytest.approx(-.07266953387249751)
    assert panel.tip_curvature_xy.value() == pytest.approx(-9.348776164576383e-12, abs=5e-16)
    assert panel.angle_rms.value() == 0.
    assert panel.energy_samples.value() == 1


def test_saved_zero_phase_boundary_is_preserved_instead_of_example_defaults(panel):
    from temsim.optics.electron_gun.tip_coherence import TipCoherence

    state = instrument()
    state.electron_gun.emitter.coherence = TipCoherence()
    panel.set_state(state)
    assert panel.tip_fwhm.value() == 5.
    assert panel.tip_energy.value() == .3
    assert panel.tip_energy_spread.value() == .2
    assert panel.tip_curvature_x.value() == panel.tip_curvature_xy.value() == 0.
    assert panel.tip_curvature_y.value() == 0.
    assert panel.tip_offset_x.value() == panel.tip_offset_y.value() == 0.
    assert "Captured tip boundary" in panel.source_info.text()


def test_gaussian_defaults_do_not_replace_curved_geometry_inputs(panel):
    state = instrument()
    state.electron_gun.emitter.curvature_nm_inv = .01
    panel.set_state(state)
    assert panel.tip_fwhm.value() == 5.
    assert panel.tip_energy.value() == .3
    assert panel.tip_curvature_x.value() == 0.
    assert state.electron_gun.emitter.curvature_nm_inv == .01
    assert not panel.source_enabled.isChecked()


def test_default_energy_mode_budget_is_not_reset_by_instrument_publication(panel):
    from temsim.optics.electron_gun.tip_coherence import TipCoherence

    assert panel.energy_samples.value() == 1
    panel.energy_samples.setValue(7)
    loaded = instrument()
    loaded.electron_gun.emitter.coherence = TipCoherence()
    panel.set_state(loaded)
    assert panel.energy_samples.value() == 7
    panel.set_state(instrument())
    assert panel.energy_samples.value() == 7


def test_snapshot_restored_custom_coherent_source_keeps_all_values(panel):
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.optics.column import default_state
    from temsim.optics.electron_gun.tip_coherence import TipCoherence

    state = default_state()
    emitter = state.electron_gun.emitter
    emitter.virtual_source_fwhm_nm = 70.
    emitter.emission_energy_ev = 2.
    emitter.energy_spread_fwhm_ev = .1
    emitter.coherence = TipCoherence(.125, 100., 17.125, 200., 1.25, -2.5, .01, -.02)
    before = capture_instrument_snapshot(state)
    restored = before.restore()
    panel.set_state(restored)
    assert panel.tip_fwhm.value() == 70.
    assert panel.tip_energy.value() == 2.
    assert panel.tip_energy_spread.value() == .1
    assert panel.tip_minimum_energy.value() == emitter.minimum_kinetic_energy_ev
    for control, value in ((panel.angle_rms, .125),
            (panel.tip_curvature_x, 100.), (panel.tip_curvature_xy, 17.125),
            (panel.tip_curvature_y, 200.), (panel.tip_offset_x, 1.25),
            (panel.tip_offset_y, -2.5), (panel.tip_tilt_x, .01), (panel.tip_tilt_y, -.02)):
        assert control.value() == value
    assert panel.energy_samples.value() == 9
    assert not panel.source_enabled.isChecked()
    assert capture_instrument_snapshot(restored).digest == before.digest


def test_nonzero_energy_spread_updates_only_automatic_mode_budget(panel):
    assert panel.energy_samples.value() == 1
    panel.tip_energy_spread.setValue(.1)
    assert panel.energy_samples.value() == 9
    panel.tip_energy_spread.setValue(0.)
    assert panel.energy_samples.value() == 1
    panel.energy_samples.setValue(7)
    panel.tip_energy_spread.setValue(.2)
    assert panel.energy_samples.value() == 7
    assert panel._worker is None and panel.result is None


def test_gun_energy_step_budget_is_explicit_and_invalidates_session(panel, qtbot):
    calculate(panel, qtbot)
    previous = panel.result
    panel.gun_energy_step.setValue(.5)
    request = panel._make_request()
    assert request.gun.maximum_fractional_energy_change == pytest.approx(.005)
    assert panel.result is previous
    assert panel._stale and not panel._session_ready
    assert panel._worker is None


def test_log_display_preserves_density_and_current_session(panel, qtbot):
    calculate(panel, qtbot)
    result_before, preview_before = panel.result, panel.preview
    saved = panel.preview.density.copy()
    generation = panel._generation
    panel.intensity_scale.setCurrentIndex(1)
    assert panel.result is result_before and panel.preview is preview_before
    np.testing.assert_array_equal(panel.preview.density, saved)
    assert panel._generation == generation and panel._session_ready
    assert panel._worker is None
    assert np.all(np.isfinite(panel.image.image))
    panel.intensity_scale.setCurrentIndex(0)
    np.testing.assert_array_equal(panel.image.image, saved)


def test_continuous_z_motion_produces_results_without_waiting_for_drag_release(panel, qtbot, monkeypatch):
    calculate(panel, qtbot)
    old_image = panel.image.image
    calls = []
    captured = panel._captured

    def solve(state, request, **_kwargs):
        calls.append((state, request.observation_z_mm))
        return result(request.observation_z_mm)

    monkeypatch.setattr(module, "simulate_tip_wave", solve)
    panel.set_selected_z(101.)
    assert panel.image.image is old_image
    # Motion arrives faster than the query interval for as long as this timer
    # runs. A restart-on-edit debounce would never execute a single request.
    motion = QTimer(panel)
    motion.setInterval(10)
    next_z = [102.]

    def move():
        panel.set_selected_z(next_z[0])
        next_z[0] += 1.

    motion.timeout.connect(move)
    motion.start()
    try:
        qtbot.wait(220)
        qtbot.waitUntil(lambda: panel.result.checkpoint.plane_z_mm > 100.)
        assert motion.isActive()
        assert calls and all(state is captured for state, _z in calls)
    finally:
        motion.stop()
    latest_z = panel._target_z_mm
    qtbot.waitUntil(lambda: panel.result.checkpoint.plane_z_mm == latest_z)
    qtbot.waitUntil(lambda: panel._worker is None)
    assert calls[-1] == (captured, latest_z)


def test_exact_cache_and_duplicate_z_do_not_recalculate(panel, qtbot, monkeypatch):
    calculate(panel, qtbot)
    calls = []
    monkeypatch.setattr(module, "simulate_tip_wave", lambda _state, request, **_kwargs:
        (calls.append(request.observation_z_mm), result(request.observation_z_mm))[1])
    first = 100.00000000012345
    panel.set_selected_z(first)
    panel.set_selected_z(first)
    qtbot.waitUntil(lambda: panel.result.checkpoint.plane_z_mm == first)
    panel.set_selected_z(100.)
    assert panel.result.checkpoint.plane_z_mm == 100.
    assert not panel.timer.isActive()
    qtbot.wait(220)
    assert calls == [first]
    panel.set_selected_z(np.nextafter(first, np.inf))
    assert panel.timer.isActive()  # Adjacent linked floats are different exact planes.


def test_follow_checkbox_prevents_linked_queries_but_manual_z_works(panel, qtbot):
    calculate(panel, qtbot)
    panel.follow_ray.setChecked(False)
    panel.set_selected_z(102.)
    assert panel._target_z_mm == 100.
    panel.target_z.setValue(103.)
    qtbot.waitUntil(lambda: panel.result.checkpoint.plane_z_mm == 103.)


@pytest.mark.parametrize("control", ["source_enabled", "angle_rms", "grid_pixels",
                                    "working_gib", "ram_cache_gib"])
def test_source_or_numerical_edit_requires_new_explicit_session(panel, qtbot, control):
    calculate(panel, qtbot)
    old = panel.result
    widget = getattr(panel, control)
    if isinstance(widget, module.QCheckBox):
        widget.setChecked(False)
    else:
        widget.setValue(widget.value()+1)
    panel.set_selected_z(105.)
    qtbot.wait(220)
    assert panel._stale
    assert not panel.timer.isActive()
    assert panel.result is old
    assert not panel._cache
    assert "Previous optics" in panel.status.text()


def test_instrument_publication_invalidates_without_launching(panel, qtbot):
    calculate(panel, qtbot)
    old = panel.result
    panel.set_state(instrument())
    panel.set_selected_z(110.)
    qtbot.wait(220)
    assert panel._stale
    assert panel.result is old
    assert panel._captured is not panel._state


def test_late_success_failure_and_progress_cannot_overwrite_current_session(panel, qtbot):
    calculate(panel, qtbot)
    old_generation, old_session = panel._generation, panel._session
    panel.mark_inputs_stale()
    status = panel.status.text()
    old = panel.result
    panel._solved(old_generation, old_session, result(100.), preview())
    panel._failed(old_generation, "Late failure")
    panel._progress(old_generation, "Late progress")
    assert panel.status.text() == status
    assert panel.result is old
    assert panel._cache == {}


def test_cancel_sets_event_and_keeps_complete_result(panel, qtbot, monkeypatch):
    calculate(panel, qtbot)
    old = panel.result
    started, released = Event(), Event()
    events = []

    def slow(_state, request, *, cancelled, **_kwargs):
        events.append(cancelled)
        started.set()
        released.wait(2.)
        return result(request.observation_z_mm)

    monkeypatch.setattr(module, "simulate_tip_wave", slow)
    panel.set_selected_z(106.)
    qtbot.waitUntil(started.is_set)
    panel.cancel()
    assert events[0]()
    assert panel.result is old
    assert "cancelled" in panel.status.text()
    released.set()
    assert panel.pool.waitForDone(2000)
    assert panel.result is old


def test_current_failure_keeps_previous_image_and_allows_retry(panel, qtbot, monkeypatch):
    calculate(panel, qtbot)
    old = panel.result
    monkeypatch.setattr(module, "simulate_tip_wave", lambda *_args, **_kwargs:
        (_ for _ in ()).throw(ValueError("Unsupported field fixture")))
    panel.set_selected_z(105.)
    qtbot.waitUntil(lambda: "Unsupported field fixture" in panel.status.text())
    assert panel.result is old
    assert "Previous complete plane" in panel.status.text()
    monkeypatch.setattr(module, "simulate_tip_wave", lambda _state, request, **_kwargs:
                        result(request.observation_z_mm))
    panel.set_selected_z(106.)
    qtbot.waitUntil(lambda: panel.result.checkpoint.plane_z_mm == 106.)


def test_failure_names_submitted_plane_and_retained_image_in_both_views(panel, qtbot, monkeypatch):
    calculate(panel, qtbot)
    mirror = module.CoherentBeamMirror(panel)
    qtbot.addWidget(mirror)
    mirror.show()
    qtbot.waitUntil(lambda: mirror.screen.image.image is not None)
    old_result = panel.result
    old_main_image = panel.image.image
    old_mirror_image = mirror.screen.image.image
    calls = []
    release_failure, release_latest = Event(), Event()

    def solve(_state, request, **_kwargs):
        calls.append(request.observation_z_mm)
        if request.observation_z_mm == 105.:
            assert release_failure.wait(5.)
            raise MemoryError("Working limit exceeded")
        assert release_latest.wait(5.)
        return result(request.observation_z_mm)

    monkeypatch.setattr(module, "simulate_tip_wave", solve)
    try:
        panel.set_selected_z(105.)
        qtbot.waitUntil(lambda: calls == [105.])
        failed_generation = panel._generation
        panel.set_selected_z(106.)
        release_failure.set()
        qtbot.waitUntil(lambda: calls == [105., 106.])
        expected = ("Coherent beam calculation failed at Z 105 mm: Working limit exceeded. "
                    "No new image was produced. Previous complete plane Z 100 mm remains displayed.")
        assert panel.status.text() == mirror.status.text() == expected
        assert panel.plane_positions.text() == mirror.plane_positions.text() == (
            "Requested Z 106 mm | Calculating Z 106 mm | Displayed Z 100 mm")
        assert panel.result is old_result
        assert panel.image.image is old_main_image
        assert mirror.screen.image.image is old_mirror_image
        panel._failed(failed_generation, "Late failure must not replace the current status")
        assert panel.status.text() == mirror.status.text() == expected
    finally:
        release_failure.set()
        release_latest.set()
    qtbot.waitUntil(lambda: panel.result.checkpoint.plane_z_mm == 106.)
    qtbot.waitUntil(lambda: panel._worker is None)
    assert "Completed exact plane Z 106 mm" in panel.status.text()
    assert mirror.status.text() == panel.status.text()


def test_unsupported_source_does_not_enter_worker_or_mutate_source(panel, qtbot, monkeypatch):
    monkeypatch.setattr(module, "wave_input_summary", lambda *_args:
        {"status": "SOURCE_UNSUPPORTED", "reason": "Fixture source domain failure"})
    panel.source_enabled.setChecked(True)
    panel.calculate()
    assert "Fixture source domain failure" in panel.status.text()
    assert panel._worker is None
    assert panel._state.electron_gun.emitter.coherence is None


def test_saved_low_energy_source_rejection_displays_real_checks_without_propagation(qtbot, monkeypatch):
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.optics.column import default_state
    from temsim.optics.electron_gun.tip_coherence import TipCoherence, TipEmission

    state = default_state()
    state.electron_gun.emitter.ray_count = 3000
    # An explicitly saved narrow, low-energy coherent input must still be
    # rejected, rather than replaced with the newly selected GUI defaults.
    state.electron_gun.emitter.coherence = TipCoherence()
    before = capture_instrument_snapshot(state).digest
    monkeypatch.setattr(module, "simulate_tip_wave", lambda *_args, **_kwargs:
                        pytest.fail("A rejected source started propagation"))
    monkeypatch.setattr(TipEmission, "modes", lambda *_args, **_kwargs:
                        pytest.fail("Source checks constructed wave arrays"))
    view = CoherentBeamPage()
    qtbot.addWidget(view)
    view.set_state(state)
    view.source_enabled.setChecked(True)
    view.calculate()

    assert "propagation did not start" in view.status.text()
    report = json.loads(view.preflight_details.toPlainText())
    summary = report["source_summary"]
    assert summary["status"] == "SOURCE_UNSUPPORTED"
    assert summary["inputs"]["virtual_source_fwhm_nm"] == 5.
    assert summary["inputs"]["emission_energy_ev"] == .3
    assert summary["inputs"]["energy_spread_fwhm_ev"] == .3
    assert summary["source_domain"]["minimum_evaluated_energy_ev"] == .01
    # Independent Gaussian/Fourier evaluation for the declared default tip.
    assert summary["source_domain"]["support_radius_over_p"] == pytest.approx(2.78989, rel=1e-5)
    assert "Particle ray count" in view.preflight_message.text()
    assert "0.198997" in view.preflight_message.text()
    assert view.preflight_details.isReadOnly()
    assert view.preflight_group.isChecked()
    assert view._worker is None and view.result is None
    assert not view.cancel_button.isEnabled()
    assert capture_instrument_snapshot(state).digest == before
    assert view.tip_fwhm.value() == 5. and view.tip_energy_spread.value() == .3
    assert view.shutdown()


def test_rejected_check_keeps_previous_image_and_marks_report_stale(panel, qtbot, monkeypatch):
    from temsim.immutable_json import freeze_json

    calculate(panel, qtbot)
    previous_result, previous_image = panel.result, panel.image.image
    summary = freeze_json({"status": "SOURCE_UNSUPPORTED", "model": "GUI source fixture",
        "reason": "Fixture source domain failure", "inputs": {"virtual_source_fwhm_nm": 5.},
        "source_domain": {"support_radius_over_p": 2.78989,
            "minimum_evaluated_energy_ev": .01, "tail_probability_budget": 1e-8,
            "paraxial_generator_relative_error_budget": .01}})
    monkeypatch.setattr(module, "wave_input_summary", lambda *_args: summary)
    panel.calculate()
    assert panel.result is previous_result and panel.image.image is previous_image
    assert "propagation did not start" in panel.status.text()
    assert "Previous result" in panel.status.text()
    assert json.loads(panel.preflight_details.toPlainText())["source_summary"] == {
        "status": "SOURCE_UNSUPPORTED", "model": "GUI source fixture",
        "reason": "Fixture source domain failure", "inputs": {"virtual_source_fwhm_nm": 5.},
        "source_domain": {"support_radius_over_p": 2.78989,
            "minimum_evaluated_energy_ev": .01, "tail_probability_budget": 1e-8,
            "paraxial_generator_relative_error_budget": .01}}
    previous_report = panel.preflight_details.toPlainText()
    panel.tip_fwhm.setValue(50.)
    assert "Previous source check" in panel.preflight_message.text()
    assert panel.preflight_details.toPlainText() == previous_report
    assert panel.result is previous_result and panel._stale
    monkeypatch.setattr(module, "wave_input_summary", lambda *_args:
        {"status": "SOURCE_READY", "model": "GUI source fixture", "mode_count": 1})
    panel.calculate()
    qtbot.waitUntil(lambda: panel._worker is None)
    assert "Source check passed" in panel.preflight_message.text()
    assert "not propagation or image qualification" in panel.preflight_message.text()
    assert json.loads(panel.preflight_details.toPlainText())["source_summary"]["status"] == "SOURCE_READY"


def test_missing_opt_in_has_copyable_settings_but_no_source_report(panel, qtbot):
    panel.calculate()
    report = json.loads(panel.preflight_details.toPlainText())
    assert report["source_summary"] is None
    assert report["tip_inputs"]["enabled"] is False
    assert "Select a coherent tip boundary" in report["error"]
    assert panel._worker is None and not panel.source_enabled.isChecked()


@pytest.mark.parametrize("domain", [
    {"support_radius_over_p": 2.78989},
    {"minimum_evaluated_energy_ev": None, "support_radius_over_p": 2.78989},
    {"support_radius_over_p": 2.78989, "paraxial_generator_relative_error_budget": -1.,
     "tail_probability_budget": 1e-8},
])
def test_partial_source_report_does_not_break_error_display(panel, monkeypatch, domain):
    monkeypatch.setattr(module, "wave_input_summary", lambda *_args:
        {"status": "SOURCE_UNSUPPORTED", "reason": "Partial source check fixture",
         "source_domain": domain})
    panel.source_enabled.setChecked(True)
    panel.calculate()
    assert "propagation did not start" in panel.status.text()
    assert "Model limit unavailable" in panel.preflight_message.text()
    assert json.loads(panel.preflight_details.toPlainText())["source_summary"]["source_domain"] == domain
    assert panel._worker is None and panel.result is None


def test_surface_assumption_is_explicit_and_values_are_passed_only_on_calculate(panel, qtbot, monkeypatch):
    surface = SimpleNamespace(coherence=SimpleNamespace(mean_energy_ev=.4,
        energy_rms_ev=.07, edge_phase_rad=.3),
        geometry=SimpleNamespace(apex_radius_nm=30.), current_na=12.)
    state = instrument(surface)
    panel.set_state(state)
    settings = []
    monkeypatch.setattr(module, "prepare_coherent_state", lambda original, options:
        (settings.append(options), deepcopy(original))[1])
    assert not panel.source_enabled.isChecked()
    assert panel.surface_mean.value() == .4
    assert panel.surface_rms.value() == .07
    assert "Explicit quantum assumption" in panel.source_assumption.text()
    assert panel.angle_rms.isHidden()
    panel.source_enabled.setChecked(True)
    panel.surface_mean.setValue(.5)
    panel.calculate()
    qtbot.waitUntil(lambda: panel.result is not None)
    assert settings[-1].surface_mean_energy_ev == .5
    assert settings[-1].surface_energy_rms_ev == .07
    assert settings[-1].surface_edge_phase_rad is None
    assert state.electron_gun.emitter.surface_model.coherence.mean_energy_ev == .4


def test_advanced_budgets_only_change_numerics(panel):
    assert panel.advanced_body.isHidden()
    panel.advanced.setChecked(True)
    assert not panel.advanced_body.isHidden()
    panel.grid_pixels.setValue(256)
    panel.energy_samples.setValue(3)
    panel.column_step.setValue(.1)
    panel.working_gib.setValue(12.)
    panel.ram_cache_gib.setValue(10.)
    panel.disk_cache_gib.setValue(128.)
    request = panel._make_request()
    assert request.source.grid_pixels == 256
    assert request.source.energy_samples == request.surface.energy_samples == 3
    assert request.column_step_mm == .1
    assert request.wave_grid.maximum_working_bytes == 12*module.GIB
    assert request.gun.maximum_checkpoint_bytes == 12*module.GIB
    assert request.surface.maximum_working_bytes == 12*module.GIB
    assert request.radial_gun.maximum_working_bytes == 12*module.GIB
    assert request.maximum_readout_bytes == 12*module.GIB
    assert request.execution.maximum_ram_cache_bytes == 10*module.GIB
    assert request.execution.maximum_disk_cache_bytes == 128*module.GIB
    assert request.execution.segmented
    assert request.inelastic.method == "trajectories"


def test_si_example_grid_and_material_budget_are_explicit_and_invalidate_session(panel, qtbot):
    default = module.TipWaveRequest()
    assert panel.maximum_grid_pixels.value() == default.wave_grid.maximum_pixels
    assert panel.material_trajectories.value() == default.inelastic.trajectories_per_mode
    assert panel.gun_bore_step.value() == default.gun.bore_step_mm
    calculate(panel, qtbot)
    previous = panel.result
    panel.maximum_grid_pixels.setValue(4096)
    panel.material_trajectories.setValue(1)
    panel.gun_bore_step.setValue(10.)
    request = panel._make_request()
    assert request.wave_grid.maximum_pixels == 4096
    assert request.inelastic.trajectories_per_mode == 1
    assert request.gun.bore_step_mm == 10.
    assert request.inelastic.method == default.inelastic.method
    assert request.inelastic.seed == default.inelastic.seed
    assert panel.result is previous and panel._stale
    assert not panel._session_ready and panel._worker is None
    assert panel._state.electron_gun.emitter.virtual_source_fwhm_nm == 5.


@pytest.mark.parametrize("surface", [False, True])
def test_working_caps_and_stage_peak_are_independent_of_large_ram_cache(panel, surface):
    if surface:
        model = SimpleNamespace(coherence=SimpleNamespace(mean_energy_ev=.3,
            energy_rms_ev=.1, edge_phase_rad=0.),
            geometry=SimpleNamespace(apex_radius_nm=30.), current_na=12.)
        panel.set_state(instrument(model))
    panel.working_gib.setValue(1.)
    panel.ram_cache_gib.setValue(80.)
    request = panel._make_request()
    assert request.gun.maximum_checkpoint_bytes == module.GIB
    assert request.surface.maximum_working_bytes == module.GIB
    assert request.radial_gun.maximum_working_bytes == module.GIB
    assert request.wave_grid.maximum_working_bytes == module.GIB
    assert request.maximum_readout_bytes == module.GIB
    assert request.execution.maximum_ram_cache_bytes == 80*module.GIB
    claim = module._wave_resource_claim(panel._state, request)
    # Surface FEM and effective radial load overlap; Gaussian stages are
    # sequential. A segmented request can add one checkpoint of <=1 GiB.
    stage_peak = (2 if surface else 1)*module.GIB
    assert claim.working_bytes == stage_peak+module.GIB+256*1024**2
    assert claim.working_bytes < 4*module.GIB


def test_cache_growth_claim_respects_small_explicit_retention_limit(panel):
    panel.working_gib.setValue(2.)
    panel.ram_cache_gib.setValue(.25)
    request = panel._make_request()
    claim = module._wave_resource_claim(panel._state, request)
    assert claim.working_bytes == 2*module.GIB+2*256*1024**2


def test_complete_source_modes_above_working_budget_are_rejected_before_job(panel, monkeypatch):
    monkeypatch.setattr(module, "wave_input_summary", lambda *_args:
        {"status": "SOURCE_READY", "model": "GUI-only fixture", "mode_count": 64,
         "minimum_initial_wave_bytes": 2*module.GIB})
    panel.working_gib.setValue(1.)
    panel.source_enabled.setChecked(True)
    panel.calculate()
    assert "Initial complete mode fields need at least" in panel.status.text()
    assert "no source modes were removed" in panel.status.text()
    assert panel._worker is None
    assert panel.result is None


def test_idle_wave_cache_is_inventoried_for_other_shared_job_owners(panel, qtbot):
    calculate(panel, qtbot)
    coordinator = panel.pool.coordinator
    assert any(owner() is panel and method == "_retained_wave_roots"
               for owner, method in coordinator._providers)
    before = coordinator._retained_bytes()
    array = np.zeros(100_000)
    panel.result.checkpoint.test_array = array
    after = coordinator._retained_bytes()
    assert after-before >= array.nbytes
    # The same checkpoint is referenced by result and its exact-Z cache.
    # Adding another reference is ownership aliasing, not another wave array.
    panel._cache[101.] = (panel.result, panel.preview)
    aliased = coordinator._retained_bytes()
    assert aliased-after < array.nbytes


def test_worker_accounts_for_retained_exact_displays_as_inventory_roots(panel, qtbot, monkeypatch):
    calculate(panel, qtbot)
    old_result, old_preview = panel.result, panel.preview
    monkeypatch.setattr(panel.pool, "start", lambda _worker: None)
    panel.set_selected_z(107.)
    panel.timer.stop()
    panel._start_query()
    retained = panel._worker.existing_result
    assert retained[-3:-1] == (old_result, old_preview)
    assert retained[-1] is panel._observation_session
    assert panel._worker.observer is panel._observation_session
    assert retained[0] == (old_result, old_preview)
    assert panel._worker.resource_claim.working_bytes > panel._request.wave_grid.maximum_working_bytes
    panel.cancel()


def test_worker_progress_is_throttled_and_cancelled_progress_suppressed(panel, monkeypatch):
    monkeypatch.setattr(module, "monotonic", lambda: 1.)
    request = panel._make_request()
    worker = module._WaveWorker(1, 1, instrument(), request, Event())
    messages = []
    worker.signals.progress.connect(lambda generation, text: messages.append((generation, text)))
    for index in range(100):
        worker._report_progress(index, 100, "Field propagation")
    worker._report_progress(100, 100, "Complete")
    assert len(messages) == 2
    assert messages[-1][1] == "Complete (100/100)"
    worker.event.set()
    worker._report_progress(100, 100, "Cancelled")
    assert len(messages) == 2


def test_cache_limit_and_old_session_publication_guard(panel, qtbot):
    calculate(panel, qtbot)
    for index in range(panel.CACHE_LIMIT+12):
        z_mm = 120.+index
        panel._target_z_mm = z_mm
        panel._solved(panel._generation, panel._session, result(z_mm), preview())
    assert len(panel._cache) == panel.CACHE_LIMIT
    assert 120. not in panel._cache
    old = panel.result
    panel._solved(panel._generation, panel._session-1, result(139.), preview())
    assert panel.result is old


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "bad", "101.5", None, True])
def test_invalid_linked_z_is_ignored(panel, value):
    panel.set_selected_z(value)
    assert panel._target_z_mm == 100.
    assert not panel.timer.isActive()


def test_first_observation_target_is_specimen_exit(panel):
    state = instrument()
    state.sample.thickness_nm = 5.
    panel.set_state(state)
    expected = 100.+5.*.5e-6
    assert panel._target_z_mm == expected
    assert panel.target_z.value() == expected
    assert panel._make_request().observation_z_mm == expected
    assert "specimen exit" in panel.target_z.toolTip()


def test_missing_tip_fields_are_explicit_and_do_not_raise_on_publication(panel):
    panel.set_state(SimpleNamespace(electron_gun=SimpleNamespace(type_key="fixture")))
    assert "No physical tip input fields" in panel.source_info.text()
    assert panel._stale


def test_active_plane_finishes_and_only_latest_pending_plane_is_queued(panel, qtbot, monkeypatch):
    calculate(panel, qtbot)
    assert panel.plane_positions.text() == "Requested Z 100 mm | Calculating Z — | Displayed Z 100 mm"
    started, released = Event(), Event()
    latest_started, latest_released = Event(), Event()
    calls = []

    def solve(state, request, *, cancelled, **_kwargs):
        calls.append((request.observation_z_mm, cancelled))
        if request.observation_z_mm == 105.:
            started.set()
            released.wait(2.)
        elif request.observation_z_mm == 125.:
            latest_started.set()
            latest_released.wait(2.)
        return result(request.observation_z_mm)

    monkeypatch.setattr(module, "simulate_tip_wave", solve)
    panel.set_selected_z(105.)
    qtbot.waitUntil(started.is_set)
    for z_mm in range(106, 126):
        panel.set_selected_z(float(z_mm))
    qtbot.wait(230)
    assert [z for z, _ in calls] == [105.]
    assert not calls[0][1]()
    assert not any(job.owner is panel.pool for job in panel.pool.coordinator.queue)
    assert panel.plane_positions.text() == "Requested Z 125 mm | Calculating Z 105 mm | Displayed Z 100 mm"
    panel._progress(panel._generation, "Phase representation at 102 mm (10/56)")
    assert panel.plane_positions.text() == "Requested Z 125 mm | Calculating Z 105 mm | Displayed Z 100 mm"
    try:
        released.set()
        qtbot.waitUntil(latest_started.is_set)
        assert panel.result.checkpoint.plane_z_mm == 105.
        assert "105" in panel.plot.getPlotItem().titleLabel.text
        assert "125" in panel.status.text()
        assert "105" in panel.status.text()
        assert [z for z, _ in calls] == [105., 125.]
        assert panel.plane_positions.text() == "Requested Z 125 mm | Calculating Z 125 mm | Displayed Z 105 mm"
    finally:
        released.set()
        latest_released.set()
    qtbot.waitUntil(lambda: panel.result.checkpoint.plane_z_mm == 125.)
    assert [z for z, _ in calls] == [105., 125.]
    qtbot.waitUntil(lambda: panel._worker is None)
    assert panel.plane_positions.text() == "Requested Z 125 mm | Calculating Z — | Displayed Z 125 mm"


def test_z_motion_during_first_calculation_keeps_authorized_session(panel, qtbot, monkeypatch):
    started, released = Event(), Event()
    calls = []

    def solve(_state, request, *, cancelled, **_kwargs):
        calls.append((request.observation_z_mm, cancelled))
        if len(calls) == 1:
            started.set()
            released.wait(2.)
        return result(request.observation_z_mm)

    monkeypatch.setattr(module, "simulate_tip_wave", solve)
    panel.source_enabled.setChecked(True)
    panel.calculate()
    qtbot.waitUntil(started.is_set)
    try:
        panel.set_selected_z(101.)
        panel.set_selected_z(102.)
        assert not calls[0][1]()
        assert not panel._stale
        assert panel.result is None
    finally:
        released.set()
    qtbot.waitUntil(lambda: panel.result is not None
                      and panel.result.checkpoint.plane_z_mm == 102.)
    assert [z for z, _ in calls] == [100., 102.]
    assert 100. in panel._cache


def test_requested_cached_plane_takes_precedence_over_late_intermediate(panel, qtbot, monkeypatch):
    calculate(panel, qtbot)
    cached_result = panel.result
    started, released = Event(), Event()
    calls = []

    def solve(_state, request, *, cancelled, **_kwargs):
        calls.append((request.observation_z_mm, cancelled))
        started.set()
        released.wait(2.)
        return result(request.observation_z_mm)

    monkeypatch.setattr(module, "simulate_tip_wave", solve)
    panel.set_selected_z(105.)
    qtbot.waitUntil(started.is_set)
    try:
        panel.set_selected_z(100.)
        assert panel.result is cached_result
        assert not calls[0][1]()
    finally:
        released.set()
    qtbot.waitUntil(lambda: panel._worker is None)
    assert panel.pool.waitForDone(2000)
    assert panel.result is cached_result
    assert panel._cache[105.][0].checkpoint.plane_z_mm == 105.
    assert [z for z, _ in calls] == [105.]
    assert "100" in panel.plot.getPlotItem().titleLabel.text


@pytest.mark.parametrize("stop", ["cancel", "mark_inputs_stale"])
def test_hard_stop_discards_pending_z_and_rejects_late_output(panel, qtbot, monkeypatch, stop):
    calculate(panel, qtbot)
    old = panel.result
    started, released = Event(), Event()
    calls = []

    def solve(_state, request, *, cancelled, **_kwargs):
        calls.append((request.observation_z_mm, cancelled))
        started.set()
        released.wait(2.)
        return result(request.observation_z_mm)

    monkeypatch.setattr(module, "simulate_tip_wave", solve)
    panel.set_selected_z(105.)
    qtbot.waitUntil(started.is_set)
    worker, context = panel._worker, panel._observation_session
    try:
        panel.set_selected_z(106.)
        getattr(panel, stop)()
        assert calls[0][1]()
        # The shared worker still owns its prepared context until it returns,
        # even when invalidation has dropped the page's reference.
        assert worker.observer is context and context in worker.existing_result
        assert (panel._observation_session is context) == (stop == "cancel")
        stopped_status = panel.status.text()
    finally:
        released.set()
    assert panel.pool.waitForDone(2000)
    qtbot.wait(100)
    assert panel.result is old
    assert panel.status.text() == stopped_status
    assert [z for z, _ in calls] == [105.]
    assert not panel.timer.isActive()


def test_browsing_range_and_slider_do_not_change_session_physics(panel, qtbot, monkeypatch):
    assert panel.z_range_start_label.text() == "Start"
    assert panel.z_range_end_label.text() == "End"
    assert panel.z_range_min.accessibleName() == "Z slider range start"
    assert panel.z_range_max.accessibleName() == "Z slider range end"
    calculate(panel, qtbot)
    captured, request, generation = panel._captured, panel._request, panel._generation
    calls = []
    monkeypatch.setattr(module, "simulate_tip_wave", lambda state, requested, **_kwargs:
        (calls.append((state, requested)), result(requested.observation_z_mm))[1])
    panel.follow_ray.setChecked(False)
    panel.z_range_min.setValue(90.)
    panel.z_range_max.setValue(110.)
    assert panel._target_z_mm == 100.
    assert panel._captured is captured and panel._request is request
    assert not panel._stale and panel._generation == generation
    assert panel.z_slider.minimum() == 0 and panel.z_slider.maximum() == 10000
    assert panel.z_slider.value() == 5000
    qtbot.wait(100)
    assert calls == []
    panel.z_slider.setValue(7500)
    qtbot.waitUntil(lambda: panel.result.checkpoint.plane_z_mm == 105.)
    assert panel._target_z_mm == panel.target_z.value() == 105.
    assert len(calls) == 1 and calls[0][0] is captured
    assert panel._request is request and not panel._stale


def test_observation_context_is_lazy_reused_for_z_and_replaced_by_calculate(panel, qtbot):
    assert panel._observation_session is None
    panel.set_selected_z(101.)
    panel.source_enabled.setChecked(True)
    assert panel._observation_session is None
    calculate(panel, qtbot)
    original = panel._observation_session
    assert original.state is panel._captured
    assert original.request is panel._request
    assert original.observed_z == [101.]
    panel.set_selected_z(102.)
    qtbot.waitUntil(lambda: panel.result.checkpoint.plane_z_mm == 102.)
    qtbot.waitUntil(lambda: panel._worker is None)
    assert panel._observation_session is original
    assert original.observed_z == [101., 102.]
    panel.cancel()
    assert panel._observation_session is original
    panel.set_selected_z(103.)
    qtbot.waitUntil(lambda: panel.result.checkpoint.plane_z_mm == 103.)
    qtbot.waitUntil(lambda: panel._worker is None)
    assert panel._observation_session is original
    calculate(panel, qtbot)
    assert panel._observation_session is not original
    assert panel._observation_session.observed_z == [103.]


@pytest.mark.parametrize("invalidate", ["source", "instrument", "shutdown"])
def test_observation_context_is_released_when_inputs_or_owner_are_invalidated(panel, qtbot, invalidate):
    calculate(panel, qtbot)
    context = panel._observation_session
    if invalidate == "source":
        panel.tip_energy.setValue(panel.tip_energy.value()+1.)
    elif invalidate == "instrument":
        panel.set_state(instrument())
    else:
        assert panel.shutdown()
    assert panel._observation_session is None
    assert context not in panel._retained_wave_roots()
    assert panel._worker is None
    assert not panel.timer.isActive()


def test_idle_observation_context_is_part_of_shared_memory_inventory(panel, qtbot):
    calculate(panel, qtbot)
    coordinator = panel.pool.coordinator
    context = panel._observation_session
    assert context in panel._retained_wave_roots()
    before = coordinator._retained_bytes()
    context.executed_array = np.zeros(100_000)
    after = coordinator._retained_bytes()
    assert after-before >= context.executed_array.nbytes


def test_return_to_inflight_plane_does_not_recompute_completed_plane(panel, qtbot, monkeypatch):
    calculate(panel, qtbot)
    started, released = Event(), Event()
    calls = []

    def solve(_state, request, *, cancelled, **_kwargs):
        calls.append((request.observation_z_mm, cancelled))
        started.set()
        released.wait(2.)
        return result(request.observation_z_mm)

    monkeypatch.setattr(module, "simulate_tip_wave", solve)
    panel.set_selected_z(105.)
    qtbot.waitUntil(started.is_set)
    try:
        panel.set_selected_z(106.)
        panel.set_selected_z(105.)
        assert not calls[0][1]()
    finally:
        released.set()
    qtbot.waitUntil(lambda: panel._worker is None)
    qtbot.wait(100)
    assert [z for z, _ in calls] == [105.]
    assert panel.result.checkpoint.plane_z_mm == 105.
    assert not panel.timer.isActive()


def test_workspace_ray_z_wiring_is_lazy_before_first_calculation(qtbot, monkeypatch):
    from test_incremental_ray_scene import _result
    from temsim.gui.visualization import VisualizationWorkspace

    view = VisualizationWorkspace()
    qtbot.addWidget(view)
    monkeypatch.setattr(view, "_update_interaction_detail", lambda: None)
    monkeypatch.setattr(view.selected_plane_readout, "select_z", lambda _z: None)
    calls = []
    monkeypatch.setattr(module, "simulate_tip_wave", lambda *_args, **_kwargs: calls.append(1))
    view.display_result(_result(), "Preview")
    view.show()
    assert view.tabs.tabText(view.tabs.indexOf(view.coherent_beam)) == "Coherent beam"
    assert view.tabs.currentWidget() is not view.coherent_beam
    assert view.ray_beam_tabs.currentWidget() is view.transverse_beam
    view.ray_beam_tabs.setCurrentWidget(view.coherent_ray_view)
    assert view.plot.isVisible() and view.coherent_ray_view.screen.isVisible()
    assert not view.transverse_beam.isVisible()
    assert view.coherent_ray_view.follow_ray.isChecked()
    view.jump_to_ray_position(15., activate_tab=False)
    assert view.coherent_beam._target_z_mm == 15.
    view.axial_cursor_item.setValue(16.)
    assert view.coherent_beam._target_z_mm == 16.
    qtbot.wait(230)
    assert calls == []
    assert not view.coherent_beam.timer.isActive()
    assert not view.coherent_beam.source_enabled.isChecked()
    view.transverse_beam_toggle.setChecked(False)
    assert not view.ray_beam_tabs.isVisible()
    view.transverse_beam_toggle.setChecked(True)
    assert view.coherent_ray_view.isVisible()
    view.mark_ray_stale(SimpleNamespace(electron_gun=SimpleNamespace(
        type_key="fixture", display_name="Electron source"), apertures=()))
    assert "No physical tip input fields" in view.coherent_beam.source_info.text()
    assert view.coherent_beam.shutdown()
    assert view.selected_plane_readout.shutdown()


def test_mirror_shares_one_session_and_retains_completed_image_while_pending(panel, qtbot, monkeypatch):
    calculate(panel, qtbot)
    mirror = module.CoherentBeamMirror(panel)
    qtbot.addWidget(mirror)
    mirror.show()
    panel.hide()
    qtbot.waitUntil(lambda: mirror.screen.image.image is not None)
    initial = mirror.screen.image.image
    initial_main = panel.image.image
    calls = []
    released = Event()

    def solve(_state, request, **kwargs):
        calls.append(request.observation_z_mm)
        assert released.wait(5.)
        assert not kwargs["cancelled"]()
        return result(request.observation_z_mm)

    monkeypatch.setattr(module, "simulate_tip_wave", solve)
    panel.set_selected_z(101.)
    qtbot.waitUntil(lambda: calls == [101.])
    try:
        panel.set_selected_z(102.)
        panel.set_selected_z(103.)
        assert mirror.screen.image.image is initial
        assert "103" in mirror.status.text() and "100" in mirror.status.text()
        assert mirror.plane_positions.text() == "Requested Z 103 mm | Calculating Z 101 mm | Displayed Z 100 mm"
    finally:
        released.set()
    qtbot.waitUntil(lambda: panel.result.checkpoint.plane_z_mm == 103.)
    qtbot.waitUntil(lambda: panel._worker is None)
    assert calls == [101., 103.]
    assert "103" in mirror.screen.plot.getPlotItem().titleLabel.text
    assert mirror.screen._pending[1] is panel.preview
    assert panel.image.image is initial_main  # Hidden page has no redundant upload.
    panel.show()
    qtbot.waitUntil(lambda: "103" in panel.plot.getPlotItem().titleLabel.text)
    assert mirror.readout.text() == panel.readout.text()


def test_workspace_continuous_ray_cursor_updates_visible_coherent_without_particle_redraw(panel, qtbot, monkeypatch):
    from test_incremental_ray_scene import _result
    from temsim.gui.visualization import VisualizationWorkspace

    monkeypatch.setattr(module, "CoherentBeamPage", lambda: panel)
    view = VisualizationWorkspace()
    qtbot.addWidget(view)
    monkeypatch.setattr(view, "_update_interaction_detail", lambda: None)
    monkeypatch.setattr(view.selected_plane_readout, "select_z", lambda _z: None)
    view.display_result(_result(), "Preview")
    view.show()
    view.ray_beam_tabs.setCurrentWidget(view.coherent_ray_view)
    view.jump_to_ray_position(11., activate_tab=False)
    view.tabs.setCurrentWidget(panel)
    calculate(panel, qtbot)
    view.show_ray_diagram()
    qtbot.waitUntil(lambda: view.coherent_ray_view.screen.image.image is not None)
    redraws, calls = [], []
    monkeypatch.setattr(view.transverse_beam, "_redraw", lambda: redraws.append(1))
    monkeypatch.setattr(module, "simulate_tip_wave", lambda _state, request, **_kwargs:
        (calls.append(request.observation_z_mm), result(request.observation_z_mm))[1])
    motion = QTimer(view)
    motion.setInterval(10)
    motion.timeout.connect(lambda: view.axial_cursor_item.setValue(
        12.+((view.axial_cursor_item.value()-12.+.025) % 2.)))
    motion.start()
    try:
        qtbot.wait(220)
        qtbot.waitUntil(lambda: panel.result.checkpoint.plane_z_mm != 11.)
        assert motion.isActive()
        assert view.plot.isVisible() and view.coherent_ray_view.screen.isVisible()
        assert not panel.isVisible()
    finally:
        motion.stop()
    latest = float(view.axial_cursor_item.value())
    qtbot.waitUntil(lambda: panel.result.checkpoint.plane_z_mm == latest)
    qtbot.waitUntil(lambda: panel._worker is None)
    assert calls[-1] == latest and redraws == []
    assert view.coherent_ray_view.screen._pending[0] is panel.result
    assert str(f"{latest:.9g}") in view.coherent_ray_view.screen.plot.getPlotItem().titleLabel.text
    assert view.selected_plane_readout.shutdown()


def test_mirror_follow_scale_cancel_and_stale_share_owner_without_new_source(panel, qtbot, monkeypatch):
    calculate(panel, qtbot)
    mirror = module.CoherentBeamMirror(panel)
    qtbot.addWidget(mirror)
    mirror.show()
    calls = []
    monkeypatch.setattr(module, "simulate_tip_wave", lambda *_args, **_kwargs: calls.append(1))
    old_session = panel._observation_session
    old_result = panel.result
    mirror.follow_ray.setChecked(False)
    assert not panel.follow_ray.isChecked()
    panel.set_selected_z(101.)
    assert panel._target_z_mm == 100.
    panel.follow_ray.setChecked(True)
    assert mirror.follow_ray.isChecked()
    mirror.intensity_scale.setCurrentIndex(1)
    assert panel.intensity_scale.currentIndex() == 1
    assert "log display" in mirror.screen.plot.getPlotItem().titleLabel.text
    assert panel._observation_session is old_session and panel.result is old_result
    panel.cancel()
    assert "cancelled" in mirror.status.text()
    assert mirror.plane_positions.text() == panel.plane_positions.text()
    assert "Calculating Z —" in mirror.plane_positions.text()
    panel.mark_inputs_stale()
    assert "Inputs changed" in mirror.status.text() and "Previous optics" in mirror.status.text()
    assert mirror.screen._pending[0] is old_result
    qtbot.wait(100)
    assert calls == []


def test_intensity_view_skips_duplicate_upload_and_preserves_zoom_for_same_domain(qtbot, monkeypatch):
    view = module.CoherentIntensityView()
    qtbot.addWidget(view)
    view.show()
    first, data = result(100.), preview()
    view.set_observation(first, data, 0)
    uploads = []
    original = view.image.setImage
    monkeypatch.setattr(view.image, "setImage", lambda *args, **kwargs:
                        (uploads.append(1), original(*args, **kwargs))[1])
    view.plot.setRange(xRange=(-.1, .1), yRange=(-.1, .1), padding=0.)
    bounds = view.plot.getViewBox().viewRange()
    view.set_observation(first, data, 0)
    assert uploads == []
    view.set_observation(result(101.), preview(), 0)
    assert uploads == [1]
    np.testing.assert_allclose(view.plot.getViewBox().viewRange(), bounds)


@pytest.mark.parametrize("scale", (0, 1))
def test_intensity_view_keeps_physical_viewport_when_z_changes_wave_domain(qtbot, scale):
    view = module.CoherentIntensityView()
    qtbot.addWidget(view)
    view.resize(600, 500)
    view.show()
    view.set_observation(result(100.), preview(), scale)
    qtbot.wait(20)
    view.plot.setRange(xRange=(-.2, .3), yRange=(-.1, .4), padding=0.)
    fixed_range = np.array(view.plot.getViewBox().viewRange())
    # A later Z may have a larger, shifted or smaller physical wave grid.
    for z_mm, bounds in ((101., ((-20., 30.), (-10., 40.))),
                         (102., ((-.02, .03), (-.01, .04)))):
        data = replace(preview(), bounds_um=np.array(bounds))
        view.set_observation(result(z_mm), data, scale)
        qtbot.wait(20)
        np.testing.assert_allclose(view.plot.getViewBox().viewRange(), fixed_range,
                                   rtol=0., atol=1e-12)
        rect = view.image.mapRectToParent(view.image.boundingRect())
        np.testing.assert_allclose((rect.left(), rect.right(), rect.top(), rect.bottom()),
                                   np.asarray(bounds).ravel())
        assert f"Exact Z {z_mm:.9g} mm" in view.plot.getPlotItem().titleLabel.text


def test_both_coherent_screens_fit_only_on_explicit_request(panel, qtbot, monkeypatch):
    calculate(panel, qtbot)
    mirror = module.CoherentBeamMirror(panel)
    qtbot.addWidget(mirror)
    mirror.show()
    qtbot.wait(20)
    original_result, original_session = panel.result, panel._observation_session
    calls = []
    monkeypatch.setattr(module, "simulate_tip_wave", lambda *_args, **_kwargs: calls.append(1))
    for host in (panel, mirror):
        view = host.screen
        view.plot.setRange(xRange=(-.1, .1), yRange=(-.1, .1), padding=0.)
        qtbot.mouseClick(view.fit_button, Qt.MouseButton.LeftButton)
        fitted = np.asarray(view.plot.getViewBox().viewRange())
        assert np.all(fitted[:, 0] <= preview().bounds_um[:, 0])
        assert np.all(fitted[:, 1] >= preview().bounds_um[:, 1])
        assert not any(view.plot.getViewBox().autoRangeEnabled())
    assert calls == [] and panel._worker is None
    assert panel.result is original_result and panel._observation_session is original_session


@pytest.mark.parametrize("scale", (0, 1))
def test_coherent_screen_fits_narrow_ray_sidebar_without_clipping(qtbot, scale):
    view = module.CoherentIntensityView()
    qtbot.addWidget(view)
    view.resize(332, 520)
    view.show()
    view.set_observation(result(1605.193957), preview(), scale)
    qtbot.wait(40)
    viewport = view.plot.viewport().rect()
    screen = view.plot.getViewBox().sceneBoundingRect()
    assert screen.right() <= viewport.right()+2
    assert screen.left() >= viewport.left()
    ranges = np.asarray(view.plot.getViewBox().viewRange())
    assert np.all(ranges[:, 0] <= -.5) and np.all(ranges[:, 1] >= .5)


def test_streamed_intensity_preview_reads_complex_modes_only_once():
    def mode(weight, shift, shape):
        return SimpleNamespace(weight_per_reference_electron=weight,
            plane=SimpleNamespace(amplitude=np.full(shape, 1j/np.sqrt(np.prod(shape))),
                basis_m=np.eye(2)*1e-6, origin_m=np.array((shift, 0.))*1e-6,
                curvature_m1=np.zeros((2, 2)), tilt_rad=np.zeros(2)))

    original = (mode(.3, 0., (4, 4)), mode(.7, 1., (8, 4)))

    class StoredModes:
        reads = 0
        geometry_reads = 0

        def geometries(self):
            for value in original:
                self.geometry_reads += 1
                yield value.plane.amplitude.shape, value.plane.basis_m, value.plane.origin_m

        def __iter__(self):
            for value in original:
                self.reads += 1
                yield value

    modes = StoredModes()
    streamed = module._intensity_preview(SimpleNamespace(beam=SimpleNamespace(
        modes=modes, content_identity="executed-test-modes")), bins=8)
    resident = module._intensity_preview(SimpleNamespace(beam=SimpleNamespace(modes=original)), bins=8)
    assert modes.geometry_reads == 2 and modes.reads == 2
    assert streamed.mode_count == 2 and streamed.probability == pytest.approx(1.)
    assert streamed.grid_shapes == ((4, 4), (8, 4))
    assert resident.grid_shapes == streamed.grid_shapes
    np.testing.assert_array_equal(streamed.density, resident.density)
    np.testing.assert_array_equal(streamed.bounds_um, resident.bounds_um)
    assert streamed.retained_bytes == streamed.density.nbytes+streamed.bounds_um.nbytes
    assert resident.retained_bytes > streamed.retained_bytes


def test_executed_wave_grid_readout_is_distinct_from_display_bins(panel):
    frame = replace(preview(), grid_shapes=((512, 1024), (2048, 2048)))
    panel._display(result(100.), frame)
    assert "Displayed wave grids: 1024 × 512; 2048 × 2048 cells" in panel.readout.text()
    assert "4 × 4 display bins" in panel.readout.text()


def test_shutdown_cannot_be_reopened_by_late_state_or_button(panel, qtbot):
    calculate(panel, qtbot)
    old = panel.result
    assert panel.shutdown()
    panel.set_state(instrument())
    panel.source_enabled.setChecked(False)
    panel.set_selected_z(120.)
    panel.calculate()
    panel._solved(panel._generation, panel._session, result(100.), preview())
    qtbot.wait(220)
    assert panel._closed
    assert panel.result is old
    assert panel._worker is None
    assert not panel.timer.isActive()
