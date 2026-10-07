"""Offline GUI fixtures, not physical-source or image acceptance."""
from dataclasses import replace

import numpy as np
import pytest

from temsim.gui.diagnostic_tabs import TransverseBeamView
from temsim.physics.tip_gun_wave import TipGunCheckpoint
from temsim.physics.wave_flux import BeamState
from temsim.optics.electron_gun.tip_coherence import TIP_REFERENCE
from test_wave_plane_observables import mode
from test_beam_analysis_modes import make_result


@pytest.fixture
def view(qtbot):
    widget = TransverseBeamView()
    qtbot.addWidget(widget)
    widget.resize(500, 900)
    widget.show()
    widget._test_qtbot = qtbot
    yield widget
    if widget.analysis.wave is not None:
        widget.analysis.wave.pool.clear()
        assert widget.analysis.wave.pool.waitForDone(5000)


def wait_for_readout(view):
    view._test_qtbot.waitUntil(lambda: not view.analysis.wave.busy, timeout=20000)


def display(view, checkpoint, **kwargs):
    view.display_wave_checkpoint(checkpoint, **kwargs)
    wait_for_readout(view)


def checkpoint():
    first = mode()
    first = replace(first, plane=replace(first.plane, curvature_m1=None, tilt_rad=None))
    second = replace(first, mode_id="fixture:two", weight_per_reference_electron=.2,
        plane=replace(first.plane, amplitude=first.plane.amplitude*1j),
        scattering_history=({"kind": "plasmon", "loss_ev": 16.},))
    return TipGunCheckpoint(BeamState((first, second), TIP_REFERENCE), 1500., 100e-9,
                            {"scope": "OFFLINE GUI FIXTURE NOT SOURCE"})


def switch(view, value):
    view.analysis.mode_combo.setCurrentIndex(view.analysis.mode_combo.findData(value))
    wait_for_readout(view)


def test_same_panel_current_and_angles_keep_absolute_tip_weights(view, monkeypatch):
    import temsim.physics.tip_wave_pipeline as pipeline
    monkeypatch.setattr(pipeline, "simulate_tip_wave", lambda *a, **k: pytest.fail("GUI must not propagate"))
    original = checkpoint()
    digest = original.digest
    display(view, original, axial_bz_t=.5)
    assert view.analysis.wave.values.sum() == pytest.approx(50000., rel=1e-12)
    assert view.initial_beam_panel.isHidden()
    switch(view, "wave_angles")
    assert view.analysis.wave.values.sum() == pytest.approx(50000., rel=1e-12)
    assert "Canonical" in view.analysis.legend.text()
    assert view.plot.getAxis("bottom").labelUnits == "mrad"
    assert original.digest == digest


def test_phase_selects_one_mode_and_keeps_relative_phase(view):
    display(view, checkpoint(), axial_bz_t=0.)
    switch(view, "wave_phase")
    assert view.analysis.colour_combo.currentData() == 0
    first = view.analysis.wave.values.copy()
    view.analysis.colour_combo.setCurrentIndex(2)
    wait_for_readout(view)
    np.testing.assert_allclose(np.exp(1j*view.analysis.wave.values), 1j*np.exp(1j*first), atol=1e-12)
    view.analysis.colour_combo.setCurrentIndex(0)
    wait_for_readout(view)
    assert view.analysis.colour_combo.currentData() == 0  # No aggregate phase option accepted.


def test_missing_plane_does_not_show_previous_wave_or_recompute(view):
    original = checkpoint()
    display(view, original, axial_bz_t=0.)
    view.focus_z(1499.)
    assert "not cached" in view.summary.text()
    assert not view.plot.getPlotItem().items
    view.focus_z(1500.)
    wait_for_readout(view)
    assert view.analysis.wave.values.sum() == pytest.approx(50000.)
    assert view.analysis.wave.checkpoint is original


def test_manual_scale_remains_fixed_for_rotation_and_plane_return(view):
    display(view, checkpoint(), axial_bz_t=0.)
    view.plot.setRange(xRange=(-.001, .001), yRange=(-.001, .001), padding=0.)
    view._manual_view_range_changed([True, True])
    wait_for_readout(view)
    before = np.array(view.plot.viewRange())
    visible = view.analysis.wave.values.sum()
    assert visible < 50000.
    view.set_projection_angle(90.)
    view.focus_z(1501.)
    view.focus_z(1500.)
    wait_for_readout(view)
    np.testing.assert_allclose(view.plot.viewRange(), before, rtol=1e-10, atol=1e-14)
    view.fit_beam.click()
    wait_for_readout(view)
    assert view.analysis.wave.values.sum() == pytest.approx(50000.)


def test_interactions_use_executed_weights_and_old_ray_view_can_be_restored(view):
    old = make_result()
    view.display_result(old)
    display(view, checkpoint(), axial_bz_t=0.)
    switch(view, "wave_interactions")
    assert view.analysis.table.rowCount() == 2
    assert view.analysis.table.item(0, 1).text() == "30"
    assert view.analysis.table.item(1, 2).text() == "20000"
    view.display_result(old, focus=("z", 1.))
    assert view.analysis.wave is None
    assert view._result is old
    assert view._scatter is not None
    assert view.analysis.colour_label.text() == "Colour by"


def test_new_checkpoint_retains_user_current_scale(view):
    original = checkpoint()
    display(view, original, axial_bz_t=0.)
    view.plot.setRange(xRange=(-.001, .001), yRange=(-.001, .001), padding=0.)
    view._manual_view_range_changed([True, True])
    wait_for_readout(view)
    before = np.array(view.plot.viewRange())
    display(view, replace(original, plane_z_mm=1501.), axial_bz_t=.1)
    np.testing.assert_allclose(view.plot.viewRange(), before, rtol=1e-10, atol=1e-14)


def test_display_budget_failure_does_not_erase_executed_wave(view):
    original = checkpoint()
    display(view, original, axial_bz_t=0., maximum_working_bytes=1)
    assert "memory budget" in view.summary.text()
    assert view.analysis.wave.checkpoint is original


@pytest.mark.parametrize("angle", (0., 31., 90.))
def test_coarse_uniform_wave_is_not_drawn_as_sparse_point_markers(view, angle):
    original = checkpoint()
    view.set_projection_angle(angle)
    display(view, original, axial_bz_t=0.)
    values = view.analysis.wave.values
    # Fixture has a continuous, uniformly sampled 32 x 32 plane. Its
    # central display must not claim the 128 x 128 grid's false zero holes.
    nx, ny = values.shape
    centre = values[nx//2-3:nx//2+3, ny//2-3:ny//2+3]
    assert centre.min() > 0
    np.testing.assert_allclose(centre, centre.mean(), rtol=1e-10)
    assert values.sum() == pytest.approx(50000., rel=1e-12)
    assert view.analysis.wave.checkpoint is original


def test_current_hover_uses_the_actual_adaptive_bin_shape(view):
    from PySide6.QtCore import QPointF
    display(view, checkpoint(), axial_bz_t=0.)
    wave = view.analysis.wave
    values, xe, ye = wave._hover
    assert len(xe) == values.shape[0]+1 and len(ye) == values.shape[1]+1
    position = view.plot.getViewBox().mapViewToScene(QPointF(0., 0.))
    wave.mouse_moved(position)
    assert "pA / bin" in view.analysis.readout.text()


@pytest.mark.parametrize("angle, expected_mrad", ((0., (.2, -.4)), (90., (-.4, -.2))))
def test_flow_sums_mixed_mode_currents_before_dividing_and_rotates_vectors(view, angle, expected_mrad):
    from PySide6.QtCore import QPointF
    original = checkpoint()
    first, second = original.beam.modes
    amplitude = np.ones_like(first.plane.amplitude)/32
    first = replace(first, plane=replace(first.plane, amplitude=amplitude, tilt_rad=np.array((.001, -.002))))
    second = replace(second, plane=replace(second.plane, amplitude=1j*amplitude, tilt_rad=-first.plane.tilt_rad))
    original = replace(original, beam=BeamState((first, second), TIP_REFERENCE))
    digest = original.digest
    view.set_projection_angle(angle)
    display(view, original, axial_bz_t=0.)
    switch(view, "wave_flow")
    wave = view.analysis.wave
    assert wave.values.sum() == pytest.approx(50000., rel=1e-12)
    np.testing.assert_allclose(wave.flow_current_pA.sum(axis=(1, 2)), np.asarray(expected_mrad)*50,
                               rtol=1e-12, atol=1e-12)
    valid = np.isfinite(wave.flow_slopes_mrad).all(axis=0)
    np.testing.assert_allclose(wave.flow_slopes_mrad[:, valid],
                               np.broadcast_to(np.asarray(expected_mrad)[:, None], (2, valid.sum())),
                               rtol=1e-12, atol=1e-12)
    wave.mouse_moved(view.plot.getViewBox().mapViewToScene(QPointF(0., 0.)))
    assert "mean flow" in view.analysis.readout.text()
    assert "not measured particle paths" in view.analysis.legend.text()
    assert original.digest == digest


def test_opposite_equal_mode_currents_cancel_without_inventing_a_mixture_phase(view):
    original = checkpoint()
    first, second = original.beam.modes
    amplitude = np.ones_like(first.plane.amplitude)/32
    first = replace(first, plane=replace(first.plane, amplitude=amplitude, tilt_rad=np.array((.001, 0.))))
    second = replace(second, weight_per_reference_electron=first.weight_per_reference_electron,
        plane=replace(second.plane, amplitude=-amplitude, tilt_rad=-first.plane.tilt_rad))
    original = replace(original, beam=BeamState((first, second), TIP_REFERENCE))
    display(view, original, axial_bz_t=0.)
    switch(view, "wave_flow")
    wave = view.analysis.wave
    assert wave.values.sum() == pytest.approx(60000., rel=1e-12)
    np.testing.assert_allclose(wave.flow_current_pA, 0., atol=1e-13)
    assert "longest arrow 0 mrad" in view.analysis.legend.text()
    switch(view, "wave_phase")
    assert view.analysis.colour_combo.currentData() == 0
    from temsim.optics.electron_gun.tip_coherence import wavelength_m
    delta = first.plane.coordinates_m()-first.plane.origin_m[:, None, None]
    carrier = 2*np.pi/wavelength_m(first.energy_kev*1000)*np.einsum("i,iyx->yx", first.plane.tilt_rad, delta)
    np.testing.assert_allclose(np.exp(1j*wave.values), np.exp(1j*carrier), atol=1e-12)


def test_unknown_magnetic_field_blocks_only_probability_flow(view):
    display(view, checkpoint(), axial_bz_t=None)
    assert view.analysis.wave.values.sum() == pytest.approx(50000.)
    switch(view, "wave_flow")
    assert view.analysis.wave.values is None
    assert "needs the recorded axial magnetic field" in view.summary.text()
    switch(view, "wave_angles")
    assert "Bz not recorded" in view.analysis.legend.text()
    assert view.analysis.wave.values.sum() == pytest.approx(50000.)
    switch(view, "wave_phase")
    assert view.analysis.wave.values.shape == (32, 32)


def test_flow_readout_uses_captured_posed_gauge_and_rejects_other_z(view):
    from scipy.constants import e
    from temsim.physics.tip_gun_wave import _momentum_velocity
    from test_wave_plane_observables import _posed_gauge
    gauge = _posed_gauge()
    original = replace(checkpoint(), plane_z_mm=gauge.plane_z_mm)
    # Zero phase gradient, so the entire integrated transverse current here
    # comes from the installed magnetic vector potential, including its offset.
    first = original.beam.modes[0]
    amplitude = np.ones_like(first.plane.amplitude)/32
    first = replace(first, plane=replace(first.plane, amplitude=amplitude))
    original = replace(original, beam=BeamState((first,), TIP_REFERENCE))
    display(view, original, axial_bz_t=None, magnetic_gauge=gauge)
    switch(view, "wave_flow")
    wave = view.analysis.wave
    potential = gauge.vector_potential_xy_t_m(first.plane.coordinates_m())
    p, _ = _momentum_velocity(first.energy_kev*1000.)
    expected = (original.reference_current_a*1e12*first.weight_per_reference_electron
                *np.sum(abs(amplitude)**2*potential, axis=(1, 2))*e/p)
    np.testing.assert_allclose(wave.flow_current_pA.sum(axis=(1, 2)), expected,
                               rtol=1e-12, atol=1e-12)
    assert np.linalg.norm(expected) > 1.
    switch(view, "wave_angles")
    assert "recorded posed-lens gauge" in view.analysis.legend.text()
    display(view, replace(original, plane_z_mm=gauge.plane_z_mm+1.),
            axial_bz_t=None, magnetic_gauge=gauge)
    switch(view, "wave_flow")
    assert "exact observation plane" in view.summary.text()


def test_all_diagnostic_derivation_runs_off_gui_thread(view, monkeypatch):
    import threading
    import temsim.gui.wave_beam_analysis as module
    gui_thread = threading.get_ident()
    observed = []
    derive = module._derive_view

    def record(request, event):
        observed.append((request.mode, threading.get_ident()))
        return derive(request, event)

    monkeypatch.setattr(module, "_derive_view", record)
    display(view, checkpoint(), axial_bz_t=.4)
    for key in ("wave_flow", "wave_phase", "wave_angles", "wave_interactions"):
        switch(view, key)
    assert {key for key, _ in observed} == {key for _, key in view.analysis.wave.MODES}
    assert all(thread != gui_thread for _, thread in observed)


def test_pending_old_readout_cannot_replace_a_new_mode_and_gui_remains_responsive(view, monkeypatch):
    from threading import Event
    from PySide6.QtCore import QTimer
    import temsim.gui.wave_beam_analysis as module
    started, release, tick = Event(), Event(), Event()
    derive = module._derive_view

    def gated(request, event):
        if request.mode == "wave_current":
            started.set()
            assert release.wait(5.)
        return derive(request, event)

    monkeypatch.setattr(module, "_derive_view", gated)
    try:
        view.display_wave_checkpoint(checkpoint(), axial_bz_t=0.)
        assert view.analysis.wave.busy
        view._test_qtbot.waitUntil(started.is_set, timeout=20000)
        QTimer.singleShot(0, tick.set)
        view._test_qtbot.waitUntil(tick.is_set, timeout=2000)
        view.analysis.mode_combo.setCurrentIndex(view.analysis.mode_combo.findData("wave_phase"))
        assert view.analysis.wave.values is None
        release.set()
        wait_for_readout(view)
        assert view.analysis.wave.mode == "wave_phase"
        assert view.analysis.wave.values.shape == (32, 32)
        assert "Single-mode phase" in view.analysis.legend.text()
    finally:
        release.set()


def test_leaving_wave_view_discards_pending_worker_result(view, monkeypatch):
    from threading import Event
    import temsim.gui.wave_beam_analysis as module
    started, release = Event(), Event()
    derive = module._derive_view

    def gated(request, event):
        started.set()
        assert release.wait(5.)
        return derive(request, event)

    monkeypatch.setattr(module, "_derive_view", gated)
    try:
        view.display_wave_checkpoint(checkpoint(), axial_bz_t=0.)
        wave = view.analysis.wave
        view._test_qtbot.waitUntil(started.is_set, timeout=20000)
        old = make_result()
        view.display_result(old, focus=("z", 1.))
        release.set()
        assert wave.pool.waitForDone(5000)
        assert view.analysis.wave is None
        assert view._result is old
        assert view._scatter is not None
    finally:
        release.set()


def test_stored_checkpoint_arrays_are_loaded_only_by_the_background_worker(view, monkeypatch, tmp_path):
    import threading
    from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
    store = ExecutedWaveStore(tmp_path, "offline-readout-fixture", 1<<20)
    restored = store.put(store.key("executed"), checkpoint())
    sequence_type = type(restored.beam.modes)
    load = sequence_type._array
    threads = []
    gui_thread = threading.get_ident()

    def recorded(sequence, item):
        threads.append(threading.get_ident())
        assert threads[-1] != gui_thread, "GUI must not load or checksum wave cache arrays"
        return load(sequence, item)

    monkeypatch.setattr(sequence_type, "_array", recorded)
    display(view, restored, axial_bz_t=.4)
    assert view.analysis.wave.values.sum() == pytest.approx(50000.)
    for key in ("wave_phase", "wave_flow", "wave_angles", "wave_interactions"):
        switch(view, key)
        assert "unavailable" not in view.summary.text()
    assert threads and all(thread != gui_thread for thread in threads)


def test_stored_mode_memory_budget_is_checked_before_array_loading(view, monkeypatch, tmp_path):
    from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
    store = ExecutedWaveStore(tmp_path, "offline-readout-budget-fixture", 1<<20)
    restored = store.put(store.key("executed"), checkpoint())
    monkeypatch.setattr(type(restored.beam.modes), "_array",
        lambda *args: pytest.fail("Budget refusal must precede reading any wave arrays"))
    display(view, restored, axial_bz_t=0., maximum_working_bytes=1)
    assert "memory budget before loading" in view.summary.text()
    assert view.analysis.wave.checkpoint is restored


def test_rapid_checkpoint_replacement_keeps_retired_worker_alive_until_return(view, monkeypatch):
    from threading import Event
    import shiboken6
    from PySide6.QtCore import QCoreApplication, QEvent
    import temsim.gui.wave_beam_analysis as module
    started, release = Event(), Event()
    derive = module._derive_view

    def gated(request, event):
        if request.checkpoint.plane_z_mm == 1500.:
            started.set()
            assert release.wait(5.)
        return derive(request, event)

    monkeypatch.setattr(module, "_derive_view", gated)
    original = checkpoint()
    try:
        view.display_wave_checkpoint(original, axial_bz_t=0.)
        retired = view.analysis.wave
        view._test_qtbot.waitUntil(started.is_set, timeout=20000)
        replacement = replace(original, plane_z_mm=1501.)
        view.display_wave_checkpoint(replacement, axial_bz_t=.1)
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert shiboken6.isValid(retired)
        assert shiboken6.isValid(retired.pool)
        release.set()
        wait_for_readout(view)
        assert view.analysis.wave.checkpoint is replacement
        assert view.analysis.wave.values.sum() == pytest.approx(50000.)
        assert "Z 1501" in view.heading.text()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert not shiboken6.isValid(retired)
    finally:
        release.set()
