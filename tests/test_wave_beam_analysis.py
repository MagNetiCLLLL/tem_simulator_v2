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
    return widget


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


def test_same_panel_current_and_angles_keep_absolute_tip_weights(view, monkeypatch):
    import temsim.physics.tip_wave_pipeline as pipeline
    monkeypatch.setattr(pipeline, "simulate_tip_wave", lambda *a, **k: pytest.fail("GUI must not propagate"))
    original = checkpoint()
    digest = original.digest
    view.display_wave_checkpoint(original, axial_bz_t=.5)
    assert view.analysis.wave.values.sum() == pytest.approx(50000., rel=1e-12)
    assert view.initial_beam_panel.isHidden()
    switch(view, "wave_angles")
    assert view.analysis.wave.values.sum() == pytest.approx(50000., rel=1e-12)
    assert "Canonical" in view.analysis.legend.text()
    assert view.plot.getAxis("bottom").labelUnits == "mrad"
    assert original.digest == digest


def test_phase_selects_one_mode_and_keeps_relative_phase(view):
    view.display_wave_checkpoint(checkpoint(), axial_bz_t=0.)
    switch(view, "wave_phase")
    assert view.analysis.colour_combo.currentData() == 0
    first = view.analysis.wave.values.copy()
    view.analysis.colour_combo.setCurrentIndex(2)
    np.testing.assert_allclose(np.exp(1j*view.analysis.wave.values), 1j*np.exp(1j*first), atol=1e-12)
    view.analysis.colour_combo.setCurrentIndex(0)
    assert view.analysis.colour_combo.currentData() == 0  # No aggregate phase option accepted.


def test_missing_plane_does_not_show_previous_wave_or_recompute(view):
    original = checkpoint()
    view.display_wave_checkpoint(original, axial_bz_t=0.)
    view.focus_z(1499.)
    assert "not cached" in view.summary.text()
    assert not view.plot.getPlotItem().items
    view.focus_z(1500.)
    assert view.analysis.wave.values.sum() == pytest.approx(50000.)
    assert view.analysis.wave.checkpoint is original


def test_manual_scale_remains_fixed_for_rotation_and_plane_return(view):
    view.display_wave_checkpoint(checkpoint(), axial_bz_t=0.)
    view.plot.setRange(xRange=(-.001, .001), yRange=(-.001, .001), padding=0.)
    view._manual_view_range_changed([True, True])
    before = np.array(view.plot.viewRange())
    visible = view.analysis.wave.values.sum()
    assert visible < 50000.
    view.set_projection_angle(90.)
    view.focus_z(1501.)
    view.focus_z(1500.)
    np.testing.assert_allclose(view.plot.viewRange(), before, rtol=1e-10, atol=1e-14)
    view.fit_beam.click()
    assert view.analysis.wave.values.sum() == pytest.approx(50000.)


def test_interactions_use_executed_weights_and_old_ray_view_can_be_restored(view):
    old = make_result()
    view.display_result(old)
    view.display_wave_checkpoint(checkpoint(), axial_bz_t=0.)
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
    view.display_wave_checkpoint(original, axial_bz_t=0.)
    view.plot.setRange(xRange=(-.001, .001), yRange=(-.001, .001), padding=0.)
    view._manual_view_range_changed([True, True])
    before = np.array(view.plot.viewRange())
    view.display_wave_checkpoint(replace(original, plane_z_mm=1501.), axial_bz_t=.1)
    np.testing.assert_allclose(view.plot.viewRange(), before, rtol=1e-10, atol=1e-14)


def test_display_budget_failure_does_not_erase_executed_wave(view):
    original = checkpoint()
    view.display_wave_checkpoint(original, axial_bz_t=0., maximum_working_bytes=1)
    assert "memory budget" in view.summary.text()
    assert view.analysis.wave.checkpoint is original


@pytest.mark.parametrize("angle", (0., 31., 90.))
def test_coarse_uniform_wave_is_not_drawn_as_sparse_point_markers(view, angle):
    original = checkpoint()
    view.set_projection_angle(angle)
    view.display_wave_checkpoint(original, axial_bz_t=0.)
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
    view.display_wave_checkpoint(checkpoint(), axial_bz_t=0.)
    wave = view.analysis.wave
    values, xe, ye = wave._hover
    assert len(xe) == values.shape[0]+1 and len(ye) == values.shape[1]+1
    position = view.plot.getViewBox().mapViewToScene(QPointF(0., 0.))
    wave.mouse_moved(position)
    assert "pA / bin" in view.analysis.readout.text()
