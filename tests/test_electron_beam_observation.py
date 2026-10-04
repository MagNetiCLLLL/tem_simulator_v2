"""Readout integration from executed fields; no tip-to-column qualification."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.gui.coherent_beam import CoherentIntensityView, _Preview, _intensity_preview
from temsim.gui.electron_beam_observation import ElectronBeamObservation
from temsim.physics.coherent_state_set import combine_intensity_previews
from test_wave_beam_analysis import checkpoint


@pytest.fixture
def view(qtbot, monkeypatch):
    import temsim.gui.coherent_beam as beam
    monkeypatch.setattr(beam, "simulate_tip_wave", lambda *a, **k: pytest.fail("A readout must not propagate"))
    widget = ElectronBeamObservation(CoherentIntensityView())
    widget.resize(950, 750)
    qtbot.addWidget(widget)
    widget.show()
    yield widget
    assert widget.shutdown(10000)


def plane(z=1500., bounds=None):
    result = SimpleNamespace(checkpoint=replace(checkpoint(), plane_z_mm=z))
    if bounds is None:
        preview = replace(_intensity_preview(result.checkpoint, bins=32), axial_bz_t=.1)
    else:
        bounds = np.array(bounds, dtype=float)
        density = np.full((8, 8), .5/np.prod(np.diff(bounds, axis=1)))
        preview = _Preview(density, bounds, .5, 1, density.nbytes, axial_bz_t=.1)
    return result, preview


def show(view, value):
    view.mode.setCurrentIndex(view.mode.findData(value))


def arrivals_ready(qtbot, view):
    qtbot.waitUntil(lambda: not view.timer.isActive() and view._worker is None
        and view.detections is not None, timeout=10000)


def diagnostic_ready(qtbot, view):
    qtbot.waitUntil(lambda: not view.diagnostics.analysis.wave.busy, timeout=15000)


def test_arrivals_preserve_absolute_probability_and_do_not_repropagate(qtbot, view):
    result, preview = plane()
    original_digest = result.checkpoint.digest
    view.set_observation(result, preview, 0)
    show(view, "arrivals")
    arrivals_ready(qtbot, view)
    assert view.detections.emitted_count == 3000
    assert view.detections.detection_probability == pytest.approx(.5)
    assert 1400 < view.detections.detected_count < 1600
    assert view.detections.detected_count + view.detections.lost_count == 3000
    assert len(view.points.points()) == view.detections.detected_count
    assert "not trajectories" in view.note.text()
    assert result.checkpoint.digest == original_digest


def test_count_and_seed_are_readout_only_with_deterministic_prefix(qtbot, view):
    result, preview = plane()
    view.set_observation(result, preview, 0)
    view.count.setValue(100)
    show(view, "arrivals")
    arrivals_ready(qtbot, view)
    first = view.detections.points_um.copy()
    view.count.setValue(200)
    arrivals_ready(qtbot, view)
    np.testing.assert_array_equal(view.detections.points_um[:len(first)], first)
    show(view, "intensity")
    view.count.setValue(0)
    show(view, "arrivals")
    arrivals_ready(qtbot, view)
    assert view.detections.emitted_count == 0
    assert len(view.points.points()) == 0
    assert view.result is result and view.preview is preview


def test_arrival_viewport_stays_fixed_across_z(qtbot, view):
    view.set_observation(*plane(bounds=((-2., 2.), (-2., 2.))), 0)
    show(view, "arrivals")
    arrivals_ready(qtbot, view)
    view.arrival_plot.setRange(xRange=(-1., 1.), yRange=(-1., 1.), padding=0.)
    # The physical zoom request is preserved. Qt may adjust the visible X
    # extent to keep equal units when wrapped status text changes plot height.
    before = np.array(view.arrival_plot.getViewBox().targetRange())
    view.set_observation(*plane(1501., ((-10., 10.), (-10., 10.))), 0)
    arrivals_ready(qtbot, view)
    np.testing.assert_allclose(view.arrival_plot.getViewBox().targetRange(), before)
    assert "1501" in view.arrival_plot.getPlotItem().titleLabel.text
    view.fit.click()
    assert view.arrival_plot.viewRange()[0][1] > 9.


def test_late_sampling_result_cannot_replace_new_plane(qtbot, view):
    view.set_observation(*plane(), 0)
    show(view, "arrivals")
    arrivals_ready(qtbot, view)
    old_generation, old_data = view._generation, view.detections
    view.set_observation(*plane(1501.), 0)
    view._detections_ready(old_generation, old_data)
    assert view.detections is None
    arrivals_ready(qtbot, view)
    assert "1501" in view.arrival_plot.getPlotItem().titleLabel.text


@pytest.mark.parametrize("mode", ("wave_current", "wave_phase", "wave_flow", "wave_angles", "wave_interactions"))
def test_complex_observations_read_same_checkpoint(qtbot, view, mode):
    result, preview = plane()
    view.set_observation(result, preview, 0)
    show(view, mode)
    diagnostic_ready(qtbot, view)
    assert view.diagnostics.analysis.wave.checkpoint is result.checkpoint
    assert view.diagnostics.analysis.mode_combo.currentData() == mode
    assert "unavailable" not in view.diagnostics.summary.text().lower()
    assert view.diagnostics.initial_beam_panel.isHidden()
    assert view.diagnostics.fit_beam.isHidden()
    assert view.diagnostics.plot.width() > 600
    assert view.diagnostics.plot.height() > 250


def test_overlay_arrivals_sum_intensities_but_phase_selects_one_state(qtbot, view):
    first, p1 = plane()
    second = SimpleNamespace(checkpoint=replace(first.checkpoint, reference_current_a=first.checkpoint.reference_current_a))
    combined, preview = combine_intensity_previews(((first, p1, 1.), (second, p1, 3.)))
    view.set_observation(combined, preview, 0)
    show(view, "arrivals")
    arrivals_ready(qtbot, view)
    assert view.detections.detection_probability == pytest.approx(.5)
    show(view, "wave_phase")
    diagnostic_ready(qtbot, view)
    assert view.member.count() == 2
    assert view.diagnostics.analysis.wave.checkpoint is first.checkpoint
    view.member.setCurrentIndex(1)
    diagnostic_ready(qtbot, view)
    assert view.diagnostics.analysis.wave.checkpoint is second.checkpoint
    assert view.diagnostics.analysis.colour_combo.currentData() == 0
    assert preview.axial_bz_t == .1


def test_unknown_magnetic_state_does_not_claim_probability_flow(qtbot, view):
    result, preview = plane()
    view.set_observation(result, replace(preview, axial_bz_t=None), 0)
    show(view, "wave_flow")
    diagnostic_ready(qtbot, view)
    assert "magnetic" in view.diagnostics.summary.text().lower()
    show(view, "wave_phase")
    diagnostic_ready(qtbot, view)
    assert view.diagnostics.analysis.wave.values is not None


def test_projection_uses_same_angle_as_ray_diagram(qtbot, view):
    view.set_projection_angle(37.)
    view.set_observation(*plane(), 0)
    show(view, "wave_current")
    diagnostic_ready(qtbot, view)
    assert view.diagnostics._projection_angle_deg == 37.


def test_dense_arrivals_show_all_counts(qtbot, view):
    view.set_observation(*plane(), 0)
    view.count.setValue(120000)
    show(view, "arrivals")
    arrivals_ready(qtbot, view)
    assert view.arrival_image.isVisible() and not view.points.isVisible()
    assert view.arrival_image.image.sum() == view.detections.detected_count
