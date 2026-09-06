"""A physically blocked specimen input must not create normalized products."""

from types import SimpleNamespace

import numpy as np

from temsim.optics.column import default_state
from temsim.physics.beam_current import sample_illumination_absent
from temsim.physics.simulation import Simulation
from temsim.simulation_pipeline import CalculationResult


def _empty_simulation():
    return Simulation(
        incident=SimpleNamespace(
            alive=np.asarray((False, False)),
            ray_weight=np.asarray((0.5, 0.5)),
        ),
        branches={}, metrics={},
    )


def test_absence_uses_traced_current_not_requested_blanker_state():
    state = default_state()
    state.nanopulser.installed = True
    state.nanopulser.blanked = False
    simulation = _empty_simulation()
    assert sample_illumination_absent(simulation, state)
    simulation.incident.alive[:] = True
    state.nanopulser.blanked = True
    state.nanopulser.voltage_v = 0.0
    assert not sample_illumination_absent(simulation, state)
    simulation.incident.ray_weight[:] = 0.0
    assert sample_illumination_absent(simulation, state)


def test_all_requested_specimen_products_skip_empty_probe_and_progress_completes(monkeypatch):
    import temsim.simulation_pipeline as pipeline

    state = default_state()
    state.ac_deflector.enabled = True
    state.ac_deflector.scan_enabled = True
    # Force every requested pipeline branch; the input beam itself is absent.
    monkeypatch.setattr(pipeline, "tem_wave_imaging_enabled", lambda _s: True)
    monkeypatch.setattr(pipeline, "_geometric_specimen_transport_requested", lambda _s: True)
    monkeypatch.setattr(pipeline, "_eds_point_requested", lambda _s: True)
    monkeypatch.setattr(pipeline, "run", lambda *_a, **_kw: _empty_simulation())
    monkeypatch.setattr(pipeline, "simulate_energy_filter", lambda *_a: None)
    monkeypatch.setattr(pipeline, "calculate_scan_geometry", lambda *_a: None)
    monkeypatch.setattr(pipeline, "calculate_scan_ray_paths", lambda *_a: None)
    monkeypatch.setattr(pipeline, "detect_all_lens_crossovers", lambda *_a: ())

    def forbidden(*_a, **_kw):
        raise AssertionError("No specimen solver may manufacture an empty probe")

    monkeypatch.setattr(pipeline, "run_specimen_interactions", forbidden)
    progress = []
    result = pipeline.calculate(
        state, progress_callback=lambda completed, total, label: progress.append(
            (completed, total, label)
        ),
    )
    assert result.wave_imaging is None
    assert result.stem_scan is None
    assert result.specimen_interactions is None
    assert result.sample_region is None
    assert result.simulation.metrics["sample_surviving_current_pa"] == 0.0
    assert result.simulation.metrics["sample_illumination_status"] == "no incident current"
    assert progress[-1][0] == progress[-1][1]
    assert progress[-1][2] == "Complete"


def test_wave_display_clears_previous_picture_and_reports_absent_current(qtbot):
    from temsim.gui.visualization import WaveImagingView

    view = WaveImagingView()
    qtbot.addWidget(view)
    view.image.setImage(np.ones((8, 8)))
    view.diffraction.setImage(np.ones((8, 8)))
    view.display_result(None, no_illumination=True)
    assert view.image.image is None
    assert view.diffraction.image is None
    assert "No incident current" in view.summary.text()


def test_eds_no_current_clears_spectrum_and_disables_acquisition(qtbot):
    from temsim.gui.eds_panel import EDSPage

    state = default_state()
    state.sample.eds_enabled = True
    page = EDSPage()
    qtbot.addWidget(page)
    page.set_state(state)
    page._spectrum_energy_kev = np.ones(4)
    page._spectrum_counts = np.ones(4)
    result = CalculationResult(
        simulation=_empty_simulation(), energy_filter=None, state_snapshot=state,
    )
    page.display_result(result)
    assert page._spectrum_counts.size == 0
    assert not hasattr(page, "eds_lines")
    assert not page.eds_acquire.isEnabled()
    assert not page.sample_region_calculation_available()
    assert "No incident current" in page.eds_summary.text()


def test_zero_current_preview_immediately_clears_previous_high_images(qtbot, monkeypatch):
    from temsim.gui.visualization import VisualizationWorkspace

    state = default_state()
    workspace = VisualizationWorkspace()
    qtbot.addWidget(workspace)
    workspace.eds_page.set_state(state)
    high = SimpleNamespace(model_signature="previous-high-state")
    workspace._high_accuracy_result = high
    workspace._high_accuracy_current = False
    workspace._sample_region_result = object()
    workspace.wave_imaging.image.setImage(np.ones((8, 8)))
    workspace.wave_imaging.diffraction.setImage(np.ones((8, 8)))
    workspace.eds_page._spectrum_counts = np.ones(4)
    for item in workspace.scan_control.detector_image_items.values():
        item.setImage(np.ones((8, 8)))
    workspace.scan_control._stem_frame = object()
    monkeypatch.setattr(workspace, "_prepare_scan_ray_playback", lambda *_a: None)
    monkeypatch.setattr(workspace, "_draw_ray_diagram", lambda *_a, **_kw: None)
    monkeypatch.setattr(workspace, "_update_sample_region_control_availability", lambda: None)
    for view in (
        workspace.physical_layout, workspace.magnetic_field,
        workspace.probe_aberrations, workspace.image_aberrations,
        workspace.optical_transfer, workspace.transverse_beam,
        workspace.sample_page, workspace.sample_interactions_3d,
    ):
        monkeypatch.setattr(view, "display_result", lambda *_a: None)
    filter_updates = []
    monkeypatch.setattr(workspace.energy_filter, "display_result", filter_updates.append)
    preview = CalculationResult(
        simulation=_empty_simulation(), energy_filter=None,
        state_snapshot=state, model_signature="blanked-preview",
    )
    workspace.display_result(preview, "Preview")
    assert filter_updates == [preview]
    assert workspace.wave_imaging.image.image is None
    assert workspace.wave_imaging.diffraction.image is None
    assert workspace.scan_control._stem_frame is None
    assert all(item.image is None for item in workspace.scan_control.detector_image_items.values())
    assert workspace.eds_page._spectrum_counts.size == 0
    assert workspace._sample_region_result is None
    assert "No incident current" in workspace.wave_imaging.summary.text()
    assert "No incident current" in workspace.scan_control.image_model_notice.text()
    assert workspace._high_accuracy_result is high
