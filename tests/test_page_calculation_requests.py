"""Page buttons request workers; navigation and display never run solvers."""

from types import SimpleNamespace

import numpy as np

from temsim.detector.stem_signal import StemScanResult
from temsim.gui.scan_panel import ScanControlView
from temsim.gui.visualization import VisualizationWorkspace
from temsim.optics.column import default_state


def _frame(value):
    x, y = np.meshgrid(np.array([-.001, .001]), np.array([-.001, .001]))
    return StemScanResult(
        scan_x_um=x, scan_y_um=y,
        fractions={key: np.full_like(x, value) for key in ("haadf", "df", "bf")},
        detector_signals={}, metrics={"model": "geometric_detector_interception"},
    )


def test_six_page_buttons_emit_only_their_scope_and_tabs_do_not_calculate(qtbot, monkeypatch):
    from temsim.specimen import interaction_engine, sample_region

    def forbidden(*_args, **_kwargs):
        raise AssertionError("GUI request must not run specimen physics")

    monkeypatch.setattr(interaction_engine, "run_specimen_interactions", forbidden)
    monkeypatch.setattr(sample_region, "simulate_sample_region", forbidden)
    workspace = VisualizationWorkspace()
    qtbot.addWidget(workspace)
    state = default_state()
    state.sample.eds_enabled = True
    workspace.eds_page.set_state(state)
    requests = []
    workspace.calculation_requested.connect(requests.append)
    for index in range(workspace.tabs.count()):
        workspace.tabs.setCurrentIndex(index)
    assert requests == []
    controls = (
        (workspace.sample_page.calculate_button, "sample"),
        (workspace.eds_page.calculate_button, "eds"),
        (workspace.scan_control.calculate_button, "stem"),
        (workspace.wave_imaging.calculate_button, "imaging"),
        (workspace.energy_filter.calculate_button, "energy_filter"),
        (workspace.sample_interactions_3d.calculate_paths, "sample_region"),
    )
    for button, scope in controls:
        assert button.isEnabled()
        button.click()
        assert requests[-1] == scope
    assert requests == [scope for _button, scope in controls]
    assert workspace.eds_page.isAncestorOf(workspace.eds_page.calculate_button)
    assert not state.sample.wave_enabled and not state.sample.stem_wave_enabled


def test_changed_page_inputs_mark_refresh_needed_without_requesting_calculation(qtbot):
    workspace = VisualizationWorkspace()
    qtbot.addWidget(workspace)
    state = default_state()
    state.sample.eds_enabled = True
    workspace.eds_page.set_state(state)
    workspace.scan_control.set_state(state)
    requests = []
    workspace.calculation_requested.connect(requests.append)
    workspace.eds_page.eds_scalar_controls["eds_support_offset_x_um"].setValue(1.)
    workspace.scan_control.parameters_changed.emit("ac_deflector.scan_pixel_size_nm")
    assert not requests
    assert "Calculate EDS" in workspace.eds_page.calculation_bar.status.text()
    assert "Inputs changed" in workspace.scan_control.calculation_bar.status.text()
    assert workspace.eds_page.calculate_button.isEnabled()


def test_explicit_stem_result_replaces_frozen_frame_without_changing_pause(qtbot):
    view = ScanControlView()
    qtbot.addWidget(view)
    state = default_state()
    view.set_state(state)
    old, automatic, explicit = _frame(.1), _frame(.2), _frame(.3)
    view.display_result(None, old, complete=True, state_snapshot=state)
    view.pause_image_refresh.setChecked(True)
    view.display_result(None, automatic, complete=True, state_snapshot=state)
    assert view._paused_display_frame is old
    view.set_bank_readout(SimpleNamespace(stem=_frame(.9), state_snapshot=state))
    view.image_source.setCurrentIndex(view.image_source.findData("bank"))
    view.display_result(None, explicit, complete=True, state_snapshot=state,
                        explicit_calculation=True)
    assert view.pause_image_refresh.isChecked()
    assert view.image_source.currentData() == "current"
    assert view._stem_frame is explicit and view._paused_display_frame is explicit
    np.testing.assert_array_equal(view.detector_image_items["bf"].image, explicit.fractions["bf"].T)


def test_scan_off_displays_executed_pixel_counts_and_unavailable_is_not_zero(qtbot):
    from temsim.detector.particle_readout import ParticleDetectorReadout
    from temsim.component_keys import HAADF_DETECTOR, BRIGHT_FIELD_DETECTOR

    view = ScanControlView()
    qtbot.addWidget(view)
    view.set_particle_signals((
        ParticleDetectorReadout(HAADF_DETECTOR, "HAADF", "AVAILABLE",
                               simulated_electrons=12.5, electrons_per_second=4e6),
        ParticleDetectorReadout(BRIGHT_FIELD_DETECTOR, "BF", "NOT_REACHED"),
    ))
    assert not view.current_pixel_table.isHidden()
    assert view.current_pixel_table.item(0, 1).text() == "12.5"
    assert view.current_pixel_table.item(0, 2).text() == "4000000"
    assert view.current_pixel_table.item(1, 1).text() == "—"
    assert "Scan is off" in view.image_model_notice.text()
    view.mark_stem_frame_stale()
    assert "Previous current-pixel" in view.current_pixel_status.text()
    view.set_particle_signals((), scan_enabled=True)
    assert view.current_pixel_table.isHidden()


def test_ray_scope_marks_missing_products_pending_and_retains_old_images(qtbot):
    workspace = VisualizationWorkspace()
    qtbot.addWidget(workspace)
    state = default_state()
    workspace.eds_page.set_state(state)
    workspace.scan_control.set_state(state)
    old = _frame(.4)
    workspace.scan_control._set_stem_frame(old)
    result = SimpleNamespace(state_snapshot=state)
    workspace._display_ray_scope_products(result, "High accuracy")
    assert workspace.scan_control._stem_frame is old
    assert workspace.scan_control._stem_frame_stale
    for page in (workspace.sample_page, workspace.eds_page, workspace.scan_control,
                 workspace.energy_filter, workspace.wave_imaging, workspace.sample_interactions_3d):
        assert "Tip-to-sample rays cached" in page.calculation_bar.status.text()
        assert "Click Calculate" in page.calculation_bar.status.text()
    assert workspace.sample_interactions_3d.calculate_paths.isEnabled()


def test_publishing_new_eds_frame_and_changing_view_do_not_mutate_checkpoints(qtbot):
    from temsim.gui.eds_panel import EDSPage

    page = EDSPage()
    qtbot.addWidget(page)
    state = default_state()
    page.set_state(state)
    old_region, new_region = object(), object()
    old = SimpleNamespace(sample_region=old_region, simulation=None,
                          calculated_products=frozenset({"sample_region"}))
    new = SimpleNamespace(sample_region=new_region, simulation=None,
                          reused_products=frozenset({"sample_region"}))
    events = []
    page.sample_region_result_ready.connect(events.append)
    page.display_result(old)
    page.display_result(new)
    assert old.sample_region is old_region and new.sample_region is new_region
    assert new.reused_products == frozenset({"sample_region"})
    page.sample_region_photons.setValue(page.sample_region_photons.value() + 1)
    assert new.sample_region is new_region
    assert not events


def test_calculation_bar_does_not_take_expanding_space_from_sample_view(qtbot):
    from PySide6.QtWidgets import QSizePolicy

    workspace = VisualizationWorkspace()
    qtbot.addWidget(workspace)
    workspace.resize(1400, 900)
    workspace.tabs.setCurrentWidget(workspace.sample_page)
    workspace.show()
    bar = workspace.sample_page.calculation_bar
    qtbot.waitUntil(lambda: bar.height() < 60)
    assert bar.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Maximum


def test_current_pixel_table_shows_all_three_detector_rows(qtbot):
    from temsim.detector.particle_readout import ParticleDetectorReadout
    from temsim.component_keys import HAADF_DETECTOR, DARK_FIELD_DETECTOR, BRIGHT_FIELD_DETECTOR

    workspace = VisualizationWorkspace()
    qtbot.addWidget(workspace)
    workspace.resize(1400, 900)
    workspace.tabs.setCurrentWidget(workspace.scanning_page)
    view = workspace.scan_control
    view.result_tabs.setCurrentIndex(1)
    view.set_particle_signals(tuple(
        ParticleDetectorReadout(key, name, "AVAILABLE", simulated_electrons=10., electrons_per_second=1e6)
        for key, name in ((HAADF_DETECTOR, "HAADF"), (DARK_FIELD_DETECTOR, "DF"),
                          (BRIGHT_FIELD_DETECTOR, "BF"))
    ))
    workspace.show()
    table = view.current_pixel_table
    qtbot.waitUntil(lambda: table.viewport().height() >= sum(table.rowHeight(row) for row in range(3)))
    assert not table.verticalScrollBar().isVisible()
    assert table.height() < 160


def test_ray_scope_distinguishes_executed_filter_from_retained_filter(qtbot, monkeypatch):
    workspace = VisualizationWorkspace()
    qtbot.addWidget(workspace)
    monkeypatch.setattr(workspace.energy_filter, "display_result", lambda _result: None)
    result = SimpleNamespace(energy_filter=object(), calculated_products=frozenset({"energy_filter"}))
    workspace._display_ray_scope_products(result, "High accuracy")
    assert "required physical transport was executed" in workspace.energy_filter.calculation_bar.status.text()
    result.calculated_products = frozenset()
    workspace._display_ray_scope_products(result, "High accuracy")
    assert "Valid cached result" in workspace.energy_filter.calculation_bar.status.text()
