"""Publication routing only: panel spies avoid transport or plotting calculations."""
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from temsim.gui.visualization import VisualizationWorkspace


class Label:
    def __init__(self):
        self.text = ""
        self.tooltip = ""

    def setText(self, text):
        self.text = text

    def setToolTip(self, text):
        self.tooltip = text

    def setStyleSheet(self, _style):
        pass


class PanelSpy:
    def __init__(self, product=None):
        self.product = product
        self.calls = []
        self.displayed = None
        self.stale = False
        self.summary = Label()
        self.eds_summary = Label()
        self.image_model_notice = Label()

    def display_result(self, result, *args, **kwargs):
        self.calls.append((result, args, kwargs))
        self.displayed = getattr(result, self.product, None) if self.product else result
        self.stale = False

    def mark_result_stale(self):
        self.stale = True

    def mark_stem_frame_stale(self):
        self.stale = True

    def publish(self, *args):
        pass

    def mark_stale(self, *args):
        pass


class ScanPanelSpy(PanelSpy):
    def __init__(self):
        super().__init__()
        self._stem_frame = None
        self.explicit_calculations = []
        self.particle_signals = None

    def display_result(self, geometry, frame=None, *, complete=False, state_snapshot=None,
                       explicit_calculation=False):
        self.calls.append((geometry, frame, complete, state_snapshot))
        self.explicit_calculations.append(explicit_calculation)
        if frame is not None or complete:
            self._stem_frame = frame
            self.displayed = frame
            self.stale = False

    def set_particle_signals(self, rows, *, scan_enabled=False):
        self.particle_signals = (rows, scan_enabled)


@pytest.fixture
def workspace():
    # Invoke the real publication and invalidation methods. Only the expensive
    # plots/panels are replaced: no source, transport or detector model is run.
    def noop(*args, **kwargs):
        pass

    view = SimpleNamespace(
        _ray_display_cache={}, _ray_display_cache_bytes=0, _last_result=None,
        _ray_flight_time_colours=SimpleNamespace(invalidate=noop),
        transverse_beam=SimpleNamespace(analysis=SimpleNamespace(
            tof=SimpleNamespace(invalidate=noop))),
        _high_accuracy_result=None, _high_accuracy_current=False,
        _sample_region_result=None, _scan_ray_paths=None,
        result_readout=PanelSpy(), ray_source_status=Label(), heading=Label(),
        hardware_tuning=SimpleNamespace(publish_result=noop, mark_result_stale=noop),
        selected_plane_readout=SimpleNamespace(set_result=noop),
        conjugate_planes=SimpleNamespace(set_result=noop),
        interactive_calculation=SimpleNamespace(calculation_timing=SimpleNamespace(set_result=noop)),
        _refresh_ray_calculation_extent=noop,
        _publish_optional_ray_panels=noop, _draw_ray_diagram=noop,
        _refresh_visible_ray_panels=noop, _set_sample_region_result=noop,
        _update_sample_region_control_availability=noop, _update_projection_text=noop,
    )
    for name in ("probe_aberrations", "image_aberrations", "optical_transfer", "sample_page",
                 "wave_imaging", "transport_adjustment_readout"):
        setattr(view, name, PanelSpy())
    view.energy_filter = PanelSpy("energy_filter")
    view.eds_page = PanelSpy("specimen_interactions")
    view.sample_interactions_3d = PanelSpy("specimen_interactions")
    view.scan_control = ScanPanelSpy()
    for page in (view.sample_page, view.eds_page, view.scan_control, view.wave_imaging,
                 view.energy_filter, view.sample_interactions_3d):
        page.calculation_bar = SimpleNamespace(mark_stale=noop, set_result_available=noop)
    view.sample_interactions_3d.calculate_paths = SimpleNamespace(setEnabled=noop)
    view.mark_high_accuracy_stale = lambda: VisualizationWorkspace.mark_high_accuracy_stale(view)
    view._prepare_scan_ray_playback = MethodType(VisualizationWorkspace._prepare_scan_ray_playback, view)
    return view


def result(*, particle=True, optical=False, reached=True, alive=True, products=False,
           stem_frame=None, model_signature="captured-state"):
    return SimpleNamespace(
        simulation=SimpleNamespace(
            metrics={"particle_tuning": particle, "optical_tuning": optical,
                     "section_sample_reference_reached": reached},
            incident=SimpleNamespace(alive=np.array([alive]), ray_weight=np.array([1.])),
        ),
        state_snapshot=SimpleNamespace(electron_gun=SimpleNamespace(type_key="fixture"),
                                       ac_deflector=SimpleNamespace(scan_enabled=stem_frame is not None)),
        model_signature=model_signature,
        energy_filter=object() if products else None,
        specimen_interactions=object() if products else None,
        sample_region=object() if products else None,
        scan_geometry=object() if stem_frame is not None else None,
        stem_scan=stem_frame,
    )


@pytest.mark.parametrize("quality", ["Preview", "Medium"])
@pytest.mark.parametrize("sample_reached", [True, False])
def test_particle_products_are_cleared_when_next_section_has_none(workspace, quality, sample_reached):
    previous = result(products=True)
    VisualizationWorkspace.display_result(workspace, previous, quality)
    assert workspace.energy_filter.displayed is previous.energy_filter
    assert workspace.eds_page.displayed is previous.specimen_interactions
    assert workspace.sample_interactions_3d.displayed is previous.specimen_interactions
    assert workspace._high_accuracy_result is None

    current = result(reached=sample_reached)
    VisualizationWorkspace.display_result(workspace, current, quality)
    for panel in (workspace.energy_filter, workspace.eds_page, workspace.sample_interactions_3d):
        assert len(panel.calls) == 2
        assert panel.displayed is None
    assert workspace.energy_filter.calls[-1][0] is current
    assert workspace.sample_interactions_3d.calls[-1][0] is current
    assert workspace._sample_region_result is None
    if not sample_reached:
        assert workspace.eds_page.calls[-1][0] is None
        assert "before the specimen" in workspace.eds_page.eds_summary.text
        assert "EDS not calculated" in workspace.eds_page.eds_summary.text


@pytest.mark.parametrize("quality", ["Preview", "Medium"])
def test_particle_stem_frame_replaces_displayed_high_accuracy_frame(workspace, quality):
    previous = result(particle=False, stem_frame=object(), model_signature="old-state")
    VisualizationWorkspace.display_result(workspace, previous, "High accuracy")
    assert workspace.scan_control.displayed is previous.stem_scan

    current = result(stem_frame=object(), model_signature="new-state")
    VisualizationWorkspace.display_result(workspace, current, quality)
    assert workspace._high_accuracy_result is previous
    assert not workspace._high_accuracy_current
    assert workspace.scan_control.displayed is current.stem_scan
    assert workspace.scan_control.calls[-1] == (
        current.scan_geometry, current.stem_scan, True, current.state_snapshot)
    assert workspace.sample_page.calls[-1][1] == (current.stem_scan,)


@pytest.mark.parametrize("quality", ["Preview", "Medium"])
def test_legacy_optical_preview_retains_existing_material_and_scan_products(workspace, quality):
    previous = result(particle=False, products=True, stem_frame=object(), model_signature="old-state")
    VisualizationWorkspace.display_result(workspace, previous, "High accuracy")
    # An empty low-count optical preview must not erase completed material
    # products or invent zero spectra/images. Their stale state is independent.
    current = result(particle=False, optical=True, alive=False, model_signature="new-state")
    VisualizationWorkspace.display_result(workspace, current, quality)
    assert workspace._last_result is current
    assert workspace._high_accuracy_result is previous
    assert workspace._sample_region_result is previous.sample_region
    assert workspace.energy_filter.displayed is previous.energy_filter
    assert workspace.eds_page.displayed is previous.specimen_interactions
    assert workspace.sample_interactions_3d.displayed is previous.specimen_interactions
    assert workspace.scan_control.displayed is previous.stem_scan
    for panel in (workspace.energy_filter, workspace.eds_page, workspace.sample_interactions_3d,
                  workspace.scan_control):
        assert len(panel.calls) == 1
        assert panel.stale


@pytest.mark.parametrize("quality", ["Preview", "Medium"])
@pytest.mark.parametrize("retained", [False, True])
def test_ray_preview_without_acquisition_keeps_stem_empty_or_preserves_prior_frame(
    workspace, qtbot, quality, retained,
):
    from temsim.detector.stem_signal import StemScanResult
    from temsim.gui.scan_panel import ScanControlView
    from temsim.optics.column import default_state

    state = default_state()
    view = ScanControlView()
    qtbot.addWidget(view)
    view.set_state(state)
    workspace.scan_control = view
    assert state.ac_deflector.scan_enabled and state.sample.stem_image_enabled
    previous = None
    if retained:
        x, y = np.meshgrid(np.arange(2) * .001, np.arange(2) * .001)
        previous = StemScanResult(
            scan_x_um=x, scan_y_um=y,
            fractions={key: np.full((2, 2), .4) for key in ("haadf", "df", "bf")},
            detector_signals={}, metrics={},
        )
        view._set_stem_frame(previous, state_snapshot=state)
    context = view._stem_frame_context
    current = result()
    current.state_snapshot = state
    current.simulation.metrics["section_scan_status"] = "not_requested"

    VisualizationWorkspace.display_result(workspace, current, quality)

    assert view._stem_frame is previous
    assert view._stem_frame_context is context
    assert not view._playback_timer.isActive()
    assert "Click Calculate STEM" in view.calculation_bar.status.text()
    assert "not been calculated" in view.summary.text()
    for key, item in view.detector_image_items.items():
        if retained:
            np.testing.assert_array_equal(item.image, previous.fractions[key].T)
        else:
            assert item.image is None
            assert view.detector_contrast_labels[key].isHidden()
    if retained:
        assert view._stem_frame_stale
        assert "previous STEM frame retained" in view.image_model_notice.text()
    else:
        assert "STEM not calculated" in view.image_model_notice.text()
        assert "STEM not calculated" in view.detector_playback_summary.text()
