"""Optional presentation is newest-only; scientific publication stays eager."""

from types import SimpleNamespace

import numpy as np
import pytest

from temsim.gui.diagnostic_tabs import MagneticFieldView, TransverseBeamView
from temsim.gui.visualization import VisualizationWorkspace


def _result(value=1.0, *, tuning=True, signature="live", detector_z=9.0):
    branch = SimpleNamespace(
        name="incident", z=np.array([0.0, 10.0]),
        x=np.array([[-1.0, 0.0, 1.0], [-0.3, 0.2, 0.5]]) * value * 1e-6,
        y=np.array([[0.5, -0.2, 0.1], [0.1, -0.1, 0.3]]) * value * 1e-6,
        blocked_z=np.full(3, np.nan),
    )
    part = SimpleNamespace(key="camera", center_z_mm=detector_z - 1.0)
    return SimpleNamespace(
        value=value, model_signature=signature,
        simulation=SimpleNamespace(incident=branch, branches={}, metrics={"optical_tuning": tuning}),
        assembly=SimpleNamespace(parts=(part,)),
        state_snapshot=SimpleNamespace(recording_planes=(SimpleNamespace(key="camera", z_mm=detector_z),)),
    )


@pytest.fixture
def workspace(qtbot, monkeypatch):
    view = VisualizationWorkspace()
    qtbot.addWidget(view)
    view.resize(1400, 900)
    # Exercise publication and real Qt visibility without a column solve or
    # unrelated image, spectrum, and instrument-layout construction.
    monkeypatch.setattr(view, "_draw_ray_diagram", lambda *_a, **_k: None)
    monkeypatch.setattr(view, "_prepare_scan_ray_playback", lambda *_a: None)
    monkeypatch.setattr(view, "_update_projection_text", lambda: None)
    monkeypatch.setattr(view, "_highlight_ray_component", lambda _p: None)
    monkeypatch.setattr(view, "_apply_component_zoom", lambda _p: None)
    monkeypatch.setattr(view, "_update_scale_notice", lambda *_a: None)
    monkeypatch.setattr(view, "_redraw_projection_items", lambda: None)
    monkeypatch.setattr("temsim.gui.visualization.sample_illumination_absent", lambda *_a: False)
    view.show()
    qtbot.waitUntil(view.transverse_beam.isVisible)
    return view


def test_hidden_panels_never_receive_display_result(workspace, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Hidden optional panel performed presentation work")
    monkeypatch.setattr(workspace.physical_layout, "display_result", forbidden)
    monkeypatch.setattr(workspace.magnetic_field, "display_result", forbidden)
    first, last = _result(1.0), _result(2.0)
    workspace.display_result(first, "Preview")
    workspace.display_result(last, "Medium")
    assert workspace.transverse_beam._result is last
    assert len(workspace._pending_ray_panels) == 2
    assert all(result is last for result in workspace._pending_ray_panels.values())
    assert "pending" in workspace.magnetic_field.diagnostic_text("camera")


def test_latest_only_flushes_on_tab_and_toggle_visibility(workspace, qtbot, monkeypatch):
    magnetic, physical = [], []
    monkeypatch.setattr(workspace.magnetic_field, "display_result", magnetic.append)
    monkeypatch.setattr(workspace.physical_layout, "display_result", physical.append)
    workspace.transverse_beam_toggle.setChecked(False)
    first, last = _result(1.0), _result(4.0)
    workspace.display_result(first, "Preview")
    workspace.display_result(last, "Preview")
    assert len(workspace._pending_ray_panels) == 3
    assert workspace.transverse_beam._result is None
    workspace.tabs.setCurrentWidget(workspace.physical_layout)
    qtbot.waitUntil(lambda: len(physical) == 1)
    assert physical == [last]
    workspace.magnetic_field_toggle.setChecked(True)
    qtbot.wait(10)
    assert magnetic == []  # Its Ray Diagram parent is still hidden.
    workspace.show_ray_diagram()
    assert magnetic == [last]
    workspace.transverse_beam_toggle.setChecked(True)
    assert workspace.transverse_beam._result is last
    assert workspace._pending_ray_panels == {}
    workspace.tabs.setCurrentWidget(workspace.physical_layout)
    workspace.show_ray_diagram()
    qtbot.wait(10)
    assert magnetic == [last] and physical == [last]


def test_hidden_workspace_flushes_after_parent_show(workspace, qtbot):
    workspace.hide()
    last = _result(3.0)
    workspace.display_result(_result(), "Preview")
    workspace.display_result(last, "Preview")
    assert workspace.transverse_beam._result is None
    workspace.show()
    qtbot.waitUntil(lambda: workspace.transverse_beam._result is last)


def test_cached_signals_subtab_also_defers_ray_panel(workspace, qtbot):
    workspace.ray_result_tabs.setCurrentIndex(1)
    last = _result(3.0)
    workspace.display_result(last, "Preview")
    assert workspace.transverse_beam._result is None
    workspace.ray_result_tabs.setCurrentIndex(0)
    qtbot.waitUntil(lambda: workspace.transverse_beam._result is last)


def test_hidden_z_and_rotation_apply_once_at_latest_result(workspace, monkeypatch):
    workspace.display_result(_result(), "Preview")
    transverse = workspace.transverse_beam
    transverse._view_scale_initialized = True
    transverse._apply_centered_view_ranges(20.0, 20.0)
    before = np.array(transverse.plot.viewRange())
    calls = []
    redraw = transverse._redraw
    monkeypatch.setattr(transverse, "_redraw", lambda: (calls.append(transverse._result), redraw()))
    workspace.transverse_beam_toggle.setChecked(False)
    workspace._focus_transverse("z", 2.0)
    workspace._set_projection_angle(90.0)
    workspace._focus_transverse("z", 7.0)
    last = _result(2.0)
    workspace.display_result(last, "Medium")
    assert calls == []
    workspace.transverse_beam_toggle.setChecked(True)
    assert calls == [last]
    assert transverse._plane_z_mm == 7.0
    assert transverse._projection_angle_deg == 90.0
    np.testing.assert_allclose(transverse.plot.viewRange(), before, rtol=0, atol=1e-10)
    # A second notification from the axial cursor must not redraw the same Z.
    workspace._focus_transverse("z", 7.0)
    assert calls == [last]


def test_hidden_component_focus_resolves_latest_recording_plane(workspace, monkeypatch):
    transverse = workspace.transverse_beam
    monkeypatch.setattr(transverse, "_add_point_spread_response", lambda: None)
    first, last = _result(detector_z=8.0), _result(detector_z=9.0)
    workspace.display_result(first, "Preview")
    workspace.transverse_beam_toggle.setChecked(False)
    workspace.focus_component(first.assembly.parts[0])
    workspace.display_result(last, "Preview")
    workspace.transverse_beam_toggle.setChecked(True)
    assert transverse._focused_component_key == "camera"
    assert transverse._plane_z_mm == 9.0
    workspace.transverse_beam_toggle.setChecked(False)
    workspace._focus_transverse("z", 6.0)
    workspace.transverse_beam_toggle.setChecked(True)
    assert transverse._focused_component_key is None
    assert transverse._plane_z_mm == 6.0


def test_dirty_physical_layout_preserves_user_ranges(workspace, qtbot, monkeypatch):
    def render(_result):
        workspace.physical_layout.plot.setRange(xRange=(0, 3000), yRange=(-100, 100), padding=0)
    monkeypatch.setattr(workspace.physical_layout, "display_result", render)
    workspace.display_result(_result(), "Preview")
    workspace.tabs.setCurrentWidget(workspace.physical_layout)
    qtbot.waitUntil(lambda: workspace.physical_layout in workspace._presented_ray_panels)
    workspace.physical_layout.plot.setRange(xRange=(1200, 1700), yRange=(-25, 35), padding=0)
    before = np.array(workspace.physical_layout.plot.viewRange())
    workspace.show_ray_diagram()
    workspace.display_result(_result(2.0), "Medium")
    workspace.tabs.setCurrentWidget(workspace.physical_layout)
    qtbot.waitUntil(lambda: workspace.physical_layout not in workspace._pending_ray_panels)
    np.testing.assert_allclose(workspace.physical_layout.plot.viewRange(), before, rtol=0, atol=1e-10)


def test_reopening_magnetic_field_keeps_current_ray_axis(workspace, qtbot, monkeypatch):
    def render(_result):
        # Simulate a field view whose first layout pass has outdated bounds.
        workspace.magnetic_field.plot.setRange(xRange=(0, 3000), yRange=(-1, 1), padding=0)
    monkeypatch.setattr(workspace.magnetic_field, "display_result", render)
    workspace.display_result(_result(), "Preview")
    workspace.magnetic_field_toggle.setChecked(True)
    workspace.magnetic_field_toggle.setChecked(False)
    workspace.plot.setRange(xRange=(4.0, 8.0), yRange=(-2, 3), padding=0)
    qtbot.wait(10)
    workspace.display_result(_result(2.0), "Medium")
    before = np.array(workspace.plot.viewRange())
    workspace.magnetic_field_toggle.setChecked(True)
    qtbot.wait(10)
    np.testing.assert_allclose(workspace.plot.viewRange(), before, rtol=0, atol=1e-10)
    np.testing.assert_allclose(workspace.magnetic_field.plot.viewRange()[0], before[0], rtol=0, atol=1e-10)


def test_removed_component_clears_hidden_detector_focus(workspace, monkeypatch):
    monkeypatch.setattr(workspace.transverse_beam, "_add_point_spread_response", lambda: None)
    first = _result()
    workspace.display_result(first, "Preview")
    workspace.focus_component(first.assembly.parts[0])
    workspace.transverse_beam_toggle.setChecked(False)
    last = _result(2.0)
    last.assembly.parts = ()
    last.state_snapshot.recording_planes = ()
    workspace.display_result(last, "Preview")
    workspace.transverse_beam_toggle.setChecked(True)
    assert workspace.transverse_beam._result is last
    assert workspace.transverse_beam._focused_component_key is None


def test_hidden_panels_do_not_defer_shared_high_accuracy_products(workspace, monkeypatch):
    products = []
    for page in (workspace.probe_aberrations, workspace.image_aberrations,
                 workspace.optical_transfer, workspace.energy_filter, workspace.scan_control,
                 workspace.sample_page, workspace.sample_interactions_3d, workspace.wave_imaging):
        monkeypatch.setattr(page, "display_result", lambda *_a, **_k: None)
    monkeypatch.setattr(workspace.eds_page, "display_result", products.append)
    monkeypatch.setattr(workspace, "_update_sample_region_control_availability", lambda: None)
    high = _result(tuning=False, signature="high")
    workspace.display_result(high, "High accuracy")
    assert products == [high] and workspace._high_accuracy_current
    live = _result(tuning=True, signature="different")
    workspace.display_result(live, "Preview")
    assert products == [high]
    assert workspace._high_accuracy_result is high
    assert not workspace._high_accuracy_current
    assert workspace._pending_ray_panels[workspace.physical_layout] is live


def test_standalone_field_diagnostic_pending_never_reports_previous_numbers(qtbot):
    view = MagneticFieldView()
    qtbot.addWidget(view)
    view._records = (SimpleNamespace(key="lens", peak_t=123.0),)
    view.mark_presentation_pending()
    assert "pending" in view.diagnostic_text("lens")
    assert "123" not in view.diagnostic_text("lens")
    view.display_result(SimpleNamespace(state_snapshot=None))
    assert not view._presentation_pending


def test_standalone_transverse_api_remains_eager(qtbot):
    view = TransverseBeamView()
    qtbot.addWidget(view)
    result = _result()
    view.display_result(result)
    assert view._scatter is not None
    view.focus_z(5.0)
    assert "Z = 5 mm" in view.heading.text()
    view.set_projection_angle(90.0)
    assert view._projection_angle_deg == 90.0
