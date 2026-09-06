"""Display-cache equivalence and GUI interaction regressions (no solver runs)."""

import gc
from types import SimpleNamespace
import weakref

import numpy as np
import pyqtgraph as pg
import pytest

from temsim.gui.visualization import VisualizationWorkspace


def _branch():
    rng = np.random.default_rng(431)
    shape = (21, 75)
    return SimpleNamespace(
        name="incident", z=np.linspace(1.0, 11.0, shape[0]),
        x=rng.normal(0.0, 1e-4, shape), y=rng.normal(0.0, 1e-4, shape),
        tx=rng.normal(0.0, 0.01, shape), ty=rng.normal(0.0, 0.01, shape),
        blocked_z=np.resize(np.array([np.nan, 0.5, 1.0, 2.25, 7.0, 15.0]), shape[1]),
    )


@pytest.fixture
def workspace(qtbot):
    view = VisualizationWorkspace()
    qtbot.addWidget(view)
    return view


def _reference(view, branch, indices=None):
    if indices is None:
        indices = view._display_ray_indices(branch)
    x, y = branch.x[:, indices], branch.y[:, indices]
    offset = view._scan_ray_offsets_m.get(branch.name)
    if offset is not None:
        x, y = x + offset[:, 0, None], y + offset[:, 1, None]
    return view._bundle_lines(
        branch.z, view._project_transverse(x, y), len(indices), branch.blocked_z[indices]
    )


@pytest.mark.parametrize("angle", [0.0, 37.25, 90.0, 180.0, 279.9, 360.0])
@pytest.mark.parametrize("scan", [False, True])
def test_cached_lines_match_exact_blocking_and_scan_interpolation(workspace, angle, scan):
    branch = _branch()
    x_before, y_before = branch.x.copy(), branch.y.copy()
    workspace._display_bundle_lines(branch)  # Prime at a different angle.
    workspace._projection_angle_deg = angle
    if scan:
        workspace._scan_ray_offsets_m[branch.name] = np.column_stack((
            np.sin(branch.z) * 2e-4, np.cos(branch.z) * 1e-4,
        ))
    actual = workspace._display_bundle_lines(branch)
    expected = _reference(workspace, branch)
    np.testing.assert_allclose(actual[0], expected[0], rtol=0, atol=0, equal_nan=True)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-12, atol=1e-13, equal_nan=True)
    np.testing.assert_array_equal(branch.x, x_before)
    np.testing.assert_array_equal(branch.y, y_before)
    assert branch.x.flags.writeable and branch.y.flags.writeable
    assert workspace.ray_display_cache_info()["hits"] == 1


def test_line_cache_is_bounded_and_zero_budget_still_draws(workspace):
    branch = _branch()
    expected = workspace._display_bundle_lines(branch)
    size = workspace.ray_display_cache_info()["used_bytes"]
    assert size > 0
    workspace.set_ray_display_cache_limit_bytes(size)
    workspace._display_bundle_lines(branch, np.array([1, 2, 3]))
    assert workspace.ray_display_cache_info()["used_bytes"] <= size
    workspace.set_ray_display_cache_limit_bytes(0)
    actual = workspace._display_bundle_lines(branch)
    np.testing.assert_allclose(actual[1], expected[1], equal_nan=True)
    assert workspace.ray_display_cache_info()["entries"] == 0
    assert workspace.ray_display_cache_info()["used_bytes"] == 0
    with pytest.raises(ValueError):
        workspace.set_ray_display_cache_limit_bytes(-1)


def test_cache_does_not_retain_physical_histories_and_replacement_misses(workspace):
    branch = _branch()
    workspace._display_bundle_lines(branch)
    old = weakref.ref(branch.x)
    branch.x = branch.x + 0.002
    gc.collect()
    assert old() is None
    actual = workspace._display_bundle_lines(branch)
    expected = _reference(workspace, branch)
    np.testing.assert_allclose(actual[1], expected[1], equal_nan=True)
    assert workspace.ray_display_cache_info()["misses"] == 2


@pytest.mark.parametrize("angle", [0.0, 32.7, 90.0, 239.8])
def test_cached_pan_slope_matches_visible_unblocked_samples(workspace, angle):
    branch = _branch()
    branch.tx[2, 0] = np.nan
    branch.ty[4, 0] = np.inf
    workspace._last_result = SimpleNamespace(simulation=SimpleNamespace(incident=branch, branches={}))
    workspace._projection_angle_deg = angle
    bounds = [[0.0, 12.0], [-1.0, 1.0]]
    workspace.plot = SimpleNamespace(getViewBox=lambda: SimpleNamespace(viewRange=lambda: bounds))
    for lo, hi in [(0.0, 12.0), (4.25, 7.1), (20.0, 22.0), (1.0, 1.0), (0.0, 12.0)]:
        bounds[0] = [lo, hi]
        in_window = (branch.z >= lo) & (branch.z <= hi)
        with np.errstate(invalid="ignore"):
            slopes = workspace._project_transverse(branch.tx[in_window], branch.ty[in_window])
        valid = np.isnan(branch.blocked_z)[None, :] | (branch.z[in_window, None] <= branch.blocked_z)
        values = np.abs(slopes[np.isfinite(slopes) & valid])
        with np.errstate(invalid="ignore"):
            actual = workspace._maximum_visible_projection_slope()
        assert actual == pytest.approx(np.max(values) if values.size else 0.0, abs=1e-14)
    assert workspace.ray_display_cache_info()["hits"] >= 3


def test_rotation_keeps_graphics_ranges_and_detector_offset_current(workspace, monkeypatch):
    branch = _branch()
    workspace._last_result = SimpleNamespace(simulation=SimpleNamespace(incident=branch, branches={}, metrics={}))
    marker = pg.InfiniteLine(pos=4.0)
    workspace.plot.addItem(marker)
    item = workspace.plot.plot(*workspace._display_bundle_lines(branch))
    workspace._ray_bundle_records = [(item, branch)]
    part = SimpleNamespace(key="bf", name="BF Detector", data={}, center_z_mm=10.0)
    detector = SimpleNamespace(key="bf", z_mm=10.0, outer_width_mm=2.0, inner_diameter_mm=0.0,
                               centre_offset_x_mm=0.5, centre_offset_y_mm=-0.25)
    result = SimpleNamespace(state_snapshot=SimpleNamespace(recording_planes=[detector]))
    monkeypatch.setattr(workspace, "_aperture_optical_plane", lambda _part: 10.0)
    workspace._add_recording_surface_range(result, part, 0)
    detector_item = workspace.recording_surface_range_items[0]
    aperture = workspace._add_aperture_stop(
        SimpleNamespace(name="Offset aperture"), 5.0,
        {"enabled": True, "installed": True, "diameter_mm": 2.0,
         "offset_x_mm": 0.5, "offset_y_mm": -0.25},
    )
    workspace.plot.setRange(xRange=(2.0, 8.0), yRange=(-2.0, 2.0), padding=0.0, disableAutoRange=True)
    before = workspace.plot.getViewBox().viewRange()
    monkeypatch.setattr(workspace, "_draw_ray_diagram", lambda *_args, **_kwargs: pytest.fail("Rotation cleared the scene"))
    workspace._set_projection_angle(90.0)
    assert marker in workspace.plot.plotItem.items
    assert workspace._ray_bundle_records[0][0] is item
    assert workspace.recording_surface_range_items[0] is detector_item
    assert detector_item.active_intervals_mm[0] == pytest.approx((-1.25, 0.75))
    assert "Allowed Y opening = [-1.25, 0.75] mm" in aperture.toolTip()
    assert aperture.label.toolTip() == aperture.toolTip()
    np.testing.assert_allclose(workspace.plot.getViewBox().viewRange(), before, rtol=0, atol=1e-12)
    np.testing.assert_allclose(item.getData()[1], _reference(workspace, branch)[1], equal_nan=True)


def test_republishing_result_invalidates_display_cache(workspace, monkeypatch):
    branch = _branch()
    result = SimpleNamespace(simulation=SimpleNamespace(metrics={}, incident=branch, branches={}))
    workspace._display_bundle_lines(branch)
    assert workspace.ray_display_cache_info()["entries"] == 1
    # Stop after the publication boundary, without invoking unrelated pages.
    class DisplayReached(Exception):
        pass
    def observe_diagram(*_args, **_kwargs):
        assert workspace.ray_display_cache_info()["entries"] == 0
        raise DisplayReached
    monkeypatch.setattr(workspace, "_draw_ray_diagram", observe_diagram)
    monkeypatch.setattr("temsim.gui.visualization.sample_illumination_absent", lambda *_: False)
    with pytest.raises(DisplayReached):
        workspace.display_result(result, "Preview")
    workspace._display_bundle_lines(branch)
    with pytest.raises(DisplayReached):
        workspace.display_result(result, "Preview")


def test_projection_finalize_flushes_latest_pending_angle_once(workspace, monkeypatch):
    workspace._last_result = object()
    angles = []
    monkeypatch.setattr(workspace, "_redraw_projection_items", lambda: angles.append(workspace._projection_angle_deg))
    monkeypatch.setattr(workspace, "_update_scale_notice", lambda: None)
    workspace._set_projection_angle(3.0, defer_redraw=True)
    workspace._set_projection_angle(51.5, defer_redraw=True)
    assert workspace._projection_redraw_timer.isActive()
    workspace._projection_finalize_timer.stop()
    workspace._finalize_projection()
    assert angles == [51.5]
    assert not workspace._projection_redraw_timer.isActive()
    workspace._finalize_projection()
    assert angles == [51.5]
