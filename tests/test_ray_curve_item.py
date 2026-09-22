"""Display-only geometry bounds; no trajectory or specimen calculation."""

import numpy as np
import pyqtgraph as pg
import pytest

from temsim.gui.ray_curve_item import RayCurveItem, finite_runs, screen_path_indices


def test_simplification_bounds_screen_error_and_keeps_narrow_tail():
    x = np.linspace(0, 10, 5001)
    y = 0.2*np.sin(x*3)
    y[2377] = 7.0  # A one-sample scattering excursion may not disappear.
    sx, sy = .012, .02
    selected = screen_path_indices(x, y, sx, sy)
    assert selected[0] == 0 and selected[-1] == len(x)-1
    assert 2377 in selected
    assert len(selected) < len(x)//10
    xy = np.column_stack((x/sx, y/sy))
    for first, last in zip(selected[:-1], selected[1:]):
        delta = xy[last] - xy[first]
        points = xy[first:last+1] - xy[first]
        fraction = np.clip(points @ delta / (delta @ delta), 0, 1)
        distances = np.linalg.norm(points - fraction[:, None]*delta, axis=1)
        assert np.max(distances) <= .35 + 1e-10


def test_zero_length_chord_and_reversal_preserve_excursion():
    x = np.r_[np.linspace(0, 1, 100), np.linspace(1, 0, 100)]
    y = np.sin(x)
    selected = screen_path_indices(x, y, .01, .01)
    assert x[selected].max() == 1
    assert selected[0] == 0 and selected[-1] == 199


@pytest.mark.parametrize("case", ("smooth", "tail", "reverse", "noisy"))
def test_compiled_and_numpy_subsets_agree(case, monkeypatch):
    import temsim.gui.ray_curve_item as module
    if module.compiled_rdp is None:
        pytest.skip("Optional native compiler is unavailable")
    x = np.linspace(0, 10, 1025)
    y = np.sin(x)
    if case == "tail":
        y[123] = 50.
    elif case == "reverse":
        x = np.sin(x)
    elif case == "noisy":
        y = np.random.default_rng(431).normal(size=len(x))
    accelerated = screen_path_indices(x, y, .01, .02)
    assert module.compiled_rdp is not None
    assert module.compiled_rdp.signatures
    monkeypatch.setattr(module, "compiled_rdp", None)
    fallback = screen_path_indices(x, y, .01, .02)
    np.testing.assert_array_equal(accelerated, fallback)


def test_failed_compilation_uses_numpy_fallback(monkeypatch):
    import temsim.gui.ray_curve_item as module
    x = np.linspace(0, 10, 1001)
    y = np.sin(x)
    monkeypatch.setattr(module, "compiled_rdp", None)
    expected = screen_path_indices(x, y, .01, .01)
    def unavailable(*_args):
        raise RuntimeError("Native compiler unavailable")
    monkeypatch.setattr(module, "compiled_rdp", unavailable)
    np.testing.assert_array_equal(screen_path_indices(x, y, .01, .01), expected)
    assert module.compiled_rdp is None


@pytest.fixture
def graph(qtbot):
    plot = pg.PlotWidget()
    qtbot.addWidget(plot)
    plot.resize(1000, 600)
    plot.show()
    plot.setRange(xRange=(0, 10), yRange=(-1, 2), padding=0, disableAutoRange=True)
    qtbot.wait(10)
    return plot


def _item(plot):
    x = np.linspace(0, 10, 5001)
    y = np.sin(x)
    # Two independent rays with overlapping Z grids, plus an internal gap.
    xx = np.r_[x, np.nan, x]
    yy = np.r_[y, np.nan, y + .5]
    yy[3200] = np.nan
    item = RayCurveItem(xx, yy, pen=pg.mkPen("#ffff00", width=1.35))
    plot.addItem(item)
    return item, xx, yy


def test_full_data_bounds_gaps_and_stops_survive_navigation(graph):
    item, x, y = _item(graph)
    original = (x.copy(), y.copy())
    assert len(finite_runs(*item.curve.getData())) == 3
    assert item.rendering_info()["rendered_points"] < len(x)//10
    for a, b in ((4, 7), (20, 30), (0, 10)):
        graph.setXRange(a, b, padding=0)
        np.testing.assert_array_equal(item.getData()[0], original[0])
        np.testing.assert_array_equal(item.getData()[1], original[1])
        assert item.dataBounds(0) == (0.0, 10.0)
    # Every independent finite run retains its actual start/end; never connect
    # a stopped ray to the next ray at a smaller Z or across a missing sample.
    draw_x, draw_y = item.curve.getData()
    expected = finite_runs(x, y)
    actual = finite_runs(draw_x, draw_y)
    assert len(actual) == len(expected)
    for (a, b), (c, d) in zip(expected, actual):
        np.testing.assert_array_equal([draw_x[c], draw_x[d-1]], [x[a], x[b-1]])
        np.testing.assert_array_equal([draw_y[c], draw_y[d-1]], [y[a], y[b-1]])


def test_clipping_retains_crossing_edges_with_no_internal_samples(graph):
    item = RayCurveItem([0., 10., np.nan, 0., 10.], [0., 1., np.nan, 1., 0.])
    graph.addItem(item)
    graph.setXRange(4, 6, padding=0)
    dx, dy = item.curve.getData()
    np.testing.assert_array_equal(dx, [0., 10., np.nan, 0., 10., np.nan])
    assert len(finite_runs(dx, dy)) == 2


def test_nonmonotonic_path_is_never_sorted_or_incorrectly_clipped(graph):
    x = np.r_[np.linspace(0, 10, 100), np.linspace(10, 0, 100)]
    y = np.r_[np.zeros(100), np.ones(100)]
    item = RayCurveItem(x, y)
    graph.addItem(item)
    graph.setXRange(3, 5, padding=0)
    dx, _ = item.curve.getData()
    assert dx[0] == dx[-2] == 0
    assert np.any(np.diff(dx) < 0)
    assert np.nanmax(dx) == 10


def test_pan_reuses_level_and_resizing_or_zoom_restores_detail(graph, monkeypatch, qtbot):
    item, _, _ = _item(graph)
    initial_points = item.rendering_info()["rendered_points"]
    initial_levels = tuple(item._ray_levels)
    import temsim.gui.ray_curve_item as module
    original = module.screen_path_indices
    calls = []
    monkeypatch.setattr(module, "screen_path_indices", lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    graph.setRange(xRange=(1, 11), yRange=(1, 4), padding=0)
    assert calls == []
    assert tuple(item._ray_levels) == initial_levels
    graph.setRange(xRange=(0, 10), yRange=(-.01, .01), padding=0)
    assert calls
    assert item.rendering_info()["rendered_points"] > initial_points
    old_levels = tuple(item._ray_levels)
    graph.resize(2000, 1800)
    qtbot.wait(20)
    assert tuple(item._ray_levels) != old_levels
    for n in range(8):
        graph.setYRange(-2.**n, 2.**n, padding=0)
    assert len(item._ray_levels) <= item.MAX_LEVELS


def test_republication_invalidates_levels_even_reusing_input(graph):
    item, x, y = _item(graph)
    y[1234] = 100.
    item.setData(x, y)
    assert item.dataBounds(1)[1] == 100.
    assert np.nanmax(item.curve.getData()[1]) == 100.
    item.setData([], [])
    assert item.dataBounds(0) == (None, None)
    assert item.rendering_info()["rendered_points"] == 0
    assert len(item._ray_levels) == 0


def test_cache_byte_limit_can_disable_retention_without_losing_paths(graph):
    item, x, y = _item(graph)
    item.MAX_CACHE_BYTES = 0
    item.setData(x, y)
    assert len(item._ray_levels) == 0
    assert item.rendering_info()["cached_bytes"] == 0
    assert len(finite_runs(*item.curve.getData())) == 3
    for n in range(5):
        graph.setXRange(n*.1, 10+n*.1, padding=0)
        assert item.rendering_info()["cached_bytes"] == 0
    np.testing.assert_array_equal(item.getData()[1], y)


def test_export_uses_full_coordinates_then_restores_screen_subset(graph):
    item, x, y = _item(graph)
    item.setExportMode(True, {"antialias": True})
    np.testing.assert_array_equal(item.curve.getData()[0], x)
    np.testing.assert_array_equal(item.curve.getData()[1], y)
    item.setExportMode(False)
    assert item.rendering_info()["rendered_points"] < len(x)//10


def test_fit_uses_full_paths_after_zoom_and_blocked_tail_is_not_extended(graph):
    x = np.linspace(0, 10, 501)
    y = np.sin(x)
    item = RayCurveItem(x, y)
    graph.addItem(item)
    graph.setXRange(3, 5, padding=0)
    graph.autoRange()
    left, right = graph.viewRange()[0]
    assert left <= 0 and right >= 10
    draw_x, _ = item.curve.getData()
    assert np.nanmax(draw_x) == 10
