"""Recorded source display only: no source sampling or beam propagation."""

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import Qt

from temsim.gui.emission_source_plot import EmissionSourcePlot


@pytest.fixture
def source_plot(qtbot):
    widget = EmissionSourcePlot()
    qtbot.addWidget(widget)
    widget.resize(430, 240)
    widget.show()
    qtbot.wait(10)
    return widget


def launch_data():
    # Two directions emitted from exactly the same surface position.
    return dict(
        ids=np.array([15, 82, 90]),
        position_m=np.array([[2., -4., -.5], [2., -4., -.5], [-3., 5., -1.]]) * 1e-9,
        brushes=[pg.mkBrush("red"), pg.mkBrush("blue"), pg.mkBrush("green")],
        azimuth_rad=np.array([0., np.pi / 2., np.nan]),
        angle_to_normal_rad=np.array([.1, .2, np.nan]),
    )


def test_actual_source_positions_keep_overlap_and_original_hover(source_plot):
    data = launch_data()
    original = data["position_m"].copy()
    source_plot.set_source(**data, status="Original source sample")
    np.testing.assert_allclose(source_plot.scatter.data["x"], [2., 2., -3.])
    np.testing.assert_allclose(source_plot.scatter.data["y"], [-4., -4., 5.])
    assert [record["source_ray_id"] for record in source_plot.scatter.data["data"]] == [15, 82, 90]
    assert [brush.color().name() for brush in source_plot.scatter.data["brush"]] == ["#ff0000", "#0000ff", "#008000"]
    hover = source_plot._hover_text(99., 99., source_plot.scatter.data["data"][1])
    assert "Source ray 82" in hover and "Original X 2 nm" in hover
    assert "Z -0.5 nm" in hover and "Launch azimuth 90°" in hover
    assert "Original source sample" in source_plot.summary.text()
    assert source_plot.summary.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse
    np.testing.assert_array_equal(data["position_m"], original)
    for name in ("bottom", "left"):
        assert source_plot.plot.getAxis(name).labelUnits == "nm"
        assert source_plot.plot.getAxis(name).autoSIPrefixScale == 1.0


def test_projection_rotates_only_display_and_keeps_original_colours(source_plot):
    source_plot.set_source(**launch_data())
    before = source_plot.plot.viewRange()
    source_plot.set_projection_angle(90.)
    np.testing.assert_allclose(source_plot.scatter.data["x"], [-4., -4., 5.])
    np.testing.assert_allclose(source_plot.scatter.data["y"], [-2., -2., 3.])
    np.testing.assert_allclose(source_plot.plot.viewRange(), before)
    assert source_plot.plot.getAxis("bottom").labelText == "Source Y"
    assert source_plot.plot.getAxis("left").labelText == "Source -X"
    assert source_plot.scatter.data["brush"][1].color().name() == "#0000ff"
    assert source_plot.scatter.data["data"][1]["position_m"] == pytest.approx((2e-9, -4e-9, -.5e-9))


def test_recolour_and_angle_changes_preserve_manual_pan_zoom(source_plot):
    data = launch_data()
    source_plot.set_source(**data)
    source_plot.plot.setRange(xRange=(30., 90.), yRange=(-60., -20.), padding=0.)
    before = np.asarray(source_plot.plot.viewRange())
    data["brushes"] = [pg.mkBrush("white")] * 3
    data["azimuth_rad"] = np.array([.5, .6, .7])
    source_plot.set_source(**data)
    np.testing.assert_allclose(source_plot.plot.viewRange(), before)
    assert all(brush.color().name() == "#ffffff" for brush in source_plot.scatter.data["brush"])
    data["position_m"] *= 100.
    source_plot.set_source(**data)
    assert not np.allclose(source_plot.plot.viewRange(), before)
    for name, column in (("x", 0), ("y", 1)):
        low, high = source_plot.plot.viewRange()[column]
        assert np.all((source_plot.scatter.data[name] >= low) & (source_plot.scatter.data[name] <= high))


def test_external_mutation_does_not_change_recorded_display(source_plot):
    data = launch_data()
    source_plot.set_source(**data)
    data["ids"][0] = 999
    data["position_m"][0] = 1.
    data["azimuth_rad"][0] = 2.
    source_plot.set_projection_angle(90.)
    first = source_plot.scatter.data["data"][0]
    assert first["source_ray_id"] == 15
    assert first["position_m"] == pytest.approx((2e-9, -4e-9, -.5e-9))
    assert first["azimuth_rad"] == 0.


def test_clear_and_partial_missing_data_do_not_keep_stale_source(source_plot):
    data = launch_data()
    data["position_m"][0] = np.nan
    source_plot.set_source(**data)
    assert len(source_plot.scatter.data) == 2
    assert "1 positions unavailable" in source_plot.summary.text()
    hover = source_plot._hover_text(0., 0., source_plot.scatter.data["data"][-1])
    assert "Launch azimuth unavailable" in hover
    source_plot.clear("Historical result has no saved emission positions.")
    assert source_plot.scatter is None
    assert not any(isinstance(item, pg.ScatterPlotItem) for item in source_plot.plot.items())
    assert "Historical result" in source_plot.summary.text()
    source_plot.set_projection_angle(37.)
    assert source_plot.scatter is None


@pytest.mark.parametrize("field,value", [
    ("ids", [1., 2., 3.]),
    ("position_m", np.zeros((3, 2))),
    ("azimuth_rad", np.zeros(2)),
    ("angle_to_normal_rad", np.zeros((3, 1))),
    ("brushes", [pg.mkBrush("red")]),
])
def test_malformed_data_does_not_replace_current_view(source_plot, field, value):
    data = launch_data()
    source_plot.set_source(**data)
    previous = source_plot.scatter
    data[field] = value
    with pytest.raises(ValueError):
        source_plot.set_source(**data)
    assert source_plot.scatter is previous


def test_empty_source_is_explicit(source_plot):
    source_plot.set_source(
        ids=np.empty(0, dtype=np.int64), position_m=np.empty((0, 3)), brushes=[],
        azimuth_rad=np.empty(0), angle_to_normal_rad=np.empty(0),
    )
    assert source_plot.scatter is None
    assert "No finite recorded emission positions" in source_plot.summary.text()


def test_fit_button_restores_source_view_without_new_data(source_plot, qtbot):
    assert not source_plot.fit_button.isEnabled()
    source_plot.fit_to_source()
    assert source_plot.scatter is None
    source_plot.set_source(**launch_data())
    assert source_plot.fit_button.isEnabled()
    fitted = np.asarray(source_plot.plot.viewRange())
    source_ids = [record["source_ray_id"] for record in source_plot.scatter.data["data"]]
    source_plot.plot.setRange(xRange=(100., 200.), yRange=(100., 200.), padding=0.)
    assert not np.allclose(source_plot.plot.viewRange(), fitted)
    qtbot.mouseClick(source_plot.fit_button, Qt.MouseButton.LeftButton)
    np.testing.assert_allclose(source_plot.plot.viewRange(), fitted)
    assert [record["source_ray_id"] for record in source_plot.scatter.data["data"]] == source_ids
    source_plot.clear("No recorded source positions.")
    assert not source_plot.fit_button.isEnabled()


def test_coincident_xy_hover_exposes_each_launch_direction_and_z(source_plot):
    data = launch_data()
    data["position_m"][1, 2] = -2e-9
    source_plot.set_source(**data)
    assert "2 overlapping samples at 1 source position" in source_plot.summary.text()
    record = source_plot.scatter.data["data"][0]
    hover = source_plot._hover_text(0., 0., record)
    assert "At this recorded X/Y: 2 displayed samples" in hover
    assert "not intensity or probability" in hover
    assert "Source ray 15 | Z -0.5 nm | Launch azimuth 0°" in hover
    assert "Source ray 82 | Z -2 nm | Launch azimuth 90°" in hover
    assert "angle to local normal 11.4592°" in hover
    np.testing.assert_allclose(source_plot.scatter.data["x"][:2], [2., 2.])
    np.testing.assert_allclose(source_plot.scatter.data["y"][:2], [-4., -4.])
    assert source_plot.scatter.data["brush"][0].color().name() == "#ff0000"
    assert source_plot.scatter.data["brush"][1].color().name() == "#0000ff"
    source_plot.set_projection_angle(43.)
    assert len(source_plot.scatter.data["data"][0]["coincident_samples"]) == 2


def test_coincident_hover_is_bounded_without_discarding_original_samples(source_plot):
    count = 12
    source_plot.set_source(
        ids=np.arange(count),
        position_m=np.column_stack((np.ones(count), np.ones(count)*2, np.arange(count))) * 1e-9,
        brushes=[pg.mkBrush("red")] * count,
        azimuth_rad=np.arange(count) * .1,
        angle_to_normal_rad=np.arange(count) * .01,
    )
    assert len(source_plot.scatter.data) == count
    assert "12 overlapping samples at 1 source position" in source_plot.summary.text()
    first = source_plot.scatter.data["data"][0]
    last = source_plot.scatter.data["data"][-1]
    assert first["coincident_samples"] is last["coincident_samples"]
    hover = source_plot._hover_text(0., 0., first)
    assert "Source ray 7 | Z 7 nm" in hover
    assert "Source ray 8 |" not in hover
    assert "Plus 4 more coincident samples." in hover
    last_hover = source_plot._hover_text(0., 0., last)
    assert "Source ray 11 | Z 11 nm" in last_hover
    assert last_hover.count("Launch azimuth") == 8


@pytest.mark.parametrize("hovered_index", [0, 1, 2])
def test_same_source_descendants_keep_each_path_time_and_launch_hover(source_plot, hovered_index):
    # One emitted electron can contribute several descendant paths, including
    # a path whose scattering history does not provide an arrival time.
    positions = np.tile([2e-9, -4e-9, -.5e-9], (3, 1))
    source_plot.set_source(
        ids=np.full(3, 15), position_m=positions,
        brushes=[pg.mkBrush("red"), pg.mkBrush("blue"), pg.mkBrush("gray")],
        azimuth_rad=np.full(3, np.pi / 2.), angle_to_normal_rad=np.full(3, .2),
        flight_time_s=np.array([1e-9, 1.00002e-9, np.nan]),
        reference_time_s=1e-9,
    )
    records = source_plot.scatter.data["data"]
    hover = source_plot._hover_text(0., 0., records[hovered_index])
    paths = [line for line in hover.splitlines() if line.startswith("Source ray 15 |")]
    assert len(paths) == 3
    assert sum("Flight time 1 ns | Delay 0 fs" in line for line in paths) == 1
    assert sum("Flight time 1.00002 ns | Delay 20 fs" in line for line in paths) == 1
    assert sum("Flight time unavailable" in line for line in paths) == 1
    assert all("Z -0.5 nm | Launch azimuth 90° | angle to local normal 11.4592°" in line for line in paths)
    assert "At this recorded X/Y: 3 displayed samples" in hover
    np.testing.assert_allclose(source_plot.scatter.data["x"], [2., 2., 2.])
    np.testing.assert_allclose(source_plot.scatter.data["y"], [-4., -4., -4.])
    np.testing.assert_array_equal(source_plot._positions, positions)


def test_same_source_path_hover_limit_includes_hovered_path_exactly_once(source_plot):
    count = 12
    source_plot.set_source(
        ids=np.full(count, 15), position_m=np.zeros((count, 3)),
        brushes=[pg.mkBrush("red")] * count,
        azimuth_rad=np.zeros(count), angle_to_normal_rad=np.zeros(count),
        flight_time_s=np.arange(1, count + 1) * 1e-9, reference_time_s=1e-9,
    )
    record = source_plot.scatter.data["data"][-1]
    hover = source_plot._hover_text(0., 0., record)
    paths = [line for line in hover.splitlines() if line.startswith("Source ray 15 |")]
    assert len(paths) == 8
    assert "Flight time 12 ns | Delay 11000000 fs" in paths[0]
    assert sum("Flight time 12 ns" in line for line in paths) == 1
    for line, time_ns in zip(paths[1:], range(1, 8)):
        assert f"Flight time {time_ns} ns | Delay" in line
    assert "Plus 4 more coincident samples." in hover
    assert len(record["coincident_samples"]) == count
