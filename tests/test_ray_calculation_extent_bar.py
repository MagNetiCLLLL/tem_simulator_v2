"""Spatial result extent follows the plot viewport, never requested progress."""

import math

import pyqtgraph as pg
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget

from temsim.gui.ray_calculation_extent import RayCalculationExtentBar


@pytest.fixture
def shown(qtbot):
    host = QWidget()
    layout = QVBoxLayout(host)
    layout.setContentsMargins(19, 11, 7, 13)
    plot = pg.PlotWidget()
    plot.setLabel("left", "Transverse position", units="mm")
    plot.setLabel("bottom", "Physical Z", units="mm")
    bar = RayCalculationExtentBar()
    layout.addWidget(plot, 1)
    layout.addWidget(bar)
    host.resize(850, 360)
    qtbot.addWidget(host)
    host.show()
    bar.bind_plot(plot)
    plot.setRange(xRange=(0., 100.), yRange=(-1., 1.), padding=0.)
    qtbot.wait(15)
    return host, plot, bar


def plot_global_x(plot, z):
    scene = plot.getViewBox().mapViewToScene(QPointF(z, 0.))
    viewport = plot.viewportTransform().map(scene)
    return plot.viewport().mapToGlobal(QPoint(0, 0)).x() + viewport.x()


def test_initial_state_and_summary_are_explicit_and_copyable(shown):
    _host, _plot, bar = shown
    assert bar.label.text() == "No completed particle calculation"
    assert bar.display_geometry()["completed_span_px"] is None
    assert bar.extent["completed_z_mm"] is None
    assert bar.label.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse
    assert bar.label.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByKeyboard
    assert 40 <= bar.height() <= 50
    assert "not elapsed time" in bar.toolTip()
    assert "individual electrons can stop earlier" in bar.toolTip()
    assert "upstream inputs must still match" in bar.toolTip()


def test_marker_coordinates_include_real_axis_margins(shown):
    _host, plot, bar = shown
    bar.set_extent(0., 45., 40., 80., quality="High accuracy")
    geometry = bar.display_geometry()
    origin = bar.mapToGlobal(QPoint(0, 0)).x()
    for marker in geometry["markers"].values():
        assert origin + marker["x_px"] == pytest.approx(plot_global_x(plot, marker["z_mm"]), abs=1e-9)
        assert marker["in_view"]
    track = geometry["track_rect"]
    assert track.left() > 20.  # The left axis consumes actual plot width.
    start, stop = geometry["completed_span_px"]
    assert start == pytest.approx(bar.map_z_to_x(0.))
    assert stop == pytest.approx(bar.map_z_to_x(45.))
    assert "Calculated 45" in bar.label.text()
    assert "High accuracy" not in bar.label.text()
    assert "Quality: High accuracy" in bar.toolTip()


def test_requested_cutoff_never_becomes_completed_transport(shown):
    _host, _plot, bar = shown
    bar.set_extent(start_z_mm=0., requested_z_mm=90.)
    geometry = bar.display_geometry()
    assert geometry["completed_span_px"] is None
    assert set(geometry["markers"]) == {"requested"}
    assert "No completed particle calculation" in bar.label.text()
    bar.set_extent(0., 30., requested_z_mm=90.)
    geometry = bar.display_geometry()
    assert geometry["completed_span_px"][1] == pytest.approx(bar.map_z_to_x(30.))
    assert geometry["markers"]["requested"]["x_px"] > geometry["completed_span_px"][1]
    bar.set_extent(0., 30., requested_z_mm=10.)
    assert bar.display_geometry()["completed_span_px"][1] == pytest.approx(bar.map_z_to_x(30.))


def test_historical_result_with_unknown_extent_is_not_called_uncalculated(shown):
    _host, _plot, bar = shown
    bar.set_extent(quality="High accuracy", requested_z_mm=80.)
    assert "Calculated range unavailable" in bar.label.text()
    assert "No completed particle calculation" not in bar.label.text()
    assert bar.display_geometry()["completed_span_px"] is None
    assert bar.extent["completed_z_mm"] is None


def test_stale_changes_colour_and_text_without_changing_completed_position(shown):
    _host, _plot, bar = shown
    bar.set_extent(0., 60., 50., 80., quality="High accuracy")
    fresh = bar.display_geometry()
    bar.set_extent(0., 60., 50., 80., stale=True, quality="High accuracy")
    stale = bar.display_geometry()
    assert stale["completed_span_px"] == fresh["completed_span_px"]
    assert stale["markers"] == fresh["markers"]
    assert stale["completed_color"] != fresh["completed_color"]
    assert stale["completed_color"] == bar.COLORS["stale"]
    assert bar.label.text().startswith("Previous")
    assert "Parameters changed" in bar.toolTip()
    assert bar.extent["stale"]


def test_zoom_pan_and_resize_remap_positions_automatically(shown, qtbot):
    host, plot, bar = shown
    bar.set_extent(0., 45., 40., 80.)
    before = bar.map_z_to_x(45.)
    plot.setXRange(20., 70., padding=0.)
    qtbot.wait(15)
    assert bar.map_z_to_x(45.) != before
    assert bar.display_geometry()["view_z_mm"] == pytest.approx((20., 70.))
    assert "right of view" in bar.label.text()
    old_width = bar.display_geometry()["track_rect"].width()
    host.resize(1150, 430)
    qtbot.wait(15)
    assert bar.display_geometry()["track_rect"].width() > old_width+100.
    origin = bar.mapToGlobal(QPoint(0, 0)).x()
    assert origin+bar.map_z_to_x(45.) == pytest.approx(plot_global_x(plot, 45.), abs=1e-9)
    plot.getViewBox().translateBy(x=30.)
    qtbot.wait(15)
    assert "Calculated 45 (left of view)" in bar.label.text()


def test_off_view_markers_retain_true_values_and_are_not_clamped(shown):
    _host, _plot, bar = shown
    bar.set_extent(-100., -10., -20., 150.)
    geometry = bar.display_geometry()
    left, right = geometry["track_rect"].left(), geometry["track_rect"].right()
    completed, requested = geometry["markers"]["completed"], geometry["markers"]["requested"]
    assert completed["z_mm"] == -10. and completed["x_px"] < left
    assert completed["offscreen"] == "left" and not completed["in_view"]
    assert requested["z_mm"] == 150. and requested["x_px"] > right
    assert requested["offscreen"] == "right" and not requested["in_view"]
    assert geometry["completed_span_px"] is None
    assert "Calculated -10 (left of view)" in bar.label.text()
    assert "Cutoff 150 (right of view)" in bar.label.text()


def test_reversed_plot_axis_maps_positions_and_offscreen_sides(shown, qtbot):
    _host, plot, bar = shown
    plot.getViewBox().invertX(True)
    bar.set_extent(0., 40., 40., 150.)
    qtbot.wait(15)
    geometry = bar.display_geometry()
    assert bar.map_z_to_x(0.) > bar.map_z_to_x(100.)
    assert geometry["markers"]["requested"]["offscreen"] == "left"
    assert "resumable" not in geometry["markers"]
    span = geometry["completed_span_px"]
    assert span == pytest.approx((bar.map_z_to_x(40.), bar.map_z_to_x(0.)))


def test_unknown_and_invalid_start_do_not_invent_a_completed_interval(shown):
    _host, _plot, bar = shown
    bar.set_extent(None, 40., requested_z_mm=80.)
    assert bar.display_geometry()["completed_span_px"] is None
    assert bar.extent["start_z_mm"] is None
    bar.set_extent(80., 40., requested_z_mm=float("nan"))
    assert bar.display_geometry()["completed_span_px"] is None
    assert bar.extent["requested_z_mm"] is None
    assert bar.map_z_to_x(math.inf) is None


def test_plot_binding_can_be_removed_without_consuming_plot_input(shown, qtbot):
    _host, plot, bar = shown
    assert bar.eventFilter(plot, QEvent(QEvent.Type.Wheel)) is False
    assert bar.eventFilter(plot, QEvent(QEvent.Type.MouseMove)) is False
    bar.set_extent(0., 30., requested_z_mm=80.)
    bar.bind_plot(None)
    plot.setXRange(10., 20., padding=0.)
    qtbot.wait(10)
    assert bar.display_geometry()["track_rect"] is None
    assert "Calculated 30" in bar.label.text()
    assert bar.map_z_to_x(30.) is None


def test_bar_paints_a_real_narrow_track_offscreen(shown):
    _host, _plot, bar = shown
    bar.set_extent(0., 45., 35., 80.)
    image = bar.grab().toImage()
    x = round(bar.map_z_to_x(20.))
    y = round(bar.TRACK_TOP+bar.TRACK_HEIGHT/2)
    assert image.pixelColor(x, y).name() == bar.COLORS["completed"]
    assert image.pixelColor(round(bar.map_z_to_x(60.)), y).name() == bar.COLORS["uncomputed"]


def test_coincident_requested_and_completed_markers_both_remain_visible(shown):
    _host, _plot, bar = shown
    bar.set_extent(0., 50., 50., 50.)
    image = bar.grab().toImage()
    x = round(bar.map_z_to_x(50.))
    assert image.pixelColor(x, 4).name() == bar.COLORS["requested"]
    assert image.pixelColor(x, 18).name() == bar.COLORS["completed_marker"]


def test_compact_text_prioritises_cutoff_and_keeps_distinct_precise_planes(shown, qtbot):
    host, plot, bar = shown
    plot.setXRange(0., 1500., padding=0.)
    host.resize(540, 350)
    qtbot.wait(15)
    bar.set_extent(0., 1000.123456, 900., 1000.123457,
                   quality="High accuracy", stale=True)
    text = bar.label.text()
    assert "Calculated 1000.123456" in text
    assert "Cutoff 1000.123457" in text
    assert text.index("Calculated") < text.index("Cutoff") < text.index("Resume")
    assert "High accuracy" not in text
    assert host.width() == 540
    assert bar.label.minimumWidth() == 0
    assert "does not mean a file has been saved" in bar.toolTip()
    assert "purple: cutoff" in bar.toolTip()
