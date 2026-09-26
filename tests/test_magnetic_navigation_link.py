"""Physical display-range linkage; no field or particle calculation is run."""
import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QDockWidget, QLabel, QMainWindow, QScrollArea, QSplitter, QVBoxLayout, QWidget

from temsim.gui.diagnostic_tabs import MagneticFieldView
from temsim.gui.test_electron_types import ElectronPath


@pytest.fixture
def linked(qtbot, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Navigation must not execute a field or trajectory calculation")

    monkeypatch.setattr("temsim.magnetic_test_particle.trace_test_electron", forbidden)
    monkeypatch.setattr("temsim.test_electron_scene.prepare_test_electron_scene", forbidden)
    source = pg.PlotWidget()
    view = MagneticFieldView()
    qtbot.addWidget(source)
    qtbot.addWidget(view)
    view.display_mode.setCurrentIndex(view.display_mode.findData("electron"))
    source.setRange(xRange=(10., 50.), yRange=(-.00002, .00003), padding=0)
    view.link_axial_axis(source)
    return source, view


def test_link_copies_both_physical_axes_without_changing_field_units(linked):
    source, view = linked
    page = view.field_lines
    assert page.link_view.isChecked()
    np.testing.assert_allclose(page.view_range_mm(), source.viewRange(), rtol=0, atol=1e-14)
    view.plot.setYRange(-.15, .25, padding=0)
    source.setRange(xRange=(20., 80.), yRange=(-.003, .002), padding=0)
    np.testing.assert_allclose(page.view_range_mm(), source.viewRange(), rtol=0, atol=1e-14)
    np.testing.assert_allclose(view.plot.viewRange()[0], (20., 80.))
    np.testing.assert_allclose(view.plot.viewRange()[1], (-.15, .25))
    assert view.plot.getAxis("left").labelUnits == "T"


def test_spatial_navigation_publishes_both_axes_and_can_unlink_then_rejoin(linked):
    source, view = linked
    page = view.field_lines
    page.set_view_range_mm((15., 25.), (-.1, .2), emit=True)
    np.testing.assert_allclose(source.viewRange(), ((15., 25.), (-.1, .2)))
    page.link_view.setChecked(False)
    page.set_view_range_mm((17., 21.), (-.002, .004), emit=True)
    np.testing.assert_allclose(source.viewRange(), ((15., 25.), (-.1, .2)))
    source.setRange(xRange=(100., 200.), yRange=(-.004, .003), padding=0)
    np.testing.assert_allclose(page.view_range_mm(), ((17., 21.), (-.002, .004)))
    page.link_view.setChecked(True)
    np.testing.assert_allclose(page.view_range_mm(), source.viewRange())


def test_field_profile_axial_navigation_respects_independent_spatial_view(linked):
    source, view = linked
    page = view.field_lines
    view.plot.setXRange(5., 12., padding=0)
    np.testing.assert_allclose(source.viewRange()[0], (5., 12.))
    np.testing.assert_allclose(page.view_range_mm(), source.viewRange())
    page.link_view.setChecked(False)
    independent = page.view_range_mm()
    view.plot.setXRange(20., 30., padding=0)
    np.testing.assert_allclose(source.viewRange()[0], (20., 30.))
    np.testing.assert_allclose(page.view_range_mm(), independent)


def test_rebinding_disconnects_previous_ray_plot(linked, qtbot):
    old, view = linked
    new = pg.PlotWidget()
    qtbot.addWidget(new)
    new.setRange(xRange=(100., 300.), yRange=(-.001, .005), padding=0)
    view.link_axial_axis(new)
    old.setRange(xRange=(-100., -50.), yRange=(-1., 1.), padding=0)
    np.testing.assert_allclose(view.field_lines.view_range_mm(), new.viewRange())
    view.field_lines.set_view_range_mm((80., 100.), (-.05, .1), emit=True)
    np.testing.assert_allclose(new.viewRange(), ((80., 100.), (-.05, .1)))
    np.testing.assert_allclose(old.viewRange(), ((-100., -50.), (-1., 1.)))


def test_mode_switch_and_projection_keep_independent_physical_ranges(linked):
    _source, view = linked
    page = view.field_lines
    page.link_view.setChecked(False)
    page.set_view_range_mm((5., 25.), (-.01, .02))
    for mode in ("electron", "3d", "2d", "electron"):
        view.display_mode.setCurrentIndex(view.display_mode.findData(mode))
        view.set_projection_angle(37.)
        np.testing.assert_allclose(page.view_range_mm(), ((5., 25.), (-.01, .02)))


def test_field_strength_fit_does_not_change_ray_spatial_range(linked):
    source, view = linked
    view.display_mode.setCurrentIndex(view.display_mode.findData("2d"))
    before = source.viewRange()
    view.field_lines.fit_button.click()
    np.testing.assert_array_equal(source.viewRange(), before)
    view.display_mode.setCurrentIndex(view.display_mode.findData("electron"))
    np.testing.assert_allclose(view.field_lines.view_range_mm(), before)


def test_linked_fit_stays_linked_when_trajectory_is_replaced(linked):
    source, view = linked
    canvas = view.field_lines.canvas
    canvas.set_electron_paths((ElectronPath("one", "Electron 1", "#ffaa66",
                                            np.array([[0., 0., 0.], [.000001, 0., .01]])),))
    view.field_lines.fit_button.click()
    expected = source.viewRange()
    np.testing.assert_allclose(canvas.view_range_mm(), expected)
    canvas.clear_electron_paths()
    canvas.set_electron_paths((ElectronPath("two", "Electron 2", "#ffaa66",
                                            np.array([[0., 0., 0.], [.000004, 0., .03]])),))
    np.testing.assert_allclose(canvas.view_range_mm(), expected)
    np.testing.assert_allclose(source.viewRange(), expected)


def test_link_respects_ray_limits_and_returns_the_accepted_range(linked):
    source, view = linked
    source.getViewBox().setLimits(xMin=0., xMax=40.)
    view.field_lines.set_view_range_mm((-100., 100.), (-.03, .03), emit=True)
    np.testing.assert_allclose(view.field_lines.view_range_mm(), source.viewRange())
    np.testing.assert_allclose(source.viewRange()[0], (0., 40.))


def _global_plot_x(plot, z):
    point = plot.getViewBox().mapViewToScene(QPointF(z, 0.))
    return plot.viewport().mapToGlobal(plot.viewportTransform().map(point)).x()


def _pixel_errors(source, view):
    canvas = view.field_lines.canvas
    return [canvas.mapToGlobal(canvas.physical_mm_to_screen(z, 0.)).x()-_global_plot_x(source, z)
            for z in np.linspace(*source.viewRange()[0], 7)]


@pytest.fixture
def stacked_linked(linked, qtbot):
    source, view = linked
    window = QMainWindow()
    window.resize(1400, 1000)
    qtbot.addWidget(window)
    # A left instrument dock and right analysis sidebar reproduce the actual
    # application nesting. Different child margins expose local/global errors.
    dock = QDockWidget("Instrument setup", window)
    dock.setWidget(QLabel("Instrument parameters"))
    dock.setMinimumWidth(170)
    window.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
    row = QSplitter(Qt.Orientation.Horizontal)
    column = QSplitter(Qt.Orientation.Vertical)
    ray_container = QWidget()
    ray_layout = QVBoxLayout(ray_container)
    ray_layout.setContentsMargins(11, 0, 7, 0)
    ray_layout.addWidget(source)
    source.getAxis("left").setWidth(112.)
    column.addWidget(ray_container)
    column.addWidget(view)
    sidebar = QLabel("Beam analysis")
    sidebar.setMinimumWidth(100)
    row.addWidget(column)
    row.addWidget(sidebar)
    row.setSizes((1000, 200))
    window.setCentralWidget(row)
    window.show()
    column.setSizes((400, 420))
    qtbot.waitUntil(lambda: max(map(abs, _pixel_errors(source, view))) < .02)
    return source, view, window, dock, row, column


def test_ray_and_electron_plot_edges_align_across_resize_sidebar_and_dock(stacked_linked, qtbot, monkeypatch):
    source, view, window, dock, row, column = stacked_linked
    canvas = view.field_lines.canvas
    canvas.set_electron_paths((ElectronPath("one", "Electron 1", "#ffaa66",
                                           np.array([[0., 0., .01], [.000001, 0., .05]])),))
    cached = canvas._electron_projections["one"]
    field_cache = canvas._colour_paths
    monkeypatch.setattr(canvas, "_rebuild_projection", lambda: pytest.fail("Layout must reuse projected paths"))
    expected = source.viewRange()
    for change in (lambda: window.resize(1250, 850),
                   lambda: row.setSizes((760, 260)),
                   lambda: dock.setMinimumWidth(260),
                   lambda: dock.hide(),
                   lambda: dock.show(),
                   lambda: source.getAxis("left").setWidth(135.),
                   lambda: column.setSizes((500, 260))):
        change()
        qtbot.wait(10)
        qtbot.waitUntil(lambda: max(map(abs, _pixel_errors(source, view))) < .02)
        np.testing.assert_allclose(canvas.view_range_mm(), expected)
        assert canvas._electron_projections["one"] is cached
        assert canvas._colour_paths is field_cache
    source.setRange(xRange=(-50., 135.), yRange=(-.02, .03), padding=0)
    qtbot.waitUntil(lambda: max(map(abs, _pixel_errors(source, view))) < .02)
    view.display_mode.setCurrentIndex(view.display_mode.findData("3d"))
    qtbot.waitUntil(lambda: max(map(abs, _pixel_errors(source, view))) < .02)


def test_pixel_alignment_tracks_scroll_reparenting_and_window_moves(stacked_linked, qtbot):
    source, view, window, _dock, row, column = stacked_linked
    scroll = QScrollArea()
    scroll.setWidgetResizable(False)
    scroll.setWidget(column)
    row.insertWidget(0, scroll)
    column.resize(max(600, row.width()+180), 1150)
    scroll.show()
    qtbot.waitUntil(lambda: scroll.horizontalScrollBar().maximum() > 0)
    for action in (lambda: scroll.horizontalScrollBar().setValue(80),
                   lambda: scroll.verticalScrollBar().setValue(80),
                   lambda: window.move(window.x()+25, window.y()+20),
                   lambda: column.resize(column.width()+70, 1250)):
        action()
        qtbot.wait(10)
        qtbot.waitUntil(lambda: max(map(abs, _pixel_errors(source, view))) < .02)


def test_separate_windows_keep_standalone_margins(linked, qtbot):
    source, view = linked
    source.show()
    view.show()
    qtbot.waitUntil(lambda: not view._alignment_timer.isActive())
    assert view.field_lines.canvas._horizontal_plot_edges is None


def test_field_profile_pixel_edges_align_without_changing_tesla_axis(stacked_linked, qtbot):
    source, view, window, _dock, row, _column = stacked_linked
    view.plot.setYRange(-.15, .25, padding=0)
    view.display_mode.setCurrentIndex(view.display_mode.findData("2d"))

    def errors():
        return [_global_plot_x(view.plot, z)-_global_plot_x(source, z)
                for z in np.linspace(*source.viewRange()[0], 7)]

    for action in (lambda: window.resize(1280, 950),
                   lambda: source.getAxis("left").setWidth(130.),
                   lambda: row.setSizes((700, 240)),
                   lambda: source.getAxis("left").setWidth(95.)):
        action()
        qtbot.wait(10)
        qtbot.waitUntil(lambda: max(map(abs, errors())) < .02)
        np.testing.assert_allclose(view.plot.viewRange()[1], (-.15, .25))
    view.display_mode.setCurrentIndex(view.display_mode.findData("electron"))
    qtbot.waitUntil(lambda: max(map(abs, _pixel_errors(source, view))) < .02)
