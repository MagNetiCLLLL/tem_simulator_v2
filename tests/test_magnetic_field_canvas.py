"""Software projection checks; synthetic line data do not qualify field physics."""

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication

from temsim.gui.magnetic_field_canvas import MagneticFieldCanvas
from temsim.gui.test_electron_types import ElectronPath


def _path(positions, *, time_s=None, key="one", label="Electron 1", colour="#ff983e", selected=False):
    return ElectronPath(key, label, colour, positions, time_s, selected)


def _scene():
    z = np.linspace(0.0, 1.0, 70)
    paths = []
    for azimuth in np.linspace(0.0, 2.0 * np.pi, 18, endpoint=False):
        radius = 0.004 + 0.014 * np.sin(np.pi * z) ** 2
        points = np.stack((radius * np.cos(azimuth), radius * np.sin(azimuth), z), axis=1)
        paths.append(np.stack((points[:-1], points[1:]), axis=1))
    segments = np.concatenate(paths)
    strength = 0.15 + 0.85 * np.cos(np.pi * segments[:, 0, 2]) ** 2
    return segments, strength


@pytest.fixture
def canvas(qtbot):
    widget = MagneticFieldCanvas()
    qtbot.addWidget(widget)
    widget.resize(850, 500)
    segments, strength = _scene()
    widget.set_geometry(segments, strength, reference_t=1.0,
                        bounds_m=np.array([[-0.02, -0.02, 0.0], [0.02, 0.02, 1.0]]),
                        direction_segments_m=segments[::19],
                        labels=(("Column start", np.array([0, 0, 0])), ("Column end", np.array([0, 0, 1]))))
    widget.set_transverse_gain(6.0)
    widget.show()
    return widget


def test_detached_physical_input_and_gain_only_changes_projection(canvas):
    segments, strength = _scene()
    canvas.set_geometry(segments, strength, reference_t=1.0,
                        bounds_m=np.array([[-0.02, -0.02, 0.0], [0.02, 0.02, 1.0]]))
    original = canvas._segments_m.copy()
    projection = canvas._project(original)
    segments[:] = 100.0
    strength[:] = 100.0
    canvas.set_transverse_gain(20.0)
    np.testing.assert_array_equal(canvas._segments_m, original)
    assert not np.allclose(canvas._project(original), projection)
    assert np.max(canvas._strengths_t) <= 1.0
    assert not canvas._segments_m.flags.writeable


@pytest.mark.parametrize("angle,hidden,visible", [(0.0, 1, 0), (90.0, 0, 1), (180.0, 1, 0), (270.0, 0, 1)])
def test_axes_match_ray_diagram_projection(canvas, angle, hidden, visible):
    canvas.set_projection_angle(angle)
    np.testing.assert_allclose(canvas._basis @ canvas._basis.T, np.eye(3), atol=1e-14)
    origin = np.array([0.0, 0.0, 0.4])
    offset = origin.copy()
    offset[hidden] += 0.01
    np.testing.assert_allclose(canvas._project(origin), canvas._project(offset), atol=1e-14)
    offset = origin.copy()
    offset[visible] += 0.01
    assert not np.allclose(canvas._project(origin), canvas._project(offset))


@pytest.mark.parametrize("angle", [0.0, 35.0, 90.0, 180.0, 270.0, 298.3])
def test_screen_coordinates_and_polarity_match_ray_diagram(canvas, angle):
    canvas.set_projection_angle(angle)
    positions = np.array([[0.002, -0.001, 0.2], [-0.003, 0.004, 0.9]])
    theta = np.deg2rad(angle)
    relative = positions - canvas._centre_m
    expected_z = relative[:, 2] / canvas._display_scale_m
    expected_u = ((relative[:, 0] * np.cos(theta) + relative[:, 1] * np.sin(theta))
                  * canvas.transverse_gain / canvas._display_scale_m)
    np.testing.assert_allclose(canvas._project(positions), np.column_stack((expected_z, -expected_u)), atol=1e-14)
    # Moving exclusively along V (the depth direction) cannot move a line in U/Z.
    depth_offset = np.array([-np.sin(theta), np.cos(theta), 0.0]) * 0.01
    np.testing.assert_allclose(canvas._project(positions + depth_offset), canvas._project(positions), atol=1e-14)


def test_angle_change_only_reprojects_cached_geometry(canvas):
    arrays = (canvas._segments_m, canvas._strengths_t, canvas._direction_segments_m)
    before = tuple(array.copy() for array in arrays)
    revision = canvas._projection_revision
    canvas._pan = QPointF(18.0, -12.0)
    canvas._zoom = 2.0
    canvas.set_projection_angle(398.0)
    assert canvas.projection_angle_deg == 38.0
    assert canvas._projection_revision == revision + 1
    for array, original in zip(arrays, before):
        np.testing.assert_array_equal(array, original)
    assert canvas._segments_m is arrays[0]
    assert canvas._strengths_t is arrays[1]
    assert canvas._direction_segments_m is arrays[2]
    assert canvas._pan == QPointF(18.0, -12.0)
    assert canvas._zoom == 2.0
    canvas.set_projection_angle(38.0)
    assert canvas._projection_revision == revision + 1
    with pytest.raises(ValueError, match="finite"):
        canvas.set_projection_angle(float("nan"))
    assert canvas.projection_angle_deg == 38.0


def test_synced_axial_range_maps_mm_to_plot_edges_and_reuses_physical_scene(canvas):
    physical = canvas._segments_m
    revision = canvas._projection_revision
    canvas.set_axial_range_mm(200.0, 650.0)
    assert canvas._axial_range_m == (0.2, 0.65)
    assert canvas._projection_revision == revision + 1
    assert canvas._segments_m is physical
    projected = canvas._project(np.array([[0.0, 0.0, 0.2], [0.0, 0.0, 0.65]]))
    transform = canvas._screen_transform()
    left, right = [transform.map(QPointF(*point)).x() for point in projected]
    assert left == pytest.approx(canvas._plot_rect().left())
    assert right == pytest.approx(canvas._plot_rect().right())
    canvas.set_axial_range_mm(200.0, 650.0)
    assert canvas._projection_revision == revision + 1
    with pytest.raises(ValueError, match="increasing"):
        canvas.set_axial_range_mm(10.0, 10.0)
    assert canvas._axial_range_m == (0.2, 0.65)


def test_vertical_fit_uses_only_clipped_visible_lines(canvas):
    # The distant, wide segment must not compress the visible small deflector.
    segments = np.array([[[0.001, 0.0, 0.2], [0.002, 0.0, 0.4]],
                         [[-0.5, 0.0, 0.8], [0.5, 0.0, 0.9]],
                         [[-0.01, 0.0, 0.0], [0.01, 0.0, 1.0]]])
    canvas.set_geometry(segments, [1.0, 1.0, 1.0], reference_t=1.0,
                        bounds_m=[[-0.5, -0.01, 0.0], [0.5, 0.01, 1.0]])
    canvas.set_axial_range_mm(200.0, 400.0)
    # Third segment intersects the window and clips at X=-0.006 and -0.002 m.
    expected_span = 0.008 * canvas.transverse_gain / canvas._display_scale_m
    assert canvas._projected_extent[1] == pytest.approx(expected_span)
    visible_endpoints = canvas._project(np.array([[-0.006, 0.0, 0.2], [0.002, 0.0, 0.4]]))
    transform = canvas._screen_transform()
    y = [transform.map(QPointF(*point)).y() for point in visible_endpoints]
    assert abs(y[1] - y[0]) == pytest.approx(canvas._plot_rect().height() * 0.94)
    # Physical tick labels and the axial title have a fixed, compact margin.
    assert canvas._plot_rect().height() / canvas.height() > 0.85


def test_default_auto_fit_fills_height_and_screen_slope_has_declared_enlargement(canvas):
    assert canvas._axial_range_m is None
    canvas.set_projection_angle(38.0)
    projected = canvas._project(canvas._segments_m)
    transform = canvas._screen_transform()
    assert np.ptp(projected[:, :, 1]) * transform.m22() == pytest.approx(canvas._plot_rect().height() * 0.94)
    # A display enlargement changes a visible slope, not its physical tangent.
    physical = np.array([[0.001, 0.002, 0.4], [0.002, -0.001, 0.6]])
    screen = [transform.map(QPointF(*point)) for point in canvas._project(physical)]
    screen_slope = (screen[1].y() - screen[0].y()) / (screen[1].x() - screen[0].x())
    delta = physical[1] - physical[0]
    theta = np.deg2rad(38.0)
    physical_slope = (delta[0] * np.cos(theta) + delta[1] * np.sin(theta)) / delta[2]
    actual_enlargement = canvas.transverse_gain * transform.m22() / transform.m11()
    assert screen_slope == pytest.approx(-physical_slope * actual_enlargement)


def _mouse(widget, event_type, position, button, buttons):
    event = QMouseEvent(event_type, QPointF(*position), QPointF(*position), button, buttons, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(widget, event)


def test_all_drag_buttons_pan_and_zoom_resize_repaint_reuse_projected_paths(canvas, qtbot):
    from PySide6.QtCore import QEvent

    revision = canvas._projection_revision
    paths = tuple(path for _, path in canvas._colour_paths)
    _mouse(canvas, QEvent.Type.MouseButtonPress, (250, 250), Qt.MouseButton.RightButton, Qt.MouseButton.RightButton)
    _mouse(canvas, QEvent.Type.MouseMove, (280, 270), Qt.MouseButton.NoButton, Qt.MouseButton.RightButton)
    _mouse(canvas, QEvent.Type.MouseButtonRelease, (280, 270), Qt.MouseButton.RightButton, Qt.MouseButton.NoButton)
    np.testing.assert_allclose([canvas._pan.x(), canvas._pan.y()], [30.0, 20.0])
    event = QWheelEvent(QPointF(300, 250), QPointF(300, 250), QPoint(), QPoint(0, 120),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(canvas, event)
    assert canvas._zoom > 1.0
    canvas.resize(900, 530)
    canvas.repaint()
    QApplication.processEvents()
    assert canvas._projection_revision == revision
    assert all(current is cached for (_, current), cached in zip(canvas._colour_paths, paths))
    _mouse(canvas, QEvent.Type.MouseButtonPress, (200, 200), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton)
    previous_pan = QPointF(canvas._pan)
    _mouse(canvas, QEvent.Type.MouseMove, (260, 225), Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)
    _mouse(canvas, QEvent.Type.MouseButtonRelease, (260, 225), Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton)
    assert canvas._pan == previous_pan + QPointF(60.0, 25.0)
    assert canvas._projection_revision == revision
    assert canvas.projection_angle_deg == 0.0
    assert all(current is cached for (_, current), cached in zip(canvas._colour_paths, paths))
    canvas.fit_view()
    assert canvas._zoom == 1.0
    assert canvas._pan == QPointF()


@pytest.mark.parametrize("field,value", [
    ("segments_m", np.array([[[np.nan, 0, 0], [0, 0, 1]]])),
    ("strengths_t", [-1.0]),
    ("reference_t", float("inf")),
    ("reference_t", 0.0),
    ("bounds_m", [[1, 0, 0], [0, 1, 1]]),
    ("direction_segments_m", [[1, 2, 3]]),
    ("labels", (("bad", [0, 0, float("nan")]),)),
])
def test_invalid_geometry_is_atomic(canvas, field, value):
    before = canvas._segments_m
    values = dict(segments_m=np.array([[[0, 0, 0], [0, 0, 1]]]), strengths_t=[1.0],
                  reference_t=1.0, bounds_m=[[0, 0, 0], [1, 1, 1]])
    values[field] = value
    with pytest.raises(ValueError):
        canvas.set_geometry(**values)
    assert canvas._segments_m is before


def test_empty_and_zero_field_scenes_render_without_opengl(canvas):
    canvas.set_geometry([], [], reference_t=0.0, bounds_m=np.zeros((2, 3)))
    assert canvas.segment_count == 0
    assert not canvas.grab().isNull()
    canvas.clear()
    assert not canvas._field_projected_arrows.size
    assert not canvas._colour_paths


def test_strength_paths_are_bounded_and_rendered(canvas, tmp_path):
    assert 1 < len(canvas._colour_paths) <= 24
    assert canvas._field_projected_arrows.size
    pixmap = canvas.grab()
    assert not pixmap.isNull()
    assert pixmap.save(str(tmp_path / "magnetic-field-canvas.png"))
    # Geometry is actually visible: blue/yellow line pixels differ from background.
    image = pixmap.toImage()
    colours = {image.pixelColor(x, y).name() for x in range(100, 750, 5) for y in range(90, 390, 5)}
    assert len(colours) > 30


def test_crowded_labels_do_not_overlap_and_omit_when_no_space(canvas):
    from types import SimpleNamespace

    canvas._projected_labels = tuple((f"Lens {i}", np.zeros(2)) for i in range(30))
    metrics = SimpleNamespace(height=lambda: 16, ascent=lambda: 12,
                              horizontalAdvance=lambda text: 90)
    placements = canvas._layout_labels(canvas._screen_transform(), metrics)
    assert 1 < len(placements) < 30
    assert all(canvas._plot_rect().contains(rectangle) for _, _, rectangle, _ in placements)
    for i, (_, _, rectangle, _) in enumerate(placements):
        assert all(not rectangle.intersects(other) for _, _, other, _ in placements[i + 1:])


def test_direction_arrow_display_is_bounded_without_changing_physical_geometry(canvas):
    segments, strength = _scene()
    canvas.set_geometry(segments, strength, reference_t=1.0,
                        bounds_m=np.array([[-0.02, -0.02, 0.0], [0.02, 0.02, 1.0]]),
                        direction_segments_m=segments)
    np.testing.assert_array_equal(canvas._direction_segments_m, segments)
    assert len(canvas._direction_draw_indices) == 160
    assert canvas._direction_draw_indices[0] == 0
    assert canvas._direction_draw_indices[-1] == len(segments) - 1


def test_electron_overlay_is_detached_and_keeps_turning_path_in_time_order(canvas):
    # Z reverses twice. Reordering by Z would create a fictitious connection.
    positions = np.array([[0.0, 0.0, 0.3], [0.001, 0.001, 0.7],
                          [0.002, 0.0, 0.5], [0.001, -0.001, 0.2]])
    times = np.array([0.0, 1.0, 2.0, 3.0]) * 1e-9
    original_positions, original_times = positions.copy(), times.copy()
    canvas.set_electron_paths([_path(positions, time_s=times)])
    positions[:] = 0.0
    times[:] = 0.0
    np.testing.assert_array_equal(canvas._electron_paths[0].positions_m, original_positions)
    np.testing.assert_array_equal(canvas._electron_paths[0].time_s, original_times)
    expected_segments = np.stack((original_positions[:-1], original_positions[1:]), axis=1)
    np.testing.assert_allclose(canvas._electron_projections["one"].segments, canvas._project(expected_segments))
    assert np.sign(np.diff(canvas._electron_paths[0].positions_m[:, 2])).tolist() == [1.0, -1.0, -1.0]
    assert not canvas._electron_paths[0].positions_m.flags.writeable
    assert not canvas._electron_paths[0].time_s.flags.writeable
    assert canvas.electron_point_count == 4


def test_electron_parameter_updates_and_visibility_reuse_field_projection(canvas, monkeypatch):
    cached = canvas._colour_paths
    physical = canvas._segments_m
    revision = canvas._projection_revision

    def forbidden_rebuild():
        raise AssertionError("An electron update must not rebuild the field batches")

    monkeypatch.setattr(canvas, "_rebuild_projection", forbidden_rebuild)
    canvas.set_electron_mode(True)
    for x in (1e-7, 2e-7, -1e-7):
        canvas.set_electron_paths([_path([[0, 0, 0], [x, 0, 1]])])
        canvas.set_field_lines_visible(False)
        canvas.set_field_lines_visible(True)
    canvas.fit_view()
    canvas.repaint()
    assert canvas._colour_paths is cached
    assert canvas._segments_m is physical
    assert canvas._projection_revision == revision


def test_fit_includes_visible_field_at_the_same_physical_scale_as_electrons(canvas):
    positions = np.array([[0.0, 0.0, 0.0], [2e-7, 0.0, 0.5], [-1e-7, 0.0, 1.0]])
    canvas.set_electron_paths([_path(positions)])
    default_span = canvas._projected_extent[1]
    canvas.set_electron_mode(True)
    assert canvas._projected_extent[1] == default_span
    field = canvas._field_projected_segments.reshape(-1, 2)
    transform = canvas._screen_transform()
    assert all(canvas._plot_rect().contains(transform.map(QPointF(*point))) for point in field)
    canvas.set_field_lines_visible(False)
    assert canvas._projected_extent[1] < default_span * 1e-4
    projected = canvas._project(positions)
    transform = canvas._screen_transform()
    assert np.ptp(projected[:, 1]) * transform.m22() == pytest.approx(canvas._plot_rect().height() * 0.94)
    canvas.set_field_lines_visible(True)
    assert canvas._projected_extent[1] == default_span
    canvas.set_field_lines_visible(False)
    assert canvas._projected_extent[1] < default_span * 1e-4


def test_electron_projection_follows_angle_and_clips_without_inventing_end_markers(canvas):
    positions = np.array([[0.0, 1e-6, 0.0], [1e-6, 2e-6, 1.0], [2e-6, 3e-6, 0.0]])
    canvas.set_electron_paths([_path(positions)])
    canvas.set_projection_angle(90.0)
    canvas.set_axial_range_mm(250.0, 750.0)
    expected = np.array([[[0.25e-6, 1.25e-6, 0.25], [0.75e-6, 1.75e-6, 0.75]],
                         [[1.25e-6, 2.25e-6, 0.75], [1.75e-6, 2.75e-6, 0.25]]])
    projected = canvas._electron_projections["one"]
    np.testing.assert_allclose(projected.segments, canvas._project(expected))
    # Neither the true start nor the true end is in the chosen slab.
    assert projected.markers == ()
    assert projected.arrows[0, 1, 0] > projected.arrows[0, 0, 0]
    assert projected.arrows[1, 1, 0] < projected.arrows[1, 0, 0]
    np.testing.assert_array_equal(canvas._electron_paths[0].positions_m, positions)


@pytest.mark.parametrize("positions,times", [
    ([[0, 0]], None),
    ([[0, 0, np.nan]], None),
    ([[0, 0, 0], [0, 0, 1]], [0]),
    ([[0, 0, 0], [0, 0, 1]], [1, 0]),
    ([[0, 0, 0]], [-1]),
])
def test_bad_electron_overlay_is_atomic(canvas, positions, times):
    canvas.set_electron_paths([_path([[0, 0, 0], [1e-6, 0, 1]])])
    previous = canvas._electron_paths[0].positions_m
    with pytest.raises(ValueError):
        canvas.set_electron_paths([_path(positions, time_s=times)])
    assert canvas._electron_paths[0].positions_m is previous


def test_field_geometry_updates_preserve_overlay_and_clear_invalidates_both(canvas):
    canvas.set_electron_paths([_path([[0, 0, 0], [1e-6, 0, 1]])])
    canvas.set_electron_mode(True)
    original = canvas._electron_paths[0].positions_m
    segments, strengths = _scene()
    canvas.set_geometry(segments, strengths, reference_t=1.0,
                        bounds_m=[[-0.02, -0.02, 0], [0.02, 0.02, 1]])
    assert canvas._electron_paths[0].positions_m is original
    assert not canvas._electron_projections["one"].path.isEmpty()
    canvas.clear_field_lines()
    assert canvas._electron_paths[0].positions_m is original
    assert not canvas._electron_projections["one"].path.isEmpty()
    assert canvas.segment_count == 0
    canvas.set_geometry(segments, strengths, reference_t=1.0,
                        bounds_m=[[-0.02, -0.02, 0], [0.02, 0.02, 1]])
    canvas.clear_electron_paths()
    assert canvas.electron_point_count == 0
    assert canvas.segment_count > 0
    canvas.set_electron_paths([_path([[0, 0, 0], [1e-6, 0, 1]])])
    canvas.clear()
    assert canvas.segment_count == canvas.electron_point_count == 0
    assert not canvas._electron_projections
    assert canvas.electron_path_count == 0


@pytest.mark.parametrize("end_z", [0.0, 1.0])
def test_zero_field_straight_electron_is_visible_with_finite_fit(canvas, end_z):
    canvas.set_geometry([], [], reference_t=0.0, bounds_m=[[0, 0, 0], [0, 0, end_z]])
    canvas.set_electron_mode(True)
    canvas.set_electron_paths([_path([[2e-6, 0, 0], [2e-6, 0, 1]])])
    transform = canvas._screen_transform()
    assert np.isfinite([transform.m11(), transform.m22()]).all()
    middle = transform.map(QPointF(*canvas._project(np.array([2e-6, 0, 0.5]))))
    assert middle.y() == pytest.approx(canvas._plot_rect().center().y())
    image = canvas.grab().toImage()
    # An orange diagnostic path is visible even when no B field lines exist.
    sample = transform.map(QPointF(*canvas._project(np.array([2e-6, 0, 0.25]))))
    colour = image.pixelColor(int(sample.x()), int(sample.y()))
    # A one-pixel antialiased stroke may straddle two screen rows.
    assert colour.red() > 100
    assert colour.green() > 60
    assert colour.blue() < 100


def test_multiple_electrons_have_independent_paths_without_cross_electron_connection(canvas):
    first = _path([[0., 0., .1], [1e-6, 0., .4], [2e-6, 0., .2]], key="a", label="A", colour="#ff5533")
    second = _path([[.003, 0., .6], [.004, 0., .9]], key="b", label="B", colour="#44ccff")
    canvas.set_electron_mode(True)
    canvas.set_electron_paths((first, second))
    assert canvas.electron_path_count == 2
    assert canvas.electron_point_count == 5
    assert set(canvas._electron_projections) == {"a", "b"}
    a, b = canvas._electron_projections["a"], canvas._electron_projections["b"]
    assert len(a.segments) == 2
    assert len(b.segments) == 1
    np.testing.assert_allclose(a.segments, canvas._project(np.stack((first.positions_m[:-1], first.positions_m[1:]), axis=1)))
    np.testing.assert_allclose(b.segments, canvas._project(np.stack((second.positions_m[:-1], second.positions_m[1:]), axis=1)))
    # Two distinct painter paths cannot draw the large, fictitious A-end → B-start line.
    assert a.path is not b.path
    assert not a.path.boundingRect().intersects(b.path.boundingRect())
    assert [name for name, _ in a.markers] == ["start", "end"]
    assert [name for name, _ in b.markers] == ["start", "end"]


def test_each_electron_retains_its_own_colour_in_rendering(canvas):
    from PySide6.QtGui import QColor
    canvas.set_field_lines_visible(False)
    canvas.set_electron_mode(True)
    paths = (_path([[-1e-6, 0, 0], [-1e-6, 0, 1]], key="red", label="Red electron", colour="#ff5533"),
             _path([[1e-6, 0, 0], [1e-6, 0, 1]], key="blue", label="Blue electron", colour="#33ccff", selected=True))
    canvas.set_electron_paths(paths)
    image = canvas.grab().toImage()
    transform = canvas._screen_transform()
    for electron in paths:
        point = electron.positions_m[0].copy()
        point[2] = .25  # Away from start, end and central arrows.
        screen = transform.map(QPointF(*canvas._project(point)))
        actual = image.pixelColor(round(screen.x()), round(screen.y()))
        expected = QColor(electron.colour)
        background = np.array(QColor("#080e1b").getRgb()[:3])
        ink = np.array(actual.getRgb()[:3])-background
        full = np.array(expected.getRgb()[:3])-background
        alpha = ink[np.argmax(full)]/max(full)
        assert alpha > .3
        np.testing.assert_allclose(ink, full*alpha, atol=2)


def test_overlay_subset_and_selection_reuse_caches_and_fit_only_visible_electrons(canvas):
    from dataclasses import replace
    first = _path([[0., 0., .1], [1e-6, 0., .9]], key="a", label="A")
    second = _path([[.01, 0., .1], [.02, 0., .9]], key="b", label="B")
    canvas.set_electron_mode(True)
    canvas.set_field_lines_visible(False)
    canvas.set_electron_paths((first, second))
    combined_span = canvas._projected_extent[1]
    cached_a = canvas._electron_projections["a"]
    cached_b = canvas._electron_projections["b"]
    field_paths = canvas._colour_paths
    canvas.set_electron_paths((replace(first, selected=True), replace(second, colour="#55ff33", label="New B")))
    assert canvas._electron_projections["a"] is cached_a
    assert canvas._electron_projections["b"] is cached_b
    assert canvas._projected_extent[1] == combined_span
    canvas.set_electron_paths((first,))
    assert canvas.electron_path_count == 1
    assert canvas.electron_point_count == 2
    assert canvas._electron_projections["a"] is cached_a
    assert canvas._projected_extent[1] < combined_span*1e-3
    assert canvas._colour_paths is field_paths
    canvas.clear_electron_paths()
    assert canvas.electron_path_count == canvas.electron_point_count == 0
    assert canvas._colour_paths is field_paths


def test_rotated_clipped_overlay_preserves_each_electrons_backward_chronology(canvas):
    paths = (_path([[0., 0., 0.], [1e-6, 2e-6, 1.]], key="forward", label="Forward"),
             _path([[2e-6, 3e-6, 1.], [3e-6, 1e-6, 0.]], key="backward", label="Backward"))
    canvas.set_electron_paths(paths)
    originals = tuple(path.positions_m.copy() for path in paths)
    canvas.set_projection_angle(90.)
    canvas.set_axial_range_mm(250., 750.)
    for electron, original in zip(paths, originals):
        np.testing.assert_array_equal(electron.positions_m, original)
        projection = canvas._electron_projections[electron.key]
        assert projection.markers == ()
        assert len(projection.segments) == 1
    first = canvas._electron_projections["forward"].segments[0]
    second = canvas._electron_projections["backward"].segments[0]
    assert first[1, 0] > first[0, 0]
    assert second[1, 0] < second[0, 0]
    forward = np.array([[.25e-6, .5e-6, .25], [.75e-6, 1.5e-6, .75]])
    np.testing.assert_allclose(first, canvas._project(forward))


def test_invalid_or_duplicate_overlay_entries_are_rejected_atomically(canvas):
    first = _path([[0, 0, 0], [1e-6, 0, 1]])
    canvas.set_electron_paths((first,))
    paths, projections = canvas._electron_paths, canvas._electron_projections
    with pytest.raises(ValueError, match="unique"):
        canvas.set_electron_paths((first, _path([[0, 0, 0]], key=first.key)))
    with pytest.raises(TypeError, match="ElectronPath"):
        canvas.set_electron_paths((first, [[0, 0, 0]]))
    assert canvas._electron_paths is paths
    assert canvas._electron_projections is projections


@pytest.mark.parametrize("change", [
    {"key": ""}, {"label": " "}, {"colour": "not-a-colour"}, {"colour": "transparent"},
    {"positions_m": [[0, np.inf, 0]]}, {"time_s": [np.nan, 1.]}, {"time_s": [1., 0.]},
    {"time_s": [0.]}, {"selected": "false"},
])
def test_electron_path_records_validate_finite_inputs_and_presentation(change):
    values = dict(key="one", label="Electron 1", colour="#ff983e", positions_m=[[0, 0, 0], [0, 0, 1]],
                  time_s=[0., 1.], selected=False)
    values.update(change)
    with pytest.raises(ValueError):
        ElectronPath(**values)


def test_electron_path_dataclass_is_frozen_and_detaches_mutable_arrays():
    from dataclasses import FrozenInstanceError
    positions = np.array([[0., 0., 0.], [0., 0., 1.]])
    times = np.array([0., 1.])
    electron = _path(positions, time_s=times)
    positions[:] = 9.
    times[:] = 9.
    np.testing.assert_array_equal(electron.positions_m, [[0., 0., 0.], [0., 0., 1.]])
    np.testing.assert_array_equal(electron.time_s, [0., 1.])
    assert not electron.positions_m.flags.writeable
    assert not electron.time_s.flags.writeable
    with pytest.raises(FrozenInstanceError):
        electron.label = "Changed"


def test_many_electron_labels_do_not_allocate_extra_plot_rows(canvas):
    paths = tuple(_path([[i*1e-6, 0., 0.], [(i+1)*1e-6, 0., 1.]], key=str(i),
                        label=f"Electron {i}: an unusually long explanatory label", selected=(i == 19))
                  for i in range(20))
    canvas.resize(420, 300)
    before = canvas._plot_rect()
    canvas.set_electron_mode(True)
    canvas.set_electron_paths(paths)
    assert canvas._plot_rect() == before
    assert canvas._plot_rect().height()/canvas.height() > .75
    assert not canvas.grab().isNull()


def _wheel(widget, point, delta=120):
    event = QWheelEvent(point, widget.mapToGlobal(point), QPoint(), QPoint(0, delta),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                        Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(widget, event)


@pytest.mark.parametrize("target,changed", [("plot", (True, True)), ("bottom", (True, False)), ("left", (False, True))])
def test_real_wheel_matches_ray_diagram_axes_and_cursor_anchor(canvas, qtbot, target, changed):
    import pyqtgraph as pg

    ray = pg.PlotWidget()
    qtbot.addWidget(ray)
    ray.resize(850, 500)
    ray.show()
    ranges = ((100., 900.), (-.004, .008))
    ray.setRange(xRange=ranges[0], yRange=ranges[1], padding=0)
    canvas.set_view_range_mm(*ranges)
    canvas.set_wheel_scale_factor(ray.getViewBox().state["wheelScaleFactor"])
    QApplication.processEvents()
    rect = canvas._plot_rect() if target == "plot" else canvas.axis_rect(target)
    point = rect.center()
    anchor = canvas.screen_to_physical_mm(point)
    cached = canvas._colour_paths
    revision = canvas._projection_revision
    emitted = []
    canvas.view_range_changed.connect(lambda x, y: emitted.append((x, y)))
    _wheel(canvas, point)
    # Deliver an actual wheel event through the same scene dispatch as Ray Diagram.
    item = ray.getViewBox() if target == "plot" else ray.getAxis(target)
    ray_point = QPointF(ray.mapFromScene(item.mapToScene(item.boundingRect().center())))
    _wheel(ray.viewport(), ray_point)
    QApplication.processEvents()
    actual = np.asarray(canvas.view_range_mm())
    expected = np.asarray(ray.viewRange())
    np.testing.assert_allclose(np.ptp(actual, axis=1), np.ptp(expected, axis=1), rtol=1e-10)
    np.testing.assert_allclose([canvas.screen_to_physical_mm(point).x(), canvas.screen_to_physical_mm(point).y()],
                               [anchor.x(), anchor.y()], atol=1e-10)
    for index, axis_changed in enumerate(changed):
        if axis_changed:
            assert np.ptp(actual[index]) < np.ptp(ranges[index])
        else:
            np.testing.assert_array_equal(actual[index], ranges[index])
    assert len(emitted) == 1
    assert canvas._colour_paths is cached
    assert canvas._projection_revision == revision


@pytest.mark.parametrize("angle", [0., 35., 90., 298.3])
def test_physical_readout_excludes_display_gain_and_retains_projection_units(canvas, angle):
    canvas.set_projection_angle(angle)
    canvas.set_view_range_mm((100., 900.), (-.006, .008))
    positions = np.array([[2e-6, -3e-6, .2], [-4e-6, 1e-6, .7]])
    original = canvas._segments_m.copy()
    theta = np.deg2rad(angle)
    for gain in (1., 6., 20.):
        canvas.set_transverse_gain(gain)
        for xyz, projected in zip(positions, canvas._project(positions)):
            screen = canvas._screen_transform().map(QPointF(*projected))
            expected = [xyz[2]*1e3, (xyz[0]*np.cos(theta)+xyz[1]*np.sin(theta))*1e3]
            readout = canvas.screen_to_physical_mm(screen)
            np.testing.assert_allclose([readout.x(), readout.y()], expected, atol=1e-12)
            assert canvas.physical_mm_to_screen(*expected) == screen
    np.testing.assert_array_equal(canvas._segments_m, original)
    ticks, title = canvas._axis_ticks("left")
    assert title == "Projected U (µm)"
    assert ticks
    np.testing.assert_allclose([float(label) for _, label in ticks], [value*1000 for value, _ in ticks])


def test_manual_physical_range_survives_resize_background_modes_and_new_geometry(canvas):
    ranges = ((200., 800.), (-.0001, .0003))
    canvas.set_view_range_mm(*ranges)
    canvas.set_electron_paths([_path([[0., 0., 0.], [2e-7, 0., 1.]])])
    for enabled in (True, False, True):
        canvas.set_electron_mode(enabled)
        canvas.set_field_lines_visible(enabled)
        canvas.resize(420, 300)
        canvas.clear_field_lines()
        canvas.set_geometry(*_scene(), reference_t=1., bounds_m=[[-.02, -.02, 0.], [.02, .02, 1.]])
        assert canvas.view_range_mm() == ranges
        assert canvas.physical_mm_to_screen(ranges[0][0], ranges[1][0]) == canvas._plot_rect().bottomLeft()
        assert not canvas.grab().isNull()
    canvas.clear()
    assert canvas.view_range_mm() == ranges


@pytest.mark.parametrize("with_electron", [False, True])
def test_fit_empty_or_straight_path_has_finite_physical_axes_and_emits(canvas, with_electron):
    canvas.set_geometry([], [], reference_t=0., bounds_m=np.zeros((2, 3)))
    canvas.set_electron_mode(True)
    if with_electron:
        canvas.set_electron_paths([_path([[2e-6, 0., 0.], [2e-6, 0., 1.]])])
    canvas.set_view_range_mm((300., 400.), (-.1, .1))
    emitted = []
    canvas.view_range_changed.connect(lambda x, y: emitted.append((x, y)))
    canvas.fit_view()
    ranges = np.asarray(canvas.view_range_mm())
    assert np.isfinite(ranges).all()
    assert (np.ptp(ranges, axis=1) > 0).all()
    np.testing.assert_allclose(emitted, [ranges])
    if with_electron:
        assert ranges[0, 0] <= 0. and ranges[0, 1] >= 1000.
        assert ranges[1, 0] < .002 < ranges[1, 1]
    assert not canvas.grab().isNull()


def test_invalid_physical_range_does_not_mutate_navigation(canvas):
    ranges = ((100., 900.), (-.01, .02))
    canvas.set_view_range_mm(*ranges)
    for invalid in ((1., 1.), (2., 1.), (0., np.inf), (np.nan, 1.), (1.,)):
        with pytest.raises(ValueError, match="finite and increasing"):
            canvas.set_view_range_mm(invalid, ranges[1])
        assert canvas.view_range_mm() == ranges


def test_fit_keeps_chosen_physical_window_until_user_fits_again(canvas):
    canvas.set_electron_mode(True)
    canvas.set_field_lines_visible(False)
    canvas.set_electron_paths([_path([[0., 0., .1], [1e-6, 0., .9]])])
    canvas.fit_view()
    fitted = canvas.view_range_mm()
    canvas.clear()
    assert canvas.view_range_mm() == fitted
    canvas.set_electron_paths([_path([[0., 0., .1], [.01, 0., .9]])])
    canvas.set_field_lines_visible(True)
    canvas.resize(420, 300)
    assert canvas.view_range_mm() == fitted
    canvas.fit_view()
    assert np.ptp(canvas.view_range_mm()[1]) > np.ptp(fitted[1]) * 1000.


def test_horizontal_pixel_edges_do_not_reproject_or_change_the_physical_window(canvas, monkeypatch):
    canvas.set_view_range_mm((20., 700.), (-.003, .005))
    before = canvas.view_range_mm()
    cache = canvas._colour_paths
    monkeypatch.setattr(canvas, "_rebuild_projection", lambda: pytest.fail("Layout must not reproject geometry"))
    canvas.set_horizontal_plot_edges((113.5, 803.25))
    assert canvas._plot_rect().left() == 113.5
    assert canvas._plot_rect().right() == 803.25
    assert canvas.physical_mm_to_screen(20., 0.).x() == pytest.approx(113.5)
    assert canvas.physical_mm_to_screen(700., 0.).x() == pytest.approx(803.25)
    assert canvas.view_range_mm() == before
    assert canvas._colour_paths is cache
    for invalid in ((1., 1.), (3., 2.), (np.nan, 4.), (1.,)):
        with pytest.raises(ValueError):
            canvas.set_horizontal_plot_edges(invalid)
    assert canvas._plot_rect().left() == 113.5
    canvas.set_horizontal_plot_edges(None)
    assert canvas._plot_rect().left() == 94.


def test_electron_strokes_are_thin_without_a_dark_halo(canvas):
    from PySide6.QtGui import QColor
    canvas.clear()
    canvas.set_electron_mode(True)
    canvas.set_field_lines_visible(False)
    canvas.set_electron_paths((
        _path([[-2e-6, 0., 0.], [-2e-6, 0., 1.]], key="one", colour="#ff5533"),
        _path([[2e-6, 0., 0.], [2e-6, 0., 1.]], key="two", colour="#33ccff", selected=True),
    ))
    canvas.set_view_range_mm((0., 1000.), (-.004, .004))
    image = canvas.grab().toImage()
    background = np.array(QColor("#080e1b").getRgb()[:3])
    for electron, expected_width in zip(canvas._electron_paths, (1., 1.4)):
        sample = canvas.physical_mm_to_screen(250., electron.positions_m[0, 0]*1e3)
        rows = np.array([image.pixelColor(int(sample.x()), y).getRgb()[:3]
                         for y in range(round(sample.y())-4, round(sample.y())+5)])
        colour = np.array(QColor(electron.colour).getRgb()[:3])
        channel = np.argmax(colour-background)
        coverage = (rows[:, channel]-background[channel])/(colour[channel]-background[channel])
        assert coverage.sum() == pytest.approx(expected_width, abs=.06)
        assert np.count_nonzero(coverage > .01) <= 3
        assert (rows >= background-1).all()  # No dark stroke around the path.


def test_magnetic_background_remains_visible_in_electron_mode(canvas):
    canvas.clear()
    canvas.set_geometry([[[.001, 0., 0.], [.001, 0., 1.]]], [1.],
                        reference_t=1., bounds_m=[[-.002, -.002, 0.], [.002, .002, 1.]])
    canvas.set_electron_paths([_path([[-1e-7, 0., 0.], [1e-7, 0., 1.]])])
    canvas.fit_view()
    sample = canvas.physical_mm_to_screen(250., 1.)
    assert canvas._plot_rect().contains(sample)
    background = np.array([8., 14., 27.])

    def ink():
        image = canvas.grab().toImage()
        return sum(np.linalg.norm(np.array(image.pixelColor(int(sample.x()), y).getRgb()[:3])-background)
                   for y in range(round(sample.y())-2, round(sample.y())+3))

    full = ink()
    canvas.set_electron_mode(True)
    faded = ink()
    assert faded/full == pytest.approx(.75, abs=.035)
    assert faded > 100.


def test_field_arrowheads_keep_small_screen_dimensions_at_extreme_transverse_gain(canvas):
    segments = np.array([[[0., 0., 0.], [0., 0., 1.]]])
    canvas.set_geometry(segments, [1.], reference_t=1.,
                        bounds_m=[[-.001, -.001, 0.], [.001, .001, 1.]],
                        direction_segments_m=segments)
    canvas.set_transverse_gain(1e6)
    canvas.set_view_range_mm((0., 1000.), (-1e-8, 1e-8))
    arrows = canvas._field_arrows_on_screen(canvas._screen_transform())
    assert arrows.boundingRect().width() == pytest.approx(5.)
    assert arrows.boundingRect().height() == pytest.approx(4.)
    assert canvas._field_arrows_on_screen(canvas._screen_transform()) is arrows
    canvas.set_view_range_mm((100., 700.), (-1., 1.))
    updated = canvas._field_arrows_on_screen(canvas._screen_transform())
    assert updated is not arrows
    assert updated.boundingRect().width() == pytest.approx(5.)
    assert updated.boundingRect().height() == pytest.approx(4.)


@pytest.mark.parametrize("axis,changed", [("bottom", 0), ("left", 1)])
def test_drag_on_axis_pans_only_its_physical_direction(canvas, axis, changed):
    from PySide6.QtCore import QEvent

    ranges = ((100., 900.), (-.004, .008))
    canvas.set_view_range_mm(*ranges)
    point = canvas.axis_rect(axis).center()
    initial = (point.x(), point.y())
    final = (point.x() + 20., point.y() + 15.)
    _mouse(canvas, QEvent.Type.MouseButtonPress, initial, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton)
    _mouse(canvas, QEvent.Type.MouseMove, final, Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)
    _mouse(canvas, QEvent.Type.MouseButtonRelease, final, Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton)
    actual = canvas.view_range_mm()
    assert actual[changed] != ranges[changed]
    assert actual[1-changed] == ranges[1-changed]
    np.testing.assert_allclose(np.ptp(actual, axis=1), np.ptp(ranges, axis=1))
