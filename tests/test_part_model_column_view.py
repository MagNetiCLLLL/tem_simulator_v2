"""Column camera conventions; no physical model, ray solve or GL is involved."""

from copy import deepcopy

import numpy as np
import pytest

from temsim.gui.part_model_view import PartModelView, _edge_segments, _section_triangles


def _panel(key="objective", y=20.0):
    return dict(
        key=key, region="housing",
        vertices=np.array([[-30.0, y, 1580.0], [30.0, y, 1580.0],
                           [30.0, y, 1620.0], [-30.0, y, 1620.0]]),
        faces=np.array([[0, 1, 2], [0, 2, 3]]),
        face_groups=("side", "side"),
        surfaces={"side": {"label": "Housing side", "normal": [0, 1, 0]}},
    )


@pytest.fixture
def view(qtbot):
    widget = PartModelView()
    qtbot.addWidget(widget)
    widget.resize(640, 800)
    widget.set_meshes([_panel()])
    return widget


@pytest.mark.parametrize("method", ["set_column_view", "set_column_isometric_view"])
def test_column_presets_are_proper_rotations_with_downstream_axis_down(view, method):
    getattr(view, method)()
    rotation = view.camera_rotation
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), rtol=0, atol=1e-14)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-14)
    points = view.project_points(view._center + np.vstack((np.zeros(3), np.eye(3))))
    x, y, z = points[1:] - points[0]
    assert x[0] > 0
    assert z[1] > 0
    assert z[0] == pytest.approx(0.0, abs=1e-12)
    if method == "set_column_view":
        assert x[1] == pytest.approx(0)
        assert y[:2] == pytest.approx([0, 0])
        assert y[2] > 0  # +Y is nearer the camera, not a left-handed mirror.
        assert z[1] == pytest.approx(x[0])
    else:
        assert abs(y[0]) > 0
        assert abs(z[2]) > 0
        assert z[1] / view._scale() > 0.9


@pytest.mark.parametrize("method", ["set_column_view", "set_column_isometric_view"])
@pytest.mark.parametrize("size", [(640, 520), (320, 900)])
def test_column_projection_has_one_physical_scale_and_no_perspective(view, method, size):
    view.resize(*size)
    getattr(view, method)()
    rotation = view.camera_rotation
    projected = view.project_points(view._center + np.vstack((np.zeros(3), np.eye(3))))
    jacobian = (projected[1:, :2] - projected[0, :2]).T
    np.testing.assert_allclose(np.linalg.svd(jacobian, compute_uv=False),
                               [view._scale()] * 2, rtol=1e-13, atol=1e-12)
    # Equal millimetre lengths along orthogonal camera-plane axes stay equal
    # at every depth. An oblique world axis is only geometrically foreshortened.
    for depth in (-3000.0, 0.0, 3000.0):
        centre = view._center + depth * rotation[2]
        coordinates = view.project_points(np.array([
            centre, centre + 7 * rotation[0], centre + 7 * rotation[1],
        ]))
        np.testing.assert_allclose(coordinates[1:, :2] - coordinates[0, :2],
            [[7 * view._scale(), 0], [0, -7 * view._scale()]], rtol=1e-12, atol=1e-10)


@pytest.mark.parametrize("method", ["set_column_view", "set_column_isometric_view"])
def test_preset_and_mesh_refresh_preserve_selection_camera_and_source_vertices(view, method):
    record = _panel()
    before = deepcopy(record)
    view.set_meshes([record])
    view.set_selection("objective", "housing")
    view.set_selection_mode("face")
    view.set_topology_selection([dict(key="objective", region="housing", kind="face",
                                      id="side", point=(0, 20, 1600))])
    part_signals, topology_signals = [], []
    view.selection_changed.connect(lambda *args: part_signals.append(args))
    view.topology_selection_changed.connect(topology_signals.append)
    view._zoom = 2.75
    view._pan = np.array([45.0, -31.0])
    camera_extent = (view._center.copy(), view._radius, view.zoom_factor, view._pan.copy())
    selected = view.topology_selection
    getattr(view, method)()
    rotation = view.camera_rotation
    projected = view.project_points(record["vertices"])
    view.set_meshes([record], preserve_view=True)
    assert view.selection == ("objective", "housing")
    assert view.topology_selection == selected
    assert not part_signals and not topology_signals
    np.testing.assert_array_equal(view.camera_rotation, rotation)
    np.testing.assert_array_equal(view._center, camera_extent[0])
    assert (view._radius, view.zoom_factor) == camera_extent[1:3]
    np.testing.assert_array_equal(view._pan, camera_extent[3])
    np.testing.assert_array_equal(view.project_points(record["vertices"]), projected)
    np.testing.assert_array_equal(record["vertices"], before["vertices"])
    np.testing.assert_array_equal(view._meshes[0].vertices, before["vertices"])
    assert not view._meshes[0].vertices.flags.writeable


def test_column_front_depth_picking_remains_right_handed_and_section_unchanged(view):
    view.set_meshes([_panel("rear", -20), _panel("front", 20)])
    view.set_column_view()
    point = view.project_points([[0, 0, 1600]])[0, :2]
    assert view.pick_at(tuple(point)) == ("front", "housing")
    view.set_section_enabled(True)
    assert len(view._triangles[0]) == 0
    assert len(view._triangles[1]) == 2
    assert view.pick_at(tuple(point)) == ("front", "housing")


def test_existing_editor_presets_and_default_labels_are_unchanged(view):
    initial_rotation = view.camera_rotation
    assert view._view_title == "3D model"
    assert view._empty_text == "Select a part to view its 3D model"
    view.set_column_view()
    view.set_isometric_view()
    np.testing.assert_array_equal(view.camera_rotation, initial_rotation)
    view.set_front_view()
    projected = view.project_points([[0, 0, 1600], [1, 0, 1600], [0, 0, 1601]])
    assert projected[1, 0] > projected[0, 0]
    assert projected[2, 1] < projected[0, 1]  # Existing editor still uses +Z up.
    view.set_axial_view()
    np.testing.assert_array_equal(view.camera_rotation, np.eye(3))


def test_custom_labels_do_not_change_camera_or_start_any_model_work(view):
    rotation = view.camera_rotation
    meshes = view._meshes
    view.set_view_labels(title="Column model", empty_text="No assembled components")
    assert view._view_title == "Column model"
    assert view._empty_text == "No assembled components"
    assert view._meshes is meshes
    np.testing.assert_array_equal(view.camera_rotation, rotation)


def test_negative_y_section_retains_original_surface_and_edge_pick_coordinates(view):
    rear, front = _panel("rear", -20), _panel("front", 20)
    rear["edges"] = [dict(id="rim", vertices=rear["vertices"][[0, 1, 2, 3, 0]])]
    original_vertices = rear["vertices"].copy()
    view.set_meshes([front, rear])
    view.set_column_view()
    view.set_section_enabled(True, keep_positive_y=False)
    assert len(view._triangles[0]) == 0
    assert np.all(view._triangles[1][..., 1] <= 0)
    position = tuple(view.project_points([[0, -20, 1600]])[0, :2])
    assert view.pick_at(position) == ("rear", "housing")
    hit = view.pick_topology_at(position, "face")
    assert hit["point"][1] == pytest.approx(-20)
    np.testing.assert_allclose(np.asarray(hit["point"])[[0, 2]], [0, 1600], atol=0.2)
    edge_position = tuple(view.project_points([[30, -20, 1600]])[0, :2])
    edge_hit = view.pick_topology_at(edge_position, "edge")
    assert edge_hit["id"] == "rim"
    np.testing.assert_allclose(edge_hit["point"], [30, -20, 1600], atol=0.2)
    np.testing.assert_array_equal(rear["vertices"], original_vertices)
    np.testing.assert_array_equal(view._meshes[1].vertices, original_vertices)
    # Changing the retained side while still enabled must rebuild clipping.
    view.set_section_enabled(True)
    assert len(view._triangles[0]) == 2 and len(view._triangles[1]) == 0
    assert view.pick_at(position) == ("front", "housing")


@pytest.mark.parametrize("positive", [False, True])
def test_section_crossings_clip_surface_and_seam_without_reflecting_coordinates(positive):
    triangles = np.array([[[-2.0, -1, 10], [2, 1, 10], [0, 1, 14]],
                          [[-2, 0, 10], [2, 0, 10], [0, 0, 14]]])
    original = triangles.copy()
    clipped, sources = _section_triangles(triangles, with_sources=True, keep_positive_y=positive)
    assert len(clipped) == (3 if positive else 2)
    assert np.all(clipped[..., 1] >= 0) if positive else np.all(clipped[..., 1] <= 0)
    assert sources[-1] == 1
    np.testing.assert_array_equal(clipped[-1], triangles[1])
    for point in clipped[sources == 0].reshape(-1, 3):
        # Original plane y=x/2+z/4-2.5; a reflection would not lie on it.
        assert point[1] == pytest.approx(point[0] / 2 + point[2] / 4 - 2.5)
    segments = _edge_segments(triangles[0], True, keep_positive_y=positive)
    assert np.all(segments[..., 1] >= 0) if positive else np.all(segments[..., 1] <= 0)
    np.testing.assert_array_equal(triangles, original)
