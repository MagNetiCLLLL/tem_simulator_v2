"""Real offscreen painting, visible-surface picking and 3D mouse gestures."""

from dataclasses import dataclass

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication

from temsim.gui.part_model_view import PartModelView


def _triangle(key="near", y=-20.0, *, region="body", color=(0.3, 0.6, 0.9, 1.0)):
    return dict(key=key, region=region, color=color,
                vertices=np.array([[-40., y, -40.], [40., y, -40.], [0., y, 40.]]),
                faces=np.array([[0, 1, 2]]))


@pytest.fixture
def view(qtbot):
    widget = PartModelView()
    qtbot.addWidget(widget)
    widget.resize(640, 520)
    widget.set_meshes([_triangle()])
    widget.set_front_view()
    widget.show()
    QApplication.processEvents()
    return widget


def _point(view, world):
    screen = view.project_points([world])[0]
    return QPoint(round(screen[0]), round(screen[1]))


def _drag(qtbot, view, button, start, end):
    qtbot.mousePress(view, button, pos=start)
    for amount in (0.25, 0.6, 1.0):
        point = start + (end - start) * amount
        event = QMouseEvent(QEvent.Type.MouseMove, QPointF(point),
                            QPointF(view.mapToGlobal(point)), Qt.MouseButton.NoButton,
                            button, Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(view, event)
        QApplication.processEvents()
    qtbot.mouseRelease(view, button, pos=end)
    QApplication.processEvents()


def test_real_offscreen_image_and_nearest_surface_pick_ignore_record_order(view, qtbot):
    for records in ([_triangle(), _triangle("far", 20)], [_triangle("far", 20), _triangle()]):
        view.set_meshes(records)
        position = _point(view, [0, -20, 0])
        assert view.pick_at(position) == ("near", "body")
        image = view.grab().toImage()
        assert image.pixelColor(position) != image.pixelColor(QPoint(2, 2))
    changes = []
    view.selection_changed.connect(lambda key, region: changes.append((key, region)))
    qtbot.mouseClick(view, Qt.MouseButton.LeftButton, pos=position)
    assert changes == [("near", "body")]
    assert view.selection == ("near", "body")


def test_intersecting_triangles_use_per_pixel_depth_not_whole_face_sort(view):
    a, b = _triangle("sloped"), _triangle("flat", 0)
    a["vertices"][:, 1] = [-30, 30, 0]
    view.set_meshes([a, b])
    assert view.pick_at(_point(view, [-20, 0, -20])) == ("sloped", "body")
    assert view.pick_at(_point(view, [20, 0, -20])) == ("flat", "body")


def test_coplanar_surfaces_have_stable_picking_without_depth_fighting(view):
    first = _triangle("first")
    first["vertices"] = np.array([[-40., 0, -40], [40., 0, -40], [40., 0, 40], [-40., 0, 40]])
    first["faces"] = np.array([[0, 1, 2], [0, 2, 3]])
    second = dict(first, key="second", faces=np.array([[0, 1, 3], [1, 2, 3]]))
    view.set_meshes([first, second])
    view.set_isometric_view()
    view.grab()
    # Identical planes with different diagonals must not produce speckled IDs.
    assert set(np.unique(view._ids)) == {-1, 0}


def test_region_picking_and_selection_highlight_only_selected_region(view, qtbot):
    a, b = _triangle("coil", region="upper"), _triangle("coil", region="lower")
    a["vertices"][:, 0] -= 50
    b["vertices"][:, 0] += 50
    view.set_meshes([a, b])
    upper, lower = _point(view, [-50, -20, 0]), _point(view, [50, -20, 0])
    before = view.grab().toImage()
    emissions = []
    view.selection_changed.connect(lambda *values: emissions.append(values))
    view.set_selection("coil", "upper")
    after = view.grab().toImage()
    assert after.pixelColor(upper) != before.pixelColor(upper)
    assert after.pixelColor(lower) == before.pixelColor(lower)
    assert emissions == []  # External tree selection cannot recurse.
    qtbot.mouseClick(view, Qt.MouseButton.LeftButton, pos=lower)
    assert emissions == [("coil", "lower")]
    qtbot.mouseClick(view, Qt.MouseButton.LeftButton, pos=QPoint(3, 40))
    assert emissions[-1] == ("", "")


def test_real_left_drag_rotates_all_three_dimensions_and_never_selects(view, qtbot):
    changes = []
    view.selection_changed.connect(lambda *args: changes.append(args))
    original = view.camera_rotation
    original_points = view.project_points([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
    _drag(qtbot, view, Qt.MouseButton.LeftButton, QPoint(300, 240), QPoint(455, 340))
    rotation = view.camera_rotation
    assert not np.allclose(rotation, original)
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert np.all(np.linalg.norm(view.project_points([[10, 0, 0], [0, 10, 0], [0, 0, 10]]) - original_points, axis=1) > 1)
    assert changes == []
    # Subsequent drags can reach the back/underside; there is no Euler-angle stop.
    for _ in range(4):
        _drag(qtbot, view, Qt.MouseButton.LeftButton, QPoint(240, 260), QPoint(410, 260))
    assert np.allclose(view.camera_rotation @ view.camera_rotation.T, np.eye(3), atol=1e-12)
    assert view.camera_rotation[2, 1] > 0.0  # Front started looking along -y.
    assert view.pick_at(_point(view, [0, -20, 0])) == ("near", "body")
    qtbot.wait(150)
    assert not view._interactive
    assert view._image.width() == view.width()


def test_right_pan_and_wheel_zoom_preserve_rotation_and_zoom_anchor(view, qtbot):
    point = np.array([10., -20., 0.])
    before = view.project_points([point])[0]
    rotation = view.camera_rotation
    _drag(qtbot, view, Qt.MouseButton.RightButton, QPoint(250, 230), QPoint(293, 267))
    after = view.project_points([point])[0]
    assert after[:2] - before[:2] == pytest.approx([43, 37])
    assert np.array_equal(rotation, view.camera_rotation)
    anchor = QPointF(after[0], after[1])
    wheel = QWheelEvent(anchor, QPointF(view.mapToGlobal(anchor.toPoint())), QPoint(), QPoint(0, 120),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(view, wheel)
    assert view.zoom_factor == pytest.approx(1.2)
    assert view.project_points([point])[0][:2] == pytest.approx(after[:2])
    assert np.array_equal(rotation, view.camera_rotation)


def test_fit_presets_resize_and_preserve_view_rebuild_the_render(view, qtbot):
    _drag(qtbot, view, Qt.MouseButton.RightButton, QPoint(250, 230), QPoint(293, 267))
    panned = view.project_points([[0, -20, 0]])
    view.set_meshes([_triangle()], preserve_view=True)
    assert view.project_points([[0, -20, 0]]) == pytest.approx(panned)
    view.fit_all()
    assert view.project_points([[0, -20, 0]])[0, :2] == pytest.approx([320, 260])
    front = view.camera_rotation
    view.set_isometric_view()
    assert not np.allclose(front, view.camera_rotation)
    view.set_front_view()
    assert np.allclose(front, view.camera_rotation)
    view.resize(800, 600)
    QApplication.processEvents()
    assert view._image.width() == 800
    assert view.pick_at(QPoint(400, 300)) == ("near", "body")


def test_section_clips_world_half_plane_and_changes_pick_without_mutating_mesh(view):
    source = [_triangle("removed", -20), _triangle("retained", 20)]
    crossing = _triangle("crossing", 0)
    crossing["vertices"][:, 1] = [-20, 20, 20]
    source.append(crossing)
    before = [record["vertices"].copy() for record in source]
    view.set_meshes(source)
    assert view.pick_at(_point(view, [0, 0, 0]))[0] == "removed"
    view.set_section_enabled(True)
    assert view.section_enabled
    assert len(view._triangles[0]) == 0
    assert len(view._triangles[2]) == 2
    assert all(np.all(triangles[..., 1] >= 0) for triangles in view._triangles)
    assert view.pick_at(_point(view, [0, 0, 0]))[0] != "removed"
    assert all(np.array_equal(record["vertices"], old) for record, old in zip(source, before))
    view.set_section_enabled(False)
    assert view.pick_at(_point(view, [0, 0, 0]))[0] == "removed"


@pytest.mark.parametrize("field,value", [
    ("vertices", [[0, 0, float("nan")]]),
    ("vertices", [[0, 0]]),
    ("faces", [[0, 1, 99]]),
    ("faces", [[0., 1., 2.]]),
    ("color", [1, float("inf"), 1]),
])
def test_invalid_replacement_is_transactional(view, field, value):
    invalid = _triangle("invalid")
    invalid[field] = value
    with pytest.raises(ValueError):
        view.set_meshes([invalid])
    assert view.pick_at(_point(view, [0, -20, 0])) == ("near", "body")


def test_record_objects_empty_scene_and_copied_input_arrays(view):
    @dataclass
    class Record:
        vertices: np.ndarray
        faces: np.ndarray
        key: str = "object"
        region: str = "upper"
        color: str = "#b87333"

    raw = _triangle()
    record = Record(raw["vertices"], raw["faces"])
    view.set_meshes([record])
    record.vertices[:] = 1_000_000
    assert view.pick_at(_point(view, [0, -20, 0])) == ("object", "upper")
    view.set_selection("object", "upper")
    view.set_meshes([])
    assert view.selection == ("", None)
    assert view.pick_at(QPoint(320, 260)) is None
    assert not view.grab().isNull()
    assert view.pick_at(QPoint(-1, 0)) is None


def test_hollow_annular_mesh_keeps_the_bore_empty_when_viewed_end_on(view):
    # Four rings: back/front inner and outer material boundaries. The axis is
    # y here so the front camera looks through the bore without a preset hack.
    count = 32
    angle = np.arange(count) * 2 * np.pi / count
    vertices = np.array([(radius * np.cos(t), axial, radius * np.sin(t))
                         for radius, axial in ((20, -10), (40, -10), (40, 10), (20, 10))
                         for t in angle])
    faces = []
    for ring in range(4):
        following = (ring + 1) % 4
        for step in range(count):
            a, b = ring * count + step, ring * count + (step + 1) % count
            c, d = following * count + step, following * count + (step + 1) % count
            faces.extend(((a, b, c), (b, d, c)))
    view.set_meshes([dict(vertices=vertices, faces=np.array(faces), key="annulus", color="#b87333")])
    assert view.pick_at(_point(view, [0, 0, 0])) is None
    assert view.pick_at(_point(view, [30, -10, 0])) == ("annulus", "body")


def test_mesh_refresh_drops_a_region_that_no_longer_exists(view):
    view.set_meshes([_triangle("coil", region="upper")])
    view.set_selection("coil", "upper")
    view.set_meshes([_triangle("coil", region="body")], preserve_view=True)
    assert view.selection == ("coil", None)


def test_axial_preset_looks_through_the_real_z_axis_bore(view, qtbot):
    from temsim.part_model_3d import revolve_section

    mesh = revolve_section([(10, 20), (10, 40), (190, 40), (190, 20)], key="coil")
    view.set_meshes([mesh])
    changes = []
    view.selection_changed.connect(lambda *selection: changes.append(selection))
    view.set_axial_view()
    assert view.camera_rotation == pytest.approx(np.eye(3))
    center = _point(view, [0, 0, 100])
    assert view.pick_at(center) is None
    qtbot.mouseClick(view, Qt.MouseButton.LeftButton, pos=center)
    assert changes == []  # The bore is empty, not a solid selectable cylinder.
    wall = _point(view, [30, 0, 190])
    assert view.pick_at(wall) == ("coil", "body")
    qtbot.mouseClick(view, Qt.MouseButton.LeftButton, pos=wall)
    assert changes == [("coil", "body")]
    assert view.project_points([[0, 0, 10], [0, 0, 190]])[:, :2] == pytest.approx(
        np.array([[view.width() / 2, view.height() / 2]] * 2))


def _semantic_square(key="panel", y=-20.0, radius=40.0, *, group="front", edges=True):
    vertices = np.array([[-radius, y, -radius], [radius, y, -radius],
                         [radius, y, radius], [-radius, y, radius]])
    paths = (("parts", key, "length_mm"),)
    return dict(key=key, region="body", vertices=vertices, faces=np.array([[0, 1, 2], [0, 2, 3]]),
                face_groups=(group, group), surfaces={group: dict(label="Planar front", parameter_paths=paths)},
                edges=[dict(id="rim", label="Outer rim", vertices=vertices[[0, 1, 2, 3, 0]],
                            surface_ids=(group,), parameter_paths=paths)] if edges else [])


def test_semantic_face_group_picks_both_triangles_and_returns_module_hit(view):
    view.set_meshes([_semantic_square(), _semantic_square("hidden", 20)])
    view.set_selection_mode("face")
    for location in ((20, -20, -20), (-20, -20, 20)):
        picked = view.pick_topology_at(_point(view, location))
        assert (picked["key"], picked["region"], picked["kind"], picked["id"]) == ("panel", "body", "face", "front")
        assert picked["parameter_paths"] == (("parts", "panel", "length_mm"),)
        assert picked["normal"] == pytest.approx([0, -1, 0])
        assert picked["point"][1] == pytest.approx(-20)
        assert picked["point"][::2] == pytest.approx(location[::2], abs=0.6)
    view.set_isometric_view()
    picked = view.pick_topology_at(_point(view, [0, -20, 0]))
    assert picked["normal"] == pytest.approx([0, -1, 0])  # Not camera coordinates.
    assert picked["point"][1] == pytest.approx(-20, abs=1e-10)


def test_face_highlight_covers_semantic_group_without_tinting_other_faces(view):
    record = _semantic_square()
    record["face_groups"] = ("first", "second")
    record["surfaces"] = {key: dict(label=key) for key in ("first", "second")}
    view.set_meshes([record])
    view.set_selection_mode("face")
    first, second = _point(view, [20, -20, -20]), _point(view, [-20, -20, 20])
    before = view.grab().toImage()
    view.set_selection("panel", "body")
    view.set_topology_selection([view.pick_topology_at(first)])
    after = view.grab().toImage()
    assert after.pixelColor(first) != before.pixelColor(first)
    assert after.pixelColor(second) == before.pixelColor(second)


@pytest.mark.parametrize("same_mesh", [False, True])
def test_semantic_edge_pick_and_highlight_cannot_reach_occluded_back_edges(view, same_mesh):
    front, back = _semantic_square(), _semantic_square("back", 20, 25, group="back")
    if same_mesh:
        front["vertices"] = np.concatenate([front["vertices"], back["vertices"]])
        front["faces"] = np.concatenate([front["faces"], back["faces"] + 4])
        front["face_groups"] += back["face_groups"]
        front["surfaces"].update(back["surfaces"])
        back["edges"][0]["id"] = "back_rim"
        front["edges"] += back["edges"]
    view.set_meshes([front] if same_mesh else [back, front])
    view.set_selection_mode("edge")
    hidden = _point(view, [25, 20, 0])
    assert view.pick_topology_at(hidden) is None
    before = view.grab().toImage()
    key, edge_id = ("panel", "back_rim") if same_mesh else ("back", "rim")
    view.set_topology_selection([dict(key=key, region="body", kind="edge", id=edge_id)])
    after = view.grab().toImage()
    assert after == before  # A programmatic hidden selection has no visible line.
    visible = _point(view, [40, -20, 0])
    hit = view.pick_topology_at(visible + QPoint(4, 0))
    assert (hit["key"], hit["id"], hit["kind"]) == ("panel", "rim", "edge")
    assert hit["point"] == pytest.approx([40, -20, 0], abs=0.3)
    assert view.pick_topology_at(visible + QPoint(8, 0)) is None
    view.set_topology_selection([hit])
    selected_image = view.grab().toImage()
    assert selected_image.pixelColor(visible) != before.pixelColor(visible)


def test_sloped_semantic_edge_uses_exact_depth_and_section_retains_original_faces(view):
    record = _semantic_square()
    record["vertices"][:, 1] = [-25, 25, 25, -25]
    record["edges"][0]["vertices"] = record["vertices"][[0, 1, 2, 3, 0]]
    view.set_meshes([record])
    view.set_selection_mode("edge")
    assert view.pick_topology_at(_point(view, [0, 0, -40]))["id"] == "rim"
    view.set_section_enabled(True)
    assert view.pick_topology_at(_point(view, [-20, -12.5, -40])) is None
    assert view.pick_topology_at(_point(view, [20, 12.5, -40]))["id"] == "rim"
    assert view.pick_topology_at(_point(view, [0, 0, 0])) is None  # No invented section seam.
    view.set_selection_mode("face")
    for point in ([20, 12.5, -20], [10, 6.25, 20]):
        picked = view.pick_topology_at(_point(view, point))
        assert picked["id"] == "front"
        assert picked["point"][1] >= 0
    assert view.pick_topology_at(_point(view, [-20, -12.5, 0])) is None


def test_topology_signal_follows_part_sync_and_ctrl_toggle_survives_mesh_replacement(view, qtbot):
    first, second = _semantic_square("left", radius=30), _semantic_square("right", radius=30)
    first["vertices"][:, 0] -= 40
    second["vertices"][:, 0] += 40
    records = [first, second]
    view.set_meshes(records)
    view.set_selection_mode("face")
    events = []

    def part_changed(key, region):
        events.append(("part", key))
        view.set_meshes(records, preserve_view=True)  # Real editor callback order.
        view.set_selection(key, region)

    view.selection_changed.connect(part_changed)
    view.topology_selection_changed.connect(lambda value: events.append(("topology", tuple(item["key"] for item in value))))
    left, right = _point(view, [-40, -20, 0]), _point(view, [40, -20, 0])
    qtbot.mouseClick(view, Qt.MouseButton.LeftButton, pos=left)
    assert events == [("part", "left"), ("topology", ("left",))]
    events.clear()
    qtbot.mouseClick(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier, pos=right)
    assert events == [("part", "right"), ("topology", ("left", "right"))]
    qtbot.mouseClick(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier, pos=right)
    assert [item["key"] for item in view.topology_selection] == ["left"]
    view.clear_topology_selection()
    assert events[-1] == ("topology", ())
    assert view.selection == ("right", "body")


def test_topology_restore_prunes_removed_ids_and_refreshes_parameter_metadata(view):
    record = _semantic_square()
    view.set_meshes([record])
    view.set_selection_mode("face")
    view.set_selection("panel", "body")
    picked = view.pick_topology_at(_point(view, [0, -20, 0]))
    view.set_topology_selection([picked])
    emissions = []
    view.topology_selection_changed.connect(emissions.append)
    record["surfaces"]["front"]["label"] = "Updated front"
    view.set_meshes([record], preserve_view=True)
    view.set_topology_selection([picked])
    assert view.topology_selection[0]["label"] == "Updated front"
    assert view.topology_selection[0]["point"] == picked["point"]
    record["face_groups"] = ("replacement", "replacement")
    record["surfaces"] = {"replacement": dict(label="Replacement")}
    view.set_meshes([record], preserve_view=True)
    assert view.topology_selection == ()
    assert emissions[-1] == ()
    view.set_topology_selection([picked])
    assert view.topology_selection == ()


@pytest.mark.parametrize("mode", ["face", "edge"])
def test_restored_hit_tracks_the_new_surface_or_edge_after_geometry_edit(view, mode):
    record = _semantic_square()
    view.set_meshes([record])
    view.set_selection_mode(mode)
    view.set_selection("panel", "body")
    position = [0, -20, 0] if mode == "face" else [40, -20, 0]
    picked = view.pick_topology_at(_point(view, position))
    view.set_topology_selection([picked])
    record["vertices"] += [10, 5, 0]
    record["edges"][0]["vertices"] += [10, 5, 0]
    view.set_meshes([record], preserve_view=True)
    view.set_topology_selection([picked])  # Editor restores its captured IDs.
    restored, = view.topology_selection
    assert restored["id"] == picked["id"]
    assert restored["point"][1] == pytest.approx(-15)
    if mode == "edge":
        assert restored["point"][0] == pytest.approx(50)
    else:
        assert restored["normal"] == pytest.approx([0, -1, 0])
        # A subsequent tilted edit must update the model-space normal too.
        rotation = np.array([[1, 0, 0], [0, 0.8, -0.6], [0, 0.6, 0.8]])
        record["vertices"] = record["vertices"] @ rotation.T
        view.set_meshes([record], preserve_view=True)
        tilted, = view.topology_selection
        assert tilted["normal"] == pytest.approx([0, -0.8, -0.6])
        assert np.dot(np.array(tilted["point"]) - record["vertices"][0], tilted["normal"]) == pytest.approx(0, abs=1e-10)


def test_topology_selection_survives_rotation_pan_zoom_and_parent_group_fit(view, qtbot):
    first, second = _semantic_square("coil"), _semantic_square("yoke", 20, 50)
    view.set_meshes([first, second])
    view.set_selection_mode("face")
    point = _point(view, [0, -20, 0])
    qtbot.mouseClick(view, Qt.MouseButton.LeftButton, pos=point)
    picked = view.topology_selection
    changes = []
    view.topology_selection_changed.connect(changes.append)
    _drag(qtbot, view, Qt.MouseButton.LeftButton, QPoint(300, 240), QPoint(455, 340))
    _drag(qtbot, view, Qt.MouseButton.RightButton, QPoint(250, 230), QPoint(293, 267))
    wheel = QWheelEvent(QPointF(300, 240), QPointF(view.mapToGlobal(QPoint(300, 240))), QPoint(), QPoint(0, 120),
                       Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(view, wheel)
    assert view.topology_selection == picked
    assert changes == []
    view.set_selection("lens", related_keys=("coil", "yoke"))
    rotation = view.camera_rotation
    assert view.fit_selection()
    assert view.camera_rotation == pytest.approx(rotation)
    assert view._center == pytest.approx([0, 0, 0])


def test_legacy_meshes_do_not_invent_topology_and_invalid_metadata_is_transactional(view):
    assert view.selection_mode == "part"
    view.set_selection_mode("face")
    assert view.pick_at(_point(view, [0, -20, 0])) == ("near", "body")
    assert view.pick_topology_at(_point(view, [0, -20, 0])) is None
    invalid = _semantic_square()
    invalid["face_groups"] = ("front",)
    with pytest.raises(ValueError, match="face_groups"):
        view.set_meshes([invalid])
    assert view.pick_at(_point(view, [0, -20, 0])) == ("near", "body")
    with pytest.raises(ValueError, match="Selection mode"):
        view.set_selection_mode("vertex")


def test_actual_coil_mesh_cap_wall_and_rim_keep_real_dimension_paths(view):
    from pathlib import Path
    import tomllib
    from temsim.part_model_3d import part_model_from_document

    path = Path(__file__).resolve().parents[1] / "configs/instruments/project_and_recording_system/EnergyFilter.toml"
    document = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    key = "intermediate_lens_excitation_coil"
    part = next(item for item in document["parts"] if item["key"] == key)
    mesh, = part_model_from_document(document, key).meshes
    inner, outer = (float(part[field]) * 0.5 for field in
                    ("mechanical_inner_diameter_mm", "mechanical_outer_diameter_mm"))
    end, start = part["local_end_z_mm"], part["local_start_z_mm"]
    view.set_meshes([mesh])
    view.set_selection_mode("face")
    view.set_axial_view()
    cap = view.pick_topology_at(_point(view, [(inner + outer) * 0.5, 0, end]))
    assert cap["id"] == "axial_positive"
    assert cap["normal"] == pytest.approx([0, 0, 1])
    assert cap["point"][2] == pytest.approx(end)
    assert {path[-1] for path in cap["parameter_paths"]} >= {
        "length_mm", "mechanical_inner_diameter_mm", "mechanical_outer_diameter_mm"}
    view.set_selection_mode("edge")
    rim = view.pick_topology_at(_point(view, [outer, 0, end]))
    assert rim is not None
    assert "mechanical_outer_diameter_mm" in {path[-1] for path in rim["parameter_paths"]}
    assert rim["point"][2] == pytest.approx(end)
    view.set_front_view()
    view.set_selection_mode("face")
    wall = view.pick_topology_at(_point(view, [0, -outer, (start + end) * 0.5]))
    assert wall["id"] == "outer"
    assert wall["normal"][1] < -0.99
    view.set_section_enabled(True)
    inner_wall = view.pick_topology_at(_point(view, [0, inner, (start + end) * 0.5]))
    assert inner_wall["id"] == "inner"
    assert "mechanical_inner_diameter_mm" in {path[-1] for path in inner_wall["parameter_paths"]}


def test_boolean_hole_face_and_front_rim_are_pickable_without_selecting_back_rim(view):
    from temsim.part_model_3d import part_model_from_document
    from temsim.part_model_features import default_model_3d

    part = dict(key="body", local_center_z_mm=100, model_3d=default_model_3d({}))
    part["model_3d"]["base"] = dict(kind="box", width_mm=10, height_mm=8, length_mm=6)
    part["model_3d"]["features"] = [dict(id="hole1", kind="hole", axis="z", center_mm=[0, 0, 0],
                                         depth_mm=20, diameter_mm=2)]
    mesh, = part_model_from_document({"parts": [part]}, "body").meshes
    view.set_meshes([mesh])
    view.set_axial_view()
    view.set_selection_mode("edge")
    assert view.pick_at(_point(view, [0, 0, 103])) is None
    rim = view.pick_topology_at(_point(view, [1, 0, 103]))
    assert rim is not None
    front_edges = {edge["id"] for edge in mesh.edges if
                   {"base:z_positive", "feature:hole1:wall"}.issubset(edge["surface_ids"])}
    assert rim["id"] in front_edges
    assert rim["point"][2] == pytest.approx(103)
    assert ("parts", "body", "model_3d", "features", 0, "diameter_mm") in rim["parameter_paths"]
    view.set_front_view()
    view.set_section_enabled(True)
    view.set_selection_mode("face")
    wall = view.pick_topology_at(_point(view, [0, 1, 100]))
    assert wall["id"] == "feature:hole1:wall"
    assert wall["normal"][1] < -0.99
    assert ("parts", "body", "model_3d", "features", 0, "diameter_mm") in wall["parameter_paths"]
