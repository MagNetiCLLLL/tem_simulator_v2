"""Real CSG solids: topology, removed volume and semantic face provenance."""

from collections import Counter
from copy import deepcopy
from pathlib import Path
import tomllib

import numpy as np
import pytest

from temsim.part_model_3d import TriangleMesh, part_dimension_specs, part_model_from_document
from temsim.part_model_features import (
    default_model_3d, feature_dimension_specs, feature_placement, validate_model_3d,
)


@pytest.fixture
def part():
    return {"key": "body", "name": "Test solid", "local_center_z_mm": 100.0,
            "model_3d": {"schema_version": 1,
                         "base": {"kind": "box", "width_mm": 10.0, "height_mm": 8.0, "length_mm": 6.0},
                         "transform": {"scale_xy": [1.0, 1.0], "offset_mm": [0.0, 0.0, 0.0],
                                       "rotation_deg": [0.0, 0.0, 0.0]}, "features": []}}


def _build(part, segments=32):
    return part_model_from_document({"parts": [part]}, part["key"], angular_segments=segments).meshes[0]


def _volume(mesh):
    triangles = mesh.vertices[mesh.faces] - mesh.vertices.mean(axis=0)
    return np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6


def _assert_solid(mesh):
    assert np.isfinite(mesh.vertices).all()
    triangle = mesh.vertices[mesh.faces]
    area = np.linalg.norm(np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]), axis=1)
    assert np.isfinite(area).all() and (area > 0).all()
    edges = Counter(tuple(sorted((int(a), int(b)))) for face in mesh.faces
                    for a, b in zip(face, np.roll(face, -1)))
    assert set(edges.values()) == {2}
    assert _volume(mesh) > 0
    assert len(mesh.face_groups) == len(mesh.faces)
    assert set(mesh.face_groups) == set(mesh.surfaces)
    for edge in mesh.edges:
        assert len(edge["vertices"]) >= 2 and np.isfinite(edge["vertices"]).all()
        assert set(edge["surface_ids"]) <= set(mesh.surfaces)


def _hole(**kwargs):
    return {"id": "hole1", "kind": "hole", "axis": "z", "center_mm": [0, 0, 0],
            "depth_mm": 20.0, "diameter_mm": 2.0, **kwargs}


def test_constructor_remains_backwards_compatible_and_default_is_independent(part):
    mesh = TriangleMesh(np.empty((0, 3)), np.empty((0, 3), int), "legacy")
    assert mesh.face_groups is None and not mesh.surfaces and not mesh.edges
    first, second = default_model_3d(part), default_model_3d(part)
    first["transform"]["scale_xy"][0] = 2
    assert second["transform"]["scale_xy"] == [1, 1]
    assert first["base"] == {"kind": "existing"} and not first["features"]


def test_box_has_six_semantic_faces_twelve_edges_and_real_dimensions(part):
    mesh = _build(part)
    _assert_solid(mesh)
    assert _volume(mesh) == pytest.approx(480)
    assert np.ptp(mesh.vertices, axis=0) == pytest.approx([10, 8, 6])
    assert mesh.vertices[:, 2].mean() == 100
    assert set(mesh.surfaces) == {"base:" + axis + side for axis in "xyz" for side in ("_negative", "_positive")}
    assert len(mesh.edges) == 12
    for axis, field in (("x", "width_mm"), ("y", "height_mm"), ("z", "length_mm")):
        for side in ("_negative", "_positive"):
            surface = mesh.surfaces["base:" + axis + side]
            assert ("parts", "body", "model_3d", "base", field) in surface["parameter_paths"]
            assert np.linalg.norm(surface["normal"]) == pytest.approx(1)


@pytest.mark.parametrize("axis,depth", [("x", 10), ("y", 8), ("z", 6)])
def test_through_hole_removes_volume_and_preserves_cut_wall_provenance(part, axis, depth):
    part["model_3d"]["features"] = [_hole(axis=axis)]
    mesh = _build(part)
    _assert_solid(mesh)
    polygon_area = 16 * np.sin(2 * np.pi / 32)
    assert _volume(mesh) == pytest.approx(480 - depth * polygon_area)
    group = "feature:hole1:wall"
    assert group in mesh.surfaces
    assert not {"feature:hole1:start", "feature:hole1:end"} & set(mesh.surfaces)
    assert ("parts", "body", "model_3d", "features", 0, "diameter_mm") in mesh.surfaces[group]["parameter_paths"]
    ring_edges = [edge for edge in mesh.edges if group in edge["surface_ids"]]
    assert len(ring_edges) == 2
    assert all(np.array_equal(edge["vertices"][0], edge["vertices"][-1]) for edge in ring_edges)
    # The walls bound removed space: their normals point towards the hole axis.
    triangles = mesh.vertices[mesh.faces[mesh.face_groups == group]]
    centers = triangles.mean(axis=1) - np.array([0, 0, 100])
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    centers[:, "xyz".index(axis)] = 0
    assert (np.einsum("ij,ij->i", centers, normals) < 0).all()


def test_blind_hole_has_actual_bottom_surface_and_correct_removed_depth(part):
    part["model_3d"]["features"] = [_hole(center_mm=[0, 0, 2], depth_mm=2)]
    mesh = _build(part)
    _assert_solid(mesh)
    assert _volume(mesh) == pytest.approx(480 - 2 * 16 * np.sin(2 * np.pi / 32))
    assert "feature:hole1:start" in mesh.surfaces
    assert "feature:hole1:end" not in mesh.surfaces
    bottom = mesh.vertices[mesh.faces[mesh.face_groups == "feature:hole1:start"]]
    assert bottom[:, :, 2] == pytest.approx(np.full(bottom[:, :, 2].shape, 101))


@pytest.mark.parametrize("length", [2, 5])
def test_capsule_slot_removes_real_material_with_rounded_ends(part, length):
    part["model_3d"]["features"] = [{"id": "slot1", "kind": "slot", "axis": "z",
                                     "center_mm": [0, 0, 0], "depth_mm": 20,
                                     "width_mm": 2, "length_mm": length, "rotation_deg": 30}]
    mesh = _build(part)
    _assert_solid(mesh)
    area = (length - 2) * 2 + 16 * np.sin(2 * np.pi / 32)
    assert _volume(mesh) == pytest.approx(480 - area * 6)
    assert "feature:slot1:wall" in mesh.surfaces


def test_intersecting_hole_and_slot_boolean_has_both_real_surface_sources(part):
    part["model_3d"]["features"] = [_hole(center_mm=[-1, 0, 0]),
        {"id": "slot2", "kind": "slot", "axis": "z", "center_mm": [1, 0, 0],
         "depth_mm": 20, "width_mm": 2, "length_mm": 5}]
    mesh = _build(part)
    _assert_solid(mesh)
    assert {"feature:hole1:wall", "feature:slot2:wall"} <= set(mesh.surfaces)
    assert _volume(mesh) < 480 - 6 * 16 * np.sin(2 * np.pi / 32)


@pytest.mark.parametrize("features,message", [
    ([_hole(center_mm=[30, 0, 0])], "does not remove material"),
    ([_hole(diameter_mm=30)], "entire solid"),
    ([_hole(), _hole(id="duplicate_geometry")], "does not remove material"),
])
def test_noop_empty_and_duplicate_cuts_are_explicit_failures(part, features, message):
    part["model_3d"]["features"] = features
    with pytest.raises(ValueError, match=message):
        _build(part)


def test_disabled_feature_is_preserved_without_cutting(part):
    part["model_3d"]["features"] = [_hole(center_mm=[30, 0, 0], enabled=False)]
    before = deepcopy(part)
    mesh = _build(part)
    assert _volume(mesh) == pytest.approx(480)
    assert not any(group.startswith("feature:") for group in mesh.surfaces)
    assert part == before


def test_elliptic_cylinder_is_non_axisymmetric_closed_solid(part):
    part["model_3d"]["base"]["kind"] = "elliptic_cylinder"
    mesh = _build(part)
    _assert_solid(mesh)
    assert np.ptp(mesh.vertices, axis=0) == pytest.approx([10, 8, 6])
    assert _volume(mesh) == pytest.approx(16 * 5 * 4 * np.sin(2 * np.pi / 32) * 6)
    assert "base:wall" in mesh.surfaces


def test_transform_moves_real_geometry_about_part_center_and_updates_normals(part):
    baseline = _build(part)
    part["model_3d"]["transform"] = {"scale_xy": [2, .5], "rotation_deg": [0, 0, 90],
                                      "offset_mm": [4, 7, -2]}
    mesh = _build(part)
    _assert_solid(mesh)
    assert np.ptp(mesh.vertices, axis=0) == pytest.approx([4, 20, 6])
    assert (mesh.vertices.min(axis=0) + mesh.vertices.max(axis=0)) / 2 == pytest.approx([4, 7, 98])
    assert _volume(mesh) == pytest.approx(_volume(baseline))
    assert mesh.surfaces["base:x_positive"]["normal"] == pytest.approx([0, 1, 0], abs=1e-12)
    assert ("parts", "body", "model_3d", "transform", "rotation_deg", 2) in mesh.surfaces["base:x_positive"]["parameter_paths"]


def test_hit_placement_inverts_scale_rotation_and_offset_without_moving_part(part):
    part["model_3d"]["transform"] = {"scale_xy": [2, .5], "rotation_deg": [0, 0, 90],
                                      "offset_mm": [4, 7, -2]}
    # Local point [5, 1, 2] -> world [3.5, 17, 100], +localX face -> world+Y.
    center, axis, depth = feature_placement(part, [3.5, 17, 100], [0, 1, 0])
    assert center == pytest.approx([0, 1, 2]) and axis == "x" and depth > 10
    assert part["local_center_z_mm"] == 100
    with pytest.raises(ValueError, match="normal"):
        feature_placement(part, [0, 0, 0], [0, 0, 0])


def test_semantic_ids_and_true_numeric_paths_survive_parameter_change_and_rebuild(part):
    part["model_3d"]["features"] = [_hole()]
    first = _build(part)
    part["model_3d"]["features"][0]["diameter_mm"] = 3
    second = _build(part)
    assert set(first.surfaces) == set(second.surfaces)
    assert {edge["id"] for edge in first.edges} == {edge["id"] for edge in second.edges}
    assert _volume(second) < _volume(first)
    specs = feature_dimension_specs(part)
    for spec in specs:
        current = part
        for segment in spec.path[2:]:
            current = current[segment]
        assert current == spec.value
    assert set(spec.path for spec in specs) <= set(spec.path for spec in part_dimension_specs({"parts": [part]}, "body"))


@pytest.mark.parametrize("field,value", [("diameter_mm", True), ("diameter_mm", "2"),
                                         ("diameter_mm", float("nan")), ("depth_mm", float("inf")),
                                         ("depth_mm", 0), ("diameter_mm", -1)])
def test_invalid_feature_numbers_fail_before_entering_csg(part, field, value):
    part["model_3d"]["features"] = [_hole(**{field: value})]
    with pytest.raises(ValueError):
        validate_model_3d(part)
    with pytest.raises(ValueError):
        _build(part)


@pytest.mark.parametrize("change", [
    lambda model: model.update(schema_version=True),
    lambda model: model.update(features=[_hole(), _hole()]),
    lambda model: model.update(features=[_hole(axis="arbitrary")]),
    lambda model: model.update(features=[_hole(enabled="false")]),
    lambda model: model["transform"].update(scale_xy=[0, 1]),
    lambda model: model["transform"].update(offset_mm=[0, 1]),
    lambda model: model["base"].update(kind="unknown"),
    lambda model: model.update(unknown_field=1),
])
def test_invalid_schema_rejected(part, change):
    change(part["model_3d"])
    with pytest.raises(ValueError):
        validate_model_3d(part)


def test_actual_existing_coil_retains_original_mesh_without_custom_model_and_accepts_real_hole():
    path = Path(__file__).resolve().parents[1] / "configs/instruments/project_and_recording_system/EnergyFilter.toml"
    original = path.read_bytes()
    document = tomllib.loads(original.decode("utf-8-sig"))
    key = "intermediate_lens_excitation_coil"
    part = next(part for part in document["parts"] if part["key"] == key)
    before = part_model_from_document(document, key).meshes[0]
    assert {"inner", "outer", "axial_negative", "axial_positive"} == set(before.surfaces)
    part["model_3d"] = default_model_3d(part)
    identity = part_model_from_document(document, key).meshes[0]
    assert identity.vertices == pytest.approx(before.vertices)
    assert np.array_equal(identity.faces, before.faces)
    part["model_3d"]["features"] = [_hole(center_mm=[42, 0, 0], depth_mm=1000)]
    after = part_model_from_document(document, key).meshes[0]
    _assert_solid(after)
    assert _volume(after) < _volume(before)
    assert "feature:hole1:wall" in after.surfaces
    assert path.read_bytes() == original


def test_unknown_envelope_cannot_be_mislabeled_as_real_cut_solid():
    part = {"key": "envelope", "mechanical_profile": "unknown_mechanism",
            "local_start_z_mm": -5, "local_center_z_mm": 0, "local_end_z_mm": 5,
            "mechanical_outer_diameter_mm": 10, "model_3d": default_model_3d({})}
    assert not _build(part).is_exact
    part["model_3d"]["features"] = [_hole()]
    with pytest.raises(ValueError, match="envelope"):
        _build(part)


def test_existing_annular_cap_exposes_axial_and_both_radial_dimensions():
    part = {"key": "coil", "mechanical_profile": "magnetic_excitation_coil",
            "local_start_z_mm": 0, "local_center_z_mm": 5, "local_end_z_mm": 10,
            "length_mm": 10, "mechanical_inner_diameter_mm": 8, "mechanical_outer_diameter_mm": 12}
    mesh = _build(part)
    for name in ("axial_negative", "axial_positive"):
        fields = {path[-1] for path in mesh.surfaces[name]["parameter_paths"]}
        assert {"length_mm", "mechanical_inner_diameter_mm", "mechanical_outer_diameter_mm"} <= fields


def test_objective_shoulder_planes_and_wall_segments_have_distinct_stable_groups():
    path = Path(__file__).resolve().parents[1] / "configs/instruments/column/C3_ProbeCorrector_ImageCorrector.toml"
    document = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    key = "objective_upper_pole"
    first = part_model_from_document(document, key).meshes[0]
    caps = [group for group, surface in first.surfaces.items() if surface["kind"] == "cap"]
    assert len(caps) >= 4
    for group in caps:
        triangles = first.vertices[first.faces[first.face_groups == group]]
        assert len(np.unique(triangles[:, :, 2])) == 1
    assert len([group for group, surface in first.surfaces.items() if surface["kind"] == "outer"]) >= 3
    part = next(part for part in document["parts"] if part["key"] == key)
    part["pole_vacuum_connector_axial_length_mm"] += 1
    second = part_model_from_document(document, key).meshes[0]
    assert set(first.surfaces) == set(second.surfaces)


@pytest.mark.parametrize("kind", ["yoke", "excitation_coil"])
def test_split_cap_surfaces_reference_actual_parent_endpoints_and_editable_controls(kind):
    path = Path(__file__).resolve().parents[1] / "configs/instruments/column/C3_ProbeCorrector_ImageCorrector.toml"
    document = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    key = "objective_lens_" + kind
    model = part_model_from_document(document, key)
    editable = {field.path for field in model.fields if field.editable}
    for mesh in model.meshes:
        for group, endpoint in (("axial_negative", "start"), ("axial_positive", "end")):
            paths = mesh.surfaces[group]["parameter_paths"]
            parent_endpoint = ("parts", "objective_lens", mesh.region + f"_yoke_{endpoint}_local_z_mm")
            assert parent_endpoint in paths
            assert ("parts", key, "length_mm") not in paths
            assert set(paths) <= editable
            if kind == "excitation_coil":
                assert ("parts", "objective_lens", "mechanical_coil_axial_inset_mm") in paths
