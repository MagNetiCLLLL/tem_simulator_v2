"""Independent mechanical insertion, placement and cross-file copy transactions."""

from copy import deepcopy
from dataclasses import replace
import shutil
import tomllib

import numpy as np
import pytest

from temsim.component_operations import (
    PartChangeSet, make_component, translated_component, validate_component_graph,
)
from temsim.part_model_document import PartModelDocument
from temsim.part_model_3d import part_model_from_document
from temsim.paths import INSTRUMENT_CONFIG_ROOT


@pytest.fixture
def documents(tmp_path):
    loaded = {}
    originals = {}
    for name, relative in {
        "recording": "project_and_recording_system/EnergyFilter.toml",
        "column": "column/C2.toml", "gun": "gun/FEG.toml",
    }.items():
        source = INSTRUMENT_CONFIG_ROOT / relative
        originals[source] = source.read_bytes()
        target = tmp_path / (name + ".toml")
        shutil.copyfile(source, target)
        loaded[name] = PartModelDocument(target)
    yield loaded
    assert all(path.read_bytes() == data for path, data in originals.items())


def _single(document, key):
    return part_model_from_document(document.document, key, include_children=False).meshes


def test_structural_changeset_is_mapping_compatible_but_not_falsely_empty():
    added = make_component(key="body")
    changes = PartChangeSet({}, (added,), (), b"original")
    assert changes and len(changes) == 0 and dict(changes) == {}
    assert changes != {} and changes != PartChangeSet()
    modified = replace(changes, fields={("parts", "old", "length_mm"): 2})
    assert modified.added_parts == changes.added_parts
    assert modified.expected_source_bytes == b"original"
    assert PartChangeSet({("parts", "old", "length_mm"): 2}) == {("parts", "old", "length_mm"): 2}
    assert PartChangeSet(removed_keys=("old",))


@pytest.mark.parametrize("module", ["gun", "column", "recording"])
@pytest.mark.parametrize("shape", ["tube", "box", "elliptic_cylinder"])
def test_new_component_roundtrip_in_any_historical_file_category(documents, module, shape):
    document = documents[module]
    original = document.path.read_bytes()
    baseline = deepcopy(document.document)
    key = document.add_component(make_component(key="custom_fixture", name="Independent body", shape=shape,
                                                center_z_mm=3000, length_mm=7,
                                                inner_diameter_mm=4, outer_diameter_mm=12))
    assert key == "custom_fixture" and document.path.read_bytes() == original
    row = document.part(key)
    assert row["mechanical_only"] and row["axial_vacuum_context_only"]
    assert row["order"] > max(p["order"] for p in baseline["parts"])
    assert len(_single(document, key)) == 1
    assert document.updates().expected_source_bytes == original
    document.undo()
    assert document.document == baseline and not document.dirty
    document.redo()
    document.save()
    assert not document.dirty
    assert tomllib.loads(document.path.read_text(encoding="utf-8")) == document.document
    assert document.document["parts"][:-1] == baseline["parts"]
    assert all(document.document[field] == baseline[field] for field in ("module", "geometry", "ports"))


@pytest.mark.parametrize("updates", [
    {"key": ""}, {"key": "body\nmalformed"}, {"length_mm": True}, {"length_mm": "4"},
    {"length_mm": float("nan")}, {"length_mm": float("inf")}, {"length_mm": -1},
    {"inner_diameter_mm": -1}, {"inner_diameter_mm": 21}, {"name": " "},
])
def test_make_component_rejects_invalid_user_values(updates):
    with pytest.raises(ValueError):
        make_component(**{"key": "body", **updates})


def test_duplicate_unknown_parent_and_self_parent_preserve_draft_history(documents):
    document = documents["recording"]
    baseline = deepcopy(document.document)
    for part in (make_component(key=baseline["parts"][0]["key"]),
                 make_component(key="new", parent_key="absent"),
                 make_component(key="new", parent_key="new")):
        with pytest.raises(ValueError):
            document.add_component(part)
        assert document.document == baseline and not document.can_undo
    row = make_component(key="new")
    row["length_mm"] = 123
    with pytest.raises(ValueError, match="length mismatch"):
        document.add_component(row)
    assert document.document == baseline and not document.can_undo


def test_place_subtree_translates_absolute_references_but_not_relative_profiles_or_cad():
    part = {"local_start_z_mm": 10, "local_center_z_mm": 15, "local_end_z_mm": 20, "length_mm": 10,
            "optical_reference_local_z_mm": 15, "interaction_centers_local_z_mm": [12, 18],
            "upper_yoke_start_local_z_mm": 11, "material_intervals_mm": [[11, 13], [17, 19]],
            "magnetic_radial_profile_mm": [[0, 1, 4], [10, 2, 5]],
            "model_3d": {"features": [{"center_mm": [0, 0, 1]}]}, "offset_mm": 3}
    original = deepcopy(part)
    moved = translated_component(part, 7)
    assert part == original
    assert moved["local_start_z_mm"] == 17 and moved["local_center_z_mm"] == 22
    assert moved["optical_reference_local_z_mm"] == 22
    assert moved["interaction_centers_local_z_mm"] == [19, 25]
    assert moved["material_intervals_mm"] == [[18, 20], [24, 26]]
    for field in ("length_mm", "magnetic_radial_profile_mm", "model_3d", "offset_mm"):
        assert moved[field] == original[field]


def test_place_is_one_undo_preserves_ports_and_optionally_leaves_children(documents):
    document = documents["recording"]
    document.add_component(make_component(key="holder", center_z_mm=3000))
    document.add_component(make_component(key="insert", center_z_mm=3002, parent_key="holder"))
    baseline = deepcopy(document.document)
    assert set(document.place_component("holder", 3100)) == {"holder", "insert"}
    assert document.part("insert")["local_center_z_mm"] == 3102
    assert document.document["ports"] == baseline["ports"]
    document.undo()
    assert document.document == baseline
    document.redo()
    document.place_component("holder", 3200, include_children=False)
    assert document.part("holder")["local_center_z_mm"] == 3200
    assert document.part("insert")["local_center_z_mm"] == 3102
    snapshot, history = deepcopy(document.document), document._history_index
    with pytest.raises(ValueError):
        document.place_component("holder", float("nan"))
    assert document.document == snapshot and document._history_index == history


def test_explicit_base_length_keeps_placement_envelope_consistent(documents):
    document = documents["recording"]
    document.add_component(make_component(key="box", shape="box", center_z_mm=3000))
    document.set_dimension(("parts", "box", "model_3d", "base", "length_mm"), 40)
    row = document.part("box")
    assert [row[f"local_{p}_z_mm"] for p in ("start", "center", "end")] == [2980, 3000, 3020]
    assert row["length_mm"] == 40
    document.set_dimension(("parts", "box", "length_mm"), 60)
    assert document.part("box")["model_3d"]["base"]["length_mm"] == 60
    assert np.ptp(_single(document, "box")[0].vertices[:, 2]) == pytest.approx(60)


def test_tube_resize_keeps_true_through_bore(documents):
    document = documents["recording"]
    document.add_component(make_component(key="tube", center_z_mm=3000, inner_diameter_mm=4, outer_diameter_mm=12))
    document.set_dimension(("parts", "tube", "length_mm"), 100)
    mesh = _single(document, "tube")[0]
    assert np.ptp(mesh.vertices[:, 2]) == pytest.approx(100)
    assert np.linalg.norm(mesh.vertices[:, :2], axis=1).min() == pytest.approx(2)
    assert document.part("tube")["model_3d"]["features"] == []


def test_copy_real_il_assembly_preserves_independent_shape_and_clears_field_bindings(documents):
    source, target = documents["recording"], documents["gun"]
    original = deepcopy(source.document)
    source_bytes, target_bytes = source.path.read_bytes(), target.path.read_bytes()
    source_key = "intermediate_lens"
    center = source.part(source_key)["local_center_z_mm"]
    root = target.copy_component_from(source, source_key, "copied_lens", 4000, name="Mechanical IL copy")
    assert root == "copied_lens" and target.part(root)["name"] == "Mechanical IL copy"
    changes = target.updates()
    assert len(changes.added_parts) >= 4
    delta = 4000 - center
    for row in changes.added_parts:
        source_row = source.part(row["copy_provenance"]["source_part_key"])
        assert row["mechanical_part_role"] == "custom_mechanical_copy"
        assert row["axial_vacuum_context_only"]
        assert not any(k.startswith("magnetic_circuit_") or k == "field_source_key" for k in row)
        assert row["local_center_z_mm"] == pytest.approx(source_row["local_center_z_mm"] + delta)
        original_meshes = _single(source, source_row["key"])
        copied_meshes = _single(target, row["key"])
        assert len(original_meshes) == len(copied_meshes)
        for before, after in zip(original_meshes, copied_meshes):
            assert after.vertices == pytest.approx(before.vertices + [0, 0, delta])
            assert np.array_equal(after.faces, before.faces)
            assert all(path[1] != source_row["key"] for surface in after.surfaces.values()
                       for path in surface["parameter_paths"] if path and path[0] == "parts")
    assert source.document == original and source.path.read_bytes() == source_bytes
    assert target.path.read_bytes() == target_bytes
    target.save()
    assert all(p["key"].startswith("copied_lens") for p in target.document["parts"] if p.get("mechanical_part_role") == "custom_mechanical_copy")


def test_standalone_objective_split_copy_owns_intervals_and_material_regions(documents):
    source, target = documents["column"], documents["recording"]
    key = "objective_lens_excitation_coil"
    source_meshes = _single(source, key)
    assert {mesh.region for mesh in source_meshes} == {"upper", "lower"}
    center = source.part(key)["local_center_z_mm"]
    target.copy_component_from(source, key, "split_coil", 4000)
    copied = target.part("split_coil")
    assert copied["parent_key"] == "" and len(copied["material_intervals_mm"]) == 2
    meshes = _single(target, "split_coil")
    assert [mesh.region for mesh in meshes] == [mesh.region for mesh in source_meshes]
    for before, after in zip(source_meshes, meshes):
        assert after.vertices == pytest.approx(before.vertices + [0, 0, 4000 - center])
    previous_length = copied["length_mm"]
    source_before_resize = deepcopy(source.document)
    target.set_dimension(("parts", "split_coil", "length_mm"), previous_length * 1.25)
    resized = _single(target, "split_coil")
    for before, after in zip(meshes, resized):
        expected = before.vertices.copy()
        expected[:, 2] = 4000 + (expected[:, 2] - 4000) * 1.25
        assert after.vertices == pytest.approx(expected)
    assert source.document == source_before_resize
    target.save()


def test_shared_external_dependencies_are_rejected_without_mutation(documents):
    source, target = documents["column"], documents["recording"]
    baseline = deepcopy(target.document)
    with pytest.raises(ValueError, match="external .* dependencies"):
        target.copy_component_from(source, "condenser_lens_1", "copy_c1", 4000)
    assert target.document == baseline and not target.can_undo


def test_same_document_copy_snapshots_source_and_one_undo_removes_whole_copy(documents):
    document = documents["recording"]
    document.add_component(make_component(key="source", center_z_mm=3000))
    document.add_component(make_component(key="source_child", shape="box", center_z_mm=3002, parent_key="source"))
    baseline = deepcopy(document.document)
    document.copy_component_from(document, "source", "duplicate", 4000)
    assert document.part("duplicate_child")["parent_key"] == "duplicate"
    assert document.part("duplicate_child")["local_center_z_mm"] == 4002
    document.undo()
    assert document.document == baseline
    document.redo()
    document.set_dimension(("parts", "duplicate", "mechanical_outer_diameter_mm"), 30)
    assert document.part("source")["mechanical_outer_diameter_mm"] == 20


def test_stale_source_and_callback_failure_preserve_additions(documents):
    document = documents["recording"]
    document.add_component(make_component(key="new", center_z_mm=3000))
    original = document.path.read_bytes()
    calls = []
    with pytest.raises(ValueError, match="did not save"):
        document.save(project_save=lambda *args: calls.append(args) or False)
    assert calls and isinstance(calls[0][1], PartChangeSet)
    assert document.path.read_bytes() == original and document.dirty
    document.path.write_bytes(original + b"\n# independent edit\n")
    with pytest.raises(ValueError, match="changed outside"):
        document.save()
    assert document.part("new") and document.dirty


def test_graph_rejects_dangling_parent_and_cycles():
    for rows in ([{"key": "child", "parent_key": "parent"}, {"key": "parent", "parent_key": "missing"}],
                 [{"key": "a", "parent_key": "b"}, {"key": "b", "parent_key": "a"}]):
        with pytest.raises(ValueError):
            validate_component_graph({"parts": rows})


def test_undo_after_first_save_persists_component_removal_without_touching_existing_parts(documents):
    document = documents["recording"]
    baseline = deepcopy(document.document)
    document.add_component(make_component(key="new", center_z_mm=3000))
    document.save()
    document.undo()
    changes = document.updates()
    assert changes.removed_keys == ("new",) and changes
    document.save()
    assert document.document == baseline
    assert tomllib.loads(document.path.read_text(encoding="utf-8")) == baseline


@pytest.mark.parametrize("branch_path", [False, True])
def test_copy_refuses_undefined_shape_and_curvilinear_branch_without_mutation(documents, branch_path):
    target = documents["recording"]
    row = {"key": "undefined", "name": "No axial body", "local_start_z_mm": 0,
           "local_center_z_mm": 0, "local_end_z_mm": 0, "length_mm": 0, "order": 1,
           "vacuum_inner_diameter_mm": 1}
    if branch_path:
        row.update(branch_path_only=True, mechanical_outer_diameter_mm=10,
                   mechanical_inner_diameter_mm=2, local_end_z_mm=2, length_mm=2)
    baseline = deepcopy(target.document)
    with pytest.raises(ValueError, match="(no defined 3D solid|branch-path)"):
        target.copy_component_from({"parts": [row]}, "undefined", "copy", 3000)
    assert target.document == baseline and not target.can_undo


def test_zero_thickness_reference_surface_is_not_copied_as_a_solid(documents):
    row = {"key": "plane", "name": "Reference plane", "local_start_z_mm": 0,
           "local_center_z_mm": 0, "local_end_z_mm": 0, "length_mm": 0, "order": 1,
           "vacuum_inner_diameter_mm": 1, "mechanical_outer_diameter_mm": 10,
           "mechanical_inner_diameter_mm": 0}
    assert part_model_from_document({"parts": [row]}, "plane").meshes
    with pytest.raises(ValueError, match="no defined 3D solid"):
        documents["recording"].copy_component_from({"parts": [row]}, "plane", "copy", 3000)


def test_copied_radial_profile_resizes_only_axial_offsets(documents):
    row = make_component(key="profiled", center_z_mm=0, length_mm=10, inner_diameter_mm=2, outer_diameter_mm=12)
    row.update(mechanical_profile="magnetic_lens_yoke",
               magnetic_radial_profile_mm=[[0, 1, 6], [4, 2, 5], [10, 1, 6]])
    target = documents["recording"]
    target.copy_component_from({"parts": [row]}, "profiled", "profile_copy", 4000)
    before = _single(target, "profile_copy")[0]
    target.set_dimension(("parts", "profile_copy", "length_mm"), 20)
    assert target.part("profile_copy")["magnetic_radial_profile_mm"] == [[0, 1, 6], [8, 2, 5], [20, 1, 6]]
    after = _single(target, "profile_copy")[0]
    expected = before.vertices.copy()
    expected[:, 2] = 4000 + (expected[:, 2] - 4000) * 2
    assert after.vertices == pytest.approx(expected)
    target.save()


def test_new_objective_parent_and_copy_of_copy_do_not_invent_split_geometry(documents):
    source, target = documents["recording"], documents["column"]
    key = "intermediate_lens_excitation_coil"
    original = _single(source, key)
    assert len(original) == 1
    target.copy_component_from(source, key, "placed_coil", 1000, parent_key="objective_lens")
    target.copy_component_from(target, "placed_coil", "placed_coil_again", 1200, parent_key="objective_lens")
    for copied_key, center in (("placed_coil", 1000), ("placed_coil_again", 1200)):
        meshes = _single(target, copied_key)
        assert len(meshes) == 1 and "material_intervals_mm" not in target.part(copied_key)
        delta = center - source.part(key)["local_center_z_mm"]
        assert meshes[0].vertices == pytest.approx(original[0].vertices + [0, 0, delta])
