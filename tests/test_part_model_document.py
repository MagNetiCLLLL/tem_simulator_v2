"""Real module-file drafts: dimensions, arrays, materials and safe persistence."""

from copy import deepcopy
from pathlib import Path
import shutil
import tomllib

import pytest

from temsim import module_manifest
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT


KEY = "intermediate_lens_excitation_coil"
MODULE = "project_and_recording_system/EnergyFilter.toml"


@pytest.fixture
def document(tmp_path):
    source = INSTRUMENT_CONFIG_ROOT / MODULE
    original = source.read_bytes()
    path = tmp_path / "model.toml"
    shutil.copyfile(source, path)
    yield PartModelDocument(path)
    assert source.read_bytes() == original


def test_dimensions_are_staged_across_parts_then_saved_with_comments_retained(document):
    original = document.path.read_bytes()
    before = deepcopy(document.document)
    center = document.part(KEY)["local_center_z_mm"]
    document.set_dimension(("parts", KEY, "length_mm"), 178.0)
    document.set_dimension(("parts", KEY, "mechanical_inner_diameter_mm"), 70.0)
    document.set_dimension(("parts", "intermediate_lens_housing", "mechanical_inner_diameter_mm"), 176.0)
    assert document.path.read_bytes() == original
    assert document.part(KEY)["local_center_z_mm"] == center
    assert document.part(KEY)["local_start_z_mm"] == center - 89.0
    assert document.dirty
    document.save()
    assert not document.dirty
    loaded = module_manifest.read_document(document.path)
    assert loaded == document.document
    for key in ("module", "geometry", "ports"):
        assert loaded[key] == before[key]
    original_comments = [line for line in original.decode().splitlines() if line.lstrip().startswith("#")]
    assert all(line in document.path.read_text(encoding="utf-8") for line in original_comments)


def test_conflicting_dimensions_stay_as_draft_and_do_not_write(document):
    original = document.path.read_bytes()
    document.set_dimension(("parts", KEY, "mechanical_outer_diameter_mm"), 110.0)
    with pytest.raises(ValueError, match="overlap"):
        document.save()
    assert document.dirty
    assert document.path.read_bytes() == original
    document.set_dimension(("parts", KEY, "mechanical_outer_diameter_mm"), 95.0)
    document.save()
    assert not document.dirty


def test_undo_redo_revert_and_two_saves_preserve_original_precision(document):
    original = deepcopy(document.document)
    document.set_dimension(("parts", KEY, "mechanical_inner_diameter_mm"), 70.0)
    document.undo()
    assert document.document == original
    document.redo()
    document.save()
    document.set_dimension(("parts", KEY, "length_mm"), 178.0)
    document.revert()
    assert document.part(KEY)["length_mm"] == next(p for p in original["parts"] if p["key"] == KEY)["length_mm"]
    assert document.part(KEY)["mechanical_inner_diameter_mm"] == 70.0
    document.set_dimension(("parts", KEY, "length_mm"), 178.0)
    document.save()
    original_coil = next(p for p in original["parts"] if p["key"] == KEY)
    assert document.part(KEY)["mechanical_outer_diameter_mm"] == original_coil["mechanical_outer_diameter_mm"]


@pytest.mark.parametrize("value", [True, "70", float("nan"), float("inf")])
def test_invalid_input_does_not_mutate_draft(document, value):
    before = deepcopy(document.document)
    with pytest.raises(ValueError):
        document.set_dimension(("parts", KEY, "mechanical_inner_diameter_mm"), value)
    assert document.document == before


def test_existing_array_dimensions_are_updated_by_index(document):
    key = "intermediate_lens_upper_pole"
    field = "pole_root_fillet_radius_range_mm"
    original = list(document.part(key)[field])
    document.set_dimension(("parts", key, field, 0), 2.5)
    assert document.part(key)[field] == [2.5, original[1]]
    assert document.updates() == {("parts", key, field): [2.5, original[1]]}


def test_source_change_is_detected_but_save_copy_can_preserve_draft(document, tmp_path):
    document.set_dimension(("parts", KEY, "mechanical_inner_diameter_mm"), 70.0)
    document.path.write_bytes(document.path.read_bytes() + b"\n# external edit\n")
    external = document.path.read_bytes()
    with pytest.raises(ValueError, match="changed outside"):
        document.save()
    path = document.path
    document.save_copy(tmp_path / "copy.toml")
    assert path.read_bytes() == external
    assert document.part(KEY)["mechanical_inner_diameter_mm"] == 70.0
    assert not document.dirty


def test_per_part_material_roundtrip_uses_new_optional_field(document):
    before = deepcopy(document.part(KEY))
    document.assign_material(KEY, "copper")
    assert document.path.read_bytes().find(b"material_regions") == -1
    document.save()
    assert document.part(KEY)["material_class"] == before["material_class"]
    assert document.part(KEY)["material_regions"]["body"]["material_key"] == "copper"
    assert document.part(KEY)["length_mm"] == before["length_mm"]
    document.undo()
    document.save()
    assert document.part(KEY)["material_regions"] == {}
    document.redo()
    document.save()
    assert document.part(KEY)["material_regions"]["body"]["material_key"] == "copper"


def test_existing_pole_bore_can_be_enlarged_without_changing_vacuum(document):
    key = "intermediate_lens_upper_pole"
    original_vacuum = document.part(key)["vacuum_inner_diameter_mm"]
    document.set_dimension(("parts", key, "mechanical_bore_diameter_mm"), 22.0)
    document.save()
    assert document.part(key)["mechanical_bore_diameter_mm"] == 22.0
    assert document.part(key)["vacuum_inner_diameter_mm"] == original_vacuum


def test_project_save_failure_keeps_draft(document):
    before = document.path.read_bytes()
    document.set_dimension(("parts", KEY, "mechanical_inner_diameter_mm"), 70.0)
    def save(path, updates):
        assert path == document.path
        assert updates[("parts", KEY, "mechanical_inner_diameter_mm")] == 70.0
        raise ValueError("Assembly load failed")
    with pytest.raises(ValueError, match="Assembly load failed"):
        document.save(project_save=save)
    assert document.dirty
    assert document.path.read_bytes() == before


def test_non_module_file_and_fixed_centre_edits_are_rejected(tmp_path, document):
    path = tmp_path / "other.toml"
    path.write_text("[profile]\nname='other'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="module TOML"):
        PartModelDocument(path)
    with pytest.raises(ValueError, match="centre is fixed"):
        document.set_dimension(("parts", KEY, "local_center_z_mm"), 0.0)


@pytest.fixture
def model_3d():
    return {
        "schema_version": 1,
        "base": {"kind": "box", "width_mm": 40.0, "height_mm": 30.0, "length_mm": 50.0},
        "transform": {"scale_xy": [1.2, 0.85], "offset_mm": [4.0, -2.0, 3.0],
                      "rotation_deg": [12.0, 25.0, 40.0]},
        "features": [
            {"id": "hole_1", "kind": "hole", "axis": "z", "center_mm": [-8.0, 0.0, 0.0],
             "depth_mm": 60.0, "diameter_mm": 8.0, "enabled": True},
            {"id": "slot_1", "kind": "slot", "axis": "z", "center_mm": [9.0, 0.0, 0.0],
             "depth_mm": 60.0, "width_mm": 4.0, "length_mm": 12.0,
             "rotation_deg": 0.0, "enabled": True},
        ],
    }


def test_model_configuration_and_nested_dimensions_are_independent_of_solver_fields(document, model_3d):
    original = deepcopy(document.part(KEY))
    expected = deepcopy(model_3d)
    document.set_model_3d(KEY, model_3d)
    model_3d["transform"]["offset_mm"][0] = 999.0
    assert document.part(KEY)["model_3d"] == expected
    document.set_dimension(("parts", KEY, "model_3d", "base", "width_mm"), 42.5)
    document.set_dimension(("parts", KEY, "model_3d", "transform", "rotation_deg", 1), 37.5)
    document.set_dimension(("parts", KEY, "model_3d", "features", 0, "diameter_mm"), 9.0)
    expected["base"]["width_mm"] = 42.5
    expected["transform"]["rotation_deg"][1] = 37.5
    expected["features"][0]["diameter_mm"] = 9.0
    assert document.part(KEY) == {**original, "model_3d": expected}
    assert document.updates() == {("parts", KEY, "model_3d"): expected}
    document.undo()
    assert document.part(KEY)["model_3d"]["features"][0]["diameter_mm"] == 8.0
    document.redo()
    document.save()
    assert PartModelDocument(document.path).part(KEY) == {**original, "model_3d": expected}
    assert not document.dirty


@pytest.mark.parametrize("suffix,value", [
    (("base", "width_mm"), -1.0),
    (("transform", "scale_xy", 0), 0.0),
    (("features", 1, "length_mm"), 1.0),
    (("features", 0, "diameter_mm"), float("nan")),
    (("features", 0, "depth_mm"), 10**1000),
    (("features", -1, "width_mm"), 6.0),
    (("features", 30, "diameter_mm"), 6.0),
    (("features", 0, "enabled"), 1.0),
    (("transform", "missing", 0), 6.0),
], ids=["negative-size", "zero-scale", "short-slot", "nan", "overflow",
        "negative-index", "missing-index", "boolean-field", "missing-path"])
def test_model_numeric_errors_leave_document_and_history_unchanged(document, model_3d, suffix, value):
    document.set_model_3d(KEY, model_3d)
    before, history = deepcopy(document.document), deepcopy(document._history)
    with pytest.raises(ValueError):
        document.set_dimension(("parts", KEY, "model_3d", *suffix), value)
    assert document.document == before
    assert document._history == history


def test_model_feature_transactions_keep_stable_order_and_undo_as_one_change(document, model_3d):
    document.set_model_3d(KEY, model_3d)
    document.update_model_3d(KEY, {"transform": {"offset_mm": [1.0, 2.0, 3.0]}})
    current = document.part(KEY)["model_3d"]
    assert current["base"] == model_3d["base"]
    assert current["features"] == model_3d["features"]
    assert current["transform"] == {**model_3d["transform"], "offset_mm": [1.0, 2.0, 3.0]}
    before = deepcopy(document.document)
    replacement = {**model_3d["features"][0], "diameter_mm": 9.0}
    document.upsert_model_feature(KEY, replacement)
    replacement["diameter_mm"] = 100.0
    assert [feature["id"] for feature in document.part(KEY)["model_3d"]["features"]] == ["hole_1", "slot_1"]
    assert document.part(KEY)["model_3d"]["features"][0]["diameter_mm"] == 9.0
    document.undo()
    assert document.document == before
    document.redo()
    before = deepcopy(document.document)
    document.remove_model_feature(KEY, "hole_1")
    assert [feature["id"] for feature in document.part(KEY)["model_3d"]["features"]] == ["slot_1"]
    document.undo()
    assert document.document == before
    with pytest.raises(ValueError, match="Unknown model feature"):
        document.remove_model_feature(KEY, "absent")
    assert document.document == before
    document.redo()
    document.save()
    document.undo()
    document.save()
    assert document.part(KEY)["model_3d"]["features"][0]["id"] == "hole_1"


@pytest.mark.parametrize("operation", ["update", "feature"])
def test_first_model_edit_includes_editable_transform_defaults(document, model_3d, operation):
    from temsim.part_model_features import default_model_3d

    defaults = default_model_3d(document.part(KEY))
    if operation == "update":
        document.update_model_3d(KEY, {"transform": {"offset_mm": [0.0, 1.0, 0.0]}})
        defaults["transform"]["offset_mm"] = [0.0, 1.0, 0.0]
    else:
        document.upsert_model_feature(KEY, model_3d["features"][0])
        defaults["features"] = [model_3d["features"][0]]
    assert document.part(KEY)["model_3d"] == defaults
    document.undo()
    assert "model_3d" not in document.part(KEY)


def test_model_removal_undo_redo_and_revert_round_trip_without_empty_table(document, model_3d):
    document.set_model_3d(KEY, model_3d)
    document.save()
    document.set_model_3d(KEY, None)
    assert document.updates() == {("parts", KEY, "model_3d"): None}
    document.revert()
    assert document.part(KEY)["model_3d"] == model_3d
    document.set_model_3d(KEY, None)
    document.save()
    assert "model_3d" not in PartModelDocument(document.path).part(KEY)
    document.undo()
    document.save()
    assert document.part(KEY)["model_3d"] == model_3d
    document.redo()
    document.save()
    assert "model_3d" not in document.part(KEY)


@pytest.mark.parametrize("outside", [True, False])
def test_boolean_failure_preserves_schema_valid_draft_and_file(document, model_3d, outside):
    feature = model_3d["features"][0]
    feature["center_mm"] = [1000.0, 0.0, 0.0] if outside else [0.0, 0.0, 0.0]
    feature["diameter_mm"] = 8.0 if outside else 100.0
    model_3d["features"] = [feature]
    before = document.path.read_bytes()
    document.set_model_3d(KEY, model_3d)
    with pytest.raises(ValueError):
        document.save()
    assert document.path.read_bytes() == before
    assert document.dirty
    assert document.part(KEY)["model_3d"] == model_3d
    feature["enabled"] = False
    document.upsert_model_feature(KEY, feature)
    document.save()
    assert document.part(KEY)["model_3d"]["features"] == [feature]


def test_invalid_schema_replacement_and_duplicate_feature_are_atomic(document, model_3d):
    document.set_model_3d(KEY, model_3d)
    before, history = deepcopy(document.document), deepcopy(document._history)
    with pytest.raises(ValueError):
        document.update_model_3d(KEY, {"base": {"kind": "existing"}})
    with pytest.raises(ValueError):
        document.upsert_model_feature(KEY, {**model_3d["features"][0], "width_mm": 2.0})
    invalid = deepcopy(model_3d)
    invalid["features"].append(deepcopy(invalid["features"][0]))
    with pytest.raises(ValueError):
        document.set_model_3d(KEY, invalid)
    assert document.document == before
    assert document._history == history


@pytest.mark.parametrize("returned,error", [(False, "did not save"), (None, "does not match")])
def test_project_callback_must_persist_the_validated_draft(document, model_3d, returned, error):
    document.set_model_3d(KEY, model_3d)
    original, baseline = document.path.read_bytes(), deepcopy(document._baseline)
    with pytest.raises(ValueError, match=error):
        document.save(project_save=lambda path, updates: returned)
    assert document.dirty
    assert document._baseline == baseline
    assert document.path.read_bytes() == original


def test_source_change_during_validation_prevents_project_callback(document, model_3d, monkeypatch):
    document.set_model_3d(KEY, model_3d)
    validate = document.validate
    callbacks = []

    def validate_and_external_edit():
        validate()
        document.path.write_bytes(document.path.read_bytes() + b"\n# external edit during validation\n")

    monkeypatch.setattr(document, "validate", validate_and_external_edit)
    with pytest.raises(ValueError, match="changed outside"):
        document.save(project_save=lambda *args: callbacks.append(args))
    assert not callbacks
    assert document.dirty
    assert document.path.read_bytes().endswith(b"# external edit during validation\n")


def test_external_reload_is_atomic_and_cannot_discard_model_draft(document, model_3d):
    document.set_model_3d(KEY, model_3d)
    document.path.write_bytes(document.path.read_bytes() + b'\n# edited externally\n')
    external = document.path.read_bytes()
    with pytest.raises(ValueError, match="current draft"):
        document.reload()
    with pytest.raises(ValueError, match="changed outside"):
        document.save()
    assert document.part(KEY)["model_3d"] == model_3d
    document.revert()
    document.reload()
    assert document._source_bytes == external
    assert not document.can_undo and not document.dirty
    baseline = deepcopy(document.document)
    document.path.write_text("[[parts]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        document.reload()
    assert document.document == baseline
    assert document._source_bytes == external


def test_model_save_copy_preserves_full_features_after_external_source_edit(document, model_3d, tmp_path):
    document.set_model_3d(KEY, model_3d)
    source = document.path
    source.write_bytes(source.read_bytes() + b"\n# user's independent edit\n")
    external = source.read_bytes()
    destination = tmp_path / "retained_model.toml"
    document.save_copy(destination)
    assert source.read_bytes() == external
    assert document.path == destination.resolve()
    assert PartModelDocument(destination).part(KEY)["model_3d"] == model_3d
    assert not document.dirty


def test_nested_model_tables_and_feature_arrays_are_replaced_without_stale_values(document, model_3d):
    import tomli_w

    text = document.path.read_text(encoding="utf-8")
    # Exercise externally authored TOML, including quoted keys, array tables,
    # retained comments and an unrelated nested table after the model.
    text = text.replace(f'key = "{KEY}"', f"key = '{KEY}' # stable component key")
    lines = text.splitlines(keepends=True)
    _, end = module_manifest._part_span(lines, KEY)
    expanded = tomli_w.dumps({"parts": [{"model_3d": model_3d}]}).split("\n", 1)[1]
    expanded = expanded.replace("parts.model_3d", 'parts."model_3d"')
    expanded = "\n# keep CAD design comment\n" + expanded
    expanded += '\n[parts.custom_geometry_notes]\nlabel = "keep this table"\n'
    lines[end:end] = [expanded + "\n"]
    document.path.write_text("".join(lines), encoding="utf-8")
    loaded = PartModelDocument(document.path)
    before = deepcopy(loaded.part(KEY))
    loaded.remove_model_feature(KEY, "hole_1")
    loaded.set_dimension(("parts", KEY, "model_3d", "base", "height_mm"), 32.0)
    loaded.save()
    written = loaded.path.read_text(encoding="utf-8")
    assert "# keep CAD design comment" in written
    assert "# stable component key" in written
    assert loaded.part(KEY)["custom_geometry_notes"] == before["custom_geometry_notes"]
    assert loaded.part(KEY)["model_3d"]["features"] == [model_3d["features"][1]]
    assert loaded.part(KEY)["model_3d"]["base"]["height_mm"] == 32.0
    assert tomllib.loads(written) == loaded.document
    loaded.set_model_3d(KEY, None)
    loaded.save()
    assert "model_3d" not in PartModelDocument(loaded.path).part(KEY)
    assert loaded.part(KEY)["custom_geometry_notes"] == before["custom_geometry_notes"]


def test_model_replacement_respects_brackets_and_headers_inside_toml_values(document, model_3d):
    model_3d["features"][0]["id"] = "hole[1"
    model_3d["features"][1]["id"] = "slot]2"
    lines = document.path.read_text(encoding="utf-8").splitlines(keepends=True)
    start, _ = module_manifest._part_span(lines, KEY)
    lines[start:start] = [
        'design_notes = """Do not interpret the following text as a part:\n',
        '[[parts]]\n',
        'key = "imaginary_part"\n',
        '"""\n',
        'labels = [\n',
        '  ["model_3d"]\n',
        ']\n',
        "model_3d = " + module_manifest._format_toml_value(model_3d) + "\n",
    ]
    document.path.write_text("".join(lines), encoding="utf-8")
    loaded = PartModelDocument(document.path)
    before = deepcopy(loaded.part(KEY))
    loaded.remove_model_feature(KEY, "slot]2")
    loaded.save()
    assert loaded.part(KEY)["design_notes"] == before["design_notes"]
    assert loaded.part(KEY)["labels"] == [["model_3d"]]
    assert loaded.part(KEY)["model_3d"]["features"] == [model_3d["features"][0]]
    loaded.set_model_3d(KEY, None)
    loaded.save()
    assert {field: value for field, value in before.items() if field != "model_3d"} == loaded.part(KEY)


def test_model_configuration_uses_real_manifest_editor_save_and_validation(tmp_path, model_3d):
    from temsim.column.state_layout import layout_configuration_from_state
    from temsim.manifest_editor import ManifestEditor, ManifestTarget
    from temsim.optics.column import default_state

    root = tmp_path / "instruments"
    source_bytes = (INSTRUMENT_CONFIG_ROOT / MODULE).read_bytes()
    shutil.copytree(INSTRUMENT_CONFIG_ROOT, root)
    editor, target = ManifestEditor(root), ManifestTarget(MODULE, KEY)
    configuration = layout_configuration_from_state(default_state())
    draft = PartModelDocument(root / MODULE)
    original = deepcopy(draft.part(KEY))
    draft.set_model_3d(KEY, model_3d)
    draft.save(project_save=lambda path, updates: editor.save(target, updates, configuration))
    assert draft.part(KEY) == {**original, "model_3d": model_3d}
    field = next(field for field in editor.fields(target) if field.label == "model_3d")
    assert field.editable and field.value == model_3d
    assert not draft.dirty
    before = draft.path.read_bytes()
    invalid = deepcopy(model_3d)
    invalid["features"][0]["center_mm"] = [999.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        editor.save(target, {("parts", KEY, "model_3d"): invalid}, configuration)
    assert draft.path.read_bytes() == before
    assert (INSTRUMENT_CONFIG_ROOT / MODULE).read_bytes() == source_bytes
