"""Structural edits preserve TOML and validate all assembly combinations first."""

from dataclasses import replace
import shutil
import tomllib

import pytest

from temsim import module_manifest
from temsim.column.state_layout import layout_configuration_from_state
from temsim.component_operations import PartChangeSet, make_component
from temsim.manifest_editor import ManifestEditor, ManifestTarget
from temsim.optics.column import default_state
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT


MODULE = "column/C3_ProbeCorrector.toml"
GUN = "gun/FEG.toml"


@pytest.fixture
def catalog(tmp_path):
    root = tmp_path / "instruments"
    shutil.copytree(INSTRUMENT_CONFIG_ROOT, root)
    return root


def _new_part(key="user_bracket"):
    row = make_component(key=key, shape="box", center_z_mm=50, length_mm=4)
    row["order"] = 999
    return row


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_append_edit_and_remove_preserve_original_tables_and_comments(newline):
    source = (INSTRUMENT_CONFIG_ROOT / MODULE).read_text(encoding="utf-8")
    source = source.replace("\r\n", "\n").replace("\n", newline)
    source += newline + '[user_notes]' + newline + 'text = "retain # text" # rationale' + newline
    added = module_manifest.stage_manifest_text(source, PartChangeSet(added_parts=(_new_part(),)))
    assert added.startswith(source)
    # An unrelated table after the added part must also survive its removal.
    added += '[other_notes]' + newline + 'value = 7' + newline
    updated = module_manifest.stage_manifest_text(
        added, PartChangeSet(fields={("parts", "user_bracket", "name"): "Changed bracket"}),
    )
    assert tomllib.loads(updated)["parts"][-1]["name"] == "Changed bracket"
    removed = module_manifest.stage_manifest_text(updated, PartChangeSet(removed_keys=("user_bracket",)))
    expected = tomllib.loads(source)
    expected["other_notes"] = {"value": 7}
    assert tomllib.loads(removed) == expected
    assert source in removed


def test_save_then_undo_removes_added_part_and_redo_restores_it(catalog):
    path = catalog / MODULE
    draft = PartModelDocument(path)
    original = tomllib.loads(path.read_text(encoding="utf-8"))
    key = draft.add_component(_new_part())
    configuration = layout_configuration_from_state(default_state())
    editor = ManifestEditor(catalog)
    save = lambda path, changes: editor.save(ManifestTarget(MODULE, key), changes, configuration)
    draft.save(project_save=save)
    assert not draft.dirty and draft.can_undo
    draft.undo()
    assert draft.updates().removed_keys == (key,)
    draft.save(project_save=save)
    assert tomllib.loads(path.read_text(encoding="utf-8")) == original
    draft.redo()
    draft.save(project_save=save)
    assert PartModelDocument(path).part(key)["name"] == "User Bracket"


def test_cross_file_key_collision_rejected_before_target_is_written(catalog):
    gun = PartModelDocument(catalog / GUN)
    gun.add_component(_new_part("user_collision"))
    gun.save()
    target = PartModelDocument(catalog / MODULE)
    original = target.path.read_bytes()
    target.add_component(_new_part("user_collision"))
    with pytest.raises(ValueError, match="Duplicate part key in assembled"):
        ManifestEditor(catalog).save(ManifestTarget(MODULE), target.updates(),
                                     layout_configuration_from_state(default_state()))
    assert target.path.read_bytes() == original
    assert target.dirty


def test_external_revision_changes_before_or_during_validation_are_not_overwritten(catalog, monkeypatch):
    draft = PartModelDocument(catalog / MODULE)
    original = draft.path.read_bytes()
    draft.add_component(_new_part())
    changes = draft.updates()
    editor = ManifestEditor(catalog)
    configuration = layout_configuration_from_state(default_state())
    with pytest.raises(ValueError, match="changed outside"):
        editor.save(ManifestTarget(MODULE), replace(changes, expected_source_bytes=b"old revision"), configuration)
    assert draft.path.read_bytes() == original
    real_validate = ManifestEditor.validate_catalog
    def concurrent_edit(candidate):
        result = real_validate(candidate)
        draft.path.write_bytes(original + b"\n# external editor\n")
        return result
    monkeypatch.setattr(ManifestEditor, "validate_catalog", concurrent_edit)
    with pytest.raises(ValueError, match="changed during validation"):
        editor.save(ManifestTarget(MODULE), changes, configuration)
    assert draft.path.read_bytes() == original + b"\n# external editor\n"


def test_structural_changes_cannot_update_added_or_removed_fields():
    source = (INSTRUMENT_CONFIG_ROOT / MODULE).read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="must not also"):
        module_manifest.stage_manifest_text(source, PartChangeSet(
            fields={("parts", "user_bracket", "name"): "Unexpected"}, added_parts=(_new_part(),),
        ))


def test_remove_part_with_nested_tables_reopened_after_unrelated_table():
    source = (INSTRUMENT_CONFIG_ROOT / MODULE).read_text(encoding="utf-8")
    rows = []
    for order, key in enumerate(("user_first", "user_second"), 999):
        part = make_component(key=key, shape="tube", center_z_mm=50)
        part.pop("model_3d")
        part["order"] = order
        rows.append(part)
    source = module_manifest.stage_manifest_text(source, PartChangeSet(added_parts=tuple(rows)))
    source += ('\n[user_notes]\ntext = "must stay" # note comment\n'
               '[parts.model_3d] # owned model comment\nschema_version = 1\n'
               'base = { kind = "box", width_mm = 11.0, height_mm = 12.0, length_mm = 10.0 }\n'
               '[second_note]\nflag = true\n'
               '[parts.model_3d.transform]\nscale_xy = [1.0, 1.0]\n')
    expected = tomllib.loads(source)
    expected["parts"] = [part for part in expected["parts"] if part["key"] != "user_second"]
    staged = module_manifest.stage_manifest_text(source, PartChangeSet(removed_keys=("user_second",)))
    assert tomllib.loads(staged) == expected
    assert "# owned model comment" in staged
    assert "# note comment" in staged
