"""Storage migration, persistent placements, transactions and physics identity."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
import tomllib

import pytest
import tomli_w

from temsim import module_manifest
from temsim.component_operations import make_component
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT
from temsim.shared_tip import copy_catalog_tree, dependencies
from temsim.subassemblies import resolve_document, resolved_origins, storage_documents

MODULE = "project_and_recording_system/EnergyFilter.toml"


@pytest.fixture
def catalog(tmp_path):
    root = tmp_path / "instruments"
    copy_catalog_tree(INSTRUMENT_CONFIG_ROOT, root)
    return root


def test_migration_keeps_all_instance_ids_and_physical_positions():
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.optics.column import default_state
    from temsim.assembly_structure import build_assembly_structure
    catalog = AssemblyCatalog()
    assembly = catalog.apply(default_state(), replace(catalog.default_selection(), recording="Energy Filter"))
    actual = build_assembly_structure(assembly).to_dict()
    previous = json.loads(Path("docs/development/evidence/default-assembly-identity-map-v1.json").read_text())
    # Display naming was later made manufacturer-neutral. Keep the historical
    # identity/position receipt immutable and compare every non-name field.
    for old, new in zip(previous["components"], actual["components"], strict=True):
        old["name"] = new["name"]
    assert actual == previous


def test_composed_module_has_no_duplicate_physical_storage():
    path = INSTRUMENT_CONFIG_ROOT / MODULE
    raw = tomllib.loads(path.read_text())
    assert "parts" not in raw
    assert len(raw["subassemblies"]) == 3
    assert len(dependencies(path)) == 3
    result = module_manifest.read_document(path)
    assert len(result["parts"]) == 53
    assert len({part["key"] for part in result["parts"]}) == 53
    assert resolved_origins(result) == {"projector_stack": 0, "detector_chamber": 772.5, "iliad_filter": 1211.5}
    restored = storage_documents(raw, result, path)
    for source, document in restored.items():
        assert document == tomllib.loads(source.read_text())


def test_dimensions_route_to_subassembly_and_undo_save(catalog):
    path = catalog / MODULE
    draft = PartModelDocument(path)
    root_bytes = path.read_bytes()
    files = dependencies(path)
    draft.set_dimension(("parts", "intermediate_lens_excitation_coil", "length_mm"), 168.0)
    draft.save()
    assert not draft.dirty
    assert path.read_bytes() == root_bytes
    changed = [p for p, data in files.items() if p.read_bytes() != data]
    assert [p.name for p in changed] == ["projector_stack.toml"]
    assert PartModelDocument(path).part("intermediate_lens_excitation_coil")["length_mm"] == 168
    draft.undo()
    draft.save()
    assert module_manifest.read_document(path)["parts"] == module_manifest.read_document(INSTRUMENT_CONFIG_ROOT / MODULE)["parts"]


def test_subassembly_comments_survive_dimension_edits(catalog):
    source = catalog.parent / "subassemblies/projector_stack.toml"
    source.write_bytes(source.read_bytes().replace(b'key = "intermediate_lens_housing"',
        b'key = "intermediate_lens_housing" # retained physical provenance'))
    draft = PartModelDocument(catalog / MODULE)
    draft.set_dimension(("parts", "intermediate_lens_housing", "length_mm"), 225.0)
    draft.save()
    assert "# retained physical provenance" in source.read_text()


def test_multi_file_failure_rolls_back_all_written_inputs(catalog, monkeypatch):
    draft = PartModelDocument(catalog / MODULE)
    draft.set_dimension(("parts", "intermediate_lens_housing", "length_mm"), 225.0)
    draft.place_subassembly("iliad_filter", {"mode": "fixed", "origin_z_mm": 1210.0})
    original = {draft.path: draft.path.read_bytes(), **dependencies(draft.path)}
    real_write = module_manifest._atomic_write_text
    calls = []
    def fail_second(path, text):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("simulated file replacement failure")
        real_write(path, text)
    monkeypatch.setattr(module_manifest, "_atomic_write_text", fail_second)
    with pytest.raises(OSError, match="simulated"):
        draft.save()
    assert all(path.read_bytes() == content for path, content in original.items())
    assert draft.dirty


def test_invalid_overlap_is_rejected_before_any_file_is_written(catalog):
    draft = PartModelDocument(catalog / MODULE)
    files = {draft.path: draft.path.read_bytes(), **dependencies(draft.path)}
    draft.set_dimension(("parts", "intermediate_lens_housing", "length_mm"), 241.0)
    with pytest.raises(ValueError, match="[Oo]verlap|clearance"):
        draft.save()
    assert all(path.read_bytes() == data for path, data in files.items())


def test_persistent_anchor_follows_edited_reference_and_preserves_filter_path(catalog):
    draft = PartModelDocument(catalog / MODULE)
    before = deepcopy(draft.document)
    draft.set_dimension(("parts", "projector_lens_2_housing", "length_mm"), 273.0)
    assert draft.part("projector_lens_2_housing")["local_center_z_mm"] == 635.0
    assert resolved_origins(draft.document)["detector_chamber"] == 771.5
    assert resolved_origins(draft.document)["iliad_filter"] == 1210.5
    old = {part["key"]: part for part in before["parts"]}
    for part in draft.document["parts"]:
        for field, value in part.items():
            if field.startswith("path_"):
                assert value == old[part["key"]][field]
    draft.save()
    assert resolved_origins(PartModelDocument(draft.path).document) == resolved_origins(draft.document)
    assert draft.part("haadf")["local_center_z_mm"] == old["haadf"]["local_center_z_mm"] - 1


def test_explicit_fixed_origin_draft_is_undoable_and_persists(catalog):
    draft = PartModelDocument(catalog / MODULE)
    draft.place_subassembly("iliad_filter", {"mode": "fixed", "origin_z_mm": 1210.0})
    assert resolved_origins(draft.document)["iliad_filter"] == 1210.0
    assert draft.updates().placement_updates
    draft.save()
    assert not draft.dirty
    assert resolved_origins(PartModelDocument(draft.path).document)["iliad_filter"] == 1210
    draft.undo()
    assert draft.document["subassemblies"][-1]["placement"]["mode"] == "anchor"
    draft.save()
    assert resolved_origins(PartModelDocument(draft.path).document)["iliad_filter"] == 1211.5


@pytest.mark.parametrize("reference", ["energy_filter", "not_a_component"])
def test_cycle_or_missing_anchor_preserves_draft(catalog, reference):
    draft = PartModelDocument(catalog / MODULE)
    before = deepcopy(draft.document)
    with pytest.raises(ValueError, match="Cyclic|Missing"):
        draft.place_subassembly("projector_stack", {"mode": "anchor", "reference_part": reference,
            "reference_point": "center", "offset_mm": 0, "local_datum_mm": 0})
    assert draft.document == before and not draft.dirty


def test_duplicate_or_missing_file_fails_explicitly(catalog):
    path = catalog / MODULE
    raw = tomllib.loads(path.read_text())
    raw["subassemblies"][1]["file"] = raw["subassemblies"][0]["file"]
    with pytest.raises(ValueError, match="only once"):
        resolve_document(raw, path)
    raw["subassemblies"][1]["file"] = "absent.toml"
    with pytest.raises(ValueError, match="unavailable"):
        resolve_document(raw, path)


def test_linked_file_change_blocks_stale_save(catalog):
    draft = PartModelDocument(catalog / MODULE)
    draft.set_dimension(("parts", "intermediate_lens_housing", "length_mm"), 225.0)
    source = catalog.parent / "subassemblies/projector_stack.toml"
    source.write_bytes(source.read_bytes() + b"\n# independent editor\n")
    originals = {draft.path: draft.path.read_bytes(), **dependencies(draft.path)}
    with pytest.raises(ValueError, match="dependency changed"):
        draft.save()
    assert all(path.read_bytes() == data for path, data in originals.items())


def test_independent_save_copy_is_flat_and_detached(catalog, tmp_path):
    draft = PartModelDocument(catalog / MODULE)
    destination = tmp_path / "independent.toml"
    draft.save_copy(destination)
    assert not dependencies(destination)
    assert "subassemblies" not in draft.document
    original = deepcopy(draft.document)
    linked = PartModelDocument(catalog / MODULE)
    linked.set_dimension(("parts", "intermediate_lens_housing", "length_mm"), 225.0)
    linked.save()
    assert PartModelDocument(destination).document == original


def test_new_parts_remain_independent_root_additions(catalog):
    draft = PartModelDocument(catalog / MODULE)
    source_bytes = dependencies(draft.path)
    draft.add_component(make_component(key="test_bracket", center_z_mm=500, length_mm=2))
    draft.save()
    raw = tomllib.loads(draft.path.read_text())
    assert [part["key"] for part in raw["parts"]] == ["test_bracket"]
    assert all(path.read_bytes() == data for path, data in source_bytes.items())
    assert len(PartModelDocument(draft.path).document["parts"]) == 54


def test_subassembly_edit_changes_consumed_input_identity(catalog):
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.optics.column import default_state
    from temsim.calculation_manifest import _external_inputs
    cat = AssemblyCatalog(catalog)
    state = default_state()
    cat.apply(state, replace(cat.default_selection(), recording="Energy Filter"))
    before = _external_inputs(state)
    rows = [row for row in before if row.role == "assembly:subassembly"]
    assert len(rows) == 3
    path = catalog.parent / "subassemblies/projector_stack.toml"
    path.write_bytes(path.read_bytes() + b"\n# changed external input\n")
    assert _external_inputs(state) != before


def test_placement_dialog_reviews_and_stages_without_writing(qtbot, catalog):
    from temsim.gui.subassembly_dialog import SubassemblyPlacementDialog
    draft = PartModelDocument(catalog / MODULE)
    files = {draft.path: draft.path.read_bytes(), **dependencies(draft.path)}
    applied = []
    dialog = SubassemblyPlacementDialog(draft, lambda: applied.append(True))
    qtbot.addWidget(dialog)
    dialog.group.setCurrentIndex(2)
    dialog.mode.setCurrentIndex(0)
    dialog.origin.setText("1210")
    assert dialog.preview.rowCount() == 20
    assert dialog.apply_button.isEnabled(), dialog.error.text()
    dialog.apply()
    assert applied and draft.dirty
    assert all(path.read_bytes() == data for path, data in files.items())
    draft.undo()
    assert not draft.dirty


def test_physical_editor_shows_storage_and_placement_entry(qtbot, catalog):
    from temsim.gui.part_model_editor import PartModelEditorPage
    page = PartModelEditorPage()
    qtbot.addWidget(page)
    assert page.open_path(catalog / MODULE, selected_key="intermediate_lens_excitation_coil")
    assert "projector_stack.toml" in page.source_label.text()
    assert page.subassembly_button.isEnabled()
    page._open_subassemblies()
    dialog = page._subassembly_dialog
    assert dialog is not None
    dialog.group.setCurrentIndex(2)
    dialog.mode.setCurrentIndex(0)
    dialog.origin.setText("1210")
    # A file switch cannot apply an obsolete dialog to a different draft.
    assert page.open_path(catalog / "column/C3.toml")
    dialog.apply()
    assert "switched files" in dialog.error.text()
    assert not page.session.dirty
    dialog.reject()
