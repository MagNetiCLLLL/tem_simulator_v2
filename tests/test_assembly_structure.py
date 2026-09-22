"""Identity/navigation migration preserves the already resolved physical state."""
from dataclasses import replace
from itertools import product
from pathlib import Path
from uuid import uuid4

import pytest

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.assembly_structure import build_assembly_structure, identity_map_from_document, stable_id
from temsim.instrument_snapshot import capture_instrument_snapshot, decode_instrument
from temsim.optics.column import default_state


@pytest.fixture
def state():
    return default_state()


@pytest.fixture
def filter_state(state):
    catalog = AssemblyCatalog()
    catalog.apply(state, replace(catalog.default_selection(), recording="Energy Filter"))
    return state


def test_mapping_does_not_mutate_physics_or_historical_snapshot(filter_state, monkeypatch):
    state = filter_state
    before = capture_instrument_snapshot(state)
    assembly = state._resolved_assembly
    def no_io(*args, **kwargs):
        pytest.fail("Captured assembly identity must not reopen current files")
    with monkeypatch.context() as guard:
        guard.setattr(Path, "open", no_io)
        structure = build_assembly_structure(assembly)
        restored = decode_instrument(before.graph)
        assert build_assembly_structure(restored._resolved_assembly).to_dict() == structure.to_dict()
    assert capture_instrument_snapshot(state).digest == before.digest
    assert len(structure.components) == len(assembly.parts)
    assert "energy_filter_slit" in structure.by_key
    for part in assembly.parts:
        row = structure.resolve(part.key)
        assert (row.start_z_mm, row.center_z_mm, row.end_z_mm) == (part.start_z_mm, part.center_z_mm, part.end_z_mm)
        assert structure.resolve(part.definition_id) == row == structure.resolve(row.instance_id)
        assert row.parent_id == (structure.by_key[part.parent_key].instance_id if part.parent_key else None)


def test_identity_survives_file_move_display_edit_and_reordering(state):
    original = state._resolved_assembly
    first = build_assembly_structure(original)
    changed = replace(original,
        modules=tuple(replace(module, source_file="moved/"+module.source_file) for module in original.modules),
        parts=tuple(replace(part, source_file="moved/"+part.source_file, name=part.name+" renamed",
                            center_z_mm=part.center_z_mm+1) for part in reversed(original.parts)))
    second = build_assembly_structure(changed)
    assert first.identity_map() == second.identity_map()
    assert first.module_ids == second.module_ids
    assert first.by_key["feg_tip"].legacy_authority != second.by_key["feg_tip"].legacy_authority
    assert first.by_key["feg_tip"].center_z_mm != second.by_key["feg_tip"].center_z_mm


def test_instances_remain_distinct_and_explicit_migration_keeps_identity(state):
    assembly = state._resolved_assembly
    initial = build_assembly_structure(assembly)
    part = assembly.part("feg_tip")
    copy = replace(part, key="tip_copy", data={**part.data, "mechanical_only": True})
    copied = replace(assembly, parts=(*assembly.parts, copy))
    result = build_assembly_structure(copied)
    assert result.by_key["tip_copy"].instance_id != result.by_key["feg_tip"].instance_id
    renamed = replace(assembly, parts=tuple(replace(p, key="renamed_tip") if p.key == "feg_tip" else p for p in assembly.parts))
    carried = {(part.module_key, "renamed_tip"): initial.by_key["feg_tip"].instance_id}
    assert build_assembly_structure(renamed, identity_map=carried).by_key["renamed_tip"].instance_id == carried[(part.module_key, "renamed_tip")]
    with pytest.raises(ValueError, match="cannot share"):
        build_assembly_structure(copied, identity_map={(copy.module_key, copy.key): result.by_key["feg_tip"].instance_id})
    with pytest.raises(ValueError, match="uninstalled"):
        build_assembly_structure(assembly, identity_map={("missing", "missing"): str(uuid4())})
    with pytest.raises(ValueError, match="collides"):
        build_assembly_structure(assembly, identity_map={(part.module_key, part.key): stable_id("group", "gun")})


def test_exported_identity_map_roundtrips_without_applying_edited_coordinates(state):
    assembly = state._resolved_assembly
    original = build_assembly_structure(assembly)
    exported = original.to_dict()
    exported["components"][0]["center_z_mm"] += 1000
    mapped = build_assembly_structure(assembly, identity_map=identity_map_from_document(exported))
    assert mapped.to_dict() == original.to_dict()
    exported["components"].append(exported["components"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        identity_map_from_document(exported)


@pytest.mark.parametrize("parent", ["feg_tip", "missing"])
def test_invalid_hierarchy_is_reported_without_silently_dropping_parts(state, parent):
    assembly = state._resolved_assembly
    invalid = replace(assembly, parts=tuple(replace(p, parent_key=parent) if p.key == "feg_tip" else p for p in assembly.parts))
    with pytest.raises(ValueError, match="Cyclic|Missing"):
        build_assembly_structure(invalid)


def test_functional_groups_do_not_split_mechanical_parents_or_filter_path(filter_state):
    state = filter_state
    structure = build_assembly_structure(state._resolved_assembly)
    groups = {key: row.group_key for key, row in structure.by_key.items()}
    assert groups["intermediate_lens"] == "imaging"
    assert groups["haadf"] == "detection"
    assert groups["energy_filter_slit"] == "energy_filter"
    assert groups["eds_detector_system"] == groups["sample"] == groups["objective_lower_pole"] == "objective"
    assert structure.by_key["sample"].parent_id == structure.by_key["sample_stage"].instance_id
    assert structure.by_key["energy_filter_slit"].path_coordinate == "curvilinear_s_mm"
    assert structure.by_key["feg_tip"].definition_reference == "../../sources/FEG_tip.toml"


_OPTIONS = list(product(("FEG", "FEG + Mono", "Thermionic"),
                       ("C2", "C3", "C3 + Probe Corrector", "C3 + Image Corrector", "C3 + Probe Corrector + Image Corrector"),
                       ("None", "Electrostatic beam blanker")))


@pytest.mark.parametrize("gun,column,blanker", _OPTIONS)
def test_all_existing_assembly_combinations_remain_representable(gun, column, blanker):
    state = default_state()
    catalog = AssemblyCatalog()
    assembly = catalog.apply(state, AssemblySelection(gun, column, "Energy Filter", blanker))
    structure = build_assembly_structure(assembly)
    assert len(structure.by_id) == len(assembly.parts)
    assert {row.key for row in structure.components} == {part.key for part in assembly.parts}
    if blanker == "Electrostatic beam blanker":
        assert structure.by_key["nanopulser_aperture"].group_key == "blanker"


def test_tree_preserves_selection_and_parent_links_on_renamed_storage(state, qtbot):
    from temsim.gui.assembly_structure_tree import AssemblyStructureTree, IDENTITY_ROLE
    from PySide6.QtCore import Qt
    tree = AssemblyStructureTree()
    qtbot.addWidget(tree)
    tree.load_assembly(state._resolved_assembly)
    assert tree.select_key("sample")
    selected = tree.current_identity()
    parent = tree.currentItem().parent()
    assert parent.data(0, Qt.ItemDataRole.UserRole).key == "sample_stage"
    assert parent.parent().data(0, Qt.ItemDataRole.UserRole).key == "objective_lens"
    moved = replace(state._resolved_assembly, parts=tuple(replace(p, source_file="renamed/"+p.source_file) for p in state._resolved_assembly.parts))
    tree.load_assembly(moved)
    assert tree.current_identity() == selected == tree.currentItem().data(0, IDENTITY_ROLE)
    assert tree.currentItem().data(0, Qt.ItemDataRole.UserRole).module_path.startswith("renamed/")
    with qtbot.waitSignal(tree.component_selected) as signal:
        tree.select_key("haadf")
    assert signal.args[0].key == "haadf"
    assert tree.currentItem().parent().text(0).startswith("Detection and recording")


def test_panel_assembly_navigation_and_export_use_same_components(filter_state, qtbot, tmp_path, monkeypatch):
    state = filter_state
    import json
    from PySide6.QtWidgets import QFileDialog
    from temsim.gui.assembly_panel import AssemblyPanel
    from temsim.runtime_parameters import runtime_targets
    catalog = AssemblyCatalog()
    panel = AssemblyPanel(catalog, catalog.selection_for_resolved(state._resolved_assembly))
    qtbot.addWidget(panel)
    panel.load_assembly(state._resolved_assembly, runtime_targets(state))
    before = capture_instrument_snapshot(state).digest
    panel.select_key("feg_tip")
    assert panel.assembly_tree.current_identity() == panel.assembly_tree.structure.by_key["feg_tip"].instance_id
    panel.component_pages.setCurrentIndex(panel.assembly_tree_index)
    with qtbot.waitSignal(panel.component_selected) as signal:
        assert panel.select_key("energy_filter_slit")
    assert signal.args[0].module_path.endswith("EnergyFilter.toml")
    assert panel.component_pages.currentIndex() == panel.assembly_tree_index
    with qtbot.waitSignal(panel.component_selected) as signal:
        panel.load_assembly(state._resolved_assembly, runtime_targets(state))
    assert signal.args[0].key == "energy_filter_slit"
    target = tmp_path/"map.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(target), "JSON"))
    panel.export_structure_button.click()
    assert json.loads(target.read_text(encoding="utf-8"))["components"] == [r.to_dict() for r in panel.assembly_tree.structure.components]
    assert capture_instrument_snapshot(state).digest == before
