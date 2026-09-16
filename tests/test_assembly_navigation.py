from dataclasses import replace
from pathlib import Path

import pytest

from temsim.assembly_navigation import assembly_sections, component_anchor, section_by_component
from temsim.assembly_catalog import AssemblyCatalog
from temsim.optics.column import default_state
from temsim.vacuum import boundary_anchors, resolve_regions, VacuumMap


@pytest.fixture
def state():
    return default_state()


def test_named_sections_cover_every_component_without_changing_physical_inputs(state):
    assembly = state._resolved_assembly
    rows = assembly_sections(assembly)
    assert [r.name for r in rows if r.kind == "subassembly"] == [
        "Imaging and projector stack", "Projection and detector chamber", "Energy filter"]
    assert set(section_by_component(assembly)) == {p.key for p in assembly.parts}
    for row in rows:
        axial = [assembly.part(k) for k in row.part_keys if not assembly.part(k).data.get("branch_path_only")]
        assert row.start_z_mm == min(p.start_z_mm for p in axial)
        assert row.end_z_mm == max(p.end_z_mm for p in axial)
    assert not state.vacuum_map.enabled
    assert len(resolve_regions(state, include_disabled=True)) == 6


def test_snapshot_carries_membership_and_resolves_anchors_without_live_files(state, monkeypatch):
    from temsim.instrument_snapshot import encode_instrument, decode_instrument
    graph = encode_instrument(state)
    expected = boundary_anchors(state)
    restored = decode_instrument(graph)
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("Frozen navigation read a live file"))
    monkeypatch.setattr(Path, "read_bytes", lambda *a, **k: pytest.fail("Frozen navigation read a live file"))
    assert boundary_anchors(restored) == expected
    assert assembly_sections(restored._resolved_assembly) == assembly_sections(state._resolved_assembly)
    assert encode_instrument(restored) == graph


def test_legacy_snapshot_uses_functional_sections_and_does_not_invent_subassembly(state):
    assembly = state._resolved_assembly
    state._resolved_assembly = replace(assembly, modules=tuple(replace(m,
        geometry={k: v for k, v in m.geometry.items() if k != "navigation_subassemblies"}) for m in assembly.modules))
    assert all(r.kind == "functional" for r in assembly_sections(state._resolved_assembly))
    assert len(resolve_regions(state, include_disabled=True)) == 6


def test_baseline_allowance_checks_metadata_and_does_not_hide_physical_changes(state):
    from copy import deepcopy
    import runpy
    from temsim.instrument_snapshot import capture_instrument_snapshot
    compare = runpy.run_path("scripts/archive_flat_tip_baseline.py")["compare_navigation_graph"]
    current = capture_instrument_snapshot(state).to_dict()["graph"]
    previous = deepcopy(current)
    for node in previous["nodes"]:
        if node["type"] == "temsim.column.module_assembly:ModuleDefinition":
            geometry = node["attributes"]["geometry"]["mapping"]
            geometry[:] = [[k, v] for k, v in geometry if k != "navigation_subassemblies"]
    result = compare(previous, current, Path.cwd())
    assert result["physical_input_graph_equal"] and result["verified_navigation_metadata"]
    altered = deepcopy(current)
    module = next(n for n in altered["nodes"] if n["type"] == "temsim.column.module_assembly:ModuleDefinition")
    module["attributes"]["length_mm"] += 1
    assert not compare(previous, altered, Path.cwd())["physical_input_graph_equal"]
    module = next(n for n in altered["nodes"] if n["type"] == "temsim.column.module_assembly:ModuleDefinition"
                  and any(k == "navigation_subassemblies" for k, v in n["attributes"]["geometry"]["mapping"]))
    rows = dict(module["attributes"]["geometry"]["mapping"])["navigation_subassemblies"]
    rows["tuple"][0]["mapping"].append(["unverified", True])
    with pytest.raises(ValueError, match="differs from its definition"):
        compare(previous, altered, Path.cwd())


def test_component_identity_anchor_survives_storage_rename_and_follows_position(state):
    assembly = state._resolved_assembly
    part = assembly.part("intermediate_lens_excitation_coil")
    key = component_anchor(part, "center")
    state._resolved_assembly = replace(assembly, parts=tuple(replace(p,
        source_file="new/location.toml", start_z_mm=p.start_z_mm+2,
        center_z_mm=p.center_z_mm+2, end_z_mm=p.end_z_mm+2) if p.key == part.key else p for p in assembly.parts))
    assert boundary_anchors(state)[key] == part.center_z_mm+2
    assert boundary_anchors(state)[part.key+".center"] == part.center_z_mm+2


def test_subassembly_anchor_follows_saved_placement_and_survives_map_roundtrip(tmp_path):
    from temsim.shared_tip import copy_catalog_tree
    from temsim.paths import INSTRUMENT_CONFIG_ROOT
    from temsim.part_model_document import PartModelDocument
    root = tmp_path/"instruments"
    copy_catalog_tree(INSTRUMENT_CONFIG_ROOT, root)
    catalog = AssemblyCatalog(root)
    state = default_state()
    catalog.apply(state, catalog.default_selection())
    section = next(r for r in assembly_sections(state._resolved_assembly) if r.name == "Energy filter")
    anchor = section.key+".origin"
    before = boundary_anchors(state)[anchor]
    draft = PartModelDocument(root/"project_and_recording_system/EnergyFilter.toml")
    draft.place_subassembly("iliad_filter", {"mode": "fixed", "origin_z_mm": 1210.0})
    draft.save()
    catalog.apply(state, catalog.default_selection())
    assert boundary_anchors(state)[anchor] == pytest.approx(before-1.5)
    region = state.vacuum_map.regions[-1]
    region.end_anchor, region.end_offset_mm = anchor, boundary_anchors(state)["column_end"]-boundary_anchors(state)[anchor]
    state.vacuum_map.save(tmp_path/"vacuum.toml")
    assert VacuumMap.load(tmp_path/"vacuum.toml") == state.vacuum_map
    assert not state.vacuum_map.enabled
    assert len(resolve_regions(state, include_disabled=True)) == 6


def test_vacuum_editor_new_boundaries_use_id_and_pressure_edit_preserves_loaded_reference(qtbot, state):
    from temsim.gui.vacuum_map_page import VacuumMapPage
    page = VacuumMapPage()
    qtbot.addWidget(page)
    page.set_state(state)
    page.select_region("column")
    before = state.vacuum_map.regions[2].start_anchor
    page.pressure.setText("4e-7")
    assert page.apply(), page.status.text()
    assert state.vacuum_map.regions[2].start_anchor == before
    key = component_anchor(state._resolved_assembly.part("condenser_lens_2"), "center")
    page.end_anchor.setCurrentIndex(page.end_anchor.findData(key))
    page.end_offset.setValue(0)
    assert page.apply(), page.status.text()
    assert state.vacuum_map.regions[2].end_anchor == key
    assert any(r.end_medium is not None for r in resolve_regions(state, include_disabled=True))
    assert not state.vacuum_map.enabled
    assert "Illumination" in page.modules_text.text()


def test_missing_reference_cannot_be_replaced_by_a_pressure_only_edit(qtbot, state):
    from temsim.gui.vacuum_map_page import VacuumMapPage
    state.vacuum_map.regions[2].end_anchor = "component:missing.center"
    page = VacuumMapPage()
    qtbot.addWidget(page)
    page.set_state(state)
    page.select_region("column")
    page.pressure.setText("4e-7")
    assert "Missing boundary" in page.end_anchor.currentText()
    assert not page.apply()
    assert state.vacuum_map.regions[2].end_anchor == "component:missing.center"


def test_primary_tree_context_menu_routes_to_existing_component(qtbot, state):
    from PySide6.QtWidgets import QMenu
    from temsim.gui.assembly_panel import AssemblyPanel
    from temsim.runtime_parameters import runtime_targets
    catalog = AssemblyCatalog()
    page = AssemblyPanel(catalog, catalog.default_selection())
    qtbot.addWidget(page)
    page.load_assembly(state._resolved_assembly, runtime_targets(state))
    page.resize(450, 750)
    page.show()
    assert page.component_pages.currentIndex() == page.assembly_tree_index
    assert page.template_controls.isHidden()
    page.template_toggle.click()
    assert not page.template_controls.isHidden()
    assert page.select_key("haadf")
    tree = page.assembly_tree
    item = tree.currentItem()
    assert "detector_chamber.toml" in item.toolTip(0)
    tree._context_menu(tree.visualItemRect(item).center())
    menu = tree.findChild(QMenu)
    with qtbot.waitSignal(page.navigation_requested) as result:
        menu.actions()[2].trigger()
    assert result.args == ["haadf", "vacuum", state._resolved_assembly.part("haadf").center_z_mm]
    menu.close()
    page.component_pages.setCurrentIndex(0)
    assert page.tree.current_key() == "haadf"
    page.component_pages.setCurrentIndex(page.assembly_tree_index)
    page.select_key("intermediate_lens_excitation_coil")
    page.component_pages.setCurrentIndex(1)
    assert page.mechanical_tree.current_key() == "intermediate_lens_excitation_coil"


def test_3d_parts_groups_preserve_parent_selection_and_do_not_edit_files(qtbot):
    from temsim.gui.part_model_editor import PartModelEditorPage
    from temsim.paths import INSTRUMENT_CONFIG_ROOT
    page = PartModelEditorPage()
    qtbot.addWidget(page)
    assert page.open_path(INSTRUMENT_CONFIG_ROOT/"project_and_recording_system/EnergyFilter.toml", selected_key="intermediate_lens_excitation_coil")
    item = page._tree_nodes["intermediate_lens_excitation_coil"]
    assert item.parent() == page._tree_nodes["intermediate_lens"]
    assert item.parent().parent() == page._subassembly_nodes["projector_stack"]
    page.tree.setCurrentItem(page._subassembly_nodes["detector_chamber"])
    assert page._selected_key == "intermediate_lens_excitation_coil"
    assert not page.session.dirty
