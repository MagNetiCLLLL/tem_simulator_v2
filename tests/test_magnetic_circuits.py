"""Synthetic magnetic-circuit tests; dimensions/materials are not OEM data."""

from dataclasses import replace
from types import SimpleNamespace
import json

import numpy as np
import pytest

from temsim.column.module_assembly import AssemblyPart
from temsim.magnetic_circuits import circuit_inventory, radial_profile_mm, validate_circuit_declarations
from temsim.physics.axisymmetric_magnetostatics import _part_mask, solve_geometry_field_map
from temsim.physics.lens_field_provider import (
    lens_geometry_binding, resolve_runtime_lens_field_provider, MappedLensFieldProvider,
    bind_imported_lens_field_map,
    CoordinateRegistration, MagneticFieldMap, FieldMapProvenance, FieldMapError, FrozenMappedField,
)


def _part(key, profile, start, end, inner, outer, parent=None, **data):
    return dict(key=key, name=key, mechanical_profile=profile, parent_key=parent,
                local_start_z_mm=start, local_center_z_mm=(start+end)/2,
                local_end_z_mm=end, length_mm=end-start,
                mechanical_inner_diameter_mm=inner, mechanical_outer_diameter_mm=outer,
                vacuum_inner_diameter_mm=2, mechanical_only=profile != "magnetic_lens_assembly",
                **data)


def _shared_rows():
    rows = []
    for key, start, end in (("a", -12, 0), ("b", 0, 12)):
        rows.append(_part(key, "magnetic_lens_assembly", start, end, 2, 30,
                         magnetic_circuit_id="shared", magnetic_circuit_topology="shared_pole_multi_gap",
                         magnetic_circuit_evidence="engineering_assumption",
                         magnetic_circuit_source="Synthetic three-pole/two-channel test only"))
    for index, (start, end) in enumerate(((-12, -10), (-1, 1), (10, 12))):
        rows.append(_part(f"pole{index}", "magnetic_pole_piece", start, end, 6, 24, "a" if index < 2 else "b",
                         mechanical_bore_diameter_mm=6, mechanical_tip_diameter_mm=14,
                         magnetic_radial_profile_mm=[[0, 3, 12], [2, 3, 12]]))
    rows.append(_part("yoke", "magnetic_lens_yoke", -12, 12, 24, 30, "a"))
    rows.append(_part("coil_a", "magnetic_excitation_coil", -9, -2, 16, 20, "a", field_source_key="a"))
    rows.append(_part("coil_b", "magnetic_excitation_coil", 2, 9, 16, 20, "b", field_source_key="b"))
    return rows


def _state(rows):
    parts = tuple(AssemblyPart("test", "synthetic.toml", row["key"], row["name"], "common",
                              row["local_start_z_mm"], row["local_center_z_mm"], row["local_end_z_mm"],
                              row["length_mm"], row["parent_key"], row) for row in rows)
    lenses = [SimpleNamespace(key=row["key"], name=row["key"], z_mm=row["local_center_z_mm"],
                              percent=100, polarity=1, enabled=True, field_support_mm=lambda: (-40., 40.))
              for row in rows if row["mechanical_profile"] == "magnetic_lens_assembly"]
    settings = dict(solver="axisymmetric_linear_fem", relative_permeability=50,
                    ampere_turns=100, radial_nodes=22, axial_nodes=40, padding_factor=3)
    return SimpleNamespace(lenses=lenses, lens_field_map_descriptors={lens.key: dict(settings) for lens in lenses},
                           _resolved_assembly=SimpleNamespace(parts=parts, modules=(), vacuum_bore_segments=(), vacuum_liner_segments=()))


def test_shared_circuit_has_two_controls_but_only_three_poles():
    rows = _shared_rows()
    validate_circuit_declarations(rows)
    circuits = circuit_inventory(rows)
    assert len(circuits) == 1
    assert circuits[0].channels == ("a", "b")
    assert circuits[0].body_keys == ("pole0", "pole1", "pole2", "yoke")
    assert circuits[0].coil_keys == ("coil_a", "coil_b")


def test_shared_linear_bases_sum_to_combined_solve_without_duplicate_coils():
    state = _state(_shared_rows())
    providers = [resolve_runtime_lens_field_provider(state, lens.key, lens) for lens in state.lenses]
    fields = [p.field_map for p in providers]
    for axis_a, axis_b in zip(fields[0].axes_m, fields[1].axes_m):
        np.testing.assert_array_equal(axis_a, axis_b)
    combined = json.loads(providers[0].binding.canonical_geometry_json)
    combined["lens_key"] = "combined"
    for part in combined["lens_assembly"]["parts"]:
        if part["data"].get("mechanical_profile") == "magnetic_excitation_coil":
            part["data"]["field_source_key"] = "combined"
    binding = SimpleNamespace(canonical_geometry_json=json.dumps(combined, sort_keys=True))
    total = solve_geometry_field_map(binding, dict(state.lens_field_map_descriptors["a"], ampere_turns=200))
    for a, b, ab in zip(fields[0].components_t, fields[1].components_t, total.components_t):
        np.testing.assert_allclose(a+b, ab, rtol=1e-10, atol=1e-12)
    assert abs(fields[0].field_at_global_positions_t([0, 0, -.005])[2]) > 1e-3
    # Strength changes reuse linear unit-current maps; no magnetic solve in a view refresh.
    state.lenses[0].percent = 25
    assert resolve_runtime_lens_field_provider(state, "a", state.lenses[0]) is providers[0]
    np.testing.assert_allclose(providers[0].field_at_global_positions_t([0, 0, -.005]),
                               .25*fields[0].field_at_global_positions_t([0, 0, -.005]))
    assert resolve_runtime_lens_field_provider(state, "b", state.lenses[1]) is providers[1]


@pytest.mark.parametrize("setting,value", [
    ("relative_permeability", 80), ("radial_nodes", 30), ("padding_factor", 4),
])
def test_shared_linear_recipes_reject_inconsistent_operator_even_when_cached(setting, value):
    state = _state(_shared_rows())
    resolve_runtime_lens_field_provider(state, "a", state.lenses[0])
    state.lens_field_map_descriptors["b"][setting] = value
    with pytest.raises(FieldMapError, match="identical material, mesh and boundary"):
        resolve_runtime_lens_field_provider(state, "a", state.lenses[0])
    with pytest.raises(FieldMapError, match="identical material, mesh and boundary"):
        resolve_runtime_lens_field_provider(state, "b", state.lenses[1])


def test_shared_pole_edit_invalidates_both_bindings_but_prose_does_not():
    state = _state(_shared_rows())
    before = [lens_geometry_binding(state, lens.key).geometry_fingerprint for lens in state.lenses]
    assembly = state._resolved_assembly
    assembly.parts = tuple(replace(p, data={**p.data, "magnetic_circuit_source": "Edited citation"}) if p.key == "a" else p for p in assembly.parts)
    assert [lens_geometry_binding(state, lens.key).geometry_fingerprint for lens in state.lenses] == before
    assembly.parts = tuple(replace(p, data={**p.data, "magnetic_radial_profile_mm": [[0, 3.1, 12], [2, 3.1, 12]]}) if p.key == "pole1" else p for p in assembly.parts)
    after = [lens_geometry_binding(state, lens.key).geometry_fingerprint for lens in state.lenses]
    assert all(a != b for a, b in zip(before, after))


def test_explicit_coil_owner_is_not_replaced_by_its_mechanical_parent():
    rows = _shared_rows()
    state = _state(rows)
    original_b = resolve_runtime_lens_field_provider(state, "b", state.lenses[1]).field_map
    for row in rows:
        if row["key"] in {"coil_a", "coil_b"}:
            row["field_source_key"] = "b" if row["key"] == "coil_a" else "a"
    validate_circuit_declarations(rows)
    swapped = _state(rows)
    changed_a = resolve_runtime_lens_field_provider(swapped, "a", swapped.lenses[0]).field_map
    for expected, actual in zip(original_b.components_t, changed_a.components_t):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)


def test_explicit_shared_membership_replaces_overlap_group_hint():
    from temsim.optics.column import default_state
    from temsim.calculation_cache import calculation_signatures
    state = default_state()
    assembly = state._resolved_assembly
    state._resolved_assembly = replace(assembly, parts=tuple(
        replace(p, data={**p.data, "mechanical_overlap_group": "independent_display_group"})
        if p.key == "c1_c2_pole_piece_cartridge" else p for p in assembly.parts))
    before = [lens_geometry_binding(state, key).geometry_fingerprint for key in ("condenser_lens_1", "condenser_lens_2", "projector_lens_1")]
    signatures = calculation_signatures(state)
    state._resolved_assembly = replace(state._resolved_assembly, parts=tuple(
        replace(p, data={**p.data, "mechanical_outer_diameter_mm": 91})
        if p.key == "c1_c2_pole_piece_cartridge" else p for p in state._resolved_assembly.parts))
    after = [lens_geometry_binding(state, key).geometry_fingerprint for key in ("condenser_lens_1", "condenser_lens_2", "projector_lens_1")]
    assert before[0] != after[0] and before[1] != after[1]
    assert before[2] == after[2]
    assert calculation_signatures(state)["incident"] != signatures["incident"]


def test_air_core_has_no_invented_poles_and_rejects_iron():
    rows = [_part("a", "magnetic_lens_assembly", -10, 10, 2, 30,
                  magnetic_circuit_id="air", magnetic_circuit_topology="air_core",
                  magnetic_circuit_evidence="engineering_assumption", magnetic_circuit_source="Synthetic test"),
            _part("coil", "magnetic_excitation_coil", -8, 8, 16, 20, "a")]
    validate_circuit_declarations(rows)
    state = _state(rows)
    provider = resolve_runtime_lens_field_provider(state, "a", state.lenses[0])
    assert provider.field_at_global_positions_t([0, 0, 0])[2] > 0
    assert circuit_inventory(rows)[0].body_keys == ()
    rows.append(_part("iron", "magnetic_lens_yoke", -10, 10, 22, 28, "a"))
    with pytest.raises(ValueError, match="air-core"):
        validate_circuit_declarations(rows)


def test_monolithic_insert_is_represented_but_linear_solve_rejected():
    rows = [_part("a", "magnetic_lens_assembly", -10, 10, 2, 30,
                  magnetic_circuit_id="mono", magnetic_circuit_topology="monolithic_saturated_insert",
                  magnetic_circuit_evidence="engineering_assumption", magnetic_circuit_source="Synthetic test"),
            _part("coil", "magnetic_excitation_coil", -8, 8, 16, 20, "a"),
            _part("insert", "magnetic_lens_yoke", -10, 10, 4, 14, "a", magnetic_part_role="saturating_insert",
                  magnetic_radial_profile_mm=[[0, 2, 7], [9, 2, 2.5], [11, 2, 2.5], [20, 2, 7]])]
    validate_circuit_declarations(rows)
    state = _state(rows)
    with pytest.raises(ValueError, match="nonlinear B-H"):
        resolve_runtime_lens_field_provider(state, "a", state.lenses[0])
    state.lens_field_map_descriptors.clear()
    lens = state.lenses[0]
    binding = lens_geometry_binding(state, "a", lens)
    field = MagneticFieldMap("axisymmetric_rz", (np.linspace(0, .003, 5), np.linspace(-.03, .03, 5)),
                             (np.zeros((5, 5)), np.full((5, 5), .2)), CoordinateRegistration(),
                             binding.geometry_fingerprint, 100, 1,
                             FieldMapProvenance("fem", "", "0"*64, "Synthetic operating-point map"))
    bind_imported_lens_field_map(state, "a", field, native_provider=lens)
    mapped = resolve_runtime_lens_field_provider(state, "a", lens)
    assert mapped.excitation_scaling == "fixed_operating_point"
    assert FrozenMappedField.from_provider(mapped).scale == 1
    lens.percent = 50
    with pytest.raises(FieldMapError, match="reference excitation"):
        resolve_runtime_lens_field_provider(state, "a", lens)


def test_saturation_map_cannot_be_linearly_rescaled_or_frozen_at_wrong_current():
    state = _state(_shared_rows())
    lens = state.lenses[0]
    binding = lens_geometry_binding(state, "a", lens)
    axis = np.linspace(-.03, .03, 5)
    field = MagneticFieldMap("axisymmetric_rz", (np.linspace(0, .003, 5), axis),
                             (np.zeros((5, 5)), np.full((5, 5), .2)), CoordinateRegistration(),
                             binding.geometry_fingerprint, 100, 1, FieldMapProvenance("fem", "", "0"*64, "Synthetic constant map"))
    provider = MappedLensFieldProvider("a", field, lens, binding, excitation_scaling="fixed_operating_point")
    assert FrozenMappedField.from_provider(provider).scale == 1
    lens.percent = 50
    with pytest.raises(FieldMapError, match="reference excitation"):
        provider.field_at_global_positions_t([0, 0, 0])
    with pytest.raises(FieldMapError, match="reference excitation"):
        FrozenMappedField.from_provider(provider)


@pytest.mark.parametrize("bad", ["unknown_topology", "air_core"])
def test_invalid_circuit_topologies_are_rejected(bad):
    rows = _shared_rows()
    rows[0]["magnetic_circuit_topology"] = bad
    with pytest.raises(ValueError):
        validate_circuit_declarations(rows)


def test_invalid_membership_and_coil_ownership_are_rejected():
    rows = _shared_rows()
    rows[2]["magnetic_lens_keys"] = ["missing"]
    with pytest.raises(ValueError, match="magnetic_lens_keys"):
        validate_circuit_declarations(rows)
    rows = _shared_rows()
    rows[-1]["field_source_key"] = "missing"
    with pytest.raises(ValueError, match="field_source_key"):
        validate_circuit_declarations(rows)


def test_radial_profile_and_material_mask_use_identical_geometry():
    data = dict(mechanical_outer_diameter_mm=20, vacuum_inner_diameter_mm=4,
                mechanical_profile="magnetic_lens_yoke",
                magnetic_radial_profile_mm=[[0, 2, 10], [5, 2, 3], [10, 2, 10]])
    profile = radial_profile_mm(data, 10)
    np.testing.assert_array_equal(profile[1], [5, 2, 3])
    p = dict(data=data, start_z_mm=20, end_z_mm=30)
    np.testing.assert_array_equal(_part_mask(p, np.array([.001, .0025, .004, .009]), np.full(4, .025)),
                                   [False, True, False, False])
    np.testing.assert_array_equal(_part_mask(p, np.array([.001, .0025, .004, .009]), np.full(4, .021)),
                                   [False, True, True, False])
    with pytest.raises(ValueError, match="increasing Z"):
        radial_profile_mm({**data, "magnetic_radial_profile_mm": [[0, 2, 10], [0, 2, 3]]}, 10)


def test_coil_fractions_cannot_duplicate_channel_excitation():
    state = _state(_shared_rows())
    state._resolved_assembly.parts = tuple(replace(p, data={**p.data, "coil_ampere_turn_fraction": 2})
                                           if p.key == "coil_a" else p for p in state._resolved_assembly.parts)
    with pytest.raises(ValueError, match="fractions must sum to one"):
        resolve_runtime_lens_field_provider(state, "a", state.lenses[0])


def test_inspector_circuit_inventory_does_not_calculate_fields(qtbot, monkeypatch):
    from temsim.optics.column import default_state
    from temsim.gui.model_inspector import ModelInspectorPage
    from temsim.physics import axisymmetric_magnetostatics
    monkeypatch.setattr(axisymmetric_magnetostatics, "solve_geometry_field_map", lambda *args: pytest.fail("View must not solve"))
    state = default_state()
    original = state.to_dict()
    page = ModelInspectorPage()
    qtbot.addWidget(page)
    page.set_state(state)
    assert page.circuit_table.rowCount() >= 4
    for row in range(page.circuit_table.rowCount()):
        assert page.circuit_table.item(row, 4).text() == "Engineering Assumption"
    page.refresh()
    assert state.to_dict() == original


def test_projector_manifest_accepts_air_core_without_a_pair_of_poles(tmp_path):
    import tomllib
    import tomli_w
    import shutil
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.optics.column import default_state
    from temsim.module_manifest import validate_document
    from temsim.paths import INSTRUMENT_CONFIG_ROOT
    path = INSTRUMENT_CONFIG_ROOT / "project_and_recording_system" / "EnergyFilter.toml"
    with path.open("rb") as stream:
        document = tomllib.load(stream)
    document["parts"] = [p for p in document["parts"] if p["key"] not in {
        "projector_lens_2_yoke", "projector_lens_2_upper_pole", "projector_lens_2_lower_pole"}]
    parent = next(p for p in document["parts"] if p["key"] == "projector_lens_2")
    parent.pop("pole_piece_topology")
    parent["magnetic_circuit_topology"] = "air_core"
    validate_document(document)
    root = tmp_path / "instruments"
    shutil.copytree(INSTRUMENT_CONFIG_ROOT, root)
    target = root / "project_and_recording_system" / "EnergyFilter.toml"
    target.write_text(tomli_w.dumps(document), encoding="utf-8")
    catalog = AssemblyCatalog(root)
    state = default_state()
    original_keys = [lens.key for lens in state.lenses]
    catalog.apply(state, catalog.default_selection())
    assert [lens.key for lens in state.lenses] == original_keys
    assert "projector_lens_2_upper_pole" not in {p.key for p in state._resolved_assembly.parts}
    state.lens_field_map_descriptors["projector_lens_2"] = dict(solver="axisymmetric_linear_fem", relative_permeability=1,
                                                               ampere_turns=100, radial_nodes=18, axial_nodes=32, padding_factor=2)
    lens = next(lens for lens in state.lenses if lens.key == "projector_lens_2")
    provider = resolve_runtime_lens_field_provider(state, lens.key, lens)
    assert provider.field_at_global_positions_t([0, 0, lens.z_mm*1e-3])[2] != 0


def test_radial_profile_draws_two_matching_material_polygons(qtbot):
    from PySide6.QtWidgets import QGraphicsPolygonItem
    from temsim.gui.diagnostic_tabs import PhysicalLayoutView
    view = PhysicalLayoutView()
    qtbot.addWidget(view)
    record = SimpleNamespace(start_z_mm=20, name="Synthetic insert", key="insert")
    profile = np.array([[0, 2, 7], [5, 2, 3], [10, 2, 7]], float)
    view._selectable_item_keys = {}
    view._add_magnetic_radial_profile(record, "#abcdef", profile)
    polygons = [item for item in view.plot.items() if isinstance(item, QGraphicsPolygonItem)]
    assert len(polygons) == 2
    for item in polygons:
        points = item.polygon()
        assert [points[i].x() for i in range(3)] == [20, 25, 30]
        assert [abs(points[i].y()) for i in range(3)] == [7, 3, 7]
