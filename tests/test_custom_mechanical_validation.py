"""CAD copies preserve declared solids without becoming optical/FEM bodies."""
from copy import deepcopy
from dataclasses import replace
import tomllib

import numpy as np
import pytest

from temsim.component_operations import added_component_document, copied_component_document, make_component
from temsim.magnetic_circuits import belongs_to_circuit, circuit_inventory, optical_owner
from temsim.module_manifest import validate_document
from temsim.parameter_impact import component_impact_summary, describe_parameter_impact
from temsim.part_materials import (
    is_magnetostatic_body, material_application_scope, material_catalog, validate_part_materials,
)
from temsim.part_model_3d import part_dimension_specs, part_model_from_document
from temsim.paths import INSTRUMENT_CONFIG_ROOT


def document(relative):
    return tomllib.loads((INSTRUMENT_CONFIG_ROOT / relative).read_text(encoding="utf-8-sig"))


def part(doc, key):
    return next(row for row in doc["parts"] if row["key"] == key)


MODULES = tuple(path for path in INSTRUMENT_CONFIG_ROOT.rglob("*.toml") if path.name != "catalog.toml")


@pytest.mark.parametrize("path", MODULES, ids=lambda path: str(path.relative_to(INSTRUMENT_CONFIG_ROOT)))
def test_each_native_module_accepts_independent_solid_without_changing_native_rows(path):
    original = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    candidate, _ = added_component_document(original, make_component(
        key="custom_fixture", inner_diameter_mm=0, center_z_mm=12))
    validate_document(candidate)
    assert candidate["parts"][:-1] == original["parts"]
    assert circuit_inventory(candidate["parts"]) == circuit_inventory(original["parts"])
    assert all(spec.path[-1] != "vacuum_inner_diameter_mm" for spec in part_dimension_specs(candidate, "custom_fixture"))


@pytest.mark.parametrize("mutation,match", [
    ({"mechanical_only": False}, "mechanical_only"),
    ({"axial_vacuum_context_only": False}, "axial_vacuum_context_only"),
    ({"local_center_z_mm": float("nan")}, "finite"),
    ({"mechanical_outer_diameter_mm": float("inf")}, "finite"),
    ({"material_intervals_mm": [[0, 2], [1, 3]]}, "disjoint"),
    ({"field_source_key": "objective_lens"}, "current/circuit"),
    ({"magnetic_lens_keys": ["objective_lens"]}, "copied components"),
    ({"parent_key": "absent"}, "unknown parent"),
])
def test_invalid_custom_data_cannot_escape_validation(mutation, match):
    candidate, key = added_component_document(document("column/C2.toml"), make_component(key="fixture"))
    part(candidate, key).update(mutation)
    with pytest.raises(ValueError, match=match):
        validate_document(candidate)


@pytest.mark.parametrize("key", ["thermionic_cathode", "objective_lens", "objective_lens_yoke", "sample", "c1", "df_s"])
def test_custom_keys_cannot_impersonate_other_runtime_variants(key):
    candidate, _ = added_component_document(document("gun/FEG.toml"), make_component(key=key))
    with pytest.raises(ValueError, match="reserved built-in"):
        validate_document(candidate)


def test_custom_addition_does_not_weaken_native_nanopulser_or_magnetic_validation():
    candidate, _ = added_component_document(document("beam_blanker/NanoPulser.toml"), make_component(key="fixture"))
    validate_document(candidate)
    candidate["parts"] = [row for row in candidate["parts"] if row["key"] != "nanopulser_aperture"]
    with pytest.raises(ValueError, match="one deflector followed by one aperture"):
        validate_document(candidate)
    candidate, _ = added_component_document(document("project_and_recording_system/EnergyFilter.toml"), make_component(key="fixture"))
    part(candidate, "intermediate_lens_yoke")["mechanical_inner_diameter_mm"] = 0
    with pytest.raises(ValueError, match="vacuum|Vacuum"):
        validate_document(candidate)


@pytest.mark.parametrize("key", ["condenser_lens_1_lower_pole", "condenser_lens_2_upper_pole", "objective_upper_pole", "objective_lower_pole"])
def test_copied_poles_keep_legacy_orientation_under_an_unrelated_key(key):
    source = document("column/C3.toml")
    before = part_model_from_document(source, key, include_children=False).meshes
    target, new_key = copied_component_document(document("gun/FEG.toml"), source, key, "unrelated_name", 4000)
    validate_document(target)
    after = part_model_from_document(target, new_key, include_children=False).meshes
    delta = 4000 - part(source, key)["local_center_z_mm"]
    for a, b in zip(before, after, strict=True):
        assert b.vertices == pytest.approx(a.vertices + [0, 0, delta])
        assert np.array_equal(b.faces, a.faces)
        assert b.key == "unrelated_name"
        assert all(path[1] == "unrelated_name" for surface in b.surfaces.values() for path in surface["parameter_paths"])


def test_standalone_split_materials_and_cap_paths_ignore_new_parent_geometry():
    source = document("column/C2.toml")
    key = "objective_lens_excitation_coil"
    copper = next(row for row in material_catalog() if row["material_key"] == "copper")
    part(source, key)["material_regions"] = {"upper": copper, "lower": deepcopy(copper)}
    copied, new_key = copied_component_document(source, source, key, "standalone_coil", 4000, parent_key="objective_lens")
    validate_document(copied)
    row = part(copied, new_key)
    meshes = part_model_from_document(copied, new_key, include_children=False).meshes
    assert {mesh.region for mesh in meshes} == {"upper", "lower"}
    for index, mesh in enumerate(meshes):
        assert np.min(mesh.vertices[:, 2]) == pytest.approx(row["material_intervals_mm"][index][0])
        assert ("parts", new_key, "material_intervals_mm", index, 0) in mesh.surfaces["axial_negative"]["parameter_paths"]
        assert all(path[1] == new_key for surface in mesh.surfaces.values() for path in surface["parameter_paths"])
    length = next(spec for spec in part_dimension_specs(copied, new_key) if spec.path[-1] == "length_mm")
    assert length.editable and "Scale the copied material sections" in length.reason
    validate_part_materials(copied["parts"])
    assert "definition only" in material_application_scope(row)
    assert optical_owner(row, {p["key"]: p for p in copied["parts"]}) is None


def test_ordinary_coil_copy_never_inherits_target_objective_split():
    source = document("project_and_recording_system/EnergyFilter.toml")
    key = "intermediate_lens_excitation_coil"
    before = part_model_from_document(source, key, include_children=False).meshes
    target, new_key = copied_component_document(document("column/C2.toml"), source, key,
                                               "ordinary_coil", 800, parent_key="objective_lens")
    validate_document(target)
    after = part_model_from_document(target, new_key, include_children=False).meshes
    assert len(after) == len(before) == 1
    assert after[0].vertices == pytest.approx(before[0].vertices + [0, 0, 800 - part(source, key)["local_center_z_mm"]])
    assert all(spec.path[1] == new_key for spec in part_dimension_specs(target, new_key))


@pytest.mark.parametrize("mode", [None, "ideal", "analytic", "linear", "nonlinear", "custom"])
def test_custom_materials_and_dimension_reports_never_claim_physics(mode):
    row = {**make_component(key="spare_coil"), "mechanical_profile": "magnetic_excitation_coil"}
    for field in ("length_mm", "material_regions", "vacuum_inner_diameter_mm", "model_3d"):
        result = describe_parameter_impact(row, ("parts", row["key"], field), simulation_mode=mode)
        assert not result.active and result.status == "inactive"
        assert set(result.effects) == {"display", "geometry"}
    summary = component_impact_summary(row, simulation_mode=mode)
    assert not summary.active_effects
    assert "excluded" in summary.detail


def test_native_lens_geometry_fingerprint_and_circuit_ignore_copies_even_as_neighbours():
    from temsim.optics.column import default_state
    from temsim.column.module_assembly import _module_vacuum_segments
    from temsim.physics.lens_field_provider import lens_geometry_binding

    state = default_state()
    key = "objective_lens"
    state.lens_field_map_descriptors = {key: {"solver": "axisymmetric_linear_fem", "padding_factor": 2}}
    before = lens_geometry_binding(state, key)
    assembly = state._resolved_assembly
    index = {p.key: p for p in assembly.parts}
    copies = []
    for source_key in (key, "objective_lens_yoke", "objective_lens_excitation_coil"):
        native = index[source_key]
        data = dict(native.data)
        data.update(mechanical_part_role="custom_mechanical_copy", mechanical_only=True,
                    axial_vacuum_context_only=True, geometry_template_key=source_key)
        for field in tuple(data):
            if field.startswith("magnetic_circuit_") or field == "field_source_key":
                del data[field]
        copied = replace(native, key="copy_" + source_key, parent_key=key, data=data)
        copies.append(copied)
        assert not is_magnetostatic_body(data)
        assert not belongs_to_circuit(copied, key, index)
    state._resolved_assembly = replace(assembly, parts=(*assembly.parts, *copies))
    after = lens_geometry_binding(state, key)
    assert after.canonical_geometry_json == before.canonical_geometry_json
    assert after.geometry_fingerprint == before.geometry_fingerprint
    assert circuit_inventory(state._resolved_assembly.parts) == circuit_inventory(assembly.parts)
    module = next(module for module in assembly.modules if module.key == index[key].module_key)
    origin = index[key].start_z_mm - next(row for row in module.parts if row.key == key).start_z_mm
    assert _module_vacuum_segments(module, origin, state._resolved_assembly.parts) == _module_vacuum_segments(module, origin, assembly.parts)
