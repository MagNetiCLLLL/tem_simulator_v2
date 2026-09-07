"""Existing-region assignments, with small synthetic static magnetic fixtures."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tomllib

import numpy as np
import pytest

from temsim.magnetic_materials import lens_material_defaults, MU0
from temsim.module_manifest import _format_toml_value, stage_manifest_text, validate_document
from temsim.part_materials import (
    material_catalog, material_for_region, part_material_updates,
    material_application_scope, validate_part_materials, validated_material_regions,
)
from temsim.physics.axisymmetric_magnetostatics import solve_geometry_field_map


ROOT = Path(__file__).parents[1] / "configs" / "instruments"


def _snapshot(key):
    return next(row for row in material_catalog() if row["material_key"] == key)


def _set_material(part, key, region="body"):
    part["material_regions"] = next(iter(part_material_updates(part, key, region).values()))


def _part(document, key):
    return next(part for part in document["parts"] if part["key"] == key)


def test_catalog_reuses_sourced_iron_and_explicit_nonmagnetic_model():
    rows = {row["material_key"]: row for row in material_catalog()}
    defaults = lens_material_defaults()
    assert rows["femm_pure_iron"]["bh_material"] == defaults["bh_material"]
    assert rows["femm_pure_iron"]["linear_source"] == defaults["linear_material_reference"]
    for key in ("copper", "aluminum", "nonmagnetic_stainless_steel"):
        assert rows[key]["relative_permeability"] == 1
        assert rows[key]["magnetic_response"] == "nonmagnetic_approximation"
        assert "No measured" in rows[key]["scope"]
    rows["femm_pure_iron"]["bh_material"]["b_t"][0] = 99
    assert _snapshot("femm_pure_iron")["bh_material"]["b_t"][0] == 0


@pytest.mark.parametrize("name", ("EnergyFilter.toml", "NoEnergyFilter.toml"))
def test_assignment_adds_only_whitelisted_toml_field_and_preserves_geometry(name):
    path = ROOT / "project_and_recording_system" / name
    text = path.read_text(encoding="utf-8")
    document = tomllib.loads(text)
    part = _part(document, "intermediate_lens_yoke")
    before = deepcopy(part)
    updates = part_material_updates(part, "femm_pure_iron")
    assert part == before
    staged = stage_manifest_text(text, updates)
    assert "".join(line for line in staged.splitlines(keepends=True)
                   if not line.startswith("material_regions = ")) == text
    result = tomllib.loads(staged)
    validate_document(result)
    edited = _part(result, part["key"])
    assignment = edited.pop("material_regions")
    assert assignment["body"]["material_key"] == "femm_pure_iron"
    assert result == document
    # Updating an existing inline snapshot must not duplicate the assignment.
    edited["material_regions"] = assignment
    twice = stage_manifest_text(staged, part_material_updates(edited, "aluminum"))
    assert twice.count("material_regions = ") == 1
    assert material_for_region(_part(tomllib.loads(twice), part["key"]))["material_key"] == "aluminum"
    with pytest.raises(ValueError, match="Missing TOML field"):
        stage_manifest_text(text, {("parts", part["key"], "invented_property"): 1})


def test_inline_table_quotes_nested_keys_and_values():
    value = {"a.b": {"space key": ['quotes " and brackets [ ]', 1.0], "\u00b5r": 1}}
    assert tomllib.loads("value = " + _format_toml_value(value))["value"] == value


def test_aperture_body_material_persists_as_definition_without_magnetic_domain():
    from temsim.part_materials import is_magnetostatic_body
    text = (ROOT / "project_and_recording_system" / "EnergyFilter.toml").read_text(encoding="utf-8")
    document = tomllib.loads(text)
    aperture = _part(document, "selected_area_aperture")
    staged = stage_manifest_text(text, part_material_updates(aperture, "femm_pure_iron"))
    recovered = _part(tomllib.loads(staged), aperture["key"])
    assert material_for_region(recovered)["material_key"] == "femm_pure_iron"
    assert not is_magnetostatic_body(recovered)
    assert "no modeled magnetostatic solid" in material_application_scope(recovered)
    recovered.pop("material_regions")
    assert recovered == aperture


def test_existing_split_regions_roundtrip_and_preserve_other_assignments():
    text = (ROOT / "column" / "C3.toml").read_text(encoding="utf-8")
    document = tomllib.loads(text)
    yoke = _part(document, "objective_lens_yoke")
    _set_material(yoke, "femm_pure_iron", "upper")
    _set_material(yoke, "aluminum", "lower")
    validate_document(document)
    staged = stage_manifest_text(text, {("parts", yoke["key"], "material_regions"): yoke["material_regions"]})
    recovered = _part(tomllib.loads(staged), yoke["key"])
    assert material_for_region(recovered, "upper")["material_key"] == "femm_pure_iron"
    assert material_for_region(recovered, "lower")["material_key"] == "aluminum"
    assert material_for_region(recovered) is None


def test_saved_snapshot_is_independent_of_catalog_and_region_reads(monkeypatch):
    import temsim.part_materials as module
    part = dict(key="yoke", mechanical_profile="magnetic_lens_yoke")
    _set_material(part, "femm_pure_iron")
    monkeypatch.setattr(module, "material_catalog", lambda: pytest.fail("Saved assignment must not re-read the catalog"))
    read = material_for_region(part, "lower")
    read["bh_material"]["b_t"][0] = 4
    assert material_for_region(part)["bh_material"]["b_t"][0] == 0


@pytest.mark.parametrize("key", ("vacuum", "femm_pure_iron"))
def test_driven_coils_cannot_be_vacuum_or_unmodelled_magnetic_windings(key):
    coil = dict(key="coil", mechanical_profile="magnetic_excitation_coil")
    with pytest.raises(ValueError, match="excitation coils require a nonmagnetic winding material"):
        part_material_updates(coil, key)
    assert "material_regions" not in coil


@pytest.mark.parametrize("change", (
    {"relative_permeability": float("nan")}, {"relative_permeability": -1},
    {"schema_version": 9}, {"scope": ""}, {"magnetic_response": "paint_only"},
))
def test_invalid_response_snapshots_are_rejected(change):
    part = dict(key="yoke", mechanical_profile="magnetic_lens_yoke",
                material_regions={"body": {**_snapshot("femm_pure_iron"), **change}})
    with pytest.raises(ValueError):
        validated_material_regions(part)


def test_material_regions_must_match_existing_geometry():
    document = tomllib.loads((ROOT / "project_and_recording_system" / "EnergyFilter.toml").read_text(encoding="utf-8"))
    part = _part(document, "intermediate_lens_yoke")
    _set_material(part, "copper", "upper")
    with pytest.raises(ValueError, match="upper/lower material assignments require existing split"):
        validate_document(document)
    with pytest.raises(ValueError, match="Unsupported material region"):
        part_material_updates(part, "copper", "new_hole")
    with pytest.raises(ValueError, match="Unknown material key"):
        part_material_updates(part, "invented_iron")


def test_air_core_rejects_adding_magnetic_housing():
    parent = dict(key="lens", mechanical_profile="magnetic_lens_assembly", magnetic_circuit_topology="air_core")
    housing = dict(key="housing", parent_key="lens", mechanical_profile="magnetic_lens_housing")
    _set_material(housing, "femm_pure_iron")
    with pytest.raises(ValueError, match="air-core circuit cannot contain a magnetic material"):
        validate_part_materials([parent, housing])


def _fixture(*, split=False):
    coil = dict(key="coil", start_z_mm=-5, end_z_mm=5,
                data=dict(mechanical_profile="magnetic_excitation_coil", field_source_key="lens",
                          mechanical_inner_diameter_mm=8, mechanical_outer_diameter_mm=12))
    yoke = dict(key="yoke", start_z_mm=-7, end_z_mm=7,
                data=dict(mechanical_profile="magnetic_lens_yoke", material_class="soft_magnetic",
                          mechanical_inner_diameter_mm=16, mechanical_outer_diameter_mm=20))
    if split:
        yoke["data"]["material_intervals_mm"] = [[-7, -1], [1, 7]]
    return [coil, yoke]


def _binding(parts):
    encoded = json.dumps(dict(lens_key="lens", lens_assembly=dict(parts=parts)), sort_keys=True, separators=(",", ":"))
    return SimpleNamespace(canonical_geometry_json=encoded,
                           geometry_fingerprint=hashlib.sha256(encoded.encode()).hexdigest())


def _settings(*, nonlinear=False):
    result = dict(solver="axisymmetric_linear_fem", relative_permeability=14872.0,
                  ampere_turns=5, radial_nodes=16, axial_nodes=24, padding_factor=2)
    if nonlinear:
        result.update(solver="axisymmetric_nonlinear_fem", bh_material=lens_material_defaults()["bh_material"],
                      channel_ampere_turns={"lens": 5})
    return result


def _vacuum_curve():
    # Analytic synthetic reference only; not a material catalog entry.
    return dict(label="Synthetic vacuum B-H", source_url="test:analytic-vacuum", source_sha256="0"*64,
                b_t=[0, 1, 10], h_a_per_m=[0, 1/MU0, 10/MU0])


@pytest.mark.parametrize("nonlinear", (False, True))
def test_explicit_iron_response_matches_independent_global_material_reference(nonlinear):
    parts, settings = _fixture(), _settings(nonlinear=nonlinear)
    reference = solve_geometry_field_map(_binding(parts), settings)
    parts[1]["data"]["material_regions"] = {"body": _snapshot("femm_pure_iron")}
    overridden = {**settings, "relative_permeability": 1.0,
                  "material_permeabilities": {"soft_magnetic": 1.0}}
    if nonlinear:
        overridden.update(bh_material=_vacuum_curve(), material_bh_overrides={"soft_magnetic": _vacuum_curve()})
    assigned = solve_geometry_field_map(_binding(parts), overridden)
    for expected, actual in zip(reference.components_t, assigned.components_t):
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-13)
    parts[1]["data"]["material_regions"] = {"body": _snapshot("aluminum")}
    nonmagnetic = solve_geometry_field_map(_binding(parts), settings)
    assert not np.allclose(assigned.components_t[1], nonmagnetic.components_t[1], rtol=1e-3, atol=1e-10)


@pytest.mark.parametrize("nonlinear", (False, True))
def test_assigned_magnetic_housing_uses_same_law_as_independent_yoke(nonlinear):
    parts, settings = _fixture(), _settings(nonlinear=nonlinear)
    reference = solve_geometry_field_map(_binding(parts), settings)
    parts[1]["data"].update(mechanical_profile="magnetic_lens_housing",
                             material_regions={"body": _snapshot("femm_pure_iron")})
    assigned = solve_geometry_field_map(_binding(parts), settings)
    for expected, actual in zip(reference.components_t, assigned.components_t):
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-13)


def test_identical_explicit_and_legacy_bh_regions_do_not_conflict():
    parts, settings = _fixture(), _settings(nonlinear=True)
    reference = solve_geometry_field_map(_binding(parts), settings)
    shared = deepcopy(parts[1])
    shared["key"] = "shared_iron_body"
    shared["data"]["material_regions"] = {"body": _snapshot("femm_pure_iron")}
    assigned = solve_geometry_field_map(_binding([*parts, shared]), settings)
    for expected, actual in zip(reference.components_t, assigned.components_t):
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-13)


@pytest.mark.parametrize("nonlinear", (False, True))
def test_split_region_assignment_matches_independent_two_body_reference(nonlinear):
    parts, settings = _fixture(split=True), _settings(nonlinear=nonlinear)
    parts[1]["data"]["material_regions"] = {"body": _snapshot("femm_pure_iron"), "upper": _snapshot("aluminum")}
    assigned = solve_geometry_field_map(_binding(parts), settings)
    independent = [deepcopy(parts[0])]
    for key, start, end, material_class in (("upper_body", -7, -1, "upper_nonmagnetic"), ("lower_body", 1, 7, "soft_magnetic")):
        row = deepcopy(parts[1])
        row.update(key=key, start_z_mm=start, end_z_mm=end)
        row["data"].pop("material_regions")
        row["data"].pop("material_intervals_mm")
        row["data"]["material_class"] = material_class
        independent.append(row)
    settings["material_permeabilities"] = {"upper_nonmagnetic": 1.0}
    if nonlinear:
        settings["material_bh_overrides"] = {"upper_nonmagnetic": _vacuum_curve()}
    reference = solve_geometry_field_map(_binding(independent), settings)
    for expected, actual in zip(reference.components_t, assigned.components_t):
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-13)


def test_magnetic_housing_is_included_in_validation_domain_and_material_scene():
    from temsim.magnetic_validation import FieldProblem, extend_problem_grid, make_scene
    from temsim.physics.axisymmetric_magnetostatics import solve_geometry_field_details
    parts, settings = _fixture(), _settings()
    housing = deepcopy(parts[1])
    housing["key"] = "housing"
    housing["data"].update(mechanical_profile="magnetic_lens_housing", mechanical_inner_diameter_mm=22,
                           mechanical_outer_diameter_mm=26, material_regions={"body": _snapshot("femm_pure_iron")})
    binding = _binding([*parts, housing])
    problem = FieldProblem("lens", binding.canonical_geometry_json, json.dumps(settings), 1.0)
    details = solve_geometry_field_details(binding, settings)
    expanded = extend_problem_grid(problem, details[0].axes_m)
    assert json.loads(expanded.settings_json)["validation_grid_axes_m"][0][-1] == pytest.approx(.026)
    scene = make_scene([problem], [details])
    r_index = np.argmin(abs(scene.r_m - .012))
    z_index = np.argmin(abs(scene.z_m))
    assert scene.regions[r_index, z_index] == 1


def test_magnetic_neighbour_housing_is_in_linear_and_joint_nonlinear_geometry():
    from temsim.column.module_assembly import AssemblyPart
    from temsim.physics.lens_field_provider import lens_geometry_binding
    from temsim.physics.nonlinear_circuits import resolve_nonlinear_provider
    physical = _fixture()
    rows = [dict(key="lens", parent_key=None, mechanical_profile="magnetic_lens_assembly"),
            dict(key="foreign", parent_key=None, mechanical_profile="magnetic_lens_assembly")]
    for part in physical:
        rows.append(dict(part["data"], key=part["key"], parent_key="lens"))
    rows.append(dict(key="foreign_housing", parent_key="foreign", mechanical_profile="magnetic_lens_housing",
                     mechanical_inner_diameter_mm=22, mechanical_outer_diameter_mm=26,
                     material_regions={"body": _snapshot("femm_pure_iron")}))
    parts = tuple(AssemblyPart("test", "synthetic.toml", row["key"], row["key"], "common", -7, 0, 7, 14,
                              row["parent_key"], dict(row, local_start_z_mm=-7, local_center_z_mm=0,
                                                     local_end_z_mm=7, length_mm=14)) for row in rows)
    lens = SimpleNamespace(key="lens", name="lens", z_mm=0, percent=100, polarity=1, enabled=True,
                           field_support_mm=lambda: (-20, 20))
    state = SimpleNamespace(lenses=[lens], lens_field_map_descriptors={"lens": _settings()},
                            _resolved_assembly=SimpleNamespace(parts=parts, modules=(), vacuum_bore_segments=(), vacuum_liner_segments=()))
    binding = lens_geometry_binding(state, "lens", lens)
    assembly = json.loads(binding.canonical_geometry_json)["lens_assembly"]
    assert [row["key"] for row in assembly["magnetostatic_neighbours"]] == ["foreign_housing"]
    state.lens_field_map_descriptors["lens"] = _settings(nonlinear=True)
    binding = lens_geometry_binding(state, "lens", lens)
    geometry, _, _ = resolve_nonlinear_provider(state, "lens", lens, binding, prepare_only=True)
    assert [row["key"] for row in geometry["lens_assembly"]["magnetostatic_neighbours"]] == ["foreign_housing"]


def test_material_edit_invalidates_field_binding_without_changing_lens_strength():
    from temsim.optics.column import default_state
    from temsim.physics.lens_field_provider import lens_geometry_binding
    state = default_state()
    key = "intermediate_lens"
    before = lens_geometry_binding(state, key)
    strengths = [(lens.key, lens.percent) for lens in state.lenses]
    assembly = state._resolved_assembly
    parts = []
    for part in assembly.parts:
        if part.key == key + "_yoke":
            data = deepcopy(dict(part.data))
            data["material_regions"] = {"body": _snapshot("femm_pure_iron")}
            part = replace(part, data=data)
        parts.append(part)
    state._resolved_assembly = replace(assembly, parts=tuple(parts))
    after = lens_geometry_binding(state, key)
    assert after.geometry_fingerprint != before.geometry_fingerprint
    assert after.assembly_fingerprint != before.assembly_fingerprint
    assert [(lens.key, lens.percent) for lens in state.lenses] == strengths
