"""Mechanical dimensions are material geometry, not vacuum or placement edits."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

import pytest

from temsim.part_geometry import AnnularPartGeometry, geometry_from_part


MODULE_PATH = "project_and_recording_system/EnergyFilter.toml"
COIL = "intermediate_lens_excitation_coil"
HOUSING = "intermediate_lens_housing"
YOKE = "intermediate_lens_yoke"


@pytest.fixture
def part():
    return {
        "key": "example_excitation_coil",
        "name": "Example coil",
        "mechanical_profile": "magnetic_excitation_coil",
        "local_start_z_mm": 10.0,
        "local_center_z_mm": 13.0,
        "local_end_z_mm": 20.0,
        "length_mm": 10.0,
        "mechanical_inner_diameter_mm": 40.0,
        "mechanical_outer_diameter_mm": 60.0,
        "vacuum_inner_diameter_mm": 20.0,
        "material_class": "insulated_copper_winding",
        "parent_key": "example_lens",
        "mechanical_overlap_group": "example_lens_assembly",
        "optical_reference_local_z_mm": 13.0,
    }


@pytest.mark.parametrize("profile, material", [
    ("magnetic_excitation_coil", "insulated_copper_winding"),
    ("magnetic_lens_housing", "non_magnetic_structural"),
    ("magnetic_lens_yoke", "soft_magnetic"),
])
def test_supported_annular_parts_keep_material_bore_and_axial_geometry(part, profile, material):
    part.update(mechanical_profile=profile, material_class=material)
    before = deepcopy(part)
    geometry = geometry_from_part(part)

    assert isinstance(geometry, AnnularPartGeometry)
    assert geometry.profile == profile
    assert geometry.material_class == material
    assert geometry.inner_diameter_mm == 40.0
    assert geometry.outer_diameter_mm == 60.0
    assert geometry.thickness_mm == 10.0
    assert geometry.vacuum_inner_diameter_mm == 20.0
    assert geometry.center_fraction == pytest.approx(0.3)
    assert (geometry.start_z_mm, geometry.center_z_mm, geometry.end_z_mm) == (10.0, 13.0, 20.0)
    assert geometry.updates_from(geometry) == {}
    assert part == before
    with pytest.raises(FrozenInstanceError):
        geometry.length_mm = 12


@pytest.mark.parametrize("dimension,value,expected_id,expected_od", [
    ("inner_diameter_mm", 44.0, 44.0, 60.0),
    ("outer_diameter_mm", 64.0, 40.0, 64.0),
])
def test_direct_diameter_edit_recomputes_radial_thickness_without_moving_other_dimension(
    part, dimension, value, expected_id, expected_od,
):
    original = geometry_from_part(part)
    changed = original.with_dimension(dimension, value)
    assert changed.inner_diameter_mm == expected_id
    assert changed.outer_diameter_mm == expected_od
    assert changed.thickness_mm == (expected_od - expected_id) / 2
    assert changed.center_z_mm == original.center_z_mm
    assert changed.length_mm == original.length_mm
    field = "mechanical_" + dimension
    assert changed.updates_from(original) == {("parts", original.key, field): value}


@pytest.mark.parametrize("anchor,expected_id,expected_od,updated_field", [
    ("inner", 40.0, 56.0, "mechanical_outer_diameter_mm"),
    ("outer", 44.0, 60.0, "mechanical_inner_diameter_mm"),
])
def test_radial_thickness_preserves_selected_diameter_anchor(
    part, anchor, expected_id, expected_od, updated_field,
):
    original = geometry_from_part(part)
    changed = original.with_dimension("thickness_mm", 8.0, thickness_anchor=anchor)
    assert (changed.inner_diameter_mm, changed.outer_diameter_mm) == (expected_id, expected_od)
    assert changed.thickness_mm == 8.0
    assert changed.updates_from(original) == {
        ("parts", original.key, updated_field): expected_od if anchor == "inner" else expected_id,
    }
    assert original.thickness_mm == 10.0


def test_length_updates_preserve_fixed_centre_and_asymmetric_envelope(part):
    original = geometry_from_part(part)
    changed = original.with_dimension("length_mm", 20.0)
    assert changed.center_z_mm == 13.0
    assert changed.center_fraction == original.center_fraction
    assert (changed.start_z_mm, changed.end_z_mm) == (7.0, 27.0)
    assert changed.updates_from(original) == {
        ("parts", original.key, "length_mm"): 20.0,
        ("parts", original.key, "local_start_z_mm"): 7.0,
        ("parts", original.key, "local_end_z_mm"): 27.0,
    }


def test_combined_dimensions_only_emit_relevant_toml_fields(part):
    original = geometry_from_part(part)
    changed = original.with_dimension("length_mm", 20).with_dimension("inner_diameter_mm", 42).with_dimension("outer_diameter_mm", 64)
    updates = changed.updates_from(original)
    staged = deepcopy(part)
    for (section, key, field), value in updates.items():
        assert (section, key) == ("parts", original.key)
        staged[field] = value
    expected = dict(part, length_mm=20.0, local_start_z_mm=7.0, local_end_z_mm=27.0,
                    mechanical_inner_diameter_mm=42.0, mechanical_outer_diameter_mm=64.0)
    assert staged == expected
    assert geometry_from_part(staged) == changed


@pytest.mark.parametrize("field,value", [
    ("key", "other_part"),
    ("profile", "magnetic_lens_yoke"),
    ("center_z_mm", 14.0),
    ("center_fraction", 0.5),
    ("vacuum_inner_diameter_mm", 21.0),
    ("material_class", "soft_magnetic"),
])
def test_updates_reject_changes_to_protected_geometry_identity(part, field, value):
    original = geometry_from_part(part)
    with pytest.raises(ValueError, match="centre|vacuum passage/material"):
        replace(original, **{field: value}).updates_from(original)


@pytest.mark.parametrize("value", [True, False, "10", float("nan"), float("inf"), float("-inf"), -1.0, 10**400])
@pytest.mark.parametrize("field", ["length_mm", "inner_diameter_mm", "outer_diameter_mm", "thickness_mm"])
def test_invalid_dimension_values_are_rejected(part, field, value):
    geometry = geometry_from_part(part)
    with pytest.raises(ValueError):
        geometry.with_dimension(field, value)


@pytest.mark.parametrize("field,value", [
    ("length_mm", 0.0),
    ("thickness_mm", 0.0),
    ("inner_diameter_mm", 60.0),
    ("inner_diameter_mm", 61.0),
    ("outer_diameter_mm", 40.0),
    ("outer_diameter_mm", 39.0),
    ("inner_diameter_mm", 19.0),
])
def test_invalid_envelopes_and_material_intrusion_into_vacuum_are_rejected(part, field, value):
    with pytest.raises(ValueError):
        geometry_from_part(part).with_dimension(field, value)


def test_outer_anchored_thickness_cannot_intrude_into_vacuum(part):
    geometry = geometry_from_part(part)
    with pytest.raises(ValueError, match="vacuum passage"):
        geometry.with_dimension("thickness_mm", 21.0, thickness_anchor="outer")
    with pytest.raises(ValueError, match="anchor"):
        geometry.with_dimension("thickness_mm", 8.0, thickness_anchor="middle")
    with pytest.raises(ValueError, match="Unsupported dimension"):
        geometry.with_dimension("vacuum_inner_diameter_mm", 25.0)


@pytest.mark.parametrize("field", ["local_start_z_mm", "local_center_z_mm", "local_end_z_mm", "length_mm",
                                   "mechanical_inner_diameter_mm", "mechanical_outer_diameter_mm", "vacuum_inner_diameter_mm"])
@pytest.mark.parametrize("value", [True, "20", float("nan"), float("inf")])
def test_loaded_geometry_requires_real_finite_numbers(part, field, value):
    part[field] = value
    with pytest.raises(ValueError):
        geometry_from_part(part)


@pytest.mark.parametrize("changes", [
    {"local_center_z_mm": 21.0},
    {"local_end_z_mm": 19.0},
    {"length_mm": 9.0},
    {"vacuum_inner_diameter_mm": 0.0},
])
def test_inconsistent_source_geometry_is_not_silently_repaired(part, changes):
    part.update(changes)
    with pytest.raises(ValueError):
        geometry_from_part(part)


@pytest.mark.parametrize("profile", ["magnetic_lens_assembly", "magnetic_pole_piece", "vacuum_bore", "unknown"])
def test_unsupported_part_profiles_fail_explicitly(part, profile):
    part["mechanical_profile"] = profile
    with pytest.raises(ValueError, match="support"):
        geometry_from_part(part)


def test_shaped_radial_profile_is_not_reinterpreted_as_an_annular_cylinder(part):
    part["magnetic_radial_profile_mm"] = [[10.0, 40.0, 60.0], [20.0, 42.0, 60.0]]
    with pytest.raises(ValueError, match="shaped radial profile"):
        geometry_from_part(part)
    del part["magnetic_radial_profile_mm"]
    del part["mechanical_inner_diameter_mm"]
    with pytest.raises(ValueError, match="Missing mechanical geometry.*mechanical_inner_diameter_mm"):
        geometry_from_part(part)


@pytest.mark.parametrize("sections", [[], [(10.0, 12.0), (18.0, 20.0)]])
def test_explicit_material_sections_cannot_be_edited_as_one_full_cylinder(part, sections):
    part["material_intervals_mm"] = sections
    with pytest.raises(ValueError, match="explicit material sections.*TOML"):
        geometry_from_part(part)


@pytest.mark.parametrize("location", ["part", "parent"])
@pytest.mark.parametrize("field,value", [
    ("shared_housing_key", "shared_housing"),
    ("magnetic_lens_keys", ["first_lens", "second_lens"]),
    ("magnetic_circuit_topology", "shared_pole_multi_gap"),
])
def test_shared_ownership_requires_assembly_editing_instead_of_individual_dimensions(
    part, location, field, value,
):
    parent = {"key": part["parent_key"], "mechanical_profile": "magnetic_lens_assembly"}
    (part if location == "part" else parent)[field] = value
    before = deepcopy((part, parent))
    with pytest.raises(ValueError, match="shared magnetic structure.*assembly in TOML"):
        geometry_from_part(part, parent=parent)
    assert (part, parent) == before


@pytest.mark.parametrize("profile", ["magnetic_excitation_coil", "magnetic_lens_yoke"])
@pytest.mark.parametrize("split_field", [
    "upper_yoke_start_local_z_mm", "upper_yoke_end_local_z_mm",
    "lower_yoke_start_local_z_mm", "lower_yoke_end_local_z_mm",
])
def test_even_partially_declared_parent_split_geometry_prevents_envelope_editing(
    part, profile, split_field,
):
    part["mechanical_profile"] = profile
    # A renamed/non-Objective owner can also define split physical material.
    parent = {"key": part["parent_key"], split_field: 12.0}
    with pytest.raises(ValueError, match="split material sections defined by its parent.*TOML"):
        geometry_from_part(part, parent=parent)


@pytest.fixture
def actual_column_parts():
    from temsim.paths import INSTRUMENT_CONFIG_ROOT

    path = Path(INSTRUMENT_CONFIG_ROOT) / "column/C3_ProbeCorrector_ImageCorrector.toml"
    return {part["key"]: part for part in tomllib.loads(path.read_text(encoding="utf-8"))["parts"]}


@pytest.mark.parametrize("key", ["objective_lens_excitation_coil", "objective_lens_yoke"])
@pytest.mark.parametrize("include_parent", [False, True])
def test_current_objective_split_layers_are_rejected_even_without_parent_context(
    actual_column_parts, key, include_parent,
):
    from temsim.magnetic_geometry import objective_layer_intervals_mm

    part = actual_column_parts[key]
    parent = actual_column_parts[part["parent_key"]]
    intervals = objective_layer_intervals_mm(
        parent, parent["local_start_z_mm"], part["mechanical_profile"],
    )
    assert len(intervals) == 2
    assert intervals[0][1] < intervals[1][0]
    assert "magnetic_radial_profile_mm" not in part
    with pytest.raises(ValueError, match="split material sections defined by its parent.*TOML"):
        geometry_from_part(part, parent=parent if include_parent else None)


def test_current_objective_housing_remains_a_supported_continuous_annulus(actual_column_parts):
    part = actual_column_parts["objective_lens_housing"]
    parent = actual_column_parts["objective_lens"]
    original = deepcopy(part)
    geometry = geometry_from_part(part, parent=parent)
    assert geometry == geometry_from_part(part)
    assert geometry.profile == "magnetic_lens_housing"
    assert geometry.start_z_mm == pytest.approx(part["local_start_z_mm"])
    assert geometry.end_z_mm == pytest.approx(part["local_end_z_mm"])
    assert geometry.to_shape_spec()["primitive"] == "annular_cylinder"
    assert part == original


@pytest.mark.parametrize("key", ["condenser_lens_1_housing", "condenser_lens_2_housing"])
def test_current_c1_c2_housing_sections_cannot_be_edited_independently(actual_column_parts, key):
    part = actual_column_parts[key]
    assert part["shared_housing_key"] == "condenser_c1_c2_shared_housing"
    assert part["shared_housing_section"] in {"upstream", "downstream"}
    with pytest.raises(ValueError, match="shared magnetic structure.*assembly in TOML"):
        geometry_from_part(part)


@pytest.mark.parametrize("key", [
    "condenser_lens_1_excitation_coil", "condenser_lens_2_excitation_coil",
    "condenser_lens_1_yoke", "condenser_lens_2_yoke",
])
def test_current_c1_c2_layer_gating_uses_its_shared_parent_context(actual_column_parts, key):
    part = actual_column_parts[key]
    parent = actual_column_parts[part["parent_key"]]
    assert not part.get("shared_housing_key")
    assert parent["shared_housing_key"] == "condenser_c1_c2_shared_housing"
    with pytest.raises(ValueError, match="shared magnetic structure.*assembly in TOML"):
        geometry_from_part(part, parent=parent)


def test_current_independent_projector_lens_layers_stay_supported_with_parent_context():
    from temsim.paths import INSTRUMENT_CONFIG_ROOT

    path = Path(INSTRUMENT_CONFIG_ROOT) / MODULE_PATH
    parts = {part["key"]: part for part in tomllib.loads(path.read_text(encoding="utf-8"))["parts"]}
    for key in (COIL, HOUSING, YOKE):
        part = parts[key]
        parent = parts[part["parent_key"]]
        assert parent["magnetic_circuit_topology"] == "two_pole_single_gap"
        assert geometry_from_part(part, parent=parent) == geometry_from_part(part)


def test_shape_spec_uses_radii_and_material_envelope_in_millimetres(part):
    geometry = geometry_from_part(part)
    assert geometry.to_shape_spec() == {
        "primitive": "annular_cylinder", "axis": "z", "units": "mm",
        "part_key": part["key"], "start_z_mm": 10.0, "end_z_mm": 20.0,
        "center_z_mm": 13.0, "inner_radius_mm": 20.0, "outer_radius_mm": 30.0,
        "material_class": "insulated_copper_winding",
    }
    shape = geometry.to_shape_spec()
    shape["inner_radius_mm"] = 0.0
    assert geometry.inner_diameter_mm == 40.0


def test_geometry_model_import_does_not_load_qt_or_rendering_engines():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import temsim.part_geometry; "
         "assert not any(name.split('.')[0] in {'PySide6', 'PyQt6', 'pyqtgraph', 'matplotlib', 'OpenGL'} "
         "for name in sys.modules)"],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("key,dimension,value", [
    (COIL, "inner_diameter_mm", 70.0),
    (COIL, "outer_diameter_mm", 95.0),
    (COIL, "thickness_mm", 10.0),
    (HOUSING, "inner_diameter_mm", 176.0),
    (YOKE, "outer_diameter_mm", 172.0),
    (COIL, "length_mm", 178.0),
])
def test_material_dimension_save_reloads_assembly_without_changing_unrelated_configuration(
    tmp_path, key, dimension, value,
):
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.calculation_manifest import resolved_assembly_geometry_fingerprint
    from temsim.column.state_layout import layout_configuration_from_state
    from temsim.manifest_editor import ManifestEditor, ManifestTarget
    from temsim.optics.column import default_state
    from temsim.paths import INSTRUMENT_CONFIG_ROOT

    # Start from the user's current configuration, including their 180 mm IL
    # coil length; never rewrite or restore a historical project default.
    source_path = Path(INSTRUMENT_CONFIG_ROOT) / MODULE_PATH
    source_bytes = source_path.read_bytes()
    root = tmp_path / "instruments"
    shutil.copytree(INSTRUMENT_CONFIG_ROOT, root)
    path = root / MODULE_PATH
    before = tomllib.loads(path.read_text(encoding="utf-8"))
    original_part = next(part for part in before["parts"] if part["key"] == key)
    original = geometry_from_part(original_part)
    changed = original.with_dimension(dimension, value)
    updates = changed.updates_from(original)
    assert updates

    catalog = AssemblyCatalog(root)
    state = default_state()
    selection = catalog.default_selection()
    initial_assembly = catalog.apply(state, selection, preserve_operating_parameters=True)
    initial_fingerprint = resolved_assembly_geometry_fingerprint(state)
    old_lens_strengths = {lens.key: lens.percent for lens in state.lenses}
    editor = ManifestEditor(root)
    originals = editor.save(ManifestTarget(MODULE_PATH, key), updates, layout_configuration_from_state(state))
    saved = tomllib.loads(path.read_text(encoding="utf-8"))
    expected = deepcopy(before)
    expected_part = next(part for part in expected["parts"] if part["key"] == key)
    expected_part.update({field: value for (_section, _key, field), value in updates.items()})
    assert saved == expected
    assert tomllib.loads(originals[MODULE_PATH]) == before

    assembly = catalog.apply(state, selection, preserve_operating_parameters=True)
    assert resolved_assembly_geometry_fingerprint(state) != initial_fingerprint
    assert {lens.key: lens.percent for lens in state.lenses} == old_lens_strengths
    for resolved in assembly.parts:
        old = initial_assembly.part(resolved.key)
        assert resolved.center_z_mm == old.center_z_mm
        if resolved.key != key:
            assert resolved.data == old.data
    reloaded = geometry_from_part(assembly.part(key).data)
    assert reloaded == changed
    assert reloaded.vacuum_inner_diameter_mm == original.vacuum_inner_diameter_mm
    assert reloaded.material_class == original.material_class
    if dimension != "length_mm":
        assert reloaded.length_mm == original.length_mm
    if key == COIL and dimension == "thickness_mm":
        assert reloaded.inner_diameter_mm == original.inner_diameter_mm
        assert reloaded.outer_diameter_mm == pytest.approx(original.inner_diameter_mm + 2 * value)
    assert source_path.read_bytes() == source_bytes
