"""Physical dimensions and clearances for editable concentric lens layers."""
from copy import deepcopy
from pathlib import Path
import tomllib

import pytest

from temsim.module_manifest import (
    _validate_simple_magnetic_layer_geometry,
    validate_document,
)


CONFIG_ROOT = Path(__file__).parents[1] / "configs" / "instruments"


@pytest.fixture(params=("EnergyFilter.toml", "NoEnergyFilter.toml"))
def recording_document(request):
    path = CONFIG_ROOT / "project_and_recording_system" / request.param
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _part(document, role):
    return next(part for part in document["parts"]
                if part["key"] == "intermediate_lens_" + role)


def test_current_instrument_manifests_validate_without_mutation():
    paths = sorted(CONFIG_ROOT.rglob("*.toml"))
    assert paths
    for path in paths:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
        if "parts" not in document:
            continue
        before = deepcopy(document)
        validate_document(document)
        assert document == before, path


@pytest.mark.parametrize("role,field,value", (
    ("excitation_coil", "mechanical_inner_diameter_mm", 77.0),
    ("excitation_coil", "mechanical_outer_diameter_mm", 95.0),
    ("housing", "mechanical_inner_diameter_mm", 176.0),
    ("yoke", "mechanical_outer_diameter_mm", 172.0),
))
def test_safe_radial_edits_preserve_axial_geometry_and_other_parts(
        recording_document, role, field, value):
    before = deepcopy(recording_document)
    _part(recording_document, role)[field] = value
    validate_document(recording_document)
    _part(before, role)[field] = value
    assert recording_document == before


@pytest.mark.parametrize("role", ("housing", "yoke", "excitation_coil"))
@pytest.mark.parametrize("field", (
    "length_mm", "mechanical_inner_diameter_mm",
    "mechanical_outer_diameter_mm", "vacuum_inner_diameter_mm",
))
@pytest.mark.parametrize("value", (float("nan"), float("inf")))
def test_layer_dimensions_must_be_finite(recording_document, role, field, value):
    _part(recording_document, role)[field] = value
    with pytest.raises(ValueError):
        validate_document(recording_document)


@pytest.mark.parametrize("role", ("housing", "yoke", "excitation_coil"))
@pytest.mark.parametrize("inner,outer", ((185.0, 180.0), (180.0, 180.0), (-1.0, 180.0)))
def test_all_layers_require_ordered_positive_wall_dimensions(recording_document, role, inner, outer):
    _part(recording_document, role).update(
        mechanical_inner_diameter_mm=inner, mechanical_outer_diameter_mm=outer)
    with pytest.raises(ValueError, match="mechanical diameters require 0 <= ID < OD"):
        validate_document(recording_document)


@pytest.mark.parametrize("role", ("housing", "yoke", "excitation_coil"))
def test_mechanical_id_must_clear_vacuum_bore(recording_document, role):
    _part(recording_document, role)["mechanical_inner_diameter_mm"] = 19.0
    with pytest.raises(ValueError, match="mechanical ID 19 mm must clear.*vacuum ID 20 mm"):
        validate_document(recording_document)


@pytest.mark.parametrize("role", ("housing", "yoke", "excitation_coil"))
def test_vacuum_mismatch_points_to_material_dimension_editor(recording_document, role):
    _part(recording_document, role)["vacuum_inner_diameter_mm"] = 25.0
    with pytest.raises(ValueError, match="vacuum ID 25 mm does not match projector stack 20 mm") as caught:
        validate_document(recording_document)
    message = str(caught.value)
    assert "vacuum ID is the beam passage, not material thickness" in message
    assert "Edit dimensions" in message
    assert "mechanical_inner_diameter_mm / mechanical_outer_diameter_mm" in message


@pytest.mark.parametrize("role", ("housing", "yoke", "excitation_coil"))
@pytest.mark.parametrize("material", (None, "", " "))
def test_layers_require_material_class_without_assigning_a_material(recording_document, role, material):
    _part(recording_document, role)["material_class"] = material
    with pytest.raises(ValueError, match="material_class must be a non-empty"):
        validate_document(recording_document)


@pytest.mark.parametrize("role,outer,neighbour", (
    ("excitation_coil", 110.0, "yoke"),
    ("yoke", 178.0, "housing"),
))
def test_overlapping_radial_layers_report_both_parts(recording_document, role, outer, neighbour):
    _part(recording_document, role)["mechanical_outer_diameter_mm"] = outer
    with pytest.raises(ValueError, match="Mechanical radial layers overlap") as caught:
        validate_document(recording_document)
    assert "intermediate_lens_" + role in str(caught.value)
    assert "intermediate_lens_" + neighbour in str(caught.value)
    assert "module Z" in str(caught.value)


@pytest.mark.parametrize("delta", (0.0, 0.5e-9))
def test_radial_contact_and_numerical_tolerance_are_allowed(recording_document, delta):
    coil, yoke = (_part(recording_document, role) for role in ("excitation_coil", "yoke"))
    coil["mechanical_outer_diameter_mm"] = yoke["mechanical_inner_diameter_mm"] + delta
    validate_document(recording_document)


def test_profiled_yoke_is_not_filled_to_its_cylindrical_envelope(recording_document):
    yoke = _part(recording_document, "yoke")
    # The envelope reaches the housing, but the actual 85 mm outer radius does not.
    yoke["mechanical_outer_diameter_mm"] = 178.0
    yoke["magnetic_radial_profile_mm"] = [[0.0, 50.0, 85.0], [yoke["length_mm"], 50.0, 85.0]]
    validate_document(recording_document)


def _tiny_layers():
    """Synthetic millimetre fixture, independent of instrument design recipes."""
    parent = dict(key="lens", mechanical_profile="magnetic_lens_assembly", local_start_z_mm=0.0)
    def layer(key, role, start, end, inner, outer):
        return dict(key=key, parent_key="lens", mechanical_profile=role,
                    local_start_z_mm=start, local_end_z_mm=end,
                    local_center_z_mm=(start + end) / 2, length_mm=end - start,
                    mechanical_inner_diameter_mm=inner, mechanical_outer_diameter_mm=outer,
                    vacuum_inner_diameter_mm=10.0, material_class="test_material")
    return [parent, layer("coil", "magnetic_excitation_coil", 0, 3, 20, 30),
            layer("yoke", "magnetic_lens_yoke", 3, 10, 25, 35)]


@pytest.mark.parametrize("start", (3.0, 3.0 - 0.5e-9, 4.0))
def test_radially_overlapping_layers_may_touch_or_separate_axially(start):
    parts = _tiny_layers()
    parts[2].update(local_start_z_mm=start, length_mm=10.0 - start)
    _validate_simple_magnetic_layer_geometry(parts)


def test_only_layers_with_same_optical_owner_are_compared():
    parts = _tiny_layers()
    parts.append(dict(key="nested", parent_key="lens", mechanical_profile="magnetic_lens_assembly",
                      local_start_z_mm=0.0))
    parts[2].update(parent_key="nested", local_start_z_mm=0.0, length_mm=10.0)
    _validate_simple_magnetic_layer_geometry(parts)


def test_shared_circuit_is_not_reduced_to_independent_cylindrical_layers():
    parts = _tiny_layers()
    parts[0]["magnetic_circuit_topology"] = "shared_pole_multi_gap"
    parts[2].update(local_start_z_mm=0.0, length_mm=10.0)
    _validate_simple_magnetic_layer_geometry(parts)


def test_objective_split_material_intervals_leave_the_central_gap_empty():
    parts = _tiny_layers()
    parts[0].update(upper_yoke_start_local_z_mm=0.0, upper_yoke_end_local_z_mm=2.0,
                    lower_yoke_start_local_z_mm=8.0, lower_yoke_end_local_z_mm=10.0)
    parts[1].update(mechanical_profile="magnetic_lens_housing", local_start_z_mm=4.0,
                    local_center_z_mm=5.0, local_end_z_mm=6.0, length_mm=2.0)
    parts[2].update(local_start_z_mm=0.0, length_mm=10.0)
    _validate_simple_magnetic_layer_geometry(parts)


def test_legacy_coil_thickness_recipe_remains_enforced():
    document = tomllib.loads((CONFIG_ROOT / "column" / "C3.toml").read_text(encoding="utf-8"))
    coil = next(part for part in document["parts"] if part["key"] == "condenser_lens_3_excitation_coil")
    coil["mechanical_inner_diameter_mm"] = 97.0
    with pytest.raises(ValueError, match="radial thickness must follow"):
        validate_document(document)
