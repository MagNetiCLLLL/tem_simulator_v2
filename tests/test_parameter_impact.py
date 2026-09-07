"""Read-only parameter consumers, checked against real saved assembly roles."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tomllib

import pytest

from temsim.magnetic_materials import lens_material_defaults
from temsim.parameter_impact import describe_parameter_impact, component_impact_summary
from temsim.part_materials import part_material_updates


ROOT = Path(__file__).resolve().parents[1] / "configs/instruments"


@pytest.fixture
def parts():
    documents = [tomllib.loads((ROOT / path).read_text(encoding="utf-8-sig")) for path in (
        "column/C3_ProbeCorrector_ImageCorrector.toml", "project_and_recording_system/EnergyFilter.toml")]
    return {row["key"]: row for document in documents for row in document["parts"]}


def _linear(**updates):
    return dict(solver="axisymmetric_linear_fem", relative_permeability=1000.0, ampere_turns=100.0,
                radial_nodes=16, axial_nodes=24, **updates)


def _nonlinear():
    return dict(solver="axisymmetric_nonlinear_fem", ampere_turns=100.0,
                bh_material=lens_material_defaults()["bh_material"], radial_nodes=16, axial_nodes=24)


def _impact(parts, key, field, mode="ideal", descriptors=None):
    return describe_parameter_impact(parts[key], ("parts", key, field), by_key=parts,
                                     simulation_mode=mode, descriptors=descriptors)


def test_c2_plate_envelope_carrier_and_operating_opening_are_different_routes(parts):
    key = "condenser_aperture_2"
    for field in ("mechanical_outer_diameter_mm", "mechanical_bore_diameter_mm", "plate_thickness_mm"):
        result = _impact(parts, key, field)
        assert result.status == "inactive"
        assert not {"magnetic_field", "beam_clearance"}.intersection(result.effects)
    length = _impact(parts, key, "length_mm")
    assert length.status == "active"
    assert "beam_clearance" in length.effects
    assert "not its plate thickness" in length.detail
    assert "narrower bore" in length.detail
    runtime = describe_parameter_impact(parts[key], ("runtime", key, "diameter_mm"), by_key=parts)
    assert runtime.active and "beam_clearance" in runtime.effects
    assert "Ray trajectories" in runtime.affected_results
    assert "separate" in runtime.detail
    limit = _impact(parts, key, "maximum_radius_mm")
    assert limit.affected_results == ("Input validation",)
    assert "beam_clearance" not in limit.effects


@pytest.mark.parametrize("mode", ["ideal", "analytical"])
@pytest.mark.parametrize("field", ["length_mm", "mechanical_inner_diameter_mm", "mechanical_outer_diameter_mm"])
def test_original_il_dimensions_do_not_rebuild_ideal_or_analytic_field(parts, mode, field):
    result = _impact(parts, "intermediate_lens_excitation_coil", field, mode,
                     {"intermediate_lens": _linear()})
    if field == "length_mm":
        assert result.active and "beam_clearance" in result.effects
        assert result.label == "Vacuum active; magnetic geometry inactive"
    else:
        assert result.status == "inactive" and result.active is False
    assert "magnetic_field" not in result.effects
    assert "focusing and magnetic rotation remain" in result.detail


@pytest.mark.parametrize("mode,recipe", [("linear_geometry", _linear), ("nonlinear_material", _nonlinear)])
@pytest.mark.parametrize("field", ["length_mm", "mechanical_inner_diameter_mm", "mechanical_outer_diameter_mm"])
def test_original_il_dimensions_enter_configured_fem(parts, mode, recipe, field):
    result = _impact(parts, "intermediate_lens_excitation_coil", field, mode,
                     {"intermediate_lens": recipe()})
    assert result.active and result.status == "active"
    assert "magnetic_field" in result.effects
    assert "intermediate_lens" in result.detail
    assert "independent prescribed inputs" in result.detail
    if field == "length_mm":
        assert "beam_clearance" in result.effects


@pytest.mark.parametrize("mode", ["linear_geometry", "nonlinear_material"])
def test_required_missing_or_invalid_recipe_is_not_reported_as_active(parts, mode):
    key = "intermediate_lens_excitation_coil"
    result = _impact(parts, key, "mechanical_outer_diameter_mm", mode)
    assert result.status == "configuration_required" and not result.active
    recipe = _linear() if mode == "linear_geometry" else _nonlinear()
    recipe["ampere_turns"] = float("nan")
    invalid = _impact(parts, key, "length_mm", mode, {"intermediate_lens": recipe})
    assert invalid.status == "configuration_required"
    assert "magnetic_field" not in invalid.effects
    assert "beam_clearance" in invalid.effects
    assert "beam-clearance route is active" in invalid.detail


def test_isolated_coil_extent_changes_the_real_vacuum_profile_without_magnetic_fem():
    from temsim.column.module_assembly import _module_vacuum_segments

    coil = dict(key="coil", name="Isolated winding", parent_key="lens", field_source_key="lens",
                mechanical_profile="magnetic_excitation_coil", vacuum_inner_diameter_mm=4,
                local_start_z_mm=10, local_center_z_mm=15, local_end_z_mm=20, length_mm=10,
                mechanical_inner_diameter_mm=6, mechanical_outer_diameter_mm=8)
    module = SimpleNamespace(key="test", entrance_z_mm=0, exit_z_mm=30,
                             geometry={"vacuum_drift_inner_diameter_mm": 10})

    def bore_at(z, row):
        part = SimpleNamespace(key=row["key"], name=row["name"], module_key="test", data=row,
                    start_z_mm=row["local_start_z_mm"], end_z_mm=row["local_end_z_mm"], length_mm=row["length_mm"])
        return next(segment.inner_diameter_mm for segment in _module_vacuum_segments(module, 0, [part])
                    if segment.start_z_mm <= z < segment.end_z_mm)

    longer = dict(coil, local_end_z_mm=25, length_mm=15)
    assert bore_at(22, coil) == 10 and bore_at(22, longer) == 4
    ideal = describe_parameter_impact(coil, ("parts", "coil", "length_mm"))
    assert ideal.active and "beam_clearance" in ideal.effects and "magnetic_field" not in ideal.effects
    configured = describe_parameter_impact(coil, ("parts", "coil", "local_end_z_mm"),
                    simulation_mode="linear_geometry", descriptors={"lens": _linear()})
    assert {"beam_clearance", "magnetic_field"}.issubset(configured.effects)
    missing = describe_parameter_impact(coil, ("parts", "coil", "length_mm"), simulation_mode="linear_geometry")
    assert missing.status == "configuration_required" and "beam_clearance" in missing.effects


def test_custom_distinguishes_analytic_generated_and_imported_routes(parts):
    key, field = "intermediate_lens_excitation_coil", "mechanical_inner_diameter_mm"
    analytic = _impact(parts, key, field, "custom")
    assert analytic.status == "inactive" and "analytic" in analytic.label.lower()
    generated = _impact(parts, key, field, "custom", {"intermediate_lens": _linear()})
    assert generated.active
    imported = _impact(parts, key, field, "custom", {"intermediate_lens": {
        "source_path": "not-read.npz", "geometry_fingerprint": "old", "map_type": "axisymmetric_rz"}})
    assert imported.status == "configuration_required"
    assert "field_map_identity" in imported.effects and "magnetic_field" not in imported.effects
    assert "not regenerated" in imported.detail and "fallback" in imported.detail


def test_cad_fields_are_ignored_by_all_scientific_routes_and_listed_in_summary(parts):
    key = "intermediate_lens_excitation_coil"
    part = deepcopy(parts[key])
    part["model_3d"] = dict(schema_version=1, base=dict(kind="box", length_mm=15),
                           transform=dict(scale_xy=[2, 1], rotation_deg=[0, 15, 0]),
                           features=[dict(id="hole1", kind="hole", diameter_mm=2, enabled=True)])
    for mode in ("ideal", "analytical", "linear_geometry", "nonlinear_material", "custom"):
        for suffix in (("base", "length_mm"), ("transform", "scale_xy", 0), ("features", 0, "diameter_mm")):
            result = describe_parameter_impact(part, ("parts", key, "model_3d", *suffix),
                        by_key=parts, simulation_mode=mode, descriptors={"intermediate_lens": _linear()})
            assert result.status == "unsupported" and result.active is False
            assert result.effects == ("display",) and result.affected_results == ("3D preview",)
            assert "not consumed" in result.detail
    summary = component_impact_summary(part, by_key=parts, simulation_mode="ideal")
    assert "model_3d.base.length_mm" in summary.ignored_cad_parameters
    assert "model_3d.transform.scale_xy[0]" in summary.ignored_cad_parameters
    assert "model_3d.features[0].diameter_mm" in summary.ignored_cad_parameters
    assert "model_3d.features[0].kind" in summary.ignored_cad_parameters
    assert "model_3d.features[0].enabled" in summary.ignored_cad_parameters
    assert "model_3d.schema_version" not in summary.ignored_cad_parameters
    assert "model_3d.features[0].id" not in summary.ignored_cad_parameters
    assert "Ignored by beam/magnetic solvers" in summary.detail


def test_split_coil_envelope_does_not_resize_the_parent_owned_winding(parts):
    key = "objective_lens_excitation_coil"
    descriptors = {"objective_lens": _linear()}
    # Context-only envelopes cannot alter either the actual source or wall.
    child = deepcopy(parts[key])
    child["axial_vacuum_context_only"] = True
    index = dict(parts, **{key: child})
    result = _impact(index, key, "length_mm", "linear_geometry", descriptors)
    assert result.status == "inactive"
    assert "parent yoke endpoints" in result.detail
    endpoint = describe_parameter_impact(child, ("parts", "objective_lens", "upper_yoke_start_local_z_mm"),
                    by_key=index, simulation_mode="linear_geometry", descriptors=descriptors)
    assert endpoint.active and "magnetic_field" in endpoint.effects
    inset = _impact(index, "objective_lens", "mechanical_coil_axial_inset_mm", "linear_geometry", descriptors)
    assert inset.active


def test_parent_summary_reports_displayed_descendant_and_shared_cad_only(parts):
    index = deepcopy(parts)
    parent = "intermediate_lens"
    coil = "intermediate_lens_excitation_coil"
    index[coil]["model_3d"] = {"transform": {"scale_xy": [1.2, 1.0]}}
    index["coil_detail"] = dict(key="coil_detail", parent_key=coil,
                                model_3d={"base": {"kind": "box", "length_mm": 3}})
    index["shared_body"] = dict(key="shared_body", magnetic_lens_keys=[parent],
                                model_3d={"features": [{"id": "hole1", "kind": "hole", "diameter_mm": 1}]})
    index["other_lens_shape"] = dict(key="other_lens_shape", parent_key="projector_lens_1",
                                     model_3d={"base": {"length_mm": 5}})
    before = deepcopy(index)
    summary = component_impact_summary(index[parent], by_key=index)
    assert f"parts.{coil}.model_3d.transform.scale_xy[0]" in summary.ignored_cad_parameters
    assert "parts.coil_detail.model_3d.base.length_mm" in summary.ignored_cad_parameters
    assert "parts.shared_body.model_3d.features[0].diameter_mm" in summary.ignored_cad_parameters
    assert not any("other_lens_shape" in item for item in summary.ignored_cad_parameters)
    assert all(path[1] == parent for path, _ in summary.parameter_impacts)
    assert index == before
    # A malformed ancestry cycle terminates; save validation owns its rejection.
    index[parent]["parent_key"] = "coil_detail"
    repeated = component_impact_summary(index[parent], by_key=index)
    assert repeated.ignored_cad_parameters == summary.ignored_cad_parameters


def test_explicit_source_shared_circuit_and_passive_references_select_the_right_recipe():
    a = dict(key="a", mechanical_profile="magnetic_lens_assembly", magnetic_circuit_id="shared")
    b = dict(key="b", mechanical_profile="magnetic_lens_assembly", magnetic_circuit_id="shared")
    coil = dict(key="winding", parent_key="a", field_source_key="b", mechanical_profile="magnetic_excitation_coil")
    by_key = {p["key"]: p for p in (a, b, coil)}
    routed = describe_parameter_impact(coil, ("parts", "winding", "mechanical_outer_diameter_mm"),
                 by_key=by_key, simulation_mode="custom", descriptors={"b": _linear()})
    assert routed.active and "route for b" in routed.detail
    assert "Analytic channels" in routed.detail
    passive = dict(key="shared_pole", mechanical_profile="magnetic_pole_piece", magnetic_lens_keys=["a", "b"])
    both = describe_parameter_impact(passive, ("parts", "shared_pole", "mechanical_bore_diameter_mm"),
                by_key=by_key, simulation_mode="linear_geometry", descriptors={"a": _linear(), "b": _linear()})
    assert both.active and "a, b" in both.detail
    lone = describe_parameter_impact(coil, ("parts", "winding", "mechanical_outer_diameter_mm"),
                simulation_mode="custom", descriptors={"b": _linear()})
    assert lone.active  # Explicit field_source_key works without a parent inventory.


def test_shared_operator_disagreement_and_missing_nonlinear_members_are_explained():
    parts = {key: dict(key=key, mechanical_profile="magnetic_lens_assembly", magnetic_circuit_id="shared") for key in ("a", "b")}
    coil = dict(key="coil", parent_key="a", mechanical_profile="magnetic_excitation_coil")
    parts["coil"] = coil
    changed = _linear()
    changed["relative_permeability"] = 500
    result = _impact(parts, "coil", "length_mm", "linear_geometry", {"a": _linear(), "b": changed})
    assert result.status == "configuration_required"
    bh = _impact(parts, "coil", "length_mm", "custom", {"a": _nonlinear()})
    assert bh.status == "configuration_required" and "Shared B-H" in bh.label


def test_material_assignments_participate_without_inventing_conductivity_or_changing_ni(parts):
    descriptors = {"intermediate_lens": _linear()}
    yoke = _impact(parts, "intermediate_lens_yoke", "material_regions", "linear_geometry", descriptors)
    assert yoke.active and "magnetic_field" in yoke.effects
    ideal = _impact(parts, "intermediate_lens_yoke", "material_regions", "ideal", descriptors)
    assert not ideal.active
    coil = _impact(parts, "intermediate_lens_excitation_coil", "material_regions", "linear_geometry", descriptors)
    assert coil.active and "magnetic_model" in coil.effects
    assert "same magnetic response" in coil.detail and "heating" in coil.detail
    aperture = _impact(parts, "condenser_aperture_2", "material_regions", "linear_geometry", descriptors)
    assert aperture.status == "inactive" and "no modeled magnetostatic solid" in aperture.detail
    assert "magnetic_field" not in aperture.effects
    metadata = describe_parameter_impact(parts["intermediate_lens_yoke"],
        ("parts", "intermediate_lens_yoke", "material_regions", "body", "label"),
        by_key=parts, simulation_mode="linear_geometry", descriptors=descriptors)
    assert not metadata.active and "field_map_identity" in metadata.effects


def test_housing_dimensions_enter_fem_only_with_a_magnetic_assignment(parts):
    key = "intermediate_lens_housing"
    descriptors = {"intermediate_lens": _linear()}
    assert _impact(parts, key, "mechanical_outer_diameter_mm", "linear_geometry", descriptors).status == "inactive"
    updated = deepcopy(parts[key])
    updated["material_regions"] = next(iter(part_material_updates(updated, "femm_pure_iron").values()))
    result = describe_parameter_impact(updated, ("parts", key, "mechanical_outer_diameter_mm"),
                                      by_key=parts, simulation_mode="linear_geometry", descriptors=descriptors)
    assert result.active  # Explicit current draft wins over a stale inventory row.


def test_vacuum_context_and_stop_specific_bores_do_not_get_conflated(parts):
    key = "condenser_aperture_2"
    assert "beam_clearance" in _impact(parts, key, "vacuum_inner_diameter_mm").effects
    context = deepcopy(parts[key])
    context["axial_vacuum_context_only"] = True
    result = describe_parameter_impact(context, ("parts", key, "vacuum_inner_diameter_mm"))
    assert result.status == "inactive" and "beam_clearance" not in result.effects
    dpa = _impact(parts, "projection_chamber_dpa_aperture", "mechanical_bore_diameter_mm")
    assert dpa.active and "beam_clearance" in dpa.effects
    fixed = describe_parameter_impact(parts["projection_chamber_dpa_aperture"],
                ("runtime", "projection_chamber_dpa_aperture", "enabled"))
    assert fixed.status == "inactive" and "cannot retract" in fixed.label


def test_derived_thickness_and_lens_operating_control_have_separate_meanings(parts):
    key = "intermediate_lens_excitation_coil"
    thickness = describe_parameter_impact(parts[key], ("derived", key, "radial_thickness_mm"), by_key=parts,
                simulation_mode="linear_geometry", descriptors={"intermediate_lens": _linear()})
    assert thickness.active and "chosen diameter boundary" in thickness.detail
    for mode in ("ideal", "analytical", "linear_geometry", "nonlinear_material", "custom"):
        descriptors = {"intermediate_lens": _nonlinear() if mode == "nonlinear_material" else _linear()}
        operating = describe_parameter_impact(parts["intermediate_lens"], ("runtime", "intermediate_lens", "percent"),
                                              by_key=parts, simulation_mode=mode, descriptors=descriptors)
        assert operating.active and "operating" in operating.effects
    incomplete = describe_parameter_impact(parts["intermediate_lens"], ("runtime", "intermediate_lens", "percent"),
                                           by_key=parts, simulation_mode="linear_geometry")
    assert incomplete.status == "configuration_required"


def test_unconnected_unknown_and_nonnumeric_rows_are_safe_and_read_only(parts):
    key = "intermediate_lens_excitation_coil"
    descriptors = {"intermediate_lens": _linear()}
    before_parts, before_descriptors = deepcopy(parts), deepcopy(descriptors)
    for field in ("length_mm", "material_class", "mechanical_profile", "aperture_plate_form", "unknown_field"):
        result = _impact(parts, key, field, None, descriptors)
        assert result.status == "unknown" and result.active is None
    for field in ("aperture_plate_form", "field_descriptor", "unknown_dimension_mm"):
        unknown = _impact(parts, "condenser_aperture_2", field)
        assert unknown.status == "unknown"
    no_parent = dict(parts[key])
    no_parent.pop("field_source_key", None)
    missing = describe_parameter_impact(no_parent, ("parts", key, "length_mm"), simulation_mode="linear_geometry")
    assert missing.status == "unknown"
    summary = component_impact_summary(parts[key], by_key=parts, simulation_mode=None, descriptors=descriptors)
    assert summary.status == "unknown" and not summary.active_effects
    assert "not connected" in summary.mode_label
    broken = deepcopy(parts[key])
    broken["material_regions"] = {"body": "incomplete"}
    assert describe_parameter_impact(broken, ("parts", key, "material_regions"), simulation_mode=None).status == "unknown"
    for endpoint in (float("inf"), float("nan"), "unfinished"):
        broken["local_end_z_mm"] = endpoint
        assert describe_parameter_impact(broken, ("parts", key, "vacuum_inner_diameter_mm")).status == "unknown"
    assert parts == before_parts and descriptors == before_descriptors
