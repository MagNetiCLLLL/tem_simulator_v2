"""A dimension's physical meaning must not be confused with its evidence."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path
import tomllib

import pytest

from temsim.parameter_semantics import CATEGORY_LABELS, SOURCE_LABELS, describe_parameter


CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "instruments"


def _document(path):
    return tomllib.loads((CONFIGS / path).read_text(encoding="utf-8-sig"))


def _parts(path):
    return {part["key"]: part for part in _document(path)["parts"]}


@pytest.mark.parametrize("field,category,label", [
    ("length_mm", "envelope", "Mechanism envelope length"),
    ("mechanical_outer_diameter_mm", "envelope", "Mechanism envelope diameter"),
    ("mechanical_bore_diameter_mm", "physical", "Carrier bore diameter"),
    ("plate_thickness_mm", "physical", "Aperture plate thickness"),
    ("maximum_radius_mm", "operating", "Maximum working opening radius"),
    ("vacuum_inner_diameter_mm", "vacuum", "Beam passage (vacuum)"),
])
def test_actual_c2_aperture_distinguishes_plate_carrier_envelope_and_operating_limit(field, category, label):
    part = _parts("column/C2.toml")["condenser_aperture_2"]
    assert part["length_mm"] == 20 and part["mechanical_outer_diameter_mm"] == 80
    assert part["mechanical_bore_diameter_mm"] == 4 and part["plate_thickness_mm"] == .2
    meaning = describe_parameter(part, ("parts", part["key"], field))
    assert (meaning.category, meaning.label, meaning.unit) == (category, label, "mm")
    assert meaning.source_kind == "unspecified"
    assert "not calibrated dimensions" in meaning.source_note


@pytest.mark.parametrize("field", ["length_mm", "mechanical_inner_diameter_mm", "mechanical_outer_diameter_mm"])
def test_actual_il_coil_dimensions_are_physical_without_invented_measurement(field):
    part = _parts("project_and_recording_system/EnergyFilter.toml")["intermediate_lens_excitation_coil"]
    assert part["length_mm"] == 180  # Preserve the user's current configuration.
    meaning = describe_parameter(part, ("parts", part["key"], field))
    assert meaning.category == "physical"
    assert meaning.source_kind == "unspecified"
    assert meaning.description and meaning.source_note


def test_split_coil_length_and_parent_material_interval_have_different_meanings():
    parts = _parts("column/C2.toml")
    coil = parts["objective_lens_excitation_coil"]
    assert describe_parameter(coil, ("parts", coil["key"], "length_mm"), by_key=parts).category == "envelope"
    for field in ("upper_yoke_start_local_z_mm", "lower_yoke_end_local_z_mm", "mechanical_coil_axial_inset_mm"):
        meaning = describe_parameter(coil, ("parts", "objective_lens", field), by_key=parts)
        assert meaning.category == "physical"
        assert meaning.source_kind == "unspecified"  # Parent photographs are not field measurements.
    assert describe_parameter(parts["objective_lens_housing"], ("parts", "objective_lens_housing", "length_mm"), by_key=parts).category == "physical"


def test_explicit_material_intervals_preserve_split_envelope_meaning_for_generic_owner():
    part = {"key": "coil", "mechanical_profile": "magnetic_excitation_coil", "material_intervals_mm": [[0, 2], [5, 7]]}
    assert describe_parameter(part, ("parts", "coil", "length_mm")).category == "envelope"
    assert describe_parameter(part, ("parts", "coil", "material_intervals_mm", 0, 1)).category == "physical"


def test_actual_pole_fillet_range_is_metadata_not_a_constructed_fillet():
    poles = [part for path in CONFIGS.rglob("*.toml")
             for part in tomllib.loads(path.read_text(encoding="utf-8-sig")).get("parts", ())
             if "pole_root_fillet_radius_range_mm" in part]
    assert poles
    for part in poles:
        meaning = describe_parameter(part, ("parts", part["key"], "pole_root_fillet_radius_range_mm", 0))
        assert meaning.category == "unknown"
        assert "sharp shoulders" in meaning.description
        assert "does not generate a fillet" in meaning.description


def test_pole_connector_material_is_not_the_inner_vacuum_passage():
    part = _parts("column/C2.toml")["objective_upper_pole"]
    for field in ("pole_vacuum_connector_outer_diameter_mm", "pole_vacuum_connector_axial_length_mm"):
        assert describe_parameter(part, ("parts", part["key"], field)).category == "physical"
    assert describe_parameter(part, ("parts", part["key"], "vacuum_inner_diameter_mm")).category == "vacuum"


def test_unknown_profiles_keep_unknown_internal_geometry_and_explicit_envelopes():
    part = {"key": "exotic", "mechanical_profile": "unimplemented_mechanism"}
    assert describe_parameter(part, ("parts", "exotic", "length_mm")).category == "envelope"
    assert describe_parameter(part, ("parts", "exotic", "mechanical_outer_diameter_mm")).category == "envelope"
    meaning = describe_parameter(part, ("parts", "exotic", "unknown_internal_radius_mm"))
    assert meaning.category == "unknown" and meaning.source_kind == "unspecified"


def test_runtime_cad_and_derived_values_do_not_claim_independent_measurements():
    part = {"key": "coil", "mechanical_profile": "magnetic_excitation_coil"}
    runtime = describe_parameter(part, ("runtime", "coil", "radius_mm"))
    assert (runtime.category, runtime.source_kind) == ("operating", "runtime")
    cad = describe_parameter(part, ("parts", "coil", "model_3d", "features", 1, "center_mm", 0))
    assert (cad.category, cad.source_kind, cad.unit) == ("cad", "user_defined", "mm")
    assert describe_parameter(part, ("parts", "coil", "model_3d", "transform", "scale_xy", 1)).unit == "×"
    derived = describe_parameter(part, ("derived", "coil", "radial_thickness_mm"))
    assert (derived.label, derived.category, derived.source_kind) == ("Radial thickness", "physical", "unspecified")
    assert "/ 2" in derived.description and "no independent thickness" in derived.source_note


@pytest.mark.parametrize("path,label,unit,category", [
    (("geometry", "length_mm"), "Length", "mm", "envelope"),
    (("ports", "entrance", "local_z_mm"), "Local z", "mm", "placement"),
    (("geometry", "vacuum_drift_inner_diameter_mm"), "Vacuum drift inner diameter", "mm", "vacuum"),
    (("geometry", "unknown_angles_deg", 0), "Unknown angles [0]", "deg", "unknown"),
])
def test_module_document_paths_keep_real_field_labels_and_units(path, label, unit, category):
    meaning = describe_parameter({}, path)
    assert (meaning.label, meaning.unit, meaning.category) == (label, unit, category)


@pytest.mark.parametrize("kind", list(SOURCE_LABELS))
def test_explicit_field_metadata_reports_declared_source_without_changing_category(kind):
    part = {"key": "c", "mechanical_profile": "magnetic_excitation_coil",
            "parameter_metadata": {"length_mm": {"source_kind": kind, "source_note": "Length recorded in fixture 12, dimension L."}}}
    meaning = describe_parameter(part, ("parts", "c", "length_mm"))
    assert meaning.source_kind == kind and meaning.category == "physical"
    assert meaning.source_label == SOURCE_LABELS[kind]


def test_specific_array_metadata_overrides_field_metadata_without_mutation():
    part = {"key": "c", "mechanical_profile": "magnetic_excitation_coil",
            "parameter_metadata": {
                "magnetic_radial_profile_mm": {"source_kind": "estimated", "source_note": "Profile reconstruction."},
                "magnetic_radial_profile_mm.1.2": {"source_kind": "measured", "source_note": "Radius measured using calibrated tool T on 2026-01-01."}}}
    before = deepcopy(part)
    path = ("parts", "c", "magnetic_radial_profile_mm", 1, 2)
    meaning = describe_parameter(part, path)
    assert meaning.source_kind == "measured" and meaning.label.endswith("[1] [2]")
    assert describe_parameter(part, (*path[:-1], 1)).source_kind == "estimated"
    assert part == before
    with pytest.raises(FrozenInstanceError):
        meaning.category = "unknown"


@pytest.mark.parametrize("metadata", [
    {"source_kind": "measured"}, {"source_kind": "documented", "source_note": " "},
    {"source_kind": "invented", "source_note": "A claim."}, {"source_kind": "estimated", "source_note": 123},
    {"source_kind": ["measured"], "source_note": "Malformed metadata must remain reviewable."},
])
def test_incomplete_field_source_declarations_stay_unspecified(metadata):
    part = {"parameter_metadata": {"length_mm": metadata}}
    assert describe_parameter(part, ("parts", "a", "length_mm")).source_kind == "unspecified"


@pytest.mark.parametrize("note", ["User photo, topology only.", "Uncalibrated image", "not_dimensionally_calibrated",
                                  "Photo only", "Not measured", "Undocumented estimate"])
def test_photo_or_negated_measurement_metadata_cannot_establish_measured_dimensions(note):
    part = {"parameter_metadata": {"length_mm": {"source_kind": "measured", "source_note": note}}}
    assert describe_parameter(part, ("parts", "a", "length_mm")).source_kind == "unspecified"


@pytest.mark.parametrize("status,expected", [("measured", "measured"), ("documented", "documented"),
                                            ("not_measured", "unspecified"), ("unmeasured", "unspecified"),
                                            ("engineering_reconstruction_not_oem", "estimated")])
def test_field_specific_status_can_support_a_dimension_but_negated_measurement_cannot(status, expected):
    part = {"mechanical_profile": "magnetic_excitation_coil", "length_status": status,
            "length_source": "Fixture or drawing dimension L; see the recorded field note."}
    assert describe_parameter(part, ("parts", "a", "length_mm")).source_kind == expected


def test_blanket_geometry_measurement_does_not_certify_each_dimension():
    part = {"mechanical_profile": "magnetic_excitation_coil", "mechanical_geometry_status": "measured",
            "mechanical_geometry_source": "Photograph and a measurement somewhere on the assembly."}
    meaning = describe_parameter(part, ("parts", "a", "length_mm"))
    assert meaning.source_kind == "unspecified"
    part["mechanical_geometry_status"] = "provisional_parameterized_non_oem"
    assert describe_parameter(part, ("parts", "a", "length_mm")).source_kind == "estimated"


def test_semantic_enums_are_stable_for_gui_consumers():
    assert set(CATEGORY_LABELS) == {"physical", "envelope", "vacuum", "operating", "placement", "material", "cad", "unknown"}
    assert set(SOURCE_LABELS) == {"measured", "documented", "estimated", "unspecified", "runtime", "user_defined"}
