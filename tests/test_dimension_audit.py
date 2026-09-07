"""Saved-catalog audits retain source locations without rewriting geometry."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from temsim.dimension_audit import audit_catalog, audit_document, audit_documents, main


ROOT = Path(__file__).resolve().parents[1]


def _part(**updates):
    return {"key": "coil", "name": "Test coil", "mechanical_profile": "magnetic_excitation_coil",
            "length_mm": 10., "mechanical_inner_diameter_mm": 4., "mechanical_outer_diameter_mm": 8., **updates}


def test_array_inventory_uses_true_paths_and_skips_booleans_and_evidence_numbers():
    document = {"geometry": {"length_mm": 40}, "ports": {"entrance": {"local_z_mm": 0}},
                "parts": [_part(magnetic_radial_profile_mm=[[0, 2, 4], [10, 2, 4]],
                                arbitrary_bool_mm=True, order=7,
                                parameter_metadata={"length_mm": {"source_kind": "measured", "source_note": "Fixture L", "source_mm": 99}})]}
    before = deepcopy(document)
    audit = audit_document(document, source="fixture.toml")
    rows = {row.path: row for row in audit.records}
    assert len(rows) == 11
    assert rows[("parts", "coil", "magnetic_radial_profile_mm", 1, 2)].value == 4
    assert rows[("parts", "coil", "length_mm")].semantics.source_kind == "measured"
    assert rows[("ports", "entrance", "local_z_mm")].semantics.category == "placement"
    assert not any("parameter_metadata" in path or "arbitrary_bool_mm" in path or "order" in path for path in rows)
    assert all(row.source_file == "fixture.toml" for row in rows.values())
    assert document == before


def test_saved_strip_audit_does_not_substitute_carrier_or_limit_for_working_opening():
    part = _part(aperture_plate_form="perforated_strip", plate_thickness_mm=.2,
                 mechanical_bore_diameter_mm=4, maximum_radius_mm=2)
    audit = audit_document({"parts": [part]})
    assert {item.code for item in audit.issues} == {"undefined_strip_outline", "undefined_working_opening"}
    assert next(item for item in audit.issues if item.code == "undefined_working_opening").path == ("parts", "coil", "radius_mm")
    part["aperture_hole_diameters_um"] = [50, 100]
    audit = audit_document({"parts": [part]})
    assert not any(item.code == "undefined_working_opening" for item in audit.issues)
    row = next(row for row in audit.records if row.path[-2:] == ("aperture_hole_diameters_um", 1))
    assert row.value == 100 and row.semantics.category == "physical" and row.semantics.unit == "µm"


def test_missing_strip_thickness_reports_real_missing_field_not_envelope_length():
    part = _part(aperture_plate_form="perforated_strip")
    audit = audit_document({"parts": [part]})
    assert any(item.code == "undefined_plate_thickness" and item.path[-1] == "plate_thickness_mm" for item in audit.issues)
    part["active_length_mm"] = .3
    assert not any(item.code == "undefined_plate_thickness" for item in audit_document({"parts": [part]}).issues)


def test_supported_radial_profile_does_not_get_misreported_as_unsupported_mechanism():
    part = _part(mechanical_profile="custom", magnetic_radial_profile_mm=[[0, 2, 4], [10, 2, 4]])
    assert not audit_document({"parts": [part]}).issues


def test_fillet_metadata_and_unsupported_solid_are_explicit_review_items():
    part = _part(mechanical_profile="magnetic_pole_piece", pole_piece_geometry_style="unsupported",
                 pole_root_fillet_radius_range_mm=[.1, .2])
    audit = audit_document({"parts": [part]})
    assert {issue.code for issue in audit.issues} == {"unimplemented_geometry_parameter", "envelope_only_geometry"}
    assert all(issue.source_file == "<memory>" and issue.part_key == "coil" for issue in audit.issues)


def test_cad_schema_issues_and_nested_values_are_retained_without_building_geometry(monkeypatch):
    import temsim.part_model_features as features
    monkeypatch.setattr(features, "apply_model_features", lambda *_args, **_kwargs: pytest.fail("Audit must not build meshes"))
    model = features.default_model_3d({})
    model["transform"]["scale_xy"] = [1.2, 1.0]
    part = _part(model_3d=model)
    audit = audit_document({"parts": [part]})
    rows = [row for row in audit.records if "model_3d" in row.path]
    assert len(rows) == 8 and all(row.semantics.category == "cad" for row in rows)
    assert all(row.semantics.source_kind == "user_defined" for row in rows)
    assert {issue.code for issue in audit.issues} == {"cad_physics_independent"}
    model["base"]["kind"] = "unsupported"
    assert "invalid_cad_configuration" in {issue.code for issue in audit_document({"parts": [part]}).issues}


@pytest.mark.parametrize("number", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_dimensions_remain_traceable_in_standards_compliant_json(number):
    audit = audit_document({"parts": [_part(length_mm=number)]}, source="invalid.toml")
    data = json.loads(audit.to_json(), parse_constant=lambda value: pytest.fail(f"Nonstandard JSON constant: {value}"))
    record = next(row for row in data["records"] if row["path"][-1] == "length_mm")
    assert record["value"] == str(number)
    assert any(issue["code"] == "nonfinite_dimension" for issue in data["issues"])


def test_combining_documents_preserves_duplicate_alternative_part_definitions():
    docs = {"one.toml": {"parts": [_part()]}, "two.toml": {"parts": [_part(length_mm=20)]}}
    mapping = audit_documents(docs)
    iterable = audit_documents(docs.items())
    assert mapping.to_dict() == iterable.to_dict()
    assert mapping.summary["module_count"] == 2 and mapping.summary["part_count"] == 2
    assert {row.source_file for row in mapping.records} == set(docs)
    assert [row.value for row in mapping.records if row.path[-1] == "length_mm"] == [10, 20]


def test_actual_catalog_covers_all_modules_parts_and_arrays_without_changing_files():
    paths = sorted((ROOT / "configs" / "instruments").rglob("*.toml"))
    before = {path: path.read_bytes() for path in paths}
    audit = audit_catalog(ROOT)
    assert (audit.module_count, audit.part_count) == (11, 482)
    assert audit.summary["parameter_count"] > 4000
    assert len({row.source_file for row in audit.records}) == 11
    assert all(Path(row.source_file).is_absolute() for row in audit.records)
    assert all(row.semantics.label and row.semantics.description and row.semantics.source_note for row in audit.records)
    assert not {"measured", "documented"}.intersection(audit.summary["by_source"])
    coil = next(row for row in audit.records if row.part_key == "intermediate_lens_excitation_coil" and row.path[-1] == "length_mm" and row.source_file.endswith("EnergyFilter.toml"))
    assert coil.value == 180 and coil.semantics.category == "physical"
    assert audit.summary["issues_by_code"]["undefined_strip_outline"] == 23
    assert len(json.loads(audit.to_json())["records"]) == len(audit.records)
    markdown = audit.to_markdown()
    assert "482 component definitions" in markdown and "Evidence gaps by component" in markdown
    assert len(markdown.splitlines()) < len(audit.records) // 2  # Grouped report, not repeated thousands of values.
    assert before == {path: path.read_bytes() for path in paths}


def test_catalog_accepts_project_configs_or_instruments_root():
    counts = [audit_catalog(root).summary["parameter_count"] for root in (
        ROOT, ROOT / "configs", ROOT / "configs" / "instruments")]
    assert len(set(counts)) == 1


def test_cli_writes_only_explicit_report_outputs_and_does_not_edit_config(tmp_path):
    configs = tmp_path / "configs" / "instruments"
    configs.mkdir(parents=True)
    source = configs / "module.toml"
    source.write_text('[[parts]]\nkey="sample"\nlength_mm=3\nmechanical_outer_diameter_mm=10\n', encoding="utf-8")
    (configs / "catalog.toml").write_text('version=1\n', encoding="utf-8")
    original = source.read_bytes()
    markdown, data = tmp_path / "reports" / "audit.md", tmp_path / "reports" / "audit.json"
    assert main(["--root", str(tmp_path), "--markdown", str(markdown), "--json", str(data)]) == 0
    assert json.loads(data.read_text(encoding="utf-8"))["summary"]["part_count"] == 1
    assert "sample" in markdown.read_text(encoding="utf-8")
    assert source.read_bytes() == original


def test_audit_and_semantics_can_be_used_without_loading_qt():
    code = "from temsim.dimension_audit import audit_catalog; import sys; audit_catalog('.'); assert not any(n.startswith(('PySide6', 'PyQt')) for n in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
