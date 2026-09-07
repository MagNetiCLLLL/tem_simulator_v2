"""Read-only dimensional meanings/evidence audit of saved module documents.

The audit inventories numeric dimension fields, including array elements and
module placement fields. It does not certify measurements, run a solver, build
meshes, or infer dimensions from photographs. JSON retains the complete field
inventory; Markdown groups repeated evidence gaps so the report stays useful.
"""

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import argparse
import json
import math
from numbers import Real
from pathlib import Path
import re
import tomllib

from temsim.parameter_semantics import ParameterSemantics, describe_parameter


@dataclass(frozen=True, slots=True)
class DimensionRecord:
    source_file: str
    part_key: str
    part_name: str
    path: tuple[str | int, ...]
    value: int | float | str
    semantics: ParameterSemantics


@dataclass(frozen=True, slots=True)
class DimensionIssue:
    code: str
    source_file: str
    part_key: str
    path: tuple[str | int, ...]
    message: str


@dataclass(frozen=True, slots=True)
class DimensionAudit:
    module_count: int
    part_count: int
    records: tuple[DimensionRecord, ...]
    issues: tuple[DimensionIssue, ...]

    @property
    def summary(self):
        return {
            "module_count": self.module_count, "part_count": self.part_count,
            "parameter_count": len(self.records), "issue_count": len(self.issues),
            "by_category": dict(sorted(Counter(row.semantics.category for row in self.records).items())),
            "by_source": dict(sorted(Counter(row.semantics.source_kind for row in self.records).items())),
            "issues_by_code": dict(sorted(Counter(row.code for row in self.issues).items())),
        }

    def to_dict(self):
        return {"schema_version": 1, "scope": "Saved numeric dimensions; runtime state and unsaved drafts excluded",
                "summary": self.summary, "records": [asdict(row) for row in self.records],
                "issues": [asdict(row) for row in self.issues]}

    def to_json(self):
        """Standards-compliant JSON, including malformed nonfinite input as text."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n"

    def to_markdown(self):
        summary = self.summary
        lines = ["# Project dimension audit", "",
                 f"{self.module_count} module documents · {self.part_count} component definitions · "
                 f"{len(self.records)} numeric dimension values · {len(self.issues)} review items.", "",
                 "Scope: saved documents only. Repeated components in alternative modules are counted separately. "
                 "Runtime openings and unsaved drafts are excluded. A physical meaning does not establish a measured source; "
                 "declared provenance is reported without independent certification. Full values and exact array paths are in the JSON report.", "",
                 "## Classification", "", "| Meaning | Values |", "| --- | ---: |"]
        lines.extend(f"| {_md(key)} | {count} |" for key, count in summary["by_category"].items())
        lines += ["", "| Source | Values |", "| --- | ---: |"]
        lines.extend(f"| {_md(key)} | {count} |" for key, count in summary["by_source"].items())
        lines += ["", "## Review items", "", "| Issue | Count |", "| --- | ---: |"]
        lines.extend(f"| {_md(key)} | {count} |" for key, count in summary["issues_by_code"].items())
        if not self.issues:
            lines.append("| None | 0 |")
        # Group identical findings within one file. All keys and true paths are
        # retained, without repeating a paragraph or each numeric value.
        grouped = defaultdict(list)
        for row in self.issues:
            grouped[(row.source_file, row.code, row.message)].append(row)
        for (source, code, message), rows in sorted(grouped.items()):
            lines += ["", f"- **{_md(code)}** — `{_md(source)}`: {_md(message)}",
                      "  Affected: " + "; ".join(f"`{_md(_path_text(row.path))}`" for row in rows) + "."]
        lines += ["", "## Evidence gaps by component", "",
                  "The following grouped counts identify values with unspecified evidence or an unclassified meaning. "
                  "They are review needs, not a claim that the configured numbers are incorrect.", "",
                  "| File | Component | Unspecified source | Unknown meaning |", "| --- | --- | ---: | ---: |"]
        gaps = defaultdict(lambda: [0, 0])
        for row in self.records:
            identity = row.source_file, row.part_key or "(module)"
            gaps[identity][0] += row.semantics.source_kind == "unspecified"
            gaps[identity][1] += row.semantics.category == "unknown"
        lines.extend(f"| `{_md(source)}` | `{_md(key)}` | {counts[0]} | {counts[1]} |"
                     for (source, key), counts in sorted(gaps.items()) if any(counts))
        return "\n".join(lines) + "\n"


def _md(value):
    return str(value).replace("|", "\\|").replace("`", "'").replace("\n", " ")


def _path_text(path):
    return ".".join(str(item) for item in path)


_DIMENSION_SUFFIX = re.compile(r"_(?:mm|um|nm|m|deg|rad|mrad|px)$")
_SKIP_CONTAINERS = {"parameter_metadata", "provenance", "evidence", "source", "metadata"}
_ANNULAR = {"magnetic_excitation_coil", "magnetic_lens_housing", "magnetic_lens_yoke"}
_BODY_PROFILES = _ANNULAR | {"magnetic_pole_piece", "circular_aperture", "vacuum_liner", "electrostatic_bias_tube"}
_POLE_STYLES = {"", "embedded_hourglass_bore", "tapered_bore_pole", "objective_vertical_back_inserted_shank_tapered_nose"}
_OUTER_FIELDS = {"mechanical_outer_diameter_mm", "mechanical_outer_radius_mm", "outer_diameter_mm", "outer_width_mm"}
_INNER_FIELDS = {"mechanical_inner_diameter_mm", "mechanical_bore_diameter_mm", "mechanical_clear_bore_diameter_mm",
                 "mechanical_bore_radius_mm", "bore_diameter_mm", "clear_bore_diameter_mm", "inner_diameter_mm"}
_OPENING_FIELDS = {"radius_mm", "diameter_mm", "opening_diameter_mm", "aperture_radius_mm",
                   "aperture_hole_diameters_mm", "aperture_hole_diameters_um", "hole_diameters_mm", "hole_diameters_um"}


def _numeric_dimensions(value, path=(), *, selected=False):
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _SKIP_CONTAINERS or str(key).endswith(("_source", "_source_urls", "_status")):
                continue
            chosen = selected or bool(_DIMENSION_SUFFIX.search(str(key))) or key in {
                "model_3d", "percent", "relative_permeability"}
            # CAD version and indexes are identifiers, not editable dimensions.
            if selected and "model_3d" in path and key in {"version", "schema_version", "enabled"}:
                continue
            yield from _numeric_dimensions(item, (*path, key), selected=chosen)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _numeric_dimensions(item, (*path, index), selected=selected)
    elif selected and isinstance(value, Real) and not isinstance(value, bool):
        yield path, value


def _part_issues(part, source):
    key = str(part.get("key", ""))
    prefix = ("parts", key)
    def issue(code, field, message):
        return DimensionIssue(code, source, key, (*prefix, field), message)
    if part.get("aperture_plate_form") == "perforated_strip":
        yield issue("undefined_strip_outline", "aperture_plate_form",
                    "Transverse strip dimensions and hole positions are not specified; the outline is schematic, not measured CAD.")
        if not _OPENING_FIELDS.intersection(part):
            yield issue("undefined_working_opening", "radius_mm",
                        "The saved file does not define a working opening. A carrier bore or maximum operating radius is not the current opening; runtime state may provide it.")
        if not {"plate_thickness_mm", "active_length_mm"}.intersection(part):
            yield issue("undefined_plate_thickness", "plate_thickness_mm",
                        "No physical plate thickness or gun active-length fallback is defined; mechanism length must not substitute for it.")
        return
    profile = part.get("mechanical_profile", "")
    if "pole_root_fillet_radius_range_mm" in part:
        yield issue("unimplemented_geometry_parameter", "pole_root_fillet_radius_range_mm",
                    "The fillet range is reference metadata; the current solid retains sharp shoulders and does not construct a radius from this range.")
    if part.get("model_3d") is not None:
        from temsim.part_model_features import validate_model_3d
        try:
            validate_model_3d(part)
        except (TypeError, ValueError) as exc:
            yield issue("invalid_cad_configuration", "model_3d", f"CAD schema validation failed: {exc}")
        yield issue("cad_physics_independent", "model_3d",
                    "Explicit CAD bases, transforms and Boolean cuts define the visual solid. Existing optical/field parameter laws do not automatically derive a new physical response from this mesh.")
        return
    if "magnetic_radial_profile_mm" in part:
        return  # This renderer supports explicit radial material profiles.
    if part.get("length_mm") == 0:
        yield issue("zero_thickness_reference", "length_mm",
                    "This component is a zero-thickness reference/envelope plane. The preview does not invent a physical plate thickness.")
        return
    supported = profile in _BODY_PROFILES
    if profile == "magnetic_pole_piece" and part.get("pole_piece_geometry_style", "") not in _POLE_STYLES:
        supported = False
    if not _OUTER_FIELDS.intersection(part):
        yield issue("undefined_outer_envelope", "mechanical_outer_diameter_mm",
                    "No supported outer radius/diameter/width is declared, so the existing preview cannot establish an outer body envelope.")
    elif not supported:
        yield issue("envelope_only_geometry", "mechanical_profile",
                    "This mechanism/profile has an envelope preview; its internal or non-axisymmetric construction is not a fully specified CAD solid.")
    elif not _INNER_FIELDS.intersection(part):
        yield issue("undefined_material_bore", "mechanical_inner_diameter_mm",
                    "No supported material bore is declared. A vacuum clearance must not silently substitute for a physical hole.")


def audit_document(document: Mapping, *, source="<memory>") -> DimensionAudit:
    """Inventory one module; paths retain part keys and true array indexes."""
    if not isinstance(document, Mapping):
        raise TypeError("A module document must be a mapping")
    source = str(source)
    parts = document.get("parts", ())
    if not isinstance(parts, (list, tuple)) or any(not isinstance(part, Mapping) for part in parts):
        raise ValueError("Module parts must be a list of mappings")
    by_key = {str(part.get("key", "")): part for part in parts}
    records, issues = [], []
    def inventory(owner, value, path, part_key="", part_name=""):
        for field_path, number in _numeric_dimensions(value, path):
            if not math.isfinite(number):
                issues.append(DimensionIssue("nonfinite_dimension", source, part_key, field_path,
                                             "A numeric dimension is not finite and cannot define valid geometry."))
                number = str(number)
            records.append(DimensionRecord(source, part_key, part_name, field_path, number,
                                           describe_parameter(owner, field_path, by_key=by_key)))
    inventory(document, {key: value for key, value in document.items() if key != "parts"}, (),
              part_name=str(document.get("name", "Module geometry")))
    for part in parts:
        key = str(part.get("key", ""))
        inventory(part, part, ("parts", key), key, str(part.get("name", key)))
        issues.extend(_part_issues(part, source))
    return DimensionAudit(1, len(parts), tuple(records), tuple(issues))


def audit_documents(documents) -> DimensionAudit:
    """Combine a source->document mapping or an iterable of (source, document)."""
    items = documents.items() if isinstance(documents, Mapping) else documents
    audits = [audit_document(document, source=source) for source, document in items]
    return DimensionAudit(sum(a.module_count for a in audits), sum(a.part_count for a in audits),
                          tuple(row for audit in audits for row in audit.records),
                          tuple(row for audit in audits for row in audit.issues))


def audit_catalog(root) -> DimensionAudit:
    """Read every module with parts beneath a project/config/instrument root.

    Catalog-selection files and operating presets without parts are excluded.
    Absolute source paths make every report record directly navigable.
    """
    root = Path(root).resolve()
    candidates = (root / "configs" / "instruments", root / "instruments", root)
    folder = next((path for path in candidates if path.is_dir()), root)
    if not folder.is_dir():
        raise FileNotFoundError(f"Instrument catalog directory does not exist: {folder}")
    documents = []
    for path in sorted(folder.rglob("*.toml")):
        with path.open("rb") as stream:
            # The editor also accepts a UTF-8 BOM in user-edited manifests.
            document = tomllib.loads(stream.read().decode("utf-8-sig"))
        if document.get("parts"):
            documents.append((str(path.resolve()), document))
    return audit_documents(documents)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    audit = audit_catalog(args.root)
    for path, content in ((args.markdown, audit.to_markdown()), (args.json, audit.to_json())):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    if args.markdown is None and args.json is None:
        print(audit.to_markdown(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
