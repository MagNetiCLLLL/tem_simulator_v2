"""Conservative parameter meanings and declared provenance, without a GUI.

Categories describe what a value means, not whether it was measured. Evidence
about a component's existence or photograph topology does not establish any of
its dimensions. A stored source declaration is reported, not independently
certified by this module.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import re


CATEGORY_LABELS = {
    "physical": "Physical dimension", "envelope": "Envelope / allocation",
    "vacuum": "Vacuum passage", "operating": "Operating value / limit",
    "placement": "Position / reference", "material": "Material definition",
    "cad": "User CAD geometry", "unknown": "Meaning not classified",
}
SOURCE_LABELS = {
    "measured": "Measured (declared)", "documented": "Documented (declared)",
    "estimated": "Estimated / reconstructed", "unspecified": "Source unspecified",
    "runtime": "Current operating value", "user_defined": "User-defined",
}


@dataclass(frozen=True, slots=True)
class ParameterSemantics:
    label: str
    category: str
    category_label: str
    unit: str
    source_kind: str
    source_label: str
    source_note: str
    description: str


_ANNULAR = {"magnetic_excitation_coil", "magnetic_lens_housing", "magnetic_lens_yoke"}
_POLE_STYLES = {"", "embedded_hourglass_bore", "tapered_bore_pole",
                "objective_vertical_back_inserted_shank_tapered_nose"}
_ENVELOPE_FIELDS = {"length_mm", "mechanical_outer_diameter_mm", "mechanical_outer_radius_mm",
                    "outer_diameter_mm", "outer_width_mm", "housing_length_mm", "electrode_length_mm"}
_DIMENSION_WORDS = ("diameter", "radius", "length", "width", "height", "thickness", "gap", "inset", "angle")
_SPLIT_ENDPOINTS = {f"{side}_yoke_{edge}_local_z_mm" for side in ("upper", "lower") for edge in ("start", "end")}
_UNITS = ("mrad", "keV", "eV", "kV", "MHz", "kHz", "Hz", "nm", "um", "mm", "deg", "rad", "m", "s")
_FRIENDLY = {
    "mechanical_inner_diameter_mm": "Material inner diameter",
    "mechanical_outer_diameter_mm": "Material outer diameter",
    "mechanical_bore_diameter_mm": "Existing bore diameter",
    "vacuum_inner_diameter_mm": "Beam passage (vacuum)",
    "plate_thickness_mm": "Aperture plate thickness",
    "radial_thickness_mm": "Radial thickness",
    "length_mm": "Length", "percent": "Excitation strength",
}


def parameter_unit(path):
    """Unit of an actual field, including nested CAD vector/list elements."""
    words = [word for word in _relative_path(path) if isinstance(word, str)]
    if "scale_xy" in words:
        return "×"
    for word in reversed(words):
        for unit in _UNITS:
            if word.endswith("_" + unit):
                return "µm" if unit == "um" else unit
        if word == "percent":
            return "%"
    return ""


def _relative_path(path):
    return path[2:] if path and path[0] in {"parts", "runtime", "derived"} else path


def _label(field, path):
    label = _FRIENDLY.get(field)
    if label is None:
        label = re.sub(r"_(?:mrad|nm|um|mm|deg|rad|m|s)$", "", str(field)).replace("_", " ").capitalize()
    for index in path[3:]:
        if isinstance(index, int) and not isinstance(index, bool):
            label += f" [{index}]"
    return label


def _meaning(part, path, by_key):
    relative = _relative_path(path)
    field = str(relative[0]) if relative else ""
    label = _label(field, ("parts", "", *relative))
    profile = part.get("mechanical_profile", "")
    if path and path[0] not in {"parts", "runtime", "derived"}:
        last_field = next((item for item in reversed(path) if isinstance(item, str)), "Parameter")
        label = _label(last_field, ("parts", "", last_field, *[item for item in path if isinstance(item, int)]))
        if path[0] == "geometry" and last_field in _ENVELOPE_FIELDS:
            return label, "envelope", "Module-level allocation or outer envelope, not the dimensions of its individual material parts."
        if path[0] == "geometry" and "vacuum" in last_field:
            return label, "vacuum", "Module-level vacuum tube/passage constraint, separate from an individual part's material body or working aperture."
        if path[0] == "ports" and last_field.endswith("_z_mm"):
            return label, "placement", "Module-local entrance/exit connection reference coordinate."
        return label, "unknown", "Module/document parameter; its material meaning is not established by the part geometry rules."
    if path and path[0] == "runtime":
        return label, "operating", "Current operating state or limit. It is separate from the saved mechanical body dimensions."
    if path and path[0] == "derived" and field == "radial_thickness_mm":
        return "Radial thickness", "physical" if profile in _ANNULAR else "unknown", (
            "Derived radial thickness = (mechanical_outer_diameter_mm − mechanical_inner_diameter_mm) / 2; no independent thickness field is stored.")
    if field == "model_3d":
        name = next((str(item) for item in reversed(path[3:]) if isinstance(item, str)), "model")
        return "3D model: " + _label(name, ("parts", "", name, *[p for p in path[3:] if isinstance(p, int)])), "cad", (
            "Explicit user CAD base, transform or Boolean feature. This controls the 3D solid; the optical/magnetic solver does not infer a new field law from the mesh.")
    if field in {"material_class", "material_regions", "bh_material", "relative_permeability"} or field.startswith("material_"):
        if field == "material_intervals_mm":
            return label, "physical", "Explicit axial material intervals, which define occupied material independently of the display envelope."
        return label, "material", "Material identity or constitutive response. Material settings are distinct from geometric dimensions and operating excitation."
    if profile == "magnetic_pole_piece" and field in {
            "pole_vacuum_connector_outer_diameter_mm", "pole_vacuum_connector_axial_length_mm"}:
        return label, "physical", "Outer material dimension of the pole's vacuum connector; this is not the diameter of the vacuum passage inside it."
    if "vacuum" in field or field in {"column_inner_diameter_mm", "clearance_to_vacuum_mm"}:
        return label, "vacuum", "Vacuum/beam-passage geometry or clearance. This is not automatically the material bore or the working aperture opening."
    if field in _SPLIT_ENDPOINTS:
        return label, "physical", "Parent-owned boundary of an actual upper/lower yoke material interval; the associated coil interval also includes its axial inset."
    if field == "mechanical_coil_axial_inset_mm":
        return label, "physical", "Axial inset of the winding from its parent material boundaries; it changes the actual coil material interval."
    if (field.startswith("local_") and field.endswith("_z_mm")) or "reference_local_z" in field or "centers_local_z" in field:
        return label, "placement", "Module-local axial placement or optical reference coordinate. A coordinate is not a measured material thickness."
    if field.startswith("offset_") or field in {"detector_axis_rotation_deg", "point_spread_rotation_deg"}:
        return label, "operating", "Configured operating alignment/response setting, separate from the nominal material envelope."
    if part.get("aperture_plate_form") == "perforated_strip":
        from temsim.part_model_apertures import dimension_semantics
        known = dimension_semantics(part)
        if field in known:
            label, description = known[field]
            category = ("envelope" if field in {"length_mm", "mechanical_outer_diameter_mm", "outer_diameter_mm"}
                        else "operating" if field == "maximum_radius_mm" else "physical")
            return label, category, description
        if field in {"radius_mm", "diameter_mm", "opening_diameter_mm", "aperture_radius_mm", "reference_operating_diameter_mm"}:
            return label, "operating", "Working or reference aperture opening; it is separate from the carrier bore and the unspecified transverse strip outline."
        if "hole_diameters" in field:
            return label, "physical", "Diameter of an explicitly listed aperture hole. The file does not thereby establish the transverse plate outline or hole locations."
    if field.startswith(("effective_", "maximum_", "minimum_", "reference_operating_", "field_", "point_spread_")) or field in {
            "percent", "cs_mm", "cc_mm", "reference_bore_diameter_mm", "reference_bore_radius_mm", "current_a", "voltage_kv"}:
        return label, "operating", "An operating/reference limit, response parameter or field-model quantity; it does not directly specify a material boundary."
    parent = by_key.get(part.get("parent_key"), {})
    split = (profile in {"magnetic_excitation_coil", "magnetic_lens_yoke"}
             and (part.get("parent_key") == "objective_lens" or _SPLIT_ENDPOINTS.intersection(parent)
                  or "material_intervals_mm" in part))
    if profile in _ANNULAR:
        if field == "length_mm" and split:
            return "Display-envelope length", "envelope", "Display allocation only: actual material occupies separate parent-owned or explicit intervals. Edit those interval endpoints to change the solid length."
        if field in {"length_mm", "mechanical_inner_diameter_mm", "mechanical_outer_diameter_mm", "magnetic_radial_profile_mm"}:
            return label, "physical", "Dimension of the configured material body; its real inner/outer surfaces remain separate from the vacuum-passage constraint."
    if profile == "magnetic_pole_piece" and part.get("pole_piece_geometry_style", "") in _POLE_STYLES:
        if field == "pole_root_fillet_radius_range_mm":
            return label, "unknown", "Reference range only. The current pole solid has sharp shoulders; this metadata does not generate a fillet or select a physical fillet radius."
        if field == "magnetic_radial_profile_mm" or any(word in field for word in _DIMENSION_WORDS):
            return label, "physical", "Configured pole material geometry, including its open bore, shoulder, mounting shank or nose. A geometric meaning does not establish measurement provenance."
    if profile == "magnetic_lens_assembly":
        if field in {"pole_gap_mm", "mechanical_coil_radial_thickness_mm"}:
            return label, "physical", "Assembly-owned spacing or material reconstruction dimension, used by its constituent magnetic parts."
        if field in _ENVELOPE_FIELDS or field in {"bore_diameter_mm", "mechanical_clear_bore_diameter_mm"}:
            return label, "envelope", "Assembly allocation/reference envelope; the housing, winding, yokes and pole pieces define the individual material solids."
    if profile in {"circular_aperture", "fixed_differential_pumping_aperture", "vacuum_liner", "electrostatic_bias_tube"}:
        if field == "length_mm" and part.get(field) == 0:
            return "Reference-plane axial extent", "placement", "A zero-thickness reference plane, not an assigned physical plate thickness."
        if any(word in field for word in _DIMENSION_WORDS):
            return label, "physical", "Explicit dimension of the configured aperture/liner/tube body, distinct from the shared vacuum constraint."
    if field in _ENVELOPE_FIELDS:
        return ("Envelope length" if field == "length_mm" else label.replace("Material", "Envelope")), "envelope", (
            "The existing preview uses this value as a component allocation or outer envelope. Its internal/non-axisymmetric material construction is not established by that surface.")
    return label, "unknown", "This field's physical meaning is not sufficiently defined by the supported geometry rules; review its schema and source before treating it as a material dimension."


def _source_from_text(status, note, *, explicit_field):
    status, note = str(status or ""), str(note or "")
    combined = (status + " " + note).lower()
    normalized = combined.replace("_", " ").replace("-", " ")
    topology_only = any(word in normalized for word in ("topology only", "not dimensionally", "uncalibrated", "not calibrated", "photo only", "photograph only"))
    if topology_only:
        return "unspecified", "The stored evidence concerns topology or an uncalibrated image, not this dimension. " + note
    estimated = any(word in combined for word in ("provisional", "reconstruction", "engineering_assumption", "engineering assumption",
                                                   "simulator assumption", "non-oem simulator", "estimated", "schematic", "assumed", "spacing-derived"))
    if estimated:
        return "estimated", note or f"Stored status: {status}."
    if explicit_field and note.strip():
        if any(word in normalized for word in ("not measured", "unmeasured", "measurement unknown", "not documented", "undocumented")):
            return "unspecified", "The stored declaration does not establish a measured/documented dimension. " + note
        if any(word in status.lower() for word in ("measured", "metrology", "calibrated_measurement")):
            return "measured", note
        if any(word in status.lower() for word in ("documented", "published_dimension", "oem_drawing", "manufacturer_dimension")):
            return "documented", note
        if "user_defined" in status.lower():
            return "user_defined", note
    return "unspecified", ("Stored source does not establish this field's dimension provenance. " + note).strip()


def _source(part, path, category):
    if path and path[0] == "runtime":
        return "runtime", "Read from the current operating state, not a mechanical measurement or a new TOML dimension."
    if len(path) > 2 and path[2] == "model_3d":
        return "user_defined", "Explicit CAD geometry authored in model_3d; no independent dimensional verification is implied."
    if path and path[0] == "derived":
        return "unspecified", "Derived from mechanical_inner_diameter_mm and mechanical_outer_diameter_mm; no independent thickness measurement or source is declared."
    relative = _relative_path(path)
    field = str(relative[0]) if relative else ""
    metadata = part.get("parameter_metadata", {})
    if isinstance(metadata, Mapping):
        candidates = [".".join(map(str, relative)), field]
        for key in dict.fromkeys(candidates):
            declared = metadata.get(key)
            if not isinstance(declared, Mapping):
                continue
            kind, note = declared.get("source_kind"), declared.get("source_note", "")
            if not isinstance(kind, str) or kind not in SOURCE_LABELS or not isinstance(note, str) or (kind != "unspecified" and not note.strip()):
                return "unspecified", "A field-level source declaration is incomplete or unsupported; source_kind and a meaningful source_note are required."
            normalized_note = note.lower().replace("_", " ").replace("-", " ")
            if kind in {"measured", "documented"} and any(word in normalized_note for word in (
                    "topology only", "uncalibrated", "not dimensionally", "not calibrated", "photo only", "photograph only",
                    "not measured", "unmeasured", "not documented", "undocumented")):
                return "unspecified", "Field metadata cites uncalibrated/topology evidence rather than a dimension. " + note
            return kind, note or "Field metadata explicitly leaves its source unspecified."
    stem = re.sub(r"_(?:mrad|nm|um|mm|deg|rad|m|s)$", "", field)
    prefixes = list(dict.fromkeys((field, stem, re.sub(r"_(?:diameter|radius|length|width|height|thickness)$", "", stem))))
    if field == "length_mm" and part.get("mechanical_profile") == "fixed_differential_pumping_aperture":
        prefixes.insert(0, "mechanical_axial_thickness")
    for prefix in prefixes:
        status, note = part.get(prefix + "_status"), part.get(prefix + "_source")
        if status is not None or note is not None:
            return _source_from_text(status, note, explicit_field=True)
    if category in {"physical", "envelope", "placement"}:
        status, note = part.get("mechanical_geometry_status"), part.get("mechanical_geometry_source")
        if status is not None or note is not None:
            return _source_from_text(status, note, explicit_field=False)
    if part.get("aperture_plate_form") == "perforated_strip" and part.get("aperture_mechanism_evidence_source"):
        return "unspecified", "The aperture evidence identifies a perforated-strip topology, not calibrated dimensions; no field-specific dimension source is declared."
    return "unspecified", "No field-specific dimensional measurement, drawing or estimation source is declared."


def describe_parameter(part: Mapping, path: tuple, *, by_key=None) -> ParameterSemantics:
    """Describe parts/runtime paths, resolving parent-owned paths when supplied."""
    part = part if isinstance(part, Mapping) else {}
    path = tuple(path)
    by_key = by_key if isinstance(by_key, Mapping) else {}
    if len(path) >= 2 and path[0] == "parts" and part.get("key") not in {None, path[1]}:
        part = by_key.get(path[1], {})
    label, category, description = _meaning(part, path, by_key)
    source, note = _source(part, path, category)
    return ParameterSemantics(label, category, CATEGORY_LABELS[category], parameter_unit(path),
                              source, SOURCE_LABELS[source], note, description)
