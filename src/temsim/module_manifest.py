"""Read mechanical component geometry from the TOML module manifests."""

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import tempfile
import tomllib

from temsim.component_keys import (
    NANOPULSER_APERTURE,
    NANOPULSER_DEFLECTOR,
    PROJECTION_CHAMBER_DPA_APERTURE,
)
from temsim.detector.eds_geometry import (
    EDS_DETECTOR_DEFINITION_FIELD,
    EDS_DETECTOR_DEFINITION_FIELDS,
    EDSDetectorArrayGeometry,
    resolve_eds_detector_part_data,
)
from temsim.mechanical_profiles import (
    C1_C2_POLE_PIECE_CARTRIDGE,
    FIXED_DIFFERENTIAL_PUMPING_APERTURE,
    MAGNETIC_EXCITATION_COIL,
    MAGNETIC_LENS_ASSEMBLY,
    MAGNETIC_LENS_HOUSING,
    MAGNETIC_LENS_MECHANICAL_PROFILES,
    MAGNETIC_LENS_YOKE,
    MAGNETIC_POLE_PIECE,
    POST_PROJECTOR_DETECTOR_CHAMBER,
    TRANSVERSE_EDS_DETECTOR_ARRAY,
    lens_mechanical_part_keys,
)
from temsim.paths import INSTRUMENT_CONFIG_ROOT

MODULE_ROOT = INSTRUMENT_CONFIG_ROOT

MAGNETIC_FIELD_POLARITY_PROFILES = frozenset({
    MAGNETIC_LENS_ASSEMBLY,
    "integrated_magnetic_lens_channel",
})
FIELD_POLARITY_STATUSES = frozenset({
    "manufacturer_documented",
    "measured_calibration",
    "provisional_model_assumption",
})
PROJECTOR_FIELD_CALIBRATION_STATUSES = frozenset({
    "manufacturer_documented",
    "measured_calibration",
    "provisional_non_oem_principle_model",
})
DETECTOR_ORIENTATION_STATUSES = frozenset({
    "uncalibrated_identity",
    "measured_calibration",
    "service_calibration",
})
DETECTOR_POINT_SPREAD_MODELS = frozenset({"none", "gaussian"})
DETECTOR_POINT_SPREAD_STATUSES = frozenset({
    "manufacturer_documented",
    "measured_calibration",
    "provisional_model_parameter",
})
PROJECTOR_LENS_KEYS = (
    "diffraction_lens",
    "intermediate_lens",
    "projector_lens_1",
    "projector_lens_2",
)
CONDENSER_FIELD_CALIBRATION_KEYS = (
    "condenser_lens_1",
    "condenser_lens_2",
)
PROJECTOR_FIELD_CALIBRATION_FIELDS = (
    "maximum_peak_field_t",
    "field_half_width_mm",
    "default_excitation_percent",
    "maximum_excitation_percent",
    "field_profile_terms",
    "field_calibration_status",
    "field_calibration_source",
)
CONDENSER_FIELD_CALIBRATION_FIELDS = (
    *PROJECTOR_FIELD_CALIBRATION_FIELDS,
    "normalise_field_profile_peak",
)
MECHANICAL_GEOMETRY_STATUSES = frozenset({
    "manufacturer_documented",
    "measured_calibration",
    "engineering_reconstruction_not_oem",
})

APERTURE_MECHANISM_PART_KEYS = frozenset({
    "feg_dpa_aperture",
    "feg_c1_aperture",
    "thermionic_anode_aperture",
    "thermionic_c1_aperture",
    "condenser_aperture_2",
    "condenser_aperture_3",
    "objective_aperture",
    "selected_area_aperture",
    "energy_filter_entrance_aperture",
})
APERTURE_MECHANISM_METADATA = {
    "aperture_plate_material": "platinum_user_identified_unverified",
    "aperture_plate_form": "perforated_strip",
    "aperture_plate_attachment": "screw_to_single_connection_rod",
    "aperture_mechanism_evidence_status": (
        "user_identified_photo_topology_not_dimensionally_calibrated"
    ),
}

ACCELERATOR_STACK_PART_KEYS = frozenset({
    "feg_accelerator",
    "thermionic_accelerator",
})
ACCELERATOR_STACK_METADATA = {
    "accelerator_electrode_stack_form": (
        "repeated_annular_electrode_stages"
    ),
    "accelerator_electrode_stack_evidence_status": (
        "user_supplied_side_view_topology_not_dimensionally_calibrated"
    ),
}

PAIRED_INTERACTION_PART_KEYS = frozenset({
    "feg_deflector",
    "thermionic_deflector",
    "beam_deflector",
    "condenser_deflector",
    "ac_deflector",
    "probe_dp12_scan_deflector",
    "descan_deflector",
    "image_diffraction_deflector",
})

REFERENCE_FREE_PART_KEYS = frozenset({
    C1_C2_POLE_PIECE_CARTRIDGE,
    PROJECTION_CHAMBER_DPA_APERTURE,
    "post_projector_detector_chamber",
    "eds_detector_system",
    "feg_accelerator",
    "thermionic_accelerator",
    "sample_stage",
    "objective_upper_pole",
    "objective_lower_pole",
    "energy_filter",
    "condenser_lens_1_lower_pole",
    "condenser_lens_2_upper_pole",
    "condenser_lens_3_upper_pole",
    "condenser_lens_3_lower_pole",
    "diffraction_lens_upper_pole",
    "diffraction_lens_lower_pole",
    "intermediate_lens_upper_pole",
    "intermediate_lens_lower_pole",
    "projector_lens_1_upper_pole",
    "projector_lens_1_lower_pole",
    "projector_lens_2_upper_pole",
    "projector_lens_2_lower_pole",
})

OBJECTIVE_LENS_REFERENCE_FIELDS = (
    "upper_field_reference_local_z_mm",
    "lower_field_reference_local_z_mm",
    "virtual_reference_local_z_mm",
)

OBJECTIVE_LENS_LOCAL_POSITION_FIELDS = (
    *OBJECTIVE_LENS_REFERENCE_FIELDS,
    "upper_yoke_start_local_z_mm",
    "upper_yoke_end_local_z_mm",
    "lower_yoke_start_local_z_mm",
    "lower_yoke_end_local_z_mm",
    "nominal_back_focal_plane_local_z_mm",
    "nominal_image_plane_local_z_mm",
)

ENERGY_FILTER_GEOMETRY_FIELDS = (
    "prism_radius_mm",
    "bend_angle_deg",
    "prism_radial_field_index",
    "entrance_multipole_s_mm",
    "prism_entrance_s_mm",
    "prism_fringe_mm",
    "pole_gap_mm",
    "sector_radial_aperture_mm",
    "exit_multipole_d_mm",
    "multipole_01_s_mm",
    "multipole_02_s_mm",
    "multipole_03_s_mm",
    "multipole_04_d_mm",
    "multipole_05_d_mm",
    "multipole_06_d_mm",
    "multipole_07_d_mm",
    "multipole_08_d_mm",
    "multipole_09_d_mm",
    "multipole_10_d_mm",
    "slit_d_mm",
    "dynamic_focus_quadrupole_d_mm",
    "bias_tube_d_mm",
    "fast_shutter_d_mm",
    "camera_deflector_d_mm",
    "output_detector_d_mm",
    "output_detector_width_mm",
    "zebra_detector_d_mm",
    "eels_plane_offset_mm",
)

ENERGY_FILTER_M12_GEOMETRY_FIELDS = (
    "mechanical_bore_radius_mm",
    "mechanical_outer_radius_mm",
    "housing_length_mm",
    "magnetic_support_length_mm",
    "entrance_soft_edge_mm",
    "exit_soft_edge_mm",
    "pole_zero_angle_deg",
)

ENERGY_FILTER_MECHANICAL_METADATA_FIELDS = (
    "m12_housing_geometry_status",
    "m12_housing_geometry_source",
)

ENERGY_FILTER_SLIT_GEOMETRY_FIELDS = (
    "clear_height_mm",
    "maximum_gap_mm",
    "blade_thickness_mm",
)

ENERGY_FILTER_PRISM_GEOMETRY_FIELDS = (
    "prism_radius_mm",
    "bend_angle_deg",
    "prism_radial_field_index",
    "fringe_length_mm",
    "pole_gap_mm",
    "radial_clear_half_width_mm",
)

ENERGY_FILTER_BRANCH_METADATA_FIELDS = (
    "public_topology_status",
    "public_topology_source",
    "multipole_family_evidence",
    "multipole_numbering_status",
    "geometry_policy",
)

ENERGY_FILTER_ZEBRA_FIELDS = (
    "strip_count",
    "pixels_per_strip",
    "strip_pixel_pitch_um",
    "strip_active_width_mm",
    "strip_active_height_mm",
    "alignment_pixels_non_dispersive",
    "alignment_pixels_dispersive",
    "alignment_active_height_mm",
    "alignment_active_width_mm",
    "maximum_spectra_per_s",
    "provisional_strip_center_pitch_mm",
)

RECORDING_PLANE_GEOMETRY_FIELDS = (
    "outer_width_mm",
    "inner_diameter_mm",
)

RECORDING_PLANE_POINT_SPREAD_FIELDS = (
    "point_spread_model",
    "point_spread_sigma_x_mm",
    "point_spread_sigma_y_mm",
    "point_spread_rotation_deg",
    "point_spread_status",
    "point_spread_source",
)

PROBE_CORRECTOR_COLUMN_KEYS = (
    "adapter_lens",
    "probe_dph2_deflector",
    "probe_qph2_quadrupole",
    "probe_hp2_hexapole",
    "probe_tl22_lens",
    "probe_dp22_deflector",
    "probe_hpc_hexapole",
    "probe_qpc_quadrupole",
    "probe_dp21_deflector",
    "probe_tl21_lens",
    "probe_dph1_deflector",
    "probe_qph1_quadrupole",
    "probe_hp1_hexapole",
    "probe_hpol_hexapole",
    "probe_qpol_quadrupole",
    "probe_dp11_deflector",
    "probe_tl12_lens",
    "probe_dp12_scan_deflector",
)

OBJECTIVE_COLUMN_KEYS = (
    "objective_lens",
    "sample_stage",
    "condenser_stigmator",
    "ac_deflector",
    "mini_condenser",
    "objective_upper_pole",
    "sample",
    "objective_aperture",
    "objective_lower_pole",
    "descan_deflector",
    "objective_stigmator",
    "image_diffraction_deflector",
)

IMAGE_CORRECTOR_COLUMN_KEYS = (
    "image_ol_post_lens",
    "image_hpol_hexapole",
    "image_qpol_quadrupole",
    "image_dp11_deflector",
    "image_tl11_lens",
    "image_dp12_deflector",
    "image_tl12_lens",
    "image_dph1_deflector",
    "image_hp1_hexapole",
    "image_dp21_deflector",
    "image_tl21_lens",
    "image_dp22_deflector",
    "image_tl22_lens",
    "image_dph2_deflector",
    "image_hp2_hexapole",
    "image_adapter_lens",
    "image_ish_deflector",
    "image_dsh_deflector",
    "image_dstg_quadrupole",
    "image_sad_plane",
)


@dataclass(frozen=True)
class PartGeometry:
    start_z_mm: float
    center_z_mm: float
    end_z_mm: float
    length_mm: float


def read_document(path):
    with Path(path).open("rb") as stream:
        return tomllib.load(stream)


def part_data(module_path, key, root=None):
    root = Path(root) if root is not None else MODULE_ROOT
    path = root / module_path
    document = read_document(path)
    matches = [
        part for part in document["parts"]
        if str(part["key"]) == str(key)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one part {key!r} in {path}, found {len(matches)}"
        )
    return dict(matches[0])


def port_z_mm(module_path, port, root=None):
    root = Path(root) if root is not None else MODULE_ROOT
    path = root / module_path
    document = read_document(path)
    try:
        return float(document["ports"][str(port)]["local_z_mm"])
    except KeyError as exc:
        raise ValueError(f"Missing {port!r} port in {path}") from exc


def part_geometry(module_path, key, root=None):
    root = Path(root) if root is not None else MODULE_ROOT
    path = root / module_path
    part = part_data(module_path, key, root)
    geometry = PartGeometry(
        float(part["local_start_z_mm"]),
        float(part["local_center_z_mm"]),
        float(part["local_end_z_mm"]),
        float(part["length_mm"]),
    )
    if not (
        geometry.start_z_mm
        <= geometry.center_z_mm
        <= geometry.end_z_mm
    ):
        raise ValueError(f"Invalid part range for {key} in {path}")
    envelope_length_mm = geometry.end_z_mm - geometry.start_z_mm
    if abs(geometry.length_mm - envelope_length_mm) > 1.0e-9:
        raise ValueError(
            f"Part length mismatch for {key} in {path}: "
            f"length_mm={geometry.length_mm}, "
            f"envelope={envelope_length_mm}"
        )
    return geometry


def part_requires_optical_reference(part):
    """Return whether an axial TOML optical-reference coordinate is needed.

    Curvilinear branch components own a path coordinate instead.  Requiring a
    fictitious axial reference at the branch entrance would silently flatten
    the Energy Filter into the main column and create a second geometry
    authority.
    """

    if isinstance(part, dict):
        if bool(part.get("branch_path_only", False)):
            return False
        key = str(part["key"])
    else:
        key = str(part)
    mechanical_suffixes = (
        "_housing",
        "_yoke",
        "_excitation_coil",
    )
    return (
        key not in REFERENCE_FREE_PART_KEYS
        and not key.endswith("_pole")
        and not key.endswith(mechanical_suffixes)
    )


def all_part_keys(root=None):
    root = Path(root) if root is not None else MODULE_ROOT
    keys = set()
    for path in root.rglob("*.toml"):
        document = read_document(path)
        keys.update(str(part["key"]) for part in document.get("parts", ()))
    return frozenset(keys)


def _format_toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("TOML table keys must be strings")
        return "{ " + ", ".join(
            f"{json.dumps(key, ensure_ascii=False)} = {_format_toml_value(item)}"
            for key, item in value.items()
        ) + " }"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not value == value or value in (float("inf"), float("-inf")):
            raise ValueError("TOML geometry values must be finite")
        return repr(value)
    raise TypeError(f"Unsupported TOML value type: {type(value).__name__}")


def _assignment_span(lines, start, end, field):
    pattern = re.compile(rf"^\s*{re.escape(field)}\s*=")
    for index in range(start, end):
        if not pattern.match(lines[index]):
            continue
        last = index + 1
        balance = lines[index].count("[") - lines[index].count("]")
        while balance > 0 and last < end:
            balance += lines[last].count("[") - lines[last].count("]")
            last += 1
        return index, last
    raise ValueError(f"Missing TOML field {field!r}")


def _section_span(lines, header):
    marker = f"[{header}]"
    for index, line in enumerate(lines):
        if line.strip() != marker:
            continue
        end = index + 1
        while end < len(lines) and not lines[end].lstrip().startswith("["):
            end += 1
        return index + 1, end
    raise ValueError(f"Missing TOML section {header!r}")


def _part_span(lines, key):
    starts = [
        first for first, _, header in _toml_statement_spans(lines, 0, len(lines))
        if lines[first].lstrip().startswith("[[") and header == ("parts",)
    ]
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        parsed = tomllib.loads("".join(lines[start:end]))
        if str(parsed["parts"][0]["key"]) == str(key):
            return start + 1, end
    raise ValueError(f"Missing TOML part {key!r}")


def _table_header_path(line):
    """Read a standalone table header, including quoted keys and comments."""
    if not line.lstrip().startswith("["):
        return None
    try:
        node = tomllib.loads(line.rstrip() + "\n__temsim_header_marker__ = 0\n")
    except tomllib.TOMLDecodeError:
        return None
    path = []
    while isinstance(node, (dict, list)):
        if isinstance(node, list):
            node = node[0]
            continue
        key = next(iter(node))
        if key == "__temsim_header_marker__":
            return tuple(path)
        path.append(key)
        node = node[key]
    return None


def _toml_statement_spans(lines, start, end):
    """Yield complete assignments/headers, ignoring brackets inside values.

    Parsing individual statements also keeps apparent headers inside multiline
    arrays or strings from being mistaken for part/model table boundaries.
    """
    index = start
    while index < end:
        if not lines[index].strip() or lines[index].lstrip().startswith("#"):
            index += 1
            continue
        header = _table_header_path(lines[index])
        if header is not None:
            yield index, index + 1, header
            index += 1
            continue
        last = index + 1
        while True:
            try:
                tomllib.loads("".join(lines[index:last]))
            except tomllib.TOMLDecodeError:
                if last >= end:
                    raise
                last += 1
            else:
                break
        yield index, last, None
        index = last


def _replace_model_3d(lines, start, end, value, newline):
    """Replace the complete optional CAD table, whether inline or expanded.

    None is a deletion marker only for this optional field. Removed-feature
    table comments remain in the source; unrelated tables retain their text.
    """
    if value is not None and not isinstance(value, dict):
        raise ValueError("model_3d must be a table or None")
    statements = list(_toml_statement_spans(lines, start, end))
    headers = [(first, header) for first, _, header in statements if header is not None]
    direct_end = headers[0][0] if headers else end
    spans = []
    assignment = re.compile(r'''^\s*(?:model_3d|"model_3d"|'model_3d')\s*(?:\.|=)''')
    for first, last, header in statements:
        if first < direct_end and header is None and assignment.match(lines[first]):
            spans.append((first, last))
    for offset, (index, path) in enumerate(headers):
        if path[:2] == ("parts", "model_3d"):
            last = headers[offset + 1][0] if offset + 1 < len(headers) else end
            spans.append((index, last))
    for first, last in sorted(spans, reverse=True):
        lines[first:last] = [line for line in lines[first:last] if line.lstrip().startswith("#")]
    if value is not None:
        lines[start:start] = [f"model_3d = {_format_toml_value(value)}{newline}"]


def stage_manifest_text(text, updates):
    """Return TOML text with targeted section/part fields replaced."""

    lines = text.splitlines(keepends=True)
    newline = "\r\n" if "\r\n" in text else "\n"
    for path, value in updates.items():
        path = tuple(path)
        if len(path) == 3 and path[0] == "parts":
            start, end = _part_span(lines, path[1])
            field = path[2]
            if field == "model_3d":
                _replace_model_3d(lines, start, end, value, newline)
                continue
        elif len(path) >= 2:
            start, end = _section_span(lines, ".".join(path[:-1]))
            field = path[-1]
        else:
            raise ValueError(f"Invalid TOML update path: {path!r}")
        try:
            first, last = _assignment_span(lines, start, end, field)
        except ValueError:
            if len(path) == 3 and path[0] == "parts" and field == "material_regions":
                first = last = start
            else:
                raise
        indent = lines[first][:len(lines[first]) - len(lines[first].lstrip())]
        lines[first:last] = [
            f"{indent}{field} = {_format_toml_value(value)}{newline}"
        ]
    staged = "".join(lines)
    document = tomllib.loads(staged)
    validate_document(document)
    return staged


def validate_document(document):
    document = dict(document)
    document["parts"] = [
        (
            resolve_eds_detector_part_data(part)
            if str(part.get("key", "")) == "eds_detector_system"
            else part
        )
        for part in document.get("parts", ())
    ]
    if document.get("coordinate_system") != "module_local_z_mm":
        raise ValueError("Invalid module coordinate system")
    parts = tuple(document.get("parts", ()))
    part_keys = [str(part["key"]) for part in parts]
    if len(set(part_keys)) != len(part_keys):
        raise ValueError("Duplicate part key in module TOML")
    part_orders = []
    for part in parts:
        order = part.get("order")
        if not isinstance(order, int) or isinstance(order, bool):
            raise ValueError(
                f"{part['key']}.order must be an integer"
            )
        part_orders.append(order)
    if len(set(part_orders)) != len(part_orders):
        raise ValueError("Duplicate part order in module TOML")
    for part in parts:
        key = str(part["key"])
        start = float(part["local_start_z_mm"])
        center = float(part["local_center_z_mm"])
        end = float(part["local_end_z_mm"])
        length = float(part["length_mm"])
        try:
            vacuum_diameter = float(part["vacuum_inner_diameter_mm"])
        except KeyError as exc:
            raise ValueError(
                f"Missing vacuum_inner_diameter_mm for {key}"
            ) from exc
        if not math.isfinite(vacuum_diameter) or vacuum_diameter <= 0.0:
            raise ValueError(
                f"Vacuum inner diameter for {key} must be finite and positive"
            )
        if part_requires_field_polarity(part):
            if "polarity" in part:
                raise ValueError(
                    f"{key} uses deprecated polarity; use field_polarity"
                )
            try:
                field_polarity = part["field_polarity"]
                status = str(part["field_polarity_status"]).strip()
                source = str(part["field_polarity_source"]).strip()
            except KeyError as exc:
                raise ValueError(
                    f"Missing {exc.args[0]} for magnetic lens {key}"
                ) from exc
            if (
                not isinstance(field_polarity, int)
                or isinstance(field_polarity, bool)
                or field_polarity not in (-1, 1)
            ):
                raise ValueError(
                    f"{key}.field_polarity must be integer +1 or -1"
                )
            if status not in FIELD_POLARITY_STATUSES:
                raise ValueError(
                    f"{key}.field_polarity_status must be one of "
                    f"{sorted(FIELD_POLARITY_STATUSES)}"
                )
            if not source:
                raise ValueError(
                    f"{key}.field_polarity_source must not be empty"
                )
        if not start <= center <= end:
            raise ValueError(f"Invalid part range for {key}")
        if abs(length - (end - start)) > 1.0e-9:
            raise ValueError(
                f"Part length mismatch for {key}: "
                f"length_mm={length}, envelope={end - start}"
            )
        if key == "objective_lens":
            references = [
                float(part[field])
                for field in OBJECTIVE_LENS_REFERENCE_FIELDS
            ]
            if not all(start <= value <= end for value in references):
                raise ValueError(
                    "Objective Lens reference planes must remain inside "
                    "its TOML envelope"
                )
            upper, lower, virtual = references
            if not upper < virtual < lower:
                raise ValueError(
                    "Objective Lens references must be ordered upper, "
                    "virtual, lower"
                )
            continue
        if not part_requires_optical_reference(part):
            continue
        try:
            reference = float(part["optical_reference_local_z_mm"])
        except KeyError as exc:
            raise ValueError(
                f"Missing optical_reference_local_z_mm for {key}"
            ) from exc
        if (
            part.get("signal_collection_surface")
            == "upstream_top_surface"
            and not math.isclose(
                reference,
                start,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError(
                f"{key} signal plane must coincide with local_start_z_mm"
            )
        if not start <= reference <= end:
            raise ValueError(
                f"Optical reference for {key} lies outside its "
                "mechanical envelope"
            )
        if key in PAIRED_INTERACTION_PART_KEYS:
            try:
                interactions = tuple(
                    float(value)
                    for value in part["interaction_centers_local_z_mm"]
                )
            except KeyError as exc:
                raise ValueError(
                    f"Missing interaction_centers_local_z_mm for {key}"
                ) from exc
            if len(interactions) != 2:
                raise ValueError(
                    f"{key} requires exactly two TOML interaction planes"
                )
            if not all(start <= value <= end for value in interactions):
                raise ValueError(
                    f"Interaction planes for {key} lie outside its "
                    "mechanical envelope"
                )
            if abs(sum(interactions) / 2.0 - center) > 1.0e-9:
                raise ValueError(
                    f"Interaction planes for {key} must be symmetric "
                    "about its mechanical centre"
                )
        elif (
            any(token in key for token in (
                "lens",
                "stigmator",
                "quadrupole",
                "hexapole",
                "deflector",
            ))
            and abs(reference - center) > 1.0e-9
        ):
            raise ValueError(
                f"Symmetric component {key} must use its mechanical "
                "centre as the TOML optical reference"
            )
    try:
        drift_diameter = float(
            document["geometry"]["vacuum_drift_inner_diameter_mm"]
        )
    except KeyError as exc:
        raise ValueError(
            "Missing geometry.vacuum_drift_inner_diameter_mm"
        ) from exc
    if not math.isfinite(drift_diameter) or drift_diameter <= 0.0:
        raise ValueError("Vacuum drift inner diameter must be finite and positive")
    try:
        liner_wall = float(
            document["geometry"]["vacuum_liner_wall_thickness_mm"]
        )
    except KeyError as exc:
        raise ValueError(
            "Missing geometry.vacuum_liner_wall_thickness_mm"
        ) from exc
    if not math.isfinite(liner_wall) or liner_wall <= 0.0:
        raise ValueError("Vacuum liner wall thickness must be positive")
    _validate_aperture_mechanism_metadata(parts)
    _validate_simple_magnetic_layer_geometry(parts)
    from temsim.part_materials import validate_part_materials
    validate_part_materials(parts)
    from temsim.magnetic_circuits import validate_circuit_declarations
    validate_circuit_declarations(parts)
    _validate_accelerator_stack_metadata(parts)
    if document.get("module", {}).get("type") == "gun":
        _validate_gun_mechanical_relationships(parts)
    if document.get("module", {}).get("type") == "beam_blanker":
        _validate_nanopulser_module(document)
    if document.get("module", {}).get("type") == "column":
        _validate_column_order(parts)
        _validate_objective_assembly(parts)
        _validate_eds_detector_geometry(parts)
        _validate_two_pole_lens_assemblies(parts)
        _validate_condenser_field_calibrations(
            parts, document["geometry"]
        )
        _validate_magnetic_lens_mechanical_parts(
            parts, document["geometry"]
        )
        _validate_shared_lens_housings(parts)
        _validate_c1_c2_cartridge_and_vacuum_tube(
            parts, document["geometry"]
        )
        _validate_column_mechanical_overlaps(parts)
    if document.get("module", {}).get("type") == "project_and_recording_system":
        _validate_projector_lens_clearances(parts, document["geometry"])
        _validate_projector_lens_geometry_provenance(parts)
        _validate_projector_field_calibrations(parts)
        _validate_two_pole_lens_assemblies(parts)
        _validate_magnetic_lens_mechanical_parts(
            parts, document["geometry"]
        )
        _validate_recording_plane_geometry(parts)
        _validate_post_projector_detector_chamber(parts)
        _validate_energy_filter_geometry(parts)
    entrance = float(document["ports"]["entrance"]["local_z_mm"])
    exit_z = float(document["ports"]["exit"]["local_z_mm"])
    length = float(document["geometry"]["length_mm"])
    if abs(length - (exit_z - entrance)) > 1.0e-9:
        raise ValueError(
            "Module length mismatch: "
            f"length_mm={length}, port_span={exit_z - entrance}"
        )
    if any("model_3d" in part for part in parts):
        from temsim.part_model_features import validate_model_3d
        from temsim.part_model_3d import part_model_from_document

        for part in parts:
            if "model_3d" not in part:
                continue
            validate_model_3d(part)
            # Confirm cuts produce real, nonempty geometry. This does not make
            # the existing axisymmetric physics consume an arbitrary CAD model.
            part_model_from_document(document, part["key"], include_children=False)
    return document


def _validate_nanopulser_module(document):
    """Keep the provisional deflector/stop assembly physically consistent."""

    module = document["module"]
    if module.get("geometry_status") != "engineering_reconstruction_not_oem":
        raise ValueError("NanoPulser dimensions must retain their non-OEM status")
    for field in (
        "geometry_source", "public_topology_source", "public_topology_source_url",
    ):
        if not str(module.get(field, "")).strip():
            raise ValueError(f"NanoPulser is missing {field}")
    entrance = float(document["ports"]["entrance"]["local_z_mm"])
    exit_z = float(document["ports"]["exit"]["local_z_mm"])
    if not math.isfinite(entrance) or not math.isfinite(exit_z) or exit_z <= entrance:
        raise ValueError("NanoPulser module must have positive finite length")
    if any(
        document["ports"][port]["interface"] != "gun_to_column"
        for port in ("entrance", "exit")
    ):
        raise ValueError("NanoPulser must connect the gun-to-column interface")
    parts = sorted(document["parts"], key=lambda part: part["order"])
    if tuple(part["key"] for part in parts) != (
        NANOPULSER_DEFLECTOR, NANOPULSER_APERTURE,
    ):
        raise ValueError("NanoPulser requires one deflector followed by one aperture")
    deflector, aperture = parts
    for part in parts:
        if not entrance <= part["local_start_z_mm"] <= part["local_end_z_mm"] <= exit_z:
            raise ValueError("NanoPulser parts must remain inside the module")
    if deflector["local_end_z_mm"] >= aperture["local_start_z_mm"]:
        raise ValueError("NanoPulser aperture must be downstream of the deflector")
    for part, names in (
        (deflector, (
            "plate_length_mm", "plate_gap_mm",
            "mechanical_outer_diameter_mm", "mechanical_clear_bore_diameter_mm",
        )),
        (aperture, (
            "aperture_radius_mm", "plate_thickness_mm", "mechanical_outer_diameter_mm",
        )),
    ):
        for field in names:
            value = float(part[field])
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"NanoPulser {field} must be finite and positive")
    if float(deflector["plate_length_mm"]) > float(deflector["length_mm"]):
        raise ValueError("NanoPulser plates must fit inside their envelope")
    if float(deflector["mechanical_clear_bore_diameter_mm"]) >= float(
        deflector["mechanical_outer_diameter_mm"]
    ):
        raise ValueError("NanoPulser bore must fit inside its body")
    radius = float(aperture["aperture_radius_mm"])
    if 2.0 * radius > float(aperture["vacuum_inner_diameter_mm"]):
        raise ValueError("NanoPulser aperture must fit inside its vacuum bore")
    if not math.isclose(2.0 * radius, float(aperture["bore_diameter_mm"])):
        raise ValueError("NanoPulser aperture radius and bore diameter disagree")


def part_requires_field_polarity(part):
    """Return whether one TOML optical parent produces an axial magnetic field."""

    return (
        not bool(part.get("mechanical_only", False))
        and part.get("mechanical_profile")
        in MAGNETIC_FIELD_POLARITY_PROFILES
    )


def _validate_aperture_mechanism_metadata(parts):
    """Require provenance without inventing dimensions for aperture rods."""

    for part in parts:
        key = str(part["key"])
        if key not in APERTURE_MECHANISM_PART_KEYS:
            continue
        missing = [
            field
            for field in (
                *APERTURE_MECHANISM_METADATA,
                "aperture_mechanism_evidence_source",
            )
            if field not in part
        ]
        if missing:
            raise ValueError(
                f"Missing {key} aperture mechanism metadata: "
                + ", ".join(missing)
            )
        for field, expected in APERTURE_MECHANISM_METADATA.items():
            actual = str(part[field]).strip()
            if actual != expected:
                raise ValueError(
                    f"{key}.{field} must be {expected!r}, got {actual!r}"
                )
        if not str(part["aperture_mechanism_evidence_source"]).strip():
            raise ValueError(
                f"{key}.aperture_mechanism_evidence_source must not be empty"
            )
        thickness_field = (
            "plate_thickness_mm"
            if "plate_thickness_mm" in part
            else "active_length_mm"
        )
        thickness = float(part[thickness_field])
        if (
            not math.isfinite(thickness)
            or thickness <= 0.0
            or thickness > float(part["length_mm"])
        ):
            raise ValueError(
                f"{key}.{thickness_field} must be positive and no greater "
                "than its mechanical envelope"
            )


def _validate_accelerator_stack_metadata(parts):
    """Validate configured stage locations and photo-topology provenance."""

    for part in parts:
        key = str(part["key"])
        if key not in ACCELERATOR_STACK_PART_KEYS:
            continue
        missing = [
            field
            for field in (
                "stage_centers_z_mm",
                *ACCELERATOR_STACK_METADATA,
                "accelerator_electrode_stack_evidence_source",
            )
            if field not in part
        ]
        if missing:
            raise ValueError(
                f"Missing {key} accelerator-stack metadata: "
                + ", ".join(missing)
            )
        for field, expected in ACCELERATOR_STACK_METADATA.items():
            actual = str(part[field]).strip()
            if actual != expected:
                raise ValueError(
                    f"{key}.{field} must be {expected!r}, got {actual!r}"
                )
        if not str(
            part["accelerator_electrode_stack_evidence_source"]
        ).strip():
            raise ValueError(
                f"{key}.accelerator_electrode_stack_evidence_source must "
                "not be empty"
            )
        centers = tuple(float(value) for value in part["stage_centers_z_mm"])
        start = float(part["local_start_z_mm"])
        end = float(part["local_end_z_mm"])
        if (
            len(centers) < 2
            or not all(math.isfinite(value) for value in centers)
            or any(
                downstream <= upstream
                for upstream, downstream in zip(centers, centers[1:])
            )
            or not all(start <= value <= end for value in centers)
        ):
            raise ValueError(
                f"{key}.stage_centers_z_mm must contain at least two "
                "ordered finite positions inside the accelerator envelope"
            )


def _validate_round_lens_field_calibration(part, key, required_fields):
    missing = [field for field in required_fields if field not in part]
    if missing:
        raise ValueError(
            f"Missing {key} TOML field calibration: "
            + ", ".join(missing)
        )
    peak_t = float(part["maximum_peak_field_t"])
    half_width_mm = float(part["field_half_width_mm"])
    default_percent = float(part["default_excitation_percent"])
    maximum_percent = float(part["maximum_excitation_percent"])
    if not math.isfinite(peak_t) or peak_t <= 0.0:
        raise ValueError(
            f"{key}.maximum_peak_field_t must be finite and positive"
        )
    if not math.isfinite(half_width_mm) or half_width_mm <= 0.0:
        raise ValueError(
            f"{key}.field_half_width_mm must be finite and positive"
        )
    if (
        not math.isfinite(maximum_percent)
        or not 0.0 < maximum_percent <= 100.0
        or not math.isfinite(default_percent)
        or not 0.0 <= default_percent <= maximum_percent
    ):
        raise ValueError(
            f"{key} excitation percentages must satisfy "
            "0 <= default <= maximum <= 100"
        )
    terms = part["field_profile_terms"]
    if not isinstance(terms, list) or not terms:
        raise ValueError(
            f"{key}.field_profile_terms must be a non-empty list"
        )
    amplitude_sum = 0.0
    for index, term in enumerate(terms):
        if not isinstance(term, list) or len(term) != 3:
            raise ValueError(
                f"{key}.field_profile_terms[{index}] must contain "
                "amplitude, offset and sigma"
            )
        amplitude, offset, sigma = (float(value) for value in term)
        if not all(
            math.isfinite(value) for value in (amplitude, offset, sigma)
        ):
            raise ValueError(f"{key}.field_profile_terms must be finite")
        if sigma <= 0.0:
            raise ValueError(f"{key} field-profile sigma must be positive")
        amplitude_sum += amplitude
    if amplitude_sum <= 0.0:
        raise ValueError(
            f"{key} field-profile amplitude sum must be positive"
        )
    status = str(part["field_calibration_status"]).strip()
    source = str(part["field_calibration_source"]).strip()
    if status not in PROJECTOR_FIELD_CALIBRATION_STATUSES:
        raise ValueError(
            f"{key}.field_calibration_status must be one of "
            f"{sorted(PROJECTOR_FIELD_CALIBRATION_STATUSES)}"
        )
    if not source:
        raise ValueError(f"{key}.field_calibration_source must not be empty")


def _validate_projector_field_calibrations(parts):
    by_key = {str(part["key"]): part for part in parts}
    for key in PROJECTOR_LENS_KEYS:
        part = by_key.get(key)
        if part is None:
            raise ValueError(f"Missing projector field source {key}")
        _validate_round_lens_field_calibration(
            part, key, PROJECTOR_FIELD_CALIBRATION_FIELDS
        )


def _validate_condenser_field_calibrations(parts, geometry):
    by_key = {str(part["key"]): part for part in parts}
    design_peak_fields = geometry.get(
        "magnetic_lens_design_peak_fields_t", {}
    )
    for key in CONDENSER_FIELD_CALIBRATION_KEYS:
        part = by_key.get(key)
        if part is None:
            raise ValueError(f"Missing condenser field source {key}")
        _validate_round_lens_field_calibration(
            part, key, CONDENSER_FIELD_CALIBRATION_FIELDS
        )
        if not isinstance(part["normalise_field_profile_peak"], bool):
            raise ValueError(
                f"{key}.normalise_field_profile_peak must be boolean"
            )
        try:
            design_peak_t = float(design_peak_fields[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Missing design peak field for {key}"
            ) from exc
        if abs(
            float(part["maximum_peak_field_t"]) - design_peak_t
        ) > 1.0e-12:
            raise ValueError(
                f"{key} runtime and mechanical design peak fields must match"
            )


def _validate_recording_plane_geometry(parts):
    by_key = {str(part["key"]): part for part in parts}
    for key in ("flu_screen", "haadf", "camera", "df", "bf"):
        part = by_key.get(key)
        if part is None:
            continue
        missing = [
            field
            for field in RECORDING_PLANE_GEOMETRY_FIELDS
            if field not in part
        ]
        if missing:
            raise ValueError(
                f"Missing {key} TOML detector geometry: "
                + ", ".join(missing)
            )
        outer = float(part["outer_width_mm"])
        inner = float(part["inner_diameter_mm"])
        if outer <= 0.0 or inner < 0.0 or inner >= outer:
            if inner == 0.0 and outer > 0.0:
                pass
            else:
                raise ValueError(
                    f"{key} detector diameters must satisfy "
                    "0 <= inner < outer"
                )
        if part.get("mechanical_part_role") != "interaction_plane":
            raise ValueError(f"{key} must be an interaction-plane row")
        if not 0.0 < float(part["length_mm"]) <= 1.0:
            raise ValueError(f"{key} active plane must remain axially thin")
        if part.get("signal_collection_surface") != "upstream_top_surface":
            raise ValueError(
                f"{key} signal collection must use the upstream top surface"
            )
        if not math.isclose(
            float(part["optical_reference_local_z_mm"]),
            float(part["local_start_z_mm"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                f"{key} signal plane must coincide with local_start_z_mm"
            )
        missing_point_spread = [
            field
            for field in RECORDING_PLANE_POINT_SPREAD_FIELDS
            if field not in part
        ]
        if missing_point_spread:
            raise ValueError(
                f"Missing {key} TOML detector point spread: "
                + ", ".join(missing_point_spread)
            )
        model = str(part["point_spread_model"]).strip().lower()
        if model not in DETECTOR_POINT_SPREAD_MODELS:
            raise ValueError(
                f"{key}.point_spread_model must be one of "
                f"{sorted(DETECTOR_POINT_SPREAD_MODELS)}"
            )
        sigma_values = []
        for field in (
            "point_spread_sigma_x_mm",
            "point_spread_sigma_y_mm",
            "point_spread_rotation_deg",
        ):
            value = part[field]
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{key}.{field} must be finite numeric")
            if field != "point_spread_rotation_deg":
                sigma_values.append(float(value))
        if any(value < 0.0 for value in sigma_values):
            raise ValueError(f"{key} point-spread sigma values cannot be negative")
        if model == "gaussian" and any(
            value <= 0.0 for value in sigma_values
        ):
            raise ValueError(
                f"{key} Gaussian point spread requires positive sigma values"
            )
        status = str(part["point_spread_status"]).strip()
        if status not in DETECTOR_POINT_SPREAD_STATUSES:
            raise ValueError(
                f"{key}.point_spread_status must be one of "
                f"{sorted(DETECTOR_POINT_SPREAD_STATUSES)}"
            )
        if not str(part["point_spread_source"]).strip():
            raise ValueError(f"{key}.point_spread_source must not be empty")
        if key != "camera":
            continue
        calibration_fields = (
            "detector_axis_rotation_deg",
            "detector_flip_x",
            "detector_flip_y",
            "detector_orientation_uncertainty_deg",
            "detector_orientation_status",
            "detector_orientation_source",
        )
        missing_calibration = [
            field for field in calibration_fields if field not in part
        ]
        if missing_calibration:
            raise ValueError(
                "Missing camera detector-orientation calibration: "
                + ", ".join(missing_calibration)
            )
        angle = part["detector_axis_rotation_deg"]
        uncertainty = part["detector_orientation_uncertainty_deg"]
        if (
            not isinstance(angle, (int, float))
            or isinstance(angle, bool)
            or not math.isfinite(float(angle))
        ):
            raise ValueError("Camera detector-axis rotation must be finite")
        if (
            not isinstance(uncertainty, (int, float))
            or isinstance(uncertainty, bool)
            or not math.isfinite(float(uncertainty))
            or not 0.0 <= float(uncertainty) <= 180.0
        ):
            raise ValueError(
                "Camera orientation uncertainty must be between 0 and 180 deg"
            )
        for field in ("detector_flip_x", "detector_flip_y"):
            if not isinstance(part[field], bool):
                raise ValueError(f"camera.{field} must be Boolean")
        status = str(part["detector_orientation_status"]).strip()
        if status not in DETECTOR_ORIENTATION_STATUSES:
            raise ValueError(
                "camera.detector_orientation_status must be one of "
                f"{sorted(DETECTOR_ORIENTATION_STATUSES)}"
            )
        if not str(part["detector_orientation_source"]).strip():
            raise ValueError(
                "camera.detector_orientation_source must not be empty"
            )

    stem_keys = ("haadf", "df", "bf")
    if all(key in by_key for key in stem_keys):
        positions = tuple(
            float(by_key[key]["optical_reference_local_z_mm"])
            for key in stem_keys
        )
        if not positions[0] < positions[1] < positions[2]:
            raise ValueError(
                "STEM detector order must remain HAADF upstream of DF "
                "upstream of BF"
            )
        if "flu_screen" in by_key:
            screen_z_mm = float(
                by_key["flu_screen"]["optical_reference_local_z_mm"]
            )
            if not positions[0] < screen_z_mm < positions[1]:
                raise ValueError(
                    "The main screen must remain between HAADF and DF"
                )
        for index, key in enumerate(stem_keys, start=1):
            part = by_key[key]
            if int(part.get("axial_order_index", 0)) != index:
                raise ValueError(f"{key}.axial_order_index must be {index}")
            if part.get("axial_order_status") != (
                "evidence_backed_relative_order_absolute_z_provisional_non_oem"
            ):
                raise ValueError(f"{key} axial-order status is missing")
            if not str(part.get("axial_order_source", "")).strip():
                raise ValueError(f"{key} axial-order source is missing")
            source_urls = part.get("axial_order_source_urls", ())
            if (
                not isinstance(source_urls, list)
                or len(source_urls) < 2
                or any(
                    not str(url).startswith("https://")
                    for url in source_urls
                )
            ):
                raise ValueError(
                    f"{key} axial-order source URLs are missing"
                )


def _validate_post_projector_detector_chamber(parts):
    """Require a mechanical-only chamber around the post-P2 detector bank."""

    by_key = {str(part["key"]): part for part in parts}
    chamber = by_key.get("post_projector_detector_chamber")
    if chamber is None:
        raise ValueError("Missing post-projector detector chamber")
    required_fields = (
        "mechanical_inner_diameter_mm",
        "mechanical_outer_diameter_mm",
        "mechanical_geometry_status",
        "mechanical_geometry_source",
        "mechanical_geometry_source_urls",
        "contained_recording_plane_keys",
        "upstream_boundary_aperture_key",
    )
    missing = [field for field in required_fields if field not in chamber]
    if missing:
        raise ValueError(
            "Missing post-projector detector-chamber metadata: "
            + ", ".join(missing)
        )
    if (
        chamber.get("mechanical_profile")
        != POST_PROJECTOR_DETECTOR_CHAMBER
        or chamber.get("mechanical_part_role")
        != "detector_chamber_housing"
        or not bool(chamber.get("mechanical_only", False))
        or not bool(chamber.get("axial_vacuum_context_only", False))
    ):
        raise ValueError(
            "Post-projector detector chamber must be a mechanical-only "
            "axial-vacuum-context housing"
        )
    if chamber["mechanical_geometry_status"] != (
        "public_titan_topology_absolute_dimensions_provisional_non_oem"
    ):
        raise ValueError(
            "Post-projector detector chamber must remain explicitly non-OEM"
        )
    if not str(chamber["mechanical_geometry_source"]).strip():
        raise ValueError(
            "Post-projector detector-chamber source must not be empty"
        )
    source_urls = chamber["mechanical_geometry_source_urls"]
    if (
        not isinstance(source_urls, list)
        or len(source_urls) < 2
        or any(not str(url).startswith("https://") for url in source_urls)
    ):
        raise ValueError(
            "Post-projector detector-chamber source URLs are missing"
        )

    p2 = by_key.get("projector_lens_2_housing")
    if p2 is None:
        raise ValueError(
            "Post-projector detector chamber requires the P2 housing"
        )
    tolerance = 1.0e-9
    start = float(chamber["local_start_z_mm"])
    end = float(chamber["local_end_z_mm"])
    inner = float(chamber["mechanical_inner_diameter_mm"])
    outer = float(chamber["mechanical_outer_diameter_mm"])
    if not math.isclose(
        start,
        float(p2["local_end_z_mm"]),
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ValueError(
            "Post-projector detector chamber must start at the P2 housing end"
        )
    if not (
        math.isfinite(inner)
        and math.isfinite(outer)
        and inner >= float(p2["mechanical_outer_diameter_mm"])
        and outer > inner
    ):
        raise ValueError(
            "Post-projector detector-chamber diameters must clear the P2 "
            "housing and satisfy outer > inner"
        )
    if not math.isclose(
        float(chamber["vacuum_inner_diameter_mm"]),
        float(p2["vacuum_inner_diameter_mm"]),
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ValueError(
            "Post-projector detector chamber must retain the projector "
            "vacuum-path diameter"
        )

    if chamber["upstream_boundary_aperture_key"] != (
        PROJECTION_CHAMBER_DPA_APERTURE
    ):
        raise ValueError(
            "Post-projector detector chamber must identify its "
            "projection-chamber DPA boundary"
        )
    dpa = by_key.get(PROJECTION_CHAMBER_DPA_APERTURE)
    if dpa is None:
        raise ValueError(
            "Missing projection-chamber differential-pumping aperture"
        )
    required_dpa_fields = (
        "maximum_radius_mm",
        "mechanical_outer_diameter_mm",
        "mechanical_bore_diameter_mm",
        "reference_bore_diameter_mm",
        "reference_bore_status",
        "aperture_adjustability",
        "mechanical_axial_thickness_status",
        "conjugate_plane_status",
        "optical_constraint_policy",
        "mechanical_geometry_status",
        "mechanical_geometry_source",
        "mechanical_geometry_source_urls",
    )
    missing_dpa = [
        field for field in required_dpa_fields if field not in dpa
    ]
    if missing_dpa:
        raise ValueError(
            "Missing projection-chamber DPA metadata: "
            + ", ".join(missing_dpa)
        )
    if (
        dpa.get("mechanical_profile")
        != FIXED_DIFFERENTIAL_PUMPING_APERTURE
        or dpa.get("mechanical_part_role")
        != "fixed_vacuum_restriction"
        or bool(dpa.get("mechanical_only", False))
        or str(dpa.get("branch")) != "common"
    ):
        raise ValueError(
            "Projection-chamber DPA must be a common, fixed, "
            "always-inserted optical vacuum restriction"
        )
    # The documented boundary location is a default, not an immutable design
    # constraint. The simulator may move the complete zero-thickness stop.
    for field in (
        "local_end_z_mm",
        "local_start_z_mm",
        "optical_reference_local_z_mm",
    ):
        if not math.isclose(
            float(dpa.get(field, float("nan"))),
            float(dpa["local_center_z_mm"]),
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise ValueError(
                "Projection-chamber DPA stop and mechanical plane must coincide"
            )
    dpa_bore = float(dpa["mechanical_bore_diameter_mm"])
    reference_bore = float(dpa["reference_bore_diameter_mm"])
    dpa_outer = float(dpa["mechanical_outer_diameter_mm"])
    maximum_radius = float(dpa["maximum_radius_mm"])
    if not (
        math.isfinite(dpa_bore)
        and math.isfinite(reference_bore)
        and math.isfinite(dpa_outer)
        and 0.0 < dpa_bore < dpa_outer
        and reference_bore > 0.0
        and math.isfinite(maximum_radius)
        and 0.5 * dpa_bore <= maximum_radius <= 0.5 * min(
            dpa_outer, float(dpa["vacuum_inner_diameter_mm"])
        )
        and dpa_outer <= inner
    ):
        raise ValueError(
            "Projection-chamber DPA diameters must define one finite "
            "restriction inside the detector chamber"
        )
    if not math.isclose(
        float(dpa["vacuum_inner_diameter_mm"]),
        float(p2["vacuum_inner_diameter_mm"]),
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ValueError(
            "Projection-chamber DPA must retain the surrounding nominal "
            "vacuum-path diameter"
        )
    if (
        dpa["aperture_adjustability"]
        != "fixed_non_retractable_hardware_toml_design_variable"
        or dpa["conjugate_plane_status"]
        != "operating_mode_dependent_not_imposed_by_mechanical_layout"
        or dpa["optical_constraint_policy"]
        != "always_inserted_hard_edge_no_automatic_preset_recalculation"
    ):
        raise ValueError(
            "Projection-chamber DPA must remain an active stop without "
            "automatic preset recalculation"
        )
    dpa_urls = dpa["mechanical_geometry_source_urls"]
    if (
        not str(dpa["reference_bore_status"]).strip()
        or not str(dpa["mechanical_axial_thickness_status"]).strip()
        or not str(dpa["mechanical_geometry_status"]).strip()
        or not str(dpa["mechanical_geometry_source"]).strip()
        or not isinstance(dpa_urls, list)
        or len(dpa_urls) < 2
        or any(not str(url).startswith("https://") for url in dpa_urls)
    ):
        raise ValueError(
            "Projection-chamber DPA provenance metadata is incomplete"
        )

    contained = tuple(chamber["contained_recording_plane_keys"])
    expected = ("haadf", "flu_screen", "df", "bf")
    if contained != expected:
        raise ValueError(
            "Post-projector detector chamber must contain HAADF, main "
            "screen, DF and BF in axial order"
        )
    for key in contained:
        part = by_key.get(key)
        if part is None:
            raise ValueError(
                f"Post-projector detector chamber is missing {key}"
            )
        reference = float(part["optical_reference_local_z_mm"])
        if not start < reference < end:
            raise ValueError(
                f"{key} active plane must lie inside the post-projector "
                "detector chamber"
            )
    camera = by_key.get("camera")
    if camera is None or not (
        float(camera["optical_reference_local_z_mm"]) > end
    ):
        raise ValueError(
            "Camera active plane must remain downstream of the schematic "
            "viewing/STEM-detector chamber"
        )


def _validate_gun_mechanical_relationships(parts):
    """Validate co-located C1 and monochromator-slit mechanics."""

    by_key = {str(part["key"]): part for part in parts}
    slit = by_key.get("feg_monochromator_slit")
    if slit is None:
        return
    c1 = by_key.get("feg_c1_aperture")
    if c1 is None:
        raise ValueError("Monochromator slit requires the C1 mechanism")
    if (
        slit.get("parent_key") != "feg_c1_aperture"
        or not bool(slit.get("mechanical_only", False))
        or slit.get("mechanical_part_role") != "slit_blade_carrier"
    ):
        raise ValueError(
            "Monochromator slit must be a mechanical child of C1"
        )
    tolerance = 1.0e-9
    if abs(
        float(slit["local_center_z_mm"])
        - float(c1["local_center_z_mm"])
    ) > tolerance:
        raise ValueError("Monochromator slit and C1 must be co-located")
    if (
        float(slit["local_start_z_mm"])
        < float(c1["local_start_z_mm"]) - tolerance
        or float(slit["local_end_z_mm"])
        > float(c1["local_end_z_mm"]) + tolerance
    ):
        raise ValueError("Monochromator slit must fit inside the C1 envelope")


def _validate_energy_filter_geometry(parts):
    by_key = {str(part["key"]): part for part in parts}
    interface = by_key.get("energy_filter")
    if interface is None:
        return

    multipole_keys = tuple(
        f"energy_filter_multipole_{index:02d}"
        for index in range(1, 11)
    )
    branch_keys = (
        "energy_filter_tapered_prism",
        *multipole_keys,
        "energy_filter_slit",
        "energy_filter_dynamic_focus_electrostatic_quadrupole",
        "energy_filter_bias_tube",
        "energy_filter_shutter",
        "energy_filter_camera_deflector",
        "energy_filter_eftem_output_plane",
        "energy_filter_zebra",
    )
    required_keys = {
        "energy_filter_entrance_aperture",
        *branch_keys,
    }
    missing_keys = sorted(required_keys - by_key.keys())
    if missing_keys:
        raise ValueError(
            "Missing Iliad Energy Filter components: "
            + ", ".join(missing_keys)
        )

    # Geometry formerly lived on the branch interface.  Reject reintroduced
    # aliases so every internal component keeps one clear TOML owner.
    duplicate_geometry = sorted(
        field for field in ENERGY_FILTER_GEOMETRY_FIELDS
        if field in interface
    )
    if duplicate_geometry:
        raise ValueError(
            "Energy Filter branch interface must not duplicate component "
            "geometry: " + ", ".join(duplicate_geometry)
        )

    required_interface_fields = (
        *ENERGY_FILTER_BRANCH_METADATA_FIELDS,
        *ENERGY_FILTER_MECHANICAL_METADATA_FIELDS,
        "confirmed_large_tapered_prism_count",
        "confirmed_multipole_count",
    )
    missing = [
        field for field in required_interface_fields
        if field not in interface
    ]
    if missing:
        raise ValueError(
            "Missing Energy Filter topology metadata: "
            + ", ".join(missing)
        )
    for field in (
        *ENERGY_FILTER_BRANCH_METADATA_FIELDS,
        *ENERGY_FILTER_MECHANICAL_METADATA_FIELDS,
    ):
        value = interface[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Energy Filter {field} must be a non-empty string"
            )
    prism_count = interface["confirmed_large_tapered_prism_count"]
    multipole_count = interface["confirmed_multipole_count"]
    if (
        not isinstance(prism_count, int)
        or isinstance(prism_count, bool)
        or prism_count != 1
    ):
        raise ValueError("Iliad requires exactly one large tapered prism")
    if (
        not isinstance(multipole_count, int)
        or isinstance(multipole_count, bool)
        or multipole_count != 10
    ):
        raise ValueError("Iliad requires exactly ten multipole elements")
    if interface["multipole_numbering_status"] != (
        "simulator_m01_m10_indices_not_public_production_labels_or_exact_order"
    ):
        raise ValueError(
            "Iliad M01-M10 labels must remain identified as simulator indices"
        )

    entrance = by_key.get("energy_filter_entrance_aperture")
    if (
        interface.get("mechanical_part_role") != "branch_interface"
        or interface.get("path_coordinate") != "curvilinear_s_mm"
        or float(interface["length_mm"]) != 0.0
    ):
        raise ValueError(
            "Energy Filter must begin at a zero-thickness curvilinear "
            "branch interface"
        )
    if abs(
        float(interface["local_center_z_mm"])
        - float(entrance["local_center_z_mm"])
    ) > 1.0e-9:
        raise ValueError(
            "Energy Filter branch interface must coincide with its entrance "
            "aperture"
        )
    try:
        reference_aperture_diameter = float(
            entrance["reference_operating_diameter_mm"]
        )
    except KeyError as exc:
        raise ValueError(
            "Iliad entrance aperture requires the public 5 mm reference "
            "operating condition"
        ) from exc
    if not math.isfinite(reference_aperture_diameter) or not math.isclose(
        reference_aperture_diameter, 5.0, abs_tol=1.0e-12
    ):
        raise ValueError(
            "Iliad entrance reference operating diameter must remain 5 mm"
        )
    if reference_aperture_diameter > 2.0 * float(
        entrance["maximum_radius_mm"]
    ):
        raise ValueError(
            "Iliad entrance reference aperture exceeds its mechanism travel"
        )

    interface_z = float(interface["local_center_z_mm"])
    for key in branch_keys:
        component = by_key[key]
        if (
            not bool(component.get("branch_path_only", False))
            or component.get("branch") != "energy_filter"
            or float(component["length_mm"]) != 0.0
            or not math.isclose(
                float(component["local_center_z_mm"]),
                interface_z,
                abs_tol=1.0e-9,
                rel_tol=0.0,
            )
        ):
            raise ValueError(
                f"{key} must be a zero-thickness curvilinear branch part"
            )
        if component.get("path_reference") not in {
            "branch_entrance", "prism_exit"
        }:
            raise ValueError(f"{key} has an invalid branch path reference")
        path_field = (
            "path_entrance_mm"
            if key == "energy_filter_tapered_prism"
            else "path_center_mm"
        )
        try:
            path_value = float(component[path_field])
        except KeyError as exc:
            raise ValueError(f"{key} requires {path_field}") from exc
        if not math.isfinite(path_value) or path_value < 0.0:
            raise ValueError(f"{key}.{path_field} must be non-negative")
        status = str(
            component.get("mechanical_geometry_status", "")
        ).strip()
        if not status:
            raise ValueError(f"{key} requires mechanical geometry status")

    prism = by_key["energy_filter_tapered_prism"]
    missing = [
        field for field in ENERGY_FILTER_PRISM_GEOMETRY_FIELDS
        if field not in prism
    ]
    if missing:
        raise ValueError(
            "Missing Iliad tapered-prism geometry: " + ", ".join(missing)
        )
    prism_values = {
        field: float(prism[field])
        for field in ENERGY_FILTER_PRISM_GEOMETRY_FIELDS
    }
    if not all(math.isfinite(value) for value in prism_values.values()):
        raise ValueError("Iliad tapered-prism geometry must be finite")
    if prism_values["prism_radius_mm"] <= 0.0:
        raise ValueError("Energy Filter prism radius must be positive")
    if not 0.0 <= prism_values["prism_radial_field_index"] < 1.0:
        raise ValueError(
            "Energy Filter prism radial field index must be in [0, 1)"
        )
    if not 0.0 < prism_values["bend_angle_deg"] <= 180.0:
        raise ValueError("Energy Filter bend angle must be in (0, 180]")
    for field in (
        "fringe_length_mm",
        "pole_gap_mm",
        "radial_clear_half_width_mm",
    ):
        if prism_values[field] <= 0.0:
            raise ValueError(
                f"Energy Filter {field} must be positive"
            )
    if prism.get("bend_angle_status") != (
        "provisional_patent_example_not_product_confirmed"
    ):
        raise ValueError(
            "Iliad prism bend angle must remain explicitly provisional"
        )

    multipole_values = {}
    for index, key in enumerate(multipole_keys, start=1):
        component = by_key[key]
        missing = [
            field for field in ENERGY_FILTER_M12_GEOMETRY_FIELDS
            if field not in component
        ]
        if missing:
            raise ValueError(
                f"Missing {key} geometry: " + ", ".join(missing)
            )
        values = {
            field: float(component[field])
            for field in ENERGY_FILTER_M12_GEOMETRY_FIELDS
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"{key} geometry must be finite")
        bore = values["mechanical_bore_radius_mm"]
        outer = values["mechanical_outer_radius_mm"]
        housing_length = values["housing_length_mm"]
        support_length = values["magnetic_support_length_mm"]
        entrance_edge = values["entrance_soft_edge_mm"]
        exit_edge = values["exit_soft_edge_mm"]
        if not 0.0 < bore < outer:
            raise ValueError(
                f"{key} radii must satisfy 0 < bore < outer"
            )
        if (
            support_length <= 0.0
            or entrance_edge <= 0.0
            or exit_edge <= 0.0
            or entrance_edge + exit_edge >= support_length
        ):
            raise ValueError(
                f"{key} soft edges must leave a positive plateau"
            )
        if housing_length < support_length:
            raise ValueError(
                f"{key} housing length cannot be shorter than its "
                "magnetic support length"
            )
        if component.get("individual_pole_assignment_status") != "not_public":
            raise ValueError(
                f"{key} must not claim a public individual pole assignment"
            )
        expected_reference = (
            "branch_entrance" if index <= 3 else "prism_exit"
        )
        if component.get("path_reference") != expected_reference:
            raise ValueError(f"{key} uses the wrong path reference")
        multipole_values[index] = (
            float(component["path_center_mm"]), values
        )

    pre_positions = tuple(
        multipole_values[index][0] for index in range(1, 4)
    )
    if not (
        pre_positions[0]
        < pre_positions[1]
        < pre_positions[2]
        < float(prism["path_entrance_mm"])
    ):
        raise ValueError("Iliad M01-M03 must be ordered before the prism")
    for upstream, downstream in zip(pre_positions, pre_positions[1:]):
        upstream_index = pre_positions.index(upstream) + 1
        housing = multipole_values[upstream_index][1]["housing_length_mm"]
        next_housing = multipole_values[upstream_index + 1][1][
            "housing_length_mm"
        ]
        if 0.5 * (housing + next_housing) > downstream - upstream:
            raise ValueError("Iliad pre-prism multipole housings overlap")

    post_positions = tuple(
        multipole_values[index][0] for index in range(4, 11)
    )
    for offset, (upstream, downstream) in enumerate(
        zip(post_positions, post_positions[1:]), start=4
    ):
        housing = multipole_values[offset][1]["housing_length_mm"]
        next_housing = multipole_values[offset + 1][1]["housing_length_mm"]
        if 0.5 * (housing + next_housing) > downstream - upstream:
            raise ValueError("Iliad post-prism multipole housings overlap")

    slit = by_key["energy_filter_slit"]
    missing = [
        field for field in ENERGY_FILTER_SLIT_GEOMETRY_FIELDS
        if field not in slit
    ]
    if missing:
        raise ValueError(
            "Missing Iliad XO/slit geometry: " + ", ".join(missing)
        )
    slit_values = tuple(
        float(slit[field]) for field in ENERGY_FILTER_SLIT_GEOMETRY_FIELDS
    )
    if (
        not all(math.isfinite(value) for value in slit_values)
        or min(slit_values) <= 0.0
    ):
        raise ValueError("Iliad energy-slit dimensions must be positive")
    if not (
        bool(slit.get("xo_crossover_plane_confirmed", False))
        and bool(slit.get("eftem_energy_selection_optional", False))
    ):
        raise ValueError(
            "Iliad slit row must identify the XO plane and optional EFTEM stop"
        )

    dynamic_quad = by_key[
        "energy_filter_dynamic_focus_electrostatic_quadrupole"
    ]
    if (
        int(dynamic_quad.get("electrode_count", 0)) != 4
        or not bool(dynamic_quad.get("mechanical_only", False))
        or dynamic_quad.get("optical_model_status")
        != "mechanical_layout_only_dynamic_focus_field_not_implemented"
    ):
        raise ValueError(
            "Iliad dynamic-focus electrostatic quadrupole must remain an "
            "explicit four-electrode, mechanical-only placeholder"
        )
    for field in (
        "housing_length_mm", "clear_bore_diameter_mm",
        "mechanical_outer_diameter_mm",
    ):
        value = float(dynamic_quad[field])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Iliad dynamic quadrupole {field} must be positive")

    bias = by_key["energy_filter_bias_tube"]
    shutter = by_key["energy_filter_shutter"]
    camera_deflector = by_key["energy_filter_camera_deflector"]
    output_plane = by_key["energy_filter_eftem_output_plane"]
    for component, fields in (
        (bias, (
            "housing_length_mm", "clear_bore_diameter_mm",
            "mechanical_outer_diameter_mm", "maximum_abs_offset_ev",
        )),
        (shutter, (
            "electrode_length_mm", "electrode_gap_mm",
            "mechanical_outer_diameter_mm",
        )),
        (camera_deflector, (
            "electrode_length_mm", "electrode_gap_mm",
            "mechanical_outer_diameter_mm",
        )),
        (output_plane, ("active_width_mm",)),
    ):
        values = tuple(float(component[field]) for field in fields)
        if not all(math.isfinite(value) for value in values) or min(values) <= 0.0:
            raise ValueError(
                f"{component['key']} mechanical dimensions must be positive"
            )
    if bias.get("offset_range_status") != (
        "provisional_simulator_limit_not_iliad_product_specification"
    ):
        raise ValueError("Iliad bias-tube range must remain marked provisional")

    zebra = by_key["energy_filter_zebra"]
    missing = [
        field for field in ENERGY_FILTER_ZEBRA_FIELDS
        if field not in zebra
    ]
    if missing:
        raise ValueError(
            "Missing Iliad Zebra detector data: " + ", ".join(missing)
        )
    if (
        int(zebra["strip_count"]) != 5
        or int(zebra["pixels_per_strip"]) != 2048
        or int(zebra["alignment_pixels_non_dispersive"]) != 256
        or int(zebra["alignment_pixels_dispersive"]) != 2048
    ):
        raise ValueError("Iliad Zebra pixel topology does not match public data")
    pixel_pitch_mm = float(zebra["strip_pixel_pitch_um"]) * 1.0e-3
    zebra_numeric = tuple(
        float(zebra[field]) for field in ENERGY_FILTER_ZEBRA_FIELDS
    )
    if not all(math.isfinite(value) and value > 0.0 for value in zebra_numeric):
        raise ValueError("Iliad Zebra detector data must be finite and positive")
    expected_width = int(zebra["pixels_per_strip"]) * pixel_pitch_mm
    expected_alignment_height = (
        int(zebra["alignment_pixels_non_dispersive"]) * pixel_pitch_mm
    )
    if not math.isclose(
        float(zebra["strip_active_width_mm"]), expected_width,
        abs_tol=1.0e-9,
    ):
        raise ValueError("Iliad Zebra strip active width is inconsistent")
    if not math.isclose(
        float(zebra["alignment_active_width_mm"]), expected_width,
        abs_tol=1.0e-9,
    ):
        raise ValueError("Iliad Zebra alignment width is inconsistent")
    if not math.isclose(
        float(zebra["alignment_active_height_mm"]),
        expected_alignment_height,
        abs_tol=1.0e-9,
    ):
        raise ValueError("Iliad Zebra alignment height is inconsistent")
    if not math.isclose(
        float(zebra["strip_active_height_mm"]), 0.800,
        abs_tol=1.0e-12,
    ):
        raise ValueError("Iliad Zebra strip active height must be 0.800 mm")
    if float(zebra["provisional_strip_center_pitch_mm"]) < float(
        zebra["strip_active_height_mm"]
    ):
        raise ValueError("Iliad Zebra provisional strip pitch causes overlap")
    if zebra.get("strip_center_pitch_status") != (
        "adjustable_unknown_not_public"
    ):
        raise ValueError("Iliad Zebra strip pitch must remain marked unknown")

    positions = {
        key: float(by_key[key]["path_center_mm"])
        for key in branch_keys
        if key != "energy_filter_tapered_prism"
    }
    if not (
        positions["energy_filter_multipole_04"]
        < positions["energy_filter_multipole_05"]
        < positions["energy_filter_multipole_06"]
        < positions["energy_filter_multipole_07"]
        < positions["energy_filter_slit"]
        < positions[
            "energy_filter_dynamic_focus_electrostatic_quadrupole"
        ]
        < positions["energy_filter_multipole_08"]
        < positions["energy_filter_multipole_09"]
        < positions["energy_filter_multipole_10"]
        < positions["energy_filter_bias_tube"]
        < positions["energy_filter_shutter"]
        < positions["energy_filter_camera_deflector"]
        < positions["energy_filter_eftem_output_plane"]
        < positions["energy_filter_zebra"]
    ):
        raise ValueError(
            "Iliad post-prism multipoles, XO/slit, dynamic-focus element, "
            "MultiEELS electrostatics, output plane and Zebra must be ordered"
        )


def _validate_eds_detector_geometry(parts):
    """Validate the installed off-axis EDS array without inventing its size."""

    by_key = {str(part["key"]): part for part in parts}
    detector = by_key.get("eds_detector_system")
    sample = by_key.get("sample")
    if detector is None:
        raise ValueError("Column TOML is missing eds_detector_system")
    if sample is None:
        raise ValueError("EDS detector geometry requires the sample part")

    required_fields = (
        EDS_DETECTOR_DEFINITION_FIELD,
        *EDS_DETECTOR_DEFINITION_FIELDS,
    )
    missing = [field for field in required_fields if field not in detector]
    if missing:
        raise ValueError(
            "Missing installed EDS geometry metadata: " + ", ".join(missing)
        )

    if (
        detector.get("mechanical_profile")
        != TRANSVERSE_EDS_DETECTOR_ARRAY
        or detector.get("mechanical_part_role")
        != "sample_adjacent_x_ray_detector_array"
        or not bool(detector.get("mechanical_only", False))
        or detector.get("branch") != "detection"
        or detector.get("parent_key") != "objective_lens"
        or detector.get("mechanical_overlap_group") != "objective_assembly"
        or detector.get("mechanical_overlap_role") != "member"
        or not bool(detector.get("axial_vacuum_context_only", False))
    ):
        raise ValueError(
            "EDS must be a transverse mechanical child of the "
            "Objective assembly"
        )
    if detector[EDS_DETECTOR_DEFINITION_FIELD] != "EDS.toml":
        raise ValueError(
            "The installed EDS system must use the single EDS.toml "
            "definition"
        )

    tolerance = 1.0e-9
    sample_z = float(sample["local_center_z_mm"])
    detector_positions = tuple(
        float(detector[field])
        for field in (
            "local_start_z_mm",
            "local_center_z_mm",
            "local_end_z_mm",
        )
    )
    if (
        abs(float(detector["length_mm"])) > tolerance
        or any(abs(value - sample_z) > tolerance for value in detector_positions)
    ):
        raise ValueError(
            "EDS aggregate must remain a zero-thickness off-axis "
            "marker at the sample plane"
        )

    geometry = EDSDetectorArrayGeometry.from_part_data(detector)
    if (
        geometry.system_key != "eds"
        or geometry.segment_count != 6
        or not geometry.windowless
        or detector["detector_technology"]
        != "windowless_silicon_drift_detector_array"
    ):
        raise ValueError(
            "The installed EDS configuration must identify a six-segment "
            "windowless array"
        )
    if not math.isclose(
        geometry.minimum_unshadowed_solid_angle_sr,
        4.45,
        abs_tol=1.0e-12,
    ) or not math.isclose(
        geometry.analytical_holder_solid_angle_sr,
        4.04,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "The installed EDS reference must retain the documented 4.45 sr lower "
            "bound and 4.04 sr analytical-holder solid angle"
        )
    if not math.isclose(
        geometry.takeoff_angle_deg, 32.06, abs_tol=0.01
    ):
        raise ValueError(
            "The EDS reference take-off angle must retain the 32.06 degree "
            "single-dataset metadata value"
        )

    expected_statuses = {
        "segment_count_status": (
            "published_instrument_report_and_user_dataset_supported_"
            "not_oem_datasheet"
        ),
        "azimuth_status": "symmetric_engineering_reconstruction_not_oem",
        "takeoff_angle_status": (
            "single_user_dataset_metadata_not_universal_oem"
        ),
        "solid_angle_status": (
            "manufacturer_documented_minimum_and_holder_configuration"
        ),
        "active_area_status": "not_public",
        "sample_to_sensor_distance_status": "not_public",
        "mechanical_envelope_status": (
            "not_public_pending_user_cross_sections"
        ),
        "mounting_topology_status": (
            "patent_family_supported_product_detail_unconfirmed"
        ),
    }
    for field, expected in expected_statuses.items():
        if str(detector[field]).strip() != expected:
            raise ValueError(
                f"eds_detector_system.{field} must be {expected!r}"
            )
    if detector["mounting_topology"] != (
        "within_objective_lens_around_sample"
    ):
        raise ValueError("EDS mounting topology must remain explicit")
    expected_azimuth_centers = tuple(float(index * 60) for index in range(6))
    if geometry.azimuth_centers_deg != expected_azimuth_centers:
        raise ValueError(
            "EDS provisional azimuth centers must retain the symmetric "
            "60 degree engineering reconstruction"
        )

    unknown_physical_dimensions = (
        "active_area_per_segment_mm2",
        "sample_to_sensor_distance_mm",
        "mechanical_outer_diameter_mm",
        "sensor_face_width_mm",
        "sensor_face_height_mm",
        "detector_package_length_mm",
    )
    invented = [
        field for field in unknown_physical_dimensions if field in detector
    ]
    if invented:
        raise ValueError(
            "EDS public sources do not support these physical dimensions: "
            + ", ".join(invented)
        )
    if (
        not str(detector["takeoff_angle_source"]).strip()
        or not str(detector["eds_geometry_source"]).strip()
        or not isinstance(detector["eds_geometry_source_urls"], list)
        or not detector["eds_geometry_source_urls"]
        or not all(
            isinstance(url, str) and url.startswith(("https://", "http://"))
            for url in detector["eds_geometry_source_urls"]
        )
    ):
        raise ValueError("EDS geometry requires non-empty provenance")


def _validate_objective_assembly(parts):
    by_key = {str(part["key"]): part for part in parts}
    required = {
        "objective_lens",
        "objective_upper_pole",
        "sample_stage",
        "sample",
        "objective_aperture",
        "objective_lower_pole",
    }
    if not required.issubset(by_key):
        return
    lens = by_key["objective_lens"]
    upper_pole = by_key["objective_upper_pole"]
    stage = by_key["sample_stage"]
    sample = by_key["sample"]
    aperture = by_key["objective_aperture"]
    lower_pole = by_key["objective_lower_pole"]
    tolerance = 1.0e-9

    stage_fields = (
        "transverse_envelope_x_mm",
        "transverse_envelope_y_mm",
        "holder_insertion_axis",
    )
    missing_stage = [field for field in stage_fields if field not in stage]
    if missing_stage:
        raise ValueError(
            "Objective sample stage is missing TOML structure: "
            + ", ".join(missing_stage)
        )
    if "mechanical_outer_diameter_mm" not in sample:
        raise ValueError(
            "Objective sample is missing mechanical_outer_diameter_mm"
        )
    if (
        float(stage["transverse_envelope_x_mm"]) <= 0.0
        or float(stage["transverse_envelope_y_mm"]) <= 0.0
        or float(sample["mechanical_outer_diameter_mm"]) <= 0.0
    ):
        raise ValueError("Objective stage and sample envelopes must be positive")

    gap = (
        float(lower_pole["local_start_z_mm"])
        - float(upper_pole["local_end_z_mm"])
    )
    declared_gap = float(lens["s_twin_pole_gap_mm"])
    if abs(gap - declared_gap) > tolerance:
        raise ValueError(
            "Objective pole positions must produce the TOML S-TWIN gap"
        )
    if declared_gap <= 0.0:
        raise ValueError("The S-TWIN pole-piece gap must be positive")
    gap_center = 0.5 * (
        float(upper_pole["local_end_z_mm"])
        + float(lower_pole["local_start_z_mm"])
    )
    sample_center = float(sample["local_center_z_mm"])
    if abs(sample_center - gap_center) > tolerance:
        raise ValueError(
            "The S-TWIN sample must remain centered in the pole gap"
        )
    if (
        stage.get("mechanical_profile") != "transverse_goniometer"
        or abs(float(stage["local_center_z_mm"]) - sample_center)
        > tolerance
        or abs(
            float(stage["local_start_z_mm"])
            - float(upper_pole["local_end_z_mm"])
        ) > tolerance
        or abs(
            float(stage["local_end_z_mm"])
            - float(lower_pole["local_start_z_mm"])
        ) > tolerance
    ):
        raise ValueError(
            "The transverse sample goniometer must cross the Objective pole "
            "gap at the sample plane"
        )

    lens_start = float(lens["local_start_z_mm"])
    lens_end = float(lens["local_end_z_mm"])
    yoke_ranges = (
        (
            float(lens["upper_yoke_start_local_z_mm"]),
            float(lens["upper_yoke_end_local_z_mm"]),
        ),
        (
            float(lens["lower_yoke_start_local_z_mm"]),
            float(lens["lower_yoke_end_local_z_mm"]),
        ),
    )
    if not all(
        lens_start <= start < end <= lens_end
        for start, end in yoke_ranges
    ):
        raise ValueError(
            "Objective yoke ranges must remain inside the TOML assembly"
        )
    if abs(
        (yoke_ranges[0][1] - yoke_ranges[0][0])
        - (yoke_ranges[1][1] - yoke_ranges[1][0])
    ) > tolerance:
        raise ValueError("The S-TWIN upper and lower yokes must be symmetric")
    if (
        abs(
            (sample_center - yoke_ranges[0][0])
            - (yoke_ranges[1][1] - sample_center)
        ) > tolerance
        or abs(
            (sample_center - yoke_ranges[0][1])
            - (yoke_ranges[1][0] - sample_center)
        ) > tolerance
    ):
        raise ValueError(
            "The S-TWIN yoke ranges must mirror about the sample"
        )

    for field in (
        "length_mm",
        "mechanical_outer_diameter_mm",
        "mechanical_tip_diameter_mm",
        "mechanical_bore_diameter_mm",
    ):
        if abs(
            float(upper_pole[field]) - float(lower_pole[field])
        ) > tolerance:
            raise ValueError(
                f"The S-TWIN pole pieces must match in {field}"
            )

    upper_reference = float(lens["upper_field_reference_local_z_mm"])
    lower_reference = float(lens["lower_field_reference_local_z_mm"])
    if abs(
        (sample_center - upper_reference)
        - (lower_reference - sample_center)
    ) > tolerance:
        raise ValueError(
            "The S-TWIN field references must be symmetric about the sample"
        )
    if float(lens["upper_peak_field_t"]) != float(
        lens["lower_peak_field_t"]
    ):
        raise ValueError("The S-TWIN peak-field calibration must be symmetric")
    if float(lens["upper_field_half_width_mm"]) != float(
        lens["lower_field_half_width_mm"]
    ):
        raise ValueError("The S-TWIN field widths must be symmetric")
    for suffix in ("amplitudes", "offsets", "sigmas"):
        upper_values = tuple(lens[f"upper_field_profile_{suffix}"])
        lower_values = tuple(lens[f"lower_field_profile_{suffix}"])
        if not upper_values or upper_values != lower_values:
            raise ValueError(
                f"The S-TWIN {suffix} profile must be non-empty and symmetric"
            )
    if any(
        float(value) <= 0.0
        for value in lens["upper_field_profile_sigmas"]
    ):
        raise ValueError("Objective field-profile sigmas must be positive")
    profile_lengths = {
        len(lens[f"upper_field_profile_{suffix}"])
        for suffix in ("amplitudes", "offsets", "sigmas")
    }
    if len(profile_lengths) != 1:
        raise ValueError("Objective field-profile arrays must align")
    if min(
        float(lens["mechanical_outer_diameter_mm"]),
        float(lens["nominal_voltage_kv"]),
        float(lens["nominal_focal_length_mm"]),
        float(lens["maximum_excitation_percent"]),
    ) <= 0.0:
        raise ValueError("Objective TOML calibration values must be positive")

    aperture_reference = float(aperture["optical_reference_local_z_mm"])
    if not sample_center < aperture_reference <= lens_end:
        raise ValueError(
            "Objective Aperture optical plane must remain downstream of the "
            "sample and inside the Objective assembly"
        )
    aperture_center = float(aperture["local_center_z_mm"])
    if abs(aperture_reference - aperture_center) > tolerance:
        raise ValueError(
            "Objective Aperture optical plane must equal its mechanical centre"
        )
    if (
        float(aperture["local_start_z_mm"]) < sample_center - tolerance
        or float(aperture["local_end_z_mm"])
        > float(lower_pole["local_start_z_mm"]) + tolerance
    ):
        raise ValueError(
            "Objective Aperture body must remain below the sample and inside "
            "the Objective pole gap"
        )
    nominal_bfp = float(lens["nominal_back_focal_plane_local_z_mm"])
    nominal_image = float(lens["nominal_image_plane_local_z_mm"])
    if not nominal_bfp < nominal_image <= lens_end:
        raise ValueError(
            "Objective TOML planes must be ordered BFP, image inside assembly"
        )

    if {"ac_deflector", "descan_deflector"}.issubset(by_key):
        ac_scan = by_key["ac_deflector"]
        descan = by_key["descan_deflector"]
        ac_distance = sample_center - float(
            ac_scan["local_center_z_mm"]
        )
        descan_distance = float(descan["local_center_z_mm"]) - sample_center
        if (
            ac_distance <= 0.0
            or descan_distance <= 0.0
            or abs(ac_distance - descan_distance) > tolerance
        ):
            raise ValueError(
                "AC Scan and Descan centres must mirror about the sample"
            )
        for field in (
            "length_mm",
            "mechanical_coil_length_mm",
            "mechanical_inter_coil_gap_mm",
            "effective_thickness_mm",
        ):
            if abs(float(ac_scan[field]) - float(descan[field])) > tolerance:
                raise ValueError(
                    f"AC Scan and Descan must match in {field}"
                )
        ac_interactions = tuple(
            float(value)
            for value in ac_scan["interaction_centers_local_z_mm"]
        )
        descan_interactions = tuple(
            float(value)
            for value in descan["interaction_centers_local_z_mm"]
        )
        if (
            len(ac_interactions) != 2
            or len(descan_interactions) != 2
            or abs(
                (ac_interactions[1] - ac_interactions[0])
                - (descan_interactions[1] - descan_interactions[0])
            ) > tolerance
        ):
            raise ValueError(
                "AC Scan and Descan optical-plane separations must match"
            )


def _expected_column_order(parts):
    keys = {str(part["key"]) for part in parts}
    expected = [
        "condenser_lens_1",
        "condenser_lens_1_lower_pole",
        "condenser_lens_2",
        "condenser_lens_2_upper_pole",
        "condenser_aperture_2",
    ]
    if "condenser_lens_3" in keys:
        expected.extend((
            "condenser_deflector",
            "condenser_lens_3",
            "condenser_lens_3_upper_pole",
            "condenser_lens_3_lower_pole",
            "condenser_aperture_3",
        ))
    expected.append("beam_deflector")
    if "adapter_lens" in keys:
        expected.extend(PROBE_CORRECTOR_COLUMN_KEYS)
    expected.extend(OBJECTIVE_COLUMN_KEYS)
    if "image_ol_post_lens" in keys:
        expected.extend(IMAGE_CORRECTOR_COLUMN_KEYS)
    return tuple(expected)


def _validate_column_order(parts):
    derived_poles = {
        f"{part['key']}_{side}_pole"
        for part in parts
        if part.get("pole_piece_topology") == "two_pole_single_gap"
        for side in ("upper", "lower")
    }
    actual = tuple(
        str(part["key"])
        for part in sorted(parts, key=lambda part: int(part["order"]))
        if (
            str(part["key"]) not in derived_poles
            and not bool(part.get("mechanical_only", False))
        )
    )
    expected = _expected_column_order(parts)
    if actual != expected:
        raise ValueError(
            "Column part order must be C1, C2, C2 Aperture, optional "
            "Condenser Deflector/C3/C3 Aperture, Beam Deflector, optional "
            "Probe Corrector, then Objective and Image assemblies"
        )
    orders = tuple(int(part["order"]) for part in parts)
    if len(set(orders)) != len(orders) or any(order < 1 for order in orders):
        raise ValueError("Column part order values must be unique and positive")


def _validate_column_mechanical_overlaps(parts):
    tolerance = 1.0e-9
    by_key = {str(part["key"]): part for part in parts}

    def is_ancestor(ancestor_key, descendant):
        parent_key = descendant.get("parent_key")
        seen = set()
        while parent_key and parent_key not in seen:
            if parent_key == ancestor_key:
                return True
            seen.add(parent_key)
            parent = by_key.get(str(parent_key))
            parent_key = parent.get("parent_key") if parent else None
        return False

    def radial_annulus(part, overlap_start, overlap_end):
        if part.get("pole_piece_geometry_style") == (
            "objective_vertical_back_inserted_shank_tapered_nose"
        ):
            start = float(part["local_start_z_mm"])
            end = float(part["local_end_z_mm"])
            length = float(part["pole_mounting_shank_axial_length_mm"])
            if str(part["key"]) == "objective_upper_pole":
                shank_start, shank_end = start, start + length
            else:
                shank_start, shank_end = end - length, end
            if (
                overlap_start >= shank_start - tolerance
                and overlap_end <= shank_end + tolerance
            ):
                return (
                    float(part["pole_mounting_shank_inner_diameter_mm"]),
                    float(part["pole_stem_outer_diameter_mm"]),
                )
        if (
            "mechanical_inner_diameter_mm" in part
            and "mechanical_outer_diameter_mm" in part
        ):
            return (
                float(part["mechanical_inner_diameter_mm"]),
                float(part["mechanical_outer_diameter_mm"]),
            )
        if (
            part.get("mechanical_profile") == MAGNETIC_POLE_PIECE
            and "mechanical_bore_diameter_mm" in part
            and "mechanical_outer_diameter_mm" in part
        ):
            return (
                float(part["mechanical_bore_diameter_mm"]),
                float(part["mechanical_outer_diameter_mm"]),
            )
        if "mechanical_outer_diameter_mm" in part:
            for bore_field in (
                "mechanical_clear_bore_diameter_mm",
                "mechanical_bore_diameter_mm",
            ):
                if bore_field in part:
                    return (
                        float(part[bore_field]),
                        float(part["mechanical_outer_diameter_mm"]),
                    )
        return None

    def radial_annuli_are_disjoint(
        first, second, overlap_start, overlap_end
    ):
        first_annulus = radial_annulus(first, overlap_start, overlap_end)
        second_annulus = radial_annulus(second, overlap_start, overlap_end)
        if first_annulus is None or second_annulus is None:
            return False
        first_inner, first_outer = first_annulus
        second_inner, second_outer = second_annulus
        return (
            first_outer <= second_inner + tolerance
            or second_outer <= first_inner + tolerance
        )

    def objective_coil_active_intervals(part):
        if part.get("mechanical_profile") != MAGNETIC_EXCITATION_COIL:
            return None
        parent = by_key.get(str(part.get("parent_key", "")))
        if not parent or str(parent.get("key")) != "objective_lens":
            return None
        inset = float(parent["mechanical_coil_axial_inset_mm"])
        return (
            (
                float(parent["upper_yoke_start_local_z_mm"]) + inset,
                float(parent["upper_yoke_end_local_z_mm"]) - inset,
            ),
            (
                float(parent["lower_yoke_start_local_z_mm"]) + inset,
                float(parent["lower_yoke_end_local_z_mm"]) - inset,
            ),
        )

    def material_axial_intervals(part, overlap_start, overlap_end):
        active_intervals = objective_coil_active_intervals(part)
        if active_intervals is None:
            return ((overlap_start, overlap_end),)
        return tuple(
            (start, end)
            for active_start, active_end in active_intervals
            for start, end in ((
                max(active_start, overlap_start),
                min(active_end, overlap_end),
            ),)
            if end - start > tolerance
        )

    def axial_envelope(part):
        active_intervals = objective_coil_active_intervals(part)
        if active_intervals is not None:
            return active_intervals[0][0], active_intervals[-1][1]
        return (
            float(part["local_start_z_mm"]),
            float(part["local_end_z_mm"]),
        )

    def materials_intersect_axially(
        first, second, overlap_start, overlap_end
    ):
        return any(
            min(first_end, second_end) - max(first_start, second_start)
            > tolerance
            for first_start, first_end in material_axial_intervals(
                first, overlap_start, overlap_end
            )
            for second_start, second_end in material_axial_intervals(
                second, overlap_start, overlap_end
            )
        )

    def excitation_coil_materials_overlap(
        first, second, overlap_start, overlap_end
    ):
        for first_start, first_end in material_axial_intervals(
            first, overlap_start, overlap_end
        ):
            for second_start, second_end in material_axial_intervals(
                second, overlap_start, overlap_end
            ):
                material_start = max(first_start, second_start)
                material_end = min(first_end, second_end)
                if material_end - material_start <= tolerance:
                    continue
                first_annulus = radial_annulus(
                    first, material_start, material_end
                )
                second_annulus = radial_annulus(
                    second, material_start, material_end
                )
                if first_annulus is None or second_annulus is None:
                    continue
                first_inner, first_outer = first_annulus
                second_inner, second_outer = second_annulus
                if (
                    first_outer > second_inner + tolerance
                    and second_outer > first_inner + tolerance
                ):
                    return True
        return False

    for part in parts:
        group = part.get("mechanical_overlap_group")
        role = part.get("mechanical_overlap_role")
        reason = str(part.get("mechanical_overlap_reason", "")).strip()
        if group is None and role is None and not reason:
            continue
        if not group or role not in {"container", "member"} or not reason:
            raise ValueError(
                f"Incomplete mechanical overlap declaration for {part['key']}"
            )
    physical = [
        part for part in parts
        if float(part["length_mm"]) > tolerance
    ]
    for index, first in enumerate(physical):
        first_start, first_end = axial_envelope(first)
        for second in physical[index + 1:]:
            second_start, second_end = axial_envelope(second)
            overlap = (
                min(first_end, second_end)
                - max(first_start, second_start)
            )
            overlap_start = max(first_start, second_start)
            overlap_end = min(first_end, second_end)
            if overlap <= tolerance:
                continue
            first_is_coil = (
                first.get("mechanical_profile")
                == MAGNETIC_EXCITATION_COIL
            )
            second_is_coil = (
                second.get("mechanical_profile")
                == MAGNETIC_EXCITATION_COIL
            )
            if (
                (first_is_coil or second_is_coil)
                and first.get("mechanical_profile")
                != MAGNETIC_LENS_ASSEMBLY
                and second.get("mechanical_profile")
                != MAGNETIC_LENS_ASSEMBLY
                and excitation_coil_materials_overlap(
                    first, second, overlap_start, overlap_end
                )
            ):
                raise ValueError(
                    "Excitation-coil material overlap between "
                    f"{first['key']} and {second['key']}"
                )
            if (
                (first_is_coil or second_is_coil)
                and not materials_intersect_axially(
                    first, second, overlap_start, overlap_end
                )
            ):
                continue
            if (
                is_ancestor(str(first["key"]), second)
                or is_ancestor(str(second["key"]), first)
            ):
                continue
            if radial_annuli_are_disjoint(
                first, second, overlap_start, overlap_end
            ):
                continue
            same_group = (
                first.get("mechanical_overlap_group")
                and first.get("mechanical_overlap_group")
                == second.get("mechanical_overlap_group")
            )
            container_overlap = (
                first.get("mechanical_overlap_role") == "container"
                or second.get("mechanical_overlap_role") == "container"
            )
            mechanical_layer_overlap = (
                bool(first.get("mechanical_only", False))
                or bool(second.get("mechanical_only", False))
            )
            optical_parent_overlap = (
                first.get("mechanical_profile") == MAGNETIC_LENS_ASSEMBLY
                or second.get("mechanical_profile")
                == MAGNETIC_LENS_ASSEMBLY
            )
            if same_group and (
                container_overlap
                or mechanical_layer_overlap
                or optical_parent_overlap
            ):
                continue
            raise ValueError(
                f"Undeclared mechanical overlap between {first['key']} "
                f"and {second['key']}: {overlap} mm"
            )


def _validate_projector_lens_clearances(parts, geometry):
    """Require non-overlapping housings and a uniform-bore D-I-P1-P2 stack.

    The configured inter-lens gap is the nominal default design value.
    Custom housing lengths may change actual gaps without moving lens centres.
    """

    tolerance = 1.0e-9
    by_key = {str(part["key"]): part for part in parts}
    required_geometry = (
        "projector_stack_inter_lens_gap_mm",
        "projector_stack_vacuum_inner_diameter_mm",
        "projector_stack_geometry_status",
        "projector_stack_geometry_source",
    )
    missing = [field for field in required_geometry if field not in geometry]
    if missing:
        raise ValueError(
            "Missing projector-stack geometry: " + ", ".join(missing)
        )
    nominal_clearance = float(geometry["projector_stack_inter_lens_gap_mm"])
    vacuum_diameter = float(
        geometry["projector_stack_vacuum_inner_diameter_mm"]
    )
    if (
        not math.isfinite(nominal_clearance)
        or not 0.0 <= nominal_clearance <= 10.0
    ):
        raise ValueError(
            "Projector-stack nominal inter-lens gap must be between 0 and 10 mm"
        )
    if not math.isfinite(vacuum_diameter) or vacuum_diameter <= 0.0:
        raise ValueError(
            "Projector-stack vacuum inner diameter must be finite and positive"
        )
    if geometry["projector_stack_geometry_status"] != (
        "user_defined_non_oem_principle_model"
    ):
        raise ValueError(
            "Projector-stack geometry must remain explicitly non-OEM"
        )
    if not str(geometry["projector_stack_geometry_source"]).strip():
        raise ValueError("Projector-stack geometry source must not be empty")
    sequence = (
        "diffraction_lens",
        "intermediate_lens",
        "projector_lens_1",
        "projector_lens_2",
    )
    envelope_sequence = tuple(f"{key}_housing" for key in sequence)
    for upstream_key, downstream_key in zip(
        envelope_sequence, envelope_sequence[1:]
    ):
        if upstream_key not in by_key or downstream_key not in by_key:
            continue
        upstream = by_key[upstream_key]
        downstream = by_key[downstream_key]
        clearance = (
            float(downstream["local_start_z_mm"])
            - float(upstream["local_end_z_mm"])
        )
        if not math.isfinite(clearance) or clearance < 0.0:
            raise ValueError(
                f"Invalid mechanical clearance between {upstream_key} "
                f"and {downstream_key}: {clearance:.9g} mm; "
                "actual housing gap must be finite and non-negative "
                f"(nominal design gap: {nominal_clearance:.9g} mm)"
            )
    stack_keys = set()
    for lens_key in sequence:
        if lens_key not in by_key:
            continue
        stack_keys.add(lens_key)
        stack_keys.update((
            f"{lens_key}_upper_pole",
            f"{lens_key}_lower_pole",
            *lens_mechanical_part_keys(lens_key),
        ))
    for key in sorted(stack_keys):
        part = by_key.get(key)
        if part is None:
            continue
        actual = float(part["vacuum_inner_diameter_mm"])
        if not math.isclose(
            actual, vacuum_diameter, rel_tol=0.0, abs_tol=tolerance
        ):
            dimension_hint = ""
            if (
                part.get("mechanical_profile") in MAGNETIC_LENS_MECHANICAL_PROFILES
                and "magnetic_radial_profile_mm" not in part
            ):
                dimension_hint = (
                    "; vacuum ID is the beam passage, not material thickness. "
                    "Use Edit dimensions to adjust mechanical_inner_diameter_mm / "
                    "mechanical_outer_diameter_mm"
                )
            raise ValueError(
                f"{key} vacuum ID {actual:g} mm does not match projector "
                f"stack {vacuum_diameter:g} mm{dimension_hint}"
            )


def _validate_projector_lens_geometry_provenance(parts):
    """Require an explicit authority level for D-I-P1-P2 dimensions."""

    by_key = {str(part["key"]): part for part in parts}
    for lens_key in PROJECTOR_LENS_KEYS:
        if lens_key not in by_key:
            continue
        lens = by_key[lens_key]
        try:
            status = str(lens["mechanical_geometry_status"]).strip()
            source = str(lens["mechanical_geometry_source"]).strip()
        except KeyError as exc:
            raise ValueError(
                f"Missing {exc.args[0]} for projector lens {lens_key}"
            ) from exc
        if status not in MECHANICAL_GEOMETRY_STATUSES:
            raise ValueError(
                f"{lens_key}.mechanical_geometry_status must be one of "
                f"{sorted(MECHANICAL_GEOMETRY_STATUSES)}"
            )
        if not source:
            raise ValueError(
                f"{lens_key}.mechanical_geometry_source must not be empty"
            )


def _validate_two_pole_lens_assemblies(parts):
    """Validate declared independent two-pole, single-gap assemblies."""

    tolerance = 1.0e-9
    by_key = {str(part["key"]): part for part in parts}
    lens_keys = tuple(
        key for key, part in by_key.items()
        if part.get("pole_piece_topology") == "two_pole_single_gap"
    )
    for lens_key in lens_keys:
        upper_key = f"{lens_key}_upper_pole"
        lower_key = f"{lens_key}_lower_pole"
        missing = {upper_key, lower_key} - by_key.keys()
        if missing:
            raise ValueError(
                f"Missing pole pieces for {lens_key}: {sorted(missing)}"
            )
        lens = by_key[lens_key]
        upper = by_key[upper_key]
        lower = by_key[lower_key]
        group = f"{lens_key}_assembly"
        allowed_groups = {
            group,
            lens.get("mechanical_overlap_group"),
        }
        for pole, pole_key in ((upper, upper_key), (lower, lower_key)):
            if (
                pole.get("parent_key") != lens_key
                or pole.get("mechanical_overlap_group") not in allowed_groups
                or pole.get("mechanical_overlap_role") != "member"
            ):
                raise ValueError(
                    f"{pole_key} must be an independent member of {lens_key}"
                )
            bore = float(pole["mechanical_bore_diameter_mm"])
            tip = float(pole["mechanical_tip_diameter_mm"])
            outer = float(pole["mechanical_outer_diameter_mm"])
            if not 0.0 < bore < tip < outer:
                raise ValueError(
                    f"{pole_key} diameters must satisfy bore < tip < outer"
                )
            vacuum = float(pole["vacuum_inner_diameter_mm"])
            if vacuum > bore + tolerance:
                raise ValueError(
                    f"{pole_key} vacuum ID must not exceed its pole bore"
                )
        if "mechanical_clear_bore_diameter_mm" in lens:
            clear_bore = float(lens["mechanical_clear_bore_diameter_mm"])
            parent_vacuum = float(lens["vacuum_inner_diameter_mm"])
            minimum_pole_bore = min(
                float(upper["mechanical_bore_diameter_mm"]),
                float(lower["mechanical_bore_diameter_mm"]),
            )
            if (
                abs(clear_bore - parent_vacuum) > tolerance
                or clear_bore > minimum_pole_bore + tolerance
            ):
                raise ValueError(
                    f"{lens_key} clear bore must equal its vacuum ID and "
                    "fit inside both pole bores"
                )
        detail_fields = (
            "pole_mounting_shank_inner_diameter_mm",
            "pole_mounting_shank_axial_length_mm",
            "pole_nose_axial_length_mm",
            "pole_cone_angle_to_axis_deg",
            "pole_face_land_axial_thickness_mm",
            "pole_root_fillet_radius_range_mm",
        )
        for field in detail_fields:
            upper_has = field in upper
            lower_has = field in lower
            if upper_has != lower_has:
                raise ValueError(
                    f"{lens_key} pole pieces must both declare {field}"
                )
            if not upper_has:
                continue
            upper_value = upper[field]
            lower_value = lower[field]
            if field.endswith("_range_mm"):
                upper_range = tuple(float(value) for value in upper_value)
                lower_range = tuple(float(value) for value in lower_value)
                if (
                    len(upper_range) != 2
                    or len(lower_range) != 2
                    or upper_range != lower_range
                    or not 0.0 < upper_range[0] <= upper_range[1]
                ):
                    raise ValueError(
                        f"{lens_key} pole pieces require one matching, "
                        f"positive {field}"
                    )
                continue
            upper_scalar = float(upper_value)
            lower_scalar = float(lower_value)
            if abs(upper_scalar - lower_scalar) > tolerance:
                raise ValueError(
                    f"{lens_key} pole pieces must match in {field}"
                )
            if field == "pole_cone_angle_to_axis_deg":
                valid = 0.0 < upper_scalar < 90.0
            elif field in {
                "pole_mounting_shank_axial_length_mm",
                "pole_nose_axial_length_mm",
            }:
                valid = (
                    0.0 < upper_scalar <= float(upper["length_mm"])
                    and upper_scalar <= float(lower["length_mm"])
                )
            else:
                valid = upper_scalar > 0.0
            if not valid:
                raise ValueError(f"Invalid {lens_key}.{field}")
        lens_start = float(lens["local_start_z_mm"])
        lens_center = float(lens["local_center_z_mm"])
        lens_end = float(lens["local_end_z_mm"])
        gap = float(lens["pole_gap_mm"])
        expected_upper_end = lens_center - 0.5 * gap
        expected_lower_start = lens_center + 0.5 * gap
        checks = (
            (float(upper["local_end_z_mm"]), expected_upper_end),
            (float(lower["local_start_z_mm"]), expected_lower_start),
        )
        poles_inside_envelope = (
            float(upper["local_start_z_mm"])
            >= lens_start - tolerance
            and float(lower["local_end_z_mm"])
            <= lens_end + tolerance
        )
        if (
            gap <= 0.0
            or not poles_inside_envelope
            or any(
                abs(actual - expected) > tolerance
                for actual, expected in checks
            )
        ):
            raise ValueError(
                f"{lens_key} pole pieces must fit its envelope and bound "
                "its declared pole gap"
            )


def _validate_simple_magnetic_layer_geometry(parts):
    """Validate physical annular layers independently of legacy sizing recipes.

    Vacuum clearance and material ID describe different boundaries. Shared
    structures and explicit radial profiles retain their dedicated geometry
    checks; their envelopes are not treated as solid concentric cylinders.
    """
    from temsim.magnetic_circuits import optical_owner
    from temsim.magnetic_geometry import objective_layer_intervals_mm

    tolerance = 1.0e-9
    by_key = {str(part["key"]): part for part in parts}
    layers_by_owner = {}
    for part in parts:
        profile = part.get("mechanical_profile")
        if profile not in MAGNETIC_LENS_MECHANICAL_PROFILES:
            continue
        key = str(part["key"])
        dimensions = {}
        for field in (
            "local_start_z_mm", "local_center_z_mm", "local_end_z_mm",
            "length_mm", "mechanical_inner_diameter_mm",
            "mechanical_outer_diameter_mm", "vacuum_inner_diameter_mm",
        ):
            try:
                value = float(part[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{key}.{field} must be a finite number in mm") from exc
            if not math.isfinite(value):
                raise ValueError(f"{key}.{field} must be a finite number in mm")
            dimensions[field] = value
        if dimensions["length_mm"] <= 0.0:
            raise ValueError(f"{key}.length_mm must be positive")
        inner = dimensions["mechanical_inner_diameter_mm"]
        outer = dimensions["mechanical_outer_diameter_mm"]
        vacuum = dimensions["vacuum_inner_diameter_mm"]
        if not 0.0 <= inner < outer:
            raise ValueError(
                f"{key}: mechanical diameters require 0 <= ID < OD "
                f"(ID={inner:g} mm, OD={outer:g} mm)"
            )
        if vacuum <= 0.0 or inner < vacuum - tolerance:
            raise ValueError(
                f"{key}: mechanical ID {inner:g} mm must clear the positive "
                f"vacuum ID {vacuum:g} mm; edit mechanical ID/OD to change wall thickness"
            )
        material = part.get("material_class")
        if not isinstance(material, str) or not material.strip():
            raise ValueError(f"{key}.material_class must be a non-empty material class")

        owner_key = optical_owner(part, by_key)
        if owner_key is None:
            continue
        owner = by_key[owner_key]
        if (
            "magnetic_radial_profile_mm" in part
            or part.get("magnetic_lens_keys")
            or part.get("shared_housing_key")
            or owner.get("shared_housing_key")
            or owner.get("magnetic_circuit_topology") == "shared_pole_multi_gap"
        ):
            continue
        intervals = ()
        if profile in {MAGNETIC_LENS_YOKE, MAGNETIC_EXCITATION_COIL}:
            parent = by_key[str(part["parent_key"])]
            intervals = objective_layer_intervals_mm(
                parent, float(parent["local_start_z_mm"]), profile
            )
        if not intervals:
            intervals = ((dimensions["local_start_z_mm"], dimensions["local_end_z_mm"]),)
        layers_by_owner.setdefault(owner_key, []).append((part, inner, outer, intervals))

    for layers in layers_by_owner.values():
        for index, (first, first_inner, first_outer, first_intervals) in enumerate(layers):
            for second, second_inner, second_outer, second_intervals in layers[index + 1:]:
                if min(first_outer, second_outer) - max(first_inner, second_inner) <= tolerance:
                    continue
                for first_start, first_end in first_intervals:
                    for second_start, second_end in second_intervals:
                        start, end = max(first_start, second_start), min(first_end, second_end)
                        if end - start > tolerance:
                            raise ValueError(
                                f"Mechanical radial layers overlap: {first['key']} "
                                f"(ID/OD {first_inner:g}/{first_outer:g} mm) and "
                                f"{second['key']} (ID/OD {second_inner:g}/{second_outer:g} mm) "
                                f"at module Z {start:g}..{end:g} mm; "
                                "adjust mechanical ID/OD or separate the parts axially"
                            )


def _validate_magnetic_lens_mechanical_parts(parts, geometry):
    """Validate the non-OEM, constant-OD magnetic-lens reconstruction."""

    tolerance = 1.0e-9
    required_geometry = (
        "magnetic_lens_external_diameter_mm",
        "magnetic_lens_coil_axial_fraction",
        "magnetic_lens_coil_radial_thickness_base_mm",
        "magnetic_lens_coil_radial_thickness_per_t_mm",
        "magnetic_lens_design_peak_fields_t",
        "magnetic_lens_geometry_status",
        "magnetic_lens_geometry_source",
    )
    missing_geometry = [
        field for field in required_geometry if field not in geometry
    ]
    if missing_geometry:
        raise ValueError(
            "Missing magnetic-lens geometry metadata: "
            f"{missing_geometry}"
        )
    external_diameter = float(
        geometry["magnetic_lens_external_diameter_mm"]
    )
    axial_fraction = float(geometry["magnetic_lens_coil_axial_fraction"])
    coil_thickness_base = float(
        geometry["magnetic_lens_coil_radial_thickness_base_mm"]
    )
    coil_thickness_per_t = float(
        geometry["magnetic_lens_coil_radial_thickness_per_t_mm"]
    )
    peak_fields = geometry["magnetic_lens_design_peak_fields_t"]
    if (
        not math.isfinite(external_diameter)
        or external_diameter <= 0.0
        or not math.isfinite(axial_fraction)
        or not 0.0 < axial_fraction < 1.0
        or not math.isfinite(coil_thickness_base)
        or coil_thickness_base <= 0.0
        or not math.isfinite(coil_thickness_per_t)
        or coil_thickness_per_t <= 0.0
        or not isinstance(peak_fields, dict)
        or geometry["magnetic_lens_geometry_status"]
        != "engineering_reconstruction_not_oem"
        or not str(geometry["magnetic_lens_geometry_source"]).strip()
    ):
        raise ValueError("Invalid magnetic-lens geometry metadata")
    by_key = {str(part["key"]): part for part in parts}
    lens_keys = tuple(
        key for key, part in by_key.items()
        if part.get("mechanical_profile") == MAGNETIC_LENS_ASSEMBLY
    )
    expected = (
        ("housing", MAGNETIC_LENS_HOUSING),
        ("yoke", MAGNETIC_LENS_YOKE),
        ("excitation_coil", MAGNETIC_EXCITATION_COIL),
    )
    objective_pole_diameters = []
    condenser_pole_diameters = []
    nested_lenses = []
    shared_peak_fields = {}
    for lens_key in lens_keys:
        lens = by_key[lens_key]
        shared_key = lens.get("shared_housing_key")
        if shared_key:
            shared_peak_fields[str(shared_key)] = max(
                shared_peak_fields.get(str(shared_key), 0.0),
                float(peak_fields[lens_key]),
            )
    for lens_key in lens_keys:
        lens = by_key[lens_key]
        if lens.get("magnetic_circuit_id"):
            # Explicit circuits are validated independently of this legacy
            # non-OEM sizing recipe. A control channel need not own two poles.
            continue
        try:
            design_peak_field = float(peak_fields[lens_key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Missing design peak field for {lens_key}"
            ) from exc
        if not math.isfinite(design_peak_field) or design_peak_field <= 0.0:
            raise ValueError(
                f"Design peak field for {lens_key} must be positive"
            )
        child_keys = lens_mechanical_part_keys(lens_key)
        missing = set(child_keys) - by_key.keys()
        if missing:
            raise ValueError(
                f"Missing magnetic-lens mechanical parts for {lens_key}: "
                f"{sorted(missing)}"
            )
        lens_start = float(lens["local_start_z_mm"])
        lens_end = float(lens["local_end_z_mm"])
        parent_outer = float(lens["mechanical_outer_diameter_mm"])
        nested_parent_key = str(lens.get("nested_lens_parent_key", ""))
        if nested_parent_key:
            nested_lenses.append((lens_key, nested_parent_key))
        elif abs(parent_outer - external_diameter) > tolerance:
            raise ValueError(
                f"{lens_key} mechanical outer diameter must equal the "
                "common column diameter"
            )
        radial_ranges = []
        children_by_profile = {}
        for child_key, (role, profile) in zip(child_keys, expected):
            child = by_key[child_key]
            children_by_profile[profile] = child
            if (
                child.get("parent_key") != lens_key
                or not bool(child.get("mechanical_only", False))
                or child.get("mechanical_part_role") != role
                or child.get("mechanical_profile") != profile
            ):
                raise ValueError(
                    f"{child_key} must be an independent {role} of {lens_key}"
                )
            start = float(child["local_start_z_mm"])
            end = float(child["local_end_z_mm"])
            inner = float(child["mechanical_inner_diameter_mm"])
            outer = float(child["mechanical_outer_diameter_mm"])
            if (
                start < lens_start - tolerance
                or end > lens_end + tolerance
                or not 0.0 < inner < outer <= parent_outer + tolerance
            ):
                raise ValueError(
                    f"{child_key} must fit inside the {lens_key} envelope"
                )
            if not str(child.get("material_class", "")).strip():
                raise ValueError(f"Missing material_class for {child_key}")
            if (
                profile == MAGNETIC_LENS_HOUSING
                and abs(outer - parent_outer) > tolerance
            ):
                raise ValueError(
                    f"{child_key} outer diameter must equal its parent "
                    "lens envelope"
                )
            radial_ranges.append((inner, outer, child_key))
        for first, second in zip(radial_ranges, radial_ranges[1:]):
            if second[1] > first[0] + tolerance:
                raise ValueError(
                    f"Magnetic-lens radial layers overlap: "
                    f"{first[2]} and {second[2]}"
                )

        coil = children_by_profile[MAGNETIC_EXCITATION_COIL]
        coil_length = (
            float(coil["local_end_z_mm"])
            - float(coil["local_start_z_mm"])
        )
        poles = [
            part for part in parts
            if part.get("parent_key") == lens_key
            and part.get("mechanical_profile") == MAGNETIC_POLE_PIECE
        ]
        if not poles:
            raise ValueError(f"Missing pole pieces for {lens_key}")
        longest_pole = max(
            float(pole["local_end_z_mm"])
            - float(pole["local_start_z_mm"])
            for pole in poles
        )
        if lens_key == "objective_lens":
            try:
                coil_inset = float(
                    lens["mechanical_coil_axial_inset_mm"]
                )
                upper_body_length = (
                    float(lens["upper_yoke_end_local_z_mm"])
                    - float(lens["upper_yoke_start_local_z_mm"])
                )
                lower_body_length = (
                    float(lens["lower_yoke_end_local_z_mm"])
                    - float(lens["lower_yoke_start_local_z_mm"])
                )
            except KeyError as exc:
                raise ValueError(
                    "Objective lens is missing split-body geometry"
                ) from exc
            active_coil_lengths = (
                upper_body_length - 2.0 * coil_inset,
                lower_body_length - 2.0 * coil_inset,
            )
            if (
                coil_inset <= 0.0
                or any(
                    not longest_pole < length < external_diameter
                    for length in active_coil_lengths
                )
            ):
                raise ValueError(
                    "Objective active coil halves must be longer than each "
                    "pole piece and shorter than the column diameter"
                )
        else:
            expected_coil_length = axial_fraction * min(
                lens_end - lens_start, external_diameter
            )
            if (
                abs(coil_length - expected_coil_length) > tolerance
                or not longest_pole < coil_length < external_diameter
            ):
                raise ValueError(
                    f"{coil['key']} axial length must be the configured "
                    "fraction of the smaller lens envelope/column diameter, "
                    "longer than each pole piece, and shorter than the column "
                    "diameter"
                )
        coil_thickness = 0.5 * (
            float(coil["mechanical_outer_diameter_mm"])
            - float(coil["mechanical_inner_diameter_mm"])
        )
        effective_peak_field = shared_peak_fields.get(
            str(lens.get("shared_housing_key", "")),
            design_peak_field,
        )
        declared_coil_thickness = lens.get(
            "mechanical_coil_radial_thickness_mm"
        )
        expected_coil_thickness = (
            float(declared_coil_thickness)
            if declared_coil_thickness is not None
            else coil_thickness_base
            + coil_thickness_per_t * effective_peak_field
        )
        if abs(coil_thickness - expected_coil_thickness) > tolerance:
            raise ValueError(
                f"{coil['key']} radial thickness must follow the declared "
                "per-lens reconstruction or design-peak-field fallback"
            )
        expected_style = (
            "objective_vertical_back_inserted_shank_tapered_nose"
            if lens_key == "objective_lens"
            else "embedded_hourglass_bore"
            if lens_key.startswith("condenser_lens_")
            or lens_key == "mini_condenser"
            else "tapered_bore_pole"
        )
        for pole in poles:
            if pole.get("pole_piece_geometry_style") != expected_style:
                raise ValueError(
                    f"{pole['key']} must use pole-piece style "
                    f"{expected_style}"
                )
            outer = float(pole["mechanical_outer_diameter_mm"])
            if lens_key == "objective_lens":
                stem = float(pole.get("pole_stem_outer_diameter_mm", 0.0))
                bore = float(pole["mechanical_bore_diameter_mm"])
                tip = float(pole["mechanical_tip_diameter_mm"])
                coil_inner = float(coil["mechanical_inner_diameter_mm"])
                shank_inner = float(
                    pole.get("pole_mounting_shank_inner_diameter_mm", 0.0)
                )
                shank_length = float(
                    pole.get("pole_mounting_shank_axial_length_mm", 0.0)
                )
                root_clearance = 0.5 * (coil_inner - stem)
                if (
                    not bore < stem <= outer
                    or not bore <= shank_inner < stem
                    or not 0.0 < shank_length < longest_pole
                    or not tip < outer
                    or abs(root_clearance) > tolerance
                ):
                    raise ValueError(
                        f"{pole['key']} objective mounting-shank OD must "
                        "equal the excitation-coil ID and its head/bore "
                        "geometry must remain valid"
                    )
                objective_pole_diameters.append(outer)
            elif (
                lens_key.startswith("condenser_lens_")
                or lens_key == "mini_condenser"
            ):
                condenser_pole_diameters.append(outer)

        if lens_key == "objective_lens":
            upper, lower = sorted(poles, key=lambda part: float(
                part["local_center_z_mm"]
            ))
            shank_length = float(
                upper["pole_mounting_shank_axial_length_mm"]
            )
            if (
                abs(
                    float(lens["upper_yoke_end_local_z_mm"])
                    - float(upper["local_start_z_mm"])
                    - shank_length
                ) > tolerance
                or abs(
                    float(lower["local_end_z_mm"])
                    - float(lens["lower_yoke_start_local_z_mm"])
                    - shank_length
                ) > tolerance
            ):
                raise ValueError(
                    "Objective mounting shanks must extend into the upper "
                    "and lower yokes by their declared axial length"
                )
    for nested_key, parent_key in nested_lenses:
        nested_housing = by_key[f"{nested_key}_housing"]
        parent_coil = by_key[f"{parent_key}_excitation_coil"]
        radial_clearance = 0.5 * (
            float(parent_coil["mechanical_inner_diameter_mm"])
            - float(nested_housing["mechanical_outer_diameter_mm"])
        )
        if radial_clearance <= 0.0:
            raise ValueError(
                f"{nested_key} housing must fit radially inside the "
                f"{parent_key} excitation-coil bore"
            )
        if parent_key == "objective_lens":
            upper = by_key["objective_upper_pole"]
            nested_lens = by_key[nested_key]
            axial_clearance = (
                float(upper["local_start_z_mm"])
                - float(nested_lens["local_end_z_mm"])
            )
            if axial_clearance <= 0.0:
                raise ValueError(
                    "The nested lens must end upstream of the Objective "
                    "mounting shank"
                )

    if (
        objective_pole_diameters
        and condenser_pole_diameters
        and min(objective_pole_diameters)
        <= max(condenser_pole_diameters) + tolerance
    ):
        raise ValueError(
            "Objective pole-piece head must be larger than condenser "
            "pole pieces"
        )


def _validate_shared_lens_housings(parts):
    """Validate one housing represented by contiguous axial sections."""

    by_key = {str(part["key"]): part for part in parts}
    grouped = {}
    for part in parts:
        shared_key = part.get("shared_housing_key")
        if shared_key:
            grouped.setdefault(str(shared_key), []).append(part)
    for shared_key, members in grouped.items():
        optical_parents = {
            str(part["key"]): part
            for part in members
            if part.get("mechanical_part_role") == "optical_parent"
        }
        housing_sections = {
            str(part.get("shared_housing_section")): part
            for part in members
            if part.get("mechanical_part_role") == "housing"
        }
        if set(optical_parents) != {
            "condenser_lens_1",
            "condenser_lens_2",
        }:
            raise ValueError(
                f"{shared_key} must be shared by the C1 and C2 lenses"
            )
        if set(housing_sections) != {"upstream", "downstream"}:
            raise ValueError(
                f"{shared_key} requires upstream and downstream housing "
                "sections"
            )
        upstream = housing_sections["upstream"]
        downstream = housing_sections["downstream"]
        if (
            upstream.get("parent_key") != "condenser_lens_1"
            or downstream.get("parent_key") != "condenser_lens_2"
        ):
            raise ValueError(
                f"{shared_key} housing sections must belong to C1 and C2"
            )
        if abs(
            float(upstream["local_end_z_mm"])
            - float(downstream["local_start_z_mm"])
        ) > 1.0e-9:
            raise ValueError(
                f"{shared_key} housing sections must be axially contiguous"
            )
        for section in housing_sections.values():
            if (
                not bool(section.get("mechanical_only", False))
                or section.get("mechanical_profile")
                != MAGNETIC_LENS_HOUSING
            ):
                raise ValueError(
                    f"{shared_key} sections must be mechanical housing rows"
                )
        c1_poles = [
            part for part in parts
            if part.get("parent_key") == "condenser_lens_1"
            and part.get("mechanical_profile") == MAGNETIC_POLE_PIECE
        ]
        c2_poles = [
            part for part in parts
            if part.get("parent_key") == "condenser_lens_2"
            and part.get("mechanical_profile") == MAGNETIC_POLE_PIECE
        ]
        if len(c1_poles) != 1 or len(c2_poles) != 1:
            raise ValueError(
                f"{shared_key} requires one C1/C2 interface pole apiece"
            )
        for field in (
            "mechanical_bore_diameter_mm",
            "mechanical_tip_diameter_mm",
            "mechanical_outer_diameter_mm",
        ):
            if abs(
                float(c1_poles[0][field]) - float(c2_poles[0][field])
            ) > 1.0e-9:
                raise ValueError(
                    f"{shared_key} C1/C2 interface poles must match in "
                    f"{field}"
                )
        c1_coil = by_key["condenser_lens_1_excitation_coil"]
        c2_coil = by_key["condenser_lens_2_excitation_coil"]
        for field in (
            "mechanical_inner_diameter_mm",
            "mechanical_outer_diameter_mm",
        ):
            if abs(float(c1_coil[field]) - float(c2_coil[field])) > 1.0e-9:
                raise ValueError(
                    f"{shared_key} C1/C2 coils must match in {field}"
                )


def _validate_c1_c2_cartridge_and_vacuum_tube(parts, geometry):
    """Validate the photo-scaled shared pole cartridge and vacuum tube."""

    tolerance = 1.0e-9
    by_key = {str(part["key"]): part for part in parts}
    required_parts = {
        "condenser_lens_1",
        "condenser_lens_2",
        "condenser_lens_1_lower_pole",
        "condenser_lens_2_upper_pole",
        "c1_c2_pole_piece_cartridge",
        "condenser_aperture_2",
        "objective_upper_pole",
        "objective_lower_pole",
    }
    missing_parts = required_parts - by_key.keys()
    if missing_parts:
        raise ValueError(
            "Missing C1/C2 cartridge structure: "
            + ", ".join(sorted(missing_parts))
        )
    required_geometry = (
        "c1_c2_objective_vacuum_tube_start_z_mm",
        "c1_c2_objective_vacuum_tube_end_z_mm",
        "c1_c2_total_lens_length_mm",
        "c1_c2_lens_length_ratio_c2_to_c1",
        "c2_aperture_service_clearance_after_cartridge_mm",
        "c1_c2_objective_vacuum_tube_inner_diameter_mm",
        "c1_c2_objective_vacuum_tube_outer_diameter_mm",
        "c1_c2_objective_vacuum_tube_inner_to_outer_ratio",
        "c1_c2_objective_vacuum_tube_outer_to_objective_pole_ratio",
        "c1_c2_objective_vacuum_tube_seal_material",
        "c1_c2_objective_vacuum_tube_geometry_status",
        "c1_c2_objective_vacuum_tube_geometry_source",
    )
    missing_geometry = [
        field for field in required_geometry if field not in geometry
    ]
    if missing_geometry:
        raise ValueError(
            "Missing C1/C2-to-objective vacuum-tube geometry: "
            + ", ".join(missing_geometry)
        )

    c1 = by_key["condenser_lens_1"]
    c2 = by_key["condenser_lens_2"]
    c1_pole = by_key["condenser_lens_1_lower_pole"]
    c2_pole = by_key["condenser_lens_2_upper_pole"]
    cartridge = by_key["c1_c2_pole_piece_cartridge"]
    c2_aperture = by_key["condenser_aperture_2"]
    upper_objective = by_key["objective_upper_pole"]
    lower_objective = by_key["objective_lower_pole"]

    c1_length = (
        float(c1["local_end_z_mm"]) - float(c1["local_start_z_mm"])
    )
    c2_length = (
        float(c2["local_end_z_mm"]) - float(c2["local_start_z_mm"])
    )
    configured_total = float(geometry["c1_c2_total_lens_length_mm"])
    configured_ratio = float(
        geometry["c1_c2_lens_length_ratio_c2_to_c1"]
    )
    if (
        not math.isfinite(configured_total)
        or configured_total <= 0.0
        or not math.isfinite(configured_ratio)
        or configured_ratio <= 0.0
        or c1_length <= 0.0
        or c2_length <= 0.0
        or abs(c1_length + c2_length - configured_total) > tolerance
        or abs(c2_length / c1_length - configured_ratio) > tolerance
    ):
        raise ValueError(
            "C1/C2 lens lengths must retain the declared total and ratio"
        )

    if (
        not bool(cartridge.get("mechanical_only", False))
        or cartridge.get("mechanical_profile")
        != C1_C2_POLE_PIECE_CARTRIDGE
        or cartridge.get("mechanical_part_role")
        != "pole_piece_cartridge"
    ):
        raise ValueError(
            "c1_c2_pole_piece_cartridge must be one mechanical cartridge"
        )
    if (
        abs(
            float(cartridge["local_start_z_mm"])
            - float(c1["local_start_z_mm"])
        ) > tolerance
        or abs(
            float(cartridge["local_end_z_mm"])
            - float(c2["local_end_z_mm"])
        ) > tolerance
    ):
        raise ValueError(
            "The C1/C2 pole-piece cartridge must span both lens bores"
        )
    cartridge_inner = float(cartridge["mechanical_inner_diameter_mm"])
    cartridge_outer = float(cartridge["mechanical_outer_diameter_mm"])
    if not 0.0 < cartridge_inner < cartridge_outer:
        raise ValueError("The C1/C2 cartridge diameters are invalid")

    aperture_clearance = float(
        geometry["c2_aperture_service_clearance_after_cartridge_mm"]
    )
    actual_aperture_clearance = (
        float(c2_aperture["local_start_z_mm"])
        - float(cartridge["local_end_z_mm"])
    )
    if (
        not math.isfinite(aperture_clearance)
        or aperture_clearance <= 0.0
        or abs(actual_aperture_clearance - aperture_clearance) > tolerance
    ):
        raise ValueError(
            "The C2 aperture must retain its declared service clearance "
            "after the C1/C2 pole-piece cartridge"
        )
    if any(
        field in c2_aperture
        for field in (
            "parent_key",
            "mechanical_overlap_group",
            "mechanical_overlap_role",
            "mechanical_overlap_reason",
        )
    ):
        raise ValueError(
            "The downstream C2 aperture must be a standalone mechanism"
        )

    tube_start = float(
        geometry["c1_c2_objective_vacuum_tube_start_z_mm"]
    )
    tube_end = float(geometry["c1_c2_objective_vacuum_tube_end_z_mm"])
    tube_inner = float(
        geometry["c1_c2_objective_vacuum_tube_inner_diameter_mm"]
    )
    tube_outer = float(
        geometry["c1_c2_objective_vacuum_tube_outer_diameter_mm"]
    )
    inner_ratio = float(
        geometry["c1_c2_objective_vacuum_tube_inner_to_outer_ratio"]
    )
    objective_ratio = float(
        geometry[
            "c1_c2_objective_vacuum_tube_outer_to_objective_pole_ratio"
        ]
    )
    if not 0.0 < tube_inner < tube_outer < cartridge_inner:
        raise ValueError(
            "The continuous vacuum tube must fit inside the C1/C2 cartridge"
        )
    if (
        abs(tube_start - float(cartridge["local_start_z_mm"])) > tolerance
        or abs(tube_end - float(upper_objective["local_start_z_mm"]))
        > tolerance
    ):
        raise ValueError(
            "The continuous vacuum tube must run from the C1/C2 cartridge "
            "entrance to the upper Objective pole tail"
        )
    if (
        abs(tube_inner - inner_ratio * tube_outer) > tolerance
        or abs(inner_ratio - 0.30) > tolerance
    ):
        raise ValueError(
            "The continuous vacuum-tube ID must be 30% of its OD"
        )
    if abs(
        tube_outer
        - objective_ratio
        * float(upper_objective["mechanical_outer_diameter_mm"])
    ) > tolerance:
        raise ValueError(
            "The vacuum-tube OD must follow the photo-scaled Objective-pole "
            "outer-diameter reference"
        )

    for pole in (c1_pole, c2_pole):
        if (
            abs(float(pole["mechanical_outer_diameter_mm"]) - cartridge_inner)
            > tolerance
            or abs(float(pole["mechanical_bore_diameter_mm"]) - tube_outer)
            > tolerance
            or abs(float(pole["vacuum_inner_diameter_mm"]) - tube_inner)
            > tolerance
        ):
            raise ValueError(
                "C1/C2 internal poles must fit the shared cartridge and "
                "surround the continuous vacuum tube"
            )
        if pole.get("mechanical_container_key") != cartridge["key"]:
            raise ValueError(
                f"{pole['key']} must identify the shared cartridge container"
            )

    for pole in (upper_objective, lower_objective):
        connector_outer = float(
            pole["pole_vacuum_connector_outer_diameter_mm"]
        )
        connector_length = float(
            pole["pole_vacuum_connector_axial_length_mm"]
        )
        if (
            abs(connector_outer - tube_outer) > tolerance
            or abs(float(pole["mechanical_bore_diameter_mm"]) - tube_inner)
            > tolerance
            or abs(
                float(pole["pole_mounting_shank_inner_diameter_mm"])
                - tube_inner
            ) > tolerance
            or not 0.0 < connector_length < float(
                pole["pole_mounting_shank_axial_length_mm"]
            )
        ):
            raise ValueError(
                "Objective pole vacuum connectors must match the continuous "
                "tube OD/ID and fit inside their mounting shanks"
            )

    if (
        not str(
            geometry["c1_c2_objective_vacuum_tube_seal_material"]
        ).strip()
        or geometry["c1_c2_objective_vacuum_tube_geometry_status"]
        != "engineering_reconstruction_not_oem"
        or not str(
            geometry["c1_c2_objective_vacuum_tube_geometry_source"]
        ).strip()
    ):
        raise ValueError(
            "The C1/C2-to-objective tube requires seal and photo provenance"
        )


def _atomic_write_text(path, text):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def restore_manifest_texts(snapshot, root=None):
    root = Path(root) if root is not None else MODULE_ROOT
    for module_path, text in snapshot.items():
        _atomic_write_text(root / module_path, text)


def update_manifest_values(updates_by_module, root=None):
    """Validate and atomically write targeted values to module TOMLs."""

    root = Path(root) if root is not None else MODULE_ROOT
    staged = {}
    originals = {}
    for module_path, updates in updates_by_module.items():
        relative = str(module_path)
        path = root / relative
        original = path.read_text(encoding="utf-8")
        originals[relative] = original
        staged[relative] = stage_manifest_text(original, updates)
    replaced = []
    try:
        for relative, text in staged.items():
            _atomic_write_text(root / relative, text)
            replaced.append(relative)
    except Exception:
        for relative in replaced:
            _atomic_write_text(root / relative, originals[relative])
        raise
    return originals
