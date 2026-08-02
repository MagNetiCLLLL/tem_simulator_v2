"""Read mechanical component geometry from the TOML module manifests."""

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import tempfile
import tomllib

from temsim.paths import INSTRUMENT_CONFIG_ROOT
from temsim.mechanical_profiles import (
    MAGNETIC_EXCITATION_COIL,
    MAGNETIC_LENS_ASSEMBLY,
    MAGNETIC_LENS_HOUSING,
    MAGNETIC_LENS_MECHANICAL_PROFILES,
    MAGNETIC_LENS_YOKE,
    lens_mechanical_part_keys,
)

MODULE_ROOT = INSTRUMENT_CONFIG_ROOT

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
    "entrance_multipole_s_mm",
    "prism_entrance_s_mm",
    "prism_fringe_mm",
    "pole_gap_mm",
    "sector_radial_aperture_mm",
    "exit_multipole_d_mm",
    "slit_d_mm",
    "output_detector_d_mm",
    "output_detector_width_mm",
    "eels_plane_offset_mm",
)

ENERGY_FILTER_M12_GEOMETRY_FIELDS = (
    "entrance_m12_bore_radius_mm",
    "entrance_m12_outer_radius_mm",
    "entrance_m12_length_mm",
    "entrance_m12_entrance_soft_edge_mm",
    "entrance_m12_exit_soft_edge_mm",
    "entrance_m12_pole_zero_angle_deg",
    "exit_m12_bore_radius_mm",
    "exit_m12_outer_radius_mm",
    "exit_m12_length_mm",
    "exit_m12_entrance_soft_edge_mm",
    "exit_m12_exit_soft_edge_mm",
    "exit_m12_pole_zero_angle_deg",
)

ENERGY_FILTER_SLIT_GEOMETRY_FIELDS = (
    "slit_clear_height_mm",
    "slit_maximum_gap_mm",
    "slit_blade_thickness_mm",
)

RECORDING_PLANE_GEOMETRY_FIELDS = (
    "outer_width_mm",
    "inner_diameter_mm",
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
    "image_hp1_hexapole",
    "image_dp21_deflector",
    "image_tl21_lens",
    "image_dp22_deflector",
    "image_tl22_lens",
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


def part_requires_optical_reference(key):
    key = str(key)
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
        index for index, line in enumerate(lines)
        if line.strip() == "[[parts]]"
    ]
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        key_pattern = re.compile(
            rf'^\s*key\s*=\s*{re.escape(json.dumps(str(key)))}\s*$'
        )
        if any(key_pattern.match(lines[index].strip()) for index in range(start, end)):
            return start + 1, end
    raise ValueError(f"Missing TOML part {key!r}")


def stage_manifest_text(text, updates):
    """Return TOML text with targeted section/part fields replaced."""

    lines = text.splitlines(keepends=True)
    newline = "\r\n" if "\r\n" in text else "\n"
    for path, value in updates.items():
        path = tuple(path)
        if len(path) == 3 and path[0] == "parts":
            start, end = _part_span(lines, path[1])
            field = path[2]
        elif len(path) >= 2:
            start, end = _section_span(lines, ".".join(path[:-1]))
            field = path[-1]
        else:
            raise ValueError(f"Invalid TOML update path: {path!r}")
        first, last = _assignment_span(lines, start, end, field)
        indent = lines[first][:len(lines[first]) - len(lines[first].lstrip())]
        lines[first:last] = [
            f"{indent}{field} = {_format_toml_value(value)}{newline}"
        ]
    staged = "".join(lines)
    document = tomllib.loads(staged)
    validate_document(document)
    return staged


def validate_document(document):
    if document.get("coordinate_system") != "module_local_z_mm":
        raise ValueError("Invalid module coordinate system")
    parts = tuple(document.get("parts", ()))
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
        if key == "objective_aperture":
            # The cartridge body remains in the S-TWIN pole gap while its
            # effective aperture plane is independently calibrated to the
            # distributed-field back focal plane.
            try:
                float(part["optical_reference_local_z_mm"])
            except KeyError as exc:
                raise ValueError(
                    "Missing Objective Aperture optical reference"
                ) from exc
            continue
        if not part_requires_optical_reference(key):
            continue
        try:
            reference = float(part["optical_reference_local_z_mm"])
        except KeyError as exc:
            raise ValueError(
                f"Missing optical_reference_local_z_mm for {key}"
            ) from exc
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
    if document.get("module", {}).get("type") == "column":
        _validate_column_order(parts)
        _validate_column_mechanical_overlaps(parts)
        _validate_objective_assembly(parts)
        _validate_two_pole_lens_assemblies(parts)
        _validate_magnetic_lens_mechanical_parts(parts)
    if document.get("module", {}).get("type") == "project_and_recording_system":
        _validate_projector_lens_clearances(parts)
        _validate_two_pole_lens_assemblies(parts)
        _validate_magnetic_lens_mechanical_parts(parts)
        _validate_recording_plane_geometry(parts)
        _validate_energy_filter_geometry(parts)
    entrance = float(document["ports"]["entrance"]["local_z_mm"])
    exit_z = float(document["ports"]["exit"]["local_z_mm"])
    length = float(document["geometry"]["length_mm"])
    if abs(length - (exit_z - entrance)) > 1.0e-9:
        raise ValueError(
            "Module length mismatch: "
            f"length_mm={length}, port_span={exit_z - entrance}"
        )
    return document


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
                continue
            raise ValueError(
                f"{key} detector diameters must satisfy "
                "0 <= inner < outer"
            )


def _validate_energy_filter_geometry(parts):
    by_key = {str(part["key"]): part for part in parts}
    part = by_key.get("energy_filter")
    if part is None:
        return
    required_fields = (
        *ENERGY_FILTER_GEOMETRY_FIELDS,
        *ENERGY_FILTER_M12_GEOMETRY_FIELDS,
        *ENERGY_FILTER_SLIT_GEOMETRY_FIELDS,
    )
    missing = [
        field for field in required_fields
        if field not in part
    ]
    if missing:
        raise ValueError(
            "Missing Energy Filter TOML geometry: "
            + ", ".join(missing)
        )
    values = {
        field: float(part[field])
        for field in required_fields
    }
    if values["prism_radius_mm"] <= 0.0:
        raise ValueError("Energy Filter prism radius must be positive")
    if not 0.0 < values["bend_angle_deg"] <= 180.0:
        raise ValueError("Energy Filter bend angle must be in (0, 180]")
    for field in (
        "prism_fringe_mm",
        "pole_gap_mm",
        "sector_radial_aperture_mm",
        "output_detector_width_mm",
    ):
        if values[field] <= 0.0:
            raise ValueError(
                f"Energy Filter {field} must be positive"
            )
    for field in (
        "entrance_multipole_s_mm",
        "prism_entrance_s_mm",
        "exit_multipole_d_mm",
        "slit_d_mm",
        "output_detector_d_mm",
        "eels_plane_offset_mm",
    ):
        if values[field] < 0.0:
            raise ValueError(
                f"Energy Filter {field} must be non-negative"
            )
    if not (
        values["entrance_multipole_s_mm"]
        < values["prism_entrance_s_mm"]
    ):
        raise ValueError(
            "Energy Filter entrance multipole must precede prism entrance"
        )
    if not (
        values["exit_multipole_d_mm"]
        < values["slit_d_mm"]
        < values["output_detector_d_mm"]
    ):
        raise ValueError(
            "Energy Filter exit geometry must be ordered M12, slit, detector"
        )
    for role in ("entrance", "exit"):
        bore = values[f"{role}_m12_bore_radius_mm"]
        outer = values[f"{role}_m12_outer_radius_mm"]
        length = values[f"{role}_m12_length_mm"]
        entrance_edge = values[
            f"{role}_m12_entrance_soft_edge_mm"
        ]
        exit_edge = values[f"{role}_m12_exit_soft_edge_mm"]
        if not 0.0 < bore < outer:
            raise ValueError(
                f"Energy Filter {role} M12 radii must satisfy "
                "0 < bore < outer"
            )
        if (
            length <= 0.0
            or entrance_edge <= 0.0
            or exit_edge <= 0.0
            or entrance_edge + exit_edge >= length
        ):
            raise ValueError(
                f"Energy Filter {role} M12 soft edges must leave "
                "a positive plateau"
            )
    if min(
        values["slit_clear_height_mm"],
        values["slit_maximum_gap_mm"],
        values["slit_blade_thickness_mm"],
    ) <= 0.0:
        raise ValueError(
            "Energy Filter slit mechanical dimensions must be positive"
        )


def _validate_objective_assembly(parts):
    by_key = {str(part["key"]): part for part in parts}
    required = {
        "objective_lens",
        "objective_upper_pole",
        "sample",
        "objective_aperture",
        "objective_lower_pole",
    }
    if not required.issubset(by_key):
        return
    lens = by_key["objective_lens"]
    upper_pole = by_key["objective_upper_pole"]
    sample = by_key["sample"]
    aperture = by_key["objective_aperture"]
    lower_pole = by_key["objective_lower_pole"]
    tolerance = 1.0e-9

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
    nominal_bfp = float(lens["nominal_back_focal_plane_local_z_mm"])
    nominal_image = float(lens["nominal_image_plane_local_z_mm"])
    if abs(aperture_reference - nominal_bfp) > tolerance:
        raise ValueError(
            "Objective Aperture optical plane must equal the TOML nominal BFP"
        )
    if not nominal_bfp < nominal_image <= lens_end:
        raise ValueError(
            "Objective TOML planes must be ordered BFP, image inside assembly"
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

    def radial_annuli_are_disjoint(first, second):
        required = (
            "mechanical_inner_diameter_mm",
            "mechanical_outer_diameter_mm",
        )
        if not all(field in first and field in second for field in required):
            return False
        first_inner = float(first[required[0]])
        first_outer = float(first[required[1]])
        second_inner = float(second[required[0]])
        second_outer = float(second[required[1]])
        return (
            first_outer <= second_inner + tolerance
            or second_outer <= first_inner + tolerance
        )

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
        first_start = float(first["local_start_z_mm"])
        first_end = float(first["local_end_z_mm"])
        for second in physical[index + 1:]:
            second_start = float(second["local_start_z_mm"])
            second_end = float(second["local_end_z_mm"])
            overlap = (
                min(first_end, second_end)
                - max(first_start, second_start)
            )
            if overlap <= tolerance:
                continue
            if (
                is_ancestor(str(first["key"]), second)
                or is_ancestor(str(second["key"]), first)
            ):
                continue
            if radial_annuli_are_disjoint(first, second):
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
            if same_group and (container_overlap or mechanical_layer_overlap):
                continue
            raise ValueError(
                f"Undeclared mechanical overlap between {first['key']} "
                f"and {second['key']}: {overlap} mm"
            )


def _validate_projector_lens_clearances(parts):
    """Reject serial projector-lens overlap in recording modules."""
    tolerance = 1.0e-9
    by_key = {str(part["key"]): part for part in parts}
    sequence = (
        "diffraction_lens",
        "intermediate_lens",
        "projector_lens_1",
        "projector_lens_2",
    )
    for upstream_key, downstream_key in zip(sequence, sequence[1:]):
        if upstream_key not in by_key or downstream_key not in by_key:
            continue
        upstream = by_key[upstream_key]
        downstream = by_key[downstream_key]
        clearance = (
            float(downstream["local_start_z_mm"])
            - float(upstream["local_end_z_mm"])
        )
        required_clearance = (
            5.0
            if (
                upstream_key == "diffraction_lens"
                and downstream_key == "intermediate_lens"
            )
            else 0.0
        )
        if clearance < required_clearance - tolerance:
            raise ValueError(
                f"Insufficient mechanical clearance between {upstream_key} "
                f"and {downstream_key}: {clearance:.9g} mm; "
                f"requires at least {required_clearance:.9g} mm"
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
        lens_start = float(lens["local_start_z_mm"])
        lens_center = float(lens["local_center_z_mm"])
        lens_end = float(lens["local_end_z_mm"])
        gap = float(lens["pole_gap_mm"])
        expected_upper_end = lens_center - 0.5 * gap
        expected_lower_start = lens_center + 0.5 * gap
        checks = (
            (float(upper["local_start_z_mm"]), lens_start),
            (float(upper["local_end_z_mm"]), expected_upper_end),
            (float(lower["local_start_z_mm"]), expected_lower_start),
            (float(lower["local_end_z_mm"]), lens_end),
        )
        if gap <= 0.0 or any(
            abs(actual - expected) > tolerance
            for actual, expected in checks
        ):
            raise ValueError(
                f"{lens_key} pole pieces must bound its declared pole gap"
            )


def _validate_magnetic_lens_mechanical_parts(parts):
    """Validate the independent housing/yoke/coil geometry of each lens."""

    tolerance = 1.0e-9
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
    for lens_key in lens_keys:
        lens = by_key[lens_key]
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
        radial_ranges = []
        for child_key, (role, profile) in zip(child_keys, expected):
            child = by_key[child_key]
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
            radial_ranges.append((inner, outer, child_key))
        for first, second in zip(radial_ranges, radial_ranges[1:]):
            if second[1] > first[0] + tolerance:
                raise ValueError(
                    f"Magnetic-lens radial layers overlap: "
                    f"{first[2]} and {second[2]}"
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


def snapshot_manifest_texts(module_paths, root=None):
    root = Path(root) if root is not None else MODULE_ROOT
    return {
        str(module_path): (root / module_path).read_text(encoding="utf-8")
        for module_path in module_paths
    }


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
