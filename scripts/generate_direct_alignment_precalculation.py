"""Generate validated Direct Alignment points and local response ratios."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from temsim.assembly_catalog import AssemblyCatalog
from temsim.operating_modes import (
    apply_operating_mode_pair,
    direct_alignment_by_key,
)
from temsim.optics.column import default_state
from temsim.optics.direct_alignment import (
    CONDENSER_KEYS,
    _get_vector,
    _validate_condenser_production,
    _validate_projector,
    _validate_projector_production,
    apply_direct_alignment,
    diffraction_reference_plane,
)
from temsim.optics.direct_alignment_precalibration import (
    precalculated_alignment_points,
    precalculated_alignment_ratios,
)


DEFAULT_OUTPUT = ROOT / "outputs" / "direct_alignment_precalculation"


def configuration_fingerprint() -> str:
    """Hash every TOML input that determines the assembled optical model."""

    digest = sha256()
    paths = sorted((ROOT / "configs").rglob("*.toml"))
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _copy_state(state):
    return type(state).from_dict(state.to_dict())


def _relative_log_error(achieved: float, target: float) -> float:
    return abs(math.log(max(float(achieved), 1.0e-15) / float(target)))


def _projector_table(base, key: str, probe_mode: str) -> dict[str, Any]:
    definition = direct_alignment_by_key(key)
    state = _copy_state(base)
    projector_mode = "imaging" if key == "image_magnification" else "diffraction"
    apply_operating_mode_pair(state, probe_mode, projector_mode)
    optimiser_step = float(definition.targets["optimiser_step_mm"])
    validation_step = float(definition.targets["validation_step_mm"])
    maximum_relative_error = float(
        definition.targets["maximum_relative_error"]
    )
    points = []
    for point in precalculated_alignment_points(definition):
        vector = np.asarray(point.strengths, dtype=float)
        coarse = _validate_projector(
            state, definition, vector, optimiser_step
        )
        fine = _validate_projector_production(
            state, definition, vector, validation_step
        )
        numerical_spread = abs(fine.value - coarse.value) / max(
            abs(fine.value), 1.0e-15
        )
        if key == "image_magnification":
            constraint_limit = float(
                definition.targets["maximum_relay_error_um"]
            )
        else:
            constraint_limit = float(
                definition.targets[
                    "maximum_diffraction_conjugacy_residual"
                ]
            )
        valid = (
            _relative_log_error(fine.value, point.target)
            <= maximum_relative_error
            and abs(fine.constraint_value) <= constraint_limit
            and numerical_spread
            <= float(definition.targets["maximum_numerical_spread"])
        )
        points.append({
            "target": point.target,
            "achieved": fine.value,
            "unit": point.unit,
            "branch": point.branch,
            "relative_log_error": _relative_log_error(
                fine.value, point.target
            ),
            "constraint_value": fine.constraint_value,
            "constraint_unit": fine.constraint_unit,
            "numerical_spread": numerical_spread,
            "valid": bool(valid),
            "strengths_percent": {
                device: value
                for device, value in zip(definition.devices, point.strengths)
            },
        })
    reference = None
    if key == "diffraction_camera_length":
        reference_key, reference_z_mm = diffraction_reference_plane(state)
        reference = {"key": reference_key, "z_mm": reference_z_mm}
    return {
        "key": key,
        "mode": f"{probe_mode}+{projector_mode}",
        "reference_plane": reference,
        "interpolation": "linear lens strength in log10(target)",
        "commit_policy": "seed only; full production validation required",
        "devices": list(definition.devices),
        "points": points,
        "ratios": [
            {
                "branch": row.branch,
                "lower_target": row.lower_target,
                "upper_target": row.upper_target,
                "target_ratio": row.target_ratio,
                "log10_span": row.log10_span,
                "strength_ratios": {
                    device: value
                    for device, value in zip(
                        definition.devices, row.strength_ratios
                    )
                },
                "strength_delta_percent_per_target_decade": {
                    device: value
                    for device, value in zip(
                        definition.devices,
                        row.strength_delta_per_decade,
                    )
                },
            }
            for row in precalculated_alignment_ratios(definition)
        ],
    }


def _nanoprobe_aperture_table(base) -> dict[str, Any]:
    definition = direct_alignment_by_key("nanoprobe_convergence")
    state = _copy_state(base)
    apply_operating_mode_pair(state, "nano_probe", "imaging")
    vector = _get_vector(state, CONDENSER_KEYS)
    step_mm = float(definition.targets["validation_step_mm"])
    diameters_um = np.linspace(10.0, 200.0, 20)
    points = []
    for diameter_um in diameters_um:
        state.condenser_aperture_2.diameter_um = float(diameter_um)
        measured = _validate_condenser_production(
            state, definition, vector, step_mm
        )
        nominal = (
            float(definition.targets["aperture_scaling_reference_mrad_per_um"])
            * float(diameter_um)
        )
        scaling_valid = (
            abs(measured.value - nominal)
            <= float(
                definition.targets[
                    "aperture_scaling_maximum_absolute_error_mrad"
                ]
            )
        )
        focus_valid = (
            abs(measured.constraint_value)
            <= float(definition.targets["maximum_waist_offset_mm"])
        )
        points.append({
            "aperture_diameter_um": float(diameter_um),
            "nominal_target_mrad": nominal,
            "achieved_mrad": measured.value,
            "angle_per_aperture_mrad_per_um": (
                measured.value / float(diameter_um)
            ),
            "waist_offset_mm": measured.constraint_value,
            "scaling_valid": bool(scaling_valid),
            "focus_valid": bool(focus_valid),
            "valid": bool(scaling_valid and focus_valid),
            "strengths_percent": {
                device: float(value)
                for device, value in zip(CONDENSER_KEYS, vector)
            },
        })
    ratios = []
    for left, right in zip(points[:-1], points[1:]):
        aperture_delta = (
            right["aperture_diameter_um"]
            - left["aperture_diameter_um"]
        )
        ratios.append({
            "lower_aperture_diameter_um": left["aperture_diameter_um"],
            "upper_aperture_diameter_um": right["aperture_diameter_um"],
            "aperture_ratio": (
                right["aperture_diameter_um"]
                / left["aperture_diameter_um"]
            ),
            "achieved_angle_ratio": (
                right["achieved_mrad"] / left["achieved_mrad"]
            ),
            "local_slope_mrad_per_um": (
                (right["achieved_mrad"] - left["achieved_mrad"])
                / aperture_delta
            ),
        })
    errors = np.asarray([
        point["achieved_mrad"] - point["nominal_target_mrad"]
        for point in points
    ])
    return {
        "key": definition.key,
        "mode": "nano_probe+imaging",
        "interpolation": "linear in aperture diameter at fixed C2/C3",
        "commit_policy": (
            "calibration relation only; aperture is not silently changed by "
            "the C2/C3 Direct Alignment control"
        ),
        "devices": list(CONDENSER_KEYS),
        "dense_grid_summary": {
            "rms_error_mrad": float(np.sqrt(np.mean(errors**2))),
            "maximum_absolute_error_mrad": float(
                np.max(np.abs(errors))
            ),
            "minimum_angle_per_aperture_mrad_per_um": min(
                point["angle_per_aperture_mrad_per_um"]
                for point in points
            ),
            "maximum_angle_per_aperture_mrad_per_um": max(
                point["angle_per_aperture_mrad_per_um"]
                for point in points
            ),
        },
        "points": points,
        "ratios": ratios,
    }


def _nanoprobe_solved_table(base) -> dict[str, Any]:
    """Solve focused C2/C3 points along the calibrated aperture path."""

    definition = direct_alignment_by_key("nanoprobe_convergence")
    state = _copy_state(base)
    apply_operating_mode_pair(state, "nano_probe", "imaging")
    reference = float(
        definition.targets["aperture_scaling_reference_mrad_per_um"]
    )
    diameters_um = (
        10.0, 20.0, 40.0, 60.0, 80.0, 100.0,
        120.0, 140.0, 160.0, 180.0, 200.0,
    )
    points = []
    for diameter_um in diameters_um:
        state.condenser_aperture_2.diameter_um = diameter_um
        target = reference * diameter_um
        started = perf_counter()
        result = apply_direct_alignment(
            state, definition.key, target, definition=definition
        )
        points.append({
            "aperture_diameter_um": diameter_um,
            "target_mrad": target,
            "achieved_mrad": result.achieved,
            "relative_log_error": _relative_log_error(
                result.achieved, target
            ),
            "waist_offset_mm": result.constraint_value,
            "numerical_spread": result.numerical_spread,
            "iterations": result.iterations,
            "solve_seconds": perf_counter() - started,
            "valid": bool(result.success),
            "strengths_percent": dict(result.strengths),
        })
    ratios = []
    valid_points = [point for point in points if point["valid"]]
    for left, right in zip(valid_points[:-1], valid_points[1:]):
        target_delta = right["target_mrad"] - left["target_mrad"]
        aperture_delta = (
            right["aperture_diameter_um"]
            - left["aperture_diameter_um"]
        )
        strength_ratios = {}
        strength_delta_per_mrad = {}
        strength_delta_per_aperture_um = {}
        for device in CONDENSER_KEYS:
            lower_value = left["strengths_percent"][device]
            upper_value = right["strengths_percent"][device]
            strength_ratios[device] = (
                None
                if abs(lower_value) <= 1.0e-12
                else upper_value / lower_value
            )
            strength_delta_per_mrad[device] = (
                (upper_value - lower_value) / target_delta
            )
            strength_delta_per_aperture_um[device] = (
                (upper_value - lower_value) / aperture_delta
            )
        ratios.append({
            "lower_aperture_diameter_um": left["aperture_diameter_um"],
            "upper_aperture_diameter_um": right["aperture_diameter_um"],
            "lower_target_mrad": left["target_mrad"],
            "upper_target_mrad": right["target_mrad"],
            "aperture_ratio": (
                right["aperture_diameter_um"]
                / left["aperture_diameter_um"]
            ),
            "target_ratio": right["target_mrad"] / left["target_mrad"],
            "achieved_angle_ratio": (
                right["achieved_mrad"] / left["achieved_mrad"]
            ),
            "strength_ratios": strength_ratios,
            "strength_delta_percent_per_mrad": strength_delta_per_mrad,
            "strength_delta_percent_per_aperture_um": (
                strength_delta_per_aperture_um
            ),
        })
    return {
        "key": definition.key,
        "mode": "nano_probe+imaging",
        "interpolation": (
            "bilinear path coordinate: aperture diameter and target angle; "
            "seed only"
        ),
        "commit_policy": "full production focus validation required",
        "devices": list(CONDENSER_KEYS),
        "solve_summary": {
            "valid_points": sum(point["valid"] for point in points),
            "total_points": len(points),
            "total_solve_seconds": sum(
                point["solve_seconds"] for point in points
            ),
        },
        "points": points,
        "ratios": ratios,
    }


def generate() -> dict[str, Any]:
    base = default_state()
    assembly_catalog = AssemblyCatalog()
    assembly_catalog.apply(base, assembly_catalog.default_selection())
    return {
        "format_version": 1,
        "calibration_status": "provisional_non_oem_engineering_model",
        "source_fingerprint_sha256": configuration_fingerprint(),
        "tables": {
            "nanoprobe_aperture_scaling": _nanoprobe_aperture_table(base),
            "nanoprobe_solved_continuation": _nanoprobe_solved_table(base),
            "image_magnification": _projector_table(
                base, "image_magnification", "micro_probe"
            ),
            "tem_diffraction_camera_length": _projector_table(
                base, "diffraction_camera_length", "micro_probe"
            ),
        },
    }


def _write_points_csv(document: dict[str, Any], path: Path) -> None:
    rows = []
    for table_name, table in document["tables"].items():
        for point in table["points"]:
            row = {"table": table_name, **point}
            strengths = row.pop("strengths_percent", {})
            row.update({f"strength_{key}_percent": value for key, value in strengths.items()})
            rows.append(row)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_ratios_csv(document: dict[str, Any], path: Path) -> None:
    rows = []
    for table_name, table in document["tables"].items():
        for ratio in table["ratios"]:
            row = {"table": table_name, **ratio}
            strength_ratios = row.pop("strength_ratios", {})
            deltas = row.pop(
                "strength_delta_percent_per_target_decade", {}
            )
            delta_per_mrad = row.pop(
                "strength_delta_percent_per_mrad", {}
            )
            delta_per_aperture = row.pop(
                "strength_delta_percent_per_aperture_um", {}
            )
            row.update({f"ratio_{key}": value for key, value in strength_ratios.items()})
            row.update({f"delta_per_decade_{key}": value for key, value in deltas.items()})
            row.update({f"delta_per_mrad_{key}": value for key, value in delta_per_mrad.items()})
            row.update({f"delta_per_aperture_um_{key}": value for key, value in delta_per_aperture.items()})
            rows.append(row)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    document = generate()
    json_path = output_dir / "direct_alignment_precalculation.json"
    points_path = output_dir / "direct_alignment_points.csv"
    ratios_path = output_dir / "direct_alignment_ratios.csv"
    json_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_points_csv(document, points_path)
    _write_ratios_csv(document, ratios_path)
    print(json.dumps({
        "json": str(json_path),
        "points_csv": str(points_path),
        "ratios_csv": str(ratios_path),
        "fingerprint": document["source_fingerprint_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
