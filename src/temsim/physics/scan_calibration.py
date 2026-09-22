"""Persistent drive calibration, not an independent downstream beam state."""
from __future__ import annotations

import json
import numpy as np


def held(state):
    mode = state.ac_deflector.calibration_mode
    if mode not in ("automatic", "held"):
        raise ValueError("Scan calibration mode must be automatic or held")
    return mode == "held"


def sample_reference_z_mm(state):
    reference = state.ac_deflector.scan_reference
    if reference == "sample_centre":
        return float(state.sample.z_mm)
    if reference == "sample_entrance":
        return float(state.sample.upper_surface_z_mm)
    raise ValueError("Unknown scan reference; choose specimen centre or entrance")


def validate_record(text):
    if not text:
        raise ValueError("No held scan calibration. Use Calibrate and hold first.")
    record = json.loads(text)
    if (not isinstance(record, dict) or type(record.get("version")) is not int
            or record["version"] != 1):
        raise ValueError("Unsupported scan calibration record")
    for key in ("ac_ratio", "descan_ratio", "command_mrad"):
        values = np.asarray(record.get(key), dtype=float)
        if values.shape != (2, 2) or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid held scan matrix: {key}")
    fov = np.asarray(record.get("fov_nm"), dtype=float)
    if fov.shape != (2,) or not np.all(np.isfinite(fov)) or np.any(fov <= 0):
        raise ValueError("Held scan FOV must have two finite positive extents")
    if not np.isfinite(float(record.get("target_z_mm", float("nan")))):
        raise ValueError("Held descan target must be finite")
    if not isinstance(record.get("descan_calibrated"), bool):
        raise ValueError("Held scan record must declare descan calibration status")
    target_key = record.get("target_key")
    if not isinstance(target_key, str):
        raise ValueError("Held scan record must declare its physical descan target key")
    if record["descan_calibrated"] and (
        not target_key.strip() or target_key == "legacy_image_reference"
    ):
        raise ValueError("Held descan calibration requires an explicit physical target key")
    if record.get("reference") not in ("sample_centre", "sample_entrance"):
        raise ValueError("Invalid held scan reference")
    return record


def capture_record(state, *, descan_calibrated: bool):
    """Capture only the calibration stages executed in the accepted solve."""
    from temsim.instrument_snapshot import capture_instrument_snapshot
    ac, descan = state.ac_deflector, state.descan_deflector
    record = dict(version=1, calibrated_snapshot_id=capture_instrument_snapshot(state).digest,
                  ac_ratio=ac.pure_shift_lower_ratio_matrix,
                  descan_ratio=descan.image_plane_lower_ratio_matrix,
                  command_mrad=ac.scan_command_matrix_mrad,
                  fov_nm=(ac.scan_field_of_view_x_nm, ac.scan_field_of_view_y_nm),
                  target_key=descan.image_plane_target_key,
                  target_z_mm=descan.image_plane_target_z_mm or float(state.sample.z_mm),
                  descan_calibrated=descan_calibrated,
                  reference=ac.scan_reference)
    encoded = json.dumps(record, sort_keys=True, allow_nan=False, separators=(",", ":"))
    validate_record(encoded)
    ac.calibration_record_json = encoded
    return encoded


def restore_held(state):
    """Restore fixed drive ratios, scaling only the requested raster extent."""
    ac, descan = state.ac_deflector, state.descan_deflector
    record = validate_record(ac.calibration_record_json)
    if record["reference"] != ac.scan_reference:
        raise ValueError("Scan reference changed. Calibrate and hold at the new specimen plane.")
    if descan.enabled and descan.scan_enabled and not record["descan_calibrated"]:
        raise ValueError("Descan was not calibrated. Enable it and use Calibrate and hold.")
    fov = np.array((ac.scan_field_of_view_x_nm, ac.scan_field_of_view_y_nm))
    command = np.asarray(record["command_mrad"]) * (fov / record["fov_nm"])[None, :]
    snapshots = [dict(c.__dict__) for c in (ac, descan)]
    try:
        ac.set_pure_shift_coupling(record["ac_ratio"])
        ac.set_scan_command_matrix_mrad(command)
        descan.set_image_plane_coupling(record["descan_ratio"],
            target_key=record["target_key"], target_z_mm=record["target_z_mm"])
        descan.set_scan_command_matrix_mrad(-command)
        ac.validate()
        descan.validate()
    except Exception:
        for component, saved in zip((ac, descan), snapshots):
            component.__dict__.clear()
            component.__dict__.update(saved)
        raise
    return command
