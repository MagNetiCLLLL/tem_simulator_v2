"""Current non-scalar gun controls, independent of TOML electrode geometry.

These records describe tip quadrature and the existing accelerator's voltage
fractions and field-edge widths. They never introduce a downstream source.
"""
from __future__ import annotations

from copy import copy
from dataclasses import asdict, dataclass
import math

from .emitter import ColdFieldEmitter, EmissionQuadrature


@dataclass(frozen=True, slots=True)
class PreparedGunProfileControls:
    gun_type_key: str
    accelerator_key: str
    tip_quadrature: EmissionQuadrature | None
    stage_values: tuple[tuple[float, float], ...]


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _expected_keys(gun):
    if gun.type_key not in {"cold_feg", "thermionic"}:
        raise ValueError("Unsupported electron gun for operating-profile controls")
    keys = {"accelerator_stages"}
    if gun.type_key == "cold_feg":
        if not isinstance(gun.emitter, ColdFieldEmitter):
            raise ValueError("Cold FEG profiles require the physical tip emitter")
        keys.add("tip_quadrature")
    return keys


def capture_gun_profile_controls(gun) -> dict:
    """Capture validated controls; an empty tip table explicitly clears quadrature."""
    keys = _expected_keys(gun)
    payload = {"accelerator_stages": [
        {"voltage_fraction": stage.voltage_fraction, "soft_edge_mm": stage.soft_edge_mm}
        for stage in gun.accelerator.stages
    ]}
    if "tip_quadrature" in keys:
        quadrature = gun.emitter.quadrature
        payload["tip_quadrature"] = {} if quadrature is None else asdict(quadrature)
    prepare_gun_profile_controls(gun, payload)
    return payload


def prepare_gun_profile_controls(gun, payload, *, emitter=None, accelerator=None) -> PreparedGunProfileControls:
    """Validate on detached candidates without changing the supplied gun.

Callers restoring scalar controls in the same transaction should pass their
pending emitter/accelerator candidates, so validation uses the final coupled
parameters rather than whichever values happen to be currently installed.
"""
    keys = _expected_keys(gun)
    if not isinstance(payload, dict) or set(payload) != keys:
        raise ValueError("Gun profile controls must contain exactly the current control tables")
    incoming_emitter = copy(gun.emitter if emitter is None else emitter)
    incoming_accelerator = copy(gun.accelerator if accelerator is None else accelerator)
    quadrature = None
    if "tip_quadrature" in keys:
        row = payload["tip_quadrature"]
        if not isinstance(row, dict) or (row and set(row) != {"spatial", "directions", "energies", "schema"}):
            raise ValueError("Tip quadrature must be empty or a complete current factor table")
        if row:
            quadrature = EmissionQuadrature(**row).validate()
        incoming_emitter.quadrature = quadrature
        incoming_emitter.validate()
        if quadrature is not None and incoming_emitter.ray_count != quadrature.total:
            raise ValueError("Tip ray count must equal the complete product quadrature")
    rows = payload["accelerator_stages"]
    if not isinstance(rows, list) or len(rows) != len(gun.accelerator.stages):
        raise ValueError("Accelerator stage controls must match the installed stage count")
    if len(incoming_accelerator.stages) != len(gun.accelerator.stages):
        raise ValueError("Pending accelerator stage count disagrees with installed geometry")
    stages = []
    for index, (original, row) in enumerate(zip(incoming_accelerator.stages, rows)):
        if not isinstance(row, dict) or set(row) != {"voltage_fraction", "soft_edge_mm"}:
            raise ValueError("Accelerator stages accept only voltage_fraction and soft_edge_mm controls")
        stage = copy(original)
        stage.voltage_fraction = _finite_number(row["voltage_fraction"], f"Accelerator stage {index} fraction")
        stage.soft_edge_mm = _finite_number(row["soft_edge_mm"], f"Accelerator stage {index} edge width")
        stages.append(stage)
    incoming_accelerator.stages = stages
    grounded = getattr(incoming_emitter, "surface_model", None) is not None
    incoming_accelerator.validate(grounded=grounded)
    if grounded and stages[-1].voltage_fraction != 1.0:
        raise ValueError("Grounded gun requires the final accelerator fraction to equal one")
    return PreparedGunProfileControls(gun.type_key, gun.accelerator.key, quadrature,
        tuple((stage.voltage_fraction, stage.soft_edge_mm) for stage in stages))


def apply_prepared_gun_profile_controls(gun, prepared: PreparedGunProfileControls) -> None:
    """Commit validated controls while retaining emitter/stage object identities."""
    if not isinstance(prepared, PreparedGunProfileControls):
        raise TypeError("Expected prepared gun profile controls")
    if (gun.type_key != prepared.gun_type_key
            or gun.accelerator.key != prepared.accelerator_key
            or len(gun.accelerator.stages) != len(prepared.stage_values)):
        raise ValueError("Prepared gun controls do not match the installed accelerator")
    if gun.type_key == "cold_feg":
        gun.emitter.quadrature = prepared.tip_quadrature
    for stage, (fraction, edge) in zip(gun.accelerator.stages, prepared.stage_values):
        stage.voltage_fraction = fraction
        stage.soft_edge_mm = edge
