"""Read-only diagnostics from retained, current-weighted incident rays.

No live State, lens evaluator, auto-focus or sample-wave reconstruction is used.
Angles follow beam_statistics: paraxial mechanical slopes relative to the
weighted chief ray, in the laboratory transverse basis (+Z downstream).
"""
from __future__ import annotations

import math
import numpy as np

from temsim.immutable_json import freeze_json
from temsim.physics.beam_statistics import transverse_beam_statistics

OBSERVABLE_SCHEMA = "weighted-incident-observables-v1"


def incident_checkpoint_observables(checkpoints, *, alive, weights):
    """Return frozen records; finite sample maximum is not a physical edge."""
    if checkpoints is None or len(checkpoints.z_mm) == 0:
        raise ValueError("An incident checkpoint plane is required")
    weights = np.asarray(weights, float)
    alive = np.asarray(alive, bool)
    shape = checkpoints.x_m[-1].shape
    if weights.shape != shape or alive.shape != shape:
        raise ValueError("Checkpoint weights and survival mask must align with rays")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Checkpoint current weights must be finite and non-negative")
    source_weight = float(weights.sum())
    if source_weight <= 0:
        raise ValueError("Checkpoint source weight must be positive")
    mask = alive.copy()
    values = [getattr(checkpoints, name)[-1] for name in ("x_m", "y_m", "tx_rad", "ty_rad")]
    for value in values:
        mask &= np.isfinite(value)
    records = {}

    def record(key, value, unit, definition, reason=""):
        available = value is not None and math.isfinite(float(value))
        records[key] = {"value": float(value) if available else None, "unit": unit,
                        "status": "AVAILABLE" if available else "UNAVAILABLE",
                        "definition_id": definition, "reason": reason}

    surviving_weight = float(weights[mask].sum())
    record("source_fraction", surviving_weight / source_weight, "1", "incident-surviving-current/source-current-v1")
    stats = transverse_beam_statistics(*values, alive=mask, weights=weights) if surviving_weight > 0 else None
    for key, attribute, unit, definition in (
        ("alpha95", "convergence_95_rad", "rad", "chief-ray-current-contained-semiangle-95-v1"),
        ("alpha99", "convergence_99_rad", "rad", "chief-ray-current-contained-semiangle-99-v1"),
        ("sampled_max_angle", "convergence_edge_rad", "rad", "chief-ray-finite-sample-maximum-v1"),
        ("radius95", "radius_95_m", "m", "chief-ray-current-contained-radius-95-v1"),
        ("centre_x", "mean_x_m", "m", "current-weighted-centre-x-v1"),
        ("centre_y", "mean_y_m", "m", "current-weighted-centre-y-v1"),
    ):
        record(key, getattr(stats, attribute) if stats is not None else None,
               unit, definition, "" if stats is not None else "No surviving current")
    record("physical_angular_edge", None, "rad", "physical-pupil-support-edge-v1",
           "Not inferable from a finite ray population")
    return freeze_json({"schema": OBSERVABLE_SCHEMA, "plane_z_mm": float(checkpoints.z_mm[-1]),
                        "basis": "laboratory mechanical slopes; +Z downstream; paraxial",
                        "validation_status": "NOT_RUN", "records": records})
