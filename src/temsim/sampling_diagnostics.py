"""Scalar observations of an executed population, never a replacement source."""
from __future__ import annotations

import math
import numpy as np

from temsim.immutable_json import freeze_json
from temsim.physics.beam_statistics import transverse_beam_statistics


def sampling_summary(arrays, *, plane_z_mm, source_current_a=None):
    """Weights retain their emitted-population normalization after clipping.

    N_eff describes concentration only. No IID errors or convergence claims
    are inferred from a deterministic population or from an empty sample.
    """
    result = dict(plane_z_mm=float(plane_z_mm), population="positive-weight incident particles at the stated plane",
        status="NOT_COMPUTED", emitted_samples=None, transmitted_samples=None,
        source_current_a=source_current_a, plane_current_a=None, transmission=None,
        effective_samples=None, maximum_weight_fraction=None, centroid_x_m=None,
        centroid_y_m=None, diameter95_m=None, alpha95_rad=None,
        reason="No retained incident population", convergence_status="NOT_RUN")
    if source_current_a is not None and (not math.isfinite(source_current_a) or source_current_a < 0):
        raise ValueError("Captured source current must be finite and non-negative")
    required = ("x_m", "y_m", "tx_rad", "ty_rad", "weight", "alive")
    if not all(key in arrays for key in required):
        return freeze_json(result)
    values = [np.asarray(arrays[key]) for key in required[:4]]
    weights, alive = np.asarray(arrays["weight"], float), np.asarray(arrays["alive"], bool)
    if weights.ndim != 1 or any(a.shape != weights.shape for a in (*values, alive)):
        raise ValueError("Sampling diagnostics require an aligned full population")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Sampling weights must be finite and non-negative")
    result["emitted_samples"] = len(weights)
    selected = alive & (weights > 0)
    result["transmitted_samples"] = int(np.count_nonzero(selected))
    total = float(np.sum(weights))
    if not math.isfinite(total):
        raise ValueError("Total sampling weight must be finite")
    if total <= 0:
        result.update(status="ZERO_TOTAL_WEIGHT", reason="No positive source weight; current fraction is undefined")
        return freeze_json(result)
    if any(np.any(~np.isfinite(a[selected])) for a in values):
        result.update(status="NUMERICAL_FAILURE", reason="Nonfinite coordinates in surviving positive-weight rays")
        return freeze_json(result)
    surviving = float(weights[selected].sum())
    fraction = surviving / total
    result.update(transmission=fraction,
                  plane_current_a=None if source_current_a is None else source_current_a * fraction)
    if surviving <= 0:
        result.update(status="NO_SAMPLED_SURVIVORS",
                      reason="No sampled transmission; this does not prove complete physical blockage")
        return freeze_json(result)
    normalized = weights[selected] / surviving
    stats = transverse_beam_statistics(*values, alive=selected, weights=weights)
    result.update(status="AVAILABLE", reason="N_eff is weight concentration, not convergence or an error bar",
        effective_samples=float(1.0 / np.sum(normalized**2)),
        maximum_weight_fraction=float(normalized.max()), centroid_x_m=stats.mean_x_m,
        centroid_y_m=stats.mean_y_m, diameter95_m=2 * stats.radius_95_m,
        alpha95_rad=stats.convergence_95_rad)
    return freeze_json(result)


def checkpoint_sampling_summary(checkpoint):
    from temsim.working_point_archive import WorkingPointArchiveIndex
    if checkpoint.is_metadata_only:
        result = dict(sampling_summary({}, plane_z_mm=checkpoint.plane_z_mm, source_current_a=None))
        summary = checkpoint.metadata.get("unverified_scalar_summary", {})
        for key in result:
            value = summary.get(key)
            if key not in {"status", "reason", "convergence_status", "plane_z_mm", "population"} and isinstance(value, (int, float)) and math.isfinite(value):
                result[key] = value
        result.update(status="METADATA_ONLY", reason="Historical scalar summary; no input assets or results to verify")
        return freeze_json(result)
    if isinstance(checkpoint, WorkingPointArchiveIndex):
        result = dict(sampling_summary({}, plane_z_mm=checkpoint.plane_z_mm,
                                       source_current_a=checkpoint.metadata.get("source_current_a")))
        # Persisted scalars help navigation, but are not accepted evidence until
        # the exact numeric payload is loaded and its checksums are verified.
        for key in result:
            value = checkpoint.index_summary.get(key)
            if key not in {"status", "reason", "convergence_status", "plane_z_mm", "population"} and isinstance(value, (int, float)) and math.isfinite(value):
                result[key] = value
        result.update(status="INDEX_ONLY", reason="Archived scalar index; load retained data to verify numeric products")
        return freeze_json(result)
    return sampling_summary(checkpoint.arrays, plane_z_mm=checkpoint.plane_z_mm,
                            source_current_a=checkpoint.metadata.get("source_current_a"))


def working_point_description(checkpoint, *, current_implementation):
    """Read raw captured labels without restoring state or evaluating optics.

    A matching implementation alone does not check external assets. Exact
    restore still performs the existing complete compatibility checks.
    Historical validation labels never confer current qualification.
    """
    graph = checkpoint.snapshot.graph
    nodes = graph["nodes"]
    root = nodes[graph["root"]["ref"]]["attributes"]
    gun_ref = root.get("electron_gun", {})
    gun = nodes[gun_ref["ref"]] if "ref" in gun_ref else {}
    assembly_ref = root.get("_resolved_assembly", {})
    assembly = nodes[assembly_ref["ref"]].get("attributes", {}) if "ref" in assembly_ref else {}
    paths = assembly.get("selected_module_paths", {}).get("tuple", ())
    assembly_label = " / ".join(str(row["tuple"][1]).replace("\\", "/").rsplit("/", 1)[-1]
                                for row in paths if "tuple" in row)
    same = checkpoint.snapshot.implementation == current_implementation
    return freeze_json(dict(source=gun.get("type", "unknown").split(":")[-1],
        assembly=assembly_label or "Unspecified assembly", mode=str(root.get("illumination_mode", "unknown")),
        simulation_mode=str(root.get("simulation_mode", "unknown")),
        quality=str(checkpoint.metadata.get("quality", "Unrecorded")),
        date=str(checkpoint.metadata.get("created_at_utc", "Unrecorded")),
        execution="METADATA_ONLY" if checkpoint.is_metadata_only else "INPUTS_ONLY" if checkpoint.is_input_design else "RETAINED_RESULT" if checkpoint.has_retained_payload else "NO_RETAINED_PAYLOAD",
        numerical="NOT_RUN", physical_validation="NOT_ESTABLISHED",
        compatibility="READ_ONLY_METADATA; assets absent" if checkpoint.is_metadata_only else "IMPLEMENTATION_MATCH; assets checked on restore" if same else "HISTORICAL_IMPLEMENTATION",
        wave_capability="PAUSED: coherent tip-to-column development is paused",
        snapshot_id=checkpoint.snapshot.digest, checkpoint_id=checkpoint.digest))
