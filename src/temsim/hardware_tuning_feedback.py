"""Read-only observations of retained particle histories for manual tuning.

No instrument restoration, field preparation or transport is performed here.
Statistics use the same weighted histories as Beam analysis; interpolated plot
histories remain display diagnostics, not full-precision continuation states.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from temsim.gui.beam_plane_data import sample_beam_plane
from temsim.hardware_tuning import resolve_task
from temsim.physics.beam_statistics import transverse_beam_statistics


@dataclass(frozen=True, slots=True)
class HardwareObservation:
    result_id: str
    model_id: str
    manifest_id: str
    z_mm: float
    status: str
    reason: str
    population: str = "Unavailable"
    ray_count: int = 0
    source_fraction: float | None = None
    current_pa: float | None = None
    centroid_x_um: float | None = None
    centroid_y_um: float | None = None
    direction_x_mrad: float | None = None
    direction_y_mrad: float | None = None
    rms_radius_um: float | None = None
    diameter95_um: float | None = None
    angular_rms_mrad: float | None = None
    angular95_mrad: float | None = None
    provenance: str = (
        "Retained history; linear interpolation between saved planes when needed. "
        "Not an integration checkpoint or numerical-convergence qualification."
    )


def observe_retained_beam(result, z_mm: float) -> HardwareObservation:
    """Return scalar readout at Z without changing or recalculating the result."""
    z = float(z_mm)
    if not math.isfinite(z):
        raise ValueError("Observation Z must be finite")
    signatures = getattr(result, "signatures", {})
    identity = signatures.get("request") if isinstance(signatures, dict) else None
    valid_identity = isinstance(identity, str) and bool(identity.strip())
    common = dict(result_id=identity if valid_identity else "UNRECORDED",
                  model_id=str(getattr(result, "model_signature", "") or "UNRECORDED"),
                  manifest_id=str(getattr(getattr(result, "calculation_manifest", None), "digest", None) or "UNRECORDED"),
                  z_mm=z)
    if result is None or getattr(result, "state_snapshot", None) is None:
        return HardwareObservation(**common, status="UNAVAILABLE", reason="No captured execution snapshot is available.")
    if not isinstance(signatures, dict) or not valid_identity:
        return HardwareObservation(**common, status="UNAVAILABLE", reason="Captured request identity is invalid.")
    try:
        plane = sample_beam_plane(result, z)
    except (ValueError, TypeError, AttributeError, IndexError, OverflowError) as exc:
        return HardwareObservation(**common, status="UNAVAILABLE", reason=f"Retained beam cannot be read: {exc}")
    common.update(population=plane.provenance, ray_count=plane.ray_count)
    diagnostics = " ".join(plane.diagnostics)
    if not plane.weights_valid:
        return HardwareObservation(**common, status="UNAVAILABLE", reason=f"{plane.status}. {diagnostics}".strip())
    common.update(source_fraction=plane.total_source_fraction, current_pa=plane.current_pa)
    if plane.ray_count == 0:
        return HardwareObservation(**common, status="NO_SAMPLED_SURVIVORS",
                                   reason="No positive-weight rays reach this plane in the retained population. "
                                          "This is not proof of complete physical blockage. " + diagnostics)
    try:
        stats = transverse_beam_statistics(plane.x_m, plane.y_m, plane.tx, plane.ty,
                                          weights=plane.source_fraction)
    except (ValueError, TypeError, OverflowError) as exc:
        return HardwareObservation(**common, status="UNAVAILABLE", reason=f"Weighted statistics unavailable: {exc}")
    common.update(centroid_x_um=stats.mean_x_m * 1e6, centroid_y_um=stats.mean_y_m * 1e6,
                  # Stored chief values are slopes, not angles. Convert the
                  # normalized mean direction to projected angles explicitly.
                  direction_x_mrad=math.atan(stats.mean_tx_rad) * 1e3,
                  direction_y_mrad=math.atan(stats.mean_ty_rad) * 1e3)
    if plane.ray_count < 2:
        return HardwareObservation(**common, status="INSUFFICIENT_POPULATION",
                                   reason="Only one positive-weight path; centroid/direction are available, "
                                          "but beam-width and angular-spread readouts require at least two. " + diagnostics)
    return HardwareObservation(**common, status="AVAILABLE", reason=diagnostics,
                               rms_radius_um=stats.radius_rms_m * 1e6,
                               diameter95_um=stats.illumination_diameter_95_um,
                               angular_rms_mrad=stats.convergence_rms_rad * 1e3,
                               angular95_mrad=stats.convergence_95_mrad)


def captured_hardware_values(state, task_key: str):
    """Detached values with stable physical component/field identities."""
    if state is None:
        return ()
    return tuple((group.target.key, field.name, group.binding.label, field.label,
                  field.unit, getattr(group.target.obj, field.name))
                 for group in resolve_task(state, task_key) if group.target is not None
                 for field in group.fields)


def baseline_difference(current: HardwareObservation, baseline: HardwareObservation):
    """Descriptive differences only; no claim that other controls were held fixed."""
    if current.z_mm != baseline.z_mm:
        return None, "Baseline uses a different observation plane."
    if current.population != baseline.population:
        return None, "Baseline uses a different population definition."
    fields = ("centroid_x_um", "centroid_y_um", "direction_x_mrad", "direction_y_mrad")
    if any(getattr(row, key) is None for row in (current, baseline) for key in fields):
        return None, "Baseline or current centroid/direction is unavailable."
    return {key: getattr(current, key) - getattr(baseline, key) for key in fields}, (
        "Descriptive difference only; other settings and transmitted populations may also differ."
    )


def recorded_waists(result, component_keys):
    """Read precomputed optical-reference waist markers; never infer image focus."""
    rows = []
    for row in getattr(result, "lens_crossovers", ()) or ():
        if not isinstance(row, dict) or row.get("source_lens_key") not in component_keys:
            continue
        try:
            z, radius = float(row["z_mm"]), float(row["rms_radius_mm"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if row.get("verified") is True and math.isfinite(z) and math.isfinite(radius) and radius >= 0:
            rows.append((str(row["source_lens_key"]), z, radius * 1e3))
    return tuple(rows)
