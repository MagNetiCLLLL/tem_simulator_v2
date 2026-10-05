"""Captured, zero-loss paraxial conjugacy at the Ray Diagram's selected Z.

This observer classifies the optical map from the specimen reference plane.
It does not create a beam source, compute Bragg intensities, or classify each
inelastic population's separate energy. Execution coverage is read from the
accepted result, never from a currently requested cutoff or drawn ray tails.
"""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass
import math
from typing import Callable

from temsim.cpu_resources import NumericalJobCancelled, numerical_job
from temsim.gui.ray_extent_data import completed_ray_extent
from temsim.optics.direct_alignment import diffraction_transfer
from temsim.physics.first_order import TransverseTransfer
from temsim.physics.scan_geometry import classify_sample_plane_transfer


@dataclass(frozen=True, slots=True)
class SelectedPlaneDiagnostic:
    z_mm: float
    kind: str
    image_residual_m_per_rad: float | None = None
    diffraction_residual: float | None = None
    detail: str = ""


def _finite(value) -> float | None:
    if isinstance(value, (bool, str, bytes)):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _requested_z(z_mm) -> float:
    if isinstance(z_mm, (bool, str, bytes)):
        raise ValueError("Selected axial Z must be a numeric value in millimetres")
    return float(z_mm)


def plane_status_without_trace(result, z_mm) -> SelectedPlaneDiagnostic | None:
    """Return an immediate boundary status, or None when a map is required.

    No field construction, numerical tracing, signatures, or particle-array
    scans belong here: this function is safe on the GUI thread during a drag.
    The shared coverage reader is pure data code and does not import Qt.
    """
    z = _requested_z(z_mm)
    if not math.isfinite(z):
        return SelectedPlaneDiagnostic(z, "unavailable", detail="Selected axial Z must be finite.")
    state = getattr(result, "state_snapshot", None)
    sample_z = _finite(getattr(getattr(state, "sample", None), "z_mm", None))
    if state is None or sample_z is None:
        return SelectedPlaneDiagnostic(z, "unavailable", detail="No captured specimen reference is available for this result.")
    try:
        extent = completed_ray_extent(result)
    except (AttributeError, TypeError, ValueError):
        extent = {}
    start, completed = extent.get("start_z_mm"), extent.get("completed_z_mm")
    if start is None or completed is None:
        return SelectedPlaneDiagnostic(z, "unavailable", detail="This result has no verified executed axial origin and endpoint; requested cutoffs and plotted ray tails are not execution evidence.")
    if z > completed or z < start:
        return SelectedPlaneDiagnostic(z, "not_calculated", detail=f"Selected Z lies outside this result's executed axial range; calculation completed through Z {completed:.9g} mm.")
    if bool(getattr(state, "energy_filter_installed", False)):
        entrance = _finite(getattr(getattr(state, "energy_filter", None), "entrance_z_mm", None))
        if entrance is None or z >= entrance:
            return SelectedPlaneDiagnostic(z, "unavailable", detail="The installed energy filter requires its transported optical response; the straight-column map cannot classify this plane.")
    if z < sample_z:
        return SelectedPlaneDiagnostic(z, "upstream", detail="Upstream of the specimen reference: sample-to-plane image/diffraction conjugacy is not defined here.")
    if z == sample_z:
        return SelectedPlaneDiagnostic(z, "specimen", detail="Specimen reference plane. This is the object plane, rather than a downstream image or diffraction plane.")
    if sample_z < start:
        return SelectedPlaneDiagnostic(z, "unavailable", detail="The verified execution range does not include the specimen reference plane.")
    voltage = _finite(getattr(state, "beam_voltage_kv", None))
    step = _finite(getattr(state, "step_mm", None))
    if (voltage is None or voltage <= 0. or step is None or step <= 0.
            or getattr(state, "lenses", None) is None):
        return SelectedPlaneDiagnostic(z, "unavailable", detail="Captured optical inputs are incomplete; no first-order solver will be started.")
    return None


def _check_cancelled(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise NumericalJobCancelled("Selected-plane diagnostic was superseded or cancelled")


def _exact_saved_transfer(simulation, source_z_mm, target_z_mm):
    # Old saved transfers may use mechanical slopes. Explicitly require the
    # shared canonical convention even when the source and target Z match.
    from temsim.physics.first_order import SPECIMEN_CANONICAL_MOMENTUM
    transfer = getattr(simulation, "sample_to_analysis_transfer", None)
    if (isinstance(transfer, TransverseTransfer)
            and getattr(transfer, "input_basis", None) == SPECIMEN_CANONICAL_MOMENTUM
            and transfer.source_z_mm == source_z_mm
            and transfer.target_z_mm == target_z_mm):
        return transfer
    return None


def _observer_state(snapshot):
    """Detach known observer writers, sharing immutable field buffers.

    Backend bookkeeping and field-provider resolution update the containers
    below in place. Lens objects are small parameter records: sync_objective
    stores derived planes on the objective, and equivalent-image calibration
    temporarily changes lens excitation. Their observer-owned copies preserve
    aliases without deep-copying field maps or executed particle arrays.
    """
    state = copy(snapshot)
    for name in ("_active_backends_used", "_runtime_lens_field_provider_cache",
                 "_runtime_nonlinear_provider_cache", "_field_provider_diagnostics",
                 "_lens_field_map_bindings"):
        value = getattr(snapshot, name, None)
        if isinstance(value, (dict, set)):
            setattr(state, name, value.copy())
    lenses = getattr(snapshot, "lenses", ())
    detached = {id(lens): copy(lens) for lens in lenses}
    state.lenses = type(lenses)(detached[id(lens)] for lens in lenses)
    # State's named lens lookups can already be cached. Rebind those aliases
    # before a dataclass equality test could retain the original component.
    for name, value in vars(snapshot).items():
        if id(value) in detached:
            setattr(state, name, detached[id(value)])
    return state


def _classified(result, z, transfer) -> SelectedPlaneDiagnostic:
    kind, image_residual, diffraction_residual = classify_sample_plane_transfer(transfer)
    voltage = float(result.state_snapshot.beam_voltage_kv)
    detail = (
        f"Captured specimen-to-Z first-order optical transfer at nominal zero-loss energy {voltage:.9g} keV. "
        "Position and canonical/Larmor specimen-angle maps include the captured lens, stigmator and deflector settings. "
        "This optical classification does not establish Bragg diffraction or classify individual inelastic energies. "
        "It does not imply that tracked particles reach this plane."
    )
    metrics = getattr(getattr(result, "simulation", None), "metrics", None) or {}
    if (metrics.get("sample_beam_surviving_rays") == 0
            or metrics.get("sample_beam_surviving_fraction") == 0.
            or metrics.get("sample_illumination_status") == "no incident current"):
        detail += " No tracked incident particles reached the specimen; conjugacy still describes the captured optical fields."
    return SelectedPlaneDiagnostic(z, kind, image_residual, diffraction_residual, detail)


def calculate_selected_plane(result, z_mm, *, cancelled=None) -> SelectedPlaneDiagnostic:
    """Evaluate one exact selected Z in a serialized, single-thread worker.

    A saved canonical analysis transfer is reused only at exactly matching
    specimen and target Z. Otherwise the same specimen-canonical observer
    evaluates this exact float64 endpoint. The caller owns latest-request
    publication and the GUI cache.
    """
    _check_cancelled(cancelled)
    boundary = plane_status_without_trace(result, z_mm)
    _check_cancelled(cancelled)
    if boundary is not None:
        return boundary
    z = _requested_z(z_mm)
    source = float(result.state_snapshot.sample.z_mm)
    transfer = _exact_saved_transfer(result.simulation, source, z)
    if transfer is None:
        state = _observer_state(result.state_snapshot)

        def solver_cancelled():
            _check_cancelled(cancelled)
            return False

        state._tuning_cancelled = solver_cancelled
        with numerical_job(1, cancelled=cancelled):
            _check_cancelled(cancelled)
            try:
                transfer = diffraction_transfer(state, z)
            except (ValueError, RuntimeError) as exc:
                _check_cancelled(cancelled)
                return SelectedPlaneDiagnostic(z, "unavailable", detail="Captured first-order optical response could not be evaluated: " + str(exc))
    _check_cancelled(cancelled)
    diagnostic = _classified(result, z, transfer)
    _check_cancelled(cancelled)
    return diagnostic
