"""Qt-free, display-only sampling of completed ray histories at one plane.

X/Y are metres, Z is millimetres, and tx/ty are dimensionless trajectory
slopes. Polar angle is atan(hypot(tx, ty)), not an azimuth or a Larmor angle.
Fractions are relative to the emitted source and are never renormalised after
clipping or histogram viewport selection. No field or specimen solver runs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np

from temsim.gui.beam_display_source import downstream_display_branches
from temsim.gui.beam_tracking_modes import branch_interaction_style
from temsim.physics.ray_identity import branch_identity, source_identity


_PROBABILITY_TOL = 1.0e-10
_INTERCEPT_TOL_MM = 1.0e-9


def _frozen(values, dtype=None):
    array = np.ascontiguousarray(values, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True, slots=True)
class BeamPlaneData:
    z_mm: float
    x_m: np.ndarray
    y_m: np.ndarray
    tx: np.ndarray
    ty: np.ndarray
    source_ray_id: np.ndarray
    source_azimuth_rad: np.ndarray
    source_fraction: np.ndarray
    interaction_key: np.ndarray
    interaction_label: np.ndarray
    interaction_rgb: np.ndarray
    interaction_symbol: np.ndarray
    column_index: np.ndarray
    total_column_count: int
    provenance: str
    status: str
    source_current_pa: float | None
    weights_valid: bool
    diagnostics: tuple[str, ...] = ()

    @property
    def ray_count(self) -> int:
        return int(self.x_m.size)

    @property
    def total_source_fraction(self) -> float | None:
        if not self.weights_valid:
            return None
        return math.fsum(float(value) for value in self.source_fraction)

    @property
    def current_pa(self) -> float | None:
        fraction = self.total_source_fraction
        if fraction is None or self.source_current_pa is None:
            return None
        return self.source_current_pa * fraction

    @property
    def theta_mrad(self) -> np.ndarray:
        return _frozen(np.arctan(np.hypot(self.tx, self.ty)) * 1000.0)


def _count(branch) -> int:
    try:
        values = np.asarray(getattr(branch, "x", ()))
        return int(values.shape[1]) if values.ndim == 2 else 0
    except (TypeError, ValueError):
        return 0


def _weights(branch, count, diagnostics, *, conditional=False):
    raw = getattr(branch, "ray_weight", None)
    if raw is None:
        if conditional:
            diagnostics.append("Detailed branch conditional weights are unavailable.")
            return np.full(count, np.nan), False
        diagnostics.append("Legacy source ray_weight=None: equal-weight Branch convention.")
        return np.full(count, 1.0 / max(count, 1)), True
    try:
        values = np.asarray(raw, dtype=float)
        if (values.shape != (count,) or np.any(~np.isfinite(values))
                or np.any(values < 0.0)):
            raise ValueError
        total = math.fsum(float(value) for value in values)
        if conditional:
            if not total > 0.0:
                raise ValueError
            if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=_PROBABILITY_TOL):
                diagnostics.append("Detailed ray weights use the full branch conditional normalisation.")
            # This is the documented conditional Branch convention, not a
            # normalisation of the rays surviving this particular plane.
            return values / total, True
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=_PROBABILITY_TOL):
            diagnostics.append("Explicit source weights do not sum to one; source fractions are ambiguous.")
            return np.full(count, np.nan), False
        return values.copy(), True
    except (TypeError, ValueError, OverflowError):
        diagnostics.append("Ray probabilities are invalid; quantitative readout is unavailable.")
        return np.full(count, np.nan), False


def _ordinary_probabilities(simulation, branches, diagnostics):
    """Use the authoritative optical-reference absolute/relative convention."""
    metrics = getattr(simulation, "metrics", {})
    if not isinstance(metrics, Mapping):
        diagnostics.append("Optical branch probability convention is unavailable.")
        return np.full(len(branches), np.nan), False
    absolute = metrics.get("branch_weights_are_absolute", False)
    if not isinstance(absolute, (bool, np.bool_)):
        diagnostics.append("Optical branch probability convention is invalid.")
        return np.full(len(branches), np.nan), False
    try:
        values = np.asarray([float(branch.weight) for branch in branches])
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError
        total = math.fsum(float(value) for value in values)
        if absolute:
            if total > 1.0 + _PROBABILITY_TOL:
                raise ValueError
            return values, True
        if not total > 0.0:
            raise ValueError
        diagnostics.append("Optical-reference branch weights use relative probability normalisation.")
        return values / total, True
    except (AttributeError, TypeError, ValueError, OverflowError):
        diagnostics.append("Optical branch probabilities are unavailable or invalid.")
        return np.full(len(branches), np.nan), False


def _blocked_mask(branch, count, selected_z):
    blocked = np.asarray(getattr(branch, "blocked_z", ()), dtype=float)
    if blocked.shape != (count,) or np.any(np.isinf(blocked)):
        raise ValueError("Missing or invalid cached interception planes.")
    return np.isnan(blocked) | (blocked >= selected_z - _INTERCEPT_TOL_MM)


def _interpolate(branch, count, selected_z):
    z = np.asarray(getattr(branch, "z", ()), dtype=float)
    if (z.ndim != 1 or not z.size or np.any(~np.isfinite(z))
            or np.any(np.diff(z) <= 0.0)):
        raise ValueError("Invalid cached axial history.")
    arrays = tuple(np.asarray(getattr(branch, key, ()))
                   for key in ("x", "y", "tx", "ty"))
    if any(array.shape != (z.size, count) or array.dtype.kind not in "fiu" for array in arrays):
        raise ValueError("Cached position and slope shapes do not agree.")
    # Strict history bounds: an arbitrary selected plane is never extrapolated.
    if selected_z < z[0] or selected_z > z[-1]:
        return None
    right = int(np.searchsorted(z, selected_z, side="left"))
    if right == 0 or z[right] == selected_z:
        return tuple(np.asarray(array[right], dtype=float).copy() for array in arrays)
    left = right - 1
    t = float((selected_z - z[left]) / (z[right] - z[left]))
    # Convert only the two selected rows. A float32 high-accuracy cache can
    # contain gigabytes of history; a display must not copy/promote all of it.
    return tuple((1.0 - t) * np.asarray(array[left], dtype=float)
                 + t * np.asarray(array[right], dtype=float) for array in arrays)


def sample_beam_plane(result, z_mm: float) -> BeamPlaneData:
    """Read every weighted ray reaching Z from one completed snapshot.

    Global column_index and total_column_count are assigned before clipping,
    so an independent fixed display subset does not change as stops are crossed.
    Invalid probability metadata keeps geometry inspectable but marks fractions
    unavailable; it never manufactures equal weights for a detailed exit.
    """
    diagnostics = []
    simulation = getattr(result, "simulation", None)
    incident = getattr(simulation, "incident", None)
    metrics = getattr(simulation, "metrics", {})
    source_current = None
    if isinstance(metrics, Mapping):
        raw_current = metrics.get("effective_source_current_pa")
        if raw_current is not None and not isinstance(raw_current, (bool, np.bool_)):
            try:
                value = float(raw_current)
                if math.isfinite(value) and value >= 0.0:
                    source_current = value
            except (TypeError, ValueError, OverflowError):
                pass
    if source_current is None:
        diagnostics.append("Cached source current is unavailable; fractions only.")
    try:
        selected_z = float(z_mm)
    except (TypeError, ValueError, OverflowError):
        selected_z = math.nan
    columns = []
    total_columns = 0
    provenance = "Unavailable"
    valid = True
    coverage = False
    if incident is None or not math.isfinite(selected_z):
        diagnostics.append("A finite plane and a cached incident beam are required.")
        valid = False
        branches = ()
    else:
        try:
            incident_z = np.asarray(getattr(incident, "z", ()), dtype=float)
        except (TypeError, ValueError):
            incident_z = np.empty(0)
        if incident_z.ndim != 1 or not incident_z.size or not np.isfinite(incident_z[-1]):
            branches = ()
            valid = False
            diagnostics.append("The cached incident boundary is invalid.")
        elif selected_z <= float(incident_z[-1]):
            branches = (incident,)
            provenance = "Incident"
        else:
            try:
                branches, provenance = downstream_display_branches(result)
            except (AttributeError, TypeError, ValueError):
                branches = ()
                valid = False
                diagnostics.append("Cached downstream history is unavailable.")
            if provenance == "Optical reference":
                diagnostics.append("Optical-reference population; not a validated detailed specimen exit.")
            if not branches and provenance != "Specimen exit":
                valid = False
                diagnostics.append("No cached downstream population is available.")

    incident_count = _count(incident)
    incident_weights, incident_valid = _weights(
        incident, incident_count, diagnostics
    ) if incident is not None and provenance in {"Incident", "Optical reference"} else (np.empty(0), False)
    branch_probabilities = None
    if provenance == "Optical reference" and branches:
        branch_probabilities, probability_valid = _ordinary_probabilities(
            simulation, branches, diagnostics
        )
        valid &= probability_valid
    elif provenance == "Specimen exit" and not branches:
        coverage = True  # A validated empty exit means no forward electrons.

    for branch_number, branch in enumerate(branches):
        count = _count(branch)
        global_indices = total_columns + np.arange(count, dtype=np.int64)
        total_columns += count
        if count == 0:
            try:
                _interpolate(branch, count, selected_z)
            except (AttributeError, TypeError, ValueError) as exc:
                valid = False
                diagnostics.append(str(exc))
            continue
        if provenance == "Incident":
            fractions = incident_weights.copy()
            valid &= incident_valid
        elif provenance == "Specimen exit":
            conditional, conditional_valid = _weights(
                branch, count, diagnostics, conditional=True
            )
            fractions = float(branch.weight) * conditional
            valid &= conditional_valid
        elif count == incident_count:
            fractions = incident_weights * branch_probabilities[branch_number]
            valid &= incident_valid
            try:
                fractions = np.where(
                    _blocked_mask(incident, incident_count, float(incident.z[-1])),
                    fractions, 0.0,
                )
            except (TypeError, ValueError):
                fractions[:] = np.nan
                valid = False
                diagnostics.append("Incident survival at the sample is unavailable.")
        else:
            fractions = np.full(count, np.nan)
            valid = False
            diagnostics.append("Reference branches do not preserve incident-column weight alignment.")
        # A zero-probability branch or support probe contributes no density.
        weighted = ~np.isfinite(fractions) | (fractions > 0.0)
        try:
            weight_source = incident if provenance == "Optical reference" and count == incident_count else branch
            raw_weights = np.asarray(getattr(weight_source, "ray_weight", None), dtype=float)
            if raw_weights.shape == (count,):
                weighted &= raw_weights != 0.0
        except (TypeError, ValueError):
            pass
        if not np.any(weighted):
            continue
        try:
            reaches = _blocked_mask(branch, count, selected_z) & weighted
            points = _interpolate(branch, count, selected_z)
            if points is None:
                if np.any(reaches):
                    valid = False
                    diagnostics.append("Selected plane lies outside a weighted cached ray history; no extrapolation.")
                continue
        except (AttributeError, TypeError, ValueError) as exc:
            valid = False
            diagnostics.append(str(exc))
            continue
        coverage = True
        finite = np.logical_and.reduce([np.isfinite(value) for value in points])
        if np.any(reaches & ~finite):
            valid = False
            diagnostics.append("Non-finite reaching coordinates or slopes were omitted.")
        keep = reaches & finite
        if not np.any(keep):
            continue
        ids, azimuths = (
            source_identity(incident, getattr(simulation, "gun_trace", None))
            if provenance == "Incident" else branch_identity(branch, simulation)
        )
        key, label, rgb, symbol = branch_interaction_style(branch)
        size = int(np.count_nonzero(keep))
        columns.append((
            *(value[keep] for value in points), ids[keep], azimuths[keep],
            fractions[keep], np.full(size, key), np.full(size, label),
            np.tile(np.asarray(rgb, dtype=np.uint8), (size, 1)),
            np.full(size, symbol), global_indices[keep],
        ))
    if columns:
        arrays = [_frozen(np.concatenate([column[index] for column in columns]))
                  for index in range(12)]
    else:
        arrays = [_frozen([], dtype=float) for _ in range(12)]
        arrays[4] = _frozen([], dtype=np.int64)
        arrays[7] = arrays[8] = arrays[10] = _frozen([], dtype="U1")
        arrays[9] = _frozen(np.empty((0, 3), dtype=np.uint8))
        arrays[11] = _frozen([], dtype=np.int64)
    if valid and math.fsum(float(value) for value in arrays[6]) > 1.0 + _PROBABILITY_TOL:
        valid = False
        diagnostics.append("Reaching source fractions exceed one; quantitative readout is unavailable.")
    status = ("Ready" if arrays[0].size else "No reaching rays") if valid else (
        "Partial data" if arrays[0].size else "Outside cached range" if not coverage else "Unavailable"
    )
    return BeamPlaneData(
        selected_z, *arrays, total_columns, provenance, status,
        source_current, bool(valid), tuple(dict.fromkeys(diagnostics)),
    )


def _histogram_bins(bins):
    if isinstance(bins, bool) or int(bins) != bins or not 1 <= int(bins) <= 1024:
        raise ValueError("Histogram bins must be an integer from 1 to 1024.")
    return int(bins)


def _histogram_range(bounds):
    low, high = (float(value) for value in bounds)
    if not math.isfinite(low) or not math.isfinite(high) or not low < high:
        raise ValueError("Histogram bounds must be finite and increasing.")
    return low, high


def spatial_histogram(data: BeamPlaneData, bins=64, *, projection_angle_deg=0.0, range_m=None):
    """Return (source-fraction bins, X edges, Y edges), with no viewport rescale."""
    if not data.weights_valid:
        raise ValueError("Source probabilities are unavailable for this plane.")
    bins = _histogram_bins(bins)
    angle = math.radians(float(projection_angle_deg))
    if not math.isfinite(angle):
        raise ValueError("Projection angle must be finite.")
    c, s = math.cos(angle), math.sin(angle)
    x, y = c * data.x_m + s * data.y_m, -s * data.x_m + c * data.y_m
    if range_m is None:
        half_x = max(float(np.max(np.abs(x))) if x.size else 0.0, 1.0e-15)
        half_y = max(float(np.max(np.abs(y))) if y.size else 0.0, 1.0e-15)
        bounds = ((-half_x, half_x), (-half_y, half_y))
    else:
        bounds = tuple(_histogram_range(axis) for axis in range_m)
        if len(bounds) != 2:
            raise ValueError("Spatial histogram needs X and Y bounds.")
    result = np.histogram2d(x, y, bins=bins, range=bounds, weights=data.source_fraction)
    return tuple(_frozen(value) for value in result)


def angular_histogram(data: BeamPlaneData, bins=64, *, range_mrad=None):
    """Return weighted polar-angle bins and edges in mrad; no signal renormalisation."""
    if not data.weights_valid:
        raise ValueError("Source probabilities are unavailable for this plane.")
    bins = _histogram_bins(bins)
    theta = data.theta_mrad
    bounds = (_histogram_range(range_mrad) if range_mrad is not None else
              (0.0, max(float(np.max(theta)) if theta.size else 0.0, 1.0e-9)))
    result = np.histogram(theta, bins=bins, range=bounds, weights=data.source_fraction)
    return tuple(_frozen(value) for value in result)
