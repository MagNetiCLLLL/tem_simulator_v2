"""Physical Camera/screen images from retained first-interception records.

No rays are traced here. Pixels contain probability per emitted electron,
not probability density and not a count of numerical rays. Expected incident
electrons use the captured source current and an explicit exposure. Detector
PSF is the configured forward response; electronic noise/QE are not inferred.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from temsim.component_keys import CAMERA, FLUORESCENT_SCREEN
from temsim.detector.particle_readout import measure_particle_detectors
from temsim.detector.point_spread import DetectorPointSpread, apply_point_spread


@dataclass(frozen=True)
class ReceiverImage:
    key: str
    name: str
    status: str
    detail: str
    ideal_probability: np.ndarray
    response_probability: np.ndarray
    x_mm: np.ndarray
    y_mm: np.ndarray
    exposure_s: float | None
    expected_electrons: np.ndarray | None
    current_pa: float | None
    metrics: dict

    def __post_init__(self):
        for name in ("ideal_probability", "response_probability", "x_mm", "y_mm",
                     "expected_electrons"):
            value = getattr(self, name)
            if value is not None:
                array = np.array(value, dtype=float, copy=True)
                array.setflags(write=False)
                object.__setattr__(self, name, array)
        object.__setattr__(self, "metrics", dict(self.metrics))


def available_receivers(state):
    """Installed imaging surfaces, including ones currently retracted.

    Selection never inserts a receiver or changes the active optical target.
    """
    return tuple(plane for plane in getattr(state, "recording_planes", ())
                 if str(plane.key) in {CAMERA, FLUORESCENT_SCREEN}
                 and bool(getattr(plane, "installed", True))
                 and bool(getattr(plane, "_layout_installed", True)))


_STATUS_DETAILS = {
    "NOT_INSERTED": "The selected receiver was retracted in this calculation.",
    "READOUT_DISABLED": "The selected receiver readout was disabled in this calculation.",
    "NOT_REACHED": "The retained calculation does not reach this receiver.",
    "NOT_CALCULATED": "No executed physical particle reception is available; calculate the particle path first.",
}


def _positions_at(branch, z_mm, count):
    z = np.asarray(branch.z, dtype=float)
    if (z.ndim != 1 or not z.size or not np.all(np.isfinite(z))
            or np.any(np.diff(z) <= 0.0)):
        raise ValueError("Receiver image requires ordered finite trajectory planes")
    if z_mm < z[0] - 1e-9 or z_mm > z[-1] + 1e-9:
        raise ValueError("Receiver image cannot extrapolate a retained trajectory")
    high = min(int(np.searchsorted(z, z_mm)), len(z) - 1)
    low = high if high == 0 or abs(float(z[high]) - z_mm) <= 1e-9 else high - 1
    fraction = 0.0 if high == low else (z_mm - z[low]) / (z[high] - z[low])
    positions = []
    for name in ("x", "y"):
        values = np.asarray(getattr(branch, name), dtype=float)
        if values.shape != (len(z), count):
            raise ValueError("Receiver coordinates do not match the particle trajectory")
        positions.append((values[low] + fraction * (values[high] - values[low])) * 1e3)
    return np.column_stack(positions)


def receiver_image(result, key, *, pixels=192, exposure_s=1.0, extent_mm=None):
    """Histogram first physical hits on one captured Camera or screen.

    The default grid covers the complete active-area bounding square in mm.
    Cropping is deliberately not a transport operation; callers may zoom the
    returned image. Missing calculations have an explicit unavailable status,
    whereas a calculated but blocked/empty receiver has AVAILABLE zero pixels.
    """
    if isinstance(pixels, bool) or int(pixels) != pixels or not 2 <= int(pixels) <= 4096:
        raise ValueError("Receiver image pixels must be an integer from 2 to 4096")
    n = int(pixels)
    if exposure_s is not None:
        exposure_s = float(exposure_s)
        if not math.isfinite(exposure_s) or exposure_s <= 0.0:
            raise ValueError("Receiver exposure must be positive and finite")
    if extent_mm is not None:
        raise ValueError("Receiver images use the full physical area; zoom the displayed image instead")
    state = getattr(result, "state_snapshot", None)
    plane = next((p for p in available_receivers(state) if str(p.key) == str(key)), None)
    if plane is None:
        raise ValueError(f"Unknown or uninstalled imaging receiver: {key}")
    width = float(plane.outer_width_mm)
    centre = np.array([getattr(plane, "centre_offset_x_mm", 0.0),
                       getattr(plane, "centre_offset_y_mm", 0.0)], dtype=float)
    z_mm = float(plane.z_mm)
    if (not math.isfinite(width) or width <= 0.0 or not np.all(np.isfinite(centre))
            or not math.isfinite(z_mm)):
        raise ValueError("Receiver dimensions and position must be finite with positive width")
    edges = np.linspace(-0.5 * width, 0.5 * width, n + 1)
    x_edges, y_edges = edges + centre[0], edges + centre[1]
    x_mm, y_mm = 0.5 * (x_edges[:-1] + x_edges[1:]), 0.5 * (y_edges[:-1] + y_edges[1:])
    ideal = np.zeros((n, n), dtype=float)
    simulation = getattr(result, "simulation", None)
    captured_metrics = getattr(simulation, "metrics", {})
    metrics = {
        "model": "retained_physical_first_hits_v1",
        "probability_convention": "probability per emitted electron per pixel",
        "axis_convention": "array[y, x]; laboratory X/Y; pixel centres in mm",
        "receiver_key": str(plane.key),
        "receiver_z_mm": z_mm,
        "receiver_width_mm": width,
        "receiver_inserted": bool(getattr(plane, "inserted", False)),
        "capture_workflow": str(getattr(result, "workflow", "")),
        "capture_request_identity": dict(getattr(result, "signatures", {})).get("request", "UNRECORDED"),
        "capture_manifest_identity": getattr(getattr(result, "calculation_manifest", None), "digest", "UNRECORDED"),
        "capture_model_signature": str(getattr(result, "model_signature", "")),
        "section_target_z_mm": captured_metrics.get("section_target_z_mm"),
        "section_physics_scope": captured_metrics.get("section_physics_scope"),
        "projector_mode": str(getattr(state, "projector_mode", "")),
        "exposure_s": exposure_s,
        "physical_area_extent_mm": (float(x_edges[0]), float(x_edges[-1]),
                                     float(y_edges[0]), float(y_edges[-1])),
        "response_scope": "incident-electron expectation with configured PSF; no electronic noise or QE model",
    }

    def unavailable(status):
        empty = np.zeros_like(ideal)
        return ReceiverImage(str(plane.key), str(plane.name), status, _STATUS_DETAILS[status],
                             empty, empty, x_mm, y_mm, exposure_s, None, None, metrics)

    if not bool(getattr(plane, "inserted", False)):
        return unavailable("NOT_INSERTED")
    if not bool(getattr(plane, "readout_enabled", True)):
        return unavailable("READOUT_DISABLED")
    if (simulation is None or str(getattr(result, "workflow", "")) == "rays"
            or bool(captured_metrics.get("optical_tuning", False))):
        return unavailable("NOT_CALCULATED")
    row = next(row for row in measure_particle_detectors(result, exposure_s=exposure_s)
               if row.key == str(plane.key))
    if row.status != "AVAILABLE":
        return unavailable(row.status)

    exit_state = getattr(result, "specimen_exit", None)
    outgoing = tuple(exit_state.branches) if exit_state is not None else tuple(simulation.branches.values())
    branches = (simulation.incident,) if z_mm <= float(state.sample.z_mm) else outgoing
    if not branches:
        from temsim.physics.beam_current import sample_illumination_absent
        if not sample_illumination_absent(simulation, state):
            return unavailable("NOT_CALCULATED")
    metrics["particle_source"] = "specimen_exit" if exit_state is not None else "simulation"
    for branch in branches:
        z = np.asarray(branch.z, dtype=float)
        if not z.size or not float(z[0]) - 1e-9 <= z_mm <= float(z[-1]) + 1e-9:
            alive = np.asarray(branch.alive, dtype=bool)
            weights = np.asarray(branch.ray_weight, dtype=float)
            if np.any(alive & (weights > 0.0)) and float(branch.weight) > 0.0:
                return unavailable("NOT_REACHED")
            continue
        weights = np.asarray(branch.ray_weight, dtype=float)
        stops = np.asarray(branch.blocked_key, dtype=str)
        blocked = np.asarray(branch.blocked_z, dtype=float)
        # measure_particle_detectors has already validated the source weights.
        hit = (stops == str(plane.key)) & np.isfinite(blocked)
        if not np.any(hit):
            continue
        if np.any(np.abs(blocked[hit] - z_mm) > 1e-7):
            raise ValueError("Recorded receiver stop position differs from its captured physical plane")
        positions = _positions_at(branch, z_mm, len(weights))[hit]
        if not np.all(np.isfinite(positions)):
            raise ValueError("Physical receiver hits must have finite positions")
        accepted = np.asarray(plane.hit_mask(positions[:, 0], positions[:, 1]), dtype=bool)
        if not np.all(accepted):
            raise ValueError("Recorded receiver hits lie outside its captured active area")
        probability = 1.0 if branch is simulation.incident else float(branch.weight)
        ideal += np.histogram2d(positions[:, 1], positions[:, 0],
                                bins=(y_edges, x_edges), weights=weights[hit] * probability)[0]
    if not math.isclose(float(ideal.sum()), float(row.fraction), rel_tol=2e-10, abs_tol=1e-12):
        raise ValueError("Receiver image does not conserve the recorded intercepted probability")
    point_spread = DetectorPointSpread.from_component(plane)
    response = apply_point_spread(ideal, point_spread,
                                 pixel_size_x_mm=width / n, pixel_size_y_mm=width / n)
    if point_spread.enabled:
        # Without spreading all binned hits have already passed the exact
        # physical mask, including pixels straddling a curved screen edge.
        active = np.asarray(plane.hit_mask(x_mm[None, :], y_mm[:, None]), dtype=bool)
        response = np.where(active, response, 0.0)
    from temsim.physics.beam_current import effective_source_current_pa
    source_current_pa = float(effective_source_current_pa(state))
    if not math.isfinite(source_current_pa) or source_current_pa < 0.0:
        raise ValueError("Captured source current must be finite and non-negative")
    expected = (None if exposure_s is None else
                response * source_current_pa * 1e-12 / 1.602176634e-19 * exposure_s)
    metrics.update(intercepted_probability=float(ideal.sum()),
                   response_probability=float(response.sum()),
                   source_current_pa=source_current_pa,
                   intercepted_current_pa=float(row.current_pa),
                   point_spread_model=point_spread.model,
                   point_spread_status=point_spread.status,
                   point_spread_source=point_spread.source,
                   coherence="classical particle reception; coherent diffraction is not inferred")
    detail = ("No electrons reached this receiver in the retained calculation."
              if not np.any(ideal) else "Actual first-interception particle distribution from the retained calculation.")
    return ReceiverImage(str(plane.key), str(plane.name), "AVAILABLE", detail, ideal, response,
                         x_mm, y_mm, exposure_s, expected, float(row.current_pa), metrics)
