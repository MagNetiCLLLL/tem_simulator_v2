"""Current-pixel electron signals from executed physical detector intercepts.

Weighted simulation counts describe the numerical bundle, not a measured dose.
Physical counts require an exposure: the always-available dose observable here
is electrons/second, derived from the emitted current and collection probability.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class ParticleDetectorReadout:
    key: str
    name: str
    status: str
    fraction: float | None = None
    simulated_electrons: float | None = None
    current_pa: float | None = None
    electrons_per_second: float | None = None
    expected_electrons: float | None = None


def measure_particle_detectors(result, *, exposure_s=None):
    """Read first physical stops; never extrapolate a truncated section."""
    from temsim.physics.beam_current import effective_source_current_pa, sample_illumination_absent
    from temsim.specimen.source import specimen_interactions_active
    from temsim.component_names import RECORDING_PLANE_NAMES
    state, simulation = result.state_snapshot, result.simulation
    if exposure_s is not None and (not math.isfinite(float(exposure_s)) or float(exposure_s) <= 0.):
        raise ValueError("Electron-count exposure must be positive and finite")
    metrics = simulation.metrics
    material = specimen_interactions_active(state.sample)
    exit_state = getattr(result, "specimen_exit", None)
    outgoing = tuple(exit_state.branches) if exit_state is not None else tuple(simulation.branches.values())
    optical_only = bool(metrics.get("optical_tuning", False))
    target = float(metrics.get("section_target_z_mm", max(
        [float(simulation.incident.z[-1])] + [float(b.z[-1]) for b in outgoing if len(b.z)])))
    current = float(effective_source_current_pa(state))
    population = int(state.electron_gun.ray_count)
    no_illumination = bool(sample_illumination_absent(simulation, state))
    rows = []
    for detector in state.recording_planes:
        key = str(detector.key)
        name = RECORDING_PLANE_NAMES.get(key, "Electron detector")
        z = float(detector.z_mm)
        status = "AVAILABLE"
        if not bool(getattr(detector, "inserted", False)):
            status = "NOT_INSERTED"
        elif not bool(getattr(detector, "readout_enabled", True)):
            status = "READOUT_DISABLED"
        elif z > target + 1e-9:
            status = "NOT_REACHED"
        elif (optical_only or (material and z > float(state.sample.z_mm)
                             and exit_state is None and not no_illumination)):
            status = "NOT_CALCULATED"
        if status != "AVAILABLE":
            rows.append(ParticleDetectorReadout(key, name, status))
            continue
        branches = (simulation.incident,) if z <= float(state.sample.z_mm) else outgoing
        fraction = 0.
        for branch in branches:
            if not len(branch.z) or not float(branch.z[0])-1e-9 <= z <= float(branch.z[-1])+1e-9:
                continue
            count = np.shape(branch.x)[1]
            weights = np.asarray(branch.ray_weight, dtype=float)
            stops = np.asarray(branch.blocked_key, dtype=str)
            blocked = np.asarray(branch.blocked_z, dtype=float)
            probability = 1. if branch is simulation.incident else float(branch.weight)
            if (weights.shape != (count,) or stops.shape != (count,) or blocked.shape != (count,)
                    or not np.all(np.isfinite(weights)) or np.any(weights < 0.)
                    or not math.isfinite(probability) or probability < 0.):
                raise ValueError("Detector readout requires valid source-normalised particle weights and stop identities")
            hit = (stops == key) & np.isfinite(blocked) & (blocked <= target + 1e-9)
            fraction += probability * float(np.sum(weights[hit]))
        if not math.isfinite(fraction) or fraction > 1. + 1e-10:
            raise ValueError("Detector interception exceeds the emitted particle probability")
        fraction = min(fraction, 1.)
        pa = current * fraction
        rate = pa * 1e-12 / 1.602176634e-19
        rows.append(ParticleDetectorReadout(key, name, status, fraction,
            population*fraction, pa, rate, None if exposure_s is None else rate*float(exposure_s)))
    energy_filter = getattr(state, "energy_filter", None)
    if energy_filter is not None and bool(energy_filter.enabled):
        # Bent filter coordinates cannot be compared with the straight-column
        # Z cursor. Consume only the actual filter solver's absorption products.
        filtered = getattr(result, "energy_filter", None)
        spectrometer = getattr(energy_filter, "zebra_detector", None)
        for key, name, inserted, field in (
            ("energy_filter_output_detector", "Energy-filter camera",
             bool(energy_filter.output_detector_inserted), "camera_recorded_fraction"),
            ("energy_loss_detector", "Electron energy-loss detector",
             bool(getattr(spectrometer, "enabled", False) and getattr(spectrometer, "inserted", False)),
             "eels_transmitted_fraction"),
        ):
            status = ("NOT_INSERTED" if not inserted else "NOT_CALCULATED" if optical_only
                      else "NOT_REACHED" if filtered is None else "AVAILABLE")
            if status != "AVAILABLE":
                rows.append(ParticleDetectorReadout(key, name, status))
                continue
            fraction = float(getattr(filtered, field))
            if not math.isfinite(fraction) or not 0. <= fraction <= 1.+1e-10:
                raise ValueError("Filter detector interception has an invalid emitted-particle probability")
            fraction = min(fraction, 1.)
            pa = current*fraction
            rate = pa*1e-12/1.602176634e-19
            rows.append(ParticleDetectorReadout(key, name, status, fraction,
                population*fraction, pa, rate, None if exposure_s is None else rate*float(exposure_s)))
    return tuple(rows)
