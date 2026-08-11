"""Observable probe state at the physical sample plane."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from temsim.physics.beam_statistics import branch_sample_statistics
from temsim.physics.core import electron


@dataclass(frozen=True, slots=True)
class ProbeEnergyBin:
    energy_offset_ev: float
    weight: float


@dataclass(frozen=True, slots=True)
class ProbeState:
    surviving_fraction: float
    surviving_current_pa: float
    centroid_nm: tuple[float, float]
    chief_angle_mrad: tuple[float, float]
    angular_covariance_mrad2: tuple[
        tuple[float, float],
        tuple[float, float],
    ]
    convergence_99_mrad: float
    radius_99_nm: float
    probe_sigma_nm: float
    wavelength_pm: float
    defocus_nm: float
    cs_mm: float
    cc_mm: float
    energy_bins: tuple[ProbeEnergyBin, ...]


def _ray_weights(branch) -> np.ndarray:
    count = int(np.asarray(branch.alive).size)
    if branch.ray_weight is None:
        return np.full(count, 1.0 / max(count, 1), dtype=float)
    values = np.asarray(branch.ray_weight, dtype=float)
    if values.shape != (count,) or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Probe ray weights must be finite and non-negative.")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("Probe ray weights must contain positive current.")
    # Emission bundles normally already sum to one.  Normalise here so custom
    # emitters cannot change the meaning of surviving_fraction.
    return values / total


def _energy_bins(branch, alive, weights, maximum_bins: int = 7) -> tuple[ProbeEnergyBin, ...]:
    energy = np.asarray(getattr(branch, "energy_offset_ev", ()), dtype=float)
    if energy.shape != alive.shape or not np.all(np.isfinite(energy[alive])):
        return ()
    selected_energy = energy[alive]
    selected_weight = weights[alive]
    total = float(selected_weight.sum())
    if total <= 0.0:
        return ()
    selected_weight = selected_weight / total
    if np.ptp(selected_energy) <= 1.0e-12:
        return (ProbeEnergyBin(float(np.sum(selected_weight * selected_energy)), 1.0),)
    count = min(int(maximum_bins), max(2, int(round(math.sqrt(selected_energy.size)))))
    edges = np.linspace(float(selected_energy.min()), float(selected_energy.max()), count + 1)
    result = []
    for index in range(count):
        mask = (selected_energy >= edges[index]) & (
            selected_energy <= edges[index + 1]
            if index == count - 1
            else selected_energy < edges[index + 1]
        )
        weight = float(selected_weight[mask].sum())
        if weight <= 0.0:
            continue
        result.append(
            ProbeEnergyBin(
                float(np.sum(selected_weight[mask] * selected_energy[mask]) / weight),
                weight,
            )
        )
    return tuple(result)


def probe_state_from_simulation(state, simulation) -> ProbeState:
    branch = simulation.incident
    alive = np.asarray(branch.alive, dtype=bool)
    if alive.ndim != 1 or not np.any(alive):
        raise ValueError("No electrons survive to define the sample probe state.")
    weights = _ray_weights(branch)
    surviving_fraction = float(weights[alive].sum())
    if surviving_fraction <= 0.0:
        raise ValueError("No positive current survives to the sample.")
    conditional = weights[alive] / surviving_fraction
    tx = np.asarray(branch.tx[-1], dtype=float)[alive]
    ty = np.asarray(branch.ty[-1], dtype=float)[alive]
    mean_tx = float(np.sum(conditional * tx))
    mean_ty = float(np.sum(conditional * ty))
    centred = np.column_stack((tx - mean_tx, ty - mean_ty))
    covariance = np.einsum("n,ni,nj->ij", conditional, centred, centred)
    statistics = branch_sample_statistics(branch)
    _charge, _momentum, wavelength_nm = electron(state)
    emitted_current_a = max(float(state.electron_gun.emitted_current_a), 0.0)
    objective = state.objective_lens
    defocus_nm = float(getattr(state.sample, "wave_defocus_nm", 0.0))
    # A circular Gaussian with this sigma has the same RMS radius as the ray
    # bundle. It is used only for virtual-density convolution, not as a claim
    # that the coherent probe itself is Gaussian.
    probe_sigma_nm = statistics.radius_rms_m * 1.0e9 / math.sqrt(2.0)
    return ProbeState(
        surviving_fraction=surviving_fraction,
        surviving_current_pa=emitted_current_a * 1.0e12 * surviving_fraction,
        centroid_nm=(statistics.mean_x_m * 1.0e9, statistics.mean_y_m * 1.0e9),
        chief_angle_mrad=(statistics.mean_tx_rad * 1.0e3, statistics.mean_ty_rad * 1.0e3),
        angular_covariance_mrad2=tuple(
            tuple(float(value * 1.0e6) for value in row) for row in covariance
        ),
        convergence_99_mrad=statistics.convergence_99_mrad,
        radius_99_nm=statistics.radius_99_m * 1.0e9,
        probe_sigma_nm=probe_sigma_nm,
        wavelength_pm=float(wavelength_nm) * 1.0e3,
        defocus_nm=defocus_nm,
        cs_mm=float(getattr(objective, "cs_mm", 0.0) or 0.0),
        cc_mm=float(getattr(objective, "cc_mm", 0.0) or 0.0),
        energy_bins=_energy_bins(branch, alive, weights),
    )

