"""Interaction and current budget at an arbitrary axial plane."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


CHANNEL_LABELS = {
    "incident": "Incident: no specimen interaction yet",
    "vacuum": "Vacuum continuation",
    "real_sample_reference": "Real-sample reference (inelastic model unavailable)",
    "real_zero_loss": "Zero loss / elastic-coherent population",
    "real_plasmon": "Single plasmon / low-loss",
    "real_ionisation": "Single core ionisation",
    "real_other_inelastic": "Single other inelastic",
    "real_plural_inelastic": "Plural inelastic (2+ events)",
    "virtual_interactions_disabled": "Virtual direct (interactions disabled)",
    "transmitted": "Virtual transmitted / direct",
    "diffraction_spots": "Virtual diffraction spots",
    "diffuse_ring": "Virtual diffuse ring",
    "gaussian_diffuse": "Virtual Gaussian diffuse",
    "arbitrary_angular": "Virtual arbitrary angular",
    "user_screened_power_law": "Virtual screened power law",
    "physical_rutherford": "Virtual screened Rutherford approximation",
    "unknown": "Unknown interaction",
}


@dataclass(frozen=True, slots=True)
class PlaneInteractionChannel:
    key: str
    label: str
    probability_at_sample: float
    source_fraction_at_plane: float
    composition_at_plane: float
    representative_loss_ev: float | None


@dataclass(frozen=True, slots=True)
class PlaneInteractionBudget:
    z_mm: float
    sample_z_mm: float
    location: str
    source_fraction_at_plane: float
    sample_incident_source_fraction: float
    pre_sample_stopped_source_fraction: float
    sample_absorbed_source_fraction: float
    downstream_stopped_source_fraction: float
    channels: tuple[PlaneInteractionChannel, ...]
    model: str
    material_name: str | None
    mean_inelastic_events: float | None
    total_inelastic_mean_free_path_nm: float | None
    warnings: tuple[str, ...]
    conservation_error: float


def _ray_weights(branch) -> np.ndarray:
    raw = getattr(branch, "ray_weight", None)
    if raw is None:
        count = int(np.asarray(branch.x).shape[1])
        return np.full(count, 1.0 / max(count, 1), dtype=float)
    weights = np.asarray(raw, dtype=float)
    if (
        weights.ndim != 1
        or weights.size != np.asarray(branch.x).shape[1]
        or np.any(~np.isfinite(weights))
        or np.any(weights < 0.0)
    ):
        raise ValueError("Ray weights must be finite, non-negative and match the bundle.")
    return weights


def _reaches_plane(branch, z_mm: float) -> np.ndarray:
    z = float(z_mm)
    branch_z = np.asarray(branch.z, dtype=float)
    count = np.asarray(branch.x).shape[1]
    if z < float(branch_z[0]) - 1.0e-9 or z > float(branch_z[-1]) + 1.0e-9:
        return np.zeros(count, dtype=bool)
    blocked = np.asarray(branch.blocked_z, dtype=float)
    # An electron intercepted at the selected plane reached that plane.  It
    # is excluded only for positions strictly downstream of the intercept.
    return np.isnan(blocked) | (blocked >= z - 1.0e-9)


def _branch_probabilities(simulation) -> dict[int, float]:
    branches = tuple(simulation.branches.values())
    raw = np.asarray(
        [max(float(getattr(branch, "weight", 1.0)), 0.0) for branch in branches],
        dtype=float,
    )
    absolute = bool(
        getattr(simulation, "metrics", {}).get(
            "branch_weights_are_absolute", False
        )
    )
    if absolute:
        if float(raw.sum()) > 1.0 + 1.0e-10:
            raise ValueError("Absolute branch probabilities exceed one.")
        probabilities = raw
    else:
        total = float(raw.sum())
        probabilities = (
            raw / total
            if total > 0.0
            else np.full(len(raw), 1.0 / max(len(raw), 1))
        )
    return {
        id(branch): float(probability)
        for branch, probability in zip(branches, probabilities)
    }


def _virtual_per_ray_probabilities(state, simulation, incident_alive):
    """Apply finite virtual density to nominal angular branch probabilities."""

    from temsim.specimen.virtual import (
        build_virtual_angular_distribution,
        virtual_density_at_scan,
    )

    sample = state.sample
    nominal = _branch_probabilities(simulation)
    distribution = build_virtual_angular_distribution(
        sample,
        beam_energy_kv=state.beam_voltage_kv,
    )
    incident = simulation.incident
    x_um = np.asarray(incident.x[-1], dtype=float)[None, :] * 1.0e6
    y_um = np.asarray(incident.y[-1], dtype=float)[None, :] * 1.0e6
    # Individual source rays already sample the probe distribution, so a
    # second Gaussian probe convolution would double-count it here.
    density = virtual_density_at_scan(
        sample,
        x_um,
        y_um,
        probe_sigma_nm=0.0,
    ).reshape(-1)
    density = np.where(incident_alive, density, 0.0)
    total_interacting = (
        distribution.scattered_probability
        + distribution.absorbed_probability
    )
    per_branch: dict[int, np.ndarray] = {}
    for branch in simulation.branches.values():
        kind = str(getattr(branch, "interaction_kind", "unknown"))
        probability = nominal[id(branch)]
        if kind == "transmitted":
            per_branch[id(branch)] = np.maximum(
                1.0 - density * total_interacting,
                0.0,
            )
        else:
            per_branch[id(branch)] = density * probability
    return per_branch, density * distribution.absorbed_probability


def plane_interaction_budget(result, z_mm: float) -> PlaneInteractionBudget:
    """Calculate source-normalised interaction fractions at selected Z."""

    simulation = result.simulation
    state = result.state_snapshot
    selected = float(z_mm)
    if not math.isfinite(selected):
        raise ValueError("Selected axial position must be finite.")
    sample_z = float(state.sample.z_mm)
    incident = simulation.incident
    incident_weights = _ray_weights(incident)
    sample_alive = _reaches_plane(incident, sample_z)
    sample_incident = float(np.sum(incident_weights[sample_alive]))

    if selected < sample_z - 1.0e-9:
        reaches = _reaches_plane(incident, selected)
        current = float(np.sum(incident_weights[reaches]))
        stopped = max(1.0 - current, 0.0)
        channel = PlaneInteractionChannel(
            key="incident",
            label=CHANNEL_LABELS["incident"],
            probability_at_sample=1.0,
            source_fraction_at_plane=current,
            composition_at_plane=1.0 if current > 0.0 else 0.0,
            representative_loss_ev=0.0,
        )
        return PlaneInteractionBudget(
            z_mm=selected,
            sample_z_mm=sample_z,
            location="upstream of sample",
            source_fraction_at_plane=current,
            sample_incident_source_fraction=sample_incident,
            pre_sample_stopped_source_fraction=stopped,
            sample_absorbed_source_fraction=0.0,
            downstream_stopped_source_fraction=0.0,
            channels=(channel,),
            model="geometric source-current transport before specimen",
            material_name=None,
            mean_inelastic_events=None,
            total_inelastic_mean_free_path_nm=None,
            warnings=(),
            conservation_error=abs(current + stopped - 1.0),
        )

    mode = str(getattr(state.sample, "specimen_mode", "atomic")).lower()
    probabilities = _branch_probabilities(simulation)
    per_ray_probabilities: dict[int, np.ndarray] = {}
    absorbed_per_ray = np.zeros_like(incident_weights)
    model = str(
        getattr(simulation, "metrics", {}).get(
            "sample_scattering_model", "unknown"
        )
    )
    warnings: list[str] = []
    material_name = None
    mean_events = None
    total_mfp = None
    real_distribution = getattr(simulation, "real_interactions", None)
    energy_by_kind: dict[str, float] = {}

    virtual_density_active = bool(
        mode == "virtual"
        and bool(getattr(state.sample, "inserted", True))
        and bool(getattr(state.sample, "diffraction_enabled", True))
    )
    if virtual_density_active:
        per_ray_probabilities, absorbed_per_ray = (
            _virtual_per_ray_probabilities(
                state, simulation, sample_alive
            )
        )
        model += "; finite virtual density evaluated at every incident ray"
    else:
        for branch in simulation.branches.values():
            per_ray_probabilities[id(branch)] = np.full(
                incident_weights.size,
                probabilities[id(branch)],
                dtype=float,
            )
        missing = max(1.0 - sum(probabilities.values()), 0.0)
        absorbed_per_ray[sample_alive] = missing

    if real_distribution is not None:
        material_name = real_distribution.material_name
        mean_events = float(real_distribution.mean_inelastic_events)
        total_mfp = float(real_distribution.total_inelastic_mean_free_path_nm)
        warnings.extend(real_distribution.warnings)
        energy_by_kind = {
            channel.key: float(channel.energy_loss_ev)
            for channel in real_distribution.channels
        }
        warnings.append(
            "Elastic diffraction is coherent multislice intensity and can coexist with every energy-loss state; it is not reported as an exclusive per-electron collision label."
        )

    group_at_sample: dict[str, float] = {}
    group_at_plane: dict[str, float] = {}
    for branch in simulation.branches.values():
        kind = str(getattr(branch, "interaction_kind", "unknown"))
        local_probability = per_ray_probabilities[id(branch)]
        at_sample = float(
            np.sum(
                incident_weights
                * local_probability
                * sample_alive.astype(float)
            )
        )
        reaches = _reaches_plane(branch, selected)
        at_plane = float(
            np.sum(
                incident_weights
                * local_probability
                * reaches.astype(float)
            )
        )
        group_at_sample[kind] = group_at_sample.get(kind, 0.0) + at_sample
        group_at_plane[kind] = group_at_plane.get(kind, 0.0) + at_plane

    absorbed = float(np.sum(incident_weights * absorbed_per_ray))
    total_at_plane = float(sum(group_at_plane.values()))
    tracked_after_sample = float(sum(group_at_sample.values()))
    pre_sample_stopped = max(1.0 - sample_incident, 0.0)
    downstream_stopped = max(tracked_after_sample - total_at_plane, 0.0)
    channels = tuple(
        PlaneInteractionChannel(
            key=kind,
            label=CHANNEL_LABELS.get(
                kind, kind.replace("_", " ").title()
            ),
            probability_at_sample=(
                source_at_sample / sample_incident
                if sample_incident > 0.0 else 0.0
            ),
            source_fraction_at_plane=group_at_plane.get(kind, 0.0),
            composition_at_plane=(
                group_at_plane.get(kind, 0.0) / total_at_plane
                if total_at_plane > 0.0 else 0.0
            ),
            representative_loss_ev=energy_by_kind.get(kind),
        )
        for kind, source_at_sample in group_at_sample.items()
        if source_at_sample > 1.0e-15
        or group_at_plane.get(kind, 0.0) > 1.0e-15
    )
    conserved = (
        pre_sample_stopped
        + absorbed
        + downstream_stopped
        + total_at_plane
    )
    return PlaneInteractionBudget(
        z_mm=selected,
        sample_z_mm=sample_z,
        location=(
            "sample exit"
            if abs(selected - sample_z) <= 1.0e-9
            else "downstream of sample"
        ),
        source_fraction_at_plane=total_at_plane,
        sample_incident_source_fraction=sample_incident,
        pre_sample_stopped_source_fraction=pre_sample_stopped,
        sample_absorbed_source_fraction=absorbed,
        downstream_stopped_source_fraction=downstream_stopped,
        channels=channels,
        model=model,
        material_name=material_name,
        mean_inelastic_events=mean_events,
        total_inelastic_mean_free_path_nm=total_mfp,
        warnings=tuple(dict.fromkeys(warnings)),
        conservation_error=abs(conserved - 1.0),
    )

