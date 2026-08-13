"""Probability-conserving real-specimen inelastic electron transport.

This module deliberately separates coherent elastic multislice scattering
from stochastic energy-loss events.  Bulk plasmon/low-loss and aggregate
core-ionisation rates are anchored to material IMFP measurements.  Independent
events follow Poisson statistics; an optional effective absorption MFP removes
electrons from the tracked transmitted population.

The compact angular branches are deterministic quadrature samples for ray and
energy-filter transport.  They are not user-defined diffraction beams and are
not a replacement for an energy-differential dielectric/EELS calculation.

Method references: Iakoubovskii et al., Phys. Rev. B 77, 104102 (2008),
DOI 10.1103/PhysRevB.77.104102 (IMFP anchors, log-ratio Poisson statistics and
relativistic collection-angle factor); Stone and Kim, Surf. Interface Anal. 37,
966 (2005), DOI 10.1002/sia.2089 (BEB equation and limitations).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from temsim.specimen.presets import (
    InelasticMaterial,
    default_specimen_preset_key,
    load_specimen_preset,
)


BOHR_RADIUS_M = 5.291_772_109_03e-11
RYDBERG_ENERGY_EV = 13.605_693_122_994


@dataclass(frozen=True, slots=True)
class RealInteractionChannel:
    key: str
    label: str
    probability: float
    mean_events: float
    energy_loss_ev: float
    characteristic_angle_mrad: float
    mean_free_path_nm: float
    approximation: str


@dataclass(frozen=True, slots=True)
class RealInteractionDistribution:
    material_key: str
    material_name: str
    thickness_nm: float
    beam_energy_kev: float
    channels: tuple[RealInteractionChannel, ...]
    absorbed_probability: float
    mean_inelastic_events: float
    total_inelastic_mean_free_path_nm: float
    plasmon_mean_free_path_nm: float
    ionisation_mean_free_path_nm: float
    other_mean_free_path_nm: float
    absorption_mean_free_path_nm: float
    reference: str
    applicability: str
    model: str
    warnings: tuple[str, ...]

    @property
    def tracked_probability(self) -> float:
        return float(sum(channel.probability for channel in self.channels))

    @property
    def total_probability(self) -> float:
        return self.tracked_probability + self.absorbed_probability

    def metrics(self) -> dict[str, object]:
        return {
            "material_key": self.material_key,
            "material_name": self.material_name,
            "thickness_nm": self.thickness_nm,
            "beam_energy_kev": self.beam_energy_kev,
            "mean_inelastic_events": self.mean_inelastic_events,
            "total_inelastic_mean_free_path_nm": (
                self.total_inelastic_mean_free_path_nm
            ),
            "plasmon_mean_free_path_nm": self.plasmon_mean_free_path_nm,
            "ionisation_mean_free_path_nm": self.ionisation_mean_free_path_nm,
            "other_mean_free_path_nm": self.other_mean_free_path_nm,
            "absorption_mean_free_path_nm": self.absorption_mean_free_path_nm,
            "absorbed_probability": self.absorbed_probability,
            "tracked_probability": self.tracked_probability,
            "probability_sum": self.total_probability,
            "channels": tuple(
                {
                    "key": channel.key,
                    "label": channel.label,
                    "probability": channel.probability,
                    "mean_events": channel.mean_events,
                    "energy_loss_ev": channel.energy_loss_ev,
                    "characteristic_angle_mrad": (
                        channel.characteristic_angle_mrad
                    ),
                    "mean_free_path_nm": channel.mean_free_path_nm,
                    "approximation": channel.approximation,
                }
                for channel in self.channels
            ),
            "reference": self.reference,
            "applicability": self.applicability,
            "model": self.model,
            "warnings": self.warnings,
        }


@dataclass(frozen=True, slots=True)
class RealInelasticRayBranch:
    name: str
    kick_x_rad: float | np.ndarray
    kick_y_rad: float | np.ndarray
    probability: float
    interaction_kind: str
    energy_loss_ev: float


def _finite_nonnegative(name: str, value: float) -> float:
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return converted


def _positive_override(sample, field: str) -> float | None:
    value = _finite_nonnegative(field, getattr(sample, field, 0.0))
    return value if value > 0.0 else None


def _material_for_state(state) -> tuple[str, str, InelasticMaterial | None, list[str]]:
    sample = state.sample
    warnings: list[str] = []
    cif_value = str(getattr(sample, "cif_path", "")).strip()
    if not cif_value:
        key = (
            str(getattr(sample, "specimen_preset_key", "")).strip()
            or default_specimen_preset_key()
        )
        preset = load_specimen_preset(key)
        return key, preset.name, preset.inelastic, warnings

    path = Path(cif_value).expanduser().resolve()
    warnings.append(
        "A custom CIF never borrows a preset's inelastic material constants; "
        "set explicit plasmon/ionisation MFP and loss-energy overrides from "
        "a measurement or validated material model."
    )
    return f"cif:{path.name}", f"Custom CIF: {path.name}", None, warnings


def relativistic_eels_factor(beam_energy_kev: float) -> float:
    """Return the conventional EELS relativistic factor F."""

    energy = float(beam_energy_kev)
    if not math.isfinite(energy) or energy <= 0.0:
        raise ValueError("Beam energy must be finite and positive.")
    return (1.0 + energy / 1022.0) / (1.0 + energy / 511.0) ** 2


def characteristic_inelastic_angle_mrad(
    loss_ev: float, beam_energy_kev: float
) -> float:
    loss = float(loss_ev)
    energy_ev = float(beam_energy_kev) * 1.0e3
    if not math.isfinite(loss) or loss < 0.0:
        raise ValueError("Energy loss must be finite and non-negative.")
    return 1.0e3 * loss / (
        2.0 * relativistic_eels_factor(beam_energy_kev) * energy_ev
    )


def _plasmon_mfp_at_energy_nm(
    reference_mfp_nm: float,
    reference_energy_kev: float,
    beam_energy_kev: float,
    loss_ev: float,
    collection_semiangle_mrad: float,
) -> float:
    """Scale a measured IMFP using the relativistic log-angle factor.

    This is the energy dependence of the Kramers--Kronig/log-ratio geometry,
    normalised to the measured material value.  It avoids treating the
    absolute free-electron model as more accurate than the experiment.
    """

    collection_rad = float(collection_semiangle_mrad) * 1.0e-3

    def transport_factor(energy_kev: float) -> float:
        factor = relativistic_eels_factor(energy_kev)
        characteristic = (
            characteristic_inelastic_angle_mrad(loss_ev, energy_kev) * 1.0e-3
        )
        angular_log = math.log1p(
            (collection_rad / max(characteristic, 1.0e-15)) ** 2
        )
        return factor * energy_kev / max(angular_log, 1.0e-15)

    return float(reference_mfp_nm) * (
        transport_factor(beam_energy_kev)
        / transport_factor(reference_energy_kev)
    )


def beb_ionisation_cross_section_m2(
    incident_energy_ev: float,
    binding_energy_ev: float,
    *,
    occupancy: float = 1.0,
    orbital_kinetic_energy_ev: float | None = None,
) -> float:
    """Binary-Encounter-Bethe cross section for one aggregate orbital.

    Implements Eq. (1) of Stone and Kim, DOI 10.1002/sia.2089.  When no
    orbital kinetic energy is available, ``U=B`` is the stated virial
    approximation.  Here BEB is used only for voltage scaling of an
    experimentally anchored core-loss rate, not as its absolute magnitude.
    """

    incident = float(incident_energy_ev)
    binding = float(binding_energy_ev)
    number = float(occupancy)
    kinetic = binding if orbital_kinetic_energy_ev is None else float(
        orbital_kinetic_energy_ev
    )
    if not all(math.isfinite(value) for value in (incident, binding, number, kinetic)):
        raise ValueError("BEB inputs must be finite.")
    if binding <= 0.0 or kinetic <= 0.0 or number <= 0.0:
        raise ValueError("BEB binding energy, kinetic energy and occupancy must be positive.")
    if incident <= binding:
        return 0.0
    t = incident / binding
    u = kinetic / binding
    scale = (
        4.0
        * math.pi
        * BOHR_RADIUS_M**2
        * number
        * (RYDBERG_ENERGY_EV / binding) ** 2
    )
    bracket = (
        0.5 * math.log(t) * (1.0 - 1.0 / t**2)
        + 1.0
        - 1.0 / t
        - math.log(t) / (t + 1.0)
    )
    return scale * bracket / (t + u + 1.0)


def _ionisation_mfp_at_energy_nm(
    reference_mfp_nm: float,
    reference_energy_kev: float,
    beam_energy_kev: float,
    representative_loss_ev: float,
) -> float:
    reference_sigma = beb_ionisation_cross_section_m2(
        reference_energy_kev * 1.0e3,
        representative_loss_ev,
    )
    current_sigma = beb_ionisation_cross_section_m2(
        beam_energy_kev * 1.0e3,
        representative_loss_ev,
    )
    if current_sigma <= 0.0:
        return math.inf
    return float(reference_mfp_nm) * reference_sigma / current_sigma


def _harmonic_mean_free_path(*values: float) -> float:
    rate = sum(
        1.0 / value
        for value in values
        if math.isfinite(value) and value > 0.0
    )
    return math.inf if rate <= 0.0 else 1.0 / rate


def _disabled_distribution(
    *,
    material_key: str,
    material_name: str,
    thickness_nm: float,
    beam_energy_kev: float,
    warnings: list[str],
    model: str,
) -> RealInteractionDistribution:
    channel = RealInteractionChannel(
        key="real_zero_loss",
        label="Zero-loss / elastic-coherent population",
        probability=1.0,
        mean_events=0.0,
        energy_loss_ev=0.0,
        characteristic_angle_mrad=0.0,
        mean_free_path_nm=math.inf,
        approximation=(
            "No stochastic energy-loss event; coherent elastic redistribution remains owned by multislice."
        ),
    )
    return RealInteractionDistribution(
        material_key=material_key,
        material_name=material_name,
        thickness_nm=thickness_nm,
        beam_energy_kev=beam_energy_kev,
        channels=(channel,),
        absorbed_probability=0.0,
        mean_inelastic_events=0.0,
        total_inelastic_mean_free_path_nm=math.inf,
        plasmon_mean_free_path_nm=math.inf,
        ionisation_mean_free_path_nm=math.inf,
        other_mean_free_path_nm=math.inf,
        absorption_mean_free_path_nm=math.inf,
        reference="",
        applicability="",
        model=model,
        warnings=tuple(warnings),
    )


def real_inelastic_distribution(state) -> RealInteractionDistribution:
    """Resolve material data and return an exclusive, conserved event budget."""

    sample = state.sample
    thickness = _finite_nonnegative("Sample thickness", sample.thickness_nm)
    energy = float(state.beam_voltage_kv)
    material_key, material_name, material, warnings = _material_for_state(state)
    active = bool(
        getattr(sample, "inserted", True)
        and str(getattr(sample, "specimen_mode", "atomic")).lower() == "atomic"
        and getattr(sample, "real_inelastic_enabled", True)
        and thickness > 0.0
    )
    if not active:
        reason = (
            "sample_not_interacting"
            if not bool(getattr(sample, "inserted", True)) or thickness <= 0.0
            else "real_inelastic_disabled"
        )
        return _disabled_distribution(
            material_key=material_key,
            material_name=material_name,
            thickness_nm=thickness,
            beam_energy_kev=energy,
            warnings=warnings,
            model=reason,
        )

    plasmon_override = _positive_override(
        sample, "real_plasmon_mean_free_path_nm"
    )
    ionisation_override = _positive_override(
        sample, "real_ionisation_mean_free_path_nm"
    )
    plasmon_loss_override = _positive_override(sample, "real_plasmon_energy_ev")
    ionisation_loss_override = _positive_override(
        sample, "real_ionisation_energy_ev"
    )
    other_mfp = _positive_override(
        sample, "real_other_inelastic_mean_free_path_nm"
    ) or math.inf
    absorption_mfp = _positive_override(
        sample, "real_absorption_mean_free_path_nm"
    ) or math.inf
    other_loss = float(getattr(sample, "real_other_inelastic_energy_ev", 50.0))
    if not math.isfinite(other_loss) or other_loss <= 0.0:
        raise ValueError("Other inelastic loss energy must be finite and positive.")

    if material is None:
        plasmon_complete = (
            plasmon_override is not None and plasmon_loss_override is not None
        )
        ionisation_complete = (
            ionisation_override is not None
            and ionisation_loss_override is not None
        )
        if (plasmon_override is None) != (plasmon_loss_override is None):
            warnings.append(
                "Incomplete custom-CIF plasmon override was ignored; both MFP and loss energy are required."
            )
        if (ionisation_override is None) != (ionisation_loss_override is None):
            warnings.append(
                "Incomplete custom-CIF ionisation override was ignored; both MFP and loss energy are required."
            )
        if not plasmon_complete:
            plasmon_override = None
            plasmon_loss_override = None
        if not ionisation_complete:
            ionisation_override = None
            ionisation_loss_override = None
        explicit_channel_active = bool(
            plasmon_complete
            or ionisation_complete
            or math.isfinite(absorption_mfp)
            or math.isfinite(other_mfp)
        )
        if not explicit_channel_active:
            warnings.append(
                "Inelastic collisions are inactive because this material has no validated anchor or complete overrides."
            )
            return _disabled_distribution(
                material_key=material_key,
                material_name=material_name,
                thickness_nm=thickness,
                beam_energy_kev=energy,
                warnings=warnings,
                model="material_data_unavailable",
            )
        warnings.append(
            "Only explicitly supplied custom-CIF channels are active; plasmon and ionisation each require both an MFP and a loss-energy override."
        )

    plasmon_loss = (
        plasmon_loss_override
        if plasmon_loss_override is not None
        else float(material.plasmon_energy_ev)
        if material is not None
        else 0.0
    )
    ionisation_loss = (
        ionisation_loss_override
        if ionisation_loss_override is not None
        else float(material.ionisation_energy_ev)
        if material is not None
        else 0.0
    )
    beam_energy_ev = energy * 1.0e3
    for label, loss in (
        ("Plasmon", plasmon_loss),
        ("Ionisation", ionisation_loss),
        ("Other inelastic", other_loss),
    ):
        if loss < 0.0 or loss >= beam_energy_ev:
            raise ValueError(
                f"{label} representative loss must be non-negative and below the beam energy."
            )

    if plasmon_override is not None:
        plasmon_mfp = plasmon_override
    elif material is not None:
        plasmon_mfp = _plasmon_mfp_at_energy_nm(
            material.plasmon_mean_free_path_nm,
            material.reference_energy_kev,
            energy,
            plasmon_loss,
            material.collection_semiangle_mrad,
        )
    else:
        plasmon_mfp = math.inf

    if ionisation_override is not None:
        ionisation_mfp = ionisation_override
    elif material is not None:
        reference_core_rate = max(
            1.0 / material.total_mean_free_path_nm
            - 1.0 / material.plasmon_mean_free_path_nm,
            0.0,
        )
        reference_core_mfp = (
            math.inf if reference_core_rate <= 0.0 else 1.0 / reference_core_rate
        )
        ionisation_mfp = _ionisation_mfp_at_energy_nm(
            reference_core_mfp,
            material.reference_energy_kev,
            energy,
            ionisation_loss,
        )
    else:
        ionisation_mfp = math.inf

    rates = {
        "real_plasmon": (
            0.0 if not math.isfinite(plasmon_mfp) else 1.0 / plasmon_mfp
        ),
        "real_ionisation": (
            0.0 if not math.isfinite(ionisation_mfp) else 1.0 / ionisation_mfp
        ),
        "real_other_inelastic": (
            0.0 if not math.isfinite(other_mfp) else 1.0 / other_mfp
        ),
    }
    means = {key: thickness * rate for key, rate in rates.items()}
    total_mean = float(sum(means.values()))
    no_event = math.exp(-total_mean)
    absorption_survival = (
        1.0
        if not math.isfinite(absorption_mfp)
        else math.exp(-thickness / absorption_mfp)
    )
    single = {key: no_event * value for key, value in means.items()}
    plural = max(1.0 - no_event * (1.0 + total_mean), 0.0)

    loss_by_key = {
        "real_plasmon": plasmon_loss,
        "real_ionisation": ionisation_loss,
        "real_other_inelastic": other_loss,
    }
    angle_by_key = {
        key: characteristic_inelastic_angle_mrad(loss, energy)
        for key, loss in loss_by_key.items()
    }
    mfp_by_key = {
        "real_plasmon": plasmon_mfp,
        "real_ionisation": ionisation_mfp,
        "real_other_inelastic": other_mfp,
    }
    labels = {
        "real_plasmon": "Single bulk plasmon / low-loss event",
        "real_ionisation": "Single core ionisation event",
        "real_other_inelastic": "Single other inelastic event",
    }
    approximations = {
        "real_plasmon": (
            "Measured plasmon-component IMFP anchor with relativistic log-angle voltage scaling."
        ),
        "real_ionisation": (
            "Measured residual core-loss rate; BEB (U=B) supplies voltage scaling only."
        ),
        "real_other_inelastic": "Explicit user-supplied effective MFP and representative loss.",
    }
    channels = [
        RealInteractionChannel(
            key="real_zero_loss",
            label="Zero-loss / elastic-coherent population",
            probability=absorption_survival * no_event,
            mean_events=0.0,
            energy_loss_ev=0.0,
            characteristic_angle_mrad=0.0,
            mean_free_path_nm=_harmonic_mean_free_path(
                plasmon_mfp, ionisation_mfp, other_mfp
            ),
            approximation=(
                "No stochastic energy-loss event; elastic diffraction remains a coherent multislice intensity, not an exclusive ray label."
            ),
        )
    ]
    for key in ("real_plasmon", "real_ionisation", "real_other_inelastic"):
        if means[key] <= 0.0:
            continue
        channels.append(
            RealInteractionChannel(
                key=key,
                label=labels[key],
                probability=absorption_survival * single[key],
                mean_events=means[key],
                energy_loss_ev=loss_by_key[key],
                characteristic_angle_mrad=angle_by_key[key],
                mean_free_path_nm=mfp_by_key[key],
                approximation=approximations[key],
            )
        )
    if plural > 0.0 and total_mean > 0.0:
        conditional_count = (
            total_mean * (1.0 - math.exp(-total_mean)) / max(plural, 1.0e-300)
        )
        mean_loss_per_event = sum(
            means[key] * loss_by_key[key] for key in means
        ) / total_mean
        rms_angle_per_event = math.sqrt(
            sum(means[key] * angle_by_key[key] ** 2 for key in means)
            / total_mean
        )
        channels.append(
            RealInteractionChannel(
                key="real_plural_inelastic",
                label="Plural inelastic events (two or more)",
                probability=absorption_survival * plural,
                mean_events=conditional_count,
                energy_loss_ev=conditional_count * mean_loss_per_event,
                characteristic_angle_mrad=(
                    math.sqrt(conditional_count) * rms_angle_per_event
                ),
                mean_free_path_nm=_harmonic_mean_free_path(
                    plasmon_mfp, ionisation_mfp, other_mfp
                ),
                approximation=(
                    "Exact Poisson probability; ray transport uses the conditional mean loss and RMS characteristic angle."
                ),
            )
        )

    absorbed = 1.0 - absorption_survival
    total_mfp = _harmonic_mean_free_path(
        plasmon_mfp, ionisation_mfp, other_mfp
    )
    if total_mean > 2.0:
        warnings.append(
            "The specimen exceeds two inelastic mean events; the plural-event probability is valid, but one mean-energy ray quadrature cannot reproduce the full EELS spectrum."
        )
    if math.isfinite(absorption_mfp):
        warnings.append(
            "Effective absorption means removal from the tracked transmitted beam; it is not surface adsorption of a high-energy TEM electron."
        )
    if material is not None and material.applicability.strip():
        applicability = material.applicability
    else:
        applicability = "User-supplied effective transport parameters."
    reference = material.reference if material is not None else "User overrides"
    result = RealInteractionDistribution(
        material_key=material_key,
        material_name=material_name,
        thickness_nm=thickness,
        beam_energy_kev=energy,
        channels=tuple(channels),
        absorbed_probability=absorbed,
        mean_inelastic_events=total_mean,
        total_inelastic_mean_free_path_nm=total_mfp,
        plasmon_mean_free_path_nm=plasmon_mfp,
        ionisation_mean_free_path_nm=ionisation_mfp,
        other_mean_free_path_nm=other_mfp,
        absorption_mean_free_path_nm=absorption_mfp,
        reference=reference,
        applicability=applicability,
        model=(
            "independent Poisson energy-loss events; measured 200-keV IMFP anchors; relativistic log-angle plasmon scaling; BEB-relative ionisation scaling"
        ),
        warnings=tuple(warnings),
    )
    if not math.isclose(
        result.total_probability, 1.0, rel_tol=0.0, abs_tol=2.0e-12
    ):
        raise RuntimeError(
            "Real inelastic interaction probabilities failed conservation: "
            f"{result.total_probability:.17g}."
        )
    return result


def real_inelastic_ray_branches(
    distribution: RealInteractionDistribution,
    *,
    ray_count: int = 1,
) -> tuple[RealInelasticRayBranch, ...]:
    """Return one weighted population per exclusive energy-loss state.

    Within each nonzero-loss population, source rays sample an evenly spaced
    azimuthal ring at that state's characteristic angle.  This is rotationally
    balanced while avoiding four duplicate full-column histories per state.
    """

    count = int(ray_count)
    if count <= 0:
        raise ValueError("Real inelastic ray transport needs a positive ray count.")
    branches: list[RealInelasticRayBranch] = []
    for channel in distribution.channels:
        if channel.key == "real_zero_loss":
            branches.append(
                RealInelasticRayBranch(
                    name="000",
                    kick_x_rad=0.0,
                    kick_y_rad=0.0,
                    probability=channel.probability,
                    interaction_kind=channel.key,
                    energy_loss_ev=0.0,
                )
            )
            continue
        if channel.probability <= 0.0:
            continue
        angle_rad = channel.characteristic_angle_mrad * 1.0e-3
        azimuth = 2.0 * math.pi * np.arange(count, dtype=float) / count
        kick_x = angle_rad * np.cos(azimuth)
        kick_y = angle_rad * np.sin(azimuth)
        kick_x.setflags(write=False)
        kick_y.setflags(write=False)
        branches.append(
            RealInelasticRayBranch(
                name=channel.key,
                kick_x_rad=kick_x,
                kick_y_rad=kick_y,
                probability=channel.probability,
                interaction_kind=channel.key,
                energy_loss_ev=channel.energy_loss_ev,
            )
        )
    if not math.isclose(
        sum(branch.probability for branch in branches)
        + distribution.absorbed_probability,
        1.0,
        rel_tol=0.0,
        abs_tol=2.0e-12,
    ):
        raise RuntimeError("Real inelastic ray quadrature failed probability conservation.")
    return tuple(branches)
