"""Probability-conserving EELS/EFTEM forward model.

The forward chain has explicit, inspectable boundaries:

1. specimen loss distribution (ZLP, single plasmon, single core loss,
   optional other loss, and plural scattering);
2. incident-energy broadening;
3. energy-filter/slit transmission;
4. detector point-spread, efficiency and optional counting noise.

The specimen probabilities come exclusively from
``real_inelastic_distribution``.  This module does not fit composition or
invent material constants.  The present loss peaks are representative-energy
transport channels, not a dielectric-function or edge fine-structure model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve
from scipy.stats import poisson

from temsim.specimen.inelastic import (
    RealInteractionDistribution,
    real_inelastic_distribution,
)


def _frozen(values) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("EELS arrays must be finite")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class EELSDetectorResponse:
    """Explicit detector boundary; defaults represent an ideal detector."""

    efficiency: float = 1.0
    energy_resolution_fwhm_ev: float = 0.0
    poisson_enabled: bool = False
    poisson_seed: int = 0
    provenance: str = "ideal scalar detector response; no OEM curve supplied"

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.efficiency)) or not (
            0.0 <= self.efficiency <= 1.0
        ):
            raise ValueError("EELS detector efficiency must be in [0, 1]")
        if (
            not math.isfinite(float(self.energy_resolution_fwhm_ev))
            or self.energy_resolution_fwhm_ev < 0.0
        ):
            raise ValueError("EELS detector resolution cannot be negative")
        if int(self.poisson_seed) < 0:
            raise ValueError("EELS Poisson seed cannot be negative")


@dataclass(frozen=True, slots=True)
class SpectrometerTransmission:
    energy_loss_ev: np.ndarray
    transmission: np.ndarray
    model: str
    provenance: str
    boundaries: tuple[str, ...]

    def __post_init__(self) -> None:
        energy = _frozen(self.energy_loss_ev)
        transmission = _frozen(self.transmission)
        if energy.ndim != 1 or transmission.shape != energy.shape:
            raise ValueError("Spectrometer transmission axis is invalid")
        if np.any((transmission < 0.0) | (transmission > 1.0)):
            raise ValueError("Spectrometer transmission must be in [0, 1]")
        object.__setattr__(self, "energy_loss_ev", energy)
        object.__setattr__(self, "transmission", transmission)


@dataclass(frozen=True, slots=True)
class EELSForwardResult:
    energy_loss_ev: np.ndarray
    generated_probability: np.ndarray
    source_broadened_probability: np.ndarray
    spectrometer_probability: np.ndarray
    detected_expected_counts: np.ndarray
    detected_sampled_counts: np.ndarray | None
    generated_components: Mapping[str, np.ndarray]
    detected_components: Mapping[str, np.ndarray]
    spectrometer_transmission: SpectrometerTransmission
    absorbed_probability: float
    metrics: dict[str, object]

    def __post_init__(self) -> None:
        axis = _frozen(self.energy_loss_ev)
        arrays = tuple(
            _frozen(value)
            for value in (
                self.generated_probability,
                self.source_broadened_probability,
                self.spectrometer_probability,
                self.detected_expected_counts,
            )
        )
        if axis.ndim != 1 or any(value.shape != axis.shape for value in arrays):
            raise ValueError("EELS result arrays must share one 1-D energy axis")
        generated = {
            str(key): _frozen(value)
            for key, value in self.generated_components.items()
        }
        detected = {
            str(key): _frozen(value)
            for key, value in self.detected_components.items()
        }
        if any(value.shape != axis.shape for value in (*generated.values(), *detected.values())):
            raise ValueError("EELS component arrays must match the energy axis")
        sampled = self.detected_sampled_counts
        if sampled is not None:
            sampled = np.ascontiguousarray(sampled, dtype=np.int64)
            if sampled.shape != axis.shape or np.any(sampled < 0):
                raise ValueError("EELS sampled counts are invalid")
            sampled.setflags(write=False)
        object.__setattr__(self, "energy_loss_ev", axis)
        object.__setattr__(self, "generated_probability", arrays[0])
        object.__setattr__(self, "source_broadened_probability", arrays[1])
        object.__setattr__(self, "spectrometer_probability", arrays[2])
        object.__setattr__(self, "detected_expected_counts", arrays[3])
        object.__setattr__(self, "detected_sampled_counts", sampled)
        object.__setattr__(self, "generated_components", generated)
        object.__setattr__(self, "detected_components", detected)

    @property
    def expected_total_counts(self) -> float:
        return float(np.sum(self.detected_expected_counts))

    @property
    def eftem_transmitted_probability(self) -> float:
        return float(np.sum(self.spectrometer_probability))


def _uniform_axis(energy_edges_ev) -> tuple[np.ndarray, float]:
    edges = np.asarray(energy_edges_ev, dtype=float)
    if (
        edges.ndim != 1
        or edges.size < 2
        or not np.all(np.isfinite(edges))
        or np.any(np.diff(edges) <= 0.0)
    ):
        raise ValueError("EELS energy edges must be finite and increasing")
    widths = np.diff(edges)
    width = float(widths[0])
    if not np.allclose(widths, width, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("EELS forward convolution requires uniform energy bins")
    return 0.5 * (edges[:-1] + edges[1:]), width


def _deposit_linear(array: np.ndarray, value_ev: float, weight: float, step_ev: float):
    coordinate = float(value_ev) / float(step_ev)
    lower = int(math.floor(coordinate))
    fraction = coordinate - lower
    if 0 <= lower < array.size:
        array[lower] += float(weight) * (1.0 - fraction)
    if fraction > 0.0 and 0 <= lower + 1 < array.size:
        array[lower + 1] += float(weight) * fraction


def _poisson_kernel(
    mean_events: float,
    loss_ev: float,
    step_ev: float,
    size: int,
    tail_probability: float,
) -> tuple[np.ndarray, float]:
    mean = float(mean_events)
    if mean <= 0.0:
        result = np.zeros(size, dtype=float)
        result[0] = 1.0
        return result, 0.0
    maximum_count = max(
        1,
        int(math.ceil(float(poisson.ppf(1.0 - tail_probability, mean)))),
    )
    counts = np.arange(maximum_count + 1, dtype=int)
    probabilities = poisson.pmf(counts, mean)
    result = np.zeros(size, dtype=float)
    for count, probability in zip(counts, probabilities, strict=True):
        _deposit_linear(
            result,
            float(count) * float(loss_ev),
            float(probability),
            step_ev,
        )
    omitted = max(1.0 - float(np.sum(probabilities)), 0.0)
    return result, omitted


def _raw_compound_poisson_components(
    distribution: RealInteractionDistribution,
    *,
    maximum_loss_ev: float,
    energy_step_ev: float,
    tail_probability: float,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    size = max(int(math.floor(maximum_loss_ev / energy_step_ev)) + 1, 1)
    channels = {
        channel.key: channel
        for channel in distribution.channels
        if channel.key in {
            "real_plasmon",
            "real_ionisation",
            "real_other_inelastic",
        }
        and channel.mean_events > 0.0
    }
    means = {
        key: float(channel.mean_events) for key, channel in channels.items()
    }
    total_mean = float(sum(means.values()))
    survival = max(1.0 - float(distribution.absorbed_probability), 0.0)
    full = np.zeros(size, dtype=float)
    full[0] = 1.0
    omitted_poisson_tail = 0.0
    for channel in channels.values():
        kernel, omitted = _poisson_kernel(
            channel.mean_events,
            channel.energy_loss_ev,
            energy_step_ev,
            size,
            tail_probability,
        )
        full = fftconvolve(full, kernel, mode="full")[:size]
        omitted_poisson_tail += omitted
    full *= survival
    zlp = np.zeros(size, dtype=float)
    zlp[0] = survival * math.exp(-total_mean)
    singles: dict[str, np.ndarray] = {}
    names = {
        "real_plasmon": "single_plasmon",
        "real_ionisation": "single_core_loss",
        "real_other_inelastic": "single_other_loss",
    }
    for key, channel in channels.items():
        contribution = np.zeros(size, dtype=float)
        probability = survival * math.exp(-total_mean) * means[key]
        _deposit_linear(
            contribution,
            channel.energy_loss_ev,
            probability,
            energy_step_ev,
        )
        singles[names[key]] = contribution
    accounted = zlp.copy()
    for contribution in singles.values():
        accounted += contribution
    plural = np.maximum(full - accounted, 0.0)
    components = {"zero_loss": zlp, **singles, "plural_scattering": plural}
    return components, {
        "poisson_tail_upper_bound": min(omitted_poisson_tail, 1.0),
        "energy_axis_omitted_probability": max(
            survival - float(np.sum(full)), 0.0
        ),
        "tracked_probability_on_raw_axis": float(np.sum(full)),
    }


def _map_raw_to_axis(
    raw: np.ndarray,
    raw_step_ev: float,
    target_edges_ev: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Conservatively histogram raw point probabilities onto target bins.

    The compound-Poisson grid stores probability masses at
    ``index * raw_step_ev``; it is not a sampled probability density.  Mapping
    those masses by interpolation between target *centres* loses half of a
    zero-loss peak when the target axis starts at 0 eV.  Instead, use the
    actual detector-bin edges: bins are left-inclusive, and the final bin also
    includes its upper edge (the same convention as ``numpy.histogram``).

    Probability outside the requested energy window is returned separately.
    It is deliberately not clipped into the first or last detector bin.
    """

    values = np.asarray(raw, dtype=float)
    edges = np.asarray(target_edges_ev, dtype=float)
    step = float(raw_step_ev)
    if (
        values.ndim != 1
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
    ):
        raise ValueError(
            "Raw EELS probabilities must be finite, non-negative 1-D data"
        )
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("Raw EELS energy step must be positive")
    if (
        edges.ndim != 1
        or edges.size < 2
        or not np.all(np.isfinite(edges))
        or np.any(np.diff(edges) <= 0.0)
    ):
        raise ValueError("Target EELS bin edges must be finite and increasing")

    target = np.zeros(edges.size - 1, dtype=float)
    populated = np.flatnonzero(values > 0.0)
    if populated.size == 0:
        return target, 0.0
    energies = populated.astype(float) * step
    bin_indices = np.searchsorted(edges, energies, side="right") - 1
    # ``numpy.histogram`` includes an observation exactly on the final edge.
    bin_indices[energies == edges[-1]] = target.size - 1
    inside = (
        (energies >= edges[0])
        & (energies <= edges[-1])
        & (bin_indices >= 0)
        & (bin_indices < target.size)
    )
    np.add.at(target, bin_indices[inside], values[populated[inside]])
    omitted = float(np.sum(values[populated[~inside]], dtype=float))
    return target, omitted


def _broaden(values: np.ndarray, fwhm_ev: float, step_ev: float) -> np.ndarray:
    if float(fwhm_ev) <= 0.0:
        return np.array(values, copy=True)
    sigma_bins = float(fwhm_ev) / (
        2.0 * math.sqrt(2.0 * math.log(2.0)) * float(step_ev)
    )
    return gaussian_filter1d(
        np.asarray(values, dtype=float),
        sigma=sigma_bins,
        mode="constant",
        cval=0.0,
        truncate=6.0,
    )


def ideal_spectrometer_transmission(
    energy_loss_ev,
) -> SpectrometerTransmission:
    energy = np.asarray(energy_loss_ev, dtype=float)
    return SpectrometerTransmission(
        energy,
        np.ones_like(energy),
        "ideal_no_filter_boundary",
        "explicit ideal transmission",
        (),
    )


def first_order_energy_filter_transmission(
    state,
    energy_loss_ev,
) -> SpectrometerTransmission:
    """Apply configured slit bounds without claiming full ray interception."""

    from temsim.optics.energy_filter import ensure_energy_filter

    ensure_energy_filter(state)
    energy_filter = state.energy_filter
    energy = np.asarray(energy_loss_ev, dtype=float)
    if not bool(energy_filter.enabled):
        return SpectrometerTransmission(
            energy,
            np.zeros_like(energy),
            "energy_filter_disabled",
            "current EnergyFilterSystem state",
            ("filter_disabled",),
        )
    slit = energy_filter.energy_slit
    transmission = np.ones_like(energy)
    boundaries = ["spectrometer_dispersion"]
    if bool(slit.inserted):
        centre = float(slit.derived_centre_loss_ev)
        half_width = 0.5 * float(slit.derived_width_ev)
        transmission *= (
            (energy >= centre - half_width)
            & (energy <= centre + half_width)
        )
        boundaries.append("physical_energy_selection_slit")
    mode = str(getattr(energy_filter, "operating_mode", "eels")).lower()
    detector = energy_filter.zebra_detector
    if mode == "eftem":
        if not bool(energy_filter.output_detector_inserted):
            transmission[:] = 0.0
            boundaries.append("eftem_output_plane_retracted")
        else:
            boundaries.append("eftem_output_image_plane")
    elif not bool(detector.enabled and detector.inserted):
        transmission[:] = 0.0
        boundaries.append("zebra_detector_retracted_or_disabled")
    else:
        boundaries.append("zebra_active_state")
    return SpectrometerTransmission(
        energy,
        transmission,
        f"first_order_calibrated_{mode}_energy_window",
        (
            "uses the current calibrated slit energy window and detector "
            "state; transverse aberration/clipping requires Boris ray tracing"
        ),
        tuple(boundaries),
    )


def source_energy_fwhm_ev(state, simulation=None) -> tuple[float, str]:
    gun_trace = getattr(simulation, "gun_trace", None)
    traced = getattr(gun_trace, "output_energy_fwhm_ev", None)
    if traced is not None and math.isfinite(float(traced)) and traced >= 0.0:
        return float(traced), "electron-gun exit trace"
    gun = getattr(state, "electron_gun", None)
    emitter = getattr(gun, "emitter", None)
    value = getattr(
        emitter,
        "energy_spread_fwhm_ev",
        getattr(gun, "energy_spread_fwhm_ev", 0.0),
    )
    return float(value), "configured electron-gun source"


def energy_axis_from_filter(state) -> np.ndarray:
    """Return Zebra pixel edges using the live calibrated dispersion."""

    from temsim.optics.energy_filter import ensure_energy_filter

    ensure_energy_filter(state)
    energy_filter = state.energy_filter
    detector = energy_filter.zebra_detector
    dispersion = abs(
        float(energy_filter.energy_slit.calibrated_dispersion_um_per_ev)
    )
    if dispersion <= 0.0:
        raise ValueError("Energy-filter dispersion must be non-zero")
    step = float(detector.pixel_size_um) / dispersion
    count = int(detector.pixels_per_strip)
    centre = float(energy_filter.energy_slit.derived_centre_loss_ev)
    return centre + (np.arange(count + 1, dtype=float) - 0.5 * count) * step


def simulate_eels_forward(
    state,
    *,
    incident_electrons: float = 1.0,
    simulation=None,
    distribution: RealInteractionDistribution | None = None,
    energy_edges_ev=None,
    spectrometer: SpectrometerTransmission | None = None,
    detector_response: EELSDetectorResponse | None = None,
    compound_poisson_tail_probability: float = 1.0e-10,
) -> EELSForwardResult:
    """Run the complete specimen-to-EELS-detector forward chain."""

    incident = float(incident_electrons)
    if not math.isfinite(incident) or incident < 0.0:
        raise ValueError("Incident EELS electron count must be non-negative")
    tail = float(compound_poisson_tail_probability)
    if not 0.0 < tail < 1.0:
        raise ValueError("Compound-Poisson tail tolerance must lie in (0, 1)")
    edges = (
        energy_axis_from_filter(state)
        if energy_edges_ev is None
        else np.asarray(energy_edges_ev, dtype=float)
    )
    centres, step = _uniform_axis(edges)
    specimen_distribution = (
        distribution
        if distribution is not None
        else real_inelastic_distribution(state)
    )
    maximum_loss = max(float(edges[-1]), step)
    raw_components, truncation = _raw_compound_poisson_components(
        specimen_distribution,
        maximum_loss_ev=maximum_loss,
        energy_step_ev=step,
        tail_probability=tail,
    )
    generated_components: dict[str, np.ndarray] = {}
    axis_mapping_omitted_by_component: dict[str, float] = {}
    for key, value in raw_components.items():
        mapped, omitted = _map_raw_to_axis(value, step, edges)
        generated_components[key] = mapped
        axis_mapping_omitted_by_component[key] = omitted
    generated = sum(
        generated_components.values(), np.zeros_like(centres)
    )
    source_fwhm, source_provenance = source_energy_fwhm_ev(
        state, simulation
    )
    source_components = {
        key: _broaden(value, source_fwhm, step)
        for key, value in generated_components.items()
    }
    source_broadened = sum(
        source_components.values(), np.zeros_like(centres)
    )
    transfer = spectrometer or first_order_energy_filter_transmission(
        state, centres
    )
    if not np.array_equal(transfer.energy_loss_ev, centres):
        raise ValueError("Spectrometer response must use the EELS energy axis")
    filtered_components = {
        key: value * transfer.transmission
        for key, value in source_components.items()
    }
    spectrometer_probability = sum(
        filtered_components.values(), np.zeros_like(centres)
    )
    response = detector_response or EELSDetectorResponse()
    detector_broadened_components = {
        key: _broaden(value, response.energy_resolution_fwhm_ev, step)
        for key, value in filtered_components.items()
    }
    detected_probability_components = {
        key: response.efficiency * value
        for key, value in detector_broadened_components.items()
    }
    detected_components = {
        key: incident * value
        for key, value in detected_probability_components.items()
    }
    expected_counts = sum(
        detected_components.values(), np.zeros_like(centres)
    )
    sampled = None
    if response.poisson_enabled:
        sampled = np.random.default_rng(response.poisson_seed).poisson(
            np.maximum(expected_counts, 0.0)
        )
    generated_sum = float(np.sum(generated))
    tracked_expected = max(
        1.0 - float(specimen_distribution.absorbed_probability), 0.0
    )
    axis_mapping_omitted = float(
        sum(axis_mapping_omitted_by_component.values())
    )
    energy_axis_omitted = max(tracked_expected - generated_sum, 0.0)
    source_edge_losses = {
        key: max(
            float(np.sum(generated_components[key])) - float(np.sum(value)),
            0.0,
        )
        for key, value in source_components.items()
    }
    detector_edge_losses = {
        key: max(
            float(np.sum(filtered_components[key])) - float(np.sum(value)),
            0.0,
        )
        for key, value in detector_broadened_components.items()
    }
    return EELSForwardResult(
        energy_loss_ev=centres,
        generated_probability=generated,
        source_broadened_probability=source_broadened,
        spectrometer_probability=spectrometer_probability,
        detected_expected_counts=expected_counts,
        detected_sampled_counts=sampled,
        generated_components=generated_components,
        detected_components=detected_components,
        spectrometer_transmission=transfer,
        absorbed_probability=float(specimen_distribution.absorbed_probability),
        metrics={
            "model": "compound-Poisson representative-loss EELS forward chain",
            "material_key": specimen_distribution.material_key,
            "specimen_distribution_source": (
                "shared_specimen_interaction_result"
                if distribution is not None
                else "direct_compatibility_resolution"
            ),
            "energy_axis_unit": "eV energy loss",
            "probability_representation": "probability per detector bin",
            "component_order": tuple(generated_components),
            "source_energy_fwhm_ev": source_fwhm,
            "source_energy_provenance": source_provenance,
            "spectrometer_model": transfer.model,
            "spectrometer_boundaries": transfer.boundaries,
            "detector_response_provenance": response.provenance,
            "detector_efficiency": response.efficiency,
            "detector_resolution_fwhm_ev": (
                response.energy_resolution_fwhm_ev
            ),
            "incident_electrons": incident,
            "generated_probability_on_axis": generated_sum,
            "tracked_probability_expected": tracked_expected,
            "raw_axis_omitted_probability": truncation[
                "energy_axis_omitted_probability"
            ],
            "axis_mapping_omitted_probability": axis_mapping_omitted,
            "axis_mapping_omitted_probability_by_component": dict(
                axis_mapping_omitted_by_component
            ),
            "energy_axis_omitted_probability": energy_axis_omitted,
            "energy_axis_probability_accounting_residual": (
                tracked_expected - generated_sum - energy_axis_omitted
            ),
            "source_broadening_edge_loss_probability": float(
                sum(source_edge_losses.values())
            ),
            "source_broadening_edge_loss_probability_by_component": (
                source_edge_losses
            ),
            "detector_broadening_edge_loss_probability": float(
                sum(detector_edge_losses.values())
            ),
            "detector_broadening_edge_loss_probability_by_component": (
                detector_edge_losses
            ),
            "detector_broadening_edge_loss_expected_counts": (
                incident
                * response.efficiency
                * float(sum(detector_edge_losses.values()))
            ),
            "eftem_transmitted_probability": float(
                np.sum(spectrometer_probability)
            ),
            "expected_detected_counts": float(np.sum(expected_counts)),
            "absorbed_probability": specimen_distribution.absorbed_probability,
            "poisson_noise_enabled": response.poisson_enabled,
            "poisson_seed": (
                response.poisson_seed if response.poisson_enabled else None
            ),
            "plural_scattering_model": (
                "independent channel compound Poisson convolution"
            ),
            "fine_structure_model": (
                "not included; core loss is the configured representative energy"
            ),
            "poisson_tail_upper_bound": truncation[
                "poisson_tail_upper_bound"
            ],
            "tracked_probability_on_raw_axis": truncation[
                "tracked_probability_on_raw_axis"
            ],
        },
    )


def apply_eftem_window(
    spectral_image,
    spectrometer: SpectrometerTransmission,
) -> np.ndarray:
    """Integrate an arbitrary per-pixel energy-loss cube through the slit.

    ``spectral_image[..., energy]`` must already be expressed as intensity per
    detector bin.  This operation introduces no separability assumption.
    """

    values = np.asarray(spectral_image, dtype=float)
    if values.ndim < 1 or values.shape[-1] != spectrometer.transmission.size:
        raise ValueError("EFTEM spectral image must end in the filter energy axis")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("EFTEM spectral intensity must be finite and non-negative")
    return np.sum(values * spectrometer.transmission, axis=-1)


def eftem_spectral_cube_from_image(
    zero_loss_image,
    forward_result: EELSForwardResult,
) -> np.ndarray:
    """Apply the shared specimen loss distribution to an image intensity.

    This is an explicitly separable first-order EFTEM boundary: it preserves
    the calculated spatial image while distributing each pixel over the same
    specimen loss spectrum.  Energy-dependent delocalisation or a full
    multislice mixed dynamic form factor requires a later model and is not
    implied here.
    """

    image = np.asarray(zero_loss_image, dtype=float)
    if image.ndim != 2 or not np.all(np.isfinite(image)) or np.any(image < 0.0):
        raise ValueError("EFTEM source image must be finite, non-negative 2-D data")
    return image[..., None] * forward_result.source_broadened_probability


def eftem_image_from_shared_spectrum(
    zero_loss_image,
    forward_result: EELSForwardResult,
) -> np.ndarray:
    """Integrate the separable EFTEM cube without allocating the full cube."""

    image = np.asarray(zero_loss_image, dtype=float)
    if image.ndim != 2 or not np.all(np.isfinite(image)) or np.any(image < 0.0):
        raise ValueError("EFTEM source image must be finite, non-negative 2-D data")
    transmitted = float(
        np.sum(
            forward_result.source_broadened_probability
            * forward_result.spectrometer_transmission.transmission
        )
    )
    result = np.ascontiguousarray(image * transmitted, dtype=float)
    result.setflags(write=False)
    return result
