"""Dose-linear expectations retained from one executed EDS calculation.

Rates hold compact numerical coefficients, never another set of material,
electron or photon records. Geometry and source identities remain in the
owning spectrum. Poisson samples are a final readout and are never normalized.
"""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass, replace
import math
from time import perf_counter

import numpy as np


_SPECTRUM_LINEAR_METRICS = (
    "total_expected_counts", "total_expected_emitted_photons",
    "spectrum_expected_counts", "counts_outside_spectrum",
    "total_expected_vacancies", "total_expected_radiative_relaxations",
    "total_expected_auger_relaxations", "total_expected_unresolved_relaxations",
    "photon_transport_expected_unattenuated_counts",
    "photon_transport_expected_detected_counts",
    "source_electrons_before_column_losses", "electrons_reaching_sample_plane",
)
_PHOTON_LINEAR_METRICS = ("total_quadrature_weight", "total_detected_weight")


def _immutable_array(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"EDS response {name} must have shape {shape}")
    if np.any(~np.isfinite(array)) or np.any(array < 0.):
        raise ValueError(f"EDS response {name} must be finite and non-negative")
    # A bytes owner prevents setflags(write=True), including after archive load.
    return np.frombuffer(array.tobytes(order="C"), dtype=np.float64).reshape(shape)


@dataclass(frozen=True, slots=True)
class EDSResponseRates:
    """Expected coefficients per electron incident on the sampled material.

    Line columns are vacancies, emitted photons and detected counts; retained
    path columns are input and detected photon weights. All coefficients are
    float64 arrays, indexed by the corresponding ordered identity tuples.
    """

    reference_incident_electrons: float
    segment_count: int
    vacancy_ids: tuple[str, ...]
    line_ids: tuple[tuple[str, str], ...]
    photon_ids: tuple[str, ...]
    emission_keys: tuple[str, ...]
    quadrature_emission_keys: tuple[str, ...]
    spectrum_metric_keys: tuple[str, ...]
    photon_metric_keys: tuple[str, ...]
    blocked_component_keys: tuple[str, ...]
    vacancy_expected: np.ndarray
    line_expected: np.ndarray
    line_per_segment: np.ndarray
    photon_path_weights: np.ndarray
    photon_per_segment: np.ndarray
    photon_per_emission: np.ndarray
    quadrature_per_emission: np.ndarray
    spectrum_metrics: np.ndarray
    photon_metrics: np.ndarray
    blocked_input_weights: np.ndarray
    has_photon_transport: bool = False
    schema: str = "executed-eds-response-rates-v1"

    def __post_init__(self):
        incident = float(self.reference_incident_electrons)
        if not math.isfinite(incident) or incident <= 0.:
            raise ValueError("EDS response reference dose must be finite and positive")
        if isinstance(self.segment_count, bool) or int(self.segment_count) != self.segment_count or self.segment_count < 1:
            raise ValueError("EDS response detector segment count must be positive")
        if self.schema != "executed-eds-response-rates-v1":
            raise ValueError("Unsupported EDS response schema")
        object.__setattr__(self, "reference_incident_electrons", incident)
        object.__setattr__(self, "segment_count", int(self.segment_count))
        for name in ("vacancy_ids", "photon_ids", "emission_keys", "quadrature_emission_keys",
                     "spectrum_metric_keys", "photon_metric_keys", "blocked_component_keys"):
            values = tuple(getattr(self, name))
            if any(not isinstance(key, str) or not key for key in values) or len(set(values)) != len(values):
                raise ValueError(f"EDS response {name} must contain unique nonempty keys")
            object.__setattr__(self, name, values)
        line_ids = tuple(tuple(row) for row in self.line_ids)
        if (any(len(row) != 2 or any(not isinstance(key, str) or not key for key in row) for row in line_ids)
                or len(set(line_ids)) != len(line_ids)):
            raise ValueError("EDS response line identities are invalid")
        object.__setattr__(self, "line_ids", line_ids)
        if not set(self.spectrum_metric_keys).issubset(_SPECTRUM_LINEAR_METRICS):
            raise ValueError("EDS response contains a non-linear spectrum metric")
        if not set(self.photon_metric_keys).issubset(_PHOTON_LINEAR_METRICS):
            raise ValueError("EDS response contains a non-linear photon metric")
        n, s = len(line_ids), self.segment_count
        shapes = {
            "vacancy_expected": (len(self.vacancy_ids),), "line_expected": (n, 3),
            "line_per_segment": (n, s), "photon_path_weights": (len(self.photon_ids), 2),
            "photon_per_segment": (s,) if self.has_photon_transport else (0,),
            "photon_per_emission": (len(self.emission_keys), s),
            "quadrature_per_emission": (len(self.quadrature_emission_keys),),
            "spectrum_metrics": (len(self.spectrum_metric_keys),),
            "photon_metrics": (len(self.photon_metric_keys),),
            "blocked_input_weights": (len(self.blocked_component_keys),),
        }
        for name, shape in shapes.items():
            object.__setattr__(self, name, _immutable_array(getattr(self, name), shape, name))


def capture_eds_response(spectrum):
    """Normalize authoritative positive-dose expectations, never samples.

    An executed positive-dose zero response is valid. Zero-dose and historical
    results without the physical expectation provenance are not a response.
    """
    metrics = spectrum.metrics
    incident = float(metrics.get("incident_electrons", 0.))
    if (not math.isfinite(incident) or incident <= 0.
            or metrics.get("signal_kind") != "characteristic_x_ray"
            or "ionisation_model" not in metrics):
        return None
    s = int(metrics["detector_segment_count"])
    vacancies, lines = spectrum.vacancies, spectrum.lines
    photons = spectrum.photon_transport
    paths = () if photons is None else photons.paths
    emission = {} if photons is None else photons.expected_detected_weight_per_emission
    quadrature = {} if photons is None else photons.quadrature_weight_per_emission
    photon_metrics = {} if photons is None else photons.metrics
    blocked = photon_metrics.get("blocked_input_weight_by_component", {})
    metric_keys = tuple(key for key in _SPECTRUM_LINEAR_METRICS if key in metrics)
    photon_keys = tuple(key for key in _PHOTON_LINEAR_METRICS if key in photon_metrics)
    def rate(values, shape):
        return np.asarray(values, dtype=np.float64).reshape(shape)/incident
    return EDSResponseRates(
        reference_incident_electrons=incident, segment_count=s,
        vacancy_ids=tuple(row.vacancy_id for row in vacancies),
        line_ids=tuple((row.vacancy_id, row.transition) for row in lines),
        photon_ids=tuple(row.photon.photon_id for row in paths),
        emission_keys=tuple(emission), quadrature_emission_keys=tuple(quadrature),
        spectrum_metric_keys=metric_keys, photon_metric_keys=photon_keys,
        blocked_component_keys=tuple(blocked),
        vacancy_expected=rate([row.expected_vacancies for row in vacancies], (len(vacancies),)),
        line_expected=rate([(row.expected_vacancies, row.expected_emitted_photons,
                             row.expected_detected_counts) for row in lines], (len(lines), 3)),
        line_per_segment=rate([row.expected_counts_per_segment for row in lines], (len(lines), s)),
        photon_path_weights=rate([(row.photon.statistical_weight, row.detected_weight)
                                  for row in paths], (len(paths), 2)),
        photon_per_segment=rate(() if photons is None else photons.expected_detected_weight_per_segment,
                                (0,) if photons is None else (s,)),
        photon_per_emission=rate(list(emission.values()), (len(emission), s)),
        quadrature_per_emission=rate(list(quadrature.values()), (len(quadrature),)),
        spectrum_metrics=rate([metrics[key] for key in metric_keys], (len(metric_keys),)),
        photon_metrics=rate([photon_metrics[key] for key in photon_keys], (len(photon_keys),)),
        blocked_input_weights=rate(list(blocked.values()), (len(blocked),)),
        has_photon_transport=photons is not None,
    )


def _axis(energy_min_ev, energy_max_ev, energy_bin_width_ev, energy_resolution_fwhm_ev, poisson_seed):
    values = tuple(float(value) for value in (
        energy_min_ev, energy_max_ev, energy_bin_width_ev, energy_resolution_fwhm_ev))
    if (not all(math.isfinite(value) for value in values) or values[0] < 0.
            or values[1] <= values[0] or values[2] <= 0. or values[3] < 0.):
        raise ValueError("EDS spectrum axis/resolution is invalid")
    if int(poisson_seed) < 0:
        raise ValueError("EDS Poisson seed cannot be negative")
    edges = np.arange(values[0], values[1]+values[2], values[2], dtype=float)
    if edges[-1] < values[1]:
        edges = np.r_[edges, values[1]]
    return values, edges, .5*(edges[:-1]+edges[1:])


def _sample(expected, enabled, seed):
    if not enabled:
        return None
    sampled = np.random.default_rng(int(seed)).poisson(np.maximum(expected, 0.))
    sampled.setflags(write=False)
    return sampled


def form_eds_spectrum(lines, *, energy_min_ev=0., energy_max_ev,
                      energy_bin_width_ev, energy_resolution_fwhm_ev,
                      poisson_enabled, poisson_seed, progress_callback=None):
    """Apply the existing axis, Gaussian response and final Poisson readout.

    Accumulation follows the supplied line order. No line is re-sorted and no
    physical line strength, efficiency or attenuation is recalculated.
    """
    from .eds_signal import _line_response_kernel, _report_phase_progress
    values, edges, centres = _axis(energy_min_ev, energy_max_ev, energy_bin_width_ev,
                                  energy_resolution_fwhm_ev, poisson_seed)
    lines = tuple(lines)
    expected, outside = np.zeros_like(centres), 0.
    sigma = values[3]/(2.*math.sqrt(2.*math.log(2.)))
    kernels = {}
    for index, line in enumerate(lines):
        if index % max(1, len(lines)//100) == 0:
            _report_phase_progress(progress_callback, .9, 1., index, len(lines),
                                   f"EDS spectrum lines {index}/{len(lines)}")
        if not values[0] <= line.energy_ev < values[1]:
            outside += line.expected_detected_counts
            continue
        if sigma <= 0.:
            target = min(max(int(np.searchsorted(edges, line.energy_ev)-1), 0), expected.size-1)
            expected[target] += line.expected_detected_counts
            continue
        if line.energy_ev not in kernels:
            kernels[line.energy_ev] = _line_response_kernel(centres, line.energy_ev, sigma)
        lower, upper, weights, total = kernels[line.energy_ev]
        if total > 0.:
            expected[lower:upper] += line.expected_detected_counts*weights/total
    sampled = _sample(expected, poisson_enabled, poisson_seed)
    centres.setflags(write=False)
    expected.setflags(write=False)
    return centres, expected, sampled, outside


def _validate_alignment(spectrum, rates):
    if type(rates) is not EDSResponseRates:
        raise ValueError("EDS response rates have an unsupported type")
    if rates.schema != "executed-eds-response-rates-v1":
        raise ValueError("Unsupported EDS response schema")
    n, s = len(rates.line_ids), rates.segment_count
    shapes = {
        "vacancy_expected": (len(rates.vacancy_ids),), "line_expected": (n, 3),
        "line_per_segment": (n, s), "photon_path_weights": (len(rates.photon_ids), 2),
        "photon_per_segment": (s,) if rates.has_photon_transport else (0,),
        "photon_per_emission": (len(rates.emission_keys), s),
        "quadrature_per_emission": (len(rates.quadrature_emission_keys),),
        "spectrum_metrics": (len(rates.spectrum_metric_keys),),
        "photon_metrics": (len(rates.photon_metric_keys),),
        "blocked_input_weights": (len(rates.blocked_component_keys),),
    }
    for name, shape in shapes.items():
        array = getattr(rates, name)
        if (not isinstance(array, np.ndarray) or array.dtype != np.float64 or array.shape != shape
                or np.any(~np.isfinite(array)) or np.any(array < 0.)):
            raise ValueError(f"EDS response {name} has invalid shape, dtype or coefficients")
    expected = (tuple(row.vacancy_id for row in spectrum.vacancies),
                tuple((row.vacancy_id, row.transition) for row in spectrum.lines))
    if expected != (rates.vacancy_ids, rates.line_ids):
        raise ValueError("EDS response rates do not match the vacancy/line identities")
    if int(spectrum.metrics.get("detector_segment_count", -1)) != rates.segment_count:
        raise ValueError("EDS response rates do not match the detector segments")
    if any(len(row.expected_counts_per_segment) != rates.segment_count for row in spectrum.lines):
        raise ValueError("EDS line response segment shape is invalid")
    photons = spectrum.photon_transport
    if (photons is not None) != rates.has_photon_transport:
        raise ValueError("EDS response photon templates are missing or unexpected")
    if photons is not None:
        if (tuple(path.photon.photon_id for path in photons.paths) != rates.photon_ids
                # JSON object order is not part of physical identity. The
                # explicit rate key tuples retain each coefficient's row;
                # decoded mappings can have a different insertion order.
                or set(photons.expected_detected_weight_per_emission) != set(rates.emission_keys)
                or set(photons.quadrature_weight_per_emission) != set(rates.quadrature_emission_keys)
                or set(photons.metrics.get("blocked_input_weight_by_component", {})) != set(rates.blocked_component_keys)
                or len(photons.expected_detected_weight_per_segment) != rates.segment_count):
            raise ValueError("EDS response rates do not match the photon identities/segments")


def _scaled_photons(template, rates, incident):
    if template is None:
        return None
    path_weights = rates.photon_path_weights*incident
    paths = []
    for index, path in enumerate(template.paths):
        # dataclasses.replace(ray) re-normalizes the already unit direction.
        # A shallow immutable-record copy preserves its exact geometry bits.
        photon = copy(path.photon)
        object.__setattr__(photon, "statistical_weight", float(path_weights[index, 0]))
        paths.append(replace(path, photon=photon, detected_weight=float(path_weights[index, 1])))
    metrics = dict(template.metrics)
    metrics.update(zip(rates.photon_metric_keys, (rates.photon_metrics*incident).tolist()))
    metrics["blocked_input_weight_by_component"] = dict(zip(
        rates.blocked_component_keys, (rates.blocked_input_weights*incident).tolist()))
    return replace(template, paths=tuple(paths), metrics=metrics,
        expected_detected_weight_per_segment=tuple(rates.photon_per_segment*incident),
        expected_detected_weight_per_emission={key: tuple(row*incident)
            for key, row in zip(rates.emission_keys, rates.photon_per_emission)},
        quadrature_weight_per_emission=dict(zip(rates.quadrature_emission_keys,
                                                (rates.quadrature_per_emission*incident).tolist())))


def replay_eds_response(spectrum, *, incident_electrons, energy_min_ev=0.,
                        energy_max_ev, energy_bin_width_ev, energy_resolution_fwhm_ev,
                        poisson_enabled, poisson_seed, progress_callback=None):
    """Reform an executed response; None requests full physics for legacy data."""
    started = perf_counter()
    incident = float(incident_electrons)
    if not math.isfinite(incident) or incident < 0.:
        raise ValueError("Incident electron count must be non-negative")
    values, _edges, requested_centres = _axis(energy_min_ev, energy_max_ev,
        energy_bin_width_ev, energy_resolution_fwhm_ev, poisson_seed)
    rates = getattr(spectrum, "response_rates", None)
    if rates is None:
        return None
    _validate_alignment(spectrum, rates)
    same_dose = incident == spectrum.metrics.get("incident_electrons")
    same_response = (same_dose and all(spectrum.metrics.get(key) == value for key, value in zip(
        ("energy_min_ev", "energy_max_ev", "energy_bin_width_ev", "energy_resolution_fwhm_ev"), values))
        and np.array_equal(requested_centres, spectrum.energy_bin_centres_ev))
    if same_dose:
        vacancies, lines, photons = spectrum.vacancies, spectrum.lines, spectrum.photon_transport
    else:
        vacancies = tuple(replace(row, expected_vacancies=float(value*incident))
                          for row, value in zip(spectrum.vacancies, rates.vacancy_expected))
        lines = tuple(replace(row, expected_vacancies=float(values_[0]*incident),
            expected_emitted_photons=float(values_[1]*incident),
            expected_detected_counts=float(values_[2]*incident),
            expected_counts_per_segment=tuple(segment*incident))
            for row, values_, segment in zip(spectrum.lines, rates.line_expected, rates.line_per_segment))
        photons = _scaled_photons(spectrum.photon_transport, rates, incident)
    if same_response:
        centres, expected = spectrum.energy_bin_centres_ev, spectrum.expected_counts
        sampled = _sample(expected, poisson_enabled, poisson_seed)
        outside = float(spectrum.metrics.get("counts_outside_spectrum", 0.))
    else:
        centres, expected, sampled, outside = form_eds_spectrum(lines,
            energy_min_ev=values[0], energy_max_ev=values[1], energy_bin_width_ev=values[2],
            energy_resolution_fwhm_ev=values[3], poisson_enabled=poisson_enabled,
            poisson_seed=poisson_seed, progress_callback=progress_callback)
    metrics = dict(spectrum.metrics)
    if not same_dose:
        metrics.update(zip(rates.spectrum_metric_keys, (rates.spectrum_metrics*incident).tolist()))
    metrics.update(incident_electrons=incident,
        energy_min_ev=values[0], energy_max_ev=values[1], energy_bin_width_ev=values[2],
        energy_resolution_fwhm_ev=values[3], poisson_noise_enabled=bool(poisson_enabled),
        poisson_seed=int(poisson_seed) if poisson_enabled else None,
        total_expected_counts=float(sum(row.expected_detected_counts for row in lines)),
        total_expected_emitted_photons=float(sum(row.expected_emitted_photons for row in lines)),
        spectrum_expected_counts=float(np.sum(expected)), counts_outside_spectrum=outside,
        total_expected_vacancies=float(sum(row.expected_vacancies for row in vacancies)),
        total_expected_radiative_relaxations=float(sum(row.expected_vacancies*row.fluorescence_yield for row in vacancies)),
        total_expected_auger_relaxations=float(sum(row.expected_vacancies*row.auger_yield for row in vacancies)),
        total_expected_unresolved_relaxations=float(sum(row.expected_vacancies*row.unresolved_relaxation_yield for row in vacancies)),
        eds_response_replayed=True, eds_response_reused=True,
        eds_response_replay_kind="noise_only" if same_response else "dose_or_spectrum",
        eds_readout_expected_reused=same_response,
        eds_response_origin_shell_cross_section_evaluation_count=int(spectrum.metrics.get(
            "eds_response_origin_shell_cross_section_evaluation_count",
            spectrum.metrics.get("shell_cross_section_evaluation_count", 0))),
        shell_cross_section_evaluation_count=0,
        eds_current_photon_geometry_trace_count=0,
        eds_photon_record_scope="retained executed quadrature; no new photon transport",
        eds_response_replay_wall_time_s=perf_counter()-started)
    if photons is not None:
        metrics["photon_transport_expected_unattenuated_counts"] = float(sum(photons.quadrature_weight_per_emission.values()))
        metrics["photon_transport_expected_detected_counts"] = float(photons.metrics["total_detected_weight"])
    from .eds_signal import _report_phase_progress
    _report_phase_progress(progress_callback, .9, 1., 1, 1, "EDS spectrum response complete")
    return replace(spectrum, energy_bin_centres_ev=centres, expected_counts=expected,
                   sampled_counts=sampled, vacancies=vacancies, lines=lines,
                   photon_transport=photons, metrics=metrics, response_rates=rates)
