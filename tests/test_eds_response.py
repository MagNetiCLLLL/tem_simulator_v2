"""Replay executed material/photons without repeating ionisation or geometry."""
from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.detector import eds_response as response
from temsim.detector import eds_signal as eds
from temsim.detector.eds_geometry import EDSDetectorArrayGeometry
from temsim.detector.eds_photon_transport import AnnularPlanarLayerOccluder
from temsim.optics.column import default_state


OPTIONS = dict(energy_min_ev=0., energy_max_ev=10000., energy_bin_width_ev=13.,
               energy_resolution_fwhm_ev=125., poisson_enabled=False, poisson_seed=17)


@pytest.fixture
def physical():
    state = default_state()
    geometry = EDSDetectorArrayGeometry(system_key="eds", segment_count=2,
        azimuth_centers_deg=(0., 180.), takeoff_angle_deg=32.,
        minimum_unshadowed_solid_angle_sr=1., analytical_holder_solid_angle_sr=1., windowless=True)
    material = eds.elemental_material(14, density_g_cm3=2.33)
    tracks = tuple(eds.ElectronTrackSegment("sample", material,
        path_length_nm=10.+index, electron_energy_ev=200000.+index,
        electron_weight=weight, source_ray_index=index)
        for index, weight in enumerate((.3, .2)))
    flights = tuple(SimpleNamespace(source_ray_index=index, source_key="sample",
        material_key=material.key, history="primary", start_nm=(0., 0., -5.),
        end_nm=(0., 0., 5.+index), electron_weight=track.electron_weight)
        for index, track in enumerate(tracks))
    transport = SimpleNamespace(material_flights=flights)
    def calculate(incident=1e7, *, blocked=False, **options):
        holders = (() if not blocked else (AnnularPlanarLayerOccluder(
            "fixture_holder", state.sample.z_mm-.6, state.sample.z_mm-.5, 2., hard_shadow=True),))
        return eds.simulate_eds_tracks(tracks, geometry, incident_electrons=incident,
            state=state, elastic_transport=transport, photon_quadrature_order=2,
            photon_maximum_stored_paths=3, photon_holder_occluders=holders,
            **(OPTIONS | options))
    return calculate, tracks, transport


def _replay(spectrum, incident=3.25e7, **options):
    return response.replay_eds_response(spectrum, incident_electrons=incident, **(OPTIONS | options))


def _close(left, right):
    np.testing.assert_allclose(left, right, rtol=2e-12, atol=1e-12)


def _expected_products_match(actual, cold):
    _close(actual.expected_counts, cold.expected_counts)
    for rows, references, scaled in (
        (actual.vacancies, cold.vacancies, {"expected_vacancies"}),
        (actual.lines, cold.lines, {"expected_vacancies", "expected_emitted_photons",
                                  "expected_detected_counts", "expected_counts_per_segment"}),
    ):
        assert len(rows) == len(references)
        for row, reference in zip(rows, references):
            for field in fields(row):
                left, right = getattr(row, field.name), getattr(reference, field.name)
                if field.name in scaled:
                    _close(left, right)
                else:
                    assert left == right
    a, b = actual.photon_transport, cold.photon_transport
    assert a is not None and b is not None
    _close(a.expected_detected_weight_per_segment, b.expected_detected_weight_per_segment)
    for name in ("expected_detected_weight_per_emission", "quadrature_weight_per_emission"):
        assert getattr(a, name).keys() == getattr(b, name).keys()
        for key, values in getattr(a, name).items():
            _close(values, getattr(b, name)[key])
    for path, reference in zip(a.paths, b.paths):
        _close(path.photon.statistical_weight, reference.photon.statistical_weight)
        _close(path.detected_weight, reference.detected_weight)
        assert path.photon.direction == reference.photon.direction
        assert path.photon.origin_mm == reference.photon.origin_mm
        assert path.material_intervals == reference.material_intervals
        assert path.detector_hit_mm == reference.detector_hit_mm
        assert path.terminal_status == reference.terminal_status
    for key in response._SPECTRUM_LINEAR_METRICS:
        if key in cold.metrics:
            _close(actual.metrics[key], cold.metrics[key])
    for key, value in b.metrics.items():
        if key in response._PHOTON_LINEAR_METRICS:
            _close(a.metrics[key], value)
        elif key == "blocked_input_weight_by_component":
            assert a.metrics[key].keys() == value.keys()
            for component, weight in value.items():
                _close(a.metrics[key][component], weight)
        else:
            assert a.metrics[key] == value


@pytest.mark.parametrize("blocked", [False, True])
def test_dose_replay_matches_full_material_and_photon_expectations(physical, blocked):
    calculate, tracks, transport = physical
    original = calculate(blocked=blocked)
    assert original.response_rates is not None
    assert original.photon_transport.metrics["photon_count"] > len(original.photon_transport.paths)
    actual, cold = _replay(original), calculate(3.25e7, blocked=blocked)
    _expected_products_match(actual, cold)
    assert actual.response_rates is original.response_rates
    assert actual.elastic_transport is transport
    for a, b in zip(actual.vacancies, original.vacancies):
        assert a.electron_weight == b.electron_weight
        assert a.expected_occurrences_per_incident_electron == b.expected_occurrences_per_incident_electron
    for a, b in zip(actual.photon_transport.paths, original.photon_transport.paths):
        assert a.photon.direction is b.photon.direction
        assert a.material_intervals is b.material_intervals
    assert actual.metrics["shell_cross_section_evaluation_count"] == 0
    assert actual.metrics["eds_response_origin_shell_cross_section_evaluation_count"] == original.metrics["shell_cross_section_evaluation_count"]
    assert actual.metrics["eds_current_photon_geometry_trace_count"] == 0
    if blocked:
        assert original.photon_transport.metrics["blocked_input_weight_by_component"]


def test_zero_then_positive_retains_the_executed_response(physical):
    calculate, _tracks, _transport = physical
    original = calculate()
    zero = _replay(original, 0.)
    assert zero.response_rates is original.response_rates
    assert np.all(zero.expected_counts == 0.)
    assert all(row.expected_vacancies == 0. for row in zero.vacancies)
    assert all(row.expected_emitted_photons == row.expected_detected_counts == 0. for row in zero.lines)
    assert all(path.photon.statistical_weight == path.detected_weight == 0. for path in zero.photon_transport.paths)
    restored = _replay(zero)
    _expected_products_match(restored, calculate(3.25e7))
    assert restored.metrics["eds_response_origin_shell_cross_section_evaluation_count"] > 0
    initial_zero = calculate(0.)
    assert initial_zero.response_rates is None
    assert _replay(initial_zero) is None


def test_point_replay_preserves_source_to_arrival_dose_through_zero(physical):
    """Supplied material tracks test point provenance, not full gun transport."""
    _calculate, tracks, transport = physical
    state = default_state()
    state.sample.eds_overlap_sampling_enabled = False
    geometry = EDSDetectorArrayGeometry(system_key="eds", segment_count=2,
        azimuth_centers_deg=(0., 180.), takeoff_angle_deg=32.,
        minimum_unshadowed_solid_angle_sr=1., analytical_holder_solid_angle_sr=1., windowless=True)
    transport.eds_tracks = tracks
    transport.metrics = {}
    bundle = SimpleNamespace(rays=(), emitted_ray_count=4, reaching_ray_count=2,
        surviving_fraction=.25, original_centroid_nm=(0., 0.), target_centroid_nm=(0., 0.),
        chief_angle_mrad=(0., 0.), energy_range_ev=(200000., 200001.))
    original = eds.simulate_eds_point(state, geometry, incident_electrons=4e7,
        incident_bundle=bundle, elastic_transport=transport, photon_quadrature_order=2)
    assert original.metrics["incident_electrons"] == 1e7
    assert original.metrics["source_electrons_before_column_losses"] == 4e7
    zero = _replay(original, 0.)
    assert zero.metrics["source_electrons_before_column_losses"] == 0.
    assert zero.metrics["electrons_reaching_sample_plane"] == 0.
    restored = _replay(zero, 2e7)
    assert restored.metrics["incident_electrons"] == 2e7
    assert restored.metrics["source_electrons_before_column_losses"] == 8e7
    assert restored.metrics["electrons_reaching_sample_plane"] == 2e7


def test_noise_only_keeps_exact_expected_arrays_and_never_repeats_physics(physical, monkeypatch):
    original = physical[0]()
    def forbidden(*_a, **_k):
        pytest.fail("Noise-only replay must not reform a spectrum or run material/photon physics")
    monkeypatch.setattr(response, "form_eds_spectrum", forbidden)
    monkeypatch.setattr(eds, "bote_ionisation_cross_section_cm2", forbidden)
    actual = _replay(original, 1e7, poisson_enabled=True, poisson_seed=43)
    assert actual.expected_counts is original.expected_counts
    assert actual.energy_bin_centres_ev is original.energy_bin_centres_ev
    assert actual.lines is original.lines and actual.vacancies is original.vacancies
    assert actual.photon_transport is original.photon_transport
    np.testing.assert_array_equal(actual.sampled_counts,
        np.random.default_rng(43).poisson(original.expected_counts))
    assert _replay(actual, 1e7, poisson_enabled=False).sampled_counts is None


def test_expanded_energy_range_restores_outside_lines_and_new_resolution(physical):
    calculate = physical[0]
    limited = calculate(energy_max_ev=500.)
    assert limited.metrics["counts_outside_spectrum"] > 0.
    actual = _replay(limited, energy_bin_width_ev=7., energy_resolution_fwhm_ev=0.)
    cold = calculate(3.25e7, energy_bin_width_ev=7., energy_resolution_fwhm_ev=0.)
    _expected_products_match(actual, cold)
    assert np.sum(actual.expected_counts) > np.sum(limited.expected_counts)*3.25
    assert actual.metrics["counts_outside_spectrum"] < limited.metrics["counts_outside_spectrum"]


def test_rates_are_numeric_immutable_and_do_not_store_record_templates(physical):
    rates = physical[0]().response_rates
    for field in fields(rates):
        value = getattr(rates, field.name)
        if isinstance(value, np.ndarray):
            assert value.dtype == np.float64
            assert not value.flags.writeable
            with pytest.raises(ValueError):
                value.setflags(write=True)
        elif isinstance(value, tuple):
            assert all(isinstance(row, (str, tuple)) for row in value)


@pytest.mark.parametrize("mismatch", ["line_order", "vacancy_id", "photon_id", "shape", "nan", "float32"])
def test_misaligned_or_malformed_rates_are_rejected(physical, mismatch):
    original = physical[0]()
    rates = replace(original.response_rates)
    if mismatch == "line_order":
        rates = replace(rates, line_ids=rates.line_ids[::-1])
    elif mismatch == "vacancy_id":
        rates = replace(rates, vacancy_ids=("wrong", *rates.vacancy_ids[1:]))
    elif mismatch == "photon_id":
        rates = replace(rates, photon_ids=("wrong", *rates.photon_ids[1:]))
    else:
        value = rates.line_expected.copy()
        if mismatch == "shape":
            value = value[:1]
        elif mismatch == "nan":
            value[0, 0] = np.nan
        else:
            value = value.astype(np.float32)
        object.__setattr__(rates, "line_expected", value)
    with pytest.raises(ValueError, match="EDS response"):
        _replay(replace(original, response_rates=rates))


def test_positive_empty_and_zero_efficiency_responses_are_valid(physical):
    calculate = physical[0]
    no_efficiency = calculate(detector_efficiency=0.)
    assert no_efficiency.response_rates is not None
    actual = _replay(no_efficiency)
    assert actual.lines and actual.vacancies
    assert np.all(actual.expected_counts == 0.)
    geometry = EDSDetectorArrayGeometry(system_key="eds", segment_count=1,
        azimuth_centers_deg=(0.,), takeoff_angle_deg=32., minimum_unshadowed_solid_angle_sr=1.,
        analytical_holder_solid_angle_sr=1., windowless=True)
    empty = eds.simulate_eds_tracks((), geometry, incident_electrons=10., **OPTIONS)
    assert empty.response_rates is not None
    changed = _replay(empty)
    assert changed.lines == changed.vacancies == ()
    assert np.all(changed.expected_counts == 0.)


def test_legacy_missing_response_falls_back_and_bad_dose_rejects(physical):
    original = physical[0]()
    assert _replay(replace(original, response_rates=None)) is None
    for incident in (-1., float("inf"), float("nan")):
        with pytest.raises(ValueError, match="Incident"):
            _replay(original, incident)


def test_progress_callback_can_cancel_replay(physical):
    original = physical[0]()
    class Cancelled(Exception):
        pass
    def cancel(*_args):
        raise Cancelled
    with pytest.raises(Cancelled):
        _replay(original, progress_callback=cancel)
