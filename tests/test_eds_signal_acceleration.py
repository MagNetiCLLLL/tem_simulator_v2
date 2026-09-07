"""Exact-value reuse and honest substage progress; no reduced physics sampling."""

from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.detector import eds_signal as eds
from temsim.detector.eds_geometry import EDSDetectorArrayGeometry
from temsim.optics.column import default_state


@pytest.fixture
def geometry():
    return EDSDetectorArrayGeometry(
        system_key="eds", segment_count=1, azimuth_centers_deg=(0.0,),
        takeoff_angle_deg=32.0, minimum_unshadowed_solid_angle_sr=1.0,
        analytical_holder_solid_angle_sr=1.0, windowless=True,
    )


def _tracks(count=4):
    return tuple(
        eds.ElectronTrackSegment(
            "sample", eds.elemental_material(14, density_g_cm3=2.33),
            path_length_nm=10.0 + i, electron_energy_ev=200_000.0 + i * 0.123,
            electron_weight=1.0 / count, source_ray_index=i,
        ) for i in range(count)
    )


def _simulation():
    return SimpleNamespace(incident=SimpleNamespace(
        alive=np.ones(2, dtype=bool),
        x=np.zeros((1, 2)), y=np.zeros((1, 2)),
        tx=np.array([[0.0, 1.0e-3]]), ty=np.zeros((1, 2)),
        energy_offset_ev=np.array([-0.1, 0.1]), ray_weight=np.array([0.5, 0.5]),
    ))


def test_attenuation_cache_uses_full_material_and_exact_energy():
    material = _tracks(1)[0].material
    cached = eds.EDSMaterial.mass_attenuation_cm2_g
    cached.cache_clear()
    for variant in (material, replace(material, mass_fractions=((29, 1.0),))):
        for energy in (1740.0, np.nextafter(1740.0, math.inf)):
            expected = cached.__wrapped__(variant, energy)
            assert cached(variant, energy) == expected
            assert cached(variant, energy) == expected
    assert cached.cache_info().misses == 4
    assert cached.cache_info().hits == 4
    assert cached.cache_info().maxsize == 4096
    with pytest.raises(ValueError):
        cached(material, -1.0)


def test_list_composition_is_immutable_and_cacheable():
    fractions = [[14, 1.0]]
    material = eds.EDSMaterial("Si", "Silicon", 2.33, fractions, "test")
    fractions[0][0] = 29
    assert material.mass_fractions == ((14, 1.0),)
    assert material.mass_attenuation_cm2_g(1740.0) == (
        eds.EDSMaterial.mass_attenuation_cm2_g.__wrapped__(material, 1740.0)
    )


def test_atomic_cache_and_gaussian_reuse_preserve_spectrum_and_poisson(
    geometry, monkeypatch,
):
    tracks = _tracks()
    options = dict(
        incident_electrons=1.0e7, energy_resolution_fwhm_ev=125.0,
        poisson_enabled=True, poisson_seed=21,
    )
    kernel = eds._line_response_kernel
    calls = []

    def counted_kernel(centres, energy, sigma):
        calls.append(energy)
        return kernel(centres, energy, sigma)

    monkeypatch.setattr(eds, "_line_response_kernel", counted_kernel)
    result = eds.simulate_eds_tracks(tracks, geometry, **options)
    assert len(calls) == len(set(line.energy_ev for line in result.lines))
    assert len(calls) < len(result.lines)
    monkeypatch.setattr(eds, "_relaxation_yields", eds._relaxation_yields.__wrapped__)
    monkeypatch.setattr(
        eds.EDSMaterial, "mass_attenuation_cm2_g",
        eds.EDSMaterial.mass_attenuation_cm2_g.__wrapped__,
    )
    reference = eds.simulate_eds_tracks(tracks, geometry, **options)
    assert result.lines == reference.lines
    assert result.vacancies == reference.vacancies
    assert result.metrics == reference.metrics
    np.testing.assert_array_equal(result.expected_counts, reference.expected_counts)
    np.testing.assert_array_equal(result.sampled_counts, reference.sampled_counts)

    # Independently accumulate uncached kernels in the original generation
    # order, rather than the energy-sorted public line order.
    vacancies = {v.vacancy_id: v for v in result.vacancies}
    ordered = sorted(result.lines, key=lambda line: (
        vacancies[line.vacancy_id].track_index,
        eds.SUBSHELL_NAMES.index(line.subshell),
        next(i for i, row in enumerate(eds._radiative_lines(
            line.atomic_number, eds.SUBSHELL_NAMES.index(line.subshell),
        )) if row[0] == line.transition),
    ))
    expected = np.zeros_like(result.expected_counts)
    centres = result.energy_bin_centres_ev
    sigma = options["energy_resolution_fwhm_ev"] / (2 * math.sqrt(2 * math.log(2)))
    for line in ordered:
        lower = max(int(np.searchsorted(centres, line.energy_ev - 5 * sigma)), 0)
        upper = min(int(np.searchsorted(centres, line.energy_ev + 5 * sigma)) + 1, centres.size)
        weights = np.exp(-0.5 * ((centres[lower:upper] - line.energy_ev) / sigma) ** 2)
        total = float(np.sum(weights))
        if total > 0:
            expected[lower:upper] += line.expected_detected_counts * weights / total
    np.testing.assert_array_equal(result.expected_counts, expected)


def test_eds_track_progress_covers_photons_and_spectrum(geometry):
    events = []
    result = eds.simulate_eds_tracks(
        _tracks(), geometry, incident_electrons=1.0e6, state=default_state(),
        progress_callback=lambda *args: events.append(args),
    )
    assert result.photon_transport is not None
    fractions = [done / total for done, total, _ in events]
    assert fractions == sorted(fractions)
    assert fractions[0] == 0.0 and fractions[-1] == 1.0
    assert any("EDS ionisation tracks 4/4" in label for _, _, label in events)
    photon_events = [event for event in events if "EDS photon emissions" in event[2]]
    assert photon_events[-1][0] == 9000
    assert photon_events[-1][2].endswith(f"{len(result.lines)}/{len(result.lines)}")
    assert any("EDS spectrum lines" in label for _, _, label in events)


def test_progress_exception_cancels_without_publishing_result(geometry):
    class Cancelled(Exception):
        pass

    def cancel(done, total, label):
        if "EDS photon emissions" in label:
            raise Cancelled

    with pytest.raises(Cancelled):
        eds.simulate_eds_tracks(
            _tracks(), geometry, incident_electrons=1.0e6,
            state=default_state(), progress_callback=cancel,
        )


def test_empty_tracks_finish_without_photon_work(geometry):
    events = []
    result = eds.simulate_eds_tracks(
        (), geometry, incident_electrons=1.0,
        progress_callback=lambda *args: events.append(args),
    )
    assert result.lines == ()
    assert events[-1][0:2] == (10000, 10000)
    assert not any("photon emissions" in label for _, _, label in events)


def test_point_progress_reserves_work_after_elastic_histories(geometry):
    state = default_state()
    state.sample.specimen_mode = "virtual"
    state.sample.specimen_preset_key = "si_110"
    state.sample.eds_transport_mode = "elastic_monte_carlo"
    simulation = _simulation()
    events = []
    result = eds.simulate_eds_point(
        state, geometry, simulation=simulation, incident_electrons=1.0e6,
        progress_callback=lambda *args: events.append(args),
    )
    fractions = [done / total for done, total, _ in events]
    assert fractions == sorted(fractions)
    histories = [event for event in events if "Elastic specimen history" in event[2]]
    assert histories[-1][:2] == (3500, 10000)
    assert histories[-1][2].endswith("2/2")
    assert events[-1][:2] == (9900, 10000)  # Final event ledger still follows.
    assert result.elastic_transport is not None
    assert result.photon_transport is not None

    cached_events = []
    repeated = eds.simulate_eds_point(
        state, geometry, simulation=simulation, incident_electrons=1.0e6,
        elastic_transport=result.elastic_transport,
        progress_callback=lambda *args: cached_events.append(args),
    )
    assert not any("Elastic specimen history" in label for _, _, label in cached_events)
    assert [d / t for d, t, _ in cached_events] == sorted(d / t for d, t, _ in cached_events)
    np.testing.assert_array_equal(result.expected_counts, repeated.expected_counts)


def test_elastic_only_progress_keeps_room_for_event_ledger():
    from temsim.specimen.interaction_engine import run_specimen_interactions
    from temsim.specimen.interaction_types import SpecimenInteractionRequest

    state = default_state()
    events = []
    result = run_specimen_interactions(
        state, _simulation(), SpecimenInteractionRequest.elastic_point(),
        progress_callback=lambda *args: events.append(args),
    )
    assert result.elastic_transport is not None
    assert result.eds_spectrum is None
    fractions = [done / total for done, total, _ in events]
    assert fractions == sorted(fractions)
    assert all(done < total for done, total, _ in events[:-1])
    assert events[-2] == (9900, 10000, "Recording specimen interaction events")
    assert events[-1] == (10000, 10000, "Specimen interactions complete")
