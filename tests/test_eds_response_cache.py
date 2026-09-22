"""Response reuse retains executed paths while rebuilding the current readout."""
from dataclasses import replace

import numpy as np
import pytest

from test_material_particle_sections import material_case
from test_particle_sections import fixture as optical_fixture
from test_particle_section_eds_reuse import eds_case
from temsim import simulation_pipeline as pipeline
from temsim.calculation_cache import calculation_signatures
from temsim.optics.column import default_state


@pytest.mark.parametrize("field,value", [
    ("eds_spectrum_max_energy_ev", 30_000.), ("eds_spectrum_bin_width_ev", 20.),
    ("eds_energy_resolution_fwhm_ev", 140.), ("eds_poisson_enabled", True),
    ("eds_poisson_seed", 321), ("column_current_limit_percent", 50.),
])
def test_readout_controls_preserve_response_but_change_complete_eds(field, value):
    state = default_state()
    before = calculation_signatures(state)
    setattr(state if field == "column_current_limit_percent" else state.sample, field, value)
    after = calculation_signatures(state)
    assert before["eds_response"] == after["eds_response"]
    assert before["elastic"] == after["elastic"]
    assert before["incident"] == after["incident"]
    assert before["eds"] != after["eds"]


@pytest.mark.parametrize("scanning", ["off", "ac", "descan"])
def test_frame_period_is_dose_only_when_both_rasters_are_off(scanning):
    state = default_state()
    state.ac_deflector.scan_enabled = scanning == "ac"
    state.descan_deflector.scan_enabled = scanning == "descan"
    before = calculation_signatures(state)
    state.ac_deflector.scan_frame_period_s *= 2.
    after = calculation_signatures(state)
    assert before["eds"] != after["eds"]
    for key in ("eds_response", "elastic", "incident", "column"):
        assert (before[key] == after[key]) is (scanning == "off"), key


@pytest.mark.parametrize("owner,field,value", [
    ("sample", "thickness_nm", 71.),
    ("sample", "specimen_orientation_quaternion_wxyz", (0.9996573249755573, 0.02617694830787315, 0., 0.)),
    ("sample", "eds_support_offset_x_um", 10.),
    ("sample", "eds_detector_efficiency", .9),
    ("sample", "eds_elastic_seed", 17),
    ("sample", "eds_overlap_sampling_points", 512),
    ("ac_deflector", "scan_pixels_x", 64),
    ("ac_deflector", "scan_lines", 64),
    ("emitter", "angular_rms_mrad", 2.),
    ("emitter", "energy_spread_fwhm_ev", .5),
    ("emitter", "emission_current_na", 7777.),
])
def test_physical_or_nonisolated_source_controls_invalidate_response(owner, field, value):
    state = default_state()
    before = calculation_signatures(state)
    target = state.electron_gun.emitter if owner == "emitter" else getattr(state, owner)
    setattr(target, field, value)
    assert calculation_signatures(state)["eds_response"] != before["eds_response"]


def _forbid_material_and_photons(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Response replay repeated executed particle/X-ray physics")
    for path in ("temsim.optics.electron_gun.source.trace_source_to_exit",
                 "temsim.specimen.elastic_transport.simulate_elastic_point_transport",
                 "temsim.detector.eds_signal.simulate_eds_point",
                 "temsim.detector.eds_signal.simulate_eds_tracks",
                 "temsim.detector.eds_photon_transport.transport_eds_photons"):
        monkeypatch.setattr(path, forbidden)


def _fresh_spectrum(case):
    from temsim.component_keys import EDS_DETECTOR_SYSTEM
    from temsim.detector.eds_geometry import EDSDetectorArrayGeometry
    from temsim.detector.eds_signal import simulate_eds_point
    interactions = case.first.specimen_interactions
    point = interactions.incident_bundle.original_centroid_nm
    geometry = EDSDetectorArrayGeometry.from_part_data(
        case.state._resolved_assembly.part(EDS_DETECTOR_SYSTEM).data)
    return simulate_eds_point(case.state, geometry, simulation=case.first.simulation,
        x_nm=point[0], y_nm=point[1], elastic_transport=interactions.elastic_transport,
        incident_bundle=interactions.incident_bundle)


@pytest.mark.parametrize("change", ["current", "dwell", "readout"])
def test_section_response_replay_matches_fresh_physics_and_keeps_original(eds_case, monkeypatch, change):
    case = eds_case
    old = case.first.specimen_interactions.eds_spectrum
    old_counts = old.expected_counts.copy()
    old_dwell = old.metrics["dwell_time_s"]
    if change == "current":
        case.state.column_current_limit_percent *= .5
    elif change == "dwell":
        case.state.ac_deflector.scan_frame_period_s *= 2.
    else:
        case.state.sample.eds_spectrum_bin_width_ev = 20.
        case.state.sample.eds_energy_resolution_fwhm_ev = 140.
        case.state.sample.eds_poisson_enabled = True
        case.state.sample.eds_poisson_seed = 321
    expected = _fresh_spectrum(case)
    _forbid_material_and_photons(monkeypatch)
    result = pipeline.calculate_particle_section(case.state, 458., existing_result=case.first)
    updated = result.specimen_interactions.eds_spectrum
    assert {"eds_response", "elastic", "inelastic"} <= result.reused_products
    assert "eds" in result.calculated_products and "eds" not in result.reused_products
    assert updated is not old
    assert updated.elastic_transport is old.elastic_transport
    assert updated.response_rates is old.response_rates
    np.testing.assert_array_equal(old.expected_counts, old_counts)
    assert old.metrics["dwell_time_s"] == old_dwell
    np.testing.assert_array_equal(updated.energy_bin_centres_ev, expected.energy_bin_centres_ev)
    np.testing.assert_allclose(updated.expected_counts, expected.expected_counts, rtol=2e-12, atol=1e-12)
    if expected.sampled_counts is not None:
        np.testing.assert_array_equal(updated.sampled_counts, expected.sampled_counts)
    assert updated.metrics["dwell_time_s"] == expected.metrics["dwell_time_s"]
    assert updated.metrics["electrons_reaching_sample_plane"] == expected.metrics["electrons_reaching_sample_plane"]
    assert result.specimen_exit.metrics["source_probability_conserved"]


@pytest.mark.parametrize("loaded", [False, True])
def test_full_pipeline_keeps_rebuilt_readout_when_elastic_is_already_reusable(eds_case, monkeypatch, loaded):
    case = eds_case
    seed = replace(case.first, specimen_interactions=None) if loaded else case.first
    if loaded:
        seed.loaded_section_only = True
    case.state.column_current_limit_percent *= .5
    before = case.first.specimen_interactions.eds_spectrum
    monkeypatch.setattr(pipeline, "run", lambda *args, **kwargs: replace(case.first.simulation))
    _forbid_material_and_photons(monkeypatch)
    progress = []
    result = pipeline.calculate(case.state, existing_result=seed,
        progress_callback=lambda *row: progress.append(row))
    after = result.specimen_interactions.eds_spectrum
    assert after is not before
    assert {"eds_response", "elastic"} <= result.reused_products
    assert "eds" in result.calculated_products
    np.testing.assert_allclose(after.expected_counts, before.expected_counts*.5, rtol=2e-12, atol=1e-12)
    assert result.specimen_interactions.events
    assert progress[-1][2] == "Complete"
    assert any(row["stage"] == "Updating EDS dose and spectral readout"
               for row in result.performance["stages"])


def test_legacy_spectrum_without_rates_allows_exact_reuse_but_not_response_replay(eds_case):
    from temsim.physics.completed_particle_section import compatible_material_eds, replay_material_eds
    case = eds_case
    cache = case.first.simulation.material_section_cache
    cache = replace(cache, eds_spectrum=replace(cache.eds_spectrum, response_rates=None))
    signatures = calculation_signatures(case.state)
    assert compatible_material_eds(cache, signatures, case.state) is cache.eds_spectrum
    case.state.column_current_limit_percent *= .5
    assert replay_material_eds(cache, calculation_signatures(case.state), case.state,
                               case.first.specimen_interactions.incident_bundle) is None


@pytest.mark.parametrize("loaded", [False, True])
def test_exact_full_eds_reuse_restores_ledger_once_without_physics(eds_case, monkeypatch, loaded):
    case = eds_case
    before = case.first.specimen_interactions
    assert before.events and before.conservation
    seed = replace(case.first, specimen_interactions=None) if loaded else case.first
    if loaded:
        seed.loaded_section_only = True
    next(lens for lens in case.state.lenses if lens.key == "projector_lens_2").percent += .1
    monkeypatch.setattr(pipeline, "run", lambda *args, **kwargs: replace(case.first.simulation))
    _forbid_material_and_photons(monkeypatch)
    calls = []
    original = pipeline.run_specimen_interactions
    def counted(*args, **kwargs):
        calls.append(kwargs.get("existing_result"))
        return original(*args, **kwargs)
    monkeypatch.setattr(pipeline, "run_specimen_interactions", counted)
    result = pipeline.calculate(case.state, existing_result=seed)
    after = result.specimen_interactions
    assert len(calls) == 1
    assert after.eds_spectrum is before.eds_spectrum
    assert after.elastic_transport is before.elastic_transport
    assert after.events == before.events
    assert after.conservation == before.conservation
    assert after.couplings == before.couplings
    assert {"elastic", "eds"} <= result.reused_products
    assert "eds" not in result.calculated_products
    assert "eds_response" not in result.reused_products


@pytest.mark.parametrize("missing", ["product", "rates"])
def test_missing_response_uses_real_eds_stage_label(eds_case, monkeypatch, missing):
    from temsim.detector import eds_signal
    case = eds_case
    cache = case.first.simulation.material_section_cache
    spectrum = None if missing == "product" else replace(cache.eds_spectrum, response_rates=None)
    simulation = replace(case.first.simulation)
    simulation.material_section_cache = replace(cache, eds_spectrum=spectrum)
    seed = replace(case.first, simulation=simulation, specimen_interactions=None)
    seed.loaded_section_only = True
    case.state.sample.eds_poisson_seed += 1
    monkeypatch.setattr(pipeline, "run", lambda *args, **kwargs: replace(simulation))
    progress, stage_at_execution = [], []
    original = eds_signal.simulate_eds_point
    def counted(*args, **kwargs):
        stage_at_execution.append(progress[-1][2])
        return original(*args, **kwargs)
    monkeypatch.setattr(eds_signal, "simulate_eds_point", counted)
    result = pipeline.calculate(case.state, existing_result=seed,
        progress_callback=lambda *row: progress.append(row))
    assert len(stage_at_execution) == 1
    assert "Resolving the EDS point spectrum" in stage_at_execution[0]
    assert not any("Updating EDS dose and spectral readout" in row[2] for row in progress)
    assert "eds" in result.calculated_products
    assert "eds_response" not in result.reused_products
    assert any(row["stage"] == "Resolving the EDS point spectrum"
               for row in result.performance["stages"])
