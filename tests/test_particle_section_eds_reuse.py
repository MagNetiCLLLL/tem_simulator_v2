"""Exact EDS reuse on bounded drift fixtures with real material/EDS physics.

The three-ray gun is an explicit fixture; real tip-origin persistence is
covered separately. No cache admission test substitutes electron counts for
the executed incident phase space.
"""
from dataclasses import replace

import numpy as np
import pytest

from test_material_particle_sections import material_case
from test_particle_sections import fixture as optical_fixture
from temsim import simulation_pipeline as pipeline
from temsim.physics.completed_particle_section import restore_material_interactions
from temsim.specimen.interaction_types import SpecimenObservable


@pytest.fixture
def eds_case(material_case):
    case = material_case
    case.state.sample.eds_enabled = True
    case.first = pipeline.calculate_particle_section(case.state, 456.)
    assert case.first.specimen_interactions.eds_spectrum is not None
    assert case.first.specimen_interactions.eds_spectrum.lines
    return case


def forbid_physics(monkeypatch):
    for path in (
        "temsim.optics.electron_gun.source.trace_source_to_exit",
        "temsim.specimen.elastic_transport.simulate_elastic_point_transport",
        "temsim.detector.eds_signal.simulate_eds_point",
    ):
        monkeypatch.setattr(path, lambda *a, **k: pytest.fail("Compatible executed physics was repeated"))


def test_section_lens_edit_reuses_complete_spectrum_and_rebuilds_ledger(eds_case, monkeypatch):
    case = eds_case
    spectrum = case.first.specimen_interactions.eds_spectrum
    assert case.first.simulation.material_section_cache.eds_spectrum is spectrum
    next(l for l in case.state.lenses if l.key == "projector_lens_2").percent += .1
    forbid_physics(monkeypatch)
    result = pipeline.calculate_particle_section(case.state, 458., existing_result=case.first)
    assert {"eds", "elastic", "inelastic"} <= result.reused_products
    assert "eds" not in result.calculated_products
    assert result.specimen_interactions.eds_spectrum is spectrum
    assert result.specimen_interactions.events == case.first.specimen_interactions.events
    assert result.specimen_interactions.conservation == case.first.specimen_interactions.conservation
    assert all(branch.z[-1] == 458. for branch in result.specimen_exit.branches)


def test_normal_pipeline_restores_eds_and_completes_progress(eds_case, monkeypatch):
    case = eds_case
    seed = replace(case.first, specimen_interactions=None)
    seed.loaded_section_only = True
    monkeypatch.setattr(pipeline, "run", lambda *a, **k: replace(case.first.simulation))
    forbid_physics(monkeypatch)
    progress = []
    result = pipeline.calculate(case.state, existing_result=seed,
        progress_callback=lambda *row: progress.append(row))
    assert {"eds", "elastic"} <= result.reused_products
    assert "eds" not in result.calculated_products
    assert result.specimen_interactions.eds_spectrum is case.first.specimen_interactions.eds_spectrum
    assert result.simulation.material_section_cache.eds_spectrum is result.specimen_interactions.eds_spectrum
    assert result.specimen_interactions.events == case.first.specimen_interactions.events
    assert progress[-1][0] == progress[-1][1] and progress[-1][2] == "Complete"
    times = result.performance["stages"]
    assert any(row["stage"] == "Restoring the cached EDS point spectrum" for row in times)
    assert times[-1]["stage"] == "Finalising optical diagnostics"


@pytest.mark.parametrize("field, value", [
    ("eds_poisson_seed", 123),
    ("eds_energy_resolution_fwhm_ev", 140.),
    ("eds_spectrum_bin_width_ev", 20.),
    ("eds_overlap_sampling_points", 512),
])
def test_eds_readout_edits_replay_response_but_sampling_recomputes_physics(eds_case, monkeypatch, field, value):
    from temsim.detector import eds_signal
    case = eds_case
    setattr(case.state.sample, field, value)
    calls = []
    original = eds_signal.simulate_eds_point
    def count(*args, **kwargs):
        calls.append(kwargs["elastic_transport"])
        return original(*args, **kwargs)
    monkeypatch.setattr(eds_signal, "simulate_eds_point", count)
    result = pipeline.calculate_particle_section(case.state, 458., existing_result=case.first)
    if field == "eds_overlap_sampling_points":
        assert calls == [case.first.specimen_interactions.elastic_transport]
        assert "eds_response" not in result.reused_products
    else:
        assert calls == []
        assert "eds_response" in result.reused_products
    assert "eds" in result.calculated_products and "eds" not in result.reused_products
    assert "elastic" in result.reused_products


@pytest.mark.parametrize("field", ["x", "y", "tx", "ty", "flight_time_s",
                                   "energy_offset_ev", "ray_weight", "source_ray_id", "alive"])
def test_same_population_does_not_admit_changed_incident_state(eds_case, field):
    case = eds_case
    incident = case.first.simulation.incident
    value = np.array(getattr(incident, field), copy=True)
    index = (-1, 0) if value.ndim == 2 else (0,)
    value[index] = not value[index] if field == "alive" else value[index] + (1 if field == "source_ray_id" else 1e-9)
    changed = replace(case.first.simulation, incident=replace(incident, **{field: value}))
    assert restore_material_interactions(case.state, changed, case.first) == (None, None)


def test_missing_old_eds_and_disabled_readout_do_not_claim_completion(eds_case):
    case = eds_case
    sim = replace(case.first.simulation)
    sim.material_section_cache = replace(case.first.simulation.material_section_cache, eds_spectrum=None)
    historical = replace(case.first, simulation=sim)
    restored, _ = restore_material_interactions(case.state, case.first.simulation, historical)
    assert restored.eds_spectrum is None
    assert SpecimenObservable.CHARACTERISTIC_X_RAY not in restored.completed_observables
    case.state.sample.eds_enabled = False
    restored, _ = restore_material_interactions(case.state, case.first.simulation, case.first)
    # The existing material signature conservatively invalidates this toggle.
    assert restored is None or restored.eds_spectrum is None
    assert restored is None or SpecimenObservable.CHARACTERISTIC_X_RAY not in restored.completed_observables


def test_changed_sample_invalidates_executed_material_and_eds(eds_case):
    case = eds_case
    case.state.sample.thickness_nm += 1.
    assert restore_material_interactions(case.state, case.first.simulation, case.first) == (None, None)


def test_scan_does_not_admit_point_material_archive(eds_case):
    case = eds_case
    case.state.ac_deflector.enabled = case.state.ac_deflector.scan_enabled = True
    assert restore_material_interactions(case.state, case.first.simulation, case.first) == (None, None)


def test_call_overrides_and_legacy_missing_identity_do_not_use_default_cache(eds_case):
    from copy import deepcopy
    from temsim.physics.completed_particle_section import compatible_material_eds
    case = eds_case
    cache = case.first.simulation.material_section_cache
    spectrum = cache.eds_spectrum
    assert compatible_material_eds(cache, cache.signatures, case.state) is spectrum
    for key, value in (("dwell_time_s_override", 2.), ("incident_electrons_override", 1000.),
                       ("custom_detector_surfaces", True), ("custom_holder_occluders", True),
                       ("photon_quadrature_order", 2), ("photon_maximum_stored_paths", 20)):
        metrics = deepcopy(spectrum.metrics)
        metrics["point_request_inputs"][key] = value
        changed = replace(cache, eds_spectrum=replace(spectrum, metrics=metrics))
        assert compatible_material_eds(changed, cache.signatures, case.state) is None
    metrics = deepcopy(spectrum.metrics)
    metrics["point_request_inputs"]["detector_geometry"]["takeoff_angle_deg"] += 1.
    assert compatible_material_eds(replace(cache, eds_spectrum=replace(spectrum, metrics=metrics)),
                                   cache.signatures, case.state) is None
    metrics.pop("point_request_inputs")
    assert compatible_material_eds(replace(cache, eds_spectrum=replace(spectrum, metrics=metrics)),
                                   cache.signatures, case.state) is None


def test_dose_pose_support_and_detector_changes_invalidate_spectrum(eds_case):
    case = eds_case
    from temsim.specimen.geometry import set_sample_orientation_euler_xyz_deg
    previous_orientation = case.state.sample.specimen_orientation_quaternion_wxyz
    set_sample_orientation_euler_xyz_deg(case.state.sample, (1., 0., 0.))
    restored, _ = restore_material_interactions(case.state, case.first.simulation, case.first)
    assert restored is None or restored.eds_spectrum is None
    case.state.sample.specimen_orientation_quaternion_wxyz = previous_orientation
    for owner, field, increment in (
        (case.state.sample, "eds_support_offset_x_um", 10.),
        (case.state.sample, "eds_detector_efficiency", -.1),
    ):
        old = getattr(owner, field)
        setattr(owner, field, old + increment)
        restored, _ = restore_material_interactions(case.state, case.first.simulation, case.first)
        assert restored is None or restored.eds_spectrum is None, field
        setattr(owner, field, old)
