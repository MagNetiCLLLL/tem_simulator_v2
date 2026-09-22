"""Short classical particle sections exercise real finite specimen scattering."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from test_particle_sections import fixture as optical_fixture
from temsim import simulation_pipeline as pipeline
from temsim.physics import particle_sections as sections
from temsim.specimen.interaction_types import SpecimenObservable, SpecimenInteractionRequest


@pytest.fixture
def material_case(optical_fixture, monkeypatch):
    state, gun, gun_calls = optical_fixture
    state.sample.inserted = True
    state.sample.wave_enabled = state.sample.stem_wave_enabled = False
    state.sample.eds_enabled = False
    state.sample.specimen_mode = "reference"
    state.sample.thickness_nm = 30.
    state.sample.size_x_nm = state.sample.size_y_nm = 10_000.
    state.sample.centre_x_nm = state.sample.centre_y_nm = 0.
    state.ac_deflector.scan_enabled = False
    state.energy_filter.enabled = False
    monkeypatch.setattr(pipeline, "apply_physical_layout_to_state", lambda s: object())
    monkeypatch.setattr(pipeline, "ensure_recording_system", lambda s: None)
    monkeypatch.setattr("temsim.calculation_manifest.solver_source_identity", lambda: "bounded-test-model")
    calls = []
    original = pipeline.run_specimen_interactions
    def interaction(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result)
        return result
    monkeypatch.setattr(pipeline, "run_specimen_interactions", interaction)
    return SimpleNamespace(state=state, gun=gun, gun_calls=gun_calls, interactions=calls)


def test_inserted_non_scanning_specimen_executes_elastic_and_inelastic(material_case):
    case = material_case
    result = pipeline.calculate_particle_section(case.state, 456.)
    assert pipeline._geometric_specimen_transport_requested(case.state)
    assert result.specimen_exit is not None
    assert result.simulation.metrics["sample_scattering_applied"]
    assert result.simulation.metrics["particle_section"]
    assert not result.simulation.metrics["optical_tuning"]
    assert result.specimen_interactions.completed_observables == frozenset((
        SpecimenObservable.ELASTIC_TRANSPORT, SpecimenObservable.STOCHASTIC_INELASTIC))
    assert result.specimen_interactions.eds_spectrum is None
    terminal = result.specimen_interactions.elastic_transport.terminal_electrons
    assert len(terminal.outcome) == 3
    assert result.specimen_exit.metrics["source_probability_conserved"]
    assert all(branch.z[-1] == 456. for branch in result.specimen_exit.branches)
    assert all(branch.interaction_kind != "optical_reference" for branch in result.simulation.branches.values())
    assert tuple(result.simulation.branches.values()) == result.specimen_exit.branches
    # Positive-loss channels retain the honest unknown event-depth clock.
    losses = [b for b in result.specimen_exit.branches if np.any(b.energy_offset_ev < 0.)]
    assert losses and all(np.all(np.isnan(branch.flight_time_s)) for branch in losses)


def test_current_pixel_preserves_actual_deflected_incident_centroid(material_case, monkeypatch):
    case = material_case
    monkeypatch.setattr(sections, "_events", lambda s: ((451., .0002, -.0001),))
    result = pipeline.calculate_particle_section(case.state, 456.)
    bundle = result.specimen_interactions.incident_bundle
    np.testing.assert_allclose(bundle.original_centroid_nm, (600., -300.), atol=1e-4)
    np.testing.assert_array_equal(bundle.target_centroid_nm, bundle.original_centroid_nm)
    assert result.specimen_interactions.request.point_x_nm == bundle.original_centroid_nm[0]


def test_post_sample_edits_reuse_executed_material_but_upstream_edits_invalidate(material_case):
    case = material_case
    first = pipeline.calculate_particle_section(case.state, 456.)
    projection = next(l for l in case.state.lenses if l.key == "projector_lens_2")
    projection.percent += .1
    second = pipeline.calculate_particle_section(case.state, 458., existing_result=first)
    assert second.specimen_interactions.elastic_transport is first.specimen_interactions.elastic_transport
    assert {"elastic", "inelastic"} <= second.reused_products
    assert second.specimen_exit is not first.specimen_exit
    assert all(b.z[-1] == 458. for b in second.specimen_exit.branches)
    same = pipeline.calculate_particle_section(case.state, 458., existing_result=second)
    assert same.specimen_exit is second.specimen_exit
    assert "sample_downstream" in same.reused_products
    case.state.lenses[0].percent += .1
    changed = pipeline.calculate_particle_section(case.state, 458., existing_result=same)
    assert changed.specimen_interactions.elastic_transport is not same.specimen_interactions.elastic_transport
    assert "elastic" in changed.calculated_products


def test_upstream_only_and_retracted_sample_do_not_invent_material_events(material_case, monkeypatch):
    case = material_case
    monkeypatch.setattr(pipeline, "run_specimen_interactions", lambda *a, **k: pytest.fail("Material was not traversed"))
    upstream = pipeline.calculate_particle_section(case.state, 452.)
    assert upstream.specimen_exit is None
    assert upstream.simulation.metrics["sample_statistics_status"] == "NOT_REACHED"
    assert upstream.signatures["incident"] != pipeline.calculation_signatures(case.state)["incident"]
    case.state.sample.inserted = False
    vacuum = pipeline.calculate_particle_section(case.state, 456.)
    assert vacuum.specimen_interactions is None
    assert vacuum.simulation.metrics["section_downstream_provenance"] == "vacuum_transport"
    assert vacuum.simulation.branches["000"].z[-1] == 456.


def test_full_path_executes_assembled_filter_after_material_once(material_case, monkeypatch):
    case = material_case
    case.state.energy_filter_installed = True
    case.state.energy_filter_mode = "energy_filter"
    case.state.energy_filter.enabled = True
    monkeypatch.setattr(sections, "section_limits", lambda s: (450.,456.))
    observed = []
    def energy_filter(state, simulation, **kwargs):
        assert simulation.metrics["energy_filter_entrance_provenance"] == "validated_specimen_exit"
        assert all(b.interaction_kind != "optical_reference" for b in simulation.branches.values())
        assert kwargs["inelastic_distribution"] is case.interactions[-1].inelastic_distribution
        observed.append(simulation)
        return object()
    monkeypatch.setattr(pipeline, "simulate_energy_filter", energy_filter)
    explicit = pipeline.calculate_particle_section(case.state, 456.)
    assert explicit.energy_filter is None and not observed
    full = pipeline.calculate_particle_section(case.state, existing_result=explicit)
    assert full.energy_filter is not None and len(observed) == 1
    assert full.simulation.metrics["section_full_path"]


def test_eds_is_an_explicit_extra_observable_without_replacing_elastic(material_case, monkeypatch):
    case = material_case
    case.state.sample.eds_enabled = True
    original = pipeline.run_specimen_interactions
    eds_calls = []
    def interaction(state, simulation, request, **kwargs):
        if SpecimenObservable.CHARACTERISTIC_X_RAY not in request.observables:
            return original(state, simulation, request, **kwargs)
        prior = kwargs["existing_result"]
        eds_calls.append(prior.elastic_transport)
        return replace(prior, eds_spectrum=SimpleNamespace(elastic_transport=prior.elastic_transport),
            completed_observables=prior.completed_observables | request.observables)
    monkeypatch.setattr(pipeline, "run_specimen_interactions", interaction)
    result = pipeline.calculate_particle_section(case.state, 456.)
    assert eds_calls == [result.specimen_interactions.elastic_transport]
    assert "eds" in result.calculated_products
    assert result.specimen_exit is not None


def _enable_small_scan(state):
    state.ac_deflector.enabled = state.ac_deflector.scan_enabled = True
    state.ac_deflector.z_mm = 451.
    state.ac_deflector.scan_pixels_x = state.ac_deflector.scan_lines = 2
    state.ac_deflector.set_scan_command_matrix_mrad(np.zeros((2, 2)))
    state.sample.stem_image_enabled = True
    for index, detector in enumerate(state.stem_detectors):
        detector.set_optical_reference_z_mm(state.selected_area_aperture.z_mm, 455. + index * .1)
        detector.inserted = detector.readout_enabled = True


def test_scan_off_and_unreached_detector_do_not_generate_frames(material_case, monkeypatch):
    case = material_case
    monkeypatch.setattr(pipeline, "calculate_stem_scan_frame", lambda *a, **k: pytest.fail("Unrequested/unreached scan"))
    off = pipeline.calculate_particle_section(case.state, 456.)
    assert off.stem_scan is None and off.simulation.metrics["section_scan_status"] == "disabled"
    _enable_small_scan(case.state)
    monkeypatch.setattr("temsim.physics.scan_geometry.calibrate_scan_system", lambda s, **kw: (np.zeros((2, 2)), 0., None))
    early = pipeline.calculate_particle_section(case.state, 454.5, existing_result=off)
    assert early.stem_scan is None
    assert early.simulation.metrics["section_scan_status"] == "detectors_not_reached"


def test_scan_uses_existing_bounded_exit_and_produces_real_2d_detector_arrays(material_case, monkeypatch):
    from temsim.detector import stem_signal
    case = material_case
    _enable_small_scan(case.state)
    # This isolated drift fixture has no installed scan pair to calibrate;
    # retain the configured commands and exercise the actual raster/readout.
    monkeypatch.setattr(stem_signal, "calibrate_scan_system", lambda state: None)
    monkeypatch.setattr("temsim.physics.scan_geometry.calibrate_scan_system", lambda s, **kw: (np.zeros((2, 2)), 0., None))
    monkeypatch.setattr(stem_signal, "build_geometric_specimen_exit", lambda *a, **k: pytest.fail("Executed bounded exit must be reused"))
    result = pipeline.calculate_particle_section(case.state, 456.)
    assert result.stem_scan is not None, result.simulation.metrics["section_scan_status"]
    assert result.scan_geometry is not None and result.scan_ray_paths is not None
    assert {detector.key for detector in case.state.stem_detectors} <= set(result.scan_geometry.plane_positions_um)
    assert all(plane.key not in result.scan_geometry.plane_positions_um
               for plane in case.state.recording_planes if plane.z_mm > 456.)
    assert result.simulation.metrics["section_scan_status"] == "calculated"
    assert result.stem_scan.metrics["shared_specimen_exit_transport_used"]
    assert result.stem_scan.metrics["finite_specimen_exit"]["downstream_stop_z_mm"] == 456.
    for values in result.stem_scan.fractions.values():
        assert values.shape == (2, 2)
        assert np.all(np.isfinite(values)) and np.all(values >= 0.)
    assert len(result.stem_scan.fractions) == len(case.state.stem_detectors)


def test_scan_invalid_exit_rebuild_stays_inside_requested_section(material_case, monkeypatch):
    from temsim.detector import stem_signal
    case = material_case
    _enable_small_scan(case.state)
    monkeypatch.setattr(stem_signal, "calibrate_scan_system", lambda state: None)
    monkeypatch.setattr("temsim.physics.scan_geometry.calibrate_scan_system", lambda s, **kw: (np.zeros((2, 2)), 0., None))
    result = pipeline.calculate_particle_section(case.state, 456.)
    calls = []
    original = stem_signal.build_geometric_specimen_exit
    def bounded(*args, **kwargs):
        calls.append(kwargs["stop_z_mm"])
        return original(*args, **kwargs)
    monkeypatch.setattr(stem_signal, "build_geometric_specimen_exit", bounded)
    frame = stem_signal.acquire_stem_scan(result.simulation, case.state,
        specimen_interactions=result.specimen_interactions,
        geometric_specimen_exit=result.specimen_exit,
        geometric_specimen_exit_signature="stale", observation_stop_z_mm=456.)
    assert calls == [456.]
    assert frame.metrics["finite_specimen_exit"]["downstream_stop_z_mm"] == 456.
    assert not frame.metrics["shared_specimen_exit_transport_used"]


def test_scan_calibration_precedes_consumption_of_reference_kicks(material_case, monkeypatch):
    case = material_case
    _enable_small_scan(case.state)
    calls = []
    def calibration(state, **kwargs):
        calls.append("calibrated")
        return np.zeros((2, 2)), 0., None
    original = sections._events
    def events(state):
        assert calls == ["calibrated"]
        return original(state)
    monkeypatch.setattr("temsim.physics.scan_geometry.calibrate_scan_system", calibration)
    monkeypatch.setattr(sections, "_events", events)
    # An earlier requested section still consumes the calibrated source kicks,
    # but does not calculate a detector frame or downstream geometry.
    result = pipeline.calculate_particle_section(case.state, 454.5)
    assert result.stem_scan is None and result.scan_geometry is None


def test_section_calibration_skips_unconsumed_coils_and_downstream_descan(material_case, monkeypatch):
    from temsim.physics import scan_geometry
    state = material_case.state
    ac, descan = state.ac_deflector, state.descan_deflector
    ac.enabled = ac.scan_enabled = descan.enabled = descan.scan_enabled = True
    first_ac = min(ac.upper_z_mm, ac.lower_z_mm)
    first_descan = min(descan.upper_z_mm, descan.lower_z_mm)
    assert first_ac < first_descan
    calls = []
    monkeypatch.setattr(scan_geometry, "calibrate_ac_scan_scale", lambda s: (calls.append("ac") or np.eye(2), 0.))
    monkeypatch.setattr(scan_geometry, "calibrate_descan_image_plane", lambda s: pytest.fail("Unconsumed downstream Descan"))
    assert scan_geometry.calibrate_scan_system(state, observation_stop_z_mm=first_ac-1.) is None
    assert not calls
    result = scan_geometry.calibrate_scan_system(state, observation_stop_z_mm=(first_ac+first_descan)/2.)
    assert calls == ["ac"] and result[2] is None


def test_scan_rejects_target_before_any_inserted_detector(material_case, monkeypatch):
    from temsim.detector import stem_signal
    case = material_case
    result = pipeline.calculate_particle_section(case.state, 456.)
    _enable_small_scan(case.state)
    with pytest.raises(ValueError, match="has not reached"):
        stem_signal.acquire_stem_scan(result.simulation, case.state,
            observation_stop_z_mm=454.5)


@pytest.mark.parametrize("eds_enabled", [False, True])
def test_high_current_pixel_matches_particle_section_without_recentering(material_case, monkeypatch, eds_enabled):
    from temsim.specimen import downstream_transport
    case = material_case
    monkeypatch.setattr(sections, "_events", lambda state: ((451., .0002, -.0001),))
    preview = pipeline.calculate_particle_section(case.state, 456.)
    # High consumes executed upstream state. The absent aggregate envelope
    # additionally verifies its material stage explicitly requests loss physics.
    upstream = replace(preview.simulation, branches={}, real_interactions=None)
    monkeypatch.setattr(pipeline, "run", lambda *args, **kwargs: upstream)
    monkeypatch.setattr(downstream_transport, "determine_tem_stop_z", lambda state: 456.)
    case.state.sample.eds_enabled = eds_enabled
    requests = []
    original = pipeline.run_specimen_interactions
    def interaction(state, simulation, request, **kwargs):
        requests.append(request)
        if SpecimenObservable.CHARACTERISTIC_X_RAY in request.observables:
            prior = kwargs["existing_result"]
            return replace(prior, eds_spectrum=SimpleNamespace(elastic_transport=prior.elastic_transport),
                completed_observables=prior.completed_observables | request.observables)
        return original(state, simulation, request, **kwargs)
    monkeypatch.setattr(pipeline, "run_specimen_interactions", interaction)
    high = pipeline.calculate(case.state)
    for request in requests:
        np.testing.assert_allclose((request.point_x_nm, request.point_y_nm), (600., -300.), atol=1e-4)
    bundle = high.specimen_interactions.incident_bundle
    np.testing.assert_array_equal(bundle.original_centroid_nm, bundle.target_centroid_nm)
    assert high.specimen_interactions.inelastic_distribution is not None
    assert SpecimenObservable.STOCHASTIC_INELASTIC in high.specimen_interactions.completed_observables
    np.testing.assert_array_equal(high.specimen_interactions.elastic_transport.terminal_electrons.position_nm,
        preview.specimen_interactions.elastic_transport.terminal_electrons.position_nm)
    assert high.specimen_exit.metrics["tracked_downstream_source_probability"] == pytest.approx(
        preview.specimen_exit.metrics["tracked_downstream_source_probability"])
