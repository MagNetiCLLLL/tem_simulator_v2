"""Workflow ownership fixtures; these do not qualify physical transport."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim import simulation_pipeline as p
from temsim import simulation_workflow as workflow
from temsim.optics.column import default_state
from temsim.specimen.interaction_types import SpecimenObservable as O
from temsim.specimen.downstream_transport import GeometricSpecimenExit


@pytest.fixture
def case(monkeypatch):
    state = default_state()
    from specimen_inputs import imported_sample
    imported_sample(state)
    state.sample.wave_enabled = state.sample.stem_wave_enabled = False
    state.sample.eds_enabled = True
    state.sample.inserted = True
    state.ac_deflector.scan_enabled = False
    state.ac_deflector.wobble_enabled = False
    state.energy_filter.enabled = False
    state._resolved_assembly = SimpleNamespace(part=lambda key: SimpleNamespace(data={}))
    n = 2
    zeros = np.zeros((2, n))
    branch = SimpleNamespace(z=np.array([450., state.sample.z_mm]),
        weight=1.,
        x=zeros, tx=zeros, y=zeros, ty=zeros, ray_weight=np.full(n, .5),
        alive=np.ones(n, bool), blocked_z=np.full(n, np.nan), blocked_key=np.full(n, ""),
        source_ray_id=np.arange(n), source_azimuth_rad=np.zeros(n))
    checkpoints = SimpleNamespace(z_mm=np.array([450., state.sample.z_mm]),
        x_m=zeros, tx_rad=zeros, y_m=zeros, ty_rad=zeros,
        flight_time_s=np.full((2, n), 1e-9), kinetic_energy_ev=np.full((2, n), 3e5))
    simulation = p.Simulation(incident=branch, branches={"000": branch},
        metrics={"sample_beam_surviving_fraction": 1.},
        gun_trace=SimpleNamespace(z_mm=np.array([0., 450.])),
        incident_plan=object(), incident_checkpoints=checkpoints)
    calls = []
    base_keys = ("request", "incident", "column", "elastic", "eds", "wave", "stem",
                 "sample_downstream", "sample_region", "scan_geometry", "scan_ray_paths", "energy_filter")
    def signatures(s):
        result = {key: key for key in base_keys}
        if s.sample.thickness_nm != state._test_thickness:
            for key in ("column", "elastic", "eds", "wave", "stem", "sample_downstream", "sample_region"):
                result[key] += ":new-specimen"
        result["incident"] += f":{s.lenses[0].percent}"
        return result
    state._test_thickness = state.sample.thickness_nm
    monkeypatch.setattr(p, "calculation_signatures", signatures)
    import temsim.calculation_cache as cache
    # This fixture uses synthetic signatures/assembly; real scan-dependency
    # admission is covered by test_stem_workflow_scan_resume.
    monkeypatch.setattr(cache, "scan_controls_only_incident_change", lambda *args: False)
    for name in ("ensure_recording_system", "ensure_energy_filter", "ensure_corrector_structure",
                 "normalise_component_names", "assert_external_input_inventory_unchanged"):
        monkeypatch.setattr(p, name, lambda *args: None)
    monkeypatch.setattr(p, "capture_external_input_identities", lambda *args: ())
    monkeypatch.setattr(p, "apply_physical_layout_to_state", lambda *args: object())
    monkeypatch.setattr(p, "sample_illumination_absent", lambda *args: False)
    monkeypatch.setattr(p, "specimen_interactions_active", lambda sample: sample.inserted)
    monkeypatch.setattr(p, "detect_all_lens_crossovers", lambda *args: ())
    monkeypatch.setattr(p, "aperture_stop_records", lambda *args: ())
    import temsim.physics.illumination as illumination
    import temsim.geometry_effects as geometry
    import temsim.physics.completed_particle_section as completed
    import temsim.specimen.elastic_transport as elastic
    import temsim.detector.eds_geometry as eds_geometry
    import temsim.detector.particle_readout as readout
    import temsim.physics.scan_geometry as scan_geometry
    monkeypatch.setattr(illumination, "illumination_config", lambda *args: None)
    monkeypatch.setattr(geometry, "admit_state_geometry", lambda *args: None)
    monkeypatch.setattr(completed, "capture_completed_particle_section", lambda *args: True)
    monkeypatch.setattr(elastic, "incident_rays_from_simulation",
        lambda *args: SimpleNamespace(original_centroid_nm=(1., 2.)))
    monkeypatch.setattr(eds_geometry.EDSDetectorArrayGeometry, "from_part_data", lambda *args: object())
    monkeypatch.setattr(readout, "measure_particle_detectors", lambda result: ("executed-pixel",))
    monkeypatch.setattr(scan_geometry, "calibrate_scan_system", lambda state: None)
    def run(s, **kwargs):
        calls.append(("run", kwargs))
        return replace(simulation, metrics=dict(simulation.metrics))
    def retain(previous, observables):
        if previous is None or not observables:
            return None
        return SimpleNamespace(**{**vars(previous),
            "elastic_transport": previous.elastic_transport if O.ELASTIC_TRANSPORT in observables else None,
            "inelastic_distribution": previous.inelastic_distribution if O.STOCHASTIC_INELASTIC in observables else None,
            "eds_spectrum": previous.eds_spectrum if O.CHARACTERISTIC_X_RAY in observables else None})
    def interact(s, sim, request, *, existing_result=None, **kwargs):
        calls.append(("interactions", request.observables))
        old = existing_result
        return SimpleNamespace(elastic_transport=getattr(old, "elastic_transport", None) or object(),
            inelastic_distribution=getattr(old, "inelastic_distribution", None) or object(),
            eds_spectrum=(getattr(old, "eds_spectrum", None) or object()
                          if O.CHARACTERISTIC_X_RAY in request.observables
                          else getattr(old, "eds_spectrum", None)),
            completed_observables=request.observables, metrics={}, conservation=())
    def downstream(s, sim, *args, **kwargs):
        calls.append(("downstream", kwargs))
        return GeometricSpecimenExit((branch,), {
            "tracked_downstream_source_probability": 1.,
            "inelastic_absorbed_source_probability": 0.},
            dependency_signature=kwargs["dependency_signature"])
    monkeypatch.setattr(p, "run", run)
    monkeypatch.setattr(p, "retain_specimen_observables", retain)
    monkeypatch.setattr(p, "run_specimen_interactions", interact)
    monkeypatch.setattr(p, "build_geometric_specimen_exit", downstream)
    monkeypatch.setattr(p, "calculate_scan_geometry", lambda *args: calls.append(("scan_geometry",)) or object())
    monkeypatch.setattr(p, "calculate_scan_ray_paths", lambda *args: calls.append(("scan_paths",)) or object())
    monkeypatch.setattr(p, "calculate_stem_scan_frame", lambda *args, **kwargs: calls.append(("stem",)) or object())
    return SimpleNamespace(state=state, calls=calls, simulation=simulation)


def test_rays_ignores_optional_sample_readouts_without_rewriting_snapshot(case):
    c = case
    c.state.sample.wave_enabled = c.state.sample.stem_wave_enabled = True
    c.state.ac_deflector.enabled = c.state.ac_deflector.scan_enabled = True
    result = p.calculate(c.state, workflow="rays")
    assert [row[0] for row in c.calls] == ["run"]
    assert c.calls[0][1]["optical_only"] is True
    assert result.state_snapshot.sample.inserted
    assert result.state_snapshot.sample.eds_enabled
    assert result.state_snapshot.sample.stem_wave_enabled
    assert result.specimen_interactions is result.stem_scan is None
    assert result.workflow == result.performance["workflow"] == "rays"


@pytest.mark.parametrize("scope", ["sample", "eds", "stem", "energy_filter", "sample_region"])
def test_dependent_page_requires_executed_ray_prerequisite(case, scope):
    with pytest.raises(ValueError, match="Run Ray Diagram"):
        p.calculate(case.state, workflow=scope)
    assert case.calls == []


def test_sample_runs_only_material_then_eds_reuses_column_and_material(case):
    c = case
    rays = p.calculate(c.state, workflow="rays")
    c.calls.clear()
    sample = p.calculate(c.state, workflow="sample", existing_result=rays)
    assert [row[0] for row in c.calls] == ["interactions", "downstream"]
    assert O.CHARACTERISTIC_X_RAY not in c.calls[0][1]
    assert sample.specimen_interactions.eds_spectrum is None
    c.calls.clear()
    eds = p.calculate(c.state, workflow="eds", existing_result=sample)
    assert [row[0] for row in c.calls] == ["interactions"]
    assert O.CHARACTERISTIC_X_RAY in c.calls[0][1]
    assert eds.specimen_interactions.elastic_transport is sample.specimen_interactions.elastic_transport
    assert eds.specimen_exit is sample.specimen_exit
    assert {"column", "incident", "elastic", "sample_downstream"} <= eds.reused_products


def test_sample_edit_reuses_exact_incident_and_rebuilds_only_downstream(case, monkeypatch):
    c = case
    rays = p.calculate(c.state, workflow="rays")
    c.state.sample.thickness_nm += 1.
    c.calls.clear()
    def rebuild(state, previous):
        assert previous is rays.simulation
        c.calls.append(("optical_downstream",))
        return replace(previous, metrics=dict(previous.metrics))
    monkeypatch.setattr(workflow, "rebuild_optical_downstream", rebuild)
    sample = p.calculate(c.state, workflow="sample", existing_result=rays)
    assert [row[0] for row in c.calls] == ["optical_downstream", "interactions", "downstream"]
    assert sample.simulation.incident is rays.simulation.incident


def test_upstream_edit_or_missing_terminal_checkpoint_cannot_retrace_tip(case):
    c = case
    rays = p.calculate(c.state, workflow="rays")
    c.calls.clear()
    c.state.lenses[0].percent += 1.
    with pytest.raises(ValueError, match="Run Ray Diagram"):
        p.calculate(c.state, workflow="sample", existing_result=rays)
    assert c.calls == []
    c.state.lenses[0].percent -= 1.
    rays.simulation.incident_checkpoints = None
    with pytest.raises(ValueError, match="Run Ray Diagram"):
        p.calculate(c.state, workflow="eds", existing_result=rays)
    assert c.calls == []


def test_stem_without_raster_returns_current_pixel_without_frame(case):
    rays = p.calculate(case.state, workflow="rays")
    case.calls.clear()
    result = p.calculate(case.state, workflow="stem", existing_result=rays)
    assert result.particle_signals == ("executed-pixel",)
    assert result.stem_scan is None
    assert not any(row[0].startswith("scan") or row[0] == "stem" for row in case.calls)


def test_stem_raster_computes_frame_but_not_eds(case):
    c = case
    c.state.ac_deflector.enabled = c.state.ac_deflector.scan_enabled = True
    rays = p.calculate(c.state, workflow="rays")
    c.calls.clear()
    result = p.calculate(c.state, workflow="stem", existing_result=rays)
    assert [row[0] for row in c.calls] == ["interactions", "downstream", "scan_geometry", "scan_paths", "stem"]
    assert result.stem_scan is not None
    assert result.specimen_interactions.eds_spectrum is None


def test_scan_change_reexecutes_incident_before_material_and_stem(case, monkeypatch):
    import temsim.calculation_cache as cache
    import temsim.physics.particle_sections as sections
    c = case
    old_signatures = p.calculation_signatures

    def signatures(state):
        result = old_signatures(state)
        if state.ac_deflector.scan_enabled:
            for key in ("incident", "column", "elastic", "stem", "sample_downstream",
                        "sample_region", "scan_geometry", "scan_ray_paths"):
                result[key] += ":scan"
        return result

    monkeypatch.setattr(p, "calculation_signatures", signatures)
    rays = p.calculate(c.state, workflow="rays")
    sample = p.calculate(c.state, workflow="sample", existing_result=rays)
    sample.simulation.section_checkpoint = SimpleNamespace(
        gun_trace=sample.simulation.gun_trace, gun_dependency_signature="executed-gun")
    monkeypatch.setattr(cache, "scan_controls_only_incident_change", lambda *args: True)
    monkeypatch.setattr(sections, "validate_section_checkpoint", lambda cp: cp)
    monkeypatch.setattr(sections, "gun_dependency_signature", lambda state: "executed-gun")
    c.state.ac_deflector.enabled = c.state.ac_deflector.scan_enabled = True
    c.calls.clear()
    result = p.calculate(c.state, workflow="stem", existing_result=sample)
    assert [row[0] for row in c.calls] == [
        "run", "interactions", "downstream", "scan_geometry", "scan_paths", "stem"]
    assert c.calls[0][1]["existing_simulation"] is sample.simulation
    assert c.calls[0][1]["optical_only"] is True
    assert {"incident", "column", "stem"} <= result.calculated_products
    assert "gun" in result.reused_products
    assert result.specimen_exit is not sample.specimen_exit
    assert result.signatures["incident"] != sample.signatures["incident"]


def test_unrequested_matching_eds_survives_sample_request_but_stale_eds_does_not(case, monkeypatch):
    c = case
    rays = p.calculate(c.state, workflow="rays")
    eds = p.calculate(c.state, workflow="eds", existing_result=rays)
    again = p.calculate(c.state, workflow="sample", existing_result=eds)
    assert again.specimen_interactions.eds_spectrum is eds.specimen_interactions.eds_spectrum
    c.state.sample.thickness_nm += 1
    monkeypatch.setattr(workflow, "rebuild_optical_downstream",
        lambda state, previous: replace(previous, metrics=dict(previous.metrics)))
    changed = p.calculate(c.state, workflow="sample", existing_result=again)
    assert changed.specimen_interactions.eds_spectrum is None


def test_rays_filter_consumes_optical_reference_and_no_sample_losses(case, monkeypatch):
    c = case
    c.state.energy_filter.enabled = True
    sources = []
    def filtered(state, source, **kwargs):
        sources.append((source, kwargs))
        return SimpleNamespace(entrance_provenance=source.metrics["energy_filter_entrance_provenance"],
            entrance_dependency_signature=source.metrics["energy_filter_entrance_dependency_signature"])
    monkeypatch.setattr(p, "simulate_energy_filter", filtered)
    rays = p.calculate(c.state, workflow="rays")
    assert sources[0][0].metrics["energy_filter_entrance_provenance"] == "optical_column_reference"
    assert sources[0][1]["inelastic_distribution"] is None
    sample = p.calculate(c.state, workflow="sample", existing_result=rays)
    assert sources[-1][0].metrics["energy_filter_entrance_provenance"] == "validated_specimen_exit"
    assert sources[-1][1]["inelastic_distribution"] is sample.specimen_interactions.inelastic_distribution


def test_imaging_keeps_paused_wave_admission_without_transport(case):
    from temsim.physics.source_admission import UnsupportedWaveSource
    with pytest.raises(UnsupportedWaveSource):
        p.calculate(case.state, workflow="imaging")
    assert case.calls == []


def test_sample_region_keeps_executed_offaxis_point(monkeypatch):
    import temsim.specimen.sample_region as module
    from temsim.specimen.interaction_types import SpecimenInteractionRequest
    state = default_state()
    state.sample.inserted = state.sample.eds_enabled = True
    previous = SimpleNamespace(request=SpecimenInteractionRequest.eds_point(x_nm=13., y_nm=-8.))
    class RequestObserved(Exception):
        pass
    def stop_after_request(s, simulation, request, **kwargs):
        assert (request.point_x_nm, request.point_y_nm) == (13., -8.)
        assert kwargs["existing_result"] is previous
        raise RequestObserved
    monkeypatch.setattr(module, "run_specimen_interactions", stop_after_request)
    with pytest.raises(RequestObserved):
        module.simulate_sample_region(state, SimpleNamespace(simulation=object()), object(),
            existing_interactions=previous)


def test_sample_region_workflow_attaches_requested_local_paths(case, monkeypatch):
    import temsim.specimen.sample_region as module
    rays = p.calculate(case.state, workflow="rays")
    seen = []
    def region(state, result, geometry, **kwargs):
        seen.append(kwargs)
        return SimpleNamespace(interactions=result.specimen_interactions,
            specimen_exit=result.specimen_exit, metrics={})
    monkeypatch.setattr(module, "simulate_sample_region", region)
    result = p.calculate(case.state, workflow="sample_region", existing_result=rays)
    assert result.sample_region is not None
    assert result.sample_region.specimen_exit is result.specimen_exit
    assert result.specimen_interactions.eds_spectrum is not None
    assert seen[0]["existing_interactions"] is result.specimen_interactions
    assert "sample_region" in result.calculated_products


def test_workflow_progress_is_monotonic_and_finishes(case):
    rays = p.calculate(case.state, workflow="rays")
    events = []
    p.calculate(case.state, workflow="sample", existing_result=rays,
                progress_callback=lambda done, total, label: events.append((done/total, label)))
    assert [value for value, _ in events] == sorted(value for value, _ in events)
    assert events[-1] == (1., "Complete")


def test_ray_workflow_forwards_optical_substeps_and_cancellation(case, monkeypatch):
    from temsim.physics.transport_progress import report_transport_progress, transport_progress_active
    original = p.run
    def run(*args, **kwargs):
        report_transport_progress("Electron gun | Z 1 mm")
        report_transport_progress("Column | 64/100 electrons")
        return original(*args, **kwargs)
    monkeypatch.setattr(p, "run", run)
    events = []
    p.calculate(case.state, workflow="rays", progress_callback=lambda *row: events.append(row))
    labels = [row[2] for row in events]
    assert "Stage 1/5 | Electron gun | Z 1 mm" in labels
    assert "Stage 1/5 | Column | 64/100 electrons" in labels
    assert not transport_progress_active()
    def cancel(done, total, label):
        if "Electron gun" in label:
            raise InterruptedError("cancel requested")
    with pytest.raises(InterruptedError, match="cancel requested"):
        p.calculate(case.state, workflow="rays", progress_callback=cancel)
    assert not transport_progress_active()


def test_real_incident_signature_ignores_cif_and_thickness_but_not_upstream_lens():
    from temsim.calculation_cache import calculation_signatures
    from specimen_inputs import SI_CIF
    from temsim.column.state_layout import apply_physical_layout_to_state
    state = default_state()
    original = calculation_signatures(state)
    state.sample.thickness_nm += 1.
    state.sample.specimen_mode = "atomic"
    state.sample.cif_path = str(SI_CIF)
    apply_physical_layout_to_state(state)
    sample_changed = calculation_signatures(state)
    assert sample_changed["incident"] == original["incident"]
    state.lenses[0].percent += 1.
    assert calculation_signatures(state)["incident"] != original["incident"]


@pytest.mark.parametrize("field", ["z_mm", "upper_field_center_z_mm", "lower_field_center_z_mm", "upper_b0_t"])
def test_actual_objective_positions_and_fields_remain_incident_dependencies(field):
    from temsim.calculation_cache import calculation_signatures
    state = default_state()
    objective = next(lens for lens in state.lenses if hasattr(lens, "upper_field_center_z_mm"))
    before = calculation_signatures(state)["incident"]
    setattr(objective, field, getattr(objective, field) + .001)
    assert calculation_signatures(state)["incident"] != before


def test_reused_column_refreshes_dose_metrics_without_retracing(case):
    rays = p.calculate(case.state, workflow="rays")
    case.state.column_current_limit_percent *= .5
    case.calls.clear()
    result = p.calculate(case.state, workflow="rays", existing_result=rays)
    assert case.calls == []
    assert result.simulation.metrics["effective_source_current_pa"] == pytest.approx(
        .5 * rays.simulation.metrics["effective_source_current_pa"])
    assert result.simulation.metrics["sample_surviving_current_pa"] == pytest.approx(
        .5 * rays.simulation.metrics["sample_surviving_current_pa"])


@pytest.mark.parametrize("scope,change,message", [
    ("energy_filter", lambda s: None, "Assemble and enable"),
    ("eds", lambda s: setattr(s.sample, "inserted", False), "Insert a specimen"),
    ("eds", lambda s: setattr(s.sample, "eds_enabled", False), "Enable EDS"),
    ("stem", lambda s: (setattr(s.ac_deflector, "enabled", True),
        setattr(s.ac_deflector, "scan_enabled", True),
        setattr(s.sample, "stem_image_enabled", False)), "Enable STEM detector images"),
])
def test_missing_requested_readout_preflight_does_not_calculate_material(case, scope, change, message):
    rays = p.calculate(case.state, workflow="rays")
    change(case.state)
    case.calls.clear()
    with pytest.raises(ValueError, match=message):
        p.calculate(case.state, workflow=scope, existing_result=rays)
    assert case.calls == []


def test_no_incident_electrons_has_explicit_zero_event_status(case, monkeypatch):
    rays = p.calculate(case.state, workflow="rays")
    monkeypatch.setattr(p, "sample_illumination_absent", lambda *args: True)
    case.calls.clear()
    result = p.calculate(case.state, workflow="eds", existing_result=rays)
    assert result.simulation.metrics["workflow_status"] == "no_illumination"
    assert result.simulation.metrics["sample_surviving_current_pa"] == 0.
    assert result.specimen_interactions is None
    assert case.calls == []


def test_partial_precision_incident_checkpoint_is_rejected(case):
    rays = p.calculate(case.state, workflow="rays")
    rays.simulation.incident_checkpoints.x_m = np.asarray(
        rays.simulation.incident_checkpoints.x_m, dtype=np.float32)
    with pytest.raises(ValueError, match="full-precision"):
        p.calculate(case.state, workflow="sample", existing_result=rays)


def test_retained_local_sample_paths_rebind_to_new_downstream(case, monkeypatch):
    rays = p.calculate(case.state, workflow="rays")
    eds = p.calculate(case.state, workflow="eds", existing_result=rays)
    old_region = SimpleNamespace(metrics={"sample_region_signature": eds.signatures["sample_region"]})
    eds.sample_region = old_region
    original_signatures = p.calculation_signatures
    def changed_projection(state):
        values = original_signatures(state)
        return {**values, "column": values["column"] + ":projection",
                "sample_downstream": values["sample_downstream"] + ":projection"}
    monkeypatch.setattr(p, "calculation_signatures", changed_projection)
    monkeypatch.setattr(workflow, "rebuild_optical_downstream",
        lambda state, previous: replace(previous, metrics=dict(previous.metrics)))
    monkeypatch.setattr(p, "_rebind_reused_sample_region", lambda region, *args: region)
    rebound = []
    def bind(region, checkpoint, interactions, **kwargs):
        assert region is old_region
        assert checkpoint is not eds.specimen_exit
        assert checkpoint.dependency_signature == kwargs["expected_signature"]
        rebound.append(checkpoint)
        return SimpleNamespace(specimen_exit=checkpoint)
    monkeypatch.setattr(p, "bind_sample_region_downstream", bind)
    result = p.calculate(case.state, workflow="sample", existing_result=eds)
    assert result.sample_region.specimen_exit is result.specimen_exit
    assert rebound == [result.specimen_exit]


