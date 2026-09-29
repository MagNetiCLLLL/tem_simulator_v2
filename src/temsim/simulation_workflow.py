"""Explicit page calculations over a dependency-checked executed incident beam.

The optical reference and the specimen products are separate executed states.
Selecting a page never changes the captured specimen or source parameters.
"""
from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import numpy as np


def require_incident_result(state, previous, signatures, *, scan_continuation=False):
    """Validate the executed prerequisite; changed scan drives need transport."""
    simulation = getattr(previous, "simulation", None)
    old = getattr(previous, "signatures", {}) or {}
    cp = getattr(simulation, "incident_checkpoints", None)
    incident = getattr(simulation, "incident", None)
    if (simulation is None or not old.get("incident")
            or not signatures.get("incident") or incident is None
            or getattr(simulation, "gun_trace", None) is None
            or getattr(simulation, "incident_plan", None) is None or cp is None
            or not len(getattr(cp, "z_mm", ()))
            or float(cp.z_mm[-1]) != float(state.sample.z_mm)
            or not len(getattr(incident, "z", ()))
            or float(incident.z[-1]) != float(state.sample.z_mm)
            or getattr(cp, "kinetic_energy_ev", None) is None
            or getattr(cp, "flight_time_s", None) is None):
        raise ValueError("Run Ray Diagram first: a matching, completed tip-to-specimen "
                         "calculation with exact incident checkpoints is required.")
    shape = (len(cp.z_mm), np.shape(incident.x)[1])
    for name in ("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s", "kinetic_energy_ev"):
        values = np.asarray(getattr(cp, name, None))
        if values.dtype != np.dtype(np.float64) or values.shape != shape:
            raise ValueError("Run Ray Diagram first: full-precision incident checkpoints are required.")
    if old.get("incident") != signatures.get("incident"):
        if not scan_continuation:
            raise ValueError("Run Ray Diagram first: upstream source, optics or numerical settings "
                             "changed since the saved incident beam was calculated.")
        from temsim.physics.particle_sections import gun_dependency_signature, validate_section_checkpoint
        checkpoint = getattr(simulation, "section_checkpoint", None)
        if checkpoint is None:
            raise ValueError("Run Ray Diagram first: changing the scan requires an executed "
                             "upstream continuation checkpoint.")
        validate_section_checkpoint(checkpoint)
        if (checkpoint.gun_trace is not simulation.gun_trace
                or checkpoint.gun_dependency_signature != gun_dependency_signature(state)):
            raise ValueError("Run Ray Diagram first: the saved electron-gun state cannot be reused "
                             "with the current scan and field settings.")
    return simulation


def rebuild_optical_downstream(state, previous):
    """Transport only after the exact executed sample-plane checkpoint.

    This is a cached boundary, never a configurable downstream source. It
    deliberately has no path to trace_source_to_exit or simulation.run.
    """
    from temsim.physics.particle_sections import _trace_segment, section_limits
    from temsim.physics.instrument_magnetic import active_column_events
    from temsim.physics.simulation import (
        configured_objective_chromatic_focal_mm, objective_chromatic_kick_from_state,
    )
    from temsim.physics.optical_tuning import tuning_metrics
    cp, incident = previous.incident_checkpoints, previous.incident
    x, tx, y, ty, time = (np.asarray(getattr(cp, name)[-1], dtype=np.float64)
                         for name in ("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s"))
    energy = np.asarray(cp.kinetic_energy_ev[-1], dtype=np.float64)
    offset = energy - float(state.beam_voltage_kv) * 1000.
    cx, cy = objective_chromatic_kick_from_state(state, x, y, offset,
        resolved_focal_mm=configured_objective_chromatic_focal_mm(state))
    start, stop = float(state.sample.z_mm), section_limits(state)[1]
    segment, _, _ = _trace_segment(state, "000", start, stop,
        (x, tx + cx, y, ty + cy, time), offset, incident.ray_weight,
        (incident.source_ray_id, incident.source_azimuth_rad),
        (incident.alive, incident.blocked_z, incident.blocked_key),
        active_column_events(state), None, float("inf"),
        initial_kicks=False, initial_kinetic_energy_ev=energy)
    segment.branch.interaction_kind = "optical_reference"
    metrics = tuning_metrics(state, incident)
    metrics["optical_execution_extent"] = {
        "coordinate_system": "column_axial_z_mm",
        "physics_scope": "optical_reference_without_specimen_interactions",
        "start_z_mm": float(previous.gun_trace.z_mm[0]), "completed_z_mm": float(stop)}
    metrics["workflow_incident_reused"] = True
    result = replace(previous, branches={"000": segment.branch}, metrics=metrics,
                     real_interactions=None, sample_to_analysis_transfer=None,
                     optical_transfers=())
    # The segment is an optical reference, not an executed specimen exit.
    # Physical continuation after material is owned by specimen_exit segments.
    result.completed_post_sections = ()
    return result


def calculate_workflow(state, *, workflow, progress_callback=None,
                       existing_result=None, diffraction_sink=None):
    """Execute one requested product; retain only compatible existing products."""
    from temsim import simulation_pipeline as p
    from temsim.calculation_workflow import admit_workflow, workflow_signatures
    from temsim.physics.illumination import illumination_config
    from temsim.geometry_effects import admit_state_geometry
    from temsim.specimen.interaction_types import SpecimenObservable as O
    started = perf_counter()
    report = p._cancellable_progress(state, progress_callback)
    report(0, 5, f"Preparing {workflow}")
    illumination_config(state)
    admit_workflow(state, workflow)
    if workflow != "rays":
        from temsim.specimen.source import validate_sample_source
        validate_sample_source(state.sample)
    p.ensure_recording_system(state)
    p.ensure_energy_filter(state)
    p.ensure_corrector_structure(state)
    p.normalise_component_names(state)
    admit_state_geometry(state)
    # Resolved mechanical coordinates and derived readbacks must be final
    # before validating or binding the executed input identity.
    layout = p.apply_physical_layout_to_state(state)
    if any(component.enabled and component.scan_enabled
           for component in (state.ac_deflector, state.descan_deflector)):
        from temsim.physics.scan_geometry import calibrate_scan_system
        calibrate_scan_system(state)
    external = p.capture_external_input_identities(state)
    signatures = p.calculation_signatures(state)
    signatures = workflow_signatures(signatures, workflow)
    p.assert_external_input_inventory_unchanged(state, external)
    old_signatures = getattr(existing_result, "signatures", None)
    reusable = p.matching_products(old_signatures, signatures)
    previous_simulation = getattr(existing_result, "simulation", None)
    scan_continuation = False
    if workflow != "rays":
        if old_signatures and old_signatures.get("incident") != signatures["incident"]:
            from temsim.calculation_cache import scan_controls_only_incident_change
            scan_continuation = scan_controls_only_incident_change(
                getattr(existing_result, "state_snapshot", None), state,
                old_signatures.get("incident"))
        require_incident_result(state, existing_result, signatures,
                                scan_continuation=scan_continuation)
    if workflow == "energy_filter" and not bool(state.energy_filter.enabled):
        raise ValueError("Assemble and enable the energy filter before calculating its response.")
    if workflow in {"eds", "sample_region"}:
        if not p.specimen_interactions_active(state.sample):
            raise ValueError("Insert a specimen with a valid structure before calculating EDS or sample-region products.")
        if not bool(state.sample.eds_enabled):
            raise ValueError("Enable EDS acquisition before calculating EDS or sample-region products.")
    if (workflow == "stem" and bool(state.ac_deflector.enabled and state.ac_deflector.scan_enabled)
            and not bool(getattr(state.sample, "stem_image_enabled", True))):
        raise ValueError("Enable STEM detector images before calculating an active raster.")
    calculated, reused = set(), set()
    progress = p._StageProgress([
        "Optical column transport", "Specimen interactions and downstream transport",
        "Requested optional readouts", "Assembled energy filter and detector readout",
        "Diagnostics and continuation checkpoint"], report, started_at=started)
    progress.report()

    def stage(index, label):
        progress.advance()

    def run_optics(**kwargs):
        from temsim.physics.transport_progress import transport_progress
        with transport_progress(progress.detail):
            return p.run(state, **kwargs)

    if scan_continuation:
        progress.update(0, 1, "Updating scan transport from the saved upstream checkpoints")
        simulation = run_optics(resolved_layout=layout,
            existing_simulation=previous_simulation, optical_only=True)
        calculated.add("column")
        reused.add("gun")
        if simulation.metrics.get("column_segment_cache", {}).get("mode") == "full_incident":
            reused.add("incident")
        else:
            calculated.add("incident")
        simulation.metrics["workflow_scan_continuation"] = True
    elif previous_simulation is not None and "column" in reusable:
        simulation = replace(previous_simulation, metrics=dict(previous_simulation.metrics))
        reused.update(("column", "incident"))
    elif workflow == "rays":
        simulation = run_optics(resolved_layout=layout,
            existing_simulation=previous_simulation, optical_only=True)
        calculated.add("column")
    else:
        simulation = rebuild_optical_downstream(state, previous_simulation)
        calculated.add("column")
        reused.add("incident")
    source_current = p.effective_source_current_pa(state)
    simulation.metrics.update(
        column_current_limit_percent=p.column_current_limit_percent(state),
        effective_source_current_pa=source_current,
        sample_surviving_current_pa=source_current * float(
            simulation.metrics.get("sample_beam_surviving_fraction", 0.)))
    stage(1, "Specimen interactions and downstream transport")
    no_illumination = p.sample_illumination_absent(simulation, state)
    if no_illumination:
        simulation.metrics.update(sample_illumination_status="no incident current",
            sample_surviving_current_pa=0., workflow_status="no_illumination",
            specimen_products_status="Zero incident electrons: no specimen or X-ray events")
    else:
        simulation.metrics["workflow_status"] = "calculated"
    material = workflow != "rays" and p.specimen_interactions_active(state.sample)

    previous_interactions = getattr(existing_result, "specimen_interactions", None)
    retain = set()
    if not no_illumination and "incident" in reusable:
        if "elastic" in reusable:
            retain.update((O.ELASTIC_TRANSPORT, O.STOCHASTIC_INELASTIC))
        if "eds" in reusable:
            retain.add(O.CHARACTERISTIC_X_RAY)
        if "wave" in reusable:
            retain.add(O.COHERENT_ELASTIC_WAVE)
    interactions = p.retain_specimen_observables(
        previous_interactions if retain else None, frozenset(retain))
    wave = (getattr(existing_result, "wave_imaging", None) if "wave" in reusable else None)
    stem = (getattr(existing_result, "stem_scan", None) if "stem" in reusable else None)
    scan_geometry = (getattr(existing_result, "scan_geometry", None)
                     if "scan_geometry" in reusable else None)
    scan_paths = (getattr(existing_result, "scan_ray_paths", None)
                  if "scan_ray_paths" in reusable else None)
    specimen_exit = p.validated_geometric_specimen_exit(
        getattr(existing_result, "specimen_exit", None), signatures["sample_downstream"])
    region = (getattr(existing_result, "sample_region", None)
              if "sample_region" in reusable else None)
    if region is not None and str((getattr(region, "metrics", {}) or {}).get(
            "sample_region_signature", "")) != signatures["sample_region"]:
        region = None
    point = {}
    if material and not no_illumination:
        from temsim.specimen.elastic_transport import incident_rays_from_simulation
        if not bool(state.ac_deflector.enabled and state.ac_deflector.scan_enabled):
            centre = incident_rays_from_simulation(state, simulation).original_centroid_nm
            point = {"point_x_nm": float(centre[0]), "point_y_nm": float(centre[1])}
        observables = {O.ELASTIC_TRANSPORT, O.STOCHASTIC_INELASTIC}
        eds_requested = workflow in {"eds", "sample_region"}
        kwargs = {}
        if eds_requested:
            from temsim.component_keys import EDS_DETECTOR_SYSTEM
            from temsim.detector.eds_geometry import EDSDetectorArrayGeometry
            geometry = EDSDetectorArrayGeometry.from_part_data(
                state._resolved_assembly.part(EDS_DETECTOR_SYSTEM).data)
            kwargs["detector_geometry"] = geometry
            observables.add(O.CHARACTERISTIC_X_RAY)
        before = interactions
        interactions = p.run_specimen_interactions(state, simulation,
            p.SpecimenInteractionRequest(observables=frozenset(observables), **point),
            existing_result=interactions, progress_callback=progress.update, **kwargs)
        if getattr(before, "elastic_transport", None) is not interactions.elastic_transport:
            # The interaction engine also validates the actual point request.
            # A replaced local state cannot keep the prior executed exit just
            # because the broad instrument parameter signature still matches.
            specimen_exit = region = stem = None
        for key, attr in (("elastic", "elastic_transport"), ("eds", "eds_spectrum")):
            if key == "eds" and not eds_requested:
                continue
            (reused if getattr(before, attr, None) is getattr(interactions, attr, None)
             and getattr(interactions, attr, None) is not None else calculated).add(key)
        if specimen_exit is None:
            from temsim.physics.particle_sections import section_limits
            distance = float(getattr(state.sample, "sample_region_downstream_distance_um", 0.))
            save_planes = ((float(state.sample.z_mm) + distance * 1e-3,) if distance > 0 else ())
            specimen_exit = p.build_geometric_specimen_exit(state, simulation,
                interactions.elastic_transport, interactions.inelastic_distribution,
                save_z_mm=save_planes, stop_z_mm=section_limits(state)[1],
                dependency_signature=signatures["sample_downstream"], progress_callback=progress.update)
            calculated.add("sample_downstream")
        else:
            reused.add("sample_downstream")
    elif workflow != "rays" and no_illumination:
        interactions = specimen_exit = region = stem = wave = None
    if region is not None:
        region = p._rebind_reused_sample_region(region, interactions, wave, signatures)
        if region is not None and specimen_exit is not None:
            region = p.bind_sample_region_downstream(region, specimen_exit, interactions,
                expected_signature=signatures["sample_downstream"], wave_imaging=wave)
        else:
            region = None
    stage(2, "Requested optional readouts")

    # The reference is still kept separately from the executed scattered rays.
    # Detector readout must not confuse the reference's optical-tuning tag with
    # the present physical calculation, including an explicitly empty beam.
    if workflow != "rays":
        simulation.metrics.update(optical_tuning=False, workflow=workflow,
            sample_scattering_applied=bool(material and not no_illumination),
            sample_scattering_model=("finite_geometry_elastic_x_inelastic_tensor_product"
                                     if material else "vacuum"))
    else:
        simulation.metrics.update(optical_tuning=True, workflow=workflow,
            sample_scattering_applied=False,
            sample_scattering_model="omitted_for_optical_reference")
    result = p.CalculationResult(simulation=simulation, energy_filter=None,
        state_snapshot=state, layout=layout, assembly=state._resolved_assembly,
        specimen_interactions=interactions, specimen_exit=specimen_exit,
        wave_imaging=wave, stem_scan=stem, scan_geometry=scan_geometry,
        scan_ray_paths=scan_paths, sample_region=region, signatures=signatures,
        external_inputs=external, workflow=workflow)
    if workflow == "sample_region":
        if not material or no_illumination:
            raise ValueError("Sample-region calculation requires an illuminated, inserted specimen.")
        if region is None:
            from temsim.specimen.sample_region import simulate_sample_region
            sample = state.sample
            region = simulate_sample_region(state, result, geometry,
                upstream_distance_um=sample.sample_region_upstream_distance_um,
                downstream_distance_um=sample.sample_region_downstream_distance_um,
                photon_path_count=sample.sample_region_photon_path_count,
                secondary_path_count=sample.sample_region_secondary_path_count,
                seed=sample.sample_region_seed, existing_interactions=interactions)
            calculated.add("sample_region")
        else:
            region = p.bind_sample_region_downstream(region, specimen_exit, interactions,
                expected_signature=signatures["sample_downstream"], wave_imaging=wave)
            reused.add("sample_region")
        result.sample_region = region
        result.specimen_interactions, result.specimen_exit = region.interactions, region.specimen_exit
        interactions, specimen_exit = result.specimen_interactions, result.specimen_exit
    if workflow == "imaging":
        # admit_workflow enforces the paused physical tip-to-wave-chain gate.
        # No particle-to-atomic-image shortcut is introduced here.
        if p.tem_wave_imaging_enabled(state) and not no_illumination and wave is None:
            interactions = p.run_specimen_interactions(state, simulation,
                p.SpecimenInteractionRequest.tem_wave(), existing_result=interactions,
                progress_callback=progress.update)
            result.specimen_interactions = interactions
            result.wave_imaging = interactions.wave_imaging
            calculated.add("wave")
    if workflow == "stem":
        raster = bool(state.ac_deflector.enabled and state.ac_deflector.scan_enabled
                      and getattr(state.sample, "stem_image_enabled", True))
        if raster:
            if scan_geometry is None:
                result.scan_geometry = p.calculate_scan_geometry(state)
                calculated.add("scan_geometry")
            if scan_paths is None:
                result.scan_ray_paths = p.calculate_scan_ray_paths(state, simulation)
                calculated.add("scan_ray_paths")
            if stem is None:
                result.stem_scan = p.calculate_stem_scan_frame(state, simulation,
                    specimen_interactions=interactions, geometric_specimen_exit=specimen_exit,
                    geometric_specimen_exit_signature=signatures["sample_downstream"],
                    progress_callback=progress.update, diffraction_sink=diffraction_sink)
                calculated.add("stem")
    stage(3, "Assembled energy filter and detector readout")

    # A physically traversed assembled filter is not an optional readout.
    if bool(state.energy_filter.enabled):
        source = p._energy_filter_source_view(simulation,
            None if workflow == "rays" else specimen_exit, signatures,
            no_illumination=no_illumination)
        old_filter = getattr(existing_result, "energy_filter", None)
        if ("energy_filter" in reusable and old_filter is not None
                and getattr(old_filter, "entrance_provenance", None)
                    == source.metrics["energy_filter_entrance_provenance"]
                and getattr(old_filter, "entrance_dependency_signature", None)
                    == source.metrics["energy_filter_entrance_dependency_signature"]):
            result.energy_filter = old_filter
            reused.add("energy_filter")
        else:
            result.energy_filter = p.simulate_energy_filter(state, source,
                inelastic_distribution=(None if workflow == "rays" else
                    getattr(interactions, "inelastic_distribution", None)))
            calculated.add("energy_filter")
    if workflow == "stem":
        from temsim.detector.particle_readout import measure_particle_detectors
        result.particle_signals = measure_particle_detectors(result)
        calculated.add("particle_signals")
    elif ("stem" in reusable and "column" in reusable
          and "energy_filter" in reusable):
        result.particle_signals = getattr(existing_result, "particle_signals", None)
    stage(4, "Diagnostics and continuation checkpoint")
    result.lens_crossovers = tuple(p.detect_all_lens_crossovers(
        [simulation.incident, *simulation.branches.values()], state.lenses))
    result.aperture_stops = p.aperture_stop_records(state)
    calculated.add("diagnostics")
    for name, value in (("wave", result.wave_imaging), ("stem", result.stem_scan),
                        ("scan_geometry", result.scan_geometry), ("scan_ray_paths", result.scan_ray_paths),
                        ("sample_region", result.sample_region),
                        ("elastic", getattr(result.specimen_interactions, "elastic_transport", None)),
                        ("eds", getattr(result.specimen_interactions, "eds_spectrum", None)),
                        ("sample_downstream", result.specimen_exit)):
        if value is not None and name in reusable and name not in calculated:
            reused.add(name)
    from temsim.physics.completed_particle_section import capture_completed_particle_section
    capture_completed_particle_section(result)
    p.assert_external_input_inventory_unchanged(state, external)
    result.calculated_products, result.reused_products = frozenset(calculated), frozenset(reused)
    stage(5, "Complete")
    result.performance = {"workflow": workflow, "pipeline_seconds": perf_counter()-started,
        "stages": tuple(progress.timings), "timing_scope": "Explicit requested workflow; retained products are not recalculated"}
    return result
