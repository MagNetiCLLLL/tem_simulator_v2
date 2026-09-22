"""Retain executed full-pipeline particle states for section continuation.

No transport is performed here. A display trajectory alone is never promoted
to a restart state: full-precision solver checkpoints and source identity are
required. Historical wave products continue through their existing path.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np


def incident_section_segment(state, simulation):
    from temsim.physics.particle_sections import ParticleSectionSegment, _digest_arrays
    gun = simulation.gun_trace
    emitted = gun.exit_bundle
    branch = simulation.incident
    plan, cp = simulation.incident_plan, simulation.incident_checkpoints
    if plan is None or cp is None or cp.z_mm[-1] != branch.z[-1]:
        raise ValueError("Completed incident transport lacks an exact terminal checkpoint")
    initial = tuple(np.asarray(getattr(emitted, name)) for name in
                    ("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s"))
    identity = (branch.source_ray_id, branch.source_azimuth_rad)
    dependency = _digest_arrays(*initial, emitted.energy_offset_ev, emitted.weight,
        emitted.alive, gun.blocked_z_mm, *identity,
        np.asarray(gun.blocked_key, dtype=str), np.asarray(""))
    keep = np.asarray(branch.z) >= plan.z_mm[0]
    phase = {}
    for name, checkpoint_name in (("x", "x_m"), ("tx", "tx_rad"),
                                  ("y", "y_m"), ("ty", "ty_rad"),
                                  ("flight_time_s", "flight_time_s")):
        values = np.asarray(getattr(branch, name)[keep], dtype=np.float64).copy()
        values[-1] = getattr(cp, checkpoint_name)[-1]
        phase[name] = values
    column = replace(branch, z=np.asarray(branch.z)[keep], **phase)
    return ParticleSectionSegment("incident", column, plan, cp, dependency)


def capture_completed_particle_section(result):
    """Attach a restartable cache to an actual completed classical pipeline.

    Returns False for legacy results without solver state. Never constructs a
    replacement source or discards the full calculation's original products.
    """
    from temsim.physics.simulation import Simulation
    from temsim.physics.particle_sections import (
        ParticleSectionCheckpoint, MaterialSectionCache, SECTION_SCHEMA,
        MATERIAL_SECTION_SCHEMA, gun_dependency_signature,
        validate_section_checkpoint, _digest_arrays,
    )
    from temsim.calculation_cache import calculation_signatures
    from temsim.calculation_manifest import solver_source_identity
    simulation, state = result.simulation, result.state_snapshot
    if (not isinstance(simulation, Simulation) or simulation.gun_trace is None
            or simulation.incident_plan is None or simulation.incident_checkpoints is None
            or getattr(state.sample, "wave_enabled", False)
            or getattr(state.sample, "stem_wave_enabled", False)):
        return False
    signature = gun_dependency_signature(state)
    if not signature:
        return False
    incident = incident_section_segment(state, simulation)
    post = tuple(getattr(simulation, "completed_post_sections", ()))
    checkpoint = ParticleSectionCheckpoint(SECTION_SCHEMA, signature,
        simulation.gun_trace, (incident, *post))
    validate_section_checkpoint(checkpoint)
    simulation.section_checkpoint = checkpoint
    target = max([float(incident.branch.z[-1]), *(float(branch.z[-1])
        for branch in simulation.branches.values())])
    resume_through = float(post[-1].branch.z[-1] if post else incident.branch.z[-1])
    interactions, specimen_exit = result.specimen_interactions, result.specimen_exit
    if (interactions is not None and interactions.elastic_transport is not None
            and interactions.inelastic_distribution is not None and specimen_exit is not None):
        from temsim.specimen.elastic_transport import incident_rays_from_simulation
        original = simulation.incident
        digest = _digest_arrays(*(getattr(original, key)[-1] for key in
            ("x", "tx", "y", "ty", "flight_time_s")), original.alive,
            original.blocked_z, original.energy_offset_ev, original.ray_weight,
            original.source_ray_id)
        point = tuple(float(v) for v in interactions.incident_bundle.target_centroid_nm)
        signatures = dict(calculation_signatures(state))
        signatures["section_material"] = hashlib.sha256(repr((MATERIAL_SECTION_SCHEMA,
            "specimen", signatures["elastic"], solver_source_identity())).encode()).hexdigest()
        simulation.material_section_cache = MaterialSectionCache(signatures, digest,
            point, interactions.elastic_transport, interactions.inelastic_distribution,
            specimen_exit, target, eds_spectrum=interactions.eds_spectrum)
        if getattr(specimen_exit, "segments", ()):
            resume_through = min(float(s.branch.z[-1]) for s in specimen_exit.segments)
    simulation.metrics.update(particle_section=True, particle_tuning=False,
        section_physics_scope="classical_particles_with_specimen_interactions",
        section_target_z_mm=float(target), section_component_keys=(),
        section_full_path=True, section_schema=SECTION_SCHEMA,
        section_resumable_through_z_mm=resume_through,
        section_resume_z_mm=float(simulation.metrics.get("column_segment_cache", {}).get(
            "resume_z_mm", state.electron_gun.exit_plane_z_mm)),
        tuning_quality="High accuracy")
    return True


def _compatible_material_eds_product(cache, signatures, state, key):
    """Admit an EDS product only after the caller validates its material.

    The selected key admits either the identical readout or a compatible
    executed response. Both retain atomic data, material and collection
    dependencies plus the actual point-call guard. The response caller then
    rebuilds dose/readout; this check alone never relabels an old spectrum.
    The archive codec restores the shared executed elastic object explicitly.
    """
    from temsim.detector.eds_signal import EDSSpectrum, eds_point_request_inputs
    spectrum = cache.eds_spectrum
    if (isinstance(spectrum, EDSSpectrum)
            and spectrum.elastic_transport is cache.elastic_transport
            and cache.signatures.get(key) == signatures.get(key)
            and signatures.get(key)):
        from temsim.component_keys import EDS_DETECTOR_SYSTEM
        from temsim.detector.eds_geometry import EDSDetectorArrayGeometry
        geometry = EDSDetectorArrayGeometry.from_part_data(
            state._resolved_assembly.part(EDS_DETECTOR_SYSTEM).data)
        if spectrum.metrics.get("point_request_inputs") == eds_point_request_inputs(geometry):
            return spectrum
    return None


def compatible_material_eds(cache, signatures, state):
    """Return an identical completed readout after material-cache validation."""
    return _compatible_material_eds_product(cache, signatures, state, "eds")


def replay_material_eds(cache, signatures, state, bundle, *, progress_callback=None):
    """Reweight a verified executed response without changing electron paths."""
    spectrum = _compatible_material_eds_product(cache, signatures, state, "eds_response")
    if spectrum is None or getattr(spectrum, "response_rates", None) is None:
        return None
    from temsim.detector.eds_response import replay_eds_response
    from temsim.detector.eds_signal import default_eds_dwell_time_s, default_eds_incident_electrons
    from temsim.physics.optical_tuning import check_tuning_cancelled
    dwell = default_eds_dwell_time_s(state)
    source_electrons = default_eds_incident_electrons(state, dwell)
    arrivals = source_electrons * bundle.surviving_fraction
    sample = state.sample
    def progress(done, total, label):
        check_tuning_cancelled(state)
        if progress_callback is not None:
            progress_callback(done, total, label)
    result = replay_eds_response(spectrum, incident_electrons=arrivals,
        energy_min_ev=0., energy_max_ev=float(sample.eds_spectrum_max_energy_ev),
        energy_bin_width_ev=float(sample.eds_spectrum_bin_width_ev),
        energy_resolution_fwhm_ev=float(sample.eds_energy_resolution_fwhm_ev),
        poisson_enabled=bool(sample.eds_poisson_enabled), poisson_seed=int(sample.eds_poisson_seed),
        progress_callback=progress)
    if result is None:
        return None
    # Dose metadata describes this readout. The kernel, relative material
    # weights and its physically executed trajectories are unchanged.
    metrics = dict(result.metrics)
    metrics.update(dwell_time_s=dwell, incident_electron_reference="emitted source current",
                   source_electrons_before_column_losses=source_electrons,
                   electrons_reaching_sample_plane=arrivals,
                   sample_x_nm=float(bundle.original_centroid_nm[0]),
                   sample_y_nm=float(bundle.original_centroid_nm[1]),
                   eds_response_replayed=True)
    return replace(result, metrics=metrics)


def restore_material_interactions(state, simulation, previous_result, *, progress_callback=None,
                                  replay_response=True):
    """Validate executed material physics against the freshly resolved beam.

    EDS additionally requires its independent exact dependency signature. Wave
    readouts are not restored here. A scan has separate placement semantics.
    """
    from temsim.physics.particle_sections import MaterialSectionCache, MATERIAL_SECTION_SCHEMA, _digest_arrays
    from temsim.calculation_cache import calculation_signatures
    from temsim.calculation_manifest import solver_source_identity
    from temsim.specimen.elastic_transport import incident_rays_from_simulation
    from temsim.specimen.interaction_types import (
        SpecimenInteractionResult, SpecimenInteractionRequest, SpecimenObservable,
    )
    from temsim.specimen.scene import SpecimenScene
    cache = getattr(getattr(previous_result, "simulation", None), "material_section_cache", None)
    if (not isinstance(cache, MaterialSectionCache) or cache.schema != MATERIAL_SECTION_SCHEMA
            or bool(state.ac_deflector.enabled and state.ac_deflector.scan_enabled)):
        return None, None
    signatures = calculation_signatures(state)
    expected = hashlib.sha256(repr((MATERIAL_SECTION_SCHEMA, "specimen",
        signatures["elastic"], solver_source_identity())).encode()).hexdigest()
    if cache.signatures.get("section_material") != expected:
        return None, None
    incident = simulation.incident
    digest = _digest_arrays(*(getattr(incident, name)[-1] for name in
        ("x", "tx", "y", "ty", "flight_time_s")), incident.alive,
        incident.blocked_z, incident.energy_offset_ev, incident.ray_weight, incident.source_ray_id)
    if digest != cache.incident_digest:
        return None, None
    bundle = incident_rays_from_simulation(state, simulation)
    point = tuple(float(v) for v in bundle.original_centroid_nm)
    if point != cache.point_xy_nm:
        return None, None
    observables = {SpecimenObservable.ELASTIC_TRANSPORT,
                   SpecimenObservable.STOCHASTIC_INELASTIC}
    spectrum = (compatible_material_eds(cache, signatures, state)
                if bool(getattr(state.sample, "eds_enabled", False)) else None)
    response_replayed = False
    if replay_response and spectrum is None and bool(getattr(state.sample, "eds_enabled", False)):
        spectrum = replay_material_eds(cache, signatures, state, bundle,
                                       progress_callback=progress_callback)
        response_replayed = spectrum is not None
    if spectrum is not None:
        observables.add(SpecimenObservable.CHARACTERISTIC_X_RAY)
    request = SpecimenInteractionRequest(observables=frozenset(observables),
        point_x_nm=point[0], point_y_nm=point[1])
    restored = SpecimenInteractionResult(request=request,
        completed_observables=request.observables, scene=SpecimenScene.from_state(state),
        incident_bundle=bundle, elastic_transport=cache.elastic_transport,
        inelastic_distribution=cache.inelastic_distribution,
        eds_spectrum=spectrum,
        metrics={"dependency_signatures": signatures, "eds_response_replayed": response_replayed})
    return restored, cache
