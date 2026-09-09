from temsim.physics.crossovers import first_crossover_after_lens

from temsim.physics.column_wall import clip_column_wall

from temsim.physics.aperture_clipping import clip_segment as _clip_aperture_segment

from dataclasses import dataclass

import numpy as np, math

from temsim.physics.chromatic import (
    configured_objective_chromatic_focal_mm,
    objective_chromatic_kick_from_state,
)

from temsim.physics.core import (
    FIELD_SIGMA_CUTOFF,
    PropagationCheckpoints,
    build_propagation_plan,
    electron,
    execute_propagation_plan,
    fields,
    propagate,
    propagation_plan_common_prefix_nodes,
)
from temsim.physics.first_order import (
    linear_map_properties,
    trace_transverse_transfer,
)

from temsim.physics.beam_waist import detect_beam_waist
from temsim.physics.beam_statistics import branch_sample_statistics
from temsim.physics.beam_current import (
    column_current_limit_percent,
    effective_source_current_pa,
)


from temsim.physics.corrector_crossovers import detect_corrector_crossovers

from temsim.physics.recording_stop import (
    tem_projection_reference_plane_z,
    determine_tem_stop_z,
)
from temsim.physics.recording_clipping import clip_recording_planes
from temsim.component_keys import CONDENSER_LENS_2, CONDENSER_LENS_3

RAY_INTERACTION_COLOURS = {
    # Colour carries interaction semantics.  Per-ray convergence is encoded
    # later by changing only the brightness of this base colour.
    'incident': (0.22, 0.74, 0.97),
    'vacuum': (0.58, 0.64, 0.72),
    'optical_reference': (0.58, 0.74, 0.84),
    'real_sample_reference': (0.58, 0.64, 0.72),
    'real_zero_loss': (0.72, 0.76, 0.82),
    'real_plasmon': (0.16, 0.82, 0.96),
    'real_ionisation': (0.98, 0.35, 0.24),
    'real_other_inelastic': (0.96, 0.73, 0.12),
    'real_plural_inelastic': (0.78, 0.36, 0.96),
    'virtual_interactions_disabled': (0.58, 0.64, 0.72),
    'transmitted': (0.29, 0.87, 0.50),
    'diffraction_spots': (0.66, 0.55, 0.98),
    'diffuse_ring': (0.96, 0.62, 0.04),
    'gaussian_diffuse': (0.98, 0.80, 0.08),
    'arbitrary_angular': (0.18, 0.83, 0.75),
    'user_screened_power_law': (0.96, 0.45, 0.71),
    'physical_rutherford': (0.98, 0.31, 0.38),
    'sample_region_primary': (0.18, 0.88, 0.72),
    'sample_region_elastic': (1.00, 0.32, 0.48),
    'unknown': (0.89, 0.91, 0.94),
}

# Small ray bundles benefit greatly from tracing all interaction quadrature
# branches together.  Large production bundles are batched to keep the
# (axial steps x rays) momentum/Larmor work arrays within a bounded peak.
MAX_VECTORIZED_POST_RAYS = 4096
INCIDENT_CHECKPOINT_SPACING_MM = 5.0
INCIDENT_CHECKPOINT_MEMORY_BUDGET_BYTES = 512 * 1024 * 1024


def _canonical_interaction_kind(kind):
    aliases = {
        'diffraction_spot': 'diffraction_spots',
        'isotropic_ring': 'diffuse_ring',
    }
    value = str(kind or 'unknown').strip().lower()
    return aliases.get(value, value)


def _interaction_colour(kind):
    return RAY_INTERACTION_COLOURS.get(
        _canonical_interaction_kind(kind),
        RAY_INTERACTION_COLOURS['unknown'],
    )


def _sample_to_stop_larmor_rotation_rad(state, stop_z_mm):
    from temsim.optics.equivalent_image_lenses import (
        equivalent_image_events,
        equivalent_image_lenses_enabled,
    )

    sample_z_mm = float(state.sample.z_mm)
    stop_z_mm = float(stop_z_mm)
    if equivalent_image_lenses_enabled(state):
        return float(sum(
            event.rotation_rad
            for event in equivalent_image_events(
                state, sample_z_mm, stop_z_mm
            )
        ))
    count = max(
        2,
        int(math.ceil(
            (stop_z_mm - sample_z_mm)
            / max(min(float(state.step_mm), 0.25), 0.01)
        )) + 1,
    )
    z_mm = np.linspace(sample_z_mm, stop_z_mm, count)
    magnetic_t = fields(z_mm, state)[0]
    charge_c, momentum, _ = electron(state)
    return float(
        -charge_c
        * np.trapezoid(magnetic_t, z_mm * 1.0e-3)
        / (2.0 * momentum)
    )

@dataclass

class Branch:
    name:str; colour:tuple; z:np.ndarray; x:np.ndarray; y:np.ndarray; tx:np.ndarray; ty:np.ndarray; alive:np.ndarray; blocked_z:np.ndarray; blocked_key:list; weight:float; energy_offset_ev:np.ndarray; ray_weight:np.ndarray|None=None; interaction_kind:str='unknown'; interaction_kick_x_rad:np.ndarray|None=None; interaction_kick_y_rad:np.ndarray|None=None
    # Display lineage only: gun ray IDs and source-position azimuth in radians.
    # Neither value is an instantaneous velocity angle or a physical weight.
    source_ray_id: np.ndarray | None = None
    source_azimuth_rad: np.ndarray | None = None

@dataclass

class Simulation:

    incident:Branch; branches:dict; metrics:dict; gun_waist:dict|None=None; c2c3_crossover:dict|None=None; corrector_crossovers:list|None=None; gun_trace:object|None=None; sample_to_analysis_transfer:object|None=None; optical_transfers:tuple=(); real_interactions:object|None=None; incident_plan:object|None=None; incident_checkpoints:PropagationCheckpoints|None=None


def _column_checkpoint_planes(start_z_mm, stop_z_mm, ray_count):
    start = float(start_z_mm)
    stop = float(stop_z_mm)
    if stop <= start:
        return ()
    desired_count = int(
        math.floor((stop - start) / INCIDENT_CHECKPOINT_SPACING_MM)
    )
    bytes_per_checkpoint = 4 * np.dtype(np.float64).itemsize * max(
        int(ray_count), 1
    )
    maximum_count = (
        INCIDENT_CHECKPOINT_MEMORY_BUDGET_BYTES // bytes_per_checkpoint
    )
    if maximum_count <= 0:
        return ()
    stride = max(1, int(math.ceil(desired_count / maximum_count)))
    spacing = INCIDENT_CHECKPOINT_SPACING_MM * stride
    count = int(math.floor((stop - start) / spacing))
    return tuple(
        start + index * spacing
        for index in range(1, count + 1)
        if start + index * spacing < stop
    )


def _gun_traces_match(previous, current):
    if previous is None or current is None:
        return False
    scalar_names = (
        "emitted_current_a", "dpa_transmitted_current_a",
        "c1_transmitted_current_a", "monochromator_transmitted_current_a",
        "output_energy_fwhm_ev", "slit_dispersion_um_per_ev",
    )
    if any(
        getattr(previous, name, None) != getattr(current, name, None)
        for name in scalar_names
    ):
        return False
    array_names = ("z_mm", "x_m", "y_m", "tx_rad", "ty_rad", "blocked_z_mm")
    if any(
        not np.array_equal(
            np.asarray(getattr(previous, name)),
            np.asarray(getattr(current, name)),
            equal_nan=True,
        )
        for name in array_names
    ):
        return False
    if tuple(previous.blocked_key) != tuple(current.blocked_key):
        return False
    old_exit = previous.exit_bundle
    new_exit = current.exit_bundle
    return all(
        np.array_equal(
            np.asarray(getattr(old_exit, name)),
            np.asarray(getattr(new_exit, name)),
            equal_nan=True,
        )
        for name in (
            "x_m", "y_m", "tx_rad", "ty_rad", "energy_offset_ev",
            "weight", "ray_id", "alive",
        )
    )


def _merge_checkpoints(previous, suffix, resume_z_mm):
    if previous is None:
        return suffix
    keep = np.asarray(previous.z_mm) < float(resume_z_mm)
    arrays = {}
    for name in ("z_mm", "x_m", "tx_rad", "y_m", "ty_rad"):
        old = np.asarray(getattr(previous, name))[keep]
        new = np.asarray(getattr(suffix, name))
        merged = np.ascontiguousarray(np.concatenate((old, new)), dtype=np.float64)
        merged.setflags(write=False)
        arrays[name] = merged
    return PropagationCheckpoints(**arrays)


def run(s, *, resolved_layout=None, existing_simulation=None, optical_only=False):
    # The low-level entry point is also public and is used directly by tests
    # and scripts, so it must enforce the same TOML-owned geometry contract as
    # the application-facing calculation pipeline.
    if resolved_layout is None:
        from temsim.column.state_layout import apply_physical_layout_to_state
        resolved_layout = apply_physical_layout_to_state(s)

    ac_scan = getattr(s, "ac_deflector", None)
    if (
        not optical_only and ac_scan is not None
        and bool(getattr(ac_scan, "enabled", False))
        and bool(getattr(ac_scan, "scan_enabled", False))
    ):
        from temsim.physics.scan_geometry import calibrate_scan_system

        calibrate_scan_system(s)

    gun=s.electron_gun.validate()
    gun_trace=gun.trace_to_exit()
    emitted=gun_trace.exit_bundle
    x,y=emitted.x_m,emitted.y_m
    tx,ty=emitted.tx_rad,emitted.ty_rad
    dE=emitted.energy_offset_ev
    n=x.size
    pre_events=[]

    post_events=[]

    for d in s.deflectors:

        if not d.enabled:

            continue

        if hasattr(d, "kick_events"):
            try:
                pair = d.kick_events(
                    time_s=float(getattr(s, "simulation_time_s", 0.0))
                )
            except TypeError:
                pair = d.kick_events()
        else:
            pair=[(d.upper_z_mm,d.upper_x_mrad*1e-3,d.upper_y_mrad*1e-3),(d.lower_z_mm,d.lower_x_mrad*1e-3,d.lower_y_mrad*1e-3)]

        for event in pair:

            (pre_events if event[0] <= s.sample.z_mm else post_events).append(event)

    for component in getattr(s, "corrector_elements", []):
        if not getattr(component, "enabled", False):
            continue
        if not hasattr(component, "kick_events"):
            continue
        try:
            events = component.kick_events(
                time_s=float(getattr(s, "simulation_time_s", 0.0))
            )
        except TypeError:
            events = component.kick_events()
        for event in events:
            (
                pre_events
                if event[0] <= s.sample.z_mm
                else post_events
            ).append(event)

    checkpoint_planes = _column_checkpoint_planes(
        gun.exit_plane_z_mm, s.sample.z_mm, n
    )
    incident_plan = build_propagation_plan(
        s, gun.exit_plane_z_mm, s.sample.z_mm, pre_events,
        checkpoint_z_mm=checkpoint_planes,
    )
    cache_mode = "none"
    resume_z_mm = float(gun.exit_plane_z_mm)
    reused_prefix_rows = 0
    reused_plan_nodes = 0
    incident_checkpoints = None
    previous_plan = getattr(existing_simulation, "incident_plan", None)
    previous_checkpoints = getattr(
        existing_simulation, "incident_checkpoints", None
    )
    source_matches = _gun_traces_match(
        getattr(existing_simulation, "gun_trace", None), gun_trace
    )

    if (
        source_matches
        and previous_plan is not None
        and previous_checkpoints is not None
        and previous_plan.signature == incident_plan.signature
    ):
        # Geometry coordinates are independent of aperture/wall clipping.  A
        # new Branch and new stop arrays are still built below.
        previous_incident = existing_simulation.incident
        # Results are isolated snapshots.  Coordinate copies prevent any
        # later display/diagnostic consumer of the replacement result from
        # mutating the previously completed high-accuracy cache entry.
        z = np.asarray(previous_incident.z).copy()
        X = np.asarray(previous_incident.x).copy()
        TX = np.asarray(previous_incident.tx).copy()
        Y = np.asarray(previous_incident.y).copy()
        TY = np.asarray(previous_incident.ty).copy()
        incident_checkpoints = previous_checkpoints
        cache_mode = "full_incident"
        resume_z_mm = float(s.sample.z_mm)
        reused_prefix_rows = int(len(z))
        reused_plan_nodes = int(len(incident_plan.z_mm))
    else:
        common_nodes = (
            propagation_plan_common_prefix_nodes(
                previous_plan, incident_plan
            )
            if source_matches else 0
        )
        resume = None
        if common_nodes > 0 and previous_checkpoints is not None:
            current_checkpoint_indices = np.asarray(
                incident_plan.checkpoint_index, dtype=np.int64
            )
            current_checkpoint_z = np.asarray(incident_plan.z_mm)[
                current_checkpoint_indices
            ]
            for old_checkpoint_row in range(
                len(previous_checkpoints.z_mm) - 1, -1, -1
            ):
                candidate_z = float(
                    previous_checkpoints.z_mm[old_checkpoint_row]
                )
                matches = np.flatnonzero(current_checkpoint_z == candidate_z)
                if not matches.size:
                    continue
                current_row = int(matches[-1])
                current_index = int(current_checkpoint_indices[current_row])
                if current_index < common_nodes:
                    resume = (
                        old_checkpoint_row, current_index, candidate_z
                    )
                    break

        if resume is None:
            column_result = execute_propagation_plan(
                s, incident_plan, x, tx, y, ty, dE
            )
            (
                z_column, X_column, TX_column, Y_column, TY_column,
                incident_checkpoints,
            ) = column_result
            z=np.r_[gun_trace.z_mm,z_column[1:]]
            X=np.vstack((gun_trace.x_m,X_column[1:]))
            TX=np.vstack((gun_trace.tx_rad,TX_column[1:]))
            Y=np.vstack((gun_trace.y_m,Y_column[1:]))
            TY=np.vstack((gun_trace.ty_rad,TY_column[1:]))
        else:
            old_row, start_index, resume_z_mm = resume
            suffix_result = execute_propagation_plan(
                s, incident_plan,
                np.asarray(previous_checkpoints.x_m[old_row]).copy(),
                np.asarray(previous_checkpoints.tx_rad[old_row]).copy(),
                np.asarray(previous_checkpoints.y_m[old_row]).copy(),
                np.asarray(previous_checkpoints.ty_rad[old_row]).copy(),
                dE, start_index=start_index,
                include_initial_plane_kicks=False,
            )
            (
                z_suffix, X_suffix, TX_suffix, Y_suffix, TY_suffix,
                suffix_checkpoints,
            ) = suffix_result
            previous_incident = existing_simulation.incident
            keep = np.asarray(previous_incident.z) < resume_z_mm
            z = np.r_[np.asarray(previous_incident.z)[keep], z_suffix]
            X = np.vstack((np.asarray(previous_incident.x)[keep], X_suffix))
            TX = np.vstack((np.asarray(previous_incident.tx)[keep], TX_suffix))
            Y = np.vstack((np.asarray(previous_incident.y)[keep], Y_suffix))
            TY = np.vstack((np.asarray(previous_incident.ty)[keep], TY_suffix))
            incident_checkpoints = _merge_checkpoints(
                previous_checkpoints, suffix_checkpoints, resume_z_mm
            )
            cache_mode = "checkpoint"
            reused_prefix_rows = int(np.count_nonzero(keep))
            reused_plan_nodes = int(start_index + 1)
    alive=emitted.alive.copy()
    blocked=gun_trace.blocked_z_mm.copy()
    keys=list(gun_trace.blocked_key)
    alive,blocked,keys=_clip_aperture_segment(
        s,z,X,Y,alive,blocked,keys
    )

    alive,blocked,keys=clip_column_wall(s,z,X,Y,alive,blocked,keys)

    incident_kind = 'incident'
    incident=Branch(
        'incident',_interaction_colour(incident_kind),z,X,Y,TX,TY,alive,
        blocked,keys,1.,dE,emitted.weight,
        interaction_kind=incident_kind,
    )
    from temsim.physics.ray_identity import source_identity

    incident.source_ray_id, incident.source_azimuth_rad = source_identity(
        incident, gun_trace
    )
    retained_checkpoint_count = (
        int(len(incident_checkpoints.z_mm))
        if incident_checkpoints is not None else 0
    )
    retained_checkpoint_bytes = (
        sum(
            int(np.asarray(getattr(incident_checkpoints, name)).nbytes)
            for name in ("x_m", "tx_rad", "y_m", "ty_rad")
        )
        if incident_checkpoints is not None else 0
    )
    segment_cache_metrics = {
        "mode": cache_mode,
        "hit": cache_mode != "none",
        "resume_z_mm": float(resume_z_mm),
        "reused_prefix_rows": int(reused_prefix_rows),
        "recomputed_history_rows": int(len(z) - reused_prefix_rows),
        "reused_integration_nodes": int(reused_plan_nodes),
        "recomputed_integration_nodes": int(
            len(incident_plan.z_mm) - reused_plan_nodes
        ),
        "checkpoint_spacing_mm": (
            float(checkpoint_planes[0]) - float(gun.exit_plane_z_mm)
            if checkpoint_planes else math.inf
        ),
        "checkpoint_count": retained_checkpoint_count,
        "checkpoint_retained_bytes": retained_checkpoint_bytes,
        "checkpoint_memory_budget_bytes": (
            INCIDENT_CHECKPOINT_MEMORY_BUDGET_BYTES
        ),
        "field_sigma_cutoff": FIELD_SIGMA_CUTOFF,
    }

    gun_field_end,gun_diagnostic_end=gun.diagnostic_waist_region_mm
    gun_waist=detect_beam_waist(
        incident,gun_field_end,gun_diagnostic_end
    )
    s.last_gun_waist_mm=float('nan') if gun_waist is None else gun_waist['z_mm']

    sample_inserted = bool(getattr(s.sample, 'inserted', True))
    specimen_mode = str(getattr(s.sample, 'specimen_mode', 'atomic')).strip().lower()
    if specimen_mode not in {'atomic', 'reference'}:
        raise ValueError("Sample specimen mode must be 'atomic' or 'reference'.")
    _,_,lam=electron(s)

    branches={}
    chromatic_focal_mm = configured_objective_chromatic_focal_mm(s)

    def branch_chromatic_kick(energy_offset_ev):
        return objective_chromatic_kick_from_state(
            s,
            X[-1],
            Y[-1],
            energy_offset_ev,
            resolved_focal_mm=chromatic_focal_mm,
        )

    real_branch_weights_are_absolute = False
    real_interactions = None
    from temsim.specimen.source import specimen_is_vacuum

    sample_is_vacuum = specimen_is_vacuum(s.sample)
    if optical_only:
        branch_specs = [('000', 0.0, 0.0, 1.0, 'optical_reference', 0.0)]
        scattering_model = 'omitted_for_optical_tuning'
    elif sample_is_vacuum:
        branch_specs = [('000', 0.0, 0.0, 1.0, 'vacuum', 0.0)]
        scattering_model = 'user_selected_vacuum_reference_plane'
    else:
        # Coherent elastic diffraction remains exclusively in multislice.
        # These branches are instead a material-derived, probability-
        # conserving quadrature of real stochastic energy-loss events.
        from temsim.specimen.inelastic import (
            real_inelastic_distribution,
            real_inelastic_ray_branches,
        )

        real_interactions = real_inelastic_distribution(s)
        real_branches = real_inelastic_ray_branches(
            real_interactions, ray_count=n
        )
        branch_specs = [
            (
                branch.name,
                branch.kick_x_rad,
                branch.kick_y_rad,
                branch.probability,
                branch.interaction_kind,
                branch.energy_loss_ev,
            )
            for branch in real_branches
        ]
        real_branch_weights_are_absolute = True
        scattering_model = 'real_material_inelastic_poisson_plus_elastic_wave'
    # Trace compact batches.  This retains energy-dependent Larmor/chromatic
    # transport, accelerates GUI-sized bundles, and bounds peak memory for
    # high-accuracy production ray counts.
    branches_per_batch=max(
        1,MAX_VECTORIZED_POST_RAYS//max(n,1)
    )
    for batch_start in range(0,len(branch_specs),branches_per_batch):
        batch_specs=branch_specs[
            batch_start:batch_start+branches_per_batch
        ]
        post_payloads=[]
        post_x=[];post_tx=[];post_y=[];post_ty=[];post_energy=[]
        for name,kick_x,kick_y,w,interaction_kind,energy_loss_ev in batch_specs:
            branch_energy_offset=dE-float(energy_loss_ev)
            chromatic_tx,chromatic_ty=branch_chromatic_kick(
                branch_energy_offset
            )
            kick_x_array=np.broadcast_to(
                np.asarray(kick_x,dtype=float),(n,)
            ).copy()
            kick_y_array=np.broadcast_to(
                np.asarray(kick_y,dtype=float),(n,)
            ).copy()
            post_payloads.append((
                name,w,interaction_kind,branch_energy_offset,
                kick_x_array,kick_y_array,
            ))
            post_x.append(X[-1])
            post_tx.append(TX[-1]+kick_x+chromatic_tx)
            post_y.append(Y[-1])
            post_ty.append(TY[-1]+kick_y+chromatic_ty)
            post_energy.append(branch_energy_offset)

        zp,XP_all,TP_all,YP_all,TYP_all=propagate(
            s,s.sample.z_mm,determine_tem_stop_z(s),
            np.concatenate(post_x),np.concatenate(post_tx),
            np.concatenate(post_y),np.concatenate(post_ty),
            post_events,np.concatenate(post_energy),
            include_initial_plane_kicks=False,
        )

        for branch_index,(name,w,interaction_kind,branch_energy_offset,kick_x_array,kick_y_array) in enumerate(post_payloads):
            branch_slice=slice(branch_index*n,(branch_index+1)*n)
            XP=XP_all[:,branch_slice];TP=TP_all[:,branch_slice]
            YP=YP_all[:,branch_slice];TYP=TYP_all[:,branch_slice]

            # Post-sample apertures and recording planes are resolved together
            # below so upstream stops always win over downstream stops.
            al=alive.copy();bl=blocked.copy();ks=list(keys)

            # Resolve detector and wall candidates, then keep the earliest
            # axial intercept. A later wall cannot hide an earlier detector.
            al,bl,ks=clip_recording_planes(s,zp,XP,YP,al,bl,ks)
            al,bl,ks=clip_column_wall(s,zp,XP,YP,al,bl,ks)

            branches[name]=Branch(
                name,_interaction_colour(interaction_kind),zp,XP,YP,TP,TYP,al,
                bl,ks,w,branch_energy_offset,emitted.weight,
                interaction_kind=interaction_kind,
                interaction_kick_x_rad=kick_x_array,
                interaction_kick_y_rad=kick_y_array,
                source_ray_id=incident.source_ray_id,
                source_azimuth_rad=incident.source_azimuth_rad,
            )

    if optical_only:
        from temsim.physics.optical_tuning import tuning_metrics
        metrics = tuning_metrics(s, incident)
        metrics['column_segment_cache'] = segment_cache_metrics
        return Simulation(incident, branches, metrics, gun_waist=gun_waist,
                          gun_trace=gun_trace, incident_plan=incident_plan,
                          incident_checkpoints=incident_checkpoints)

    analysis_reference_key=None
    from temsim.optics.direct_alignment import diffraction_transfer
    if s.projector_mode == 'image':
        analysis_stop_z=tem_projection_reference_plane_z(s)
        sample_transfer=diffraction_transfer(s,analysis_stop_z)
    else:
        from temsim.optics.direct_alignment import (
            diffraction_focus_depth_diagnostic,
            diffraction_reference_plane,
        )
        analysis_reference_key,analysis_stop_z=diffraction_reference_plane(s)
        sample_transfer=diffraction_transfer(s,analysis_stop_z)
    image_properties=linear_map_properties(sample_transfer.j_img)
    diffraction_properties=linear_map_properties(
        sample_transfer.j_diff_m_per_rad
    )
    image_larmor_rotation_rad = _sample_to_stop_larmor_rotation_rad(
        s, analysis_stop_z
    )
    cosine = math.cos(-image_larmor_rotation_rad)
    sine = math.sin(-image_larmor_rotation_rad)
    derotation = np.array(((cosine, -sine), (sine, cosine)))
    derotated_image = derotation @ sample_transfer.j_img
    signed_image_magnification = float(
        0.5 * np.trace(derotated_image)
    )
    reference_component=min(
        (s.fluorescent_screen,s.camera),
        key=lambda component: abs(
            float(component.z_mm)-float(analysis_stop_z)
        ),
    )
    recording_width_mm=float(getattr(
        reference_component,
        'width_mm',
        getattr(reference_component,'outer_width_mm'),
    ))
    half=recording_width_mm/2

    if s.projector_mode=='image':

        plane_name='objective_image_plane'
        plane_z=s.objective_image_plane_z_mm
        plane_map=(
            trace_transverse_transfer(s,plane_z,analysis_stop_z)
            if plane_z is not None else None
        )
        relay_error=(
            float(np.linalg.norm(plane_map.j_diff_m_per_rad,ord=2))
            if plane_map is not None else math.inf
        )
        plane_magnification=(
            linear_map_properties(plane_map.j_img).isotropic_scale
            if plane_map is not None else 0.0
        )
        magnification=max(image_properties.isotropic_scale,1e-15)
        metrics={'mode':'image','magnification':magnification,'object_full_m':recording_width_mm*1e-3/magnification,'relay_error':relay_error,'conjugate_plane':plane_name,'conjugate_plane_z_mm':plane_z,'conjugate_plane_magnification':plane_magnification}

    else:
        from temsim.optics.direct_alignment import (
            projector_field_calibration_rows,
        )
        plane_name=(
            analysis_reference_key
            if analysis_reference_key is not None
            else 'active_recording_plane'
        )
        plane_z=analysis_stop_z
        relay_error=float(np.linalg.norm(sample_transfer.j_img,ord=2))
        plane_magnification=image_properties.isotropic_scale
        L=max(diffraction_properties.isotropic_scale,1e-15);mrad_half=half/L;metrics={'mode':'diffraction','effective_camera_length_m':L,'mrad_half':mrad_half,'g_half_inv_nm':mrad_half*1e-3/lam,'relay_error':relay_error,'conjugate_plane':plane_name,'conjugate_plane_z_mm':plane_z,'conjugate_plane_magnification':plane_magnification}
        metrics['projector_field_calibration']=(
            projector_field_calibration_rows(s)
        )
        if analysis_reference_key is not None:
            focus_depth=diffraction_focus_depth_diagnostic(s)
            metrics.update({
                'diffraction_focus_depth_mm':focus_depth['full_depth_mm'],
                'diffraction_best_focus_offset_mm':focus_depth['best_focus_offset_mm'],
                'diffraction_best_focus_residual':focus_depth['best_residual'],
                'diffraction_focus_tolerance':focus_depth['tolerance'],
                'diffraction_focus_depth_model':focus_depth['model'],
            })

    # A mechanically valid trace may still lose every ray before the sample
    # (for example a deliberately coarse diagnostic trace through a small
    # aperture).  Keep that simulation result inspectable while making the
    # user-level beam observables explicitly unavailable.  Invalid weights on
    # surviving rays are still rejected by ``branch_sample_statistics``.
    sample_beam = (
        branch_sample_statistics(incident)
        if np.any(np.asarray(incident.alive, dtype=bool))
        else None
    )
    metrics.update({
        'column_segment_cache': segment_cache_metrics,
        'sample_inserted': sample_inserted,
        'sample_scattering_applied': bool(
            real_interactions is not None
            and (
                real_interactions.mean_inelastic_events > 0.0
                or real_interactions.absorbed_probability > 0.0
            )
        ),
        'specimen_mode': specimen_mode,
        'sample_scattering_model': scattering_model,
        'branch_weights_are_absolute': bool(
            sample_inserted and real_branch_weights_are_absolute
        ),
        'sample_absorbed_probability': (
            float(real_interactions.absorbed_probability)
            if real_interactions is not None else 0.0
        ),
        'real_inelastic_interactions': (
            real_interactions.metrics()
            if real_interactions is not None else None
        ),
        'ray_interaction_types': tuple(dict.fromkeys(
            branch.interaction_kind for branch in branches.values()
        )),
        'lambda_nm':lam,
        'transfer_coordinate_order':('x','y','theta_x','theta_y'),
        'transfer_analysis_plane_z_mm':analysis_stop_z,
        'transfer_analysis_plane_key':(
            analysis_reference_key
        ),
        'j_img':sample_transfer.j_img.tolist(),
        'j_diff_m_per_rad':sample_transfer.j_diff_m_per_rad.tolist(),
        'image_rotation_deg':image_properties.orientation_deg,
        'image_larmor_rotation_deg':math.degrees(
            image_larmor_rotation_rad
        ),
        'signed_image_magnification':signed_image_magnification,
        'image_inversion':(
            'inverted' if signed_image_magnification < 0.0 else 'upright'
        ),
        'diffraction_rotation_deg':diffraction_properties.orientation_deg,
        'image_handedness':(
            'mirrored' if image_properties.mirrored else 'preserved'
        ),
        'diffraction_handedness':(
            'mirrored' if diffraction_properties.mirrored else 'preserved'
        ),
        'image_anisotropy_ratio':image_properties.anisotropy_ratio,
        'diffraction_anisotropy_ratio':(
            diffraction_properties.anisotropy_ratio
        ),
        'image_conjugacy_residual_m_per_rad':float(
            np.linalg.norm(sample_transfer.j_diff_m_per_rad,ord=2)
        ),
        'diffraction_conjugacy_residual':float(
            np.linalg.norm(sample_transfer.j_img,ord=2)
        ),
        'sample_convergence_95_mrad':(
            sample_beam.convergence_95_mrad
            if sample_beam is not None else math.nan
        ),
        'sample_convergence_99_mrad':(
            sample_beam.convergence_99_mrad
            if sample_beam is not None else math.nan
        ),
        'sample_illumination_diameter_95_um':(
            sample_beam.illumination_diameter_95_um
            if sample_beam is not None else math.nan
        ),
        'sample_wavefront_curvature_per_m':(
            sample_beam.radial_wavefront_curvature_per_m
            if sample_beam is not None else math.nan
        ),
        'sample_waist_offset_mm':(
            sample_beam.waist_offset_m * 1.0e3
            if sample_beam is not None else math.nan
        ),
        'sample_beam_surviving_rays':(
            sample_beam.surviving_rays if sample_beam is not None else 0
        ),
        'sample_beam_surviving_fraction':(
            sample_beam.surviving_fraction
            if sample_beam is not None else 0.0
        ),
        'column_current_limit_percent':column_current_limit_percent(s),
        'effective_source_current_pa':effective_source_current_pa(s),
        'sample_surviving_current_pa':(
            effective_source_current_pa(s) * sample_beam.surviving_fraction
            if sample_beam is not None else 0.0
        ),
    })

    crossovers=detect_corrector_crossovers(incident,getattr(s,"corrector_crossover_targets_mm",[810.0,853.0,963.0]))

    lens_map={lens.key:lens for lens in s.lenses}

    c2c3=None

    condenser_lens_3 = lens_map.get(CONDENSER_LENS_3)
    condenser_lens_3_enabled = (
        condenser_lens_3 is not None
        and getattr(condenser_lens_3, "enabled", True)
    )
    if (
        getattr(s,"column_mode","three_lens")=="three_lens"
        and condenser_lens_3_enabled
        and CONDENSER_LENS_2 in lens_map
    ):
        c2c3=first_crossover_after_lens(
            incident,lens_map[CONDENSER_LENS_2].z_mm,
            "C2-C3 intermediate image crossover",
            stop_z_mm=condenser_lens_3.z_mm)

    from temsim.diagnostics import optical_transfer_records
    result=Simulation(
        incident=incident, branches=branches, metrics=metrics, gun_waist=gun_waist,
        c2c3_crossover=c2c3, corrector_crossovers=crossovers,
        gun_trace=gun_trace,
        sample_to_analysis_transfer=sample_transfer,
        optical_transfers=optical_transfer_records(s),
        real_interactions=real_interactions,
        incident_plan=incident_plan,
        incident_checkpoints=incident_checkpoints,
    )

    return result
