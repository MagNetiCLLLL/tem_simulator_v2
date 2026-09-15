"""Explicit particle transport recovery, not a probe/image-focus calibration.

Only C1/C2/C3 excitation may change. The executed gun, all stops, specimen
settings and downstream optics are retained. Linear maps propose candidates;
the normal nonlinear optical-only pipeline decides whether they pass.
"""
from dataclasses import asdict
import math
from types import SimpleNamespace

import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

from temsim.immutable_json import json_digest
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.operating_modes import DirectAlignmentDefinition


KEY = "column_transport"
LENSES = ("condenser_lens_1", "condenser_lens_2", "condenser_lens_3")
DEFINITION = DirectAlignmentDefinition(
    key=KEY, name="Match column transport", family="condenser", mode_key="nano_probe",
    unit="source fraction", minimum=.01, maximum=.99, default_value=.01,
    devices=LENSES, observable="positive_current_past_projection_chamber_entrance",
    constraint="finite_clipped_particle_paths", calibration_status="bounded_transport_search_not_focus_alignment",
    calibration_reference="Current executed tip-origin particles; no aperture or field removal.",
    targets={"search_rays":49, "validation_rays":193, "minimum_survivors":4,
             "search_step_mm":.2, "validation_step_mm":.1, "refinement_step_mm":.05,
             "maximum_current_spread":.02, "maximum_envelope_spread":.02},
    applies_to_modes=("nano_probe", "micro_probe"), state_parameters=(),
)


def target_plane(state):
    """Use the installed permanent projection-chamber entrance aperture."""
    aperture = next((a for a in state.apertures
                     if a.key == "projection_chamber_dpa_aperture"), None)
    if aperture is None or not aperture.enabled or not getattr(aperture, "installed", True):
        raise ValueError("Match transport requires the installed projection-chamber entrance aperture")
    z = float(aperture.z_mm)
    if not np.isfinite(z) or z <= state.sample.z_mm:
        raise ValueError("Projection-chamber entrance must be downstream of the specimen")
    return z


def transport_measurement(simulation, z_mm):
    """Count positive-current rays after the aperture, not a drawn envelope.

An eventual detector hit is a successful transmission through an upstream
plane even though its final alive flag is false. Earlier stops never pass.
"""
    branch = simulation.branches.get("000")
    if branch is None or not branch.z[0] <= z_mm <= branch.z[-1]:
        raise ValueError("Optical transport result does not cover the projection chamber")
    weights = np.asarray(branch.ray_weight, float)
    live = (weights > 0) & (~np.isfinite(branch.blocked_z) | (branch.blocked_z > z_mm + 1e-9))
    row = min(int(np.searchsorted(branch.z, z_mm)), len(branch.z)-1)
    visible = ((~np.isfinite(branch.blocked_z)[None,:]
                | (branch.z[:,None] <= branch.blocked_z[None,:]))
               & (branch.z[:,None] <= z_mm) & live[None,:])
    finite = all(np.all(np.isfinite(values) | ~visible)
                 for values in (branch.x,branch.y,branch.tx,branch.ty))
    incident = simulation.incident
    upstream_visible = ((~np.isfinite(incident.blocked_z)[None,:]
                         | (incident.z[:,None] <= incident.blocked_z[None,:])) & live[None,:])
    finite &= all(np.all(np.isfinite(values) | ~upstream_visible)
                  for values in (incident.x,incident.y,incident.tx,incident.ty))
    radius = np.hypot(branch.x[row,live],branch.y[row,live])
    return dict(rays=int(live.sum()), source_fraction=float(weights[live].sum()*branch.weight),
                radius_mm=float(radius.max(initial=0)*1e3), finite=bool(finite), z_mm=float(z_mm))


def _candidate_vectors(state, source, weights, check):
    from temsim.optics.direct_alignment import _LiveFirstOrderModel
    start, stop = float(state.electron_gun.exit_plane_z_mm),float(state.sample.z_mm)
    aperture = state.condenser_aperture_2
    if not start < aperture.z_mm < stop or aperture.radius_mm <= 0:
        raise ValueError("C2 aperture must have a positive opening between gun and specimen")
    captures = np.unique(np.r_[np.arange(start,stop,10.),aperture.z_mm,stop])
    model = _LiveFirstOrderModel(state,start,stop,LENSES,
        step_mm=float(DEFINITION.targets["search_step_mm"]),capture_z_mm=captures)
    if model.vector_maps:
        raise ValueError("Transport search for imported/coupled vector fields is not yet qualified")
    weights = np.sqrt(weights/weights.sum())
    aperture_row = np.searchsorted(captures,aperture.z_mm)
    # This smooth proposal cost is NOT clipping or a physical acceptance mask.
    # Bore/offset details are tested by production tracing before application.
    bore = min(segment.inner_diameter_mm for segment in state._resolved_assembly.vacuum_bore_segments
               if segment.end_z_mm > start and segment.start_z_mm < stop)*.5e-3
    aperture_radius = aperture.radius_mm*1e-3

    def residual(values):
        check()
        positions = (model.matrices_at(values,captures) @ source)[:,:2,:]
        final = (positions[aperture_row]-np.array([aperture.offset_x_mm,aperture.offset_y_mm])[:,None]*1e-3)
        wall = np.maximum(np.hypot(positions[:,0],positions[:,1])-.85*bore,0)*weights/aperture_radius
        downstream = positions[captures>aperture.z_mm]*weights/max(.1*bore,aperture_radius)
        return np.r_[(final*weights/aperture_radius).ravel(),wall.ravel(),downstream.ravel()]

    current = np.array([lens.percent for lens in model.lenses])
    seeds = [current]
    seeds.extend([c1,c2,c3] for c1,c2 in ((3.,8.5),(11.,14.),(11.,76.),(51.,25.),(89.,14.))
                 for c3 in (10.,21.,35.,55.))
    candidates=[]
    # Small repeated 4x4 maps are latency-bound; avoid launching a BLAS pool
    # for each objective evaluation. The actual particle backend is unchanged.
    with threadpool_limits(limits=1):
        for seed in seeds:
            check()
            fitted=least_squares(residual,np.clip(seed,0,model.upper),
                bounds=(np.zeros(3),model.upper),max_nfev=80,ftol=1e-7,xtol=1e-7,gtol=1e-7)
            if np.all(np.isfinite(fitted.x)) and np.all(np.isfinite(fitted.fun)):
                candidates.append((float(np.linalg.norm(fitted.fun)),fitted.x))
    candidates.sort(key=lambda pair:pair[0])
    selected=[]
    for _,vector in candidates:
        if not any(np.linalg.norm(vector-previous)<.05 for previous in selected):
            selected.append(vector)
        if len(selected)==8:
            break
    return selected


def solve_transport_candidate(request, *, cancelled=lambda:False):
    from temsim.alignment_transaction import AlignmentCancelled, AlignmentCandidate, _allowed_state
    from temsim.calculation_manifest import capture_calculation_manifest
    from temsim.optics.direct_alignment import DirectAlignmentResult
    from temsim.optics.electron_gun.source import trace_source_to_exit
    from temsim.physics.simulation import run
    from temsim.simulation_pipeline import CalculationResult, aperture_stop_records
    from temsim.physics.all_lens_crossovers import detect_all_lens_crossovers
    from temsim.working_point import WorkingPointCheckpoint

    def check():
        if cancelled():
            raise AlignmentCancelled("Transport matching cancelled; previous working point retained")
    check()
    if request.registry_digest != json_digest(asdict(DEFINITION)):
        raise ValueError("Transport matching definition changed")
    scratch=request.start_snapshot.restore()
    from temsim.physics.beam_current import effective_source_current_a
    if effective_source_current_a(scratch) <= 0:
        raise ValueError("No source current is enabled; transport matching cannot create electrons")
    if scratch.electron_gun.source_representation != "classical_particles":
        raise ValueError("Transport recovery currently requires classical tip particles; coherent development remains paused")
    z=target_plane(scratch)
    scratch.electron_gun.emitter.ray_count=int(DEFINITION.targets["search_rays"])
    scratch.step_mm=float(DEFINITION.targets["search_step_mm"])
    scratch.history_step_mm=.5
    scratch._optical_tuning=True
    scratch._tuning_cancelled=cancelled
    gun=trace_source_to_exit(scratch)
    check()
    emitted=gun.exit_bundle
    mask=emitted.alive & (emitted.weight>0)
    minimum_rays=int(DEFINITION.targets["minimum_survivors"])
    if np.count_nonzero(mask)<minimum_rays:
        raise ValueError("Fewer than four current-carrying rays exit the gun; correct gun fields/alignment first")
    source=np.array([emitted.x_m,emitted.y_m,emitted.tx_rad,emitted.ty_rad])[:,mask]
    candidates=_candidate_vectors(scratch,source,emitted.weight[mask],check)
    lenses={lens.key:lens for lens in scratch.lenses}
    attempts=[]
    for vector in candidates:
        check()
        strengths=dict(zip(LENSES,map(float,vector)))
        for key,value in strengths.items():
            lenses[key].percent=value
        try:
            coarse=run(scratch,resolved_layout=scratch._resolved_optics_layout,optical_only=True)
            measure=transport_measurement(coarse,z)
        except (ValueError,FloatingPointError) as exc:
            attempts.append({"strengths":strengths,"rejected":str(exc)})
            continue
        attempts.append({"strengths":strengths,"measurement":measure})
        if not measure["finite"] or measure["rays"]<minimum_rays or measure["source_fraction"]<request.target:
            continue
        # Reconstruct from the complete captured graph, changing only approved
        # controls. Numerical budgets below belong only to validation.
        candidate_state=_allowed_state(request,strengths)
        candidate_snapshot=capture_instrument_snapshot(candidate_state)
        validation=candidate_snapshot.restore()
        validation.electron_gun.emitter.ray_count=int(DEFINITION.targets["validation_rays"])
        validation.history_step_mm=.5
        validation._optical_tuning=True
        validation._tuning_cancelled=cancelled
        validation.step_mm=float(DEFINITION.targets["validation_step_mm"])
        comparison=run(validation,resolved_layout=validation._resolved_optics_layout,optical_only=True)
        reference=transport_measurement(comparison,z)
        check()
        validation.step_mm=float(DEFINITION.targets["refinement_step_mm"])
        # Callbacks are runtime-only and must not enter a persistent snapshot.
        del validation._tuning_cancelled
        manifest=capture_calculation_manifest(validation)
        refined=run(validation,resolved_layout=validation._resolved_optics_layout,optical_only=True)
        measured=transport_measurement(refined,z)
        check()
        spread=abs(measured["source_fraction"]-reference["source_fraction"])/max(measured["source_fraction"],1e-30)
        radius_spread=abs(measured["radius_mm"]-reference["radius_mm"])/max(measured["radius_mm"],1e-12)
        passed=(measured["finite"] and reference["finite"] and measured["rays"]>=minimum_rays
                and reference["rays"]>=minimum_rays and measured["source_fraction"]>=request.target
                and reference["source_fraction"]>=request.target
                and spread<=float(DEFINITION.targets["maximum_current_spread"])
                and radius_spread<=float(DEFINITION.targets["maximum_envelope_spread"]))
        if not passed:
            attempts[-1]["refinement_rejected"]={"reference":reference,"refined":measured}
            continue
        message=(f"Transport matched: {measured['rays']} / 193 rays, {100*measured['source_fraction']:.3g}% of source current "
                 "past the projection-chamber entrance. Probe/image focus is not calibrated.")
        result=DirectAlignmentResult(KEY,True,request.target,measured["source_fraction"],"source fraction",
            measured["radius_mm"],"mm",strengths,len(attempts),validation.step_mm,spread,message,
            target_plane_key="projection_chamber_dpa_aperture",target_plane_z_mm=z)
        checkpoint=WorkingPointCheckpoint.from_result(SimpleNamespace(simulation=refined,
            calculation_manifest=manifest,signatures=manifest.calculation_signatures),parent_id=request.start_snapshot.digest)
        display=CalculationResult(refined,None,state_snapshot=validation,layout=validation._resolved_optics_layout,
            assembly=validation._resolved_assembly,
            lens_crossovers=tuple(detect_all_lens_crossovers([refined.incident,*refined.branches.values()],validation.lenses)),
            aperture_stops=aperture_stop_records(validation),calculation_manifest=manifest,
            signatures=manifest.calculation_signatures,calculated_products=frozenset({"incident","column"}))
        return AlignmentCandidate(request,result,checkpoint,{"status":"PASS",
            "candidate_snapshot":candidate_snapshot.to_dict(),"forward_snapshot_id":manifest.instrument_snapshot.digest,
            "reference":reference,"refined":measured,"current_spread":spread,"envelope_spread":radius_spread,
            "attempts":attempts,"scope":"optical transport only; no specimen/image/focus qualification"},
            "READY_TO_APPLY",ray_result=display)
    result=DirectAlignmentResult(KEY,False,request.target,0.,"source fraction",math.nan,"mm",{},
        len(attempts),.05,math.nan,"No forward-validated transport match found; all settings retained.")
    return AlignmentCandidate(request,result,None,{"status":"FAIL","attempts":attempts},"FAILED")
