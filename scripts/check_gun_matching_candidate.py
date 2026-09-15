"""Execute a detached classical gun/column candidate; never install a preset.

Input geometry updates both the solved field and physical vacuum bores. Reports
contain only scalar evidence and input settings, never particle arrays. A fast
axial proposal or a small displayed spot is not a qualified finite-source probe.
"""
from dataclasses import asdict, replace
import argparse
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from temsim.optics.column import default_state
from temsim.operating_modes import apply_operating_mode_pair
from temsim.optics.gun_matching import candidate_with_gun_geometry
from temsim.optics.beam_path_audit import (
    emission_measurement, incident_checkpoints, spot_measurement, crossover_candidates,
    gun_planes_coordinates, optical_component_planes, require_same_topology, refine_crossover_candidate,
    partition_surface_crossovers,
    uniform_cap_footprint,
)
from temsim.optics.surface_probe_focus import measure_surface_focus


def inspect_candidate(state, *, step_mm=.05, reference=None, refine=False):
    start = perf_counter()
    gun = state.electron_gun
    end = float(state.sample.upper_surface_z_mm)
    planes = np.unique(np.r_[np.arange(gun.exit_plane_z_mm, end, .5),
                             [l.z_mm for l in state.lenses if gun.exit_plane_z_mm < l.z_mm < end], end])
    trace, cp, mask = incident_checkpoints(state, planes, step_mm=step_mm)
    weight = trace.exit_bundle.weight
    names = [("Gun exit",gun.exit_plane_z_mm),("Sample entrance",end)]
    names.extend((l.name,float(l.z_mm)) for l in state.lenses if l.key in ("condenser_lens_1","condenser_lens_2"))
    spots = [asdict(emission_measurement(gun.emit(gun.emitter.ray_count)))]
    for name,z in names:
        j = int(np.flatnonzero(np.abs(cp.z_mm-z)<1e-10)[0])
        spots.append(asdict(spot_measurement(name,z,
            (cp.x_m[j],cp.y_m[j],cp.tx_rad[j],cp.ty_rad[j]),weight,mask[j])))
    crossings = crossover_candidates(cp,mask,weight)
    if refine:
        crossings = [refine_crossover_candidate(state,row,step_mm=step_mm) for row in crossings]
    component_planes = optical_component_planes(state)
    intermediate, terminal = partition_surface_crossovers(crossings, end)
    topology = "NOT_COMPARED"
    if reference is not None:
        try:
            reference_end = next(z for key,z in reference["component_planes"] if key == "sample")
            # Prefer the actual entrance from the report, not the sample centre.
            if reference.get("focus"):
                reference_end = reference["focus"]["surface_z_mm"]
            elif reference.get("scope_z_mm"):
                reference_end = reference["scope_z_mm"][-1]
            reference_intermediate, _ = partition_surface_crossovers(reference["crossovers"], reference_end)
            require_same_topology([r["z_mm"] for r in reference_intermediate],
                [r["z_mm"] for r in intermediate], reference["component_planes"],
                candidate_component_planes=component_planes)
            topology = "SAME_INCIDENT_INTERVALS_AT_THIS_BUDGET; not gun or post-specimen qualification"
        except ValueError as error:
            topology = str(error)
    # These gun histories are interpolated separately, never concatenated into
    # the precise column checkpoints or treated as a new configurable source.
    gz = np.arange(gun.electrostatic_lens.mechanical_center_from_tip_mm+
                   .5*gun.electrostatic_lens.mechanical_length_mm,gun.exit_plane_z_mm,.5)
    arrays, gun_masks = gun_planes_coordinates(trace,gz)
    history = SimpleNamespace(z_mm=gz,x_m=arrays[:,0],y_m=arrays[:,1],
                              tx_rad=arrays[:,2],ty_rad=arrays[:,3])
    gun_crossings = crossover_candidates(history,gun_masks,weight)
    focus = None
    if mask[-1].sum() >= 2:
        focus = asdict(measure_surface_focus(state,step_mm=step_mm))
    return dict(seconds=perf_counter()-start, qualification="DIAGNOSTIC_ONLY; no preset changed",
        rays=gun.emitter.ray_count,step_mm=step_mm,tip_model=gun.emitter.surface_model.to_dict(),
        exact_tip_footprint=uniform_cap_footprint(gun.emitter.surface_model),
        extractor_mm=gun.extractor.mechanical_center_from_tip_mm,
        lens_mm=gun.electrostatic_lens.mechanical_center_from_tip_mm,
        accelerator_mm=gun.accelerator.mechanical_center_from_tip_mm,
        gun_dpa_mm=float(gun.dpa_aperture.z_mm),
        accelerator_stage_mm=[s.center_from_tip_mm for s in gun.accelerator.stages],
        extractor_kv=gun.extractor.voltage_kv,lens_kv=gun.electrostatic_lens.voltage_kv,
        lens_reference=gun.electrostatic_lens.voltage_reference,
        lens_percent={l.key:float(l.percent) for l in state.lenses},
        field_report=dict(trace.surface_model_report),spots=spots,focus=focus,
        crossovers=crossings,component_planes=component_planes,topology=topology,
        full_component_planes=optical_component_planes(state, full_path=True),
        intermediate_crossovers=intermediate,terminal_surface_crossovers=terminal,
        terminal_focus_tolerance_nm=1.,
        gun_history_crossovers=gun_crossings,
        gun_topology="NOT_QUALIFIED; interpolated gun history, compare historical trace separately")


def clean(value):
    if isinstance(value, dict):
        return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [clean(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):
        return None
    return value


def write_report(report,path):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as stream:
        json.dump(clean(report),stream,indent=2,allow_nan=False)
        stream.write("\n")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lens-mm",type=float,required=True)
    p.add_argument("--extractor-mm",type=float,default=8.)
    p.add_argument("--accelerator-mm",type=float)
    p.add_argument("--electrode-cells",type=int,choices=(8,16,32,64),default=8)
    p.add_argument("--extractor-kv",type=float,default=4.5)
    p.add_argument("--lens-kv",type=float,default=1.1)
    p.add_argument("--rays",type=int,default=193)
    p.add_argument("--step-mm",type=float,default=.05)
    p.add_argument("--c1",type=float)
    p.add_argument("--c2",type=float)
    p.add_argument("--c3",type=float)
    p.add_argument("--objective",type=float)
    p.add_argument("--reference-report",type=Path)
    p.add_argument("--refine-crossovers",action="store_true")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        p.error("Choose a new evidence file; no existing output is overwritten")
    if not 4<=args.extractor_kv<=5 or not 1<=args.lens_kv<=1.2 or args.rays<73:
        p.error("Require 4-5 kV extraction, 1-1.2 kV gun lens and at least 73 tip samples")
    state=default_state()
    apply_operating_mode_pair(state,"nano_probe","diffraction")
    state.vacuum_map.enabled=False
    state=candidate_with_gun_geometry(state,extractor_center_mm=args.extractor_mm,lens_center_mm=args.lens_mm,
                                     accelerator_center_mm=args.accelerator_mm)
    gun=state.electron_gun
    gun.extractor.voltage_kv=args.extractor_kv
    gun.electrostatic_lens.voltage_kv=args.lens_kv
    gun.emitter.ray_count=args.rays
    gun.emitter.surface_model=replace(gun.emitter.surface_model,
        emission=replace(gun.emitter.surface_model.emission,spatial_sampling="apex_stratified_v1"),
        field_numerics=replace(gun.emitter.surface_model.field_numerics,
                              electrode_cells_per_bore=args.electrode_cells))
    for key,value in (("condenser_lens_1",args.c1),("condenser_lens_2",args.c2),
                      ("condenser_lens_3",args.c3),("objective_lens",args.objective)):
        if value is not None:
            lens=next(l for l in state.lenses if l.key==key)
            if not np.isfinite(value) or not 0<=value<=lens.max_percent:
                p.error("Lens excitation must be finite and within its configured limits")
            lens.percent=value
    state._optical_tuning=True
    reference=json.loads(args.reference_report.read_text(encoding="utf-8")) if args.reference_report else None
    print(f"Executing {args.rays} tip-origin particles; no image calculation",flush=True)
    report=inspect_candidate(state,step_mm=args.step_mm,reference=reference,refine=args.refine_crossovers)
    write_report(report,args.output)
    print(json.dumps(clean(report),allow_nan=False),flush=True)


if __name__=="__main__":
    main()
