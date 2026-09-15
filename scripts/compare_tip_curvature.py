"""Compare equal-D95 tip particles inside every accelerator stage.

No wave calculation, default changes, cached arrays or automatic installation.
Lightweight reports contain all candidate inputs and real transport summaries.
Flat/curved use different field and emission laws: this is not a curvature-only
comparison. Curved-to-curved comparisons retain the same local emission law.
"""
import argparse
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
import json

import numpy as np

from temsim.optics.column import default_state
from temsim.operating_modes import apply_operating_mode_pair
from temsim.optics.tip_curvature_comparison import flat_tip_d95_nm, matched_curved_candidate
from temsim.optics.beam_path_audit import (
    emission_measurement, gun_planes_coordinates, spot_measurement, uniform_cap_footprint,
    incident_checkpoints, crossover_candidates, optical_component_planes,
)
from temsim.optics.electron_gun.source import trace_source_to_exit
from check_gun_matching_candidate import write_report, clean


def baseline(rays):
    state = default_state()
    apply_operating_mode_pair(state, "nano_probe", "diffraction")
    state.electron_gun.emitter.ray_count = rays
    state.electron_gun.history_step_mm = .2
    return state


def inspect(state, *, column=False, step_mm=.05):
    start = perf_counter()
    gun = state.electron_gun
    trace = trace_source_to_exit(state)
    assembly = state._resolved_assembly
    accelerator = assembly.part("feg_accelerator")
    tip = assembly.part("feg_tip")
    origin = tip.start_z_mm-float(tip.data["local_start_z_mm"])
    named = [("Accelerator entrance", accelerator.start_z_mm)]
    named += [(f"Accelerator stage {i+1}", s.center_from_tip_mm+origin)
              for i, s in enumerate(gun.accelerator.stages)]
    named += [("Accelerator exit", accelerator.end_z_mm), ("Gun exit", gun.exit_plane_z_mm)]
    planes = np.array([z for _, z in named])
    coordinates, masks = gun_planes_coordinates(trace, planes)
    spots = []
    for (name, z), values, mask in zip(named, coordinates, masks):
        row = asdict(spot_measurement(name, z, values, trace.exit_bundle.weight, mask,
            precision="executed gun history interpolation; not nm-focus acceptance"))
        row["current_na"] = row["source_current_fraction"]*gun.emitted_current_a*1e9
        segments = [s for s in assembly.vacuum_bore_segments if s.start_z_mm <= z <= s.end_z_mm]
        row["bore_diameter_mm"] = min((s.inner_diameter_mm for s in segments), default=None)
        row["maximum_axis_radius_mm"] = float(np.max(np.hypot(values[0,mask], values[1,mask]))*1e3) if mask.any() else None
        spots.append(row)
    model = gun.emitter.surface_model
    report = dict(qualification="DIAGNOSTIC_ONLY", rays=gun.emitter.ray_count,
        field_model="grounded electrode field" if model else "legacy analytic gun field",
        inputs=model.to_dict() if model else {"flat_fwhm_nm":gun.emitter.virtual_source_fwhm_nm,
            "angular_rms_mrad":gun.emitter.angular_rms_mrad,"angular_cutoff_mrad":gun.emitter.angular_cutoff_mrad},
        expected_d95_nm=uniform_cap_footprint(model)["diameter95_nm"] if model else flat_tip_d95_nm(gun.emitter),
        source_current_na=gun.emitted_current_a*1e9,
        emitted=asdict(emission_measurement(gun.emit(gun.emitter.ray_count))),
        extractor_kv=gun.extractor.voltage_kv,gun_lens_kv=gun.electrostatic_lens.voltage_kv,
        gun_lens_reference=gun.electrostatic_lens.voltage_reference,
        lens_percent={l.key:float(l.percent) for l in state.lenses},
        trace_step_mm=gun.trace_step_mm,history_step_mm=gun.history_step_mm,
        field_report=dict(getattr(trace,"surface_model_report",{}) or {}), spots=spots,
        stops=dict(Counter(k for k in trace.blocked_key if k)))
    if column:
        end = state.sample.upper_surface_z_mm
        planes = np.unique(np.r_[np.arange(gun.exit_plane_z_mm, end, 1.),
            [l.z_mm for l in state.lenses if gun.exit_plane_z_mm < l.z_mm < end], end])
        _, cp, masks = incident_checkpoints(state, planes, step_mm=step_mm)
        report["column_spots"] = [asdict(spot_measurement(str(cp.z_mm[i]),float(cp.z_mm[i]),
            (cp.x_m[i],cp.y_m[i],cp.tx_rad[i],cp.ty_rad[i]),trace.exit_bundle.weight,masks[i]))
            for i in [0,len(planes)-1]]
        report["column_crossovers"] = crossover_candidates(cp,masks,trace.exit_bundle.weight)
        report["component_planes"] = optical_component_planes(state,full_path=True)
        report["column_step_mm"] = step_mm
    report["seconds"] = perf_counter()-start
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rays",type=int,default=193)
    parser.add_argument("--radii-nm",type=float,nargs="+",default=[100.,200.,500.])
    parser.add_argument("--flat-only",action="store_true")
    parser.add_argument("--column",action="store_true")
    parser.add_argument("--electrode-cells",type=int,choices=range(4,65),default=8)
    parser.add_argument("--trace-step-mm",type=float,default=.2)
    parser.add_argument("--history-step-mm",type=float,default=.2)
    parser.add_argument("--export-inputs-only",action="store_true",
        help="Write detached .temwp input designs without running transport")
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists() or args.rays < 73:
        parser.error("Choose a new output and at least 73 particle samples")
    state = baseline(args.rays)
    cases = [("Flat",state)] + ([] if args.flat_only else
        [(f"Curved R={r:g} nm",matched_curved_candidate(state,r)) for r in args.radii_nm])
    reports = {}
    for name,candidate in cases:
        candidate.electron_gun.trace_step_mm = args.trace_step_mm
        candidate.electron_gun.history_step_mm = args.history_step_mm
        model = candidate.electron_gun.emitter.surface_model
        if model is not None:
            candidate.electron_gun.emitter.surface_model = replace(model,
                field_numerics=replace(model.field_numerics,electrode_cells_per_bore=args.electrode_cells))
        if args.export_inputs_only:
            from temsim.optics.gun_matching import input_working_point
            suffix = "flat" if model is None else f"r{model.geometry.apex_radius_nm:g}"
            path = args.output.with_name(args.output.stem+"_"+suffix+".temwp")
            path.parent.mkdir(parents=True,exist_ok=True)
            input_working_point(candidate,label=f"{name} matched D95 / Flat preset - NOT QUALIFIED").write_package(path)
            reports[name] = dict(input_package=str(path),status="INPUTS_ONLY; NO TRANSPORT")
            continue
        print(f"Executing {name}, {args.rays} tip particles",flush=True)
        reports[name] = inspect(candidate,column=args.column)
        print(json.dumps(clean({"case":name,"seconds":reports[name]["seconds"],
            "accelerator_entrance":reports[name]["spots"][0],
            "accelerator_exit":reports[name]["spots"][-2],
            "sample":reports[name].get("column_spots",[None])[-1]})),flush=True)
    write_report(dict(match="analytic projected source D95", comparison_caveat=
        "Flat and curved differ in field/spatial/angular laws and current. Not curvature alone.",
        reports=reports),args.output)


if __name__ == "__main__":
    main()
