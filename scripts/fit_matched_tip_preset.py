"""Bounded detached preset exploration from Flat Nanoprobe + Diffraction.

First-order proposals accelerate the search; every recorded final spot uses
the executed tip and real column. A failed topology/focus gate never installs
a preset. Source/current/geometry are fixed after the equal-D95 tip selection.
"""
import argparse
from dataclasses import asdict
from pathlib import Path
import json
import warnings

import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

from compare_tip_curvature import baseline, inspect
from check_gun_matching_candidate import clean, write_report
from fit_gun_column_candidate import downstream_proposals
from temsim.optics.tip_curvature_comparison import matched_curved_candidate
from temsim.optics.electron_gun.source import trace_source_to_exit
from temsim.optics.direct_alignment import _LiveFirstOrderModel
from temsim.optics.beam_path_audit import incident_checkpoints, spot_measurement
from temsim.optics.surface_probe_focus import refine_surface_focus, measure_surface_focus


def controls(state,keys,values):
    for key,value in zip(keys,values):
        lens = next(l for l in state.lenses if l.key == key)
        if not np.isfinite(value) or not 0 <= value <= lens.max_percent:
            raise ValueError("Lens controls must be finite and within their configured limits")
        lens.percent = float(value)


def compare_topology(candidate_report, reference_path):
    from temsim.optics.beam_path_audit import require_same_topology
    reference = json.loads(reference_path.read_text(encoding="utf-8"))["reports"]["Flat"]
    try:
        require_same_topology([c["z_mm"] for c in reference["column_crossovers"]],
            [c["z_mm"] for c in candidate_report["column_crossovers"]],
            reference["component_planes"],
            candidate_component_planes=candidate_report["component_planes"])
        return "SAME_INCIDENT_INTERVALS_AT_FIXED_BUDGET; gun and post-specimen NOT QUALIFIED"
    except ValueError as error:
        return str(error)


def pupil_proposals(state):
    trace = trace_source_to_exit(state)
    e = trace.exit_bundle
    alive = e.alive & (e.weight > 0)
    source = np.array([e.x_m,e.y_m,e.tx_rad,e.ty_rad])[:,alive]
    weight = np.sqrt(e.weight[alive]/e.weight[alive].sum())
    stop = state.condenser_aperture_2.z_mm
    keys = ("condenser_lens_1","condenser_lens_2")
    model = _LiveFirstOrderModel(state,state.electron_gun.exit_plane_z_mm,stop,
        keys,step_mm=.05,capture_z_mm=(stop,))
    initial = [next(l for l in state.lenses if l.key == k).percent for k in keys]
    def residual(values):
        return ((model.matrices_at(values,[stop])[0]@source)[:2]*weight/5e-5).ravel()
    found = []
    with threadpool_limits(1):
        for seed in [initial,[45.,20.],[35.,15.],[60.,30.]]:
            fit = least_squares(residual,seed,bounds=([20.,5.],[70.,40.]),max_nfev=80,
                ftol=1e-10,xtol=1e-10,gtol=1e-10)
            if not any(np.linalg.norm(fit.x-row[1]) < .01 for row in found):
                found.append((float(np.linalg.norm(fit.fun)),fit.x))
    found.sort(key=lambda r:r[0])
    print(json.dumps(dict(pupil_proposals=[dict(score=v,values=x.tolist()) for v,x in found])),flush=True)
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--radius-nm",type=float,default=500.)
    p.add_argument("--rays",type=int,default=193)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--initial-controls",type=float,nargs=4,
                   help="Explicit C1 C2 C3 Objective continuation from a recorded proposal")
    p.add_argument("--inspect-only",action="store_true",
        help="Audit and export the explicitly supplied controls; never fit or install")
    args = p.parse_args()
    if args.output.exists() or args.rays < 73:
        p.error("New output and at least 73 rays required")
    state = matched_curved_candidate(baseline(args.rays),args.radius_nm)
    if args.inspect_only:
        if args.initial_controls is None:
            p.error("Inspection requires four explicit initial controls")
        controls(state,("condenser_lens_1","condenser_lens_2","condenser_lens_3","objective_lens"),args.initial_controls)
        candidate_report = inspect(state,column=True)
        report = dict(status="NOT_QUALIFIED",candidate=candidate_report,
            incident_topology=compare_topology(candidate_report,args.reference),
            focus=asdict(measure_surface_focus(state,step_mm=.05)))
        from temsim.optics.gun_matching import input_working_point
        inputs_path = args.output.with_suffix(".temwp")
        input_working_point(state,label=f"R{args.radius_nm:g} preset trial - NOT QUALIFIED").write_package(inputs_path)
        report["input_package"] = str(inputs_path)
        write_report(report,args.output)
        print(json.dumps(clean(report)),flush=True)
        return
    # Warnings from trajectories already outside a real aperture are counted,
    # not hidden or taken as converged coordinates. Final survivor tests remain.
    report = dict(status="EXPLORATION_ONLY",radius_nm=args.radius_nm,rays=args.rays,attempts=[])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always",RuntimeWarning)
        pupils = [(0.,np.array(args.initial_controls[:2]))] if args.initial_controls else pupil_proposals(state)[:2]
        for score,values in pupils:
            controls(state,("condenser_lens_1","condenser_lens_2"),values)
            pupil_z = state.condenser_aperture_2.z_mm
            gun,cp,masks = incident_checkpoints(state,[pupil_z],step_mm=.05)
            pupil = asdict(spot_measurement("C2 aperture",pupil_z,
                (cp.x_m[0],cp.y_m[0],cp.tx_rad[0],cp.ty_rad[0]),gun.exit_bundle.weight,masks[0]))
            attempt = dict(c1_c2=values.tolist(),pupil=pupil)
            report["attempts"].append(attempt)
            print(json.dumps(clean(attempt)),flush=True)
            if pupil["rays"] < 16:
                continue
            proposals = ([(0.,np.array(args.initial_controls[2:]))] if args.initial_controls
                         else downstream_proposals(state))
            unique = []
            for cost,vector in proposals:
                if not any(np.linalg.norm(vector-other[1]) < .01 for other in unique):
                    unique.append((cost,vector))
            attempt["focus_proposals"] = []
            for cost,focus in unique[:2]:
                controls(state,("condenser_lens_3","objective_lens"),focus)
                measurement = measure_surface_focus(state,step_mm=.05)
                row = dict(c3_objective=focus.tolist(),measurement=asdict(measurement))
                attempt["focus_proposals"].append(row)
                print(json.dumps(clean(row)),flush=True)
                if measurement.statistics.surviving_rays < 16:
                    continue
                candidate,evidence = refine_surface_focus(state,30.,maximum_evaluations=30)
                evidence = {k:asdict(v) if k in {"reference","refined"} else v
                            for k,v in evidence.items()}
                row["refinement"] = evidence
                if candidate is not None:
                    report["candidate"] = inspect(candidate,column=True)
                    report["focus_evidence"] = evidence
                    report["incident_topology"] = compare_topology(report["candidate"],args.reference)
                    from temsim.optics.gun_matching import input_working_point
                    inputs_path = args.output.with_suffix(".temwp")
                    input_working_point(candidate,label=f"Matched-D95 R{args.radius_nm:g} candidate - NOT QUALIFIED").write_package(inputs_path)
                    report["input_package"] = str(inputs_path)
                    break
            if "candidate" in report:
                break
        report["warnings"] = dict((str(w.message),sum(str(v.message)==str(w.message) for v in caught)) for w in caught)
    write_report(report,args.output)
    print(json.dumps(clean(report),default=lambda o:asdict(o)),flush=True)


if __name__ == "__main__":
    main()
