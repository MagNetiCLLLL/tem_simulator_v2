"""Detached finite-source column fit with the strong objective branch retained.

Each proposal executes the ordinary production particle chain. A fit is not a
mesh/sampling/topology qualification and never installs operating settings.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares

from temsim.optics.column import default_state
from temsim.operating_modes import apply_operating_mode_pair
from temsim.optics.gun_matching import candidate_with_gun_geometry
from temsim.optics.surface_probe_focus import measure_surface_focus
from check_gun_matching_candidate import clean, inspect_candidate, write_report


def pupil_proposal(state):
    from temsim.optics.direct_alignment import _LiveFirstOrderModel
    from temsim.optics.electron_gun.source import trace_source_to_exit
    from threadpoolctl import threadpool_limits
    trace = trace_source_to_exit(state)
    e = trace.exit_bundle
    live = e.alive & (e.weight > 0)
    source = np.array([e.x_m, e.y_m, e.tx_rad, e.ty_rad])[:,live]
    weight = np.sqrt(e.weight[live]/e.weight[live].sum())
    stop = float(state.condenser_aperture_2.z_mm)
    model = _LiveFirstOrderModel(state, state.electron_gun.exit_plane_z_mm, stop,
        ("condenser_lens_1", "condenser_lens_2"), step_mm=.05, capture_z_mm=(stop,))
    def residual(values):
        return ((model.matrices_at(values,[stop])[0]@source)[:2]*weight/5e-5).ravel()
    candidates = []
    with threadpool_limits(1):
        for seed in ([25.,10.],[30.,15.],[35.,20.],[40.,25.]):
            fit = least_squares(residual, seed, bounds=([20.,5.],[45.,30.]), max_nfev=80,
                                ftol=1e-10, xtol=1e-10, gtol=1e-10)
            candidates.append((np.linalg.norm(fit.fun),fit.x))
    score, vector = min(candidates, key=lambda row:row[0])
    print(json.dumps(dict(scope="PUPIL_PROPOSAL_ONLY", values=vector.tolist(), score=score)),flush=True)
    return vector


def downstream_proposals(state):
    """Proposal from the executed C2 checkpoint, never an active new source."""
    from temsim.optics.beam_path_audit import incident_checkpoints
    from temsim.optics.direct_alignment import _LiveFirstOrderModel
    from temsim.physics.beam_statistics import transverse_beam_statistics
    from threadpoolctl import threadpool_limits
    start = float(state.condenser_aperture_2.z_mm)
    end = float(state.sample.upper_surface_z_mm)
    trace, cp, masks = incident_checkpoints(state,[start],step_mm=.05)
    live = masks[0] & (trace.exit_bundle.weight > 0)
    if live.sum() < 4:
        raise ValueError("Insufficient real C2-aperture population for a proposal")
    source = np.array([cp.x_m[0],cp.y_m[0],cp.tx_rad[0],cp.ty_rad[0]])[:,live]
    weights = trace.exit_bundle.weight[live]
    captures = np.r_[np.arange(start,end,10.),end]
    model = _LiveFirstOrderModel(state,start,end,("condenser_lens_3","objective_lens"),
                                 step_mm=.05,capture_z_mm=captures)
    def residual(values):
        points = model.matrices_at(values,captures)@source
        stats = transverse_beam_statistics(*points[-1],weights=weights)
        wall = np.maximum(np.hypot(points[:,0],points[:,1])-2e-3,0).max()/1e-3
        return [np.log(stats.convergence_95_mrad/30.),stats.waist_offset_m/1e-5,wall]
    candidates = []
    with threadpool_limits(1):
        for c3 in (12.,20.,28.,36.,44.,52.,60.):
            fit = least_squares(residual,[c3,68.98],bounds=([10.,68.5],[65.,69.4]),
                                x_scale=[3.,.03],max_nfev=80,diff_step=1e-7)
            candidates.append((float(np.linalg.norm(fit.fun)),fit.x))
    candidates.sort(key=lambda row:row[0])
    print(json.dumps(dict(scope="EXECUTED_C2_CHECKPOINT_PROPOSALS_ONLY",
        candidates=[dict(score=score,values=values.tolist()) for score,values in candidates])),flush=True)
    return candidates


def main():
    global LAST_STATE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rays", type=int, default=193)
    parser.add_argument("--c1", type=float, default=30.68886068895488)
    parser.add_argument("--c2", type=float, default=12.05)
    parser.add_argument("--c3", type=float, default=24.45)
    parser.add_argument("--objective", type=float, default=68.9668)
    parser.add_argument("--lens-mm", type=float, default=22.817987616230624)
    parser.add_argument("--extractor-mm", type=float, default=8.)
    parser.add_argument("--reference", choices=("tip","extractor"), default="extractor")
    parser.add_argument("--auto-pupil", action="store_true")
    parser.add_argument("--c2-factor", type=float, default=1.)
    parser.add_argument("--accelerator-mm", type=float, default=200.)
    parser.add_argument("--dpa-mm", type=float, default=None)
    parser.add_argument("--cells", type=int, default=8)
    parser.add_argument("--corner-cells", type=int, default=0)
    parser.add_argument("--apex-cells", type=int, default=20)
    parser.add_argument("--radial-nodes", type=int, default=160)
    parser.add_argument("--axial-nodes", type=int, default=320)
    parser.add_argument("--tangent-refinement-gain", type=float, default=None)
    parser.add_argument("--directions-per-site", type=int, default=8)
    parser.add_argument("--cap-allocation", type=int, nargs=9, default=(),
                        help="Numerical site priorities for all nine full-cap area strata")
    parser.add_argument("--tangent-width-sigma", type=float, default=.1)
    parser.add_argument("--tangent-allocation", type=int, nargs=9, default=(),
                        help="Numerical priorities for all nine tangent CDF cells; centre is cell five")
    parser.add_argument("--tangent-rule", choices=("tangent_stratified_v1","tangent_stratified_v2"),
                        default="tangent_stratified_v2")
    parser.add_argument("--fixed", action="store_true", help="Observe these exact settings without refitting")
    parser.add_argument("--preview-only", action="store_true", help="Execute the ordinary small optical preview, not focus acceptance")
    parser.add_argument("--fit-pair", choices=("c3-objective","c2-objective"),default="c3-objective")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("Choose a new report path")
    state = default_state()
    apply_operating_mode_pair(state, "nano_probe", "diffraction")
    state.vacuum_map.enabled = False
    state = candidate_with_gun_geometry(state, extractor_center_mm=args.extractor_mm, lens_center_mm=args.lens_mm,
                                       accelerator_center_mm=args.accelerator_mm, dpa_center_mm=args.dpa_mm)
    gun = state.electron_gun
    gun.extractor.voltage_kv = 4.5
    gun.electrostatic_lens.voltage_kv = 1.1
    gun.electrostatic_lens.voltage_reference = args.reference
    gun.emitter.ray_count = args.rays
    model = gun.emitter.surface_model
    gun.emitter.surface_model = replace(model,
        emission=replace(model.emission, spatial_sampling="apex_stratified_v1",
            spatial_stratum_allocation=tuple(args.cap_allocation),
            directions_per_position=args.directions_per_site,
            angular_sampling="uniform_cdf" if args.tangent_refinement_gain is None else args.tangent_rule,
            angular_refinement_gain=0. if args.tangent_refinement_gain is None else args.tangent_refinement_gain,
            angular_refinement_width_sigma=args.tangent_width_sigma,
            angular_stratum_allocation=tuple(args.tangent_allocation)),
        field_numerics=replace(model.field_numerics, electrode_cells_per_bore=args.cells,
                              electrode_corner_cells=args.corner_cells,
                              apex_cells_per_radius=args.apex_cells, radial_nodes=args.radial_nodes,
                              axial_nodes=args.axial_nodes))
    state._optical_tuning = True
    LAST_STATE = state  # python -i: retain the executed upstream cache between bounded fits.
    lenses = {l.key:l for l in state.lenses}
    lenses["condenser_lens_1"].percent = args.c1
    lenses["condenser_lens_2"].percent = args.c2
    lenses["condenser_lens_3"].percent = args.c3
    lenses["objective_lens"].percent = args.objective
    if args.preview_only:
        from temsim.gui.calculation_request import apply_request_numerics
        from temsim.physics.optical_tuning import prepare_tuning_snapshot
        from temsim.physics.simulation import run
        from temsim.optics.transport_matching import transport_measurement,target_plane
        apply_request_numerics(state,"Preview",args.rays,1.)
        prepare_tuning_snapshot(state,"Preview")
        started = perf_counter()
        result = run(state,resolved_layout=state._resolved_optics_layout,optical_only=True)
        evidence = dict(scope="ACTUAL_PREVIEW_NOT_NANOMETRE_FOCUS_ACCEPTANCE",
            seconds=perf_counter()-started,requested_rays=args.rays,executed_rays=state.electron_gun.emitter.ray_count,
            projection=transport_measurement(result,target_plane(state)),metrics=result.metrics)
        write_report(evidence,args.output)
        print(json.dumps(clean(evidence)),flush=True)
        return
    if args.auto_pupil:
        c1, c2 = pupil_proposal(state)
        lenses["condenser_lens_1"].percent = float(c1)
        lenses["condenser_lens_2"].percent = float(c2*args.c2_factor)
    start = perf_counter()
    evaluations = []
    fit_key = "condenser_lens_2" if args.fit_pair == "c2-objective" else "condenser_lens_3"
    initial = np.array([lenses[fit_key].percent,args.objective])
    bounds = ([5.,68.5],[30.,69.4]) if args.fit_pair == "c2-objective" else ([10.,68.5],[60.,69.4])
    def objective(values):
        lenses[fit_key].percent, lenses["objective_lens"].percent = map(float, values)
        row = measure_surface_focus(state, step_mm=.05)
        record = dict(values=values.tolist(), measurement=asdict(row), seconds=perf_counter()-start)
        evaluations.append(record)
        print(json.dumps(clean(dict(values=values.tolist(), rays=row.statistics.surviving_rays,
            fraction=row.statistics.surviving_fraction, effective_rays=row.effective_rays,
            diameter95_nm=2e9*row.statistics.radius_95_m, alpha95_mrad=row.statistics.convergence_95_mrad,
            waist_nm=row.local_waist_offset_nm, seconds=perf_counter()-start))), flush=True)
        return [np.log(row.statistics.convergence_95_mrad/30.), row.local_waist_offset_nm/1000.]
    try:
        if args.fixed:
            objective(initial)
            report = inspect_candidate(state,step_mm=.025)
            report["fit"] = dict(scope="FIXED_SETTINGS_NO_REFIT",evaluations=evaluations)
            write_report(report,args.output)
            return
        fit = least_squares(objective, initial, bounds=bounds,
                            x_scale=[3.,.03], diff_step=1e-8, max_nfev=40, ftol=1e-9, xtol=1e-10, gtol=1e-9)
    except ValueError as error:
        report = inspect_candidate(state, step_mm=.05)
        report["fit"] = dict(success=False, error=str(error), evaluations=evaluations)
        write_report(report, args.output)
        print(json.dumps(report["fit"]),flush=True)
        return
    objective(fit.x)
    report = inspect_candidate(state, step_mm=.025)
    report["fit"] = dict(values=fit.x.tolist(), success=bool(fit.success), message=fit.message,
                         controls=[fit_key,"objective_lens"],evaluations=evaluations)
    write_report(report, args.output)
    print(json.dumps(clean(report["focus"])), flush=True)


if __name__ == "__main__":
    main()
