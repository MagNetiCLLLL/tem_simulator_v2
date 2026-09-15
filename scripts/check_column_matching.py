"""Bounded, detached particle-column matching diagnostic (no array exports).

First-order maps only propose lens settings. Acceptance uses the ordinary
nonlinear, clipped ray pipeline and the same executed tip-to-gun calculation.
No source parameters, aperture sizes, walls or corrector fields are changed.
"""
from collections import Counter
import argparse
import json
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

from temsim.optics.column import default_state
from temsim.optics.direct_alignment import _LiveFirstOrderModel
from temsim.optics.electron_gun.source import trace_source_to_exit
from temsim.physics.simulation import run


def summary(result):
    rows = []
    for branch in (result.incident, *result.branches.values()):
        planes = []
        for z in (450., 550., 650., 765., 1010., 1599.2, 1814.4, 2586.9, 2800.):
            if not branch.z[0] <= z <= branch.z[-1]:
                continue
            live = (~np.isfinite(branch.blocked_z) | (branch.blocked_z > z)) & (branch.ray_weight > 0)
            j = min(np.searchsorted(branch.z, z), len(branch.z)-1)
            planes.append(dict(z_mm=z, rays=int(live.sum()),
                current_fraction=float(branch.ray_weight[live].sum()),
                radius_mm=float(np.max(np.hypot(branch.x[j,live], branch.y[j,live])*1e3, initial=0))))
        rows.append(dict(branch=branch.name, planes=planes, stops=dict(Counter(branch.blocked_key))))
    return rows


def audit(rays=49, preset=False):
    state = default_state()
    if preset:
        from temsim.operating_modes import apply_operating_mode_pair
        apply_operating_mode_pair(state,'nano_probe','diffraction')
    state.electron_gun.emitter.ray_count = rays
    state.step_mm = .2
    state.history_step_mm = .5
    state.acceleration_enabled = True
    state._optical_tuning = True
    started = perf_counter()
    gun = trace_source_to_exit(state)
    print('Executed gun', perf_counter()-started, flush=True)
    emitted = gun.exit_bundle
    live = emitted.alive & (emitted.weight > 0)
    source = np.array([emitted.x_m, emitted.y_m, emitted.tx_rad, emitted.ty_rad])[:,live]
    weights = np.sqrt(emitted.weight[live] / emitted.weight[live].sum())
    keys = ('condenser_lens_1', 'condenser_lens_2','condenser_lens_3')
    lens_map = {lens.key:lens for lens in state.lenses}
    captures = np.unique(np.r_[np.arange(450.,state.sample.z_mm,10.),765.,state.sample.z_mm])
    model = _LiveFirstOrderModel(state, 450., state.sample.z_mm, keys, step_mm=.2, capture_z_mm=captures)

    def residual(values):
        positions = (model.matrices_at(values,captures) @ source)[:,:2,:]
        # Minimise the footprint at the actual 100 um C2 aperture, while
        # penalising approaches to the retained 5.76 mm upstream tube bore.
        final = (positions[np.searchsorted(captures,765.)]*weights/50e-6).ravel()
        wall = np.maximum(np.hypot(positions[:,0],positions[:,1])-2.4e-3,0)*weights/50e-6
        downstream = positions[captures>765.]*weights/2.5e-4
        return np.r_[final,wall.ravel(),downstream.ravel()]

    candidates = []
    with threadpool_limits(limits=1):
        for c1,c2 in ((3.,8.5),(11.,14.),(11.,76.),(51.,25.),(89.,14.)):
            for c3 in (10.,21.,35.,55.):
                fit = least_squares(residual,[c1,c2,c3],bounds=(np.zeros(3),np.full(3,99.)),max_nfev=80,
                                    ftol=1e-7,xtol=1e-7,gtol=1e-7)
                candidates.append((float(np.linalg.norm(fit.fun)),fit.x))
        candidates.sort(key=lambda item:item[0])
    tested = []
    for cost,vector in candidates:
        if any(np.linalg.norm(vector-previous)<.05 for previous in tested):
            continue
        tested.append(vector)
        for key,value in zip(keys,vector):
            lens_map[key].percent=float(value)
        result = run(state,optical_only=True)
        print(json.dumps(dict(cost=cost,c1_c2=vector.tolist(),result=summary(result))),flush=True)
        if len(tested)>=6:
            break
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rays',type=int,default=49)
    parser.add_argument('--preset',action='store_true')
    parser.add_argument('--recover',action='store_true',help='Run the production transactional recovery and refinement')
    parser.add_argument('--mode',choices=('ideal','custom'),default='custom')
    args=parser.parse_args()
    if args.recover:
        from temsim.alignment_transaction import AlignmentRequest, AlignmentCommitGate, solve_alignment_candidate
        from temsim.operating_modes import apply_operating_mode_pair
        from temsim.simulation_modes import switch_mode
        state=default_state()
        if args.preset:
            apply_operating_mode_pair(state,'nano_probe','diffraction')
        switch_mode(state,args.mode)
        started=perf_counter()
        request=AlignmentRequest.capture(state,'column_transport',.01,revision=0)
        candidate=solve_alignment_candidate(request)
        print(json.dumps(dict(status=candidate.status,seconds=perf_counter()-started,
            message=candidate.result.message,strengths=dict(candidate.result.strengths),
            reference=dict(candidate.validation.get('reference',{})),
            refined=dict(candidate.validation.get('refined',{}))),indent=2),flush=True)
        if not candidate.result.success:
            raise RuntimeError('No validated recovery; original state retained')
        updated=AlignmentCommitGate().apply(state,candidate,revision=0)
        print(json.dumps(summary(candidate.ray_result.simulation),indent=2),flush=True)
        return updated
    return audit(args.rays,args.preset)


if __name__ == '__main__':
    state = main()
