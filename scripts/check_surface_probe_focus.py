"""Detached tip-origin particle focus search; no calculated-array export.

Proposal maps are not physical acceptance. Final measurements propagate the
executed gun particles to the exact upper surface with real apertures/walls.
"""
from dataclasses import asdict
import argparse
import json
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

from temsim.optics.direct_alignment import _LiveFirstOrderModel
from temsim.physics.beam_statistics import transverse_beam_statistics
from temsim.optics.surface_probe_focus import measure_surface_focus, refine_surface_focus


VARIABLES=('condenser_lens_3','objective_lens')


def set_vector(state, values):
    lenses={lens.key:lens for lens in state.lenses}
    for key,value in zip(VARIABLES,values):
        lenses[key].percent=float(value)


def measure(state,gun,values,step=.1):
    """Use the same exact-plane acceptance implementation as refinement."""
    set_vector(state,values)
    # Reuse is owned by trace_source_to_exit; a caller cannot replace the source
    # by handing this diagnostic a different gun-exit bundle.
    return measure_surface_focus(state,step_mm=step).statistics


class Proposal:
    def __init__(self,state,gun):
        self.state=state
        self.z=float(state.sample.upper_surface_z_mm)
        self.captures=np.unique(np.r_[np.arange(state.electron_gun.exit_plane_z_mm,self.z,10.),
            state.condenser_aperture_2.z_mm,self.z])
        self.model=_LiveFirstOrderModel(state,state.electron_gun.exit_plane_z_mm,self.z,VARIABLES,
                                       step_mm=.1,capture_z_mm=self.captures)
        e=gun.exit_bundle
        positive=e.alive&(e.weight>0)
        self.source=np.array([e.x_m,e.y_m,e.tx_rad,e.ty_rad])[:,positive]
        self.weights=e.weight[positive]
        initial=[lens.percent for lens in self.model.lenses]
        initial_maps=self.model.matrices_at(initial,self.captures)
        cap=self.state.condenser_aperture_2
        xy=(initial_maps@self.source)[np.searchsorted(self.captures,cap.z_mm),:2]
        # The C1/C2-controlled pupil stays fixed in this C3/objective search.
        selected=np.hypot(xy[0]-cap.offset_x_mm*1e-3,xy[1]-cap.offset_y_mm*1e-3)<=cap.radius_mm*1e-3
        self.source=self.source[:,selected]
        self.weights=self.weights[selected]
        print('Proposal positive pupil',len(self.weights),flush=True)
        if len(self.weights)<4:
            raise ValueError('Insufficient incident pupil for focus search')

    def statistics(self,vector):
        states=self.model.matrices_at(vector,self.captures)@self.source
        stats=transverse_beam_statistics(*states[-1],weights=self.weights)
        return stats,states

    def residual(self,vector,target=30.):
        stats,states=self.statistics(vector)
        # Smooth mapping is a proposal, not the physical clipped distribution.
        wall=np.maximum(np.hypot(states[:,0],states[:,1])-2e-3,0).max()/1e-3
        return np.array([np.log(max(stats.convergence_95_mrad,1e-12)/target),
                         stats.waist_offset_m/1e-5,wall])

    def search(self,target=30.):
        found=[]
        with threadpool_limits(limits=1):
            for c3 in (15.,21.,30.,40.3,50.,65.,80.):
                for obj in (50.,65.,69.,80.,95.):
                    fit=least_squares(lambda v:self.residual(v,target),[c3,obj],
                        bounds=(np.zeros(2),self.model.upper),max_nfev=90,
                        ftol=1e-10,xtol=1e-10,gtol=1e-10,diff_step=1e-5)
                    cost=float(np.linalg.norm(self.residual(fit.x,target)))
                    if not any(np.linalg.norm(fit.x-p[1])<.02 for p in found):
                        found.append((cost,fit.x))
            found.sort(key=lambda v:v[0])
        return found


def main():
    from temsim.optics.column import default_state
    from temsim.operating_modes import apply_operating_mode_pair
    from temsim.simulation_modes import switch_mode
    from temsim.optics.electron_gun.source import trace_source_to_exit
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rays',type=int,default=193)
    parser.add_argument('--target',type=float,default=30.)
    parser.add_argument('--mode',choices=('ideal','custom'),default='ideal')
    parser.add_argument('--refine',action='store_true',help='Refine a supplied C3/objective seed with production particles')
    parser.add_argument('--seed',type=float,nargs=2,default=(32.72507667778783,25.260664949735187))
    parser.add_argument('--reference-rays',type=int,help='Independently check the fitted controls at this particle count')
    args=parser.parse_args()
    state=default_state()
    apply_operating_mode_pair(state,'nano_probe','diffraction')
    switch_mode(state,args.mode)
    state.electron_gun.emitter.ray_count=args.rays
    state.history_step_mm=.5
    state._optical_tuning=True
    lenses={lens.key:lens for lens in state.lenses}
    # Previous executed transport recovery supplies only a starting guess.
    for key,value in zip(('condenser_lens_1','condenser_lens_2','condenser_lens_3'),
                         (8.012821601837457,84.12563904641223,40.29774021130246)):
        lenses[key].percent=value
    started=perf_counter()
    gun=trace_source_to_exit(state)
    if args.refine:
        from temsim.instrument_snapshot import capture_instrument_snapshot
        from temsim.optics.transport_matching import transport_measurement
        from temsim.physics.simulation import run
        candidate,evidence=refine_surface_focus(state,args.target,initial=args.seed)
        print(json.dumps(evidence,default=asdict,indent=2),flush=True)
        if candidate is None:
            raise RuntimeError('Fixed-budget angle/focus/step validation failed; no state applied')
        result=run(candidate,resolved_layout=candidate._resolved_optics_layout,optical_only=True)
        print('Projection entrance',transport_measurement(result,
            next(a.z_mm for a in candidate.apertures if a.key=='projection_chamber_dpa_aperture')),flush=True)
        if args.reference_rays:
            reference=capture_instrument_snapshot(candidate).restore()
            reference.electron_gun.emitter.ray_count=args.reference_rays
            measured=measure_surface_focus(reference,step_mm=.025)
            print(json.dumps(dict(reference_rays=args.reference_rays,
                sampling_check='PASS' if measured.accepts(args.target) else 'FAIL',
                measurement=measured),default=asdict,indent=2),flush=True)
        print('Seconds',perf_counter()-started,flush=True)
        return candidate,evidence
    proposal=Proposal(state,gun)
    for cost,vector in proposal.search(args.target)[:6]:
        print(json.dumps(dict(cost=cost,vector=vector.tolist(),
            production=asdict(measure(state,gun,vector)))) ,flush=True)
    print('Seconds',perf_counter()-started,flush=True)


if __name__=='__main__':
    main()
