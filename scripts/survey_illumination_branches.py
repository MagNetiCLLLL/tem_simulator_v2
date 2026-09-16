"""Survey source-specific condenser branches; no defaults are installed."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.calculation_manifest import solver_source_identity
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state
from temsim.optics.assembly_illumination import TARGETS, seed_illumination, diameter_gate, _focus_branch
from temsim.optics.illumination_current import aperture_gate, set_calibration_flux
from temsim.simulation_modes import mode_key as simulation_mode_key
from temsim.optics.illumination_checkpoint import IlluminationCheckpoint
from temsim.optics.illumination_search import focus_grid_proposals
from temsim.optics.surface_probe_focus import measure_surface_focus
from calibrate_assembly_illumination import finite_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gun', default='FEG')
    parser.add_argument('--column', default='C2')
    parser.add_argument('--blanker', default='None')
    parser.add_argument('--mode', choices=('nano_probe','micro_probe'), default='nano_probe')
    parser.add_argument('--primary', default='condenser_lens_2')
    parser.add_argument('--rays', type=int, default=512)
    parser.add_argument('--mini', type=float, nargs='+', default=[0.,35.,-35.])
    parser.add_argument('--aperture', type=float, default=.12)
    parser.add_argument('--fixed', action='append', default=[], metavar='LENS=PERCENT',
                        help='Fixed installed pre-specimen lens excitation (not either varied lens)')
    parser.add_argument('--primary-points', type=int, default=49)
    parser.add_argument('--objective-points', type=int, default=65)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Choose a new report path')
    selection = AssemblySelection(args.gun,args.column,'Energy Filter',args.blanker)
    document = dict(implementation=solver_source_identity(), status='NOT_INSTALLED', records=[])
    target = TARGETS[args.mode]

    def save():
        if solver_source_identity() != document['implementation']:
            raise RuntimeError('Solver changed during calculation')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(finite_json(document),indent=2,allow_nan=False)+'\n',encoding='utf-8')

    with threadpool_limits(1):
        for mini in args.mini:
            start = perf_counter()
            state = default_state()
            AssemblyCatalog().apply(state,selection)
            seed_illumination(state,args.mode)
            state.electron_gun.emitter.ray_count=args.rays
            state.condenser_aperture_2.diameter_mm=args.aperture
            if not aperture_gate(state)['passed']:
                parser.error('The physical C2 aperture must be enabled with diameter 20-250 um')
            lenses={l.key:l for l in state.lenses}
            lenses['mini_condenser'].percent=abs(mini)
            lenses['mini_condenser'].polarity=-1 if mini<0 else 1
            keys=(args.primary,'objective_lens')
            for assignment in args.fixed:
                key, value = assignment.split('=', 1)
                value = float(value)
                if (key not in lenses or key in keys or key == 'mini_condenser'
                        or not lenses[key].enabled or lenses[key].z_mm > state.sample.z_mm
                        or not np.isfinite(value) or not 0 <= value <= lenses[key].max_percent):
                    parser.error('Fixed values require an installed pre-specimen lens outside the varied pair')
                lenses[key].percent = value
            upper=np.array([lenses[k].max_percent for k in keys])
            print(f'{selection}: {args.mode}, {keys}, mini {mini}',flush=True)
            proposals,audit=focus_grid_proposals(state,keys,args.mode,
                primary_points=args.primary_points, objective_points=args.objective_points,
                progress=lambda m:print(m,flush=True))
            record=dict(assembly=asdict(selection), mode=args.mode,rays=args.rays,mini_signed_percent=mini,
                        simulation_mode=simulation_mode_key(state),
                        controls=keys, fixed=args.fixed, c2_aperture_diameter_mm=args.aperture,
                        proposals=[asdict(p) for p in proposals],search=audit,attempts=[],status='NOT_QUALIFIED')
            document['records'].append(record);save()
            checkpoint=IlluminationCheckpoint(state,keys)
            for proposal in proposals[:4]:
                print('Testing',proposal,flush=True)
                seen={}
                count=0
                def measure(v):
                    nonlocal count
                    identity=tuple(map(float,v))
                    if identity not in seen:
                        if count>=90: raise RuntimeError('Physical evaluation budget')
                        count+=1
                        seen[identity]=checkpoint.measure(v)
                    return seen[identity]
                try:
                    if args.mode=='nano_probe':
                        vector=_focus_branch(measure,target,np.array(proposal.values),upper)
                    else:
                        fit=least_squares(lambda v:target.residual(measure(v).statistics),proposal.values,
                            bounds=(np.zeros(2),upper),diff_step=1e-6,x_scale='jac',max_nfev=30,
                            ftol=1e-10,xtol=1e-10,gtol=1e-10)
                        vector=fit.x
                    m=measure(vector)
                    for k,v in zip(keys,vector): lenses[k].percent=float(v)
                    fine=measure_surface_focus(state,step_mm=.025)
                    flux=set_calibration_flux(state,fine)
                    passed=target.accepts(m) and target.accepts(fine) and diameter_gate(state,fine,args.mode)['passed']
                    row=dict(values=list(vector),fine=asdict(fine),focus_angle_pass=passed,flux_selection=flux)
                    record['attempts'].append(row)
                    print('Actual',fine.statistics.convergence_95_mrad,fine.statistics.illumination_diameter_95_um,
                          fine.local_waist_offset_nm,'pass',passed,flush=True)
                    if passed:
                        record.update(strengths={l.key:l.percent for l in state.lenses if l.enabled and l.z_mm<=state.sample.z_mm},
                            polarities={l.key:l.polarity for l in state.lenses if l.enabled and l.z_mm<=state.sample.z_mm},
                            c2_aperture_diameter_mm=state.condenser_aperture_2.diameter_mm,fine=asdict(fine),
                            column_current_limit_percent=state.column_current_limit_percent,flux_selection=flux,
                            focus_angle_pass=True,executed_input_digest=capture_instrument_snapshot(state).physical_digest)
                        break
                except (ValueError,RuntimeError,FloatingPointError) as error:
                    record['attempts'].append(dict(error=str(error),evaluations=count))
                save()
            record['seconds']=perf_counter()-start
            save()
            if record.get('focus_angle_pass'): break


if __name__=='__main__':
    main()
