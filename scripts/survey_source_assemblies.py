"""Bounded source-specific searches across all upstream assemblies.

This exploratory command preserves the installed source/gun/geometry and
reports unresolved selections explicitly. Candidate values are not defaults.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

from threadpoolctl import threadpool_limits

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.calculation_manifest import solver_source_identity
from temsim.optics.column import default_state
from temsim.optics.assembly_illumination import seed_illumination, TARGETS, diameter_gate
from temsim.optics.illumination_current import aperture_gate, set_calibration_flux
from temsim.simulation_modes import mode_key as simulation_mode_key
from temsim.optics.illumination_search import focus_grid_proposals
from temsim.optics.surface_probe_focus import measure_surface_focus
from calibrate_assembly_illumination import finite_json
from validate_assembly_illumination import fit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gun',choices=('FEG + Mono','Thermionic'),required=True)
    parser.add_argument('--rays',type=int,default=1024)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error('Choose a new report path')
    implementation=solver_source_identity()
    document=dict(implementation=implementation,status='NOT_INSTALLED',records=[])
    def save():
        if solver_source_identity()!=implementation: raise RuntimeError('Solver changed during survey')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(finite_json(document),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with threadpool_limits(1):
        for blanker in ('None','Electrostatic beam blanker'):
            for column in ('C2','C3','C3 + Probe Corrector'):
                for mode in ('nano_probe','micro_probe'):
                    selection=AssemblySelection(args.gun,column,'Energy Filter',blanker)
                    record=dict(assembly=asdict(selection),mode=mode,rays=args.rays,
                                status='NOT_QUALIFIED',attempts=[])
                    document['records'].append(record);save()
                    start=perf_counter()
                    # Low C1 excitations expose the transmitted branch of the
                    # executed broad/monochromated gun; no source is replaced.
                    for c1 in (5.95,3.0):
                        state=default_state();AssemblyCatalog().apply(state,selection)
                        seed_illumination(state,mode)
                        state.electron_gun.emitter.ray_count=args.rays
                        state.condenser_aperture_2.diameter_mm=.25
                        record['simulation_mode']=simulation_mode_key(state)
                        lenses={l.key:l for l in state.lenses}
                        lenses['condenser_lens_1'].percent=c1
                        primary='condenser_lens_2' if column=='C2' else 'condenser_lens_3'
                        keys=(primary,'objective_lens')
                        print('Survey',selection,mode,'C1',c1,flush=True)
                        try:
                            proposals,audit=focus_grid_proposals(state,keys,mode,primary_points=33,
                                objective_points=65,keep=3)
                            attempt=dict(c1_percent=c1,controls=keys,search=audit,
                                         proposals=[asdict(p) for p in proposals],physical=[])
                            record['attempts'].append(attempt);save()
                            for proposal in proposals:
                                for k,v in zip(keys,proposal.values): lenses[k].percent=v
                                try:
                                    count=fit(state,mode,keys,100)
                                    m=measure_surface_focus(state,step_mm=.025)
                                    flux=set_calibration_flux(state,m)
                                    passed=TARGETS[mode].accepts(m) and diameter_gate(state,m,mode)['passed']
                                    attempt['physical'].append(dict(evaluations=count,measurement=asdict(m),passed=passed,
                                        flux_selection=flux,aperture_gate=aperture_gate(state)))
                                    print('Actual',m.statistics.convergence_95_mrad,
                                          m.statistics.illumination_diameter_95_um,passed,flush=True)
                                    if passed:
                                        record.update(controls=keys,strengths={l.key:l.percent for l in state.lenses if l.enabled and l.z_mm<=state.sample.z_mm},
                                            polarities={l.key:l.polarity for l in state.lenses if l.enabled and l.z_mm<=state.sample.z_mm},
                                            c2_aperture_diameter_mm=state.condenser_aperture_2.diameter_mm,
                                            column_current_limit_percent=state.column_current_limit_percent,flux_selection=flux,
                                            fine=asdict(m),focus_angle_pass=True)
                                        break
                                except (ValueError,RuntimeError,FloatingPointError) as error:
                                    attempt['physical'].append(dict(error=str(error)))
                                save()
                        except (ValueError,RuntimeError,FloatingPointError) as error:
                            record['attempts'].append(dict(c1_percent=c1,error=str(error)));save()
                        if record.get('focus_angle_pass'): break
                    record['seconds']=perf_counter()-start;save()
                    print('Complete',selection,mode,'fixed-budget pass',record.get('focus_angle_pass',False),flush=True)


if __name__=='__main__':
    main()
