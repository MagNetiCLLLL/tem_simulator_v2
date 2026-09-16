"""Refit seed reports and validate independent particle/step budgets.

Runs detached classical particle optics, reusing the executed physical gun
within this process. Writes scalar evidence, never ray arrays or GUI settings.
New crossover baselines are explicitly allowed for nonhistorical assemblies.
"""
import argparse
from dataclasses import asdict
import json
import tomllib
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
from temsim.optics.illumination_current import aperture_gate, current_gate, set_calibration_flux
from temsim.simulation_modes import mode_key as simulation_mode_key
from temsim.optics.illumination_checkpoint import IlluminationCheckpoint
from temsim.optics.surface_probe_focus import measure_surface_focus
from temsim.optics.beam_path_audit import (
    incident_checkpoints, crossover_candidates, optical_component_planes,
    refine_crossover_candidate, crossover_intervals, require_same_topology,
)
from calibrate_assembly_illumination import finite_json


def restore_seed(record, *, blanker=None):
    selection = dict(record['assembly'])
    if blanker is not None:
        selection['beam_blanker'] = blanker
    state = default_state()
    AssemblyCatalog().apply(state, AssemblySelection(**selection))
    seed_illumination(state, record['mode'])
    state.condenser_aperture_2.diameter_mm = record['c2_aperture_diameter_mm']
    state.column_current_limit_percent = record.get('column_current_limit_percent', 100.)
    for lens in state.lenses:
        if lens.key in record['strengths']:
            lens.percent = record['strengths'][lens.key]
        if lens.key in record.get('polarities', {}):
            lens.polarity = record['polarities'][lens.key]
    return state, selection


def fit(state, mode, keys, maximum_evaluations):
    checkpoint = IlluminationCheckpoint(state, keys, step_mm=.05)
    lenses = {l.key: l for l in state.lenses}
    upper = np.array([lenses[k].max_percent for k in keys])
    initial = np.array([lenses[k].percent for k in keys])
    cache, count = {}, 0
    target = TARGETS[mode]

    def measure(vector):
        nonlocal count
        identity = tuple(map(float, vector))
        if identity not in cache:
            if count >= maximum_evaluations:
                raise RuntimeError('Physical trajectory budget exhausted')
            count += 1
            cache[identity] = checkpoint.measure(vector)
        return cache[identity]

    if mode == 'nano_probe':
        vector = _focus_branch(measure, target, initial, upper)
    else:
        solution = least_squares(lambda v: target.residual(measure(v).statistics), initial,
            bounds=(np.zeros(2), upper), diff_step=1e-6, x_scale='jac',
            max_nfev=maximum_evaluations, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        vector = solution.x
    for key, value in zip(keys, vector):
        lenses[key].percent = float(value)
    return count


def convergence_gate(observations, mode):
    """Compare fixed settings, never refit each sampling/step observation."""
    target = TARGETS[mode]
    values = np.array([[o['measurement']['statistics']['convergence_95_rad'],
                        o['measurement']['statistics']['radius_95_m'],
                        o['measurement']['statistics']['surviving_fraction']]
                       for o in observations], dtype=float)
    if len(values) < 2 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        return dict(passed=False, reason='At least two finite positive measurements are required')
    variation = np.ptp(values, axis=0) / np.max(values, axis=0)
    return dict(maximum_relative_change=target.relative_tolerance,
        alpha_relative_change=float(variation[0]), diameter_relative_change=float(variation[1]),
        transmission_relative_change=float(variation[2]),
        passed=bool(np.all(variation <= target.relative_tolerance)))


def topology(state):
    end = state.sample.upper_surface_z_mm
    planes = np.unique(np.r_[np.arange(state.electron_gun.exit_plane_z_mm, end, 1.), end])
    gun, cp, masks = incident_checkpoints(state, planes, step_mm=.05)
    proposals = crossover_candidates(cp, masks, gun.exit_bundle.weight)
    rows = [refine_crossover_candidate(state, p, step_mm=.05) for p in proposals]
    components = optical_component_planes(state)
    return dict(crossovers=rows, component_planes=components,
                intervals=crossover_intervals([r['z_mm'] for r in rows], components))


def reference_microprobe_needs_baseline_approval(selection, mode, approved=False):
    return (not approved and selection['gun']=='FEG'
        and selection['column']=='C3 + Probe Corrector'
        and selection['beam_blanker']=='None' and mode=='micro_probe')


def reference_microprobe_baseline():
    from temsim.paths import OPERATING_MODE_CONFIG_ROOT
    with (OPERATING_MODE_CONFIG_ROOT/'illumination_targets.toml').open('rb') as stream:
        return tomllib.load(stream)['crossover_baselines']['feg_c3_probe_corrector_microprobe']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=Path, nargs='+', required=True)
    parser.add_argument('--rays', type=int, nargs='+', default=[2048,4096])
    parser.add_argument('--fit-rays', type=int, default=4096)
    parser.add_argument('--both-blankers', action='store_true')
    parser.add_argument('--topology', action='store_true')
    parser.add_argument('--refine-gun-step', action='store_true')
    parser.add_argument('--allow-reference-microprobe-baseline',action='store_true',
        help='Explicit approval override; the recorded six-crossover baseline must still match')
    parser.add_argument('--max-evaluations', type=int, default=160)
    parser.add_argument('--column', nargs='+', help='Validate only these installed upstream columns')
    parser.add_argument('--mode', choices=('nano_probe','micro_probe'), help='Validate only this illumination mode')
    parser.add_argument('--skip-refit', action='store_true', help='Keep the seed lens values fixed; still select explicit source flux once')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Choose a new evidence path')
    if len(set(args.rays)) < 2 or min(args.rays) < 32 or args.fit_rays < 32:
        parser.error('At least two distinct positive validation ray budgets are required')
    # Later explicit seed reports supersede earlier ones for the same assembly/mode.
    seeds = {}
    for path in args.seeds:
        for row in json.loads(path.read_text(encoding='utf-8'))['records']:
            if args.column and row['assembly']['column'] not in args.column: continue
            if args.mode and row['mode'] != args.mode: continue
            if (row.get('focus_angle_pass') or row.get('sampling_and_column_step_pass')) and 'strengths' in row:
                key = tuple(row['assembly'][k] for k in ('gun','column','beam_blanker','recording'))+(row['mode'],)
                seeds[key] = row
    if not seeds:
        parser.error('No eligible executed candidate matched the selected assembly/mode')
    document = dict(schema='assembly-illumination-validation-v1', implementation=solver_source_identity(),
                    status='NOT_INSTALLED', records=[])

    def save():
        if solver_source_identity() != document['implementation']:
            raise RuntimeError('Solver changed during validation')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(finite_json(document), indent=2, allow_nan=False)+'\n', encoding='utf-8')

    with threadpool_limits(1):
        for seed in seeds.values():
            blankers = ('None','Electrostatic beam blanker') if args.both_blankers else (None,)
            for blanker in blankers:
                start = perf_counter()
                state, selection = restore_seed(seed, blanker=blanker)
                mode = seed['mode']
                keys = seed.get('controls', seed.get('variable_lenses'))
                if keys is None: raise ValueError('Seed lacks declared fitting controls')
                row = dict(assembly=selection, mode=mode, controls=keys, observations=[], status='NOT_QUALIFIED',
                    simulation_mode=simulation_mode_key(state),
                    modelling_scope='Classical tip-origin particle illumination; no wave-image validation')
                document['records'].append(row); save()
                print('Fitting', selection, mode, 'rays', args.fit_rays, flush=True)
                try:
                    state.electron_gun.emitter.ray_count = args.fit_rays
                    if not aperture_gate(state)['passed']:
                        raise ValueError('The physical C2 aperture is outside the enabled 20-250 um range')
                    row['fit_evaluations'] = 0 if args.skip_refit else fit(state, mode, keys, args.max_evaluations)
                    calibration = measure_surface_focus(state, step_mm=.05)
                    row['flux_selection'] = set_calibration_flux(state, calibration)
                    row['column_current_limit_percent'] = state.column_current_limit_percent
                    row['aperture_gate'] = aperture_gate(state)
                    row.update(strengths={l.key:l.percent for l in state.lenses if l.enabled and l.z_mm<=state.sample.z_mm},
                        polarities={l.key:l.polarity for l in state.lenses if l.enabled and l.z_mm<=state.sample.z_mm},
                        c2_aperture_diameter_mm=state.condenser_aperture_2.diameter_mm)
                    save()
                    for rays in sorted(set(args.rays)):
                        state.electron_gun.emitter.ray_count = rays
                        for step in (.05,.025):
                            m = measure_surface_focus(state, step_mm=step)
                            gate = diameter_gate(state, m, mode)
                            result = dict(rays=rays, gun_step_mm=state.electron_gun.trace_step_mm,
                                measurement=asdict(m), target_pass=TARGETS[mode].accepts(m), diameter_gate=gate,
                                current_gate=current_gate(state,m),
                                executed_input_digest=capture_instrument_snapshot(state).physical_digest)
                            row['observations'].append(result); save()
                            print('Measured',mode,rays,step,'alpha',m.statistics.convergence_95_mrad,
                                  'D95 um',m.statistics.illumination_diameter_95_um,'target',result['target_pass'],flush=True)
                    row['convergence_gate'] = convergence_gate(row['observations'], mode)
                    row['sampling_and_column_step_pass'] = all(o['target_pass'] and o['diameter_gate']['passed']
                        and o['current_gate']['passed'] for o in row['observations']) and row['convergence_gate']['passed']
                    if args.refine_gun_step:
                        state.electron_gun.trace_step_mm *= .5
                        m = measure_surface_focus(state, step_mm=.025)
                        gate = diameter_gate(state,m,mode)
                        row['gun_step_check'] = dict(rays=state.electron_gun.emitter.ray_count,
                            gun_step_mm=state.electron_gun.trace_step_mm, measurement=asdict(m),
                            current_gate=current_gate(state,m),
                            passed=TARGETS[mode].accepts(m) and gate['passed'] and current_gate(state,m)['passed'])
                        row['gun_step_check']['convergence_gate'] = convergence_gate(
                            [row['observations'][-1],row['gun_step_check']],mode)
                        row['gun_step_check']['passed'] &= row['gun_step_check']['convergence_gate']['passed']
                        state.electron_gun.trace_step_mm *= 2
                        save()
                    if args.topology:
                        row['topology'] = topology(state)
                        row['topology_policy'] = 'NEW_ASSEMBLY_BASELINE_AUTHORISED_2026_09_17'
                        if (selection['gun']=='FEG' and selection['column']=='C3 + Probe Corrector'
                                and selection['beam_blanker']=='None' and mode=='nano_probe'):
                            reference=json.loads(Path('docs/development/tip_curvature_193_20260915.json').read_text(encoding='utf-8'))['reports']['Flat']
                            # The terminal surface focus has its own signed 1 nm gate;
                            # compare the five established intermediate crossovers.
                            reference_roots=[p['z_mm'] for p in reference['column_crossovers'] if p['z_mm']<state.sample.z_mm-.01]
                            actual_roots=[p['z_mm'] for p in row['topology']['crossovers'] if p['z_mm']<state.sample.z_mm-.01]
                            require_same_topology(reference_roots,actual_roots,row['topology']['component_planes'])
                            row['topology_policy']='HISTORICAL_INTERMEDIATE_TOPOLOGY_PRESERVED'
                        save()
                    row['pending_gates'] = ([] if args.topology else ['crossover_topology'])+([] if args.refine_gun_step else ['gun_step'])
                    if reference_microprobe_needs_baseline_approval(selection,mode):
                        baseline=reference_microprobe_baseline()
                        approved=args.allow_reference_microprobe_baseline or baseline.get('authorised') is True
                        if not approved:
                            row['pending_gates'].append('reference_microprobe_baseline_approval')
                            row['topology_policy']='REFERENCE_MICROPROBE_BASELINE_REQUIRES_CONFIRMATION'
                        elif args.topology:
                            if (len(row['topology']['crossovers']) != baseline['count']
                                    or json.loads(json.dumps(row['topology']['intervals'])) != baseline['intervals']):
                                raise ValueError('The authorised microprobe crossover count/order/intervals changed')
                            row['topology_policy']='USER_AUTHORISED_INDEPENDENT_MICROPROBE_BASELINE_2026_09_17'
                    row['failed_gates'] = ([] if row['sampling_and_column_step_pass'] else ['sampling_or_illumination'])
                    if args.refine_gun_step and not row['gun_step_check']['passed']:
                        row['failed_gates'].append('gun_step')
                    if not row['pending_gates'] and not row['failed_gates']:
                        row['status']='VALIDATED_PARTICLE_ILLUMINATION'
                except (ValueError,RuntimeError,FloatingPointError,KeyError) as error:
                    row['error']=str(error)
                row['seconds']=perf_counter()-start; save()
                print('Result',row['status'],row.get('error',row.get('failed_gates')),flush=True)


if __name__=='__main__':
    main()
