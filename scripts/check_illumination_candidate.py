"""Re-execute a reported candidate; scalar evidence only, never install it."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.optics.column import default_state
from temsim.optics.assembly_illumination import seed_illumination, TARGETS, diameter_gate
from temsim.optics.illumination_current import aperture_gate, current_gate
from temsim.simulation_modes import mode_key
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.calculation_manifest import solver_source_identity
from temsim.optics.surface_probe_focus import measure_surface_focus
from temsim.optics.beam_path_audit import (
    incident_checkpoints, crossover_candidates, optical_component_planes,
    refine_crossover_candidate, crossover_intervals,
)
from calibrate_assembly_illumination import finite_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--rays', type=int, nargs='+', default=[1024, 2048])
    parser.add_argument('--topology', action='store_true')
    parser.add_argument('--paired', action='store_true', help='Check the other illumination mode at the final ray budget, reusing its identical physical gun')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new evidence path')
    record = json.loads(args.candidate.read_text(encoding='utf-8'))['records'][args.index]
    state = default_state()
    AssemblyCatalog().apply(state, AssemblySelection(**record['assembly']))
    seed_illumination(state, record['mode'])
    state.condenser_aperture_2.diameter_mm = record['c2_aperture_diameter_mm']
    state.column_current_limit_percent = record.get('column_current_limit_percent',100.)
    for lens in state.lenses:
        if lens.key in record['strengths']:
            lens.percent = record['strengths'][lens.key]
        if lens.key in record.get('polarities', {}):
            lens.polarity = record['polarities'][lens.key]
    implementation = solver_source_identity()
    report = dict(status='NOT_INSTALLED', implementation=implementation,
                  candidate=str(args.candidate), index=args.index,
                  assembly=record['assembly'], mode=record['mode'], observations=[])

    def evidence(state, mode, rays, measurement):
        if solver_source_identity() != implementation:
            raise RuntimeError('Solver source changed during validation; rerun with fixed inputs')
        return dict(mode=mode, rays=rays, measurement=asdict(measurement),
                    simulation_mode=mode_key(state), aperture_gate=aperture_gate(state),
                    current_gate=current_gate(state,measurement),
                    executed_input_digest=capture_instrument_snapshot(state).physical_digest,
                    focus_angle_pass=TARGETS[mode].accepts(measurement),
                    diameter_gate=diameter_gate(state, measurement, mode))

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(finite_json(report), indent=2, allow_nan=False)+'\n', encoding='utf-8')

    with threadpool_limits(1):
        for rays in args.rays:
            state.electron_gun.emitter.ray_count = rays
            for step in (.05, .025):
                measurement = measure_surface_focus(state, step_mm=step)
                row = evidence(state, record['mode'], rays, measurement)
                report['observations'].append(row)
                print(json.dumps(finite_json(row)), flush=True)
                save()
        if args.topology:
            end = state.sample.upper_surface_z_mm
            planes = np.unique(np.r_[np.arange(state.electron_gun.exit_plane_z_mm, end, 1.), end])
            gun, checkpoints, masks = incident_checkpoints(state, planes, step_mm=.05)
            proposals = crossover_candidates(checkpoints, masks, gun.exit_bundle.weight)
            report['crossover_proposals'] = proposals
            report['component_planes'] = optical_component_planes(state)
            report['crossovers'] = []
            save()
            for proposal in proposals:
                row = refine_crossover_candidate(state, proposal, step_mm=.05)
                report['crossovers'].append(row)
                print(json.dumps(finite_json(row)), flush=True)
                save()
            report['intervals'] = crossover_intervals([r['z_mm'] for r in report['crossovers']],
                                                     report['component_planes'])
            save()
        if args.paired:
            report['paired'] = []
            source_records = json.loads(args.candidate.read_text(encoding='utf-8'))['records']
            for other in source_records:
                if other['assembly'] != record['assembly'] or other['mode'] == record['mode']:
                    continue
                paired = default_state()
                AssemblyCatalog().apply(paired, AssemblySelection(**other['assembly']))
                seed_illumination(paired, other['mode'])
                paired.electron_gun.emitter.ray_count = args.rays[-1]
                paired.condenser_aperture_2.diameter_mm = other['c2_aperture_diameter_mm']
                paired.column_current_limit_percent = other.get('column_current_limit_percent',100.)
                for lens in paired.lenses:
                    if lens.key in other['strengths']:
                        lens.percent = other['strengths'][lens.key]
                    if lens.key in other.get('polarities', {}):
                        lens.polarity = other['polarities'][lens.key]
                for step in (.05, .025):
                    m = measure_surface_focus(paired, step_mm=step)
                    row = evidence(paired, other['mode'], args.rays[-1], m)
                    report['paired'].append(row)
                    print(json.dumps(finite_json(row)), flush=True)
                    save()


if __name__ == '__main__':
    main()
