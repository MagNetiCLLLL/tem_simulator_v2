"""Prove fixed-setting upstream operator equality for downstream variants.

An equal signature is permission to reuse an *executed* illumination result,
not a substitute for that execution. No field beyond the sample is certified.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.calculation_manifest import solver_source_identity
from temsim.immutable_json import freeze_json, json_digest
from temsim.optics.column import default_state
from temsim.optics.assembly_illumination import seed_illumination
from temsim.optics.direct_alignment import _pre_sample_kick_events
from temsim.optics.surface_probe_focus import surface_z_mm
from temsim.physics.core import build_propagation_plan
from temsim.physics.column_wall import _vacuum_segments
from temsim.vacuum import bind_gun_environment
from validate_assembly_illumination import restore_seed


def operator_identity(state, *, step_mm):
    """Exact current optical plan, emitted-gun input and clipping geometry.

    Limited deliberately to classical default optics with vacuum disabled.
    Mapped fields and arbitrary transmission-mask extensions fail closed.
    """
    from temsim.optics.condenser_aperture import ContinuousApertureComponent
    from temsim.simulation_modes import mode_key
    if mode_key(state) not in ('ideal','analytical','custom') or state.vacuum_map.enabled:
        raise ValueError('Equivalence audit requires analytic column fields and inactive vacuum transport')
    if state.electron_gun.source_representation!='classical_particles':
        raise ValueError('Particle illumination equivalence requires the physical classical gun')
    start,end=state.electron_gun.exit_plane_z_mm,surface_z_mm(state)
    planes=(end-.0002,end-.0001,end)
    apertures=list(state.apertures)
    if state.nanopulser.installed: apertures.append(state.nanopulser.aperture)
    active=[a for a in apertures if a.enabled and getattr(a,'installed',True) and start<=a.z_mm<=end]
    contracts=[]
    for aperture in active:
        if hasattr(aperture,'transmission_mask') and not isinstance(aperture,ContinuousApertureComponent):
            raise ValueError('Unsupported custom aperture transmission law in equivalence audit')
        contracts.append(dict(key=aperture.key,z_mm=aperture.z_mm,radius_mm=aperture.radius_mm,
                              offset_x_mm=aperture.offset_x_mm,offset_y_mm=aperture.offset_y_mm))
    plan=build_propagation_plan(state,start,end,events=_pre_sample_kick_events(state),
        save_z_mm=tuple(a.z_mm for a in state.apertures if start<a.z_mm<end)+planes,
        checkpoint_z_mm=planes,maximum_step_mm=step_mm)
    if plan.mapped_fields: raise ValueError('Mapped optical equality requires a separate audit')
    walls=[(max(start,s.start_z_mm),min(end,s.end_z_mm),s.inner_diameter_mm)
           for s in _vacuum_segments(state,[start,end]) if s.end_z_mm>start and s.start_z_mm<end]
    bind_gun_environment(state)
    state.electron_gun.validate()
    # This key contains physical gun geometry, emission and numerical inputs.
    # It is NOT used here to construct a source or replace any gun transport.
    gun=hashlib.sha256(state.electron_gun._cache_key(None).encode()).hexdigest()
    payload=dict(plan=plan.signature,gun=gun,walls=walls,apertures=contracts,
                 upper_surface_mm=end,step_mm=step_mm,
                 column_current_limit_percent=state.column_current_limit_percent,
                 simulation_mode=mode_key(state))
    return dict(signature=json_digest(freeze_json(payload)),**payload)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error('Choose a new evidence path')
    document=dict(implementation=solver_source_identity(),scope='UPSTREAM_FIXED_SETTING_EQUIVALENCE_ONLY',records=[])
    records=json.loads(args.candidates.read_text(encoding='utf-8'))['records']
    for record in records:
        if 'strengths' not in record: continue
        base,selection=restore_seed(record)
        base.electron_gun.emitter.ray_count=record.get('rays',4096)
        reference=[operator_identity(base,step_mm=s) for s in (.05,.025)]
        columns=[selection['column']]
        if selection['column']=='C3': columns.append('C3 + Image Corrector')
        if selection['column']=='C3 + Probe Corrector': columns.append('C3 + Probe Corrector + Image Corrector')
        for column in columns:
            for recording in ('Energy Filter','No Energy Filter'):
                variant=dict(record,assembly=dict(selection,column=column,recording=recording))
                state,chosen=restore_seed(variant)
                state.electron_gun.emitter.ray_count=base.electron_gun.emitter.ray_count
                measured=[operator_identity(state,step_mm=s) for s in (.05,.025)]
                same=all(a['signature']==b['signature'] for a,b in zip(reference,measured))
                row=dict(reference_assembly=selection,assembly=chosen,mode=record['mode'],
                    equivalent=same,reference=reference,operators=measured)
                document['records'].append(row)
                print(chosen,record['mode'],'equivalent',same,flush=True)
                args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.write_text(json.dumps(document,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
