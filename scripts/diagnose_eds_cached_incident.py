"""Quantify a cached App incident bundle with an explicit baseline EDS state.

Cached incident data contains no complete state. This is not a reconstruction
of the user's EDS settings or an identification of the cached objective value.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import shutil
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from temsim.assembly_catalog import AssemblyCatalog
from temsim.component_keys import EDS_DETECTOR_SYSTEM
from temsim.detector.eds_geometry import EDSDetectorArrayGeometry
from temsim.detector.eds_signal import simulate_eds_point,default_eds_dwell_time_s,default_eds_incident_electrons
from temsim.optics.column import default_state
from temsim.profile_io import read_profile,apply_profile_values
from temsim.specimen.elastic_transport import incident_rays_from_simulation,simulate_elastic_point_transport


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=int,default=0)
    parser.add_argument('--origin',choices=['center','raw'],default='center')
    parser.add_argument('--field-percent',type=float,default=68.0)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/eds_defocus_diagnostic/latest_cached_baseline_eds')
    args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    refs=json.loads((ROOT/'outputs/eds_defocus_diagnostic/recent_actual_cached_incident.json').read_text())
    manifest_path=Path(refs[args.index]['path']);manifest=json.loads(manifest_path.read_text())
    def load(name):
        return np.load(manifest_path.parent/manifest['arrays']['incident.'+name]['file'],mmap_mode='r',allow_pickle=False)
    rows={key:np.asarray(load(key)[-1:]).copy() for key in ('x','y','tx','ty')}
    rows.update({key:np.asarray(load(key)).copy() for key in ('alive','blocked_z','energy_offset_ev','ray_weight')})
    rows['z']=np.asarray(load('z')[-1:]).copy()
    np.savez_compressed(output/'cached_incident_sample_plane.npz',**rows)
    state=default_state();catalog=AssemblyCatalog()
    profile=ROOT/'outputs/haadf_dpa_clearance/si110_haadf_dpa_12mm.toml'
    selection,values=read_profile(profile);assembly=catalog.apply(state,selection)
    assert apply_profile_values(state,values)==[]
    shutil.copyfile(profile,output/'baseline_profile.toml')
    state.sample.size_x_nm=state.sample.size_y_nm=10.;state.sample.thickness_nm=5.
    state.sample.centre_x_nm=state.sample.centre_y_nm=0.
    state.sample.envelope_shape='disk';state.sample.inserted=True
    state.sample.eds_support_material_key='vacuum';state.sample.eds_poisson_enabled=False
    state.objective_lens.percent=args.field_percent
    state.ac_deflector.scan_enabled=False
    cached_sample_z=float(rows['z'][0])
    baseline_sample_z=float(state.sample.z_mm)
    # The cache does not retain its assembly/state. Reanchor the phase-space
    # reference plane to the intact baseline assembly, never move only sample
    # relative to its poles. This is a disclosed hybrid, not the cached state.
    rows['z']=np.asarray([baseline_sample_z])
    geometry=EDSDetectorArrayGeometry.from_part_data(assembly.part(EDS_DETECTOR_SYSTEM).data)
    targets=dict(target_x_nm=0.,target_y_nm=0.) if args.origin=='center' else {}
    bundle=incident_rays_from_simulation(state,SimpleNamespace(incident=SimpleNamespace(**rows)),**targets)
    started=perf_counter();last=[-20.]
    def progress(done,total,message):
        elapsed=perf_counter()-started
        if elapsed-last[0]<10 and done!=total:return
        last[0]=elapsed
        record=dict(elapsed_s=elapsed,done=done,total=total,message=message)
        with (output/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    result=simulate_elastic_point_transport(state,incident_rays=bundle.rays,stored_trajectory_count=49,progress_callback=progress)
    ids={row.source_ray_index for row in result.eds_tracks if row.source_key=='sample' and row.path_length_nm>0}
    preliminary=dict(scope='Actual cached sample-plane rays plus saved baseline Si/field/current/dwell/detector settings; no cached state available, objective identity unknown.',
        cached_manifest=str(manifest_path),cached_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        baseline_profile=str(profile),baseline_field_objective_percent=args.field_percent,
        cached_sample_z_mm=cached_sample_z,baseline_sample_z_mm=baseline_sample_z,
        global_z_reanchor_mm=baseline_sample_z-cached_sample_z,
        ray_count=bundle.emitted_ray_count,reaching_count=bundle.reaching_ray_count,
        surviving_fraction=bundle.surviving_fraction,original_centroid_nm=bundle.original_centroid_nm,
        point_origin=args.origin,target_centroid_nm=bundle.target_centroid_nm,material_hit_count=len(ids),eds_track_count=len(result.eds_tracks),
        dwell_time_s=default_eds_dwell_time_s(state),emitted_electrons=default_eds_incident_electrons(state,default_eds_dwell_time_s(state)),
        elastic_metrics=result.metrics,eds_tracks=[asdict(row) for row in result.eds_tracks])
    (output/'diagnostic.json').write_text(json.dumps(preliminary,indent=2),encoding='utf-8')
    print(json.dumps({key:value for key,value in preliminary.items() if key!='elastic_metrics'}),flush=True)
    spectrum=simulate_eds_point(state,geometry,incident_bundle=bundle,elastic_transport=result,
        x_nm=0.,y_nm=0.,photon_maximum_stored_paths=0,progress_callback=progress)
    np.savez_compressed(output/'expected_spectrum.npz',energy_ev=spectrum.energy_bin_centres_ev,expected_counts=spectrum.expected_counts)
    preliminary.update(eds_metrics=spectrum.metrics,
        photon_metrics=spectrum.photon_transport.metrics if spectrum.photon_transport is not None else None,
        total_expected_counts=spectrum.total_expected_counts,
        positive_expected_bins=int(np.count_nonzero(spectrum.expected_counts>0)),line_contribution_count=len(spectrum.lines),
        poisson_zero_total_probability=math.exp(-spectrum.total_expected_counts),elapsed_total_s=perf_counter()-started)
    (output/'diagnostic.json').write_text(json.dumps(preliminary,indent=2),encoding='utf-8')
    print(json.dumps({key:value for key,value in preliminary.items() if key not in ('elastic_metrics','eds_metrics','photon_metrics','eds_tracks')}),flush=True)


if __name__=='__main__':main()
