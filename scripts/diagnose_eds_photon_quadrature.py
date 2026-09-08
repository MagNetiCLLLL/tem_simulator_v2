"""Reuse the one saved broad-beam Si track to inspect angular quadrature."""
from pathlib import Path
from dataclasses import asdict
import json
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from temsim.assembly_catalog import AssemblyCatalog
from temsim.component_keys import EDS_DETECTOR_SYSTEM
from temsim.detector.eds_geometry import EDSDetectorArrayGeometry
from temsim.detector.eds_signal import EDSMaterial,ElectronTrackSegment,simulate_eds_tracks
from temsim.detector.eds_photon_transport import objective_pole_occluders_from_state
from temsim.optics.column import default_state
from temsim.profile_io import read_profile,apply_profile_values


def main():
    output=ROOT/'outputs/eds_defocus_diagnostic/photon_quadrature_consistent'
    output.mkdir(parents=True,exist_ok=True)
    source=ROOT/'outputs/eds_defocus_diagnostic/broad_cached_hybrid_consistent_center/diagnostic.json'
    saved=json.loads(source.read_text())
    state=default_state();catalog=AssemblyCatalog()
    selection,values=read_profile(saved['baseline_profile']);assembly=catalog.apply(state,selection)
    assert apply_profile_values(state,values)==[]
    profile_sample_z=float(state.sample.z_mm)
    state.sample.size_x_nm=state.sample.size_y_nm=10.;state.sample.thickness_nm=5.
    state.sample.centre_x_nm=state.sample.centre_y_nm=0.
    state.sample.envelope_shape='disk';state.sample.inserted=True
    state.sample.eds_support_material_key='vacuum';state.sample.eds_poisson_enabled=False
    state.objective_lens.percent=saved['baseline_field_objective_percent']
    cached_sample_z=saved['cached_sample_z_mm']
    assert state.sample.z_mm==saved['baseline_sample_z_mm']
    geometry=EDSDetectorArrayGeometry.from_part_data(assembly.part(EDS_DETECTOR_SYSTEM).data)
    tracks=[]
    for row in saved['eds_tracks']:
        raw=dict(row);material=dict(raw.pop('material'))
        material['mass_fractions']=tuple(tuple(x) for x in material['mass_fractions'])
        tracks.append(ElectronTrackSegment(material=EDSMaterial(**material),**raw))
    part=assembly.part('sample')
    report=dict(scope='Same saved single material track and baseline dose, no electron retracing. Cached full state unknown.',
        source=str(source),profile_sample_z_mm=profile_sample_z,cached_sample_z_mm=cached_sample_z,
        assembly_sample={key:getattr(part,key,None) for key in ('center_z_mm','start_z_mm','end_z_mm')},
        objective_poles=[asdict(row) for row in objective_pole_occluders_from_state(state)],
        detector_geometry=asdict(geometry),cases=[])
    for order in (1,3,5):
        started=perf_counter()
        spectrum=simulate_eds_tracks(tracks,geometry,
            incident_electrons=saved['emitted_electrons']*saved['surviving_fraction'],
            state=state,photon_origin_xy_nm=(0.,0.),photon_quadrature_order=order,
            photon_maximum_stored_paths=96)
        row=dict(order=order,total_expected_counts=spectrum.total_expected_counts,
            positive_expected_bins=int(np.count_nonzero(spectrum.expected_counts)),
            spectrum_metrics=spectrum.metrics,photon_metrics=spectrum.photon_transport.metrics,
            path_statuses=sorted(set(path.terminal_status for path in spectrum.photon_transport.paths)),
            elapsed_s=perf_counter()-started)
        report['cases'].append(row)
        (output/'diagnostic.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(dict(order=order,expected=row['total_expected_counts'],elapsed_s=row['elapsed_s'],
            blocked_by=row['photon_metrics'].get('blocked_by_component_counts'),
            photon_count=row['photon_metrics']['photon_count'])),flush=True)
    print(json.dumps({key:value for key,value in report.items() if key not in ('cases','objective_poles','detector_geometry')}),flush=True)


if __name__=='__main__':main()
