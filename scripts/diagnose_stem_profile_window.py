"""Measure incident-probe edge energy and real detector masks, without multislice."""
from __future__ import annotations
import gc
import json
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from temsim.assembly_catalog import AssemblyCatalog
from temsim.optics.column import default_state
from temsim.physics.stem_wave_imaging import _probe_spectrum
from temsim.physics.record_plane import build_record_plane_plan,prepare_record_plane_detector_masks
from temsim.physics.scan_geometry import raster_sample_grid
from temsim.profile_io import read_profile,apply_profile_values


def diagnose(folder):
    folder=Path(folder)
    p=json.loads((folder/'parameters.json').read_text())
    m=json.loads((folder/'metrics.json').read_text())['production_metrics']
    raw=np.load(folder/'raw_scan.npz')
    state=default_state(); catalog=AssemblyCatalog()
    selection,values=read_profile(folder/'operating_profile.toml')
    catalog.apply(state,selection); assert not apply_profile_values(state,values)
    n=int(m['grid_pixels_x']); length=float(m['field_of_view_angstrom'])
    spacing=length/n; wavelength=float(m['wavelength_angstrom'])
    frequencies=np.fft.fftshift(np.fft.fftfreq(n,d=spacing))
    spectrum=_probe_spectrum(state,p['probe'],frequencies,frequencies,wavelength)
    fx,fy=np.meshgrid(frequencies,frequencies,indexing='xy')
    axis_nm=(np.arange(n)-n//2)*spacing*.1
    edge_dist=np.minimum(axis_nm+length*.05,length*.05-axis_nm)
    edge_masks={str(width):np.minimum(edge_dist[:,None],edge_dist[None,:])<width for width in (.5,1.)}
    positions=[(1,1),(0,0),(0,2),(2,0),(2,2)]
    entries=[]
    for iy,ix in positions:
        x=float(raw['scan_x_nm'][iy,ix]);y=float(raw['scan_y_nm'][iy,ix])
        shifted=spectrum*np.exp(-2j*np.pi*(fx*x*10+fy*y*10))
        wave=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(shifted)))
        intensity=abs(wave)**2;intensity/=intensity.sum()
        entries.append(dict(index=[iy,ix],position_nm=[x,y],
            edge_strip_energy={width:float(intensity[mask].sum()) for width,mask in edge_masks.items()}))
    del spectrum,shifted,wave,intensity,edge_masks
    gc.collect()
    angles=np.stack((np.arcsin(np.clip(wavelength*fx,-1,1)),np.arcsin(np.clip(wavelength*fy,-1,1))),axis=-1)
    valid=np.hypot(fx,fy)<=np.sin(m['maximum_isotropic_angle_mrad']*.001)/wavelength
    _,_,times=raster_sample_grid(state.ac_deflector,maximum_count=None)
    plan=build_record_plane_plan(state,scan_times_s=times,recalibrate_scan=True)
    masks=prepare_record_plane_detector_masks(plan,angles)
    origin=np.asarray(m['specimen_calculation_roi_centre_nm'])
    dtheta=wavelength/length*1000
    for entry,(iy,ix) in zip(entries,positions):
        pos=np.asarray(entry['position_nm'])+origin
        all_masks=masks(pos.reshape(1,1,1,2)*1e-9,scan_slice=slice(iy*3+ix,iy*3+ix+1))
        entry['detectors']={key:dict(pixel_count=int(np.count_nonzero(all_masks[key]&valid)),
            pixel_count_times_small_angle_area_mrad2=float(np.count_nonzero(all_masks[key]&valid)*dtheta**2))
            for key in ('haadf','df','bf')}
    result=dict(scope='Production incident _probe_spectrum and record-plane masks; no specimen propagation. Edge energy measures periodic-window strips, not an exact outside-domain probability.',
        grid=n,fov_nm=length*.1,reciprocal_pixel_step_mrad_small_angle=dtheta,
        reciprocal_pixel_area_mrad2_small_angle=dtheta**2,positions=entries)
    (folder/'probe_boundary_mask_diagnostic.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(folder=str(folder),**result),indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        for folder in sys.argv[1:]:
            diagnose(folder)
