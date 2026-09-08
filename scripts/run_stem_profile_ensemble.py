"""Run independent single-phonon STEM rasters sequentially and average them.

Every constituent uses the unchanged production CUDA acquisition. The exact
ensemble recipe is this script plus each saved constituent profile; the App
profile with four configurations is explicitly only sampling-equivalent.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter, sleep
from types import SimpleNamespace
import tomllib
import tomli_w
import numpy as np
from run_stem_profile_scan import save_images
from generate_si110_cif_stem_scan import _write_json

ROOT=Path(__file__).resolve().parents[1]


def merge(output, folders, seeds, elapsed):
    all_arrays=[]; records=[]; parameters=[]
    for folder in folders:
        with np.load(folder/'raw_scan.npz') as raw:
            all_arrays.append({key:raw[key].copy() for key in raw.files})
        records.append(json.loads((folder/'metrics.json').read_text()))
        parameters.append(json.loads((folder/'parameters.json').read_text()))
    for key in ('scan_x_nm','scan_y_nm'):
        assert all(np.array_equal(row[key],all_arrays[0][key]) for row in all_arrays)
    means={}; sem={}
    for key in all_arrays[0]:
        if key in ('scan_x_nm','scan_y_nm'):
            means[key]=all_arrays[0][key]
        else:
            stack=np.stack([row[key] for row in all_arrays])
            means[key]=np.mean(stack,axis=0,dtype=np.float64)
            sem[key+'_sem']=np.std(stack,axis=0,ddof=1)/np.sqrt(len(seeds))
    p=dict(parameters[0],mode='run',phonons=len(seeds),
        phonon_label=f"{len(seeds)} independent frozen-phonon configurations; seeds {', '.join(map(str,seeds))}",
        ensemble_seeds=seeds,constituent_directories=[str(folder) for folder in folders],
        ensemble_recipe='Arithmetic intensity mean of complete single-configuration production rasters. Not the same random-realisation sequence as one seed with four configurations.',
        command=[sys.executable,*sys.argv],
        ensemble_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    # The first state is retained as a representative constituent, never
    # presented as a state that produced the whole independent-seed ensemble.
    p['state_scope']='First constituent; exact states are in the constituent parameters.json files.'
    pm=dict(records[0]['production_metrics'])
    for key in list(pm):
        if key.startswith('cuda_'):
            pm.pop(key)
    pm.update(specimen_configuration_count=len(seeds),specimen_thermal_seed=None,
        specimen_thermal_seeds=seeds,ensemble_execution='sequential_independent_single_configuration_rasters',
        cuda_resident_pipeline=all(row['production_metrics']['cuda_resident_pipeline'] for row in records),
        cuda_pipeline_elapsed_s=sum(row['production_metrics']['cuda_pipeline_elapsed_s'] for row in records))
    frame=SimpleNamespace(scan_x_um=means['scan_x_nm']*.001,scan_y_um=means['scan_y_nm']*.001,
        fractions={key:means[key+'_fraction'] for key in ('haadf','df','bf')},
        high_angle_tail_fraction={key:means[key+'_tail_fraction'] for key in ('haadf','df','bf')},
        uncollected_fraction=means['uncollected_fraction'],absorbed_fraction=means['absorbed_fraction'],
        truncated_fraction=means['truncated_fraction'],metrics=pm)
    summary=save_images(frame,output,p)
    # Preserve arithmetic means of every independently stored observable.
    np.savez_compressed(output/'raw_scan.npz',**means)
    np.savez_compressed(output/'ensemble_sem.npz',**sem)
    pm['maximum_probability_conservation_error']=summary['probability_error']
    pm['real_probability_conserved']=summary['probability_error']<=1e-10
    metrics=dict(elapsed_total_s=elapsed,elapsed_acquisition_s=sum(row['elapsed_acquisition_s'] for row in records),
        production_metrics=pm,constituent_metrics_paths=[str(folder/'metrics.json') for folder in folders],
        sem_scope='Standard error of four independent thermal configurations; excludes discretisation, physical-model and counting uncertainty.',**summary)
    _write_json(output/'parameters.json',p);_write_json(output/'metrics.json',metrics)
    nominal=tomllib.loads((folders[0]/'operating_profile.toml').read_text())
    nominal['devices']['sample']['wave_frozen_phonon_configurations']=len(seeds)
    (output/'ensemble_equivalent_profile.toml').write_text(tomli_w.dumps(nominal),encoding='utf-8')
    shutil.copyfile(folders[0]/'input.cif',output/'input.cif')
    (output/'README.md').write_text(
        f"# STEM underfocus ensemble\n\n{p['actual_scan_side']} × {p['actual_scan_side']} pixels, {p['actual_scan_step_nm']:g} nm step; effective C1={p['focus']['effective_defocus_mm']*1e6:g} nm.\n\n"
        f"Arithmetic mean of four independent single-phonon complete rasters, seeds {seeds}. Each child directory contains the exact operating profile, input CIF, raw arrays and metrics. "
        "The ensemble-equivalent App profile reproduces the physical settings and configuration count, but not this independent-seed random-realisation sequence. Rerun this script for the exact recipe.\n\n"
        "raw_scan.npz contains the mean of every total/coherent/tail/uncollected/absorbed/truncated array. ensemble_sem.npz contains thermal-configuration standard errors, not convergence or physical-model error bars. "
        "PNGs are independently min/max scaled and flipped for laboratory +Y-up viewing; TIFF/NPZ retain minimum-Y-first rows and unnormalised emitted-electron fractions. "
        "haadf_df_bf_absolute_scale.png starts each labelled scale at zero; no smoothing or sharpening is applied.\n",
        encoding='utf-8')
    print(json.dumps(dict(complete=True,output=str(output),elapsed_total_s=elapsed,**summary)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[707,708,709,710])
    parser.add_argument('--scan-side',type=int,default=16)
    parser.add_argument('--step-nm',type=float,default=.08)
    parser.add_argument('--grid',type=int,default=2048)
    parser.add_argument('--fov-angstrom',type=float,default=80.)
    parser.add_argument('--padding-factor',type=float,default=1.3)
    parser.add_argument('--effective-c1-nm',type=float,default=-100.)
    args=parser.parse_args()
    if len(args.seeds)<2 or len(set(args.seeds))!=len(args.seeds):parser.error('Distinct seeds are required')
    output=args.output.resolve()
    if output.exists() and any(output.iterdir()):raise FileExistsError('Choose a fresh ensemble output directory')
    output.mkdir(parents=True,exist_ok=True)
    started=perf_counter();folders=[]
    _write_json(output/'ensemble_recipe.json',dict(profile=str(args.profile.resolve()),seeds=args.seeds,
        scan_side=args.scan_side,step_nm=args.step_nm,grid=args.grid,fov_angstrom=args.fov_angstrom,
        effective_c1_nm=args.effective_c1_nm,padding_factor=args.padding_factor,
        execution='sequential complete-raster single-configuration production CUDA acquisitions'))
    for seed in args.seeds:
        folder=output/f'seed_{seed}';folder.mkdir();folders.append(folder)
        command=[sys.executable,str(ROOT/'scripts/run_stem_profile_scan.py'),'--profile',str(args.profile.resolve()),
            '--output',str(folder),'--mode','run','--phonons','1','--seed',str(seed),
            '--scan-side',str(args.scan_side),'--step-nm',str(args.step_nm),'--grid',str(args.grid),
            '--fov-angstrom',str(args.fov_angstrom),'--padding-factor',str(args.padding_factor),
            '--effective-c1-nm',str(args.effective_c1_nm)]
        print(json.dumps(dict(starting_seed=seed,output=str(folder))),flush=True)
        with (folder/'run.log').open('w',encoding='utf-8') as stream:
            process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            last_line=None
            while process.poll() is None:
                path=folder/'progress.jsonl'
                if path.exists():
                    lines=path.read_text(encoding='utf-8').splitlines()
                    if lines and lines[-1]!=last_line:
                        last_line=lines[-1]
                        print(json.dumps(dict(seed=seed,progress=json.loads(last_line))),flush=True)
                sleep(5)
            if process.returncode:
                raise RuntimeError(f'Constituent seed {seed} failed; inspect {folder}/run.log. No partial average was produced.')
    merge(output,folders,args.seeds,perf_counter()-started)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
