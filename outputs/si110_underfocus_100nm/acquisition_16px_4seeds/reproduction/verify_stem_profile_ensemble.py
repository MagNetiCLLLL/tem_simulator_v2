"""Verify a completed independent-seed STEM ensemble and scope its metadata."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile
import numpy as np
from PIL import Image
import tifffile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from temsim.assembly_catalog import AssemblyCatalog
from temsim.optics.column import default_state
from temsim.profile_io import read_profile,apply_profile_values


def main(folder):
    output=Path(folder).resolve()
    recipe=json.loads((output/'ensemble_recipe.json').read_text())
    folders=[output/f'seed_{seed}' for seed in recipe['seeds']]
    params=[json.loads((path/'parameters.json').read_text()) for path in folders]
    metrics=[json.loads((path/'metrics.json').read_text()) for path in folders]
    arrays=[]
    for path in folders:
        with np.load(path/'raw_scan.npz') as data:
            arrays.append({key:data[key].copy() for key in data.files})
    with np.load(output/'raw_scan.npz') as data:
        mean={key:data[key].copy() for key in data.files}
    with np.load(output/'ensemble_sem.npz') as data:
        sem={key:data[key].copy() for key in data.files}
    side=recipe['scan_side'];step=recipe['step_nm']
    expected_shape=(side,side)
    assert mean['scan_x_nm'].shape==mean['scan_y_nm'].shape==expected_shape
    assert np.allclose(np.diff(mean['scan_x_nm'],axis=1),step,atol=1e-9,rtol=1e-6)
    assert np.allclose(np.diff(mean['scan_y_nm'],axis=0),step,atol=1e-9,rtol=1e-6)
    for key in mean:
        assert mean[key].shape==expected_shape and np.isfinite(mean[key]).all()
        if key in ('scan_x_nm','scan_y_nm'):
            assert all(np.array_equal(row[key],mean[key]) for row in arrays)
        else:
            expected=np.mean([row[key] for row in arrays],axis=0,dtype=np.float64)
            assert np.array_equal(expected,mean[key]),key
            expected_sem=np.std([row[key] for row in arrays],axis=0,ddof=1)/np.sqrt(len(folders))
            assert np.array_equal(expected_sem,sem[key+'_sem']),key
            assert mean[key].min()>=0
    exports={}
    for key in ('haadf','df','bf'):
        values=mean[key+'_fraction']; tail=mean[key+'_tail_fraction']; coherent=mean[key+'_coherent_scaled_fraction']
        assert np.allclose(values,tail+coherent,atol=1e-15,rtol=1e-13)
        assert np.array_equal(tifffile.imread(output/f'{key}.tiff'),values.astype(np.float32))
        scaled=np.round(np.clip((values-values.min())/max(values.max()-values.min(),1e-30),0,1)*65535).astype(np.uint16)
        assert np.array_equal(np.asarray(Image.open(output/f'{key}.png')),np.flipud(scaled))
        exports[key]=dict(float32_tiff_exact=True,png_linear_mapping_and_y_flip_exact=True,
            coherent_tail_max_sum_error=float(np.max(abs(values-tail-coherent))))
    budget=sum(mean[key+'_fraction'] for key in ('haadf','df','bf'))+mean['uncollected_fraction']+mean['absorbed_fraction']
    budget_error=float(np.max(abs(budget-1)))
    assert budget_error<=1e-10
    expected_hashes=params[0]['implementation']['sha256_by_file']
    combined=json.loads((output/'metrics.json').read_text())
    combined_params=json.loads((output/'parameters.json').read_text())
    archive=output/'reproduction/implementation_snapshot.zip'
    with zipfile.ZipFile(archive) as z:
        assert all(hashlib.sha256(z.read(name)).hexdigest()==digest for name,digest in expected_hashes.items())
        assert all(hashlib.sha256(z.read('scripts/run_stem_profile_scan.py')).hexdigest()==p['script_sha256'] for p in params)
        assert hashlib.sha256(z.read('scripts/run_stem_profile_ensemble.py')).hexdigest()==combined_params['ensemble_script_sha256']
    assert hashlib.sha256((output/'input.cif').read_bytes()).hexdigest()==params[0]['cif_sha256']
    source_cif=Path(params[0]['source_cif'])
    assert hashlib.sha256(source_cif.read_bytes()).hexdigest()==params[0]['cif_sha256']
    profiles=[]
    for path,p,m,seed in zip(folders,params,metrics,recipe['seeds']):
        assert p['implementation']['sha256_by_file']==expected_hashes
        assert p['seed']==seed and p['phonons']==1
        assert m['production_metrics']['cuda_resident_pipeline'] is True
        assert m['production_metrics']['wave_compute_backend']=='CuPy CUDA'
        assert m['production_metrics']['specimen_configuration_count']==1
        assert abs(m['production_metrics']['probe_effective_defocus_nm']-recipe['effective_c1_nm'])<1e-8
        assert abs(p['configured_c1_nm']-p['probe']['waist_offset_m']*1e9-recipe['effective_c1_nm'])<1e-8
        assert hashlib.sha256((path/'input.cif').read_bytes()).hexdigest()==p['cif_sha256']==params[0]['cif_sha256']
        state=default_state();catalog=AssemblyCatalog();selection,values=read_profile(path/'operating_profile.toml')
        catalog.apply(state,selection); skipped=apply_profile_values(state,values);assert skipped==[]
        assert state.sample.wave_frozen_phonon_seed==seed
        assert state.sample.wave_frozen_phonon_configurations==1
        assert state.ac_deflector.scan_pixels_x==side and state.ac_deflector.scan_lines==side
        assert state.ac_deflector.scan_pixel_size_nm==step
        assert state.sample.wave_grid_pixels==recipe['grid']
        assert state.sample.wave_field_of_view_angstrom==recipe['fov_angstrom']
        assert abs(state.sample.wave_defocus_nm-p['configured_c1_nm'])<1e-10
        restored=json.loads(json.dumps(state.to_dict()))
        preserved_sections=('lenses','apertures','deflectors','sample','recording_planes',
                            'probe_aberrations','image_aberrations')
        assert all(restored[key]==p['state'][key] for key in preserved_sections)
        profiles.append(dict(seed=seed,zero_skipped=True,effective_c1_nm=m['production_metrics']['probe_effective_defocus_nm'],
            configured_c1_nm=state.sample.wave_defocus_nm,all_startup_source_hashes_match=True,
            restored_state_sections_exact=list(preserved_sections)))
    selection,values=read_profile(output/'ensemble_equivalent_profile.toml')
    state=default_state();AssemblyCatalog().apply(state,selection);assert apply_profile_values(state,values)==[]
    assert state.sample.wave_frozen_phonon_configurations==len(folders)
    equivalent=json.loads(json.dumps(state.to_dict()))
    equivalent['sample']['wave_frozen_phonon_configurations']=1
    assert all(equivalent[key]==params[0]['state'][key] for key in preserved_sections)
    pm=combined['production_metrics']
    maxima=[key for key in pm if ('maximum' in key or key.startswith('max_'))
            and all(isinstance(row['production_metrics'].get(key),(float,int)) and not isinstance(row['production_metrics'].get(key),bool) for row in metrics)]
    for key in maxima:
        pm[key]=max(row['production_metrics'][key] for row in metrics)
    pm['maximum_probability_conservation_error']=budget_error
    pm['mean_uncollected_fraction']=float(mean['uncollected_fraction'].mean())
    pm['mean_truncated_source_fraction']=float(mean['truncated_fraction'].mean())
    pm['detector_configuration_relative_standard_error']={key:float(np.linalg.norm(sem[key+'_fraction_sem'])/max(np.linalg.norm(mean[key+'_fraction']),1e-30)) for key in ('haadf','df','bf')}
    pm['metadata_scope']={
        'shared_physical_settings':'Grid, geometry, C1, specimen model and detector bands match the constituent runs.',
        'maxima_over_all_constituents':maxima,
        'ensemble_statistics':'Image means, source uncollected/truncated means, conservation and thermal relative SEM are computed from the four-constituent ensemble.',
        'remaining_diagnostics':'Other diagnostics describe the first constituent; individual complete diagnostics are in constituent_metrics_paths.'}
    (output/'metrics.json').write_text(json.dumps(combined,indent=2),encoding='utf-8')
    verification=dict(passed=True,shape=list(expected_shape),step_nm=step,configuration_seeds=recipe['seeds'],
        arithmetic_means_exact=True,standard_errors_exact=True,all_arrays_finite=True,all_signal_arrays_nonnegative=True,
        probability_error=budget_error,exports=exports,profiles=profiles,equivalent_profile_zero_skipped=True,
        equivalent_profile_scope='Physical settings/configuration count only, not the independent-seed realisation sequence.',
        source_cif_sha256=params[0]['cif_sha256'],source_cif_path=str(source_cif),source_cif_copy_exact=True,
        startup_hash_count=len(expected_hashes),all_constituent_source_hashes_match=True,
        archived_acquisition_and_ensemble_script_hashes_match=True,
        verifier_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        artifact_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in output.iterdir() if path.is_file() and path.suffix in ('.npz','.png','.tiff','.toml')})
    (output/'output_verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    shutil.copyfile(__file__,output/'reproduction/verify_stem_profile_ensemble.py')
    print(json.dumps({key:value for key,value in verification.items() if key!='artifact_sha256'},indent=2))


if __name__=='__main__':
    main(sys.argv[1])
