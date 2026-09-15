"""Summarise executed scalar reports; no transport, fitting or cached ray output."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from temsim.optics.beam_path_audit import require_same_topology
from check_gun_matching_candidate import write_report


CRITERIA = dict(target_alpha95_mrad=30., angle_relative_tolerance=.01,
    waist_tolerance_nm=1., maximum_diameter95_nm=5.,
    diameter_relative_change_limit=.10, current_relative_change_limit=.05,
    minimum_effective_rays=16)


def focus_row(report, reference=None):
    """Predeclared engineering tolerances, not an experimental certification."""
    f = report['focus']
    s = f['statistics']
    row = dict(rays=s['surviving_rays'], effective_rays=f['effective_rays'],
        alpha95_mrad=s['convergence_95_rad']*1000,
        diameter95_nm=s['radius_95_m']*2e9,
        source_fraction=s['surviving_fraction'],
        local_waist_offset_nm=f['local_waist_offset_nm'],
        covariance_waist_offset_nm=s['waist_offset_m']*1e9,
        surface_z_mm=f['surface_z_mm'], step_mm=f['step_mm'])
    if not np.isfinite(list(row.values())).all():
        raise ValueError('Nonfinite focus evidence')
    c = CRITERIA
    passed = (abs(row['alpha95_mrad']/c['target_alpha95_mrad']-1) <= c['angle_relative_tolerance']
        and max(abs(row[k]) for k in ('local_waist_offset_nm','covariance_waist_offset_nm')) <= c['waist_tolerance_nm']
        and 0 < row['diameter95_nm'] <= c['maximum_diameter95_nm']
        and 0 < row['source_fraction'] <= 1
        and row['effective_rays'] >= c['minimum_effective_rays']
        and np.isfinite(f['variance_second_derivative']) and f['variance_second_derivative'] > 0)
    if reference is not None:
        row['diameter_relative_change'] = row['diameter95_nm']/reference['diameter95_nm']-1
        row['current_relative_change'] = row['source_fraction']/reference['source_fraction']-1
        passed &= (row['surface_z_mm'] == reference['surface_z_mm']
            and abs(row['diameter_relative_change']) <= c['diameter_relative_change_limit']
            and abs(row['current_relative_change']) <= c['current_relative_change_limit'])
    row['passed'] = bool(passed)
    return row


def controls(report):
    values = report.get('lens_percent')
    return ([values[k] for k in ('condenser_lens_3','objective_lens')]
            if values is not None else report['controls'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports',type=Path,default=Path('outputs/crossover-audit-20260914'))
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    evidence = []
    def read(name):
        path = args.reports/name
        data = path.read_bytes()
        evidence.append(dict(path=path.as_posix(),sha256=sha256(data).hexdigest()))
        return json.loads(data)
    nominal = read('tip-reference-dpa210-global80-tangent10369-fit.json')
    target = controls(nominal)
    reference = focus_row(nominal)
    rows = [dict(check='Nominal fitted quadrature',**reference)]
    for label,name in (
        ('Twice the angular samples per site; no refit','tip-reference-dpa210-global80-tangent20737-sharedfit.json'),
        ('Twice the emission sites, half directions per site; no refit','tip-reference-dpa210-global80-spatial288-fixed.json'),
        ('Finer tip and global field mesh; no refit','tip-reference-dpa210-global128-dense-sharedfit.json'),
        ('Finer field mesh and half column step; no refit','tip-reference-dpa210-global128-halfstep.json'),
        ('Ordinary CUDA backend, same executed gun; no refit','tip-reference-dpa210-ordinary-backend.json')):
        report = read(name)
        if controls(report) != target:
            raise ValueError('A comparison changed the fitted controls')
        rows.append(dict(check=label,**focus_row(report,reference)))
    historical = read('historical-production-column769.json')['rows'][0]
    old_planes = read('historical-full-component-layout.json')['component_planes']
    old_z = [historical['production_gun_waist']['z_mm']]+[m['z_mm'] for m in historical['lens_crossovers']]
    transport = []
    for n in (10369,20737):
        report = read(f'tip-reference-dpa210-dense{n}-full-transport.json')
        if controls(report) != target:
            raise ValueError('Transport evidence changed the fitted controls')
        z = [report['gun_waist']['z_mm']]+[m['z_mm'] for m in report['lens_crossovers']]
        topology = require_same_topology(old_z,z,old_planes,
            candidate_component_planes=report['full_component_planes'])
        signal = report['projection']
        passed = bool(len(z)==9 and signal['finite'] and signal['rays']>0
            and np.isfinite(signal['source_fraction']) and 0<signal['source_fraction']<=1)
        transport.append(dict(rays=n,displayed_marker_count=len(z),topology=topology,
            markers=[report['gun_waist'],*report['lens_crossovers']],projection=signal,passed=passed))
    passed = all(r['passed'] for r in rows+transport)
    result = dict(status='PASS_WITHIN_DECLARED_PARTICLE_TOLERANCES' if passed else 'FAIL',
        scope='Classical tip-origin optical transport at this design only; no specimen interaction or imaging validation',
        criteria=CRITERIA, controls_percent=nominal['lens_percent'],
        geometry_mm={k:nominal[k] for k in ('extractor_mm','lens_mm','accelerator_mm','gun_dpa_mm')},
        voltages=dict(extractor_kv=nominal['extractor_kv'],gun_lens_kv=nominal['lens_kv'],
            gun_lens_reference=nominal['lens_reference'],OEM_definition='UNCONFIRMED'),
        tip_model=nominal['tip_model'],exact_tip_footprint=nominal['exact_tip_footprint'],
        fixed_settings_comparisons=rows,full_column_transport=transport,
        intermediate_waist_diagnostics=nominal['crossovers'],
        final_survivor_c1_diagnostic=read('tip-reference-dpa210-dense-cohort-c1.json'),
        whole_cap_diagnostic=read('tip-reference-dpa210-global80-whole-gun4097.json'),
        caveats=['Spatial repartition is an independent fixed-total-budget check, not uniform refinement of every dimension.',
            'Importance quadrature resolves the aperture-selected specimen population; whole-gun moments need separate sampling.',
            'Nine existing displayed crossover intervals are preserved; raw gun-envelope extrema and the terminal focus are separate diagnostics.',
            'No diffraction-limited spot, coherence, TEM/STEM image, space-charge or real-instrument calibration is certified.'],
        evidence=evidence)
    write_report(result,args.output)
    print(json.dumps(dict(status=result['status'],focus_comparisons=rows),indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
