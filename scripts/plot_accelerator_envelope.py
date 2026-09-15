"""Plot scalar accelerator audits; no trace or instrument settings are changed."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def rows(report):
    by_z = {r['z_mm']: r for r in report['accelerator_envelope']}
    by_z.update({r['z_mm']: r for r in report['planes']})
    return [by_z[z] for z in sorted(by_z)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    def read(name):
        return json.loads((args.input_dir / f'accelerator-{name}.json').read_text())
    curved, planar, mesh = read('curved-769'), read('planar-193'), read('mesh2-193')
    c, p, m = rows(curved), rows(planar), rows(mesh)
    z = np.array([r['z_mm'] for r in c])
    r95, rmax = (np.array([r[k] for r in c]) for k in ('r95_mm', 'rmax_mm'))
    fig = plt.figure(figsize=(12, 8.8), layout='constrained')
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.])
    whole, near, comparison = fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    for ax in (whole, near):
        segments = sorted(curved['bore_segments'], key=lambda s: s['start_z_mm'])
        previous = None
        for segment in segments:
            a, b = segment['start_z_mm'], min(450., segment['end_z_mm'])
            if a > 450:
                continue
            radius = segment['inner_diameter_mm'] / 2
            ax.plot([a, b], [radius, radius], color='#596779', lw=2,
                    label='Local bore radius' if previous is None else None)
            if previous and abs(previous[0]-a) < 1e-9:
                ax.plot([a, a], [previous[1], radius], color='#596779', lw=2)
            previous = b, radius
        ax.plot(z, rmax, color='#e47722', ls='--', lw=1.6, label='Sampled outermost radius (769 rays)')
        ax.plot(z, r95, color='#007c91', lw=2.2, label='95% current radius (769 rays)')
        ax.set(ylabel='Radius (mm)', xlabel='Z from tip apex (mm)', ylim=(0, 5.5))
        ax.grid(alpha=.18)
    whole.set(xlim=(0, 450), title='A. Most of the accelerator has substantial radial clearance')
    whole.axvspan(30, 370, color='#d3e8ec', alpha=.25, zorder=-1)
    whole.text(220, 5.17, 'Accelerator: Z = 30–370 mm | bulk bore diameter = 10 mm',
               ha='center', fontsize=10)
    whole.plot([120, 120], [2, 3], color='#a12865', lw=3)
    whole.annotate('DPA: active radius 2 mm', (120, 2), (143, .9),
                   arrowprops=dict(arrowstyle='->', color='#a12865'), color='#a12865', fontsize=10)
    whole.legend(loc='lower right', fontsize=9, framealpha=.93)
    near.set(xlim=(0, 35), title='B. Tight spot is before the accelerator')
    maximum = next(r for r in c if r['z_mm'] == 22.)['rmax_mm']
    near.scatter([22], [maximum], color='#ba2525', s=35, zorder=5)
    near.annotate(f'Z = 22 mm\noutermost r = {maximum:.3f} mm\nbore r = 2.500 mm',
                  (22, maximum), (11, 4.25), fontsize=9, color='#a32424',
                  arrowprops=dict(arrowstyle='->', color='#a32424'))
    near.axvspan(6, 10, color='#aebac8', alpha=.18)
    near.axvspan(14, 22, color='#aebac8', alpha=.18)
    near.text(8, .32, 'Extractor', fontsize=9, ha='center')
    near.text(18, .7, 'Gun lens', fontsize=9, ha='center')
    comparison.semilogy(z, [r['d95_mm'] for r in c], color='#007c91', lw=2,
                        label='Curved: 769 rays')
    comparison.semilogy([r['z_mm'] for r in m], [r['d95_mm'] for r in m],
                        color='#77a9ab', ls=':', lw=2, label='Curved: refined field, 193 rays')
    comparison.semilogy([r['z_mm'] for r in p], [r['d95_mm'] for r in p],
                        color='#754697', lw=2, label='Legacy planar: 193 rays')
    comparison.set(xlim=(0, 450), ylim=(1e-4, 10), xlabel='Z from tip apex (mm)',
                   ylabel='95% current diameter (mm, log scale)',
                   title='C. These are different source and field models')
    comparison.grid(alpha=.18, which='both')
    comparison.legend(fontsize=8.5, loc='lower right')
    fig.suptitle('Classical FEG envelope audit | 300 kV, extractor 4 kV, gun lens +1.2 kV relative to extractor', fontsize=13)
    fig.supxlabel('Gun-only first crossings before assembly-wall clipping; current-weighted and conditional on upstream aperture stops.\n'
                  'Numerical model results, not a measured or OEM-calibrated microscope beam profile.', fontsize=10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
