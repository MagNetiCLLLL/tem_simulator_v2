from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from numba import set_num_threads

from temsim.assembly_catalog import AssemblyCatalog
from temsim.operating_modes import apply_operating_mode_pair
from temsim.optics.column import default_state
from temsim.optics.probe_calibration import IncidentProbeModel

set_num_threads(4)
state = default_state()
catalog = AssemblyCatalog()
catalog.apply(state, catalog.default_selection())
old_fields = [(h.strength_m3, h.orientation_rad) for h in (state.hp2_hexapole, state.hp1_hexapole)]
apply_operating_mode_pair(state, 'nano_probe', 'diffraction')
calibrated_settings = {
    'lenses': {lens.key: lens.percent for lens in state.lenses},
    'hexapoles': {h.key: {'strength_m3': h.strength_m3, 'orientation_rad': h.orientation_rad}
                  for h in (state.hp2_hexapole, state.hp1_hexapole)},
}
state.electron_gun.emitter.ray_count = 15000
state.acceleration_enabled = True
state.acceleration_backend = 'Numba CPU'
model = IncidentProbeModel(state)
results = {}
for step in (0.1, 0.05, 0.025):
    state.step_mm = step
    probe = model.trace()
    results[str(step)] = asdict(probe.statistics)
    print(step, results[str(step)], flush=True)
    if step == 0.05:
        corrected = probe
state.step_mm = 0.05
for h, (strength, orientation) in zip((state.hp2_hexapole, state.hp1_hexapole), old_fields):
    h.strength_m3 = strength
    h.orientation_rad = orientation
before = model.trace()
out = Path('outputs/probe_corrector_validation')
out.mkdir(parents=True, exist_ok=True)
report = {
    'scope': 'Geometric incident source rays at the physical sample plane; installed preset without refitting',
    'calibrated_settings': calibrated_settings,
    'emitted_ray_count': 15000,
    'voltage_kv': state.beam_voltage_kv,
    'sample_z_mm': state.sample.z_mm,
    'c2_aperture_diameter_um': state.condenser_aperture_2.diameter_um,
    'corrected_steps_mm': results,
    'old_hexapoles_at_same_lens_focus': asdict(before.statistics),
}
(out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
radius = before.statistics.radius_99_m*1e9*1.2
for ax, title, sample, limit in zip(axes,
        ('Previous hexapole settings', 'Calibrated hexapoles · same scale', 'Calibrated probe · enlarged'),
        (before, corrected, corrected), (radius, radius, 0.45)):
    s = sample.statistics
    ax.scatter((sample.x_m[sample.alive]-s.mean_x_m)*1e9,
               (sample.y_m[sample.alive]-s.mean_y_m)*1e9,
               s=2, alpha=.35, c='#087fbe')
    ax.set(xlim=(-limit,limit), ylim=(-limit,limit), aspect='equal', xlabel='X (nm)', ylabel='Y (nm)')
    ax.set_title(f'{title}\nRMS radius {s.radius_rms_m*1e9:.3f} nm · D95 {s.radius_95_m*2e9:.3f} nm')
    ax.grid(alpha=.15)
fig.suptitle('Same source, sample plane and round-lens settings · 300 kV · 100 µm aperture\n15,000 emitted rays; geometric ray distributions', fontsize=12)
fig.savefig(out/'comparison.png', dpi=170)
plt.close(fig)
print(out.resolve(), flush=True)
