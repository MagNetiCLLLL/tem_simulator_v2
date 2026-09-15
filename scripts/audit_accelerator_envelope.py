"""Classical tip-origin accelerator envelope and fixed-input convergence audit.

No settings are saved or retuned. Plane radii use first crossings in the
particle time history, with positive current and physical stops retained.
Outputs are scalar summaries; arrays/caches are not exported.
"""
from dataclasses import asdict, replace
from collections import Counter
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from temsim.assembly_catalog import AssemblyCatalog
from temsim.optics.column import default_state
from temsim.optics.electron_gun.source import trace_source_to_exit
from temsim.operating_modes import apply_operating_mode_pair
from temsim.physics.column_wall import clip_column_wall


def quantile(values, weights, fraction):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(np.interp(fraction*cumulative[-1], cumulative, values[order]))


def crossing(trace, z):
    history = trace.equal_time_history
    keep = ((~np.isfinite(trace.blocked_z_mm) | (trace.blocked_z_mm >= z))
            & (trace.exit_bundle.weight > 0))
    indices, samples = [], []
    for i in np.flatnonzero(keep):
        reached = np.flatnonzero(history.z_mm[:, i] >= z-8*np.spacing(max(1., abs(z))))
        if not len(reached):
            raise ValueError(f"Missing first crossing of {z} mm for ray {i}")
        hi = int(reached[0])
        lo = max(0, hi-1)
        a, b = history.z_mm[[lo, hi], i]
        t = 0. if a == b else np.clip((z-a)/(b-a), 0., 1.)
        samples.append([v[lo, i]+t*(v[hi, i]-v[lo, i])
                        for v in (history.x_m, history.y_m, history.tx_rad, history.ty_rad)])
        indices.append(i)
    return np.asarray(indices, int), np.asarray(samples).reshape((-1, 4))


def plane(trace, z, assembly):
    indices, samples = crossing(trace, z)
    radii = [s.inner_diameter_mm/2 for s in assembly.vacuum_bore_segments
             if s.start_z_mm <= z <= s.end_z_mm]
    bore = min(radii) if radii else None
    row = dict(z_mm=float(z), positive_rays=len(indices), wall_radius_mm=bore)
    if not len(indices):
        return row
    x, y, tx, ty = samples.T
    w = trace.exit_bundle.weight[indices]
    radius = np.hypot(x, y)*1000
    r95 = quantile(radius, w, .95)
    rmax = float(np.max(radius))
    row.update(current_fraction=float(w.sum()), r95_mm=r95,
               d95_mm=2*r95, rmax_mm=rmax,
               rms_radius_mm=float(np.sqrt(np.average(radius**2, weights=w))),
               alpha95_mrad=1000*quantile(np.arctan(np.hypot(tx, ty)), w, .95),
               minimum_sampled_wall_clearance_mm=None if bore is None else bore-rmax,
               r95_wall_fraction=None if bore is None else r95/bore)
    return row


def audit(case):
    state = default_state()
    catalog = AssemblyCatalog()
    assembly = catalog.apply(state, catalog.default_selection())
    apply_operating_mode_pair(state, 'nano_probe', 'diffraction')
    gun = state.electron_gun
    rays = 769 if case == 'curved_769' else 193
    gun.emitter.ray_count = rays
    gun.history_step_mm = gun.trace_step_mm
    if case == 'planar_193':
        gun.emitter.surface_model = None
    elif case == 'curved_mesh2_193':
        model = gun.emitter.surface_model
        f = model.field_numerics
        gun.emitter.surface_model = replace(model, field_numerics=replace(f,
            radial_nodes=2*f.radial_nodes, axial_nodes=2*f.axial_nodes,
            apex_cells_per_radius=2*f.apex_cells_per_radius,
            electrode_cells_per_bore=2*f.electrode_cells_per_bore))
    elif case == 'curved_step_half_193':
        gun.trace_step_mm *= .5
        gun.drift_step_mm *= .5
        gun.history_step_mm = gun.trace_step_mm
    elif case not in ('curved_193', 'curved_769'):
        raise ValueError(case)
    started = perf_counter()
    field = gun.electric_field
    build_s = perf_counter()-started
    print(f'{case}: field built in {build_s:.2f} s; tracing {rays} particles', flush=True)
    trace = trace_source_to_exit(state)
    end = perf_counter()
    print(f'{case}: trace finished in {end-started-build_s:.2f} s; measuring planes', flush=True)
    zs = sorted(set([1e-6, .001, .01, .1, 1., 2., 4., 5.9, 6.1, 8., 10.1, 14., 18., 22.,
                     30., 42., 70., 100., 119.9, 120.1, 140., 200., 300., 362., 370., 398., 444., 450.]))
    rows = [plane(trace, z, assembly) for z in zs]
    envelope = [plane(trace, float(z), assembly) for z in np.linspace(30., 370., 171)]
    pos = np.zeros((len(zs), 3))
    pos[:, 2] = np.array(zs)*.001
    model = gun.emitter.surface_model
    potential = (field.potential_rise_v_at_global_positions(pos) if model else
                 field.potential_v_at_global_positions(pos))
    wall_alive, wall_z, wall_key = clip_column_wall(state, trace.z_mm, trace.x_m,
        trace.y_m, trace.exit_bundle.alive, trace.blocked_z_mm, trace.blocked_key)
    def exit_summary(alive):
        bundle = trace.exit_bundle
        alive = alive & (bundle.weight > 0)
        w = bundle.weight[alive]
        r = np.hypot(bundle.x_m[alive], bundle.y_m[alive])*1000
        return dict(positive_rays=int(alive.sum()), current_fraction=float(w.sum()),
            d95_mm=2*quantile(r, w, .95) if len(r) else None,
            rmax_mm=float(r.max()) if len(r) else None)
    apertures = [dict(key=a.key, z_mm=a.z_mm, radius_mm=a.radius_mm,
                      mechanical_bore_diameter_mm=a.mechanical_bore_diameter_mm)
                 for a in (gun.dpa_aperture, gun.c1_aperture)]
    report = dict(case=case, ray_count=rays, field_build_s=build_s, trace_s=end-started-build_s,
        input=gun.to_dict(), field_report=getattr(field, 'report', None),
        surface_report=getattr(trace, 'surface_model_report', None),
        exit_current_fraction=float(trace.exit_bundle.weight[trace.exit_bundle.alive].sum()),
        accepted_gun_exit=exit_summary(trace.exit_bundle.alive),
        accepted_exit_after_assembly_walls=exit_summary(wall_alive),
        assembly_wall_stops=dict(Counter(wall_key)),
        earliest_assembly_wall_stop_mm=(float(np.nanmin(wall_z[np.array(wall_key)=='column_wall']))
            if 'column_wall' in wall_key else None),
        active_apertures=apertures,
        gun_stop_positions_mm={key:sorted(set(map(float, trace.blocked_z_mm[np.array(trace.blocked_key)==key])))
                               for key in set(trace.blocked_key) if key},
        arrival_summaries=[dict(key=p.key, z_mm=p.z_mm, reached=int(p.reached.sum()),
            transmitted=int(p.transmitted.sum()),
            transmitted_current_fraction=float(trace.exit_bundle.weight[p.transmitted].sum()))
            for p in trace.plane_arrivals],
        stops=dict(Counter(trace.blocked_key)), planes=rows, accelerator_envelope=envelope,
        axial_potential_rise_v=[dict(z_mm=z, rise_v=float(v)) for z, v in zip(zs, potential)],
        bore_segments=[asdict(s) for s in assembly.vacuum_bore_segments if s.start_z_mm < 451],
        scope=('Gun-only first-crossing planes, conditional on gun aperture/body stops; at a stop plane '
               'the arriving beam is counted before its mask. Nominal walls are compared at each plane; '
               'the separate exit-after-assembly-walls summary applies production wall clipping '
               'to saved gun trajectories. No downstream condenser matching or specimen calculation.'))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', required=True, choices=('planar_193', 'curved_193', 'curved_769',
        'curved_mesh2_193', 'curved_step_half_193'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Choose a new output file')
    report = audit(args.case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('case', 'field_build_s', 'trace_s',
        'exit_current_fraction', 'stops', 'planes')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
