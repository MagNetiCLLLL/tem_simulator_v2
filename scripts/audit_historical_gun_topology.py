"""Read-only isolated historical transport, with current history diagnostics.

The archive must be supplied explicitly. No historical source or field enters
an active instrument. Reports are new scalar files, not calculation caches.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--rays", type=int, default=193)
    parser.add_argument("--steps", type=float, nargs="+", default=[.2, .1, .05])
    parser.add_argument("--full-column", action="store_true",
                        help="Also observe the historical optical-only column and displayed waist labels")
    parser.add_argument("--layout-only", action="store_true", help="Read archived full-path component planes without executing rays")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new report path")
    root = args.archive.resolve(strict=True)
    sys.path.insert(0, str(root/"src"))
    os.environ["TEMSIM_PROJECT_ROOT"] = str(root)
    spec = importlib.util.spec_from_file_location("current_audit",
        Path(__file__).resolve().parents[1]/"src/temsim/optics/beam_path_audit.py")
    audit = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = audit
    spec.loader.exec_module(audit)
    from temsim.optics.column import default_state
    from temsim.operating_modes import apply_operating_mode_pair
    from temsim.optics.electron_gun.source import trace_source_to_exit
    if args.layout_only:
        state = default_state()
        apply_operating_mode_pair(state, "nano_probe", "diffraction")
        document = dict(archive=str(root), scope="HISTORICAL_LAYOUT_ONLY_NO_NEW_PHYSICS",
                        component_planes=audit.optical_component_planes(state, full_path=True))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2)
        print(json.dumps(document), flush=True)
        return
    rows = []
    for step in args.steps:
        start = perf_counter()
        state = default_state()
        apply_operating_mode_pair(state, "nano_probe", "diffraction")
        gun = state.electron_gun
        gun.emitter.ray_count = args.rays
        gun.trace_step_mm = step
        gun.drift_step_mm = step
        gun.history_step_mm = step
        trace = trace_source_to_exit(state)
        z = np.arange(24., gun.exit_plane_z_mm+1e-8, .25)
        a, masks = audit.gun_planes_coordinates(trace, z)
        cp = SimpleNamespace(z_mm=z, x_m=a[:,0], y_m=a[:,1], tx_rad=a[:,2], ty_rad=a[:,3])
        roots = audit.crossover_candidates(cp, masks, trace.exit_bundle.weight)
        # A local envelope minimum need not invert the bundle. Retain every
        # raw root and report its neighbouring sizes/ID-preserving orientation;
        # this classification must not silently alter the topology gate.
        neighbourhoods = []
        for root_row in roots:
            plane = root_row["z_mm"]
            locations = plane+np.array([-5.,-1.,0.,1.,5.])
            positions, live = audit.gun_planes_coordinates(trace, locations)
            measurements = [audit.spot_measurement("Historical waist neighbourhood",float(zz),pp,
                trace.exit_bundle.weight,mm) for zz,pp,mm in zip(locations,positions,live)]
            common = live[0]&live[-1]
            weight = trace.exit_bundle.weight[common]
            before, after = positions[0,:2,common],positions[-1,:2,common]
            # NumPy advanced indexing yields (rays, xy) here.
            inverted = np.einsum("ij,ij->i",before,after) < 0
            neighbourhoods.append(dict(z_mm=plane, planes_mm=locations.tolist(),
                rms_radius_nm=[m.rms_radius_nm for m in measurements],
                diameter95_nm=[m.diameter95_nm for m in measurements],
                opposite_side_fraction=float(weight[inverted].sum()/weight.sum())))
        row = dict(step_mm=step, roots=roots, seconds=perf_counter()-start,
                   neighbourhoods=neighbourhoods,
                   rays=args.rays, exit_alive=int(trace.exit_bundle.alive.sum()),
                   field_class=type(gun.electric_field).__name__)
        if args.full_column:
            from temsim.physics.simulation import run
            from temsim.physics.all_lens_crossovers import detect_all_lens_crossovers
            state.step_mm = step
            state.history_step_mm = .5
            state._optical_tuning = True
            result = run(state, resolved_layout=state._resolved_optics_layout, optical_only=True)
            row["production_gun_waist"] = result.gun_waist
            row["lens_crossovers"] = detect_all_lens_crossovers(
                [result.incident, *result.branches.values()], state.lenses)
            row["corrector_crossovers"] = result.corrector_crossovers
            row["c2c3_crossover"] = result.c2c3_crossover
            row["component_planes"] = audit.optical_component_planes(state)
            row["full_component_planes"] = audit.optical_component_planes(state, full_path=True)
            row["physical_scope"] = "historical optical-only particles; no sample/image calculation"
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(dict(archive=str(root), rows=rows, scope="HISTORICAL_MODEL_DIAGNOSTIC"), stream, indent=2)


if __name__ == "__main__":
    main()
