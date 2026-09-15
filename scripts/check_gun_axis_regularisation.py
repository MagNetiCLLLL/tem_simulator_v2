"""Bounded tip-origin comparison of raw and axis-regular field interpolation.

No profile changes, wave work or detector images. Scalar report only. Uses
the historical Nanoprobe/Diffraction lens branch, not the transport-recovery
branch that changed crossover order. Numerical sampling does not change the
physical emission law. A completed run is not probe qualification.
"""
from dataclasses import asdict, replace
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from temsim.optics.column import default_state
from temsim.operating_modes import apply_operating_mode_pair
from temsim.optics.beam_path_audit import incident_checkpoints, spot_measurement, crossover_candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rays", type=int, default=193)
    parser.add_argument("--fraction", type=float, action="append")
    parser.add_argument("--electrode-cells", type=int, default=8)
    parser.add_argument("--sampling", choices=("uniform_area", "apex_stratified_v1"), default="apex_stratified_v1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rays < 73:
        parser.error("Use at least 73 rays to represent nine cap strata with eight directions per site")
    if args.output and args.output.exists():
        raise FileExistsError("Choose a new output path; previous evidence will not be overwritten")
    reports = []
    for fraction in args.fraction or [0., .01, .005]:
        state = default_state()
        apply_operating_mode_pair(state, "nano_probe", "diffraction")
        state.vacuum_map.enabled = False
        state._optical_tuning = True
        state.step_mm = .05
        gun = state.electron_gun
        gun.emitter.ray_count = args.rays
        model = gun.emitter.surface_model
        gun.emitter.surface_model = replace(model,
            field_numerics=replace(model.field_numerics, axis_core_fraction=fraction,
                                  electrode_cells_per_bore=args.electrode_cells),
            emission=replace(model.emission, spatial_sampling=args.sampling))
        start = perf_counter()
        print(f"Axis fraction {fraction}: executing {args.rays} physical tip samples", flush=True)
        end = float(state.sample.upper_surface_z_mm)
        lens_z = {lens.key: float(lens.z_mm) for lens in state.lenses}
        checkpoints = [("Gun exit", gun.exit_plane_z_mm),
                       ("C1 centre", lens_z["condenser_lens_1"]),
                       ("C2 centre", lens_z["condenser_lens_2"]), ("Specimen entrance", end)]
        planes = np.unique(np.r_[np.arange(gun.exit_plane_z_mm, end, .5), [z for _, z in checkpoints]])
        trace, cp, mask = incident_checkpoints(state, planes, step_mm=.05)
        e = trace.exit_bundle
        spots = []
        for name, z in checkpoints:
            matches = np.flatnonzero(abs(cp.z_mm-z) < 1e-10)
            if len(matches) != 1:
                raise ValueError(f"Missing exact diagnostic plane: {name}")
            index = int(matches[0])
            spots.append(asdict(spot_measurement(name, float(cp.z_mm[index]),
                (cp.x_m[index], cp.y_m[index], cp.tx_rad[index], cp.ty_rad[index]), e.weight, mask[index])))
        report = dict(axis_core_fraction=fraction, electrode_cells_per_bore=args.electrode_cells,
            rays=args.rays, sampling=args.sampling,
            elapsed_s=perf_counter()-start, tip_model=gun.emitter.surface_model.to_dict(),
            field_report=dict(trace.surface_model_report),
            spots=spots, crossovers=crossover_candidates(cp, mask, e.weight),
            qualification="DIAGNOSTIC_ONLY; not step/sampling-converged probe qualification")
        reports.append(report)
        print(json.dumps(clean(report), allow_nan=False), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(clean(reports), stream, indent=2, allow_nan=False)
            stream.write("\n")


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    main()
