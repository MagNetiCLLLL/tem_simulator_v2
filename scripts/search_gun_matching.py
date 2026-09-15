"""Bounded gun-position screening, followed separately by physical particles.

Does not edit defaults/profiles or write arrays. The axial variational map is
only a proposal, not an independently configurable downstream source.
"""
import argparse
from dataclasses import replace
import json
from time import perf_counter
from pathlib import Path

import numpy as np

from temsim.optics.column import default_state
from temsim.physics.grounded_tip_field import grounded_field, _cached_field
from temsim.optics.gun_matching import axis_variational_map, candidate_with_gun_geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extractor-mm", type=float, nargs="+", default=[2.5, 4., 8.])
    parser.add_argument("--gap-mm", type=float, nargs="+", default=[.5, 3., 7., 10.])
    parser.add_argument("--reference", choices=["tip", "extractor"], default="extractor")
    parser.add_argument("--extractor-kv", type=float, default=4.5)
    parser.add_argument("--lens-kv", type=float, default=1.1)
    parser.add_argument("--intervals", type=int, default=4000)
    parser.add_argument("--electrode-cells", type=int, choices=(0,8,16,32,64), default=8)
    parser.add_argument("--corner-cells", type=int, default=0)
    parser.add_argument("--accelerator-mm", type=float, default=200.)
    parser.add_argument("--plane-mm", type=float, default=450.)
    parser.add_argument("--output", help="Optional new scalar report; no particle arrays")
    args = parser.parse_args()
    if args.output and Path(args.output).exists():
        parser.error("Choose a new report path")
    if not 4<=args.extractor_kv<=5 or not 1<=args.lens_kv<=1.2:
        parser.error("Keep the requested extraction 4-5 kV and gun lens 1-1.2 kV")
    state = default_state()
    gun = state.electron_gun
    gun.extractor.voltage_kv = args.extractor_kv
    gun.electrostatic_lens.voltage_kv = args.lens_kv
    gun.electrostatic_lens.voltage_reference = args.reference
    model=gun.emitter.surface_model
    gun.emitter.surface_model=replace(model,field_numerics=replace(model.field_numerics,
        electrode_cells_per_bore=args.electrode_cells, electrode_corner_cells=args.corner_cells))
    rows = []
    for ext in args.extractor_mm:
        for gap in args.gap_mm:
            start = perf_counter()
            lens = ext+.5*(gun.extractor.mechanical_length_mm+gun.electrostatic_lens.mechanical_length_mm)+gap
            try:
                candidate = candidate_with_gun_geometry(state, extractor_center_mm=ext, lens_center_mm=lens,
                    accelerator_center_mm=args.accelerator_mm)
            except ValueError as error:
                print(json.dumps(dict(extractor_mm=ext, lens_mm=lens, rejected=str(error))), flush=True)
                continue
            field = grounded_field(candidate.electron_gun)
            matrix, evidence = axis_variational_map(field,
                emission_energy_ev=gun.emitter.surface_model.emission.mean_energy_ev,
                scale_m=gun.emitter.surface_model.geometry.apex_radius_nm*1e-9,
                intervals=args.intervals, end_m=args.plane_mm*1e-3)
            # First-order image of the local normal: theta_tip=x_tip/R.
            # The actual emission includes multiple independent local directions.
            normal = matrix @ np.ones(2)
            row = dict(extractor_mm=ext, lens_mm=lens, gap_mm=gap,
                accelerator_mm=args.accelerator_mm, plane_mm=args.plane_mm,
                extractor_kv=args.extractor_kv, lens_kv=args.lens_kv, reference=args.reference,
                matrix=matrix.tolist(), axial_normal_transfer=normal.tolist(),
                electrode_cells=args.electrode_cells, corner_cells=args.corner_cells,
                evidence=evidence, seconds=perf_counter()-start)
            print(json.dumps(row), flush=True)
            rows.append(row)
            del field
            _cached_field.cache_clear()
    if args.output:
        from check_gun_matching_candidate import write_report
        write_report(dict(inputs=vars(args), rows=rows, qualification="PROPOSALS_ONLY"),args.output)


if __name__ == "__main__":
    main()
