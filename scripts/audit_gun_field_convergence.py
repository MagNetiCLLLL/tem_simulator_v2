"""Fixed-input electrostatic mesh comparison, never an automatic retuning."""
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
from time import perf_counter

from temsim.optics.column import default_state
from temsim.optics.gun_matching import candidate_with_gun_geometry, axis_variational_map
from temsim.physics.grounded_tip_field import grounded_field, _cached_field
from check_gun_matching_candidate import write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lens-mm", type=float, default=22.817987616230624)
    parser.add_argument("--extractor-mm", type=float, default=8.)
    parser.add_argument("--accelerator-mm", type=float, default=200.)
    parser.add_argument("--reference", choices=("tip","extractor"), default="extractor")
    parser.add_argument("--cells", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--intervals", type=int, default=32000)
    parser.add_argument("--corner-cells", type=int, default=0)
    parser.add_argument("--apex-cells", type=int, default=20)
    parser.add_argument("--radial-nodes", type=int, default=160)
    parser.add_argument("--axial-nodes", type=int, default=320)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("Choose a new report path")
    state = candidate_with_gun_geometry(default_state(), extractor_center_mm=args.extractor_mm,
        lens_center_mm=args.lens_mm, accelerator_center_mm=args.accelerator_mm)
    gun = state.electron_gun
    gun.extractor.voltage_kv = 4.5
    gun.electrostatic_lens.voltage_kv = 1.1
    gun.electrostatic_lens.voltage_reference = args.reference
    rows = []
    for cells in args.cells:
        start = perf_counter()
        model = gun.emitter.surface_model
        gun.emitter.surface_model = replace(model, field_numerics=replace(model.field_numerics,
            electrode_cells_per_bore=cells, electrode_corner_cells=args.corner_cells,
            apex_cells_per_radius=args.apex_cells, radial_nodes=args.radial_nodes, axial_nodes=args.axial_nodes))
        try:
            field = grounded_field(gun)
        except (MemoryError, ValueError) as error:
            rows.append(dict(cells=cells, failed=type(error).__name__, error=str(error),
                             seconds=perf_counter()-start))
            write_report(dict(inputs=vars(args), rows=rows, qualification="INCOMPLETE_DIAGNOSTIC"), args.output)
            raise
        matrix, evidence = axis_variational_map(field, emission_energy_ev=.3, scale_m=1e-7,
                                                intervals=args.intervals)
        row = dict(cells=cells, matrix=matrix.tolist(), evidence=evidence,
                   field=field.report, seconds=perf_counter()-start)
        rows.append(row)
        print(json.dumps(row), flush=True)
        del field
        _cached_field.cache_clear()
        gc.collect()
    write_report(dict(inputs=vars(args), rows=rows, qualification="FIXED_INPUT_DIAGNOSTIC"), args.output)


if __name__ == "__main__":
    main()
