"""Field-only mesh audit; no rays, images, profile edits or cache files."""
from dataclasses import replace
import argparse
import json
from time import perf_counter

import numpy as np

from temsim.optics.column import default_state
from temsim.physics.grounded_tip_field import grounded_field, _cached_field


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=[0, 8, 16])
    parser.add_argument("--global-scale", type=int, default=1)
    args = parser.parse_args()
    gun = default_state().electron_gun
    model = gun.emitter.surface_model
    z = np.array([.001, .008, .018, .05, .2])
    positions = np.column_stack((z*0+1e-8, z*0, z))
    for count in args.cells:
        numerics = replace(model.field_numerics, electrode_cells_per_bore=count,
            radial_nodes=model.field_numerics.radial_nodes*args.global_scale,
            axial_nodes=model.field_numerics.axial_nodes*args.global_scale,
            apex_cells_per_radius=model.field_numerics.apex_cells_per_radius*args.global_scale)
        gun.emitter.surface_model = replace(model, field_numerics=numerics).validate()
        start = perf_counter()
        print(f"Solving electrode mesh {count}; global scale {args.global_scale}", flush=True)
        field = grounded_field(gun)
        potential, force = field._interpolate(positions)
        print(json.dumps(dict(cells_per_bore=count, global_scale=args.global_scale,
            seconds=perf_counter()-start, z_mm=(z*1000).tolist(),
            potential_rise_v=potential.tolist(), er_over_r_v_per_m2=(force[:, 0]/1e-8).tolist(),
            ez_v_per_m=force[:, 2].tolist(), field_report=field.report)), flush=True)
        del field
        _cached_field.cache_clear()


if __name__ == "__main__":
    main()
