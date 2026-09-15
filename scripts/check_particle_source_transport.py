"""Bounded classical Ray Diagram audit; no wave/specimen image or cache files.

Executes the real source/extraction/acceleration and the normal ray pipeline.
Only in-memory condenser controls are changed. Output is a lightweight JSON
report; finite live histories do not imply near-axis physical validity.
"""
import argparse
from collections import Counter
import json
import time

import numpy as np

from temsim.optics.column import default_state
from temsim.physics.simulation import run
from temsim.physics.ray_identity import emission_colour_values


def audit(rays, step):
    state = default_state()
    state.electron_gun.emitter.ray_count = rays
    state.step_mm = step
    state.history_step_mm = step
    model = state.electron_gun.emitter.surface_model
    for c1, c2 in ((90., 35.), (5., 5.)):
        lenses = {lens.key: lens for lens in state.lenses}
        lenses["condenser_lens_1"].percent = c1
        lenses["condenser_lens_2"].percent = c2
        started = time.perf_counter()
        result = run(state, optical_only=True)
        assert lenses["condenser_lens_1"].percent == c1
        assert lenses["condenser_lens_2"].percent == c2
        trace, branch = result.gun_trace, result.incident
        reference = trace.emission_reference
        positive = trace.exit_bundle.weight > 0
        live = ((np.isnan(branch.blocked_z)[None, :]
                 | (branch.z[:, None] <= branch.blocked_z[None, :])) & positive[None, :])
        finite = np.isfinite(branch.x) & np.isfinite(branch.y) & np.isfinite(branch.tx) & np.isfinite(branch.ty)
        slopes = np.hypot(branch.tx, branch.ty)
        position = reference["position_m"]
        _, sizes = np.unique(position[positive], axis=0, return_counts=True)
        # The gun uses full momentum; only the post-gun column is paraxial.
        column_live = live & (branch.z[:, None] >= state.electron_gun.exit_plane_z_mm)
        row = {
            "c1_percent": c1, "c2_percent": c2, "rays": rays,
            "seconds": time.perf_counter()-started,
            "surface_sites": int(sizes.size), "directions_per_site_min": int(sizes.min()),
            "directions_per_site_max": int(sizes.max()),
            "tip_current_na": model.current_na,
            "gun_transmitted": int(np.count_nonzero(trace.exit_bundle.alive & positive)),
            "sample_transmitted": int(np.count_nonzero(branch.alive & positive)),
            "stop_counts": dict(Counter(key for key in branch.blocked_key if key)),
            "finite_live_coordinates": bool(np.all(finite[live])),
            "maximum_live_column_angle_deg": float(np.degrees(np.arctan(np.max(slopes[column_live], initial=0)))),
            "exit_energy_report": trace.surface_model_report,
            "emission_azimuth_available": int(np.count_nonzero(np.isfinite(
                emission_colour_values(result, branch.source_ray_id, "emission_direction")))),
        }
        print(json.dumps(row, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rays", type=int, default=49)
    parser.add_argument("--step-mm", type=float, default=.5)
    args = parser.parse_args()
    audit(args.rays, args.step_mm)


if __name__ == "__main__":
    main()
