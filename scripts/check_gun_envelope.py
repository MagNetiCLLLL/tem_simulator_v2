"""Small classical gun-only sensitivity audit; no saved settings or array files.

The normal-only and narrow-cap cases are explicitly artificial sensitivity
controls, not replacement sources or calibrated operating recommendations.
"""
import argparse
from collections import Counter
from dataclasses import replace
import json
from time import perf_counter

import numpy as np

from temsim.optics.column import default_state


def plane_summary(trace, z_mm):
    """Interpolate original particle histories, not the coarse display grid."""
    live = (np.isnan(trace.blocked_z_mm) | (trace.blocked_z_mm >= z_mm)) & (trace.exit_bundle.weight > 0)
    indices = np.flatnonzero(live)
    if not indices.size:
        return {"z_mm": z_mm, "reaching_rays": 0}
    history = trace.equal_time_history
    crossings = []
    z_roundoff = 8 * np.spacing(max(1., abs(z_mm)))
    for i in indices:
        reached = np.flatnonzero(history.z_mm[:, i] >= z_mm-z_roundoff)
        if not reached.size:
            raise ValueError(f"Ray {i} has no recorded crossing of Z={z_mm} mm; "
                             f"last recorded Z={np.max(history.z_mm[:, i])} mm")
        upper = int(reached[0])
        lower = max(0, upper - 1)
        z0, z1 = history.z_mm[[lower, upper], i]
        fraction = 0. if z1 == z0 else np.clip((z_mm-z0)/(z1-z0), 0., 1.)
        crossings.append([values[lower, i] + fraction*(values[upper, i]-values[lower, i])
                          for values in (history.x_m, history.y_m, history.tx_rad, history.ty_rad)])
    x, y, tx, ty = np.asarray(crossings).T
    weights = trace.exit_bundle.weight[indices]
    return {"z_mm": z_mm, "reaching_rays": len(indices),
            "x_span_mm": float(np.ptp(x) * 1e3),
            "axis_centred_envelope_diameter_mm": float(2 * np.max(np.hypot(x, y)) * 1e3),
            "axis_centred_rms_radius_mm": float(np.sqrt(np.average(x*x+y*y, weights=weights))*1e3),
            "max_polar_angle_deg": float(np.degrees(np.arctan(np.max(np.hypot(tx, ty)))))}


def audit(case, rays, *, extractor_kv=None, lens_kv=None,
          extractor_center_mm=None, lens_center_mm=None, mesh_scale=1, lens_reference=None):
    state = default_state()
    gun = state.electron_gun
    gun.emitter.ray_count = rays
    gun.history_step_mm = gun.trace_step_mm
    model = gun.emitter.surface_model
    if case == "normal-only":
        model = replace(model, emission=replace(model.emission, maximum_angle_deg=0.))
    elif case == "narrow-cap":
        model = replace(model, emission=replace(model.emission, cap_half_angle_deg=1.))
    elif case == "refined-field":
        model = replace(model, field_numerics=replace(model.field_numerics,
            radial_nodes=2*model.field_numerics.radial_nodes,
            axial_nodes=2*model.field_numerics.axial_nodes,
            apex_cells_per_radius=2*model.field_numerics.apex_cells_per_radius))
    elif case == "lens-nearer":
        # Keep the full 8 mm lens body after the extractor (which ends at 10 mm).
        gun.electrostatic_lens.mechanical_center_from_tip_mm = 14.5
    elif case == "front-nearer":
        # Change positions only: body lengths, bores and voltages stay unchanged.
        gun.extractor.mechanical_center_from_tip_mm = 4.
        gun.electrostatic_lens.mechanical_center_from_tip_mm = 11.
    if extractor_kv is not None:
        if not 4. <= extractor_kv <= 5.:
            raise ValueError("This matching audit constrains extraction to 4-5 kV")
        gun.extractor.voltage_kv = extractor_kv
    if lens_kv is not None:
        if not 1. <= lens_kv <= 1.2:
            raise ValueError("This matching audit constrains gun-lens control to 1-1.2 kV")
        gun.electrostatic_lens.voltage_kv = lens_kv
    if lens_reference is not None:
        gun.electrostatic_lens.voltage_reference = lens_reference
    if extractor_center_mm is not None:
        gun.extractor.mechanical_center_from_tip_mm = extractor_center_mm
    if lens_center_mm is not None:
        gun.electrostatic_lens.mechanical_center_from_tip_mm = lens_center_mm
    if mesh_scale != 1:
        model = replace(model,field_numerics=replace(model.field_numerics,
            radial_nodes=model.field_numerics.radial_nodes*mesh_scale,
            axial_nodes=model.field_numerics.axial_nodes*mesh_scale,
            apex_cells_per_radius=model.field_numerics.apex_cells_per_radius*mesh_scale))
    gun.emitter.surface_model = model
    started = perf_counter()
    trace = gun.trace_to_exit()
    field = gun.electric_field
    z_values = [1e-5, .001, .1, 1., gun.extractor.mechanical_center_from_tip_mm,
                gun.electrostatic_lens.mechanical_center_from_tip_mm, 30., 70., gun.exit_plane_z_mm]
    report = {"case": case, "rays": rays, "seconds": perf_counter()-started,
              "tip_radius_nm": model.geometry.apex_radius_nm,
              "cap_half_angle_deg": model.emission.cap_half_angle_deg,
              "local_max_angle_deg": model.emission.maximum_angle_deg,
              "current_na": model.current_na,
              "passed_exit_rays": int(np.count_nonzero(trace.exit_bundle.alive)),
              "passed_exit_fraction": float(np.sum(trace.exit_bundle.weight[trace.exit_bundle.alive])),
              "voltage_reference": f"Extractor relative to tip; gun lens relative to {gun.electrostatic_lens.voltage_reference}",
              "trace_step_mm": gun.trace_step_mm,
              "scope": "Gun-only; surrounding column drift-wall clipping is applied separately in Ray Diagram",
              "electrodes": field.request["rings"][:2],
              "planes": [plane_summary(trace, z) for z in z_values],
              "stops": dict(Counter(trace.blocked_key)),
              "numerical_report": trace.surface_model_report}
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rays", type=int, default=49)
    parser.add_argument("--case", action="append", choices=(
        "default", "normal-only", "narrow-cap", "refined-field", "lens-nearer", "front-nearer"))
    parser.add_argument("--extractor-kv",type=float)
    parser.add_argument("--lens-kv",type=float)
    parser.add_argument("--lens-reference",choices=("tip","extractor","ground"))
    parser.add_argument("--extractor-center-mm",type=float)
    parser.add_argument("--lens-center-mm",type=float)
    parser.add_argument("--mesh-scale",type=int,choices=(1,2,3,4),default=1)
    args = parser.parse_args()
    for case in args.case or ["default"]:
        audit(case, args.rays,extractor_kv=args.extractor_kv,lens_kv=args.lens_kv,
              extractor_center_mm=args.extractor_center_mm,lens_center_mm=args.lens_center_mm,
              mesh_scale=args.mesh_scale,lens_reference=args.lens_reference)


if __name__ == "__main__":
    main()
