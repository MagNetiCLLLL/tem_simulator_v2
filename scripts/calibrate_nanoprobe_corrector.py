"""Reproduce the non-OEM, source-based nanoprobe calibration.

Writes a candidate report and a sample-plane scatter plot; never overwrites
instrument TOMLs.  Use the report only after its step-refinement check passes.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from temsim.assembly_catalog import AssemblyCatalog
from temsim.operating_modes import apply_operating_mode_pair
from temsim.optics.column import default_state
from temsim.optics.probe_calibration import IncidentProbeModel, fit_probe_hexapoles


def focus(model, target_mrad, *, initial=False):
    state = model.state
    if initial:
        from temsim.optics.direct_alignment import apply_direct_alignment
        result = apply_direct_alignment(state, "nanoprobe_convergence", target_mrad)
        print("convergence alignment", result, flush=True)
        if not result.success:
            raise RuntimeError(f"Convergence alignment failed: {result}")
    lenses = {lens.key: lens for lens in state.lenses}
    c2, c3 = (lenses[key] for key in ("condenser_lens_2", "condenser_lens_3"))
    original = float(c3.percent)

    def waist(percent):
        c3.percent = float(percent)
        stats = model.trace().statistics
        return stats.waist_offset_m * 1e9

    for width in (0.0001, 0.001, 0.01, 0.1):
        lower, upper = max(0.0, original - width), min(c3.max_percent, original + width)
        if waist(lower) * waist(upper) <= 0:
            c3.percent = float(brentq(waist, lower, upper, xtol=1e-11))
            break
    else:
        c3.percent = original
        raise RuntimeError("No local sample-focus root found")
    stats = model.trace().statistics
    print("focus", c2.percent, c3.percent, stats.convergence_95_mrad,
          "waist_nm", stats.waist_offset_m * 1e9, flush=True)
    if abs(stats.convergence_95_mrad / target_mrad - 1) > 0.08:
        raise RuntimeError("Corrector calibration moved convergence outside tolerance")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/probe_calibration"))
    parser.add_argument("--rays", type=int, default=512)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--target", type=float, default=25.0)
    parser.add_argument("--skip-focus", action="store_true")
    parser.add_argument("--keep-convergence", action="store_true",
                        help="Keep C2 and polish only the nearby C3 focus root")
    parser.add_argument("--initial", type=Path)
    args = parser.parse_args()
    from numba import set_num_threads
    set_num_threads(4)
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, catalog.default_selection())
    apply_operating_mode_pair(state, "nano_probe", "diffraction")
    state.electron_gun.emitter.ray_count = args.rays
    state.step_mm = args.step
    state.acceleration_enabled = True
    state.acceleration_backend = "Numba CPU"
    if args.initial:
        previous = json.loads(args.initial.read_text(encoding="utf-8"))
        for lens in state.lenses:
            if lens.key in previous["lenses"]:
                lens.percent = previous["lenses"][lens.key]
        for key, values in previous["hexapoles"].items():
            component = next(item for item in state.corrector_elements if item.key == key)
            for name, value in values.items():
                setattr(component, name, value)
    model = IncidentProbeModel(state)
    initial_probe = model.trace()
    print("initial", asdict(initial_probe.statistics), flush=True)
    if not args.skip_focus:
        focus(model, args.target, initial=not args.keep_convergence)
    before = model.trace()
    fitted = fit_probe_hexapoles(model)
    fits = [fitted]
    print("fit", fitted, flush=True)
    if not args.skip_focus:
        focus(model, args.target)
        fitted = fit_probe_hexapoles(model)
        fits.append(fitted)
    after = model.trace()
    refined = model.trace(step_mm=args.step / 2.0)
    print("after", asdict(after.statistics), flush=True)
    print("refined", asdict(refined.statistics), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    coarse_stats, fine_stats = after.statistics, refined.statistics
    step_radius_error = abs(coarse_stats.radius_rms_m - fine_stats.radius_rms_m)
    step_waist_error_nm = abs(coarse_stats.waist_offset_m - fine_stats.waist_offset_m) * 1e9
    step_converged = (
        step_radius_error <= max(0.005 * fine_stats.radius_rms_m, 0.005e-9)
        and step_waist_error_nm <= 0.5
    )
    focus_valid = abs(fine_stats.waist_offset_m) <= 1.0e-9
    convergence_valid = abs(fine_stats.convergence_95_mrad / args.target - 1) <= 0.08
    candidate_accepted = (
        step_converged and focus_valid and convergence_valid and not args.skip_focus
    )
    report = {
        "voltage_kv": state.beam_voltage_kv,
        "sample_z_mm": state.sample.z_mm,
        "rays": args.rays, "step_mm": args.step,
        "target_convergence_95_mrad": args.target,
        "lenses": {lens.key: lens.percent for lens in state.lenses},
        "hexapoles": {
            item.key: {"strength_m3": item.strength_m3,
                       "orientation_rad": item.orientation_rad}
            for item in (state.hp2_hexapole, state.hp1_hexapole)
        },
        "fit": fitted,
        "fit_stages": fits,
        "step_converged": bool(step_converged),
        "focus_valid": bool(focus_valid),
        "convergence_valid": bool(convergence_valid),
        "candidate_accepted": bool(candidate_accepted),
        "step_radius_difference_nm": float(step_radius_error * 1e9),
        "step_waist_difference_nm": float(step_waist_error_nm),
        "initial": asdict(initial_probe.statistics),
        "before": asdict(before.statistics),
        "after": asdict(after.statistics),
        "refined": asdict(refined.statistics),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez(args.output / "rays.npz", before_x=before.x_m[before.alive],
             before_y=before.y_m[before.alive], after_x=after.x_m[after.alive],
             after_y=after.y_m[after.alive])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
    radius_nm = 1e9 * max(before.statistics.radius_99_m, after.statistics.radius_99_m) * 1.2
    for ax, label, probe in zip(axes, ("Before multipole calibration", "After calibration"),
                                 (before, after)):
        stats = probe.statistics
        ax.scatter((probe.x_m[probe.alive] - stats.mean_x_m) * 1e9,
                   (probe.y_m[probe.alive] - stats.mean_y_m) * 1e9,
                   s=3, alpha=0.55)
        ax.set(xlabel="X (nm)", ylabel="Y (nm)", xlim=(-radius_nm, radius_nm),
               ylim=(-radius_nm, radius_nm), aspect="equal")
        ax.set_title(f"{label}\nRMS radius {stats.radius_rms_m * 1e9:.3f} nm")
        ax.grid(alpha=0.15)
    fig.suptitle(f"Sample plane Z = {state.sample.z_mm:.3f} mm; geometric incident rays")
    fig.savefig(args.output / "comparison.png", dpi=160)
    plt.close(fig)
    if not candidate_accepted:
        raise RuntimeError(
            "Candidate saved for diagnosis only: final focus, convergence and step "
            "validation must pass; --skip-focus never accepts a calibration"
        )


if __name__ == "__main__":
    main()
