"""Summarise saved mechanism experiments without replaying any gun transport."""
import argparse
import json
from pathlib import Path

import numpy as np

from temsim.cpu_resources import numerical_job
from scripts.analyze_accelerator_turns import trajectory_metrics


def render(previous, current):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    names = ("analytic_width2", "analytic_width4", "analytic_width8", "analytic_width16",
             "analytic_width4_halfstep", "reference8", "reference16", "reference32",
             "reference16_flat", "reference16_halfstep", "reference16_exit100", "reference16_exit200")
    rows, summary = {}, {}
    for name, folder in [("original_geometry32", previous)] + [(n, current) for n in names]:
        file_name = "planar32" if name == "original_geometry32" else name
        with np.load(folder/(file_name+".npz"), allow_pickle=False) as a:
            rows[name] = {k: a[k] for k in ("z_m", "state", "ray_id", "weights", "probe_m", "phi_v", "electric_v_per_m")}
        row = rows[name]
        if name != "original_geometry32":
            base = rows["original_geometry32"]
            for key in ("ray_id", "weights"):
                np.testing.assert_array_equal(row[key], base[key])
            np.testing.assert_array_equal(row["state"][0], base["state"][0])
        interior = (row["probe_m"][:, 2] >= .08) & (row["probe_m"][:, 2] <= .34)
        data = {"er_abs_max_at_1um_v_per_m": float(np.max(abs(row["electric_v_per_m"][interior, 0]))),
                "ez_min_at_1um_v_per_m": float(np.min(row["electric_v_per_m"][interior, 2])),
                "ez_max_at_1um_v_per_m": float(np.max(row["electric_v_per_m"][interior, 2])),
                "intervals": {}}
        for low, high in ((30, 370), (80, 340)):
            record = trajectory_metrics(row["z_m"], row["state"], row["weights"],
                interval_m=(low*.001, high*.001), slope_deadband_rad=1e-8)
            # Keep per-ray diagnostics in the analysis script; the table here
            # contains population summaries with an explicit 0.01 urad floor.
            compact = {"longitudinal": record["longitudinal"],
                       "peak_transverse_slope_rad": record["peak_transverse_slope_rad"]}
            for key in ("x_slope_turns", "y_slope_turns", "radial_extrema", "x_axis_crossings", "y_axis_crossings"):
                if key in record:
                    compact[key] = {mode: {k: v for k, v in value.items() if k != "per_particle"}
                                    for mode, value in record[key].items()}
            data["intervals"][f"{low}-{high}mm"] = compact
        summary[name] = data
    (current/"mechanism-summary.json").write_text(json.dumps({
        "identical_emission_states_ray_ids_and_weights": True,
        "slope_deadband_rad": 1e-8, "field_probe_radius_m": 1e-6,
        "field_probe_interval_m": [.08, .34], "runs": summary}, indent=2), encoding="utf-8")
    # Deterministically choose the same source ray for all panels and models.
    baseline = rows["analytic_width4"]
    inside = (baseline["z_m"] >= .03) & (baseline["z_m"] <= .37)
    ray = int(np.argmax(np.max(abs(baseline["state"][inside, :, 0]), axis=0)))
    ray_id = int(baseline["ray_id"][ray])
    display = [("analytic_width4", "Current analytic field", "#bc3c29"),
               ("original_geometry32", "Current geometry, coupled field", "#0072b2"),
               ("reference32", "Proportional reference, shaped", "#009e73"),
               ("reference16_flat", "Proportional reference, flat", "#a05db0")]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for name, label, colour in display:
        row = rows[name]
        z = row["probe_m"][:, 2]*1000
        show = (z >= 30) & (z <= 370)
        axs[0, 0].plot(z[show], -row["electric_v_per_m"][show, 2]/1e6, color=colour, label=label)
        axs[0, 1].plot(z[show], row["electric_v_per_m"][show, 0], color=colour)
        z = row["z_m"]*1000
        show = (z >= 30) & (z <= 370)
        axs[1, 0].plot(z[show], row["state"][show, ray, 0]*1e6, color=colour)
        axs[1, 1].plot(z[show], row["state"][show, ray, 2]/row["state"][show, ray, 4]*1e6, color=colour)
    axs[0, 0].set_ylabel("Axial accelerating field -Ez (MV/m)")
    axs[0, 1].set_ylabel("Er at r = 1 um (V/m)")
    axs[1, 0].set_ylabel(f"Ray {ray_id}: X (um)")
    axs[1, 1].set_ylabel(f"Ray {ray_id}: dX/dZ (urad)")
    for ax in axs.flat:
        ax.set_xlim(30, 370)
        ax.set_xlabel("Z (mm)")
        ax.axhline(0, color="#777777", lw=.6)
        ax.grid(alpha=.2)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("Accelerator mechanism comparison\nOriginal emission is identical; axes use different physical units/scales")
    fig.savefig(current/"mechanism-comparison.png", dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    for width, colour in zip((2, 4, 8, 16), ("#d55e00", "#cc79a7", "#0072b2", "#009e73")):
        row = rows[f"analytic_width{width}"]
        z = row["probe_m"][:, 2]*1000
        show = (z >= 65) & (z <= 190)
        axs[0].plot(z[show], row["electric_v_per_m"][show, 0], color=colour, label=f"Half-width {width} mm")
        z = row["z_m"]*1000
        show = (z >= 65) & (z <= 190)
        axs[1].plot(z[show], row["state"][show, ray, 2]/row["state"][show, ray, 4]*1e6, color=colour)
    axs[0].set_ylabel("Er at r = 1 um (V/m)")
    axs[1].set_ylabel(f"Ray {ray_id}: dX/dZ (urad)")
    axs[1].set_xlabel("Z (mm)")
    axs[0].legend(ncol=2)
    for ax in axs:
        ax.set_xlim(65, 190)
        ax.axhline(0, color="#777777", lw=.6)
        ax.grid(alpha=.2)
    fig.suptitle("Only the analytic transition width changes\nSame stage positions, voltages, source and all other gun controls")
    fig.savefig(current/"width-sensitivity.png", dpi=150)
    plt.close(fig)

    report = json.loads((current/"reference-comparison.json").read_text(encoding="utf-8"))
    fig, axs = plt.subplots(2, 1, figsize=(12, 9), sharex=True, sharey=True, constrained_layout=True)
    for ax, name in zip(axs, ("reference16", "reference16_flat")):
        request = report["runs"][name]["field"]
        for ring in request["rings"]:
            z0, z1 = ring["start_m"]*1000, ring["stop_m"]*1000
            r0, r1 = ring["inner_m"]*1000, ring["outer_m"]*1000
            colour = plt.colormaps["viridis"](ring["potential_rise_v"]/300000)
            for side in (-1, 1):
                lower = r0 if side > 0 else -r1
                ax.add_patch(Rectangle((z0, lower), z1-z0, r1-r0, facecolor=colour, edgecolor="#333333", lw=.5))
        ax.axvline(0, color="black", lw=2, label="Ideal planar cathode")
        ax.axhline(0, color="#555555", linestyle="--", lw=.7)
        ax.axvline(450, color="#d55e00", linestyle=":", label="Numerical boundary")
        ax.set_aspect("equal", adjustable="box")
        ax.set_ylim(-135, 135)
        ax.set_xlim(-10, 460)
        ax.set_ylabel("Signed radius (mm)")
        ax.set_title("Shaped intermediate electrodes" if name == "reference16" else "Flat intermediate electrodes: same webs, minimum bores and voltages")
        ax.legend(loc="upper right", fontsize=8)
    axs[-1].set_xlabel("Z from original emission plane (mm)")
    fig.suptitle("Approximate proportional reference geometry\nExtractor upstream enclosure and dielectric housing are not reconstructed")
    fig.savefig(current/"reference-geometry.png", dpi=150)
    plt.close(fig)
    print("Saved mechanism summary and three figures; source arrays match exactly", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, default=Path("tmp/planar-gun-20260926"))
    parser.add_argument("--current", type=Path, default=Path("tmp/accelerator-mechanism-20260926"))
    args = parser.parse_args()
    with numerical_job(requested=1):
        render(args.previous, args.current)
