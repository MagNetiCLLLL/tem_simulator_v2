"""Recompose the archived Si frames after the finite-envelope tail fix.

This reuses their saved coherent multislice signals; it does not run a new
wave acquisition. The old Gaussian ROI overlap and both probability scales
are reconstructed explicitly. Original acquisition files are never modified.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import tifffile


def correct(source: Path, destination: Path, *, figures=False):
    parameters = json.loads((source / "parameters.json").read_text())
    old_metrics = json.loads((source / "metrics.json").read_text())
    metrics = old_metrics["production_metrics"]
    data = np.load(source / "raw_scan.npz")
    sample = parameters["state"]["sample"]
    assert sample["envelope_shape"] == "disk"
    assert sample["size_x_nm"] == sample["size_y_nm"] == 10.0
    x, y = data["scan_x_nm"], data["scan_y_nm"]
    sigma = parameters["actual_probe"]["radius_rms_m"] * 1e9 / np.sqrt(2)
    window = metrics["specimen_wave_window_bounds_nm"]
    origin = np.array([(window[0] + window[1]) / 2, (window[2] + window[3]) / 2])
    # These three known Si acquisitions lie well inside one homogeneous disk.
    radius = np.hypot(x + origin[0] - sample["centre_x_nm"],
                      y + origin[1] - sample["centre_y_nm"])
    assert float(radius.max()) + 10 * sigma < 5.0
    dx = float(np.median(np.abs(np.diff(x, axis=1))))
    dy = float(np.median(np.abs(np.diff(y, axis=0))))
    old_overlap = np.clip(gaussian_filter(np.ones_like(x),
        sigma=(sigma / dy, sigma / dx), mode="constant", cval=0.), 0., 1.)
    from types import SimpleNamespace
    from temsim.specimen.rutherford import finite_sample_gaussian_overlap
    new_overlap = finite_sample_gaussian_overlap(SimpleNamespace(**sample),
        (x + origin[0]) * 1e-3, (y + origin[1]) * 1e-3, probe_sigma_nm=sigma)
    assert np.allclose(new_overlap, 1., rtol=0., atol=1e-12)
    probability = metrics["rutherford_tail"]["scattered_probability"]
    incident = metrics["incident_sample_fraction"]
    tracked = metrics["tracked_probability_after_inelastic_absorption"]
    available = incident * tracked
    old_scale = 1. - probability * old_overlap
    new_scale = 1. - probability * new_overlap
    ratio = new_scale / old_scale
    arrays = {"scan_x_nm": x, "scan_y_nm": y,
              "old_overlap": old_overlap, "corrected_overlap": new_overlap}
    statistics = {}
    old_collected_tail = np.zeros_like(x)
    new_collected_tail = np.zeros_like(x)
    for key in ("haadf", "df", "bf"):
        old_tail = data[key + "_tail_fraction"]
        tail = old_tail * new_overlap / old_overlap
        coherent = data[key + "_coherent_scaled_fraction"] * ratio
        total = coherent + tail
        arrays[key + "_fraction"] = total
        arrays[key + "_tail_fraction"] = tail
        arrays[key + "_coherent_scaled_fraction"] = coherent
        old_collected_tail += old_tail
        new_collected_tail += tail
        statistics[key] = dict(old_mean=float(data[key + "_fraction"].mean()),
            corrected_mean=float(total.mean()), min_fraction=float(total.min()),
            max_fraction=float(total.max()), mean_tail_fraction=float(tail.mean()),
            tail_share=float(tail.sum() / total.sum()))
    old_tail_uncollected = probability * old_overlap * available - old_collected_tail
    new_tail_uncollected = probability * new_overlap * available - new_collected_tail
    pre_sample_loss = max(1. - incident, 0.)
    arrays["uncollected_fraction"] = (
        (data["uncollected_fraction"] - pre_sample_loss - old_tail_uncollected) * ratio
        + pre_sample_loss + new_tail_uncollected)
    arrays["absorbed_fraction"] = data["absorbed_fraction"]
    arrays["truncated_fraction"] = data["truncated_fraction"] * ratio
    budget = arrays["uncollected_fraction"] + arrays["absorbed_fraction"]
    for key in ("haadf", "df", "bf"):
        budget = budget + arrays[key + "_fraction"]
    error = float(np.max(np.abs(budget - 1.)))
    assert error < 1e-12
    for key, value in arrays.items():
        assert np.isfinite(value).all()
        if key.endswith("fraction"):
            assert value.min() >= -1e-12
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "raw_scan.npz", **arrays)
    report = dict(method="Probability recomposition using archived coherent multislice; no new wave propagation",
        source_directory=str(source.resolve()),
        input_sha256={name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                      for name in ("raw_scan.npz", "parameters.json", "metrics.json")},
        sigma_nm=sigma, old_overlap_min=float(old_overlap.min()),
        old_overlap_max=float(old_overlap.max()), new_overlap_min=float(new_overlap.min()),
        old_coherent_scale="1 - retained_tail_probability * old_overlap",
        corrected_coherent_scale="1 - retained_tail_probability * physical_envelope_overlap",
        maximum_probability_conservation_error=error, images=statistics,
        original_DPA_diameter_mm=.2, source_wave_grid=1024 if figures else parameters["grid"],
        no_new_wave_acquisition=True)
    (destination / "correction.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if figures:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.3), layout="constrained")
        extent = [x.min() - dx / 2, x.max() + dx / 2,
                  y.min() - dy / 2, y.max() + dy / 2]
        for key, ax in zip(("haadf", "df", "bf"), axes):
            value = arrays[key + "_fraction"]
            scaled = np.rint((value - value.min()) / np.ptp(value) * 65535).astype(np.uint16)
            Image.fromarray(np.flipud(scaled)).save(destination / f"{key}.png")
            tifffile.imwrite(destination / f"{key}.tiff", value.astype(np.float32), metadata={
                "axes": "YX", "pixel_size_nm": dx, "row_zero": "minimum sample Y",
                "unit": "fraction_of_emitted_electrons", "method": report["method"]})
            plot = ax.imshow(value, origin="lower", extent=extent, cmap="gray", interpolation="nearest")
            angle = parameters["physical_detector_reference_angles"][key]
            suffix = " (low-angle annulus)" if key == "df" else ""
            ax.set_title(f"{key.upper()} {angle['inner_mrad']:.2f}–{angle['outer_mrad']:.2f} mrad{suffix}\n"
                         f"Tail share {statistics[key]['tail_share']:.1%}", fontsize=9)
            ax.set(xlabel="Sample X (nm)", ylabel="Sample Y (nm)")
            fig.colorbar(plot, ax=ax, shrink=.8, label="Fraction of emitted electrons")
        fig.suptitle("Near-focus Si [110] · 64 × 64, 0.02 nm/pixel · 10 nm diameter, 5 nm thickness\n"
                     "Corrected finite-envelope tail; archived coherent multislice reused (no new wave acquisition)")
        fig.savefig(destination / "haadf_df_bf_comparison.png", dpi=180)
        plt.close(fig)
    return report


if __name__ == "__main__":
    source_root = ROOT / "outputs/si110_cif_5nm_64px_002nm"
    output_root = ROOT / "outputs/si110_underfocus_100nm"
    results = {}
    for source_name, label in (("benchmark_512_gpu_calibrated", "benchmark_512_tail_corrected"),
                               ("benchmark_1024_gpu_calibrated", "benchmark_1024_tail_corrected"),
                               ("acquisition_gpu_1024_final", "near_focus_tail_corrected")):
        results[label] = correct(source_root / source_name, output_root / label,
                                 figures=label.startswith("near_focus"))
    a, b = (results[f"benchmark_{grid}_tail_corrected"]["images"] for grid in (512, 1024))
    comparison = {key: dict(mean_512=a[key]["corrected_mean"], mean_1024=b[key]["corrected_mean"],
        relative_mean_difference=a[key]["corrected_mean"] / b[key]["corrected_mean"] - 1.)
        for key in ("haadf", "df", "bf")}
    (output_root / "corrected_grid_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))
