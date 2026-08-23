"""Generate reproducible 30 mrad Si[110] STEM HAADF/BF images.

The condenser preset is traced first.  Because the current project calibration
does not reach 30 mrad at the specimen, the last-plane ray bundle is explicitly
recalibrated to a focused 30 mrad (current-weighted 95 percent) probe.  Detector
masks remain those of the physical HAADF and BF planes in the current column.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from temsim.detector.stem_signal import physical_angular_detectors
from temsim.operating_modes import apply_operating_mode_pair
from temsim.optics.column import default_state
from temsim.physics.beam_statistics import branch_sample_statistics
from temsim.physics.simulation import run
from temsim.physics.stem_wave_imaging import simulate_angle_resolved_stem


TARGET_ALPHA_RAD = 30.0e-3
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "si110_30mrad"


def _impose_focused_probe(incident, target_alpha_rad: float) -> tuple[float, float]:
    """Set the specimen-plane waist and scale centred slopes to target alpha95."""

    initial = branch_sample_statistics(incident)
    incident.x[-1, :] = 0.0
    incident.y[-1, :] = 0.0
    incident.tx[-1, :] -= initial.mean_tx_rad
    incident.ty[-1, :] -= initial.mean_ty_rad
    for _ in range(4):
        current = branch_sample_statistics(incident)
        if current.convergence_95_rad <= 0.0:
            raise RuntimeError("Cannot calibrate a zero-angle incident bundle")
        scale = target_alpha_rad / current.convergence_95_rad
        incident.tx[-1, :] *= scale
        incident.ty[-1, :] *= scale
    final = branch_sample_statistics(incident)
    return initial.convergence_95_rad, final.convergence_95_rad


def _limits(image: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(image, (0.5, 99.5))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(np.min(image)), float(np.max(image))
    return float(low), float(high)


def _save_single(
    image: np.ndarray,
    extent: list[float],
    name: str,
    title: str,
    cmap: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    low, high = _limits(image)
    plotted = ax.imshow(
        image,
        origin="lower",
        extent=extent,
        cmap=cmap,
        interpolation="nearest",
        vmin=low,
        vmax=high,
    )
    ax.set_title(title)
    ax.set_xlabel("X at specimen (nm)")
    ax.set_ylabel("Y at specimen (nm)")
    fig.colorbar(plotted, ax=ax, label="Collected probability fraction")
    fig.savefig(OUTPUT_DIR / name, dpi=180)
    plt.close(fig)


def main() -> None:
    started = perf_counter()
    state = default_state()
    apply_operating_mode_pair(state, "nano_probe", "diffraction")
    state.sample.inserted = True
    state.sample.specimen_mode = "atomic"
    state.sample.specimen_preset_key = "si_110"
    state.sample.thickness_nm = 10.0
    state.sample.stem_wave_enabled = True
    state.sample.wave_grid_pixels = 512
    state.sample.wave_field_of_view_angstrom = 40.0
    # One static 10 nm IAM slice keeps the full reciprocal-grid support.  This
    # is an atomistic projected-phase calculation, not a channeling/frozen-
    # phonon production multislice result.
    state.sample.wave_multislice_enabled = True
    state.sample.wave_atomistic_enabled = True
    state.sample.wave_slice_thickness_angstrom = 100.0
    state.sample.wave_bandwidth_fraction = 1.0
    state.sample.wave_frozen_phonon_enabled = False
    state.sample.real_high_angle_tail_enabled = False
    state.sample.stem_poisson_enabled = False
    state.sample.wave_defocus_nm = 0.0
    state.acceleration_enabled = True
    state.acceleration_backend = "CUDA GPU"

    simulation = run(state)
    alpha_before, alpha_after = _impose_focused_probe(
        simulation.incident, TARGET_ALPHA_RAD
    )

    selected = tuple(
        detector
        for detector in state.stem_detectors
        if detector.key in {"haadf", "bf"}
    )
    physical_detectors, _angles = physical_angular_detectors(state, selected)

    pixels = 64
    pitch_nm = 0.025
    axis_nm = (np.arange(pixels, dtype=float) - (pixels - 1) / 2.0) * pitch_nm
    scan_x_nm, scan_y_nm = np.meshgrid(axis_nm, axis_nm, indexing="xy")
    result = simulate_angle_resolved_stem(
        state,
        simulation,
        physical_detectors,
        scan_x_nm * 1.0e-3,
        scan_y_nm * 1.0e-3,
    )

    haadf = np.asarray(result.fractions["haadf"], dtype=float)
    bf = np.asarray(result.fractions["bf"], dtype=float)
    total = np.asarray(result.uncollected_fraction, dtype=float) + haadf + bf
    conservation_error = float(np.max(np.abs(total - 1.0)))
    detector_ranges = {
        key: [float(values[0]), float(values[1])]
        for key, values in result.detector_ranges_mrad.items()
    }
    max_angle = float(result.maximum_isotropic_angle_mrad)
    effective_ranges = {
        key: [inner, min(outer, max_angle)]
        for key, (inner, outer) in detector_ranges.items()
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_DIR / "si110_30mrad_raw.npz",
        scan_x_nm=scan_x_nm,
        scan_y_nm=scan_y_nm,
        haadf_fraction=haadf,
        bf_fraction=bf,
    )
    extent = [axis_nm[0], axis_nm[-1], axis_nm[0], axis_nm[-1]]
    haadf_effective = effective_ranges["haadf"]
    bf_effective = effective_ranges["bf"]
    _save_single(
        haadf,
        extent,
        "si110_30mrad_haadf.png",
        (
            "Si [110] HAADF, 30.00 mrad, "
            f"{haadf_effective[0]:.1f}-{haadf_effective[1]:.1f} mrad represented"
        ),
        "inferno",
    )
    _save_single(
        bf,
        extent,
        "si110_30mrad_bf.png",
        (
            "Si [110] BF, 30.00 mrad, "
            f"{bf_effective[0]:.1f}-{bf_effective[1]:.1f} mrad"
        ),
        "gray",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8), constrained_layout=True)
    for ax, image, label, cmap in (
        (axes[0], haadf, "HAADF", "inferno"),
        (axes[1], bf, "BF", "gray"),
    ):
        low, high = _limits(image)
        plotted = ax.imshow(
            image,
            origin="lower",
            extent=extent,
            cmap=cmap,
            interpolation="nearest",
            vmin=low,
            vmax=high,
        )
        ax.set_title(label)
        ax.set_xlabel("X (nm)")
        ax.set_ylabel("Y (nm)")
        fig.colorbar(plotted, ax=ax, label="Collected probability fraction")
    fig.suptitle(
        "Si [110], imposed focused 30.00 mrad probe, 300 kV, "
        "static atomistic projected-phase"
    )
    fig.savefig(OUTPUT_DIR / "si110_30mrad_haadf_bf.png", dpi=180)
    plt.close(fig)

    metadata = {
        "specimen": "Silicon [110]",
        "model": result.metrics.get("specimen_model"),
        "potential_model": result.metrics.get("specimen_potential_model"),
        "atomistic_applied": result.metrics.get("specimen_atomistic_applied"),
        "atom_count": result.metrics.get("specimen_atom_count"),
        "beam_energy_kv": state.beam_voltage_kv,
        "thickness_nm": state.sample.thickness_nm,
        "probe_convergence_95_mrad_before_imposed_calibration": alpha_before * 1e3,
        "probe_convergence_95_mrad": alpha_after * 1e3,
        "probe_waist_offset_nm": result.metrics.get("probe_ray_waist_offset_nm"),
        "configured_defocus_nm": result.metrics.get("probe_configured_defocus_nm"),
        "effective_defocus_nm": result.metrics.get("probe_effective_defocus_nm"),
        "scan_shape_yx": list(haadf.shape),
        "scan_pixel_size_nm": pitch_nm,
        "scan_fov_x_nm": float(np.ptp(scan_x_nm)),
        "scan_fov_y_nm": float(np.ptp(scan_y_nm)),
        "wave_grid_pixels": state.sample.wave_grid_pixels,
        "wave_field_of_view_angstrom": result.metrics.get("field_of_view_angstrom"),
        "maximum_isotropic_angle_mrad": max_angle,
        "physical_detector_ranges_mrad": detector_ranges,
        "represented_detector_ranges_mrad": effective_ranges,
        "truncated_detector_keys": list(result.metrics.get("truncated_detector_keys", ())),
        "haadf_mean_fraction": float(np.mean(haadf)),
        "haadf_min_fraction": float(np.min(haadf)),
        "haadf_max_fraction": float(np.max(haadf)),
        "bf_mean_fraction": float(np.mean(bf)),
        "bf_min_fraction": float(np.min(bf)),
        "bf_max_fraction": float(np.max(bf)),
        "maximum_probability_conservation_error": conservation_error,
        "real_probability_conserved": conservation_error <= 5.0e-10,
        "wave_compute_backend": result.metrics.get("wave_compute_backend"),
        "elapsed_s": perf_counter() - started,
        "scope": (
            "qualitative static atomistic projected-phase Si[110]; physical "
            "detector masks; no frozen phonons/channeling; HAADF outer angle "
            "truncated by reciprocal-grid angular support"
        ),
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
