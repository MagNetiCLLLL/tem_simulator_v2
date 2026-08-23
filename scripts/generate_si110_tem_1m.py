"""Calculate the current 1,000,000x TEM image of the Si[110] preset."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from temsim.assembly_catalog import AssemblyCatalog
from temsim.operating_modes import apply_operating_mode_pair
from temsim.optics.column import default_state
from temsim.optics.direct_alignment import apply_direct_alignment
from temsim.physics.simulation import run
from temsim.physics.wave_imaging import (
    estimate_tem_wave_memory_bytes,
    simulate_wave_image,
)
from temsim.specimen.atomistic import atomistic_capability


OUTPUT_DIR = ROOT / "outputs" / "si110_tem_1m"
TARGET_MAGNIFICATION = 1_000_000.0
HIGH_ACCURACY_RAYS = 15_000
HIGH_ACCURACY_STEP_MM = 0.1


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _percentile_limits(array: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(np.asarray(array, dtype=float), (0.5, 99.5))
    if not math.isfinite(float(low)) or not math.isfinite(float(high)):
        return float(np.nanmin(array)), float(np.nanmax(array))
    if high <= low:
        return float(np.min(array)), float(np.max(array))
    return float(low), float(high)


def _save_images(result) -> None:
    image = np.asarray(result.image_intensity, dtype=float)
    diffraction = np.asarray(result.diffraction_intensity, dtype=float)
    potential = np.asarray(
        result.projected_potential_v_angstrom, dtype=float
    )
    extent_image = [
        float(result.camera_x_mm[0]),
        float(result.camera_x_mm[-1]),
        float(result.camera_y_mm[0]),
        float(result.camera_y_mm[-1]),
    ]
    extent_diffraction = [
        float(result.spatial_frequency_inv_angstrom[0]),
        float(result.spatial_frequency_inv_angstrom[-1]),
        float(result.spatial_frequency_y_inv_angstrom[0]),
        float(result.spatial_frequency_y_inv_angstrom[-1]),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 6.4), constrained_layout=True)
    low, high = _percentile_limits(image)
    plotted = ax.imshow(
        image,
        origin="lower",
        extent=extent_image,
        cmap="gray",
        interpolation="nearest",
        vmin=low,
        vmax=high,
    )
    ax.set_title("Si [110] physical Camera image, Direct Alignment 1,000,000x")
    ax.set_xlabel("Camera X (mm)")
    ax.set_ylabel("Camera Y (mm)")
    fig.colorbar(plotted, ax=ax, label="Display-normalised intensity")
    fig.savefig(OUTPUT_DIR / "si110_tem_1m.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(
        1, 3, figsize=(17.5, 5.5), constrained_layout=True
    )
    specimen_extent = [
        float(result.x_angstrom[0]),
        float(result.x_angstrom[-1]),
        float(result.y_angstrom[0]),
        float(result.y_angstrom[-1]),
    ]
    panels = (
        (potential, specimen_extent, "Projected potential", "viridis"),
        (image, extent_image, "Physical Camera image", "gray"),
        (
            diffraction,
            extent_diffraction,
            "Log-scaled diffraction",
            "magma",
        ),
    )
    for axis, (array, extent, title, cmap) in zip(axes, panels):
        low, high = _percentile_limits(array)
        plotted = axis.imshow(
            array,
            origin="lower",
            extent=extent,
            cmap=cmap,
            interpolation="nearest",
            vmin=low,
            vmax=high,
        )
        axis.set_title(title)
        if array is diffraction:
            axis.set_xlabel("Spatial frequency (1/angstrom)")
            axis.set_ylabel("Spatial frequency (1/angstrom)")
        elif array is image:
            axis.set_xlabel("Camera X (mm)")
            axis.set_ylabel("Camera Y (mm)")
        else:
            axis.set_xlabel("Specimen X (angstrom)")
            axis.set_ylabel("Specimen Y (angstrom)")
        fig.colorbar(plotted, ax=axis)
    fig.suptitle(
        "Current Si [110] TEM calculation at 1,000,000x projector setting"
    )
    fig.savefig(OUTPUT_DIR / "si110_tem_1m_overview.png", dpi=180)
    plt.close(fig)


def main() -> int:
    started = perf_counter()
    capability = atomistic_capability()
    if not capability.available:
        raise RuntimeError(capability.detail)

    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, catalog.default_selection())
    apply_operating_mode_pair(state, "micro_probe", "imaging")
    alignment = apply_direct_alignment(
        state, "image_magnification", TARGET_MAGNIFICATION
    )
    if not alignment.success:
        raise RuntimeError(alignment.message)

    # A physical Camera exposure requires the Camera in the beam.  Retract the
    # upstream recording surfaces for this dedicated TEM Camera calculation.
    for plane in state.recording_planes:
        plane.inserted = plane.key == "camera"

    emitter = getattr(state.electron_gun, "emitter", None)
    if emitter is not None:
        emitter.ray_count = HIGH_ACCURACY_RAYS
    else:
        state.electron_gun.ray_count = HIGH_ACCURACY_RAYS
    state.step_mm = HIGH_ACCURACY_STEP_MM
    state.history_step_mm = 0.5

    sample = state.sample
    sample.inserted = True
    sample.specimen_mode = "atomic"
    sample.specimen_preset_key = "si_110"
    sample.thickness_nm = 10.0
    sample.wave_enabled = True
    sample.wave_grid_pixels = 256
    sample.wave_field_of_view_angstrom = 40.0
    sample.wave_defocus_nm = 0.0
    sample.wave_multislice_enabled = True
    sample.wave_atomistic_enabled = True
    sample.wave_slice_thickness_angstrom = 2.0
    sample.wave_bandwidth_fraction = 2.0 / 3.0
    sample.wave_frozen_phonon_enabled = False

    simulation = run(state)
    wave = simulate_wave_image(state, simulation)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        OUTPUT_DIR / "si110_tem_1m_raw.npz",
        x_angstrom=wave.x_angstrom,
        y_angstrom=wave.y_angstrom,
        camera_x_mm=wave.camera_x_mm,
        camera_y_mm=wave.camera_y_mm,
        image_intensity=wave.image_intensity,
        camera_electron_optical_intensity=(
            wave.camera_electron_optical_intensity
        ),
        projected_potential_v_angstrom=(
            wave.projected_potential_v_angstrom
        ),
        exit_wave=wave.exit_wave,
        linear_diffraction_probability=wave.linear_diffraction_probability,
        diffraction_intensity=wave.diffraction_intensity,
        spatial_frequency_x_inv_angstrom=(
            wave.spatial_frequency_inv_angstrom
        ),
        spatial_frequency_y_inv_angstrom=(
            wave.spatial_frequency_y_inv_angstrom
        ),
    )
    tifffile.imwrite(
        OUTPUT_DIR / "si110_tem_1m_image_float32.tif",
        np.asarray(wave.image_intensity, dtype=np.float32),
        metadata={
            "axes": "YX",
            "unit": "display_normalised_intensity",
            "pixel_size_mm_xy": wave.metrics["camera_pixel_size_mm_xy"],
        },
    )
    _save_images(wave)

    image = np.asarray(wave.image_intensity, dtype=float)
    metrics = _jsonable(wave.metrics)
    metadata = {
        "specimen": wave.preset_name,
        "zone_axis": "[110]",
        "beam_energy_kv": state.beam_voltage_kv,
        "illumination_mode": state.illumination_mode,
        "projector_mode": state.projector_mode,
        "requested_magnification": TARGET_MAGNIFICATION,
        "achieved_magnification": alignment.achieved,
        "magnification_relative_error": abs(
            alignment.achieved / TARGET_MAGNIFICATION - 1.0
        ),
        "image_conjugacy_residual_um": alignment.constraint_value,
        "direct_alignment_iterations": alignment.iterations,
        "lens_strengths_percent": alignment.strengths,
        "thickness_nm": sample.thickness_nm,
        "wave_grid_shape_yx": list(image.shape),
        "wave_grid_pixels_requested": sample.wave_grid_pixels,
        "wave_field_of_view_angstrom_requested": (
            sample.wave_field_of_view_angstrom
        ),
        "wave_slice_thickness_angstrom_requested": (
            sample.wave_slice_thickness_angstrom
        ),
        "wave_bandwidth_fraction": sample.wave_bandwidth_fraction,
        "frozen_phonon_enabled": sample.wave_frozen_phonon_enabled,
        "high_accuracy_rays": HIGH_ACCURACY_RAYS,
        "ray_integration_step_mm": HIGH_ACCURACY_STEP_MM,
        "estimated_tem_wave_memory_bytes": estimate_tem_wave_memory_bytes(
            state
        ),
        "image_min": float(np.min(image)),
        "image_max": float(np.max(image)),
        "image_mean": float(np.mean(image)),
        "image_std": float(np.std(image)),
        "atomistic_capability": capability.detail,
        "wave_metrics": metrics,
        "elapsed_s": perf_counter() - started,
        "scope": (
            "Current non-OEM engineering simulation. The source-to-specimen "
            "ray state conditions the incident wave; static atomistic Lobato "
            "IAM multislice is followed by the Objective pupil, the complete "
            "Objective/D/I/P1/P2 projector transfer, affine post-specimen "
            "deflection, physical Camera sampling and the configured Camera "
            "point-spread response. Frozen phonons are disabled."
        ),
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
