"""Lossless numeric TEM export, independent of display contrast controls."""
from pathlib import Path

import numpy as np


def export_wave_image(result, path):
    """Write an allow_pickle=False NPZ with raw arrays and execution metadata."""
    if result.camera_intensity is None or result.absolute_diffraction_probability is None:
        raise ValueError("Recalculate this legacy TEM result to export explicit raw probabilities")
    from temsim.immutable_json import canonical_json_bytes
    execution = result.metrics.get("wave_execution_manifest")
    if execution is None:
        raise ValueError("TEM result has no execution manifest")
    dx_m = float(result.camera_x_mm[1] - result.camera_x_mm[0]) * 1e-3
    dy_m = float(result.camera_y_mm[1] - result.camera_y_mm[0]) * 1e-3
    metadata = {
        "schema": "tem-raw-probability-export-v1",
        "preset_key": result.preset_key,
        "execution": execution,
        "request_manifest": result.request_manifest,
        "source_request_digest": result.metrics.get("wave_source_request_digest"),
        "pixel_area_m2": dx_m * dy_m,
        "probability_reference": execution["reference_plane"],
        "count_conversion": "I_ref[A] * exposure[s] / elementary_charge[C] * pixel_probability; I_ref is current in the named conditional branch",
        "zero_loss_probability_per_sample_incident": result.metrics.get("zero_loss_probability_per_sample_incident"),
        "source_compute_backend": result.metrics.get("wave_compute_backend"),
        "source_numeric_precision": result.metrics.get("specimen_numeric_precision"),
        "display_arrays": ["image_display", "diffraction_display"],
        "exit_wave_scope": result.metrics.get("exit_wave_representation"),
    }
    # Store metadata as Unicode, never a pickled Python object.
    encoded = canonical_json_bytes(metadata).decode("utf-8")
    with Path(path).open("wb") as stream:
        np.savez_compressed(stream,
            camera_x_mm=result.camera_x_mm, camera_y_mm=result.camera_y_mm,
            camera_density_pre_psf_per_m2=result.camera_electron_optical_intensity,
            camera_density_post_psf_per_m2=result.camera_intensity,
            camera_pixel_probability=result.camera_intensity * dx_m * dy_m,
            absolute_exit_diffraction_probability=result.absolute_diffraction_probability,
            specimen_x_angstrom=result.x_angstrom, specimen_y_angstrom=result.y_angstrom,
            spatial_frequency_x_inv_angstrom=result.spatial_frequency_inv_angstrom,
            spatial_frequency_y_inv_angstrom=result.spatial_frequency_y_inv_angstrom,
            image_display=result.image_intensity, diffraction_display=result.diffraction_intensity,
            metadata_json=np.array(encoded))
    return Path(path)
