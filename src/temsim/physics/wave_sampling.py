"""Allocation-free planning of a padded wave domain at fixed spatial sampling.

The specimen envelope limits atoms, not the FFT domain. A dynamically enlarged
illumination/scan window must gain pixels rather than silently lose resolution.
Potential storage and conservative wave working memory are estimated separately;
neither is a promise about whole-application peak RAM.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class WaveSamplingPlan:
    field_of_view_angstrom: float
    pixels: int
    sampling_angstrom: float
    reference_sampling_angstrom: float
    estimated_potential_bytes: int
    estimated_working_bytes: int
    expanded: bool


def plan_wave_sampling(
    *,
    reference_fov_angstrom: float,
    reference_pixels: int,
    requested_fov_angstrom: float,
    thickness_angstrom: float,
    target_slice_thickness_angstrom: float,
    configuration_count: int = 1,
    atomistic: bool = True,
    max_potential_bytes: int = 4 * 1024**3,
    max_working_bytes: int = 8 * 1024**3,
) -> WaveSamplingPlan:
    """Preserve the configured FOV/grid spacing when a beam needs more space.

    User inputs remain unchanged. The returned grid is an execution detail;
    shrinking the requested window never reduces the user's pixel count.
    Finite windows expanded here have an even grid, keeping zero at N//2.
    """
    for name, value in (
        ("Reference wave FOV", reference_fov_angstrom),
        ("Required wave FOV", requested_fov_angstrom),
        ("Target slice thickness", target_slice_thickness_angstrom),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    if not math.isfinite(thickness_angstrom) or thickness_angstrom < 0.0:
        raise ValueError("Wave specimen thickness must be finite and non-negative.")
    if (not math.isfinite(reference_pixels)
            or int(reference_pixels) != reference_pixels or reference_pixels < 32):
        raise ValueError("Wave grid must contain at least 32 integer pixels.")
    if (not math.isfinite(configuration_count)
            or int(configuration_count) != configuration_count or configuration_count < 1):
        raise ValueError("Wave configuration count must be a positive integer.")
    if not math.isfinite(max_potential_bytes) or max_potential_bytes <= 0:
        raise ValueError("Wave potential storage limit must be positive.")
    if not math.isfinite(max_working_bytes) or max_working_bytes <= 0:
        raise ValueError("Wave working-memory limit must be positive.")

    sampling = reference_fov_angstrom / reference_pixels
    if not math.isfinite(sampling) or sampling <= 0.0:
        raise ValueError("Reference wave sampling is outside the numerical range.")
    required_pixels = requested_fov_angstrom / sampling
    if not math.isfinite(required_pixels):
        raise ValueError("Required wave FOV/grid ratio exceeds the numerical range.")
    pixels = max(int(reference_pixels), math.ceil(required_pixels))
    expanded = pixels > reference_pixels
    if expanded:
        pixels += pixels % 2
    slice_count = thickness_angstrom / target_slice_thickness_angstrom
    if not math.isfinite(slice_count):
        raise ValueError("Wave slice count is outside the numerical range.")
    slices = max(1, math.ceil(slice_count))
    bytes_per_pixel = 4 * slices * int(configuration_count) + 8 if atomistic else 8
    storage = pixels * pixels * bytes_per_pixel
    if storage > max_potential_bytes:
        raise ValueError(
            f"Wave window {requested_fov_angstrom * 0.1:.6g} nm needs "
            f"{pixels:,} x {pixels:,} pixels to retain "
            f"{sampling * 100.0:.6g} pm sampling; potential storage alone is "
            f"{storage / 1024**3:.3g} GiB (limit "
            f"{max_potential_bytes / 1024**3:.3g} GiB). "
            "Reduce probe defocus or scan extent, or explicitly revise the "
            "wave sampling. Reducing specimen diameter only reduces atoms; "
            "the illuminated vacuum still needs wave propagation."
        )
    # Reserve eight CPU probes plus FFT/coordinate/mask scratch, retained
    # configuration waves and two potential copies. Dynamic STEM grids are
    # unknown to the controller's reference-grid preflight. Bound this extra
    # working set here even for vacuum (whose potential alone is very small).
    working = pixels * pixels * (1024 + 48 * int(configuration_count)) + 2 * storage
    if working > max_working_bytes:
        raise ValueError(
            f"Wave grid {pixels:,} x {pixels:,} needs approximately "
            f"{working / 1024**3:.3g} GiB of wave working memory "
            f"(limit {max_working_bytes / 1024**3:.3g} GiB), including FFT/probe "
            "buffers. Reduce probe defocus or scan extent; the grid will not "
            "be silently coarsened."
        )
    return WaveSamplingPlan(
        field_of_view_angstrom=float(requested_fov_angstrom),
        pixels=pixels,
        sampling_angstrom=requested_fov_angstrom / pixels,
        reference_sampling_angstrom=sampling,
        estimated_potential_bytes=storage,
        estimated_working_bytes=working,
        expanded=expanded,
    )
