"""Bound STEM GPU batches by scratch space, not by the number of scan pixels.

These are working-set estimates, not an allocation guarantee. The existing
whole-frame CPU retry remains authoritative if CUDA cannot allocate a plan.
No physical sampling parameter is changed by this policy.
"""
from __future__ import annotations


def estimate_stem_cuda_batch_size(
    scan_positions: int,
    grid_shape: tuple[int, int],
    *,
    free_device_bytes: int,
    potential_bytes: int,
    detector_count: int,
    recording_plane_count: int = 0,
    dynamic_masks: bool = False,
) -> int:
    if scan_positions < 1 or len(grid_shape) != 2 or min(grid_shape) < 1:
        raise ValueError("STEM scan and wave-grid sizes must be positive")
    if min(free_device_bytes, potential_bytes, detector_count, recording_plane_count) < 0:
        raise ValueError("STEM memory estimates and component counts cannot be negative")
    grid_points = int(grid_shape[0]) * int(grid_shape[1])
    mib = 1024**2
    # Keep at least half the current free VRAM for the application, other
    # calculations and FFT work areas. Account for potential + plan storage.
    device_budget = max(0, min(2048 * mib,
        free_device_bytes // 2 - 2 * potential_bytes - 64 * mib))
    # Complex waves, FFT buffers/workspace, real reductions and Boolean masks.
    device_per_probe = grid_points * (160 + detector_count)
    capacity = min(128, scan_positions, max(1, device_budget // device_per_probe))
    if dynamic_masks:
        # Prepared angular maps are shared across probes; only detector masks
        # and current-plane coordinates are materialised for each batch.
        host_fixed = grid_points * 16 * recording_plane_count
        host_per_probe = grid_points * (128 + detector_count)
        host_budget = max(0, 1024 * mib - host_fixed)
        capacity = min(capacity, max(1, host_budget // host_per_probe))
    # Stable power-of-two batches reduce FFT-plan variants as raster sizes vary.
    return min(scan_positions, 1 << (int(capacity).bit_length() - 1))


def resident_stem_batch_size(scan_positions, grid_shape, **kwargs) -> int:
    """Discover free GPU memory lazily; keep the old small batch on failure."""
    try:
        from temsim.physics.compute_backend import cupy_module

        cp = cupy_module()
        free_bytes, _total_bytes = cp.cuda.runtime.memGetInfo()
    except Exception:
        return min(8, scan_positions)
    return estimate_stem_cuda_batch_size(
        scan_positions, grid_shape, free_device_bytes=int(free_bytes), **kwargs)
