"""Resource policy changes throughput only, never physical sampling."""
import pytest

from temsim.physics import stem_batching
from temsim.physics.stem_batching import estimate_stem_cuda_batch_size


def _batch(**overrides):
    values = dict(scan_positions=10000, grid_shape=(256, 256),
                  free_device_bytes=30 * 1024**3, potential_bytes=50 * 256**2 * 4,
                  detector_count=3)
    values.update(overrides)
    return estimate_stem_cuda_batch_size(**values)


def test_large_gpu_uses_more_than_eight_probes_but_not_whole_raster():
    assert _batch() == 128
    assert 8 < _batch(dynamic_masks=True, recording_plane_count=4) <= 128
    assert _batch(scan_positions=3) <= 3


def test_memory_and_physical_router_bound_batch_size():
    small = _batch(free_device_bytes=256 * 1024**2)
    assert 1 <= small < _batch()
    assert _batch(free_device_bytes=0) == 1
    assert _batch(potential_bytes=16 * 1024**3) == 1
    assert _batch(grid_shape=(2048, 2048), dynamic_masks=True,
                  recording_plane_count=10) == 1
    assert _batch(dynamic_masks=True, recording_plane_count=768) < _batch(
        dynamic_masks=True, recording_plane_count=4)


def test_no_cuda_discovery_keeps_small_batch(monkeypatch):
    from temsim.physics import compute_backend

    def unavailable():
        raise RuntimeError("No GPU")

    monkeypatch.setattr(compute_backend, "cupy_module", unavailable)
    assert stem_batching.resident_stem_batch_size(10000, (256, 256)) == 8
    assert stem_batching.resident_stem_batch_size(3, (256, 256)) == 3


@pytest.mark.parametrize("invalid", [dict(scan_positions=0), dict(grid_shape=(0, 32)),
                                     dict(free_device_bytes=-1), dict(detector_count=-1)])
def test_invalid_memory_estimates_rejected(invalid):
    with pytest.raises(ValueError):
        _batch(**invalid)
