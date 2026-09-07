"""Prepared signal masks agree exactly with the diagnostic recording router."""

from dataclasses import replace

import numpy as np
import pytest

from temsim.physics.first_order import TransverseTransfer
from temsim.physics.record_plane import (
    PlaneStop,
    RecordPlanePlan,
    prepare_record_plane_detector_masks,
    route_record_planes,
)


def _fixture(*, dynamic=False):
    planes = (
        PlaneStop("ap", "Shifted aperture", 1, "aperture", "disk", radius_mm=0.14,
                  offset_x_mm=0.015, offset_y_mm=-0.008),
        PlaneStop("monitor", "Nonblocking monitor", 2, "detector", "square",
                  outer_width_mm=0.1, readout_enabled=True, non_blocking=True),
        PlaneStop("disabled", "Disabled readout", 3, "detector", "disk",
                  outer_width_mm=0.06, readout_enabled=False),
        PlaneStop("annular", "Annular detector", 4, "detector", "annulus",
                  outer_width_mm=0.16, inner_diameter_mm=0.07, readout_enabled=True,
                  offset_x_mm=-0.006, offset_y_mm=0.003),
        PlaneStop("camera", "Camera", 5, "detector", "camera",
                  outer_width_mm=0.8, readout_enabled=True),
    )
    transfers = tuple(
        TransverseTransfer(
            0.0, plane.z_mm,
            np.array([[1.2 + index * 0.1, -0.25], [0.3, -0.9 - index * 0.05]]),
            np.array([[-0.034 - index * 0.002, 0.007], [-0.004, 0.041 + index * 0.002]]),
            np.array([[0.02, -0.01], [-0.03, 0.04]]),
            np.array([[0.9, 0.1], [-0.2, 1.1]]),
            position_offset_m=(index * 2e-6, -index * 1e-6),
            angle_offset_rad=(1e-4, -2e-4),
        )
        for index, plane in enumerate(planes)
    )
    times = np.arange(7, dtype=float)[None] * 0.1
    offsets = tuple(
        np.stack((np.linspace(-2e-5, 2e-5, 7) * (index + 1) / 5,
                  np.sin(times.ravel()) * 8e-6), axis=-1)[None]
        for index in range(len(planes))
    ) if dynamic else ()
    plan = RecordPlanePlan(
        0.0, planes, transfers, "a" * 64, "b" * 64,
        scan_times_s=times if dynamic else None,
        scan_position_offsets_m=offsets,
        time_dependent_deflection=dynamic,
    )
    angle_x, angle_y = np.meshgrid(np.linspace(-0.004, 0.004, 13), np.linspace(-0.003, 0.003, 11))
    angles = np.stack((angle_x, angle_y), axis=-1)[None]
    positions = np.stack((np.linspace(-1e-5, 1e-5, 7), np.linspace(2e-6, -3e-6, 7)), axis=-1)[:, None, None]
    return plan, positions, angles


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("batch_size", [1, 3, 7])
def test_prepared_masks_exactly_match_signed_mixed_plane_reference(dynamic, batch_size):
    plan, positions, angles = _fixture(dynamic=dynamic)
    prepared = prepare_record_plane_detector_masks(plan, angles)
    actual = []
    for start in range(0, len(positions), batch_size):
        stop = min(start + batch_size, len(positions))
        scan_slice = slice(start, stop)
        masks = prepared(positions[start:stop], scan_slice=scan_slice)
        reference = route_record_planes(plan, positions[start:stop], angles, scan_slice=scan_slice)
        detector_interactions = [item for item in reference.interactions if item.plane.kind == "detector"]
        assert set(masks) == {item.plane.key for item in detector_interactions}
        for item in detector_interactions:
            np.testing.assert_array_equal(masks[item.plane.key], item.signal_mask)
            assert not masks[item.plane.key].flags.writeable
        actual.append(masks)
    assert sum(np.count_nonzero(item["monitor"]) for item in actual) > 0
    assert sum(np.count_nonzero(item["annular"]) for item in actual) > 0
    assert sum(np.count_nonzero(item["camera"]) for item in actual) > 0
    assert not any(np.any(item["disabled"]) for item in actual)


def test_disabled_readout_still_blocks_and_nonblocking_monitor_does_not():
    plan, positions, angles = _fixture()
    result = prepare_record_plane_detector_masks(plan, angles)(positions)
    nonblocking_disabled = replace(plan, planes=tuple(
        replace(plane, non_blocking=True) if plane.key == "disabled" else plane
        for plane in plan.planes
    ))
    changed = prepare_record_plane_detector_masks(nonblocking_disabled, angles)(positions)
    assert sum(np.count_nonzero(changed[key]) for key in ("annular", "camera")) > sum(
        np.count_nonzero(result[key]) for key in ("annular", "camera")
    )
    blocking_monitor = replace(plan, planes=tuple(
        replace(plane, non_blocking=False) if plane.key == "monitor" else plane
        for plane in plan.planes
    ))
    blocked = prepare_record_plane_detector_masks(blocking_monitor, angles)(positions)
    assert sum(np.count_nonzero(blocked[key]) for key in ("annular", "camera")) < sum(
        np.count_nonzero(result[key]) for key in ("annular", "camera")
    )


def test_preparation_reuses_angular_maps_and_owns_its_input_snapshot(monkeypatch):
    plan, positions, angles = _fixture()
    baseline = prepare_record_plane_detector_masks(plan, angles)(positions)
    original_einsum = np.einsum
    calls = []

    def counted(*args, **kwargs):
        calls.append(np.shape(args[-1]))
        return original_einsum(*args, **kwargs)

    monkeypatch.setattr(np, "einsum", counted)
    prepared = prepare_record_plane_detector_masks(plan, angles)
    assert len(calls) == len(plan.planes)
    angles[:] = 0.0
    for _ in range(2):
        masks = prepared(positions)
        for key in baseline:
            np.testing.assert_array_equal(masks[key], baseline[key])
    # Only J_img multiplies the small (B,1,1,2) position input per batch.
    assert len(calls) == 3 * len(plan.planes)
    assert all(shape == positions.shape for shape in calls[len(plan.planes):])


@pytest.mark.parametrize("angle", [np.zeros((3, 3)), np.array([np.nan, 0.0])])
def test_preparation_rejects_invalid_angles(angle):
    plan, _, _ = _fixture()
    with pytest.raises(ValueError, match="dimension 2|finite"):
        prepare_record_plane_detector_masks(plan, angle)


@pytest.mark.parametrize("position", [
    np.zeros((3, 3)), np.array([np.inf, 0.0]), np.zeros((4, 4, 2)),
])
def test_prepared_router_rejects_invalid_positions(position):
    plan, _, angles = _fixture()
    prepared = prepare_record_plane_detector_masks(plan, angles)
    with pytest.raises(ValueError, match="dimension 2|finite|broadcast-compatible"):
        prepared(position)


def test_prepared_router_rejects_mismatched_raster_slice():
    plan, positions, angles = _fixture(dynamic=True)
    prepared = prepare_record_plane_detector_masks(plan, angles)
    with pytest.raises(ValueError, match="offsets do not match"):
        prepared(positions[:3], scan_slice=slice(0, 2))
    with pytest.raises(ValueError, match="offsets do not match"):
        prepared(positions[:3])


def test_zero_aperture_and_empty_plan_are_supported():
    plan, positions, angles = _fixture()
    closed = replace(plan, planes=(replace(plan.planes[0], radius_mm=0.0), *plan.planes[1:]))
    assert not any(np.any(mask) for mask in prepare_record_plane_detector_masks(closed, angles)(positions).values())
    empty = replace(plan, planes=(), transfers=())
    assert prepare_record_plane_detector_masks(empty, angles)(positions) == {}
