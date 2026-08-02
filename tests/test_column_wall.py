from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics.column_wall import COLUMN_WALL_KEY, clip_column_wall


def test_column_wall_uses_exact_first_segment_intersection():
    state = SimpleNamespace(column_inner_diameter_mm=2.0)
    z_mm = np.array([0.0, 10.0, 20.0])
    x_m = np.array([[0.0], [0.0005], [0.0015]])
    y_m = np.zeros_like(x_m)

    alive, blocked_z, blocked_key = clip_column_wall(
        state, z_mm, x_m, y_m
    )

    assert alive.tolist() == [False]
    assert blocked_z.tolist() == pytest.approx([15.0])
    assert blocked_key == [COLUMN_WALL_KEY]


def test_touching_column_wall_counts_as_a_stop_in_either_axis():
    state = SimpleNamespace(column_inner_diameter_mm=2.0)
    z_mm = np.array([0.0, 4.0, 8.0])
    x_m = np.zeros((3, 2))
    y_m = np.array([
        [0.0, 0.0],
        [0.001, 0.0002],
        [0.0012, 0.0014],
    ])

    alive, blocked_z, blocked_key = clip_column_wall(
        state, z_mm, x_m, y_m
    )

    assert alive.tolist() == [False, False]
    assert blocked_z[0] == pytest.approx(4.0)
    assert 4.0 < blocked_z[1] < 8.0
    assert blocked_key == [COLUMN_WALL_KEY, COLUMN_WALL_KEY]


def test_vacuum_bore_step_stops_a_ray_at_the_narrowing_shoulder():
    segments = (
        SimpleNamespace(
            start_z_mm=0.0, end_z_mm=10.0, inner_diameter_mm=4.0
        ),
        SimpleNamespace(
            start_z_mm=10.0, end_z_mm=20.0, inner_diameter_mm=2.0
        ),
    )
    state = SimpleNamespace(
        _resolved_assembly=SimpleNamespace(vacuum_bore_segments=segments)
    )
    z_mm = np.array([0.0, 20.0])
    x_m = np.array([[0.0015], [0.0015]])
    y_m = np.zeros_like(x_m)

    alive, blocked_z, blocked_key = clip_column_wall(
        state, z_mm, x_m, y_m
    )

    assert alive.tolist() == [False]
    assert blocked_z.tolist() == pytest.approx([10.0])
    assert blocked_key == [COLUMN_WALL_KEY]


def test_column_wall_only_clips_where_a_wall_segment_exists():
    segments = (
        SimpleNamespace(
            start_z_mm=0.0, end_z_mm=10.0, inner_diameter_mm=2.0
        ),
    )
    state = SimpleNamespace(
        _resolved_assembly=SimpleNamespace(vacuum_bore_segments=segments)
    )
    z_mm = np.array([0.0, 10.0, 20.0])
    x_m = np.array([[0.0], [0.0005], [0.004]])
    y_m = np.zeros_like(x_m)

    alive, blocked_z, blocked_key = clip_column_wall(
        state, z_mm, x_m, y_m
    )

    assert alive.tolist() == [True]
    assert np.isnan(blocked_z[0])
    assert blocked_key == [""]


def test_column_wall_preserves_an_existing_upstream_stop():
    state = SimpleNamespace(column_inner_diameter_mm=2.0)
    alive, blocked_z, blocked_key = clip_column_wall(
        state,
        np.array([0.0, 10.0]),
        np.array([[0.0], [0.002]]),
        np.zeros((2, 1)),
        alive=np.array([False]),
        blocked_z=np.array([2.0]),
        blocked_key=["aperture"],
    )

    assert alive.tolist() == [False]
    assert blocked_z.tolist() == pytest.approx([2.0])
    assert blocked_key == ["aperture"]


def test_column_wall_replaces_an_existing_downstream_stop():
    state = SimpleNamespace(column_inner_diameter_mm=2.0)
    alive, blocked_z, blocked_key = clip_column_wall(
        state,
        np.array([0.0, 10.0, 20.0]),
        np.array([[0.0], [0.0005], [0.0015]]),
        np.zeros((3, 1)),
        alive=np.array([False]),
        blocked_z=np.array([18.0]),
        blocked_key=["detector"],
    )

    assert alive.tolist() == [False]
    assert blocked_z.tolist() == pytest.approx([15.0])
    assert blocked_key == [COLUMN_WALL_KEY]


@pytest.mark.parametrize("diameter", (0.0, -1.0, float("nan")))
def test_column_wall_rejects_invalid_diameter(diameter):
    state = SimpleNamespace(column_inner_diameter_mm=diameter)
    with pytest.raises(ValueError, match="finite and positive"):
        clip_column_wall(
            state,
            np.array([0.0, 1.0]),
            np.zeros((2, 1)),
            np.zeros((2, 1)),
        )
