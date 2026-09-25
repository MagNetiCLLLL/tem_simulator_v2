"""Display interpolation checks on synthetic paths, without particle jobs."""

import numpy as np
import pytest

from temsim.gui.ray_curve_item import finite_runs
from temsim.gui.ray_scalar_colours import (
    UNKNOWN_SCALAR_BIN, UNKNOWN_SCALAR_RGB, scalar_colour_groups,
    scalar_colour_indices, scalar_colour_palette, scalar_rgb,
)


def test_sparse_linear_ramp_preserves_all_colour_bins_and_endpoints():
    groups = scalar_colour_groups([0., 8.], [1., 5.], [0., 2.], 2., bins=8)
    assert tuple(groups) == tuple(range(8))
    for colour, (x, y) in groups.items():
        np.testing.assert_allclose(x, [colour, colour + 1, np.nan])
        np.testing.assert_allclose(y, [.5 * colour + 1, .5 * (colour + 1) + 1, np.nan])


def test_constant_dense_path_stays_one_simplifiable_run():
    x = np.linspace(0., 10., 10001)
    groups = scalar_colour_groups(x, 2. * x, np.full(len(x), .4), 1., bins=8)
    assert tuple(groups) == (3,)
    xx, yy = groups[3]
    assert finite_runs(xx, yy) == ((0, len(x)),)
    np.testing.assert_array_equal(xx[:-1], x)
    assert len(xx) == len(x) + 1  # No separator at each integration step.


def test_nonmonotonic_scalar_keeps_disconnected_crossings_in_order():
    groups = scalar_colour_groups([0., 1., 2.], [0., 1., 0.], [0., 1., 0.], 1., bins=4)
    xx, yy = groups[0]
    np.testing.assert_allclose(xx, [0., .25, np.nan, 1.75, 2., np.nan])
    np.testing.assert_allclose(yy, [0., .25, np.nan, .25, 0., np.nan])
    # At the apex the same-bin halves form one continuous run.
    np.testing.assert_allclose(groups[3][0], [.75, 1., 1.25, np.nan])


def test_supplied_stop_endpoint_is_never_extended():
    groups = scalar_colour_groups([0., 2.5], [0., 5.], [0., .25], 1., bins=4)
    assert tuple(groups) == (0,)
    np.testing.assert_allclose(groups[0][0], [0., 2.5, np.nan])
    np.testing.assert_allclose(groups[0][1], [0., 5., np.nan])


def test_missing_scalar_is_grey_but_missing_geometry_never_connects():
    groups = scalar_colour_groups(
        [0., 1., 2., np.nan, 4., 5., 6., 7.],
        [0., 0., 0., np.nan, 1., 1., 1., 1.],
        [0., np.nan, .3, 0., .4, -1., np.inf, .7], 1., bins=4,
    )
    assert tuple(groups) == (UNKNOWN_SCALAR_BIN,)
    xx, yy = groups[UNKNOWN_SCALAR_BIN]
    np.testing.assert_allclose(xx, [0., 1., 2., np.nan, 4., 5., 6., 7., np.nan])
    assert finite_runs(xx, yy) == ((0, 3), (4, 8))


def test_zero_scalar_and_zero_scale_remain_known():
    np.testing.assert_array_equal(scalar_colour_indices([0., 2., np.nan, -1.], 0.), [0, 0, -1, -1])
    groups = scalar_colour_groups([0., 1.], [0., 0.], [0., 0.], 0.)
    assert tuple(groups) == (0,)
    np.testing.assert_allclose(groups[0][0], [0., 1., np.nan])


def test_palette_mapping_and_saturation_are_shared():
    np.testing.assert_array_equal(scalar_colour_indices([0., .25, .5, 1., 4., np.inf], 1., bins=4),
                                  [0, 1, 2, 3, 3, -1])
    palette = scalar_colour_palette(bins=4)
    assert len(palette) == 4
    assert all(scalar_rgb(i, bins=4) == rgb for i, rgb in enumerate(palette))
    assert scalar_rgb(-1, bins=4) == UNKNOWN_SCALAR_RGB
    assert palette[0] != palette[-1]


def test_above_scale_segment_crossings_follow_original_scalar_not_clipped_endpoints():
    groups = scalar_colour_groups([0., 2.], [0., 0.], [0., 2.], 1., bins=4)
    np.testing.assert_allclose(groups[0][0], [0., .25, np.nan])
    np.testing.assert_allclose(groups[3][0], [.75, 2., np.nan])


def test_inputs_are_not_modified_and_reversed_geometry_is_preserved():
    x = np.array([2., 0., 2., np.nan, 5.])
    y = np.array([0., 0., 1., np.nan, 2.])
    times = np.array([0., 1., 2., np.nan, 3.])
    saved = [a.copy() for a in (x, y, times)]
    groups = scalar_colour_groups(x, y, times, 2., bins=4)
    np.testing.assert_allclose(groups[0][0], [2., 1., np.nan])
    np.testing.assert_allclose(groups[2][0], [0., 1., np.nan])
    for actual, expected in zip((x, y, times), saved):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("maximum,bins", [(-1., 4), (np.nan, 4), (np.inf, 4), (1., 1), (1., True), (1., 3.5)])
def test_invalid_scale_or_palette_rejected(maximum, bins):
    with pytest.raises(ValueError):
        scalar_colour_groups([0., 1.], [0., 1.], [0., 1.], maximum, bins=bins)


def test_invalid_shapes_and_no_line_segments():
    with pytest.raises(ValueError, match="aligned"):
        scalar_colour_groups([0., 1.], [0.], [0., 1.], 1.)
    with pytest.raises(ValueError, match="one-dimensional"):
        scalar_colour_groups([[0.]], [[0.]], [[0.]], 1.)
    assert scalar_colour_groups([], [], [], 1.) == {}
    assert scalar_colour_groups([1.], [2.], [0.], 1.) == {}
