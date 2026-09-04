import math

import numpy as np
import pytest

from temsim.operating_modes import direct_alignment_by_key
from temsim.optics.direct_alignment_precalibration import (
    interpolated_precalculated_seed,
    interpolated_nanoprobe_seed,
    precalculated_alignment_points,
    precalculated_alignment_ratios,
)


def test_image_precalculation_keeps_lm_and_normal_branches_separate():
    definition = direct_alignment_by_key("image_magnification")
    points = precalculated_alignment_points(definition)

    assert [point.branch for point in points] == [
        "lm", "lm", "lm", "normal", "normal", "normal"
    ]
    ratios = precalculated_alignment_ratios(definition)
    assert len(ratios) == 4
    assert not any(
        row.lower_target == 1000.0 and row.upper_target == 10000.0
        for row in ratios
    )


def test_log_midpoint_seed_is_the_vector_midpoint():
    definition = direct_alignment_by_key("diffraction_camera_length")
    points = precalculated_alignment_points(definition)
    left = points[1]
    right = points[2]
    target = math.sqrt(left.target * right.target)
    lower = np.zeros(4)
    upper = np.full(4, 100.0)

    seed = interpolated_precalculated_seed(
        definition, target, lower, upper
    )

    assert seed is not None
    assert seed == pytest.approx(
        0.5 * (
            np.asarray(left.strengths) + np.asarray(right.strengths)
        )
    )


def test_image_precalculation_does_not_extrapolate_across_branch_gap():
    definition = direct_alignment_by_key("image_magnification")
    seed = interpolated_precalculated_seed(
        definition,
        2_000.0,
        np.zeros(5),
        np.full(5, 100.0),
    )

    assert seed is None


def test_ratio_table_reports_target_ratio_and_per_decade_delta():
    definition = direct_alignment_by_key("diffraction_camera_length")
    first = precalculated_alignment_ratios(definition)[0]
    points = precalculated_alignment_points(definition)

    assert first.target_ratio == pytest.approx(10.0)
    assert first.log10_span == pytest.approx(1.0)
    assert first.strength_delta_per_decade == pytest.approx(
        np.asarray(points[1].strengths) - np.asarray(points[0].strengths)
    )


def test_nanoprobe_seed_requires_the_calibrated_aperture_path():
    definition = direct_alignment_by_key("nanoprobe_convergence")
    lower = np.zeros(2)
    upper = np.full(2, 100.0)

    matched = interpolated_nanoprobe_seed(
        definition, 33.0, 132.0, lower, upper
    )
    mismatched = interpolated_nanoprobe_seed(
        definition, 33.0, 100.0, lower, upper
    )

    assert matched is not None
    assert matched == pytest.approx(
        0.5 * np.asarray((
            definition.targets["precalculation_vectors"][5],
            definition.targets["precalculation_vectors"][6],
        )).sum(axis=0)
    )
    assert mismatched is None
