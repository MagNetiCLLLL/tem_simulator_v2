"""Read-only error classification must never phase-fit acceptance values."""
import numpy as np
import pytest

from temsim.physics.global_embedded_refinement import boundary_difference_details


def test_pure_phase_error_is_still_rejected_by_full_complex_difference():
    a = np.array([[1.+2j, -.3j], [.2, .4j]])
    b = a*np.exp(.3j)
    result = boundary_difference_details((a, 2j*a), (b, 2j*b))
    assert result["maximum_relative_difference"] == pytest.approx(2*np.sin(.15))
    for row in result["worst_boundaries"]:
        assert row["overlap_phase_rad_diagnostic"] == pytest.approx(.3)
        assert row["normalised_overlap_magnitude_diagnostic"] == pytest.approx(1.)
        assert row["norm_relative_change_diagnostic"] < 1e-15


def test_shape_and_amplitude_errors_are_not_removed():
    a = np.array([[1.+0j, 0.]])
    b = np.array([[0., 2.+0j]])
    result = boundary_difference_details((a, a), (b, b))
    assert result["maximum_relative_difference"] == pytest.approx(np.sqrt(5)/2)
    row = result["worst_boundaries"][0]
    assert row["normalised_overlap_magnitude_diagnostic"] == 0.
    assert row["overlap_phase_rad_diagnostic"] is None
    assert row["norm_relative_change_diagnostic"] == .5


def test_no_phase_is_fabricated_for_zero_field():
    a, b = np.zeros((1, 2), complex), np.ones((1, 2), complex)
    result = boundary_difference_details((a, a), (a, b))
    assert result["maximum_relative_difference"] == 1.
    assert all(row["overlap_phase_rad_diagnostic"] is None for row in result["worst_boundaries"])
    assert all(row["normalised_overlap_magnitude_diagnostic"] is None for row in result["worst_boundaries"])
