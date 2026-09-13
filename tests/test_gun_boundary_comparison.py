"""The diagnostic distinguishes phase error from a coordinate change."""
import numpy as np
import pytest
from scripts.compare_gun_boundaries import boundary_error, boundary_metrics


def gaussian(width):
    # Analytic projection of exp(-r^2/2)/sqrt(pi) into width-b Laguerre modes.
    ratio = (width*width-1)/(1+width*width)
    values = 2*width/(1+width*width)*ratio**np.arange(64)
    return {"z_nm": 1., "width_nm": width, "reference_k_per_nm": 3.,
        "curvature_per_nm": 0., "coefficients_real": values.tolist(),
        "coefficients_imag": np.zeros(64).tolist()}


def test_physical_field_is_compared_not_the_basis_coefficients():
    assert boundary_error(gaussian(1.), gaussian(.5)) < 1e-12
    assert boundary_error(gaussian(1.), gaussian(2.)) < 1e-12


def test_comparison_does_not_remove_constant_or_curvature_phase():
    first = gaussian(1.)
    second = gaussian(1.)
    second["coefficients_imag"] = second["coefficients_real"]
    second["coefficients_real"] = np.zeros(64).tolist()
    assert boundary_error(first, second) == pytest.approx(np.sqrt(2), abs=1e-12)
    metrics = boundary_metrics(first, second)
    assert metrics["relative_amplitude_l2"] < 1e-14
    assert metrics["relative_density_l1"] < 1e-14
    second = {**gaussian(1.), "curvature_per_nm": .2}
    assert boundary_error(first, second) > .1
