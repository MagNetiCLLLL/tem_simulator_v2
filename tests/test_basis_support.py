"""Best-approximation diagnostics must not fit away lost wave content."""
import numpy as np
import pytest

from scripts.inspect_basis_support import projection_defect
from tests.test_gun_boundary_comparison import gaussian


def test_equivalent_gaussian_coordinates_are_representable():
    for width in (.5, 1., 2.):
        result = projection_defect(gaussian(1.), gaussian(width))
        assert result["best_approximation_relative_l2"] < 1e-6


def test_excluded_orthogonal_mode_is_not_called_physical_absorption():
    executed = gaussian(1.)
    values = np.zeros(64)
    values[4] = 1.
    executed["coefficients_real"] = values.tolist()
    destination = {**gaussian(1.), "coefficients_real": [0.]*4, "coefficients_imag": [0.]*4}
    result = projection_defect(executed, destination)
    assert result["best_approximation_relative_l2"] == pytest.approx(1., abs=1e-8)
    with pytest.raises(ValueError, match="same physical plane"):
        projection_defect(executed, {**destination, "z_nm": 2.})
