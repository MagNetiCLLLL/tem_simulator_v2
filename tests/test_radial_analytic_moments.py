"""Analytic finite-basis integrals versus independent positive quadrature."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from threadpoolctl import threadpool_limits
from temsim.physics.radial_gun_wave import (aperture_projection, resolved_radial_quadrature,
    piecewise_radial_potential, KINETIC_NM2_PER_EV, REST_EV)


@pytest.mark.parametrize("count", (16, 64, 128))
@pytest.mark.parametrize("limit", (.1, 4., 50.))
def test_analytic_disk_retains_the_same_mask_integral(count, limit):
    path = Path(__file__).parents[1]/"scripts"/"inspect_laguerre_aperture.py"
    spec = importlib.util.spec_from_file_location("disk_reference", path)
    reference = importlib.util.module_from_spec(spec); spec.loader.exec_module(reference)
    with threadpool_limits(1):
        expected = reference.positive_quadrature(limit, count)
        actual = aperture_projection(np.sqrt(limit), 1., count, 0)
    np.testing.assert_allclose(actual, expected, rtol=0., atol=8e-13)
    assert np.linalg.eigvalsh(actual).min() > -8e-13
    assert np.linalg.eigvalsh(actual).max() < 1+8e-13


@pytest.mark.parametrize("count", (8, 32, 64))
@pytest.mark.parametrize("energy", (20., 300000.))
def test_piecewise_relativistic_potential_has_no_removed_high_order_coupling(count, energy):
    width = 3.4
    knots = width*np.r_[0., .3, .8, 1.3, 2.7, 5., np.sqrt(4*count+160)]
    kinetic = energy+np.array([0., .04, -.01, .2, -.3, .7, .8])
    with threadpool_limits(1):
        r, weighted = resolved_radial_quadrature(width, count, knots, 128)
        local = np.interp((r/width)**2, (knots/width)**2, kinetic)
        difference = KINETIC_NM2_PER_EV*(local-energy)*(1+(local+energy)/(2*REST_EV))
        expected = (weighted*difference)@weighted.T
        actual = piecewise_radial_potential(width, count, knots, kinetic, energy)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=2e-9)
    assert abs(actual[0, -1]) > 1e-4


def test_scaled_disk_recurrence_handles_a_very_wide_bore():
    np.testing.assert_array_equal(aperture_projection(1000., 1., 256, 0), np.eye(256))
