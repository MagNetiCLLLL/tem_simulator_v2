"""Batched analytic moments versus scalar and independent positive quadrature."""
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from temsim.physics.radial_potential_batch import disk_projections, piecewise_potential
from temsim.physics.radial_gun_wave import (
    aperture_projection, resolved_radial_quadrature,
    KINETIC_NM2_PER_EV, REST_EV, laguerre_operators)


def scalar_potential(width, count, radii, energy, axis_energy):
    x = (radii/width)**2
    coordinate = laguerre_operators(count+2)[0]
    result, lower = np.zeros_like(coordinate), np.zeros_like(coordinate)
    for i, upper_x in enumerate(x[1:]):
        upper = aperture_projection(np.sqrt(upper_x), 1., count+2, 0)
        interval = upper-lower
        slope = (energy[i+1]-energy[i])/(upper_x-x[i])
        intercept = energy[i]-slope*x[i]
        a = KINETIC_NM2_PER_EV*(intercept-axis_energy)*(1+(intercept+axis_energy)/(2*REST_EV))
        b = KINETIC_NM2_PER_EV*slope*(1+intercept/REST_EV)
        c = KINETIC_NM2_PER_EV*slope*slope/(2*REST_EV)
        result += a*interval+b*interval@coordinate+c*interval@coordinate@coordinate
        lower = upper
    return ((result+result.T)/2)[:count, :count]


@pytest.mark.parametrize("count", [8, 64, 128, 256])
def test_batch_keeps_every_disk_boundary_and_scaled_recurrence(count):
    limits = np.array([0., 1e-12, .1, 4., 50., 1e3, 1e6])
    actual = disk_projections(limits, count)
    expected = np.array([aperture_projection(np.sqrt(x), 1., count, 0) for x in limits])
    np.testing.assert_allclose(actual, expected, atol=5e-13, rtol=5e-13)


@pytest.mark.parametrize("count", [1, 8, 64, 128, 256])
def test_compiled_disk_integral_matches_vectorized_reference_without_clipping(count):
    from temsim.physics.radial_potential_batch import _disk_projections_numpy
    limits = np.r_[0., np.geomspace(1e-15, 1e6, 113)]
    expected = _disk_projections_numpy(limits, count)
    actual = disk_projections(limits, count)
    np.testing.assert_allclose(actual, expected, atol=5e-13, rtol=5e-13)


def test_no_numba_uses_same_numpy_formula(monkeypatch):
    from temsim.physics import radial_integral_kernel as kernel
    from temsim.physics.radial_potential_batch import _disk_projections_numpy
    monkeypatch.setattr(kernel, "disk_projections_compiled", None)
    limits = np.array([.001, 20., 1000.])
    np.testing.assert_array_equal(disk_projections(limits, 12), _disk_projections_numpy(limits, 12))


@pytest.mark.parametrize("energy", [.3, 20., 300000.])
@pytest.mark.parametrize("count", [8, 64, 128])
def test_batched_potential_retains_complex_mode_coupling(energy, count):
    width = 3.4
    limits = np.r_[0., np.geomspace(.001, 4*count+160, 81)]
    knots = width*np.sqrt(limits)
    energies = energy+.04*np.sin(limits/8)+.01*np.cos(limits)
    with threadpool_limits(1):
        expected = scalar_potential(width, count, knots, energies, energy)
        actual = piecewise_potential(width, count, knots, energies, energy, KINETIC_NM2_PER_EV, REST_EV)
        r, weighted = resolved_radial_quadrature(width, count, knots, 128)
        local = np.interp((r/width)**2, limits, energies)
        q = KINETIC_NM2_PER_EV*(local-energy)*(1+(local+energy)/(2*REST_EV))
        quadrature = (weighted*q)@weighted.T
    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-10)
    np.testing.assert_allclose(actual, quadrature, atol=2e-9, rtol=2e-9)
