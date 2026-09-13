"""A numerical chart transition must preserve physical complex wave transport."""
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from temsim.physics.radial_coordinates import RadialCoordinateBlend, blended_radial_chart
from scripts.inspect_moving_radial_frame import calculate


def test_blended_chart_derivatives_include_the_transition_rate():
    transition = RadialCoordinateBlend(.6, 20e-6, 80e-6).validate()
    def chart(z):
        q = 1j/100/(1+1j*z/100)
        target = (1j/36)/(1+1j*z/36)
        return blended_radial_chart(q, -q*q, target, -target*target, 100., z, transition)
    for z in (0., 20., 30., 50., 70., 80., 100.):
        value, low, high = chart(z), chart(z-1e-4), chart(z+1e-4)
        assert (np.log(high[0])-np.log(low[0]))/2e-4 == pytest.approx(value[2], abs=1e-10)
        assert (high[1]-low[1])/2e-4 == pytest.approx(value[3], abs=1e-11)


def test_smooth_chart_change_converges_to_independent_nonparaxial_wave():
    transition = RadialCoordinateBlend(.6, 20e-6, 80e-6)
    with threadpool_limits(1):
        errors = [calculate(step, count=24, transition=transition)["relative_complex_error"]
                  for step in (2., 1., .5)]
        # The carrier/backward-wave scale is much shorter than these steps.
        # Do not assume uniform second order through its pre-asymptotic
        # plateau: also require a substantially finer absolute field check.
        fine = calculate(.0125, count=24, transition=transition)["relative_complex_error"]
    assert errors[-1] < 1e-5
    assert all(coarse/fine > 3.5 for coarse, fine in zip(errors, errors[1:]))
    assert fine < 2e-8


@pytest.mark.parametrize("values", ((0., .1, .2), (.2, .2, .1), (.2, .1, .1),
                                    (.2, float("nan"), .2), (True, .1, .2)))
def test_invalid_chart_is_rejected(values):
    with pytest.raises(ValueError):
        RadialCoordinateBlend(*values).validate()
