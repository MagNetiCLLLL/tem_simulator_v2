"""Independent gradient and complex-field checks, not full column acceptance."""
import numpy as np
import pytest

from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.multipole_wave import apply_multipole_phase, multipole_action


def wave():
    axis = np.arange(128)-64
    x, y = np.meshgrid(axis, axis)
    amplitude = np.exp(-(x*x+y*y)/(4*8**2)).astype(complex)
    amplitude /= np.sqrt(np.sum(abs(amplitude)**2))
    return PlaneWave(amplitude, np.array(((2e-9, .2e-9), (-.1e-9, 2e-9))),
                     np.array((4e-8, -3e-8)), np.array(((3e2, 1e2), (1e2, -2e2))), np.array((1e-4, -2e-4)))


@pytest.mark.parametrize("strengths", [dict(normal_m2=2e8), dict(skew_m2=-3e8),
                                      dict(spherical_m3=4e15),
                                      dict(normal_m2=2e8, skew_m2=-3e8, spherical_m3=4e15)])
def test_factored_carriers_equal_the_complete_nonlinear_complex_phase(strengths):
    source = wave()
    wavelength = 2e-12
    result = apply_multipole_phase(source, wavelength, **strengths)
    x, y = source.coordinates_m()
    expected = source.full_amplitude(wavelength)*np.exp(2j*np.pi*multipole_action(x, y, **strengths)/wavelength)
    np.testing.assert_allclose(result.full_amplitude(wavelength), expected, atol=2e-13, rtol=2e-12)
    assert result.probability == pytest.approx(source.probability, abs=1e-14)


def test_phase_gradient_has_the_existing_ray_kick_signs_and_axes():
    x, y, delta = 2e-5, -1e-5, 1e-10
    hn, hs, cs = 2e5, -3e5, 4e12
    kw = dict(normal_m2=hn, skew_m2=hs, spherical_m3=cs)
    gx = (multipole_action(x+delta, y, **kw)-multipole_action(x-delta, y, **kw))/(2*delta)
    gy = (multipole_action(x, y+delta, **kw)-multipole_action(x, y-delta, **kw))/(2*delta)
    expected_x = -hn*(x*x-y*y)-hs*2*x*y-cs*(x*x+y*y)*x
    expected_y = hn*2*x*y-hs*(x*x-y*y)-cs*(x*x+y*y)*y
    np.testing.assert_allclose((gx, gy), (expected_x, expected_y), rtol=1e-9)


def test_nonlinear_phase_cannot_alias_or_change_a_zero_strength_wave():
    source = wave()
    assert apply_multipole_phase(source, 2e-12) is source
    with pytest.raises(ValueError, match="undersampled"):
        apply_multipole_phase(source, 2e-12, spherical_m3=1e26)
