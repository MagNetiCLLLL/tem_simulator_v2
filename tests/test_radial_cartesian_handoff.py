"""Independent analytic representation tests, not physical source admission."""
from dataclasses import replace

import numpy as np
import pytest

from temsim.physics.radial_cartesian_handoff import RadialColumnNumerics, radial_to_cartesian
from temsim.physics.radial_column_wave import RadialWave
from temsim.physics.wave_grid import WaveGridNumerics


def field(phase=0., quartic=0.):
    r = np.geomspace(1e-12, 50., 65536, endpoint=False)
    return RadialWave(r, .6*np.exp(-r*r/2+1j*phase+1j*quartic*r**4)/np.sqrt(np.pi), .25)


@pytest.mark.parametrize("phase,quartic", [(0., 0.), (.73, 0.), (-1.15, .1)])
def test_full_complex_field_and_current_without_fitted_phase_or_normalisation(phase, quartic):
    radial = field(phase, quartic)
    plane, record = radial_to_cartesian(radial, wavelength_m=2*np.pi,
        numerics=RadialColumnNumerics(initial_cartesian_pixels=32),
        grid_numerics=WaveGridNumerics(maximum_pixels=2048, maximum_working_bytes=2*1024**3))
    x, y = plane.coordinates_m()
    r2 = x*x+y*y
    dx = plane.basis_m[0, 0]
    exact = .6*np.exp(-r2/2+1j*phase+1j*quartic*r2*r2)/np.sqrt(np.pi)*dx
    constant = .125*float(plane.origin_m@plane.origin_m)
    assert np.linalg.norm(plane.amplitude-exact*np.exp(1j*constant))/.6 < 1e-7
    assert plane.probability == pytest.approx(.36, rel=1e-8)
    assert plane.probability == record["output_norm"]
    assert record["normalisation_correction"] is False
    assert record["phase_fit"] is False
    assert record["complex_checks"][-1]["successive_passes"] == 2
    np.testing.assert_array_equal(plane.curvature_m1, np.eye(2)*.25)
    # The factored curvature is preserved as well, not just the envelope.
    np.testing.assert_allclose(plane.full_amplitude(2*np.pi), exact*np.exp(.125j*r2), atol=1e-9)


def test_refuses_unresolved_phase_under_user_budget():
    with pytest.raises(ValueError, match="budget exceeded"):
        radial_to_cartesian(field(quartic=5.), wavelength_m=2*np.pi, numerics=RadialColumnNumerics(initial_cartesian_pixels=32),
            grid_numerics=WaveGridNumerics(maximum_pixels=64))


def test_refuses_current_mismatch_instead_of_rescaling(monkeypatch):
    # Corrupted original-grid quadrature cannot be concealed by a unit-norm image.
    from scipy.interpolate import CubicSpline
    import temsim.physics.radial_cartesian_handoff as module
    class WrongInterpolation:
        def __init__(self, *a, **kw): self.actual = CubicSpline(*a, **kw)
        def __call__(self, x): return self.actual(x)*1.01
    monkeypatch.setattr(module, "CubicSpline", WrongInterpolation)
    with pytest.raises(ValueError, match="budget exceeded"):
        radial_to_cartesian(field(), wavelength_m=2*np.pi, numerics=RadialColumnNumerics(initial_cartesian_pixels=32),
                            grid_numerics=WaveGridNumerics(maximum_pixels=128))


def test_cancelled_readout_publishes_nothing():
    with pytest.raises(InterruptedError, match="cancelled"):
        radial_to_cartesian(field(), wavelength_m=2*np.pi, cancelled=lambda: True)


def test_zero_and_invalid_fields():
    original = field()
    zero = replace(original, amplitude=np.zeros_like(original.amplitude))
    plane, record = radial_to_cartesian(zero, wavelength_m=2*np.pi)
    assert plane.probability == 0 and record["zero_field"]
    with pytest.raises(ValueError, match="uniform log-radius"):
        radial_to_cartesian(replace(original, radius_m=original.radius_m+1.), wavelength_m=2*np.pi)


@pytest.mark.parametrize("values", [{"complex_tolerance": 0.}, {"current_tolerance": float("nan")},
    {"maximum_samples": 32}, {"initial_samples": True}, {"backend": "unknown"}])
def test_invalid_numerics(values):
    with pytest.raises(ValueError):
        replace(RadialColumnNumerics(), **values).validate()
