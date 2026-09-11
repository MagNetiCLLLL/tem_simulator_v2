"""Numerical complex-field checks, not tip-to-image source qualification."""
from dataclasses import replace

import numpy as np
import pytest

from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.multipole_wave import apply_multipole_phase, multipole_action
from temsim.physics.wave_grid import (WaveGridNumerics, WaveSamplingError,
                                      apply_resolved_operator, refine_plane_wave)


@pytest.mark.parametrize("shape", [(32, 32), (33, 35), (32, 35)])
def test_refinement_retains_all_complex_fourier_coefficients_and_affine_carriers(shape):
    rng = np.random.default_rng(937)
    amplitude = rng.normal(size=shape)+1j*rng.normal(size=shape)
    amplitude *= np.sqrt(.37/np.sum(abs(amplitude)**2))
    wave = PlaneWave(amplitude, np.array(((2e-8, .3e-8), (-.2e-8, 3e-8))),
                     np.array((7e-6, -8e-6)), np.array(((.2, -.3), (-.3, .4))), np.array((.01, -.02)))
    result = refine_plane_wave(wave, tuple(2*n for n in shape))
    np.testing.assert_allclose(result.basis_m, wave.basis_m/2, atol=0, rtol=0)
    for field in ("origin_m", "curvature_m1", "tilt_rad"):
        np.testing.assert_array_equal(getattr(result, field), getattr(wave, field))
    starts = tuple(n % 2 for n in shape)
    selected = result.amplitude[starts[0]::2, starts[1]::2]
    np.testing.assert_allclose(selected*2, wave.amplitude, atol=1e-16, rtol=1e-13)
    assert result.probability == pytest.approx(.37, abs=3e-16)
    assert not result.amplitude.flags.writeable


def test_even_negative_nyquist_bin_keeps_its_complex_phase_and_norm():
    n, factor = 32, 4
    x = np.arange(n)-n//2
    a = np.tile(np.exp(-1j*np.pi*x), (n, 1))/n
    wave = PlaneWave(a, np.eye(2), np.zeros(2))
    result = refine_plane_wave(wave, (n*factor, n*factor))
    fine_x = (np.arange(n*factor)-n*factor//2)/factor
    expected = np.tile(np.exp(-1j*np.pi*fine_x), (n*factor, 1))/(n*factor)
    np.testing.assert_allclose(result.amplitude, expected, atol=5e-17)
    assert result.probability == pytest.approx(1., abs=2e-15)


def gaussian(n):
    axis = (np.arange(n)-n//2)*(128/n)*4e-9
    xx, yy = np.meshgrid(axis, axis)
    a = np.exp(-(xx*xx+yy*yy)/(4*(16e-9)**2)).astype(complex)
    a /= np.linalg.norm(a)
    return PlaneWave(a, np.eye(2)*(128/n)*4e-9, np.zeros(2))


def test_adaptive_cs_complex_field_and_diffraction_match_independent_fine_grid():
    source, lam, cs = gaussian(128), 2e-12, 4e17
    operator = lambda w: apply_multipole_phase(w, lam, spherical_m3=cs)
    with pytest.raises(WaveSamplingError):
        operator(source)
    result, rows = apply_resolved_operator(source, operator)
    assert rows and result.amplitude.shape[0] > 128
    assert result.probability == pytest.approx(source.probability, abs=1e-14)
    reference = gaussian(result.amplitude.shape[0]*2)
    x, y = reference.coordinates_m()
    # Independent original quartic law, without factoring/fitting any phase.
    expected = reference.amplitude*np.exp(-.5j*np.pi*cs*(x*x+y*y)**2/lam)
    np.testing.assert_allclose(result.amplitude, expected[::2, ::2]*2, atol=2e-12, rtol=2e-9)
    def free_field(a, basis, length=1e-5):
        fy, fx = np.meshgrid(np.fft.fftfreq(len(a)), np.fft.fftfreq(len(a)), indexing="ij")
        f = np.einsum("ij,jyx->iyx", np.linalg.inv(basis).T, np.stack((fx, fy)))
        return np.fft.ifft2(np.fft.fft2(a)*np.exp(-1j*np.pi*lam*length*(f*f).sum(axis=0)))
    propagated = free_field(result.amplitude, result.basis_m)
    fine = free_field(expected, reference.basis_m)
    assert np.linalg.norm(propagated-2*fine[::2, ::2]) < 2e-8


def test_budget_failure_precedes_allocation_and_does_not_apply_or_ignore_an_operator(monkeypatch):
    source = gaussian(128)
    calls = []
    def operator(w):
        calls.append(w)
        raise WaveSamplingError("known unresolved optical operator", 16)
    def allocation(*a, **kw):
        raise AssertionError("large allocation must be checked first")
    monkeypatch.setattr(np, "zeros", allocation)
    with pytest.raises(ValueError, match="budget exceeded"):
        apply_resolved_operator(source, operator, numerics=WaveGridNumerics(maximum_pixels=256))
    assert calls == [source]
    with pytest.raises(WaveSamplingError, match="unresolved"):
        apply_resolved_operator(source, operator, numerics=WaveGridNumerics(automatic_refinement=False))
    assert calls == [source, source]
    with pytest.raises(InterruptedError):
        apply_resolved_operator(source, operator, cancelled=lambda: True)
    assert len(calls) == 2


def test_non_sampling_errors_are_not_retried_and_invalid_budgets_are_rejected():
    def wrong_operator(w):
        raise ValueError("physical field is unavailable")
    with pytest.raises(ValueError, match="field is unavailable"):
        apply_resolved_operator(gaussian(128), wrong_operator)
    for bad in (WaveGridNumerics(maximum_pixels=True), WaveGridNumerics(maximum_pixels=16),
                WaveGridNumerics(automatic_refinement=1), WaveGridNumerics(maximum_working_bytes=0)):
        with pytest.raises(ValueError):
            bad.validate()


def test_resolved_wave_zero_does_not_masquerade_as_aliasing():
    wave = gaussian(128)
    x, y = wave.coordinates_m()
    a = wave.amplitude*(x/16e-9+1j*y/16e-9)
    a /= np.linalg.norm(a)
    vortex = replace(wave, amplitude=a)
    result = apply_multipole_phase(vortex, 2e-12, spherical_m3=4e14)
    expected = a*np.exp(2j*np.pi*multipole_action(x, y, spherical_m3=4e14)/2e-12)
    np.testing.assert_allclose(result.amplitude, expected, atol=1e-15)
