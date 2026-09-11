"""Phase operators for the column's existing normal/skew hexapole and Cs laws.

Coefficients and signs follow ray_integrator.canonical_rk4_step and its discrete
Cs kick. These functions add no optics or sources. They retain the full cubic
and quartic phase, including non-quadratic residuals, on an affine wave lattice.
The stationary column_wave development path executes these at physical nodes.
"""
from dataclasses import replace
import math

import numpy as np

from temsim.physics.wave_flux import check_lossless_norm
from temsim.physics.wave_grid import WaveSamplingError


def multipole_action(x, y, *, normal_m2=0., skew_m2=0., spherical_m3=0.):
    """Optical action in metres; its gradient is the canonical momentum kick.

    normal_m2/skew_m2 are integrated distributed strengths (h*hn and h*hs),
    whereas spherical_m3 is the existing discrete radial-cubic kick strength.
    """
    return (-normal_m2*(x*x*x/3-x*y*y)
            -skew_m2*(x*x*y-y*y*y/3)
            -spherical_m3*(x*x+y*y)**2/4)


def apply_multipole_phase(wave, wavelength_m, *, normal_m2=0., skew_m2=0., spherical_m3=0.):
    for value in (wavelength_m, normal_m2, skew_m2, spherical_m3):
        if not math.isfinite(value):
            raise ValueError("Multipole phase inputs must be finite")
    if wavelength_m <= 0:
        raise ValueError("Multipole phase requires positive wavelength")
    if normal_m2 == skew_m2 == spherical_m3 == 0:
        return wave
    x, y = wave.coordinates_m()
    x0, y0 = wave.origin_m
    u, v = x-x0, y-y0
    hn, hs, cs = normal_m2, skew_m2, spherical_m3
    r2 = x0*x0+y0*y0
    constant = float(multipole_action(x0, y0, normal_m2=hn, skew_m2=hs, spherical_m3=cs))
    gradient = np.array((-hn*(x0*x0-y0*y0)-2*hs*x0*y0-cs*r2*x0,
                          2*hn*x0*y0-hs*(x0*x0-y0*y0)-cs*r2*y0))
    cross = 2*hn*y0-2*hs*x0-2*cs*x0*y0
    hessian = np.array(((-2*hn*x0-2*hs*y0-cs*(3*x0*x0+y0*y0), cross),
                        (cross, 2*hn*x0+2*hs*y0-cs*(x0*x0+3*y0*y0))))
    # Exact Taylor remainder of this degree-four polynomial. Evaluating it
    # directly avoids subtracting nearly equal large off-axis phase carriers.
    residual = multipole_action(u, v, normal_m2=hn, skew_m2=hs, spherical_m3=cs)
    residual -= cs*(x0*u+y0*v)*(u*u+v*v)
    phase = (2*np.pi/wavelength_m)*residual
    amplitude = wave.amplitude
    occupied = abs(amplitude) > np.max(abs(amplitude))*1e-8
    required_scale, largest = 1., 0.
    # Bound represented envelope frequencies by marginal spectral probability.
    # A wrapped neighbour phase is not a frequency bound: zeros and vortices
    # legitimately have pi jumps. No spectrum is removed by this check.
    spectrum = abs(np.fft.fft2(amplitude, norm="ortho"))**2
    total = float(spectrum.sum())
    for axis in (0, 1):
        n = amplitude.shape[axis]
        low = np.take(occupied, np.arange(n-1), axis=axis)
        high = np.take(occupied, np.arange(1, n), axis=axis)
        operator_difference = abs(np.diff(phase, axis=axis))[low & high]
        added = float(operator_difference.max(initial=0.))
        frequencies = abs(2*np.pi*np.fft.fftfreq(n))
        marginal = spectrum.sum(axis=1-axis)
        order = np.argsort(frequencies)
        tail = np.cumsum(marginal[order][::-1])[::-1]
        support = frequencies[order][tail > total*1e-12]
        bandwidth = float(support.max(initial=0.))
        largest = max(largest, added+bandwidth)
        required_scale = max(required_scale, (added+bandwidth)/(.8*np.pi))
    if required_scale > 1:
        raise WaveSamplingError(f"Nonlinear column phase is undersampled (operator increment plus envelope bandwidth "
                                f"{largest:.6g} rad, budget 0.8*pi); refine the physical wave grid", required_scale)
    curvature = np.zeros((2, 2)) if wave.curvature_m1 is None else wave.curvature_m1
    tilt = np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
    result = replace(wave, amplitude=amplitude*np.exp(1j*(phase+2*np.pi*constant/wavelength_m)),
                     curvature_m1=curvature+hessian, tilt_rad=tilt+gradient)
    check_lossless_norm(wave.probability, result.probability, context="Column multipole phase")
    return result
