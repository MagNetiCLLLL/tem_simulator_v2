"""Checked change of representation of an executed axisymmetric wave.

No source is fitted, and neither current nor a global phase is corrected.
The retained quadratic carrier stays analytical. Numerical window, sampling
and interpolation errors are reported separately from physical absorption.
This interface alone does not qualify the upstream gun or radial propagation.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.interpolate import CubicSpline

from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.wave_grid import WaveGridNumerics, refine_plane_wave


@dataclass(frozen=True)
class RadialColumnNumerics:
    initial_samples: int = 65536
    maximum_samples: int = 8_388_608
    initial_cartesian_pixels: int = 256
    complex_tolerance: float = 1e-6
    current_tolerance: float = 1e-8
    backend: str = "cpu"

    def validate(self):
        for name in ("initial_samples", "maximum_samples", "initial_cartesian_pixels"):
            value = getattr(self, name)
            if type(value) is not int or value < 32:
                raise ValueError(f"Radial numerical {name} must be an integer of at least 32")
        if self.maximum_samples < self.initial_samples:
            raise ValueError("Maximum radial samples cannot be smaller than initial samples")
        for name in ("complex_tolerance", "current_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value < .01:
                raise ValueError(f"Radial handoff {name} must be positive and below 0.01")
        if self.backend not in ("cpu", "cuda"):
            raise ValueError("Radial transform backend must be cpu or cuda")
        return self


def _refined_at_cell_centres(coarse, pixels, grid_numerics, retained_bytes, wavelength_m):
    """Same Fourier interpolant at the finer cell centres, not a phase fit."""
    fine = refine_plane_wave(coarse, (pixels, pixels), numerics=grid_numerics,
                             retained_bytes=retained_bytes)
    dx = fine.basis_m[0, 0]
    displacement = (np.array((.5*dx, .5*dx))-fine.origin_m)/dx
    frequency = np.fft.fftfreq(pixels)
    phase = 2*np.pi*(frequency[:, None]*displacement[1]+frequency[None, :]*displacement[0])
    new_origin = np.full(2, .5*dx)
    # Quadratic carriers are defined relative to each PlaneWave origin.
    # Shift their known constant phase analytically; do not estimate a phase
    # from either field or fit the comparison to a better-looking result.
    curvature = coarse.curvature_m1
    constant = np.pi/wavelength_m*(new_origin@curvature@new_origin
                                   - coarse.origin_m@curvature@coarse.origin_m)
    return np.fft.ifft2(np.fft.fft2(fine.amplitude)*np.exp(1j*phase))*np.exp(1j*constant)


def radial_to_cartesian(wave, *, wavelength_m, numerics=RadialColumnNumerics(),
                        grid_numerics=WaveGridNumerics(), retained_bytes=0,
                        cancelled=lambda: False, progress_callback=None):
    """Resolve a single finite radial field on a cell-centred square lattice.

    Cubic interpolation acts on the original COMPLEX log-radius envelope.
    A fixed window excludes at most the explicitly recorded numerical tail.
    Two successive doubled-grid complex comparisons, without fitted phase,
    and comparison with the original radial integral are required. The
    returned norm is the sampled norm, never restored to the radial norm.
    Independent upstream radial-grid/domain convergence is still required.
    """
    numerics.validate(); grid_numerics.validate()
    if isinstance(wavelength_m, bool) or not math.isfinite(wavelength_m) or wavelength_m <= 0:
        raise ValueError("Handoff requires the wavelength of the executed mode")
    r, value = np.asarray(wave.radius_m), np.asarray(wave.amplitude)
    if (r.ndim != 1 or len(r) < 4 or value.shape != r.shape or np.any(r <= 0)
            or not np.all(np.isfinite(r)) or not np.all(np.isfinite(value))
            or not np.all(np.diff(r) > 0) or not math.isfinite(wave.curvature_m1)):
        raise ValueError("Handoff requires the complete finite increasing radial field")
    log_r = np.log(r)
    dln = float(log_r[1]-log_r[0])
    if not np.allclose(np.diff(log_r), dln, rtol=1e-8, atol=1e-13):
        raise ValueError("Handoff requires the executed uniform log-radius grid")
    shell = 2*np.pi*dln*abs(r*value)**2
    original_norm = float(shell.sum())
    if not math.isfinite(original_norm):
        raise ValueError("Nonfinite executed radial norm")
    if cancelled():
        raise InterruptedError("Radial handoff cancelled")
    if original_norm == 0:
        pixels = numerics.initial_cartesian_pixels
        grid_numerics.check((pixels, pixels), retained_bytes=retained_bytes)
        dx = 2*r[-1]/pixels
        origin = np.full(2, .5*dx)
        return PlaneWave(np.zeros((pixels, pixels), complex), np.eye(2)*dx,
            origin, np.eye(2)*wave.curvature_m1, wave.curvature_m1*origin), {
            "schema": "radial-cartesian-handoff-v1", "input_norm": 0., "output_norm": 0.,
            "complex_checks": (), "zero_field": True, "numerical_tail_fraction": 0.}
    # Reserve at most one tenth of the amplitude error for the finite window.
    tail_budget = min((numerics.complex_tolerance*.1)**2, numerics.current_tolerance*.01)
    tail = np.cumsum(shell[::-1])[::-1]
    occupied = np.flatnonzero(tail > original_norm*tail_budget)
    edge = min(len(r)-1, int(occupied[-1])+1)
    radius = float(r[edge])
    excluded = float(tail[edge+1]/original_norm) if edge+1 < len(r) else 0.
    interpolation = CubicSpline(log_r, value, extrapolate=False)

    def sample(pixels):
        if cancelled():
            raise InterruptedError("Radial handoff cancelled")
        grid_numerics.check((pixels, pixels), retained_bytes=retained_bytes)
        dx = 2*radius/pixels
        axis = (np.arange(pixels)-pixels//2+.5)*dx
        rho = np.hypot(axis[:, None], axis[None, :])
        if float(rho.min()) < r[0]:
            raise ValueError("Cartesian handoff reaches below the executed radial domain; expand and replay the radial calculation")
        amplitude = np.zeros((pixels, pixels), complex)
        inside = rho <= r[-1]
        amplitude[inside] = interpolation(np.log(rho[inside]))*dx
        origin = np.full(2, .5*dx)
        constant = np.pi/wavelength_m*wave.curvature_m1*float(origin@origin)
        return PlaneWave(amplitude*np.exp(1j*constant), np.eye(2)*dx, origin,
                         np.eye(2)*wave.curvature_m1, wave.curvature_m1*origin)

    pixels, successive, checks = numerics.initial_cartesian_pixels, 0, []
    coarse = sample(pixels)
    while True:
        pixels *= 2
        fine = sample(pixels)
        interpolated = _refined_at_cell_centres(coarse, pixels, grid_numerics,
                                               retained_bytes+fine.amplitude.nbytes, wavelength_m)
        error = float(np.linalg.norm(interpolated-fine.amplitude)/math.sqrt(original_norm))
        current_error = abs(fine.probability-original_norm)/original_norm
        passed = error <= numerics.complex_tolerance and current_error <= numerics.current_tolerance
        successive = successive+1 if passed else 0
        checks.append({"pixels": pixels, "relative_complex_change": error,
                       "relative_current_error": current_error, "successive_passes": successive})
        if progress_callback:
            progress_callback(pixels, grid_numerics.maximum_pixels,
                f"Radial/2-D handoff: complex change {error:.4g}; current error {current_error:.4g}")
        if cancelled():
            raise InterruptedError("Radial handoff cancelled before publication")
        if successive >= 2:
            return fine, {"schema": "radial-cartesian-handoff-v1", "input_norm": original_norm,
                "output_norm": fine.probability, "complex_checks": checks,
                "numerical_tail_fraction": excluded, "window_half_width_m": radius,
                "normalisation_correction": False, "phase_fit": False,
                "scope": "Representation check only; upstream gun/radial convergence is separate"}
        if not grid_numerics.automatic_refinement:
            raise ValueError("Radial/2-D handoff needs numerical refinement; no wave was published")
        coarse = fine
