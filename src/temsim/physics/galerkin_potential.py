"""Non-circular finite-basis potential action, with an exponential error bound.

The wave lives in N Fourier modes per axis. R embeds those modes isometrically
in a >=2N quadrature grid on the SAME physical period. H=R* diag(phase) R is
Hermitian for a real potential, and exp(i H) is unitary. Its non-circular
Fourier convolution does not wrap unresolved scattering into a low-angle bin.

This is a Galerkin approximation, not an infinite-bandwidth specimen solution.
Converge both wave basis and independently regenerated potential quadrature.
No high-frequency wave samples are filtered or renormalised by this operator.
"""
import math

import numpy as np

from temsim.physics.wave_flux import check_lossless_norm
from temsim.physics.wave_execution import check_available_memory


def potential_action(amplitude, phase_quadrature, *, tolerance=1e-13,
                     maximum_applications=8192, maximum_working_bytes=72*1024**3,
                     cancelled=lambda: False):
    """Return exp(i R* phase R) applied to complex cell amplitudes, and evidence.

    phase_quadrature is dimensionless and centered at the same physical origin
    as amplitude, on the same affine period. Fourier sign is exp(-i k.x), with
    unitary FFTs. At least twice as many samples per axis are required so all
    differences of retained wave frequencies have distinct quadrature bins.

    A shifted/scaled Taylor exponential has a PRIOR remainder bound using
    ||H-cI|| <= max|phase-c|. Deterministic degree selection avoids random norm
    estimation and keeps interrupted stochastic histories reproducible. The
    bound concerns the exponential evaluation only, not basis/grid error.
    """
    a = np.asarray(amplitude, dtype=complex)
    phase = np.asarray(phase_quadrature)
    if a.ndim != 2 or min(a.shape) < 2 or not np.all(np.isfinite(a)):
        raise ValueError("Galerkin wave must be a finite two-dimensional complex array")
    if (phase.ndim != 2 or np.iscomplexobj(phase) or not np.all(np.isfinite(phase))
            or any(p < 2*n for p, n in zip(phase.shape, a.shape))):
        raise ValueError("Real potential quadrature needs at least twice the wave samples on each axis")
    if not math.isfinite(tolerance) or not 1e-15 <= tolerance <= 1e-12:
        raise ValueError("Exponential evaluation tolerance must be between 1e-15 and 1e-12")
    if type(maximum_applications) is not int or maximum_applications < 1:
        raise ValueError("Galerkin application budget must be a positive integer")
    if type(maximum_working_bytes) is not int or maximum_working_bytes <= 0:
        raise ValueError("Galerkin memory budget must be a positive integer")
    required = 128*phase.size+160*a.size
    if required > maximum_working_bytes:
        raise MemoryError(f"Galerkin potential needs approximately {required} working bytes")
    check_available_memory(required)
    if cancelled():
        raise InterruptedError("Galerkin potential cancelled")
    # Centre the spectral interval; cI commutes exactly and carries an exact
    # scalar phase. Isometry of R bounds H without a dense matrix or RNG.
    low, high = float(phase.min()), float(phase.max())
    centre = low/2+high/2
    bound = max(abs(high-centre), abs(low-centre))
    steps = max(1, math.ceil(bound))
    if steps > maximum_applications:
        raise ValueError("Galerkin potential exponential exceeds its application budget")
    radius = bound/steps
    degree, term = 0, 1.
    while True:
        remainder = math.exp(radius)*term*radius/(degree+1)
        if steps*remainder <= tolerance/4:
            break
        degree += 1
        term *= radius/degree
    if steps*degree > maximum_applications:
        raise ValueError("Galerkin potential exponential exceeds its application budget")
    phase = (np.asarray(phase, float)-centre)/steps
    slices = tuple(slice(p//2-n//2, p//2-n//2+n) for p, n in zip(phase.shape, a.shape))
    def h_action(coefficients):
        if cancelled():
            raise InterruptedError("Galerkin potential cancelled")
        padded = np.zeros(phase.shape, complex)
        padded[slices] = coefficients
        fine = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(padded), norm="ortho"))
        product = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(phase*fine), norm="ortho"))
        return product[slices].copy()
    coefficients = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(a), norm="ortho"))
    for _ in range(steps):
        term_array = coefficients.copy()
        summed = coefficients.copy()
        for order in range(1, degree+1):
            term_array = (1j/order)*h_action(term_array)
            summed += term_array
        coefficients = summed
    result = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(coefficients), norm="ortho"))*np.exp(1j*centre)
    before, after = float(np.vdot(a, a).real), float(np.vdot(result, result).real)
    check_lossless_norm(before, after, context="Hermitian Galerkin potential exponential")
    return result, {"method": "hermitian-fourier-galerkin-v1", "wave_shape": a.shape,
        "potential_quadrature_shape": phase.shape, "applications": steps*degree,
        "exponential_error_bound": math.expm1(steps*math.log1p(remainder)),
        "norm_residual": after-before, "renormalised": False,
        "finite_basis_convergence": "REQUIRES_INDEPENDENT_REFINEMENT"}
