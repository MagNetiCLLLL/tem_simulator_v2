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
from scipy import fft

from temsim.cpu_resources import numerical_thread_budget
from temsim.physics.wave_flux import check_lossless_norm
from temsim.physics.wave_execution import check_available_memory


def potential_action(amplitude, phase_quadrature, *, tolerance=1e-13,
                     maximum_applications=8192, maximum_working_bytes=72*1024**3,
                     backend="cpu", maximum_device_working_bytes=24*1024**3,
                     cancelled=lambda: False):
    """Execute one complete potential operator, retrying only recoverable GPU failures."""
    from temsim.physics.wave_device import array_module, normalise_backend
    from temsim.physics.compute_backend import gpu_failure_category, gpu_retry_reason
    options = dict(tolerance=tolerance, maximum_applications=maximum_applications,
        maximum_working_bytes=maximum_working_bytes, backend=backend,
        maximum_device_working_bytes=maximum_device_working_bytes, cancelled=cancelled)
    try:
        return _potential_action_once(amplitude, phase_quadrature, **options)
    except Exception as error:
        # A partial Taylor sum is never a restart state. Keep device inputs
        # resident; their owning stage decides whether a full CPU replay exists.
        if (array_module(amplitude) is not np or array_module(phase_quadrature) is not np
                or normalise_backend(backend) in ("CPU", "Numba CPU")
                or gpu_failure_category(error) not in ("unavailable", "out_of_memory")):
            raise
        reason = gpu_retry_reason(error, normalise_backend(backend), stage="galerkin_potential")
        result, record = _potential_action_once(amplitude, phase_quadrature,
            **{**options, "backend": "cpu"})
        return result, {**record, "fallback_reason": reason}


def _potential_action_once(amplitude, phase_quadrature, *, tolerance,
                          maximum_applications, maximum_working_bytes,
                          backend, maximum_device_working_bytes, cancelled):
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
    from temsim.physics.wave_device import array_module, device_scope, to_host
    input_xp, phase_xp = array_module(amplitude), array_module(phase_quadrature)
    a = input_xp.asarray(amplitude, dtype=input_xp.complex128)
    phase = phase_xp.asarray(phase_quadrature)
    if a.ndim != 2 or min(a.shape) < 2 or not bool(input_xp.all(input_xp.isfinite(a))):
        raise ValueError("Galerkin wave must be a finite two-dimensional complex array")
    if (phase.ndim != 2 or phase_xp.iscomplexobj(phase) or not bool(phase_xp.all(phase_xp.isfinite(phase)))
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
    if cancelled():
        raise InterruptedError("Galerkin potential cancelled")
    resident = input_xp is not np or phase_xp is not np
    requested = "Require GPU" if resident else backend
    with device_scope(requested, maximum_working_bytes=maximum_device_working_bytes,
                      work_items=phase.size, required_bytes=required) as scope:
        xp = scope.xp
        if resident and xp is np:
            raise MemoryError("A resident material wave cannot silently leave the GPU: "+str(scope.fallback_reason))
        if xp is np:
            check_available_memory(required)
        # One upload precedes all Taylor applications. GPU arrays and FFTs
        # stay resident throughout the exponential, including its norm check.
        a = xp.asarray(a, dtype=xp.complex128)
        phase = xp.asarray(phase, dtype=xp.float64)
        result, record = _resident_potential_action(a, phase, xp=xp,
            tolerance=tolerance, maximum_applications=maximum_applications,
            cancelled=cancelled)
        if input_xp is np and xp is not np:
            result = to_host(result)
        return result, {**record, "compute_backend": "numpy" if xp is np else "cupy",
            "numeric_precision": "complex128 / float64", "fallback_reason": scope.fallback_reason,
            "estimated_working_bytes": required}


def _resident_potential_action(a, phase, *, xp, tolerance, maximum_applications, cancelled):
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
    phase = (phase-centre)/steps
    slices = tuple(slice(p//2-n//2, p//2-n//2+n) for p, n in zip(phase.shape, a.shape))
    # This material operator runs modes serially; use only its caller's CPU
    # allowance and cap internal FFT parallelism to bound scratch/workers.
    fft_workers = min(4, numerical_thread_budget()) if xp is np else None
    def transform(name, value):
        if xp is np:
            return getattr(fft, name)(value, norm="ortho", workers=fft_workers)
        return getattr(xp.fft, name)(value, norm="ortho")
    def h_action(coefficients):
        if cancelled():
            raise InterruptedError("Galerkin potential cancelled")
        padded = xp.zeros(phase.shape, dtype=xp.complex128)
        padded[slices] = coefficients
        fine = xp.fft.fftshift(transform("ifft2", xp.fft.ifftshift(padded)))
        product = xp.fft.fftshift(transform("fft2", xp.fft.ifftshift(phase*fine)))
        return product[slices].copy()
    coefficients = xp.fft.fftshift(transform("fft2", xp.fft.ifftshift(a)))
    for _ in range(steps):
        term_array = coefficients.copy()
        summed = coefficients.copy()
        for order in range(1, degree+1):
            term_array = (1j/order)*h_action(term_array)
            summed += term_array
        coefficients = summed
    result = xp.fft.fftshift(transform("ifft2", xp.fft.ifftshift(coefficients)))*np.exp(1j*centre)
    before, after = float(xp.vdot(a, a).real), float(xp.vdot(result, result).real)
    check_lossless_norm(before, after, context="Hermitian Galerkin potential exponential")
    return result, {"method": "hermitian-fourier-galerkin-v1", "wave_shape": a.shape,
        "potential_quadrature_shape": phase.shape, "applications": steps*degree,
        "exponential_error_bound": math.expm1(steps*math.log1p(remainder)),
        "norm_residual": after-before, "renormalised": False, "fft_workers": fft_workers,
        "finite_basis_convergence": "REQUIRES_INDEPENDENT_REFINEMENT"}
