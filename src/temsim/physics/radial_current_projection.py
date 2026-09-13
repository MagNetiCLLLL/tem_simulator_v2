"""Read-only X-Z/Y-Z current projection of an executed axisymmetric wave.

At fixed X, integrate Im(psi* d_z psi) over all Y, including the moving-chart
derivative outside the propagated subspace. The common quadratic/quartic phase
cancels in this product. Gauss-Hermite order 2*N+1 integrates the remaining
degree-4*N polynomial exactly in exact arithmetic (NIST DLMF 3.5(v)).
This is neither a particle trajectory nor a new source. Signed current is
retained; no clipping or renormalisation of the visible window is performed.
"""
from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from scipy.special import roots_hermite

from temsim.physics.radial_wave_observables import (
    _readonly, executed_boundary_parameters, normal_derivative_coefficients, radial_mode_observables)


@lru_cache(maxsize=16)
def hermite_root_weights(order):
    """Square roots of Gauss weights, without first underflowing their squares.

    w_i = 1 / (n*p_(n-1)(x_i)**2), with orthonormal Hermite p. A scaled
    polynomial recurrence avoids huge Hermite values at the outer nodes.
    The square roots can remain representable when SciPy's weights are zero.
    """
    if type(order) is not int or order < 1:
        raise ValueError("Positive integer quadrature order required")
    nodes = roots_hermite(order)[0]
    previous = np.zeros_like(nodes)
    value = np.full_like(nodes, np.pi**(-.25))
    logarithm = np.zeros_like(nodes)
    for n in range(1, order):
        current = math.sqrt(2/n)*nodes*value-math.sqrt((n-1)/n)*previous
        scale = np.maximum(np.maximum(abs(current), abs(value)), 1.)
        previous, value = value/scale, current/scale
        logarithm += np.log(scale)
    root_weights = np.exp(-.5*math.log(order)-np.log(abs(value))-logarithm)
    if not np.all(np.isfinite(root_weights)) or np.any(root_weights <= 0):
        raise ValueError("Requested current projection exceeds floating-point quadrature support")
    return _readonly(nodes), _readonly(root_weights)


@dataclass(frozen=True)
class RadialCurrentProjection:
    x_nm: np.ndarray
    axial_flux_fraction_per_nm: np.ndarray
    integrated_axial_flux_fraction: float  # All X; not the visible-window sum.


def radial_current_projection(x_nm, coefficients, covariant_derivatives, *,
        width_nm, curvature_per_nm, reference_k_per_nm, width_log_rate_per_nm,
        curvature_prime_per_nm2, quartic_phase_per_nm4, quartic_prime_per_nm5,
        maximum_working_bytes=128*1024**2):
    x = np.asarray(x_nm, float)
    if (x.ndim != 1 or not len(x) or not np.all(np.isfinite(x))
            or np.any(np.diff(x) <= 0)):
        raise ValueError("Projected current requires increasing finite X coordinates")
    parameters = dict(width_nm=width_nm, curvature_per_nm=curvature_per_nm,
        reference_k_per_nm=reference_k_per_nm, width_log_rate_per_nm=width_log_rate_per_nm,
        curvature_prime_per_nm2=curvature_prime_per_nm2,
        quartic_phase_per_nm4=quartic_phase_per_nm4, quartic_prime_per_nm5=quartic_prime_per_nm5)
    validated = radial_mode_observables([0.], coefficients, covariant_derivatives, **parameters)
    a, d = np.asarray(coefficients, complex), np.asarray(covariant_derivatives, complex)
    extended = normal_derivative_coefficients(a, d, width_nm=width_nm,
        reference_k_per_nm=reference_k_per_nm, width_log_rate_per_nm=width_log_rate_per_nm,
        curvature_prime_per_nm2=curvature_prime_per_nm2, quartic_prime_per_nm5=quartic_prime_per_nm5)
    order = 2*len(a)+1
    # Chunked evaluation changes no nodes/modes or physics. Budget includes
    # live complex accumulators, recurrence temporaries and returned profile.
    if type(maximum_working_bytes) is not int or maximum_working_bytes < 256*order+32*x.size:
        raise MemoryError("Current projection working budget is too small")
    from temsim.physics.wave_execution import check_available_memory
    chunk = min(len(x), (maximum_working_bytes-32*x.size)//(256*order))
    check_available_memory(256*order*chunk+32*x.size)
    nodes, root_weights = hermite_root_weights(order)
    flux = np.empty_like(x)
    for start in range(0, len(x), chunk):
        t = x[start:start+chunk, None]/width_nm
        radial_square = t*t+nodes[None, :]**2
        value = np.exp(-.5*t*t)*root_weights[None, :]
        previous = np.zeros_like(value)
        field, derivative = np.zeros_like(value, complex), np.zeros_like(value, complex)
        for n in range(len(extended)):
            if n < len(a):
                field += a[n]*value
            derivative += extended[n]*value
            if n+1 < len(extended):
                current = ((2*n+1-radial_square)*value-n*previous)/(n+1)
                previous, value = value, current
        flux[start:start+chunk] = np.sum((field.conj()*derivative).imag, axis=1)/(np.pi*width_nm)
    if not np.all(np.isfinite(flux)):
        raise ValueError("Nonfinite projected current; no display was published")
    return RadialCurrentProjection(_readonly(x), _readonly(flux), validated.integrated_axial_flux_fraction)


@dataclass(frozen=True)
class ExecutedRadialCurrentDiagram:
    boundary_indices: tuple[int, ...]
    z_nm: np.ndarray
    x_nm: np.ndarray
    axial_current_a_per_nm: np.ndarray  # (stored Z boundaries, projected X)
    total_axial_current_a: np.ndarray  # All transverse space, not window-fit.
    source_mixture_weight: float


def project_executed_boundaries(history, indices, x_nm, *, reference_k_per_nm,
        reference_current_a, source_mixture_weight, maximum_working_bytes=128*1024**2,
        cancelled=lambda: False):
    """Project actual cached boundaries with original tip mixture/current weights.

    No propagation, arbitrary-Z interpolation, trajectory sampling or phase
    aggregation occurs here. The selected indices are display sampling only;
    this function does not delete any retained history or physical effects.
    Both sides of a stop at identical Z remain separate rows.
    """
    indices = tuple(indices)
    if not indices or any(type(i) is not int for i in indices) or any(
            b <= a for a, b in zip(indices, indices[1:])):
        raise ValueError("Select increasing executed boundary indices")
    if (not math.isfinite(reference_current_a) or reference_current_a < 0
            or not math.isfinite(source_mixture_weight) or not 0 <= source_mixture_weight <= 1):
        raise ValueError("Readout requires the original nonnegative tip current and mixture weight")
    x = np.asarray(x_nm, float)
    if x.ndim != 1:
        raise ValueError("Projected X coordinates must be one-dimensional")
    output_bytes = 16*len(indices)*len(x)+32*len(indices)+16*len(x)
    if type(maximum_working_bytes) is not int or maximum_working_bytes <= output_bytes:
        raise MemoryError("Executed current diagram exceeds its display memory budget")
    from temsim.physics.wave_execution import check_available_memory
    check_available_memory(output_bytes)
    rows, totals, z = [], [], []
    scale = reference_current_a*source_mixture_weight
    for index in indices:
        if cancelled():
            raise InterruptedError("Current projection cancelled; executed history is unchanged")
        a, d, parameters = executed_boundary_parameters(history, index, reference_k_per_nm=reference_k_per_nm)
        result = radial_current_projection(x, a, d,
            maximum_working_bytes=maximum_working_bytes-output_bytes, **parameters)
        rows.append(scale*result.axial_flux_fraction_per_nm)
        totals.append(scale*result.integrated_axial_flux_fraction)
        z.append(history["z_nm"][index])
    return ExecutedRadialCurrentDiagram(indices, _readonly(z), _readonly(x),
        _readonly(rows), _readonly(totals), float(source_mixture_weight))
