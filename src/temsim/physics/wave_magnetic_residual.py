"""Hermitian spectral magnetic correction on a sampled complex envelope.

The supplied real coefficients define H = W + sum({v_j, D_j/k}/2), with
D=-i grad in the laboratory XY coordinates and k=2*pi/lambda. Coefficients
already include any analytical-carrier contribution; this operator does not
sample, change or remove that carrier. It advances the finite periodic-grid
Hamiltonian. Its exponential-series bound does NOT certify coefficient
sampling, physical-domain size, splitting, or the underlying field model.
"""
from dataclasses import replace
import math

import numpy as np

from temsim.physics.wave_device import array_module


_MAX_SUBSTEPS = 1024
_MAX_ORDER = 64
_MAX_OPERATOR_APPLICATIONS = 32768
# Includes coefficients, frequency grids, Taylor vectors, simultaneous
# complex derivative/flux batches, output copies and FFT scratch headroom.
# Captured upstream fields and the caller's coefficient-builder temporaries
# are additional retained memory, not included in this per-pixel estimate.
WORKING_BYTES_PER_PIXEL = 512


def _series_plan(kappa, error_budget):
    """Bound total operator error, including composition of approximate steps."""
    if not math.isfinite(kappa):
        raise ValueError("Magnetic residual exponential has an unbounded generator")
    if kappa == 0:
        return 0, 0, 0.
    if error_budget < np.finfo(float).eps:
        raise ValueError("Magnetic residual nonzero propagation error budget is below "
                         "float64 machine precision; this precision cannot be promised")
    # ||-ik dz H/m|| <= 1 keeps direct Taylor evaluation well conditioned.
    if kappa > _MAX_SUBSTEPS:
        raise ValueError("Magnetic residual exponential exceeds its substep budget; "
                         "reduce the propagation step before applying the operator")
    substeps = max(1, math.ceil(kappa))
    local = kappa/substeps
    target = min(error_budget, 1e-12)
    log_target = math.log(target)
    for order in range(1, _MAX_ORDER+1):
        # Taylor remainder <= exp(local) local**(order+1)/(order+1)!.
        log_tail = local+(order+1)*math.log(local)-math.lgamma(order+2)
        # For m steps with exact unitary U and error <= tail in each,
        # ||T**m-U**m|| <= (1+tail)**m-1 <= m*tail*exp(m*tail).
        tail = math.exp(log_tail)
        log_total = math.log(substeps)+log_tail+substeps*tail
        if log_total <= log_target:
            if substeps*order > _MAX_OPERATOR_APPLICATIONS:
                raise ValueError("Magnetic residual exponential exceeds its operator budget; "
                                 "reduce the propagation step")
            bound = float(np.nextafter(math.exp(log_total), np.inf))
            if bound > target:
                continue
            return substeps, order, bound
    raise ValueError("Magnetic residual exponential cannot satisfy the requested error "
                     "within its Taylor-order budget")


def _apply_hermitian(amplitude, velocity, scalar, frequency, xp):
    """Self-adjoint discrete operator; real Fourier symbols on any affine grid."""
    transformed = xp.fft.fft2(amplitude)
    derivatives = xp.fft.ifft2(frequency*transformed[None, :, :], axes=(-2, -1))
    flux_derivatives = xp.fft.ifft2(
        frequency*xp.fft.fft2(velocity*amplitude[None, :, :], axes=(-2, -1)),
        axes=(-2, -1))
    return scalar*amplitude+.5*xp.sum(velocity*derivatives+flux_derivatives, axis=0)


def apply_magnetic_residual(wave, velocity, scalar, wavelength, distance, *,
                            error_budget, cancelled=lambda: False):
    """Return ``(wave, record)`` after exp(-i k distance H) on its envelope.

    ``velocity`` has shape (2, ny, nx); ``scalar`` has shape (ny, nx).
    Both are real, finite, dimensionless laboratory-coordinate coefficients.
    Wavelength and signed propagation distance are in metres. Error budget
    bounds the exponential-series *relative amplitude operator* truncation;
    separately reported floating-point allowance is used for the norm check.
    A nonzero operator rejects budgets below float64 machine precision; a
    series bound alone is not a claim of achieved total numerical accuracy.
    Input arrays, physical lattice, origin and phase carriers are unchanged.
    """
    if cancelled():
        raise InterruptedError("Magnetic residual propagation cancelled")
    if not math.isfinite(wavelength) or wavelength <= 0:
        raise ValueError("Magnetic residual wavelength must be finite and positive")
    if not math.isfinite(distance):
        raise ValueError("Magnetic residual distance must be finite")
    if not math.isfinite(error_budget) or error_budget <= 0:
        raise ValueError("Magnetic residual error budget must be finite and positive")
    xp = array_module(wave.amplitude)
    raw_velocity, raw_scalar = xp.asarray(velocity), xp.asarray(scalar)
    if xp.iscomplexobj(raw_velocity) or xp.iscomplexobj(raw_scalar):
        raise ValueError("Magnetic residual coefficients must be real")
    if (raw_velocity.shape != (2, *wave.amplitude.shape)
            or raw_scalar.shape != wave.amplitude.shape):
        raise ValueError("Magnetic residual coefficient shapes must match the wave lattice")
    velocity = xp.asarray(raw_velocity, dtype=xp.float64)
    scalar = xp.asarray(raw_scalar, dtype=xp.float64)
    if not bool(xp.all(xp.isfinite(velocity))) or not bool(xp.all(xp.isfinite(scalar))):
        raise ValueError("Magnetic residual coefficients must be finite")
    ny, nx = wave.amplitude.shape
    fy, fx = xp.meshgrid(xp.fft.fftfreq(ny), xp.fft.fftfreq(nx), indexing="ij")
    frequency = wavelength*xp.einsum("ij,jyx->iyx",
        xp.asarray(np.linalg.inv(wave.basis_m).T), xp.stack((fx, fy)))
    coefficient_bound = float(xp.max(abs(scalar)))
    coefficient_bound += sum(float(xp.max(abs(velocity[j])))*float(xp.max(abs(frequency[j])))
                             for j in range(2))
    phase_distance = 2*math.pi*(distance/wavelength)
    kappa = abs(phase_distance)*coefficient_bound
    substeps, order, truncation_bound = _series_plan(kappa, error_budget)
    before = wave.probability
    amplitude = wave.amplitude
    if substeps and before:
        factor = -1j*phase_distance/substeps
        for step in range(substeps):
            if cancelled():
                raise InterruptedError("Magnetic residual propagation cancelled")
            term = amplitude
            total = amplitude.copy()
            for number in range(1, order+1):
                if number % 4 == 0 and cancelled():
                    raise InterruptedError("Magnetic residual propagation cancelled")
                term = (factor/number)*_apply_hermitian(term, velocity, scalar, frequency, xp)
                total = total+term
            amplitude = total
    after = float(xp.sum(abs(amplitude)**2))
    # FFT and pointwise arithmetic have finite rounding error in addition to
    # the exact-arithmetic Taylor bound. Report this allowance separately;
    # it is not presented as an independent rigorous floating-point proof.
    applications = substeps*order
    roundoff_allowance = 64*np.finfo(float).eps*max(1, applications)*(1+math.log2(nx*ny))
    norm_tolerance = before*(2*truncation_bound+truncation_bound**2+roundoff_allowance)
    if not math.isfinite(after) or abs(after-before) > norm_tolerance:
        raise ValueError("Magnetic residual lossless norm check failed: "
                         f"before {before:.17g}, after {after:.17g}, "
                         f"allowed absolute change {norm_tolerance:.6g}; no normalization was applied")
    record = {"method": "Hermitian spectral magnetic residual / bounded Taylor exponential",
        "compute_backend": "numpy" if xp is np else "cupy",
        "distance_m": float(distance), "generator_norm_bound": kappa,
        "hamiltonian_norm_bound": coefficient_bound,
        "substeps": substeps, "taylor_order": order, "operator_applications": applications,
        "requested_error_budget": float(error_budget),
        "series_error_bound": truncation_bound,
        "roundoff_norm_relative_allowance": roundoff_allowance,
        "input_probability": before, "output_probability": after,
        "scope": "finite-grid exponential series only; coefficient sampling, physical domain and axial splitting require separate convergence"}
    return replace(wave, amplitude=amplitude), record
