"""Read-only observables of an executed, flux-normalised radial gun mode.

This does not emit particles, define a downstream source, or admit imaging.
In the gun convention integral Im(psi* d_z psi) dA is a source-current
fraction. |psi|^2 is scalar wave intensity, NOT transmitted current density
or a normalised probability density. Lengths are nm; +Z is downstream.
"""
from dataclasses import dataclass
import math

import numpy as np

from temsim.physics.quartic_radial_phase import radial_envelope


def _readonly(value):
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def normal_derivative_coefficients(a, d, *, width_nm, reference_k_per_nm,
        width_log_rate_per_nm, curvature_prime_per_nm2, quartic_prime_per_nm5):
    """Complete derivative of a validated boundary state in its executed chart."""
    from temsim.physics.radial_gun_wave import laguerre_operators
    count = len(a)
    x, dilation, _ = laguerre_operators(count+2)
    connection = (.5*reference_k_per_nm*curvature_prime_per_nm2*width_nm**2*x
        +quartic_prime_per_nm5*width_nm**4*(x@x)+1j*width_log_rate_per_nm*dilation)
    return np.r_[d, 1j*connection[count:, :count]@a]


@dataclass(frozen=True)
class RadialModeObservables:
    radius_nm: np.ndarray
    amplitude: np.ndarray
    normal_derivative_per_nm: np.ndarray
    scalar_intensity: np.ndarray
    axial_flux_fraction_per_nm2: np.ndarray
    phase_rad: np.ndarray  # NaN at an exactly zero amplitude; one mode only.
    integrated_axial_flux_fraction: float


def radial_mode_observables(radius_nm, coefficients, covariant_derivatives, *,
                            width_nm, curvature_per_nm, reference_k_per_nm,
                            width_log_rate_per_nm, curvature_prime_per_nm2,
                            quartic_phase_per_nm4, quartic_prime_per_nm5):
    """Evaluate the same finite-basis field, including its outside-basis derivative.

    For psi=B_P a and d=a'+i G_PP a, the physical derivative is
    B_P d + i B_Q G_QP a, not just B_P d. G has bandwidth two for
    the quadratic/quartic moving chart, so two additional derivative
    coefficients suffice exactly. No extra propagated/source modes are added.
    All chart rates must come from the executed coordinate frame.
    """
    radius = np.asarray(radius_nm, float)
    a, d = np.asarray(coefficients, complex), np.asarray(covariant_derivatives, complex)
    parameters = (width_nm, curvature_per_nm, reference_k_per_nm,
        width_log_rate_per_nm, curvature_prime_per_nm2,
        quartic_phase_per_nm4, quartic_prime_per_nm5)
    if (radius.ndim != 1 or not len(radius) or np.any(radius < 0)
            or not np.all(np.isfinite(radius)) or np.any(np.diff(radius) <= 0)):
        raise ValueError("Wave observables need increasing, finite, nonnegative radial coordinates")
    if (a.ndim != 1 or not len(a) or d.shape != a.shape
            or not np.all(np.isfinite(a)) or not np.all(np.isfinite(d))):
        raise ValueError("Wave observables need matching finite complex boundary states")
    if not all(math.isfinite(v) for v in parameters) or min(width_nm, reference_k_per_nm) <= 0:
        raise ValueError("Wave observables need the finite executed chart and positive width/wave number")
    extended_derivative = normal_derivative_coefficients(a, d, width_nm=width_nm,
        reference_k_per_nm=reference_k_per_nm, width_log_rate_per_nm=width_log_rate_per_nm,
        curvature_prime_per_nm2=curvature_prime_per_nm2, quartic_prime_per_nm5=quartic_prime_per_nm5)
    phase = np.exp(1j*(.5*reference_k_per_nm*curvature_per_nm*radius**2
                       +quartic_phase_per_nm4*radius**4))
    amplitude = radial_envelope(radius, width_nm, a)*phase
    derivative = radial_envelope(radius, width_nm, extended_derivative)*phase
    intensity = abs(amplitude)**2
    flux = (amplitude.conj()*derivative).imag
    angle = np.where(intensity == 0., np.nan, np.angle(amplitude))
    # Orthogonality cancels the Q contribution in the complete radial integral,
    # although that contribution is essential to the LOCAL flux distribution.
    integrated = float(np.vdot(a, d).imag)
    return RadialModeObservables(*map(_readonly,
        (radius, amplitude, derivative, intensity, flux, angle)), integrated)


def incoherent_radial_observables(modes, weights, *, reference_current_a):
    """Sum observables, never amplitudes/phases of incoherent energy modes.

    Weights are the original source mixture weights, not exit-normalised
    weights; absorption is already present in each executed field. A selected
    subset may sum to less than one and is never renormalised here.
    """
    modes, weights = tuple(modes), np.asarray(weights, float)
    if (not modes or weights.shape != (len(modes),) or not np.all(np.isfinite(weights))
            or np.any(weights < 0) or weights.sum() > 1.+1e-12):
        raise ValueError("Mixture readout requires source weights without exit renormalisation")
    if not math.isfinite(reference_current_a) or reference_current_a < 0:
        raise ValueError("Reference source current must be finite and nonnegative")
    radius = modes[0].radius_nm
    if any(not np.array_equal(mode.radius_nm, radius) for mode in modes):
        raise ValueError("Mixture observables require the same radial readout coordinates")
    intensity = sum(weight*mode.scalar_intensity for weight, mode in zip(weights, modes))
    density = reference_current_a*sum(weight*mode.axial_flux_fraction_per_nm2
                                    for weight, mode in zip(weights, modes))
    current = reference_current_a*sum(weight*mode.integrated_axial_flux_fraction
                                    for weight, mode in zip(weights, modes))
    return {"radius_nm": _readonly(radius), "scalar_intensity": _readonly(intensity),
        "axial_current_density_a_per_nm2": _readonly(density),
        "integrated_axial_current_a": float(current),
        "phase_scope": "No aggregate phase for an incoherent mixture; inspect individual modes"}


def executed_boundary_parameters(history, index, *, reference_k_per_nm):
    """Read exact stored coordinates, including both sides of coincident stops."""
    if "chart_derivatives" not in history:
        raise ValueError("This historical wave has no exact chart derivatives for local-current readout")
    length = len(history["z_nm"])
    expected = {"width_curvature": (length, 2), "quartic_phase_per_nm4": (length,),
                "chart_derivatives": (length, 3)}
    if any(np.shape(history[key]) != shape for key, shape in expected.items()):
        raise ValueError("Executed wave history has inconsistent boundary coordinates")
    a, d = np.asarray(history["coefficients"]), np.asarray(history["covariant_derivatives"])
    if a.ndim != 2 or d.shape != a.shape or len(a) != length:
        raise ValueError("Executed wave history has inconsistent complex boundary states")
    if type(index) is not int or not 0 <= index < length:
        raise ValueError("Choose an executed boundary index; arbitrary Z requires propagation")
    width, curvature = history["width_curvature"][index]
    rate, curvature_prime, quartic_prime = history["chart_derivatives"][index]
    return a[index], d[index], dict(width_nm=width,
        curvature_per_nm=curvature, reference_k_per_nm=reference_k_per_nm,
        width_log_rate_per_nm=rate, curvature_prime_per_nm2=curvature_prime,
        quartic_phase_per_nm4=history["quartic_phase_per_nm4"][index], quartic_prime_per_nm5=quartic_prime)


def observe_executed_boundary(history, index, radius_nm, *, reference_k_per_nm):
    """Read one stored plane; missing chart rates never become inferred zeros."""
    a, d, parameters = executed_boundary_parameters(history, index, reference_k_per_nm=reference_k_per_nm)
    return radial_mode_observables(radius_nm, a, d, **parameters)
