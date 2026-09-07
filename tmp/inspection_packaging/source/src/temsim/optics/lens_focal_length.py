"""Equivalent focal length for each isolated three-Gaussian magnetic lens."""
from __future__ import annotations
import math
import numpy as np

from temsim.optics.excitation_policy import (
    DEFAULT_OPERATING_TARGET_PERCENT,
    is_saturated_excitation,
)

E = 1.602176634e-19
M = 9.1093837015e-31
C = 299792458.0

def electron_momentum(voltage_kv):
    kinetic = E * float(voltage_kv) * 1000.0
    rest = M * C * C
    return math.sqrt(kinetic * kinetic + 2.0 * kinetic * rest) / C

def _integrate_trapezoid(y, x):
    """NumPy-version-independent trapezoidal integration.

    Uses numpy.trapezoid when present. Otherwise performs the same 1-D
    trapezoidal sum directly, avoiding any dependency on numpy.trapz.
    """
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return float(trapezoid(y, x))
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.size < 2:
        return 0.0
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))

def _unit_field_samples(lens, samples=4001):
    if not lens.gaussian:
        return np.array([0.0]), np.array([0.0])
    reaches = [
        abs(g.offset * lens.a_mm) + 7.0 * abs(g.sigma * lens.a_mm)
        for g in lens.gaussian
    ]
    half_width_mm = max(max(reaches), abs(lens.a_mm), 1.0e-6)
    z_mm = np.linspace(-half_width_mm, half_width_mm, int(samples))
    field = np.zeros_like(z_mm)
    for g in lens.gaussian:
        sigma_mm = max(abs(g.sigma * lens.a_mm), 1.0e-12)
        centre_mm = g.offset * lens.a_mm
        field += g.amplitude * np.exp(-0.5 * ((z_mm - centre_mm) / sigma_mm) ** 2)
    if bool(getattr(lens, "normalise_profile_peak", False)):
        field /= max(float(np.max(np.abs(field))), 1e-15)
    return z_mm, field

def unit_field_peak(lens, samples=4001):
    _, field = _unit_field_samples(lens, samples)
    return float(np.max(np.abs(field)))

def unit_field_integral(lens, samples=4001):
    z_mm, field = _unit_field_samples(lens, samples)
    return _integrate_trapezoid(field * field, z_mm * 1.0e-3)

def focal_length_m(lens, voltage_kv):
    if hasattr(lens, "focal_length_for_voltage_mm"):
        value_mm = lens.focal_length_for_voltage_mm(voltage_kv)
        return (
            math.inf
            if not math.isfinite(value_mm)
            else value_mm * 1.0e-3
        )
    if not bool(getattr(lens, "enabled", True)):
        return math.inf
    excitation = abs(float(lens.percent)) / 100.0
    b0 = abs(float(lens.b0_t))
    if excitation <= 0.0 or b0 <= 0.0:
        return math.inf
    momentum = electron_momentum(voltage_kv)
    power = (E / (2.0 * momentum)) ** 2 * (b0 * excitation) ** 2 * unit_field_integral(lens)
    return math.inf if power <= 0.0 else 1.0 / power

def focal_length_mm(lens, voltage_kv):
    value = focal_length_m(lens, voltage_kv)
    return math.inf if not math.isfinite(value) else value * 1000.0

def required_actual_field_scale(lens, voltage_kv, target_focal_mm):
    target_m = float(target_focal_mm) * 1.0e-3
    if not math.isfinite(target_m) or target_m <= 0.0:
        raise ValueError("Focal length must be a positive finite value")
    integral = unit_field_integral(lens)
    if integral <= 0.0:
        raise ValueError(f"{lens.name} has no usable Gaussian field profile")
    momentum = electron_momentum(voltage_kv)
    coefficient = (E / (2.0 * momentum)) ** 2 * integral
    return math.sqrt(1.0 / (target_m * coefficient))

def set_focal_length(lens, voltage_kv, target_focal_mm):
    if hasattr(lens, "set_focal_length_for_voltage_mm"):
        return lens.set_focal_length_for_voltage_mm(
            voltage_kv, target_focal_mm
        )
    required_scale = required_actual_field_scale(lens, voltage_kv, target_focal_mm)
    b0 = abs(float(lens.b0_t))
    required_percent = math.inf if b0 <= 0.0 else 100.0 * required_scale / b0

    if (
        required_percent <= float(lens.max_percent)
        and not is_saturated_excitation(
            required_percent, float(lens.max_percent)
        )
    ):
        lens.percent = required_percent
        return "percentage"

    lens.percent = DEFAULT_OPERATING_TARGET_PERCENT
    lens.b0_t = (
        required_scale * 100.0 / DEFAULT_OPERATING_TARGET_PERCENT
    )
    return "maximum field"
