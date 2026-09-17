"""Twofold stigmation in the fixed column X/Y frame.

Each channel is a trace-free quadrupole. Its mechanical orientation enters
with twice the angle; 0/45 degree channels span both independent components.
The co-located Gaussian model is an ideal effective field, not an OEM coil map.
"""
from __future__ import annotations

import numpy as np

LEGACY = "legacy_difference"
DUAL = "normal_skew"
MODELS = (LEGACY, DUAL)


def validate_stigmator_field(component):
    if component.field_model not in MODELS:
        raise ValueError(f"Unknown stigmator field model: {component.field_model}")
    values = (component.channel_x_angle_deg, component.channel_y_angle_deg,
              component.strength_x_percent, component.strength_y_percent,
              component.max_strength_m2, component.length_mm)
    if not np.all(np.isfinite(values)):
        raise ValueError("Stigmator angles and strengths must be finite")
    if component.max_strength_m2 <= 0 or component.length_mm <= 0:
        raise ValueError("Stigmator field length and strength scale must be positive")
    separation = np.deg2rad(2 * (component.channel_y_angle_deg - component.channel_x_angle_deg))
    if abs(np.sin(separation)) < 1e-6:
        raise ValueError("Stigmator structural channels must be independent quadrupoles")


def quadrupole_tensor_components(component, z_mm):
    """Return Kxx, Kyy, Kxy in m^-2; theta' = -K @ (x,y)."""
    validate_stigmator_field(component)
    z = np.asarray(z_mm, dtype=float)
    zero = np.zeros_like(z)
    if not component.enabled:
        return zero, zero, zero
    envelope = np.exp(-0.5 * ((z - component.z_mm) / (component.length_mm / 2.355)) ** 2)
    x = component.max_strength_m2 * component.strength_x_percent / 100.0
    y = component.max_strength_m2 * component.strength_y_percent / 100.0
    if component.field_model == LEGACY:
        normal = 0.5 * (x - y) * envelope
        return normal, -normal, zero
    ax, ay = np.deg2rad(2 * np.array([component.channel_x_angle_deg, component.channel_y_angle_deg]))
    normal = (x * np.cos(ax) + y * np.cos(ay)) * envelope
    skew = (x * np.sin(ax) + y * np.sin(ay)) * envelope
    return normal, -normal, skew
