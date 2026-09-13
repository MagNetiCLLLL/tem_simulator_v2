"""Spherical emitting-cap geometry, shared by particle launch and the editor.

The apex is (0, 0, 0), +Z is downstream, and the sphere centre is (0, 0, -R).
The patch half-angle is measured at that centre. It is not the independent
particle angle relative to the local surface normal.
"""
import math

import numpy as np


def patch_dimensions(geometry, half_angle_deg: float) -> dict[str, float]:
    """Derived geometry only; no independently adjustable source size."""
    geometry.validate()
    if (isinstance(half_angle_deg, bool) or not math.isfinite(half_angle_deg)
            or not 0 < half_angle_deg < 90 - geometry.cone_half_angle_deg):
        raise ValueError("Emission half-angle must lie inside the spherical tip cap")
    angle = math.radians(half_angle_deg)
    radius = geometry.apex_radius_nm
    sine_half = math.sin(angle / 2)
    return {
        "half_angle_rad": angle,
        "full_angle_deg": 2 * half_angle_deg,
        "apex_curvature_nm_inv": 1 / radius,
        "projected_diameter_nm": 2 * radius * math.sin(angle),
        "cap_depth_nm": 2 * radius * sine_half**2,
        "apex_to_edge_arc_nm": radius * angle,
        "surface_area_nm2": 4 * math.pi * radius**2 * sine_half**2,
    }


def sample_cap_frame(geometry, half_angle_deg, area_quantiles, azimuth_quantiles):
    """Return (positions [m], normals, tangent1, tangent2), each (N, 3).

    Uniform area means uniform 1-cos(theta), not uniform theta. Half-angle
    identities preserve transverse coordinates and cap depth for narrow caps.
    They do not move particles to a planar or accelerated downstream source.
    """
    dimensions = patch_dimensions(geometry, half_angle_deg)
    u, a = np.asarray(area_quantiles, float), np.asarray(azimuth_quantiles, float)
    if (u.ndim != 1 or a.shape != u.shape or not np.all(np.isfinite([u, a]))
            or np.any(u < 0) or np.any(u > 1) or np.any(a < 0) or np.any(a >= 1)):
        raise ValueError("Cap sampling requires finite area/azimuth quantiles in [0, 1]")
    sine_half_squared = u * math.sin(dimensions["half_angle_rad"] / 2)**2
    cosine = 1 - 2 * sine_half_squared
    sine = 2 * np.sqrt(sine_half_squared * (1 - sine_half_squared))
    phi = 2 * np.pi * a
    normals = np.column_stack((sine * np.cos(phi), sine * np.sin(phi), cosine))
    radius = geometry.apex_radius_nm * 1e-9
    positions = radius * normals
    positions[:, 2] = -2 * radius * sine_half_squared
    tangent1 = np.column_stack((cosine * np.cos(phi), cosine * np.sin(phi), -sine))
    tangent2 = np.column_stack((-np.sin(phi), np.cos(phi), np.zeros_like(u)))
    return positions, normals, tangent1, tangent2
