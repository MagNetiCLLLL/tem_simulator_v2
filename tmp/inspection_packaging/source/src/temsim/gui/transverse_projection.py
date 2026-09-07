"""Shared display basis for transverse electron-beam projections."""

from __future__ import annotations

import numpy as np


def format_projection_angle(angle_deg: float) -> str:
    """Format a projection angle without insignificant trailing zeroes."""

    return f"{float(angle_deg):.2f}".rstrip("0").rstrip(".")


def projection_axis_name(angle_deg: float) -> str:
    """Name the in-plane axis shown by the Ray Diagram projection."""

    angle = float(angle_deg) % 360.0
    for cardinal_angle, name in (
        (0.0, "X"),
        (90.0, "Y"),
        (180.0, "-X"),
        (270.0, "-Y"),
    ):
        if np.isclose(angle, cardinal_angle, atol=0.05):
            return name
    return f"U({format_projection_angle(angle)} deg)"


def orthogonal_axis_name(angle_deg: float) -> str:
    """Name the transverse axis 90 degrees CCW from the projection axis."""

    angle = float(angle_deg) % 360.0
    for cardinal_angle, name in (
        (0.0, "Y"),
        (90.0, "-X"),
        (180.0, "-Y"),
        (270.0, "X"),
    ):
        if np.isclose(angle, cardinal_angle, atol=0.05):
            return name
    return f"V({format_projection_angle(angle)} deg)"


def project_transverse_values(x, y, angle_deg: float) -> np.ndarray:
    """Project physical X/Y values onto the Ray Diagram U axis."""

    angle_rad = np.deg2rad(float(angle_deg))
    return (
        np.asarray(x, dtype=float) * np.cos(angle_rad)
        + np.asarray(y, dtype=float) * np.sin(angle_rad)
    )


def transverse_view_coordinates(
    x,
    y,
    angle_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate physical X/Y into the matching right-handed U/V view basis.

    U is the Ray Diagram projection axis and V is its in-plane orthogonal:

    ``U = X cos(phi) + Y sin(phi)``
    ``V = -X sin(phi) + Y cos(phi)``
    """

    angle_rad = np.deg2rad(float(angle_deg))
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    return (
        x_values * cosine + y_values * sine,
        -x_values * sine + y_values * cosine,
    )
