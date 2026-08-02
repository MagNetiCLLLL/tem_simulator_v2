"""Measured optical properties at the Energy Filter slit plane."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from temsim.optics.energy_filter import ensure_energy_filter
from temsim.optics.energy_filter_raytrace import (
    trace_energy_filter_batch,
)


@dataclass(frozen=True)
class SlitPlaneMetrics:
    """Finite-ray measurements; these are not thin-lens coefficients."""

    dispersion_um_per_ev: float
    second_order_dispersion_um_per_ev2: float
    astigmatism_um: float
    spectral_line_bow_um: float
    non_isochromaticity_ev_rms: float
    non_isochromaticity_ev_peak_to_peak: float
    reference_dispersive_um: float
    reference_non_dispersive_um: float
    field_radius_mm: float
    energy_span_ev: float
    all_rays_reached_slit: bool

    def summary(self):
        return (
            f"dispersion {self.dispersion_um_per_ev:.4g} µm/eV | "
            f"2nd order {self.second_order_dispersion_um_per_ev2:.4g} "
            f"µm/eV² | astigmatism {self.astigmatism_um:.4g} µm | "
            f"line bow {self.spectral_line_bow_um:.4g} µm | "
            f"non-isochromaticity RMS "
            f"{self.non_isochromaticity_ev_rms:.4g} eV"
        )


def measure_slit_plane_metrics(
    state,
    *,
    energy_span_ev=10.0,
    field_radius_mm=0.25,
    angular_radius_mrad=0.25,
):
    """Trace a diagnostic ray bundle and measure the slit-plane optics."""

    ensure_energy_filter(state)
    energy_span = float(energy_span_ev)
    field_radius = float(field_radius_mm)
    angular_radius = float(angular_radius_mrad)
    if not all(math.isfinite(value) and value > 0.0 for value in (
        energy_span,
        field_radius,
        angular_radius,
    )):
        raise ValueError("Metric ray-bundle spans must be positive.")

    # First three rays measure x(loss).  The next 3x3 grid measures
    # isochromaticity and line bow.  The last four equal-angle rays compare
    # the two principal focal sections.
    loss = np.array([-energy_span, 0.0, energy_span])
    field_xy = np.array([
        (-field_radius, -field_radius),
        (0.0, -field_radius),
        (field_radius, -field_radius),
        (-field_radius, 0.0),
        (0.0, 0.0),
        (field_radius, 0.0),
        (-field_radius, field_radius),
        (0.0, field_radius),
        (field_radius, field_radius),
    ])
    ray_count = 3 + len(field_xy) + 4
    x_mm = np.zeros(ray_count)
    y_mm = np.zeros(ray_count)
    tx_rad = np.zeros(ray_count)
    ty_rad = np.zeros(ray_count)
    energy_offset_ev = np.zeros(ray_count)
    energy_offset_ev[:3] = -loss
    x_mm[3:12] = field_xy[:, 0]
    y_mm[3:12] = field_xy[:, 1]
    angle = angular_radius * 1.0e-3
    tx_rad[12:14] = (-angle, angle)
    ty_rad[14:16] = (-angle, angle)

    batch = trace_energy_filter_batch(
        state,
        x_mm,
        y_mm,
        tx_rad,
        ty_rad,
        energy_offset_ev,
        apply_slit=False,
        stop_at_slit=True,
    )
    reached = np.asarray(batch.reached_slit, dtype=bool)
    if not np.all(reached[:3]):
        raise RuntimeError(
            "Reference energy rays did not reach the slit plane."
        )
    dispersive_um = batch.slit_dispersive_m * 1.0e6
    non_dispersive_um = batch.slit_non_dispersive_m * 1.0e6
    quadratic, linear, intercept = np.polyfit(
        loss,
        dispersive_um[:3],
        2,
    )
    if abs(linear) <= 1.0e-12:
        raise RuntimeError(
            "Energy dispersion is too small to measure "
            "non-isochromaticity."
        )

    field_dispersive = dispersive_um[3:12]
    equivalent_loss = (
        field_dispersive - intercept
    ) / linear
    valid_field = reached[3:12] & np.isfinite(equivalent_loss)
    if np.any(valid_field):
        equivalent_loss = equivalent_loss[valid_field]
        non_iso_rms = float(
            np.sqrt(np.mean(equivalent_loss**2))
        )
        non_iso_ptp = float(np.ptp(equivalent_loss))
    else:
        non_iso_rms = math.inf
        non_iso_ptp = math.inf

    # At x=0, compare the average top/bottom spectral-line position to
    # the centre.  This reports the bow over the requested field radius.
    line_bow = float(
        0.5 * (field_dispersive[1] + field_dispersive[7])
        - field_dispersive[4]
    )
    x_angle = dispersive_um[12:14]
    y_angle = non_dispersive_um[14:16]
    x_half_span = 0.5 * abs(float(x_angle[1] - x_angle[0]))
    y_half_span = 0.5 * abs(float(y_angle[1] - y_angle[0]))

    return SlitPlaneMetrics(
        dispersion_um_per_ev=float(linear),
        second_order_dispersion_um_per_ev2=float(quadratic),
        astigmatism_um=float(x_half_span - y_half_span),
        spectral_line_bow_um=line_bow,
        non_isochromaticity_ev_rms=non_iso_rms,
        non_isochromaticity_ev_peak_to_peak=non_iso_ptp,
        reference_dispersive_um=float(intercept),
        reference_non_dispersive_um=float(
            non_dispersive_um[4 + 3]
        ),
        field_radius_mm=field_radius,
        energy_span_ev=energy_span,
        all_rays_reached_slit=bool(np.all(reached)),
    )

