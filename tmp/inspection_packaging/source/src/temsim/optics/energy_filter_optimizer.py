"""Explicit joint tuning of the seven pre-slit Iliad multipoles."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import least_squares

from temsim.optics.energy_filter import (
    configure_energy_slit_from_software,
    energy_filter_voltage_match_status,
    ensure_energy_filter,
)
from temsim.optics.energy_filter_metrics import (
    SlitPlaneMetrics,
    measure_slit_plane_metrics,
)


@dataclass(frozen=True)
class EnergyFilterOptimizationResult:
    success: bool
    message: str
    function_evaluations: int
    optimized_orders: tuple
    initial_metrics: SlitPlaneMetrics
    final_metrics: SlitPlaneMetrics
    initial_residual_rms: float
    final_residual_rms: float


def _coefficient_parameters(elements, orders, reference_radius_m):
    values = []
    for element in elements:
        for family in ("normal", "skew"):
            coefficients = getattr(
                element.multipole_field,
                f"{family}_coefficients",
            )
            for order in orders:
                values.append(
                    coefficients[order - 1]
                    * reference_radius_m ** (order - 1)
                )
    return np.asarray(values, dtype=float)


def _load_coefficient_parameters(
    elements,
    orders,
    reference_radius_m,
    parameters,
):
    index = 0
    for element in elements:
        for family in ("normal", "skew"):
            for order in orders:
                coefficient = (
                    float(parameters[index])
                    / reference_radius_m ** (order - 1)
                )
                element.multipole_field.set_component(
                    order,
                    **{family: coefficient},
                )
                index += 1


def _metric_residuals(metrics, target_dispersion, parameters, bound_t):
    span = float(metrics.energy_span_ev)
    target_scale = max(abs(float(target_dispersion)), 1.0e-6)
    optical = np.array([
        metrics.reference_dispersive_um / 10.0,
        metrics.reference_non_dispersive_um / 10.0,
        metrics.astigmatism_um / 100.0,
        metrics.spectral_line_bow_um / 10.0,
        metrics.non_isochromaticity_ev_rms / 100.0,
        metrics.non_isochromaticity_ev_peak_to_peak / 200.0,
        (
            metrics.second_order_dispersion_um_per_ev2
            * span**2
            / 10.0
        ),
        (
            metrics.dispersion_um_per_ev - target_dispersion
        ) / target_scale,
    ])
    if not metrics.all_rays_reached_slit or not np.all(
        np.isfinite(optical)
    ):
        optical = np.full(8, 1.0e4)
    # A weak physical regularizer prevents underdetermined families from
    # acquiring large cancelling fields.
    regularizer = 0.02 * np.asarray(parameters) / float(bound_t)
    return np.concatenate((optical, regularizer))


def optimize_energy_filter_m12(
    state,
    *,
    orders=(1, 2, 3),
    reference_radius_mm=2.0,
    maximum_field_at_reference_t=0.02,
    max_function_evaluations=12,
):
    """Jointly tune M01-M07; called only by an explicit user action."""

    ensure_energy_filter(state)
    energy_filter = state.energy_filter
    current_voltage = float(state.beam_voltage_kv)
    if not math.isclose(
        current_voltage,
        float(energy_filter.matched_voltage_kv),
        abs_tol=1.0e-12,
        rel_tol=0.0,
    ):
        raise ValueError(
            energy_filter_voltage_match_status(state)
            + " Match the Energy Filter before optimizing M12."
        )
    orders = tuple(sorted({int(order) for order in orders}))
    if not orders or orders[0] < 1 or orders[-1] > 6:
        raise ValueError("Optimized M12 orders must be within n=1...6.")
    reference_radius_m = float(reference_radius_mm) * 1.0e-3
    bound_t = float(maximum_field_at_reference_t)
    if (
        not math.isfinite(reference_radius_m)
        or reference_radius_m <= 0.0
        or not math.isfinite(bound_t)
        or bound_t <= 0.0
    ):
        raise ValueError("Optimizer radius and field bound must be positive.")
    elements = tuple(energy_filter.multipoles[:7])
    initial = _coefficient_parameters(
        elements,
        orders,
        reference_radius_m,
    )
    original_normal = [
        element.multipole_field.normal_coefficients
        for element in elements
    ]
    original_skew = [
        element.multipole_field.skew_coefficients
        for element in elements
    ]
    initial_metrics = measure_slit_plane_metrics(state)
    target_dispersion = initial_metrics.dispersion_um_per_ev
    initial_residual = _metric_residuals(
        initial_metrics,
        target_dispersion,
        initial,
        bound_t,
    )

    def residual(parameters):
        _load_coefficient_parameters(
            elements,
            orders,
            reference_radius_m,
            parameters,
        )
        try:
            metrics = measure_slit_plane_metrics(state)
        except (RuntimeError, ValueError, FloatingPointError):
            return np.full(initial_residual.shape, 1.0e4)
        return _metric_residuals(
            metrics,
            target_dispersion,
            parameters,
            bound_t,
        )

    try:
        solution = least_squares(
            residual,
            initial,
            bounds=(-bound_t, bound_t),
            max_nfev=max(1, int(max_function_evaluations)),
            x_scale=max(bound_t * 0.1, 1.0e-6),
            diff_step=2.0e-3,
        )
        _load_coefficient_parameters(
            elements,
            orders,
            reference_radius_m,
            solution.x,
        )
        final_metrics = measure_slit_plane_metrics(state)
    except Exception:
        for element, normal, skew in zip(
            elements,
            original_normal,
            original_skew,
        ):
            for order in range(1, 7):
                element.multipole_field.set_component(
                    order,
                    normal=normal[order - 1],
                    skew=skew[order - 1],
                )
        raise

    final_residual = _metric_residuals(
        final_metrics,
        target_dispersion,
        solution.x,
        bound_t,
    )
    initial_rms = float(np.sqrt(np.mean(initial_residual**2)))
    final_rms = float(np.sqrt(np.mean(final_residual**2)))
    accepted = math.isfinite(final_rms) and final_rms < initial_rms
    if accepted:
        for element in elements:
            element.calibration.set_reference_from_field(
                element.multipole_field,
                current_voltage,
            )
        energy_filter._last_slit_metrics = final_metrics
        energy_filter.energy_slit.calibrated_dispersion_um_per_ev = (
            final_metrics.dispersion_um_per_ev
        )
        configure_energy_slit_from_software(energy_filter)
    else:
        for element, normal, skew in zip(
            elements,
            original_normal,
            original_skew,
        ):
            for order in range(1, 7):
                element.multipole_field.set_component(
                    order,
                    normal=normal[order - 1],
                    skew=skew[order - 1],
                )
        final_metrics = initial_metrics
        final_rms = initial_rms
    return EnergyFilterOptimizationResult(
        success=bool(accepted),
        message=(
            str(solution.message)
            if accepted
            else "No improving multipole solution was applied."
        ),
        function_evaluations=int(solution.nfev),
        optimized_orders=orders,
        initial_metrics=initial_metrics,
        final_metrics=final_metrics,
        initial_residual_rms=initial_rms,
        final_residual_rms=final_rms,
    )
