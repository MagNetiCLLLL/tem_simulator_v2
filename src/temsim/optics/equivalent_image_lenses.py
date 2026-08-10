"""Equivalent focal-length events for coordinated TEM image presets.

The projector column is mechanically reconstructed rather than calibrated from
an OEM current table.  In image mode we therefore offer an explicit paraxial
engineering model: each round lens is reduced to a thin focusing event whose
power is derived from its isolated ``integral(Bz**2 dz)``.  Its signed Larmor
rotation is retained as a separate event.  Diffraction mode continues to use
the distributed magnetic fields because effective camera length is calibrated
against the Objective back-focal plane instead.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from temsim.component_keys import (
    DIFFRACTION_LENS,
    INTERMEDIATE_LENS,
    OBJECTIVE_LENS,
    PROJECTOR_LENS_1,
    PROJECTOR_LENS_2,
)
from temsim.optics.lens_focal_length import E, electron_momentum


IMAGE_LENS_KEYS = (
    OBJECTIVE_LENS,
    DIFFRACTION_LENS,
    INTERMEDIATE_LENS,
    PROJECTOR_LENS_1,
    PROJECTOR_LENS_2,
)


@dataclass(frozen=True, slots=True)
class EquivalentImageLensCalibration:
    key: str
    z_mm: float
    maximum_percent: float
    power_at_100_percent_m1: float
    rotation_at_100_percent_rad: float

    def event(self, percent: float) -> "EquivalentImageLensEvent":
        fraction = float(percent) / 100.0
        return EquivalentImageLensEvent(
            key=self.key,
            z_mm=self.z_mm,
            power_m1=self.power_at_100_percent_m1 * fraction**2,
            rotation_rad=self.rotation_at_100_percent_rad * fraction,
        )


@dataclass(frozen=True, slots=True)
class EquivalentImageLensEvent:
    key: str
    z_mm: float
    power_m1: float
    rotation_rad: float


def equivalent_image_lenses_enabled(state) -> bool:
    return (
        str(getattr(state, "projector_mode", "")).lower() == "image"
        and bool(getattr(state, "equivalent_image_lenses_enabled", False))
    )


def _trapezoid(values, coordinates) -> float:
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return float(trapezoid(values, coordinates))
    return float(np.trapz(values, coordinates))  # pragma: no cover - NumPy < 2


def _unit_field_samples(lens, sample_z_mm: float, stop_z_mm: float):
    if not hasattr(lens, "magnetic_field_t"):
        raise ValueError(f"{lens.name} has no axial magnetic-field model")
    if hasattr(lens, "field_support_mm"):
        lower, upper = lens.field_support_mm()
    else:  # pragma: no cover - current five image lenses expose support.
        width = max(abs(float(getattr(lens, "a_mm", 1.0))), 1.0)
        lower = float(lens.z_mm) - 8.0 * width
        upper = float(lens.z_mm) + 8.0 * width
    lower = max(float(sample_z_mm), float(lower))
    upper = min(float(stop_z_mm), float(upper))
    if upper <= lower:
        raise ValueError(f"{lens.name} has no post-sample field support")
    z_mm = np.linspace(lower, upper, 4001, dtype=float)
    original_percent = float(lens.percent)
    try:
        lens.percent = 100.0
        field_t = np.asarray(lens.magnetic_field_t(z_mm), dtype=float)
    finally:
        lens.percent = original_percent
    return z_mm, field_t


def _calibration_signature(state, sample_z_mm: float, stop_z_mm: float):
    lenses = {str(lens.key): lens for lens in state.lenses}
    lens_values = []
    for key in IMAGE_LENS_KEYS:
        lens = lenses[key]
        gaussian_values = []
        for attribute in ("gaussian", "upper_gaussian", "lower_gaussian"):
            for term in getattr(lens, attribute, ()):
                gaussian_values.append((
                    float(term.amplitude),
                    float(term.offset),
                    float(term.sigma),
                ))
        lens_values.append((
            id(lens),
            key,
            float(getattr(lens, "z_mm", 0.0)),
            float(getattr(lens, "b0_t", 0.0)),
            float(getattr(lens, "upper_b0_t", 0.0)),
            float(getattr(lens, "lower_b0_t", 0.0)),
            float(getattr(lens, "a_mm", 0.0)),
            float(getattr(lens, "upper_a_mm", 0.0)),
            float(getattr(lens, "lower_a_mm", 0.0)),
            int(getattr(lens, "polarity", 1)),
            tuple(gaussian_values),
        ))
    return (
        float(state.beam_voltage_kv),
        float(sample_z_mm),
        float(stop_z_mm),
        tuple(lens_values),
    )


def equivalent_image_calibrations(
    state,
    sample_z_mm: float,
    stop_z_mm: float,
) -> tuple[EquivalentImageLensCalibration, ...]:
    """Return five immutable current-to-focal-event calibrations."""

    signature = _calibration_signature(state, sample_z_mm, stop_z_mm)
    cached = getattr(state, "_equivalent_image_calibration_cache", None)
    if cached is not None and cached[0] == signature:
        return cached[1]
    lenses = {str(lens.key): lens for lens in state.lenses}
    momentum = electron_momentum(float(state.beam_voltage_kv))
    field_to_g_m1 = E / (2.0 * momentum)
    calibrations = []
    for key in IMAGE_LENS_KEYS:
        try:
            lens = lenses[key]
        except KeyError as exc:
            raise ValueError(f"Image preset is missing lens {key!r}") from exc
        if not bool(getattr(lens, "enabled", True)):
            raise ValueError(f"Image preset lens {key!r} must be enabled")
        z_mm, unit_field_t = _unit_field_samples(
            lens, sample_z_mm, stop_z_mm
        )
        z_m = z_mm * 1.0e-3
        squared_integral = _trapezoid(unit_field_t**2, z_m)
        absolute_weight = unit_field_t**2
        weight_integral = _trapezoid(absolute_weight, z_mm)
        if squared_integral <= 0.0 or weight_integral <= 0.0:
            raise ValueError(f"Image preset lens {key!r} has zero field power")
        effective_z_mm = _trapezoid(
            z_mm * absolute_weight, z_mm
        ) / weight_integral
        field_integral_t_m = _trapezoid(unit_field_t, z_m)
        calibrations.append(EquivalentImageLensCalibration(
            key=key,
            z_mm=float(effective_z_mm),
            maximum_percent=float(lens.max_percent),
            power_at_100_percent_m1=(
                field_to_g_m1**2 * squared_integral
            ),
            # d(phi)/dz = -g for the signed convention used by core.py.
            rotation_at_100_percent_rad=(
                field_to_g_m1 * field_integral_t_m
            ),
        ))
    result = tuple(sorted(calibrations, key=lambda item: item.z_mm))
    state._equivalent_image_calibration_cache = (signature, result)
    return result


def equivalent_image_events(
    state,
    start_z_mm: float,
    stop_z_mm: float,
) -> tuple[EquivalentImageLensEvent, ...]:
    """Return current image-lens events within one propagation interval."""

    if not equivalent_image_lenses_enabled(state):
        return ()
    sample_z_mm = float(state.sample.z_mm)
    if float(stop_z_mm) <= sample_z_mm:
        return ()
    lenses = {str(lens.key): lens for lens in state.lenses}
    calibration_stop_z_mm = float(stop_z_mm)
    for key in IMAGE_LENS_KEYS:
        lens = lenses[key]
        if hasattr(lens, "field_support_mm"):
            calibration_stop_z_mm = max(
                calibration_stop_z_mm,
                float(lens.field_support_mm()[1]),
            )
    calibrations = equivalent_image_calibrations(
        state, sample_z_mm, calibration_stop_z_mm
    )
    lower = float(start_z_mm)
    upper = float(stop_z_mm)
    return tuple(
        calibration.event(float(lenses[calibration.key].percent))
        for calibration in calibrations
        if lower <= calibration.z_mm <= upper
    )


def drift_matrix(distance_m: float) -> np.ndarray:
    identity = np.eye(2, dtype=float)
    return np.block([
        [identity, float(distance_m) * identity],
        [np.zeros((2, 2), dtype=float), identity],
    ])


def lens_event_matrix(power_m1: float, rotation_rad: float) -> np.ndarray:
    identity = np.eye(2, dtype=float)
    focus = np.block([
        [identity, np.zeros((2, 2), dtype=float)],
        [-float(power_m1) * identity, identity],
    ])
    cosine = math.cos(float(rotation_rad))
    sine = math.sin(float(rotation_rad))
    rotation = np.array(
        ((cosine, -sine), (sine, cosine)), dtype=float
    )
    return np.block([
        [rotation, np.zeros((2, 2), dtype=float)],
        [np.zeros((2, 2), dtype=float), rotation],
    ]) @ focus


def equivalent_image_transfer_matrix(
    calibrations: tuple[EquivalentImageLensCalibration, ...],
    percentages,
    source_z_mm: float,
    target_z_mm: float,
) -> np.ndarray:
    """Compose D(z) and calibrated L(f) events in laboratory coordinates."""

    values = np.asarray(percentages, dtype=float)
    if values.shape != (len(calibrations),):
        raise ValueError("Equivalent image-lens vector has the wrong shape")
    matrix = np.eye(4, dtype=float)
    cursor = float(source_z_mm)
    for calibration, percent in zip(calibrations, values):
        if calibration.z_mm < cursor:
            raise ValueError("Equivalent image lenses must be Z ordered")
        matrix = drift_matrix(
            (calibration.z_mm - cursor) * 1.0e-3
        ) @ matrix
        event = calibration.event(float(percent))
        matrix = lens_event_matrix(
            event.power_m1, event.rotation_rad
        ) @ matrix
        cursor = calibration.z_mm
    matrix = drift_matrix((float(target_z_mm) - cursor) * 1.0e-3) @ matrix
    return matrix
