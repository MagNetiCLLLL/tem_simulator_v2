"""Forward point-spread models for physical detector readout planes."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.signal import fftconvolve


POINT_SPREAD_MODELS = frozenset({"none", "gaussian"})
POINT_SPREAD_STATUSES = frozenset({
    "manufacturer_documented",
    "measured_calibration",
    "provisional_model_parameter",
})


@dataclass(frozen=True)
class DetectorPointSpread:
    """A detector-plane PSF expressed in physical millimetres.

    ``sigma_x_mm`` and ``sigma_y_mm`` are one-standard-deviation widths along
    the PSF principal axes.  ``rotation_deg`` rotates the first principal axis
    counter-clockwise from detector +X.  This model is deliberately separate
    from electron-optical aberrations and the objective-lens CTF.
    """

    model: str
    sigma_x_mm: float
    sigma_y_mm: float
    rotation_deg: float
    status: str
    source: str

    @classmethod
    def from_component(cls, component) -> "DetectorPointSpread":
        return cls(
            model=str(component.point_spread_model),
            sigma_x_mm=float(component.point_spread_sigma_x_mm),
            sigma_y_mm=float(component.point_spread_sigma_y_mm),
            rotation_deg=float(component.point_spread_rotation_deg),
            status=str(component.point_spread_status),
            source=str(component.point_spread_source),
        ).validated()

    def validated(self) -> "DetectorPointSpread":
        model = self.model.strip().lower()
        if model not in POINT_SPREAD_MODELS:
            raise ValueError(
                f"Point-spread model must be one of {sorted(POINT_SPREAD_MODELS)}."
            )
        values = (self.sigma_x_mm, self.sigma_y_mm, self.rotation_deg)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Point-spread widths and rotation must be finite.")
        if self.sigma_x_mm < 0.0 or self.sigma_y_mm < 0.0:
            raise ValueError("Point-spread sigma values cannot be negative.")
        if model == "gaussian" and (
            self.sigma_x_mm <= 0.0 or self.sigma_y_mm <= 0.0
        ):
            raise ValueError(
                "Gaussian point spread requires positive sigma_x_mm and sigma_y_mm."
            )
        if self.status not in POINT_SPREAD_STATUSES:
            raise ValueError(
                "Point-spread status must be one of "
                f"{sorted(POINT_SPREAD_STATUSES)}."
            )
        if not self.source.strip():
            raise ValueError("Point-spread source must not be empty.")
        return self

    @property
    def enabled(self) -> bool:
        return self.model.strip().lower() != "none"


def validate_component_point_spread(component) -> DetectorPointSpread:
    """Validate and return the TOML-backed PSF attached to a component."""

    return DetectorPointSpread.from_component(component)


def gaussian_kernel(
    point_spread: DetectorPointSpread,
    *,
    pixel_size_x_mm: float,
    pixel_size_y_mm: float,
    maximum_half_pixels: int | None = None,
) -> np.ndarray:
    """Return a unit-sum sampled Gaussian kernel on a detector image grid."""

    point_spread.validated()
    dx = float(pixel_size_x_mm)
    dy = float(pixel_size_y_mm)
    if not math.isfinite(dx) or not math.isfinite(dy) or dx <= 0.0 or dy <= 0.0:
        raise ValueError("Detector-image pixel sizes must be finite and positive.")
    if not point_spread.enabled:
        return np.ones((1, 1), dtype=float)

    sigma_max = max(point_spread.sigma_x_mm, point_spread.sigma_y_mm)
    half_x = max(1, int(math.ceil(4.0 * sigma_max / dx)))
    half_y = max(1, int(math.ceil(4.0 * sigma_max / dy)))
    if maximum_half_pixels is not None:
        limit = max(1, int(maximum_half_pixels))
        half_x = min(half_x, limit)
        half_y = min(half_y, limit)
    x = np.arange(-half_x, half_x + 1, dtype=float) * dx
    y = np.arange(-half_y, half_y + 1, dtype=float) * dy
    xx, yy = np.meshgrid(x, y)
    angle = math.radians(point_spread.rotation_deg)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    principal_x = cosine * xx + sine * yy
    principal_y = -sine * xx + cosine * yy
    kernel = np.exp(-0.5 * (
        (principal_x / point_spread.sigma_x_mm) ** 2
        + (principal_y / point_spread.sigma_y_mm) ** 2
    ))
    total = float(kernel.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("Point-spread kernel has no finite positive support.")
    return kernel / total


def apply_point_spread(
    intensity,
    point_spread: DetectorPointSpread,
    *,
    pixel_size_x_mm: float,
    pixel_size_y_mm: float,
) -> np.ndarray:
    """Forward-convolve a non-negative detector image with zero padding.

    Zero padding represents response that leaves the finite rendered field;
    callers can apply the detector active-area mask and report the retained
    weight.  No inverse filtering or deconvolution is performed.
    """

    values = np.asarray(intensity, dtype=float)
    if values.ndim != 2:
        raise ValueError("Point spread requires a two-dimensional image.")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Detector intensity must be finite and non-negative.")
    point_spread.validated()
    if not point_spread.enabled or values.size == 0:
        return values.copy()
    kernel = gaussian_kernel(
        point_spread,
        pixel_size_x_mm=pixel_size_x_mm,
        pixel_size_y_mm=pixel_size_y_mm,
        maximum_half_pixels=2 * max(values.shape),
    )
    response = fftconvolve(values, kernel, mode="same")
    # FFT round-off can produce tiny negative values around exact zeros.
    return np.maximum(np.asarray(response, dtype=float), 0.0)
