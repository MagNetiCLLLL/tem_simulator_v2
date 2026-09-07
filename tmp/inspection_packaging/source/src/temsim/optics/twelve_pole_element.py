"""Mechanical carrier and coordinate transform for a twelve-pole field."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Protocol

import numpy as np


class LocalMagneticFieldBackend(Protocol):
    """Interchangeable analytic or field-map backend interface."""

    @property
    def length_m(self) -> float:
        ...

    def field_at_local_positions_t(self, positions_m):
        ...


@dataclass(frozen=True)
class LocalCoordinateFrame:
    """Rigid right-handed transform from local coordinates to global."""

    origin_m: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )
    rotation_local_to_global: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=float)
    )

    def __post_init__(self):
        origin = np.asarray(self.origin_m, dtype=float)
        rotation = np.asarray(
            self.rotation_local_to_global,
            dtype=float,
        )
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ValueError("Local-frame origin must be one finite 3-vector.")
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise ValueError(
                "Local-frame rotation must be one finite 3x3 matrix."
            )
        if not np.allclose(
            rotation.T @ rotation,
            np.eye(3),
            atol=1.0e-12,
            rtol=0.0,
        ) or not math.isclose(
            float(np.linalg.det(rotation)),
            1.0,
            abs_tol=1.0e-12,
            rel_tol=0.0,
        ):
            raise ValueError(
                "Local-frame rotation must be proper and orthonormal."
            )
        origin = origin.copy()
        rotation = rotation.copy()
        origin.setflags(write=False)
        rotation.setflags(write=False)
        object.__setattr__(self, "origin_m", origin)
        object.__setattr__(self, "rotation_local_to_global", rotation)

    def points_to_local_m(self, global_positions_m):
        positions = np.asarray(global_positions_m, dtype=float)
        if positions.ndim == 0 or positions.shape[-1] != 3:
            raise ValueError(
                "Global positions must have a final three-vector axis."
            )
        return (
            positions - self.origin_m
        ) @ self.rotation_local_to_global

    def points_to_global_m(self, local_positions_m):
        positions = np.asarray(local_positions_m, dtype=float)
        if positions.ndim == 0 or positions.shape[-1] != 3:
            raise ValueError(
                "Local positions must have a final three-vector axis."
            )
        return (
            positions @ self.rotation_local_to_global.T
            + self.origin_m
        )

    def vectors_to_global(self, local_vectors):
        vectors = np.asarray(local_vectors, dtype=float)
        if vectors.ndim == 0 or vectors.shape[-1] != 3:
            raise ValueError(
                "Local vectors must have a final three-vector axis."
            )
        return vectors @ self.rotation_local_to_global.T
@dataclass
class TwelvePoleElement:
    """Finite physical carrier for one interchangeable magnetic field."""

    name: str
    key: str
    field_backend: LocalMagneticFieldBackend
    frame: LocalCoordinateFrame = field(
        default_factory=LocalCoordinateFrame
    )
    bore_radius_m: float = 5.0e-3
    outer_radius_m: float = 25.0e-3
    pole_zero_angle_rad: float = 0.0
    enabled: bool = True

    POLE_COUNT = 12
    KIND = "twelve_pole"
    INTERACTION_KIND = "finite_magnetic_field"

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Twelve-pole name must not be empty.")
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError("Twelve-pole key must not be empty.")
        if (
            not hasattr(self.field_backend, "length_m")
            or not callable(
                getattr(
                    self.field_backend,
                    "field_at_local_positions_t",
                    None,
                )
            )
        ):
            raise TypeError(
                "TwelvePoleElement requires a local magnetic-field backend."
            )
        if not isinstance(self.frame, LocalCoordinateFrame):
            raise TypeError(
                "TwelvePoleElement requires a LocalCoordinateFrame."
            )
        dimensions = (
            self.bore_radius_m,
            self.outer_radius_m,
            self.pole_zero_angle_rad,
            self.field_backend.length_m,
        )
        if not all(math.isfinite(float(value)) for value in dimensions):
            raise ValueError("Twelve-pole dimensions must be finite.")
        if not 0.0 < self.bore_radius_m < self.outer_radius_m:
            raise ValueError(
                "Twelve-pole bore radius must fit inside its outer radius."
            )
        if self.field_backend.length_m <= 0.0:
            raise ValueError("Twelve-pole field length must be positive.")

    @property
    def length_m(self):
        return float(self.field_backend.length_m)

    @property
    def pole_angles_rad(self):
        return (
            float(self.pole_zero_angle_rad)
            + 2.0
            * math.pi
            * np.arange(self.POLE_COUNT, dtype=float)
            / self.POLE_COUNT
        )

    def local_positions_m(self, global_positions_m):
        return self.frame.points_to_local_m(global_positions_m)

    def local_field_t(self, local_positions_m):
        positions = np.asarray(local_positions_m, dtype=float)
        if not self.enabled:
            if positions.ndim == 0 or positions.shape[-1] != 3:
                raise ValueError(
                    "Local positions must have a final three-vector axis."
                )
            return np.zeros_like(positions)
        return self.field_backend.field_at_local_positions_t(positions)

    def field_at_global_positions_t(self, global_positions_m):
        local_positions = self.local_positions_m(global_positions_m)
        local_field = self.local_field_t(local_positions)
        return self.frame.vectors_to_global(local_field)
