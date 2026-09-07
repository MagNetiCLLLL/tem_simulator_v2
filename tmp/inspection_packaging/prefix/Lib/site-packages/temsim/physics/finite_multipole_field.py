"""Finite-length analytic multipole fields with compact soft edges."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from numpy.polynomial import polynomial

from temsim.physics.multipole_field import MultipoleField


@dataclass(frozen=True)
class SoftEdgeEnvelope:
    """Compact C6 entrance/plateau/exit field envelope.

    ``length_m`` is the complete non-zero field interval.  Entrance and exit
    transition widths lie inside that interval, leaving an exactly flat
    central plateau.  The transition is the regularised integral of
    ``t**6 * (1-t)**6`` so its first six derivatives join continuously to the
    zero and unit regions.
    """

    length_m: float
    entrance_soft_edge_m: float
    exit_soft_edge_m: float

    SMOOTHNESS_ORDER = 6
    MAX_DERIVATIVE = SMOOTHNESS_ORDER

    def __post_init__(self):
        values = (
            self.length_m,
            self.entrance_soft_edge_m,
            self.exit_soft_edge_m,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Soft-edge dimensions must be finite.")
        if self.length_m <= 0.0:
            raise ValueError("Finite field length must be positive.")
        if (
            self.entrance_soft_edge_m <= 0.0
            or self.exit_soft_edge_m <= 0.0
        ):
            raise ValueError("Entrance and exit soft edges must be positive.")
        if (
            self.entrance_soft_edge_m + self.exit_soft_edge_m
            >= self.length_m
        ):
            raise ValueError(
                "Entrance and exit soft edges must leave a central plateau."
            )

    @classmethod
    @lru_cache(maxsize=None)
    def _transition_coefficients(cls):
        order = cls.SMOOTHNESS_ORDER
        coefficients = np.zeros(2 * order + 2, dtype=float)
        normalisation = (
            math.factorial(2 * order + 1)
            / math.factorial(order) ** 2
        )
        for j in range(order + 1):
            power = order + j + 1
            coefficients[power] = (
                normalisation
                * (-1.0) ** j
                * math.comb(order, j)
                / power
            )
        return coefficients

    @classmethod
    @lru_cache(maxsize=None)
    def _transition_derivative_coefficients(cls, order):
        return polynomial.polyder(
            cls._transition_coefficients(),
            m=order,
        )

    @property
    def start_m(self):
        return -0.5 * float(self.length_m)

    @property
    def end_m(self):
        return 0.5 * float(self.length_m)

    @property
    def entrance_end_m(self):
        return self.start_m + float(self.entrance_soft_edge_m)

    @property
    def exit_start_m(self):
        return self.end_m - float(self.exit_soft_edge_m)

    @property
    def plateau_length_m(self):
        return (
            float(self.length_m)
            - float(self.entrance_soft_edge_m)
            - float(self.exit_soft_edge_m)
        )

    def derivative(self, local_s_m, order=0):
        """Return an axial derivative of the envelope in SI units."""

        if isinstance(order, bool) or not isinstance(
            order, (int, np.integer)
        ):
            raise TypeError("Envelope derivative order must be an integer.")
        order = int(order)
        if not 0 <= order <= self.MAX_DERIVATIVE:
            raise ValueError(
                f"Envelope derivatives are available from 0 through "
                f"{self.MAX_DERIVATIVE}."
            )
        s = np.asarray(local_s_m, dtype=float)
        result = np.zeros_like(s)
        coefficients = self._transition_derivative_coefficients(
            order
        )

        entrance = (s >= self.start_m) & (s <= self.entrance_end_m)
        if np.any(entrance):
            t = (
                (s[entrance] - self.start_m)
                / float(self.entrance_soft_edge_m)
            )
            result[entrance] = polynomial.polyval(t, coefficients) / (
                float(self.entrance_soft_edge_m) ** order
            )

        plateau = (s > self.entrance_end_m) & (s < self.exit_start_m)
        if order == 0:
            result[plateau] = 1.0

        exit_edge = (s >= self.exit_start_m) & (s <= self.end_m)
        if np.any(exit_edge):
            t = (
                (self.end_m - s[exit_edge])
                / float(self.exit_soft_edge_m)
            )
            result[exit_edge] = polynomial.polyval(t, coefficients) * (
                (-1.0 / float(self.exit_soft_edge_m)) ** order
            )
        return result

    def value(self, local_s_m):
        return self.derivative(local_s_m, order=0)


@dataclass
class FiniteMultipoleField:
    """Analytic finite multipole field in local ``(x, y, s)`` coordinates.

    The field is derived from a truncated three-dimensional magnetic scalar
    potential.  This makes the analytic field curl-free.  Radial correction
    terms cancel ``div(B)`` successively; the default order 2 retains terms
    through ``r**4`` and leaves only the next omitted radial order.  A future
    measured field-map backend can implement the same
    ``field_at_local_positions_t`` interface.
    """

    multipole_field: MultipoleField
    envelope: SoftEdgeEnvelope
    fringe_expansion_order: int = 2

    MAX_FRINGE_EXPANSION_ORDER = 2

    def __post_init__(self):
        if not isinstance(self.multipole_field, MultipoleField):
            raise TypeError(
                "FiniteMultipoleField requires a MultipoleField backend."
            )
        if not isinstance(self.envelope, SoftEdgeEnvelope):
            raise TypeError(
                "FiniteMultipoleField requires a SoftEdgeEnvelope."
            )
        if isinstance(self.fringe_expansion_order, bool) or not isinstance(
            self.fringe_expansion_order, (int, np.integer)
        ):
            raise TypeError("Fringe expansion order must be an integer.")
        self.fringe_expansion_order = int(self.fringe_expansion_order)
        if not (
            0
            <= self.fringe_expansion_order
            <= self.MAX_FRINGE_EXPANSION_ORDER
        ):
            raise ValueError(
                "Fringe expansion order must be between 0 and "
                f"{self.MAX_FRINGE_EXPANSION_ORDER}."
            )

    @property
    def length_m(self):
        return float(self.envelope.length_m)

    def field_at_local_positions_t(self, positions_m):
        """Return local ``[..., (B_x, B_y, B_s)]`` values in tesla."""

        positions = np.asarray(positions_m, dtype=float)
        if positions.ndim == 0 or positions.shape[-1] != 3:
            raise ValueError(
                "Local positions must have a final (x, y, s) axis."
            )
        x = positions[..., 0]
        y = positions[..., 1]
        s = positions[..., 2]
        coordinate = x + 1j * y
        radius_squared = x * x + y * y
        derivatives = [
            self.envelope.derivative(s, derivative_order)
            for derivative_order in range(
                2 * self.fringe_expansion_order + 2
            )
        ]

        bx = np.zeros_like(x)
        by = np.zeros_like(y)
        bs = np.zeros_like(s)
        for order, coefficient in enumerate(
            self.multipole_field.complex_coefficients,
            start=1,
        ):
            if coefficient == 0.0:
                continue
            base_complex_field = coefficient * coordinate ** (order - 1)
            base_bx = base_complex_field.imag
            base_by = base_complex_field.real
            harmonic = (coefficient * coordinate ** order).imag

            radial_power = np.ones_like(radius_squared)
            scalar_coefficient = 1.0
            for correction in range(
                self.fringe_expansion_order + 1
            ):
                if correction:
                    scalar_coefficient *= -1.0 / (
                        4.0
                        * correction
                        * (order + correction)
                    )
                    previous_radial_power = (
                        radius_squared ** (correction - 1)
                    )
                    gradient_x = (
                        2.0
                        * correction
                        * x
                        * previous_radial_power
                        * harmonic
                    )
                    gradient_y = (
                        2.0
                        * correction
                        * y
                        * previous_radial_power
                        * harmonic
                    )
                    radial_power = radius_squared ** correction
                else:
                    gradient_x = np.zeros_like(x)
                    gradient_y = np.zeros_like(y)

                transverse_scale = (
                    scalar_coefficient
                    * derivatives[2 * correction]
                    / order
                )
                bx += transverse_scale * (
                    gradient_x
                    + order * radial_power * base_bx
                )
                by += transverse_scale * (
                    gradient_y
                    + order * radial_power * base_by
                )
                bs += (
                    scalar_coefficient
                    * derivatives[2 * correction + 1]
                    * radial_power
                    * harmonic
                    / order
                )
        return np.stack((bx, by, bs), axis=-1)
