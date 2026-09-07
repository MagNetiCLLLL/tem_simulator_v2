"""Ideal two-dimensional magnetic multipole fields.

This module contains the field calculation only.  It deliberately knows
nothing about pole-piece rendering, element length, fringe fields, beam
apertures, or a particular ray integrator.

The local transverse-field convention is

    B_y + i B_x = sum(C_n * (x + i y)**(n - 1)),

where ``C_n = normal_n + i * skew_n`` and ``n`` runs from 1 through 6.
Coordinates are in metres and the order-n coefficients therefore have SI
units T / m**(n - 1).  With this convention a positive normal dipole points
along +y and a positive skew dipole points along +x.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral

import numpy as np


class MultipoleField:
    """Superposition of normal and skew multipoles from order 1 to 6.

    Positive ``orientation_rad`` is an active counter-clockwise rotation of
    the physical field pattern in the local x-y plane.  Rotating order ``n``
    by an angle ``theta`` multiplies its complex coefficient by
    ``exp(-i * n * theta)``.
    """

    MIN_ORDER = 1
    MAX_ORDER = 6
    ORDER_COUNT = MAX_ORDER - MIN_ORDER + 1

    def __init__(
        self,
        normal: Mapping[int, float] | Sequence[float] | None = None,
        skew: Mapping[int, float] | Sequence[float] | None = None,
    ):
        self._normal = self._coerce_coefficients(normal, "normal")
        self._skew = self._coerce_coefficients(skew, "skew")

    @classmethod
    def _coerce_coefficients(cls, values, name):
        coefficients = np.zeros(cls.ORDER_COUNT, dtype=float)
        if values is None:
            return coefficients
        if isinstance(values, Mapping):
            for order, value in values.items():
                index = cls._order_index(order)
                coefficients[index] = float(value)
        else:
            coefficients = np.asarray(values, dtype=float)
            if coefficients.shape != (cls.ORDER_COUNT,):
                raise ValueError(
                    f"{name} coefficients must contain exactly "
                    f"{cls.ORDER_COUNT} values for orders "
                    f"{cls.MIN_ORDER} through {cls.MAX_ORDER}."
                )
            coefficients = coefficients.copy()
        if not np.all(np.isfinite(coefficients)):
            raise ValueError(f"{name} coefficients must all be finite.")
        return coefficients

    @classmethod
    def _order_index(cls, order):
        if isinstance(order, bool) or not isinstance(order, Integral):
            raise TypeError("Multipole order must be an integer.")
        order = int(order)
        if not cls.MIN_ORDER <= order <= cls.MAX_ORDER:
            raise ValueError(
                f"Multipole order must be between {cls.MIN_ORDER} "
                f"and {cls.MAX_ORDER}."
            )
        return order - cls.MIN_ORDER

    @staticmethod
    def _finite_float(value, name):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")
        return value

    @property
    def normal_coefficients(self):
        """Return a copy ordered as n=1, ..., 6."""

        return self._normal.copy()

    @property
    def skew_coefficients(self):
        """Return a copy ordered as n=1, ..., 6."""

        return self._skew.copy()

    @property
    def complex_coefficients(self):
        """Return ``normal + i*skew`` coefficients ordered as n=1, ..., 6."""

        return self._normal + 1j * self._skew

    def component(self, order):
        """Return the ``(normal, skew)`` coefficient pair for one order."""

        index = self._order_index(order)
        return float(self._normal[index]), float(self._skew[index])

    def set_component(self, order, *, normal=None, skew=None):
        """Set either or both Cartesian coefficients and return ``self``."""

        index = self._order_index(order)
        if normal is not None:
            self._normal[index] = self._finite_float(normal, "normal")
        if skew is not None:
            self._skew[index] = self._finite_float(skew, "skew")
        return self

    def set_polar_component(self, order, strength, orientation_rad=0.0):
        """Set one order from signed strength and physical orientation."""

        index = self._order_index(order)
        strength = self._finite_float(strength, "strength")
        orientation = self._finite_float(
            orientation_rad, "orientation_rad"
        )
        phase = -int(order) * orientation
        self._normal[index] = strength * math.cos(phase)
        self._skew[index] = strength * math.sin(phase)
        return self

    def component_strength(self, order):
        """Return the non-negative magnitude of one complex coefficient."""

        normal, skew = self.component(order)
        return math.hypot(normal, skew)

    def component_orientation_rad(self, order):
        """Return the principal physical orientation of one field order.

        A zero-strength component has no physical orientation; this method
        returns zero for that case.
        """

        normal, skew = self.component(order)
        if normal == 0.0 and skew == 0.0:
            return 0.0
        return -math.atan2(skew, normal) / int(order)

    def rotate_component(self, order, angle_rad):
        """Actively rotate one multipole pattern and return ``self``."""

        index = self._order_index(order)
        angle = self._finite_float(angle_rad, "angle_rad")
        coefficient = complex(
            self._normal[index], self._skew[index]
        ) * np.exp(-1j * int(order) * angle)
        self._normal[index] = coefficient.real
        self._skew[index] = coefficient.imag
        return self

    def complex_transverse_field_t(self, x_m, y_m):
        """Return ``B_y + i*B_x`` in tesla at local transverse coordinates."""

        x, y = np.broadcast_arrays(
            np.asarray(x_m, dtype=float),
            np.asarray(y_m, dtype=float),
        )
        coordinate = x + 1j * y
        field = np.zeros(coordinate.shape, dtype=complex)
        power = np.ones(coordinate.shape, dtype=complex)
        for coefficient in self.complex_coefficients:
            field += coefficient * power
            power *= coordinate
        return field

    def field_t(self, x_m, y_m):
        """Return local ``[..., (B_x, B_y, B_z)]`` values in tesla.

        This generic transverse model has ``B_z = 0``.  The finite-length
        element added in the next implementation stage will own its axial
        envelope and Maxwell-consistent fringe-field completion.
        """

        transverse = self.complex_transverse_field_t(x_m, y_m)
        zeros = np.zeros(transverse.shape, dtype=float)
        return np.stack(
            (transverse.imag, transverse.real, zeros),
            axis=-1,
        )

    def field_at_local_positions_t(self, positions_m):
        """Evaluate the ideal field at local ``[..., (x, y, s)]`` points.

        The ideal cross-section is independent of ``s``.  This common
        position-vector interface lets a finite analytic backend and a future
        measured field-map backend be exchanged without changing callers.
        """

        positions = np.asarray(positions_m, dtype=float)
        if positions.ndim == 0 or positions.shape[-1] != 3:
            raise ValueError(
                "Local positions must have a final (x, y, s) axis."
            )
        return self.field_t(positions[..., 0], positions[..., 1])

    def component_field_t(self, order, x_m, y_m):
        """Return the field of one selected order without other components."""

        index = self._order_index(order)
        x, y = np.broadcast_arrays(
            np.asarray(x_m, dtype=float),
            np.asarray(y_m, dtype=float),
        )
        coordinate = x + 1j * y
        coefficient = complex(
            self._normal[index], self._skew[index]
        )
        transverse = coefficient * coordinate ** (int(order) - 1)
        zeros = np.zeros(transverse.shape, dtype=float)
        return np.stack(
            (transverse.imag, transverse.real, zeros),
            axis=-1,
        )
