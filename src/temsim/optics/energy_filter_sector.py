"""Finite sector-magnet geometry and Energy Filter local frames."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

from temsim.optics.twelve_pole_element import LocalCoordinateFrame
from temsim.physics.finite_multipole_field import SoftEdgeEnvelope
from temsim.optics.energy_filter_m12 import magnetic_rigidity_t_m
from temsim.physics.relativistic_lorentz import (
    RelativisticPhaseSpace,
    boris_step,
    momentum_from_kinetic_energy_ev,
    velocity_from_momentum_m_per_s,
)


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    magnitude = float(np.linalg.norm(vector))
    if magnitude <= 0.0:
        raise ValueError("Coordinate-frame axis must be non-zero.")
    return vector / magnitude


def beam_frame(origin_m, tangent):
    """Build a right-handed frame with local s along ``tangent``.

    Global Y is the non-dispersive direction. Local X is chosen so
    ``local_x cross local_y = local_s``.
    """

    local_s = _unit(tangent)
    local_y = np.array([0.0, 1.0, 0.0])
    local_x = _unit(np.cross(local_y, local_s))
    rotation = np.column_stack((local_x, local_y, local_s))
    return LocalCoordinateFrame(
        origin_m=np.asarray(origin_m, dtype=float),
        rotation_local_to_global=rotation,
    )


@dataclass
class SectorMagnetElement:
    """Uniform-sector body with compact C6 entrance and exit fringes."""

    entrance_point_m: np.ndarray
    radius_m: float
    bend_angle_rad: float
    plateau_field_t: float
    fringe_length_m: float
    pole_gap_m: float
    radial_aperture_m: float
    soft_edges_enabled: bool = False
    enabled: bool = True

    def __post_init__(self):
        self.entrance_point_m = np.asarray(
            self.entrance_point_m,
            dtype=float,
        ).copy()
        if self.entrance_point_m.shape != (3,):
            raise ValueError("Sector entrance must be one three-vector.")
        dimensions = (
            self.radius_m,
            self.bend_angle_rad,
            self.plateau_field_t,
            self.fringe_length_m,
            self.pole_gap_m,
            self.radial_aperture_m,
        )
        if not all(math.isfinite(float(value)) for value in dimensions):
            raise ValueError("Sector parameters must be finite.")
        if self.radius_m <= 0.0 or self.bend_angle_rad <= 0.0:
            raise ValueError("Sector radius and bend angle must be positive.")
        if self.fringe_length_m <= 0.0:
            raise ValueError("Sector fringe length must be positive.")
        if 2.0 * self.fringe_length_m >= self.arc_length_m:
            raise ValueError("Sector fringes must leave a central plateau.")
        if self.pole_gap_m <= 0.0 or self.radial_aperture_m <= 0.0:
            raise ValueError("Sector apertures must be positive.")
        self._envelope = SoftEdgeEnvelope(
            self.arc_length_m,
            self.fringe_length_m,
            self.fringe_length_m,
        )

    @property
    def arc_length_m(self):
        return float(self.radius_m) * float(self.bend_angle_rad)

    @property
    def effective_field_length_m(self):
        # Symmetric smoothstep transitions integrate to half their width.
        return self.arc_length_m - self.fringe_length_m

    @property
    def centre_m(self):
        return self.entrance_point_m + np.array(
            [0.0, 0.0, -self.radius_m]
        )

    @property
    def exit_point_m(self):
        angle = float(self.bend_angle_rad)
        return self.entrance_point_m + np.array(
            [
                self.radius_m * math.sin(angle),
                0.0,
                -self.radius_m * (1.0 - math.cos(angle)),
            ]
        )

    @property
    def entrance_tangent(self):
        return np.array([1.0, 0.0, 0.0])

    @property
    def exit_tangent(self):
        angle = float(self.bend_angle_rad)
        return np.array(
            [math.cos(angle), 0.0, -math.sin(angle)]
        )

    @property
    def entrance_frame(self):
        return beam_frame(self.entrance_point_m, self.entrance_tangent)

    @property
    def exit_frame(self):
        return beam_frame(self.exit_point_m, self.exit_tangent)

    def _cylindrical_coordinates(self, positions_m):
        positions = np.asarray(positions_m, dtype=float)
        if positions.ndim == 0 or positions.shape[-1] != 3:
            raise ValueError(
                "Sector positions must have a final three-vector axis."
            )
        relative = positions - self.centre_m
        radial_x = relative[..., 0]
        radial_z = relative[..., 2]
        radius = np.hypot(radial_x, radial_z)
        angle = np.arctan2(radial_x, radial_z)
        arc_coordinate = self.radius_m * angle
        return positions, radius, angle, arc_coordinate

    def envelope_value(self, positions_m):
        _, _, _, arc_coordinate = self._cylindrical_coordinates(
            positions_m
        )
        centred_s = arc_coordinate - 0.5 * self.arc_length_m
        return self._envelope.value(centred_s)

    def field_at_global_positions_t(self, positions_m):
        positions, radius, angle, _ = self._cylindrical_coordinates(
            positions_m
        )
        field = np.zeros_like(positions)
        if not self.enabled:
            return field
        inside_angle = (angle >= 0.0) & (
            angle <= self.bend_angle_rad
        )
        inside_radial_map = (
            np.abs(radius - self.radius_m)
            <= 1.5 * self.radial_aperture_m
        )
        envelope = (
            self.envelope_value(positions)
            if self.soft_edges_enabled
            else np.ones_like(radius)
        )
        field[..., 1] = (
            float(self.plateau_field_t)
            * envelope
            * inside_angle
            * inside_radial_map
        )
        return field

    def aperture_blocked_mask(self, positions_m):
        positions, radius, angle, _ = self._cylindrical_coordinates(
            positions_m
        )
        within_body = (angle >= 0.0) & (
            angle <= self.bend_angle_rad
        )
        outside_gap = (
            np.abs(positions[..., 1] - self.entrance_point_m[1])
            > 0.5 * self.pole_gap_m
        )
        outside_radial = (
            np.abs(radius - self.radius_m)
            > self.radial_aperture_m
        )
        return within_body & (outside_gap | outside_radial)


def sector_plateau_field_t(
    voltage_kv,
    radius_m,
    bend_angle_rad,
    fringe_length_m,
):
    """Return the soft-edge field matched to the sector reference orbit.

    The C6 fringe changes the effective magnetic length, so ``B rho / R``
    is only the hard-edge starting value.  The geometry-only correction is
    obtained once with the same relativistic Boris integration used by the
    production tracer and then cached.
    """

    scale = _soft_edge_reference_scale(
        round(float(radius_m), 12),
        round(float(bend_angle_rad), 12),
        round(float(fringe_length_m), 12),
    )
    return (
        magnetic_rigidity_t_m(voltage_kv)
        / float(radius_m)
        * scale
    )


@lru_cache(maxsize=64)
def _soft_edge_reference_scale(
    radius_m,
    bend_angle_rad,
    fringe_length_m,
):
    """Numerically match the central-ray exit direction for one geometry."""

    reference_voltage_kv = 300.0
    base_field_t = (
        magnetic_rigidity_t_m(reference_voltage_kv) / radius_m
    )
    entrance = np.zeros(3)
    step_m = min(0.00025, fringe_length_m / 40.0)
    step_m = max(step_m, 0.00002)

    def exit_angle(scale):
        sector = SectorMagnetElement(
            entrance_point_m=entrance,
            radius_m=radius_m,
            bend_angle_rad=bend_angle_rad,
            plateau_field_t=base_field_t * scale,
            fringe_length_m=fringe_length_m,
            pole_gap_m=max(0.03, radius_m),
            radial_aperture_m=max(0.03, 0.25 * radius_m),
            soft_edges_enabled=True,
        )
        momentum = momentum_from_kinetic_energy_ev(
            np.array([reference_voltage_kv * 1000.0]),
            np.array([[1.0, 0.0, 0.0]]),
        )
        phase = RelativisticPhaseSpace(
            np.array([[0.0, 0.0, 0.0]]),
            momentum,
        )
        speed = float(
            np.linalg.norm(
                velocity_from_momentum_m_per_s(momentum)[0]
            )
        )
        dt = step_m / speed
        path_limit_m = (
            radius_m * bend_angle_rad
            + 4.0 * fringe_length_m
        )
        for _ in range(int(math.ceil(path_limit_m / step_m))):
            phase = boris_step(phase, dt, sector)
        direction = velocity_from_momentum_m_per_s(
            phase.momentum_kg_m_per_s
        )[0]
        return math.atan2(-direction[2], direction[0])

    lower = 0.75
    upper = 1.35
    lower_error = exit_angle(lower) - bend_angle_rad
    upper_error = exit_angle(upper) - bend_angle_rad
    if lower_error * upper_error > 0.0:
        raise RuntimeError(
            "Could not bracket the Energy Filter sector-field match."
        )
    for _ in range(36):
        midpoint = 0.5 * (lower + upper)
        error = exit_angle(midpoint) - bend_angle_rad
        if lower_error * error <= 0.0:
            upper = midpoint
            upper_error = error
        else:
            lower = midpoint
            lower_error = error
    return 0.5 * (lower + upper)


def sector_from_energy_filter(energy_filter):
    return SectorMagnetElement(
        entrance_point_m=np.array(
            [
                float(energy_filter.prism_entrance_s_mm) * 1.0e-3,
                0.0,
                0.0,
            ]
        ),
        radius_m=float(energy_filter.prism_radius_mm) * 1.0e-3,
        bend_angle_rad=math.radians(
            float(energy_filter.bend_angle_deg)
        ),
        plateau_field_t=float(energy_filter.sector_field_t),
        fringe_length_m=float(energy_filter.prism_fringe_mm) * 1.0e-3,
        pole_gap_m=float(energy_filter.pole_gap_mm) * 1.0e-3,
        radial_aperture_m=(
            float(energy_filter.sector_radial_aperture_mm) * 1.0e-3
        ),
        soft_edges_enabled=bool(
            energy_filter.sector_soft_edges_enabled
        ),
    )


def place_m12_in_sector_frames(energy_filter):
    """Place both independent M12s without coupling their settings."""

    sector = sector_from_energy_filter(energy_filter)
    entrance_origin = np.array(
        [
            float(energy_filter.entrance_multipole_s_mm) * 1.0e-3,
            0.0,
            0.0,
        ]
    )
    exit_origin = (
        sector.exit_point_m
        + sector.exit_tangent
        * float(energy_filter.exit_multipole_d_mm)
        * 1.0e-3
    )
    energy_filter.entrance_m12.frame = beam_frame(
        entrance_origin,
        sector.entrance_tangent,
    )
    energy_filter.exit_m12.frame = beam_frame(
        exit_origin,
        sector.exit_tangent,
    )
    energy_filter.m12_frames_placed = True
    return energy_filter.entrance_m12, energy_filter.exit_m12


@dataclass
class EnergyFilterMagneticField:
    """Superposed sector, Entrance M12 and Exit M12 fields."""

    sector: SectorMagnetElement
    entrance_m12: object
    exit_m12: object

    def field_at_global_positions_t(self, positions_m):
        return (
            self.sector.field_at_global_positions_t(positions_m)
            + self.entrance_m12.field_at_global_positions_t(positions_m)
            + self.exit_m12.field_at_global_positions_t(positions_m)
        )

    def component_fields_t(self, positions_m):
        return {
            "sector": self.sector.field_at_global_positions_t(
                positions_m
            ),
            "entrance_m12": (
                self.entrance_m12.field_at_global_positions_t(
                    positions_m
                )
            ),
            "exit_m12": self.exit_m12.field_at_global_positions_t(
                positions_m
            ),
        }


def magnetic_field_from_energy_filter(energy_filter):
    return EnergyFilterMagneticField(
        sector=sector_from_energy_filter(energy_filter),
        entrance_m12=energy_filter.entrance_m12,
        exit_m12=energy_filter.exit_m12,
    )
