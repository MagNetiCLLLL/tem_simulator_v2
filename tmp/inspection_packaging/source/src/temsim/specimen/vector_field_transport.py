"""Specimen-local SI transport using the column's registered vector fields.

Positions exposed here are specimen-local nm; the Lorentz integrator operates
in local metres to avoid subtracting metre-scale column coordinates afterwards.
Static magnetic fields do no work. Collisions remain the responsibility of the
material transport kernel.
"""

from __future__ import annotations

import math
import numpy as np

from temsim.component_keys import CONDENSER_LENS_KEYS
from temsim.physics.lens_field_provider import resolve_runtime_lens_field_provider
from temsim.physics.relativistic_lorentz import (
    ELECTRON, RelativisticPhaseSpace, boris_step,
    momentum_from_kinetic_energy_ev, velocity_from_momentum_m_per_s,
)


class SpecimenFieldTransport:
    """One resolved field context per calculation, never per collision."""

    def __init__(self, state):
        from temsim.physics.core import electron
        self.state = state
        charge, momentum, _ = electron(state)
        self.reference_momentum_over_charge = momentum / charge
        self.origin_m = np.array((0.0, 0.0, float(state.sample.z_mm) * 1e-3))
        self.providers = tuple(
            resolve_runtime_lens_field_provider(
                state, lens.key,
                state.condenser_system[lens.key]
                if lens.key in CONDENSER_LENS_KEYS else lens,
            )
            for lens in state.lenses if lens.enabled
        )
        # This context is owned by one immutable calculation. Support geometry
        # cannot change between collisions, so do not reconstruct it per query.
        self._provider_supports = tuple(
            (provider, *provider.field_support_mm()) for provider in self.providers
        )
        # Resolve spatial variation of every imported grid, including rotated
        # grids. Analytic fields are resolved relative to their axial support.
        scales = []
        for provider, low, high in self._provider_supports:
            field_map = getattr(provider, "field_map", None)
            if field_map is not None:
                scales.extend(float(np.min(np.diff(axis))) * 1e9 / 4
                              for axis in field_map.axes_m)
            else:
                scales.append(max((high - low) * 1e6 / 128, 1e-6))
        self.spatial_step_nm = min(scales, default=1e6)

    def field_at_global_positions_t(self, local_positions_m):
        positions = np.asarray(local_positions_m, dtype=float) + self.origin_m
        total = np.zeros_like(positions)
        for provider, low, high in self._provider_supports:
            if np.all((positions[..., 2] * 1e3 < low) | (positions[..., 2] * 1e3 > high)):
                continue
            total += provider.field_at_global_positions_t(positions)
        if hasattr(self, "state"):
            from temsim.physics.core import multipole_focusing_fields, hexapole_field_components
            z = positions[..., 2] * 1e3
            kx, ky = multipole_focusing_fields(z, self.state)
            hn, hs = hexapole_field_components(z, self.state)
            x, y = positions[..., 0], positions[..., 1]
            u, v = x*x-y*y, 2*x*y
            total[..., 0] += self.reference_momentum_over_charge * (-ky*y + hn*v - hs*u)
            total[..., 1] += self.reference_momentum_over_charge * (kx*x + hn*u + hs*v)
        return total

    def advance(self, position_nm, direction, path_length_nm, *, energy_ev):
        position = np.asarray(position_nm, dtype=float) * 1e-9
        momentum = momentum_from_kinetic_energy_ev(energy_ev, direction)
        speed = float(np.linalg.norm(velocity_from_momentum_m_per_s(momentum)))
        length = float(path_length_nm)
        if not math.isfinite(length):
            raise ValueError("Magnetic flight length must be finite")
        phase = RelativisticPhaseSpace(position, momentum)
        remaining = abs(length)
        sign = math.copysign(1.0, length)
        for _ in range(100000):
            if remaining <= 1e-12:
                return phase.position_m * 1e9, phase.momentum_kg_m_per_s / np.linalg.norm(phase.momentum_kg_m_per_s)
            field = float(np.linalg.norm(self.field_at_global_positions_t(phase.position_m)))
            curvature = abs(ELECTRON.charge_c) * field / np.linalg.norm(momentum) * 1e-9
            step = min(remaining, self.spatial_step_nm, 1e-3 / max(curvature, 1e-30))
            phase = boris_step(phase, sign * step * 1e-9 / speed, self)
            remaining -= step
        raise ValueError("Specimen field transport exceeded its step budget")

    def to_plane(self, position_nm, direction, target_z_nm, *, energy_ev):
        """Match a Z plane by Newton shooting, including changing longitudinal velocity."""
        position = np.asarray(position_nm, dtype=float)
        unit = np.asarray(direction, dtype=float)
        unit = unit / np.linalg.norm(unit)
        if abs(unit[2]) < 1e-10:
            raise ValueError("Cannot match a reference plane with a grazing ray")
        length = (float(target_z_nm) - position[2]) / unit[2]
        for _ in range(16):
            endpoint, final = self.advance(position, unit, length, energy_ev=energy_ev)
            residual = endpoint[2] - target_z_nm
            if abs(residual) < 1e-7:
                return endpoint, final
            if abs(final[2]) < 1e-10:
                break
            length -= residual / final[2]
        raise ValueError("Magnetic trajectory did not converge to the reference plane")

    def plane_polyline(self, position_nm, direction, target_z_nm, *, energy_ev, point_count=9):
        planes = np.linspace(float(position_nm[2]), float(target_z_nm), max(point_count, 2))
        values = [self.to_plane(position_nm, direction, z, energy_ev=energy_ev) for z in planes]
        return np.asarray([value[0] for value in values]), values[-1][1]

    def boundary_distance(self, geometry, position, direction, region, *, maximum_distance_nm, energy_ev):
        """Intersect short magnetic arcs with the authoritative material geometry.

        Each chord's sagitta is limited to one quarter of the geometry epsilon.
        A tangent intercept only bounds the trial step: the magnetic arc and
        its chord are still evaluated before accepting a material crossing.
        No tangent intercept does NOT imply that a curved ray misses matter.
        """
        travelled = 0.0
        point, unit = np.asarray(position), np.asarray(direction)
        bounds = geometry.material_z_bounds_nm
        if bounds is None:
            return None
        momentum = float(np.linalg.norm(momentum_from_kinetic_energy_ev(energy_ev, unit)))
        for _ in range(100000):
            if region is None and (
                (point[2] > bounds[1] and unit[2] > 0)
                or (point[2] < bounds[0] and unit[2] < 0)
            ):
                # Beyond the finite specimen envelope, the downstream column
                # owns the trajectory, including any later magnetic turning.
                return None
            remaining = maximum_distance_nm - travelled
            if remaining <= 1e-12:
                return None
            field = float(np.linalg.norm(np.cross(unit, self.field_at_global_positions_t(point * 1e-9))))
            curvature = abs(ELECTRON.charge_c) * field / momentum * 1e-9
            step = min(remaining, self.spatial_step_nm,
                       math.sqrt(2 * geometry.epsilon_nm / max(curvature, 1e-30)))
            tangent_hit = geometry.next_region_boundary_distance_nm(
                point, unit, region, maximum_distance_nm=step,
            )
            if tangent_hit is not None:
                # Do not integrate micrometres just to find a boundary a few
                # picometres away. A small overshoot avoids asymptotic steps
                # on approach; the original spatial/sagitta limits still win.
                step = min(step, tangent_hit + max(
                    8.0 * geometry.epsilon_nm, 0.01 * tangent_hit,
                ))
            endpoint, final = self.advance(point, unit, step, energy_ev=energy_ev)
            chord = endpoint - point
            norm = float(np.linalg.norm(chord))
            if norm <= 0:
                raise ValueError("Magnetic boundary search made no progress")
            hit = geometry.next_region_boundary_distance_nm(
                point, chord / norm, region, maximum_distance_nm=norm,
            )
            if hit is not None:
                return travelled + step * hit / norm
            travelled += step
            point, unit = endpoint, final
        raise ValueError("Magnetic boundary search exceeded its step budget")
