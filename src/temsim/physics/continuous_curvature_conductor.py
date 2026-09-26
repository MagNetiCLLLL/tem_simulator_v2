"""Conductor geometry for the existing continuous-curvature particle emitter.

This adapter never constructs an emission model or edits its sampled particles.
The existing spherical emission cap is continued into the configured tangent
cone, truncated at the configured mechanical back plane. A large sphere may be
cut before its cone begins, which permits a continuous weak-curvature limit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ContinuousCurvatureConductor:
    apex_radius_nm: float
    cone_half_angle_deg: float
    shank_length_um: float
    emission_support_radius_nm: float

    def validate(self):
        values = (self.apex_radius_nm, self.cone_half_angle_deg,
                  self.shank_length_um, self.emission_support_radius_nm)
        if (not all(math.isfinite(value) for value in values)
                or self.apex_radius_nm <= 0 or self.shank_length_um <= 0
                or self.emission_support_radius_nm < 0
                or not 0 < self.cone_half_angle_deg < 90):
            raise ValueError("Continuous-curvature conductor dimensions must be finite and physical")
        angle = math.radians(self.cone_half_angle_deg)
        join_radius_nm = self.apex_radius_nm * math.cos(angle)
        if self.emission_support_radius_nm >= join_radius_nm:
            raise ValueError("Existing emitting support must remain inside the spherical cap before its tangent cone")
        support_z = float(self.surface_z_m(self.emission_support_radius_nm * 1e-9))
        if support_z <= -self.shank_length_um * 1e-6:
            raise ValueError("Existing emitting support extends beyond the conductor back plane")
        if not math.isfinite(float(self.radius_m(-self.shank_length_um * 1e-6))):
            raise ValueError("Continuous-curvature conductor is outside representable geometry")
        return self

    @property
    def curvature_nm_inv(self):
        return 1.0 / self.apex_radius_nm

    @property
    def back_z_m(self):
        return -self.shank_length_um * 1e-6

    def surface_z_m(self, radius_m):
        """Untruncated cap/cone upper surface, measured from the original apex.

        The finite conductor additionally requires z >= back_z_m. Returning
        the underlying graph lets boundary-conforming solvers intersect it
        with their existing back plane without moving emitted positions.
        """
        r = np.asarray(radius_m, dtype=float)
        if not np.all(np.isfinite(r)) or np.any(r < 0):
            raise ValueError("Conductor radii must be finite and non-negative")
        radius = self.apex_radius_nm * 1e-9
        angle = math.radians(self.cone_half_angle_deg)
        ratio = r / radius
        root = np.sqrt(np.maximum(0.0, 1.0 - ratio**2))
        # Rationalized sag is stable as curvature approaches zero.
        cap = -(r * ratio) / (1.0 + root)
        join_r = radius * math.cos(angle)
        join_z = -radius * (1.0 - math.sin(angle))
        cone = join_z - (r - join_r) / math.tan(angle)
        return np.where(r <= join_r, cap, cone)

    def radius_m(self, z_m):
        """Finite conductor radius at an axial position; zero outside its ends."""
        z = np.asarray(z_m, dtype=float)
        if not np.all(np.isfinite(z)):
            raise ValueError("Conductor axial positions must be finite")
        radius = self.apex_radius_nm * 1e-9
        angle = math.radians(self.cone_half_angle_deg)
        join_z = -radius * (1.0 - math.sin(angle))
        depth = np.maximum(0.0, -z)
        # Unlike R²-(z+R)², this retains the cap radius when |z| << R.
        cap = np.sqrt(depth) * np.sqrt(np.maximum(0.0, 2.0 * radius - depth))
        cone = radius * math.cos(angle) + (join_z - z) * math.tan(angle)
        physical = (z <= 0.0) & (z >= self.back_z_m)
        return np.where(physical, np.where(z >= join_z, cap, cone), 0.0)

    def normal_at_positions(self, positions_m):
        """Outward unit normal on the cap/cone graph, without altering positions."""
        positions = np.asarray(positions_m, dtype=float)
        if positions.shape[-1:] != (3,) or not np.all(np.isfinite(positions)):
            raise ValueError("Conductor positions must be finite xyz metres")
        radius = self.apex_radius_nm * 1e-9
        r = np.hypot(positions[..., 0], positions[..., 1])
        angle = math.radians(self.cone_half_angle_deg)
        cap = r <= radius * math.cos(angle)
        radial = np.where(cap, r / radius, math.cos(angle))
        axial = np.where(cap, np.sqrt(np.maximum(0.0, 1.0 - (r / radius)**2)),
                         math.sin(angle))
        multiplier = np.divide(radial, r, out=np.zeros_like(r), where=r > 0)
        normal = np.empty_like(positions)
        normal[..., :2] = positions[..., :2] * multiplier[..., None]
        normal[..., 2] = axial
        return normal

    def material_mask(self, positions_m):
        """Exact finite-metal membership; the apex and cap boundary are metal."""
        positions = np.asarray(positions_m, dtype=float)
        if positions.shape[-1:] != (3,) or not np.all(np.isfinite(positions)):
            raise ValueError("Conductor positions must be finite xyz metres")
        r = np.hypot(positions[..., 0], positions[..., 1])
        z = positions[..., 2]
        return (z >= self.back_z_m) & (z <= self.surface_z_m(r))

    def to_dict(self):
        return asdict(self)


def continuous_curvature_conductor(emitter):
    """Adapt the same emitter geometry without converting or resampling it."""
    from temsim.optics.electron_gun.tip_curvature import support_radius_nm, validate_curvature

    validate_curvature(emitter)
    curvature = float(emitter.curvature_nm_inv)
    if curvature == 0:
        raise ValueError("A zero-curvature emitter uses its separate planar conductor boundary")
    return ContinuousCurvatureConductor(
        apex_radius_nm=1.0 / curvature,
        cone_half_angle_deg=float(emitter.tip_cone_half_angle_deg),
        shank_length_um=float(emitter.mechanical_length_mm) * 1000.0,
        emission_support_radius_nm=float(support_radius_nm(emitter)),
    ).validate()
