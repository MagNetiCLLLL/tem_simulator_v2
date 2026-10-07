"""Finite, tilted circular aperture plates on the common global-Z mesh.

The plate is opaque outside its cylindrical opening. Its actual declared
thickness is required; no display thickness or ellipse is substituted. Wave
absorption is a swept-step boundary discretisation, to be refined with the
propagation step near an edge. The identical geometry provides ray contact
fractions. It is not an oblique-plane Fourier propagation operator.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from temsim.physics.lens_field_provider import CoordinateRegistration
from temsim.physics.posed_wave_hardware import PosedWaveBore
from temsim.physics.wave_device import array_module


@dataclass(frozen=True)
class PosedWaveAperture(PosedWaveBore):
    """An absorbing circular plate, with registration centred on its hole.

    The inherited local Z coordinates retain the original global-mm origin.
    Registration includes the operating hole offset in the physically placed
    aperture frame, so body translation and hole movement are applied once.
    """

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (
                self.local_start_z_mm, self.local_end_z_mm, self.radius_mm)):
            raise ValueError("Aperture dimensions must be finite")
        if self.radius_mm < 0 or self.local_end_z_mm <= self.local_start_z_mm:
            raise ValueError("Aperture radius must be nonnegative and plate thickness positive")

    @classmethod
    def from_component(cls, state, aperture, registration):
        thickness = getattr(aperture, "plate_thickness_mm", None)
        if thickness is None:
            parts = getattr(getattr(state, "_resolved_assembly", None), "parts", ())
            part = next((p for p in parts if p.key == aperture.key), None)
            thickness = None if part is None else part.data.get("plate_thickness_mm")
        if thickness is None or not math.isfinite(float(thickness)) or float(thickness) <= 0:
            raise ValueError(f"Tilted aperture {aperture.key} requires a declared positive plate_thickness_mm; "
                             "a zero-length or schematic plate cannot define coherent absorption")
        # These installed components use a circular hole. Do not silently turn
        # an arbitrary component's custom/slit mask into a circular model.
        if callable(getattr(aperture, "transmission_mask", None)):
            from temsim.optics.condenser_aperture import ContinuousApertureComponent
            from temsim.optics.objective_aperture import ObjectiveApertureComponent
            from temsim.optics.selected_area_aperture import SelectedAreaApertureComponent
            if not isinstance(aperture, (ContinuousApertureComponent, ObjectiveApertureComponent,
                                         SelectedAreaApertureComponent)):
                raise ValueError(f"Tilted aperture {aperture.key}: its non-circular/custom opening is unsupported")
        centre, radius = float(aperture.z_mm), float(aperture.radius_mm)
        hole_offset = np.array((float(getattr(aperture, "offset_x_mm", 0.)),
                                float(getattr(aperture, "offset_y_mm", 0.)), 0.))
        if not np.all(np.isfinite(hole_offset)):
            raise ValueError("Aperture opening offsets must be finite")
        origin = registration.origin_array_m + registration.rotation_array @ (hole_offset*1e-3)
        placed_hole = CoordinateRegistration(tuple(origin), registration.rotation_local_to_global)
        return cls(str(aperture.key), centre-float(thickness)*.5,
                   centre+float(thickness)*.5, radius, placed_hole)

    @property
    def identity(self):
        return ("posed-wave-circular-aperture-v1", *super().identity[1:])

    def transmission_mask(self, x_mm, y_mm, z_mm, previous_z_mm=None, xp=None):
        if previous_z_mm is not None:
            return self.swept_transmission_mask(x_mm, y_mm, previous_z_mm, z_mm, xp=xp)
        xp = array_module(x_mm) if xp is None else xp
        x, y, z = self._local(x_mm, y_mm, z_mm)
        active = (z >= self.local_start_z_mm) & (z <= self.local_end_z_mm)
        return ~active | ((self.radius_mm > 0) & (xp.hypot(x, y) <= self.radius_mm))

    def swept_transmission_mask(self, x_mm, y_mm, start_z_mm, end_z_mm, *, xp=None):
        if not math.isfinite(start_z_mm) or not math.isfinite(end_z_mm):
            raise ValueError("Aperture marching bounds must be finite")
        xp = array_module(x_mm) if xp is None else xp
        x, y, z = self._local(x_mm, y_mm, start_z_mm)
        delta = (end_z_mm-start_z_mm)*self.registration.rotation_array[2]
        if abs(float(delta[2])) <= 1e-14:
            active = (z >= self.local_start_z_mm) & (z <= self.local_end_z_mm)
            first, last = 0., 1.
        else:
            a = (self.local_start_z_mm-z)/delta[2]
            b = (self.local_end_z_mm-z)/delta[2]
            first, last = xp.maximum(0., xp.minimum(a, b)), xp.minimum(1., xp.maximum(a, b))
            active = last >= first
        r_first = xp.hypot(x+first*delta[0], y+first*delta[1])
        r_last = xp.hypot(x+last*delta[0], y+last*delta[1])
        return ~active | ((self.radius_mm > 0) & (r_first <= self.radius_mm) & (r_last <= self.radius_mm))

    def first_contact_fraction(self, start_xyz_mm, end_xyz_mm, xp=None):
        """First opaque-material contact on arbitrary ray chords, or infinity.

        Endpoint arrays have shape (..., 3). The finite local slab is intersected
        first, then the exact cylinder exit is solved. This includes shoulders
        between saved global-Z nodes, in either propagation direction.
        """
        xp = array_module(start_xyz_mm) if xp is None else xp
        start = xp.asarray(start_xyz_mm, dtype=float)
        end = xp.asarray(end_xyz_mm, dtype=float)
        rotation = xp.asarray(self.registration.rotation_array)
        origin = xp.asarray(self.registration.origin_array_m)*1e3
        local = (start-origin) @ rotation
        delta = (end-start) @ rotation
        moving = xp.abs(delta[..., 2]) > 1e-14
        safe_dz = xp.where(moving, delta[..., 2], 1.)
        a = (self.local_start_z_mm-local[..., 2])/safe_dz
        b = (self.local_end_z_mm-local[..., 2])/safe_dz
        first = xp.where(moving, xp.maximum(0., xp.minimum(a, b)), 0.)
        last = xp.where(moving, xp.minimum(1., xp.maximum(a, b)), 1.)
        inside_slab = (local[..., 2] >= self.local_start_z_mm) & (local[..., 2] <= self.local_end_z_mm)
        active = xp.where(moving, last >= first, inside_slab)
        active &= xp.all(xp.isfinite(local), axis=-1) & xp.all(xp.isfinite(delta), axis=-1)
        entry = local[..., :2] + first[..., None]*delta[..., :2]
        exit_xy = local[..., :2] + last[..., None]*delta[..., :2]
        entry_out = xp.sum(entry*entry, axis=-1) > self.radius_mm**2
        exit_out = xp.sum(exit_xy*exit_xy, axis=-1) > self.radius_mm**2
        qa = xp.sum(delta[..., :2]**2, axis=-1)
        qb = 2*xp.sum(entry*delta[..., :2], axis=-1)
        qc = xp.sum(entry*entry, axis=-1)-self.radius_mm**2
        # Positive root, measured from entry. The other root enters the hole.
        root = (-qb+xp.sqrt(xp.maximum(0., qb*qb-4*qa*qc))) / xp.where(qa > 0, 2*qa, 1.)
        hit = xp.where(entry_out | (self.radius_mm == 0), first, first+root)
        blocked = active & (entry_out | exit_out | (self.radius_mm == 0))
        return xp.where(blocked, xp.clip(hit, first, last), xp.inf)
