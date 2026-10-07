"""Absorbing bore geometry shared with the ray solver on global Z slices.

Coordinates supplied to masks are in mm; affine wave-grid inputs use metres.
The material model is the existing ray law: a finite local axial slab, opaque
everywhere outside its circular bore. No finite outer radius is invented.

A swept mask checks a whole marching interval, not just its endpoints. It is
exact for straight global-Z characteristics and catches oblique entrance
shoulders between slices. Applied between wave propagation steps, it is an
absorbing-boundary discretisation whose diffraction accuracy still requires
step refinement. It is not propagation on a rotated wave observation plane.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from temsim.physics.lens_field_provider import CoordinateRegistration
from temsim.physics.wave_device import array_module


def _grid_corners_mm(shape, basis_m, origin_m):
    ny, nx = shape
    if ny <= 0 or nx <= 0:
        raise ValueError("Wave grid dimensions must be positive")
    basis, origin = np.asarray(basis_m, dtype=float), np.asarray(origin_m, dtype=float)
    if (basis.shape != (2, 2) or origin.shape != (2,)
            or not np.all(np.isfinite(basis)) or not np.all(np.isfinite(origin))):
        raise ValueError("Wave grid must have a finite 2-D basis and origin")
    corners = np.array(((-(nx//2), -(ny//2)), (nx-1-nx//2, -(ny//2)),
                        (nx-1-nx//2, ny-1-ny//2), (-(nx//2), ny-1-ny//2)))
    return (origin[:, None] + basis @ corners.T).T * 1e3


@dataclass(frozen=True)
class PosedWaveBore:
    """An installed bore in its original local frame and shared rigid pose."""

    key: str
    local_start_z_mm: float
    local_end_z_mm: float
    radius_mm: float
    registration: CoordinateRegistration

    def __post_init__(self):
        values = (self.local_start_z_mm, self.local_end_z_mm, self.radius_mm)
        if not all(math.isfinite(v) for v in values) or self.radius_mm <= 0:
            raise ValueError("Bore dimensions must be finite with positive radius")
        if self.local_end_z_mm <= self.local_start_z_mm:
            raise ValueError("Bore axial extent must be positive")

    @classmethod
    def from_segment(cls, segment, registration):
        return cls(str(getattr(segment, "key", "column_wall")),
                   float(segment.start_z_mm), float(segment.end_z_mm),
                   .5*float(segment.inner_diameter_mm), registration)

    @property
    def identity(self):
        return ("posed-wave-bore-v1", self.key, self.local_start_z_mm,
                self.local_end_z_mm, self.radius_mm,
                self.registration.origin_global_m,
                self.registration.rotation_local_to_global)

    def _local(self, x_mm, y_mm, z_mm):
        r = self.registration.rotation_array
        t = self.registration.origin_array_m * 1e3
        x, y, z = x_mm-t[0], y_mm-t[1], z_mm-t[2]
        return tuple(x*r[0, k] + y*r[1, k] + z*r[2, k] for k in range(3))

    def transmission_mask(self, x_mm, y_mm, z_mm, previous_z_mm=None, xp=None):
        """Global XY mask; include the complete previous interval if given."""
        if previous_z_mm is not None:
            return self.swept_transmission_mask(x_mm, y_mm, previous_z_mm, z_mm, xp=xp)
        xp = array_module(x_mm) if xp is None else xp
        x, y, z = self._local(x_mm, y_mm, z_mm)
        active = (z >= self.local_start_z_mm) & (z <= self.local_end_z_mm)
        return ~active | (xp.hypot(x, y) < self.radius_mm)

    def swept_transmission_mask(self, x_mm, y_mm, start_z_mm, end_z_mm, *, xp=None):
        """Transmit only if a complete vertical marching segment is clear.

        The local axial-slab entry and exit are solved before checking radial
        contact. Squared radial distance is convex on a line, so its maximum
        over the intersected interval occurs at one of those two endpoints.
        This is the same shoulder/wall geometry as ray bore clipping, without
        a device-to-host copy. Both marching directions are supported.
        """
        if not math.isfinite(start_z_mm) or not math.isfinite(end_z_mm):
            raise ValueError("Bore marching bounds must be finite")
        xp = array_module(x_mm) if xp is None else xp
        x, y, z = self._local(x_mm, y_mm, start_z_mm)
        delta = (end_z_mm-start_z_mm)*self.registration.rotation_array[2]
        dz = float(delta[2])
        if abs(dz) <= 1e-14:
            active = (z >= self.local_start_z_mm) & (z <= self.local_end_z_mm)
            first, last = 0., 1.
        else:
            a, b = (self.local_start_z_mm-z)/dz, (self.local_end_z_mm-z)/dz
            first = xp.maximum(0., xp.minimum(a, b))
            last = xp.minimum(1., xp.maximum(a, b))
            active = last >= first
        first_radius = xp.hypot(x+first*delta[0], y+first*delta[1])
        last_radius = xp.hypot(x+last*delta[0], y+last*delta[1])
        return ~active | ((first_radius < self.radius_mm) & (last_radius < self.radius_mm))

    def clears_grid(self, shape, basis_m, origin_m, z_mm, previous_z_mm=None):
        """Conservatively prove clearance of the entire grid prism.

        Not merely a corner test at the output plane: the proof bounds all
        intermediate Z values and all samples of an affine grid. A false
        result requests actual masking; it does not claim an interception.
        """
        start_z_mm = z_mm if previous_z_mm is None else previous_z_mm
        end_z_mm = z_mm
        corners = _grid_corners_mm(shape, basis_m, origin_m)
        xyz = np.concatenate((np.column_stack((corners, np.full(4, start_z_mm))),
                              np.column_stack((corners, np.full(4, end_z_mm)))))
        local = (xyz-self.registration.origin_array_m*1e3) @ self.registration.rotation_array
        magnitude = max(1., float(np.max(abs(local))))
        slack = 32*np.finfo(float).eps*magnitude
        if (np.max(local[:, 2])+slack < self.local_start_z_mm
                or np.min(local[:, 2])-slack > self.local_end_z_mm):
            return True
        # A norm is convex, so its maximum on this affine prism occurs at a
        # vertex. Ignoring the slab for this second proof is conservative.
        return bool(np.nextafter(np.max(np.hypot(local[:, 0], local[:, 1]))+slack,
                                 np.inf) < self.radius_mm)

    def grid_axial_bounds_mm(self, shape, basis_m, origin_m):
        """Z range in which the local slab can intersect this transverse grid.

        A tilted slab has no finite global Z range without a bounded XY
        domain. These bounds must be recomputed if the wave grid changes.
        A slab parallel to global Z may be active for all Z.
        """
        corners = _grid_corners_mm(shape, basis_m, origin_m)
        r, t = self.registration.rotation_array, self.registration.origin_array_m*1e3
        if abs(r[2, 2]) <= 1e-14:
            return -np.inf, np.inf
        shift = (corners[:, 0]-t[0])*r[0, 2] + (corners[:, 1]-t[1])*r[1, 2]
        lower = t[2] + (self.local_start_z_mm-shift)/r[2, 2]
        upper = t[2] + (self.local_end_z_mm-shift)/r[2, 2]
        slack = 32*np.finfo(float).eps*max(1., float(np.max(abs(np.r_[lower, upper]))))
        return float(min(lower.min(), upper.min())-slack), float(max(lower.max(), upper.max())+slack)

    def event_z_mm(self):
        """Finite opening-face extrema for useful plan events, not admission.

        These bracket each tilted circular *opening*. Opaque shoulders extend
        further in Z over a larger XY grid, so never use these events alone to
        remove a bore constraint from the marching plan.
        """
        r, t = self.registration.rotation_array, self.registration.origin_array_m*1e3
        spread = self.radius_mm*math.hypot(r[2, 0], r[2, 1])
        return tuple(sorted(z*r[2, 2]+t[2]+offset
                            for z in (self.local_start_z_mm, self.local_end_z_mm)
                            for offset in (-spread, 0., spread)))

    def maximum_step_mm(self, basis_m, *, pixel_fraction=.5):
        """Geometry-only guide limiting bore-axis motion to part of a pixel.

        Swept masks still apply even with this guide. It neither resolves wave
        phase nor certifies hard-edge diffraction convergence.
        """
        if not math.isfinite(pixel_fraction) or not 0 < pixel_fraction <= 1:
            raise ValueError("Geometry pixel fraction must be in (0, 1]")
        basis = np.asarray(basis_m, dtype=float)
        if basis.shape != (2, 2) or not np.all(np.isfinite(basis)):
            raise ValueError("Wave basis must be a finite 2-D matrix")
        pixel_mm = float(np.linalg.svd(basis, compute_uv=False)[-1])*1e3
        if pixel_mm <= 0:
            raise ValueError("Wave basis must be invertible")
        axis = self.registration.rotation_array[:, 2]
        transverse = math.hypot(axis[0], axis[1])
        return np.inf if transverse == 0 else pixel_fraction*pixel_mm*abs(axis[2])/transverse


def placed_wave_bores(state, start_z_mm, stop_z_mm):
    """Return stationary profiles and all installed posed bore constraints.

    Shares the ray inventory, including fixed wider tubes recovered when a
    narrower component moves. Never derive hardware presence from excitation.
    """
    from temsim.physics.column_wall import _partition_vacuum_segments, _vacuum_segments
    stationary, placed = _partition_vacuum_segments(
        state, _vacuum_segments(state, np.array((start_z_mm, stop_z_mm))))
    return stationary, tuple(PosedWaveBore.from_segment(segment, registration)
                             for segment, registration in placed)
