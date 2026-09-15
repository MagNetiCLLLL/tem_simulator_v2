"""Stable near-axis interpolation of an executed axisymmetric potential solve.

The globally tip-refined tensor grid contains nanometre radial cells even
hundreds of millimetres downstream. Differences between their kilovolt nodal
potentials can be below floating-point resolution. Dividing those differences
by r**2 manufactures a large, sometimes sign-reversed radial force.

Only the interpolation's innermost radial interval is coarsened, in s=r**2.
At each axial NODE, interpolate from the solved axis potential to a resolved
radial node within a small fraction of the nearest tip/bore scale. Outside
that interval retain the original field. Then interpolate between the two
axial nodes. This ordering preserves potential continuity when the radial
support changes with Z. Both force components differentiate this same scalar
potential; no independently fitted focusing force is added.

This is a controlled interpolation approximation, not a new Laplace solve or
an electrode change. Its radial support and the field mesh need independent
convergence checks. Boundary-conforming tip cells are never replaced.
"""
from __future__ import annotations

import numpy as np


class AxisRegularPotential:
    """SI coordinates, potential rise in volts, electric field in volts/metre."""

    def __init__(self, solved_field, *, bore_radius_m: float, fraction: float, compiled=True):
        if (isinstance(fraction, (bool, np.bool_))
                or not np.isfinite(fraction) or not 0 <= fraction <= .05
                or not np.isfinite(bore_radius_m) or bore_radius_m <= 0):
            raise ValueError("Axis interpolation needs a positive bore and fraction in [0, 0.05]")
        self.field = solved_field
        self.fraction = float(fraction)
        self.compiled = bool(compiled)
        r, z = solved_field.r, solved_field.z
        limit = fraction*np.minimum(np.maximum(z, 0.), bore_radius_m)
        indices = np.maximum(0, np.searchsorted(r, limit, side="right")-1)
        # If the original axis cell is already larger, leave it untouched.
        indices[indices < 2] = 0
        self.radius_m = r[indices]
        self.s = self.radius_m**2
        values = solved_field.nodal_voltage
        difference = values[indices, np.arange(len(z))]-values[0]
        self.slope = np.divide(difference, self.s,
                               out=np.zeros_like(difference), where=self.s > 0)
        for value in (self.radius_m, self.s, self.slope):
            value.setflags(write=False)

    def interpolate(self, positions):
        """Retain cut-cell and off-axis results; replace only the axis core."""
        p = np.asarray(positions, float)
        from .axis_field_interpolation import compiled_evaluate
        if self.compiled and compiled_evaluate is not None:
            f = self.field
            if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
                raise ValueError("Field positions must be finite xyz coordinates")
            if (np.any(np.hypot(p[:,0],p[:,1]) > f.r[-1])
                    or np.any(p[:,2] < f.z[0]) or np.any(p[:,2] > f.z[-1])):
                raise ValueError("Requested position is outside the conforming field")
            return compiled_evaluate(p, f.r, f.z, f.nodal_voltage, f.cut_cells, f.lookup,
                f.origin, f.inverse, f.gradient, f.phi0, self.s, self.slope)
        potential, electric = self.field.interpolate(p)
        if not np.any(self.s > 0):
            return potential, electric
        radius = np.hypot(p[:, 0], p[:, 1])
        s = radius**2
        r, z = self.field.r, self.field.z
        j = np.minimum(np.searchsorted(z, p[:, 2], side="right")-1, len(z)-2)
        i = np.minimum(np.searchsorted(r, radius, side="right")-1, len(r)-2)
        selected = np.flatnonzero(((s < self.s[j]) | (s < self.s[j+1]))
                                  & ~self.field.cut_cells[i, j])
        if not len(selected):
            return potential, electric
        i, j, s = i[selected], j[selected], s[selected]
        radial_width = r[i+1]**2-r[i]**2
        u = (s-r[i]**2)/radial_width
        dz = z[j+1]-z[j]
        v = (p[selected, 2]-z[j])/dz
        values = self.field.nodal_voltage
        a, b = values[i, j], values[i+1, j]
        c, d = values[i, j+1], values[i+1, j+1]
        bottom = (1-u)*a+u*b
        top = (1-u)*c+u*d
        bottom_s, top_s = (b-a)/radial_width, (d-c)/radial_width
        for row, point, derivative in ((j, bottom, bottom_s), (j+1, top, top_s)):
            core = s < self.s[row]
            point[core] = values[0, row[core]]+s[core]*self.slope[row[core]]
            derivative[core] = self.slope[row[core]]
        potential[selected] = (1-v)*bottom+v*top
        derivative_s = (1-v)*bottom_s+v*top_s
        electric[selected, :2] = -2*p[selected, :2]*derivative_s[:, None]
        electric[selected, 2] = -(top-bottom)/dz
        return potential, electric
