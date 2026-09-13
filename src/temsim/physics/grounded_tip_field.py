"""Axisymmetric vacuum Laplace field for an idealised, grounded FEG assembly.

Nonuniform cylindrical finite-volume conductances; metal nodes are Dirichlet.
The numerical radial/back boundaries are insulating (zero normal derivative),
not extra grounded electrodes. The exit boundary and final accelerator ring
are grounded. No charge density, image potential or tunnelling is included.
Electrode contours are reference annuli, not a claim about an OEM gun.
"""
from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
import json

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

from temsim.optics.electron_gun.tip_surface import TipSurfaceModel


def field_request(gun):
    """Include every consumed geometry, voltage and numerical input in the key."""
    model = gun.emitter.surface_model
    if model is None:
        raise ValueError("A grounded surface model must be selected")
    model.validate()
    ht = float(gun.accelerator.high_tension_kv) * 1000
    ext = float(gun.extractor.voltage_kv) * 1000
    if not 0 <= ext < ht:
        raise ValueError("Extraction voltage must lie between zero and high tension")
    if not gun.accelerator.stages or gun.accelerator.stages[-1].voltage_fraction != 1.0:
        raise ValueError("The final accelerating anode is fixed at ground: its voltage fraction must be exactly 1")
    rings = []
    for component, potential in ((gun.extractor, ext),
            (gun.electrostatic_lens, ext + float(gun.electrostatic_lens.voltage_kv)*1000)):
        center = float(component.mechanical_center_from_tip_mm)*1e-3
        half = float(component.mechanical_length_mm)*.5e-3
        rings.append((component.key, center-half, center+half,
                      float(component.mechanical_clear_bore_diameter_mm)*.5e-3,
                      float(component.mechanical_outer_diameter_mm)*.5e-3, potential))
    for index, stage in enumerate(gun.accelerator.stages):
        center = float(stage.center_from_tip_mm)*1e-3
        half = model.field_numerics.accelerator_ring_thickness_mm*.5e-3
        rings.append((f"accelerator:{index}", center-half, center+half,
                      float(gun.accelerator.mechanical_clear_bore_diameter_mm)*.5e-3,
                      float(gun.accelerator.mechanical_outer_diameter_mm)*.5e-3,
                      ext + stage.voltage_fraction*(ht-ext)))
    # Offsets/soft field windows are legacy analytic parameters, not electrode
    # locations. Moving metal now changes the field through its boundary.
    return {"schema": "axisymmetric-grounded-feg-laplace-v1",
            "potential_interpolation": "bilinear-radius-squared-z-v1",
            "geometry": asdict(model.geometry), "numerics": asdict(model.field_numerics),
            "rings": rings, "high_tension_v": ht, "exit_m": gun.exit_plane_z_mm*1e-3}


def grounded_field(gun):
    return _cached_field(json.dumps(field_request(gun), sort_keys=True, allow_nan=False))


@lru_cache(maxsize=8)
def _cached_field(key):
    return GroundedTipField(json.loads(key))


def solve_axisymmetric_laplace(r_m, z_m, fixed, fixed_v, *, tolerance=1e-9, tip_geometry=None):
    """Solve on (r,z) nodes, with Neumann zero on unspecified outer faces.

    Returns potential and row-scaled residual. Fixed voltages must include a
    nonempty Dirichlet boundary; all axes and numerical failure are checked.
    """
    r, z = np.asarray(r_m, float), np.asarray(z_m, float)
    shape = (r.size, z.size)
    fixed, values = np.asarray(fixed, bool), np.asarray(fixed_v, float)
    if (r.size < 3 or z.size < 3 or r[0] != 0 or not np.all(np.isfinite(r))
            or not np.all(np.isfinite(z)) or np.any(np.diff(r) <= 0) or np.any(np.diff(z) <= 0)
            or fixed.shape != shape or values.shape != shape or not fixed.any()
            or not np.all(np.isfinite(values)) or not 0 < tolerance <= 1e-6):
        raise ValueError("Invalid axisymmetric Laplace grid or boundary")
    nr, nz = shape
    ids = np.arange(nr*nz).reshape(shape)
    rf = np.r_[0., .5*(r[1:]+r[:-1]), r[-1]]
    zf = np.r_[z[0], .5*(z[1:]+z[:-1]), z[-1]]
    area = .5*np.diff(rf**2)
    radial = .5*(r[1:]+r[:-1])[:, None]*np.diff(zf)[None, :]/np.diff(r)[:, None]
    axial = area[:, None]/np.diff(z)[None, :]
    if tip_geometry is not None:
        # Embedded Dirichlet boundary: shorten vacuum-to-metal edge lengths to
        # the analytic spherical-cap/cone intersection. This avoids the false
        # apex enhancement from treating the last metal node as an isolated
        # point on a staircase. The remaining metric error is mesh tested.
        metal_r = tip_geometry.radius_m(z)
        crossing_r = (z[None, :] <= 0) & (r[:-1, None] <= metal_r) & (r[1:, None] > metal_r)
        distance_r = r[1:, None]-metal_r
        radial[crossing_r] *= np.broadcast_to(np.diff(r)[:, None], radial.shape)[crossing_r]/distance_r[crossing_r]
        radius = tip_geometry.apex_radius_nm*1e-9
        angle = np.deg2rad(tip_geometry.cone_half_angle_deg)
        join_r, join_z = radius*np.cos(angle), -radius*(1-np.sin(angle))
        surface_z = np.where(r <= join_r,
            -radius+np.sqrt(np.maximum(0., radius**2-r**2)),
            join_z-(r-join_r)/np.tan(angle))
        crossing_z = (z[:-1][None, :] <= surface_z[:, None]) & (z[1:][None, :] > surface_z[:, None])
        distance_z = z[1:][None, :]-surface_z[:, None]
        axial[crossing_z] *= np.broadcast_to(np.diff(z)[None, :], axial.shape)[crossing_z]/distance_z[crossing_z]
    left = np.r_[ids[:-1].ravel(), ids[:, :-1].ravel()]
    right = np.r_[ids[1:].ravel(), ids[:, 1:].ravel()]
    weights = np.r_[radial.ravel(), axial.ravel()]
    rows = np.r_[left, right, left, right]
    cols = np.r_[left, right, right, left]
    data = np.r_[weights, weights, -weights, -weights]
    matrix = coo_matrix((data, (rows, cols)), shape=(nr*nz, nr*nz)).tocsr()
    known = fixed.ravel()
    unknown = ~known
    block = matrix[unknown][:, unknown]
    rhs = -matrix[unknown][:, known] @ values.ravel()[known]
    # Scaling each equation is conditioning, not a modification of the field.
    scale = 1/block.diagonal()
    block = diags(scale) @ block
    rhs = scale*rhs
    solved = spsolve(block.tocsc(), rhs)
    residual = float(np.max(np.abs(block@solved-rhs), initial=0)/max(1., np.max(np.abs(values))))
    if not np.all(np.isfinite(solved)) or residual > tolerance:
        raise ValueError(f"Laplace solve failed: scaled residual {residual:.3g} > {tolerance:.3g}")
    result = values.copy().ravel()
    result[unknown] = solved
    return result.reshape(shape), residual


def _axis(endpoint, minimum, count):
    return np.r_[0., np.geomspace(minimum, endpoint, count-1)]


class GroundedTipField:
    """Potential in V relative to ground; electric field in V/m."""

    def __init__(self, request):
        from temsim.optics.electron_gun.tip_surface import TipGeometry, TipFieldNumerics
        self.request = request
        geometry = TipGeometry(**request["geometry"]).validate()
        numerics = TipFieldNumerics(**request["numerics"]).validate()
        rings = request["rings"]
        ht = float(request["high_tension_v"])
        end = float(request["exit_m"])
        if not rings or not np.isfinite(ht) or ht <= 0 or not np.isfinite(end) or end <= 0:
            raise ValueError("Invalid grounded gun boundary")
        for key, start, stop, inner, outer, potential in rings:
            if (not np.all(np.isfinite([start, stop, inner, outer, potential]))
                    or not 0 < start < stop < end or not 0 < inner < outer):
                raise ValueError(f"Invalid conductor boundary: {key}")
        radius = geometry.apex_radius_nm*1e-9
        step = radius/numerics.apex_cells_per_radius
        rmax = max(row[4] for row in rings)*numerics.outer_radius_factor
        self.r = np.unique(np.r_[_axis(rmax, step, numerics.radial_nodes),
            [v for row in rings for v in row[3:5]]])
        shank = geometry.shank_length_um*1e-6
        if geometry.radius_m(-shank) >= rmax:
            raise ValueError("The numerical radial boundary must enclose the full reference tip shank")
        self.z = np.unique(np.r_[-_axis(shank, step, numerics.axial_nodes//3)[::-1],
            _axis(end, step, numerics.axial_nodes),
            [v for row in rings for v in (row[1], .5*(row[1]+row[2]), row[2])]])
        rr, zz = np.meshgrid(self.r, self.z, indexing="ij")
        tip = (zz <= 0) & (rr <= geometry.radius_m(zz))
        fixed = tip.copy()
        rise = np.zeros_like(rr)
        for key, start, stop, inner, outer, potential in rings:
            metal = (zz >= start) & (zz <= stop) & (rr >= inner) & (rr <= outer)
            if np.any(metal & fixed):
                raise ValueError(f"Overlapping electrode boundaries: {key}")
            if not np.any(metal):
                raise ValueError(f"Grid did not resolve electrode {key}")
            fixed |= metal
            rise[metal] = potential
        fixed[:, -1] = True
        rise[:, -1] = ht
        self.rise, residual = solve_axisymmetric_laplace(self.r, self.z, fixed, rise,
            tolerance=numerics.linear_residual_tolerance, tip_geometry=geometry)
        self.high_tension_v = ht
        self.geometry = geometry
        # A regular axisymmetric scalar potential is even in radius. Use r^2
        # rather than r for interpolation; a linear-r interpolant introduces
        # an unphysical |r| cusp and finite radial force arbitrarily near the
        # axis. Electric field remains the derivative of this SAME potential.
        self.report = {"schema": request["schema"], "potential_reference": "final_anode_ground_0V",
            "tip_potential_v": -ht, "extractor_potential_v": -ht+rings[0][-1],
            "final_anode_potential_v": 0., "grid_shape": list(rr.shape),
            "linear_residual": residual, "space_charge": "neglected",
            "potential_interpolation": request["potential_interpolation"],
            "boundary_model": "reference annuli; insulating radial/back boundary; grounded exit plane",
            "convergence": "not_certified_by_linear_residual"}
        for array in (self.r, self.z, self.rise):
            array.setflags(write=False)

    def _interpolate(self, positions_m):
        points = np.asarray(positions_m, float)
        if points.shape[-1:] != (3,) or not np.all(np.isfinite(points)):
            raise ValueError("Field positions must be finite xyz coordinates in metres")
        shape = points.shape[:-1]
        p = points.reshape(-1, 3)
        radius, z = np.hypot(p[:, 0], p[:, 1]), p[:, 2]
        if np.any(radius > self.r[-1]) or np.any(z < self.z[0]):
            raise ValueError("Requested position is outside the solved gun field")
        downstream = z >= self.z[-1]
        z = np.minimum(z, self.z[-1])
        i = np.minimum(np.searchsorted(self.r, radius, side="right")-1, len(self.r)-2)
        j = np.minimum(np.searchsorted(self.z, z, side="right")-1, len(self.z)-2)
        dr2, dz = self.r[i+1]**2-self.r[i]**2, self.z[j+1]-self.z[j]
        u, v = (radius**2-self.r[i]**2)/dr2, (z-self.z[j])/dz
        a, b = self.rise[i, j], self.rise[i+1, j]
        c, d = self.rise[i, j+1], self.rise[i+1, j+1]
        value = (1-u)*(1-v)*a + u*(1-v)*b + (1-u)*v*c + u*v*d
        er, ez = -2*radius*((1-v)*(b-a)+v*(d-c))/dr2, -((1-u)*(c-a)+u*(d-b))/dz
        field = np.zeros_like(p)
        field[:, :2] = p[:, :2] * np.divide(er, radius, out=np.zeros_like(er), where=radius>0)[:, None]
        field[:, 2] = ez
        field[downstream] = 0.
        value[downstream] = self.high_tension_v
        return value.reshape(shape), field.reshape(points.shape)

    def potential_v_at_global_positions(self, positions_m):
        return self._interpolate(positions_m)[0]-self.high_tension_v

    def potential_rise_v_at_global_positions(self, positions_m):
        """Stable work differences; public electrode voltages remain grounded."""
        return self._interpolate(positions_m)[0]

    def _metal_surface_z(self, radius):
        """Upper face of the represented tip metal, never another electrode."""
        i = np.minimum(np.searchsorted(self.r, radius, side="right")-1, len(self.r)-2)
        surface_z = np.full(np.shape(radius), -np.inf)
        for index, row in enumerate(i):
            nodes = np.flatnonzero((self.z <= 0) & (self.rise[row] == 0) & (self.rise[row+1] == 0))
            if nodes.size:
                surface_z[index] = self.z[nodes[-1]]
        return surface_z

    def tip_material_mask(self, positions_m):
        """Absorb returning rays in the same discretised metal used at launch."""
        p = np.asarray(positions_m, float)
        result = np.zeros(p.shape[0], dtype=bool)
        near_tip = p[:, 2] < 0
        if np.any(near_tip):
            surface_z = self._metal_surface_z(np.hypot(p[near_tip, 0], p[near_tip, 1]))
            result[near_tip] = p[near_tip, 2] < surface_z - self.geometry.apex_radius_nm*1e-19
        return result

    def surface_mesh_positions(self, positions_m):
        """Represent the prescribed curved surface on the solved metal grid.

        Project inward onto its local zero-rise boundary, never onto an
        independently selectable accelerated launch plane. Report this shape
        discretisation separately; it must converge with the electrode mesh.
        """
        points = np.array(positions_m, dtype=float, copy=True)
        radius = np.hypot(points[:, 0], points[:, 1])
        points[:, 2] = self._metal_surface_z(radius)
        if not np.all(np.isfinite(points)):
            raise ValueError("Emission patch is outside the resolved tip surface")
        error = np.max(np.abs(points[:, 2]-np.asarray(positions_m)[:, 2]), initial=0)
        if error > .1*self.geometry.apex_radius_nm*1e-9:
            raise ValueError("Tip boundary displacement exceeds 10% of apex radius; refine the field grid")
        return points, float(error)

    def field_at_global_positions_v_per_m(self, positions_m):
        return self._interpolate(positions_m)[1]

    def axial_potential_v(self, z_mm):
        z = np.asarray(z_mm, float)
        p = np.zeros((*z.shape, 3))
        p[..., 2] = z*1e-3
        return self.potential_v_at_global_positions(p)

    def axial_potential_v_and_derivatives_per_mm(self, z_mm):
        raise ValueError("Grounded surface field needs grid-aware non-paraxial transport; legacy radial Taylor coefficients are unavailable")
