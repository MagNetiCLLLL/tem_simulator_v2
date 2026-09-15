"""Boundary-conforming P1 electrostatics in (s=r**2, z) coordinates.

The axisymmetric vacuum weak form (apart from a constant factor) is
integral(4*s*phi_s*v_s + phi_z*v_z) ds dz = 0. Linear triangular elements
therefore have exactly integrable stiffness and a regular E_r=-2*r*phi_s.
Grid triangles intersecting the tip are clipped at its analytic contour;
the same piecewise-linear boundary is used for emission and absorption.
No virtual source, fitted field multiplier or downstream launch is introduced.
Uncut grid cells reconstruct the nodal solution bilinearly in (s,z). This
removes unnecessary internal-diagonal force jumps while matching every cell
edge exactly; cut cells retain their boundary-conforming triangle potential.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve


def tip_surface_z(s, geometry):
    """Analytic spherical-cap/tangent-cone upper metal boundary, metres."""
    s = np.asarray(s, float)
    radius = geometry.apex_radius_nm * 1e-9
    angle = np.deg2rad(geometry.cone_half_angle_deg)
    join_r = radius * np.cos(angle)
    join_z = -radius * (1 - np.sin(angle))
    # Stable at the apex: sqrt(R**2-s)-R loses the small cap depth.
    root = np.sqrt(np.maximum(0., radius*radius-s))
    cap = -s / (radius+root)
    cone = join_z - (np.sqrt(s)-join_r)/np.tan(angle)
    return np.where(s <= join_r*join_r, cap, cone)


class AxisymmetricCutField:
    """Solve a graded rectangular grid clipped by an optional tip conductor."""

    def __init__(self, r, z, fixed, values, *, geometry=None, tolerance=1e-9):
        self.r, self.z = np.asarray(r, float), np.asarray(z, float)
        nr, nz = len(r), len(z)
        if (nr < 3 or nz < 3 or self.r[0] != 0 or not np.all(np.isfinite(self.r))
                or not np.all(np.isfinite(self.z)) or np.any(np.diff(self.r) <= 0)
                or np.any(np.diff(self.z) <= 0) or np.shape(fixed) != (nr,nz)
                or np.shape(values) != (nr,nz) or not np.any(fixed)
                or not np.all(np.isfinite(values)) or not 0 < tolerance <= 1e-6):
            raise ValueError("Invalid conforming Laplace grid or boundary")
        s = self.r**2
        ss, zz = np.meshgrid(s, self.z, indexing="ij")
        vertices = np.column_stack((ss.ravel(), zz.ravel()))
        ids = np.arange(nr*nz).reshape(nr, nz)
        a, b = ids[:-1, :-1].ravel(), ids[1:, :-1].ravel()
        c, d = ids[1:, 1:].ravel(), ids[:-1, 1:].ravel()
        base = np.stack((np.column_stack((a,b,c)), np.column_stack((a,c,d))), axis=1).reshape(-1,3)
        original_ids = np.arange(len(base))
        self.lookup = np.full((len(base), 2), -1, dtype=np.int64)
        known = np.asarray(fixed, bool).ravel().copy()
        voltage = np.asarray(values, float).ravel().copy()
        if geometry is None:
            triangles, parents = base, original_ids
            self.cut_cells = np.zeros((nr-1,nz-1),bool)
        else:
            signed = vertices[:,1] - tip_surface_z(vertices[:,0], geometry)
            metal = signed <= 0.
            known[metal] = True
            voltage[metal] = 0.
            inside = metal[base]
            self.cut_cells = inside.any(axis=1).reshape(nr-1,nz-1,2).any(axis=2)
            clean = ~inside.any(axis=1)
            cut = inside.any(axis=1) & ~inside.all(axis=1)
            cut_ids = original_ids[cut]
            edges = np.sort(base[cut][:, [[0,1],[1,2],[2,0]]].reshape(-1,2), axis=1)
            edges = np.unique(edges[metal[edges[:,0]] != metal[edges[:,1]]], axis=0)
            lo, hi = np.zeros(len(edges)), np.ones(len(edges))
            start, delta = vertices[edges[:,0]], vertices[edges[:,1]]-vertices[edges[:,0]]
            sign_start = signed[edges[:,0]] <= 0.
            for _ in range(56):
                mid = .5*(lo+hi)
                point = start+mid[:,None]*delta
                same = (point[:,1] <= tip_surface_z(point[:,0], geometry)) == sign_start
                lo, hi = np.where(same, mid, lo), np.where(same, hi, mid)
            fractions = .5*(lo+hi)
            fractions[signed[edges[:,0]] == 0.] = 0.
            fractions[signed[edges[:,1]] == 0.] = 1.
            intersections = start+fractions[:,None]*delta
            edge_ids = {}
            added = []
            for pair, fraction, point in zip(edges, fractions, intersections):
                if fraction == 0.:
                    index = int(pair[0])
                elif fraction == 1.:
                    index = int(pair[1])
                else:
                    index = len(vertices)+len(added)
                    added.append(point)
                edge_ids[tuple(pair)] = index
            surface = np.vstack((intersections, vertices[signed == 0.]))
            order = np.argsort(surface[:,0])
            self.boundary_s, unique = np.unique(surface[order,0], return_index=True)
            self.boundary_z = surface[order,1][unique]
            triangles = [base[clean]]
            parents = [original_ids[clean]]
            new_triangles, new_parents = [], []
            for parent in cut_ids:
                polygon = []
                tri = base[parent]
                for left, right in zip(tri, np.roll(tri,-1)):
                    if not metal[left]:
                        polygon.append(int(left))
                    if metal[left] != metal[right]:
                        polygon.append(edge_ids[tuple(sorted((int(left),int(right))))])
                polygon = list(dict.fromkeys(polygon))
                for k in range(1, len(polygon)-1):
                    new_triangles.append((polygon[0],polygon[k],polygon[k+1]))
                    new_parents.append(parent)
            if new_triangles:
                triangles.append(np.asarray(new_triangles))
                parents.append(np.asarray(new_parents))
            triangles, parents = np.concatenate(triangles), np.concatenate(parents)
            if added:
                vertices = np.vstack((vertices, added))
                known = np.r_[known, np.ones(len(added),bool)]
                voltage = np.r_[voltage, np.zeros(len(added))]
        points = vertices[triangles]
        e1, e2 = points[:,1]-points[:,0], points[:,2]-points[:,0]
        determinant = e1[:,0]*e2[:,1]-e1[:,1]*e2[:,0]
        if np.any(determinant <= 0) or not np.all(np.isfinite(determinant)):
            raise ValueError("Invalid conforming electrostatic triangle")
        inverse = np.stack((np.column_stack((e2[:,1],-e2[:,0])),
                            np.column_stack((-e1[:,1],e1[:,0]))),axis=1)/determinant[:,None,None]
        gradients = np.concatenate((-inverse.sum(axis=1)[:,None,:], inverse),axis=1)
        area = .5*determinant
        stiffness = area[:,None,None]*(4*points[:,:,0].mean(axis=1)[:,None,None]
            *gradients[:,:,0,None]*gradients[:,None,:,0]
            +gradients[:,:,1,None]*gradients[:,None,:,1])
        rows = np.broadcast_to(triangles[:,:,None], stiffness.shape).ravel()
        cols = np.broadcast_to(triangles[:,None,:], stiffness.shape).ravel()
        matrix = coo_matrix((stiffness.ravel(),(rows,cols)),shape=(len(vertices),len(vertices))).tocsr()
        unknown = ~known
        block = matrix[unknown][:,unknown]
        rhs = -matrix[unknown][:,known] @ voltage[known]
        diagonal = block.diagonal()
        if np.any(diagonal <= 0):
            raise ValueError("Disconnected electrostatic vacuum node")
        scale = diags(1/diagonal)
        block, rhs = scale@block, rhs/diagonal
        solved = spsolve(block.tocsc(),rhs)
        self.residual = float(np.max(np.abs(block@solved-rhs),initial=0)/max(1.,np.max(np.abs(voltage))))
        if not np.all(np.isfinite(solved)) or self.residual > tolerance:
            raise ValueError(f"Conforming Laplace solve failed: residual {self.residual:g}")
        voltage[unknown] = solved
        self.nodal_voltage = voltage[:nr*nz].reshape(nr,nz)
        self.origin = points[:,0]
        self.inverse = inverse
        differences = voltage[triangles[:,1:]]-voltage[triangles[:,0,None]]
        self.gradient = np.sum(differences[:,:,None]*inverse,axis=1)
        self.phi0 = voltage[triangles[:,0]]
        # Every clipped parent triangle has at most two vacuum triangles.
        order = np.argsort(parents,kind="stable")
        ordered = parents[order]
        slot = np.zeros(len(order),dtype=int)
        slot[1:] = (ordered[1:] == ordered[:-1]).astype(int)
        self.lookup[ordered,slot] = order
        self.element_count = len(triangles)
        self.vertex_count = len(vertices)
        for item in vars(self).values():
            if isinstance(item,np.ndarray):
                item.setflags(write=False)

    def surface_z(self, radius):
        return np.interp(np.asarray(radius)**2,self.boundary_s,self.boundary_z)

    def interpolate(self, positions):
        """Return potential rise (V) and its exact negative gradient (V/m)."""
        p = np.asarray(positions,float)
        if p.ndim != 2 or p.shape[1] != 3 or not np.all(np.isfinite(p)):
            raise ValueError("Field positions must be finite xyz coordinates")
        radius, z = np.hypot(p[:,0],p[:,1]), p[:,2]
        if np.any(radius > self.r[-1]) or np.any(z < self.z[0]) or np.any(z > self.z[-1]):
            raise ValueError("Requested position is outside the conforming field")
        s = radius**2
        i = np.minimum(np.searchsorted(self.r,radius,side="right")-1,len(self.r)-2)
        j = np.minimum(np.searchsorted(self.z,z,side="right")-1,len(self.z)-2)
        u = (s-self.r[i]**2)/(self.r[i+1]**2-self.r[i]**2)
        v = (z-self.z[j])/(self.z[j+1]-self.z[j])
        parent = 2*(i*(len(self.z)-1)+j)+(v>u).astype(int)
        points = np.column_stack((s,z))
        a, b = self.nodal_voltage[i,j], self.nodal_voltage[i+1,j]
        c, d = self.nodal_voltage[i,j+1], self.nodal_voltage[i+1,j+1]
        potential = (1-u)*(1-v)*a+u*(1-v)*b+(1-u)*v*c+u*v*d
        gradient = np.column_stack((((1-v)*(b-a)+v*(d-c))/(self.r[i+1]**2-self.r[i]**2),
                                    ((1-u)*(c-a)+u*(d-b))/(self.z[j+1]-self.z[j])))
        assigned = ~self.cut_cells[i,j]
        potential[~assigned] = 0.
        gradient[~assigned] = 0.
        for slot in (0,1):
            selected = np.flatnonzero((self.lookup[parent,slot] >= 0)&~assigned)
            index = self.lookup[parent[selected],slot]
            delta = points[selected]-self.origin[index]
            bary = np.einsum("nij,nj->ni",self.inverse[index],delta)
            inside = (bary.min(axis=1)>=-1e-9)&(bary.sum(axis=1)<=1+1e-9)
            selected, index, delta = selected[inside], index[inside], delta[inside]
            potential[selected] = self.phi0[index]+np.sum(self.gradient[index]*delta,axis=1)
            gradient[selected] = self.gradient[index]
            assigned[selected] = True
        field = np.empty_like(p)
        field[:,:2] = -2*p[:,:2]*gradient[:,0,None]
        field[:,2] = -gradient[:,1]
        return potential,field
