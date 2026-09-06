"""Static isotropic A-phi P1 FEM with a consistent tangent and damped Newton.

SI units. Three-point triangle quadrature, material IDs at centroids, A=0
on the symmetry axis and remote boundary. No hysteresis or thermal feedback.
Mesh preparation is bounded and reusable; fields are never current-scaled.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from temsim.magnetic_materials import BHCurve, MU0


@dataclass(frozen=True)
class FEMMesh:
    r: np.ndarray
    z: np.ndarray
    triangles: np.ndarray
    centres: np.ndarray
    curl: np.ndarray
    weights: np.ndarray
    shapes: np.ndarray
    free: np.ndarray

    @property
    def size(self):
        return self.r.size * self.z.size

    def assemble_matrix(self, local):
        return coo_matrix((local.ravel(), (
            np.broadcast_to(self.triangles[:, :, None], local.shape).ravel(),
            np.broadcast_to(self.triangles[:, None, :], local.shape).ravel())),
            shape=(self.size, self.size)).tocsr()

    def assemble_vector(self, local):
        result = np.zeros(self.size)
        np.add.at(result, self.triangles.ravel(), local.ravel())
        return result

    def forcing(self, current):
        current = np.broadcast_to(np.asarray(current, float), (len(self.triangles),))
        if not np.all(np.isfinite(current)):
            raise ValueError("Current density must be finite")
        return self.assemble_vector(np.einsum("tq,qi,t->ti", self.weights, self.shapes, current))


@lru_cache(maxsize=4)
def prepared_mesh(r_values: tuple, z_values: tuple) -> FEMMesh:
    r, z = np.array(r_values, float), np.array(z_values, float)
    if (r.ndim != 1 or z.ndim != 1 or min(r.size, z.size) < 4
            or r[0] != 0 or np.any(np.diff(r) <= 0) or np.any(np.diff(z) <= 0)
            or not np.all(np.isfinite(r)) or not np.all(np.isfinite(z))):
        raise ValueError("Magnetostatic axes must be increasing SI arrays, starting at r=0")
    rr, zz = np.meshgrid(r, z, indexing="ij")
    points = np.column_stack((rr.ravel(), zz.ravel()))
    ids = np.arange(rr.size).reshape(rr.shape)
    lo = ids[:-1, :-1].ravel()
    triangles = np.vstack((np.column_stack((lo, lo+z.size, lo+z.size+1)),
                           np.column_stack((lo, lo+z.size+1, lo+1))))
    vertices = points[triangles]
    determinant = ((vertices[:, 1, 0]-vertices[:, 0, 0])*(vertices[:, 2, 1]-vertices[:, 0, 1])
                   - (vertices[:, 2, 0]-vertices[:, 0, 0])*(vertices[:, 1, 1]-vertices[:, 0, 1]))
    grad_r = (vertices[:, [1, 2, 0], 1]-vertices[:, [2, 0, 1], 1])/determinant[:, None]
    grad_z = (vertices[:, [2, 0, 1], 0]-vertices[:, [1, 2, 0], 0])/determinant[:, None]
    shapes = np.array(((2/3, 1/6, 1/6), (1/6, 2/3, 1/6), (1/6, 1/6, 2/3)))
    radius = vertices[:, :, 0] @ shapes.T
    curl = np.empty((len(triangles), 3, 2, 3))
    curl[:, :, 0, :] = -grad_z[:, None, :]
    curl[:, :, 1, :] = grad_r[:, None, :] + shapes[None, :, :]/radius[:, :, None]
    weights = determinant[:, None]*radius/6
    centres, free = vertices.mean(axis=1), ids[1:-1, 1:-1].ravel()
    for array in (r, z, triangles, centres, curl, weights, shapes, free):
        array.setflags(write=False)
    return FEMMesh(r, z, triangles, centres, curl, weights, shapes, free)


def _solution(mesh, potential, residual, iterations=1, peak_material_t=0):
    from temsim.physics.axisymmetric_magnetostatics import MagnetostaticSolution
    a = potential.reshape((len(mesh.r), len(mesh.z)))
    br = -np.gradient(a, mesh.z, axis=1, edge_order=2)
    derivative = np.gradient(a, mesh.r, axis=0, edge_order=2)
    bz = derivative.copy()
    bz[1:] += a[1:]/mesh.r[1:, None]
    bz[0] = 2*derivative[0]
    br[0] = 0
    return MagnetostaticSolution(mesh.r, mesh.z, a, br, bz, residual, iterations, peak_material_t)


def solve_linear(r, z, permeability, current):
    mesh = prepared_mesh(tuple(r), tuple(z))
    mur = np.broadcast_to(permeability(*mesh.centres.T), (len(mesh.triangles),))
    if np.any(mur <= 0) or not np.all(np.isfinite(mur)):
        raise ValueError("Permeability must be positive and finite")
    local = np.einsum("tqci,tqcj,tq,t->tij", mesh.curl, mesh.curl, mesh.weights, 1/(MU0*mur))
    matrix = mesh.assemble_matrix(local)[mesh.free][:, mesh.free]
    rhs = mesh.forcing(current(*mesh.centres.T))
    potential = np.zeros(mesh.size)
    potential[mesh.free] = spsolve(matrix, rhs[mesh.free])
    residual = np.linalg.norm(matrix@potential[mesh.free]-rhs[mesh.free])/max(np.linalg.norm(rhs[mesh.free]), 1e-30)
    if not np.all(np.isfinite(potential)) or residual > 1e-7:
        raise ValueError("Magnetostatic linear solve failed its residual check")
    return _solution(mesh, potential, float(residual))


def solve_nonlinear(r, z, material_ids, current, materials, *, relative_tolerance=1e-7, max_iterations=80):
    if not (np.isfinite(relative_tolerance) and 1e-12 <= relative_tolerance <= 1e-3
            and isinstance(max_iterations, (int, np.integer)) and 1 <= max_iterations <= 500):
        raise ValueError("Invalid nonlinear tolerance or iteration limit")
    mesh = prepared_mesh(tuple(r), tuple(z))
    curves = tuple(BHCurve(row) for row in materials)
    labels = np.broadcast_to(np.asarray(material_ids(*mesh.centres.T)), (len(mesh.triangles),))
    if np.any(labels != labels.astype(int)) or np.any(labels < -1) or np.any(labels >= len(curves)):
        raise ValueError("Unknown nonlinear material region")
    masks = [labels == index for index in range(len(curves))]
    rhs = mesh.forcing(current(*mesh.centres.T))
    rhs_norm = max(float(np.linalg.norm(rhs[mesh.free])), 1e-30)

    def system(potential, tangent=True, strict=False):
        flux = np.einsum("tqci,ti->tqc", mesh.curl, potential[mesh.triangles])
        magnitude = np.linalg.norm(flux, axis=2)
        nu = np.full(magnitude.shape, 1/MU0)
        slope = nu.copy()
        for mask, curve in zip(masks, curves):
            h, derivative = curve.evaluate(magnitude[mask], allow_iteration_extension=not strict)
            nu[mask] = np.divide(h, magnitude[mask], out=np.full(h.shape, curve.slopes[0]), where=magnitude[mask] > 0)
            slope[mask] = derivative
        residual = mesh.assemble_vector(np.einsum("tqci,tqc,tq->ti", mesh.curl, flux*nu[:, :, None], mesh.weights))-rhs
        if not tangent:
            return residual
        unit = np.divide(flux, magnitude[:, :, None], out=np.zeros_like(flux), where=magnitude[:, :, None] > 0)
        differential = nu[:, :, None, None]*np.eye(2) + (slope-nu)[:, :, None, None]*unit[:, :, :, None]*unit[:, :, None, :]
        local = np.einsum("tqci,tqcd,tqdj,tq->tij", mesh.curl, differential, mesh.curl, mesh.weights)
        return residual, mesh.assemble_matrix(local), magnitude

    potential = np.zeros(mesh.size)
    for iteration in range(max_iterations+1):
        residual, tangent, magnitude = system(potential)
        norm = float(np.linalg.norm(residual[mesh.free]))
        relative = norm/rhs_norm
        if relative <= relative_tolerance:
            system(potential, tangent=False, strict=True)
            peak = float(np.max(magnitude[labels >= 0], initial=0))
            return _solution(mesh, potential, relative, iteration, peak)
        if iteration == max_iterations:
            break
        direction = np.zeros(mesh.size)
        direction[mesh.free] = spsolve(tangent[mesh.free][:, mesh.free], -residual[mesh.free])
        if not np.all(np.isfinite(direction)):
            raise ValueError("Nonlinear magnetostatic Newton step is not finite")
        for power in range(24):
            alpha = 0.5**power
            trial = potential + alpha*direction
            trial_norm = np.linalg.norm(system(trial, tangent=False)[mesh.free])
            if trial_norm <= (1-1e-4*alpha)*norm or trial_norm/rhs_norm <= relative_tolerance:
                potential = trial
                break
        else:
            raise ValueError(f"Nonlinear magnetostatic line search failed (relative residual {relative:.3g})")
    raise ValueError(f"Nonlinear magnetostatic solve did not converge in {max_iterations} iterations (relative residual {relative:.3g})")
