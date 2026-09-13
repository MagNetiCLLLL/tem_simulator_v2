"""Quadratic, body-fitted cylindrical finite elements in nanometres.

The radial measure is 2*pi*r. Geometry, potential, amplitude and boundary
traces use the same six-node triangle/three-node edge basis. These operators
do not define an electron source or a downstream matching condition.
"""
import math

import numpy as np
from scipy.sparse import coo_matrix


def mesh(radius_nm, bottom_nm, top_nm, axial_nodes):
    r, bottom = np.asarray(radius_nm, float), np.asarray(bottom_nm, float)
    if (r.ndim != 1 or len(r) < 3 or len(r) % 2 != 1 or r[0] != 0
            or np.any(np.diff(r) <= 0) or not np.all(np.isfinite(r))
            or bottom.shape != r.shape or not np.all(np.isfinite(bottom))
            or type(axial_nodes) is not int or axial_nodes < 3 or axial_nodes % 2 != 1
            or not math.isfinite(top_nm) or np.any(bottom >= top_nm)):
        raise ValueError("Quadratic cylindrical mesh requires odd, ordered node counts and finite geometry")
    z = bottom[:, None] + (top_nm-bottom[:, None])*np.linspace(0., 1., axial_nodes)
    points = np.column_stack((np.broadcast_to(r[:, None], z.shape).ravel(), z.ravel()))
    ids = np.arange(len(points)).reshape(z.shape)
    a, b, c, d = (ids[:-2:2, :-2:2].ravel(), ids[2::2, :-2:2].ravel(),
                   ids[2::2, 2::2].ravel(), ids[:-2:2, 2::2].ravel())
    ab, bc, cd, da, ac = (ids[1::2, :-2:2].ravel(), ids[2::2, 1::2].ravel(),
                          ids[1::2, 2::2].ravel(), ids[:-2:2, 1::2].ravel(),
                          ids[1::2, 1::2].ravel())
    triangles = np.vstack((np.column_stack((a, b, c, ab, bc, ac)),
                           np.column_stack((a, c, d, ac, cd, da))))
    edges = {"source": np.column_stack((ids[:-2:2, 0], ids[2::2, 0], ids[1::2, 0])),
             "top": np.column_stack((ids[:-2:2, -1], ids[2::2, -1], ids[1::2, -1])),
             "side": np.column_stack((ids[-1, :-2:2], ids[-1, 2::2], ids[-1, 1::2]))}
    return points, triangles, edges, z


def triangle_basis(u, v):
    l = np.array((1-u-v, u, v))
    dl = np.array(((-1., -1.), (1., 0.), (0., 1.)))
    shape = np.r_[l*(2*l-1), 4*l[[0, 1, 2]]*l[[1, 2, 0]]]
    derivative = np.vstack(((4*l-1)[:, None]*dl,
        4*(l[[0, 1, 2], None]*dl[[1, 2, 0]] + l[[1, 2, 0], None]*dl[[0, 1, 2]])))
    return shape, derivative


def _assemble(values, nodes, count):
    return coo_matrix((values.ravel(),
        (np.broadcast_to(nodes[:, :, None], values.shape).ravel(),
         np.broadcast_to(nodes[:, None, :], values.shape).ravel())),
         shape=(count, count)).tocsc()


def volume(points, triangles, nodal_potential):
    """Real symmetric stiffness, mass and potential matrices (no lumping)."""
    p, potential = np.asarray(points)[triangles], np.asarray(nodal_potential)[triangles]
    matrices = [np.zeros((len(triangles), 6, 6)) for _ in range(3)]
    nodes, weights = np.polynomial.legendre.leggauss(6)
    for a, wa in zip((nodes+1)/2, weights/2):
        for b, wb in zip((nodes+1)/2, weights/2):
            shape, derivative = triangle_basis(a*(1-b), a*b)
            jacobian = np.einsum("tni,nj->tij", p, derivative)
            determinant = np.linalg.det(jacobian)
            if np.any(determinant <= 0):
                raise ValueError("Inverted quadratic cylindrical element")
            grad = np.einsum("ni,tij->tnj", derivative, np.linalg.inv(jacobian))
            factor = 2*math.pi*a*wa*wb*determinant*(p[:, :, 0]@shape)
            matrices[0] += np.einsum("tij,tkj->tik", grad, grad)*factor[:, None, None]
            mass = np.outer(shape, shape)[None]*factor[:, None, None]
            matrices[1] += mass
            matrices[2] += mass*(potential@shape)[:, None, None]
    return tuple(_assemble(value, triangles, len(points)) for value in matrices)


def boundary(points, edges, nodal_wave_number):
    """Boundary flux form for a prescribed real local outgoing wavenumber."""
    p, wave_number = np.asarray(points)[edges], np.asarray(nodal_wave_number)[edges]
    matrix = np.zeros((len(edges), 3, 3))
    nodes, weights = np.polynomial.legendre.leggauss(8)
    for u, w in zip((nodes+1)/2, weights/2):
        basis = np.array(((1-u)*(1-2*u), u*(2*u-1), 4*u*(1-u)))
        derivative = np.array((4*u-3, 4*u-1, 4-8*u))
        radius = p[:, :, 0]@basis
        k = wave_number@basis
        if np.any(k < 0) or not np.all(np.isfinite(k)):
            raise ValueError("Quadratic boundary wavenumber must be finite and nonnegative")
        length = np.linalg.norm(np.einsum("tni,n->ti", p, derivative), axis=1)
        matrix += (2*math.pi*w*radius*k*length)[:, None, None]*np.outer(basis, basis)
    return _assemble(matrix, edges, len(points))
