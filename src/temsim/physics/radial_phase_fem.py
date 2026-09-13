"""Exact radial phase-coordinate terms for the cylindrical weak equation.

For psi=exp(i*gamma*r^2/2)*u, the physical gradient becomes
(grad+i*gamma*r*e_r)u. Retain both the antisymmetric derivative term and
the squared-gradient mass. This is a numerical gauge, not a source phase
fit, a lens correction, an electromagnetic gauge change or paraxial physics.
Coordinates use nm, hence gamma has units nm^-2.
"""
import math

import numpy as np
from scipy.sparse import coo_matrix


def radial_phase_terms(points, triangles):
    from temsim.physics.quadratic_axisymmetric_fem import triangle_basis
    p = np.asarray(points)[triangles]
    order = triangles.shape[1]
    if order not in (3, 6):
        raise ValueError("Radial phase coordinates require linear or quadratic triangles")
    linear = np.zeros((len(triangles), order, order))
    square = np.zeros_like(linear)
    nodes, weights = np.polynomial.legendre.leggauss(6)
    for a, wa in zip((nodes+1)/2, weights/2):
        for b, wb in zip((nodes+1)/2, weights/2):
            u, v = a*(1-b), a*b
            if order == 6:
                shape, derivative = triangle_basis(u, v)
            else:
                shape = np.array((1-u-v, u, v))
                derivative = np.array(((-1., -1.), (1., 0.), (0., 1.)))
            jacobian = np.einsum("tni,nj->tij", p, derivative)
            determinant = np.linalg.det(jacobian)
            if np.any(determinant <= 0):
                raise ValueError("Inverted radial phase element")
            grad = np.einsum("ni,tij->tnj", derivative, np.linalg.inv(jacobian))
            radius = p[:, :, 0]@shape
            measure = 2*math.pi*a*wa*wb*determinant*radius
            radial = grad[:, :, 0]
            linear += (radial[:, :, None]*shape[None, None, :]
                       -shape[None, :, None]*radial[:, None, :])*(measure*radius)[:, None, None]
            square += np.outer(shape, shape)[None]*(measure*radius**2)[:, None, None]
    rows = np.broadcast_to(triangles[:, :, None], linear.shape).ravel()
    cols = np.broadcast_to(triangles[:, None, :], linear.shape).ravel()
    return tuple(coo_matrix((v.ravel(), (rows, cols)), shape=(len(points), len(points))).tocsc()
                 for v in (linear, square))


def interface_gram(points, edges, trace):
    """FE trace inner product in the SAME numerical phase coordinates.

    A unitary analytic phase cancels in this mass product. A large defect
    diagnoses unresolved radial modes; it must not be repaired by amplitude
    renormalisation or a fitted downstream source.
    """
    from temsim.physics.surface_wave import robin_port, KINETIC_NM2_PER_EV
    mass = robin_port(points, edges, np.ones(len(points))/KINETIC_NM2_PER_EV)
    top = np.unique(edges)
    if trace.shape[0] != len(top):
        raise ValueError("Interface trace must follow the ordered boundary nodes")
    return trace.conj().T@(mass[top][:, top]@trace)
