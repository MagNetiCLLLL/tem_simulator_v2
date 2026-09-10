"""Independent discretizations and analytic fields for validation, not production.

The flux finite-volume solver below does not call the A-phi FEM mesh, assembly,
field extraction or interpolation implementation. Both prescribe zero remote
boundary potential; this agreement tests discretizations, not real materials.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

MU0 = 4*np.pi*1e-7


def finite_solenoid_axis(z, *, inner_m=.004, outer_m=.006, length_m=.01, ampere_turns=100):
    """Biot–Savart integral for a uniformly filled annular winding, in tesla."""
    points, weights = np.polynomial.legendre.leggauss(96)
    r = (inner_m+outer_m)/2 + points*(outer_m-inner_m)/2
    z = np.asarray(z)[..., None]
    a, b = z+length_m/2, z-length_m/2
    integrand = a/np.sqrt(r*r+a*a) - b/np.sqrt(r*r+b*b)
    current_density = ampere_turns / (length_m*(outer_m-inner_m))
    return MU0*current_density/2 * (outer_m-inner_m)/2 * (integrand @ weights)


def independent_flux_fvm(r, z, permeability, current):
    """Nonuniform finite volumes of -div(nu/r grad(psi))=J, psi=r*Aphi."""
    r, z = np.asarray(r), np.asarray(z)
    rr, zz = np.meshgrid(r[1:-1], z[1:-1], indexing="ij")
    ni, nj = rr.shape
    indices = np.arange(ni*nj).reshape(ni, nj)
    dr = (r[2:]-r[:-2])/2
    dz = (z[2:]-z[:-2])/2
    rows, columns, data = [], [], []
    diagonal = np.zeros_like(rr)
    for direction, sign in ((0,-1), (0,1), (1,-1), (1,1)):
        if direction == 0:
            other_r = r[:-2] if sign < 0 else r[2:]
            face_r = (r[1:-1]+other_r)/2
            face_r, face_z = np.meshgrid(face_r, z[1:-1], indexing="ij")
            delta = (other_r-r[1:-1])[:,None]
            resistance = sum(permeability(face_r+s*delta,face_z)*(face_r+s*delta) for s in (-.25,.25))/2
            conductance = dz[None,:] / (MU0*resistance*np.abs(delta))
        else:
            other_z = z[:-2] if sign < 0 else z[2:]
            face_r, face_z = np.meshgrid(r[1:-1], (z[1:-1]+other_z)/2, indexing="ij")
            delta = (other_z-z[1:-1])[None,:]
            resistance = sum(permeability(face_r,face_z+s*delta)*face_r for s in (-.25,.25))/2
            conductance = dr[:,None] / (MU0*resistance*np.abs(delta))
        diagonal += conductance
        i, j = np.indices(rr.shape)
        ip, jp = i + (sign if direction == 0 else 0), j + (sign if direction == 1 else 0)
        valid = (ip >= 0) & (ip < ni) & (jp >= 0) & (jp < nj)
        rows.extend(indices[valid].ravel())
        columns.extend(indices[ip[valid],jp[valid]].ravel())
        data.extend(-conductance[valid].ravel())
    rows.extend(indices.ravel()); columns.extend(indices.ravel()); data.extend(diagonal.ravel())
    matrix = coo_matrix((data,(rows,columns)), shape=(ni*nj, ni*nj)).tocsr()
    # Integrate the source over each dual cell; sampling a discontinuous coil
    # exactly on its edge would spuriously include a whole extra winding row.
    rhs = sum(current(rr+a*dr[:,None],zz+b*dz[None,:]) for a in (-.25,.25) for b in (-.25,.25))/4
    rhs *= dr[:,None] * dz[None,:]
    psi = np.zeros((len(r),len(z)))
    psi[1:-1,1:-1] = spsolve(matrix, rhs.ravel()).reshape(rr.shape)
    return 2*psi[1,:]/r[1]**2
