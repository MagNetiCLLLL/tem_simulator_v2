"""CPU reference transport for imported vector fields in the column.

Coordinates are metres and mechanical slopes dx/dz, dy/dz.  Unmapped round
lenses retain the existing paraxial canonical equations.  Imported fields
enter once through q/p * sqrt(1+tx**2+ty**2) times the full Lorentz slope
equations, evaluated at each ray's actual XYZ at every RK stage.  The magnetic
term uses constant relativistic momentum magnitude: a static B does no work.
Z is the independent variable, so this solver covers forward column rays;
turning trajectories must be handled by a time-domain particle solver.
"""

import math
import numpy as np

from temsim.physics.ray_integrator import canonical_rk4_step


def vector_map_rk4(
    kx, ky, hn, hs, larmor_axis, inverse_momentum, cs_kick,
    thin_power, thin_rotation, step_m, x, tx, y, ty,
    kickx, kicky, save_index, checkpoint_index, *, z_mm, mapped_fields,
):
    nr, ns, nc = x.size, save_index.size, checkpoint_index.size
    # Finite-difference transfer Jacobians need float64 even in saved history.
    X, TX, Y, TY = (np.empty((ns, nr), np.float64) for _ in range(4))
    CX, CTX, CY, CTY = (np.empty((nc, nr), np.float64) for _ in range(4))
    saved = captured = 0
    charge_over_p = -1.602176634e-19 * inverse_momentum
    supports = tuple((item, *item.field_map.field_support_mm)
                     for item in mapped_fields if item.scale != 0.0)
    for j in range(step_m.size + 1):
        tx, ty = tx - thin_power[j] * x, ty - thin_power[j] * y
        if thin_rotation[j] != 0.0:
            co, si = math.cos(thin_rotation[j]), math.sin(thin_rotation[j])
            x, y = co * x - si * y, si * x + co * y
            tx, ty = co * tx - si * ty, si * tx + co * ty
        tx, ty = tx + kickx[j], ty + kicky[j]
        radial = cs_kick[j] * (x*x + y*y)
        tx, ty = tx - radial*x, ty - radial*y
        if saved < ns and j == save_index[saved]:
            X[saved], TX[saved], Y[saved], TY[saved] = x, tx, y, ty
            saved += 1
        if captured < nc and j == checkpoint_index[captured]:
            CX[captured], CTX[captured], CY[captured], CTY[captured] = x, tx, y, ty
            captured += 1
        if j == step_m.size:
            continue
        a, b, c = 2*j, 2*j+1, 2*j+2
        g = larmor_axis[[a,b,c], None] * inverse_momentum[None, :]
        active = tuple(item for item, lower, upper in supports
                       if lower < z_mm[j+1] and upper > z_mm[j])
        if not active:
            x, tx, y, ty = canonical_rk4_step(
                x, tx, y, ty, step_m[j], *g,
                kx[a], kx[b], kx[c], ky[a], ky[b], ky[c],
                hn[a], hn[b], hn[c], hs[a], hs[b], hs[c],
            )
            continue

        def derivative(values, stage, axial_mm):
            xx, px, yy, py = values
            gg = g[stage]
            ux, uy = px + gg*yy, py - gg*xx
            pos = np.column_stack((xx, yy, np.full(nr, axial_mm*1e-3)))
            field = sum((item.field_at_global_positions_t(pos) for item in active),
                        start=np.zeros((nr,3)))
            bx, by, bz = field.T
            factor = charge_over_p * np.sqrt(1.0 + ux*ux + uy*uy)
            fx = factor * (uy*bz - (1.0+ux*ux)*by + ux*uy*bx)
            fy = factor * ((1.0+uy*uy)*bx - ux*bz - ux*uy*by)
            index = (a,b,c)[stage]
            hu, hv = xx*xx - yy*yy, 2.0*xx*yy
            return np.array((ux,
                -(kx[index]+gg*gg)*xx + gg*py - hn[index]*hu - hs[index]*hv + fx,
                uy,
                -(ky[index]+gg*gg)*yy - gg*px + hn[index]*hv - hs[index]*hu + fy))

        initial = np.array((x, tx-g[0]*y, y, ty+g[0]*x))
        h = float(step_m[j])
        # A map is zero outside its volume. Sample boundary nodes from the
        # interval interior to avoid giving a discontinuous edge finite weight
        # in the neighbouring vacuum interval.
        za = np.nextafter(float(z_mm[j]), float(z_mm[j+1]))
        zc = np.nextafter(float(z_mm[j+1]), float(z_mm[j]))
        zm = 0.5*(float(z_mm[j])+float(z_mm[j+1]))
        k1 = derivative(initial, 0, za)
        k2 = derivative(initial+0.5*h*k1, 1, zm)
        k3 = derivative(initial+0.5*h*k2, 1, zm)
        k4 = derivative(initial+h*k3, 2, zc)
        xx, px, yy, py = initial + h*(k1+2*k2+2*k3+k4)/6.0
        x, tx, y, ty = xx, px+g[2]*yy, yy, py-g[2]*xx
        if (not np.all(np.isfinite((x,tx,y,ty)))
                or np.any(np.hypot(tx,ty) > 100.0)):
            raise ValueError("Vector field trajectory leaves the forward-Z domain; reduce the step or use time-domain transport")
    return X, TX, Y, TY, CX, CTX, CY, CTY
