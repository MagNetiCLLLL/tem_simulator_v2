"""Fourth-order post-gun transport in canonical transverse coordinates.

Continuous coefficient arrays interleave exact nodes and exact midpoints.
The public phase space is always (x, theta_x, y, theta_y); only an individual
RK4 interval uses px = theta_x - g*y and py = theta_y + g*x.  This removes
the numerical derivative of the magnetic field from the ray equations.
"""

import math

import numpy as np

try:
    from numba import cuda, njit, prange
    from numba.extending import register_jitable
    NUMBA_AVAILABLE = True
except ImportError:
    cuda = None
    NUMBA_AVAILABLE = False
    prange = range

    def njit(*args, **kwargs):
        return lambda function: function

    def register_jitable(*args, **kwargs):
        return lambda function: function


@register_jitable(inline="always")
def _canonical_rk4_stages(
    x, tx, y, ty, h,
    g0, gm, g1, kx0, kxm, kx1, ky0, kym, ky1,
    hn0, hnm, hn1, hs0, hsm, hs1, kxy0=0.0, kxym=0.0, kxy1=0.0,
):
    """Advance one interval with exact coefficient values at all RK stages.

    This scalar/array arithmetic is shared by NumPy, Numba and CUDA.  The
    canonical transformation is valid at nonzero-field source/checkpoint
    planes too; returning slopes preserves after-kick checkpoint semantics.
    """
    px = tx - g0 * y
    py = ty + g0 * x

    ax = px + g0 * y
    ay = py - g0 * x
    hu, hv = x * x - y * y, 2.0 * x * y
    apx = -(kx0 + g0 * g0) * x - kxy0 * y + g0 * py - hn0 * hu - hs0 * hv
    apy = -(ky0 + g0 * g0) * y - kxy0 * x - g0 * px + hn0 * hv - hs0 * hu

    bx, by = x + 0.5 * h * ax, y + 0.5 * h * ay
    bpx, bpy = px + 0.5 * h * apx, py + 0.5 * h * apy
    bx1, by1 = bpx + gm * by, bpy - gm * bx
    hu, hv = bx * bx - by * by, 2.0 * bx * by
    bpx1 = -(kxm + gm * gm) * bx - kxym * by + gm * bpy - hnm * hu - hsm * hv
    bpy1 = -(kym + gm * gm) * by - kxym * bx - gm * bpx + hnm * hv - hsm * hu

    cx, cy = x + 0.5 * h * bx1, y + 0.5 * h * by1
    cpx, cpy = px + 0.5 * h * bpx1, py + 0.5 * h * bpy1
    cx1, cy1 = cpx + gm * cy, cpy - gm * cx
    hu, hv = cx * cx - cy * cy, 2.0 * cx * cy
    cpx1 = -(kxm + gm * gm) * cx - kxym * cy + gm * cpy - hnm * hu - hsm * hv
    cpy1 = -(kym + gm * gm) * cy - kxym * cx - gm * cpx + hnm * hv - hsm * hu

    dx, dy = x + h * cx1, y + h * cy1
    dpx, dpy = px + h * cpx1, py + h * cpy1
    dx1, dy1 = dpx + g1 * dy, dpy - g1 * dx
    hu, hv = dx * dx - dy * dy, 2.0 * dx * dy
    dpx1 = -(kx1 + g1 * g1) * dx - kxy1 * dy + g1 * dpy - hn1 * hu - hs1 * hv
    dpy1 = -(ky1 + g1 * g1) * dy - kxy1 * dx - g1 * dpx + hn1 * hv - hs1 * hu

    x = x + h * (ax + 2.0 * bx1 + 2.0 * cx1 + dx1) / 6.0
    y = y + h * (ay + 2.0 * by1 + 2.0 * cy1 + dy1) / 6.0
    px = px + h * (apx + 2.0 * bpx1 + 2.0 * cpx1 + dpx1) / 6.0
    py = py + h * (apy + 2.0 * bpy1 + 2.0 * cpy1 + dpy1) / 6.0
    return x, px + g1 * y, y, py - g1 * x, ax, ay, bx1, by1, cx1, cy1, dx1, dy1


def canonical_rk4_step(
    x, tx, y, ty, h,
    g0, gm, g1, kx0, kxm, kx1, ky0, kym, ky1,
    hn0, hnm, hn1, hs0, hsm, hs1, kxy0=0.0, kxym=0.0, kxy1=0.0,
):
    """Existing ray-only API, using the same four RK stage states."""
    return _canonical_rk4_stages(
        x, tx, y, ty, h, g0, gm, g1, kx0, kxm, kx1, ky0, kym, ky1,
        hn0, hnm, hn1, hs0, hsm, hs1, kxy0, kxym, kxy1,
    )[:4]


@register_jitable(inline="always")
def _rk4_flight_increment(h, inverse_speed, ax, ay, bx, by, cx, cy, dx, dy):
    """Integrate dt/dz from the mechanical slopes at the executed RK stages."""
    return h*inverse_speed*(
        (1.+ax*ax+ay*ay)**.5 + 2.*(1.+bx*bx+by*by)**.5
        + 2.*(1.+cx*cx+cy*cy)**.5 + (1.+dx*dx+dy*dy)**.5
    )/6.


def canonical_rk4_step_with_time(
    x, tx, y, ty, h,
    g0, gm, g1, kx0, kxm, kx1, ky0, kym, ky1,
    hn0, hnm, hn1, hs0, hsm, hs1, kxy0, kxym, kxy1, inverse_speed,
):
    values = _canonical_rk4_stages(
        x, tx, y, ty, h, g0, gm, g1, kx0, kxm, kx1, ky0, kym, ky1,
        hn0, hnm, hn1, hs0, hsm, hs1, kxy0, kxym, kxy1,
    )
    elapsed = _rk4_flight_increment(h, inverse_speed, values[4], values[5],
        values[6], values[7], values[8], values[9], values[10], values[11])
    return values[0], values[1], values[2], values[3], elapsed


_canonical_step_numba = njit(cache=True, inline="always")(canonical_rk4_step)
_canonical_time_numba = njit(cache=True, inline="always")(canonical_rk4_step_with_time)


@njit(cache=True, parallel=True)
def _parallel_rk4(
    kx, ky, hn, hs, larmor_axis, inverse_momentum, cs_kick,
    thin_power, thin_rotation, step_m, x0, tx0, y0, ty0,
    kickx, kicky, save_index, checkpoint_index, kxy=None,
    initial_time_s=None, inverse_speed=None,
):
    if kxy is None:
        kxy = np.zeros_like(kx)
    nr, ns, nc = x0.size, save_index.size, checkpoint_index.size
    X = np.empty((ns, nr), np.float32)
    TX = np.empty((ns, nr), np.float32)
    Y = np.empty((ns, nr), np.float32)
    TY = np.empty((ns, nr), np.float32)
    CX = np.empty((nc, nr), np.float64)
    CTX = np.empty((nc, nr), np.float64)
    CY = np.empty((nc, nr), np.float64)
    CTY = np.empty((nc, nr), np.float64)
    T = np.empty((ns, nr), np.float64) if initial_time_s is not None else np.empty((0, 0), np.float64)
    CT = np.empty((nc, nr), np.float64) if initial_time_s is not None else np.empty((0, 0), np.float64)
    for ray in prange(nr):
        x, tx, y, ty = x0[ray], tx0[ray], y0[ray], ty0[ray]
        saved, captured = 0, 0
        inv_p = inverse_momentum[ray]
        time = initial_time_s[ray] if initial_time_s is not None else 0.
        for j in range(step_m.size + 1):
            tx -= thin_power[j] * x
            ty -= thin_power[j] * y
            if thin_rotation[j] != 0.0:
                co, si = math.cos(thin_rotation[j]), math.sin(thin_rotation[j])
                x, y = co * x - si * y, si * x + co * y
                tx, ty = co * tx - si * ty, si * tx + co * ty
            tx += kickx[j]
            ty += kicky[j]
            if cs_kick[j] != 0.0:
                radial = cs_kick[j] * (x * x + y * y)
                tx -= radial * x
                ty -= radial * y
            if saved < ns and j == save_index[saved]:
                X[saved, ray], TX[saved, ray] = x, tx
                Y[saved, ray], TY[saved, ray] = y, ty
                if initial_time_s is not None:
                    T[saved, ray] = time
                saved += 1
            if captured < nc and j == checkpoint_index[captured]:
                CX[captured, ray], CTX[captured, ray] = x, tx
                CY[captured, ray], CTY[captured, ray] = y, ty
                if initial_time_s is not None:
                    CT[captured, ray] = time
                captured += 1
            if j == step_m.size:
                continue
            a, b, c = 2 * j, 2 * j + 1, 2 * j + 2
            if initial_time_s is not None:
                x, tx, y, ty, elapsed = _canonical_time_numba(
                    x, tx, y, ty, step_m[j],
                    larmor_axis[a]*inv_p, larmor_axis[b]*inv_p, larmor_axis[c]*inv_p,
                    kx[a], kx[b], kx[c], ky[a], ky[b], ky[c],
                    hn[a], hn[b], hn[c], hs[a], hs[b], hs[c],
                    kxy[a], kxy[b], kxy[c], inverse_speed[ray],
                )
                time += elapsed
                continue
            x, tx, y, ty = _canonical_step_numba(
                x, tx, y, ty, step_m[j],
                larmor_axis[a] * inv_p, larmor_axis[b] * inv_p,
                larmor_axis[c] * inv_p,
                kx[a], kx[b], kx[c], ky[a], ky[b], ky[c],
                hn[a], hn[b], hn[c], hs[a], hs[b], hs[c],
                kxy[a], kxy[b], kxy[c],
            )
    return X, TX, Y, TY, CX, CTX, CY, CTY, T, CT


# The exact same kernel without thread-pool overhead for small tuning bundles.
# prange is an ordinary serial range when parallel=False; no equations differ.
_serial_rk4 = (njit(cache=True, nogil=True)(_parallel_rk4.py_func)
               if NUMBA_AVAILABLE else _parallel_rk4)


def parallel_rk4(*inputs, initial_time_s=None, inverse_speed=None):
    result = _parallel_rk4(*inputs, initial_time_s=initial_time_s, inverse_speed=inverse_speed)
    return result if initial_time_s is not None else result[:8]


def serial_rk4(*inputs, initial_time_s=None, inverse_speed=None):
    result = _serial_rk4(*inputs, initial_time_s=initial_time_s, inverse_speed=inverse_speed)
    return result if initial_time_s is not None else result[:8]


def vectorised_rk4(
    kx, ky, hn, hs, larmor_axis, inverse_momentum, cs_kick,
    thin_power, thin_rotation, step_m, x, tx, y, ty,
    kickx, kicky, save_index, checkpoint_index, kxy=None, *, step_operator=None,
    initial_time_s=None, inverse_speed=None,
):
    if kxy is None:
        kxy = np.zeros_like(kx)
    nr, ns, nc = x.size, save_index.size, checkpoint_index.size
    X, TX, Y, TY = (np.empty((ns, nr), np.float32) for _ in range(4))
    CX, CTX, CY, CTY = (np.empty((nc, nr), np.float64) for _ in range(4))
    saved, captured = 0, 0
    time = np.array(initial_time_s, copy=True) if initial_time_s is not None else None
    T = np.empty((ns, nr), np.float64) if time is not None else None
    CT = np.empty((nc, nr), np.float64) if time is not None else None
    for j in range(step_m.size + 1):
        tx = tx - thin_power[j] * x
        ty = ty - thin_power[j] * y
        if thin_rotation[j] != 0.0:
            co, si = math.cos(thin_rotation[j]), math.sin(thin_rotation[j])
            x, y = co * x - si * y, si * x + co * y
            tx, ty = co * tx - si * ty, si * tx + co * ty
        tx = tx + kickx[j]
        ty = ty + kicky[j]
        if cs_kick[j] != 0.0:
            radial = cs_kick[j] * (x * x + y * y)
            tx = tx - radial * x
            ty = ty - radial * y
        if saved < ns and j == save_index[saved]:
            X[saved], TX[saved], Y[saved], TY[saved] = x, tx, y, ty
            if time is not None:
                T[saved] = time
            saved += 1
        if captured < nc and j == checkpoint_index[captured]:
            CX[captured], CTX[captured], CY[captured], CTY[captured] = x, tx, y, ty
            if time is not None:
                CT[captured] = time
            captured += 1
        if j == step_m.size:
            continue
        a, b, c = 2 * j, 2 * j + 1, 2 * j + 2
        before = (x.copy(), tx.copy(), y.copy(), ty.copy()) if step_operator is not None else None
        step = canonical_rk4_step_with_time if time is not None else canonical_rk4_step
        result = step(
            x, tx, y, ty, step_m[j],
            larmor_axis[a] * inverse_momentum,
            larmor_axis[b] * inverse_momentum,
            larmor_axis[c] * inverse_momentum,
            kx[a], kx[b], kx[c], ky[a], ky[b], ky[c],
            hn[a], hn[b], hn[c], hs[a], hs[b], hs[c],
            kxy[a], kxy[b], kxy[c],
            *((inverse_speed,) if time is not None else ()),
        )
        x, tx, y, ty = result[:4]
        if time is not None:
            time += result[4]
        if step_operator is not None:
            x, tx, y, ty = step_operator(j, before, (x, tx, y, ty))
    result = X, TX, Y, TY, CX, CTX, CY, CTY
    return (*result, T, CT) if time is not None else result


if NUMBA_AVAILABLE:
    _canonical_stages_cuda = cuda.jit(device=True)(_canonical_rk4_stages)
    _rk4_flight_cuda = cuda.jit(device=True)(_rk4_flight_increment)

    @cuda.jit
    def _cuda_rk4_kernel(
        kx, ky, hn, hs, larmor_axis, inverse_momentum, cs_kick,
        thin_power, thin_rotation, step_m, x0, tx0, y0, ty0,
        kickx, kicky, save_index, checkpoint_index, kxy,
        X, TX, Y, TY, CX, CTX, CY, CTY,
    ):
        ray = cuda.grid(1)
        if ray >= x0.size:
            return
        x, tx, y, ty = x0[ray], tx0[ray], y0[ray], ty0[ray]
        saved, captured = 0, 0
        inv_p = inverse_momentum[ray]
        for j in range(step_m.size + 1):
            tx -= thin_power[j] * x
            ty -= thin_power[j] * y
            if thin_rotation[j] != 0.0:
                co, si = math.cos(thin_rotation[j]), math.sin(thin_rotation[j])
                x, y = co * x - si * y, si * x + co * y
                tx, ty = co * tx - si * ty, si * tx + co * ty
            tx += kickx[j]
            ty += kicky[j]
            if cs_kick[j] != 0.0:
                radial = cs_kick[j] * (x * x + y * y)
                tx -= radial * x
                ty -= radial * y
            if saved < save_index.size and j == save_index[saved]:
                X[saved, ray], TX[saved, ray] = x, tx
                Y[saved, ray], TY[saved, ray] = y, ty
                saved += 1
            if captured < checkpoint_index.size and j == checkpoint_index[captured]:
                CX[captured, ray], CTX[captured, ray] = x, tx
                CY[captured, ray], CTY[captured, ray] = y, ty
                captured += 1
            if j == step_m.size:
                continue
            a, b, c = 2 * j, 2 * j + 1, 2 * j + 2
            values = _canonical_stages_cuda(
                x, tx, y, ty, step_m[j],
                larmor_axis[a] * inv_p, larmor_axis[b] * inv_p,
                larmor_axis[c] * inv_p,
                kx[a], kx[b], kx[c], ky[a], ky[b], ky[c],
                hn[a], hn[b], hn[c], hs[a], hs[b], hs[c],
                kxy[a], kxy[b], kxy[c],
            )
            x, tx, y, ty = values[:4]

    @cuda.jit
    def _cuda_time_rk4_kernel(
        kx, ky, hn, hs, larmor_axis, inverse_momentum, cs_kick,
        thin_power, thin_rotation, step_m, x0, tx0, y0, ty0,
        kickx, kicky, save_index, checkpoint_index, kxy, initial_time_s, inverse_speed,
        X, TX, Y, TY, CX, CTX, CY, CTY, T, CT,
    ):
        ray = cuda.grid(1)
        if ray >= x0.size:
            return
        x, tx, y, ty = x0[ray], tx0[ray], y0[ray], ty0[ray]
        time = initial_time_s[ray]
        inv_p, inv_v = inverse_momentum[ray], inverse_speed[ray]
        saved, captured = 0, 0
        for j in range(step_m.size+1):
            tx -= thin_power[j]*x
            ty -= thin_power[j]*y
            if thin_rotation[j] != 0.:
                co, si = math.cos(thin_rotation[j]), math.sin(thin_rotation[j])
                x, y = co*x-si*y, si*x+co*y
                tx, ty = co*tx-si*ty, si*tx+co*ty
            tx += kickx[j]
            ty += kicky[j]
            if cs_kick[j] != 0.:
                radial = cs_kick[j]*(x*x+y*y)
                tx -= radial*x
                ty -= radial*y
            if saved < save_index.size and j == save_index[saved]:
                X[saved, ray], TX[saved, ray] = x, tx
                Y[saved, ray], TY[saved, ray] = y, ty
                T[saved, ray] = time
                saved += 1
            if captured < checkpoint_index.size and j == checkpoint_index[captured]:
                CX[captured, ray], CTX[captured, ray] = x, tx
                CY[captured, ray], CTY[captured, ray] = y, ty
                CT[captured, ray] = time
                captured += 1
            if j == step_m.size:
                continue
            a, b, c = 2*j, 2*j+1, 2*j+2
            values = _canonical_stages_cuda(
                x, tx, y, ty, step_m[j],
                larmor_axis[a]*inv_p, larmor_axis[b]*inv_p, larmor_axis[c]*inv_p,
                kx[a], kx[b], kx[c], ky[a], ky[b], ky[c],
                hn[a], hn[b], hn[c], hs[a], hs[b], hs[c],
                kxy[a], kxy[b], kxy[c],
            )
            x, tx, y, ty = values[:4]
            time += _rk4_flight_cuda(step_m[j], inv_v, values[4], values[5],
                values[6], values[7], values[8], values[9], values[10], values[11])
else:
    _cuda_rk4_kernel = None
    _cuda_time_rk4_kernel = None


def cuda_rk4(*inputs, initial_time_s=None, inverse_speed=None):
    if cuda is None or _cuda_rk4_kernel is None:
        raise RuntimeError("Numba CUDA support is not importable")
    from temsim.physics.ray_device_cache import DEVICE_CACHE
    if initial_time_s is not None:
        return DEVICE_CACHE.execute(cuda, _cuda_time_rk4_kernel,
                                    (*inputs, initial_time_s, inverse_speed))
    return DEVICE_CACHE.execute(cuda, _cuda_rk4_kernel, inputs)
