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
    NUMBA_AVAILABLE = True
except ImportError:
    cuda = None
    NUMBA_AVAILABLE = False
    prange = range

    def njit(*args, **kwargs):
        return lambda function: function


def canonical_rk4_step(
    x, tx, y, ty, h,
    g0, gm, g1, kx0, kxm, kx1, ky0, kym, ky1,
    hn0, hnm, hn1, hs0, hsm, hs1,
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
    apx = -(kx0 + g0 * g0) * x + g0 * py - hn0 * hu - hs0 * hv
    apy = -(ky0 + g0 * g0) * y - g0 * px + hn0 * hv - hs0 * hu

    bx, by = x + 0.5 * h * ax, y + 0.5 * h * ay
    bpx, bpy = px + 0.5 * h * apx, py + 0.5 * h * apy
    bx1, by1 = bpx + gm * by, bpy - gm * bx
    hu, hv = bx * bx - by * by, 2.0 * bx * by
    bpx1 = -(kxm + gm * gm) * bx + gm * bpy - hnm * hu - hsm * hv
    bpy1 = -(kym + gm * gm) * by - gm * bpx + hnm * hv - hsm * hu

    cx, cy = x + 0.5 * h * bx1, y + 0.5 * h * by1
    cpx, cpy = px + 0.5 * h * bpx1, py + 0.5 * h * bpy1
    cx1, cy1 = cpx + gm * cy, cpy - gm * cx
    hu, hv = cx * cx - cy * cy, 2.0 * cx * cy
    cpx1 = -(kxm + gm * gm) * cx + gm * cpy - hnm * hu - hsm * hv
    cpy1 = -(kym + gm * gm) * cy - gm * cpx + hnm * hv - hsm * hu

    dx, dy = x + h * cx1, y + h * cy1
    dpx, dpy = px + h * cpx1, py + h * cpy1
    dx1, dy1 = dpx + g1 * dy, dpy - g1 * dx
    hu, hv = dx * dx - dy * dy, 2.0 * dx * dy
    dpx1 = -(kx1 + g1 * g1) * dx + g1 * dpy - hn1 * hu - hs1 * hv
    dpy1 = -(ky1 + g1 * g1) * dy - g1 * dpx + hn1 * hv - hs1 * hu

    x = x + h * (ax + 2.0 * bx1 + 2.0 * cx1 + dx1) / 6.0
    y = y + h * (ay + 2.0 * by1 + 2.0 * cy1 + dy1) / 6.0
    px = px + h * (apx + 2.0 * bpx1 + 2.0 * cpx1 + dpx1) / 6.0
    py = py + h * (apy + 2.0 * bpy1 + 2.0 * cpy1 + dpy1) / 6.0
    return x, px + g1 * y, y, py - g1 * x


_canonical_step_numba = njit(cache=True, inline="always")(canonical_rk4_step)


@njit(cache=True, parallel=True)
def parallel_rk4(
    kx, ky, hn, hs, larmor_axis, inverse_momentum, cs_kick,
    thin_power, thin_rotation, step_m, x0, tx0, y0, ty0,
    kickx, kicky, save_index, checkpoint_index,
):
    nr, ns, nc = x0.size, save_index.size, checkpoint_index.size
    X = np.empty((ns, nr), np.float32)
    TX = np.empty((ns, nr), np.float32)
    Y = np.empty((ns, nr), np.float32)
    TY = np.empty((ns, nr), np.float32)
    CX = np.empty((nc, nr), np.float64)
    CTX = np.empty((nc, nr), np.float64)
    CY = np.empty((nc, nr), np.float64)
    CTY = np.empty((nc, nr), np.float64)
    for ray in prange(nr):
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
            if saved < ns and j == save_index[saved]:
                X[saved, ray], TX[saved, ray] = x, tx
                Y[saved, ray], TY[saved, ray] = y, ty
                saved += 1
            if captured < nc and j == checkpoint_index[captured]:
                CX[captured, ray], CTX[captured, ray] = x, tx
                CY[captured, ray], CTY[captured, ray] = y, ty
                captured += 1
            if j == step_m.size:
                continue
            a, b, c = 2 * j, 2 * j + 1, 2 * j + 2
            x, tx, y, ty = _canonical_step_numba(
                x, tx, y, ty, step_m[j],
                larmor_axis[a] * inv_p, larmor_axis[b] * inv_p,
                larmor_axis[c] * inv_p,
                kx[a], kx[b], kx[c], ky[a], ky[b], ky[c],
                hn[a], hn[b], hn[c], hs[a], hs[b], hs[c],
            )
    return X, TX, Y, TY, CX, CTX, CY, CTY


# The exact same kernel without thread-pool overhead for small tuning bundles.
# prange is an ordinary serial range when parallel=False; no equations differ.
serial_rk4 = (njit(cache=True, nogil=True)(parallel_rk4.py_func)
              if NUMBA_AVAILABLE else parallel_rk4)


def vectorised_rk4(
    kx, ky, hn, hs, larmor_axis, inverse_momentum, cs_kick,
    thin_power, thin_rotation, step_m, x, tx, y, ty,
    kickx, kicky, save_index, checkpoint_index,
):
    nr, ns, nc = x.size, save_index.size, checkpoint_index.size
    X, TX, Y, TY = (np.empty((ns, nr), np.float32) for _ in range(4))
    CX, CTX, CY, CTY = (np.empty((nc, nr), np.float64) for _ in range(4))
    saved, captured = 0, 0
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
            saved += 1
        if captured < nc and j == checkpoint_index[captured]:
            CX[captured], CTX[captured], CY[captured], CTY[captured] = x, tx, y, ty
            captured += 1
        if j == step_m.size:
            continue
        a, b, c = 2 * j, 2 * j + 1, 2 * j + 2
        x, tx, y, ty = canonical_rk4_step(
            x, tx, y, ty, step_m[j],
            larmor_axis[a] * inverse_momentum,
            larmor_axis[b] * inverse_momentum,
            larmor_axis[c] * inverse_momentum,
            kx[a], kx[b], kx[c], ky[a], ky[b], ky[c],
            hn[a], hn[b], hn[c], hs[a], hs[b], hs[c],
        )
    return X, TX, Y, TY, CX, CTX, CY, CTY


if NUMBA_AVAILABLE:
    _canonical_step_cuda = cuda.jit(device=True)(canonical_rk4_step)

    @cuda.jit
    def _cuda_rk4_kernel(
        kx, ky, hn, hs, larmor_axis, inverse_momentum, cs_kick,
        thin_power, thin_rotation, step_m, x0, tx0, y0, ty0,
        kickx, kicky, save_index, checkpoint_index,
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
            x, tx, y, ty = _canonical_step_cuda(
                x, tx, y, ty, step_m[j],
                larmor_axis[a] * inv_p, larmor_axis[b] * inv_p,
                larmor_axis[c] * inv_p,
                kx[a], kx[b], kx[c], ky[a], ky[b], ky[c],
                hn[a], hn[b], hn[c], hs[a], hs[b], hs[c],
            )
else:
    _cuda_rk4_kernel = None


def cuda_rk4(*inputs):
    if cuda is None or _cuda_rk4_kernel is None:
        raise RuntimeError("Numba CUDA support is not importable")
    ray_count = inputs[10].size
    device_inputs = [cuda.to_device(value) for value in inputs]
    outputs = [
        cuda.device_array((inputs[16].size, ray_count), dtype=np.float32)
        for _ in range(4)
    ]
    outputs.extend(
        cuda.device_array((inputs[17].size, ray_count), dtype=np.float64)
        for _ in range(4)
    )
    _cuda_rk4_kernel[(ray_count + 127) // 128, 128](*device_inputs, *outputs)
    cuda.synchronize()
    return tuple(output.copy_to_host() for output in outputs)
