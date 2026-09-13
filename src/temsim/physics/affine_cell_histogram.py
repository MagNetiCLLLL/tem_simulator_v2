"""Conservative display of cell-integrated values on an affine wave lattice.

Each native cell has constant display density. Polygon/rectangle overlap
redistributes its integral, without smoothing, point splatting, phase changes,
or restoring cropped current. This display does not refine the physical wave.
"""
import logging
import math

import numpy as np


def _overlap(vertices, ix, iy):
    # Shift before the area sum to avoid cancellation far from the origin.
    a, b = np.empty((12, 2)), np.empty((12, 2))
    for j in range(4):
        a[j, 0], a[j, 1] = vertices[j, 0]-ix, vertices[j, 1]-iy
    count = 4
    for edge in range(4):
        axis, bound = edge//2, float(edge % 2)
        sign = 1. if edge % 2 == 0 else -1.
        used = 0
        for j in range(count):
            previous = (j+count-1) % count
            d0, d1 = sign*(a[previous, axis]-bound), sign*(a[j, axis]-bound)
            if (d0 >= 0.) != (d1 >= 0.):
                fraction = d0/(d0-d1)
                b[used, 0] = a[previous, 0]+fraction*(a[j, 0]-a[previous, 0])
                b[used, 1] = a[previous, 1]+fraction*(a[j, 1]-a[previous, 1])
                used += 1
            if d1 >= 0.:
                b[used] = a[j]
                used += 1
        a, b, count = b, a, used
        if count == 0:
            return 0.
    area = 0.
    for j in range(count):
        following = (j+1) % count
        area += a[j, 0]*a[following, 1]-a[j, 1]*a[following, 0]
    return abs(area)*.5


def _deposit(weights, matrix, start, nx, ny, area_function):
    output = np.zeros((nx, ny))
    area = abs(matrix[0, 0]*matrix[1, 1]-matrix[0, 1]*matrix[1, 0])
    offsets = np.empty((4, 2))
    for j, (x, y) in enumerate(((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5))):
        offsets[j, 0] = matrix[0, 0]*x+matrix[0, 1]*y
        offsets[j, 1] = matrix[1, 0]*x+matrix[1, 1]*y
    extent = np.abs(matrix).sum(axis=1)*.5
    vertices = np.empty((4, 2))
    for iy in range(weights.shape[0]):
        for ix in range(weights.shape[1]):
            weight = weights[iy, ix]
            if weight == 0.:
                continue
            x = start[0]+matrix[0, 0]*ix+matrix[0, 1]*iy
            y = start[1]+matrix[1, 0]*ix+matrix[1, 1]*iy
            lo_x, hi_x = math.floor(x-extent[0]), math.floor(x+extent[0])
            lo_y, hi_y = math.floor(y-extent[1]), math.floor(y+extent[1])
            if hi_x < 0 or lo_x >= nx or hi_y < 0 or lo_y >= ny:
                continue
            if lo_x == hi_x and lo_y == hi_y:
                output[lo_x, lo_y] += weight
                continue
            for j in range(4):
                vertices[j, 0], vertices[j, 1] = x+offsets[j, 0], y+offsets[j, 1]
            for bx in range(max(0, lo_x), min(nx-1, hi_x)+1):
                for by in range(max(0, lo_y), min(ny-1, hi_y)+1):
                    output[bx, by] += weight*area_function(vertices, bx, by)/area
    return output


_COMPILED = None
_COMPILE_UNAVAILABLE = False


def affine_cell_histogram(weights, basis, origin, bounds, bins=128, *, backend="auto"):
    """Return X/Y bin integrals and a display-only execution record.

    weights is (Y,X); origin is the centre at index (ny//2,nx//2).
    basis columns are X/Y cell vectors, in the same units as bounds.
    CPU/Numba use the same clipping algorithm; no CUDA transfer is needed.
    """
    global _COMPILED, _COMPILE_UNAVAILABLE
    weights, basis, origin, bounds = (np.asarray(value, float) for value in (weights, basis, origin, bounds))
    if (weights.ndim != 2 or 0 in weights.shape or basis.shape != (2, 2)
            or origin.shape != (2,) or bounds.shape != (2, 2)
            or any(not np.all(np.isfinite(a)) for a in (weights, basis, origin, bounds))
            or np.any(bounds[:, 1] <= bounds[:, 0])):
        raise ValueError("Affine current display needs finite cell weights, geometry and ordered bounds")
    if isinstance(bins, int) and not isinstance(bins, bool):
        bins = (bins, bins)
    if not isinstance(bins, tuple) or len(bins) != 2 or any(type(n) is not int or not 1 <= n <= 8192 for n in bins):
        raise ValueError("Display bins must be positive integer dimensions, at most 8192")
    if backend not in ("auto", "python", "numba"):
        raise ValueError("Display backend must be auto, python or numba")
    scale = np.asarray(bins)/(bounds[:, 1]-bounds[:, 0])
    matrix = basis*scale[:, None]
    start = (origin-basis@np.array((weights.shape[1]//2, weights.shape[0]//2))-bounds[:, 0])*scale
    if not np.isfinite(np.linalg.det(matrix)) or abs(np.linalg.det(matrix)) <= np.finfo(float).tiny:
        raise ValueError("Affine current display needs a nonsingular cell basis")
    if backend != "python" and _COMPILED is None and not _COMPILE_UNAVAILABLE:
        try:
            from numba import njit
            _COMPILED = (njit(cache=True)(_deposit), njit(cache=True)(_overlap))
        except ImportError:
            _COMPILE_UNAVAILABLE = True
    actual = "python"
    if backend != "python" and _COMPILED is not None:
        try:
            result = _COMPILED[0](weights, matrix, start, *bins, _COMPILED[1])
            actual = "numba"
        except Exception:
            if backend == "numba":
                raise
            logging.getLogger(__name__).warning("Compiled display unavailable; using identical CPU clipping", exc_info=True)
            _COMPILED, _COMPILE_UNAVAILABLE = None, True
    elif backend == "numba":
        raise RuntimeError("Numba display backend is unavailable")
    if actual == "python":
        result = _deposit(weights, matrix, start, *bins, _overlap)
    return result, {"backend": actual, "method": "piecewise-constant affine cell overlap",
                    "scope": "display only; physical wave and cropped current are not renormalised"}
