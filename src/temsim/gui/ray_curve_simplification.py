"""Optional single-thread compilation of a display-only RDP traversal.

No parallel pools, fast-math, physics state, or scientific sampling is used.
The caller keeps an independent NumPy fallback for unavailable compilation.
"""

import numpy as np


def _rdp_loop(xx, yy, limit_squared):
    count = len(xx)
    keep = np.zeros(count, dtype=np.bool_)
    keep[0] = keep[-1] = True
    first_stack = np.empty(count, dtype=np.int64)
    last_stack = np.empty(count, dtype=np.int64)
    first_stack[0], last_stack[0] = 0, count - 1
    pending = 1
    while pending:
        pending -= 1
        first, last = first_stack[pending], last_stack[pending]
        if last - first <= 1:
            continue
        dx, dy = xx[last] - xx[first], yy[last] - yy[first]
        length_squared = dx*dx + dy*dy
        maximum, split = -1.0, first + 1
        for index in range(first + 1, last):
            px, py = xx[index] - xx[first], yy[index] - yy[first]
            if length_squared > 0.0:
                fraction = min(1.0, max(0.0, (px*dx + py*dy) / length_squared))
                ex, ey = px - fraction*dx, py - fraction*dy
                distance_squared = ex*ex + ey*ey
            else:
                distance_squared = px*px + py*py
            if distance_squared > maximum:
                maximum, split = distance_squared, index
        if maximum > limit_squared:
            keep[split] = True
            first_stack[pending], last_stack[pending] = first, split
            first_stack[pending+1], last_stack[pending+1] = split, last
            pending += 2
    return np.flatnonzero(keep)


try:
    from numba import njit
except ImportError:
    compiled_rdp = None
else:
    compiled_rdp = njit(cache=True, fastmath=False)(_rdp_loop)
