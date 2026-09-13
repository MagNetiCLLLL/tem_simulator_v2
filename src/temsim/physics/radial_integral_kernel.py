"""Compiled float64 form of the same analytic Laguerre disk identities.

No fast-math, quadrature reduction, radial truncation or eigenvalue clipping.
The NumPy implementation remains the fallback when Numba is unavailable.
"""
import math

import numpy as np

try:
    from numba import njit
except ImportError:
    disk_projections_compiled = None
else:
    @njit(cache=True, nogil=True, fastmath=False)
    def disk_projections_compiled(limits, count):
        result = np.zeros((len(limits), count, count))
        for boundary in range(len(limits)):
            limit = limits[boundary]
            values, derivative = np.zeros(count), np.zeros(count)
            previous, current, log_scale = 0., 1., 0.
            for n in range(count):
                if current != 0.:
                    values[n] = math.copysign(math.exp(math.log(abs(current))+log_scale-limit/2), current)
                following = ((2*n+1-limit)*current-n*previous)/(n+1)
                previous, current = current, following
                size = max(abs(previous), abs(current))
                if size > 1e100:
                    previous /= size
                    current /= size
                    log_scale += math.log(size)
                if n > 0:
                    derivative[n] = derivative[n-1]-values[n-1]
            for i in range(count):
                lower_sum = 0.
                for j in range(i):
                    element = limit*(derivative[i]*values[j]-values[i]*derivative[j])/(j-i)
                    result[boundary, i, j] = element
                    result[boundary, j, i] = element
                    lower_sum += element
                result[boundary, i, i] = (-math.expm1(-limit) if i == 0
                                          else 1-values[i]*values[i]-2*lower_sum)
        return result
