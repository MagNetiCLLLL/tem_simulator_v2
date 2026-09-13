"""Batched analytic radial integrals, not a separate physics model.

All potential knots and Laguerre modes remain present. The finite-basis
polynomial moments are collected before the two exact coordinate products;
no potential, phase, reflected component or wave mode is sampled away.
"""
import numpy as np


def disk_projections(limits, count):
    """Exact Laguerre disk integrals, batched over dimensionless squared radii."""
    limits = np.asarray(limits, dtype=float)
    if limits.ndim != 1 or np.any(limits < 0) or not np.all(np.isfinite(limits)):
        raise ValueError("Disk integration limits must be finite and nonnegative")
    if type(count) is not int or count < 1:
        raise ValueError("Radial mode count must be a positive integer")
    from temsim.physics.radial_integral_kernel import disk_projections_compiled
    if disk_projections_compiled is not None:
        return disk_projections_compiled(np.ascontiguousarray(limits), count)
    return _disk_projections_numpy(limits, count)


def _disk_projections_numpy(limits, count):
    """Unchanged vectorized fallback, also used for independent kernel checks."""
    values = np.zeros((len(limits), count))
    previous, current, log_scale = np.zeros(len(limits)), np.ones(len(limits)), np.zeros(len(limits))
    for n in range(count):
        nonzero = current != 0
        values[nonzero, n] = np.sign(current[nonzero])*np.exp(
            np.log(abs(current[nonzero]))+log_scale[nonzero]-limits[nonzero]/2)
        following = ((2*n+1-limits)*current-n*previous)/(n+1)
        previous, current = current, following
        size = np.maximum(abs(previous), abs(current))
        large = size > 1e100
        previous[large] /= size[large]
        current[large] /= size[large]
        log_scale[large] += np.log(size[large])
    derivative = np.column_stack((np.zeros(len(limits)), -np.cumsum(values[:, :-1], axis=1)))
    order = np.arange(count)
    difference = order[None, :]-order[:, None]
    matrix = np.divide(limits[:, None, None]*(derivative[:, :, None]*values[:, None, :]
                       -values[:, :, None]*derivative[:, None, :]), difference,
                       out=np.zeros((len(limits), count, count)), where=difference != 0)
    diagonal = -np.expm1(-limits)[:, None]*np.ones((1, count))
    diagonal[:, 1:] = 1-values[:, 1:]**2-2*np.tril(matrix, -1).sum(axis=2)[:, 1:]
    matrix[:, order, order] = diagonal
    return matrix


def multiply_x(matrix):
    n = np.arange(len(matrix))
    result = matrix*(2*n+1.)[None, :]
    result[:, 1:] -= matrix[:, :-1]*n[1:][None, :]
    result[:, :-1] -= matrix[:, 1:]*n[1:][None, :]
    return result


def piecewise_potential(width, count, radii, energies, axis_energy, kinetic_constant, rest_energy):
    """Same relativistic polynomial integral, collecting moments before X products."""
    x, energy = (np.asarray(radii)/width)**2, np.asarray(energies)
    if (x.ndim != 1 or len(x) < 2 or x[0] != 0 or np.any(np.diff(x) <= 0)
            or energy.shape != x.shape or np.any(energy <= 0)
            or not np.all(np.isfinite((x, energy)))):
        raise ValueError("Radial potential needs increasing knots and positive finite kinetic energies")
    slopes = np.diff(energy)/np.diff(x)
    intercepts = energy[:-1]-slopes*x[:-1]
    coefficients = np.array((
        kinetic_constant*(intercepts-axis_energy)*(1+(intercepts+axis_energy)/(2*rest_energy)),
        kinetic_constant*slopes*(1+intercepts/rest_energy),
        kinetic_constant*slopes*slopes/(2*rest_energy)))
    n = count+2  # Preserve both X products before final projection.
    moments, lower = np.zeros((3, n, n)), np.zeros((n, n))
    for begin in range(0, len(slopes), 16):
        end = min(begin+16, len(slopes))
        upper = disk_projections(x[begin+1:end+1], n)
        intervals = np.diff(np.concatenate((lower[None], upper)), axis=0)
        moments += (coefficients[:, begin:end]@intervals.reshape(end-begin, -1)).reshape(3, n, n)
        lower = upper[-1]
    result = (moments[0]+multiply_x(moments[1])+multiply_x(multiply_x(moments[2])))[:count, :count]
    return (result+result.T)/2
