"""Independent closed-form aperture matrix check; not image acceptance."""
import math
import time
import numpy as np
from scipy.special import eval_laguerre, eval_genlaguerre
from threadpoolctl import threadpool_limits
from temsim.physics.radial_gun_wave import aperture_projection


def positive_quadrature(limit, count):
    nodes, weights = np.polynomial.legendre.leggauss(max(64, 4*count))
    edges = np.linspace(0., limit, math.ceil(limit/4)+1)
    half = np.diff(edges)/2
    x = ((edges[:-1]+edges[1:])[:, None]/2+half[:, None]*nodes).ravel()
    w = (half[:, None]*weights).ravel()
    values = np.array([eval_laguerre(n, x) for n in range(count)])*np.exp(-x/2)
    return (values*w)@values.T


def closed_form(limit, count):
    order = np.arange(count)
    values = np.exp(-limit/2)*eval_laguerre(order, limit)
    derivative = np.zeros(count)
    derivative[1:] = -np.exp(-limit/2)*eval_genlaguerre(order[1:]-1, 1, limit)
    difference = order[None, :]-order[:, None]
    result = np.divide(limit*(derivative[:, None]*values[None, :]-values[:, None]*derivative[None, :]),
        difference, out=np.zeros((count, count)), where=difference != 0)
    diagonal = 1-values*values-2*np.tril(result, -1).sum(axis=1)
    np.fill_diagonal(result, diagonal)
    return result


if __name__ == "__main__":
    with threadpool_limits(1):
        for count in (16, 64, 128):
            for limit in (.1, 4., 50., 4*count):
                start = time.perf_counter()
                exact = closed_form(limit, count)
                exact_time = time.perf_counter()-start
                start = time.perf_counter()
                quadrature = positive_quadrature(limit, count)
                elapsed = time.perf_counter()-start
                print(count, limit, np.max(abs(exact-quadrature)), exact_time, elapsed, flush=True)
