"""CF6:5Opt in the complete two-way covariant current chart.

Coefficients: Alvermann and Fehske, JCP 230 (2011), Table 6 and Eq. (60),
https://doi.org/10.1016/j.jcp.2011.04.006 (arXiv:1102.5071).
The five exponentials include a negative numerical substep. This is not a
physical reverse flight or an omitted reflected/evanescent channel. Independent
axial convergence and the full source-boundary solve are still required.
"""
from functools import lru_cache
import math

import numpy as np
from numpy.polynomial.legendre import leggauss, legvander

from temsim.physics.scattering_load import compose, hermitian_slab


@lru_cache(maxsize=1)
def quadrature():
    nodes, weights = leggauss(4)
    first = np.array([.1714, .15409059414309687213, .11947178242929061641, .07195])
    second = np.array([.37496374319946236513, .13813675394387646682,
                       -.13090674649282935743, -.21123356253315514306])
    middle = np.array([1-2*second[0]-2*first[0], 0., -2*second[2]-2*first[2], 0.])
    parity = np.array([1., -1., 1., -1.])
    # The paper writes U=exp(Omega1)...exp(Omega5). In physical execution
    # order the rightmost exponential acts first (Redheffer left-to-right).
    legendre = np.array([first, second, middle, second*parity, first*parity])[::-1]
    fractions = (nodes+1)/2
    coefficients = legendre@(legvander(nodes, 3)*np.arange(1, 8, 2)).T*(weights/2)
    for value in (fractions, coefficients):
        value.setflags(write=False)
    return fractions, coefficients


def cf6_slab(samples, width, reference_k, *, cancelled=lambda: False):
    """Four action-Gauss samples of (residual k^2, connection), same chart.

Normalize each combination by its constant-generator coefficient and put
that coefficient in the signed step length. This keeps the large carrier
analytic instead of subtracting nearly equal kinetic-energy matrices.
"""
    if len(samples) != 4 or not math.isfinite(width) or not math.isfinite(reference_k) or reference_k <= 0:
        raise ValueError("CF6 needs four samples, a finite width and positive finite reference k")
    _, weights = quadrature()
    result, records = None, []
    for row in weights:
        if cancelled():
            raise InterruptedError("CF6 wave propagation cancelled")
        fraction = float(row.sum())
        normalised = row/fraction
        residual = sum(a*value[0] for a, value in zip(normalised, samples))
        connection = sum(a*value[1] for a, value in zip(normalised, samples))
        piece, record = hermitian_slab(residual, connection, width*fraction,
                                       reference_k, carrier_k=reference_k)
        result = piece if result is None else compose(result, piece)
        records.append(record)
    return result, {"axial_integrator": "cf6-5opt", "action_gauss_fractions": quadrature()[0].tolist(),
                    "covariant_substeps": records, "negative_numerical_substep": True}
