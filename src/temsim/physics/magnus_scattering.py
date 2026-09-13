"""Fourth-order commutator-free Magnus step in a common flux chart.

The independent variable is the caller's action coordinate, not physical z.
Two Gauss nodes and two exact covariant slabs retain forward/backward waves.
Blanes and Moan, Applied Numerical Mathematics 56 (2006), 1519-1537,
https://doi.org/10.1016/j.apnum.2005.11.004 .
This is an axial discretisation, not a source or a convergence certificate.
"""
import math

from temsim.physics.scattering_load import compose, hermitian_slab


GAUSS_FRACTIONS = ((1-1/math.sqrt(3))/2, (1+1/math.sqrt(3))/2)


def cf4_slab(first, second, width, reference_k, *, cancelled=lambda: False):
    """Nodes are (residual k^2, connection), in the SAME moving coordinates.

    The first-order covariant generator is linear in both matrices. Do not
    average physical-z operators here before the Liouville transformation.
    """
    low = (3-2*math.sqrt(3))/6
    high = (3+2*math.sqrt(3))/6
    operators, diagnostics = [], []
    for a, b in ((high, low), (low, high)):
        if cancelled():
            raise InterruptedError("CF4 wave propagation cancelled")
        op, diagnostic = hermitian_slab(a*first[0]+b*second[0],
            a*first[1]+b*second[1], width/2, reference_k,
            carrier_k=reference_k)
        operators.append(op)
        diagnostics.append(diagnostic)
    return compose(*operators), {"axial_integrator": "cf4",
        "action_gauss_fractions": GAUSS_FRACTIONS,
        "covariant_substeps": diagnostics}
