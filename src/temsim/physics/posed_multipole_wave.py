"""Remainders of the captured, rigidly posed native multipole vector potential.

The provider owns A_local=(0,0,P(x,y) f(z)), its polynomial coefficients,
finite axial support and registration. These helpers neither refit its
strength nor replace the hard support with a smooth fringe field.
"""
import math

import numpy as np


def _gaussian_bounds(centre, sigma, lower, upper):
    """Suprema of |d**j f/dz**j|, j=0..3, over an interval."""
    lo, hi = (lower-centre)/sigma, (upper-centre)/sigma
    extrema = ((0.,), (-1., 1.), (0., -math.sqrt(3.), math.sqrt(3.)),
        tuple(sign*math.sqrt(3.+s*math.sqrt(6.)) for s in (-1., 1.) for sign in (-1., 1.)))
    values = []
    for order, roots in enumerate(extrema):
        candidates = (lo, hi, *(r for r in roots if lo <= r <= hi))
        def derivative(u):
            polynomial = (1., -u, u*u-1., 3*u-u**3)[order]
            return abs(polynomial)*math.exp(-.5*u*u)/sigma**order
        values.append(max(derivative(u) for u in candidates))
    return values


def _polynomial_directional_bound(terms, coordinate_bound, displacement_bound, order):
    result = 0.
    for px, py, coefficient in terms:
        for ix in range(min(px, order)+1):
            iy = order-ix
            if 0 <= iy <= py:
                result += (abs(coefficient)*math.comb(px, ix)*math.comb(py, iy)
                    *coordinate_bound[0]**(px-ix)*coordinate_bound[1]**(py-iy)
                    *displacement_bound[0]**ix*displacement_bound[1]**iy)
    return math.factorial(order)*result


def multipole_potential_remainder_bound(field, position_m, xy_bound):
    """Bound ||A-A0-A1-A2|| and |Az-Az0-Az1-Az2| on a global XY box.

    Inside a smooth region use an interval bound on the third directional
    derivative. A box crossing finite-support edges instead uses a full
    field-plus-polynomial bound; the caller must resolve its actual masked
    residual, rather than assuming a Taylor series across that discontinuity.
    """
    rotation = field.registration.rotation_array
    local = field.registration.positions_global_to_local_m(position_m)
    delta = abs(rotation[:2, :].T)@np.asarray(xy_bound)
    lower, upper = local[2]-delta[2], local[2]+delta[2]
    support_lower, support_upper = np.asarray(field.native_support_mm)*1e-3
    if upper < support_lower or lower > support_upper:
        return 0., 0.
    radius = abs(local[:2])+delta[:2]
    crosses = lower < support_lower or upper > support_upper
    if field.envelope_kind == "uniform":
        profile_bounds = (1., 0., 0., 0.)
    else:
        profile_bounds = _gaussian_bounds(field.center_z_m, field.sigma_m,
            max(lower, support_lower), min(upper, support_upper))
    if crosses:
        maximum = _polynomial_directional_bound(field.polynomial_terms, radius, delta[:2], 0)*profile_bounds[0]
        a, d, dd = field.vector_potential_jet(position_m)
        bound = (maximum+np.linalg.norm(a)+np.linalg.norm(abs(d[:, :2])@xy_bound)
            +.5*np.linalg.norm(np.einsum("ijk,j,k->i", abs(dd[:, :2, :2]), xy_bound, xy_bound)))
    else:
        bound = sum(math.comb(3, order)*_polynomial_directional_bound(
            field.polynomial_terms, radius, delta[:2], order)
            *profile_bounds[3-order]*delta[2]**(3-order) for order in range(4))/6.
    return float(bound), float(abs(rotation[2, 2])*bound)


def _polynomial_local_expansion(terms, centre, delta, xp):
    """Exact homogeneous degree 0,1,2,3 terms in a local displacement."""
    pieces = [xp.zeros_like(delta[0]) for _ in range(4)]
    for px, py, coefficient in terms:
        for ix in range(px+1):
            for iy in range(py+1):
                pieces[ix+iy] += (coefficient*math.comb(px, ix)*math.comb(py, iy)
                    *centre[0]**(px-ix)*centre[1]**(py-iy)*delta[0]**ix*delta[1]**iy)
    return pieces


def multipole_vector_potential_remainders(field, position_m, delta_xy, xp):
    """Exact A-A0-A1 and A-A0-A1-A2, evaluated stably on CPU or GPU."""
    from temsim.physics.posed_lens_wave import _exponential_tail
    rotation = field.registration.rotation_array
    centre = field.registration.positions_global_to_local_m(position_m)
    delta = xp.einsum("ij,jyx->iyx", xp.asarray(rotation[:2, :].T), delta_xy)
    p0, p1, p2, p3 = _polynomial_local_expansion(field.polynomial_terms, centre, delta, xp)
    lower, upper = np.asarray(field.native_support_mm)*1e-3
    inside_centre = lower <= centre[2] <= upper
    local_z = centre[2]+delta[2]
    inside = (local_z >= lower) & (local_z <= upper)
    if field.envelope_kind == "uniform":
        actual = inside.astype(xp.float64)
        difference1 = actual-float(inside_centre)
        difference2 = difference3 = difference1
    else:
        actual = xp.where(inside, xp.exp(-.5*((local_z-field.center_z_m)/field.sigma_m)**2), 0.)
        if inside_centre:
            u = (centre[2]-field.center_z_m)/field.sigma_m
            t = delta[2]/field.sigma_m
            exponent = -u*t-.5*t*t
            f0 = math.exp(-.5*u*u)
            f1 = -f0*u*t
            f2 = .5*f0*(u*u-1)*t*t
            # expm1 avoids losing physically small higher orders. Outside
            # the local interval use direct values, avoiding 0*exp(large).
            small = abs(exponent) < .5
            bounded_exponent = xp.clip(exponent, -.5, .5)
            difference1 = xp.where(inside & small, f0*xp.expm1(bounded_exponent), actual-f0)
            difference2 = xp.where(inside & small,
                f0*(_exponential_tail(bounded_exponent, 1, xp)-.5*t*t), actual-f0-f1)
            difference3 = xp.where(inside & small,
                f0*(_exponential_tail(bounded_exponent, 2, xp)+.5*u*t**3+.125*t**4), actual-f0-f1-f2)
        else:
            # All captured derivatives vanish outside the native support.
            difference1 = difference2 = difference3 = actual
    second = p0*difference2+p1*difference1+(p2+p3)*actual
    third = p0*difference3+p1*difference2+p2*difference1+p3*actual
    direction = xp.asarray(rotation[:, 2])[:, None, None]
    return direction*second, direction*third
