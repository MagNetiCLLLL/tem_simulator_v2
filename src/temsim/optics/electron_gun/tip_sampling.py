"""Importance quadrature of the same uniform physical emitting cap."""
import numpy as np


def tangent_cell_ids(count, allocation=()):
    """Numerical counts for all nine Cartesian CDF cells; no cell is omitted."""
    if type(count) is not int or count < 9:
        raise ValueError("Tangent quadrature needs at least nine samples")
    if not allocation:
        return np.arange(count)%9
    if len(allocation) != 9 or any(type(v) is not int or not 1 <= v <= 1000 for v in allocation):
        raise ValueError("Tangent allocation requires nine positive integer numerical priorities")
    quota = (count-9)*np.asarray(allocation,float)/sum(allocation)
    population = 1+np.floor(quota).astype(int)
    remainder = count-int(population.sum())
    population[np.argsort(-(quota-np.floor(quota)),kind="stable")[:remainder]] += 1
    return np.concatenate([np.flatnonzero(population > j) for j in range(population.max())])


def stratified_cap_area(site_count, allocation=()):
    """Return area/azimuth coordinates and weights for nine disjoint strata.

The first cell covers [0, 1e-8] of total emitting area, followed by decades
up to 1. No area is omitted. Samples in a stratum use a local low-discrepancy
sequence; changing site budget refines every stratum. The zero-area apex is
not assigned a finite current. This rule does not prescribe a new cap radius.
"""
    from temsim.optics.electron_gun.emitter import _halton_dimensions
    if type(site_count) is not int or site_count < 9:
        raise ValueError("Apex-stratified quadrature needs at least nine emission sites")
    edges = np.r_[0., np.logspace(-8, 0, 9)]
    count = edges.size-1
    if allocation:
        if (len(allocation) != count or any(type(v) is not int or not 1 <= v <= 1000 for v in allocation)):
            raise ValueError("Cap allocation requires nine positive integer numerical priorities")
        # At least one site in EVERY stratum, then Hamilton allocation of the
        # remaining budget. These priorities affect sample density, not flux.
        quotas = (site_count-count)*np.asarray(allocation, float)/sum(allocation)
        population = 1+np.floor(quotas).astype(int)
        remainder = site_count-int(population.sum())
        population[np.argsort(-(quotas-np.floor(quotas)), kind="stable")[:remainder]] += 1
        strata = np.concatenate([np.flatnonzero(population > j) for j in range(population.max())])
    else:
        strata = np.arange(site_count) % count
        population = np.bincount(strata, minlength=count)
    coordinates = np.empty(site_count)
    azimuth = np.empty(site_count)
    weights = np.empty(site_count)
    for index in range(count):
        selected = strata == index
        local, phi = _halton_dimensions(int(population[index]), (2, 3))
        width = edges[index+1]-edges[index]
        coordinates[selected] = edges[index]+width*local
        # Taking every ninth point of a base-3 Halton azimuth would trap each
        # annulus in one sector! Each stratum needs its OWN complete azimuth law.
        azimuth[selected] = (phi+index*(np.sqrt(5.)-1)*.5) % 1.
        weights[selected] = width/population[index]
    return coordinates, azimuth, weights


def stratified_tangent_momenta(count, *, sigma_sqrt_ev, radial_centre_sigma=0., halfwidth_sigma=.1,
                              sequence_offset=0, allocation=()):
    """Exact-probability quadrature of two independent zero-mean Gaussians.

    Units of the two returned amplitudes are sqrt(eV). Their squares sum to
    tangential kinetic energy. Three CDF intervals on each axis cover the
    entire Gaussian plane, including both tails. Moving the numerical central
    box does NOT move the physical mean or omit the exterior. Each of the nine
    boxes receives its exact probability, independent of the sample count.
    The first return is an independent normal-energy CDF coordinate.
    """
    from scipy.special import ndtr, ndtri
    from temsim.optics.electron_gun.emitter import _radical_inverse
    if (type(count) is not int or count < 9 or not np.isfinite(
            [sigma_sqrt_ev,radial_centre_sigma,halfwidth_sigma]).all()
            or sigma_sqrt_ev <= 0 or halfwidth_sigma <= 0):
        raise ValueError("Tangent quadrature needs nine samples and positive finite widths")
    if type(sequence_offset) is not int or sequence_offset < 0:
        raise ValueError("Tangent sequence offset must be a nonnegative integer")
    edges = [np.r_[0.,ndtr(centre+np.array([-halfwidth_sigma,halfwidth_sigma])),1.]
             for centre in (radial_centre_sigma,0.)]
    if any(np.any(np.diff(edge) <= 0) for edge in edges):
        raise ValueError("Tangent quadrature CDF bins are unresolved; adjust numerical refinement")
    ids = tangent_cell_ids(count, allocation)
    population = np.bincount(ids,minlength=9)
    en, t1, t2, weight = (np.empty(count) for _ in range(4))
    for cell in range(9):
        selected = ids == cell
        sequence = np.arange(18+sequence_offset,18+sequence_offset+population[cell])
        qn,q1,q2 = (_radical_inverse(sequence,base) for base in (5,7,11))
        first, second = divmod(cell,3)
        lo1,hi1 = edges[0][first:first+2]
        lo2,hi2 = edges[1][second:second+2]
        p1,p2 = lo1+(hi1-lo1)*q1,lo2+(hi2-lo2)*q2
        if np.any((p1 <= 0)|(p1 >= 1)|(p2 <= 0)|(p2 >= 1)):
            raise ValueError("Tangent quadrature quantiles are unresolved")
        en[selected] = qn
        t1[selected],t2[selected] = sigma_sqrt_ev*ndtri(p1),sigma_sqrt_ev*ndtri(p2)
        weight[selected] = (hi1-lo1)*(hi2-lo2)/population[cell]
    return en,t1,t2,weight
