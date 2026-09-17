"""Importance quadrature of the same uniform physical emitting cap."""
import numpy as np


def _product_indices(plan):
    plan.validate()
    return (np.repeat(np.arange(plan.spatial), plan.directions*plan.energies),
            np.tile(np.repeat(np.arange(plan.directions), plan.energies), plan.spatial),
            np.tile(np.arange(plan.energies), plan.spatial*plan.directions))


def flat_product_bundle(emitter, plan):
    """Independent marginals of the existing flat/curved legacy tip law."""
    from temsim.optics.electron_gun.emitter import _halton_dimensions, _truncated_gaussian_disk
    from temsim.optics.electron_gun.base import EmissionBundle
    from temsim.physics.chromatic import cold_feg_energy_offsets
    from temsim.optics.electron_gun.tip_curvature import curve_bundle
    si, di, ei = _product_indices(plan)
    u, phi = _halton_dimensions(plan.spatial, (2, 3))
    sigma = emitter.virtual_source_fwhm_nm / 2.354820045 * 1e-9
    x, y = _truncated_gaussian_disk(u, phi, sigma, 3*sigma)
    u, phi = _halton_dimensions(plan.directions, (5, 7))
    tx, ty = _truncated_gaussian_disk(u, phi, emitter.angular_rms_mrad*1e-3, emitter.angular_cutoff_mrad*1e-3)
    energy = cold_feg_energy_offsets(plan.energies, emitter.energy_spread_fwhm_ev,
        emitter.energy_half_range_ev, emitter.young_decay_width_ev, emitter.boersch_sigma_ev,
        quantiles=_halton_dimensions(plan.energies, (11, 13)), mean_kinetic_energy_ev=emitter.emission_energy_ev,
        minimum_kinetic_energy_ev=emitter.minimum_kinetic_energy_ev)
    return curve_bundle(EmissionBundle(x[si], y[si], tx[di], ty[di], energy[ei],
        np.full(plan.total, 1/plan.total), np.arange(plan.total, dtype=np.int64)), emitter)


def surface_product_samples(model, plan):
    """Full-cap area, local direction and conditional positive-energy CDFs.

    For exponential normal/tangential energies, set s=T/(N+T). The joint
    density is E*exp(-lambda(s)*E)/(a*b), lambda=(1-s)/a+s/b. Its angular
    marginal is proportional to lambda^-2, and E conditional on s is Gamma(2,
    1/lambda). Truncating s to sin(maximum_angle)^2 is exactly the existing
    local angular conditioning. Direction/energy correlation is preserved.
    The product factors refine these conditional integrals independently.
    """
    from scipy.special import gammaincinv
    from temsim.optics.electron_gun.emitter import _halton_dimensions
    from temsim.optics.electron_gun.tip_patch import sample_cap_frame
    model.validate()
    if model.coherence is not None:
        raise ValueError("Surface product quadrature is classical only")
    si, di, ei = _product_indices(plan)
    emission = model.emission
    if emission.spatial_sampling == "apex_stratified_v1":
        area, azimuth, site_weight = stratified_cap_area(plan.spatial, emission.spatial_stratum_allocation)
    else:
        area, azimuth = _halton_dimensions(plan.spatial, (2, 3))
        site_weight = np.full(plan.spatial, 1/plan.spatial)
    positions, normals, tangent1, tangent2 = sample_cap_frame(model.geometry, emission.cap_half_angle_deg, area, azimuth)
    u, phi = _halton_dimensions(plan.directions, (5, 7))
    energy_u, = _halton_dimensions(plan.energies, (11,))
    maximum = np.deg2rad(emission.maximum_angle_deg)
    if emission.energy_distribution == "normal_tangential_exponential":
        a, b = emission.normal_mean_energy_ev, emission.tangential_mean_energy_ev
        if b == 0:
            cosine = np.ones(plan.directions)
            energy = -a*np.log1p(-energy_u[ei])
        else:
            smax = np.sin(maximum)**2
            # Algebraic inverse CDF avoids subtracting nearly equal inverses.
            s = u*smax*b / (b+(a-b)*smax*(1-u))
            cosine = np.sqrt(1-s)
            rate = (1-s)/a+s/b
            energy = gammaincinv(2., energy_u[ei])/rate[di]
    else:
        cosine = 1-u*(1-np.cos(maximum))
        energy = np.full(plan.total, emission.kinetic_mean_ev)
        if emission.energy_distribution == "gamma":
            shape = (emission.kinetic_mean_ev/emission.kinetic_sigma_ev)**2
            energy = gammaincinv(shape, energy_u[ei])*emission.kinetic_sigma_ev**2/emission.kinetic_mean_ev
    tangent = np.cos(2*np.pi*phi[di])[:, None]*tangent1[si] + np.sin(2*np.pi*phi[di])[:, None]*tangent2[si]
    direction = cosine[di, None]*normals[si] + np.sqrt(np.maximum(0, 1-cosine[di]**2))[:, None]*tangent
    weight = site_weight[si]/(plan.directions*plan.energies)
    if np.any(~np.isfinite(energy)) or np.any(energy <= 0):
        raise ValueError("Conditional surface energy quadrature is out of its numerical domain")
    return positions[si], direction, energy, weight


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
