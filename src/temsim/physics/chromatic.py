import numpy as np


def cold_feg_energy_offsets(
    count,
    fwhm_ev=0.30,
    half_range_ev=1.0,
    decay_width_ev=0.20,
    boersch_sigma_ev=0.10,
    quantiles=None,
    mean_kinetic_energy_ev=None,
    minimum_kinetic_energy_ev=0.0,
):
    """Return deterministic, bounded cold-FEG energy offsets in eV.

    Cold field emission has Young's one-sided exponential low-energy tail.
    Coulomb (Boersch) broadening is represented by a zero-mean Gaussian.  The
    quantile construction makes a recalculation reproducible for a fixed ray
    count while retaining one independent physical energy per electron.
    """
    n=max(int(count),1)
    fwhm=max(float(fwhm_ev),0.0)
    decay=max(float(decay_width_ev),0.0)
    sigma=max(float(boersch_sigma_ev),0.0)
    limit=max(float(half_range_ev),0.0)
    if fwhm==0.0 or (decay==0.0 and sigma==0.0):
        return np.zeros(n)
    if quantiles is None:
        young_u=(np.arange(n,dtype=float)+0.5)/n
        gaussian_u=young_u[::-1]
    else:
        young_u=np.asarray(quantiles[0],dtype=float)
        gaussian_u=np.asarray(quantiles[1],dtype=float)
        if young_u.shape != (n,) or gaussian_u.shape != (n,):
            raise ValueError("Energy quantiles must each contain one value per ray.")
    young=decay*np.log(np.clip(young_u,1e-15,1.0))
    gaussian=_gaussian_like_quantiles(gaussian_u,sigma)
    offsets=young+gaussian
    offsets-=offsets.mean()
    offsets*=fwhm/(2.354820045*max(offsets.std(),1e-15))
    if limit>0.0:
        offsets=np.clip(offsets,-limit,limit)
    if mean_kinetic_energy_ev is not None:
        offsets = _physical_kinetic_energy_offsets(
            offsets,
            mean_kinetic_energy_ev,
            fwhm,
            limit,
            minimum_kinetic_energy_ev,
        )
    return offsets


def _physical_kinetic_energy_offsets(
    shape,
    mean_kinetic_energy_ev,
    fwhm_ev,
    half_range_ev,
    minimum_kinetic_energy_ev,
):
    """Map an energy-spread shape to a positive kinetic-energy distribution.

    Energy offsets are later added to the sub-eV launch energy of a cold FEG.
    A centred Gaussian-like tail can therefore imply negative launch energy.
    Clamping those samples after emission creates a pile-up of almost stationary
    electrons and numerically unstable trajectories.  This monotonic mapping
    retains the deterministic quantile ordering while matching the requested
    mean and RMS-equivalent FWHM on physically valid kinetic energies.
    """

    values = np.asarray(shape, dtype=float)
    mean = float(mean_kinetic_energy_ev)
    minimum = float(minimum_kinetic_energy_ev)
    target_std = max(float(fwhm_ev), 0.0) / 2.354820045
    if not np.isfinite(mean) or mean <= 0.0:
        raise ValueError("Mean cold-FEG kinetic energy must be positive")
    if not np.isfinite(minimum) or minimum < 0.0 or minimum >= mean:
        raise ValueError(
            "Minimum cold-FEG kinetic energy must lie in [0, mean)"
        )
    if target_std == 0.0 or values.size < 2:
        return np.zeros_like(values)

    centered = values - float(np.mean(values))
    scale = float(np.std(centered))
    if not np.isfinite(scale) or scale <= 0.0:
        return np.zeros_like(values)
    standardized = centered / scale

    limit = max(float(half_range_ev), 0.0)
    lower = minimum if limit == 0.0 else max(minimum, mean - limit)
    if lower >= mean:
        raise ValueError(
            "Cold-FEG energy range leaves no room below the mean energy"
        )

    if limit == 0.0:
        kinetic = _lower_bounded_distribution(
            standardized, lower, mean, target_std
        )
    else:
        upper = mean + limit
        maximum_std = np.sqrt((mean - lower) * (upper - mean))
        if target_std > maximum_std * (1.0 + 1.0e-12):
            raise ValueError(
                "Cold-FEG energy spread is incompatible with its kinetic-"
                "energy mean, minimum and half range"
            )
        kinetic = _bounded_distribution(
            standardized, lower, upper, mean, target_std
        )
    return kinetic - mean


def _lower_bounded_distribution(shape, lower, mean, target_std):
    available_mean = mean - lower

    def distribution(beta):
        logarithm = beta * shape
        logarithm -= float(np.max(logarithm))
        weights = np.exp(logarithm)
        weights /= float(np.mean(weights))
        return lower + available_mean * weights

    low, high = 0.0, 1.0
    while float(np.std(distribution(high))) < target_std:
        high *= 2.0
        if high > 1024.0:
            raise ValueError("Unable to construct cold-FEG energy spread")
    for _ in range(56):
        middle = 0.5 * (low + high)
        if float(np.std(distribution(middle))) < target_std:
            low = middle
        else:
            high = middle
    return distribution(0.5 * (low + high))


def _bounded_distribution(shape, lower, upper, mean, target_std):
    span = upper - lower
    mean_fraction = (mean - lower) / span

    def sigmoid(value):
        return np.exp(-np.logaddexp(0.0, -np.asarray(value, dtype=float)))

    def distribution(beta):
        location_low = -60.0 - beta * float(np.max(shape))
        location_high = 60.0 - beta * float(np.min(shape))
        for _ in range(48):
            location = 0.5 * (location_low + location_high)
            fraction = sigmoid(location + beta * shape)
            if float(np.mean(fraction)) < mean_fraction:
                location_low = location
            else:
                location_high = location
        location = 0.5 * (location_low + location_high)
        return lower + span * sigmoid(location + beta * shape)

    low, high = 0.0, 1.0
    while float(np.std(distribution(high))) < target_std:
        high *= 2.0
        if high > 1024.0:
            raise ValueError("Unable to construct bounded cold-FEG energy spread")
    for _ in range(48):
        middle = 0.5 * (low + high)
        if float(np.std(distribution(middle))) < target_std:
            low = middle
        else:
            high = middle
    return distribution(0.5 * (low + high))


def _gaussian_like_quantiles(u, sigma):
    """Stable inverse-normal approximation used without a SciPy dependency."""
    if sigma <= 0.0:
        return np.zeros_like(np.asarray(u,dtype=float))
    q=np.clip(np.asarray(u,dtype=float),1e-12,1.0-1e-12)
    x=np.log(q/(1.0-q))/1.813799364
    x-=x.mean()
    x/=max(x.std(),1e-15)
    return x*float(sigma)


def deterministic_energy_offsets(count, fwhm_ev):

    """Deterministic Gaussian-like source-energy samples, FWHM in eV."""

    n=max(int(count),1);f=max(float(fwhm_ev),0.0)

    if f==0:return np.zeros(n)

    # Symmetric quantiles without scipy; bounded approximation is stable in GUI.

    u=(np.arange(n,dtype=float)+0.5)/n

    x=np.log(u/(1.0-u))/1.813799364

    x-=x.mean();x/=max(x.std(),1e-15)

    sigma=f/2.354820045

    return x*sigma



def objective_chromatic_kick(x_m,y_m,energy_offset_ev,kinetic_energy_ev,cc_mm,focal_mm):

    """First-order objective chromatic focal-spread kick.


    Delta f = Cc * Delta E / E. The equivalent thin-lens perturbation is

    Delta theta = r * Delta f / f^2. This is an ideal paraxial approximation.

    """

    E=max(float(kinetic_energy_ev),1e-12);f=max(abs(float(focal_mm))*1e-3,1e-12)

    df=float(cc_mm)*1e-3*np.asarray(energy_offset_ev,float)/E

    return np.asarray(x_m,float)*df/(f*f),np.asarray(y_m,float)*df/(f*f)
