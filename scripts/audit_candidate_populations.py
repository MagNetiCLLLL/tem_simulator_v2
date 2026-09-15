"""Observe all-current and identical final-survivor populations separately."""
from dataclasses import asdict, replace

import numpy as np
from scipy.optimize import brentq

from temsim.optics.beam_path_audit import (
    emission_measurement, incident_checkpoints, spot_measurement, uniform_cap_footprint,
)


def population_report(state, labelled_planes, *, step_mm=.025):
    """No input changes or discarded source current; only masks for diagnostics."""
    names, planes = zip(*labelled_planes)
    if planes[-1] != state.sample.upper_surface_z_mm:
        raise ValueError("The last population checkpoint must be the sample entrance")
    trace, checkpoints, masks = incident_checkpoints(state, planes, step_mm=step_mm)
    exit_bundle = trace.exit_bundle
    launch = state.electron_gun.emit(state.electron_gun.emitter.ray_count)
    if not np.array_equal(launch.ray_id,exit_bundle.ray_id):
        raise ValueError("Emission and executed gun ray identities differ")
    final = masks[-1]
    label = "same IDs eventually reaching sample; observation only"
    rows = [asdict(emission_measurement(launch)),
            asdict(replace(emission_measurement(launch,alive=final),population=label))]
    for i,(name,z) in enumerate(labelled_planes):
        values = tuple(getattr(checkpoints,key)[i] for key in ('x_m','y_m','tx_rad','ty_rad'))
        rows.append(asdict(spot_measurement(name,z,values,exit_bundle.weight,masks[i])))
        rows.append(asdict(spot_measurement(name,z,values,exit_bundle.weight,final,population=label)))
    return dict(scope="OBSERVATION_OF_EXECUTED_PARTICLES_NOT_A_NEW_SOURCE",
        exact_tip=uniform_cap_footprint(state.electron_gun.emitter.surface_model),
        step_mm=step_mm,rows=rows)


def cohort_waist(state, bracket_mm, *, step_mm=.025):
    """Locate the C1 waist of eventual specimen survivors, without clipping them early."""
    trace,_,masks = incident_checkpoints(state,[state.sample.upper_surface_z_mm],step_mm=step_mm)
    final = masks[-1]
    measured = {}
    def observation(z):
        if z not in measured:
            _,cp,mask = incident_checkpoints(state,[z],step_mm=step_mm)
            if np.any(final & ~mask[0]):
                raise ValueError("Final survivor IDs are absent at an earlier plane")
            measured[z] = spot_measurement('C1 waist of final specimen cohort',z,
                tuple(getattr(cp,key)[0] for key in ('x_m','y_m','tx_rad','ty_rad')),
                trace.exit_bundle.weight,final,
                population='same IDs eventually reaching sample; not an additional full-beam crossover')
        return measured[z]
    lo,hi = bracket_mm
    def offset(z):
        return -observation(float(z)).covariance_waist_offset_mm
    if not offset(lo) < 0 < offset(hi):
        raise ValueError('The cohort waist is not bracketed')
    root = float(brentq(offset,lo,hi,xtol=1e-9,rtol=1e-13))
    return asdict(observation(root))
