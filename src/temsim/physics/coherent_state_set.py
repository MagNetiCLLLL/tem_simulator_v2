"""Named tip states in shared optics, with incoherent intensity comparison.

A state record is an input snapshot, never an independently configured
downstream source. Each member is propagated normally from the physical tip.
The display combines probabilities; members retain their own complex fields.
"""
from dataclasses import dataclass, replace
from copy import deepcopy
import math
from numbers import Real

import numpy as np

from temsim.immutable_json import json_digest
from temsim.instrument_snapshot import (
    InstrumentSnapshot, capture_instrument_snapshot, decode_instrument, encode_instrument,
)
from temsim.physics.coherent_inputs import (
    candidate_tip_emission, prepare_coherent_state, source_settings_from_state,
)


def _normalised_state(state):
    """Detach and mask only the explicitly supported physical emission inputs."""
    detached = decode_instrument(encode_instrument(state))
    # Readiness/model errors remain errors, rather than weakening the identity.
    source_settings_from_state(detached)
    _mask_emission_inputs(detached.electron_gun.emitter)
    return detached


def _mask_emission_inputs(emitter):
    """Mask only fields in the supported emission editor on an owned copy."""
    surface = emitter.surface_model
    if surface is None:
        emitter.virtual_source_fwhm_nm = 100.
        emitter.emission_energy_ev = 1.
        emitter.minimum_kinetic_energy_ev = .01
        emitter.energy_spread_fwhm_ev = .1
        if emitter.coherence is not None:
            emitter.coherence = replace(emitter.coherence,
                incoherent_angle_rms_mrad=0., curvature_x_m1=0., curvature_xy_m1=0.,
                curvature_y_m1=0., offset_x_nm=0., offset_y_nm=0.,
                tilt_x_mrad=0., tilt_y_mrad=0.)
    else:
        if surface.coherence is not None and not surface.shared_boundary:
            raise ValueError("Historical surface reservoirs cannot join a tip state set")
        if surface.shared_boundary:
            # Gamma with zero width is represented by its monoenergetic limit.
            # This choice is derived from the editable mean/RMS inputs.
            emitter.surface_model = replace(surface,
                emission=replace(surface.emission, kinetic_mean_ev=1.,
                    kinetic_sigma_ev=.1, energy_distribution="gamma"),
                coherence=replace(surface.coherence, edge_phase_rad=0.))


def common_state_identity(state):
    """Identity of shared geometry, fields, current, material and numerics.

    Only supported tip emission-state values are masked. The established
    read-only virtual observation Z is omitted by ``physical_digest``; unknown
    fields and all other source/hardware values remain part of the identity.
    No input or cached executed state is changed.
    """
    return capture_instrument_snapshot(_normalised_state(state)).physical_digest


def tip_geometry_identity(state):
    """Conservative tip/model identity for explicit execution in new optics.

    Emission inputs are masked as above. Particle sample count/quadrature are
    execution choices, not metal geometry. Other emitter attributes, including
    current, emitting-cap shape, source family and field numerics, remain bound.
    """
    source_settings_from_state(state)
    emitter = deepcopy(state.electron_gun.emitter)
    _mask_emission_inputs(emitter)
    emitter.ray_count = 1
    emitter.quadrature = None
    return json_digest({"gun_type": state.electron_gun.type_key,
                        "emitter": encode_instrument(emitter)})


def restore_tip_state(saved_snapshot, current_state):
    """Restore a stored state only while its complete common inputs still match."""
    if not isinstance(saved_snapshot, InstrumentSnapshot):
        raise TypeError("A named tip state requires an instrument input snapshot")
    saved = saved_snapshot.restore()
    if common_state_identity(saved) != common_state_identity(current_state):
        raise ValueError("Shared optics or tip geometry changed; calculate the states again")
    return saved


def rebase_tip_state(saved_snapshot, current_state):
    """Explicitly prepare stored tip emission for a new current-optics execution.

    This creates a complete canonical instrument input for either particle or
    wave propagation. It never reuses an executed field, replaces tip geometry,
    or modifies the live instrument. The normal wave capture guard is retained.
    """
    if not isinstance(saved_snapshot, InstrumentSnapshot):
        raise TypeError("A named tip state requires an instrument input snapshot")
    saved = saved_snapshot.restore()
    if tip_geometry_identity(saved) != tip_geometry_identity(current_state):
        raise ValueError("Tip geometry, current or source model changed; recapture states")
    settings = source_settings_from_state(saved)
    current = capture_instrument_snapshot(current_state).restore()
    candidate = candidate_tip_emission(current, settings)
    # Assembly models may share the emitter by object reference. Publishing
    # fields on the detached owner keeps those references canonical; replacing
    # the object would strand an old source elsewhere in the captured graph.
    vars(current.electron_gun.emitter).clear()
    vars(current.electron_gun.emitter).update(vars(candidate.emitter))
    current.electron_gun.source_representation = candidate.source_representation
    current.electron_gun._trace_cache = current.electron_gun._trace_cache_key = None
    return prepare_coherent_state(current, settings)


@dataclass(frozen=True)
class EnsembleCheckpoint:
    """Display location and common current; deliberately no aggregate wave."""
    plane_z_mm: float
    reference_current_a: float


@dataclass(frozen=True)
class EnsembleObservation:
    checkpoint: EnsembleCheckpoint
    members: tuple
    relative_weights: tuple[float, ...]


@dataclass(frozen=True)
class EnsemblePreview:
    density: np.ndarray
    bounds_um: np.ndarray
    probability: float
    mode_count: int
    retained_bytes: int
    grid_shapes: tuple[tuple[int, int], ...] = ()
    pair_token: None = None
    comparison: None = None
    comparison_error: None = None
    axial_bz_t: float | None = None


def _preview_geometry(preview):
    density = np.asarray(preview.density, dtype=float)
    bounds = np.asarray(preview.bounds_um, dtype=float)
    if (density.ndim != 2 or 0 in density.shape or bounds.shape != (2, 2)
            or not np.all(np.isfinite(density)) or np.any(density < 0.)
            or not np.all(np.isfinite(bounds))
            or np.any(bounds[:, 1] <= bounds[:, 0])):
        raise ValueError("State previews need finite non-negative density and physical bounds")
    return density, bounds


def combine_intensity_previews(items):
    """Return an observation and a conservative normalized intensity mixture.

    ``items`` contains ``(raw_result, preview, relative_weight)`` tuples. Display
    arrays use X/Y order, matching the existing wave display. Rebinning uses
    cell overlap, never complex-field addition or an interpolation of phase.
    The caller must also verify the members' shared input/optical identity.
    """
    from temsim.physics.affine_cell_histogram import affine_cell_histogram

    rows = tuple(items)
    if not rows:
        raise ValueError("Select at least one electron state")
    weights, geometries = [], []
    z_mm = current_a = None
    for result, preview, weight in rows:
        if (isinstance(weight, bool) or not isinstance(weight, Real)
                or not math.isfinite(weight) or weight < 0.):
            raise ValueError("State weights must be finite and non-negative")
        weights.append(float(weight))
        geometries.append(_preview_geometry(preview))
        z = float(result.checkpoint.plane_z_mm)
        current = float(result.checkpoint.reference_current_a)
        if not math.isfinite(z) or not math.isfinite(current) or current < 0.:
            raise ValueError("State results need a finite plane and reference current")
        if z_mm is None:
            z_mm, current_a = z, current
        elif z != z_mm:
            raise ValueError("All electron states must be evaluated at exactly the same Z")
        elif current != current_a:
            raise ValueError("All electron states must have the same tip reference current")
    largest = max(weights)
    if largest == 0.:
        raise ValueError("At least one state weight must be positive")
    # Stable even when several individually finite weights would overflow sum.
    weights = np.asarray(weights, dtype=float)/largest
    weights /= weights.sum()
    observation = EnsembleObservation(EnsembleCheckpoint(z_mm, current_a),
        tuple(row[0] for row in rows), tuple(float(weight) for weight in weights))
    if len(rows) == 1:
        return observation, rows[0][1]

    active = tuple(index for index, weight in enumerate(weights) if weight > 0.)
    bounds = np.array(((min(geometries[i][1][0, 0] for i in active),
                        max(geometries[i][1][0, 1] for i in active)),
                       (min(geometries[i][1][1, 0] for i in active),
                        max(geometries[i][1][1, 1] for i in active))))
    bins = tuple(max(geometries[i][0].shape[axis] for i in active) for axis in (0, 1))
    probability = np.zeros(bins)
    for index in active:
        density, source_bounds = geometries[index]
        cell_size = (source_bounds[:, 1]-source_bounds[:, 0])/density.shape
        if density.shape == bins and np.array_equal(source_bounds, bounds):
            contribution = density*np.prod(cell_size)
        else:
            # affine_cell_histogram accepts Y/X weights but returns X/Y bins.
            origin = source_bounds[:, 0]+(np.asarray(density.shape)//2+.5)*cell_size
            contribution, _ = affine_cell_histogram(
                (density*np.prod(cell_size)).T, np.diag(cell_size), origin, bounds, bins)
        probability += weights[index]*contribution
    cell_area = np.prod((bounds[:, 1]-bounds[:, 0])/bins)
    density = probability/cell_area
    density.setflags(write=False)
    bounds.setflags(write=False)
    unique_previews = {id(row[1]): row[1] for row in rows}
    preview = EnsemblePreview(density, bounds, float(probability.sum()),
        sum(int(rows[i][1].mode_count) for i in active),
        sum(int(item.retained_bytes) for item in unique_previews.values())+density.nbytes+bounds.nbytes,
        tuple(sorted({shape for i in active for shape in rows[i][1].grid_shapes})),
        axial_bz_t=(getattr(rows[active[0]][1], "axial_bz_t", None)
            if all(getattr(rows[i][1], "axial_bz_t", None) ==
                   getattr(rows[active[0]][1], "axial_bz_t", None) for i in active)
            else None))
    return observation, preview
