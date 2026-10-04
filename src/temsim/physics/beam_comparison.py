"""Read-only, full-population moments for a matched particle/wave plane.

The caller owns captured-input identity. This module checks the observation
plane and current reference; it cannot establish provenance from equal numbers.
It never compares display subsets, normalises away lost electrons, or adds
complex amplitudes belonging to different incoherent modes.
"""
from __future__ import annotations

import math
import numpy as np

from temsim.immutable_json import freeze_json
from temsim.optics.electron_gun.tip_coherence import TIP_REFERENCE
from temsim.physics.multiplane_wave import PlaneWave


def _scalar(value, name, *, nonnegative=False):
    if isinstance(value, (bool, np.bool_)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    value = float(value)
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _summary(z, reference, weight, centre, covariance, energy, energy_variance, **extra):
    if weight < 0 or weight > 1 + 1e-8:
        raise ValueError("Plane probability must remain relative to the emitted current")
    return freeze_json(dict(z_mm=z, reference_current_a=reference,
        current_a=reference*weight, source_fraction=weight,
        centroid_xy_m=None if centre is None else tuple(map(float, centre)),
        rms_xy_m=None if covariance is None else tuple(np.sqrt(np.maximum(np.diag(covariance), 0.))),
        radial_rms_m=None if covariance is None else math.sqrt(max(float(np.trace(covariance)), 0.)),
        mean_energy_ev=energy,
        rms_energy_ev=None if energy_variance is None else math.sqrt(max(energy_variance, 0.)),
        coordinate_frame="laboratory XY; +Z downstream", **extra))


def particle_plane_moments(plane):
    """All retained rays crossing Z, including their executed kinetic energy."""
    if not plane.weights_valid or plane.source_current_pa is None:
        raise ValueError("Comparison requires valid particle weights and source current")
    if plane.coordinate_frame != "column":
        raise ValueError("Comparison requires the laboratory column coordinate frame")
    z = _scalar(plane.z_mm, "Particle Z")
    reference = _scalar(plane.source_current_pa, "Particle source current", nonnegative=True)*1e-12
    x, y, weights = (np.asarray(a, dtype=float) for a in (plane.x_m, plane.y_m, plane.source_fraction))
    if (x.ndim != 1 or y.shape != x.shape or weights.shape != x.shape
            or np.any(~np.isfinite(x)) or np.any(~np.isfinite(y))
            or np.any(~np.isfinite(weights)) or np.any(weights < 0)):
        raise ValueError("Particle comparison arrays must be finite and aligned")
    total = float(weights.sum())
    centre = covariance = energy = variance = None
    if total > 0:
        xy = np.stack((x, y))
        centre = xy @ weights / total
        delta = xy-centre[:, None]
        covariance = (delta*weights) @ delta.T / total
        raw = getattr(plane, "kinetic_energy_ev", None)
        if raw is not None:
            values = np.asarray(raw, float)
            positive = weights > 0
            if values.shape != weights.shape:
                raise ValueError("Particle energy must align with retained rays")
            if np.all(np.isfinite(values[positive])) and np.all(values[positive] > 0):
                energy = float(np.dot(weights[positive], values[positive])/total)
                variance = float(np.dot(weights[positive], (values[positive]-energy)**2)/total)
    return _summary(z, reference, total, centre, covariance, energy, variance,
        representation="classical particles", provenance=plane.provenance,
        energy_definition="executed kinetic energy; interpolated only between retained planes",
        population_count=plane.ray_count)


def _mode_moments(plane):
    """Affine cell moments without a full coordinate mesh or a remapped image."""
    if not isinstance(plane, PlaneWave):
        raise ValueError("This comparison needs a forward Cartesian wave checkpoint; two-way near fields need flux readout")
    ny, nx = plane.amplitude.shape
    xi = np.arange(nx, dtype=float)-nx//2
    yi = np.arange(ny, dtype=float)-ny//2
    # Row chunks cap temporary memory independently of the saved wave size.
    xweights = np.zeros(nx)
    yweights = np.zeros(ny)
    for first in range(0, ny, 64):
        probability = abs(plane.amplitude[first:first+64])**2
        xweights += probability.sum(axis=0)
        yweights[first:first+64] = probability.sum(axis=1)
    total = float(xweights.sum())
    if total <= 0:
        return total, None, None
    centre_index = np.array((xi @ xweights, yi @ yweights))/total
    dx, dy = xi-centre_index[0], yi-centre_index[1]
    cross = 0.
    for first in range(0, ny, 64):
        probability = abs(plane.amplitude[first:first+64])**2
        cross += float(dy[first:first+64] @ probability @ dx)
    covariance_index = np.array(((dx**2 @ xweights, cross), (cross, dy**2 @ yweights)))/total
    return (total, plane.origin_m+plane.basis_m @ centre_index,
            plane.basis_m @ covariance_index @ plane.basis_m.T)


def wave_plane_moments(checkpoint):
    """Add per-mode intensities, retaining source loss and between-mode width."""
    if checkpoint.beam.reference_plane != TIP_REFERENCE:
        raise ValueError("Wave comparison requires an executed physical-tip reference")
    z = _scalar(checkpoint.plane_z_mm, "Wave Z")
    reference = _scalar(checkpoint.reference_current_a, "Wave source current", nonnegative=True)
    total = 0.
    centre = np.zeros(2)
    covariance_sum = np.zeros((2, 2))
    energy = variance_sum = 0.
    count = 0
    for mode in checkpoint.beam.modes:
        count += 1
        norm, mode_centre, mode_covariance = _mode_moments(mode.plane)
        weight = _scalar(mode.weight_per_reference_electron, "Mode probability", nonnegative=True)*norm
        if weight == 0:
            continue
        mode_energy = _scalar(mode.energy_kev, "Mode reference energy", nonnegative=True)*1000.
        if mode_energy <= 0:
            raise ValueError("Mode reference energy must be positive")
        combined = total+weight
        delta = mode_centre-centre
        covariance_sum += weight*mode_covariance + total*weight/combined*np.outer(delta, delta)
        centre += weight/combined*delta
        de = mode_energy-energy
        variance_sum += total*weight/combined*de*de
        energy += weight/combined*de
        total = combined
    return _summary(z, reference, total, centre if total else None,
        covariance_sum/total if total else None, energy if total else None,
        variance_sum/total if total else None,
        representation="incoherent sum of propagated coherent modes", population_count=count,
        energy_definition="forward-mode reference kinetic energy; not a local energy map inside a transverse electric field")


def compare_beam_planes(particle_plane, wave_checkpoint):
    """Compare geometry/current, not phase or equality of quantum/classical laws.

    The workflow must additionally match its frozen input identity. Historical
    particle energies remain unavailable rather than using nominal voltage.
    """
    if not math.isclose(float(particle_plane.z_mm), float(wave_checkpoint.plane_z_mm),
                        rel_tol=0., abs_tol=1e-9):
        raise ValueError("Particle and wave observations must be at the same Z")
    particles = particle_plane_moments(particle_plane)
    wave = wave_plane_moments(wave_checkpoint)
    if not math.isclose(particles["reference_current_a"], wave["reference_current_a"],
                        rel_tol=1e-10, abs_tol=1e-24):
        raise ValueError("Particle and wave results must share the emitted current reference")
    return freeze_json({"schema": "same-plane-beam-moments-v1", "particle": particles, "wave": wave,
        "scope": "Observable comparison only; input provenance is checked by the paired workflow. "
                 "Different interference, diffraction, sampling and model approximations may produce different widths. "
                 "No aggregate phase is defined for an incoherent mixture."})
