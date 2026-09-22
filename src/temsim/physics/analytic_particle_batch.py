"""Bounded consecutive analytic gun steps between physical/history events.

Every accepted step retains the existing spatial and impulse caps, doubled
Boris error test and separate static-energy projection. A possible boundary
event ends the batch and is resolved by the original tracing routines. This
is execution-local scratch work, never an alternative electron source.
"""
from __future__ import annotations

import math
import threading

import numpy as np

from . import analytic_particle_step as stepping
from .analytic_gun_field import njit
from .relativistic_lorentz import RelativisticPhaseSpace


MAX_BATCH_STEPS = 16
BATCH_EXECUTION_SCHEMA = "analytic-no-event-batch-v1"
E, M, C = stepping.E, stepping.M, stepping.C


def _outer_projection(position, momentum, indices, energy, terms):
    # Preserve the NumPy outer projection's operation order, including d2's
    # SI conversion before the radial product and division before scaling p.
    for local in range(len(indices)):
        i = indices[local]
        x, y, z = position[i]
        phi, d2 = 0., 0.
        for start, end, amplitude in terms:
            length = max(end-start, 1e-9)
            u = (z*1000.-start)/length
            t = min(1., max(0., u))
            phi += amplitude*t**3*(10.-15.*t+6.*t**2)
            if 0. < u < 1.:
                d2 += amplitude*(60.*t-180.*t**2+120.*t**3)/length**2
        d2_m = d2*1e6
        potential = phi-.25*(x*x+y*y)*d2_m
        target = max(energy[local]+potential, 1e-6)
        if not math.isfinite(target):
            raise ValueError("Kinetic energy must be finite and non-negative.")
        px, py, pz = momentum[i]
        norm = math.sqrt(px*px+py*py+pz*pz)
        if norm <= 0.:
            raise ValueError("Momentum direction must be non-zero.")
        energy_j = target*E
        rest_energy_j = M*C**2
        magnitude = math.sqrt(energy_j*(energy_j+2.*rest_energy_j))/C
        momentum[i, 0] = px/norm*magnitude
        momentum[i, 1] = py/norm*magnitude
        momentum[i, 2] = pz/norm*magnitude


def _spatial_dt(position, momentum, indices, supports, trace_step, drift_step):
    low, high, maximum_vz = math.inf, -math.inf, -math.inf
    for i in indices:
        px, py, pz = momentum[i]
        gamma = math.sqrt(1.+(px*px+py*py+pz*pz)/(M*C)**2)
        vz = pz/(gamma*M)
        if vz <= 0.:
            return 0.  # The original outer loop owns backstream bookkeeping.
        maximum_vz = max(maximum_vz, vz)
        z = position[i, 2]*1000.
        low, high = min(low, z), max(high, z)
    step = drift_step
    for start, end in supports:
        if low <= end and high >= start:
            step = min(step, trace_step)
        elif high < start:
            step = min(step, max(start-high, 1e-10))
    return step*1e-3/max(maximum_vz, 1.)


def _boundary_candidate(old, new, momentum, indices, bores, planes,
                        dpa_passed, c1_passed):
    for i in indices:
        if momentum[i, 2] <= 0.:
            return True
        old_z, new_z = old[i, 2], new[i, 2]
        if ((not dpa_passed[i] and old_z <= planes[0] <= new_z)
                or (not c1_passed[i] and old_z <= planes[1] <= new_z)
                or old_z <= planes[2] <= new_z):
            return True
        radius = math.hypot(new[i, 0], new[i, 1])*1000.
        z_mm = new_z*1000.
        for center, half_length, radius_limit in bores:
            # Guard conservatively near a boundary: NumPy's hypot and libm's
            # hypot need not round identically. False positives only return to
            # the exact original NumPy test; no particle is stopped here.
            if (abs(z_mm-center) <= half_length
                    and radius >= radius_limit-max(abs(radius_limit)*1e-12, 1e-15)):
                return True
    return False


def _batch(position, momentum, time_s, indices, energy, terms, magnetic,
           supports, trace_step, drift_step, bores, planes, dpa_passed,
           c1_passed, impulse, maximum_steps, status, half, limits, invalid,
           parallel):
    current_x, current_p = position.copy(), momentum.copy()
    next_x, next_p = position.copy(), momentum.copy()
    previous_time, dt, count = time_s, 0., 0
    for _ in range(maximum_steps):
        proposed = _spatial_dt(current_x, current_p, indices, supports,
                               trace_step, drift_step)
        if proposed <= 0.:
            break
        bounded = stepping._compiled_prepared_time_step(current_x, current_p,
            indices, proposed, impulse, terms, magnetic, limits, invalid, parallel)
        if not math.isfinite(bounded) or bounded == 0.:
            raise ValueError("Time step must be finite and non-zero.")
        dt = stepping._compiled_prepared_step(current_x, current_p, bounded,
            indices, energy, terms, magnetic, next_x, next_p, status, half, parallel)
        _outer_projection(next_x, next_p, indices, energy, terms)
        previous_time = time_s
        time_s += dt
        count += 1
        event = _boundary_candidate(current_x, next_x, next_p, indices, bores,
                                    planes, dpa_passed, c1_passed)
        current_x, next_x = next_x, current_x
        current_p, next_p = next_p, current_p
        if event:
            break
    return current_x, current_p, time_s, next_x, next_p, previous_time, dt, count


if njit is not None:
    _outer_projection = njit(cache=True, nogil=True)(_outer_projection)
    _spatial_dt = njit(cache=True, nogil=True)(_spatial_dt)
    _boundary_candidate = njit(cache=True, nogil=True)(_boundary_candidate)
    _compiled_batch = njit(cache=True, nogil=True)(_batch)
else:
    _compiled_batch = None


def _readonly_cancellation(callback):
    """Unknown callbacks may mutate inputs or depend on per-step invocation."""
    return (callback is None or (
        type(getattr(callback, "__self__", None)) is threading.Event
        and getattr(callback, "__func__", None) is threading.Event.is_set
        and "is_set" not in callback.__self__.__dict__))


def _unmodified_gun(gun):
    # Imported at execution time to avoid field_emission -> tracing cycles.
    from temsim.optics.electron_gun.field_emission import FieldEmissionGun
    from temsim.optics.electron_gun.electrostatic import AcceleratorStage
    if type(gun) is not FieldEmissionGun:
        return False
    if any(type(stage) is not AcceleratorStage for stage in gun.accelerator.stages):
        return False
    for name in ("integration_step_mm_at", "field_supports_mm", "bore_components",
                 "monochromator_installed", "electric_field", "magnetic_field",
                 "exit_plane_z_mm"):
        if name in gun.__dict__:
            return False
        value = FieldEmissionGun.__dict__[name]
        function = value.fget if isinstance(value, property) else value
        if (getattr(function, "__module__", None) != FieldEmissionGun.__module__
                or getattr(function, "__qualname__", None) != "FieldEmissionGun."+name
                or hasattr(function, "__wrapped__")):
            return False
    from temsim.optics.electron_gun.aperture import GunAperture
    for aperture in (gun.dpa_aperture, gun.c1_aperture):
        if type(aperture) is not GunAperture:
            return False
        for name in ("transmission_mask", "z_mm", "optical_reference_from_tip_mm"):
            value = GunAperture.__dict__[name]
            function = value.fget if isinstance(value, property) else value
            if (name in aperture.__dict__
                    or getattr(function, "__module__", None) != GunAperture.__module__
                    or getattr(function, "__qualname__", None) != "GunAperture."+name
                    or hasattr(function, "__wrapped__")):
                return False
    return (getattr(gun.emitter, "surface_model", None) is None
            and not gun.monochromator_installed
            and not getattr(gun, "_vacuum_regions", ())
            and getattr(gun, "compiled_particle_batches", True))


class AnalyticParticleBatch:
    """One trace's bounded accelerator, with current inputs read each batch."""

    def __init__(self, gun, execution, cancelled):
        self.gun, self.execution, self.cancelled = gun, execution, cancelled
        # Small local counters expose actual admission to diagnostic callers;
        # they never influence physics, history cadence, or persistent caches.
        self.accepted_steps = 0
        self.completed_batches = 0
        self.maximum_batch_steps = 0

    def advance(self, phase, active, invariant_energy, dpa_passed, c1_passed,
                *, maximum_steps, impulse):
        from temsim.optics.electron_gun import tracing
        if (not _unmodified_gun(self.gun)
                or not _readonly_cancellation(self.cancelled)
                or any(getattr(tracing, name) is not original
                       for name, original in tracing._BATCH_ORIGINAL_FUNCTIONS.items())
                or not getattr(self.gun.electric_field, "compiled_axial", True)):
            return None
        execution = self.execution
        if (type(execution) is not stepping.AnalyticParticleExecution
                or any(name in execution.__dict__ or getattr(type(execution), name) is not original
                       for name, original in stepping._ORIGINAL_EXECUTION_METHODS.items())):
            return None
        if not execution.begin_step(phase, active) or not execution._active_count:
            return None
        indices = execution._active_indices
        energy = np.asarray(invariant_energy, dtype=np.float64)
        if energy.shape != active.shape or not np.all(np.isfinite(energy)):
            return None
        gun = self.gun
        supports = np.asarray(gun.field_supports_mm, dtype=np.float64).reshape(-1, 2)
        bores = np.asarray([(part.mechanical_center_from_tip_mm,
                             .5*part.mechanical_length_mm,
                             .5*part.mechanical_clear_bore_diameter_mm)
                            for part in gun.bore_components], dtype=np.float64).reshape(-1, 3)
        planes = np.array([gun.dpa_aperture.z_mm*1e-3,
                           gun.c1_aperture.z_mm*1e-3, gun.exit_plane_z_mm*1e-3])
        trace_step, drift_step = float(gun.trace_step_mm), float(gun.drift_step_mm)
        if (not np.all(np.isfinite(supports)) or not np.all(np.isfinite(bores))
                or not np.all(np.isfinite(planes)) or not math.isfinite(trace_step)
                or not math.isfinite(drift_step) or trace_step <= 0. or drift_step <= 0.):
            return None
        maximum_steps = min(MAX_BATCH_STEPS, int(maximum_steps))
        if maximum_steps <= 0:
            return None
        execution._ready = False
        execution._phase = None
        values = _compiled_batch(phase.position_m, phase.momentum_kg_m_per_s,
            phase.time_s, indices, energy[indices], *execution._parameters,
            supports, trace_step, drift_step, bores, planes, dpa_passed,
            c1_passed, float(impulse), maximum_steps, execution._active_status,
            execution._active_half, execution._active_limits,
            execution._active_invalid, execution._parallel)
        x, p, time_s, old_x, old_p, old_time, dt, count = values
        if count == 0:
            return None
        self.accepted_steps += count
        self.completed_batches += 1
        self.maximum_batch_steps = max(self.maximum_batch_steps, count)
        return (RelativisticPhaseSpace(x, p, time_s), old_x, old_p,
                old_time, dt, count)


def prepare_analytic_batch(gun, execution, cancelled):
    if (_compiled_batch is None or execution is None
            or not _unmodified_gun(gun) or not _readonly_cancellation(cancelled)):
        return None
    return AnalyticParticleBatch(gun, execution, cancelled)
