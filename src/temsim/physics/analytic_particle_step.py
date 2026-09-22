"""Optional compiled form of the analytic gun's existing adaptive Boris step.

Positions and mechanical momenta are SI float64 arrays with shape (N, 3).
The same two-half-step acceptance, static-energy projection, and error budgets
as the NumPy reference are retained. Only the original analytic field providers
are supported; custom/combined fields continue through the general reference.
The compatibility functions read current fields on every call. An optional
execution-local workspace refreshes its fields at each begin_step boundary;
it is never a persistent source or a cache shared between gun executions.
"""

from __future__ import annotations

import math

import numpy as np

from temsim.optics.electron_gun.alignment import (
    FegMagneticField, GunDeflector, GunStigmator,
)
from temsim.optics.electron_gun.electrostatic import (
    AcceleratorColumn, ElectrostaticGunLens, ExtractorElectrode,
    FegElectrostaticField,
)
from .analytic_gun_field import njit
if njit is not None:
    from numba import get_num_threads, prange
else:
    prange = range
from .relativistic_lorentz import (
    ELEMENTARY_CHARGE_C as E,
    ELECTRON_MASS_KG as M,
    SPEED_OF_LIGHT_M_PER_S as C,
    RelativisticPhaseSpace,
)


_METHODS = {
    FegElectrostaticField: (
        "axial_potential_v_and_derivatives_per_mm",
        "potential_v_at_global_positions", "field_at_global_positions_v_per_m",
    ),
    ExtractorElectrode: ("axial_potential_v_and_derivatives_per_mm",),
    ElectrostaticGunLens: ("axial_potential_v_and_derivatives_per_mm",),
    AcceleratorColumn: ("normalized_potential_and_derivatives_per_mm",),
    FegMagneticField: ("field_at_global_positions_t",),
    GunDeflector: ("field_at_global_positions_t",),
    GunStigmator: ("field_at_global_positions_t",),
}
_ORIGINAL_METHODS = {
    cls: tuple(getattr(cls, name) for name in names)
    for cls, names in _METHODS.items()
}


def _original_provider(provider, cls):
    return (type(provider) is cls and all(
        name not in provider.__dict__ and getattr(cls, name) is original
        for name, original in zip(_METHODS[cls], _ORIGINAL_METHODS[cls])
    ))


def _electric(x, y, z, terms):
    """The compact quintic potential and its radial-order-two field."""
    phi, d1, d2, d3 = 0., 0., 0., 0.
    for start, end, amplitude in terms:
        length = max(end-start, 1e-9)
        u = (z*1000.-start)/length
        t = min(1., max(0., u))
        phi += amplitude*t**3*(10.-15.*t+6.*t**2)
        if 0. < u < 1.:
            d1 += amplitude*(30.*t**2-60.*t**3+30.*t**4)/length
            d2 += amplitude*(60.*t-180.*t**2+120.*t**3)/length**2
            d3 += amplitude*(60.-360.*t+360.*t**2)/length**3
    radius2 = x*x+y*y
    return (phi-.25*radius2*d2*1e6,
            .5*x*d2*1e6, .5*y*d2*1e6, -d1*1e3+.25*radius2*d3*1e9)


def _window(z, center, half, edge):
    edge = max(edge, 1e-6)
    left = min(1., max(0., (z-center+half+edge)/(2.*edge)))
    right = min(1., max(0., (z-center-half+edge)/(2.*edge)))
    return left**3*(10.-15.*left+6.*left**2)-right**3*(10.-15.*right+6.*right**2)


def _magnetic(x, y, z, parameters):
    """Finite deflector coils and rotated quadrupole, including blanking."""
    bx, by = 0., 0.
    for j in (0, 5):
        center, half, edge, field_x, field_y = parameters[j:j+5]
        if field_x != 0. or field_y != 0.:
            envelope = _window(z*1000., center, half, edge)
            bx += field_x*envelope
            by += field_y*envelope
    center, half, edge, gradient, angle = parameters[10:15]
    if gradient != 0.:
        envelope = _window(z*1000., center, half, edge)
        cosine, sine = math.cos(angle), math.sin(angle)
        u, v = cosine*x+sine*y, -sine*x+cosine*y
        bu, bv = gradient*v*envelope, gradient*u*envelope
        bx += cosine*bu-sine*bv
        by += sine*bu+cosine*bv
    return bx, by


def _advance_values(x, y, z, px, py, pz, dt, energy, terms, magnetic):
    gamma = math.sqrt(1.+(px*px+py*py+pz*pz)/(M*C)**2)
    mx, my, mz = (x+.5*dt*px/(gamma*M), y+.5*dt*py/(gamma*M),
                  z+.5*dt*pz/(gamma*M))
    _, ex, ey, ez = _electric(mx, my, mz, terms)
    bx, by = _magnetic(mx, my, mz, magnetic)
    ix, iy, iz = -.5*E*dt*ex, -.5*E*dt*ey, -.5*E*dt*ez
    ux, uy, uz = px+ix, py+iy, pz+iz
    gamma = math.sqrt(1.+(ux*ux+uy*uy+uz*uz)/(M*C)**2)
    tx, ty = -E*dt*bx/(2.*M*gamma), -E*dt*by/(2.*M*gamma)
    factor = 2./(1.+tx*tx+ty*ty)
    sx, sy = factor*tx, factor*ty
    vx, vy, vz = ux-uz*ty, uy+uz*tx, uz+ux*ty-uy*tx
    px, py, pz = ux-vz*sy+ix, uy+vz*sx+iy, uz+vx*sy-vy*sx+iz
    gamma = math.sqrt(1.+(px*px+py*py+pz*pz)/(M*C)**2)
    x, y, z = mx+.5*dt*px/(gamma*M), my+.5*dt*py/(gamma*M), mz+.5*dt*pz/(gamma*M)
    potential, _, _, _ = _electric(x, y, z, terms)
    energy_j = max(energy+potential, 1e-6)*E
    norm = math.sqrt(px*px+py*py+pz*pz)
    if norm <= 0.:
        return x, y, z, px, py, pz, 2
    magnitude = math.sqrt(energy_j*(energy_j+2.*M*C**2))/C
    px, py, pz = px/norm*magnitude, py/norm*magnitude, pz/norm*magnitude
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)
            and math.isfinite(px) and math.isfinite(py) and math.isfinite(pz)):
        return x, y, z, px, py, pz, 3
    return x, y, z, px, py, pz, 0


def _advance(x, y, z, px, py, pz, dt, energy, terms, magnetic):
    values = _advance_values(x, y, z, px, py, pz, dt, energy, terms, magnetic)
    if values[6] == 2:
        raise ValueError("Momentum direction must be non-zero.")
    if values[6] == 3:
        raise ValueError("Relativistic phase-space values must be finite.")
    return values[:6]


def _parallel_attempt(position, momentum, indices, energy, dt, terms, magnetic,
                      out_x, out_p, status, half_cache, reuse_half):
    # Never raise inside prange: Numba does not guarantee propagation from
    # independent worker iterations. Every active particle reports its status.
    for local_index in prange(len(indices)):
        i = indices[local_index]
        x, y, z = position[i]
        px, py, pz = momentum[i]
        invariant = energy[local_index]
        # A rejected trial halves dt. Its already validated first half-step is
        # exactly the next trial's full step: same initial state, static field
        # snapshot, duration and energy projection. No accepted step is reused.
        if reuse_half:
            full = (half_cache[local_index, 0], half_cache[local_index, 1],
                    half_cache[local_index, 2], half_cache[local_index, 3],
                    half_cache[local_index, 4], half_cache[local_index, 5], 0)
        else:
            full = _advance_values(x, y, z, px, py, pz, dt, invariant, terms, magnetic)
        if full[6] != 0:
            status[local_index] = full[6]
            continue
        half = _advance_values(x, y, z, px, py, pz, .5*dt, invariant, terms, magnetic)
        if half[6] != 0:
            status[local_index] = half[6]
            continue
        half_values = half[:6]
        for coordinate in range(6):
            half_cache[local_index, coordinate] = half_values[coordinate]
        fine = _advance_values(*half[:6], .5*dt, invariant, terms, magnetic)
        if fine[6] != 0:
            status[local_index] = fine[6]
            continue
        fx, fy, fz, fpx, fpy, fpz = fine[:6]
        dx, dy = fx-full[0], fy-full[1]
        dpx, dpy, dpz = fpx-full[3], fpy-full[4], fpz-full[5]
        pscale = max(math.sqrt(fpx*fpx+fpy*fpy+fpz*fpz), 1e-30)
        transverse_scale = max(math.sqrt(fpx*fpx+fpy*fpy), pscale*1e-5)
        status[local_index] = int(
            math.sqrt(dpx*dpx+dpy*dpy)/transverse_scale > 1e-5
            or math.sqrt(dpx*dpx+dpy*dpy+dpz*dpz)/pscale > 1e-6
            or math.sqrt(dx*dx+dy*dy) > 1e-13+1e-6*math.sqrt(fx*fx+fy*fy))
        out_x[i, 0], out_x[i, 1], out_x[i, 2] = fx, fy, fz
        out_p[i, 0], out_p[i, 1], out_p[i, 2] = fpx, fpy, fpz


def _parallel_step(position, momentum, dt, active, energy, terms, magnetic):
    indices = np.flatnonzero(active)
    status = np.empty(len(indices), np.uint8)
    half_cache = np.empty((len(indices), 6), np.float64)
    out_x, out_p = position.copy(), momentum.copy()
    for attempt in range(32):
        _parallel_attempt(position, momentum, indices, energy, dt, terms, magnetic,
                          out_x, out_p, status, half_cache, attempt > 0)
        accepted = True
        for code in status:
            if code == 2:
                raise ValueError("Momentum direction must be non-zero.")
            if code == 3:
                raise ValueError("Relativistic phase-space values must be finite.")
            if code == 1:
                accepted = False
        if accepted:
            return out_x, out_p, dt
        dt *= .5
    raise ValueError("Analytic gun integration did not meet the transverse error budget")


def _impulse_value(position, momentum, i, dt, impulse, terms, magnetic):
    px, py, pz = momentum[i]
    pnorm = math.sqrt(px*px+py*py+pz*pz)
    gamma = math.sqrt(1.+(px*px+py*py+pz*pz)/(M*C)**2)
    vx, vy, vz = px/(gamma*M), py/(gamma*M), pz/(gamma*M)
    maximum_force, invalid = 0., 0
    for fraction in (0., .5):
        x = position[i, 0]+fraction*dt*vx
        y = position[i, 1]+fraction*dt*vy
        z = position[i, 2]+fraction*dt*vz
        _, ex, ey, ez = _electric(x, y, z, terms)
        bx, by = _magnetic(x, y, z, magnetic)
        if not (math.isfinite(ex) and math.isfinite(ey) and math.isfinite(ez)
                and math.isfinite(bx) and math.isfinite(by)):
            invalid = 1
        fx, fy, fz = -E*(ex-vz*by), -E*(ey+vz*bx), -E*(ez+vx*by-vy*bx)
        maximum_force = max(maximum_force, math.sqrt(fx*fx+fy*fy+fz*fz))
    limit = impulse*pnorm/maximum_force if maximum_force > 0. else math.inf
    return limit, invalid


def _impulse_limits(position, momentum, indices, dt, impulse, terms, magnetic):
    limits = np.empty(len(indices), np.float64)
    invalid = np.empty(len(indices), np.uint8)
    for local_index in prange(len(indices)):
        limits[local_index], invalid[local_index] = _impulse_value(
            position, momentum, indices[local_index], dt, impulse, terms, magnetic)
    return limits, invalid


def _impulse_limits_into(position, momentum, indices, dt, impulse, terms, magnetic,
                         limits, invalid):
    for local_index in prange(len(indices)):
        limits[local_index], invalid[local_index] = _impulse_value(
            position, momentum, indices[local_index], dt, impulse, terms, magnetic)


def _active_indices_into(active, indices):
    count = 0
    for i in range(len(active)):
        if active[i]:
            indices[count] = i
            count += 1
    return count


def _prepared_time_step(position, momentum, indices, dt, impulse, terms, magnetic,
                        limits, invalid, parallel):
    if parallel:
        _parallel_impulse_limits_into(position, momentum, indices, dt, impulse,
            terms, magnetic, limits, invalid)
    else:
        _serial_impulse_limits_into(position, momentum, indices, dt, impulse,
            terms, magnetic, limits, invalid)
    # Reduce in ascending particle order after every field has been checked,
    # retaining the compatibility path's failure semantics for any bad field.
    bounded = dt
    for i in range(len(indices)):
        if invalid[i]:
            raise ValueError("Electric and magnetic field values must be finite.")
        bounded = min(bounded, limits[i])
    return bounded


def _prepared_step(position, momentum, dt, indices, energy, terms, magnetic,
                   out_x, out_p, status, half_cache, parallel):
    # Scratch outputs never alias caller-owned phase arrays. The public phase
    # constructor copies these scratch arrays before a subsequent step can
    # reuse them, so event interpolation and retained histories stay unchanged.
    out_x[:, :] = position
    out_p[:, :] = momentum
    for attempt in range(32):
        if parallel:
            _parallel_attempt(position, momentum, indices, energy, dt, terms,
                magnetic, out_x, out_p, status, half_cache, attempt > 0)
        else:
            _serial_attempt(position, momentum, indices, energy, dt, terms,
                magnetic, out_x, out_p, status, half_cache, attempt > 0)
        accepted = True
        for code in status:
            if code == 2:
                raise ValueError("Momentum direction must be non-zero.")
            if code == 3:
                raise ValueError("Relativistic phase-space values must be finite.")
            if code == 1:
                accepted = False
        if accepted:
            return dt
        dt *= .5
    raise ValueError("Analytic gun integration did not meet the transverse error budget")


def _step(position, momentum, dt, active, energy, terms, magnetic):
    out_x, out_p = position.copy(), momentum.copy()
    half_cache = np.empty((len(energy), 6), np.float64)
    for attempt in range(32):
        accepted = True
        local_index = 0
        for i in range(len(position)):
            if not active[i]:
                continue
            x, y, z = position[i]
            px, py, pz = momentum[i]
            invariant = energy[local_index]
            if attempt:
                full = (half_cache[local_index, 0], half_cache[local_index, 1],
                        half_cache[local_index, 2], half_cache[local_index, 3],
                        half_cache[local_index, 4], half_cache[local_index, 5])
            else:
                full = _advance(x, y, z, px, py, pz, dt, invariant, terms, magnetic)
            half = _advance(x, y, z, px, py, pz, .5*dt, invariant, terms, magnetic)
            for coordinate in range(6):
                half_cache[local_index, coordinate] = half[coordinate]
            local_index += 1
            fine = _advance(*half, .5*dt, invariant, terms, magnetic)
            fx, fy, fz, fpx, fpy, fpz = fine
            dx, dy = fx-full[0], fy-full[1]
            dpx, dpy, dpz = fpx-full[3], fpy-full[4], fpz-full[5]
            pscale = max(math.sqrt(fpx*fpx+fpy*fpy+fpz*fpz), 1e-30)
            transverse_scale = max(math.sqrt(fpx*fpx+fpy*fpy), pscale*1e-5)
            if (math.sqrt(dpx*dpx+dpy*dpy)/transverse_scale > 1e-5
                    or math.sqrt(dpx*dpx+dpy*dpy+dpz*dpz)/pscale > 1e-6
                    or math.sqrt(dx*dx+dy*dy) > 1e-13+1e-6*math.sqrt(fx*fx+fy*fy)):
                accepted = False
            out_x[i, 0], out_x[i, 1], out_x[i, 2] = fx, fy, fz
            out_p[i, 0], out_p[i, 1], out_p[i, 2] = fpx, fpy, fpz
        if accepted:
            return out_x, out_p, dt
        dt *= .5
    raise ValueError("Analytic gun integration did not meet the transverse error budget")


if njit is not None:
    _electric = njit(cache=True)(_electric)
    _window = njit(cache=True)(_window)
    _magnetic = njit(cache=True)(_magnetic)
    _advance_values = njit(cache=True)(_advance_values)
    _advance = njit(cache=True)(_advance)
    _impulse_value = njit(cache=True, inline="always")(_impulse_value)
    _compiled_step = njit(cache=True, nogil=True)(_step)
    _serial_attempt = njit(cache=True, nogil=True)(_parallel_attempt)
    _parallel_attempt = njit(cache=True, parallel=True, nogil=True)(_parallel_attempt)
    _compiled_parallel_step = njit(cache=True, nogil=True)(_parallel_step)
    _compiled_impulse_limits = njit(cache=True, nogil=True)(_impulse_limits)
    _parallel_impulse_limits = njit(cache=True, parallel=True, nogil=True)(_impulse_limits)
    _serial_impulse_limits_into = njit(cache=True, nogil=True)(_impulse_limits_into)
    _parallel_impulse_limits_into = njit(cache=True, parallel=True, nogil=True)(_impulse_limits_into)
    _compiled_active_indices_into = njit(cache=True, nogil=True)(_active_indices_into)
    _compiled_prepared_time_step = njit(cache=True, nogil=True)(_prepared_time_step)
    _compiled_prepared_step = njit(cache=True, nogil=True)(_prepared_step)
else:
    _compiled_step = None
    _compiled_parallel_step = None
    _compiled_impulse_limits = None
    _parallel_impulse_limits = None
    _compiled_active_indices_into = None
    _compiled_prepared_time_step = None
    _compiled_prepared_step = None


def _field_parameters(electric, magnetic):
    """Read current model inputs without caching or accepting added physics."""
    if (not getattr(electric, "compiled_particle_steps", True)
            or not getattr(magnetic, "compiled_particle_steps", True)
            or not _original_provider(electric, FegElectrostaticField)
            or not _original_provider(magnetic, FegMagneticField)):
        return None
    extractor, lens, accelerator = electric.extractor, electric.electrostatic_lens, electric.accelerator
    d, s = magnetic.deflector, magnetic.stigmator
    for obj, cls in ((extractor, ExtractorElectrode), (lens, ElectrostaticGunLens),
                     (accelerator, AcceleratorColumn), (d, GunDeflector), (s, GunStigmator)):
        if not _original_provider(obj, cls):
            return None
    start = extractor.transition_start_mm+extractor.field_center_offset_mm
    end = extractor.transition_end_mm+extractor.field_center_offset_mm
    terms = [(start, end, extractor.voltage_kv*1000.)]
    center = lens.optical_reference_from_tip_mm
    half, edge = .5*lens.mechanical_length_mm, max(lens.soft_edge_mm, 1e-6)
    amplitude = lens.voltage_kv*1000.*lens.potential_scale
    terms += [(center-half-edge, center-half+edge, amplitude),
              (center+half-edge, center+half+edge, -amplitude)]
    gain = accelerator.high_tension_kv*1000.-electric.emitter.emission_energy_ev-extractor.voltage_kv*1000.
    previous = 0.
    for stage in accelerator.stages:
        center = stage.center_from_tip_mm+accelerator.field_center_offset_mm
        terms.append((center-stage.soft_edge_mm, center+stage.soft_edge_mm,
                      gain*(stage.voltage_fraction-previous)))
        previous = stage.voltage_fraction
    upper = (d.upper_field_x_mt, d.upper_field_y_mt) if d.enabled else (0., 0.)
    lower = (d.lower_field_x_mt, d.lower_field_y_mt) if d.enabled else (0., 0.)
    if d.beam_blanked:
        upper, lower = (0., d.blanking_field_y_mt), (0., 0.)
    parameters = np.array([
        d.upper_center_from_tip_mm+d.field_center_offset_mm, .5*d.coil_length_mm,
        d.soft_edge_mm, upper[0]*1e-3, upper[1]*1e-3,
        d.lower_center_from_tip_mm+d.field_center_offset_mm, .5*d.coil_length_mm,
        d.soft_edge_mm, lower[0]*1e-3, lower[1]*1e-3,
        s.optical_reference_from_tip_mm, .5*s.effective_length_mm, s.soft_edge_mm,
        s.gradient_t_per_m if s.enabled else 0., math.radians(s.rotation_deg)])
    terms = np.asarray(terms, dtype=float)
    if not (np.all(np.isfinite(terms)) and np.all(np.isfinite(parameters))):
        return None  # General field validation owns invalid/custom inputs.
    return terms, parameters


def _current_field_key(gun, electric, magnetic):
    """Cheap exact consumed-input key, with current provider admission checks.

    Read on every outer step, including flags and nested component identity.
    A changed field is repacked; a custom/overridden provider is rejected.
    This key controls only execution-local scratch data, never source reuse.
    """
    if (not getattr(gun, "compiled_particle_steps", True)
            or not getattr(electric, "compiled_particle_steps", True)
            or not getattr(magnetic, "compiled_particle_steps", True)
            or not _original_provider(electric, FegElectrostaticField)
            or not _original_provider(magnetic, FegMagneticField)):
        return None
    projection = gun.electric_field
    if (not _original_provider(projection, FegElectrostaticField)
            or not getattr(projection, "compiled_particle_steps", True)
            or any(getattr(projection, name) is not getattr(electric, name)
                   for name in ("emitter", "extractor", "electrostatic_lens", "accelerator"))):
        return None
    extractor, lens, accelerator = electric.extractor, electric.electrostatic_lens, electric.accelerator
    d, s = magnetic.deflector, magnetic.stigmator
    for obj, cls in ((extractor, ExtractorElectrode), (lens, ElectrostaticGunLens),
                     (accelerator, AcceleratorColumn), (d, GunDeflector), (s, GunStigmator)):
        if not _original_provider(obj, cls):
            return None
    return (
        id(electric.emitter), id(extractor), id(lens), id(accelerator), id(d), id(s),
        electric.emitter.emission_energy_ev,
        extractor.transition_start_mm, extractor.transition_end_mm,
        extractor.field_center_offset_mm, extractor.voltage_kv,
        lens.optical_reference_from_tip_mm, lens.mechanical_length_mm,
        lens.soft_edge_mm, lens.voltage_kv, lens.potential_scale,
        accelerator.high_tension_kv, accelerator.field_center_offset_mm,
        tuple((stage.center_from_tip_mm, stage.soft_edge_mm, stage.voltage_fraction)
              for stage in accelerator.stages),
        d.upper_center_from_tip_mm, d.lower_center_from_tip_mm,
        d.field_center_offset_mm, d.coil_length_mm, d.soft_edge_mm,
        d.enabled, d.beam_blanked, d.blanking_field_y_mt,
        d.upper_field_x_mt, d.upper_field_y_mt, d.lower_field_x_mt, d.lower_field_y_mt,
        s.optical_reference_from_tip_mm, s.effective_length_mm, s.soft_edge_mm,
        s.enabled, s.gradient_t_per_m, s.rotation_deg,
    )


class AnalyticParticleExecution:
    """Scratch workspace for one analytic gun execution, never shared globally.

    ``begin_step`` reads current supported provider inputs and the active mask,
    then ``time_step`` and ``step`` consume that one outer-step snapshot. Call
    begin_step again after any accepted step or model/mask change; step consumes
    its prepared state. No trajectory or input array is retained as an output
    buffer: returned phase arrays are independently owned by their caller.

    This object is intentionally not thread safe or serializable as a source.
    Keep it local to a single trace_source_to_exit invocation.
    """

    def __init__(self, gun, magnetic, electric, ray_count, key, parameters):
        self._gun, self._magnetic, self._electric = gun, magnetic, electric
        self._ray_count = ray_count
        self._field_key, self._parameters = key, parameters
        self._indices = np.empty(ray_count, np.int64)
        self._status = np.empty(ray_count, np.uint8)
        self._half = np.empty((ray_count, 6), np.float64)
        self._limits = np.empty(ray_count, np.float64)
        self._invalid = np.empty(ray_count, np.uint8)
        self._out_x = np.empty((ray_count, 3), np.float64)
        self._out_p = np.empty((ray_count, 3), np.float64)
        self._active_count = -1
        self._ready = False
        self._phase = None

    def begin_step(self, phase, active):
        """Prepare this exact phase/mask, returning False for general fallback."""
        self._ready = False
        self._phase = None
        if (_compiled_prepared_step is None
                or phase.position_m.shape != (self._ray_count, 3)
                or phase.momentum_kg_m_per_s.shape != (self._ray_count, 3)
                or phase.position_m.dtype != np.float64
                or phase.momentum_kg_m_per_s.dtype != np.float64):
            return False
        active = np.asarray(active)
        if active.dtype != np.bool_ or active.shape != (self._ray_count,):
            return False
        key = _current_field_key(self._gun, self._electric, self._magnetic)
        if key is None:
            self._field_key, self._parameters = None, None
            return False
        if key != self._field_key or self._parameters is None:
            parameters = _field_parameters(self._electric, self._magnetic)
            if parameters is None:
                self._field_key, self._parameters = None, None
                return False
            self._field_key, self._parameters = key, parameters
        count = _compiled_active_indices_into(active, self._indices)
        if count != self._active_count:
            self._active_count = count
            self._active_indices = self._indices[:count]
            self._active_status = self._status[:count]
            self._active_half = self._half[:count]
            self._active_limits = self._limits[:count]
            self._active_invalid = self._invalid[:count]
        self._parallel = count >= 1024 and get_num_threads() > 1
        self._phase, self._ready = phase, True
        return True

    def time_step(self, dt, impulse):
        """Apply the unchanged current/midpoint impulse bound to this step."""
        if not self._ready:
            return None
        dt, impulse = float(dt), float(impulse)
        if not math.isfinite(dt) or dt <= 0. or not math.isfinite(impulse) or impulse <= 0.:
            return None
        phase = self._phase
        return _compiled_prepared_time_step(phase.position_m,
            phase.momentum_kg_m_per_s, self._active_indices, dt, impulse,
            *self._parameters, self._active_limits, self._active_invalid, self._parallel)

    def step(self, dt, invariant_energy):
        """Return independent phase arrays with the identical acceptance budget."""
        if not self._ready:
            return None
        self._ready = False
        phase, self._phase = self._phase, None
        dt = float(dt)
        if not math.isfinite(dt) or dt == 0.:
            raise ValueError("Time step must be finite and non-zero.")
        energy = np.asarray(invariant_energy, dtype=float)
        try:
            energy = np.broadcast_to(energy, (self._active_count,))
        except ValueError:
            return None
        if not np.all(np.isfinite(energy)):
            raise ValueError("Kinetic energy must be finite and non-negative.")
        accepted_dt = _compiled_prepared_step(phase.position_m,
            phase.momentum_kg_m_per_s, dt, self._active_indices, energy,
            *self._parameters, self._out_x, self._out_p, self._active_status,
            self._active_half, self._parallel)
        return RelativisticPhaseSpace(self._out_x, self._out_p,
            phase.time_s+accepted_dt), accepted_dt

    def project_momentum(self, position, momentum, invariant_energy):
        """Preserve the separate NumPy energy projection, or defer with None.

        The axial evaluator and subsequent NumPy operation order are the same
        as FegElectrostaticField.potential_v_at_global_positions followed by
        tracing._enforce_static_field_energy. This does not omit or fuse that
        second projection. Current parameters are checked again because this
        method is callable independently of the accepted step.
        """
        from .analytic_gun_field import compiled_evaluate
        from .relativistic_lorentz import momentum_from_kinetic_energy_ev
        if (compiled_evaluate is None
                or not getattr(self._gun.electric_field, "compiled_axial", True)):
            return None
        key = _current_field_key(self._gun, self._electric, self._magnetic)
        if key is None:
            return None
        if key != self._field_key or self._parameters is None:
            parameters = _field_parameters(self._electric, self._magnetic)
            if parameters is None:
                return None
            self._field_key, self._parameters = key, parameters
        position = np.asarray(position, dtype=float)
        if position.ndim != 2 or position.shape[-1] != 3:
            return None
        potential, _, d2_mm, _ = compiled_evaluate(position[:, 2]*1000., self._parameters[0])
        d2_m = d2_mm * 1.0e6
        radius_squared = position[:, 0] ** 2 + position[:, 1] ** 2
        potential = potential - 0.25 * radius_squared * d2_m
        energy = np.asarray(invariant_energy, dtype=float) + potential
        return momentum_from_kinetic_energy_ev(np.maximum(energy, 1e-6), momentum)


def prepare_analytic_execution(gun, magnetic, electric, ray_count):
    """Create local scratch state, or None for the unchanged general path.

    Fields remain live between begin_step calls; no source/gun state is cached
    across executions. Existing try_analytic_* interfaces remain independent.
    """
    if (_compiled_prepared_step is None or type(ray_count) not in (int, np.int64, np.int32)
            or ray_count < 0):
        return None
    key = _current_field_key(gun, electric, magnetic)
    if key is None:
        return None
    parameters = _field_parameters(electric, magnetic)
    if parameters is None:
        return None
    return AnalyticParticleExecution(gun, magnetic, electric, int(ray_count), key, parameters)


# A consecutive-step caller must not bypass per-step custom execution hooks.
_ORIGINAL_EXECUTION_METHODS = {name: getattr(AnalyticParticleExecution, name)
    for name in ("begin_step", "time_step", "step", "project_momentum")}


def try_analytic_time_step(phase, dt, active, magnetic, electric, impulse):
    """Same current/predicted-midpoint Lorentz impulse cap, or None to defer.

    This only evaluates the existing step bound. It does not change the
    spatial cap, error controller, field samples, or impulse fraction.
    """
    if _compiled_impulse_limits is None or phase.position_m.ndim != 2:
        return None
    parameters = _field_parameters(electric, magnetic)
    active = np.asarray(active)
    if parameters is None or active.dtype != np.bool_ or active.shape != (len(phase.position_m),):
        return None
    dt, impulse = float(dt), float(impulse)
    if not math.isfinite(dt) or dt <= 0. or not math.isfinite(impulse) or impulse <= 0.:
        return None
    indices = np.flatnonzero(active)
    kernel = (_parallel_impulse_limits if len(indices) >= 1024 and get_num_threads() > 1
              else _compiled_impulse_limits)
    limits, invalid = kernel(phase.position_m, phase.momentum_kg_m_per_s, indices,
                             dt, impulse, *parameters)
    if np.any(invalid):
        raise ValueError("Electric and magnetic field values must be finite.")
    return min(dt, float(np.min(limits, initial=math.inf)))


def try_analytic_step(gun, phase, dt, active, magnetic, electric, invariant_energy):
    """Return ``(phase, accepted_dt)`` or None for the NumPy reference path.

    ``invariant_energy`` is K - phi in eV, one value per active particle.
    All physical fields and accuracy settings are the reference settings.
    """
    if (_compiled_step is None or phase.position_m.ndim != 2
            or not getattr(gun, "compiled_particle_steps", True)):
        return None
    parameters = _field_parameters(electric, magnetic)
    if parameters is None:
        return None
    # Projection in the reference obtains its field from gun.electric_field.
    # Never substitute the stepping field for a different projection provider.
    projection = gun.electric_field
    if (not _original_provider(projection, FegElectrostaticField)
            or not getattr(projection, "compiled_particle_steps", True)
            or any(getattr(projection, name) is not getattr(electric, name)
                   for name in ("emitter", "extractor", "electrostatic_lens", "accelerator"))):
        return None
    dt = float(dt)
    if not math.isfinite(dt) or dt == 0.:
        raise ValueError("Time step must be finite and non-zero.")
    active = np.asarray(active)
    energy = np.asarray(invariant_energy, dtype=float)
    if active.dtype != np.bool_ or active.shape != (len(phase.position_m),):
        return None
    try:
        energy = np.broadcast_to(energy, (int(np.count_nonzero(active)),))
    except ValueError:
        return None
    if not np.all(np.isfinite(energy)):
        raise ValueError("Kinetic energy must be finite and non-negative.")
    step = (_compiled_parallel_step if np.count_nonzero(active) >= 1024
            and get_num_threads() > 1 else _compiled_step)
    x, p, accepted_dt = step(
        phase.position_m, phase.momentum_kg_m_per_s, dt, active,
        energy, *parameters,
    )
    return RelativisticPhaseSpace(x, p, phase.time_s+accepted_dt), accepted_dt
