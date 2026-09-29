"""Compiled evaluation of the existing discrete-gradient gun step.

Same solved potential, Lorentz equation, stopping tolerance and captured magnetic
sum as the Python reference. No fast-math, voltage scaling or beam matching.
Other field providers (including the Wien assembly) use the general reference.
"""
import math
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from threading import RLock
import numpy as np
from .relativistic_lorentz import (
    SPEED_OF_LIGHT_M_PER_S as c,
    ELEMENTARY_CHARGE_C as e,
    ELECTRON_MASS_KG as m_e,
)
from .axis_field_interpolation import compiled_evaluate, njit
from .compiled_magnetic_field import compiled_magnetic_batch, prepare_compiled_magnetic_sources
from .instrument_magnetic import InstrumentMagneticField
from .discrete_gradient import discrete_gradient_update, NORMALIZED_MOMENTUM_FLOOR


_PACKED_INSTRUMENT_FIELDS = OrderedDict()
_PACKED_INSTRUMENT_LOCK = RLock()
_INSTRUMENT_FIELD_METHOD = InstrumentMagneticField.field_at_global_positions_t
_STEP_POOL = ContextVar("gun_discrete_gradient_pool", default=None)


class _StepPool:
    """Parallel rows, with the original global iteration convergence barrier."""
    def __init__(self):
        self.executor = None
        self.workers = 1

    def step(self, x0, p0, dt, tolerance, iterations, *field_args):
        from temsim.cpu_resources import numerical_thread_budget
        workers = min(numerical_thread_budget(), len(x0)//2048)
        if workers < 2:
            return _step(x0, p0, dt, tolerance, iterations, *field_args)
        if self.executor is None:
            self.workers = workers
            self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gun-step")
        # Every child executes serial, nogil native code. It must not acquire
        # the parent's numerical-job lease or create another numerical pool.
        phi0, u0, gamma0, u1 = _prepare_step(x0, p0, dt, *field_args[:5])
        cuts = np.linspace(0, len(x0), min(workers, self.workers)+1, dtype=int)
        slices = [slice(int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:])]
        momenta = [u1[sl] for sl in slices]
        for _ in range(iterations):
            futures = [self.executor.submit(_step_iteration,
                x0[sl], u0[sl], previous, gamma0[sl], phi0[sl], dt, *field_args)
                for sl, previous in zip(slices, momenta)]
            # Resolve every child even on failure, before a retry can submit
            # another set or a cancelled calculation can release its budget.
            results, error = [], None
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as exc:
                    if error is None:
                        error = exc
            if error is not None:
                raise error
            if max(row[2] for row in results) <= tolerance:
                return (np.concatenate([row[0] for row in results]),
                        np.concatenate([row[1] for row in results])*(m_e*c))
            momenta = [row[1] for row in results]
        raise ValueError("Discrete-gradient Lorentz iteration did not converge")

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)


@contextmanager
def gun_step_workers():
    if _STEP_POOL.get() is not None:
        yield
        return
    from temsim.cpu_resources import numerical_job
    # GUI calculations already own this reentrant lease. Direct gun callers
    # need it too, so concurrent traces cannot multiply the process CPU cap.
    with numerical_job():
        pool = _StepPool()
        token = _STEP_POOL.set(pool)
        try:
            yield
        finally:
            try:
                pool.close()
            finally:
                _STEP_POOL.reset(token)


def _packed_instrument_field(provider):
    """Memoize immutable captures, never arrays derived from live hardware."""
    key = id(provider)
    with _PACKED_INSTRUMENT_LOCK:
        cached = _PACKED_INSTRUMENT_FIELDS.get(key)
        if cached is not None and cached[0] is provider:
            _PACKED_INSTRUMENT_FIELDS.move_to_end(key)
            return cached[1]
        packed = prepare_compiled_magnetic_sources(provider._sources)
        _PACKED_INSTRUMENT_FIELDS[key] = (provider, packed)
        while len(_PACKED_INSTRUMENT_FIELDS) > 4:
            _PACKED_INSTRUMENT_FIELDS.popitem(last=False)
        return packed


def _window(z, center, half, edge):
    edge = max(edge, 1e-6)
    left = min(1., max(0., (z-center+half+edge)/(2*edge)))
    right = min(1., max(0., (z-center-half+edge)/(2*edge)))
    return left**3*(10-15*left+6*left**2)-right**3*(10-15*right+6*right**2)


def _magnetic(p, parameters):
    result = np.zeros_like(p)
    for i in range(len(p)):
        z = p[i, 2]*1000
        for j in (0, 5):
            center, half, edge, bx, by = parameters[j:j+5]
            if bx != 0 or by != 0:
                envelope = _window(z, center, half, edge)
                result[i, 0] += bx*envelope
                result[i, 1] += by*envelope
        center, half, edge, gradient, angle = parameters[10:15]
        if gradient != 0:
            envelope = _window(z, center, half, edge)
            cosine, sine = math.cos(angle), math.sin(angle)
            u = cosine*p[i,0]+sine*p[i,1]
            v = -sine*p[i,0]+cosine*p[i,1]
            bu, bv = gradient*v*envelope, gradient*u*envelope
            result[i,0] += cosine*bu-sine*bv
            result[i,1] += sine*bu+cosine*bv
    return result


def _electric(p, data, ht, strict_domain=False, planar_cathode=False):
    query = p.copy()
    for i in range(len(p)):
        if not np.isfinite(p[i]).all():
            raise ValueError("Field positions must be finite xyz coordinates in metres")
        if math.hypot(p[i,0], p[i,1]) > data[0][-1] or (p[i,2] < data[1][0] and not planar_cathode):
            raise ValueError("Requested position is outside the solved gun field")
        if strict_domain and p[i,2] > data[1][-1]:
            raise ValueError("Requested position is outside the solved gun field")
        query[i,2] = min(p[i,2], data[1][-1])
        if planar_cathode:
            query[i,2] = max(query[i,2], 0.)
    potential, field = compiled_evaluate(query, *data)
    for i in range(len(p)):
        if not strict_domain and p[i,2] >= data[1][-1]:
            potential[i] = ht
            field[i,:] = 0.
        if planar_cathode and p[i,2] < 0:
            potential[i] = 0.
            field[i,:] = 0.
    return potential, field


def _prepare_step(x0, p0, dt, data, ht, magnetic, strict_domain=False, planar_cathode=False):
    phi0, e0 = _electric(x0, data, ht, strict_domain, planar_cathode)
    u0 = p0/(m_e*c)
    gamma0 = np.sqrt(1+np.sum(u0*u0, axis=1))
    u1 = u0-e*dt/(m_e*c)*e0
    return phi0, u0, gamma0, u1


def _step_iteration(x0, u0, u1, gamma0, phi0, dt, data, ht, magnetic, strict_domain=False,
                    planar_cathode=False, instrument_sources=None, instrument_terms=None):
    gamma1 = np.sqrt(1+np.sum(u1*u1, axis=1))
    vbar = c*(u1+u0)/(gamma1+gamma0).reshape((-1,1))
    dx = dt*vbar
    x1 = x0+dx
    midpoint = .5*(x0+x1)
    _, emid = _electric(midpoint, data, ht, strict_domain, planar_cathode)
    bmid = (_magnetic(midpoint, magnetic) if instrument_sources is None else
            compiled_magnetic_batch(midpoint, instrument_sources, instrument_terms))
    phi1, _ = _electric(x1, data, ht, strict_domain, planar_cathode)
    updated = np.empty_like(u0)
    error = 0.
    for i in range(len(x0)):
        ux, uy, uz = u0[i]
        dx0, dx1, dx2 = dx[i]
        ex, ey, ez = emid[i]
        bx, by, bz = bmid[i]
        vx, vy, vz = discrete_gradient_update(ux, uy, uz, gamma1[i]+gamma0[i],
            dx0, dx1, dx2, ex, ey, ez, bx, by, bz, phi0[i], phi1[i], dt)
        updated[i, 0], updated[i, 1], updated[i, 2] = vx, vy, vz
        if not np.isfinite(updated[i]).all():
            raise ValueError("Non-finite discrete-gradient iteration")
        scale = max(np.sqrt(np.sum(u0[i]**2)), np.sqrt(np.sum(updated[i]**2)), NORMALIZED_MOMENTUM_FLOOR)
        error = max(error, np.sqrt(np.sum((updated[i]-u1[i])**2))/scale)
    return x1, updated, error


def _step(x0, p0, dt, tolerance, iterations, data, ht, magnetic, strict_domain=False,
          planar_cathode=False, instrument_sources=None, instrument_terms=None):
    phi0, u0, gamma0, u1 = _prepare_step(x0, p0, dt, data, ht, magnetic, strict_domain, planar_cathode)
    for _ in range(iterations):
        x1, updated, error = _step_iteration(x0, u0, u1, gamma0, phi0, dt, data, ht, magnetic,
            strict_domain, planar_cathode, instrument_sources, instrument_terms)
        if error <= tolerance:
            return x1, updated*(m_e*c)
        u1 = updated
    raise ValueError("Discrete-gradient Lorentz iteration did not converge")


if njit is not None:
    _window = njit(cache=True)(_window)
    _magnetic = njit(cache=True)(_magnetic)
    _electric = njit(cache=True)(_electric)
    _prepare_step = njit(cache=True, nogil=True)(_prepare_step)
    _step_iteration = njit(cache=True, nogil=True)(_step_iteration)
    # A large emission bundle can spend most of the job in this kernel. Release
    # the interpreter so the GUI can paint progress and accept cancellation.
    _step = njit(cache=True, nogil=True)(_step)


def try_step(phase, dt, magnetic, electric, tolerance, iterations):
    from temsim.optics.electron_gun.alignment import FegMagneticField, GunDeflector, GunStigmator
    from .grounded_tip_field import GroundedTipField
    from .closed_gun_field import ClosedGunField
    from .continuous_gun_field import ContinuousGunField
    planar = type(electric) is ClosedGunField
    closed = type(electric) in (ClosedGunField, ContinuousGunField)
    # Exact providers only: never substitute for added/overridden physics.
    if (njit is None or phase.position_m.ndim != 2 or type(electric) not in (GroundedTipField, ClosedGunField, ContinuousGunField)
            or not getattr(electric, 'compiled_particle_steps', True)
            or (not planar and (not hasattr(electric, '_regular') or not electric._regular.compiled))):
        return None
    for provider, names in (
            (electric, ('potential_v_at_global_positions', 'potential_rise_v_at_global_positions',
                        'field_at_global_positions_v_per_m', '_interpolate', 'interpolate')),
            (magnetic, ('field_at_global_positions_t',))):
        if any(name in provider.__dict__ for name in names):
            return None
    instrument_sources = instrument_terms = None
    if type(magnetic) is InstrumentMagneticField:
        if getattr(magnetic.field_at_global_positions_t, '__func__', None) is not _INSTRUMENT_FIELD_METHOD:
            return None
        packed = _packed_instrument_field(magnetic)
        if packed is None:
            return None
        instrument_sources, instrument_terms = packed
        parameters = np.empty(0)
    elif (type(magnetic) is FegMagneticField and type(magnetic.deflector) is GunDeflector
            and type(magnetic.stigmator) is GunStigmator):
        d, s = magnetic.deflector, magnetic.stigmator
        if any('field_at_global_positions_t' in provider.__dict__ for provider in (d, s)):
            return None
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
    else:
        return None
    if planar:
        data = _closed_field_data(electric)
        high_tension = float(electric.request['high_tension_v'])
    else:
        f, regular = electric._fem, electric._regular
        data = (f.r, f.z, f.nodal_voltage, f.cut_cells, f.lookup, f.origin,
                f.inverse, f.gradient, f.phi0, regular.s, regular.slope)
        high_tension = (float(electric.request['high_tension_v']) if closed
                        else electric.high_tension_v)
    from .relativistic_lorentz import RelativisticPhaseSpace
    pool = _STEP_POOL.get()
    step = _step if pool is None else pool.step
    x, p = step(phase.position_m, phase.momentum_kg_m_per_s, dt,
                 tolerance, iterations, data, high_tension, parameters, closed, planar,
                 instrument_sources, instrument_terms)
    return RelativisticPhaseSpace(x, p, phase.time_s+dt)


def _closed_field_data(electric):
    """Use the same bilinear potential as the reference, without axis fitting."""
    data = getattr(electric, '_compiled_field_data', None)
    if data is None:
        r, z = electric.r, electric.z
        data = (r, z, electric.voltage,
                np.zeros((len(r)-1, len(z)-1), dtype=np.bool_),
                np.empty((0, 2), dtype=np.int64), np.empty((0, 2)),
                np.empty((0, 2, 2)), np.empty((0, 2)), np.empty(0),
                np.zeros(len(z)), np.zeros(len(z)))
        electric._compiled_field_data = data
    return data
