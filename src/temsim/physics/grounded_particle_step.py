"""Compiled evaluation of the existing discrete-gradient gun step.

Same solved potential, Lorentz equation, stopping tolerance and captured magnetic
sum as the Python reference. No fast-math, voltage scaling or beam matching.
Other field providers (including the Wien assembly) use the general reference.
"""
import math
from collections import OrderedDict
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


def _step(x0, p0, dt, tolerance, iterations, data, ht, magnetic, strict_domain=False,
          planar_cathode=False, instrument_sources=None, instrument_terms=None):
    phi0, e0 = _electric(x0, data, ht, strict_domain, planar_cathode)
    u0 = p0/(m_e*c)
    gamma0 = np.sqrt(1+np.sum(u0*u0, axis=1))
    u1 = u0-e*dt/(m_e*c)*e0
    for _ in range(iterations):
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
        if error <= tolerance:
            return x1, updated*(m_e*c)
        u1 = updated
    raise ValueError("Discrete-gradient Lorentz iteration did not converge")


if njit is not None:
    _window = njit(cache=True)(_window)
    _magnetic = njit(cache=True)(_magnetic)
    _electric = njit(cache=True)(_electric)
    _step = njit(cache=True)(_step)


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
    x, p = _step(phase.position_m, phase.momentum_kg_m_per_s, dt,
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
