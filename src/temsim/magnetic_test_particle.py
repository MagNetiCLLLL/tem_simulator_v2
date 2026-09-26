"""An isolated relativistic test electron in captured electromagnetic fields.

This diagnostic never supplies a microscope source or transport checkpoint.
It solves dx/dt = p/(gamma*m), dp/dt = -e*(E + v cross B) in right-handed
global XYZ metres; +Z is downstream. E is V/m, B tesla and potential volts.
Scene-owned hardware intercepts stop the path; specimen and detector signal
interactions are excluded. No field solve or source substitution occurs here.

Static electric fields use a symmetric discrete-gradient Lorentz step, with
step-doubling error control. It conserves K-e*phi to iteration tolerance without
rescaling momentum. The pure-B specialization uses a symmetric Boris rotation.
Both permit turning and integrate flight time and travelled path separately.
See Higuera & Cary (2017), https://arxiv.org/abs/1701.05605 for relativistic
particle pushers and the pure-magnetic energy invariant. No field solve or
thread pool is started here; GUI callers own cancellation and CPU budgeting.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from math import cos, radians, sin, sqrt
from time import monotonic
from typing import Callable
import logging

import numpy as np

from temsim.physics.relativistic_lorentz import (
    ELEMENTARY_CHARGE_C, ELECTRON_MASS_KG, SPEED_OF_LIGHT_M_PER_S,
)


@dataclass(frozen=True)
class TestElectronSettings:
    kinetic_energy_ev: float = .3
    position_m: tuple[float, float, float] = (0., 0., 0.)
    polar_angle_deg: float = 0.
    azimuth_angle_deg: float = 0.
    max_path_length_m: float = .05
    step_m: float = 1e-4
    max_steps: int = 12_000
    relative_tolerance: float = 1e-4
    position_tolerance_m: float = 1e-12

    def __post_init__(self):
        for name in ("kinetic_energy_ev", "max_path_length_m", "step_m"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        point = tuple(float(v) for v in self.position_m)
        if len(point) != 3 or not np.isfinite(point).all():
            raise ValueError("position_m must contain three finite XYZ metres")
        object.__setattr__(self, "position_m", point)
        polar, azimuth = float(self.polar_angle_deg), float(self.azimuth_angle_deg)
        if not np.isfinite((polar, azimuth)).all() or not 0. <= polar <= 180.:
            raise ValueError("Polar angle must be 0 to 180 degrees from +Z and azimuth must be finite")
        object.__setattr__(self, "polar_angle_deg", polar)
        object.__setattr__(self, "azimuth_angle_deg", azimuth % 360.)
        if (isinstance(self.max_steps, bool) or not np.isfinite(self.max_steps)
                or int(self.max_steps) != self.max_steps or not 1 <= self.max_steps <= 200_000):
            raise ValueError("max_steps must be an integer from 1 to 200000")
        object.__setattr__(self, "max_steps", int(self.max_steps))
        if not np.isfinite(self.relative_tolerance) or not 1e-9 <= self.relative_tolerance <= .01:
            raise ValueError("relative_tolerance must lie from 1e-9 to 0.01")
        if not np.isfinite(self.position_tolerance_m) or self.position_tolerance_m <= 0.:
            raise ValueError("position_tolerance_m must be finite and positive")


@dataclass(frozen=True)
class TestElectronTrajectory:
    positions_m: np.ndarray
    directions: np.ndarray
    time_s: np.ndarray
    path_length_m: np.ndarray
    energy_invariant_error_ev: float
    energy_invariant_relative_error: float
    reason: str
    completed: bool
    steps: int
    kinetic_energy_ev: np.ndarray
    electrostatic_potential_v: np.ndarray
    speed_m_per_s: np.ndarray
    momentum_kg_m_per_s: np.ndarray
    notes: tuple[str, ...]


# These classes are inputs/results, not pytest test collections.
TestElectronSettings.__test__ = False
TestElectronTrajectory.__test__ = False


def _frozen(values):
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def electron_momentum_and_speed(kinetic_energy_ev):
    """Stable SI relativistic p and v for positive kinetic energy in eV."""
    energy = float(kinetic_energy_ev) * ELEMENTARY_CHARGE_C
    rest = ELECTRON_MASS_KG * SPEED_OF_LIGHT_M_PER_S**2
    if not np.isfinite(energy) or energy <= 0.:
        raise ValueError("Electron kinetic energy must be finite and positive")
    ratio = energy / rest
    momentum = ELECTRON_MASS_KG * SPEED_OF_LIGHT_M_PER_S * sqrt(ratio) * sqrt(ratio+2.)
    speed = SPEED_OF_LIGHT_M_PER_S * sqrt(ratio/(ratio+1.)) * sqrt((ratio+2.)/(ratio+1.))
    if not np.isfinite(momentum) or momentum <= 0.:
        raise ValueError("Electron kinetic energy is outside the numerical range")
    return momentum, speed


def _boris_direction(direction, magnetic_field, factor):
    # n' = n + n cross t; n_new = n + n' cross [2t/(1+t.t)].
    # Scalar arithmetic avoids allocating tiny cross-product helper arrays.
    tx, ty, tz = factor * magnetic_field
    x, y, z = direction
    ax, ay, az = x+y*tz-z*ty, y+z*tx-x*tz, z+x*ty-y*tx
    scale = 2./(1.+tx*tx+ty*ty+tz*tz)
    return np.array((x+scale*(ay*tz-az*ty), y+scale*(az*tx-ax*tz), z+scale*(ax*ty-ay*tx)))


class _Sampler:
    def __init__(self, scene, *, use_compiled=False):
        self.scene = scene
        bounds = getattr(scene, "diagnostic_bounds_m", None)
        self.bounds = np.asarray(scene.bounds_m if bounds is None else bounds, dtype=float)
        if self.bounds.shape != (2, 3) or not np.isfinite(self.bounds).all() or np.any(self.bounds[1] <= self.bounds[0]):
            raise ValueError("Magnetic diagnostic bounds must be a finite box with positive extent")
        self.scalar_query = getattr(scene, "diagnostic_field_at_global_position_t", None)
        self.scalar_valid = getattr(scene, "diagnostic_position_is_valid", None)
        self.joint_query = getattr(scene, "diagnostic_fields_at_global_position", None)
        self.segment_stop = getattr(scene, "diagnostic_segment_stop", None)
        self.regions = tuple(np.asarray(region.bounds_m, dtype=float) for region in getattr(scene, "source_regions", ()))
        self.edges = np.unique([v for box in self.regions for v in box[:, 2]])
        self.sampling_regions = tuple(getattr(scene, "diagnostic_sampling_regions", ()))
        self.local_spatial_step = getattr(scene, "diagnostic_spatial_step_m", None)
        # The field scene is frozen for one trace. Discrete-gradient iterations
        # and step doubling often revisit exactly the same float64 positions.
        # Reuse those values only; no coordinate rounding or interpolation.
        self._field_cache = OrderedDict()
        self.compiled_data = None
        if use_compiled:
            from temsim.test_electron_compiled import prepare_compiled_fields
            self.compiled_data = prepare_compiled_fields(scene)

    def compilation_fallback(self, error):
        """Unavailable compilation must retain every generic physical callback."""
        from numba.core.errors import NumbaError
        if not isinstance(error, NumbaError):
            return False
        self.compiled_data = None
        logging.getLogger(__name__).warning(
            "Virtual electron compiled backend unavailable; using the complete reference solver: %s", error)
        return True

    def fields(self, position):
        if self.compiled_data is not None:
            from temsim.test_electron_compiled import compiled_fields
            try:
                valid, magnetic, electric, potential = compiled_fields(position, self.compiled_data)
                return (magnetic, electric, potential) if valid else None
            except Exception as error:
                if not self.compilation_fallback(error):
                    raise
        key = np.asarray(position, dtype=float).tobytes()
        if key in self._field_cache:
            self._field_cache.move_to_end(key)
            return self._field_cache[key]
        if not callable(self.joint_query):
            magnetic = self.sample(position)
            values = None if magnetic is None else (magnetic, np.zeros(3), 0.)
        else:
            values = self.joint_query(position)
        if values is None:
            self._cache_fields(key, None)
            return None
        magnetic, electric, potential = values
        magnetic, electric = np.asarray(magnetic, float), np.asarray(electric, float)
        potential = float(potential)
        if (magnetic.shape != (3,) or electric.shape != (3,) or
                not np.isfinite(magnetic).all() or not np.isfinite(electric).all() or not np.isfinite(potential)):
            raise ValueError("Electromagnetic provider returned invalid SI fields or potential")
        result = _frozen(magnetic), _frozen(electric), potential
        self._cache_fields(key, result)
        return result

    def _cache_fields(self, key, values):
        self._field_cache[key] = values
        self._field_cache.move_to_end(key)
        while len(self._field_cache) > 64:
            self._field_cache.popitem(last=False)

    def intercept(self, start, end):
        if not callable(self.segment_stop):
            return None
        value = self.segment_stop(start, end)
        if value is None:
            return None
        fraction, reason = value
        if not np.isfinite(fraction) or not 0. <= fraction <= 1.:
            raise ValueError("Hardware intercept fraction must lie from zero to one")
        return float(fraction), str(reason)

    def is_valid(self, position):
        if callable(self.scalar_valid):
            return bool(self.scalar_valid(position))
        if np.any(position < self.bounds[0]) or np.any(position > self.bounds[1]):
            return False
        admitted = np.asarray(self.scene.contains(position[None, :]), dtype=bool)
        if admitted.shape != (1,):
            raise ValueError("Magnetic domain query must return one flag per point")
        return bool(admitted[0])

    def sample(self, position):
        if callable(self.scalar_query):
            values = self.scalar_query(position)
            if values is None:
                return None
            values = np.asarray(values, dtype=float)
        else:
            if not self.is_valid(position):
                return None
            values = np.asarray(self.scene.field_at_global_positions_t(position[None, :]), dtype=float)
            if values.shape != (1, 3):
                raise ValueError("Magnetic provider must return one three-component B vector per point")
            values = values[0]
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError("Magnetic provider returned invalid XYZ field values")
        return values

    def spatial_step(self, position, direction, requested):
        # Resolve finite field entrances/exits even when a user requests a
        # much larger step. Regions are model support, never hardware walls.
        z, nz = float(position[2]), float(direction[2])
        step = requested
        if abs(nz) > 1e-14:
            ahead = (self.edges-z)/nz
            ahead = ahead[ahead > max(1e-13, requested*1e-10)]
            if len(ahead):
                step = min(step, float(ahead.min()))
            for box in self.regions:
                if box[0, 2] <= z <= box[1, 2]:
                    step = min(step, (box[1, 2]-box[0, 2])/(96.*abs(nz)))
        transverse_direction = sqrt(float(direction[0]**2+direction[1]**2))
        for region in self.sampling_regions:
            if region.bounds_m[0, 2] <= z <= region.bounds_m[1, 2]:
                step = min(step, region.step_m)
                if transverse_direction > 1e-14:
                    step = min(step, region.transverse_step_m/transverse_direction)
        if callable(self.local_spatial_step):
            local = float(self.local_spatial_step(position, direction, step))
            if not np.isfinite(local) or local <= 0.:
                raise ValueError("Local field step must be finite and positive")
            step = min(step, local)
        return step

    def curved_support_step(self, position, direction, curvature, requested):
        """Bound Z motion toward every support face, including from vz=0.

        |d(n_z)/ds| <= |q B|/p. Therefore the displacement toward a face is
        bounded by v_axis*s + curvature*s**2/2. Limiting the trial step by
        this envelope prevents its curved endpoint from crossing a thin coil
        into an otherwise valid zero-field gap. A straight-ray-only face
        calculation cannot provide that guarantee near transverse emission.
        """
        if curvature <= 0. or not len(self.edges):
            return requested
        z, nz = float(position[2]), float(direction[2])
        offsets = self.edges-z
        # Ignore a face already reached to numerical precision, allowing a
        # step into the adjacent region. Midpoint sampling then chooses its
        # own field. This tolerance is far below native display-field scales.
        epsilon = max(2e-15, np.spacing(abs(z))*16.)
        step = requested
        for sign in (1., -1.):
            distances = sign*offsets
            distances = distances[distances > epsilon]
            if not len(distances):
                continue
            distance = float(distances.min())
            velocity = sign*nz
            discriminant = sqrt(velocity*velocity+2.*curvature*distance)
            limit = (2.*distance/(velocity+discriminant) if velocity >= 0.
                     else (discriminant-velocity)/curvature)
            step = min(step, limit)
        return step


def _trace_magnetic_electron(scene, settings: TestElectronSettings, *, cancelled: Callable[[], bool] | None = None,
                             progress=None, progress_interval_s=.08):
    """Trace only the test electron, without modifying its captured field.

    ``path_limit`` and ``domain_exit`` are completed diagnostic stops, not
    accepted microscope calculations. ``step_limit`` exposes truncation and
    ``cancelled`` exposes interruption. An unsupported initial position returns
    ``initial_outside_domain`` without inventing a field or moving the electron.
    ``path_length_m`` integrates v dt; it is not the saved polyline length.
    """
    if not isinstance(settings, TestElectronSettings):
        raise TypeError("settings must be TestElectronSettings")
    sampler = _Sampler(scene)
    momentum, speed = electron_momentum_and_speed(settings.kinetic_energy_ev)
    signed_q_over_p = -ELEMENTARY_CHARGE_C/momentum
    theta, phi = radians(settings.polar_angle_deg), radians(settings.azimuth_angle_deg)
    direction = np.array((sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)))
    position = np.array(settings.position_m)
    positions, directions, distances = [position.copy()], [direction.copy()], [0.]
    reason = "step_limit"
    cancelled = cancelled or (lambda: False)
    # The gyrophase limit complements spatial sampling. It is not an a priori
    # trajectory error guarantee; the exposed step permits convergence checks.
    max_rotation_rad = .05
    min_step = max(1e-14, settings.step_m*1e-10)
    path_tolerance = max(np.finfo(float).tiny, settings.max_path_length_m*2e-14)
    next_progress = monotonic()+progress_interval_s
    if cancelled():
        reason = "cancelled"
    elif sampler.sample(position) is None:
        reason = "initial_outside_domain"
    else:
        for _ in range(settings.max_steps):
            if cancelled():
                reason = "cancelled"
                break
            remaining = settings.max_path_length_m-distances[-1]
            if remaining <= path_tolerance:
                reason = "path_limit"
                break
            ds = sampler.spatial_step(position, direction, min(settings.step_m, remaining))
            accepted = False
            for _attempt in range(48):
                if cancelled():
                    reason = "cancelled"
                    break
                midpoint = position+.5*ds*direction
                field = sampler.sample(midpoint)
                if field is None:
                    ds *= .5
                else:
                    curvature = abs(signed_q_over_p)*sqrt(float(field@field))
                    curved_step = sampler.curved_support_step(position, direction, curvature, ds)
                    if curved_step < ds*(1.-1e-12):
                        ds = curved_step
                        continue
                    omega_ds = curvature*ds
                    if omega_ds > max_rotation_rad*(1.+1e-12):
                        ds *= max_rotation_rad/omega_ds
                        continue
                    next_direction = _boris_direction(direction, field, .5*signed_q_over_p*ds)
                    next_position = midpoint+.5*ds*next_direction
                    if not sampler.is_valid(next_position):
                        ds *= .5
                    else:
                        accepted = True
                        break
                if ds < min_step:
                    reason = "domain_exit"
                    break
            if not accepted:
                if reason not in {"cancelled", "domain_exit"}:
                    reason = "domain_exit"
                break
            position, direction = next_position, next_direction
            positions.append(position.copy())
            directions.append(direction.copy())
            distances.append(distances[-1]+ds)
            if progress is not None and monotonic() >= next_progress:
                progress(_magnetic_result(settings, momentum, speed, positions, directions, distances, "in_progress"))
                next_progress = monotonic()+progress_interval_s
        if reason == "step_limit" and settings.max_path_length_m-distances[-1] <= path_tolerance:
            reason = "path_limit"
    return _magnetic_result(settings, momentum, speed, positions, directions, distances, reason)


def _magnetic_result(settings, momentum, speed, positions, directions, distances, reason):
    direction_values = np.asarray(directions)
    # Use p^2/(sqrt(m^2 c^4+p^2 c^2)+mc^2), avoiding low-energy cancellation.
    rest = ELECTRON_MASS_KG*SPEED_OF_LIGHT_M_PER_S**2
    p_squared_c_squared = np.sum(direction_values**2, axis=1)*(momentum*SPEED_OF_LIGHT_M_PER_S)**2
    energy = p_squared_c_squared/(np.sqrt(rest*rest+p_squared_c_squared)+rest)
    energy_ev = energy/ELEMENTARY_CHARGE_C
    error_ev = float(np.max(np.abs(energy_ev-energy_ev[0])))
    error = error_ev/settings.kinetic_energy_ev
    notes = ("Magnetic-only virtual test electron: no electric acceleration, material, apertures or detector interactions.",
             "Field and any equivalent deflector calibration are frozen at the captured microscope settings.",
             "Leaving field validity is a model-domain stop, not a physical collision.",
             "Second-order pure-B Boris; requested step, native cell spacing, field support and 0.05 rad gyro limit; check step convergence.")
    return TestElectronTrajectory(_frozen(positions), _frozen(directions), _frozen(np.asarray(distances)/speed),
                                  _frozen(distances), error_ev, error, reason, reason in {"path_limit", "domain_exit"},
                                  len(distances)-1, _frozen(energy_ev), _frozen(np.zeros(len(distances))),
                                  _frozen(np.full(len(distances), speed)), _frozen(direction_values*momentum), notes)


class _TrialOutside(Exception):
    def __init__(self, stop_reason=None):
        self.stop_reason = stop_reason


class _TrialConvergence(Exception):
    pass


class _TraceCancelled(Exception):
    pass


def _velocity_u(u):
    return SPEED_OF_LIGHT_M_PER_S*u/sqrt(1.+float(u@u))


def _cross3(first, second):
    x, y, z = first
    a, b, c = second
    return np.array((y*c-z*b, z*a-x*c, x*b-y*a))


def _collinear_force(velocity, electric, magnetic):
    """Exact axial/parallel motion has no magnetic rotation to resolve."""
    norm = float(np.linalg.norm(velocity))
    axis = velocity/norm if norm > 0. else electric
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm == 0.:
        return True
    for field in (electric, magnetic):
        field_norm = float(np.linalg.norm(field))
        if np.linalg.norm(_cross3(axis, field)) > 1e-14*axis_norm*field_norm:
            return False
    return True


def _discrete_gradient_step(sampler, x0, u0, fields0, dt, cancelled):
    """Single-electron form of physics.static_energy_lorentz.static_energy_step.

    The same Gonzalez discrete gradient enforces Ebar.dx = -Delta(phi).
    The magnetic part of its fixed-point update is solved as a Cayley rotation
    instead of iterating the linear cross product. Momentum is never rescaled.
    u=p/(mc) avoids operations on subnormal squared SI momenta.
    """
    if sampler.compiled_data is not None:
        from temsim.test_electron_compiled import compiled_step
        if cancelled():
            raise _TraceCancelled
        try:
            status, x1, u1, magnetic, electric, potential, path = compiled_step(
                sampler.compiled_data, x0, u0, fields0[1], fields0[2], dt)
        except Exception as error:
            if not sampler.compilation_fallback(error):
                raise
        else:
            if status == 1:
                hit = sampler.intercept(x0, x1)
                raise _TrialOutside(hit[1] if hit is not None and hit[1].startswith("unsupported_field:") else None)
            if status == 2:
                raise _TrialConvergence
            return x1, u1, (magnetic, electric, potential), path
    c = SPEED_OF_LIGHT_M_PER_S
    alpha = -ELEMENTARY_CHARGE_C*dt/(ELECTRON_MASS_KG*c)
    phi0 = fields0[2]
    gamma0 = sqrt(1.+float(u0@u0))
    # An electric predictor permits rapid acceleration from sub-eV emission.
    u1 = u0+alpha*fields0[1]
    for _ in range(32):
        if cancelled():
            raise _TraceCancelled
        gamma1 = sqrt(1.+float(u1@u1))
        vbar = c*(u1+u0)/(gamma1+gamma0)
        dx = dt*vbar
        x1 = x0+dx
        midpoint = sampler.fields(.5*(x0+x1))
        endpoint = sampler.fields(x1)
        if midpoint is None or endpoint is None:
            hit = sampler.intercept(x0, x1)
            stop_reason = hit[1] if hit is not None and hit[1].startswith("unsupported_field:") else None
            raise _TrialOutside(stop_reason)
        if (not _collinear_force(vbar, midpoint[1], midpoint[0])
                and ELEMENTARY_CHARGE_C*dt*float(np.linalg.norm(midpoint[0]))
                /(ELECTRON_MASS_KG*.5*(gamma1+gamma0)) > .05*(1.+1e-12)):
            # Initial alignment does not justify a large rotation after the
            # particle enters a differently directed field within the trial.
            raise _TrialConvergence
        electric = midpoint[1]
        length2 = float(dx@dx)
        potential_difference = endpoint[2]-phi0
        midpoint_work = float(electric@dx)
        potential_resolution = 64.*np.finfo(float).eps*max(abs(endpoint[2]), abs(phi0), 1.)
        if length2 > 0. and abs(potential_difference)+abs(midpoint_work) > potential_resolution:
            defect = potential_difference+midpoint_work
            electric = electric-(defect/length2)*dx
        # At a turning point, a displacement can round back to x0 and its
        # potential difference becomes unresolvable. The discrete gradient's
        # limiting value is E(midpoint), not zero. Dividing that roundoff by
        # dx**2 would cancel the force and artificially strand the electron.
        t = alpha*c*midpoint[0]/(gamma1+gamma0)
        a = 2.*u0+alpha*electric
        # w - w cross t = a, w=u1+u0. This inverse preserves the same
        # discrete-gradient equation and is exact for its linear B term.
        updated = (a+_cross3(a, t)+t*float(a@t))/(1.+float(t@t))-u0
        if not np.isfinite(updated).all():
            raise _TrialConvergence
        scale = max(float(np.linalg.norm(u0)), float(np.linalg.norm(updated)), 1e-9)
        if float(np.linalg.norm(updated-u1)) <= 2e-11*scale:
            speed0 = float(np.linalg.norm(_velocity_u(u0)))
            speed1 = float(np.linalg.norm(_velocity_u(updated)))
            speed_mid = float(np.linalg.norm(_velocity_u(.5*(u0+updated))))
            path = dt*(speed0+4.*speed_mid+speed1)/6.
            return x1, updated, endpoint, path
        u1 = updated
    raise _TrialConvergence


def _electromagnetic_result(settings, positions, momenta_u, times, distances, potentials, reason):
    u = np.asarray(momenta_u)
    magnitude = np.linalg.norm(u, axis=1)
    gamma = np.sqrt(1.+magnitude*magnitude)
    rest_ev = ELECTRON_MASS_KG*SPEED_OF_LIGHT_M_PER_S**2/ELEMENTARY_CHARGE_C
    energy = rest_ev*magnitude*magnitude/(gamma+1.)
    potential = np.asarray(potentials)
    # Stable gauge: compare changes rather than subtracting two large total
    # energies. For an electron, energy in eV has invariant K_eV - phi_V.
    known_potential = bool(np.isfinite(potential).all())
    defect = (energy-energy[0])-(potential-potential[0])
    error_ev = float(np.max(np.abs(defect))) if known_potential else np.nan
    reference_ev = max(float(np.max(energy)), settings.kinetic_energy_ev)
    if known_potential:
        reference_ev = max(reference_ev, float(np.max(np.abs(potential-potential[0]))))
    directions = np.divide(u, magnitude[:, None], out=np.zeros_like(u), where=magnitude[:, None] > 0.)
    failure_reasons = {"initial_outside_domain", "step_limit", "cancelled", "numerical_limit", "in_progress"}
    notes = ("Virtual test electron in captured electric and magnetic fields; not a microscope source or continuation checkpoint.",
             "Extraction, electrostatic focusing, acceleration, magnetic optics and scene-owned hardware stops are included; specimen and detector signals are excluded.",
             "Electron static-field invariant: kinetic energy in eV minus electric potential in V; no forced exit energy or momentum rescaling.",
             "Second-order discrete-gradient Lorentz integration with step doubling, local field resolution and support-boundary limits.",
             "Time and travelled path are integrated with changing speed; zero momentum has no defined direction.")
    return TestElectronTrajectory(_frozen(positions), _frozen(directions), _frozen(times), _frozen(distances),
                                  error_ev, error_ev/reference_ev, reason,
                                  reason not in failure_reasons and not reason.startswith("unsupported_field:"),
                                  len(times)-1, _frozen(energy), _frozen(potential),
                                  _frozen(SPEED_OF_LIGHT_M_PER_S*magnitude/gamma),
                                  _frozen(u*(ELECTRON_MASS_KG*SPEED_OF_LIGHT_M_PER_S)), notes)


def _trace_electromagnetic_electron(scene, settings, cancelled, progress=None, progress_interval_s=.08, use_compiled=True):
    sampler = _Sampler(scene, use_compiled=use_compiled)
    if sampler.compiled_data is not None:
        from temsim.test_electron_intercepts import prepare_compiled_intercepts
        prepared_intercepts = prepare_compiled_intercepts(scene)
        if prepared_intercepts is not None:
            try:
                return _trace_compiled_electromagnetic_electron(
                    sampler, settings, cancelled, progress, progress_interval_s, prepared_intercepts)
            except Exception as error:
                if not sampler.compilation_fallback(error):
                    raise
    p0, _ = electron_momentum_and_speed(settings.kinetic_energy_ev)
    theta, phi = radians(settings.polar_angle_deg), radians(settings.azimuth_angle_deg)
    u = p0/(ELECTRON_MASS_KG*SPEED_OF_LIGHT_M_PER_S)*np.array((sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)))
    x = np.asarray(settings.position_m)
    positions, momenta_u, times, distances = [x.copy()], [u.copy()], [0.], [0.]
    fields = None if cancelled() else sampler.fields(x)
    potentials = [np.nan if fields is None else fields[2]]
    if cancelled() or fields is None:
        reason = "cancelled" if cancelled() else "initial_outside_domain"
        return _electromagnetic_result(settings, positions, momenta_u, times, distances, potentials, reason)
    u_floor = max(float(np.linalg.norm(u))*.1, 1e-9)
    suggested_dt = np.inf
    reason = "step_limit"
    path_tolerance = max(settings.max_path_length_m*2e-12, settings.position_tolerance_m*.01)
    next_progress = monotonic()+progress_interval_s
    for _ in range(settings.max_steps):
        if cancelled():
            reason = "cancelled"
            break
        remaining = settings.max_path_length_m-distances[-1]
        if remaining <= path_tolerance:
            reason = "path_limit"
            break
        velocity = _velocity_u(u)
        speed = float(np.linalg.norm(velocity))
        direction = velocity/speed if speed > 0. else np.zeros(3)
        requested = sampler.spatial_step(x, direction, min(settings.step_m, remaining))
        electric_norm, magnetic_norm = float(np.linalg.norm(fields[1])), float(np.linalg.norm(fields[0]))
        collinear = _collinear_force(velocity, fields[1], fields[0])
        rotation_field = 0. if collinear else magnetic_norm
        acceleration_bound = ELEMENTARY_CHARGE_C/ELECTRON_MASS_KG*(electric_norm+SPEED_OF_LIGHT_M_PER_S*rotation_field)
        if acceleration_bound > 0.:
            dt = 2.*requested/(speed+sqrt(speed*speed+2.*acceleration_bound*requested))
        elif speed > 0.:
            dt = requested/speed
        else:
            reason = "numerical_limit"
            break
        if electric_norm > 0.:
            impulse_time = (.05*max(float(np.linalg.norm(u)), u_floor)
                            *ELECTRON_MASS_KG*SPEED_OF_LIGHT_M_PER_S/(ELEMENTARY_CHARGE_C*electric_norm))
            dt = min(dt, impulse_time)
        if rotation_field > 0.:
            dt = min(dt, .05*sqrt(1.+float(u@u))*ELECTRON_MASS_KG/(ELEMENTARY_CHARGE_C*magnetic_norm))
        dt = min(dt, suggested_dt)
        dt = sampler.curved_support_step(x, velocity, acceleration_bound, dt)
        accepted = False
        pending_stop = None
        last_failure = "numerical_limit"
        for _attempt in range(64):
            pending_stop = None
            if cancelled():
                reason = "cancelled"
                break
            if dt <= max(np.spacing(times[-1])*8., 1e-30):
                reason = last_failure
                break
            try:
                full = _discrete_gradient_step(sampler, x, u, fields, dt, cancelled)
                first = _discrete_gradient_step(sampler, x, u, fields, .5*dt, cancelled)
                second = _discrete_gradient_step(sampler, first[0], first[1], first[2], .5*dt, cancelled)
            except _TraceCancelled:
                reason = "cancelled"
                break
            except _TrialOutside as outside:
                last_failure = outside.stop_reason or "domain_exit"
                dt *= .5
                continue
            except _TrialConvergence:
                last_failure = "numerical_limit"
                dt *= .5
                continue
            path = first[3]+second[3]
            position_scale = settings.position_tolerance_m+settings.relative_tolerance*max(path, full[3])
            momentum_scale = settings.relative_tolerance*max(float(np.linalg.norm(u)), float(np.linalg.norm(second[1])), u_floor*.01)
            error = max(float(np.linalg.norm(second[0]-full[0]))/position_scale,
                        float(np.linalg.norm(second[1]-full[1]))/momentum_scale,
                        abs(path-full[3])/position_scale)
            if error > 1.:
                dt *= max(.1, .8*error**(-1./3.))
                continue
            if path > remaining+path_tolerance:
                dt *= max(.1, remaining/path)
                continue
            # Inspect both integrated halves, including trajectories that turn
            # around. Hardware contacts are time ordered, never sorted by Z.
            hit_first = sampler.intercept(x, first[0])
            hit_second = None if hit_first is not None else sampler.intercept(first[0], second[0])
            if hit_first is not None or hit_second is not None:
                hit = hit_first if hit_first is not None else hit_second
                fraction = .5*hit[0] if hit_first is not None else .5+.5*hit[0]
                pending_stop = hit[1]
                if fraction <= 1e-12:
                    reason = pending_stop
                    break
                if (1.-fraction)*path > settings.position_tolerance_m:
                    dt *= fraction
                    continue
            accepted = True
            # An almost exact, geometry-limited step does not establish a tiny
            # physical time scale for the next cell. Recompute its local field
            # and impulse bounds instead of spending dozens of doubling steps
            # recovering from a face reached to floating-point precision.
            suggested_dt = (np.inf if error < 1e-6 else
                            dt*min(2., max(.5, .9*error**(-1./3.))))
            break
        if not accepted:
            if reason == "step_limit":
                reason = last_failure
            break
        x, u, fields = second[:3]
        positions.append(x.copy())
        momenta_u.append(u.copy())
        times.append(times[-1]+dt)
        distances.append(distances[-1]+first[3]+second[3])
        potentials.append(fields[2])
        if progress is not None and monotonic() >= next_progress:
            progress(_electromagnetic_result(settings, positions, momenta_u, times, distances, potentials, "in_progress"))
            next_progress = monotonic()+progress_interval_s
        if pending_stop is not None:
            reason = pending_stop
            break
    if reason == "step_limit" and settings.max_path_length_m-distances[-1] <= path_tolerance:
        reason = "path_limit"
    return _electromagnetic_result(settings, positions, momenta_u, times, distances, potentials, reason)


def _trace_compiled_electromagnetic_electron(sampler, settings, cancelled, progress, interval, prepared_intercepts):
    """Same accepted adaptive integration, returning to Python every 64 steps.

    Blocks bound cancellation latency and publish actual prefixes. They neither
    replace the physical source nor relax a requested numerical tolerance.
    """
    from temsim.test_electron_compiled import compiled_block
    data, reasons = prepared_intercepts
    unsupported = np.array([reason.startswith("unsupported_field:") for reason in reasons], dtype=np.bool_)
    sampling = (np.asarray(sampler.regions, dtype=float).reshape(-1, 2, 3), sampler.edges,
                np.asarray([region.bounds_m for region in sampler.sampling_regions], dtype=float).reshape(-1, 2, 3),
                np.asarray([(region.step_m, region.transverse_step_m) for region in sampler.sampling_regions], dtype=float).reshape(-1, 2))
    p0, _ = electron_momentum_and_speed(settings.kinetic_energy_ev)
    theta, phi = radians(settings.polar_angle_deg), radians(settings.azimuth_angle_deg)
    u = p0/(ELECTRON_MASS_KG*SPEED_OF_LIGHT_M_PER_S)*np.array((sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)))
    x = np.asarray(settings.position_m)
    fields = None if cancelled() else sampler.fields(x)
    rows = np.empty((settings.max_steps+1, 9))
    rows[0, :3], rows[0, 3:6] = x, u
    rows[0, 6:8], rows[0, 8] = 0., np.nan if fields is None else fields[2]
    count, suggested_dt = 1, np.inf
    reason = "cancelled" if cancelled() else "initial_outside_domain" if fields is None else "step_limit"
    u_floor = max(float(np.linalg.norm(u))*.1, 1e-9)
    next_progress = monotonic()+interval

    def snapshot(reason):
        values = rows[:count]
        return _electromagnetic_result(settings, values[:, :3], values[:, 3:6], values[:, 6],
                                       values[:, 7], values[:, 8], reason)

    while fields is not None and count <= settings.max_steps:
        if cancelled():
            reason = "cancelled"
            break
        previous = rows[count-1]
        block, magnetic, electric, potential, suggested_dt, status = compiled_block(
            sampler.compiled_data, sampling, data, unsupported, previous[:3].copy(), previous[3:6].copy(),
            fields[0], fields[1], fields[2], previous[6], previous[7], suggested_dt,
            settings.max_path_length_m, settings.step_m, settings.relative_tolerance,
            settings.position_tolerance_m, u_floor, min(64, settings.max_steps-count+1))
        rows[count:count+len(block)] = block
        count += len(block)
        fields = magnetic, electric, potential
        if cancelled():
            reason = "cancelled"
            break
        if progress is not None and len(block) and monotonic() >= next_progress:
            progress(snapshot("in_progress"))
            next_progress = monotonic()+interval
        if status:
            reason = ({1: "path_limit", 4: "domain_exit", 5: "numerical_limit"}.get(status)
                      if status < 100 else reasons[status-100])
            break
    return snapshot(reason)


def trace_test_electron(scene, settings: TestElectronSettings, *, cancelled: Callable[[], bool] | None = None,
                        progress: Callable[[TestElectronTrajectory], None] | None = None,
                        progress_interval_s: float = .08, use_compiled: bool = True):
    """Trace one diagnostic electron through the complete captured E+B scene.

    Pure magnetostatic fixtures use the faster energy-preserving Boris special
    case. A scene that declares electric fields must supply their physical
    potential and field together; the solver never invents acceleration from
    a target energy. Hardware-stop callbacks remain separate from field validity.

    Supported immutable electric grids and analytic magnetic components run
    the same algorithm in compiled bounded blocks. ``use_compiled=False``
    selects the general reference implementation for parity/convergence checks;
    unsupported scene types retain that implementation automatically. A
    ``progress`` callback receives read-only accepted prefixes marked
    ``in_progress``; it never receives interpolated or predicted future states.
    """
    if not isinstance(settings, TestElectronSettings):
        raise TypeError("settings must be TestElectronSettings")
    if not np.isfinite(progress_interval_s) or progress_interval_s < 0.:
        raise ValueError("Progress interval must be finite and nonnegative")
    if bool(getattr(scene, "has_electric_field", False)):
        if not callable(getattr(scene, "diagnostic_fields_at_global_position", None)):
            raise TypeError("Electric scene must expose captured E, B and potential")
        return _trace_electromagnetic_electron(scene, settings, cancelled or (lambda: False), progress, progress_interval_s, use_compiled)
    return _trace_magnetic_electron(scene, settings, cancelled=cancelled,
                                    progress=progress, progress_interval_s=progress_interval_s)
