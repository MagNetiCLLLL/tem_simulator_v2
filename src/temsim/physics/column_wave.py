"""Distributed stationary scalar column operator acting on an executed beam.

Internal stage function, not a configurable source. Public tip-to-detector
requests execute the gun first. Every quadratic field, nonlinear multipole,
physical kick, aperture and sampled vacuum bore is owned by the shared plan.
"""
from dataclasses import asdict, dataclass, fields as dataclass_fields, is_dataclass, replace
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from contextvars import copy_context
from functools import cached_property
import math
from threading import Event, Lock

import numpy as np
from scipy.linalg import expm

from temsim.cpu_resources import initialize_numerical_thread, numerical_thread_budget
from temsim.physics.canonical_action import CanonicalPath
from temsim.physics.core import build_propagation_plan, electron
from temsim.physics.column_wall import _vacuum_segments, _expanded_profile_axis, _partition_vacuum_segments
from temsim.physics.multiplane_wave import propagate_plane_wave, reselect_phase_carrier
from temsim.physics.multipole_wave import apply_multipole_phase
from temsim.physics.tip_gun_wave import TipGunCheckpoint
from temsim.physics.wave_flux import BeamState, WaveMode
from temsim.physics.wave_grid import WaveGridNumerics, WaveMemoryBudgetError, apply_resolved_operator
from temsim.physics.wave_device import array_module, device_scope, normalise_backend, to_host
from temsim.physics.posed_wave_hardware import PosedWaveBore
from temsim.physics.posed_wave_aperture import PosedWaveAperture


@dataclass
class _PosedColumnPath(CanonicalPath):
    """A canonical path with its executed magnetic approximation contract."""

    correction: object
    support_majorant: np.ndarray
    generator: np.ndarray
    force: np.ndarray
    distance_m: float


@dataclass
class _ColumnPath(CanonicalPath):
    """Keep the step generator for conservative oblique-bore clearance."""

    generator: np.ndarray
    force: np.ndarray
    distance_m: float

    @cached_property
    def support_majorant(self):
        extended = np.zeros((5, 5))
        extended[:4, :4] = abs(self.generator)
        extended[:4, 4] = abs(self.force)
        return expm(extended*self.distance_m)


def _phase_space_box(shape, basis, origin, curvature, tilt, wavelength, *, center=None):
    """Entire lattice and full residual Nyquist support, without RMS fitting."""
    half_shape = np.array((shape[1]//2, shape[0]//2))
    local_extent = abs(basis)@half_shape
    center = np.zeros(4) if center is None else np.asarray(center)
    position = abs(origin-center[:2])+local_extent
    momentum = abs(tilt-center[2:])+abs(curvature)@local_extent
    momentum += .5*wavelength*np.sum(abs(np.linalg.inv(basis).T), axis=1)
    return np.r_[position, momentum]


def _posed_path(generator, force, correction, distance_m):
    # Bound deviations from the fixed expansion centre through the WHOLE
    # step. Its chief motion is included in the affine forcing term.
    majorant = np.zeros((5, 5))
    majorant[:4, :4] = abs(generator)
    majorant[:4, 4] = abs(force+generator@correction.center_phase_space)
    path = _PosedColumnPath(1e-3, correction, expm(majorant*distance_m),
                            generator, force, distance_m)
    _linear_factor(path, generator, distance_m, force=force,
                   constant_h=correction.constant_h)
    return path


def _recenter_posed_path(path, origin, tilt):
    """Expand about this mode's carrier centre without moving physical fields."""
    if not isinstance(path, _PosedColumnPath):
        return path
    correction = path.correction.reexpanded(np.r_[origin, tilt])
    if np.array_equal(correction.center_phase_space, path.correction.center_phase_space):
        return path
    return _posed_path(path.generator-path.correction.generator+correction.generator,
        path.force-path.correction.force+correction.force, correction, path.distance_m)


def _check_posed_step(path, box, distance_m, wavelength, phase_budget_per_m, *, allow_residual=False):
    """Bound the discarded magnetic action and forward near-axis domain.

    The positive-system majorant encloses the complete constant-Hamiltonian
    step, including an interior extremum. It does not establish axial-step
    convergence or accuracy of the original analytical lens model.
    """
    if not isinstance(path, _PosedColumnPath):
        return None
    bound = (path.support_majorant@np.r_[box, 1.])[:4]
    error = path.correction.phase_error_bound(bound[:2], bound[2:], distance_m, wavelength)
    allowed = phase_budget_per_m*distance_m
    electric_error = path.correction.unresolved_electric_phase_error_bound(
        bound[:2], bound[2:], distance_m, wavelength)
    if not math.isfinite(electric_error) or electric_error > allowed:
        raise ValueError("Tilted-lens electric-magnetic remainder exceeds the phase budget "
                         f"at Z {path.correction.z_m*1e3:.9g} mm: bound {electric_error:.6g} rad, "
                         f"allowed {allowed:.6g} rad. The unsupported operator was not applied.")
    support = path.correction.support_bounds(bound[:2], bound[2:])
    local_max = max((item["transverse_over_longitudinal_bound"] for item in support["local"]), default=0.)
    global_max = support["global_transverse_over_longitudinal_bound"]
    # An explicit paraxial domain, not an assertion of a corresponding total
    # phase accuracy. At slope .1 the kinetic expansion omits about .25% of
    # its transverse kinetic contribution; the magnetic Taylor bound is separate.
    if max(global_max, local_max) > .1 or any(
            item["forward_direction_lower_bound"] <= 0 for item in support["local"]):
        raise ValueError("Tilted-lens wave support exceeds the forward paraxial domain "
                         f"(global slope bound {global_max:.6g}, local {local_max:.6g}; limit 0.1). "
                         "The unsupported operator was not applied.")
    residual = error > allowed
    can_correct = allow_residual and path.correction.supports_magnetic_residual
    if not math.isfinite(error) or (residual and not can_correct):
        raise ValueError("Tilted-lens magnetic Taylor remainder exceeds the phase budget "
                         f"at Z {path.correction.z_m*1e3:.9g} mm: bound {error:.6g} rad, "
                         f"allowed {allowed:.6g} rad. A higher-order spatial field operator is required.")
    return {"phase_error_bound_rad": electric_error if residual else error, "distance_m": distance_m,
            "electric_magnetic_remainder_bound_rad": electric_error,
            "quadratic_phase_remainder_rad": error, "magnetic_residual_required": residual,
            "global_slope_bound": global_max, "local_slope_bound": local_max,
            "position_bound_m": abs(path.correction.center_phase_space[:2])+bound[:2]}


def _component_events(state, start, stop, *, arrival_time=None):
    events, owners, seen = [], [], set()
    from temsim.physics.instrument_magnetic import column_dipole_fields
    for component in (*state.deflectors, *getattr(state, "stigmators", ()), *getattr(state, "corrector_elements", ())):
        if not getattr(component, "enabled", False):
            continue
        if component.key in seen:
            raise ValueError(f"Duplicate column component {component.key}")
        seen.add(component.key)
        if (getattr(component, "scan_enabled", False) or getattr(component, "wobble_enabled", False)) and hasattr(component, "validate"):
            component.validate()
    for coil in column_dipole_fields(state):
        if not (coil.field_support_mm[0] < stop and coil.field_support_mm[1] > start):
            continue
        z, x, y = coil.event_z_mm, coil.event_dx_rad, coil.event_dy_rad
        time = None
        if coil.dynamic and arrival_time is not None:
            time = float(arrival_time(coil.arrival_z_mm))
            actual = next((item for item in column_dipole_fields(state, time_s=time)
                           if item.key == coil.key), None)
            if actual is None or (actual.lower_m, actual.upper_m, actual.event_z_mm) != (coil.lower_m, coil.upper_m, z):
                raise ValueError("Dynamic coil geometry changed with time")
            x, y = actual.event_dx_rad, actual.event_dy_rad
        events.append((z, x, y))
        owners.append({"component": coil.key.rsplit(":", 1)[0], "drive_keys": coil.drive_keys,
                       "z_mm": z, "kick_rad": (x, y), "dynamic": coil.dynamic,
                       "arrival_z_mm": coil.arrival_z_mm,
                       "arrival_time_s": time, "finite_field": True,
                       "effective_thickness_mm": (coil.upper_m-coil.lower_m)*1e3})
    return events, owners


@dataclass(frozen=True)
class _ParallelBore:
    """A translated physical bore; its circular wall is a strict boundary."""

    key: str
    start_z_mm: float
    end_z_mm: float
    radius_mm: float
    offset_x_mm: float
    offset_y_mm: float

    def clears_grid(self, shape, basis_m, origin_m):
        origin = origin_m - np.array((self.offset_x_mm, self.offset_y_mm))*1e-3
        return _inside_bore(shape, basis_m, origin, self.radius_mm)


@dataclass(frozen=True)
class _PlacedAperture:
    """Translate/rotate a transverse aperture in its own physical plane."""

    component: object
    registration: object
    z_mm: float

    @property
    def key(self):
        return self.component.key

    @property
    def radius_mm(self):
        return self.component.radius_mm

    def transmission_mask(self, x_mm, y_mm):
        from temsim.physics.aperture_clipping import _transmits
        offset = self.registration.origin_array_m*1e3
        rotation = self.registration.rotation_array
        x, y = x_mm-offset[0], y_mm-offset[1]
        local_x = x*rotation[0, 0] + y*rotation[1, 0]
        local_y = x*rotation[0, 1] + y*rotation[1, 1]
        return _transmits(self.component, local_x, local_y)


def _wave_apertures(state, start, stop):
    from temsim.physics.aperture_clipping import posed_aperture_registration
    apertures = []
    for aperture in state.apertures:
        if not all(bool(getattr(aperture, n, True)) for n in ("installed", "enabled", "inserted")):
            continue
        registration = posed_aperture_registration(state, aperture)
        if registration is not None:
            if not np.array_equal(registration.rotation_array[2], (0., 0., 1.)):
                apertures.append(PosedWaveAperture.from_component(state, aperture, registration))
                continue
            aperture = _PlacedAperture(aperture, registration,
                aperture.z_mm+registration.origin_global_m[2]*1e3)
        if start < aperture.z_mm <= stop:
            apertures.append(aperture)
    return apertures


def _wave_bores(state, start, stop):
    """Use the particle solver's installed bore inventory, even without B.

    Tilted constraints use the same finite local-Z slab as the ray solver.
    Swept slice masks resolve oblique shoulders without inventing a finite
    outer radius. Keep these constraints even outside the opening's nominal
    global-Z extent; a sufficiently off-axis wave can still strike the slab.
    """
    stationary, placed = _partition_vacuum_segments(
        state, _vacuum_segments(state, np.array((start, stop))))
    bores = []
    for segment, registration in placed:
        radius = .5*float(segment.inner_diameter_mm)
        if not math.isfinite(radius) or radius <= 0.:
            raise ValueError("Vacuum inner diameter must be finite and positive")
        rotation = registration.rotation_array
        if not np.array_equal(rotation[2], (0., 0., 1.)):
            bores.append(PosedWaveBore.from_segment(segment, registration))
            continue
        offset = registration.origin_array_m*1e3
        lower, upper = segment.start_z_mm+offset[2], segment.end_z_mm+offset[2]
        if lower <= stop and upper >= start:
            bores.append(_ParallelBore(segment.key, lower, upper,
                radius, float(offset[0]), float(offset[1])))
    return stationary, tuple(bores)


def _prepare_column(state, start, stop, maximum_step_mm):
    if not math.isfinite(stop) or stop <= start:
        raise ValueError("Column wave output must follow the executed input plane")
    if not math.isfinite(maximum_step_mm) or maximum_step_mm <= 0:
        raise ValueError("Column step must be finite and positive")
    if math.ceil((stop-start)/min(float(state.step_mm), maximum_step_mm)) > 1_000_000:
        raise ValueError("Column wave grid exceeds one million steps; choose a feasible numerical budget")
    if bool(getattr(state, "equivalent_image_lenses_enabled", False)):
        raise ValueError("Tip wave transport needs distributed image optics; only executed checkpoints can replace them")
    nano = getattr(state, "nanopulser", None)
    if nano is not None and nano.installed:
        raise ValueError("The installed electrostatic beam blanker needs time-energy wavepacket transport; it cannot be omitted")
    apertures = _wave_apertures(state, start, stop)
    if len({a.key for a in apertures}) != len(apertures):
        raise ValueError("Duplicate physical column aperture")
    walls, placed_bores = _wave_bores(state, start, stop)
    boundaries = [z for w in (*walls, *placed_bores)
                  for z in (w.event_z_mm() if isinstance(w, PosedWaveBore)
                            else (w.start_z_mm, w.end_z_mm)) if start < z < stop]
    boundaries.extend(z for aperture in apertures if isinstance(aperture, PosedWaveAperture)
                      for z in aperture.event_z_mm() if start < z < stop)
    events, owners = _component_events(state, start, stop)
    plan = build_propagation_plan(state, start, stop, events,
        save_z_mm=[a.z_mm for a in apertures if not isinstance(a, PosedWaveAperture)]+boundaries,
        maximum_step_mm=maximum_step_mm)
    from temsim.physics.wave_field_admission import require_supported_column_wave_fields
    require_supported_column_wave_fields(state, start, stop, plan)
    from temsim.physics.posed_lens_wave import require_analytic_fields
    require_analytic_fields(plan.mapped_fields)
    if np.any(plan.thin_power_m1) or np.any(plan.thin_rotation_rad):
        raise ValueError("Unexecuted equivalent image maps are not admitted")
    axis, _, radii = _expanded_profile_axis(plan.z_mm, walls)
    if not np.array_equal(axis, plan.z_mm):
        raise ValueError("Column wave plan is missing a mechanical boundary")
    stops = {}
    for aperture in apertures:
        if isinstance(aperture, PosedWaveAperture):
            for index in range(len(plan.z_mm)):
                stops.setdefault(index, []).append(aperture)
            continue
        index = int(np.argmin(abs(plan.z_mm-aperture.z_mm)))
        if abs(plan.z_mm[index]-aperture.z_mm) > 1e-9:
            raise ValueError("Column wave plan is missing an aperture plane")
        stops.setdefault(index, []).append(aperture)
    for bore in placed_bores:
        # Mechanical contacts stay active throughout the real axial span and
        # are never inferred from active magnetic fields or nominal centres.
        indices = (range(len(plan.z_mm)) if isinstance(bore, PosedWaveBore)
                   else np.flatnonzero((plan.z_mm >= bore.start_z_mm) & (plan.z_mm <= bore.end_z_mm)))
        for index in indices:
            stops.setdefault(int(index), []).append(bore)
    if placed_bores or any(isinstance(a, (_PlacedAperture, PosedWaveAperture)) for a in apertures):
        from temsim.immutable_json import json_digest
        # Field plans can be identical with B switched off. A moved physical
        # mask must still invalidate a stored column continuation.
        plan = replace(plan, signature=json_digest((plan.signature,
            tuple(asdict(bore) for bore in placed_bores),
            tuple(a.identity for a in apertures if isinstance(a, PosedWaveAperture)),
            tuple((a.key, a.z_mm, asdict(a.registration))
                  for a in apertures if isinstance(a, _PlacedAperture)))))
    return plan, radii, stops, owners


def _linear_factor(path, generator, distance, depth=0, *, force=None, constant_h=0.):
    """Lift a known constant Hamiltonian; bisect without changing its map."""
    try:
        if force is None or not np.any(force):
            path.append(expm(generator*distance), action_m=-constant_h*distance)
        else:
            # Integrate the driven centre and its central Weyl action in the
            # same matrix exponential. a'=1/2(F_p.q-F_q.p), not a phase fitted
            # to the final ray displacement. A uniform force then has the
            # independently known a=F^2 L^3/12 in a unit drift.
            extended = np.zeros((6, 6))
            extended[:4, :4] = generator
            extended[:4, 5] = force
            extended[4, :4] = .5*np.r_[force[2:], -force[:2]]
            result = expm(extended*distance)
            path.append(result[:4, :4], result[:4, 5],
                        action_m=float(result[4, 5])-constant_h*distance)
    except ValueError as error:
        if "Canonical phase path is undersampled" not in str(error) or depth >= 24:
            raise
        _linear_factor(path, generator, distance/2, depth+1, force=force, constant_h=constant_h)
        _linear_factor(path, generator, distance/2, depth+1, force=force, constant_h=constant_h)


def _electric_coefficients(plan):
    field = getattr(plan, "electric_field", None)
    if field is None:
        count = len(plan.step_m)
        return np.zeros(count+1), np.zeros(count), np.zeros((count, 2)), np.zeros((count, 2, 2))
    from temsim.physics.wave_field_admission import sample_wave_electric
    nodes = sample_wave_electric(field, plan.z_mm)[0]
    phi, gradient, hessian = sample_wave_electric(field, .5*(plan.z_mm[:-1]+plan.z_mm[1:]))
    return nodes, phi, gradient, hessian


def _column_transports(plan, energy_kev, *, electric=None, dipoles=None):
    """Symplectic maps in fixed entrance-p canonical units during acceleration.

    H/p0=(P+A)^2/(2 p(z) p0)-p(phi(x,y,z))/p0 plus the shared quadrupole
    potentials. Expanding the scalar p(phi) to second order includes both
    electrostatic curvature and the negative relativistic second derivative.
    Lens/stigmator coefficients were defined at instrument reference momentum;
    convert that force to this mode's fixed canonical units rather than changing
    the captured magnetic field when energy changes.
    """
    from scipy.constants import e, m_e
    from temsim.physics.tip_gun_wave import _momentum_velocity
    nodes, phi, gradient, hessian = _electric_coefficients(plan) if electric is None else electric
    entrance_p, _ = _momentum_velocity(energy_kev*1000.)
    momentum, velocity = _momentum_velocity(energy_kev*1000.+phi-nodes[0])
    reference_p = float(getattr(plan, "reference_momentum_kg_m_s", entrance_p))
    scale = reference_p/entrance_p
    strength = (e*hessian/velocity[:, None, None]
        -(m_e*m_e*e*e/momentum**3)[:, None, None]*gradient[:, :, None]*gradient[:, None, :])/entrance_p
    bx = np.asarray(getattr(plan, "dipole_bx_t", np.zeros(3*len(plan.step_m))))[1::3]
    by = np.asarray(getattr(plan, "dipole_by_t", np.zeros(3*len(plan.step_m))))[1::3]
    if dipoles is not None:
        bx, by = dipoles
    posed = getattr(plan, "mapped_fields", ())
    if posed:
        from temsim.physics.posed_lens_wave import posed_lens_correction
    for i, dz in enumerate(plan.step_m):
        b = -e*plan.midpoint_magnetic_t[i]/2
        g = b/momentum[i]
        rotation = np.array(((0., g), (-g, 0.)))
        stiffness = np.diag((scale*plan.midpoint_sx_m2[i]+b*b/(entrance_p*momentum[i]),
                             scale*plan.midpoint_sy_m2[i]+b*b/(entrance_p*momentum[i])))
        stiffness[0, 1] = stiffness[1, 0] = scale*plan.midpoint_sxy_m2[i]
        generator = np.block([[rotation, np.eye(2)*entrance_p/momentum[i]],
                              [strength[i]-stiffness, rotation]])
        force = np.r_[np.zeros(2), e*(gradient[i]/velocity[i]+np.array((by[i], -bx[i])))/entrance_p]
        constant_h = 0.
        if posed:
            correction = posed_lens_correction(posed, .5*(plan.z_mm[i]+plan.z_mm[i+1])*1e-3,
                float(momentum[i]), float(entrance_p), aligned_b_t=float(plan.midpoint_magnetic_t[i]),
                potential_gradient_v_m=gradient[i], potential_hessian_v_m2=hessian[i],
                kinetic_energy_ev=float(energy_kev*1000.+phi[i]-nodes[0]),
                axis_momentum_magnetic=True)
            generator += correction.generator
            force += correction.force
            constant_h = correction.constant_h
        if posed:
            yield _posed_path(generator, force, correction, float(dz))
            continue
        else:
            path = _ColumnPath(1e-3, generator, force, float(dz))
        if not posed and not np.any(rotation) and not np.any(strength[i]-stiffness) and not np.any(force):
            matrix = np.eye(4)
            matrix[:2, 2:] = np.eye(2)*float(dz)*entrance_p/momentum[i]
            path.append(matrix)
        else:
            _linear_factor(path, generator, float(dz), force=force, constant_h=constant_h)
        yield path


def _mode_dipoles(plan, owners, mode_owners, state):
    """Re-evaluate finite driven coils without adding a second centre kick."""
    from scipy.constants import e
    count = len(plan.step_m)
    bx = np.asarray(getattr(plan, "dipole_bx_t", np.zeros(3*count)))[1::3].copy()
    by = np.asarray(getattr(plan, "dipole_by_t", np.zeros(3*count)))[1::3].copy()
    reference_p = electron(state)[1]
    z = .5*(plan.z_mm[:-1]+plan.z_mm[1:])
    updated = {(row["component"], row["z_mm"]): row for row in mode_owners}
    posed_drives = {field.component_key for field in getattr(plan, "mapped_fields", ())
                    if getattr(field, "event_z_mm", None) is not None}
    for old in owners:
        if old["component"] in posed_drives:
            continue  # The rotated vector provider owns this entire field.
        new = updated[(old["component"], old["z_mm"])]
        if not old.get("finite_field", False):
            continue
        length = old["effective_thickness_mm"]
        active = abs(z-old["z_mm"]) < .5*length
        dx, dy = np.subtract(new["kick_rad"], old["kick_rad"])
        bx[active] -= reference_p*dy/(e*length*1e-3)
        by[active] += reference_p*dx/(e*length*1e-3)
    return bx, by


def _inside_bore(shape, basis_m, origin_m, radius_mm):
    """Prove every sample of an affine grid is strictly inside a circular bore."""
    if not math.isfinite(radius_mm):
        return True  # The existing wall operator is inactive for nonfinite radii.
    ny, nx = shape
    corners = np.array(((-(nx//2), -(ny//2)), (nx-1-nx//2, -(ny//2)),
                        (nx-1-nx//2, ny-1-ny//2), (-(nx//2), ny-1-ny//2)), dtype=float)
    corner_xy = (origin_m[:, None]+basis_m@corners.T)*1e3
    coordinate_bound = (abs(origin_m)+abs(basis_m)@np.max(abs(corners), axis=0))*1e3
    margin = 16*np.finfo(float).eps*np.hypot(*coordinate_bound)
    return np.nextafter(np.max(np.hypot(*corner_xy))+margin, np.inf) < radius_mm


def _linear_run(wave, wavelength, paths, first, plan, radii, stops, kick_x, kick_y,
                *, posed_checks=None, phase_budget_per_m=.01):
    """Compose resolved quadratic steps only while every intermediate grid is safe.

    This changes the numerical carrier schedule, never the integrated fields.
    No event or possibly active wall is crossed. All tests use the actual input
    lattice after carrier selection, not a Gaussian fit or a ray envelope.
    """
    from temsim.physics.canonical_action import principal_reference_phase
    curvature = np.zeros((2, 2)) if wave.curvature_m1 is None else wave.curvature_m1
    tilt = np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
    inverse_basis = np.linalg.inv(wave.basis_m)
    basis_condition = np.linalg.cond(wave.basis_m)
    theta = wavelength*np.linalg.norm(inverse_basis, ord=2)
    extent = np.linalg.norm(wave.basis_m, ord=2)*min(wave.amplitude.shape)
    nxny = np.array(wave.amplitude.shape[::-1])
    combined = CanonicalPath(1e-3)
    accepted, last = None, first
    checks = []
    from temsim.physics.posed_aberrations import kick_indices
    spherical_events = kick_indices(getattr(plan, "posed_spherical_kicks", ()), plan.z_mm,
                                    include_initial=False)
    for i in range(first, len(paths)):
        j = i+1
        if (plan.midpoint_hex_normal_m3[i] or plan.midpoint_hex_skew_m3[i]
                or plan.cs_kick_m3[j] or j in spherical_events or kick_x[j] or kick_y[j]
                or any(not isinstance(stop, (_ParallelBore, PosedWaveBore)) for stop in stops.get(j, ()))):
            break
        child = paths[i]
        old_matrix, old_offset = combined.matrix.copy(), combined.offset.copy()
        if isinstance(child, _PosedColumnPath):
            oa, ob = old_matrix[:2, :2], old_matrix[:2, 2:]
            effective_before = oa+ob@curvature
            if np.linalg.cond(effective_before) >= 1e8:
                break
            previous_curvature = np.linalg.solve(effective_before.T,
                (old_matrix[2:, :2]+old_matrix[2:, 2:]@curvature).T).T
            previous_origin = oa@wave.origin_m+ob@tilt+old_offset[:2]
            previous_tilt = old_matrix[2:, :2]@wave.origin_m+old_matrix[2:, 2:]@tilt+old_offset[2:]
            child = _recenter_posed_path(child, previous_origin, previous_tilt)
            box = _phase_space_box(wave.amplitude.shape, effective_before@wave.basis_m,
                                   previous_origin, previous_curvature, previous_tilt, wavelength,
                                   center=child.correction.center_phase_space)
            try:
                check = _check_posed_step(child, box, float(plan.step_m[i]), wavelength, phase_budget_per_m)
            except ValueError:
                # A grouped representation can have a larger conservative
                # Fourier box. Retry this step on the actual executed wave.
                break
        else:
            check = None
        # A child with an internally lifted full turn cannot be reconstructed
        # from its endpoint. Keep the stepwise operator for that case.
        if abs(child.reference_phase_rad-principal_reference_phase(
                child.matrix, child.reference_length_m)) > 1e-10:
            break
        try:
            combined.append(child.matrix, child.offset, action_m=child.action_m)
        except ValueError as error:
            if "Canonical phase path is undersampled" not in str(error):
                raise
            break
        a, b = combined.matrix[:2, :2], combined.matrix[:2, 2:]
        effective = a+b@curvature
        condition = np.linalg.cond(effective) if np.isfinite(effective).all() else np.inf
        if condition >= 1e8:
            break
        drift = np.linalg.solve(effective, b)
        # Retain the existing angular-spectrum bound, plus a bound in each
        # sampled direction for skew or highly anisotropic lattices.
        index_drift = wavelength*inverse_basis@drift@inverse_basis.T
        # Do not admit a boundary case merely because inverses/products rounded
        # inward. Ill-conditioned lattices conservatively keep the step path.
        slack = 64*np.finfo(float).eps*max(1., basis_condition, condition)
        if (np.linalg.norm(drift, ord=2)*theta >= (1-slack)*.25*extent
                or np.any(.5*np.sum(abs(index_drift), axis=1) >= (1-slack)*.25*nxny)):
            break
        basis = effective@wave.basis_m
        origin = a@wave.origin_m+b@tilt+combined.offset[:2]
        if not _inside_bore(wave.amplitude.shape, basis, origin, radii[j]):
            break
        def clears(bore):
            if isinstance(bore, PosedWaveBore):
                if check is not None:
                    bound = check["position_bound_m"]
                elif isinstance(child, _ColumnPath):
                    oa, ob = old_matrix[:2, :2], old_matrix[:2, 2:]
                    oe = oa+ob@curvature
                    if np.linalg.cond(oe) >= 1e8:
                        return False
                    oq = np.linalg.solve(oe.T,
                        (old_matrix[2:, :2]+old_matrix[2:, 2:]@curvature).T).T
                    oo = oa@wave.origin_m+ob@tilt+old_offset[:2]
                    ot = old_matrix[2:, :2]@wave.origin_m+old_matrix[2:, 2:]@tilt+old_offset[2:]
                    box = _phase_space_box(wave.amplitude.shape, oe@wave.basis_m, oo, oq, ot, wavelength)
                    bound = (child.support_majorant@np.r_[box, 1.])[:2]
                else:
                    return False  # No complete-step bound, so keep the step.
                # A two-by-two lattice whose corners enclose the complete
                # position box. This covers an interior excursion, not just
                # the input and output grid projections.
                return bore.clears_grid((2, 2), np.diag(2*bound), bound,
                                       plan.z_mm[j], previous_z_mm=plan.z_mm[i])
            return bore.clears_grid(wave.amplitude.shape, basis, origin)
        if any(not clears(bore) for bore in stops.get(j, ())):
            break
        # append mutates its owner, so the last accepted path needs a copy.
        accepted = CanonicalPath(combined.reference_length_m)
        accepted.matrix, accepted.offset = combined.matrix.copy(), combined.offset.copy()
        accepted.action_m, accepted.reference_phase_rad = combined.action_m, combined.reference_phase_rad
        last = j
        if check is not None:
            checks.append(check)
    if posed_checks is not None:
        posed_checks.extend(checks)
    return last, accepted


def _clip(wave, radius_mm, apertures, z_mm, prior, *, previous_z_mm=None):
    xp = array_module(wave.amplitude)
    xy = None
    rows = []
    if math.isfinite(radius_mm):
        inside_bore = _inside_bore(wave.amplitude.shape, wave.basis_m, wave.origin_m, radius_mm)
        if not inside_bore:
            xy = wave.coordinates_m()*1e3
            before = wave.probability
            mask = xp.hypot(xy[0], xy[1]) < radius_mm
            if not bool(xp.all(mask)):
                wave = replace(wave, amplitude=xp.where(mask, wave.amplitude, 0j))
            if wave.probability < before:
                rows.append({"component": "column_wall", "z_mm": z_mm,
                             "lost_weight": prior*(before-wave.probability)})
    for a in apertures:
        if isinstance(a, _ParallelBore) and a.clears_grid(wave.amplitude.shape, wave.basis_m, wave.origin_m):
            continue
        if isinstance(a, PosedWaveBore) and a.clears_grid(wave.amplitude.shape,
                wave.basis_m, wave.origin_m, z_mm, previous_z_mm=previous_z_mm):
            continue
        if xy is None:
            xy = wave.coordinates_m()*1e3
        if isinstance(a, PosedWaveBore):
            # Even a closed tilted plate occupies only its local slab, not
            # the entire laboratory XY observation plane.
            mask = a.transmission_mask(xy[0], xy[1], z_mm, previous_z_mm=previous_z_mm, xp=xp)
        elif float(getattr(a, "radius_mm", 1.)) <= 0:
            mask = xp.zeros_like(xy[0], dtype=bool)
        elif isinstance(a, _ParallelBore):
            mask = xp.hypot(xy[0]-a.offset_x_mm, xy[1]-a.offset_y_mm) < a.radius_mm
        elif hasattr(a, "transmission_mask"):
            # Runtime apertures can supply non-circular/slit geometry. Keep
            # their authoritative mask; only transfer geometry at this event.
            mask = xp.asarray(a.transmission_mask(to_host(xy[0]), to_host(xy[1])), dtype=bool)
        else:
            mask = xp.hypot(xy[0]-a.offset_x_mm, xy[1]-a.offset_y_mm) <= a.radius_mm
        if mask.shape != wave.amplitude.shape:
            raise ValueError("Column aperture mask has the wrong shape")
        before = wave.probability
        wave = replace(wave, amplitude=xp.where(mask, wave.amplitude, 0j))
        rows.append({"component": a.key, "z_mm": z_mm,
            "input_weight": prior*before, "output_weight": prior*wave.probability,
            "lost_weight": prior*(before-wave.probability)})
    return wave, rows


def _maximum_mode_pixels(checkpoint):
    modes = checkpoint.beam.modes
    if hasattr(modes, "rows"):
        # This is only a resource estimate. The normal mode reader still
        # verifies checksums and validates every array before execution.
        return max((math.prod(row["arrays"]["amplitude"]["shape"]) for row in modes.rows), default=0)
    return max((mode.plane.amplitude.size for mode in modes), default=0)


def _propagate_column(state, checkpoint, stop_z_mm, *, maximum_step_mm=.5,
                      grid_numerics=WaveGridNumerics(), retained_bytes=0,
                      tip_time_s=None, _prepared=None, _combine_linear=True,
                      cancelled=lambda: False, progress_callback=None):
    """Execute a segment with resident device fields and host checkpoints."""
    from temsim.physics.compute_backend import gpu_failure_category, gpu_retry_reason
    prepared = (_prepare_column(state, checkpoint.plane_z_mm, stop_z_mm, maximum_step_mm)
                if _prepared is None else _prepared)
    largest = _maximum_mode_pixels(checkpoint)
    arguments = dict(maximum_step_mm=maximum_step_mm, grid_numerics=grid_numerics,
        retained_bytes=retained_bytes, tip_time_s=tip_time_s, _prepared=prepared,
        _combine_linear=_combine_linear, cancelled=cancelled, progress_callback=progress_callback)
    try:
        with device_scope(grid_numerics.compute_backend,
                acceleration_enabled=grid_numerics.acceleration_enabled,
                maximum_working_bytes=grid_numerics.maximum_device_working_bytes,
                work_items=largest*len(prepared[0].step_m), required_bytes=256*largest) as device:
            return _propagate_column_impl(state, checkpoint, stop_z_mm, _device=device, **arguments)
    except Exception as error:
        # Physics, grid and cancellation failures never become CPU successes.
        if gpu_failure_category(error) is None:
            raise
        reason = gpu_retry_reason(error, normalise_backend(grid_numerics.compute_backend),
                                  stage="coherent_column")
        error.__traceback__ = None
        if cancelled():
            raise InterruptedError("Column wave propagation cancelled") from None
    if progress_callback:
        progress_callback(0, len(prepared[0].step_m), "GPU memory unavailable; retrying the unchanged column segment on CPU")
    with device_scope("CPU") as device:
        device.fallback_reason = reason
        return _propagate_column_impl(state, checkpoint, stop_z_mm, _device=device, **arguments)


def _propagate_column_impl(state, checkpoint, stop_z_mm, *, maximum_step_mm=.5,
                      grid_numerics=WaveGridNumerics(), retained_bytes=0,
                      tip_time_s=None, _prepared=None, _combine_linear=True, _device,
                      cancelled=lambda: False, progress_callback=None):
    from temsim.optics.electron_gun.tip_coherence import wavelength_m
    plan, radii, stops, owners = (_prepare_column(state, checkpoint.plane_z_mm, stop_z_mm, maximum_step_mm)
                                 if _prepared is None else _prepared)
    outputs, records, maps = [], [], {}
    electric = _electric_coefficients(plan)
    from temsim.physics.posed_aberrations import kick_indices
    spherical_events = kick_indices(getattr(plan, "posed_spherical_kicks", ()), plan.z_mm,
                                    include_initial=False)
    spherical_electric = {}
    if spherical_events and getattr(plan, "electric_field", None) is not None:
        from temsim.physics.wave_field_admission import sample_spherical_electric
        event_nodes = tuple(spherical_events)
        gradients, hessians = sample_spherical_electric(plan.electric_field,
                                                       plan.z_mm[list(event_nodes)])
        spherical_electric = dict(zip(event_nodes, zip(gradients, hessians)))
    from temsim.physics.wave_checkpoint_store import resident_wave_bytes
    retained_bytes += resident_wave_bytes(checkpoint.beam)
    map_bytes = (2048 if getattr(plan, "mapped_fields", ()) else 512)*len(plan.step_m)
    retained_bytes += map_bytes
    for mode in checkpoint.beam.modes:
        grid_numerics.check(mode.plane.amplitude.shape, retained_bytes=retained_bytes)
    for index, mode in enumerate(checkpoint.beam.modes):
        if cancelled():
            raise InterruptedError("Column wave propagation cancelled")
        from temsim.physics.tip_gun_wave import _momentum_velocity
        entrance_p, entrance_v = _momentum_velocity(mode.energy_kev*1000)
        nodes, phi, _, _ = electric
        energies = mode.energy_kev*1000+phi-nodes[0]
        momentum, velocity = _momentum_velocity(energies)
        exit_energy = mode.energy_kev*1000+nodes[-1]-nodes[0]
        exit_p, _ = _momentum_velocity(exit_energy)
        elapsed = np.r_[0., np.cumsum(plan.step_m/velocity)]
        flight_increment = float(elapsed[-1])
        action_increment = float(np.sum(plan.step_m*momentum))
        dynamic_actions = any(row.get("dynamic", False) for row in owners)
        mode_owners, kick_x, kick_y = owners, plan.kick_x_rad, plan.kick_y_rad
        mode_plan = plan
        if dynamic_actions:
            if mode.axial_reference is None:
                raise ValueError("Dynamic coil transport requires the executed flight time from the tip")
            epoch = float(getattr(state, "simulation_time_s", 0.) if tip_time_s is None else tip_time_s)
            if not math.isfinite(epoch):
                raise ValueError("Tip emission clock must be finite")
            arrival_time = lambda z: epoch+mode.axial_reference.flight_time_s+(
                (float(np.interp(z, plan.z_mm, elapsed)) if plan.z_mm[0] <= z <= plan.z_mm[-1]
                 else (z-checkpoint.plane_z_mm)*1e-3/entrance_v))
            events, mode_owners = _component_events(state, checkpoint.plane_z_mm, stop_z_mm,
                arrival_time=arrival_time)
            from temsim.physics.posed_column_fields import FrozenPosedMultipole, capture_posed_column_fields
            if any(isinstance(field, FrozenPosedMultipole) and field.dynamic
                   for field in getattr(plan, "mapped_fields", ())):
                from temsim.simulation_modes import is_ideal
                captured = {field.lens_key: field for field in capture_posed_column_fields(
                    state, include_hexapole=not is_ideal(state), arrival_time=arrival_time)}
                mode_fields = tuple(captured[field.lens_key] if isinstance(field, FrozenPosedMultipole)
                                    else field for field in plan.mapped_fields)
                if is_dataclass(plan):
                    mode_plan = replace(plan, mapped_fields=mode_fields)
                else:
                    from types import SimpleNamespace
                    mode_plan = SimpleNamespace(**{**vars(plan), "mapped_fields": mode_fields})
            kick_x, kick_y = np.zeros(len(plan.z_mm)), np.zeros(len(plan.z_mm))
            for (z, x, y), owner in zip(events, mode_owners):
                if owner.get("finite_field", False):
                    continue
                node = int(np.argmin(abs(plan.z_mm-z)))
                if abs(plan.z_mm[node]-z) > 1e-9:
                    raise ValueError("Dynamic coil is missing from the physical integration grid")
                kick_x[node] += x; kick_y[node] += y
        if mode.energy_kev not in maps or dynamic_actions:
            maps.clear()  # only one energy's current segment, never all past paths
            maps[mode.energy_kev] = tuple(_column_transports(mode_plan, mode.energy_kev, electric=electric,
                dipoles=_mode_dipoles(plan, owners, mode_owners, state) if dynamic_actions else None))
        wave, losses, refinements, linear_runs = mode.plane, [], [], []
        residual_records = []
        posed_checks = []
        spherical_actions = []
        if _device.backend == "cupy":
            wave = replace(wave, amplitude=_device.xp.asarray(wave.amplitude))
        wavelength = float(wavelength_m(mode.energy_kev*1000))
        def multipole(wave, z, **strengths):
            if not any(strengths.values()):
                return wave
            try:
                result, rows = apply_resolved_operator(wave,
                    lambda value: apply_multipole_phase(value, wavelength, **strengths),
                    numerics=grid_numerics, retained_bytes=retained_bytes, cancelled=cancelled)
            except ValueError as error:
                raise ValueError(f"Column z={z:.9g} mm, mode {mode.mode_id}, strengths={strengths}: {error}") from error
            for row in rows:
                row.update(z_mm=float(z), strengths=strengths)
            refinements.extend(rows)
            if rows and progress_callback:
                progress_callback(index*len(plan.step_m)+i, len(checkpoint.beam.modes)*len(plan.step_m),
                                  f"Column wave refined {rows[0]['from_shape']} -> {rows[-1]['to_shape']} at {z:.9g} mm")
            return result
        def linear(wave, path, z, *, input_z, operator="distributed_linear", selected_gauge=None):
            output_rows = []
            try:
                if selected_gauge is None:
                    wave, gauge_rows = reselect_phase_carrier(wave, wavelength,
                        grid_numerics=grid_numerics, retained_bytes=retained_bytes, cancelled=cancelled)
                else:
                    wave, gauge_rows = selected_gauge
                result, rows = apply_resolved_operator(wave,
                    lambda value: propagate_plane_wave(value, path.matrix, path.offset,
                        wavelength, **path.phase_kwargs(), grid_numerics=grid_numerics,
                        retained_bytes=retained_bytes, cancelled=cancelled,
                        sampling_records=output_rows),
                    numerics=grid_numerics, retained_bytes=retained_bytes, cancelled=cancelled)
            except ValueError as error:
                raise ValueError(f"Column linear z={z:.9g} mm, mode {mode.mode_id}: {error}") from error
            for row in gauge_rows:
                row.update(z_mm=float(input_z), input_z_mm=float(input_z), output_z_mm=float(z),
                           mode_id=mode.mode_id, operator="phase_carrier_representation")
            rows = gauge_rows+rows+output_rows
            for row in rows[len(gauge_rows):]:
                row.update(z_mm=float(z), input_z_mm=float(input_z), output_z_mm=float(z),
                           mode_id=mode.mode_id, operator=operator)
            refinements.extend(rows)
            if rows and progress_callback:
                changed = rows[0]['from_shape'] != rows[-1]['to_shape']
                message = (f"Column wave grid {rows[0]['from_shape']} -> {rows[-1]['to_shape']}" if changed
                           else "Column wave phase representation checked")
                progress_callback(index*len(plan.step_m)+i, len(checkpoint.beam.modes)*len(plan.step_m),
                    f"{message} at {z:.9g} mm")
            return result

        def magnetic_residual(wave, path, dz, z):
            from temsim.physics.wave_magnetic_residual import apply_magnetic_residual, WORKING_BYTES_PER_PIXEL
            from temsim.physics.wave_grid import WaveSamplingError
            evidence = []
            def apply(value):
                grid_numerics.check(value.amplitude.shape, retained_bytes=retained_bytes,
                    working_bytes_per_pixel=WORKING_BYTES_PER_PIXEL)
                box = _phase_space_box(value.amplitude.shape, value.basis_m, value.origin_m,
                    np.zeros((2, 2)) if value.curvature_m1 is None else value.curvature_m1,
                    np.zeros(2) if value.tilt_rad is None else value.tilt_rad,
                    wavelength, center=path.correction.center_phase_space)
                # Refinement increases the represented Fourier bandwidth.
                # Recheck the actual lattice before applying its residual.
                support_check = _check_posed_step(path, box, abs(float(dz)), wavelength,
                    grid_numerics.maximum_posed_lens_phase_error_rad_per_m, allow_residual=True)
                xp = array_module(value.amplitude)
                velocity, scalar = path.correction.residual_coefficients(value, wavelength)
                # These are local sampling checks, separate from the bounded
                # exponential-series error and from axial/grid convergence.
                phase = -2*np.pi/wavelength*dz*scalar
                increment = max(float(xp.max(abs(xp.diff(phase, axis=axis)))) for axis in (0, 1))
                displacement = xp.einsum("ij,jyx->iyx", xp.asarray(np.linalg.inv(value.basis_m)), velocity)*dz
                travel = float(xp.max(abs(displacement)))
                if travel > .125*min(value.amplitude.shape):
                    raise ValueError("Magnetic residual displacement exceeds the physical wave-grid domain; "
                                     "a wider represented domain is required")
                if increment > .2*np.pi:
                    raise WaveSamplingError("Magnetic residual exceeds the phase/displacement sampling guard",
                                            increment/(.2*np.pi))
                remaining_budget = (grid_numerics.maximum_posed_lens_phase_error_rad_per_m*abs(dz)
                                    -support_check["electric_magnetic_remainder_bound_rad"])
                if remaining_budget <= 0:
                    raise ValueError("Electric-magnetic remainder leaves no magnetic residual error budget")
                result, record = apply_magnetic_residual(value, velocity, scalar, wavelength, dz,
                    error_budget=remaining_budget,
                    cancelled=cancelled)
                record.update(z_mm=float(z), maximum_phase_increment_rad=increment,
                              maximum_lattice_displacement=travel,
                              electric_magnetic_remainder_bound_rad=support_check["electric_magnetic_remainder_bound_rad"],
                              global_slope_bound=support_check["global_slope_bound"],
                              local_slope_bound=support_check["local_slope_bound"])
                evidence.append(record)
                return result
            try:
                result, rows = apply_resolved_operator(wave, apply,
                    numerics=grid_numerics, retained_bytes=retained_bytes, cancelled=cancelled)
            except ValueError as error:
                raise ValueError(f"Tilted-lens phase correction at Z {z:.9g} mm: {error}") from error
            refinements.extend(dict(row, z_mm=float(z), operator="magnetic_residual") for row in rows)
            residual_records.extend(evidence[-1:])
            return result
        i = 0
        # A resumed input may already touch the moved wall. Do not propagate
        # that amplitude through material before applying its first mask.
        if stops.get(0):
            wave, rows = _clip(wave, radii[0], stops[0], float(plan.z_mm[0]), mode.weight_per_reference_electron)
            losses.extend(rows)
        paths = maps[mode.energy_kev]
        while i < len(paths):
            dz, path = plan.step_m[i], paths[i]
            if i % 64 == 0:
                if cancelled():
                    raise InterruptedError("Column wave propagation cancelled")
                if progress_callback:
                    progress_callback(index*len(plan.step_m)+i, len(checkpoint.beam.modes)*len(plan.step_m), "Distributed column wave")
            if wave.probability == 0:
                break
            multipole_scale = float(getattr(plan, "reference_momentum_kg_m_s", entrance_p))/entrance_p
            hn, hs = plan.midpoint_hex_normal_m3[i]*dz*multipole_scale/2, plan.midpoint_hex_skew_m3[i]*dz*multipole_scale/2
            wave = multipole(wave, plan.z_mm[i], normal_m2=hn, skew_m2=hs)
            selected = None
            if _combine_linear and not hn and not hs:
                selected = reselect_phase_carrier(wave, wavelength,
                    grid_numerics=grid_numerics, retained_bytes=retained_bytes, cancelled=cancelled)
                group_checks = []
                last, combined = _linear_run(selected[0], wavelength, paths, i, plan, radii, stops, kick_x, kick_y,
                    posed_checks=group_checks,
                    phase_budget_per_m=grid_numerics.maximum_posed_lens_phase_error_rad_per_m)
                if last > i+1:
                    wave = linear(wave, combined, plan.z_mm[last], input_z=plan.z_mm[i], selected_gauge=selected)
                    posed_checks.extend(group_checks)
                    linear_runs.append({"first_step": i, "last_step_exclusive": last})
                    i = last
                    continue
            if isinstance(path, _PosedColumnPath):
                checked_wave = selected[0] if selected is not None else wave
                checked_tilt = np.zeros(2) if checked_wave.tilt_rad is None else checked_wave.tilt_rad
                path = _recenter_posed_path(path, checked_wave.origin_m, checked_tilt)
                box = _phase_space_box(checked_wave.amplitude.shape, checked_wave.basis_m, checked_wave.origin_m,
                    np.zeros((2, 2)) if checked_wave.curvature_m1 is None else checked_wave.curvature_m1,
                    checked_tilt, wavelength, center=path.correction.center_phase_space)
                check = _check_posed_step(path, box, float(dz), wavelength,
                    grid_numerics.maximum_posed_lens_phase_error_rad_per_m, allow_residual=True)
                posed_checks.append(check)
                if check["magnetic_residual_required"]:
                    wave = checked_wave
                    wave = magnetic_residual(wave, path, float(dz)/2, plan.z_mm[i])
                    selected = None  # The envelope changed; reselect its carrier normally.
            wave = linear(wave, path, plan.z_mm[i+1], input_z=plan.z_mm[i], selected_gauge=selected)
            if isinstance(path, _PosedColumnPath) and check["magnetic_residual_required"]:
                wave = magnetic_residual(wave, path, float(dz)/2, plan.z_mm[i+1])
            wave = multipole(wave, plan.z_mm[i+1], normal_m2=hn, skew_m2=hs)
            j = i+1
            if kick_x[j] or kick_y[j]:
                kick = CanonicalPath(1e-3)
                local_p, _ = _momentum_velocity(mode.energy_kev*1000+nodes[j]-nodes[0])
                kick.append(np.eye(4), np.array((0., 0., kick_x[j], kick_y[j]))*local_p/entrance_p)
                wave = linear(wave, kick, plan.z_mm[j], input_z=plan.z_mm[j], operator="physical_impulse")
            local_p, _ = _momentum_velocity(mode.energy_kev*1000+nodes[j]-nodes[0])
            wave = multipole(wave, plan.z_mm[j], spherical_m3=plan.cs_kick_m3[j]*local_p/entrance_p)
            for spherical in spherical_events.get(j, ()):
                if spherical.strength_m3 == 0:
                    continue
                from temsim.physics.posed_aberration_wave import apply_posed_spherical
                evidence = []

                def apply_spherical(value):
                    # Include the paired trial fields and spectral residual
                    # scratch used by the obliquity/splitting checks.
                    grid_numerics.check(value.amplitude.shape, retained_bytes=retained_bytes,
                                        working_bytes_per_pixel=1024)
                    result, record = apply_posed_spherical(value, wavelength, spherical,
                        momentum_kg_m_s=float(local_p), reference_momentum_kg_m_s=float(entrance_p),
                        fields=getattr(mode_plan, "mapped_fields", ()),
                        aligned_bz_t=float(plan.magnetic_t[j]), error_budget=1e-10,
                        electrostatic_gradient_v_m=spherical_electric.get(j, (None, None))[0],
                        electrostatic_hessian_v_m2=spherical_electric.get(j, (None, None))[1],
                        maximum_obliquity_phase_error_rad=grid_numerics.maximum_spherical_obliquity_phase_error_rad,
                        cancelled=cancelled)
                    evidence.append(record)
                    return result

                try:
                    wave, rows = apply_resolved_operator(wave, apply_spherical,
                        numerics=grid_numerics, retained_bytes=retained_bytes, cancelled=cancelled)
                except ValueError as error:
                    raise ValueError(f"Spherical aberration of {spherical.lens_key} at "
                                     f"Z {plan.z_mm[j]:.9g} mm: {error}") from error
                refinements.extend(dict(row, z_mm=float(plan.z_mm[j]), operator="posed_spherical")
                                   for row in rows)
                spherical_actions.extend(dict(row, z_mm=float(plan.z_mm[j]),
                                              component=spherical.lens_key) for row in evidence[-1:])
            wave, rows = _clip(wave, radii[j], stops.get(j, ()), float(plan.z_mm[j]),
                               mode.weight_per_reference_electron, previous_z_mm=float(plan.z_mm[i]))
            losses.extend(rows)
            i += 1
        norm = wave.probability
        reference = None if mode.axial_reference is None else mode.axial_reference.advance(flight_increment, action_increment)
        ratio = float(entrance_p/exit_p)
        amplitude = wave.amplitude/math.sqrt(norm) if norm else wave.amplitude
        output = replace(mode, plane=replace(wave, amplitude=to_host(amplitude),
                         curvature_m1=None if wave.curvature_m1 is None else wave.curvature_m1*ratio,
                         tilt_rad=None if wave.tilt_rad is None else wave.tilt_rad*ratio),
                         weight_per_reference_electron=mode.weight_per_reference_electron*norm,
                         energy_kev=float(exit_energy)*1e-3,
                         axial_reference=reference)
        outputs.append(output)
        retained_bytes += output.plane.amplitude.nbytes
        records.append({"mode_id": mode.mode_id, "energy_kev": mode.energy_kev, **_device.evidence(),
                        "input_weight": mode.weight_per_reference_electron,
                        "output_energy_kev": output.energy_kev,
                        "reference_flight_time_increment_s": flight_increment,
                        "reference_longitudinal_action_increment_j_s": action_increment,
                        "axial_reference": None if reference is None else asdict(reference),
                        "deflector_actions": mode_owners,
                        "posed_column_fields": [asdict(field) for field in getattr(mode_plan, "mapped_fields", ())
                                                if hasattr(field, "component_key")],
                        "output_weight": output.weight_per_reference_electron, "losses": losses,
                        "posed_lens_approximation": {"checked_steps": len(posed_checks),
                            "magnetic_phase_error_bound_rad": sum(row["phase_error_bound_rad"] for row in posed_checks),
                            "electric_magnetic_remainder_bound_rad": sum(row["electric_magnetic_remainder_bound_rad"] for row in posed_checks),
                            "phase_budget_rad_per_m": grid_numerics.maximum_posed_lens_phase_error_rad_per_m,
                            "maximum_global_slope_bound": max((row["global_slope_bound"] for row in (*posed_checks, *residual_records)), default=0.),
                            "maximum_local_slope_bound": max((row["local_slope_bound"] for row in (*posed_checks, *residual_records)), default=0.),
                            "magnetic_residual_series_error_bound": sum(row["series_error_bound"] for row in residual_records),
                            "scope": "omitted magnetic Taylor/inverse-momentum action and finite-grid residual exponential series recorded separately; roundoff, scalar field, axial splitting, grid and paraxial convergence separate"},
                        "magnetic_residual_steps": residual_records,
                        "posed_spherical_actions": spherical_actions,
                        "grid_refinements": refinements, "quadratic_runs": linear_runs, "output_shape": wave.amplitude.shape})
    return TipGunCheckpoint(BeamState(tuple(outputs), checkpoint.beam.reference_plane), stop_z_mm,
        checkpoint.reference_current_a, {"schema": "executed-column-wave-v1", "upstream_digest": checkpoint.digest,
        "upstream": checkpoint.record, "field_plan_signature": plan.signature, "step_count": len(plan.step_m),
        "maximum_step_mm": maximum_step_mm, "deflector_actions": owners, "modes": records,
        "grid_numerics": grid_numerics.column_identity(),
        "integrator": "second-order midpoint affine quadratic Hamiltonian with variable longitudinal momentum / symmetric magnetic residual and cubic splits / discrete axial and canonical first-inclination Cs",
        "posed_spherical_model": "canonical-first-inclination-cs-v1; per-event approximation and exponential/splitting errors recorded separately",
        "coordinate_basis": "laboratory canonical normalized to each mode entrance momentum; output phase carriers converted to exit momentum; continuous symmetric axial-field gauge",
        "electric_model": "captured static scalar field, near-axis second-order transverse expansion; axis energy, axial action and flight time carried",
        "dipole_model": "shared finite field supports; driven coils frozen at per-mode axial arrival time, no longitudinal pulse envelope",
        "posed_lens_model": "rigid analytical vector potential with Az and affine scalar action; carrier-centred axis-momentum magnetic Hamiltonian with bounded transverse electric coupling omission and Hermitian spectral magnetic residual; swept oblique bore absorption",
        "time_model": "actual coil laws at per-mode axial arrival times; frozen transverse slices, no longitudinal pulse wavepacket",
        "validation_status": "DEVELOPMENT"})


def _slice_prepared(prepared, first, last):
    """Slice existing integration nodes; no new steps or repeated entrance kick."""
    from types import SimpleNamespace
    from temsim.immutable_json import json_digest
    plan, radii, stops, owners = prepared
    values = {name: getattr(plan, name)[first:last] for name in (
        "step_m", "midpoint_magnetic_t", "midpoint_sx_m2", "midpoint_sy_m2",
        "midpoint_sxy_m2",
        "midpoint_hex_normal_m3", "midpoint_hex_skew_m3")}
    values.update({name: getattr(plan, name)[first:last+1] for name in ("z_mm", "kick_x_rad", "kick_y_rad", "cs_kick_m3")})
    if hasattr(plan, "magnetic_t"):
        # Discrete posed Cs uses the same node gauge as its particle action.
        values["magnetic_t"] = plan.magnetic_t[first:last+1]
    values.update({name: getattr(plan, name)[3*first:3*last] for name in ("dipole_bx_t", "dipole_by_t")
                   if hasattr(plan, name)})
    for name in ("electric_field", "electric_field_identity", "reference_momentum_kg_m_s", "mapped_fields"):
        if hasattr(plan, name):
            values[name] = getattr(plan, name)
    # The checkpoint at the left boundary already contains its discrete
    # action. A lens at the right boundary belongs to this segment exactly once.
    values["posed_spherical_kicks"] = tuple(
        kick for kick in getattr(plan, "posed_spherical_kicks", ())
        if plan.z_mm[first] < kick.z_mm <= plan.z_mm[last])
    values["signature"] = json_digest((plan.signature, first, last))
    sliced_stops = {i-first: value for i, value in stops.items() if first < i <= last}
    entrance_bores = [value for value in stops.get(first, ()) if isinstance(value, (_ParallelBore, PosedWaveBore))]
    if entrance_bores:
        # Bore masks are idempotent. Retain them at every resumed entrance;
        # discrete aperture events and kicks still execute only downstream.
        sliced_stops[0] = entrance_bores
    return (SimpleNamespace(**values), radii[first:last+1],
            sliced_stops,
            [r for r in owners if (
                r["z_mm"]+.5*r.get("effective_thickness_mm", 0.) > plan.z_mm[first]
                and r["z_mm"]-.5*r.get("effective_thickness_mm", 0.) < plan.z_mm[last])
             if r.get("finite_field", False)] +
            [r for r in owners if not r.get("finite_field", False)
             and plan.z_mm[first] < r["z_mm"] <= plan.z_mm[last]])


def _column_mode_workers(checkpoint, requested=None):
    """Capture the owning numerical job's budget before starting threads."""
    return min(8, len(checkpoint.beam.modes), numerical_thread_budget(requested))


def _column_shared_bytes(checkpoint, prepared):
    """Resident inputs shared by workers, without loading stored amplitudes."""
    from temsim.physics.wave_checkpoint_store import resident_wave_bytes
    resident = resident_wave_bytes(checkpoint.beam)
    if not hasattr(checkpoint.beam, "content_identity"):
        resident += sum(value.nbytes for value in checkpoint.auxiliary_arrays.values())
    plan = prepared[0]
    values = ((getattr(plan, item.name) for item in dataclass_fields(plan))
              if is_dataclass(plan) else vars(plan).values())
    arrays = [value for value in values if isinstance(value, np.ndarray)]
    arrays.append(prepared[1])
    return resident+sum(value.nbytes for value in arrays)


def _memory_budget_cause(error):
    """Follow explicit contextual wrappers; never infer from error strings."""
    seen = set()
    while error is not None and id(error) not in seen:
        if isinstance(error, WaveMemoryBudgetError):
            return error
        seen.add(id(error))
        error = error.__cause__
    return None


def _write_column_modes(state, checkpoint, stop_z_mm, writer, *, prepared, maximum_step_mm,
                        grid_numerics, workers, shared_bytes, tip_time_s, cancelled,
                        progress_callback, parent_digest):
    """Bounded slots include in-flight inputs, scratch and pending outputs.

    Each batch has at most one mode per slot. Finished modes retain their slot
    until the main thread writes them in source order; no additional work is
    submitted while those outputs are pending. The 256-byte-per-pixel working
    estimate therefore covers both active scratch and a completed slot's
    immutable output. Shared inputs and plan arrays are reserved separately.
    """
    stop = Event()
    progress_lock = Lock()
    remaining = grid_numerics.maximum_working_bytes-shared_bytes
    if remaining < workers:
        raise WaveMemoryBudgetError("Resident column inputs and plan leave no working-memory partition")
    worker_bytes = remaining//workers
    local_numerics = replace(grid_numerics, maximum_working_bytes=worker_bytes)
    records = []

    def report(*args):
        if progress_callback is not None:
            with progress_lock:
                progress_callback(*args)

    def execute(index, *, threaded=False):
        if threaded:
            # BLAS is already serial in the owning numerical job. Numba's
            # thread-local mask must also be one in each mode worker.
            initialize_numerical_thread(1)
        if cancelled() or stop.is_set():
            raise InterruptedError("Column mode propagation cancelled")
        mode = checkpoint.beam.modes[index]
        local = TipGunCheckpoint(BeamState((mode,), checkpoint.beam.reference_plane), checkpoint.plane_z_mm,
            checkpoint.reference_current_a, {"executed_parent": parent_digest})
        return _propagate_column(state, local, stop_z_mm, maximum_step_mm=maximum_step_mm,
            grid_numerics=local_numerics, tip_time_s=tip_time_s, _prepared=prepared,
            cancelled=lambda: cancelled() or stop.is_set(), progress_callback=report)

    def append(transported):
        if cancelled():
            raise InterruptedError("Column modes cancelled before writing checkpoint")
        writer.append(transported.beam.modes[0])
        records.extend(transported.record["modes"])

    if workers == 1:
        for index in range(len(checkpoint.beam.modes)):
            transported = execute(index)
            append(transported)
            del transported
        return records, worker_bytes

    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="temsim-wave-mode")
    futures = []
    errors = []
    try:
        for first in range(0, len(checkpoint.beam.modes), workers):
            futures = [pool.submit(copy_context().run, execute, index, threaded=True)
                       for index in range(first, min(first+workers, len(checkpoint.beam.modes)))]
            errors = []
            positions = {future: index for index, future in enumerate(futures)}
            completed, next_write = set(), 0
            for future in as_completed(futures):
                try:
                    future.result()
                except CancelledError:
                    pass
                except BaseException as error:
                    errors.append(error)
                    stop.set()
                    for pending in futures:
                        pending.cancel()
                else:
                    completed.add(positions[future])
                    while not errors and next_write in completed:
                        transported = futures[next_write].result()
                        append(transported)
                        del transported
                        next_write += 1
            if cancelled():
                raise InterruptedError("Column mode propagation cancelled")
            if errors:
                # A genuine physical/grid error must not be hidden by a
                # concurrent partition failure or its cooperative cancellation.
                genuine = [error for error in errors
                           if _memory_budget_cause(error) is None and not isinstance(error, InterruptedError)]
                if genuine:
                    raise genuine[0]
                memory = next((_memory_budget_cause(error) for error in errors
                               if _memory_budget_cause(error) is not None), None)
                if memory is not None:
                    message = str(memory)
                    # Release failed-worker frames before the serial retry;
                    # tracebacks otherwise retain their large intermediate grids.
                    for error in errors:
                        while error is not None:
                            error.__traceback__ = None
                            error = error.__cause__
                    raise WaveMemoryBudgetError(message) from None
                raise errors[0]
            positions.clear()
            futures.clear()
            del future
        return records, worker_bytes
    finally:
        stop.set()
        for future in futures:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        futures.clear()
        errors.clear()


def _propagate_column_segmented(state, checkpoint, stop_z_mm, *, store, segment_steps,
                               maximum_step_mm=.5, grid_numerics=WaveGridNumerics(), tip_time_s=None,
                               cancelled=lambda: False, progress_callback=None, verify=lambda: None, use_cache=True,
                               checkpoint_callback=None, _prepared=None):
    from temsim.immutable_json import json_digest
    prepared = (_prepare_column(state, checkpoint.plane_z_mm, stop_z_mm, maximum_step_mm)
                if _prepared is None else _prepared)
    plan = prepared[0]
    if plan.z_mm[0] != checkpoint.plane_z_mm or plan.z_mm[-1] != stop_z_mm:
        raise ValueError("Prepared column plan must match the exact executed interval")
    result, hit = checkpoint, False
    for first in range(0, len(plan.step_m), segment_steps):
        if cancelled():
            raise InterruptedError("Segmented column propagation cancelled")
        last = min(first+segment_steps, len(plan.step_m))
        segment = _slice_prepared(prepared, first, last)
        time_key = tip_time_s if any(row["dynamic"] for row in segment[3]) else None
        parent_digest = result.digest
        key = store.key("column-segment", parent_digest, segment[0].signature, grid_numerics.column_identity(), time_key)
        cached = store.get(key) if use_cache else None
        if cached is not None:
            result, hit = cached, True
            if checkpoint_callback is not None:
                checkpoint_callback(result)
            continue
        shared_bytes = _column_shared_bytes(result, prepared)
        workers = _column_mode_workers(result)
        # The device pool belongs to one wave mode at a time. CPU mode workers
        # must not each independently claim the complete GPU memory budget.
        from temsim.physics.compute_backend import choose_wave_backend, WAVE_BACKEND_CUPY
        largest = _maximum_mode_pixels(result)
        selected, _ = choose_wave_backend(normalise_backend(grid_numerics.compute_backend),
            acceleration_enabled=grid_numerics.acceleration_enabled, work_items=largest*len(segment[0].step_m))
        if selected == WAVE_BACKEND_CUPY:
            workers = 1
        retry_serial = False
        while True:
            writer = store.writer(key, result.beam.reference_plane)
            try:
                records, worker_bytes = _write_column_modes(state, result, float(plan.z_mm[last]), writer,
                    prepared=segment, maximum_step_mm=maximum_step_mm, grid_numerics=grid_numerics,
                    workers=workers, shared_bytes=shared_bytes, tip_time_s=tip_time_s,
                    cancelled=cancelled, progress_callback=progress_callback, parent_digest=parent_digest)
                verify()
                if cancelled():
                    raise InterruptedError("Segment cancelled before checkpoint commit")
                result = writer.finish(float(plan.z_mm[last]), result.reference_current_a,
                    {"schema": "executed-column-segment-v1", "upstream_digest": parent_digest,
                     "upstream": result.record, "parent_plan_signature": plan.signature,
                     "node_span": (first, last), "modes": records,
                     "mode_execution": {"workers": workers, "retry_serial_for_memory": retry_serial,
                         "maximum_working_bytes": grid_numerics.maximum_working_bytes,
                         "shared_retained_bytes": shared_bytes, "worker_working_bytes": worker_bytes},
                     "memory_policy": "bounded mode slots; shared inputs reserved; ordered streaming commit"})
                break
            except WaveMemoryBudgetError as error:
                if workers == 1 or cancelled():
                    raise
                error.__traceback__ = None
                error.__cause__ = None
                error.__context__ = None
                # The pool has joined and no partial checkpoint is published.
                # Retry the same modes/operators with the full remaining budget.
                workers, retry_serial = 1, True
                if progress_callback:
                    progress_callback(first, len(plan.step_m),
                        "Column mode memory partition too small; continuing serially")
            finally:
                writer.abort()
        if checkpoint_callback is not None:
            checkpoint_callback(result)
        if progress_callback:
            progress_callback(last, len(plan.step_m), "Column segment saved; previous wave buffers released")
    return result, hit
