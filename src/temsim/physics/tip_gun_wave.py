"""Executed tip-to-exit wave transport in the installed FEG fields.

This is a scalar, stationary, second-order paraxial Hamiltonian solver. It
retains variable longitudinal momentum, electrostatic focusing, both gun
deflector coils, the rotated magnetic stigmator, the analytic Wien electric
and magnetic fields, DPA/C1/slit masks and sampled absorbing body bores.
The actual solved electrostatic provider is expanded to quadratic order about
the optical axis. Captured column-lens tails and overlapping quadratic magnets
are included when an instrument is supplied. This is not full TEM/STEM
qualification or admission of arbitrary imported fields.

Canonical coordinates use the constant *launch* momentum p_ref, so the map
remains symplectic during acceleration. At the exit only the phase-carrier
units change to p_exit; no new amplitude, source current or source is fitted.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
import math
from types import MappingProxyType

import numpy as np
from scipy.constants import c, e, h, m_e

from temsim.immutable_json import freeze_json, json_digest
from temsim.optics.electron_gun.tip_coherence import (
    TIP_REFERENCE, TipWaveNumerics, generate_tip_boundary_emission, generate_tip_emission, wavelength_m,
)
from temsim.physics.canonical_action import CanonicalPath
from temsim.physics.multiplane_wave import propagate_plane_wave
from temsim.physics.wave_flux import BeamState, WaveMode


@dataclass(frozen=True)
class GunWaveNumerics:
    field_step_mm: float = 0.05
    bore_step_mm: float = 1.0
    max_steps: int = 100_000
    maximum_checkpoint_bytes: int = 8 * 1024**3
    maximum_fractional_energy_change: float = 0.05

    def validate(self):
        for name in ("field_step_mm", "bore_step_mm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Gun wave {name} must be positive and finite")
        fraction = self.maximum_fractional_energy_change
        if (isinstance(fraction, bool) or not math.isfinite(fraction)
                or not 0 < fraction <= 1):
            raise ValueError("Gun wave maximum_fractional_energy_change must lie in (0, 1]")
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int) or not 1 <= self.max_steps <= 1_000_000:
            raise ValueError("Gun wave max_steps must be a positive bounded integer")
        if isinstance(self.maximum_checkpoint_bytes, bool) or not isinstance(self.maximum_checkpoint_bytes, int) or self.maximum_checkpoint_bytes <= 0:
            raise ValueError("Gun checkpoint memory budget must be a positive integer")
        return self


@dataclass(frozen=True)
class TipGunCheckpoint:
    beam: BeamState
    plane_z_mm: float
    reference_current_a: float
    record: object
    auxiliary_arrays: object = field(default_factory=dict)

    def __post_init__(self):
        if self.beam.reference_plane != TIP_REFERENCE:
            raise ValueError("A physical gun checkpoint must retain its tip electron reference")
        if not math.isfinite(self.plane_z_mm) or self.plane_z_mm <= 0:
            raise ValueError("A gun checkpoint must be downstream of the tip")
        if not math.isfinite(self.reference_current_a) or self.reference_current_a < 0:
            raise ValueError("Invalid tip reference current")
        if self.beam.total_weight > 1+1e-10:
            raise ValueError("Gun propagation created electron probability")
        object.__setattr__(self, "record", freeze_json(self.record))
        # Committed disk checkpoints expose a checksum-checked immutable lazy
        # mapping. Their manifest already binds every auxiliary payload; do
        # not read all near-tip arrays during an unrelated Z browse.
        if hasattr(self.beam, "content_identity"):
            return
        auxiliary = {}
        for name, value in self.auxiliary_arrays.items():
            if (not isinstance(name, str) or not name or len(name) > 80
                    or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name)):
                raise ValueError("Checkpoint auxiliary state needs a plain array name")
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise ValueError("Checkpoint auxiliary state cannot contain Python objects")
            auxiliary[name] = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
        object.__setattr__(self, "auxiliary_arrays", MappingProxyType(auxiliary))

    @property
    def digest(self):
        if hasattr(self.beam, "content_identity"):
            return json_digest({"record": self.record, "stored_payload": self.beam.content_identity,
                                "plane_z_mm": self.plane_z_mm, "current_a": self.reference_current_a})
        digest = sha256()
        for name, array in sorted(self.auxiliary_arrays.items()):
            digest.update(name.encode())
            digest.update(str((array.dtype.str, array.shape)).encode())
            digest.update(array.tobytes())
        for mode in self.beam.modes:
            for name in ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad"):
                value = getattr(mode.plane, name)
                digest.update(b"none" if value is None else np.ascontiguousarray(value).tobytes())
            digest.update(json_digest((mode.mode_id, mode.energy_kev, mode.weight_per_reference_electron,
                                       None if mode.axial_reference is None else asdict(mode.axial_reference),
                                       mode.scattering_history)).encode())
        return json_digest({"record": self.record, "payload": digest.hexdigest(),
                            "plane_z_mm": self.plane_z_mm, "current_a": self.reference_current_a})

    @property
    def transmitted_current_a(self):
        return self.reference_current_a*self.beam.total_weight


def _momentum_velocity(energy_ev):
    kinetic = np.asarray(energy_ev, dtype=float)*e
    if np.any(~np.isfinite(kinetic)) or np.any(kinetic <= 0):
        raise ValueError("Gun potential has a turning/forbidden region; forward paraxial transport is unavailable")
    momentum = np.sqrt(kinetic*(kinetic+2*m_e*c*c))/c
    return momentum, momentum*c*c/(kinetic+m_e*c*c)


def _require_supported_gun_fields(gun):
    """Reject unsupported providers before any field or mesh query."""
    from temsim.optics.electron_gun.field_emission import FieldEmissionGun
    from temsim.optics.electron_gun.monochromator import AnalyticWienField
    if type(gun) is not FieldEmissionGun or gun.type_key != "cold_feg":
        raise ValueError("Tip wave gun transport currently requires the physical cold-FEG assembly")
    if gun.monochromator_installed and type(gun.monochromator.field_provider) is not AnalyticWienField:
        raise ValueError("Imported Wien fields need a validated non-quadratic wave operator; the installed provider cannot be skipped")


def _field_coefficients(gun, z_mm, *, electric_provider=None):
    """Read the installed solved E and supported analytic magnetic/Wien fields.

    The electric provider is expanded to second order about the physical
    optical axis. Magnetic gun dipoles and quadrupoles retain their exact
    transverse law. Arbitrary providers fail closed rather than being
    substituted with a matched Wien or an independently defined exit wave.
    """
    _require_supported_gun_fields(gun)
    z = np.asarray(z_mm, dtype=float)
    positions = np.zeros((len(z), 3))
    positions[:, 2] = z*1e-3
    electric = gun.electric_field if electric_provider is None else electric_provider
    magnetic = gun.magnetic_field
    from temsim.physics.wave_field_admission import sample_wave_electric
    phi, gradient, hessian = sample_wave_electric(electric, z)
    magnetic0 = magnetic.field_at_global_positions_t(positions)
    bj = np.empty((len(z), 2, 2))
    delta = 1e-4
    for axis in range(2):
        plus, minus = positions.copy(), positions.copy()
        plus[:, axis], minus[:, axis] = delta, -delta
        bj[:, :, axis] = (magnetic.field_at_global_positions_t(plus)[:, :2]
                         - magnetic.field_at_global_positions_t(minus)[:, :2])/(2*delta)
    if np.any(magnetic0[:, 2] != 0):
        raise ValueError("An axial gun magnetic field requires its vector-potential rotation operator")
    # phi gradient/Hessian and grad(A_z)=(-B_y,B_x) for q=-e.
    magnetic_force = np.stack((magnetic0[:, 1], -magnetic0[:, 0]), axis=1)
    magnetic_hessian = np.stack((bj[:, 1], -bj[:, 0]), axis=1)
    for array in (hessian, magnetic_hessian):
        if not np.allclose(array, array.transpose(0, 2, 1), rtol=1e-12, atol=1e-12):
            raise ValueError("Installed gun fields do not yield a symmetric canonical phase Hessian")
    return phi, gradient, hessian, magnetic_force, magnetic_hessian


def _refine_energy_grid(z_mm, electric, launch_energy_ev, numerics, cancelled=lambda: False):
    """Bound axial energy change without replacing the installed field.

    Midpoint quadrature in p(K), 1/p(K) and 1/v(K) needs an energy scale as
    well as a distance scale near emission. Retain every original physical
    node and insert midpoint nodes until the endpoint/midpoint energy range
    is at most maximum_fractional_energy_change of its minimum. For a linear rising
    non-relativistic potential the .05 default bounds the local 1/sqrt(K)
    midpoint relative error by 3*fraction**2/32 (<0.024%). This is a local
    quadrature bound, not a field or complete gun convergence certificate.
    """
    numerics.validate()
    grid = np.asarray(z_mm, dtype=float)
    if (grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all()
            or grid[0] != 0 or np.any(np.diff(grid) <= 0)):
        raise ValueError("Gun energy refinement needs a finite ordered grid beginning at the tip")
    if len(grid)-1 > numerics.max_steps:
        raise ValueError("Gun energy refinement exceeds max_steps")
    query = getattr(electric, "potential_rise_v_at_global_positions", None)
    if query is None:
        query = electric.potential_v_at_global_positions

    def potentials(nodes):
        points = np.zeros((len(nodes), 3))
        points[:, 2] = nodes*1e-3
        values = np.asarray(query(points), dtype=float)
        if values.shape != nodes.shape or not np.isfinite(values).all():
            raise ValueError("Gun electric provider returned invalid axial potentials")
        return values

    if cancelled():
        raise InterruptedError("Tip-to-exit wave energy refinement cancelled")
    phi = potentials(grid)
    energy = float(launch_energy_ev)+phi-phi[0]
    _momentum_velocity(energy)  # Reject turning/forbidden nodes before transport.
    for _ in range(64):
        if cancelled():
            raise InterruptedError("Tip-to-exit wave energy refinement cancelled")
        mids = grid[:-1]+.5*np.diff(grid)
        mid_energy = float(launch_energy_ev)+potentials(mids)-phi[0]
        _momentum_velocity(mid_energy)
        low = np.minimum(np.minimum(energy[:-1], energy[1:]), mid_energy)
        high = np.maximum(np.maximum(energy[:-1], energy[1:]), mid_energy)
        refine = (high-low)/low > numerics.maximum_fractional_energy_change
        count = int(np.count_nonzero(refine))
        if not count:
            return grid
        if len(grid)-1+count > numerics.max_steps:
            raise ValueError("Gun energy refinement exceeds max_steps; increase the explicit numerical budget")
        if np.any((mids[refine] <= grid[:-1][refine]) | (mids[refine] >= grid[1:][refine])):
            raise ValueError("Gun energy refinement cannot resolve the requested tolerance in finite Z precision")
        new_grid = np.sort(np.r_[grid, mids[refine]])
        # Retain exact queried endpoint/midpoint values; no field interpolation
        # or potential fit is introduced by the quadrature refinement.
        positions = np.searchsorted(new_grid, grid)
        new_energy = np.empty(len(new_grid))
        new_energy[positions] = energy
        new_energy[np.searchsorted(new_grid, mids[refine])] = mid_energy[refine]
        grid, energy = new_grid, new_energy
    raise ValueError("Gun energy refinement did not converge within its bounded refinement depth")


def _axial_grid(gun, numerics, *, exact_z_mm=(), extra_mask_planes=(),
                electric_provider=None, minimum_energy_ev=None, cancelled=lambda: False,
                start_z_mm=0.):
    stop = float(gun.exit_plane_z_mm)
    if not math.isfinite(stop) or stop <= 0:
        raise ValueError("Physical gun exit must follow the tip")
    if not math.isfinite(start_z_mm) or not 0 <= start_z_mm < stop:
        raise ValueError("Executed gun continuation must begin before the gun exit")
    # Boundaries come from geometry and field providers, never a source plane.
    events = {0., start_z_mm, stop, float(gun.dpa_aperture.z_mm), float(gun.c1_aperture.z_mm)}
    masks = set(events)
    masks.update(float(value) for value in extra_mask_planes)
    events.update(float(value) for value in exact_z_mm)
    for component in gun.bore_components:
        start = component.mechanical_center_from_tip_mm-.5*component.mechanical_length_mm
        end = component.mechanical_center_from_tip_mm+.5*component.mechanical_length_mm
        count = math.ceil((end-start)/numerics.bore_step_mm)
        if count > numerics.max_steps:
            raise ValueError("Gun bore grid exceeds max_steps")
        masks.update(np.linspace(start, end, count+1).tolist())
    events.update(masks)
    for start, end in gun.field_supports_mm:
        events.update((float(start), float(end)))
    # Extractor support has its own field offset (the legacy step selector's
    # support list does not include it); explicitly include both true edges.
    events.update((gun.extractor.transition_start_mm+gun.extractor.field_center_offset_mm,
                   gun.extractor.transition_end_mm+gun.extractor.field_center_offset_mm))
    ordered = sorted(value for value in events if 0 <= value <= stop)
    count = sum(math.ceil((b-a)/numerics.field_step_mm) for a, b in zip(ordered, ordered[1:]))
    if count > numerics.max_steps:
        raise ValueError(f"Gun wave grid needs {count} steps, above max_steps={numerics.max_steps}")
    grid = [ordered[0]]
    for a, b in zip(ordered, ordered[1:]):
        grid.extend(np.linspace(a, b, math.ceil((b-a)/numerics.field_step_mm)+1)[1:].tolist())
    grid = np.array(grid)
    if electric_provider is not None or minimum_energy_ev is not None:
        # The radial surface-boundary solver also uses this geometry helper;
        # its different reference energy must not be silently substituted by
        # the Gaussian tip's launch energy. The executed Gaussian gun always
        # supplies its actual provider and minimum quadrature energy below.
        electric = gun.electric_field if electric_provider is None else electric_provider
        minimum = gun.emitter.emission_energy_ev if minimum_energy_ev is None else minimum_energy_ev
        grid = _refine_energy_grid(grid, electric, minimum, numerics, cancelled)
    # A non-paraxial prefix has already executed every earlier interval. Keep
    # its exact face and all later physical events without double propagation.
    grid = grid[grid >= start_z_mm]
    return grid, frozenset(v for v in masks if start_z_mm < v <= stop)


def _kick(path, strength, force, depth=0):
    matrix = np.eye(4)
    matrix[2:, :2] = strength
    try:
        path.append(matrix, np.r_[np.zeros(2), force])
    except ValueError as error:
        if "Canonical phase path is undersampled" not in str(error) or depth >= 24:
            raise
        # Subdivide the KNOWN kick Hamiltonian, not an arbitrary endpoint map.
        # Both factors commute, so this resolves the lift without changing
        # the physical kick, wavefront, field grid or accuracy order.
        _kick(path, strength/2, force/2, depth+1)
        _kick(path, strength/2, force/2, depth+1)


def _drift(path, distance, depth=0):
    matrix = np.eye(4)
    matrix[:2, 2:] = np.eye(2)*distance
    try:
        path.append(matrix)
    except ValueError as error:
        if "Canonical phase path is undersampled" not in str(error) or depth >= 24:
            raise
        _drift(path, distance/2, depth+1)
        _drift(path, distance/2, depth+1)


def _energy_transport(gun, z, mask_planes, coefficients, launch_energy, cancelled, axial_b_t=None,
                      electric_provider=None):
    """Symmetric kick/drift/kick, retaining the continuous Weyl action and lift."""
    phi, gradient, hessian, magnetic_force, magnetic_hessian = coefficients
    electric = gun.electric_field if electric_provider is None else electric_provider
    query = getattr(electric, "potential_rise_v_at_global_positions", None)
    if query is None:
        query = electric.potential_v_at_global_positions
    launch_position = np.array(((0., 0., float(z[0])*1e-3),))
    launch_phi = float(query(launch_position)[0])
    energies = launch_energy+phi-launch_phi
    momentum, velocity = _momentum_velocity(energies)
    p_ref, _ = _momentum_velocity(launch_energy)
    # Second derivative of p(E) is -m_e^2/p^3. The electric dipole term
    # supplies a real quadratic contribution, including in the Wien filter.
    strength = (e*hessian/velocity[:, None, None]
                - (m_e*m_e*e*e/momentum**3)[:, None, None]
                * gradient[:, :, None]*gradient[:, None, :]
                + e*magnetic_hessian)/p_ref
    force = e*(gradient/velocity[:, None]+magnetic_force)/p_ref
    # Shared column lens tails use A=(-By/2,Bx/2,0), q=-e, and the SAME
    # constant launch canonical momentum. This includes both diamagnetic
    # focusing and Larmor rotation while longitudinal momentum accelerates.
    g_ref = np.zeros_like(momentum) if axial_b_t is None else -e*np.asarray(axial_b_t)/(2*p_ref)
    strength -= (p_ref/momentum*g_ref*g_ref)[:, None, None]*np.eye(2)
    path = CanonicalPath(1e-3)
    overall = CanonicalPath(1e-3)
    segments = []
    flight_time = 0.
    longitudinal_action = 0.
    for i, dz_mm in enumerate(np.diff(z)):
        if i % 128 == 0 and cancelled():
            raise InterruptedError("Tip-to-exit wave propagation cancelled")
        dz = float(dz_mm)*1e-3
        # Each factor has a known Hamiltonian and phase, rather than adding
        # phase after a completed trajectory calculation.
        for active in (path, overall):
            _kick(active, strength[i]*dz/2, force[i]*dz/2)
            angle = float(g_ref[i]*p_ref/momentum[i]*dz)
            if angle:
                rotation = np.array(((math.cos(angle), math.sin(angle)), (-math.sin(angle), math.cos(angle))))
                matrix = np.zeros((4, 4))
                matrix[:2, :2] = matrix[2:, 2:] = rotation
                active.append(matrix)
            _drift(active, float(dz*p_ref/momentum[i]))
            _kick(active, strength[i]*dz/2, force[i]*dz/2)
        flight_time += dz/velocity[i]
        longitudinal_action += float(momentum[i]*dz)
        if float(z[i+1]) in mask_planes:
            segments.append((float(z[i+1]), path.matrix, path.offset, path.phase_kwargs()))
            path = CanonicalPath(1e-3)
    exit_position = np.array(((0., 0., float(z[-1])*1e-3),))
    exit_energy = launch_energy+float(query(exit_position)[0])-launch_phi
    p_exit, _ = _momentum_velocity(exit_energy)
    return segments, {
        "launch_energy_ev": launch_energy, "exit_axial_energy_ev": exit_energy,
        "executed_start_z_mm": float(z[0]),
        "reference_flight_time_s": float(flight_time), "momentum_ratio_tip_to_exit": float(p_ref/p_exit),
        "reference_longitudinal_action_j_s": longitudinal_action,
        "longitudinal_carrier_phase_rad": 2*np.pi*longitudinal_action/h,
        "canonical_map": overall.matrix.tolist(), "canonical_offset": overall.offset.tolist(),
        "weyl_action_m": overall.action_m, "metaplectic_reference_phase_rad": overall.reference_phase_rad,
        "phase_path_digest": json_digest([(plane, phase) for plane, _, _, phase in segments])}


def _mask_plane(gun, wave, z_mm, prior, column_apertures=()):
    xy = wave.coordinates_m()*1e3
    rows = []
    for component in gun.bore_components:
        half = .5*component.mechanical_length_mm
        if abs(z_mm-component.mechanical_center_from_tip_mm) <= half+1e-10:
            radius = .5*component.mechanical_clear_bore_diameter_mm
            mask = np.hypot(xy[0], xy[1]) < radius
            before = wave.probability
            if not np.all(mask):
                wave = replace(wave, amplitude=np.where(mask, wave.amplitude, 0j))
            loss = prior*(before-wave.probability)
            if loss > 0:
                rows.append({"component": component.key, "z_mm": z_mm, "lost_probability": loss, "kind": "body_bore"})
    for aperture in (gun.dpa_aperture, gun.c1_aperture):
        if abs(z_mm-aperture.z_mm) < 1e-10:
            is_slit = aperture.kind == "energy_selection_slit"
            if aperture.enabled and not is_slit and aperture.radius_mm <= 0:
                # A zero-area opening transmits zero probability; a sampled
                # pixel whose centre equals the aperture centre is not an area.
                mask = np.zeros(wave.amplitude.shape, dtype=bool)
            else:
                mask = aperture.transmission_mask(xy[0], xy[1])
                if is_slit and gun.monochromator.slit.inserted:
                    gap_m = gun.monochromator.slit.gap_um*1e-6
                    cell_x = float(np.sum(abs(wave.basis_m[0])))
                    if gap_m < 2*cell_x:
                        raise ValueError(f"C1 monochromator slit is undersampled (gap {gap_m*1e6:.6g} um, cell width {cell_x*1e6:.6g} um); increase the tip wave grid before computing its transmission")
            before = wave.probability
            wave = replace(wave, amplitude=np.where(mask, wave.amplitude, 0j))
            rows.append({"component": aperture.key, "z_mm": z_mm, "kind": aperture.interaction_kind,
                         "incoming_probability": prior*before, "outgoing_probability": prior*wave.probability,
                         "lost_probability": prior*(before-wave.probability)})
    # A column component moved into the gun is still a real absorbing opening.
    # Gun-owned DPA/C1 masks above must not be applied a second time.
    gun_keys = {gun.dpa_aperture.key, gun.c1_aperture.key}
    for aperture in column_apertures:
        if aperture.key in gun_keys or abs(z_mm-float(aperture.z_mm)) >= 1e-10:
            continue
        if float(aperture.radius_mm) <= 0:
            mask = np.zeros(wave.amplitude.shape, dtype=bool)
        elif hasattr(aperture, "transmission_mask"):
            mask = np.asarray(aperture.transmission_mask(xy[0], xy[1]), dtype=bool)
        else:
            mask = np.hypot(xy[0]-aperture.offset_x_mm, xy[1]-aperture.offset_y_mm) <= aperture.radius_mm
        if mask.shape != wave.amplitude.shape:
            raise ValueError("Column aperture mask has the wrong shape inside the gun")
        before = wave.probability
        wave = replace(wave, amplitude=np.where(mask, wave.amplitude, 0j))
        rows.append({"component": aperture.key, "z_mm": z_mm, "kind": "column_aperture",
                     "incoming_probability": prior*before, "outgoing_probability": prior*wave.probability,
                     "lost_probability": prior*(before-wave.probability)})
    return wave, rows


def _column_magnetic_coefficients(column, midpoints):
    """Actual column B terms not already present in the local gun provider.

    Stigmator coefficients in the shared particle model use its nominal p/q.
    Convert those coefficients back to B derivatives before combining with
    the accelerating gun's fixed launch-momentum canonical Hamiltonian.
    """
    from temsim.physics.core import fields, multipole_focusing_fields, skew_quadrupole_field, electron
    from temsim.physics.instrument_magnetic import column_dipole_fields, gun_paraxial_fields
    axial = fields(midpoints, column)[0]
    if fields(np.array((0.,)), column)[0][0] != 0:
        raise ValueError("A magnetic field at the tip needs a magnetic emission boundary model")
    positions = np.zeros((len(midpoints), 3))
    positions[:, 2] = midpoints*1e-3
    dipoles = sum((coil.field_at_global_positions_t(positions) for coil in column_dipole_fields(column)),
                  np.zeros_like(positions))
    sx, sy = multipole_focusing_fields(midpoints, column)
    sxy = skew_quadrupole_field(midpoints, column)-gun_paraxial_fields(column, midpoints)[4]
    tensor = np.empty((len(midpoints), 2, 2))
    tensor[:, 0, 0], tensor[:, 1, 1] = sx, sy
    tensor[:, 0, 1] = tensor[:, 1, 0] = sxy
    tensor *= -electron(column)[1]/e
    force = np.stack((dipoles[:, 1], -dipoles[:, 0]), axis=1)
    return axial, force, tensor


_CACHE = OrderedDict()


def build_tip_gun_checkpoint(gun, *, source_numerics=TipWaveNumerics(),
                             numerics=GunWaveNumerics(), use_cache=True,
                             cancelled=lambda: False, progress_callback=None,
                             _column_state=None):
    """Execute from tip parameters and actual optics, or reuse the exact result.

    There is deliberately no intermediate-wave argument. Preparation consumes
    actual fields before cache lookup; a configuration label is not transport.
    """
    from temsim.calculation_manifest import solver_source_identity
    from temsim.instrument_snapshot import encode_instrument, decode_instrument
    from temsim.optics.electron_gun.source_policy import require_physical_gun_source
    require_physical_gun_source(gun)
    if cancelled():
        raise InterruptedError("Tip-to-exit wave propagation cancelled")
    numerics.validate()
    working = deepcopy(gun)
    working.validate()
    driven = getattr(working.emitter.coherence, "boundary_model", "forward_gaussian_schell") == "driven_gaussian_schell"
    emission = (generate_tip_boundary_emission(working, source_numerics) if driven
                else generate_tip_emission(working, source_numerics))
    if driven and _column_state is None:
        raise ValueError("The driven Tip boundary requires the captured instrument so all shared electric and magnetic fields are retained")
    near_numerics = None
    if driven:
        from temsim.physics.accelerating_tip_boundary import AcceleratingTipNumerics
        near_numerics = AcceleratingTipNumerics(maximum_working_bytes=numerics.maximum_checkpoint_bytes)
    start_z_mm = 0. if near_numerics is None else near_numerics.end_z_nm*1e-9*1e3
    _require_supported_gun_fields(working)
    estimated_bytes = (emission.record["mode_count"]+8)*source_numerics.grid_pixels**2*16
    if estimated_bytes > numerics.maximum_checkpoint_bytes:
        raise ValueError(f"Gun checkpoint and wave buffers need approximately {estimated_bytes} bytes, above maximum_checkpoint_bytes={numerics.maximum_checkpoint_bytes}")
    from temsim.physics.wave_execution import check_available_memory
    check_available_memory(estimated_bytes)
    column_record, axial_b, electric, column = None, None, None, None
    column_apertures, exact_z = (), ()
    if _column_state is not None:
        from temsim.physics.core import build_propagation_plan
        from temsim.physics.instrument_electric import capture_instrument_electric_field, configure_instrument_electric_domain
        from temsim.physics.instrument_magnetic import active_column_events
        column = decode_instrument(encode_instrument(_column_state))
        if json_digest(encode_instrument(column.electron_gun)) != json_digest(encode_instrument(working)):
            raise ValueError("Shared column and gun inputs do not describe the same installed gun")
        # Capture the SAME fixed full-instrument electrostatic solve used by
        # the particles and subsequent column stages. A standalone gun domain
        # must not silently replace this solve inside the accelerated stage.
        electric = capture_instrument_electric_field(column)
        configure_instrument_electric_domain(working, float(electric.bounds_m[1, 2])*1e3)
        stop = float(working.exit_plane_z_mm)
        shared = build_propagation_plan(column, 0., stop, active_column_events(column),
                                       maximum_step_mm=numerics.field_step_mm)
        # Quadratic column lenses, stigmators and finite dipoles are included
        # below. Non-quadratic maps/correctors still fail explicitly.
        if shared.mapped_fields or any(np.any(getattr(shared, name)) for name in (
                "hex_normal_m3", "hex_skew_m3", "midpoint_hex_normal_m3", "midpoint_hex_skew_m3",
                "cs_kick_m3", "thin_power_m1", "thin_rotation_rad", "kick_x_rad", "kick_y_rad")):
            raise ValueError("Non-quadratic column components inside the accelerating gun require a joint operator; they cannot be skipped")
        exact_z = tuple(shared.z_mm)
        column_apertures = tuple(a for a in column.apertures
            if all(bool(getattr(a, name, True)) for name in ("installed", "enabled", "inserted"))
            and 0 < float(a.z_mm) <= stop)
        column_record = {"inputs": encode_instrument(column), "plan_signature": shared.signature,
                         "electric_field_identity": electric.numerical_identity,
                         "boundary_gauge": "A=(-Bz*y/2,Bz*x/2,0); canonical momentum continuous"}
    if electric is None:
        electric = working.electric_field
    base_electric = getattr(electric, "base_field", electric)
    exact_z += tuple(float(value)*1e3 for value in getattr(base_electric, "z", ()))
    z, masks = _axial_grid(working, numerics, exact_z_mm=exact_z,
                          extra_mask_planes=(a.z_mm for a in column_apertures),
                          electric_provider=electric,
                          minimum_energy_ev=min(row["energy_ev"] for row in emission.record["energy_modes"]),
                          cancelled=cancelled, start_z_mm=start_z_mm)
    midpoints = (z[1:]+z[:-1])*.5
    coefficients = _field_coefficients(working, midpoints, electric_provider=electric)
    if column is not None:
        axial_b, column_force, column_tensor = _column_magnetic_coefficients(column, midpoints)
        phi, gradient, hessian, magnetic_force, magnetic_hessian = coefficients
        coefficients = (phi, gradient, hessian,
                        magnetic_force+column_force, magnetic_hessian+column_tensor)
    field_digest = sha256()
    for values in (z, *coefficients):
        field_digest.update(np.ascontiguousarray(values).tobytes())
    if axial_b is not None:
        field_digest.update(np.ascontiguousarray(axial_b).tobytes())
    implementation = solver_source_identity()
    dependency = json_digest({"gun": encode_instrument(working), "source": emission.digest,
        "numerics": asdict(numerics), "sampled_fields": field_digest.hexdigest(),
        "implementation": implementation, "shared_column": column_record,
        "accelerating_tip_numerics": None if near_numerics is None else asdict(near_numerics),
        "schema": "executed-tip-gun-v1"})
    if use_cache and dependency in _CACHE:
        if cancelled():
            raise InterruptedError("Tip-to-exit wave propagation cancelled")
        _CACHE.move_to_end(dependency)
        return _CACHE[dependency]
    near = None
    auxiliary = {}
    if driven:
        from temsim.physics.accelerating_tip_boundary import execute_accelerating_tip
        near = execute_accelerating_tip(column, emission, electric, numerics=near_numerics,
            cancelled=cancelled, progress_callback=progress_callback)
        if near.plane_z_mm != float(z[0]):
            raise ValueError("Executed near-tip endpoint does not match the gun continuation face")
        for index, (mode, original) in enumerate(zip(near.modes, emission.modes())):
            prefix = f"near_tip_m{index}_"
            for name, values in (("spectral_amplitude", near.spectral_amplitudes[index]),
                                 ("spectral_derivative", near.spectral_derivatives_per_m[index]),
                                 ("log_transmission", near.logarithmic_transmissions[index]),
                                 ("source_amplitude", original.plane.full_amplitude(float(wavelength_m(original.energy_kev*1e3)))),
                                 ("source_basis_m", original.plane.basis_m),
                                 ("source_origin_m", original.plane.origin_m)):
                auxiliary[prefix+name] = values
            for name in ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad"):
                value = getattr(mode.plane, name)
                if value is not None:
                    auxiliary[prefix+"exit_"+name] = value
        required = sum(a.nbytes for a in auxiliary.values())+estimated_bytes
        if required > numerics.maximum_checkpoint_bytes:
            raise ValueError(f"Retained near-tip state and gun buffers need {required} bytes, above maximum_checkpoint_bytes={numerics.maximum_checkpoint_bytes}")
    energy_maps, energy_records, outputs, records = {}, {}, [], []
    total = emission.record["mode_count"]
    for index, mode in enumerate(emission.modes() if near is None else near.modes):
        if cancelled():
            raise InterruptedError("Tip-to-exit wave propagation cancelled")
        if progress_callback is not None:
            progress_callback(index, total, "FEG tip through extraction, acceleration and gun optics")
        energy = mode.energy_kev*1000
        if energy not in energy_maps:
            energy_maps.clear()
            energy_maps[energy] = _energy_transport(working, z, masks, coefficients, energy, cancelled,
                                                    axial_b, electric_provider=electric)
        segments, transport = energy_maps[energy]
        energy_records[energy] = transport
        wave, losses = mode.plane, []
        for plane, matrix, offset, phase in segments:
            if cancelled():
                raise InterruptedError("Tip-to-exit wave propagation cancelled")
            if wave.probability > 0:
                wave = propagate_plane_wave(wave, matrix, offset, float(wavelength_m(energy)), **phase)
                wave, rows = _mask_plane(working, wave, plane, mode.weight_per_reference_electron,
                                         column_apertures)
                losses.extend(rows)
        norm = wave.probability
        # This is a unit conversion of phase carriers, preserving their phase
        # in radians and all previously accumulated diffraction/aperture losses.
        ratio = transport["momentum_ratio_tip_to_exit"]
        wave = replace(wave, amplitude=wave.amplitude/math.sqrt(norm) if norm else wave.amplitude,
                       curvature_m1=None if wave.curvature_m1 is None else wave.curvature_m1*ratio,
                       tilt_rad=None if wave.tilt_rad is None else wave.tilt_rad*ratio)
        outputs.append(replace(mode, plane=wave, weight_per_reference_electron=mode.weight_per_reference_electron*norm,
            energy_kev=transport["exit_axial_energy_ev"]*1e-3,
            axial_reference=mode.axial_reference.advance(transport["reference_flight_time_s"], transport["reference_longitudinal_action_j_s"])))
        records.append({"mode_id": mode.mode_id, "input_weight": mode.weight_per_reference_electron,
                        "output_weight": mode.weight_per_reference_electron*norm, "losses": losses})
    checkpoint = TipGunCheckpoint(BeamState(tuple(outputs), TIP_REFERENCE), float(z[-1]),
        emission.reference_current_a, {"schema": "executed-tip-gun-v1", "dependency_digest": dependency,
        "tip_emission_id": emission.digest, "tip_emission": emission.record,
        "field_digest": field_digest.hexdigest(), "field_steps": len(z)-1,
        "shared_column": column_record,
        "implementation": implementation,
        "numerics": asdict(numerics), "source_numerics": asdict(source_numerics),
        "physical_components": [component.key for component in working.components],
        "energy_transport": list(energy_records.values()), "mode_records": records,
        "accelerating_tip": None if near is None else {"digest": near.digest, "plane_z_mm": near.plane_z_mm,
            "record": near.record, "retained_auxiliary_arrays": sorted(auxiliary)},
        "physics": ("Two-way scalar driven Tip boundary followed by an executed second-order paraxial continuation in the installed fields"
                    if driven else "Stationary scalar second-order paraxial Hamiltonian in the installed solved gun electric field and captured magnetic components"),
        "integrator": "Energy-refined symmetric potential kick / variable-momentum drift / potential kick",
        "current_reference": "Physical tip emission before all gun losses",
        "limitations": ["After the bounded non-paraxial prefix, higher-order/non-paraxial gun dynamics are not included" if driven
                         else "Higher-order/non-paraxial gun dynamics not included in this operator",
                        "Electric potential is expanded to second transverse order about the optical axis",
                        "Imported Wien fields require a separate validated operator",
                        "Body bores use axial absorbing projections; refine bore_step_mm",
                        "Reference flight time is axial, not a pulsed longitudinal wave packet"],
        "validation_status": "DEVELOPMENT_NOT_FULL_TEM_STEM_ACCEPTANCE"}, auxiliary)
    if cancelled():
        raise InterruptedError("Tip-to-exit wave propagation cancelled; result not cached")
    if solver_source_identity() != implementation:
        raise RuntimeError("Solver source changed during gun wave transport; result not published or cached")
    if use_cache:
        _CACHE[dependency] = checkpoint
        _CACHE.move_to_end(dependency)
        while len(_CACHE) > 4 or sum(sum(m.plane.amplitude.nbytes for m in v.beam.modes)
                +sum(a.nbytes for a in v.auxiliary_arrays.values()) for v in _CACHE.values()) > numerics.maximum_checkpoint_bytes:
            _CACHE.popitem(last=False)
    if progress_callback is not None:
        progress_callback(total, total, "FEG coherent gun checkpoint computed")
    return checkpoint
