"""Executed tip-to-exit wave transport in the installed analytic FEG fields.

This is a scalar, stationary, second-order paraxial Hamiltonian solver. It
retains variable longitudinal momentum, electrostatic focusing, both gun
deflector coils, the rotated magnetic stigmator, the analytic Wien electric
and magnetic fields, DPA/C1/slit masks and sampled absorbing body bores.
It is not yet admission for full TEM/STEM or for arbitrary imported fields.

Canonical coordinates use the constant *launch* momentum p_ref, so the map
remains symplectic during acceleration. At the exit only the phase-carrier
units change to p_exit; no new amplitude, source current or source is fitted.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import math

import numpy as np
from scipy.constants import c, e, h, m_e

from temsim.immutable_json import freeze_json, json_digest
from temsim.optics.electron_gun.tip_coherence import (
    TIP_REFERENCE, TipWaveNumerics, generate_tip_emission, wavelength_m,
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

    def validate(self):
        for name in ("field_step_mm", "bore_step_mm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Gun wave {name} must be positive and finite")
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

    @property
    def digest(self):
        if hasattr(self.beam, "content_identity"):
            return json_digest({"record": self.record, "stored_payload": self.beam.content_identity,
                                "plane_z_mm": self.plane_z_mm, "current_a": self.reference_current_a})
        digest = sha256()
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


def _field_coefficients(gun, z_mm):
    """Read the installed analytic providers, including the Wien fringe model.

    E_perp and B_perp are affine functions of x,y in these providers; their
    derivatives are exact divided differences. Arbitrary providers fail closed
    rather than being silently linearised or substituted with a matched Wien.
    """
    from temsim.optics.electron_gun.field_emission import FieldEmissionGun
    from temsim.optics.electron_gun.monochromator import AnalyticWienField
    if type(gun) is not FieldEmissionGun or gun.type_key != "cold_feg":
        raise ValueError("Tip wave gun transport currently requires the physical analytic FEG assembly")
    if gun.monochromator_installed and type(gun.monochromator.field_provider) is not AnalyticWienField:
        raise ValueError("Imported Wien fields need a validated non-quadratic wave operator; the installed provider cannot be skipped")
    z = np.asarray(z_mm, dtype=float)
    positions = np.zeros((len(z), 3))
    positions[:, 2] = z*1e-3
    electric, magnetic = gun.electric_field, gun.magnetic_field
    phi = electric.potential_v_at_global_positions(positions)
    electric0 = electric.field_at_global_positions_v_per_m(positions)
    magnetic0 = magnetic.field_at_global_positions_t(positions)
    ej, bj = np.empty((len(z), 2, 2)), np.empty((len(z), 2, 2))
    delta = 1e-4
    for axis in range(2):
        plus, minus = positions.copy(), positions.copy()
        plus[:, axis], minus[:, axis] = delta, -delta
        ej[:, :, axis] = (electric.field_at_global_positions_v_per_m(plus)[:, :2]
                         - electric.field_at_global_positions_v_per_m(minus)[:, :2])/(2*delta)
        bj[:, :, axis] = (magnetic.field_at_global_positions_t(plus)[:, :2]
                         - magnetic.field_at_global_positions_t(minus)[:, :2])/(2*delta)
    if np.any(magnetic0[:, 2] != 0):
        raise ValueError("An axial gun magnetic field requires its vector-potential rotation operator")
    # phi gradient/Hessian and grad(A_z)=(-B_y,B_x) for q=-e.
    gradient = -electric0[:, :2]
    hessian = -ej
    magnetic_force = np.stack((magnetic0[:, 1], -magnetic0[:, 0]), axis=1)
    magnetic_hessian = np.stack((bj[:, 1], -bj[:, 0]), axis=1)
    for array in (hessian, magnetic_hessian):
        if not np.allclose(array, array.transpose(0, 2, 1), rtol=1e-12, atol=1e-12):
            raise ValueError("Installed gun fields do not yield a symmetric canonical phase Hessian")
    return phi, gradient, hessian, magnetic_force, magnetic_hessian


def _axial_grid(gun, numerics):
    stop = float(gun.exit_plane_z_mm)
    if not math.isfinite(stop) or stop <= 0:
        raise ValueError("Physical gun exit must follow the tip")
    # Boundaries come from geometry and field providers, never a source plane.
    events = {0., stop, float(gun.dpa_aperture.z_mm), float(gun.c1_aperture.z_mm)}
    masks = set(events)
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
    return np.array(grid), frozenset(v for v in masks if 0 < v <= stop)


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


def _energy_transport(gun, z, mask_planes, coefficients, launch_energy, cancelled, axial_b_t=None):
    """Symmetric kick/drift/kick, retaining the continuous Weyl action and lift."""
    phi, gradient, hessian, magnetic_force, magnetic_hessian = coefficients
    launch_phi = float(gun.electric_field.potential_v_at_global_positions(np.zeros((1, 3)))[0])
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
    exit_energy = launch_energy+float(gun.electric_field.potential_v_at_global_positions(exit_position)[0])-launch_phi
    p_exit, _ = _momentum_velocity(exit_energy)
    return segments, {
        "launch_energy_ev": launch_energy, "exit_axial_energy_ev": exit_energy,
        "reference_flight_time_s": float(flight_time), "momentum_ratio_tip_to_exit": float(p_ref/p_exit),
        "reference_longitudinal_action_j_s": longitudinal_action,
        "longitudinal_carrier_phase_rad": 2*np.pi*longitudinal_action/h,
        "canonical_map": overall.matrix.tolist(), "canonical_offset": overall.offset.tolist(),
        "weyl_action_m": overall.action_m, "metaplectic_reference_phase_rad": overall.reference_phase_rad,
        "phase_path_digest": json_digest([(plane, phase) for plane, _, _, phase in segments])}


def _mask_plane(gun, wave, z_mm, prior):
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
    return wave, rows


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
    emission = generate_tip_emission(working, source_numerics)
    estimated_bytes = (emission.record["mode_count"]+8)*source_numerics.grid_pixels**2*16
    if estimated_bytes > numerics.maximum_checkpoint_bytes:
        raise ValueError(f"Gun checkpoint and wave buffers need approximately {estimated_bytes} bytes, above maximum_checkpoint_bytes={numerics.maximum_checkpoint_bytes}")
    from temsim.physics.wave_execution import check_available_memory
    check_available_memory(estimated_bytes)
    z, masks = _axial_grid(working, numerics)
    coefficients = _field_coefficients(working, (z[1:]+z[:-1])*.5)
    column_record, axial_b = None, None
    if _column_state is not None:
        from temsim.physics.core import build_propagation_plan, fields
        column = decode_instrument(encode_instrument(_column_state))
        if json_digest(encode_instrument(column.electron_gun)) != json_digest(encode_instrument(working)):
            raise ValueError("Shared column and gun inputs do not describe the same installed gun")
        shared = build_propagation_plan(column, 0., float(z[-1]), save_z_mm=z)
        if shared.mapped_fields or any(np.any(getattr(shared, name)) for name in (
                "sx_m2", "sy_m2", "midpoint_sx_m2", "midpoint_sy_m2",
                "hex_normal_m3", "hex_skew_m3", "midpoint_hex_normal_m3", "midpoint_hex_skew_m3",
                "cs_kick_m3", "thin_power_m1", "thin_rotation_rad", "kick_x_rad", "kick_y_rad")):
            raise ValueError("Non-axial column components inside the accelerating gun require a joint operator; they cannot be skipped")
        # Sample on the gun's own grid so no post-hoc boundary phase is fitted.
        axial_b = fields((z[1:]+z[:-1])*.5, column)[0]
        if fields(np.array((0.,)), column)[0][0] != 0:
            raise ValueError("A magnetic field at the tip needs a magnetic emission boundary model")
        column_record = {"inputs": encode_instrument(column), "plan_signature": shared.signature,
                         "boundary_gauge": "A=(-Bz*y/2,Bz*x/2,0); canonical momentum continuous"}
    field_digest = sha256()
    for values in (z, *coefficients):
        field_digest.update(np.ascontiguousarray(values).tobytes())
    if axial_b is not None:
        field_digest.update(np.ascontiguousarray(axial_b).tobytes())
    implementation = solver_source_identity()
    dependency = json_digest({"gun": encode_instrument(working), "source": emission.digest,
        "numerics": asdict(numerics), "sampled_fields": field_digest.hexdigest(),
        "implementation": implementation, "shared_column": column_record, "schema": "executed-tip-gun-v1"})
    if use_cache and dependency in _CACHE:
        if cancelled():
            raise InterruptedError("Tip-to-exit wave propagation cancelled")
        _CACHE.move_to_end(dependency)
        return _CACHE[dependency]
    energy_maps, energy_records, outputs, records = {}, {}, [], []
    total = emission.record["mode_count"]
    for index, mode in enumerate(emission.modes()):
        if cancelled():
            raise InterruptedError("Tip-to-exit wave propagation cancelled")
        if progress_callback is not None:
            progress_callback(index, total, "FEG tip through extraction, acceleration and gun optics")
        energy = mode.energy_kev*1000
        if energy not in energy_maps:
            energy_maps.clear()
            energy_maps[energy] = _energy_transport(working, z, masks, coefficients, energy, cancelled, axial_b)
        segments, transport = energy_maps[energy]
        energy_records[energy] = transport
        wave, losses = mode.plane, []
        for plane, matrix, offset, phase in segments:
            if cancelled():
                raise InterruptedError("Tip-to-exit wave propagation cancelled")
            if wave.probability > 0:
                wave = propagate_plane_wave(wave, matrix, offset, float(wavelength_m(energy)), **phase)
                wave, rows = _mask_plane(working, wave, plane, mode.weight_per_reference_electron)
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
        "physics": "Stationary scalar quadratic paraxial Hamiltonian in installed analytic gun fields",
        "integrator": "Symmetric potential kick / variable-momentum drift / potential kick",
        "current_reference": "Physical tip emission before all gun losses",
        "limitations": ["Higher-order/non-paraxial gun dynamics not included in this operator",
                        "Imported Wien fields require a separate validated operator",
                        "Body bores use axial absorbing projections; refine bore_step_mm",
                        "Reference flight time is axial, not a pulsed longitudinal wave packet"],
        "validation_status": "DEVELOPMENT_NOT_FULL_TEM_STEM_ACCEPTANCE"})
    if cancelled():
        raise InterruptedError("Tip-to-exit wave propagation cancelled; result not cached")
    if solver_source_identity() != implementation:
        raise RuntimeError("Solver source changed during gun wave transport; result not published or cached")
    if use_cache:
        _CACHE[dependency] = checkpoint
        _CACHE.move_to_end(dependency)
        while len(_CACHE) > 4 or sum(sum(m.plane.amplitude.nbytes for m in v.beam.modes) for v in _CACHE.values()) > numerics.maximum_checkpoint_bytes:
            _CACHE.popitem(last=False)
    if progress_callback is not None:
        progress_callback(total, total, "FEG coherent gun checkpoint computed")
    return checkpoint
