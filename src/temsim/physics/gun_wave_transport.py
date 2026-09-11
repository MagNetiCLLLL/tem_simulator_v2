"""Historical exit-wave mathematics through the linear column field graph.

Production entry points reject the retired exit source. The separate development
tip_gun_wave operator is not yet connected here; this historical path stays closed.

The first supported domain is quadratic paraxial optics (including continuous
round-lens rotation, quadrupoles and physical deflector kicks). Nonlinear or
3-D mapped fields are explicitly rejected until their phase operators have
independent validation. No fitted specimen pupil is created here.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
from scipy.linalg import expm

from temsim.immutable_json import freeze_json, json_digest
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.electron_gun.effective_source import generate_gun_emission, wavelength_m
from temsim.physics.canonical_phase import validate_canonical_map
from temsim.physics.canonical_action import CanonicalPath
from temsim.physics.multiplane_wave import PlaneWave, propagate_plane_wave
from temsim.physics.wave_flux import BeamState, WaveMode


def _generator(g, kx, ky):
    # Exactly the quadratic part of ray_integrator.canonical_rk4_step.
    rotation = np.array(((0., g), (-g, 0.)))
    return np.block([[rotation, np.eye(2)], [-np.diag((kx+g*g, ky+g*g)), rotation]])


def _gauss_values(start, midpoint, end):
    offset = math.sqrt(3)/6
    def at(t):
        return 2*(t-.5)*(t-1)*start + 4*t*(1-t)*midpoint + 2*t*(t-.5)*end
    return at(.5-offset), at(.5+offset)


def canonical_magnus_step(h_m, g_nodes, kx_nodes, ky_nodes):
    """Fourth-order Magnus integrator; Hamiltonian exponent, no matrix repair."""
    gs, xs, ys = (_gauss_values(*v) for v in (g_nodes, kx_nodes, ky_nodes))
    a, b = _generator(gs[0], xs[0], ys[0]), _generator(gs[1], xs[1], ys[1])
    omega = h_m*(a+b)/2 - math.sqrt(3)*h_m*h_m*(a@b-b@a)/12
    matrix = expm(omega)
    validate_canonical_map(matrix)
    return matrix


def specimen_entrance_z_mm(state):
    """Declared upper face of the finite, lab-Z specimen envelope.

    Crystal orientation rotates the lattice within that envelope. Retraction
    does not move its reference plane. There is no arbitrary wave-input Z.
    """
    thickness = float(state.sample.thickness_nm)
    centre = float(state.sample.z_mm)
    if not math.isfinite(thickness) or thickness < 0 or not math.isfinite(centre):
        raise ValueError("The specimen entrance needs a finite centre and non-negative thickness")
    return centre-thickness*0.5e-6


def upstream_component_events(state):
    """Read actual hardware kicks. The specimen is not a free scan input."""
    events, owners = [], []
    stop = specimen_entrance_z_mm(state)
    seen = set()
    for component in (*state.deflectors, *getattr(state, "corrector_elements", ())):
        if not getattr(component, "enabled", False):
            continue
        if component.key in seen:
            raise ValueError(f"Duplicate deflector in the source graph: {component.key}")
        seen.add(component.key)
        if hasattr(component, "kick_events"):
            try:
                rows = component.kick_events(time_s=float(getattr(state, "simulation_time_s", 0.)))
            except TypeError:
                rows = component.kick_events()
        elif all(hasattr(component, name) for name in ("upper_z_mm", "lower_z_mm", "upper_x_mrad", "upper_y_mrad", "lower_x_mrad", "lower_y_mrad")):
            rows = [(component.upper_z_mm, component.upper_x_mrad*1e-3, component.upper_y_mrad*1e-3),
                    (component.lower_z_mm, component.lower_x_mrad*1e-3, component.lower_y_mrad*1e-3)]
        else:
            # Corrector field terms live in the shared field plan, not kicks.
            continue
        for z, tx, ty in rows:
            if state.electron_gun.exit_plane_z_mm <= z <= stop:
                events.append((z, tx, ty))
                owners.append({"component_id": component.key, "plane_z_mm": z, "kick_rad": (tx, ty)})
    return events, owners


@dataclass(frozen=True)
class GunWaveCheckpoint:
    snapshot: object
    emission: object
    beam: BeamState
    plane_z_mm: float
    execution: object

    def __post_init__(self):
        from temsim.instrument_snapshot import InstrumentSnapshot
        from temsim.optics.electron_gun.effective_source import GunEmissionState, REFERENCE_ID
        if not isinstance(self.snapshot, InstrumentSnapshot) or not isinstance(self.emission, GunEmissionState):
            raise TypeError("A gun-wave checkpoint needs its captured instrument and generated gun emission")
        if not isinstance(self.beam, BeamState) or self.beam.reference_plane != REFERENCE_ID:
            raise ValueError("A gun-wave checkpoint needs the gun-exit electron reference")
        if not math.isfinite(self.plane_z_mm) or self.plane_z_mm <= self.emission.plane_z_mm:
            raise ValueError("A gun-wave checkpoint must be downstream of its source")
        if self.execution.get("source_id") != self.emission.digest or self.execution.get("source_snapshot_id") != self.snapshot.digest:
            raise ValueError("Gun-wave execution and input identities do not match")
        if self.beam.total_weight > 1+1e-10:
            raise ValueError("Gun-wave flux exceeds its exit source reference")
        object.__setattr__(self, "execution", freeze_json(self.execution))

    @property
    def digest(self):
        # Payload bytes, not just a source label, enter checkpoint identity.
        from hashlib import sha256
        payload = sha256()
        for mode in self.beam.modes:
            for field in ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad"):
                value = getattr(mode.plane, field)
                payload.update(b"none" if value is None else np.ascontiguousarray(value).tobytes())
            payload.update(json_digest((mode.mode_id, mode.energy_kev, mode.weight_per_reference_electron)).encode())
        return json_digest({"snapshot": self.snapshot.digest, "emission": self.emission.digest,
                            "plane_z_mm": self.plane_z_mm, "execution": self.execution, "payload": payload.hexdigest()})


@dataclass(frozen=True)
class _GunWavePlan:
    snapshot: object
    working: object
    emission: object
    plan: object
    apertures: tuple
    stops: object
    wall_radii: np.ndarray
    endpoints: frozenset
    owners: tuple

    @property
    def stage_signature(self):
        """Exact consumed numerical inputs, not a mechanical-position guess.

        Field samples come from all installed/shared providers, including
        downstream lens tails. Unsupported maps/nonlinearity were rejected
        during preparation. All non-specimen external model bytes are pinned.
        Specimen composition is not an input to this *upper-face* operator.
        """
        from hashlib import sha256
        from temsim.instrument_snapshot import encode_instrument
        from temsim.simulation_modes import mode_key
        digest = sha256()
        for name in ("z_mm", "step_m", "magnetic_t", "sx_m2", "sy_m2",
                     "midpoint_magnetic_t", "midpoint_sx_m2", "midpoint_sy_m2",
                     "hex_normal_m3", "hex_skew_m3", "midpoint_hex_normal_m3",
                     "midpoint_hex_skew_m3", "cs_kick_m3", "thin_power_m1",
                     "thin_rotation_rad", "kick_x_rad", "kick_y_rad"):
            array = np.ascontiguousarray(getattr(self.plan, name))
            digest.update(json_digest((name, array.dtype.str, array.shape)).encode())
            digest.update(array.tobytes())
        digest.update(np.ascontiguousarray(self.wall_radii).tobytes())
        return json_digest({"schema": "gun-upper-face-operator-dependencies-v1",
            "implementation": self.snapshot.implementation, "source": self.emission.digest,
            "voltage_kv": float(self.working.beam_voltage_kv), "model": mode_key(self.working),
            "sampled_operator": digest.hexdigest(), "deflectors": self.owners,
            "apertures": [encode_instrument(item) for item in self.apertures],
            "model_inputs": [row for row in self.snapshot.external_inputs
                             if not row["role"].startswith("specimen:")]})


def _prepare_gun_wave_plan(snapshot):
    """Read actual fields and boundaries without propagating any modes."""
    from temsim.physics.core import build_propagation_plan
    from temsim.physics.column_wall import _vacuum_segments, _expanded_profile_axis
    working = snapshot.restore()
    gun = working.electron_gun
    from temsim.optics.electron_gun.source_policy import require_tip_coherent_source
    require_tip_coherent_source(gun)
    if gun.source_representation != "effective_gaussian_schell":
        raise ValueError("No validated tip-to-exit coherent producer is available")
    emission = generate_gun_emission(gun)
    start, stop = emission.plane_z_mm, specimen_entrance_z_mm(working)
    if stop <= start:
        raise ValueError("The specimen entrance must be after the installed gun exit")
    physical_apertures = list(working.apertures)
    nanopulser = getattr(working, "nanopulser", None)
    if nanopulser is not None and nanopulser.installed:
        nanopulser.validate()
        physical_apertures.append(nanopulser.aperture)
    apertures = [aperture for aperture in physical_apertures
        if getattr(aperture, "enabled", True) and getattr(aperture, "installed", True)
        and getattr(aperture, "inserted", True) and start < aperture.z_mm <= stop]
    if len({a.key for a in apertures}) != len(apertures):
        raise ValueError("Duplicate physical aperture identity in gun propagation graph")
    events, owners = upstream_component_events(working)
    if nanopulser is not None and nanopulser.installed:
        # build_propagation_plan adds these actions itself, exactly once.
        owners.extend({"component_id": "nanopulser", "plane_z_mm": z, "kick_rad": (tx, ty)}
                      for z, tx, ty in nanopulser.kick_events(working.beam_voltage_kv) if start <= z <= stop)
    walls = _vacuum_segments(working, np.array((start, stop)))
    wall_boundaries = [z for wall in walls for z in (wall.start_z_mm, wall.end_z_mm) if start < z < stop]
    plan = build_propagation_plan(working, start, stop, events=events,
                                 save_z_mm=[a.z_mm for a in apertures]+wall_boundaries)
    if plan.mapped_fields:
        raise ValueError("3-D imported-field coherent operators are not validated; no pupil fallback")
    if any(np.any(getattr(plan, field)) for field in ("hex_normal_m3", "hex_skew_m3", "cs_kick_m3")):
        raise ValueError("Nonlinear column phase operators are not validated in this source-chain version")
    if any(np.any(getattr(plan, field)) for field in ("thin_power_m1", "thin_rotation_rad")):
        raise ValueError("Equivalent thin-lens actions are not admitted in the distributed gun phase chain")
    # The bound source already describes the gun exit, including its own exit
    # aperture. Re-executing that aperture would count its calibration twice.
    stops = {}
    for aperture in apertures:
        index = int(np.argmin(abs(plan.z_mm-aperture.z_mm)))
        if abs(plan.z_mm[index]-aperture.z_mm) > 1e-9:
            raise ValueError("A physical aperture is missing from the exact field grid")
        stops.setdefault(index, []).append(aperture)
    wall_axis, _, wall_radii = _expanded_profile_axis(plan.z_mm, walls)
    if not np.array_equal(wall_axis, plan.z_mm):
        raise ValueError("A vacuum-bore boundary is missing from the exact field grid")
    endpoints = set(stops) | {len(plan.z_mm)-1}
    endpoints.update(np.flatnonzero(np.isfinite(wall_radii)).tolist())
    endpoints.discard(0)  # The calibrated exit boundary is already the source.
    return _GunWavePlan(snapshot, working, emission, plan, tuple(apertures), stops,
                        wall_radii, frozenset(endpoints), tuple(owners))


def build_gun_wave_checkpoint(state, *, cancelled=lambda: False, progress_callback=None):
    """Execute from gun inputs; callers cannot supply an intermediate wave."""
    if cancelled():
        raise InterruptedError("Gun phase propagation cancelled; no partial checkpoint published")
    prepared = _prepare_gun_wave_plan(capture_instrument_snapshot(state))
    return _execute_gun_wave_plan(prepared, cancelled=cancelled, progress_callback=progress_callback)


def _execute_gun_wave_plan(prepared, *, cancelled=lambda: False, progress_callback=None):
    from temsim.optics.electron_gun.source_policy import require_tip_coherent_source
    require_tip_coherent_source(prepared.working.electron_gun)
    from temsim.physics.core import electron
    from temsim.simulation_modes import is_ideal
    snapshot, working, emission, plan = prepared.snapshot, prepared.working, prepared.emission, prepared.plan
    stops, wall_radii, endpoints, owners = prepared.stops, prepared.wall_radii, prepared.endpoints, prepared.owners
    stop = float(plan.z_mm[-1])
    if cancelled():
        raise InterruptedError("Gun phase propagation cancelled; no partial checkpoint published")
    nominal_momentum = electron(working)[1]
    fields = (plan.magnetic_t, plan.sx_m2, plan.sy_m2)
    midpoints = (plan.midpoint_magnetic_t, plan.midpoint_sx_m2, plan.midpoint_sy_m2)
    modes, records, transport_cache, phase_records = [], [], {}, []
    energy_count = 1 if is_ideal(working) else len(emission.energies_ev)
    work_total = energy_count*len(plan.step_m) + emission.record["mode_count"]*len(endpoints)
    work_done = 0
    def progress(label):
        if progress_callback is not None:
            progress_callback(work_done, work_total, label)
    progress("Gun coherent transport: field maps")
    for mode_index, mode in enumerate(emission.modes()):
        if cancelled():
            raise InterruptedError("Gun phase propagation cancelled; no partial checkpoint published")
        # Ideal Optics explicitly neglects chromatic lens focusing, as does the
        # common particle integrator. The wave still retains its actual energy.
        energy = float(working.beam_voltage_kv) if is_ideal(working) else mode.energy_kev
        if energy not in transport_cache:
            from types import SimpleNamespace
            q, momentum, _ = electron(SimpleNamespace(beam_voltage_kv=energy))
            gscale = q/(2*momentum)
            # A fixed numerical reference chart, not a replacement source.
            # Follow the lift at every actual field step, including caustics;
            # record the scalar Weyl action when physical kicks are grouped.
            path, segments = CanonicalPath(1e-3), []
            for i, h in enumerate(plan.step_m):
                if i % 256 == 0 and cancelled():
                    raise InterruptedError("Gun phase propagation cancelled; no partial checkpoint published")
                nodes = [(v[i], m[i], v[i+1]) for v, m in zip(fields, midpoints)]
                step = canonical_magnus_step(h, np.array(nodes[0])*gscale, nodes[1], nodes[2])
                work_done += 1
                if i % 256 == 0:
                    progress("Gun coherent transport: field maps")
                if i == 0:
                    path.append(np.eye(4), np.array((0., 0., plan.kick_x_rad[0], plan.kick_y_rad[0])))
                path.append(step)
                j = i+1
                if plan.kick_x_rad[j] != 0 or plan.kick_y_rad[j] != 0:
                    path.append(np.eye(4), np.array((0., 0., plan.kick_x_rad[j], plan.kick_y_rad[j])))
                if j in endpoints:
                    validate_canonical_map(path.matrix)
                    segments.append((j, path.matrix, path.offset, path.phase_kwargs()))
                    path = CanonicalPath(path.reference_length_m)
            transport_cache[energy] = segments
            phase_records.append({"optical_energy_kev": energy,
                "reference_length_m": path.reference_length_m, "segments": len(segments),
                "phase_path_digest": json_digest([(int(j), phase) for j, _, _, phase in segments])})
        wave = mode.plane
        rows = []
        wall_loss = 0.
        for j, matrix, offset, phase in transport_cache[energy]:
            if cancelled():
                raise InterruptedError("Gun phase propagation cancelled; no partial checkpoint published")
            wave = propagate_plane_wave(wave, matrix, offset, wavelength_m(mode.energy_kev*1000), **phase)
            if math.isfinite(wall_radii[j]):
                xy = wave.coordinates_m()
                inside = np.hypot(xy[0], xy[1]) < wall_radii[j]*1e-3
                if not np.all(inside):
                    before = wave.probability
                    wave = replace(wave, amplitude=np.where(inside, wave.amplitude, 0j))
                    wall_loss += before-wave.probability
            for aperture in stops.get(j, ()):
                xy = wave.coordinates_m()
                # Execute the installed aperture's own geometry in its native
                # mm units. Do not replace a non-circular stop with a disk.
                x_mm, y_mm = xy[0]*1e3, xy[1]*1e3
                if float(getattr(aperture, "radius_mm", 1.)) <= 0:
                    mask = np.zeros_like(x_mm, dtype=bool)
                elif hasattr(aperture, "transmission_mask"):
                    mask = np.asarray(aperture.transmission_mask(x_mm, y_mm), dtype=bool)
                else:
                    mask = np.hypot(x_mm-aperture.offset_x_mm, y_mm-aperture.offset_y_mm) <= aperture.radius_mm
                if mask.shape != wave.amplitude.shape:
                    raise ValueError("Physical aperture returned a mask with the wrong shape")
                before = wave.probability
                wave = replace(wave, amplitude=np.where(mask, wave.amplitude, 0j))
                rows.append({"physical_element_id": "aperture:"+aperture.key, "plane_z_mm": aperture.z_mm,
                             "incoming_probability": mode.weight_per_reference_electron*before,
                             "outgoing_probability": mode.weight_per_reference_electron*wave.probability,
                             "strategy": "physical_plane_mask"})
            work_done += 1
            if work_done % 128 == 0 or work_done == work_total:
                progress(f"Gun coherent transport: mode {mode_index+1}/{emission.record['mode_count']}")
        probability = wave.probability
        # Factor norm into a mode weight, never renormalise the beam current.
        plane = replace(wave, amplitude=wave.amplitude/math.sqrt(probability)) if probability > 0 else wave
        modes.append(WaveMode(plane, mode.weight_per_reference_electron*probability,
                              mode.reference_plane, mode.mode_id, mode.energy_kev))
        records.append({"mode_id": mode.mode_id, "energy_kev": mode.energy_kev,
                        "input_weight": mode.weight_per_reference_electron,
                        "output_weight": mode.weight_per_reference_electron*probability, "apertures": rows,
                        "distributed_wall_loss": mode.weight_per_reference_electron*wall_loss})
    progress("Gun coherent transport complete")
    return GunWaveCheckpoint(snapshot, emission, BeamState(tuple(modes), emission.record["reference_id"]), stop,
        {"schema": "gun-quadratic-field-graph-v3", "field_plan_signature": plan.signature,
         "stage_signature": prepared.stage_signature,
         "output_boundary": "finite-specimen-upper-face",
         "specimen_centre_z_mm": float(working.sample.z_mm),
         "specimen_thickness_nm": float(working.sample.thickness_nm),
         "source_id": emission.digest, "source_snapshot_id": snapshot.digest,
         "reference_momentum_kg_m_s": nominal_momentum, "coordinate_basis": "laboratory-normalized-canonical",
         "integrator": "fourth-order Hamiltonian Magnus / affine LCT", "step_mm": working.step_mm,
         "phase_convention": "Continuous reference-Gaussian metaplectic lift + Weyl scalar action",
         "phase_paths": phase_records,
         "deflector_actions": owners, "mode_records": records,
         "physical_scope": "Bound effective gun + quadratic distributed fields + apertures + sampled absorbing bore",
         "column_wall_model": "Absorbing projection at every finite-bore field node; refine axial sampling independently",
         "validation_status": "NOT_VALIDATED"})
