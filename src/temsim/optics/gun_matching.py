"""Small-signal proposals from the executed electrostatic gun field.

Not a source, beam cache or acceptance calculation. This variational model
linearises about the axial electron from the tip apex, at a declared kinetic
energy. It cannot represent the finite curved cap or its large local angles.
Every candidate needs ordinary tip-origin, non-paraxial particle validation.
"""
from __future__ import annotations

import numpy as np
from scipy.constants import m_e, c, e


def input_working_point(state, *, label):
    """Package this complete design for explicit GUI restore, without ray caches.

    A diagnostic execution flag is not an operating setting. Clear it on a
    detached full graph; all source, component and numerical inputs are retained.
    This package is not an executed downstream source or a simulation result.
    """
    from temsim.instrument_snapshot import capture_instrument_snapshot, encode_instrument, decode_instrument
    from temsim.working_point import WorkingPointCheckpoint
    if state.electron_gun.source_representation != "classical_particles":
        raise ValueError("Gun design input export requires classical tip particles")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("Gun design inputs need a descriptive label")
    # Components can refer back to their owning State. A shallow copy would
    # leave an old owner (and its diagnostic callbacks) reachable in the graph.
    detached = decode_instrument(encode_instrument(state))
    for key in ("_optical_tuning", "_tuning_quality", "_tuning_kernel", "_tuning_cancelled"):
        vars(detached).pop(key, None)
    snapshot = capture_instrument_snapshot(detached)
    return WorkingPointCheckpoint(snapshot, {}, float(state.sample.upper_surface_z_mm),
        "inputs-only:"+snapshot.digest,
        {"label": label, "package_kind": "INSTRUMENT_INPUTS_ONLY",
         "source_representation": "classical-tip-particle-inputs (no cached rays)",
         "phase_status": "NOT_COMPUTED", "validation_status": "INPUTS_ONLY",
         "quality": "NOT_COMPUTED"})


def candidate_with_gun_geometry(state, *, extractor_center_mm, lens_center_mm,
                                accelerator_center_mm=None, dpa_center_mm=None):
    """Detached physical placement, including bores, layout and field identity.

    Keep source, body sizes, voltage references and all column parts. This
    does not save to TOML or qualify a design. Moving only a runtime electrode
    while retaining the old assembly's vacuum walls is not a valid candidate.
    """
    from dataclasses import replace
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.column.module_assembly import _freeze, _module_vacuum_segments, VacuumLinerSegment
    from temsim.column.state_layout import apply_physical_layout_to_state
    centres = [extractor_center_mm, lens_center_mm]
    if accelerator_center_mm is not None:
        centres.append(accelerator_center_mm)
    if dpa_center_mm is not None:
        centres.append(dpa_center_mm)
    if not np.isfinite(centres).all():
        raise ValueError("Gun electrode centres must be finite")
    candidate = capture_instrument_snapshot(state).restore()
    if (candidate.electron_gun.type_key != "cold_feg"
            or candidate.electron_gun.source_representation != "classical_particles"):
        raise ValueError("Gun placement matching requires classical FEG particles")
    assembly = candidate._resolved_assembly
    old_tip = assembly.part("feg_tip")
    module = next(m for m in assembly.modules if m.key == old_tip.module_key)
    origin = old_tip.start_z_mm-float(old_tip.data["local_start_z_mm"])
    changed = {}
    placements = {"feg_extractor": extractor_center_mm,
                  "feg_electrostatic_lens": lens_center_mm}
    if accelerator_center_mm is not None:
        placements["feg_accelerator"] = accelerator_center_mm
        shift = float(accelerator_center_mm)+origin-assembly.part("feg_accelerator").center_z_mm
        descendants = {"feg_accelerator"}
        while True:
            children = {p.key for p in assembly.parts if p.data.get("parent_key") in descendants}
            if children <= descendants:
                break
            descendants |= children
        for key in descendants-{"feg_accelerator"}:
            placements[key] = assembly.part(key).center_z_mm-origin+shift
    if dpa_center_mm is not None:
        placements["feg_dpa_aperture"] = float(dpa_center_mm)
    for key, centre in placements.items():
        part = assembly.part(key)
        shift = float(centre)-part.center_z_mm+origin
        data = dict(part.data)
        for name in ("local_start_z_mm", "local_center_z_mm", "local_end_z_mm", "optical_reference_local_z_mm"):
            if name in data:
                data[name] = float(data[name])+shift
        if "stage_centers_z_mm" in data:
            data["stage_centers_z_mm"] = [float(z)+shift for z in data["stage_centers_z_mm"]]
        changed[key] = replace(part, start_z_mm=part.start_z_mm+shift,
            center_z_mm=part.center_z_mm+shift, end_z_mm=part.end_z_mm+shift, data=_freeze(data))
    ext, lens = changed["feg_extractor"], changed["feg_electrostatic_lens"]
    accelerator = changed.get("feg_accelerator", assembly.part("feg_accelerator"))
    dpa = changed.get("feg_dpa_aperture", assembly.part("feg_dpa_aperture"))
    if not (origin < ext.start_z_mm < ext.end_z_mm < lens.start_z_mm
            < lens.end_z_mm < accelerator.start_z_mm
            < accelerator.end_z_mm < assembly.part("feg_deflector").start_z_mm):
        raise ValueError("Retain tip/extractor/gun-lens/accelerator order and separated bodies")
    if not accelerator.start_z_mm < dpa.start_z_mm < dpa.end_z_mm < accelerator.end_z_mm:
        raise ValueError("Gun DPA must remain inside its parent accelerator")
    # A separately moved aperture cannot intersect a voltage-carrying ring.
    if dpa_center_mm is not None:
        half = .5*float(accelerator.data.get("electrode_thickness_mm", 1.))
        for z in accelerator.data["stage_centers_z_mm"]:
            if dpa.start_z_mm < origin+float(z)+half and dpa.end_z_mm > origin+float(z)-half:
                raise ValueError("Gun DPA body overlaps an accelerator electrode")
    parts = tuple(changed.get(p.key, p) for p in assembly.parts)
    local_parts = tuple(replace(p, start_z_mm=changed[p.key].start_z_mm-origin,
        center_z_mm=changed[p.key].center_z_mm-origin, end_z_mm=changed[p.key].end_z_mm-origin,
        data=changed[p.key].data) if p.key in changed else p for p in module.parts)
    module = replace(module, parts=local_parts)
    segments = tuple(_module_vacuum_segments(module, origin, parts))
    old_segments = _module_vacuum_segments(next(m for m in assembly.modules if m.key == module.key),
                                          origin, assembly.parts)
    old_keys = {s.key for s in old_segments}
    wall = float(module.geometry["vacuum_liner_wall_thickness_mm"])
    liners = tuple(VacuumLinerSegment("@vacuum_liner:"+s.key, s.name+" vacuum liner",
        s.start_z_mm, s.end_z_mm, s.inner_diameter_mm, s.inner_diameter_mm+2*wall, wall)
        for s in segments)
    assembly = replace(assembly, parts=parts,
        modules=tuple(module if m.key == module.key else m for m in assembly.modules),
        vacuum_bore_segments=tuple(sorted(
            tuple(s for s in assembly.vacuum_bore_segments if s.key not in old_keys)+segments,
            key=lambda s: s.start_z_mm)),
        vacuum_liner_segments=tuple(sorted(
            tuple(s for s in assembly.vacuum_liner_segments if s.key not in {"@vacuum_liner:"+k for k in old_keys})+liners,
            key=lambda s: s.start_z_mm)))
    apply_physical_layout_to_state(candidate, assembly=assembly, preserve_operating_parameters=True)
    return candidate


def _integrate(coefficients, widths):
    matrix = np.eye(2)
    for j, h in enumerate(widths):
        a, b, d = coefficients[2*j:2*j+3]
        def rhs(v, pair):
            return np.stack((pair[0]*v[1], pair[1]*v[0]))
        k1 = rhs(matrix, a)
        k2 = rhs(matrix+.5*h*k1, b)
        k3 = rhs(matrix+.5*h*k2, b)
        k4 = rhs(matrix+h*k3, d)
        matrix = matrix+h*(k1+2*k2+2*k3+k4)/6
    return matrix


def axis_variational_map(field, *, emission_energy_ev, scale_m, intervals=4000, end_m=None):
    """Map (x/scale, px/p_tip) from Z=0 to the gun exit, dimensionless.

    SI equations with q=-e: dx/dz=px/pz and dpx/dz=-e Er/vz.
    E_r=k(z)*x gives a traceless canonical generator. The exact determinant
    is one, even during acceleration; this is an independent numerical check.
    Only the already-executed scalar field supplies acceleration and focusing.
    No aperture transmission, finite-source size or nonlinear spot is inferred.
    """
    if (not np.isfinite([emission_energy_ev, scale_m]).all()
            or emission_energy_ev <= 0 or scale_m <= 0
            or type(intervals) is not int or intervals < 100):
        raise ValueError("Positive tip energy, scale and at least 100 intervals required")
    end = float(field.z[-1] if end_m is None else end_m)
    if not np.isfinite(end) or not 0 < end <= field.z[-1]:
        raise ValueError("Variational endpoint must be inside the executed field")
    # Logarithmic independent coordinate resolves the full tip-to-exit range.
    final_t = np.log1p(end/scale_m)
    t = np.unique(np.r_[np.linspace(0., final_t, intervals+1),
                        np.log1p(field.z[(field.z >= 0) & (field.z <= end)]/scale_m)])
    sample_t = np.empty(2*len(t)-1)
    sample_t[::2] = t
    sample_t[1::2] = .5*(t[:-1]+t[1:])
    z = np.minimum(scale_m*np.expm1(sample_t), end)
    # Axis coefficient limit at a radius far inside the first vacuum cell.
    radius = min(float(field.r[1])*1e-3, scale_m*1e-5)
    positions = np.column_stack((np.full_like(z, radius), z*0, z))
    _, electric = field._interpolate(positions)
    positions[:, 0] = 0.
    potential, _ = field._interpolate(positions)
    energy = emission_energy_ev+potential
    if np.any(energy <= 0):
        raise ValueError("Axial reference electron cannot traverse this potential")
    rest = m_e*c*c/e
    momentum_c_ev = np.sqrt(energy*(energy+2*rest))
    start_momentum_c_ev = np.sqrt(emission_energy_ev*(emission_energy_ev+2*rest))
    velocity_over_c = momentum_c_ev/(energy+rest)
    dz_dt = scale_m*np.exp(sample_t)
    coefficients = np.column_stack((
        start_momentum_c_ev/momentum_c_ev/scale_m*dz_dt,
        -(electric[:, 0]/radius)*scale_m/(velocity_over_c*start_momentum_c_ev)*dz_dt))
    matrix = _integrate(coefficients, np.diff(t))
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite gun variational proposal")
    return matrix, dict(determinant=float(np.linalg.det(matrix)), intervals=len(t)-1,
        end_m=end,
        emission_energy_ev=float(emission_energy_ev), scale_m=float(scale_m),
        exit_momentum_ratio=float(momentum_c_ev[-1]/start_momentum_c_ev),
        scope="AXIAL_VARIATIONAL_PROPOSAL_ONLY; not finite-source or aperture acceptance")
