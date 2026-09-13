"""Joint curved-tip reservoir / two-way grounded round-gun calculation.

There is no API to inject a gun-exit or specimen-plane source. The full
installed input is consumed to build the downstream load BEFORE solving the
physical tip boundary. This development scalar orbital operator is not a
calibrated commercial-gun model or general non-axisymmetric gun admission.
"""
from dataclasses import asdict, replace
import math

import numpy as np
from scipy.constants import hbar
from scipy.linalg import eigh
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from temsim.immutable_json import freeze_json, json_digest
from temsim.physics.radial_gun_wave import (REST_EV, RadialGunNumerics,
    prepare_round_gun, squared_wave_number, aperture_projection)
from temsim.physics.radial_mask_ledger import mask_loss_budget, physical_to_chart
from temsim.physics.wave_following_chart import chart_from_executed_wave
from temsim.physics.surface_wave import (SurfaceWaveNumerics,
    prepare_surface_problem, SurfaceWaveCheckpoint, SurfaceWaveMode, _immutable)


def validate_joint_radial_domain(gun, surface, radial):
    """Reject a missing interface region before the expensive gun execution."""
    from temsim.physics.radial_gun_wave import aperture_projection
    surface.validate(); radial.validate()
    model = gun.emitter.surface_model
    cap = model.geometry.apex_radius_nm*math.sin(math.radians(model.emission.cap_half_angle_deg))
    width = cap*radial.coordinate_width_over_cap_radius
    domain = aperture_projection(cap*surface.outer_radius_factor, width, radial.radial_modes, 0)
    error = float(np.linalg.norm(np.eye(radial.radial_modes)-domain, 2))
    if error > surface.flux_tolerance:
        raise ValueError(f"Joined radial basis extends outside the near-field face (defect {error:.6g}); "
                         "enlarge the numerical domain or refine the coordinate basis before executing the gun")
    return error


def solve_joint_boundary(problem, energy, load, frame, *, cancelled=lambda: False,
                         linear_tolerance=1e-9, flux_tolerance=1e-8, factor_radial_phase=True,
                         condensation_cache=None, assembly_cache=None, maximum_working_bytes=8*1024**3):
    """Galerkin continuity of the complex trace and normal derivative.

    The top trace is constrained to the retained radial basis. It is solved
    with the near field, not projected from an independently outgoing result.
    Reflection at every represented downstream operation returns through this
    interface. Converge the radial basis independently of the FEM mesh.
    """
    if cancelled():
        raise InterruptedError("Joint tip/gun boundary cancelled")
    count = load.input_admittance.shape[0]
    from temsim.physics.joint_boundary_assembly import prepare_joint_assembly
    assembled, assembly_hit = prepare_joint_assembly(problem, energy, count,
        frame["initial_width_nm"], frame["initial_curvature_per_nm"], load.kappa, factor_radial_phase,
        cache=assembly_cache, maximum_working_bytes=maximum_working_bytes, cancelled=cancelled)
    reduced, rhs, transform, source, side, incoming, phase, gamma = (assembled[name] for name in
        ("reduced", "rhs", "transform", "source", "side", "incoming", "phase", "gamma"))
    domain_error = assembled["domain_error"]
    if domain_error > flux_tolerance:
        raise ValueError(f"The radial gun basis extends beyond its joined near-field face (defect {domain_error:.6g}); "
                         "enlarge the numerical near domain or refine its coordinate basis; do not discard that channel")
    ids = len(rhs)-count+np.arange(count)
    boundary = coo_matrix((load.input_admittance.ravel(),
        (np.repeat(ids, count), np.tile(ids, count))), shape=reduced.shape).tocsc()
    linear_record = {"method": "direct-coupled-sparse-solve"}
    if condensation_cache is None:
        reduced -= boundary
        solved = splu(reduced).solve(rhs)
        residual = float(np.linalg.norm(reduced@solved-rhs)/np.linalg.norm(rhs))
    else:
        from temsim.physics.condensed_wave_boundary import solve_cached_boundary
        solved, linear_record = solve_cached_boundary(reduced, rhs, load.input_admittance, condensation_cache,
            tolerance=linear_tolerance, maximum_working_bytes=maximum_working_bytes-assembled["retained_bytes"], cancelled=cancelled)
        residual = linear_record["linear_residual"]
    linear_record = {**linear_record, "assembly_cache_hit": assembly_hit,
        "assembly_inputs_digest": assembled["matrix_inputs_digest"], "assembly_retained_bytes": assembled["retained_bytes"]}
    if cancelled():
        raise InterruptedError("Joint tip/gun boundary cancelled")
    if not np.all(np.isfinite(solved)) or residual > linear_tolerance:
        raise ValueError(f"Joint tip/gun linear residual {residual:.3g} exceeds {linear_tolerance:.3g}")
    psi = transform@solved
    coefficients = solved[-count:]
    reflected = psi-incoming
    outgoing = float(np.vdot(coefficients, load.input_admittance@coefficients).imag)
    flux = {"incoming": 1., "reflected": float(np.vdot(reflected, source@reflected).real),
            "side": float(np.vdot(psi, side@psi).real), "top": outgoing,
            "linear_residual": residual, "interface_gram_error": assembled["gram_error"],
            "interface_domain_error": domain_error,
            "radial_phase_gradient_nm2": gamma, "boundary_linear_solve": linear_record}
    flux["balance_error"] = abs(1-flux["reflected"]-flux["side"]-flux["top"])
    if flux["balance_error"] > flux_tolerance or min(flux[k] for k in ("reflected", "side", "top")) < -flux_tolerance:
        raise ValueError(f"Joint tip/gun flux balance failed: {flux}")
    return psi*np.exp(1j*phase), coefficients, flux


def build_surface_gun_checkpoint(gun, *, surface=SurfaceWaveNumerics(element_order=2),
        radial=RadialGunNumerics(), grid_pixels=256, column_state=None,
        cancelled=lambda: False, progress_callback=None, _mode_completed=None, _energy_cache=None,
        _reuse_energy_cache=True):
    """Execute a physical tip-origin gun and retain its jointly solved near field.

    Disabling reuse does not disable mandatory history writes to the supplied
    execution store. Direct diagnostics may instead consume _mode_completed.
    """
    if type(_reuse_energy_cache) is not bool:
        raise ValueError("Energy-cache reuse must be a boolean")
    from temsim.calculation_manifest import solver_source_identity
    from temsim.instrument_snapshot import encode_instrument
    from temsim.optics.electron_gun.source_policy import require_physical_gun_source
    from temsim.optics.electron_gun.tip_coherence import TIP_REFERENCE
    from temsim.physics.multiplane_wave import PlaneWave
    from temsim.physics.tip_gun_wave import TipGunCheckpoint
    from temsim.physics.wave_flux import BeamState, WaveMode
    from temsim.physics.wave_reference import AxialWaveReference
    from copy import deepcopy
    require_physical_gun_source(gun)
    working = deepcopy(gun)
    validate_joint_radial_domain(working, surface, radial)
    graph = encode_instrument(working)
    implementation = solver_source_identity()
    column_snapshot = None
    column_guard = None
    if column_state is not None:
        from temsim.instrument_snapshot import capture_instrument_snapshot
        if json_digest(encode_instrument(column_state.electron_gun)) != json_digest(graph):
            raise ValueError("The joint column and gun must consume the same physical tip and gun inputs")
        column_snapshot = capture_instrument_snapshot(column_state)
        column_state = column_snapshot.restore()
        from temsim.physics.surface_wave import guard_surface_column
        column_guard = guard_surface_column(column_state, surface)
    problem = prepare_surface_problem(working, surface, cancelled=cancelled)
    seed_energy = radial.occupied_refinement.initial_mesh_energy_ev
    if radial.occupied_refinement.initial_mesh:
        if seed_energy is None and len(problem["energies"]) > 1:
            raise ValueError("A multi-energy gun needs an explicit emission energy for its numerical mesh seed")
        if seed_energy is not None and seed_energy not in problem["energies"]:
            raise ValueError("The mesh-seed emission energy is not one of the current source energies")
    def verify_inputs():
        if cancelled():
            raise InterruptedError("Joint coherent gun cancelled before publication")
        if solver_source_identity() != implementation or json_digest(encode_instrument(gun)) != json_digest(graph):
            raise RuntimeError("Gun/source implementation changed during the joint wave calculation")
        if column_snapshot is not None:
            column_snapshot.restore()
    modes, near_modes, records, radial_payload, energy_histories = [], [], [], [], []
    for index, (energy, weight) in enumerate(zip(problem["energies"], problem["weights"])):
        identity = {"schema": "executed-surface-gun-mode-v1", "implementation": implementation,
            "gun_inputs": graph, "column_snapshot": None if column_snapshot is None else column_snapshot.digest,
            "surface_numerics": asdict(surface), "radial_numerics": asdict(radial),
            "grid_pixels": grid_pixels, "mode_index": index,
            "energy_ev": float(energy), "mixture_weight": float(weight)}
        identity = freeze_json({**identity, "dependency_digest": json_digest(identity)})
        if _energy_cache is not None:
            from temsim.physics.surface_mode_cache import mode_key
            energy_histories.append({"cache_key": mode_key(_energy_cache, identity),
                "executed_identity_digest": json_digest(identity), "mode_id": f"surface-energy:{index}",
                "energy_ev": float(energy), "source_mixture_weight": float(weight)})
        if _energy_cache is not None and _reuse_energy_cache:
            from temsim.physics.surface_mode_cache import restore_mode
            verify_inputs()
            completed = restore_mode(_energy_cache, identity)
            if completed is not None:
                mode, near_mode, record, payload, history = completed
                modes.append(mode); near_modes.append(near_mode)
                records.append(record); radial_payload.append(payload)
                if _mode_completed is not None:
                    _mode_completed(mode, near_mode, record, payload, history, identity)
                if progress_callback:
                    progress_callback(index+1, len(problem["energies"]),
                        "Reused complete input-bound tip/gun energy; full mixture still required")
                del completed, history
                continue
        if progress_callback:
            progress_callback(index, len(problem["energies"]), "Building coupled physical tip/gun mode")
        chart = None
        chart_iterations = []
        axial_records = []
        boundary_cache, assembly_cache = {}, {}
        for iteration in range(radial.wave_following.iterations+1):
            if progress_callback and radial.wave_following.iterations:
                progress_callback(iteration, radial.wave_following.iterations+1,
                    f"Re-solving full tip/gun mode {index+1}; numerical chart {iteration}")
            refined_chart = radial.occupied_refinement.applies_to_chart(iteration, radial.wave_following.iterations)
            refinement_plan = {} if refined_chart else None
            load, frame = prepare_round_gun(working, surface.exit_height_nm, float(energy), radial,
                column_state=column_state, executed_chart=chart,
                refinement_plan=refinement_plan,
                cancelled=cancelled, progress_callback=progress_callback)
            def boundary_solver(current_load):
                return solve_joint_boundary(problem, float(energy), current_load, frame,
                    cancelled=cancelled, linear_tolerance=surface.linear_tolerance, flux_tolerance=surface.flux_tolerance,
                    factor_radial_phase=surface.joint_radial_phase,
                    condensation_cache=boundary_cache, assembly_cache=assembly_cache,
                    maximum_working_bytes=surface.maximum_working_bytes)
            if refinement_plan is None:
                psi, coefficients, flux = boundary_solver(load)
            else:
                from temsim.physics.occupied_axial_refinement import refine_joint_mode
                del load
                planes = [frame["start_nm"], *[row["z_nm"] for row in frame["rows"]]]
                refinement_plan["interval_z_nm"] = {j: (planes[j], planes[j+1])
                    for j in refinement_plan["samplers"]}
                load, solution, axial_record = refine_joint_mode(refinement_plan, boundary_solver,
                    radial.occupied_refinement.for_energy(float(energy)),
                    cancelled=cancelled, progress_callback=progress_callback)
                psi, coefficients, flux = solution
                axial_records.append({"chart_iteration": iteration, **axial_record})
                del refinement_plan
            fields, derivatives = load.propagate(coefficients)
            if iteration < radial.wave_following.iterations:
                chart, entry = chart_from_executed_wave(fields, derivatives, frame, radial.wave_following)
                chart_iterations.append({**entry, "iteration": iteration, "near_flux": flux,
                    "exit_net_current": float(np.vdot(fields[-1], derivatives[-1]).imag),
                    "scope": "Numerical coordinate pilot only; never a published source or accepted current",
                    "occupied_axial_refined": refined_chart})
                # Release completed O(steps*N^2) operators BEFORE rebuilding.
                del load, fields, derivatives
        currents = np.imag(np.einsum("ij,ij->i", fields.conj(), derivatives))
        # The exact positive outgoing spectral form avoids cancellation of
        # nearly equal incident/reflected chart amplitudes at high energies.
        values, vectors = eigh(load.output_wave_number.real if np.max(abs(load.output_wave_number.imag)) < 1e-12
                               else load.output_wave_number, check_finite=False)
        if values.min() <= 0:
            raise ValueError("Gun exit includes non-propagating channels requiring retained evanescent output")
        flux_coefficients = (vectors*np.sqrt(values))@vectors.conj().T@fields[-1]
        exit_current = float(np.vdot(flux_coefficients, flux_coefficients).real)
        if exit_current > flux["top"]+surface.flux_tolerance or exit_current < 0:
            raise ValueError("The two-way gun created electron current")
        losses = []
        boundary_states = []
        mask_indices = {}
        for j, row in enumerate(frame["rows"]):
            if row["kind"] != "vacuum_fields":
                mask_indices.setdefault(row["component"], []).append(j)
        selected_boundaries = {0, len(fields)-1}
        selected_boundaries.update(j+1 for j, row in enumerate(frame["rows"]) if row.get("diagnostic_checkpoint"))
        for indices in mask_indices.values():
            selected_boundaries.update((indices[0], indices[0]+1, indices[-1], indices[-1]+1))
        for j in sorted(selected_boundaries):
            width_j, curvature_j = ((frame["initial_width_nm"], frame["initial_curvature_per_nm"])
                if j == 0 else frame["frames"][j-1])
            boundary_states.append({"boundary_index": j,
                "z_nm": surface.exit_height_nm if j == 0 else frame["rows"][j-1]["z_nm"],
                "width_nm": width_j, "curvature_per_nm": curvature_j,
                "quartic_phase_per_nm4": 0. if j == 0 else frame["quartic_phases_per_nm4"][j-1],
                "chart_derivatives": frame["chart_derivatives"][j],
                "reference_k_per_nm": load.kappa, "net_current_fraction": float(currents[j]),
                "basis_tail_l2_fraction": float(np.linalg.norm(fields[j, -min(8, radial.radial_modes):])**2
                    / max(np.linalg.norm(fields[j])**2, np.finfo(float).tiny)),
                "coefficients_real": fields[j].real.tolist(), "coefficients_imag": fields[j].imag.tolist(),
                "derivative_real": derivatives[j].real.tolist(), "derivative_imag": derivatives[j].imag.tolist(),
                "phase_scope": "Physical complex coefficients with axial carrier retained; not an independent source"})
        for j, row in enumerate(frame["rows"]):
            if row["kind"] != "vacuum_fields":
                left, _ = physical_to_chart(fields[j], derivatives[j], load.alpha[j],
                    load.log_amplitude_derivative[j], load.kappa)
                _, right = physical_to_chart(fields[j+1], derivatives[j+1], load.alpha[j+1],
                    load.log_amplitude_derivative[j+1], load.kappa)
                projection = aperture_projection(row["radius_nm"], frame["frames"][j][0],
                    radial.radial_modes, radial.potential_quadrature)
                budget = mask_loss_budget(projection, left, right, load.kappa)
                closure_error = abs(budget["total_removed"]-float(currents[j]-currents[j+1]))
                if closure_error > surface.flux_tolerance:
                    raise ValueError(f"Mask flux ledger failed at {row['component']}: {closure_error:.3g}")
                losses.append({**row, "net_incoming": float(currents[j]),
                    "net_outgoing": float(currents[j+1]), "absorbed": float(currents[j]-currents[j+1]),
                    **budget, "loss_decomposition_error": closure_error,
                    "absorbed_legacy_scope": "Total finite-basis removal; use mask_absorbed for disk absorption"})
        width = frame["exit_width_nm"]
        extent = math.sqrt(4*radial.radial_modes+160)*width
        step = 2*extent/grid_pixels
        axis = (np.arange(grid_pixels)-grid_pixels//2)*step
        yy, xx = np.meshgrid(axis, axis, indexing="ij")
        from temsim.physics.quartic_radial_phase import apply_quartic_on_grid, radial_envelope
        radius_grid = np.hypot(xx, yy)
        amplitude = radial_envelope(radius_grid, width, flux_coefficients)*step
        amplitude = apply_quartic_on_grid(amplitude, radius_grid, step, frame["exit_quartic_phase_per_nm4"])
        norm = float(np.sum(abs(amplitude)**2))
        if not math.isclose(norm, exit_current, rel_tol=1e-8, abs_tol=1e-13):
            raise ValueError(f"Gun-exit Cartesian representation is undersampled: norm={norm}, flux={exit_current}; increase grid_pixels")
        k_exit = math.sqrt(float(squared_wave_number(frame["exit_energy_ev"])))
        plane = PlaneWave(amplitude/math.sqrt(norm) if norm else amplitude,
            np.eye(2)*step*1e-9, np.zeros(2),
            curvature_m1=np.eye(2)*frame["exit_curvature_per_nm"]*1e9*load.kappa/k_exit)
        # Reference flight time/action include the near-tip electrostatic gain.
        # They are axial stationary references, not a reflected pulse delay.
        from temsim.physics.grounded_tip_field import grounded_field
        from scipy.constants import m_e
        nodes, weights = np.polynomial.legendre.leggauss(32)
        zs = (nodes+1)*surface.exit_height_nm/2
        xyz = np.column_stack((np.zeros(len(zs)), np.zeros(len(zs)), zs))*1e-9
        kinetic = energy+grounded_field(working).potential_rise_v_at_global_positions(xyz)
        ks = np.sqrt(squared_wave_number(kinetic))
        dt = surface.exit_height_nm/2*np.dot(weights, m_e*(1+kinetic/REST_EV)/(hbar*ks*1e9))*1e-9
        action = hbar*surface.exit_height_nm/2*np.dot(weights, ks)
        reference = AxialWaveReference(float(dt+frame["reference_flight_time_s"]),
                                      float(action+frame["reference_longitudinal_action_j_s"]))
        # The existing column/detector contract stores the axial carrier in
        # AxialWaveReference, outside the complex transverse envelope. Remove
        # that carrier exactly once; do not change any relative spatial phase.
        envelope_phase = np.exp(-1j*(action/hbar+frame["reference_carrier_phase_mod_rad"]))
        flux_coefficients *= envelope_phase
        plane = replace(plane, amplitude=plane.amplitude*envelope_phase)
        modes.append(WaveMode(plane, float(weight)*norm, TIP_REFERENCE, f"surface-energy:{index}",
            frame["exit_energy_ev"]*.001, reference))
        radial_payload.append({"mode_id": f"surface-energy:{index}", "width_nm": width,
            "coefficients_real": (flux_coefficients/math.sqrt(norm)).real.tolist() if norm else flux_coefficients.real.tolist(),
            "coefficients_imag": (flux_coefficients/math.sqrt(norm)).imag.tolist() if norm else flux_coefficients.imag.tolist(),
            "curvature_m1": frame["exit_curvature_per_nm"]*1e9*load.kappa/k_exit,
            "quartic_phase_per_nm4": frame["exit_quartic_phase_per_nm4"],
            "representation": "The same executed unit mode as the Cartesian plane, not a second source"})
        near_modes.append(SurfaceWaveMode(float(energy), float(weight), _immutable(psi.reshape(problem["z"].shape)), freeze_json(flux)))
        records.append({"energy_ev": float(energy), "mixture_weight": float(weight), "near_flux": flux,
            "wave_following_iterations": chart_iterations,
            "occupied_axial_refinement": axial_records,
            "exit_fraction": exit_current, "absorbed_fraction": flux["top"]-exit_current,
            "mask_absorbed_fraction": sum(row["mask_absorbed"] for row in losses),
            "unresolved_mask_fraction": sum(row["unresolved_transmitted"] for row in losses),
            "boundary_states": boundary_states,
            "axial_refinement_summary": {"step_evaluations": sum(row.get("step_evaluations", 0) for row in frame["rows"]),
                "accepted_substeps": sum(row.get("accepted_substeps", 0) for row in frame["rows"]),
                "summed_local_indicator": sum(row.get("summed_local_indicator", 0.) for row in frame["rows"])},
            "mask_losses": losses, "far_field": {k: v for k, v in frame.items()
                if k not in ("frames", "rows", "quartic_phases_per_nm4", "chart_derivatives")}})
        if _mode_completed is not None or _energy_cache is not None:
            # Output-only development evidence. This cannot replace the next
            # energy, admit a source, or inject a downstream wave. Preserve all
            # complex boundary traces before releasing their operator buffers.
            verify_inputs()
            history = {"z_nm": np.r_[frame["start_nm"], [row["z_nm"] for row in frame["rows"]]],
                "width_curvature": np.array([(frame["initial_width_nm"], frame["initial_curvature_per_nm"]), *frame["frames"]]),
                "quartic_phase_per_nm4": np.r_[0., frame["quartic_phases_per_nm4"]],
                "chart_derivatives": np.asarray(frame["chart_derivatives"]),
                "coefficients": fields, "covariant_derivatives": derivatives,
                "near_radius_nm": problem["radius_nm"], "near_z_nm": problem["z"],
                "near_potential_rise_v": problem["potential"].reshape(problem["z"].shape)}
            if _energy_cache is not None:
                from temsim.physics.surface_mode_cache import preserve_mode
                preserve_mode(_energy_cache, identity, modes[-1], near_modes[-1], records[-1],
                    radial_payload[-1], history, z_mm=working.exit_plane_z_mm,
                    current_a=problem["model"].emission.current_na*1e-9, verify=verify_inputs)
            if _mode_completed is not None:
                _mode_completed(modes[-1], near_modes[-1], freeze_json(records[-1]),
                    freeze_json(radial_payload[-1]), {key: _immutable(value) for key, value in history.items()}, identity)
            del history
        del load, fields, derivatives
    verify_inputs()
    near = SurfaceWaveCheckpoint(_immutable(problem["radius_nm"]), _immutable(problem["z"]),
        _immutable(problem["potential"].reshape(problem["z"].shape)), tuple(near_modes),
        problem["model"].emission.current_na*1e-9, freeze_json({"schema": "joint-tip-gun-near-field-v1",
        "source": problem["model"].to_dict(), "grounded_field": problem["grounded_field"],
        "implementation": implementation, "numerics": asdict(surface), "radial_numerics": asdict(radial),
        "ports": "Driven physical tip, local side port, two-way complete round-gun top load",
        "phase_coordinates": "Each mode stores full complex nodal values. Its FE envelope is recovered by removing exp(i*gamma*r^2/2), with gamma in mode.flux.radial_phase_gradient_nm2; retain this analytic phase between nodes.",
        "scope": "Jointly solved near field, not an independent downstream source"}))
    checkpoint = TipGunCheckpoint(BeamState(tuple(modes), TIP_REFERENCE), working.exit_plane_z_mm,
        near.reference_current_a, {"schema": "executed-grounded-round-surface-gun-v1",
        "source": problem["model"].to_dict(), "gun_inputs": graph, "near_field_digest": near.digest,
        "implementation": implementation, "surface_numerics": asdict(surface), "radial_numerics": asdict(radial),
        "mode_records": records, "physical_components": [component.key for component in working.components],
        "near_column_guard": column_guard,
        "column_snapshot": None if column_snapshot is None else column_snapshot.digest,
        "unresolved_side_fraction": sum(m.weight*m.flux["side"] for m in near_modes),
        "loss_scope": "Near-field side escape is a numerical-domain channel, NOT physical absorption. Finite-basis mask loss also includes unresolved modes.",
        "radial_output_modes": radial_payload,
        "executed_energy_histories": energy_histories,
        "phase": "Complex envelope from the driven tip; axial carrier stored separately in AxialWaveReference; no aggregate mixture phase",
        "validation_status": "DEVELOPMENT_REQUIRES_FULL_CHAIN_CONVERGENCE"})
    return checkpoint, near
