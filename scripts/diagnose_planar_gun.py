"""Opt-in, classical tip-to-exit validation of a reference planar cathode.

Run with PYTHONPATH=src and the project Python. This does not select a new
production field or source. Output archives are diagnostic data, not GUI
result files. Every trajectory begins at the original flat emission samples.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.integrate import solve_ivp

from temsim.cpu_resources import numerical_job
from temsim.physics.grounded_tip_field import merge_axis_nodes
from temsim.physics.relativistic_lorentz import (
    ELECTRON_MASS_KG as MASS, ELEMENTARY_CHARGE_C as CHARGE,
    SPEED_OF_LIGHT_M_PER_S as LIGHT, momentum_from_kinetic_energy_ev,
)


def array_digest(**arrays):
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        value = np.ascontiguousarray(value)
        digest.update(json.dumps([name, value.dtype.str, list(value.shape)]).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def kinetic_energy(state):
    u2 = np.sum(state[..., 2:5]**2, axis=-1)
    return (MASS*LIGHT**2/CHARGE)*u2/(np.sqrt(1+u2)+1)


def positions(z, state):
    return np.column_stack((state[:, :2], np.full(len(state), z)))


def trace(field, initial, start_m, end_m, boundaries_m=(), *,
          max_step_m=0.00025, rtol=2e-9, sample_planes=()):
    """Independent relativistic Z integration, for forward vacuum rays only.

    Backstreaming or leaving the solved field fails the diagnostic; neither
    event is converted into a successful downstream state. A production
    general-purpose transport would need terminal interception/turning events.
    """
    initial = np.asarray(initial, dtype=float)
    if (initial.ndim != 2 or initial.shape[1] != 6 or not len(initial)
            or not np.isfinite(initial).all() or np.any(initial[:, 4] <= 0)
            or not 0 <= start_m < end_m or not np.isfinite(end_m)
            or not np.isfinite(max_step_m) or max_step_m <= 0
            or not 0 < rtol < 1):
        raise ValueError("Invalid forward particle diagnostic input")
    count = len(initial)
    axes = np.asarray(boundaries_m, dtype=float)
    if axes.ndim != 1 or not np.isfinite(axes).all():
        raise ValueError("Integration boundaries must be a finite one-dimensional array")
    axes = axes[(axes > start_m) & (axes < end_m)]
    bounds = merge_axis_nodes(axes, boundaries=[start_m, end_m])
    extras = np.asarray(sample_planes, dtype=float)
    if not np.isfinite(extras).all():
        raise ValueError("Sample planes must be finite")
    force_lo, force_hi = start_m, end_m

    def ode(z, flat):
        state = flat.reshape(count, 6)
        momentum = state[:, 2:5]
        gamma = np.sqrt(1+np.sum(momentum**2, axis=1))
        if not np.isfinite(state).all() or np.any(momentum[:, 2] <= 0):
            raise ValueError("Backstreaming or nonfinite state; diagnostic cannot continue")
        sample_z = float(np.clip(z, force_lo, force_hi))
        electric = field.field_at_global_positions_v_per_m(positions(sample_z, state))
        return np.column_stack((
            momentum[:, :2]/momentum[:, 2, None],
            -CHARGE*electric*gamma[:, None]/(MASS*LIGHT**2*momentum[:, 2, None]),
            gamma/(LIGHT*momentum[:, 2]),
        )).ravel()

    state = initial.copy()
    zs, states, nfev = [start_m], [state.copy()], 0
    atol = np.tile([1e-14, 1e-14, 1e-12, 1e-12, 1e-12, 1e-20], count)
    with numerical_job(requested=1):
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            # The field is the derivative of a continuous cellwise potential.
            # Use this cell's one-sided force for RK stages at its endpoints.
            force_lo = lo+8*np.spacing(max(abs(lo), 1e-6))
            force_hi = hi-8*np.spacing(max(abs(hi), 1e-6))
            if force_lo >= force_hi:
                raise ValueError("Unresolved interval in particle diagnostic")
            # Sub-eV emission can accelerate over nanometres even in a smooth
            # field. Bound the first trial by 5% axial momentum change so the
            # solver's generic initial-step estimate cannot jump across that
            # launch scale and create a spurious negative RK-stage momentum.
            derivative = ode(lo, state.ravel()).reshape(count, 6)
            changing = abs(derivative[:, 4]) > 0
            momentum_scale = (np.min(abs(state[changing, 4]/derivative[changing, 4]))
                              if np.any(changing) else np.inf)
            first_step = min(hi-lo, max_step_m, .05*momentum_scale)
            solution = solve_ivp(ode, (lo, hi), state.ravel(), method="DOP853",
                rtol=rtol, atol=atol, first_step=first_step,
                max_step=max_step_m, dense_output=True)
            if not solution.success:
                raise RuntimeError(solution.message)
            nfev += solution.nfev
            sampled = merge_axis_nodes(np.r_[
                np.linspace(lo, hi, max(2, int(np.ceil((hi-lo)/0.00025))+1))[1:],
                extras[(extras > lo) & (extras < hi)]], boundaries=[hi])
            values = solution.sol(sampled).T.reshape(-1, count, 6)
            # Archive the executed endpoint, not its dense interpolant.
            state = solution.y[:, -1].reshape(count, 6).copy()
            values[-1] = state
            zs.extend(sampled)
            states.extend(values)
    return np.asarray(zs), np.asarray(states), nfev


def sample_states(z, state, planes):
    return np.stack([np.column_stack([
        np.interp(planes, z, state[:, i, component]) for i in range(state.shape[1])
    ]) for component in range(6)], axis=-1)


def compare(first, second, weights):
    radius_a = np.sqrt(np.sum(weights*np.sum(first[..., :2]**2, axis=-1), axis=-1))
    radius_b = np.sqrt(np.sum(weights*np.sum(second[..., :2]**2, axis=-1), axis=-1))
    slope_a = first[..., 2:4]/first[..., 4, None]
    slope_b = second[..., 2:4]/second[..., 4, None]
    return {
        "envelope_relative_to_peak": float(np.max(abs(radius_a-radius_b))/max(np.max(radius_b), 1e-30)),
        "slope_relative_to_peak": float(np.max(np.linalg.norm(slope_a-slope_b, axis=-1))/max(np.max(np.linalg.norm(slope_b, axis=-1)), 1e-30)),
        "max_xy_difference_nm": float(np.max(np.linalg.norm(first[..., :2]-second[..., :2], axis=-1))*1e9),
        "max_tof_difference_fs": float(np.max(abs(first[..., 5]-second[..., 5]))*1e15),
    }


def physical_clearance(gun, z, states):
    """Sampled clearance with exact aperture planes; abort on any interception."""
    body = []
    for component in gun.bore_components:
        inside = abs(z*1000-component.mechanical_center_from_tip_mm) <= component.mechanical_length_mm/2
        if np.any(inside):
            margin = component.mechanical_clear_bore_diameter_mm/2-np.max(np.linalg.norm(states[inside, :, :2], axis=-1))*1000
            if margin <= 0:
                raise ValueError("Possible body interception: exact event tracing needed before accepting this diagnostic")
            body.append({"key": component.key, "sampled_margin_mm": float(margin)})
    apertures = []
    for aperture in (gun.dpa_aperture, gun.c1_aperture):
        plane = aperture.z_mm*0.001
        index = int(np.argmin(abs(z-plane)))
        if abs(z[index]-plane) > 1e-14:
            raise ValueError("Aperture is not an executed integration endpoint")
        xy = states[index, :, :2]*1000
        passed = aperture.transmission_mask(xy[:, 0], xy[:, 1])
        if not np.all(passed):
            raise ValueError("Aperture interception: exact terminal-event tracing needed")
        apertures.append({"key": aperture.key, "passed": int(np.sum(passed))})
    return {"sampled_body_clearances": body, "apertures": apertures,
            "scope": "Near-axis diagnostic; body clearance sampled, not general collision-event qualification"}


def run(args):
    from temsim.optics.column import default_state
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.physics.planar_gun_field import build_planar_gun_field

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    instrument = default_state()
    gun = instrument.electron_gun
    if (gun.emitter.surface_model is not None or gun.emitter.curvature_nm_inv
            or gun.emitter.coherence is not None or gun.monochromator_installed):
        raise ValueError("This diagnostic requires the unchanged flat, classical, no-monochromator source")
    # The diagnostic equation includes electric forces only. Reject active
    # default magnetic controls rather than silently omitting their physics.
    probe = np.column_stack((np.full(1001, 1e-6), np.zeros(1001), np.linspace(0, .45, 1001)))
    if np.any(gun.magnetic_field.field_at_global_positions_t(probe)):
        raise ValueError("Magnetic gun controls are active; electric-only diagnostic rejected")
    emitted = gun.emit(args.particles)
    if len(emitted.ray_id) != args.particles:
        raise ValueError("Unexpected emission population; no support probes may be silently added")
    weights = emitted.weight/np.sum(emitted.weight)
    momenta = momentum_from_kinetic_energy_ev(gun.emitter.emission_energy_ev+emitted.energy_offset_ev,
        np.column_stack((emitted.tx_rad, emitted.ty_rad, np.ones(args.particles))))
    initial = np.column_stack((emitted.x_m, emitted.y_m, momenta/(MASS*LIGHT), np.zeros(args.particles)))
    snapshot = capture_instrument_snapshot(instrument).to_dict()
    (output/"instrument-input-snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    report = {
        "scope": "Coupled reference electrode field with explicitly idealized planar cathode; not default GUI model or full gun qualification",
        "particles": args.particles, "cpu_threads": 1,
        "declared_targets": {"mesh_envelope_relative": .01, "mesh_slope_relative": .01,
            "energy_invariant_error_ev": .001, "continuation_xy_m": 1e-11, "continuation_slope_rad": 1e-8},
        "input_snapshot_digest": snapshot["digest"],
        "emission_sha256": array_digest(state=initial, ray_id=emitted.ray_id, weight=emitted.weight),
        "state_columns": ["x_m", "y_m", "px_over_mc", "py_over_mc", "pz_over_mc", "time_since_emission_s"],
        "runs": {}, "comparisons": {},
        "limitations": ["Ideal planar cathode does not define the current cone-shaped tip conductor.",
            "Vacuum Laplace field; no space charge or self-consistent emission current.",
            "Apertures and liners have no electrical connection metadata; clipping retained, electrode potentials not invented.",
            "No downstream column or specimen calculation. Checkpoints are diagnostic archives, not GUI files."],
    }

    def persist():
        (output/"comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    # Mesh gates declared above; boundary and control sensitivities reported
    # separately, never converted into a mesh-convergence pass.
    definitions = [
        ("analytic", None, 2., 0., None, .00025),
        ("planar8", 8, 2., 0., None, .00025),
        ("planar16", 16, 2., 0., None, .00025),
        ("planar32", 32, 2., 0., None, .00025),
        ("outer1_25", 16, 1.25, 0., None, .00025),
        ("outer3", 16, 3., 0., None, .00025),
        ("exit_plus100", 16, 2., 100., None, .00025),
        ("exit_plus200", 16, 2., 200., None, .00025),
        ("lens_plus10pct", 16, 2., 0., "lens", .00025),
        ("extractor_plus10pct", 16, 2., 0., "extractor", .00025),
        ("step_halved", 16, 2., 0., None, .000125),
    ]
    if args.quick:
        definitions = definitions[:3]
    references = np.unique(np.r_[0., np.geomspace(1e-9, 1e-4, 101), np.linspace(.0001, .45, 2001)])
    observations = {}
    for name, cells, outer, extension, control, step in definitions:
        start = time.perf_counter()
        model = copy.deepcopy(gun)
        if control == "lens":
            model.electrostatic_lens.voltage_kv *= 1.1
        elif control == "extractor":
            model.extractor.voltage_kv *= 1.1
        if cells is None:
            field = gun.electric_field
            field_report = {"provider": "current analytic"}
            boundaries = np.array([v*.001 for pair in gun.field_supports_mm for v in pair])
        else:
            field = build_planar_gun_field(model, cathode_boundary="planar_equipotential",
                cells_per_bore=cells, outer_factor=outer, exit_extension_mm=extension,
                cache_dir=output/"field-cache")
            field_report = dict(field.report)
            warm = build_planar_gun_field(model, cathode_boundary="planar_equipotential",
                cells_per_bore=cells, outer_factor=outer, exit_extension_mm=extension,
                cache_dir=output/"field-cache")
            np.testing.assert_array_equal(field.field_at_global_positions_v_per_m(probe),
                                          warm.field_at_global_positions_v_per_m(probe))
            field_report["repeat_cache_report"] = dict(warm.report)
            boundaries = field.z
        boundaries = np.r_[boundaries, model.dpa_aperture.z_mm*.001, model.c1_aperture.z_mm*.001]
        print(f"Field ready: {name}", flush=True)
        z, state, nfev = trace(field, initial, 0., .45, boundaries,
            max_step_m=step, sample_planes=references)
        clearance = physical_clearance(model, z, state)
        invariant_error = 0.
        initial_invariant = kinetic_energy(initial)-field.potential_v_at_global_positions(positions(0., initial))
        for index in np.unique(np.r_[np.arange(0, len(z), max(1, len(z)//300)), len(z)-1]):
            invariant = kinetic_energy(state[index])-field.potential_v_at_global_positions(positions(z[index], state[index]))
            invariant_error = max(invariant_error, float(np.max(abs(invariant-initial_invariant))))
        samples = sample_states(z, state, references)
        observations[name] = samples
        phi, electric = (field.potential_v_at_global_positions(probe), field.field_at_global_positions_v_per_m(probe))
        record = {
            "field": field_report, "seconds": time.perf_counter()-start, "nfev": nfev,
            "calculated_endpoint_mm": float(z[-1]*1000), "survivors": args.particles,
            "max_energy_invariant_error_ev_sampled": invariant_error,
            "exit_rms_radius_um": float(np.sqrt(np.sum(weights*np.sum(state[-1, :, :2]**2, axis=-1)))*1e6),
            "exit_mean_energy_ev": float(np.sum(weights*kinetic_energy(state[-1]))),
            "exit_mean_tof_ns": float(np.sum(weights*state[-1, :, 5])*1e9),
            "source_ez_v_per_m": float(electric[0, 2]), **clearance,
        }
        if name == "planar16":
            launch_planes = references[references <= .001]
            launch_z, launch_state, launch_nfev = trace(field, initial, 0., .001,
                boundaries, max_step_m=step/2, rtol=2e-11, sample_planes=launch_planes)
            record["launch_tolerance_check"] = compare(
                sample_states(z, state, launch_planes),
                sample_states(launch_z, launch_state, launch_planes), weights)
            record["launch_tolerance_check"]["nfev"] = launch_nfev
            record["launch_tolerance_check"]["scope_mm"] = [0., 1.]
            continuations = []
            for cutoff in (.026, .030, .034):
                zz, yy, _ = trace(field, initial, 0., cutoff, boundaries, max_step_m=step)
                endpoint = yy[-1].copy()
                identity = {"field_request": field.request, "input_snapshot_digest": snapshot["digest"],
                    "emission_sha256": report["emission_sha256"], "particles": args.particles,
                    "cutoff_m": cutoff, "numerics": "DOP853-z-rtol2e-9-xy1e-14-u1e-12-t1e-20-maxstep0.25mm",
                    "tracer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "state_sha256": array_digest(state=endpoint)}
                path = output/f"checkpoint-{cutoff*1000:g}mm.npz"
                np.savez_compressed(path, state=endpoint, ray_id=emitted.ray_id,
                    weights=emitted.weight, identity_json=json.dumps(identity, sort_keys=True))
                with np.load(path, allow_pickle=False) as archive:
                    restored = archive["state"]
                    saved_identity = json.loads(str(archive["identity_json"]))
                    if saved_identity != identity or array_digest(state=restored) != identity["state_sha256"]:
                        raise ValueError("Checkpoint identity mismatch")
                    np.testing.assert_array_equal(restored, endpoint)
                    np.testing.assert_array_equal(archive["ray_id"], emitted.ray_id)
                    np.testing.assert_array_equal(archive["weights"], emitted.weight)
                _, after, _ = trace(field, restored, cutoff, .45, boundaries, max_step_m=step)
                delta = compare(after[-1:], state[-1:], weights)
                delta["cutoff_mm"] = cutoff*1000
                delta["max_slope_difference_rad"] = float(np.max(np.linalg.norm(
                    after[-1, :, 2:4]/after[-1, :, 4, None]-state[-1, :, 2:4]/state[-1, :, 4, None], axis=-1)))
                continuations.append(delta)
            record["saved_continuation_checks"] = continuations
        np.savez_compressed(output/f"{name}.npz", z_m=z, state=state, reference_z_m=references,
            samples=samples, probe_m=probe, phi_v=phi, electric_v_per_m=electric,
            ray_id=emitted.ray_id, weights=emitted.weight,
            input_snapshot_digest=snapshot["digest"])
        record["archive_path"] = str(output/f"{name}.npz")
        report["runs"][name] = record
        persist()
        print(f"Completed {name}: {record['seconds']:.2f} s, exit RMS {record['exit_rms_radius_um']:.6g} um", flush=True)
    for name in observations:
        if name == "planar16":
            continue
        report["comparisons"]["planar16__"+name] = compare(observations["planar16"], observations[name], weights)
    if "planar32" in observations:
        mesh = report["comparisons"]["planar16__planar32"]
        report["mesh_target_met"] = mesh["envelope_relative_to_peak"] < .01 and mesh["slope_relative_to_peak"] < .01
    report["energy_target_met"] = all(row["max_energy_invariant_error_ev_sampled"] < .001 for row in report["runs"].values())
    report["continuation_target_met"] = all(row["max_xy_difference_nm"] < .01 and row["max_slope_difference_rad"] < 1e-8
        for row in report["runs"]["planar16"]["saved_continuation_checks"])
    persist()
    render(output, report)
    print("Diagnostic completed; production default unchanged", flush=True)
    return report


def render(output, report):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(4, 1, figsize=(11, 11), sharex=True, constrained_layout=True)
    for name in ("analytic", "planar8", "planar16", "planar32"):
        if name not in report["runs"]:
            continue
        with np.load(output/f"{name}.npz", allow_pickle=False) as data:
            z = data["probe_m"][:, 2]*1000
            axs[0].plot(z, data["phi_v"]/1000, label=name)
            axs[1].plot(z, -data["electric_v_per_m"][:, 2]/1e6, label=name)
            samples = data["samples"]
            weights = data["weights"]/np.sum(data["weights"])
            rms = np.sqrt(np.sum(weights*np.sum(samples[..., :2]**2, axis=-1), axis=-1))
            axs[2].plot(data["reference_z_m"]*1000, rms*1e6, label=name)
            axs[3].plot(data["reference_z_m"]*1000, np.sum(weights*samples[..., 5], axis=-1)*1e9, label=name)
    for ax, label in zip(axs, ("Potential rise (kV)", "-Ez at r=1 um (MV/m)", "RMS radius (um)", "Mean time since emission (ns)")):
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axs[-1].set_xlabel("Axial Z (mm)")
    fig.suptitle("Coupled reference electrode field: explicitly idealized planar cathode\nSame tip emission samples; no artificial accelerator inlet; not production default")
    fig.savefig(output/"planar-gun-comparison.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("tmp/planar-gun-20260926"))
    parser.add_argument("--particles", type=int, default=193)
    parser.add_argument("--quick", action="store_true", help="Analytic and two mesh levels only; not full convergence evidence")
    args = parser.parse_args()
    if args.particles < 9:
        parser.error("At least nine particles are required")
    with numerical_job(requested=1):
        run(args)


if __name__ == "__main__":
    main()
