"""Controlled, tip-origin particle tests of accelerator turning mechanisms.

This is an opt-in diagnostic. It changes no production defaults or GUI settings.
All fields retain extraction, focusing and acceleration. Existing physical
apertures and downstream gun parts remain checked. Generated data stay local.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from temsim.cpu_resources import numerical_job
from scripts.diagnose_planar_gun import (
    CHARGE, LIGHT, MASS, array_digest, compare, kinetic_energy, positions,
    sample_states, trace,
)


def trajectory_metrics(z, state, field, initial):
    """Check every saved state, not just the rendered vertices or mean energy."""
    initial_invariant = kinetic_energy(initial)-field.potential_v_at_global_positions(positions(0., initial))
    invariant_error, max_ez = 0., -np.inf
    for lo in range(0, len(z), 64):
        sl = slice(lo, lo+64)
        q = np.concatenate((state[sl, :, :2],
            np.broadcast_to(z[sl, None, None], (*state[sl].shape[:2], 1))), axis=-1)
        phi = field.potential_v_at_global_positions(q.reshape(-1, 3)).reshape(q.shape[:-1])
        invariant_error = max(invariant_error, float(np.max(abs(kinetic_energy(state[sl])-phi-initial_invariant))))
        electric = field.field_at_global_positions_v_per_m(q.reshape(-1, 3)).reshape(q.shape)
        max_ez = max(max_ez, float(np.max(electric[..., 2])))
    return {"max_energy_invariant_error_ev_all_saved": invariant_error,
            "minimum_pz_over_mc_all_saved": float(np.min(state[..., 4])),
            "nonforward_saved_states": int(np.sum(state[..., 4] <= 0)),
            "max_ez_v_per_m_on_saved_paths": max_ez}


def check_clearances(gun, field, z, state, *, reference):
    """Near-axis clearance checks; no accepted case may hide an interception."""
    radial_mm = np.linalg.norm(state[..., :2], axis=-1)*1000
    margins = []
    replaced = {"feg_extractor", "feg_electrostatic_lens", "feg_accelerator"} if reference else set()
    for part in gun.bore_components:
        if part.key in replaced:
            continue
        mask = abs(z*1000-part.mechanical_center_from_tip_mm) <= part.mechanical_length_mm/2
        if np.any(mask):
            margin = part.mechanical_clear_bore_diameter_mm/2-float(np.max(radial_mm[mask]))
            if margin <= 0:
                raise ValueError("Body interception needs a terminal-event calculation: "+part.key)
            margins.append({"part": part.key, "sampled_margin_mm": margin})
    if reference:
        for part in field.request["rings"]:
            mask = (z >= part["start_m"]) & (z <= part["stop_m"])
            if np.any(mask):
                margin = part["inner_m"]*1000-float(np.max(radial_mm[mask]))
                if margin <= 0:
                    raise ValueError("Reference electrode interception needs terminal events: "+part["key"])
                margins.append({"part": part["key"], "sampled_margin_mm": margin})
    apertures = []
    for aperture in (gun.dpa_aperture, gun.c1_aperture):
        index = int(np.argmin(abs(z-aperture.z_mm*.001)))
        if abs(z[index]-aperture.z_mm*.001) > 1e-14:
            raise ValueError("Physical aperture must be an executed endpoint")
        xy = state[index, :, :2]*1000
        mask = aperture.transmission_mask(xy[:, 0], xy[:, 1])
        if not np.all(mask):
            raise ValueError("Aperture interception needs terminal-event calculation")
        apertures.append({"part": aperture.key, "passed": int(np.sum(mask))})
    return {"sampled_body_clearances": margins, "apertures": apertures,
            "scope": "Near-axis paths with sampled body clearance; not general collision tracking"}


def definitions(group):
    if group == "ramps":
        return [(f"analytic_width{width}", "analytic", width, .00025) for width in (2, 4, 8, 16)] + [
            ("analytic_width4_halfstep", "analytic", 4, .000125)]
    if group == "reference":
        return [(f"reference{n}", "reference", n, .00025) for n in (8, 16, 32)] + [
            ("reference16_flat", "reference_flat", 16, .00025),
            ("reference16_halfstep", "reference", 16, .000125),
            ("reference16_exit100", "reference_exit100", 16, .00025),
            ("reference16_exit200", "reference_exit200", 16, .00025)]
    raise ValueError("Unknown diagnostic group")


def run(args):
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.optics.column import default_state
    from temsim.physics.relativistic_lorentz import momentum_from_kinetic_energy_ev

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    instrument = default_state()
    gun = instrument.electron_gun
    if (gun.emitter.surface_model is not None or gun.emitter.curvature_nm_inv
            or gun.emitter.coherence is not None or gun.monochromator_installed):
        raise ValueError("Only original flat classical emission is admitted by this diagnostic")
    probe = np.column_stack((np.full(4501, 1e-6), np.zeros(4501), np.linspace(0., .45, 4501)))
    if np.any(gun.magnetic_field.field_at_global_positions_t(probe)):
        raise ValueError("Active magnetic controls require their full force in the diagnostic")
    emitted = gun.emit(args.particles)
    if len(emitted.ray_id) != args.particles:
        raise ValueError("Unexpected emission population")
    momentum = momentum_from_kinetic_energy_ev(gun.emitter.emission_energy_ev+emitted.energy_offset_ev,
        np.column_stack((emitted.tx_rad, emitted.ty_rad, np.ones(args.particles))))/(MASS*LIGHT)
    initial = np.column_stack((emitted.x_m, emitted.y_m, momentum, np.zeros(args.particles)))
    weights = emitted.weight/np.sum(emitted.weight)
    snapshot = capture_instrument_snapshot(instrument).to_dict()
    emission_digest = array_digest(state=initial, ray_id=emitted.ray_id, weight=emitted.weight)
    (output/"instrument-input-snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    report = {"scope": "Classical tip-origin accelerator mechanism diagnostic; no default field change",
        "group": args.group, "particles": args.particles, "cpu_threads": 1,
        "input_snapshot_digest": snapshot["digest"], "emission_sha256": emission_digest,
        "targets": {"energy_invariant_ev": .001, "mesh_envelope_relative_to_peak": .01,
                    "mesh_slope_relative_to_peak": .01},
        "runs": {}, "comparisons": {},
        "limitations": ["Reference conductors and numerical boundaries are explicit assumptions, not commercial calibration.",
            "Planar cathode, no space charge or self-consistent extraction-current feedback.",
            "Existing aperture electrical potentials remain unspecified; their interception is retained.",
            "No coherent wave or downstream column calculation. Archives are diagnostic, not GUI resume files."]}
    observations = {}
    reference_z = np.unique(np.r_[0., np.geomspace(1e-9, 1e-4, 101), np.linspace(.0001, .45, 4501)])
    report_path = output/(args.group+"-comparison.json")
    for name, kind, value, step in definitions(args.group):
        started = time.perf_counter()
        model = copy.deepcopy(gun)
        if kind == "analytic":
            for stage in model.accelerator.stages:
                stage.soft_edge_mm = value
            field = model.electric_field
            boundaries = np.array([v*.001 for pair in model.field_supports_mm for v in pair])
            request = {"provider": "existing_analytic", "stage_soft_edge_half_width_mm": value,
                "input_snapshot_digest": snapshot["digest"],
                "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                    Path("src/temsim/physics/analytic_gun_field.py"),
                    Path("src/temsim/optics/electron_gun/electrostatic.py"))}}
        else:
            from temsim.physics.patent_gun_reference import build_patent_gun_field
            field = build_patent_gun_field(model, cathode_boundary="planar_equipotential",
                cells_per_bore=value, cache_dir=output/"field-cache",
                contour="flat" if kind == "reference_flat" else "stepped",
                exit_extension_mm={"reference_exit100": 100., "reference_exit200": 200.}.get(kind, 0.))
            request, boundaries = field.request, field.z
        boundaries = np.r_[boundaries, gun.dpa_aperture.z_mm*.001, gun.c1_aperture.z_mm*.001]
        print("Field ready:", name, flush=True)
        try:
            z, state, nfev = trace(field, initial, 0., .45, boundaries,
                max_step_m=step, sample_planes=reference_z)
            checks = trajectory_metrics(z, state, field, initial)
            clearances = check_clearances(model, field, z, state, reference=kind != "analytic")
            if checks["max_energy_invariant_error_ev_all_saved"] >= .001:
                raise ValueError("Energy invariant exceeded predeclared 0.001 eV bound")
            samples = sample_states(z, state, reference_z)
            observations[name] = samples
            phi = field.potential_v_at_global_positions(probe)
            electric = field.field_at_global_positions_v_per_m(probe)
            identity = {"field": request, "emission_sha256": emission_digest,
                "numerics": {"max_step_m": step, "rtol": 2e-9},
                "tracer_sha256": hashlib.sha256(Path("scripts/diagnose_planar_gun.py").read_bytes()).hexdigest(),
                "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "state_sha256": array_digest(z=z, state=state),
                "calculated_endpoint_mm": 450., "saved_utc": datetime.now(timezone.utc).isoformat()}
            archive = output/(name+".npz")
            np.savez_compressed(archive, z_m=z, state=state, reference_z_m=reference_z,
                samples=samples, probe_m=probe, phi_v=phi, electric_v_per_m=electric,
                ray_id=emitted.ray_id, weights=emitted.weight,
                input_snapshot_digest=snapshot["digest"], identity_json=json.dumps(identity, sort_keys=True))
            record = {"status": "completed", "seconds": time.perf_counter()-started,
                "field": request, "field_report": getattr(field, "report", {}),
                "archive_path": str(archive), "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "identity": identity, "nfev": nfev, "calculated_endpoint_mm": float(z[-1]*1000),
                "survivors": args.particles, **checks, **clearances,
                "exit_mean_energy_ev": float(np.sum(weights*kinetic_energy(state[-1]))),
                "exit_mean_tof_ns": float(np.sum(weights*state[-1, :, 5])*1e9),
                "exit_rms_radius_um": float(np.sqrt(np.sum(weights*np.sum(state[-1, :, :2]**2, axis=-1)))*1e6)}
        except (ValueError, RuntimeError) as exc:
            report["runs"][name] = {"status": "failed", "error": str(exc), "field": request}
            report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            raise
        report["runs"][name] = record
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Completed {name}: {record['seconds']:.2f}s; invariant {checks['max_energy_invariant_error_ev_all_saved']:.3g} eV", flush=True)
    base = "analytic_width4" if args.group == "ramps" else "reference16"
    for name, samples in observations.items():
        if name != base:
            report["comparisons"][base+"__"+name] = compare(observations[base], samples, weights)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("tmp/accelerator-mechanism-20260926"))
    parser.add_argument("--group", choices=("ramps", "reference"), default="ramps")
    parser.add_argument("--particles", type=int, default=193)
    args = parser.parse_args()
    if not 9 <= args.particles <= 5000:
        parser.error("Diagnostic population must be between 9 and 5000")
    with numerical_job(requested=1):
        run(args)


if __name__ == "__main__":
    main()
