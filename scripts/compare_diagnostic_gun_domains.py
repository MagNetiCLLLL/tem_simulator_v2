"""Bounded, serial comparison of production and extended gun electric domains.

Default: three detached input cases, two original tip rays, four field solves
and twelve reference traces per case. --describe prepares requests only.
No coherent execution, source substitution, field replacement or qualification
is performed. Numerical arrays and caches are written only below --output.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import traceback
from types import SimpleNamespace

import numpy as np

from temsim.cpu_resources import numerical_job
from temsim.diagnostic_gun_comparison import (
    REFERENCE_CHECKS, compare_populations, emission_samples, field_difference,
    trajectory_measurements,
)
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.magnetic_field_scene import MagneticSourceRegion, prepare_magnetic_scene
from temsim.magnetic_test_particle import TestElectronSettings, trace_test_electron
from temsim.optics.column import default_state
from temsim.physics.planar_gun_field import request_digest
from temsim.test_electron_scene import TestElectronScene, _active_apertures, _physical_bores


CASES = ("flat", "curved", "near_aperture")
COMPARISON_PAIRS = (("domain_coarse", "production_coarse", "extended_coarse"),
                   ("domain_fine", "production_fine", "extended_fine"),
                   ("production_grid", "production_coarse", "production_fine"),
                   ("extended_grid", "extended_coarse", "extended_fine"),
                   ("production_step", "production_coarse", "production_coarse_step_refined"),
                   ("extended_step", "extended_coarse", "extended_coarse_step_refined"))


def source_file_hashes():
    """On-disk provenance, distinct from already imported function bytecode."""
    import temsim
    root = Path(temsim.__file__).parent
    names = ("diagnostic_gun_comparison.py", "magnetic_test_particle.py", "test_electron_scene.py",
             "magnetic_field_scene.py", "physics/closed_gun_field.py", "physics/continuous_gun_field.py",
             "physics/planar_gun_field.py", "physics/axisymmetric_cut_field.py", "physics/axis_regular_potential.py")
    return {**{name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names},
            "scripts/compare_diagnostic_gun_domains.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def inputs(case):
    if case not in CASES:
        raise ValueError("Unknown gun-domain comparison case")
    state = default_state()
    gun = state.electron_gun
    gun.emitter.curvature_nm_inv = .01 if case == "curved" else 0.
    # An explicitly detached mechanical perturbation puts the optical axis
    # within 0.1% of the real aperture rim. It is not a downstream launch.
    if case == "near_aperture":
        gun.dpa_aperture.offset_x_mm = .999*gun.dpa_aperture.radius_mm
    if gun.monochromator_installed:
        raise ValueError("This bounded default-input experiment has not declared an installed selector case")
    return state


def requests(gun, *, cells, refined_cells, extended_stop_mm):
    curved = float(gun.emitter.curvature_nm_inv) != 0.
    if curved:
        from temsim.physics.continuous_gun_field import continuous_field_request as request
        from temsim.physics.continuous_gun_field import mesh_axes
    else:
        from temsim.physics.closed_gun_field import closed_field_request as request, mesh_axes
    original = request(gun)
    baseline_extension = float(original["domain"]["exit_m"])*1e3-float(gun.exit_plane_z_mm)
    extended = float(extended_stop_mm)-float(gun.exit_plane_z_mm)
    if extended <= baseline_extension:
        raise ValueError("Extended endpoint must lie beyond the production numerical field domain")
    result = {}
    for grid, count in (("coarse", cells), ("fine", refined_cells)):
        for domain, extension in (("production", baseline_extension), ("extended", extended)):
            options = dict(cells_per_bore=count, exit_extension_mm=extension)
            if curved and grid == "fine":
                options.update(apex_cells_per_radius=64, tip_nodes=256)
            item = request(gun, **options)
            r, z = mesh_axes(item)
            result[f"{domain}_{grid}"] = {"options": options, "request": item,
                "request_sha256": request_digest(item), "grid_shape": [len(r), len(z)],
                "mesh_nodes": len(r)*len(z)}
    return result


def _scene(state, magnetic, field):
    gun = state.electron_gun
    bounds = np.array(((-field.r[-1], -field.r[-1], field.z[0]),
                       (field.r[-1], field.r[-1], field.z[-1])))
    diagnostic = bounds.copy()
    diagnostic[1, 2] = gun.exit_plane_z_mm*1e-3
    if gun.emitter.curvature_nm_inv == 0:
        diagnostic[0, 2] = min(diagnostic[0, 2], gun.emitter.mechanical_center_from_tip_mm*1e-3)
    region = MagneticSourceRegion("gun_electric_field", "Gun electrostatic field", "electric", bounds)
    return TestElectronScene(replace(magnetic, diagnostic_bounds_m=diagnostic), field, field,
        diagnostic, bounds, (0., 0., 0.), gun.emitter.emission_energy_ev,
        float(gun.exit_plane_z_mm)*1e-3, ("Isolated numerical-domain comparison; original tip samples",),
        _active_apertures(state, gun), _physical_bores(state, gun, field), (region,),
        _flat_cathode=gun.emitter.curvature_nm_inv == 0.)


def _probes(request):
    """Common physical points; near-rim samples stay separate from axis probes."""
    main, near_wall = [], []
    for ring in request["rings"]:
        if ring["start_m"] >= request["domain"]["gun_exit_m"]:
            continue
        z = .5*(ring["start_m"]+ring["stop_m"])
        if z <= 0:
            continue
        free_radius = min(row["inner_m"] for row in request["rings"] if row["start_m"] <= z <= row["stop_m"])
        for fraction in (0., .1):
            main.append((fraction*free_radius, 0., z))
        near_wall.append((.98*free_radius, 0., z))
    return {"axis_and_offaxis": np.unique(main, axis=0),
            "near_electrode_rim_98_percent": np.unique(near_wall, axis=0)}


def _planes(gun):
    values = [("extractor", gun.extractor.mechanical_center_from_tip_mm),
              ("gun_lens", gun.electrostatic_lens.mechanical_center_from_tip_mm),
              ("dpa_aperture", gun.dpa_aperture.z_mm)]
    values.extend((f"accelerator_stage_{index+1}", stage.center_from_tip_mm)
                  for index, stage in enumerate(gun.accelerator.stages))
    values.append(("gun_exit", gun.exit_plane_z_mm))
    return [(name, float(z)*1e-3) for name, z in values if 0 < z <= gun.exit_plane_z_mm]


def run_case(case, args, output):
    state = inputs(case)
    gun = state.electron_gun
    selected = emission_samples(gun, 193, args.samples)
    prepared = requests(gun, cells=args.cells_per_bore, refined_cells=args.refined_cells_per_bore,
                        extended_stop_mm=args.extended_stop_mm)
    report = {"case": case, "inputs": capture_instrument_snapshot(state).to_dict(),
              "source": selected, "requests": prepared, "variants": {}, "field_comparisons": {}}
    if args.describe:
        return report
    output.mkdir(parents=True, exist_ok=True)
    magnetic = prepare_magnetic_scene(state, z_limits_mm=(0., float(gun.exit_plane_z_mm)))
    fields, trajectories = {}, {}
    planes = _planes(gun)
    report["observation_planes_m"] = dict(planes)
    for name, item in prepared.items():
        if case == "curved":
            from temsim.physics.continuous_gun_field import build_continuous_gun_field as build
        else:
            from temsim.physics.closed_gun_field import build_closed_gun_field as build
        started = time.perf_counter()
        # The rim-risk case changes an unassigned aperture, not the scalar
        # boundary. Reuse matching field arrays across cases by request digest.
        field = build(gun, **item["options"], cache_dir=output.parent/"fields")
        # The continuous potential reference avoids JIT interpolation too.
        if hasattr(field, "_regular"):
            field._regular.compiled = False
        if request_digest(field.request) != item["request_sha256"]:
            raise RuntimeError("Prepared field identity changed during the comparison")
        fields[name] = field
        item["preparation_seconds"] = time.perf_counter()-started
        item["solver_report"] = field.report
        residual = field.report.get("linear_residual")
        item["reference_fixture_linear_residual_check"] = (None if residual is None else
            bool(float(residual) < REFERENCE_CHECKS["field_linear_residual"]))
        if hasattr(field, "launch_boundary_report"):
            item["launch_boundary"] = field.launch_boundary_report(np.array([row["position_m"] for row in selected["samples"]]))
            item["reference_fixture_launch_check"] = bool(item["launch_boundary"]["maximum_launch_potential_error_v"]
                < REFERENCE_CHECKS["curved_launch_potential_absolute_v"])
        scene = _scene(state, magnetic, field)
        budgets = (("", args.step_mm), ("_step_refined", args.step_mm/2)) if name.endswith("coarse") else (("", args.step_mm),)
        for suffix, step in budgets:
            variant = name+suffix
            rows = []
            for sample in selected["samples"]:
                settings = TestElectronSettings(kinetic_energy_ev=sample["kinetic_energy_ev"],
                    position_m=tuple(sample["position_m"]), polar_angle_deg=sample["polar_angle_deg"],
                    azimuth_angle_deg=sample["azimuth_angle_deg"], max_path_length_m=2*gun.exit_plane_z_mm*1e-3,
                    step_m=step*1e-3, max_steps=args.max_steps,
                    relative_tolerance=1e-5 if not suffix else 2.5e-6)
                started = time.perf_counter()
                result = trace_test_electron(scene, settings, use_compiled=False)
                measurements = trajectory_measurements(result, planes, scene._apertures)
                rows.append({**measurements, "ray_id": sample["ray_id"], "original_weight": sample["original_weight"],
                             "settings": asdict(settings), "seconds": time.perf_counter()-started})
                np.savez_compressed(output/f"{variant}-ray{sample['ray_id']}.npz",
                    position_m=result.positions_m, direction=result.directions,
                    time_s=result.time_s, path_length_m=result.path_length_m,
                    kinetic_energy_ev=result.kinetic_energy_ev, potential_v=result.electrostatic_potential_v,
                    momentum_kg_m_per_s=result.momentum_kg_m_per_s)
                print(json.dumps({"case": case, "variant": variant, "ray_id": sample["ray_id"],
                                  "reason": result.reason, "steps": result.steps, "seconds": rows[-1]["seconds"]}), flush=True)
            trajectories[variant] = rows
            report["variants"][variant] = rows
            (output/"partial.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    report["transport_comparisons"] = {name: compare_populations(trajectories[first], trajectories[second])
                                       for name, first, second in COMPARISON_PAIRS}
    for name, first, second in COMPARISON_PAIRS[:4]:
        report["field_comparisons"][name] = {label: field_difference(fields[first], fields[second], points)
            for label, points in _probes(prepared["production_coarse"]["request"]).items()}
    report["source_unchanged"] = selected == emission_samples(gun, 193, args.samples)
    if not report["source_unchanged"]:
        raise AssertionError("Comparison changed the tip emission")
    return report


def reanalyse(output):
    """Re-measure saved accepted arrays without field solving or propagation."""
    destination = output/"report.json"
    original = destination.read_bytes()
    report = json.loads(original)
    if report.get("schema") != "diagnostic-gun-domain-comparison-v1" or report.get("status") not in (
            "COMPARISON_COMPLETE_REVIEW_REQUIRED", "INCOMPLETE_TRANSPORT_REVIEW_REQUIRED"):
        raise ValueError("Reanalysis requires a finished comparison; it cannot edit a running or failed run")
    checksums = {}
    for case, data in report["cases"].items():
        if case not in CASES:
            raise ValueError("Unrecognised comparison input case")
        planes = tuple(data["observation_planes_m"].items())
        sources = {row["ray_id"]: row for row in data["source"]["samples"]}
        aperture_nodes = []
        for node in data["inputs"]["graph"]["nodes"]:
            if node["type"] not in ("temsim.optics.electron_gun.aperture:GunAperture", "temsim.optics.model:Aperture"):
                continue
            attributes = node["attributes"]
            z_mm = (attributes["z_mm"] if "z_mm" in attributes else
                    attributes["mechanical_center_from_tip_mm"]+attributes["field_center_offset_mm"])
            aperture_nodes.append(SimpleNamespace(key=attributes["key"], z_mm=z_mm,
                radius_mm=attributes["radius_mm"], offset_x_mm=attributes["offset_x_mm"], offset_y_mm=attributes["offset_y_mm"]))
        for variant, rows in data["variants"].items():
            if variant not in {name for pair in COMPARISON_PAIRS for name in pair[1:]}:
                raise ValueError("Unrecognised comparison variant")
            for row in rows:
                source = sources[row["ray_id"]]
                archive = output/case/f"{variant}-ray{row['ray_id']}.npz"
                before = hashlib.sha256(archive.read_bytes()).hexdigest()
                with np.load(archive, allow_pickle=False) as saved:
                    expected = {"position_m", "direction", "time_s", "path_length_m", "kinetic_energy_ev", "potential_v", "momentum_kg_m_per_s"}
                    if set(saved.files) != expected:
                        raise ValueError("Unexpected accepted-state archive arrays")
                    arrays = {name: np.array(saved[name], copy=True) for name in expected}
                size = row["steps"]+1
                for name, array in arrays.items():
                    shape = (size, 3) if name in ("position_m", "direction", "momentum_kg_m_per_s") else (size,)
                    if array.shape != shape or array.dtype != np.float64 or not np.isfinite(array).all():
                        raise ValueError("Invalid accepted-state archive shape, precision or values")
                if not np.array_equal(arrays["position_m"][0], source["position_m"]):
                    raise ValueError("Saved trajectory no longer matches its original tip position")
                if arrays["time_s"][0] != 0. or np.any(np.diff(arrays["time_s"]) < 0.):
                    raise ValueError("Saved trajectory chronology is invalid")
                trajectory = SimpleNamespace(positions_m=arrays["position_m"], directions=arrays["direction"],
                    time_s=arrays["time_s"], kinetic_energy_ev=arrays["kinetic_energy_ev"],
                    electrostatic_potential_v=arrays["potential_v"], reason=row["reason"],
                    completed=row["completed"], steps=row["steps"])
                # Read the original captured aperture definitions; current
                # defaults must not silently redefine a saved experiment.
                original_keys = {entry["key"] for entry in row["aperture_clearances"]}
                apertures = [item for item in aperture_nodes if item.key in original_keys]
                measurements = trajectory_measurements(trajectory, planes, apertures)
                revised_clearances = {entry["key"]: entry for entry in measurements.pop("aperture_clearances")}
                row["aperture_clearances"] = [revised_clearances.get(entry["key"], entry) for entry in row["aperture_clearances"]]
                row["reanalysed_aperture_clearance_keys"] = sorted(revised_clearances)
                row.update(measurements)
                after = hashlib.sha256(archive.read_bytes()).hexdigest()
                if before != after:
                    raise RuntimeError("Accepted-state archive changed during reanalysis")
                checksums[str(archive.relative_to(output))] = before
        data["transport_comparisons"] = {name: compare_populations(data["variants"][first], data["variants"][second])
                                         for name, first, second in COMPARISON_PAIRS}
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    previous = output/f"report-before-reanalysis-{timestamp}.json"
    previous.write_bytes(original)
    from temsim import diagnostic_gun_comparison
    report["measurement_reanalysis"] = {"utc": timestamp, "prior_report": previous.name,
        "prior_report_sha256": hashlib.sha256(original).hexdigest(), "raw_arrays_unchanged": True,
        "archive_sha256": checksums, "field_solves": 0, "transport_executions": 0,
        "measurement_source_sha256": hashlib.sha256(Path(diagnostic_gun_comparison.__file__).read_bytes()).hexdigest(),
        "current_source_files": source_file_hashes(),
        "original_trace_source_provenance": ("See original source_provenance; current hashes are not original execution hashes"
            if "source_provenance" in report else "NOT_CAPTURED: original run did not record per-file trace hashes; do not infer from current files or Git HEAD"),
        "measurement_change": "Terminal-roundoff measurement was corrected while the original run was active. This reanalysis reads saved accepted arrays without re-integrating; it does not certify unchanged runtime source.",
        "terminal_contact_rule": "Completed forward domain/hardware/aperture stop within 64 float64 spacings; actual endpoint and signed plane offset retained"}
    temporary = destination.with_suffix(".reanalysis.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps({"status": report["status"], "reanalysed": len(checksums), "raw_arrays_unchanged": True,
                      "domain_equivalence": "NOT_ESTABLISHED"}), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", choices=CASES, nargs="+", default=list(CASES))
    parser.add_argument("--samples", type=int, choices=range(1, 4), default=2)
    parser.add_argument("--cells-per-bore", type=int, default=8)
    parser.add_argument("--refined-cells-per-bore", type=int, default=12)
    parser.add_argument("--extended-stop-mm", type=float, default=3026.4)
    parser.add_argument("--step-mm", type=float, default=.1)
    parser.add_argument("--max-steps", type=int, default=40000)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--reanalyse", action="store_true", help="Re-measure completed saved trajectories; never solve or integrate")
    args = parser.parse_args()
    if args.reanalyse:
        return reanalyse(args.output)
    if not np.isfinite((args.step_mm, args.extended_stop_mm)).all() or args.step_mm <= 0:
        parser.error("Step and extended endpoint must be finite; step must be positive")
    if not 1 <= args.max_steps <= 200000:
        parser.error("Max steps must lie from 1 to 200000")
    if args.refined_cells_per_bore <= args.cells_per_bore:
        parser.error("Refined grid must have more cells per bore")
    args.cases = list(dict.fromkeys(args.cases))
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output/"report.json"
    if destination.exists():
        parser.error("Choose a new output folder to preserve previous evidence")
    report = {"schema": "diagnostic-gun-domain-comparison-v1", "status": "PREPARING",
        "started_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_provenance": {"before_execution": source_file_hashes(),
            "scope": "On-disk hashes at run boundaries; already imported code is not retroactively replaced by file edits"},
        "reference_checks_declared_before_execution": REFERENCE_CHECKS,
        "arguments": {**vars(args), "output": str(args.output.resolve())}, "cases": {},
        "scope": "Numerical-domain effect with the same diagnostic reference integrator and unchanged original tip population; not production-integrator equivalence",
        "limitations": ["Classical particles only; no wave or downstream source.",
            "Plane measurements interpolate accepted states; step refinement measures their resolution too.",
            "Near-aperture case is a detached rim-risk input; report actual stops and signed clearances.",
            "Two grid levels and two step budgets expose sensitivity, not asymptotic convergence proof.",
            "Reference fixture checks do not define full-gun domain equivalence or authorize replacing the production field."]}
    destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    try:
        with numerical_job(requested=1) as cpu:
            report["cpu_resources"] = cpu.to_dict()
            for case in args.cases:
                report["cases"][case] = run_case(case, args, args.output/case)
                destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        complete = all(row["completed"] for case in report["cases"].values()
                       for variant in case["variants"].values() for row in variant)
        report["status"] = ("REQUESTS_ONLY_NO_EXECUTION" if args.describe else
                            "COMPARISON_COMPLETE_REVIEW_REQUIRED" if complete else "INCOMPLETE_TRANSPORT_REVIEW_REQUIRED")
    except Exception:
        report.update(status="FAILED", traceback=traceback.format_exc())
        raise
    finally:
        report["source_provenance"]["after_execution"] = source_file_hashes()
        report["source_provenance"]["files_unchanged_at_run_boundaries"] = (
            report["source_provenance"]["before_execution"] == report["source_provenance"]["after_execution"])
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(destination.resolve()),
                     "domain_equivalence": "NOT_ESTABLISHED"}), flush=True)
    return 2 if report["status"] == "INCOMPLETE_TRANSPORT_REVIEW_REQUIRED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
