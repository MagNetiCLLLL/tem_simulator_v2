"""Execute the reference coherent surface through the round column prefix.

Numerical development evidence, not full TEM/STEM acceptance. No optical
settings, historical profiles or source widths are tuned by this driver.
Completed numerical checks stream to refinement.jsonl. Creating cancel.request
in the new output directory requests cooperative cancellation and diagnostics.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import traceback
from time import perf_counter

from threadpoolctl import threadpool_limits
from temsim.calculation_manifest import solver_source_identity
from temsim.optics.column import default_state
from temsim.optics.electron_gun.tip_surface import SurfaceCoherence, load_tip_surface_reference
from temsim.physics.surface_gun_wave import build_surface_gun_checkpoint
from temsim.physics.surface_wave import SurfaceWaveNumerics
from temsim.physics.radial_gun_wave import RadialGunNumerics
from temsim.physics.adaptive_scattering import AxialRefinement
from temsim.physics.radial_coordinates import RadialCoordinateBlend
from temsim.physics.wave_following_chart import WaveFollowingNumerics
from temsim.physics.occupied_axial_refinement import OccupiedAxialRefinement
from temsim.physics.radial_column_wave import round_column_prefix
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.immutable_json import freeze_json, thaw_json
import numpy as np


def _persist_report(output, report):
    """Publish diagnostics atomically before large failed frames are released."""
    temporary = output/"report.pending.json"
    temporary.write_text(json.dumps(thaw_json(freeze_json(report)), indent=2)+"\n", encoding="utf-8")
    temporary.replace(output/"report.json")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--maximum-radial-samples", type=int, default=8_388_608)
    p.add_argument("--stop", choices=("gun", "round-prefix"), default="round-prefix")
    p.add_argument("--radial-backend", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--direct-radial-phase", action="store_true")
    p.add_argument("--coordinate-width", type=float, default=.2,
                   help="Numerical Laguerre width / physical emitting-cap radius; not a source width")
    p.add_argument("--working-gib", type=float, default=16.)
    p.add_argument("--surface-radial-nodes", type=int, default=65)
    p.add_argument("--surface-axial-nodes", type=int, default=129)
    p.add_argument("--surface-domain", type=float, default=1.5)
    p.add_argument("--radial-modes", type=int, default=8)
    p.add_argument("--quadrature", type=int, default=48)
    p.add_argument("--gun-pixels", type=int, default=256)
    p.add_argument("--field-refinement", type=int, default=1)
    p.add_argument("--field-step", type=float, default=.5)
    p.add_argument("--relative-step", type=float, default=.1)
    p.add_argument("--axial-integrator", choices=("midpoint", "cf4", "adaptive_cf4"), default="midpoint")
    p.add_argument("--axial-tolerance", type=float, default=1e-6)
    p.add_argument("--adaptive-until-mm", type=float)
    p.add_argument("--diagnostic-planes-mm", type=float, nargs="*", default=[])
    p.add_argument("--coordinate-blend", type=float, nargs=3, metavar=("TARGET_WIDTH", "START_MM", "END_MM"),
                   help="Smooth numerical chart transition; no source or physical component is changed")
    p.add_argument("--timeout", type=float, default=600.)
    p.add_argument("--energy-cache-directory", type=Path,
                   help="Optional private executed-energy cache; read-only evidence exports are not imported")
    p.add_argument("--energy-cache-gib", type=float, default=192.)
    p.add_argument("--wave-following-iterations", type=int, default=0,
                   help="Development chart corrections; each re-executes the complete tip/gun problem")
    p.add_argument("--wave-following-width-multiplier", type=float, default=1.)
    p.add_argument("--wave-following-phase-order", type=int, choices=(2, 4), default=2)
    p.add_argument("--chart-smoothing-log-z", type=float, default=0.,
                   help="Numerical coordinate smoothing only; complete physical wave is re-solved and verified")
    p.add_argument("--occupied-axial-tolerance", type=float)
    p.add_argument("--occupied-refinement-strategy", choices=("local_sum", "global_embedded", "spatial_embedded"), default="local_sum")
    p.add_argument("--occupied-axial-rounds", type=int, default=12)
    p.add_argument("--occupied-mesh-seed", type=Path,
                   help="Numeric subdivision hints only; re-executes every field and convergence check")
    p.add_argument("--occupied-mesh-energy-ev", type=float,
                   help="Exact source emission energy to which the numerical seed applies; other modes execute normally")
    p.add_argument("--occupied-axial-evaluations", type=int, default=200_000)
    p.add_argument("--occupied-axial-workers", type=int, default=1)
    p.add_argument("--occupied-axial-executor", choices=("thread", "process"), default="thread")
    p.add_argument("--occupied-axial-integrator", choices=("cf4", "cf6"), default="cf4")
    p.add_argument("--occupied-budget-allocation", choices=("uniform", "initial_indicator"), default="uniform")
    p.add_argument("--refine-pilot-charts", action="store_true",
                   help="Also refine unpublished coordinate-selection pilots; the final physical solve is always checked")
    args = p.parse_args()
    from temsim.physics.axial_mesh_seed import read_mesh_seed
    mesh_seed = read_mesh_seed(args.occupied_mesh_seed)
    args.output.mkdir(parents=True, exist_ok=False)
    state = default_state()
    state.electron_gun.emitter.surface_model = replace(load_tip_surface_reference(), coherence=SurfaceCoherence())
    if args.field_refinement < 1:
        p.error("--field-refinement must be a positive integer")
    model = state.electron_gun.emitter.surface_model
    field = model.field_numerics
    state.electron_gun.emitter.surface_model = replace(model, field_numerics=replace(field,
        radial_nodes=field.radial_nodes*args.field_refinement,
        axial_nodes=field.axial_nodes*args.field_refinement,
        apex_cells_per_radius=field.apex_cells_per_radius*args.field_refinement))
    implementation = solver_source_identity()
    start = perf_counter()
    instrument = capture_instrument_snapshot(state)
    from temsim.immutable_json import json_digest
    report = {"scope": "DEVELOPMENT_ROUND_PREFIX_NOT_IMAGE_ACCEPTANCE", "implementation": implementation,
              "instrument_digest": instrument.digest, "instrument_snapshot": instrument.to_dict(),
              "instrument_settings_digest": json_digest({"graph": instrument.graph, "external_inputs": instrument.external_inputs})}
    from hashlib import sha256
    report["driver_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
    energy_cache = None
    if args.energy_cache_directory is not None:
        if not np.isfinite(args.energy_cache_gib) or args.energy_cache_gib <= 0:
            p.error("--energy-cache-gib must be positive and finite")
        from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
        energy_cache = ExecutedWaveStore(args.energy_cache_directory, instrument.digest,
                                         int(args.energy_cache_gib*1024**3))
    last_progress = [0., None]
    current_mode = [None]
    def progress(done, total, text):
        if text == "Building coupled physical tip/gun mode":
            current_mode[0] = done
        elapsed = perf_counter()-start
        if elapsed-last_progress[0] >= 10 or text != last_progress[1]:
            print(f"{elapsed:.2f}s {done}/{total}: {text}", flush=True)
            last_progress[:] = elapsed, text
    def record_refinement(method, row):
        data = {"scope": "IN_PROGRESS_NUMERICAL_CHECK_NOT_SOURCE", "mode_index": current_mode[0],
                "method": method, "elapsed_s": perf_counter()-start, "diagnostic": thaw_json(row)}
        with (args.output/"refinement.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(data, allow_nan=False)+"\n")
    progress.record_refinement = record_refinement
    def cancelled():
        return perf_counter()-start > args.timeout or (args.output/"cancel.request").exists()
    def completed_mode(*values):
        from scripts.surface_mode_evidence import preserve_mode
        receipt = preserve_mode(args.output, *values)
        report.setdefault("completed_modes", []).append(receipt)
        print(f"Completed energy mode {receipt['mode_index']+1}: complete complex state saved; not full-source acceptance", flush=True)
    try:
        with threadpool_limits(1):
            checkpoint, near = build_surface_gun_checkpoint(state.electron_gun,
                surface=SurfaceWaveNumerics(element_order=2, radial_nodes=args.surface_radial_nodes,
                    axial_nodes=args.surface_axial_nodes, outer_radius_factor=args.surface_domain,
                    joint_radial_phase=not args.direct_radial_phase),
                radial=RadialGunNumerics(radial_modes=args.radial_modes, potential_quadrature=args.quadrature,
                    relative_axial_step=args.relative_step, field_step_mm=args.field_step,
                    coordinate_width_over_cap_radius=args.coordinate_width,
                    axial_integrator=args.axial_integrator,
                    axial_refinement=AxialRefinement(tolerance=args.axial_tolerance),
                    adaptive_until_mm=args.adaptive_until_mm,
                    diagnostic_planes_mm=tuple(args.diagnostic_planes_mm),
                    coordinate_blend=None if args.coordinate_blend is None else RadialCoordinateBlend(*args.coordinate_blend),
                    wave_following=WaveFollowingNumerics(iterations=args.wave_following_iterations,
                        width_multiplier=args.wave_following_width_multiplier, phase_order=args.wave_following_phase_order,
                        smoothing_log_z_width=args.chart_smoothing_log_z),
                    occupied_refinement=OccupiedAxialRefinement(enabled=args.occupied_axial_tolerance is not None,
                        initial_mesh=mesh_seed,
                        initial_mesh_energy_ev=args.occupied_mesh_energy_ev,
                        strategy=args.occupied_refinement_strategy, maximum_rounds=args.occupied_axial_rounds,
                        tolerance=.001 if args.occupied_axial_tolerance is None else args.occupied_axial_tolerance,
                        maximum_evaluations=args.occupied_axial_evaluations,
                        workers=args.occupied_axial_workers,
                        executor=args.occupied_axial_executor,
                        integrator=args.occupied_axial_integrator,
                        error_budget_allocation=args.occupied_budget_allocation,
                        refine_pilot_charts=args.refine_pilot_charts,
                        maximum_working_bytes=int(args.working_gib*1024**3)),
                    maximum_working_bytes=int(args.working_gib*1024**3)),
                grid_pixels=args.gun_pixels,
                column_state=state, cancelled=cancelled, progress_callback=progress,
                _mode_completed=completed_mode, _energy_cache=energy_cache)
            report["gun"] = {"digest": checkpoint.digest, "current_a": checkpoint.transmitted_current_a,
                "record": checkpoint.record,
                "near_flux": [dict(mode.flux) for mode in near.modes],
                "mode_records": [dict(row) for row in checkpoint.record["mode_records"]],
                "unresolved_side_fraction": checkpoint.record["unresolved_side_fraction"],
                "radial_output_modes": [dict(row) for row in checkpoint.record["radial_output_modes"]]}
            arrays = {"near_r_nm": near.radius_nm, "near_z_nm": near.z_nm, "near_potential_v": near.potential_rise_v}
            for index, (mode, near_mode) in enumerate(zip(checkpoint.beam.modes, near.modes)):
                arrays[f"gun_mode_{index}"] = mode.plane.amplitude
                arrays[f"gun_basis_{index}_m"] = mode.plane.basis_m
                arrays[f"gun_curvature_{index}_m1"] = mode.plane.curvature_m1
                arrays[f"near_mode_{index}"] = near_mode.amplitude
            np.savez(args.output/"complex_states.npz", **arrays)
            print(json.dumps({key: report["gun"][key] for key in ("digest", "current_a", "unresolved_side_fraction")}), flush=True)
            report["status"] = "DIAGNOSTIC_GUN_ONLY_NOT_IMAGES"
            if args.stop == "round-prefix":
                z, modes, record = round_column_prefix(state, checkpoint, state.sample.z_mm,
                    maximum_samples=args.maximum_radial_samples, backend=args.radial_backend,
                    progress_callback=progress, cancelled=cancelled)
                report["prefix"] = record
                report["stop_z_mm"] = z
                report["mode_probability"] = [wave.probability for mode, wave in modes]
                report["status"] = "ROUND_PREFIX_COMPLETE_NOT_IMAGES"
    except Exception as error:
        report["status"] = "FAILED"
        report["error"] = str(error)
        if hasattr(error, "refinement_diagnostic"):
            report["refinement_diagnostic"] = error.refinement_diagnostic
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], flush=True)
        report["elapsed_s"] = perf_counter()-start
        report["implementation_unchanged"] = solver_source_identity() == implementation
        report["diagnostics_saved_before_failed_buffer_cleanup"] = True
        _persist_report(args.output, report)
    report["elapsed_s"] = perf_counter()-start
    report["implementation_unchanged"] = solver_source_identity() == implementation
    _persist_report(args.output, report)
    return int(report["status"] == "FAILED")


if __name__ == "__main__":
    raise SystemExit(main())
