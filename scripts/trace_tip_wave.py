"""Export a development tip-to-detector calculation from an operating profile.

This does not enable TEM/STEM admission, fit a downstream source, or change the
profile. Archives are inspectable numeric evidence; they are not live sources.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import zipfile
from pathlib import Path
from time import perf_counter

import numpy as np

from temsim.assembly_catalog import AssemblyCatalog
from temsim.immutable_json import thaw_json
from temsim.calculation_manifest import solver_source_identity
from temsim.optics.column import default_state
from temsim.optics.electron_gun.tip_coherence import TipWaveNumerics, generate_tip_emission
from temsim.physics.tip_gun_wave import GunWaveNumerics
from temsim.physics.tip_wave_pipeline import TipWaveRequest, simulate_tip_wave
from temsim.physics.wave_grid import WaveGridNumerics
from temsim.physics.wave_execution import WaveExecutionOptions, InelasticWaveNumerics, ScanWaveNumerics
from temsim.physics.tip_wave_scan import simulate_tip_scan, ScanResponseAccumulator
from temsim.detector.wave_readout import WaveReadoutOptions
from temsim.profile_io import apply_profile_values, read_profile, save_profile


def _stream_npz(path, arrays):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name, value in arrays:
            with archive.open(name+".npy", "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, np.asarray(value), allow_pickle=False)


def export_calculation(calculation, output, state, selection, elapsed_s):
    result, modes = calculation.checkpoint, []
    output.mkdir(parents=True, exist_ok=False)
    save_profile(output/"request.toml", state, selection)
    def arrays():
        for index, mode in enumerate(result.beam.modes):
            fields = []
            for name in ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad"):
                value = getattr(mode.plane, name)
                if value is not None:
                    fields.append(name)
                    yield f"mode_{index}_{name}", value
            modes.append({"index": index, "fields": fields, "mode_id": mode.mode_id,
                "energy_kev": mode.energy_kev, "tip_electron_weight": mode.weight_per_reference_electron,
                "axial_reference": None if mode.axial_reference is None else asdict(mode.axial_reference),
                "scattering_history": thaw_json(mode.scattering_history)})
            del mode
    _stream_npz(output/(calculation.request.stop+"_wave.npz"), arrays())
    receipt = {"schema": "tip-wave-development-export-v3", "status": "COMPUTED_DEVELOPMENT_SCOPE",
        "full_tem_stem": "NOT_VALIDATED; paraxial columns, arrival-time coils and material-model conditional waves",
        "elapsed_s": elapsed_s, "request": asdict(calculation.request), "instrument_digest": calculation.instrument_digest,
        "checkpoint_digest": result.digest, "plane_z_mm": result.plane_z_mm,
        "tip_current_a": result.reference_current_a, "plane_current_a": result.transmitted_current_a,
        "transmitted_probability": result.beam.total_weight, "modes": modes,
        "execution": thaw_json(result.record), "archive_use": "Read-only evidence; pipeline resumes only its dependency-bound executed cache"}
    if calculation.request.stop == "gun_exit":
        receipt["exit_current_a"] = result.transmitted_current_a
    if calculation.detector is not None:
        def readout_arrays():
            for name in ("x_mm", "y_mm", "optical_probability", "detected_probability"):
                value = getattr(calculation.detector, name)
                if value is not None:
                    yield name, value
            for index, mode in enumerate(calculation.detector.modes):
                for name in ("phase_rad", "phase_valid", "complex_cell_amplitude", "canonical_covariance"):
                    value = getattr(mode, name)
                    if value is not None:
                        yield f"mode_{index}_{name}", value
                del mode
        _stream_npz(output/"detector_readout.npz", readout_arrays())
        receipt["detector"] = thaw_json(calculation.detector.record)
    (output/"receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True, help="Operating profile with explicit FEG tip coherence")
    parser.add_argument("--output", type=Path, help="New directory for computed complex arrays and evidence")
    parser.add_argument("--describe", action="store_true", help="Inspect tip inputs without propagating or writing results")
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--energy-samples", type=int, default=9)
    parser.add_argument("--mode-tail", type=float, default=1e-6)
    parser.add_argument("--maximum-modes", type=int, default=1024)
    parser.add_argument("--field-step-mm", type=float, default=.05)
    parser.add_argument("--bore-step-mm", type=float, default=1.)
    parser.add_argument("--stop", choices=("gun_exit", "specimen_entrance", "specimen_exit", "detector"), default="gun_exit")
    parser.add_argument("--observables", nargs="+", choices=("intensity", "phase", "complex", "covariance"), default=["intensity"])
    parser.add_argument("--column-step-mm", type=float, default=.5)
    parser.add_argument("--no-wave-refinement", action="store_true", help="Reject unresolved operators without automatically refining the wave grid")
    parser.add_argument("--maximum-wave-grid", type=int, default=32768)
    parser.add_argument("--wave-budget-mb", type=int, default=72*1024, help="Estimated live column wave buffers; not total process memory")
    parser.add_argument("--detector-pixels", type=int, default=512)
    parser.add_argument("--detector", help="Installed detector component key; default is the first inserted detector with readout enabled")
    parser.add_argument("--readout-budget-mb", type=int, default=72*1024)
    parser.add_argument("--gun-budget-mb", type=int, default=8*1024)
    parser.add_argument("--ram-cache-gb", type=int, default=8)
    parser.add_argument("--disk-cache-gb", type=int, default=192)
    parser.add_argument("--cache-directory", default=".temsim-wave-cache")
    parser.add_argument("--segment-steps", type=int, default=128)
    parser.add_argument("--no-segmentation", action="store_true", help="Use in-memory column/stage execution; conditional inelastic trajectories still save slice checkpoints")
    parser.add_argument("--inelastic-method", choices=("zero_loss", "trajectories"), default="trajectories")
    parser.add_argument("--trajectories", type=int, default=32)
    parser.add_argument("--trajectory-seed", type=int, default=0)
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--scan-stride", type=int, default=1)
    parser.add_argument("--dwell-samples", type=int, default=1)
    parser.add_argument("--maximum-scan-positions", type=int, default=4096)
    parser.add_argument("--scan-frame", type=int, default=0)
    args = parser.parse_args(argv)
    if not args.describe and args.output is None:
        parser.error("--output is required unless --describe is selected")
    if not args.describe and args.output.exists():
        parser.error("Output already exists; choose a new directory")
    state = default_state()
    catalog = AssemblyCatalog()
    selection, values = read_profile(args.profile)
    catalog.apply(state, selection)
    skipped = apply_profile_values(state, values)
    if skipped:
        raise ValueError(f"Profile contains unsupported fields: {skipped}")
    source_numerics = TipWaveNumerics(grid_pixels=args.grid, energy_samples=args.energy_samples,
        mode_tail_tolerance=args.mode_tail, maximum_modes=args.maximum_modes).validate()
    numerics = GunWaveNumerics(field_step_mm=args.field_step_mm, bore_step_mm=args.bore_step_mm,
                              maximum_checkpoint_bytes=args.gun_budget_mb*1024**2).validate()
    request = TipWaveRequest(stop=args.stop, source=source_numerics, gun=numerics,
        column_step_mm=args.column_step_mm, detector_pixels=args.detector_pixels, detector_key=args.detector,
        wave_grid=WaveGridNumerics(automatic_refinement=not args.no_wave_refinement,
            maximum_pixels=args.maximum_wave_grid, maximum_working_bytes=args.wave_budget_mb*1024**2),
        maximum_readout_bytes=args.readout_budget_mb*1024**2,
        execution=WaveExecutionOptions(segmented=not args.no_segmentation, segment_steps=args.segment_steps,
            cache_directory=args.cache_directory, maximum_ram_cache_bytes=args.ram_cache_gb*1024**3,
            maximum_disk_cache_bytes=args.disk_cache_gb*1024**3),
        inelastic=InelasticWaveNumerics(method=args.inelastic_method, trajectories_per_mode=args.trajectories, seed=args.trajectory_seed),
        readout=WaveReadoutOptions(intensity="intensity" in args.observables, phase="phase" in args.observables,
            complex_amplitude="complex" in args.observables, covariance="covariance" in args.observables)).validate()
    emission = generate_tip_emission(state.electron_gun, source_numerics)
    scan_numerics = ScanWaveNumerics(stride=args.scan_stride, dwell_samples=args.dwell_samples,
        maximum_positions=args.maximum_scan_positions, frame_index=args.scan_frame).validate() if args.scan else None
    if args.describe:
        print(json.dumps({"status": "CONFIGURATION_ONLY_NOT_PROPAGATED", "tip": thaw_json(emission.parameters),
            "source_record": thaw_json(emission.record), "source_numerics": asdict(source_numerics),
            "gun_numerics": asdict(numerics), "request": asdict(request),
            "scan_numerics": None if scan_numerics is None else asdict(scan_numerics),
            "full_tem_stem": "NOT_VALIDATED; arrival-time scans and conditional material-model waves"}, indent=2))
        return 0

    last_progress = None
    def progress(done, total, label):
        nonlocal last_progress
        last_progress = {"progress": done, "total": total, "stage": label}
        print(json.dumps(last_progress), flush=True)

    start = perf_counter()
    implementation = solver_source_identity()
    try:
        if args.scan:
            samples = []
            responses = ScanResponseAccumulator()
            args.output.mkdir(parents=True, exist_ok=False)
            save_profile(args.output/"request.toml", state, selection)
            for sample in simulate_tip_scan(state, request, scan_numerics, progress_callback=progress):
                directory = args.output/f"row_{sample.row}_column_{sample.column}_dwell_{sample.dwell_index}"
                export_calculation(sample.calculation, directory, state, selection, perf_counter()-start)
                responses.add(sample)
                samples.append({"row": sample.row, "column": sample.column, "dwell": sample.dwell_index,
                    "tip_time_s": sample.tip_time_s, "dwell_weight": sample.dwell_weight, "directory": directory.name,
                    "received_weight": None if sample.calculation.detector is None else sample.calculation.detector.record.get("response_weight")})
                del sample
                (args.output/"scan_progress.json").write_text(json.dumps({"samples": samples, "complete": False}), encoding="utf-8")
            _stream_npz(args.output/"scan_response.npz", responses.arrays().items())
            (args.output/"scan_progress.json").write_text(json.dumps({"samples": samples, "complete": True}), encoding="utf-8")
            (args.output/"scan_receipt.json").write_text(json.dumps({"status": "COMPUTED_DEVELOPMENT_SCOPE", "samples": samples,
                "numerics": asdict(scan_numerics), "unsampled_pixels": "not computed; never zero-filled",
                "response": "dwell-weighted probability per emitted tip electron; NaN where intensity was not computed",
                "time_coordinates": "tip emission raster labels; actual coil values evaluated at energy-dependent arrival time",
                "statistics": "common inelastic random numbers across scan positions; no independent-error combination across dwell samples",
                "phase": "per position, dwell and environmental trajectory; no averaging of phases"}, indent=2), encoding="utf-8")
            print(json.dumps({"status": "COMPUTED_DEVELOPMENT_SCOPE", "samples": len(samples),
                "output": str(args.output.resolve()), "elapsed_s": perf_counter()-start}), flush=True)
            return 0
        calculation = simulate_tip_wave(state, request, progress_callback=progress)
    except (Exception, KeyboardInterrupt) as error:
        # A failed request is inspectable evidence, never a computed wave or
        # an input source. Do not lose its original settings/error position.
        causes, seen, cause = [], set(), error
        while cause is not None and id(cause) not in seen:
            seen.add(id(cause))
            causes.append({"type": type(cause).__name__, "message": str(cause)})
            cause = cause.__cause__ or cause.__context__
        failure = {"schema": "tip-wave-development-failure-v1",
            "status": "CANCELLED" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "FAILED",
            "elapsed_s": perf_counter()-start, "request": asdict(request),
            "implementation_at_start": implementation, "implementation_at_failure": solver_source_identity(),
            "error_type": type(error).__name__, "error": str(error), "causes": causes, "last_progress": last_progress,
            "requested_checkpoint_published": False,
            "archive_use": "Failure diagnostics only; contains no replacement electron source"}
        args.output.mkdir(parents=True, exist_ok=args.scan)
        save_profile(args.output/"request.toml", state, selection)
        (args.output/"failure.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        print(json.dumps({"status": failure["status"], "output": str(args.output.resolve()),
                          "error": str(error)}), flush=True)
        raise
    receipt = export_calculation(calculation, args.output, state, selection, perf_counter()-start)
    print(json.dumps({"status": receipt["status"], "full_tem_stem": receipt["full_tem_stem"],
                      "output": str(args.output.resolve()), "elapsed_s": receipt["elapsed_s"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
