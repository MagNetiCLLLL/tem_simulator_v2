"""Reproducible Si[110] CIF STEM acquisition through the production column.

The default action only prepares inputs and measures the actual ray probe.
``--mode benchmark`` runs a labelled 4x4 timing sample; ``--mode run`` acquires
the requested 64x64 raster at 0.02 nm. No incident coordinates/slopes are edited.
Physical sequential HAADF/DF/BF recording planes are used by acquire_stem_scan.
Finite-grid multislice and an explicitly reported structure-derived Rutherford
high-angle extension remain separate contributions in the saved raw data.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter, sleep
from types import SimpleNamespace
import tomllib
import tomli_w

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tifffile

from temsim.assembly_catalog import AssemblyCatalog
from temsim.detector.stem_signal import acquire_stem_scan, measure_sample_current, physical_angular_detectors
from temsim.operating_modes import apply_operating_mode_pair
from temsim.optics.column import default_state
from temsim.physics.beam_statistics import branch_sample_statistics
from temsim.physics.simulation import run
from temsim.profile_io import save_profile
from temsim.specimen.geometry import quaternion_from_zone_axes, set_sample_orientation


DETECTORS = ("haadf", "df", "bf")
OUTPUT = ROOT / "outputs" / "si110_cif_5nm_64px_002nm"


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path, value):
    path.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _hardware():
    result = {"platform": platform.platform(), "cpu": platform.processor(), "logical_cpus": os.cpu_count(),
              "python": sys.version, "numpy": np.__version__, "cupy_installed": importlib.util.find_spec("cupy") is not None}
    result["scientific_packages"] = {
        item.metadata["Name"]: item.version for item in importlib.metadata.distributions()
        if item.metadata.get("Name", "").lower().startswith(("cupy", "cuda", "nvidia", "numpy", "scipy", "numba", "abtem", "ase"))}
    try:
        result["gpu_inventory"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"],
            text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        result["gpu_inventory"] = "Unavailable"
    return result


def _implementation_sources():
    folders = (ROOT / "src" / "temsim" / "physics", ROOT / "src" / "temsim" / "specimen", ROOT / "src" / "temsim" / "detector")
    files = sorted(path for folder in folders for path in folder.rglob("*.py"))
    files += [ROOT / "src" / "temsim" / "optics" / "model.py", ROOT / "configs" / "operating_modes" / "catalog.toml"]
    result = {"sha256_by_file": {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}}
    try:
        result["git_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        result["git_revision"] = None
    return result


def build_inputs(args, output):
    from ase.io import read
    source = args.cif.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"The explicit input CIF is missing: {source}")
    source_bytes = source.read_bytes()
    unit = read(source)
    if set(unit.get_chemical_symbols()) != {"Si"}:
        raise ValueError("This acquisition script expects the requested elemental Si CIF")
    cif_copy = output / "input_Si.cif"
    shutil.copyfile(source, cif_copy)
    state = default_state()
    apply_operating_mode_pair(state, "nano_probe", "diffraction")
    if args.recording_calibration is not None:
        calibration = json.loads(args.recording_calibration.read_text(encoding="utf-8"))
        permitted = {"diffraction_lens", "intermediate_lens", "projector_lens_1", "projector_lens_2"}
        if set(calibration["strengths"]) != permitted:
            raise ValueError("Recording calibration must specify exactly D/I/P1/P2 strengths")
        for lens in state.lenses:
            if lens.key in permitted:
                lens.percent = float(calibration["strengths"][lens.key])
    if not np.isclose(state.beam_voltage_kv, 300.0):
        raise ValueError("The calibrated default assembly must supply the requested 300 kV beam")
    sample = state.sample
    sample.inserted = True
    sample.specimen_mode = "atomic"
    sample.specimen_preset_key = "si_110"
    sample.cif_path = str(cif_copy.resolve())
    sample.envelope_shape = "disk"
    sample.size_x_nm = sample.size_y_nm = 10.0
    sample.thickness_nm = 5.0
    sample.centre_x_nm = sample.centre_y_nm = 0.0
    sample.scan_origin_x_nm = sample.scan_origin_y_nm = 0.0
    sample.zone_axis_uvw = (1, 1, 0)
    sample.in_plane_axis_uvw = (1, -1, 0)
    set_sample_orientation(sample, quaternion_from_zone_axes(unit.cell.array, sample.zone_axis_uvw, sample.in_plane_axis_uvw))
    sample.stem_wave_enabled = True
    sample.wave_atomistic_enabled = True
    sample.wave_multislice_enabled = True
    sample.wave_grid_pixels = args.grid
    sample.wave_field_of_view_angstrom = 40.0
    sample.wave_slice_thickness_angstrom = 2.0
    sample.wave_bandwidth_fraction = 2.0 / 3.0
    sample.wave_frozen_phonon_enabled = True
    sample.wave_frozen_phonon_configurations = args.phonons
    sample.wave_frozen_phonon_seed = args.seed
    # Explicit displacement assumption for imported Si, not inferred from CIF.
    sample.wave_frozen_phonon_sigma_by_element_angstrom = {"Si": 0.085}
    sample.wave_frozen_phonon_sigma_angstrom = 0.0
    sample.wave_defocus_nm = 0.0
    sample.stem_poisson_enabled = False
    sample.stem_fourdstem_enabled = False
    sample.real_high_angle_tail_enabled = True
    # Fail visibly if the integrated structure-derived tail API is unavailable.
    for field in ("real_tail_material_source", "real_tail_screening_source"):
        if not hasattr(sample, field):
            if args.mode != "prepare":
                raise RuntimeError(f"Wait for integrated production API: Sample.{field} is not available")
        else:
            setattr(sample, field, "structure" if field == "real_tail_material_source" else "moliere")
    sample.real_tail_max_angle_mrad = 360.0
    state.acceleration_enabled = True
    state.acceleration_backend = args.backend
    # The stored nanoprobe preset was calibrated at 0.05 mm (validated 0.025),
    # not at the interactive default's 0.5 mm ray-integration step.
    state.step_mm = .05
    scan = state.ac_deflector
    scan.scan_pixels_x = scan.scan_lines = 64 if args.mode != "benchmark" else 4
    scan.scan_pixel_size_nm = .02
    scan.scan_frame_period_s = 1.0
    scan.wobble_enabled = False
    scan.scan_enabled = True
    state.descan_deflector.wobble_enabled = False
    # Keep the installed calibrated optical path, including its descan policy.
    for detector in state.stem_detectors:
        if detector.key in DETECTORS:
            detector.inserted = detector.readout_enabled = True
    selection = AssemblyCatalog().default_selection()
    # The source input files are archived, so later geometry/preset edits do
    # not erase which column and numerical defaults produced this acquisition.
    catalog = AssemblyCatalog()
    for relative in ("catalog.toml", *catalog.selected_paths(selection).values()):
        archived = output / "instrument_inputs" / relative
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(catalog.root / relative, archived)
    provenance = {
        "source_file": str(source), "archived_source_file": str(cif_copy.resolve()),
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
        "cell_vectors_angstrom": unit.cell.array, "cell_lengths_angstrom": unit.cell.lengths(),
        "expanded_unit_cell_atom_count": len(unit), "symbols": unit.get_chemical_symbols(),
        "zone_axis_uvw": sample.zone_axis_uvw, "in_plane_axis_uvw": sample.in_plane_axis_uvw,
        "orientation_quaternion_wxyz": sample.specimen_orientation_quaternion_wxyz,
        "frozen_phonon_one_axis_rms_angstrom": {"Si": .085}, "frozen_phonon_seed": args.seed,
        "frozen_phonon_rms_source": "Explicit Si near-room-temperature assumption retained from configs/specimens/10_si_110.toml: Loane, Xu and Silcox, Acta Cryst. A47 (1991), DOI 10.1107/S0108767391000375; not a displacement measured from the input CIF.",
    }
    if args.recording_calibration is not None:
        shutil.copyfile(args.recording_calibration, output / "recording_calibration.json")
    return state, selection, provenance


def _progress(output):
    started = perf_counter()
    last = [-100.0]
    def update(done, total, message):
        elapsed = perf_counter() - started
        if elapsed - last[0] < 20 and done != total:
            return
        last[0] = elapsed
        line = {"elapsed_s": elapsed, "done": done, "total": total, "message": message}
        with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(_jsonable(line), ensure_ascii=False) + "\n")
        print(json.dumps(_jsonable(line), ensure_ascii=False), flush=True)
    return update


def _save_images(frame, output, parameters):
    scan_x_nm = np.asarray(frame.scan_x_um) * 1e3
    scan_y_nm = np.asarray(frame.scan_y_um) * 1e3
    expected_side = parameters["scan_pixels"]
    if scan_x_nm.shape != (expected_side, expected_side) or scan_y_nm.shape != scan_x_nm.shape:
        raise ValueError("The production acquisition returned an unexpected raster shape")
    if not (np.allclose(np.diff(scan_x_nm, axis=1), .02, rtol=1e-6, atol=1e-9)
            and np.allclose(np.diff(scan_y_nm, axis=0), .02, rtol=1e-6, atol=1e-9)):
        raise ValueError("The calibrated physical raster does not reproduce the requested 0.02 nm pitch")
    extent = [scan_x_nm.min() - .01, scan_x_nm.max() + .01, scan_y_nm.min() - .01, scan_y_nm.max() + .01]
    arrays = {"scan_x_nm": scan_x_nm, "scan_y_nm": scan_y_nm}
    stats = {}
    tails = frame.high_angle_tail_fraction or {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), constrained_layout=True)
    for ax, key in zip(axes, DETECTORS):
        values = np.asarray(frame.fractions[key], dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < -1e-12):
            raise ValueError(f"{key} image contains invalid probabilities")
        tail = np.asarray(tails.get(key, np.zeros_like(values)), dtype=np.float64)
        arrays[f"{key}_fraction"] = values
        arrays[f"{key}_tail_fraction"] = tail
        arrays[f"{key}_coherent_scaled_fraction"] = values - tail
        low, high = float(values.min()), float(values.max())
        image16 = np.round(np.clip((values - low) / max(high - low, 1e-30), 0, 1) * 65535).astype(np.uint16)
        Image.fromarray(image16).save(output / f"{key}.png")
        tifffile.imwrite(output / f"{key}.tiff", values.astype(np.float32), metadata={
            "axes": "YX", "unit": "fraction_of_emitted_electrons", "pixel_size_nm": .02,
            "row_zero": "minimum sample y", "source_cif_sha256": parameters["specimen"]["sha256"]})
        stats[key] = {"min_fraction": low, "max_fraction": high, "mean_fraction": float(values.mean()),
                      "std_fraction": float(values.std()), "mean_tail_fraction": float(tail.mean()),
                      "tail_share_of_detector_signal": float(tail.sum() / max(values.sum(), 1e-30)),
                      "png_encoding": {"min_maps_to_0": low, "max_maps_to_65535": high, "bits": 16},
                      "tiff_encoding": "float32 unnormalised probability fraction"}
        rendered = ax.imshow(values, origin="lower", extent=extent, cmap="gray", interpolation="nearest", vmin=low, vmax=high)
        angles = parameters["physical_detector_reference_angles"][key]
        suffix = " (low-angle annulus)" if key == "df" else ""
        ax.set_title(f"{key.upper()} detector · {angles['inner_mrad']:.2f}–{angles['outer_mrad']:.2f} mrad{suffix}\n"
                     f"Tail share {stats[key]['tail_share_of_detector_signal']:.1%}", fontsize=10)
        ax.set_xlabel("Sample X (nm)")
        ax.set_ylabel("Sample Y (nm)")
        fig.colorbar(rendered, ax=ax, label="Emitted-electron probability fraction", shrink=.85)
    for field in ("uncollected_fraction", "absorbed_fraction", "truncated_fraction"):
        value = getattr(frame, field, None)
        if value is not None:
            arrays[field] = np.asarray(value)
    np.savez_compressed(output / "raw_scan.npz", **arrays)
    cutoff = frame.metrics.get("maximum_isotropic_angle_mrad", frame.metrics.get("maximum_scattering_angle_mrad"))
    cutoff = frame.metrics.get("detector_sampling", {}).get("maximum_simulated_angle_mrad", cutoff)
    mode_label = "4x4 TIMING SAMPLE — not the final acquisition" if parameters["mode"] == "benchmark" else "64 x 64, 0.02 nm / pixel"
    fig.suptitle(f"Si [110] CIF | 300 kV | 10 nm diameter, 5 nm thickness | {mode_label}\n"
                 f"{parameters['phonons']} frozen phonons; 2 Å slices; wave cutoff {cutoff:.1f} mrad + structure-derived Rutherford extension")
    fig.savefig(output / "haadf_df_bf_comparison.png", dpi=180)
    plt.close(fig)
    return stats


def _merge_ensemble(output, child_dirs, seeds, args, elapsed):
    frames = []
    child_metrics = []
    for folder in child_dirs:
        with np.load(folder / "raw_scan.npz") as archive:
            frames.append({key: archive[key].copy() for key in archive.files})
        child_metrics.append(json.loads((folder / "metrics.json").read_text(encoding="utf-8")))
    if len({tuple(sorted(frame)) for frame in frames}) != 1:
        raise ValueError("Independent configurations returned different observables")
    for key in ("scan_x_nm", "scan_y_nm"):
        if not all(np.array_equal(frame[key], frames[0][key]) for frame in frames):
            raise ValueError("Independent configurations must use the identical full physical raster")
    averaged = {key: np.mean([frame[key] for frame in frames], axis=0, dtype=np.float64) for key in frames[0]}
    metrics = dict(child_metrics[0]["production_metrics"])
    metrics.update({"specimen_configuration_count": len(seeds), "specimen_thermal_seed": None,
                    "specimen_thermal_seeds": seeds, "specimen_intensity_ensemble_average": True,
                    "ensemble_execution": "independent complete-raster single-configuration production acquisitions, averaged linearly",
                    "detector_configuration_relative_standard_error": {}})
    fractions = {key: averaged[f"{key}_fraction"] for key in DETECTORS}
    total = sum(fractions.values()) + averaged["uncollected_fraction"] + averaged["absorbed_fraction"]
    metrics["maximum_probability_conservation_error"] = float(np.max(np.abs(total - 1.0)))
    metrics["real_probability_conserved"] = metrics["maximum_probability_conservation_error"] <= 5e-10
    if not metrics["real_probability_conserved"]:
        raise ValueError("Averaged configurations failed probability conservation")
    frame = SimpleNamespace(fractions=fractions, high_angle_tail_fraction={key: averaged[f"{key}_tail_fraction"] for key in DETECTORS},
                            scan_x_um=averaged["scan_x_nm"] * 1e-3, scan_y_um=averaged["scan_y_nm"] * 1e-3,
                            metrics=metrics, uncollected_fraction=averaged["uncollected_fraction"],
                            absorbed_fraction=averaged["absorbed_fraction"], truncated_fraction=averaged.get("truncated_fraction"))
    parameters = json.loads((child_dirs[0] / "parameters.json").read_text(encoding="utf-8"))
    parameters.pop("state", None)
    parameters.update({"phonons": len(seeds), "command": [sys.executable, *sys.argv],
                       "ensemble_root_seeds": seeds, "ensemble_workers": args.workers,
                       "ensemble_sampling": "Independent isotropic Gaussian frozen-phonon configurations; full raster and potential ROI per worker, no spatial tiling",
                       "individual_profiles": [str((folder / "operating_profile.toml").resolve()) for folder in child_dirs],
                       "merged_profile_reproduction": "Equivalent sampling inputs only. Exact reproduction uses the listed individual profiles/seeds and this script; a single base seed with config_count=4 uses a different SeedSequence child set."})
    shutil.copyfile(child_dirs[0] / "input_Si.cif", output / "input_Si.cif")
    shutil.copytree(child_dirs[0] / "instrument_inputs", output / "instrument_inputs", dirs_exist_ok=True)
    parameters["specimen"]["archived_source_file"] = str((output / "input_Si.cif").resolve())
    profile = tomllib.loads((child_dirs[0] / "operating_profile.toml").read_text(encoding="utf-8"))
    profile["devices"]["sample"]["wave_frozen_phonon_configurations"] = len(seeds)
    profile["devices"]["sample"]["wave_frozen_phonon_seed"] = args.seed
    profile["devices"]["sample"]["cif_path"] = str((output / "input_Si.cif").resolve())
    (output / "operating_profile.toml").write_text(
        "# Equivalent ensemble inputs. For the exact realised average use each config_*/operating_profile.toml and the recorded individual seeds.\n"
        + tomli_w.dumps(profile), encoding="utf-8")
    _write_json(output / "parameters.json", parameters)
    statistics = _save_images(frame, output, parameters)
    standard_errors = {}
    for key in DETECTORS:
        samples = np.stack([raw[f"{key}_fraction"] for raw in frames])
        sem = np.std(samples, axis=0, ddof=1) / np.sqrt(len(frames))
        standard_errors[f"{key}_standard_error_fraction"] = sem
        relative = float(np.sqrt(np.sum(sem ** 2)) / max(np.sqrt(np.sum(fractions[key] ** 2)), 1e-30))
        metrics["detector_configuration_relative_standard_error"][key] = relative
        statistics[key]["configuration_relative_standard_error"] = relative
    np.savez_compressed(output / "ensemble_standard_error.npz", **standard_errors)
    combined = {"mode": "run", "elapsed_parallel_wall_s": elapsed, "scan_positions": 4096,
                "configuration_count": len(seeds), "individual_root_seeds": seeds,
                "sum_individual_acquisition_s": sum(item["elapsed_acquisition_s"] for item in child_metrics),
                "images": statistics, "production_metrics": metrics,
                "individual_metrics": [str((folder / "metrics.json").resolve()) for folder in child_dirs]}
    _write_json(output / "metrics.json", combined)
    (output / "README.md").write_text(
        "# Si [110] STEM: 64×64 at 0.02 nm\n\n"
        "This is the linear probability average of four independent, complete-raster, single-frozen-phonon production acquisitions. "
        "No image was synthesised and no spatial tiles were stitched. Each configuration has its own raw data, input CIF, profile and progress log. "
        f"Root seeds: {seeds}. Physical specimen: 10 nm diameter, 5 nm thickness; 300 kV; Si [110].\n\n"
        "`operating_profile.toml` restores equivalent ensemble settings in the App. It does not reproduce the exact four-seed realisation with one base seed. "
        "Exact reproduction uses the individual config_*/operating_profile.toml files or the command in parameters.json.\n\n"
        "TIFFs are float32 source-probability fractions. PNGs use a documented per-channel 16-bit min/max display mapping. "
        "raw_scan.npz preserves total, scaled coherent and Rutherford-tail signals separately. ensemble_standard_error.npz is the standard error across the four independent configurations. "
        "The comparison figure reports the actual wave cutoff and tail signal shares; the tail is a structure-derived screened Rutherford approximation, not high-angle multislice or Mott intensity. "
        "A four-configuration ensemble is not a convergence study. Explicit Si one-axis RMS=0.085 Å; no Poisson noise.\n",
        encoding="utf-8")
    print(json.dumps(_jsonable({"output": str(output), "elapsed_parallel_wall_s": elapsed, "images": statistics}), indent=2), flush=True)


def _run_ensemble(args, output):
    if args.mode != "run" or args.phonons < 2:
        raise ValueError("Parallel ensemble execution requires --mode run and at least two configurations")
    child_dirs = [output / f"config_{index + 1:02d}_seed_{args.seed + index}" for index in range(args.phonons)]
    seeds = [args.seed + index for index in range(args.phonons)]
    started = perf_counter()
    active = []
    completed = []
    pending = list(zip(child_dirs, seeds))
    environment = dict(os.environ, NUMBA_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    try:
        while pending or active:
            while pending and len(active) < args.workers:
                folder, seed = pending.pop(0)
                folder.mkdir(parents=True, exist_ok=True)
                stream = (folder / "run.log").open("w", encoding="utf-8")
                command = [sys.executable, str(Path(__file__).resolve()), "--mode", "run", "--grid", str(args.grid),
                           "--phonons", "1", "--seed", str(seed), "--workers", "1", "--backend", args.backend,
                           "--cif", str(args.cif.resolve()), "--output", str(folder)]
                if args.recording_calibration is not None:
                    command += ["--recording-calibration", str(args.recording_calibration.resolve())]
                process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                           env=environment, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                active.append((process, stream, folder, seed))
                print(json.dumps({"launched_seed": seed, "pid": process.pid, "output": str(folder)}), flush=True)
            remaining = []
            for process, stream, folder, seed in active:
                code = process.poll()
                if code is None:
                    remaining.append((process, stream, folder, seed))
                else:
                    stream.close()
                    if code != 0:
                        raise RuntimeError(f"Configuration seed {seed} failed; inspect {folder / 'run.log'}")
                    completed.append(seed)
            active = remaining
            if active:
                progress = {}
                for _process, _stream, folder, seed in active:
                    path = folder / "progress.jsonl"
                    if path.exists():
                        try:
                            progress[str(seed)] = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
                        except (ValueError, IndexError):
                            pass
                print(json.dumps(_jsonable({"elapsed_s": perf_counter() - started, "completed_seeds": completed, "progress": progress})), flush=True)
                sleep(20)
    finally:
        for process, stream, _folder, _seed in active:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=15)
            stream.close()
    _merge_ensemble(output, child_dirs, seeds, args, perf_counter() - started)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "benchmark", "run"), default="prepare")
    parser.add_argument("--cif", type=Path, default=ROOT / "configs" / "reference_samples" / "Si.cif")
    parser.add_argument("--grid", type=int, default=1024)
    parser.add_argument("--phonons", type=int, default=4)
    parser.add_argument("--seed", type=int, default=707)
    parser.add_argument("--workers", type=int, default=1, help="Parallel independent complete-raster single-configuration acquisitions; exact seeds are saved separately")
    parser.add_argument("--backend", choices=("Numba CPU", "CUDA GPU"), default="Numba CPU")
    parser.add_argument("--recording-calibration", type=Path, help="Output JSON from calibrate_si110_stem_recording.py; retains every real aperture")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.grid < 32 or args.phonons < 1 or args.seed < 0 or not 1 <= args.workers <= 8:
        parser.error("Grid must be at least 32; frozen-phonon configurations at least 1; seed nonnegative; workers between 1 and 8")
    output = (args.output or OUTPUT / args.mode).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "raw_scan.npz").exists():
        raise FileExistsError(f"An acquisition already exists in {output}; choose a fresh --output")
    if args.workers > 1:
        return _run_ensemble(args, output)
    started = perf_counter()
    state, selection, specimen = build_inputs(args, output)
    parameters = {"mode": args.mode, "created_utc": datetime.now(timezone.utc).isoformat(), "specimen": specimen,
                  "grid": args.grid, "fov_angstrom": 40., "phonons": args.phonons,
                  "slice_thickness_angstrom": 2., "nominal_slice_count": 25,
                  "bandwidth_fraction": 2 / 3, "beam_voltage_kv": 300.,
                  "scan_pixels": state.ac_deflector.scan_pixels_x, "scan_pixel_size_nm": .02,
                  "physical_sample_diameter_nm": 10., "physical_sample_thickness_nm": 5.,
                  "probe": "Production nano_probe/diffraction presets; unchanged propagated incident bundle",
                  "recording_calibration": None if args.recording_calibration is None else str(args.recording_calibration.resolve()),
                  "requested_backend": args.backend,
                  "tail": "Structure-derived multi-element screened Rutherford; Moliere screening; up to 360 mrad; approximate high-angle extension, not Mott or multislice signal",
                  "hardware": _hardware(), "command": [sys.executable, *sys.argv],
                  "implementation": _implementation_sources(),
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    _write_json(output / "parameters.json", parameters)
    print("Tracing the production incident beam (no probe override)...", flush=True)
    ray_started = perf_counter()
    simulation = run(state)
    ray_elapsed = perf_counter() - ray_started
    incident = branch_sample_statistics(simulation.incident)
    parameters["actual_probe"] = asdict(incident)
    parameters["incident_sample_fraction"] = measure_sample_current(simulation, state).fraction
    parameters["ray_calculation_elapsed_s"] = ray_elapsed
    _, angles = physical_angular_detectors(state, state.stem_detectors)
    parameters["physical_detector_reference_angles"] = {key: asdict(value) for key, value in angles.items()}
    parameters["state"] = state.to_dict()
    save_profile(output / "operating_profile.toml", state, selection)
    _write_json(output / "parameters.json", parameters)
    print(json.dumps(_jsonable({"alpha95_mrad": incident.convergence_95_mrad,
                               "waist_offset_nm": incident.waist_offset_m * 1e9,
                               "incident_sample_fraction": parameters["incident_sample_fraction"],
                               "ray_elapsed_s": ray_elapsed}), indent=2), flush=True)
    if args.mode == "prepare":
        return 0
    acquisition_started = perf_counter()
    frame = acquire_stem_scan(simulation, state, detector_keys=DETECTORS, progress_callback=_progress(output))
    if frame is None:
        raise RuntimeError("No specimen illumination; no STEM image was acquired")
    if frame.metrics.get("real_probability_conserved") is False:
        raise ValueError("The production acquisition failed its probability-conservation check")
    elapsed = perf_counter() - acquisition_started
    statistics = _save_images(frame, output, parameters)
    metrics = {"mode": args.mode, "elapsed_acquisition_s": elapsed, "elapsed_total_s": perf_counter() - started,
               "scan_positions": state.ac_deflector.scan_pixels_x ** 2, "images": statistics,
               "production_metrics": frame.metrics}
    metrics["interpretation"] = {
        "df": "The installed DF detector is a low-angle annulus inside the approximately 24.8 mrad illumination disk, not pure dark-field excluding the direct-beam disk.",
        "bf_sampling": "The small physical BF disk is sampled by a finite reciprocal grid; no angular/FOV or frozen-phonon convergence study is claimed.",
        "reciprocal_pixel_spacing_mrad_small_angle": float(frame.metrics["wavelength_angstrom"]) / float(frame.metrics["field_of_view_angstrom"]) * 1000}
    if args.mode == "benchmark":
        metrics["estimated_64x64_acquisition_s_linear_upper_estimate"] = elapsed * 4096 / 16
        metrics["estimate_scope"] = "Includes potential/plan setup scaled by 256, so pessimistic; excludes new hardware contention. This file is a timing sample, not the requested image."
    _write_json(output / "metrics.json", metrics)
    (output / "README.md").write_text(
        "# Si [110] STEM acquisition\n\n"
        + ("This is a 4×4 performance sample, not the requested final 64×64 image.\n\n" if args.mode == "benchmark" else "64×64 pixels at 0.02 nm; 300 kV; 10 nm disk diameter and 5 nm thickness.\n\n")
        + "The specimen is the archived user CIF, oriented [110] along the beam and [1 -1 0] along X. "
        "The actual production nanoprobe/diffraction preset is used with a 0.05 mm ray-integration step. No incident coordinates or slopes were edited.\n\n"
        "Open `operating_profile.toml` in the App to restore inputs. `raw_scan.npz` retains unnormalised source-probability fractions, "
        "coherent and tail contributions, and actual raster coordinates. TIFFs contain float32 fractions. "
        "The 16-bit PNGs map each channel's min/max to 0/65535; exact mapping is in metrics.json. "
        "The comparison figure uses independently labelled probability scales.\n\n"
        "Finite-grid elastic frozen-phonon multislice and the structure-derived screened Rutherford high-angle extension are separate models. "
        "The extension is not a Mott calculation or recovered high-angle multislice interference. "
        "The displayed cutoff and detector tail fractions disclose that distinction. Four configurations are a finite ensemble, not an established convergence study. "
        "The explicit Si RMS assumption is 0.085 Å and is not a thermal parameter measured from this CIF. No shot noise is added.\n\n"
        "Parameters, script/source hashes, physical detector angles and hardware are in parameters.json; "
        "production diagnostics and image statistics are in metrics.json. The archived instrument_inputs preserve the selected input TOMLs.\n",
        encoding="utf-8")
    print(json.dumps(_jsonable({"output": str(output), "elapsed_acquisition_s": elapsed, "images": statistics}), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
