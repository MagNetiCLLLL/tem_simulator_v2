"""Acquire an explicitly configured STEM raster from an operating profile.

Benchmark uses a representative regular grid containing the final raster's
extreme pixel centres (and its centre for odd benchmark sides). It is not the
requested full image. CUDA failure aborts instead of starting a CPU retry.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from time import perf_counter
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

from generate_si110_cif_stem_scan import _hardware, _implementation_sources, _write_json
from temsim.assembly_catalog import AssemblyCatalog
from temsim.detector.stem_signal import acquire_stem_scan, measure_sample_current, physical_angular_detectors
from temsim.optics.column import default_state
from temsim.physics.beam_statistics import branch_sample_statistics
from temsim.physics.compute_backend import choose_wave_backend, WAVE_BACKEND_CUPY
from temsim.physics.core import electron
from temsim.physics.record_plane import build_record_plane_plan
from temsim.physics.simulation import run
from temsim.physics.stem_wave_imaging import probe_focus_aberrations
from temsim.physics.wave_imaging import _weighted_ray_statistics
from temsim.physics.wave_sampling import plan_wave_sampling
from temsim.profile_io import apply_profile_values, read_profile, save_profile

DETECTORS = ("haadf", "df", "bf")


def progress_callback(output):
    started = perf_counter()
    last = [-100.]
    def update(done, total, message):
        # The production callback reports this before beginning its CPU retry.
        if "CPU" in message or message.startswith("STEM probes "):
            raise RuntimeError("CUDA-required acquisition aborted: " + message)
        now = perf_counter() - started
        if now - last[0] < 10 and done != total:
            return
        last[0] = now
        row = dict(elapsed_s=now, done=done, total=total, message=message)
        with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    return update


def save_images(frame, output, parameters):
    x = np.asarray(frame.scan_x_um) * 1e3
    y = np.asarray(frame.scan_y_um) * 1e3
    side = parameters["actual_scan_side"]
    step = parameters["actual_scan_step_nm"]
    assert x.shape == y.shape == (side, side)
    assert np.allclose(np.diff(x, axis=1), step, rtol=1e-6, atol=1e-9)
    assert np.allclose(np.diff(y, axis=0), step, rtol=1e-6, atol=1e-9)
    arrays = {"scan_x_nm": x, "scan_y_nm": y}
    statistics = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), constrained_layout=True)
    extent = (x.min() - step / 2, x.max() + step / 2,
              y.min() - step / 2, y.max() + step / 2)
    for key, ax in zip(DETECTORS, axes):
        values = np.asarray(frame.fractions[key], dtype=np.float64)
        tail = np.asarray((frame.high_angle_tail_fraction or {}).get(key, np.zeros_like(values)))
        if values.shape != x.shape or not np.isfinite(values).all() or np.min(values) < 0:
            raise ValueError(f"Invalid {key} detector array")
        arrays[key + "_fraction"] = values
        arrays[key + "_tail_fraction"] = tail
        arrays[key + "_coherent_scaled_fraction"] = values - tail
        low, high = float(values.min()), float(values.max())
        scaled = np.round(np.clip((values - low) / max(high - low, 1e-30), 0, 1) * 65535).astype(np.uint16)
        # PNG viewers place row zero at the top: show laboratory +Y upwards.
        Image.fromarray(np.flipud(scaled)).save(output / f"{key}.png")
        tifffile.imwrite(output / f"{key}.tiff", values.astype(np.float32), metadata={
            "axes": "YX", "row_zero": "minimum sample Y", "pixel_size_nm": step,
            "unit": "fraction_of_emitted_electrons"})
        statistics[key] = dict(min_fraction=low, max_fraction=high,
            mean_fraction=float(values.mean()), std_fraction=float(values.std()),
            mean_tail_fraction=float(tail.mean()),
            tail_share=float(tail.sum() / max(values.sum(), 1e-30)),
            png=dict(min_maps_to_0=low, max_maps_to_65535=high, row_zero="maximum sample Y"))
        art = ax.imshow(values, cmap="gray", origin="lower", interpolation="nearest", extent=extent)
        angle = parameters["physical_detector_reference_angles"][key]
        extra = "\nLow-angle annulus inside illumination disk" if key == "df" and angle["outer_mrad"] < parameters["probe"]["convergence_95_rad"] * 1000 else ""
        ax.set_title(f"{key.upper()} · {angle['inner_mrad']:.2f}–{angle['outer_mrad']:.2f} mrad{extra}\nTail share {statistics[key]['tail_share']:.1%}", fontsize=9)
        ax.set_xlabel("Sample X (nm)"); ax.set_ylabel("Sample Y (nm)")
        fig.colorbar(art, ax=ax, label="Fraction of emitted electrons", shrink=.8)
    for name in ("uncollected_fraction", "absorbed_fraction", "truncated_fraction"):
        value = getattr(frame, name, None)
        if value is not None:
            arrays[name] = np.asarray(value)
    budget = sum(arrays[key + "_fraction"] for key in DETECTORS) + arrays["uncollected_fraction"] + arrays["absorbed_fraction"]
    error = float(np.max(np.abs(budget - 1)))
    if error > 1e-10:
        raise ValueError(f"Probability budget failed: {error}")
    np.savez_compressed(output / "raw_scan.npz", **arrays)
    fp = f"{parameters['phonons']} frozen phonons, seed {parameters['seed']}" if parameters["phonons"] else "Static ordered CIF (no frozen phonons)"
    scope = "REPRESENTATIVE-POINT BENCHMARK" if parameters["mode"] == "benchmark" else "STEM acquisition"
    fig.suptitle(f"{scope} · {side} × {side}, {step:g} nm step · effective C1 {parameters['focus']['effective_defocus_mm'] * 1e6:.3f} nm\n"
                 f"{fp} · wave {frame.metrics['grid_pixels_x']} × {frame.metrics['grid_pixels_y']}, {frame.metrics['field_of_view_angstrom'] / 10:g} nm FOV")
    fig.savefig(output / "haadf_df_bf_comparison.png", dpi=180)
    plt.close(fig)
    return dict(images=statistics, probability_error=error)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("prepare", "benchmark", "run"), default="prepare")
    parser.add_argument("--effective-c1-nm", type=float, default=-100.)
    parser.add_argument("--grid", type=int, default=2048)
    parser.add_argument("--fov-angstrom", type=float, default=80.)
    parser.add_argument("--padding-factor", type=float, default=1.3)
    parser.add_argument("--scan-side", type=int, default=16)
    parser.add_argument("--step-nm", type=float, default=.08)
    parser.add_argument("--phonons", type=int, default=4, help="0 selects a static ordered CIF")
    parser.add_argument("--seed", type=int, default=707)
    parser.add_argument("--benchmark-side", type=int, default=3)
    args = parser.parse_args(argv)
    if min(args.grid, args.scan_side, args.benchmark_side) < 2 or args.phonons < 0 or args.seed < 0:
        parser.error("Positive grids/scan sides and nonnegative phonon count/seed are required")
    if not all(math.isfinite(v) for v in (args.effective_c1_nm, args.fov_angstrom, args.step_nm, args.padding_factor)) or min(args.fov_angstrom, args.step_nm) <= 0 or args.padding_factor < 0:
        parser.error("Finite positive FOV/pitch and nonnegative padding are required")
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    if (output / "parameters.json").exists():
        raise FileExistsError("Choose a fresh output directory")
    started = perf_counter()
    catalog = AssemblyCatalog(); selection, values = read_profile(args.profile)
    state = default_state(); catalog.apply(state, selection)
    skipped = apply_profile_values(state, values)
    if skipped:
        raise ValueError(f"Profile has unsupported fields: {skipped}")
    shutil.copyfile(args.profile, output / "input_profile.toml")
    source = Path(state.sample.cif_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Explicit profile CIF unavailable: {source}")
    shutil.copyfile(source, output / "input.cif")
    state.sample.cif_path = str(output / "input.cif")
    sample = state.sample
    sample.wave_grid_pixels = args.grid; sample.wave_field_of_view_angstrom = args.fov_angstrom
    sample.wave_probe_padding_factor = args.padding_factor
    sample.wave_frozen_phonon_enabled = args.phonons > 0
    sample.wave_frozen_phonon_configurations = max(1, args.phonons)
    sample.wave_frozen_phonon_seed = args.seed
    sample.stem_fourdstem_enabled = False; sample.stem_poisson_enabled = False
    state.acceleration_backend = "CUDA GPU"; state.acceleration_enabled = True
    backend, reason = choose_wave_backend("CUDA GPU", acceleration_enabled=True, work_items=1)
    if backend != WAVE_BACKEND_CUPY:
        raise RuntimeError(f"CUDA is required: {reason}")
    scan = state.ac_deflector
    actual_side = args.benchmark_side if args.mode == "benchmark" else args.scan_side
    step = (args.scan_side - 1) * args.step_nm / (actual_side - 1) if args.mode == "benchmark" else args.step_nm
    scan.scan_pixels_x = scan.scan_lines = actual_side
    scan.scan_pixel_size_nm = step; scan.scan_enabled = True; scan.wobble_enabled = False
    for detector in state.stem_detectors:
        if detector.key in DETECTORS:
            detector.inserted = detector.readout_enabled = True
    print("Tracing actual profile incident beam...", flush=True)
    simulation = run(state)
    ray_stats = _weighted_ray_statistics(simulation.incident)
    ray_defocus_nm = -ray_stats["waist_offset_m"] * 1e9
    state.probe_aberrations = dict(state.probe_aberrations or {})
    state.probe_aberrations.pop("c1_mm", None)
    sample.wave_defocus_nm = args.effective_c1_nm - ray_defocus_nm
    _, focus = probe_focus_aberrations(state, ray_stats)
    if not math.isclose(focus.effective_defocus_mm * 1e6, args.effective_c1_nm, abs_tol=1e-9):
        raise ValueError("Effective focus differs from requested target")
    probe = branch_sample_statistics(simulation.incident)
    _, _, wavelength_nm = electron(state)
    radius = max(1.22 * wavelength_nm / probe.convergence_95_rad,
                 abs(args.effective_c1_nm) * math.tan(probe.convergence_95_rad))
    required_fov = max(args.fov_angstrom, ((args.scan_side - 1) * args.step_nm + 2 * args.padding_factor * radius) * 10)
    sampling_plan = plan_wave_sampling(reference_fov_angstrom=args.fov_angstrom,
        reference_pixels=args.grid, requested_fov_angstrom=required_fov,
        thickness_angstrom=sample.thickness_nm * 10,
        target_slice_thickness_angstrom=sample.wave_slice_thickness_angstrom,
        configuration_count=max(args.phonons, 1))
    _, angles = physical_angular_detectors(state, state.stem_detectors)
    plan = build_record_plane_plan(state)
    parameters = dict(mode=args.mode, created_utc=datetime.now(timezone.utc).isoformat(),
        command=[sys.executable, *sys.argv], source_profile=str(args.profile.resolve()),
        source_cif=str(source), cif_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        requested_scan_side=args.scan_side, requested_step_nm=args.step_nm,
        actual_scan_side=actual_side, actual_scan_step_nm=step,
        representative_point_span_nm=(args.scan_side-1)*args.step_nm,
        phonons=args.phonons, seed=args.seed, configured_c1_nm=sample.wave_defocus_nm,
        focus=asdict(focus), probe=asdict(probe), incident_sample_fraction=measure_sample_current(simulation,state).fraction,
        sampling_plan=asdict(sampling_plan), padding_factor=args.padding_factor,
        physical_detector_reference_angles={key:asdict(value) for key,value in angles.items()},
        recording_planes=[dict(key=plane.key,z_mm=plane.z_mm) for plane in plan.planes],
        state=state.to_dict(), hardware=_hardware(), implementation=_implementation_sources(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        gpu_failure_policy="Abort in production progress callback before its CPU retry; require resident metrics.")
    _write_json(output / "parameters.json", parameters)
    save_profile(output / "operating_profile_raw.toml", state, selection)
    profile = tomllib.loads((output / "operating_profile_raw.toml").read_text())
    for key,field in (("simulation","last_gun_waist_mm"),("descan_deflector","wobble_enabled")):
        profile.get("devices",{}).get(key,{}).pop(field,None)
    (output / "operating_profile.toml").write_text(tomli_w.dumps(profile),encoding="utf-8")
    for relative in ("catalog.toml",*catalog.selected_paths(selection).values()):
        target=output / "instrument_inputs" / relative; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(catalog.root / relative,target)
    print(json.dumps(dict(configured_c1_nm=sample.wave_defocus_nm,effective_c1_nm=focus.effective_defocus_mm*1e6,
        alpha_mrad=probe.convergence_95_mrad,sampling_plan=asdict(sampling_plan))),flush=True)
    if args.mode == "prepare":
        return 0
    acquire_started=perf_counter()
    frame=acquire_stem_scan(simulation,state,detector_keys=DETECTORS,progress_callback=progress_callback(output))
    if frame is None or not frame.metrics.get("cuda_resident_pipeline"):
        raise RuntimeError("No resident CUDA STEM frame was returned")
    summary=save_images(frame,output,parameters)
    metrics=dict(elapsed_acquisition_s=perf_counter()-acquire_started,elapsed_total_s=perf_counter()-started,
        production_metrics=frame.metrics,**summary)
    _write_json(output / "metrics.json",metrics)
    (output / "README.md").write_text(
        f"{args.mode}: {actual_side} × {actual_side}, {step:g} nm step. Effective C1={args.effective_c1_nm:g} nm.\n\n"
        "Negative C1 denotes a converging probe whose geometric focus is downstream of the sample. "
        "This is an explicitly configured coherent-probe defocus; the traced incident rays and lens excitations are retained.\n\n"
        "NPZ/TIFF rows increase with sample Y. PNG rows are reversed for +Y-up viewing, matching the comparison figure. "
        "PNGs use independent linear min/max scaling; black is the image minimum, not zero electrons. "
        "The raw data retain unscaled emitted-electron fractions and coherent/tail components.\n\n"
        "Benchmark is representative centre/edge/corner sampling, not a full image. "
        "Static comparisons omit thermal displacements to isolate wave-window changes. "
        "All physical detector stops remain active. No CPU retry is permitted.\n",encoding="utf-8")
    print(json.dumps(dict(output=str(output),**{key:metrics[key] for key in ('elapsed_acquisition_s','images','probability_error')})),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
