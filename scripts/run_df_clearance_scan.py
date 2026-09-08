"""Recompute an archived Si STEM raster with an independently owned DF annulus.

No global instrument files are edited. Acquisition timing and lens strengths
come from the archived state; the 100 pA / 10 us Poisson exposure is readout only.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import tomli_w
import tifffile
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate_si110_cif_stem_scan import _hardware, _implementation_sources, _write_json
from run_stem_profile_scan import progress_callback, save_images
from temsim.assembly_catalog import AssemblyCatalog
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.detector.stem_signal import acquire_stem_scan, measure_sample_current, physical_angular_detectors
from temsim.optics.model import State
from temsim.physics.beam_statistics import branch_sample_statistics
from temsim.physics.compute_backend import choose_wave_backend, WAVE_BACKEND_CUPY
from temsim.physics.record_plane import build_record_plane_plan, prepare_record_plane_detector_masks
from temsim.physics.scan_geometry import calibrate_scan_system, paired_kick_response, raster_sample_grid
from temsim.physics.simulation import run
from temsim.physics.stem_wave_imaging import probe_focus_aberrations
from temsim.physics.wave_imaging import _weighted_ray_statistics
from temsim.profile_io import apply_profile_values, read_profile, save_profile

DETECTORS = ("haadf", "df", "bf")
ARCHIVE = ROOT / "outputs/si110_cif_5nm_64px_002nm/acquisition_gpu_1024_final"
ELECTRON_CHARGE = 1.602176634e-19


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_inputs(archive, output):
    original = json.loads((archive / "parameters.json").read_text())
    shutil.copyfile(archive / "parameters.json", output / "original_parameters.json")
    shutil.copyfile(archive / "operating_profile.toml", output / "original_operating_profile.toml")
    shutil.copyfile(archive / "input_Si.cif", output / "input_Si.cif")
    source_root = archive / "instrument_inputs"
    input_root = output / "instrument_inputs"
    shutil.copytree(source_root, input_root)
    shutil.copyfile(source_root / "catalog.toml", output / "original_catalog.toml")
    catalog_doc = tomllib.loads((input_root / "catalog.toml").read_text())
    removed = {}
    for group in ("gun_variants", "beam_blanker_variants", "column_variants", "project_and_recording_system_variants"):
        rows = catalog_doc.get(group, [])
        kept = [row for row in rows if (input_root / row["file"]).is_file()]
        removed[group] = [row for row in rows if row not in kept]
        if kept:
            catalog_doc[group] = kept
        else:
            catalog_doc.pop(group, None)
    catalog_doc["assembly"]["order"] = ["gun", "column", "project_and_recording_system"]
    (input_root / "catalog.toml").write_text(tomli_w.dumps(catalog_doc), encoding="utf-8")
    recording_file = input_root / "project_and_recording_system/EnergyFilter.toml"
    document = tomllib.loads(recording_file.read_text())
    df = next(part for part in document["parts"] if part["key"] == "df")
    before = deepcopy(df)
    df["inner_diameter_mm"] = 60.0
    df["outer_width_mm"] = 100.0
    # Keep any explicit aliases consistent, without inventing a new housing.
    for field in ("outer_diameter_mm", "mechanical_outer_diameter_mm"):
        if field in df:
            df[field] = 100.0
    if "mechanical_inner_diameter_mm" in df:
        df["mechanical_inner_diameter_mm"] = 60.0
    recording_file.write_text(tomli_w.dumps(document), encoding="utf-8")
    catalog = AssemblyCatalog(root=input_root)
    selection, values = read_profile(output / "original_operating_profile.toml")
    state = State.from_dict(original["state"])
    catalog.apply(state, selection, preserve_operating_parameters=True)
    skipped = apply_profile_values(state, values)
    if skipped:
        raise ValueError(f"Archived profile has unsupported fields: {skipped}")
    # Profile load retains operating fields. Geometry is independently owned by
    # this validated assembly, never by the process-default catalog.
    apply_physical_layout_to_state(state, assembly_root=input_root, preserve_operating_parameters=True)
    state.sample.cif_path = str((output / "input_Si.cif").resolve())
    state.sample.stem_fourdstem_enabled = False
    state.sample.stem_poisson_enabled = False
    state.sample.stem_wave_enabled = True
    state.acceleration_backend = "CUDA GPU"
    state.acceleration_enabled = True
    state.ac_deflector.scan_pixels_x = state.ac_deflector.scan_lines = 32
    state.ac_deflector.scan_pixel_size_nm = .04
    old_lenses = {row["key"]: row for row in original["state"]["lenses"]}
    lens_checks = {}
    for lens in state.lenses:
        if lens.key not in old_lenses:
            continue
        row = old_lenses[lens.key]
        lens_checks[lens.key] = {}
        for field in ("percent", "b0_t", "z_mm"):
            if field in row and hasattr(lens, field):
                actual = float(getattr(lens, field))
                wanted = float(row[field])
                lens_checks[lens.key][field] = dict(original=wanted, actual=actual)
                if not np.isclose(actual, wanted, rtol=1e-11, atol=1e-10):
                    raise ValueError(f"Archived lens changed: {lens.key}.{field} {wanted} -> {actual}")
    sample = state.sample
    for field, expected in (("size_x_nm", 10.), ("size_y_nm", 10.), ("thickness_nm", 5.),
                            ("wave_grid_pixels", 1024), ("wave_field_of_view_angstrom", 40.),
                            ("wave_frozen_phonon_configurations", 4), ("wave_frozen_phonon_seed", 707)):
        if getattr(sample, field) != expected:
            raise ValueError(f"Unexpected {field}={getattr(sample, field)}")
    if sample.wave_defocus_nm != original["state"]["sample"]["wave_defocus_nm"]:
        raise ValueError("Configured wave focus changed")
    archived_sample_z = original["state"]["sample"].get("z_mm")
    if archived_sample_z is not None and not np.isclose(sample.z_mm, archived_sample_z, rtol=0, atol=1e-10):
        raise ValueError("Archived sample plane changed")
    detector = next(d for d in state.stem_detectors if d.key == "df")
    if (detector.inner_diameter_mm, detector.outer_diameter_mm) != (60., 100.):
        raise ValueError(f"DF geometry was not applied: {detector}")
    provenance = dict(catalog_pruned_unarchived_modules=removed, df_before=before, df_after=df,
        lens_checks=lens_checks, source_cif_sha256=sha(output / "input_Si.cif"),
        original_parameters_sha256=sha(archive / "parameters.json"),
        source_archive=str(archive), acquisition_frame_period_s=state.ac_deflector.scan_frame_period_s,
        instrument_sha256={p.relative_to(input_root).as_posix(): sha(p) for p in input_root.rglob("*.toml")})
    _write_json(output / "input_provenance.json", provenance)
    return state, selection, catalog, original, provenance


def preflight_masks(state, simulation):
    calibrate_scan_system(state)
    xf, yf, times = raster_sample_grid(state.ac_deflector, maximum_count=None)
    factors = np.stack((xf, yf), axis=-1)
    kicks = np.einsum("ij,...j->...i", state.ac_deflector.scan_command_matrix_mrad, factors)
    response = 1e3 * paired_kick_response(state, state.ac_deflector, state.sample.z_mm)
    offsets_um = (kicks * 1e-3) @ response.T * 1e3
    base_kick = np.asarray(state.ac_deflector.scan_kick_mrad(float(getattr(state, "simulation_time_s", 0.))))
    base_um = (base_kick * 1e-3) @ response.T * 1e3
    stats = _weighted_ray_statistics(simulation.incident)
    origin_um = np.array([stats["mean_x_m"], stats["mean_y_m"]]) * 1e6 - base_um
    positions_m = (offsets_um + origin_um + np.array([state.sample.scan_origin_x_nm, state.sample.scan_origin_y_nm]) * 1e-3) * 1e-6
    plan = build_record_plane_plan(state, scan_times_s=times)
    azimuth = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    radii = np.r_[np.linspace(0, 27, 55), np.linspace(30, 51, 43)]
    # Absolute specimen angle coordinates, exactly as used by the wave router.
    angles = radii[:, None, None] * np.stack((np.cos(azimuth), np.sin(azimuth)), axis=-1)[None, ...] * 1e-3
    evaluate = prepare_record_plane_detector_masks(plan, angles.reshape(1, -1, 2))
    direct_hits = 0
    inner_band_hits = 0
    inner_band_total = 0
    angle_limits = []
    for first in range(0, 1024, 8):
        masks = evaluate(positions_m.reshape(-1, 2)[first:first + 8, None, :], scan_slice=slice(first, first + 8))
        hits = masks["df"].reshape(-1, len(radii), len(azimuth))
        direct_hits += int(hits[:, radii <= 27].sum())
        interior = hits[:, (radii >= 32) & (radii <= 48)]
        inner_band_hits += int(interior.sum())
        inner_band_total += int(interior.size)
        accepted = radii[np.any(hits, axis=(0, 2))]
        if accepted.size:
            angle_limits.append([float(accepted.min()), float(accepted.max())])
    if direct_hits:
        raise ValueError(f"DF still accepts direct angles <=27 mrad: {direct_hits}")
    if not inner_band_hits:
        raise ValueError("No 32-48 mrad rays reach DF through the actual sequential stops")
    return dict(direct_hits_at_or_below_27_mrad=direct_hits, tested_scan_points=1024,
        azimuth_samples=360, radial_samples_mrad=radii, interior_32_48_mrad_accepted_fraction=inner_band_hits / inner_band_total,
        sampled_accepted_limits_mrad=[min(x[0] for x in angle_limits), max(x[1] for x in angle_limits)],
        note="Finite polar sampling; all actual aperture and blocking detector surfaces included. No shadow compensation.",
        plan_fingerprint=plan.fingerprint, planes=[dict(key=p.key, z_mm=p.z_mm,
            j_diff_m_per_rad=t.j_diff_m_per_rad, j_img=t.j_img, position_offset_m=t.position_offset_m,
            stop=asdict(p)) for p, t in zip(plan.planes, plan.transfers)])


def archive_implementation(output, implementation):
    targets = list(dict.fromkeys(list((ROOT / "src/temsim").rglob("*.py")) + [Path(__file__),
        ROOT / "scripts/run_stem_profile_scan.py", ROOT / "scripts/generate_si110_cif_stem_scan.py", ROOT / "pyproject.toml"]
        + [ROOT / relative for relative in implementation["sha256_by_file"]]))
    path = output / "implementation_snapshot.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for source in targets:
            archive.write(source, source.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(path) as archive:
        for relative, expected in implementation["sha256_by_file"].items():
            if hashlib.sha256(archive.read(relative)).hexdigest() != expected:
                raise ValueError(f"Source changed while archiving: {relative}")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (output / "pip_freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    return dict(path=path.name, sha256=sha(path), source_hashes_verified=True)


def add_counting_readout(frame, output):
    current_pa, dwell_s, seed = 100., 10e-6, 42
    source_electrons = current_pa * 1e-12 * dwell_s / ELECTRON_CHARGE
    rng = np.random.default_rng(seed)
    with np.load(output / "raw_scan.npz") as raw:
        arrays = {key: raw[key] for key in raw.files}
    summary = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    x, y = arrays["scan_x_nm"], arrays["scan_y_nm"]
    extent = (x.min()-.02, x.max()+.02, y.min()-.02, y.max()+.02)
    for key, ax in zip(DETECTORS, axes):
        expected = np.asarray(frame.fractions[key]) * source_electrons
        counts = rng.poisson(expected)
        arrays[key + "_expected_electrons"] = expected
        arrays[key + "_poisson_counts"] = counts
        tifffile.imwrite(output / f"{key}_poisson_counts.tiff", counts.astype(np.uint32), metadata={
            "axes": "YX", "unit": "detected electrons / scan pixel", "pixel_size_nm": .04,
            "row_zero": "minimum sample Y", "source_current_pa": current_pa, "dwell_s": dwell_s, "seed": seed})
        high = max(int(counts.max()), 1)
        display = np.rint(counts / high * 65535).astype(np.uint16)
        Image.fromarray(np.flipud(display)).save(output / f"{key}_poisson_counts.png")
        art = ax.imshow(counts, cmap="gray", origin="lower", interpolation="nearest", extent=extent, vmin=0, vmax=high)
        ax.set_title(f"{key.upper()} · Poisson counts\nExpected mean {expected.mean():.3g} e− / pixel")
        ax.set_xlabel("Sample X (nm)"); ax.set_ylabel("Sample Y (nm)")
        fig.colorbar(art, ax=ax, label="Detected electrons / pixel")
        summary[key] = dict(expected_mean=float(expected.mean()), count_mean=float(counts.mean()),
            count_min=int(counts.min()), count_max=int(counts.max()), png_zero=0, png_max=high)
    fig.suptitle("32 × 32 STEM · 0.04 nm step · readout-only 100 pA, 10 µs/pixel, seed 42\n"
                 "Independent ideal Poisson electron counting; no detector efficiency or electronic noise model")
    fig.savefig(output / "haadf_df_bf_poisson_comparison.png", dpi=180)
    plt.close(fig)
    np.savez_compressed(output / "raw_scan.npz", **arrays)
    return dict(source_current_pa=current_pa, dwell_time_s=dwell_s, seed=seed,
        source_electrons_per_pixel=source_electrons, channel_rng_order=DETECTORS, images=summary,
        scope="Postprocessing of emitted-source-normalized fractions only; acquisition scan timing and incident beam are unchanged.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    state, selection, catalog, original, provenance = prepare_inputs(args.archive.resolve(), output)
    backend, reason = choose_wave_backend("CUDA GPU", acceleration_enabled=True, work_items=1)
    if backend != WAVE_BACKEND_CUPY:
        raise RuntimeError(f"CUDA unavailable: {reason}")
    print("Tracing archived physical illumination with independently owned DF geometry...", flush=True)
    simulation = run(state)
    stats = _weighted_ray_statistics(simulation.incident)
    _, focus = probe_focus_aberrations(state, stats)
    probe = branch_sample_statistics(simulation.incident)
    _, angles = physical_angular_detectors(state, state.stem_detectors)
    print(json.dumps(dict(effective_c1_nm=focus.effective_defocus_mm * 1e6, probe=asdict(probe),
        detector_angles={key: asdict(value) for key, value in angles.items()})), flush=True)
    masks = preflight_masks(state, simulation)
    _write_json(output / "mask_preflight.json", masks)
    parameters = dict(mode="run", created_utc=datetime.now(timezone.utc).isoformat(), command=[sys.executable, *sys.argv],
        actual_scan_side=32, actual_scan_step_nm=.04, phonons=4, seed=707,
        configured_c1_nm=state.sample.wave_defocus_nm, focus=asdict(focus), probe=asdict(probe),
        incident_sample_fraction=measure_sample_current(simulation, state).fraction,
        physical_detector_reference_angles={key: asdict(value) for key, value in angles.items()},
        acquisition_frame_period_s=state.ac_deflector.scan_frame_period_s, state=state.to_dict(),
        hardware=_hardware(), implementation=_implementation_sources(), script_sha256=sha(__file__),
        input_provenance=provenance, archive_probe_comparison=original.get("actual_probe"),
        gpu_failure_policy="Abort before CPU retry; require resident CuPy and four configurations.")
    parameters["implementation_archive"] = archive_implementation(output, parameters["implementation"])
    _write_json(output / "parameters.json", parameters)
    save_profile(output / "operating_profile_raw.toml", state, selection)
    profile = tomllib.loads((output / "operating_profile_raw.toml").read_text())
    for key, field in (("simulation", "last_gun_waist_mm"), ("descan_deflector", "wobble_enabled")):
        profile.get("devices", {}).get(key, {}).pop(field, None)
    (output / "operating_profile.toml").write_text(tomli_w.dumps(profile), encoding="utf-8")
    print(json.dumps(dict(preflight="passed", direct_hits=masks["direct_hits_at_or_below_27_mrad"],
        interior_transmission=masks["interior_32_48_mrad_accepted_fraction"], output=str(output))), flush=True)
    if args.prepare_only:
        return 0
    acquisition_started = perf_counter()
    frame = acquire_stem_scan(simulation, state, detector_keys=DETECTORS, progress_callback=progress_callback(output))
    elapsed = perf_counter() - acquisition_started
    if frame is None or not frame.metrics.get("cuda_resident_pipeline"):
        raise RuntimeError("A resident CUDA scan was not returned")
    for field, expected in (("grid_pixels_x", 1024), ("grid_pixels_y", 1024),
                            ("specimen_configuration_count", 4), ("cuda_configuration_count", 4)):
        if frame.metrics.get(field) != expected:
            raise ValueError(f"Actual {field}: {frame.metrics.get(field)} != {expected}")
    summary = save_images(frame, output, parameters)
    readout = add_counting_readout(frame, output)
    metrics = dict(elapsed_acquisition_s=elapsed, elapsed_total_s=perf_counter()-started,
        production_metrics=frame.metrics, readout=readout, **summary)
    _write_json(output / "metrics.json", metrics)
    _write_json(output / "final_acquisition_state.json", state.to_dict())
    (output / "README.md").write_text(
        "# Si[110] STEM with DF direct-disk clearance\n\n"
        "Fresh simultaneous HAADF/DF/BF CUDA multislice acquisition: 32 × 32 pixels, 0.04 nm pitch, 1.28 nm pixel FOV. "
        "The archived 300 kV illumination, focus, lens excitations and detector Z positions are retained. Only DF ID/OD are changed to 60/100 mm in an independent validated instrument catalog. "
        "The original catalog listed modules absent from its archive; the derived catalog lists only the three archived installed modules. No global TOML was edited.\n\n"
        "Si is the exact archived user CIF, diameter 10 nm, thickness 5 nm, [110], explicit 0.085 Å one-axis frozen-phonon RMS; 4 configurations, seed 707. Wave grid 1024², 4 nm FOV. "
        "Current production numerical code is archived here, including the corrected sample-boundary Rutherford overlap; this is a new calculation, not reuse of the historical integrated DF image. "
        "The coherent wave plus approximate screened-Rutherford high-angle tail and complete sequential physical stops are retained. No shadow correction is applied.\n\n"
        "mask_preflight.json records the physical Jdiff matrices and sampled acceptance at all raster positions; angles ≤27 mrad have no DF hits. Reference collection angles are specific to this archived camera-length calibration, not universal detector specifications. "
        "This 32-pixel raster is a preview at coarser spatial sampling than the historical 64-pixel image; it does not establish wave-grid or phonon convergence.\n\n"
        "raw_scan.npz contains source-normalized fractions, coherent/tail parts, loss budget and separate expected-electron/Poisson-count arrays. "
        "The 100 pA, 10 µs per-pixel, seed 42 counting exposure is readout-only and does not change the acquisition period or traced illumination. It assumes ideal independent electron detection and omits efficiency, electronic noise and saturation. "
        "NPZ/TIFF row zero is minimum Y; PNG row zero is maximum Y. Fraction PNGs use individual min/max; counts PNGs use zero-to-labelled-maximum. No smoothing or sharpening.\n\n"
        "Reproduce with the archived source/scripts, local .venv dependencies in pip_freeze.txt, and this script's --archive path pointing at the original self-contained acquisition archive. "
        "The exported profile must be loaded with this directory's instrument_inputs catalog to retain DF geometry.\n",
        encoding="utf-8")
    print(json.dumps(dict(completed=str(output), elapsed_acquisition_s=elapsed, probability_error=summary["probability_error"], images=summary["images"])), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
