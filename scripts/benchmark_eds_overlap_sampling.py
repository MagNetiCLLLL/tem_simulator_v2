"""Compare EDS-only overlap integration on one real cached broad ray bundle.

The cache has no full state: its final phase space is reanchored to the intact
saved baseline assembly. This is a disclosed hybrid diagnostic, not a recovery
of the user's full calculation. The original elastic transport is computed once.
"""
from dataclasses import asdict, fields
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import math
import shutil
from time import perf_counter

import diagnose_eds_cached_incident as d

np = d.np
ROOT = d.ROOT


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def terminal_digest(terminal):
    digest = hashlib.sha256()
    for field in fields(terminal):
        value = np.asarray(getattr(terminal, field.name))
        digest.update(field.name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/eds_overlap_sampling/broad_cached_raw_comparison")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), output / Path(__file__).name)
    references = json.loads((ROOT / "outputs/eds_defocus_diagnostic/recent_actual_cached_incident.json").read_text())
    manifest_path = Path(references[1]["path"])
    manifest = json.loads(manifest_path.read_text())
    def load(name):
        return np.load(manifest_path.parent / manifest["arrays"]["incident." + name]["file"], mmap_mode="r", allow_pickle=False)
    rows = {key: np.asarray(load(key)[-1:]).copy() for key in ("x", "y", "tx", "ty")}
    rows.update({key: np.asarray(load(key)).copy() for key in ("alive", "blocked_z", "energy_offset_ev", "ray_weight")})
    rows["z"] = np.asarray(load("z")[-1:]).copy()
    np.savez_compressed(output / "original_cached_sample_plane.npz", **rows)
    state = d.default_state()
    profile = ROOT / "outputs/haadf_dpa_clearance/si110_haadf_dpa_12mm.toml"
    selection, values = d.read_profile(profile)
    assembly = d.AssemblyCatalog().apply(state, selection)
    assert d.apply_profile_values(state, values) == []
    shutil.copyfile(profile, output / "baseline_profile.toml")
    state.sample.size_x_nm = state.sample.size_y_nm = 10.0
    state.sample.thickness_nm = 5.0
    state.sample.centre_x_nm = state.sample.centre_y_nm = 0.0
    state.sample.envelope_shape = "disk"
    state.sample.inserted = True
    state.sample.eds_support_material_key = "vacuum"
    state.sample.eds_poisson_enabled = False
    state.sample.eds_overlap_sampling_enabled = False
    state.objective_lens.percent = 68.0
    state.ac_deflector.scan_enabled = False
    cached_z = float(rows["z"][0])
    rows["z"] = np.asarray([state.sample.z_mm])
    bundle = d.incident_rays_from_simulation(state, SimpleNamespace(incident=SimpleNamespace(**rows)))
    assert bundle.reaching_ray_count == 7635
    geometry = d.EDSDetectorArrayGeometry.from_part_data(assembly.part(d.EDS_DETECTOR_SYSTEM).data)
    started = perf_counter()
    last = [-20.0]
    phase = ["original elastic"]
    def progress(done, total, message):
        elapsed = perf_counter() - started
        if elapsed - last[0] < 10 and done != total:
            return
        last[0] = elapsed
        row = dict(elapsed_s=elapsed, phase=phase[0], done=done, total=total, message=message)
        with (output / "progress.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    original = d.simulate_elastic_point_transport(state, incident_rays=bundle.rays, stored_trajectory_count=49, progress_callback=progress)
    elastic_elapsed = perf_counter() - started
    terminal_before = terminal_digest(original.terminal_electrons)
    tracks_before = original.eds_tracks
    positions = np.asarray([ray.position_xy_nm for ray in bundle.rays])
    weights = np.asarray([ray.weight for ray in bundle.rays])
    np.savez_compressed(output / "incident_footprint.npz", position_xy_nm=positions, conditional_weights=weights)
    source_files = ["scripts/benchmark_eds_overlap_sampling.py", "scripts/diagnose_eds_cached_incident.py", "src/temsim/specimen/overlap_sampling.py", "src/temsim/specimen/elastic_transport.py", "src/temsim/detector/eds_signal.py", "src/temsim/detector/eds_photon_transport.py"]
    report = dict(
        scope="Actual broad cached final phase space plus explicit baseline Si/field/dose/detector geometry; not a reconstruction of the unknown cached state.",
        cached_manifest=str(manifest_path), cached_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        baseline_profile=str(profile), baseline_sample_z_mm=state.sample.z_mm, cached_sample_z_mm=cached_z,
        global_z_reanchor_mm=float(state.sample.z_mm)-cached_z, original_phase_space_xy_preserved=True,
        objective_percent=68.0, sample_diameter_nm=10.0, sample_thickness_nm=5.0, support="vacuum",
        emitted_ray_count=bundle.emitted_ray_count, reaching_ray_count=bundle.reaching_ray_count,
        survival_fraction=bundle.surviving_fraction, original_centroid_nm=bundle.original_centroid_nm,
        radial_rms_about_centroid_nm=float(np.sqrt(np.average(np.sum((positions-np.average(positions, axis=0, weights=weights))**2, axis=1), weights=weights))),
        minimum_raw_radius_nm=float(np.min(np.linalg.norm(positions, axis=1))),
        original_elastic_elapsed_s=elastic_elapsed, original_elastic_metrics=dict(original.metrics),
        original_terminal_sha256=terminal_before, original_eds_track_count=len(tracks_before),
        dwell_time_s=d.default_eds_dwell_time_s(state), source_electrons=d.default_eds_incident_electrons(state, d.default_eds_dwell_time_s(state)),
        dose_scope="Baseline profile current/dwell, not recovered user EDS settings; expected counts, Poisson sampling disabled.",
        source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_files if (ROOT/name).is_file()},
        cases={}, limitations=["Scott KDE with Gaussian tails is a source-density approximation, not proof of resolved physical overlap.", "Parent-conditioned directions and energies, thin specimen/weak local field guards apply.", "Installed detector photon quadrature order remains the profile default; this comparison does not establish photon angular or KDE-model convergence."])
    write_json(output / "diagnostic.json", report)
    spectra = {}
    for label, enabled, count in (("disabled", False, 256), ("points_256", True, 256), ("points_1024", True, 1024)):
        phase[0] = label
        state.sample.eds_overlap_sampling_enabled = enabled
        state.sample.eds_overlap_sampling_points = count
        before = perf_counter()
        spectrum = d.simulate_eds_point(state, geometry, incident_bundle=bundle, elastic_transport=original,
            x_nm=0.0, y_nm=0.0, photon_maximum_stored_paths=0, progress_callback=progress)
        assert spectrum.elastic_transport is original
        assert original.eds_tracks is tracks_before
        assert terminal_digest(original.terminal_electrons) == terminal_before
        assert np.all(np.isfinite(spectrum.expected_counts)) and np.all(spectrum.expected_counts >= 0)
        np.savez_compressed(output / (label + "_spectrum.npz"), energy_ev=spectrum.energy_bin_centres_ev, expected_counts=spectrum.expected_counts)
        spectra[label] = spectrum
        row = dict(enabled=enabled, requested_points=count, elapsed_s=perf_counter()-before,
            total_expected_counts=spectrum.total_expected_counts,
            positive_bins=int(np.count_nonzero(spectrum.expected_counts > 0)),
            poisson_zero_count_probability=math.exp(-spectrum.total_expected_counts),
            original_transport_reused=True, original_terminal_sha256=terminal_digest(original.terminal_electrons),
            original_eds_tracks_unchanged=True, eds_metrics=spectrum.metrics,
            photon_metrics=spectrum.photon_transport.metrics if spectrum.photon_transport is not None else None)
        report["cases"][label] = row
        report["elapsed_total_s"] = perf_counter()-started
        write_json(output / "diagnostic.json", report)
        print(json.dumps({key: value for key,value in row.items() if key not in ("eds_metrics", "photon_metrics")}), flush=True)
    a = report["cases"]["points_256"]["total_expected_counts"]
    b = report["cases"]["points_1024"]["total_expected_counts"]
    report["relative_total_difference_256_vs_1024"] = (a/b-1) if b else None
    report["original_transport_computation_count"] = 1
    report["elapsed_total_s"] = perf_counter()-started
    write_json(output / "diagnostic.json", report)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, ax = plt.subplots(figsize=(8.5, 4.5), layout="constrained")
    for label, spectrum in spectra.items():
        ax.step(spectrum.energy_bin_centres_ev/1000, spectrum.expected_counts, where="mid",
            label=f"{label}: total {spectrum.total_expected_counts:.6g} counts")
    ax.set(xlim=(0, 2.2), xlabel="Photon energy (keV)", ylabel="Expected counts / 10 eV bin", ylim=(0,None),
        title="EDS-only overlap integration · actual broad rays + baseline Si 10 × 5 nm\nRaw incident coordinates; one unchanged original elastic transport")
    ax.legend()
    figure.savefig(output / "expected_spectrum_comparison.png", dpi=170)
    plt.close(figure)
    print(json.dumps({"output":str(output), "elapsed_total_s":report["elapsed_total_s"], "relative_difference":report["relative_total_difference_256_vs_1024"]}), flush=True)


if __name__ == "__main__":
    main()
