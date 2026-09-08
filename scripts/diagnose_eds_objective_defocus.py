"""Bounded optical-ray/elastic-path diagnostic for EDS under objective defocus.

The saved profile is a reproducible baseline, not the user's unknown live state.
No multislice, STEM raster or EDS photon Monte Carlo is run.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np

from temsim.assembly_catalog import AssemblyCatalog
from temsim.optics.column import default_state
from temsim.physics.beam_statistics import branch_sample_statistics
from temsim.physics.simulation import run
from temsim.profile_io import apply_profile_values, read_profile
from temsim.specimen.elastic_transport import (
    ElasticTransportGeometry,
    incident_rays_from_simulation,
    simulate_elastic_point_transport,
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=ROOT / "outputs/haadf_dpa_clearance/si110_haadf_dpa_12mm.toml")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/eds_defocus_diagnostic/baseline")
    parser.add_argument("--counts", type=int, nargs="+", default=[49, 1000])
    parser.add_argument("--percent", type=float, nargs="+", default=[68.9801, 68.98055])
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "diagnostic.json").exists():
        raise FileExistsError("Choose a new output directory")
    started = perf_counter()
    state = default_state()
    catalog = AssemblyCatalog()
    selection, values = read_profile(args.profile)
    catalog.apply(state, selection)
    skipped = apply_profile_values(state, values)
    if skipped:
        raise ValueError(f"Profile has skipped fields: {skipped}")
    baseline_percent = float(state.objective_lens.percent)
    percentages = list(dict.fromkeys([baseline_percent, *args.percent]))
    shutil.copyfile(args.profile, output / "input_profile.toml")
    source_cif = Path(state.sample.cif_path).resolve()
    shutil.copyfile(source_cif, output / "input.cif")
    sample = state.sample
    sample.mode = "atomic"
    sample.cif_path = str(output / "input.cif")
    sample.inserted = True
    sample.envelope_shape = "disk"
    sample.size_x_nm = sample.size_y_nm = 10.0
    sample.thickness_nm = 5.0
    sample.centre_x_nm = sample.centre_y_nm = 0.0
    sample.eds_support_material_key = "vacuum"
    state.ac_deflector.scan_enabled = False
    state.ac_deflector.wobble_enabled = False
    state.ac_deflector.kick_x_mrad = state.ac_deflector.kick_y_mrad = 0.0
    # Public optical_only currently expects the GUI's descriptive quality tag.
    # This does not replace the profile's integration step or source quadrature.
    state._tuning_quality = "EDS diagnostic: saved step and explicit ray count"
    geometry = ElasticTransportGeometry.from_state(state)
    assert geometry.sample_size_xy_nm == (10.0, 10.0)
    assert geometry.sample_thickness_nm == 5.0 and geometry.sample_material is not None
    data = dict(
        scope="Reproducible saved-profile baseline, not the unknown live App state.",
        profile=str(args.profile.resolve()), baseline_percent=baseline_percent,
        requested_percentages=args.percent, deduplicated_percentages=percentages,
        point_scope="AC scan and wobble disabled. Incident centroid explicitly translated to specimen centre (0,0) by production point-acquisition API; unshifted centroid is also recorded.",
        sample=dict(shape=geometry.sample_envelope_shape, size_xy_nm=geometry.sample_size_xy_nm,
                    thickness_nm=geometry.sample_thickness_nm, material=asdict(geometry.sample_material),
                    support_material="vacuum", cif_sha256=hashlib.sha256((output / "input.cif").read_bytes()).hexdigest()),
        voltage_kv=state.beam_voltage_kv, integration_step_mm=state.step_mm,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
            "src/temsim/physics/simulation.py", "src/temsim/specimen/elastic_transport.py",
            "src/temsim/specimen/vector_field_transport.py", "src/temsim/specimen/scene.py",
            "src/temsim/specimen/geometry.py", "src/temsim/detector/eds_signal.py")}, cases=[])
    write_json(output / "diagnostic.json", data)
    for count in args.counts:
        state.electron_gun.emitter.ray_count = count
        previous = None
        for percent in percentages:
            case_started = perf_counter()
            state.objective_lens.percent = percent
            print(json.dumps(dict(starting_count=count, objective_percent=percent)), flush=True)
            simulation = run(state, optical_only=True, existing_simulation=previous)
            optical_seconds = perf_counter()-case_started
            previous = simulation
            bundle = incident_rays_from_simulation(state, simulation, target_x_nm=0., target_y_nm=0.)
            xy = np.array([ray.position_xy_nm for ray in bundle.rays])
            weights = np.array([ray.weight for ray in bundle.rays])
            radius = np.linalg.norm(xy, axis=1)
            unshifted_radius = np.linalg.norm(xy + bundle.original_centroid_nm, axis=1)
            ids = np.array([ray.source_ray_index for ray in bundle.rays])
            transport_started = perf_counter()
            result = simulate_elastic_point_transport(state, incident_rays=bundle.rays, stored_trajectory_count=min(len(bundle.rays), 49))
            transport_seconds = perf_counter()-transport_started
            hit_ids = {track.source_ray_index for track in result.eds_tracks if track.source_key == "sample" and track.path_length_nm > 0}
            hit_mask = np.isin(ids, list(hit_ids))
            ray_paths = {int(key):0. for key in ids}
            for track in result.eds_tracks:
                if track.source_key == "sample":
                    ray_paths[track.source_ray_index] += track.path_length_nm
            case = dict(count_requested=count, objective_percent=percent, baseline=(percent==baseline_percent),
                emitted_count=bundle.emitted_ray_count, reaching_count=bundle.reaching_ray_count,
                reaching_emitted_weight=bundle.surviving_fraction,
                original_centroid_nm=bundle.original_centroid_nm, target_centroid_nm=bundle.target_centroid_nm,
                spot_rms_about_centroid_nm=float(np.sqrt(np.sum(weights*radius**2))),
                minimum_radius_nm=float(radius.min()), maximum_radius_nm=float(radius.max()),
                unshifted_minimum_radius_nm=float(unshifted_radius.min()),
                midpoint_inside_radius_5nm_count=int((radius<=5).sum()),
                midpoint_inside_radius_5nm_conditional_weight=float(weights[radius<=5].sum()),
                sample_material_hit_count=len(hit_ids), sample_material_hit_conditional_weight=float(weights[hit_mask].sum()),
                sample_material_hit_emitted_weight=float(weights[hit_mask].sum()*bundle.surviving_fraction),
                sample_material_path_weighted_nm=float(sum(track.path_length_nm*track.electron_weight for track in result.eds_tracks if track.source_key=="sample")),
                sample_material_path_max_nm=max(ray_paths.values(),default=0.),
                eds_track_count=len(result.eds_tracks), elastic_metrics=result.metrics,
                beam_statistics=asdict(branch_sample_statistics(simulation.incident)),
                optical_metrics={key:value for key,value in simulation.metrics.items() if any(word in key for word in ("reuse", "prefix", "backend", "defocus"))},
                elapsed_optical_s=optical_seconds, elapsed_transport_s=transport_seconds,
                elapsed_case_s=perf_counter()-case_started)
            tag=f"rays_{count}_objective_{percent:.8f}"
            np.savez_compressed(output/f"{tag}.npz", source_ray_index=ids, xy_nm=xy, conditional_weight=weights,
                direction=np.array([ray.direction for ray in bundle.rays]), material_hit=hit_mask,
                material_path_nm=np.array([ray_paths[int(key)] for key in ids]))
            data["cases"].append(case)
            data["elapsed_s"] = perf_counter()-started
            write_json(output/"diagnostic.json",data)
            print(json.dumps({key:value for key,value in case.items() if key not in ("elastic_metrics", "beam_statistics", "optical_metrics")}),flush=True)
    print(json.dumps(dict(complete=True, output=str(output), elapsed_s=perf_counter()-started)),flush=True)


if __name__ == "__main__":
    main()
