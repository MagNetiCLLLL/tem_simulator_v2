"""Single-threaded production gun validation; all generated arrays stay local.

Execute the ordinary unmodified source, gun fields, magnetic controls and
interception. The optional history recorder copies accepted phase states at
the existing history boundary, before their public display resampling.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from temsim.cpu_resources import numerical_job
from temsim.optics.electron_gun.field_emission import FieldEmissionGun
from temsim.optics.electron_gun import tracing
from temsim.physics.relativistic_lorentz import momentum_from_kinetic_energy_ev


def array_digest(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        value = np.ascontiguousarray(value)
        digest.update(str((value.shape, value.dtype.str)).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def emission_arrays(emitted):
    return tuple(getattr(emitted, key) for key in
                 ("x_m", "y_m", "tx_rad", "ty_rad", "energy_offset_ev", "weight", "ray_id"))


@contextmanager
def record_history(enabled):
    recorded = {}
    original = tracing._finalize_gun_history
    if enabled:
        def capture(positions, momenta, times, alive, completed, **kwargs):
            recorded.update(position_m=np.asarray(positions), momentum_kg_m_per_s=np.asarray(momenta),
                            time_s=np.asarray(times), alive=np.asarray(alive), completed=np.asarray(completed))
            return original(positions, momenta, times, alive, completed, **kwargs)
        tracing._finalize_gun_history = capture
    try:
        yield recorded
    finally:
        tracing._finalize_gun_history = original


def run_case(gun, count, label, output, *, capture=False):
    before = gun.emit(count)
    source_hash = array_digest(*emission_arrays(before))
    started = time.perf_counter()
    with record_history(capture) as recorded:
        result = gun.trace_to_exit(count)
    seconds = time.perf_counter()-started
    after_hash = array_digest(*emission_arrays(gun.emit(count)))
    if source_hash != after_hash:
        raise AssertionError("The executed calculation changed source emission")
    field_report = result.electrostatic_model_report["electrostatic_field"]
    payload = {
        "label": label, "particle_count": count, "trace_step_mm": gun.trace_step_mm,
        "seconds": seconds, "trace_cache_reuse_seconds": None,
        "trace_cache_same_object": None, "trace_cache_check": "pending",
        "source_sha256_before": source_hash, "source_sha256_after": after_hash,
        "source_positions_angles_energies_weights_ids_unchanged": True,
        "arrivals": int(np.count_nonzero(result.exit_bundle.alive)),
        "transmitted_weight": float(np.sum(result.exit_bundle.weight[result.exit_bundle.alive])),
        "field_request_sha256": field_report["request_sha256"],
        "field_report": field_report,
        "electrostatic_report": result.electrostatic_model_report,
        "minimum_exit_offset_ev": float(np.min(result.exit_bundle.energy_offset_ev[result.exit_bundle.alive])),
        "maximum_exit_offset_ev": float(np.max(result.exit_bundle.energy_offset_ev[result.exit_bundle.alive])),
        "blocked_keys": sorted(set(key for key in result.blocked_key if key)),
    }
    if capture:
        file = output/f"{label}-accepted-history.npz"
        np.savez_compressed(file, **recorded, ray_id=before.ray_id, weight=before.weight,
                            source_x_m=before.x_m, source_y_m=before.y_m,
                            source_tx_rad=before.tx_rad, source_ty_rad=before.ty_rad,
                            source_energy_offset_ev=before.energy_offset_ev)
        payload["accepted_history_file"] = str(file.resolve())
        payload["accepted_history_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
    report_path = output/f"{label}.json"
    report_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": "trace_completed_and_saved", "label": label,
                      "seconds": seconds, "arrivals": payload["arrivals"]}), flush=True)
    if gun._cache_key(count) != gun._trace_cache_key:
        payload["trace_cache_check"] = "skipped_inputs_or_implementation_changed_after_execution"
    else:
        started = time.perf_counter()
        repeated = gun.trace_to_exit(count)
        payload["trace_cache_reuse_seconds"] = time.perf_counter()-started
        payload["trace_cache_same_object"] = repeated is result
        payload["trace_cache_check"] = "verified" if repeated is result else "failed"
    report_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    if payload["trace_cache_check"] == "failed":
        raise AssertionError("Identical completed gun calculation did not reuse its trace")
    print(json.dumps({key: payload[key] for key in
          ("label", "particle_count", "seconds", "arrivals", "trace_cache_reuse_seconds")}), flush=True)
    return payload, result, recorded


def compare(first, second, energy_ev):
    left, right = first.exit_bundle, second.exit_bundle
    if not np.array_equal(left.alive, right.alive):
        raise AssertionError("Refined step changed the exit population")
    hit = left.alive
    positions = [np.column_stack((item.x_m[hit], item.y_m[hit])) for item in (left, right)]
    momenta = [momentum_from_kinetic_energy_ev(energy_ev+item.energy_offset_ev[hit],
               np.column_stack((item.tx_rad[hit], item.ty_rad[hit], np.ones(np.count_nonzero(hit)))))
               for item in (left, right)]
    delta = np.linalg.norm(momenta[1]-momenta[0], axis=1)
    result = {
        "exit_population_identical": True,
        "maximum_exit_xy_difference_m": float(np.max(np.linalg.norm(positions[1]-positions[0], axis=1))),
        "maximum_exit_radial_difference_m": float(np.max(np.abs(np.linalg.norm(positions[1], axis=1)-np.linalg.norm(positions[0], axis=1)))),
        "maximum_exit_slope_difference_rad": float(np.max(np.hypot(right.tx_rad[hit]-left.tx_rad[hit], right.ty_rad[hit]-left.ty_rad[hit]))),
        "maximum_exit_momentum_relative_difference": float(np.max(delta/np.linalg.norm(momenta[1], axis=1))),
        "maximum_exit_energy_difference_ev": float(np.max(np.abs(right.energy_offset_ev[hit]-left.energy_offset_ev[hit]))),
        "maximum_exit_flight_time_difference_s": float(np.max(np.abs(right.flight_time_s[hit]-left.flight_time_s[hit]))),
        "physical_plane_positions": [],
        "scope": "Actual integrated plane crossings and endpoint momentum; no display-path smoothing or matching.",
    }
    for a, b in zip(first.plane_arrivals, second.plane_arrivals):
        valid = a.reached & b.reached
        difference = np.hypot(a.x_m[valid]-b.x_m[valid], a.y_m[valid]-b.y_m[valid])
        result["physical_plane_positions"].append({"key": a.key, "z_mm": a.z_mm,
            "same_reached_population": bool(np.array_equal(a.reached, b.reached)),
            "maximum_xy_difference_m": float(difference.max(initial=0.))})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("tmp/gun-repair-20260926"))
    parser.add_argument("--large-count", type=int, default=5000)
    parser.add_argument("--mode", choices=("all", "large", "steps"), default="all")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with numerical_job(requested=1) as cpu:
        gun = FieldEmissionGun().validate()
        started = time.perf_counter()
        field = gun.base_electric_field
        field_seconds = time.perf_counter()-started
        started = time.perf_counter()
        gun.trace_to_exit(9)  # Warm the actual production numerical kernels.
        warmup_seconds = time.perf_counter()-started
        if args.mode == "large":
            run_case(gun, args.large_count, f"flat{args.large_count}-step0p2", args.output)
            return
        reports = []
        first, coarse, coarse_history = run_case(gun, 193, "flat193-step0p2", args.output, capture=True)
        reports.append(first)
        if args.mode == "all":
            large, _, _ = run_case(gun, args.large_count, f"flat{args.large_count}-step0p2", args.output)
            reports.append(large)
        gun.trace_step_mm = .1
        fine, refined, fine_history = run_case(gun, 193, "flat193-step0p1", args.output, capture=True)
        reports.append(fine)
        if first["source_sha256_before"] != fine["source_sha256_before"]:
            raise AssertionError("Step refinement changed the original source")
        if first["field_request_sha256"] != fine["field_request_sha256"]:
            raise AssertionError("Field implementation/mesh changed during step comparison; rerun after edits finish")
        outcome = {"schema": "production-closed-gun-validation-v1", "cpu_resources": cpu.to_dict(),
            "field_preparation_seconds": field_seconds, "kernel_warmup_trace_seconds": warmup_seconds,
            "field_shared_across_cases": field is gun.base_electric_field,
            "cases": reports, "step_comparison": compare(coarse, refined, gun.nominal_exit_energy_ev),
            "limits": ["Classical prescribed flat source; coherent work remains paused.",
                       "This validates the complete gun and its physical handoff, not a specimen/detector simulation.",
                       "Accepted raw histories are retained locally; plotting samples do not establish convergence."]}
        (args.output/"production-transport.json").write_text(json.dumps(outcome, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(outcome["step_comparison"], indent=2), flush=True)


if __name__ == "__main__":
    main()
