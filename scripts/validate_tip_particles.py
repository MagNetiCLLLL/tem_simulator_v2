"""Reproducible tip-to-exit particle energy checks, keeping all gun elements.

This writes scalar reports and input settings, never wave/particle array caches.
It is an energy/transport check, not a spatial mesh-convergence certificate.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from temsim.optics.electron_gun.field_emission import FieldEmissionGun
from temsim.paths import INSTRUMENT_CONFIG_ROOT


def check_case(voltage_kv, extraction_kv, count, gun_lens_kv=None):
    gun = FieldEmissionGun()
    gun.accelerator.high_tension_kv = voltage_kv
    gun.extractor.voltage_kv = extraction_kv
    if gun_lens_kv is not None:
        gun.electrostatic_lens.voltage_kv = gun_lens_kv
    start = perf_counter()
    trace = gun.trace_to_exit(count)
    report = dict(trace.surface_model_report)
    report.update(elapsed_seconds=perf_counter()-start, emitted_particles=count,
                  emitted_current_a=trace.emitted_current_a,
                  exit_current_a=trace.c1_transmitted_current_a,
                  blocked_counts=dict(Counter(key for key in trace.blocked_key if key)),
                  inputs=gun.to_dict())
    # These points straddle the old analytic support windows. The actual
    # full-domain field is sampled; no artificial interpolation is added here.
    sample_z_mm = np.array([.01, 1., 6., 10., 12., 22., 26., 30., 40.])
    xyz = np.column_stack((np.zeros((len(sample_z_mm), 2)), sample_z_mm*.001))
    field = gun.electric_field
    report["connected_field_samples"] = [
        {"z_mm": float(z), "potential_v": float(v), "electric_z_v_per_m": float(e)}
        for z, v, e in zip(sample_z_mm, field.potential_v_at_global_positions(xyz),
                           field.field_at_global_positions_v_per_m(xyz)[:, 2])]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=9)
    parser.add_argument("--voltages-kv", nargs="+", type=float, default=[300.])
    parser.add_argument("--extraction-kv", type=float, default=4.)
    parser.add_argument("--gun-lens-kv", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = INSTRUMENT_CONFIG_ROOT / "gun/FEG.toml"
    from temsim.shared_tip import dependencies
    result = {"scope": "classical tip-to-exit particles; coherent imaging paused",
              "manifest": str(manifest), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "shared_definitions": {str(path): hashlib.sha256(data).hexdigest()
                                     for path, data in dependencies(manifest).items()},
              "mesh_convergence_certified": False, "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for voltage in args.voltages_kv:
        print(f"Tracing {args.count} particles: HT {voltage:g} kV, extraction {args.extraction_kv:g} kV", flush=True)
        case = check_case(voltage, args.extraction_kv, args.count, args.gun_lens_kv)
        result["cases"].append(case)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Exit: {case['transmitted_particle_count']} particles; max energy error {case['maximum_exit_energy_error_ev']} eV; blocked {case['blocked_counts']}", flush=True)
    if any(case["exit_energy_status"] != "verified" for case in result["cases"]):
        raise SystemExit("No particles reached the exit in at least one case; report retained, energy acceptance not demonstrated")


if __name__ == "__main__":
    main()
