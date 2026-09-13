"""Time full two-way solvers on a retained actual gun slab, not a source."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from temsim.calculation_manifest import solver_source_identity
from temsim.physics.scattering_load import covariant_carrier_slab


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slab", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=21)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 1000:
        parser.error("Choose 1 to 1000 repetitions")
    args.output.mkdir(parents=True, exist_ok=False)
    identity = solver_source_identity()
    digest = sha256(args.slab.read_bytes()).hexdigest()
    with np.load(args.slab, allow_pickle=False) as data:
        q, g = data["residual"], data["connection"]
        width, kappa, carrier = data["width_kappa_carrier"]
    report = {"scope": "Retained actual constant operator; NOT_SOURCE_ACCEPTANCE",
        "input_file": str(args.slab.resolve()), "input_sha256": digest,
        "solver_identity": identity, "blas_threads": 1, "repeats": args.repeats,
        "timing_caveat": "Other frozen source jobs are running on this host", "results": []}
    with threadpool_limits(1):
        for factor in (1., .5, -.25, .01):
            results, durations, records = {}, {"auto": [], "doubling": []}, {}
            for method in ("auto", "doubling"):
                results[method], records[method] = covariant_carrier_slab(q, g,
                    width*factor, kappa, carrier, mixed_coordinates=method)
            # Alternate order to avoid assigning a persistent warm-up/load bias.
            for repeat in range(args.repeats):
                for method in (("auto", "doubling") if repeat % 2 else ("doubling", "auto")):
                    started = perf_counter()
                    result, _ = covariant_carrier_slab(q, g, width*factor, kappa, carrier,
                                                      mixed_coordinates=method)
                    durations[method].append(perf_counter()-started)
                    np.testing.assert_array_equal(result, results[method])
            difference = float(np.linalg.norm(np.asarray(results["auto"])-results["doubling"]))
            medians = {method: float(np.median(values)) for method, values in durations.items()}
            report["results"].append({"width_factor": factor, "full_complex_difference": difference,
                "seconds": durations, "median_s": medians, "diagnostics": records,
                "speedup": medians["doubling"]/medians["auto"], "within_1e_9": difference < 1e-9})
    report["inputs_unchanged"] = (identity == solver_source_identity()
        and digest == sha256(args.slab.read_bytes()).hexdigest())
    (args.output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"inputs_unchanged": report["inputs_unchanged"], "results": [
        {key: row[key] for key in ("width_factor", "full_complex_difference", "median_s", "speedup", "within_1e_9")}
        for row in report["results"]]}, indent=2), flush=True)
    return 0 if report["inputs_unchanged"] and all(row["within_1e_9"] for row in report["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
