"""Compare exact constant-slab solvers on a retained real-gun operator.

This is a kernel timing/equivalence check, not a physical source or image run.
No upstream component or input is changed and no wave is published.
"""
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
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    if args.repeats < 1 or args.repeats > 1000:
        parser.error("Choose 1 to 1000 repetitions")
    args.output.mkdir(parents=True, exist_ok=False)
    identity = solver_source_identity()
    source_hash = sha256(args.slab.read_bytes()).hexdigest()
    with np.load(args.slab, allow_pickle=False) as data:
        q, g = data["residual"], data["connection"]
        width, kappa, carrier = data["width_kappa_carrier"]
    report = {"scope": "Retained actual constant operator; NOT_SOURCE_ACCEPTANCE",
        "solver_identity": identity, "input_file": str(args.slab.resolve()), "input_sha256": source_hash,
        "repeats": args.repeats, "blas_threads": 1, "results": {}}
    baseline = None
    with threadpool_limits(1):
        for method in ("spectral", "graph", "riccati"):
            try:
                result, diagnostics = covariant_carrier_slab(q, g, width, kappa, carrier, current_coordinates=method)
                if baseline is None:
                    baseline = result
                durations = []
                for _ in range(args.repeats):
                    started = perf_counter()
                    repeated, _ = covariant_carrier_slab(q, g, width, kappa, carrier, current_coordinates=method)
                    durations.append(perf_counter()-started)
                difference = float(np.linalg.norm(np.asarray(result)-baseline))
                report["results"][method] = {"seconds": durations, "median_s": float(np.median(durations)),
                    "full_complex_difference_from_spectral": difference,
                    "repeat_difference": float(np.linalg.norm(np.asarray(repeated)-result)),
                    "diagnostics": diagnostics}
            except Exception as error:
                report["results"][method] = {"error": f"{type(error).__name__}: {error}"}
    report["inputs_unchanged"] = (identity == solver_source_identity()
        and source_hash == sha256(args.slab.read_bytes()).hexdigest())
    (args.output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["inputs_unchanged"] and all("error" not in row for row in report["results"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
