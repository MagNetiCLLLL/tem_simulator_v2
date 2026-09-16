"""Generate lightweight, non-installed assembly illumination candidate reports.

No image, wave or array cache is written. Failed and incomplete calibrations
are retained as NOT_QUALIFIED, never substituted for usable default presets.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from temsim.assembly_catalog import AssemblyCatalog
from temsim.optics.column import default_state
from temsim.optics.assembly_illumination import assembly_combinations, calibrate_illumination
from temsim.calculation_manifest import solver_source_identity


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gun", default="FEG")
    parser.add_argument("--column", default="C3 + Probe Corrector")
    parser.add_argument("--blanker", default="None")
    parser.add_argument("--recording", default="Energy Filter")
    parser.add_argument("--mode", choices=("nano_probe", "micro_probe", "both"), default="both")
    parser.add_argument("--all", action="store_true", help="Audit every full assembly separately")
    parser.add_argument("--upstream-only", action="store_true",
        help="Calculate the 18 pre-specimen combinations (no downstream equivalence claim)")
    parser.add_argument("--rays", type=int, default=193)
    parser.add_argument("--max-evaluations", type=int, default=30)
    parser.add_argument("--controls", nargs=2, help="Two installed pre-specimen lens keys")
    parser.add_argument("--c2-aperture-mm", type=float, help="Explicit physical C2 aperture diameter for candidate exploration")
    parser.add_argument("--mini-polarity", type=int, choices=(-1,1), help="Explicit mini condenser field polarity")
    parser.add_argument("--search-branches", action="store_true", help="Explore first-order branch seeds; all remain unqualified until topology checks")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new report path; existing evidence is not overwritten")
    catalog = AssemblyCatalog()
    selections = [s for s in assembly_combinations(catalog) if args.all or
        (args.upstream_only and s.column in ('C2', 'C3', 'C3 + Probe Corrector')
         and s.recording == 'Energy Filter') or
        (not args.upstream_only and
        (s.gun, s.column, s.beam_blanker, s.recording) ==
        (args.gun, args.column, args.blanker, args.recording))]
    if not selections:
        parser.error("No matching assembly")
    modes = ("nano_probe", "micro_probe") if args.mode == "both" else (args.mode,)
    implementation = solver_source_identity()
    document = dict(schema="assembly-illumination-candidates-v1", status="NOT_INSTALLED",
                    implementation=implementation, records=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for selection in selections:
        state = default_state()
        catalog.apply(state, selection)
        for mode in modes:
            start = perf_counter()
            print(json.dumps(dict(assembly=asdict(selection), mode=mode)), flush=True)
            try:
                _, record = calibrate_illumination(state, mode, rays=args.rays,
                    maximum_evaluations=args.max_evaluations,
                    controls=args.controls,
                    search_branches=args.search_branches,
                    c2_aperture_mm=args.c2_aperture_mm,
                    mini_polarity=args.mini_polarity,
                    progress=lambda text: print(text, flush=True))
            except Exception as error:
                record = dict(status="NOT_QUALIFIED", error=str(error))
            record.update(assembly=asdict(selection), mode=mode, seconds=perf_counter()-start)
            if solver_source_identity() != implementation:
                raise RuntimeError('Solver source changed during calibration; retain earlier records and rerun')
            document["records"].append(record)
            args.output.write_text(json.dumps(finite_json(document), indent=2, allow_nan=False)+"\n", encoding="utf-8")
            print(json.dumps(dict(status=record["status"], mode=mode, seconds=record["seconds"])), flush=True)


if __name__ == "__main__":
    main()
