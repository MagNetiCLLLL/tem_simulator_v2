"""Report transmitted NanoPulser condenser presets on the actual TOML geometry.

This writes a numerical report and never overwrites instrument or mode TOMLs.
The optional module length is a declared simulator assumption, not an OEM
NanoPulser dimension.  Re-run after changing its geometry or the source.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

from numba import set_num_threads

from temsim.assembly_catalog import AssemblyCatalog
from temsim.operating_modes import apply_operating_mode_pair
from temsim.optics.column import default_state
from temsim.optics.probe_calibration import IncidentProbeModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/nanopulser_calibration/report.json"))
    parser.add_argument("--mode", choices=("nano_probe", "micro_probe", "both"),
                        default="both")
    parser.add_argument("--rays", type=int, default=512)
    args = parser.parse_args()
    set_num_threads(4)
    modes = ("nano_probe", "micro_probe") if args.mode == "both" else (args.mode,)
    catalog = AssemblyCatalog()
    rows = []
    for mode in modes:
        state = default_state()
        state.electron_gun.emitter.ray_count = args.rays
        selection = replace(catalog.default_selection(), beam_blanker="NanoPulser")
        catalog.apply(state, selection)
        print("Recalculating", mode, "sample Z", state.sample.z_mm, flush=True)
        started = time.perf_counter()
        result = apply_operating_mode_pair(state, mode, "diffraction")
        elapsed = time.perf_counter() - started
        print(result.summary, "elapsed", elapsed, flush=True)
        state.acceleration_enabled = True
        state.acceleration_backend = "Numba CPU"
        state.electron_gun.emitter.ray_count = args.rays
        model = IncidentProbeModel(state)
        coarse = model.trace(step_mm=0.05).statistics
        fine = model.trace(step_mm=0.025).statistics
        row = {
            "mode": mode, "sample_z_mm": state.sample.z_mm,
            "blanker_z_mm": state.nanopulser.z_mm,
            "stop_z_mm": state.nanopulser.stop_z_mm,
            "beam_voltage_kv": state.beam_voltage_kv,
            "rays": args.rays, "calculation_seconds": elapsed,
            "preset": asdict(result.condenser),
            "coarse_step_mm": 0.05, "fine_step_mm": 0.025,
            "coarse_statistics": asdict(coarse),
            "fine_statistics": asdict(fine),
        }
        print(json.dumps(row, indent=2), flush=True)
        rows.append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
