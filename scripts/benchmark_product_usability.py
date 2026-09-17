"""Bounded classical tip-to-specimen benchmark; scalar receipts only.

Run with the project interpreter. Nested timings are never added to wall time.
The small default case checks plumbing, not numerical or physical convergence.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
import statistics
import subprocess
import tempfile
from time import perf_counter
from unittest.mock import patch

import numpy as np

from temsim.assembly_catalog import AssemblyCatalog
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state
from temsim.optics.beam_path_audit import incident_checkpoints
from temsim.optics.electron_gun import source
from temsim.physics import core
from temsim.working_point import WorkingPointCheckpoint


def benchmark(rays=9, repeats=5):
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, catalog.default_selection())
    state.electron_gun.emitter.ray_count = rays
    state.acceleration_backend = "CPU"
    state.step_mm = 1.0
    state.history_step_mm = 5.0
    source_function, column_function = source.trace_source_to_exit, core.propagate
    gun_class = type(state.electron_gun)
    electric_field = gun_class.electric_field.fget
    magnetic_field = gun_class.magnetic_field.fget
    runs = []
    checkpoint = None
    for index in range(repeats + 1):
        timings = dict(electric_field_construction_or_cache_s=0., magnetic_field_construction_or_cache_s=0.)

        def measured(name, function, *args, **kwargs):
            start = perf_counter()
            value = function(*args, **kwargs)
            timings[name] = timings.get(name, 0.0) + perf_counter() - start
            return value

        wall = perf_counter()
        snapshot = measured("capture_s", capture_instrument_snapshot, state)
        detached = measured("restore_s", snapshot.restore)
        with patch.object(source, "trace_source_to_exit", side_effect=lambda *a, **k:
                          measured("gun_including_fields_s", source_function, *a, **k)), \
                patch.object(core, "propagate", side_effect=lambda *a, **k:
                             measured("column_s", column_function, *a, **k)), \
                patch.object(gun_class, "electric_field", property(lambda gun:
                             measured("electric_field_construction_or_cache_s", electric_field, gun))), \
                patch.object(gun_class, "magnetic_field", property(lambda gun:
                             measured("magnetic_field_construction_or_cache_s", magnetic_field, gun))):
            gun, planes, mask = measured("audit_including_gun_and_column_s", incident_checkpoints,
                detached, [detached.sample.upper_surface_z_mm], step_mm=state.step_mm)
        arrays = {key: getattr(planes, key)[-1] for key in ("x_m", "y_m", "tx_rad", "ty_rad")}
        arrays.update(weight=gun.exit_bundle.weight, alive=mask[-1],
                      gun_ray_id=gun.exit_bundle.ray_id, energy_offset_ev=gun.exit_bundle.energy_offset_ev)
        checkpoint = measured("freeze_checkpoint_s", WorkingPointCheckpoint, snapshot, arrays,
            float(planes.z_mm[-1]), snapshot.physical_digest,
            {"source_representation": "gun-derived-particles", "scope": "incident-audit"})
        measured("hash_s", lambda: checkpoint.digest)
        with tempfile.TemporaryDirectory(prefix="temsim-benchmark-") as directory:
            path = Path(directory) / "bounded.temwp"
            measured("archive_write_s", checkpoint.write_package, path)
            loaded = measured("archive_read_s", WorkingPointCheckpoint.read_package, path)
            assert loaded.digest == checkpoint.digest
        timings["wall_s"] = perf_counter() - wall
        runs.append({"kind": "cold" if index == 0 else "warm", **timings})
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from temsim.gui.working_point_panel import WorkingPointPanel
    app = QApplication.instance() or QApplication([])
    panel = WorkingPointPanel()
    panel.add_checkpoint(checkpoint)
    ui_samples = []
    for _ in range(20):
        start = perf_counter()
        panel._select(0)
        app.processEvents()
        ui_samples.append(perf_counter() - start)
    panel.close()
    return {
        "schema": "product-usability-benchmark-v1",
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "host": {"platform": platform.platform(), "processor": platform.processor(),
                 "logical_cpus": os.cpu_count(), "python": platform.python_version()},
        "scope": "CPU classical physical gun through specimen entrance; no sample/detector/wave qualification",
        "input_identity": checkpoint.snapshot.digest,
        "implementation_identity": checkpoint.snapshot.implementation,
        "rays": rays, "column_step_mm": state.step_mm,
        "survivors": int(np.count_nonzero(checkpoint.arrays["alive"])),
        "runs": runs,
        "warm_medians_s": {key: statistics.median(row[key] for row in runs[1:])
                           for key in runs[0] if key != "kind"},
        "gui_detail_refresh": {"samples_s": ui_samples, "median_s": statistics.median(ui_samples),
                               "p95_s": float(np.percentile(ui_samples, 95))},
        "timing_semantics": "Field provider construction/cache lookup is nested in gun; field evaluation during trajectory integration is still in gun. Gun and column are nested in audit; all stages are nested in wall. Do not sum them.",
        "not_measured": ["GPU transfers (CPU case)",
                         "Large asset capture and peak memory", "Interactive main-window event latency"],
        "validation_status": "NOT_RUN: bounded execution is not convergence evidence",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rays", type=int, default=9, choices=range(9, 129))
    parser.add_argument("--repeats", type=int, default=5, choices=range(5, 11))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = benchmark(args.rays, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["warm_medians_s"], indent=2))
