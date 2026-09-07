"""Small, reproducible STEM wave benchmark; no GUI state or cache is changed.

Uses a prescribed 30 mrad incident bundle, Si [110], 10 nm thickness and a
synthetic signed recording map. This measures the wave/detector stage, not a
complete microscope calculation or the user's current unsaved CIF settings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from temsim.optics.column import default_state
from temsim.physics.first_order import TransverseTransfer
from temsim.physics.record_plane import PlaneStop, RecordPlanePlan
from temsim.physics.stem_wave_imaging import AngularDetector, simulate_angle_resolved_stem


def benchmark_inputs(scan_side=8, grid=256, configurations=1):
    state = default_state()
    state.illumination_mode = "STEM"
    state.sample.specimen_preset_key = "si_110"
    state.sample.thickness_nm = 10.0
    state.sample.wave_grid_pixels = grid
    state.sample.wave_field_of_view_angstrom = 20.0
    state.sample.wave_multislice_enabled = True
    state.sample.wave_atomistic_enabled = True
    state.sample.wave_slice_thickness_angstrom = 2.0
    state.sample.wave_frozen_phonon_enabled = configurations > 1
    state.sample.wave_frozen_phonon_configurations = configurations
    state.sample.wave_frozen_phonon_seed = 707
    incident = SimpleNamespace(
        alive=np.ones(5, dtype=bool), ray_weight=np.array([.6, .1, .1, .1, .1]),
        x=np.zeros((1, 5)), y=np.zeros((1, 5)),
        tx=np.array([[0., .03, -.03, 0., 0.]]),
        ty=np.array([[0., 0., 0., .03, -.03]]),
    )
    axis_um = (np.arange(scan_side) - (scan_side - 1) / 2) * .02e-3
    scan_x, scan_y = np.meshgrid(axis_um, axis_um)
    z = float(state.sample.z_mm)
    planes = (
        PlaneStop("ap", "Aperture", z + 10, "aperture", "disk", radius_mm=3.),
        PlaneStop("haadf", "HAADF", z + 20, "detector", "annulus",
                  outer_width_mm=5., inner_diameter_mm=1.2, readout_enabled=True),
        PlaneStop("df", "DF", z + 30, "detector", "annulus",
                  outer_width_mm=1.2, inner_diameter_mm=.32, readout_enabled=True),
        PlaneStop("bf", "BF", z + 40, "detector", "disk",
                  outer_width_mm=.2, readout_enabled=True),
    )
    transfers = tuple(TransverseTransfer(
        z, plane.z_mm, np.eye(2), np.array([[.01, .001], [-.001, .01]]),
        np.zeros((2, 2)), np.eye(2),
    ) for plane in planes)
    plan = RecordPlanePlan(z, planes, transfers, "0" * 64, "1" * 64)
    detectors = (AngularDetector("haadf", 60., 250.), AngularDetector("df", 16., 60.),
                 AngularDetector("bf", 0., 10.))
    return state, SimpleNamespace(incident=incident), detectors, scan_x, scan_y, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", default=["CPU", "CUDA GPU"])
    parser.add_argument("--scan-side", type=int, default=8)
    parser.add_argument("--grid", type=int, default=256)
    parser.add_argument("--configurations", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()
    if min(args.scan_side, args.grid, args.configurations, args.repeat) < 1:
        parser.error("All sizes and repeat counts must be positive.")
    reference = None
    for backend in args.backends:
        for repeat in range(args.repeat):
            state, simulation, detectors, x, y, plan = benchmark_inputs(
                args.scan_side, args.grid, args.configurations)
            state.acceleration_backend = backend
            state.acceleration_enabled = backend != "CPU"
            start = time.perf_counter()
            result = simulate_angle_resolved_stem(
                state, simulation, detectors, x, y, record_plane_plan=plan)
            elapsed = time.perf_counter() - start
            if reference is None and backend == "CPU":
                reference = result
            if reference is not None:
                for key in result.fractions:
                    np.testing.assert_allclose(result.fractions[key], reference.fractions[key],
                                               rtol=2e-4, atol=2e-7)
                np.testing.assert_allclose(result.uncollected_fraction, reference.uncollected_fraction,
                                           rtol=2e-4, atol=2e-7)
                np.testing.assert_allclose(result.truncated_fraction, reference.truncated_fraction,
                                           rtol=2e-4, atol=2e-7)
            errors = None if reference is None else {
                key: float(np.max(np.abs(values - reference.fractions[key])))
                for key, values in result.fractions.items()
            }
            print(json.dumps({
                "requested_backend": backend, "run": repeat + 1,
                "elapsed_s": elapsed, "scan_positions": x.size,
                "positions_per_s": x.size / elapsed,
                "actual_backend": result.metrics["wave_compute_backend"],
                "grid": [result.metrics["grid_pixels_y"], result.metrics["grid_pixels_x"]],
                "configurations": args.configurations,
                "probe_batch_size": result.metrics.get("cuda_probe_batch_size"),
                "resident_pipeline_s": result.metrics.get("cuda_pipeline_elapsed_s"),
                "fallback": result.metrics.get("cuda_pipeline_fallback_reason"),
                "max_abs_fraction_error_vs_cpu": errors,
            }), flush=True)


if __name__ == "__main__":
    main()
