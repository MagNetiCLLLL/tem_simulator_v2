"""Small deterministic specimen-transport benchmark, not an instrument calibration.

Run before and after an optimisation with the same arguments. NPZ outputs keep
terminal electrons and material paths for numerical comparison; no microscope
settings, presets, or running GUI results are changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from temsim.optics.column import default_state
from temsim.simulation_modes import switch_mode
from temsim.specimen.elastic_transport import (
    IncidentElectronRay,
    simulate_elastic_point_transport,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rays", type=int, default=64)
    parser.add_argument("--tilt-mrad", type=float, default=0.0)
    parser.add_argument("--preset", default="si_110")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rays <= 0 or not np.isfinite(args.tilt_mrad):
        parser.error("Positive ray count and finite tilt are required")
    if args.output.exists():
        parser.error("Output already exists; choose a new benchmark artifact")

    state = default_state()
    switch_mode(state, "ideal")
    state.sample.specimen_mode = "virtual"
    state.sample.specimen_preset_key = args.preset
    state.sample.thickness_nm = 10.0
    state.sample.eds_support_material_key = "vacuum"
    direction = np.asarray((args.tilt_mrad * 1.0e-3, 0.0, 1.0))
    direction /= np.linalg.norm(direction)
    rays = tuple(
        IncidentElectronRay(i, (0.0, 0.0), tuple(direction), 300_000.0, 1 / args.rays)
        for i in range(args.rays)
    )
    started = perf_counter()
    result = simulate_elastic_point_transport(
        state, incident_rays=rays, seed=73, stored_trajectory_count=args.rays,
    )
    elapsed = perf_counter() - started
    terminal = result.terminal_electrons
    payload = {
        name: np.asarray(getattr(terminal, name)) for name in (
            "source_ray_index", "position_nm", "direction", "kinetic_energy_ev",
            "weight", "outcome", "event_count", "has_scattered",
        )
    }
    payload["track_path_nm"] = np.asarray([t.path_length_nm for t in result.eds_tracks])
    payload["track_source"] = np.asarray([t.source_key for t in result.eds_tracks])
    payload["track_material"] = np.asarray([t.material.key for t in result.eds_tracks])
    payload["history_offsets"] = np.cumsum([0] + [len(t.points_nm) for t in result.trajectories])
    payload["history_points_nm"] = np.concatenate([t.points_nm for t in result.trajectories])
    metadata = {
        "seconds": elapsed, "rays": args.rays, "tilt_mrad": args.tilt_mrad,
        "preset": args.preset, "thickness_nm": 10.0, "support": "vacuum",
        "energy_ev": 300_000.0, "seed": 73, "mode": "ideal",
    }
    payload["metadata_json"] = np.asarray(json.dumps(metadata))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, **payload)
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
