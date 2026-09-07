"""Benchmark a tiny CPU specimen potential, never a full TEM/STEM acquisition.

Constructs a new in-memory Si [110] fixture. It does not load/edit user state,
solve lens presets, trace the column, use CUDA, or write output files. JSON goes
to stdout. First backend startup and subsequent cache timings are separate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from temsim.optics.model import Sample
from temsim.physics.prepared_specimen_cache import (
    clear_prepared_specimen_cache,
    prepared_specimen_cache_info,
)
from temsim.physics.wave_imaging import (
    _prepare_specimen_potentials_uncached,
    prepare_specimen_potentials,
)
from temsim.specimen.presets import load_specimen_preset


def _assert_equal(actual, reference):
    for name in ("x_angstrom", "y_angstrom", "mean_projected_potential_v_angstrom",
                 "slice_thicknesses_angstrom"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(reference, name))
    for actual_array, reference_array in zip(
        actual.potential_configurations_v_angstrom,
        reference.potential_configurations_v_angstrom, strict=True,
    ):
        np.testing.assert_array_equal(actual_array, reference_array)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hits", type=int, default=3, choices=range(1, 11),
                        help="Number of warmed cache hits (1-10; default 3).")
    args = parser.parse_args()
    sample = Sample()
    sample.specimen_mode = "virtual"
    sample.specimen_preset_key = "si_110"
    sample.thickness_nm = 1.0
    sample.wave_grid_pixels = 64
    sample.wave_field_of_view_angstrom = 16.0
    sample.wave_slice_thickness_angstrom = 2.0
    sample.wave_atomistic_enabled = True
    sample.wave_multislice_enabled = True
    sample.wave_frozen_phonon_enabled = False
    sample.wave_frozen_phonon_seed = 100
    state = SimpleNamespace(sample=sample)
    preset = load_specimen_preset("si_110")
    clear_prepared_specimen_cache()
    started = perf_counter()
    reference = _prepare_specimen_potentials_uncached(state, preset)
    first_uncached = perf_counter() - started
    if not reference.metrics.get("atomistic_applied", False):
        raise RuntimeError("This benchmark requires the actual abTEM/ASE potential backend.")
    started = perf_counter()
    cold = prepare_specimen_potentials(state, preset)
    cold_seconds = perf_counter() - started
    _assert_equal(cold, reference)
    if cold.metrics["prepared_specimen_cache_hit"]:
        raise AssertionError("The first cache request unexpectedly hit a retained entry")
    hits = []
    for _ in range(args.hits):
        started = perf_counter()
        actual = prepare_specimen_potentials(state, preset)
        hits.append(perf_counter() - started)
        if not actual.metrics["prepared_specimen_cache_hit"]:
            raise AssertionError("An identical specimen request missed the prepared cache")
        _assert_equal(actual, reference)
    print(json.dumps({
        "scope": "CPU potential only; no column, propagation, detector image or GPU",
        "reference_first_call_seconds": first_uncached,
        "cold_cache_after_backend_warm_seconds": cold_seconds,
        "warm_hit_seconds": hits,
        "warm_hit_median_seconds": statistics.median(hits),
        "backend": reference.metrics["potential_builder_backend"],
        "fixture": {"preset": "si_110", "thickness_nm": 1.0,
                    "requested_pixels": 64, "requested_fov_angstrom": 16.0,
                    "slice_angstrom": 2.0, "frozen_phonons": False, "seed": 100},
        "atom_count": reference.metrics["atom_count"],
        "grid_shape_yx": reference.mean_projected_potential_v_angstrom.shape,
        "potential_shape_zyx": reference.potential_configurations_v_angstrom[0].shape,
        "exact_array_equality": True,
        "cache": prepared_specimen_cache_info(),
    }, indent=2))


if __name__ == "__main__":
    main()
