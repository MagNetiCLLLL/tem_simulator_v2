"""Compare foreground request preparation with capture plus background work.

This deterministic CPU benchmark does not trace rays, draw widgets, calculate
images or solve lens presets. Background work is timed synchronously here to
separate its cost from GUI capture, not to claim an end-to-end speedup.
"""
from __future__ import annotations

import argparse
import json
from statistics import median
from threading import Event
from time import perf_counter
from pathlib import Path

from temsim.assembly_catalog import AssemblyCatalog
from temsim.calculation_cache import calculation_signatures, state_model_signature
from temsim.gui.calculation_controller import CalculationController
from temsim.gui.calculation_request import CapturedCalculationRequest
from temsim.optics.column import default_state


def benchmark_assets(repeats):
    """Serialization fixtures only; not physical field/column qualification."""
    from copy import deepcopy
    import gc
    import platform
    import tracemalloc
    import numpy as np
    from temsim.input_assets import InputAssetStore
    from temsim.instrument_snapshot import encode_instrument, decode_instrument
    from temsim.immutable_json import json_digest
    from temsim.calculation_manifest import solver_source_identity
    from temsim.physics.lens_field_provider import MagneticFieldMap, CoordinateRegistration, FieldMapProvenance

    cases = []
    for side in (32, 128):
        state = default_state()
        axes = tuple(np.linspace(0., .01, side) for _ in range(3))
        shape = (side,)*3
        field = MagneticFieldMap("cartesian_xyz", axes,
            (np.zeros(shape), np.zeros(shape), np.ones(shape)), CoordinateRegistration(),
            "serialization-fixture", 100., 1,
            FieldMapProvenance("fem", "serialization-fixture", "0"*64, "Serialization only; not a calibrated optical field"))
        # A supported extra input exercises full-graph capture without asking
        # an uncalibrated fixture to act as the instrument's physical field.
        state.serialization_fixture_field = field
        store = InputAssetStore(budget_bytes=512*1024**2)
        lease = store.capture()
        inline = encode_instrument(state)
        started = perf_counter()
        compact = encode_instrument(state, asset_store=lease)
        cold = perf_counter()-started
        prepared_request = CapturedCalculationRequest.capture(state, "High accuracy", 9, 1.)
        timings = {}
        operations = {
            "inline_encode": lambda: encode_instrument(state),
            "asset_encode_warm": lambda: encode_instrument(state, asset_store=lease),
            "inline_decode": lambda: decode_instrument(inline),
            "asset_decode": lambda: decode_instrument(compact, assets=lease),
            "deepcopy_field": lambda: deepcopy(field),
            "inline_hash": lambda: json_digest(inline),
            "asset_hash": lambda: json_digest(compact),
            "high_request_capture": lambda: CapturedCalculationRequest.capture(state, "High accuracy", 9, 1.),
            "high_request_prepare": lambda: prepared_request.prepare(Event()),
        }
        for name, operation in operations.items():
            durations, peaks = [], []
            for _ in range(repeats):
                gc.collect()
                tracemalloc.start()
                started = perf_counter()
                result = operation()
                durations.append(perf_counter()-started)
                peaks.append(tracemalloc.get_traced_memory()[1])
                tracemalloc.stop()
                if isinstance(result, CapturedCalculationRequest):
                    result._input_assets.close()
                del result
            timings[name] = dict(samples_s=durations, median_s=median(durations),
                p95_s=float(np.percentile(durations, 95)), peak_traced_bytes=max(peaks))
        restored = decode_instrument(compact, assets=lease)
        assert json_digest(encode_instrument(restored)) == json_digest(inline)
        cases.append(dict(grid_shape=list(shape), component_bytes=sum(a.nbytes for a in field.components_t),
            cold_asset_encode_s=cold, timings=timings, asset_statistics=store.statistics(), exact_inline_roundtrip=True))
        lease.close()
        prepared_request._input_assets.close()
    return dict(schema="input-asset-capture-benchmark-v1", implementation=solver_source_identity(),
        host=platform.platform(), processor=platform.processor(), repeats=repeats, cases=cases,
        scope="Supported MagneticFieldMap serialization fixtures, 0.75 MiB and 48 MiB components; no physics, GPU, UI paint or model calibration acceptance",
        memory_scope="Tracemalloc peak incremental allocations per operation; not whole-process RSS or VRAM",
        limits="Local timings with tracing enabled; no claim of end-to-end simulator acceleration")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--assets", action="store_true", help="Measure immutable field-map capture/encode/decode/hash fixtures")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.assets:
        report = json.dumps(benchmark_assets(args.repeats), indent=2)
        if args.output:
            args.output.write_text(report+"\n", encoding="utf-8")
        print(report)
        return
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, catalog.default_selection())
    old_times, capture_times, background_times = [], [], []
    for i in range(args.repeats + 1):
        start = perf_counter()
        expected_model = state_model_signature(state)
        snapshot = CalculationController._calculation_snapshot(state, "Preview", 49, 1.)
        expected_signatures = calculation_signatures(snapshot)
        old_duration = perf_counter() - start
        start = perf_counter()
        captured = CapturedCalculationRequest.capture(state, "Preview", 49, 1.)
        capture_duration = perf_counter() - start
        start = perf_counter()
        prepared = captured.prepare(Event())
        background_duration = perf_counter() - start
        assert prepared.model_signature == expected_model
        assert prepared.request_signatures == expected_signatures
        assert prepared.snapshot.to_dict() == snapshot.to_dict()
        if i:
            old_times.append(old_duration)
            capture_times.append(capture_duration)
            background_times.append(background_duration)
    print(json.dumps({
        "scope": "CPU request preparation only; assembled default; warm medians; no ray solve or Qt painting",
        "repeats": args.repeats,
        "previous_foreground_ms": median(old_times) * 1000.,
        "new_foreground_capture_ms": median(capture_times) * 1000.,
        "new_background_preparation_ms": median(background_times) * 1000.,
        "identities_and_snapshots_equal": True,
    }, indent=2))


if __name__ == "__main__":
    main()
