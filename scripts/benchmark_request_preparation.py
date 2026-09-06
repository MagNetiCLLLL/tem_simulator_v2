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

from temsim.assembly_catalog import AssemblyCatalog
from temsim.calculation_cache import calculation_signatures, state_model_signature
from temsim.gui.calculation_controller import CalculationController
from temsim.gui.calculation_request import CapturedCalculationRequest
from temsim.optics.column import default_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
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
