"""Compare live scalar edit preparation; excludes ray calculation and painting."""
from __future__ import annotations

import argparse
import statistics
from time import perf_counter

from temsim.assembly_catalog import AssemblyCatalog
from temsim.gui.calculation_controller import CalculationController
from temsim.interactive_calculation import (
    CalculationRange, _assign, apply_live_tuning_values, available_controls,
)
from temsim.optics.column import default_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    catalog, state = AssemblyCatalog(), default_state()
    catalog.apply(state, catalog.default_selection())
    control = next(c for c in available_controls(state) if c.key == "objective_lens" and c.field == "percent")
    axis = CalculationRange(control, 60., 75.)

    def old(value):
        snapshot = CalculationController._calculation_snapshot(state, "Preview", 49, 1.)
        assert axis.control.identity in {c.identity for c in available_controls(snapshot)}
        _assign(snapshot, axis.control, axis.validate_value(value))
        _assign(state, axis.control, value)

    def new(value):
        apply_live_tuning_values(state, ((axis, value),))

    for label, operation in (("Previous full-snapshot edit", old), ("Scalar validated edit", new)):
        operation(65.)
        times = []
        for i in range(args.repeats):
            value = 65. + i % 10 / 10
            started = perf_counter()
            operation(value)
            times.append(perf_counter() - started)
            assert state.objective_lens.percent == value
        print(f"{label}: median {1000 * statistics.median(times):.4f} ms ({args.repeats} repeats)")
    print("CPU control preparation only; not full-column or GUI-frame latency.")


if __name__ == "__main__":
    main()
