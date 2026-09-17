"""Bounded real grounded-field refinement receipt; never installs a preset."""
import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state
from temsim.optics.electron_gun.tip_assembly import model_from_part
from temsim.sampling_convergence import ConvergenceRequest, run_convergence, write_evidence
from temsim.working_point import WorkingPointCheckpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-seconds", type=float, default=120.)
    parser.add_argument("--axis", choices=("field_radial", "field_axial", "field_apex"), default="field_radial")
    args = parser.parse_args()
    if args.maximum_seconds <= 0:
        parser.error("Choose a positive time bound")
    state = default_state()
    emitter = state.electron_gun.emitter
    model = model_from_part(state._resolved_assembly.part("feg_tip").data)
    emitter.surface_model = replace(model, field_numerics=replace(model.field_numerics,
        radial_nodes=32, axial_nodes=64, apex_cells_per_radius=4))
    emitter.ray_count = 9
    state.step_mm = 1.
    captured = capture_instrument_snapshot(state)
    point = WorkingPointCheckpoint(captured, {}, state.sample.z_mm, "inputs", {})
    start = perf_counter()
    try:
        evidence = run_convergence(ConvergenceRequest(point, args.axis, 9),
            cancelled=lambda: perf_counter()-start > args.maximum_seconds,
            progress=lambda message: print(message, flush=True))
        receipt = dict(status="EXECUTED", elapsed_s=perf_counter()-start, evidence=evidence)
    except Exception as exc:
        receipt = dict(status="UNRESOLVED", elapsed_s=perf_counter()-start,
            error_type=type(exc).__name__, error=str(exc))
    receipt.update(schema="bounded-grounded-field-comparison-v1", input_id=captured.digest,
        implementation=captured.implementation,
        scope=f"Detached 9-ray coarse grounded field, independently doubled {args.axis}; baseline radial 32, axial 64, apex 4; exact existing tip/gun/incident-column path",
        model_qualification="NOT_ESTABLISHED", full_numerical_qualification="NOT_ESTABLISHED",
        limits="Coarse numerical fixture; no production defaults changed and no source/field acceptance inferred")
    write_evidence(receipt, args.output)
    print(receipt["status"], flush=True)


if __name__ == "__main__":
    main()
