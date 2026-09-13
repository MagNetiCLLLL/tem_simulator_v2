"""Re-execute the current tip/gun and compare local CF4/CF6 operators.

The first energy is solved with its complete physical reflected load. The
diagnostic observes that solved state's incoming ports and continuous chart;
it deliberately publishes no source or image. All arguments are passed to
inspect_surface_column, which defines the physical/numerical reference.
"""
from hashlib import sha256
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
import traceback

import numpy as np

from temsim.physics.magnus_scattering import cf4_slab, GAUSS_FRACTIONS
from temsim.physics.magnus_sixth import cf6_slab, quadrature
from temsim.physics.scattering_load import compose, outgoing_load
from temsim.physics.liouville_wave import PhysicalCoordinateLoad
from temsim.physics.radial_mask_ledger import physical_to_chart
from temsim.physics.occupied_axial_refinement import apply_two_port


class AuditComplete(BaseException):
    """Stop a diagnostic before any source publication, not solver failure."""


def compare_interval(sample, width, kappa, left, right, *, divisions=(1, 4, 16, 64, 256),
                     cancelled=lambda: False, progress=None):
    results, rows = {}, []
    norm = np.linalg.norm(np.r_[left, right])
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("A nonzero executed incoming state is required")
    for method in ("cf4", "cf6"):
        for count in divisions:
            started, result = perf_counter(), None
            for i in range(count):
                if cancelled():
                    raise InterruptedError("Local gun-integrator comparison cancelled")
                fractions = GAUSS_FRACTIONS if method == "cf4" else quadrature()[0]
                samples = [sample((i+f)/count) for f in fractions]
                piece = (cf4_slab(*samples, width/count, kappa, cancelled=cancelled) if method == "cf4"
                         else cf6_slab(samples, width/count, kappa, cancelled=cancelled))[0]
                result = piece if result is None else compose(result, piece)
            results[method, count] = result
            rows.append({"method": method, "subdivisions": count, "elapsed_s": perf_counter()-started,
                         "exponentials": count*(2 if method == "cf4" else 5)})
            if progress:
                progress(method, count)
    reference = results["cf6", divisions[-1]]
    for row in rows:
        actual = results[row["method"], row["subdivisions"]]
        difference = tuple(a-b for a, b in zip(actual, reference))
        row["occupied_difference_from_finest_cf6"] = float(np.linalg.norm(apply_two_port(difference, left, right))/norm)
        row["full_port_difference_from_finest_cf6"] = float(np.linalg.norm(np.block(
            [[difference[0], difference[1]], [difference[2], difference[3]]])))
    return rows


def main():
    from scripts import inspect_surface_column as driver
    from temsim.physics import occupied_axial_refinement as refinement
    from temsim.calculation_manifest import solver_source_identity
    from temsim.immutable_json import thaw_json, freeze_json
    local = argparse.ArgumentParser(add_help=False)
    local.add_argument("--audit-planes-nm", type=float, nargs="+", default=(100., 300., 400., 1000.))
    local.add_argument("--audit-subdivisions", type=int, nargs="+", default=(1, 4, 16, 64, 256))
    choices, driver_arguments = local.parse_known_args(sys.argv[1:])
    if (any(not np.isfinite(v) or v <= 0 for v in choices.audit_planes_nm)
            or any(v <= 0 for v in choices.audit_subdivisions)
            or list(choices.audit_subdivisions) != sorted(set(choices.audit_subdivisions))):
        local.error("Audit planes must be positive finite; subdivisions must be increasing positive integers")
    output = Path(driver_arguments[driver_arguments.index("--output")+1])
    original_arguments = sys.argv
    started, identity = perf_counter(), solver_source_identity()
    report = {"scope": "LOCAL_ACTUAL_GUN_COMPARISON_NOT_SOURCE_OR_IMAGE_ACCEPTANCE",
              "implementation": identity, "command": sys.argv,
              "driver_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
    original, capture = refinement.refine_joint_mode, driver.capture_instrument_snapshot
    def captured(state):
        result = capture(state)
        report["instrument_snapshot"] = result.to_dict()
        return result
    def observed(plan, boundary_solver, settings, *, cancelled, progress_callback):
        load = PhysicalCoordinateLoad(outgoing_load(plan["operators"], plan["exit_q"], plan["kappa"],
            cancelled=cancelled), plan["alpha"], plan["log_derivative"])
        solved = boundary_solver(load)
        field, derivative = load.propagate(solved[1])
        incoming = [physical_to_chart(f, d, a, l, plan["kappa"]) for f, d, a, l in
                    zip(field, derivative, plan["alpha"], plan["log_derivative"])]
        report["intervals"] = []
        for target in choices.audit_planes_nm:
            index = next(i for i, bounds in plan["interval_z_nm"].items() if bounds[0] <= target < bounds[1])
            sample, width = plan["samplers"][index]
            row = {"root_z_interval_nm": plan["interval_z_nm"][index], "action_width_nm": width,
                   "kappa_per_nm": plan["kappa"]}
            row["comparisons"] = compare_interval(sample, width, plan["kappa"], incoming[index][0],
                incoming[index+1][1], divisions=choices.audit_subdivisions, cancelled=cancelled,
                progress=lambda method, n: print(f"{perf_counter()-started:.2f}s: {target:g} nm {method}/{n}", flush=True))
            report["intervals"].append(row)
            print(json.dumps(row), flush=True)
        raise AuditComplete()
    refinement.refine_joint_mode, driver.capture_instrument_snapshot = observed, captured
    sys.argv = [sys.argv[0], *driver_arguments]
    try:
        returned = driver.main()
        report["status"] = "FAILED_AUDIT_NOT_REACHED"
        report["driver_exit_code"] = returned
    except AuditComplete:
        report["status"] = "LOCAL_DIAGNOSTIC_COMPLETE_NOT_SOURCE"
    except Exception:
        report.update(status="FAILED", traceback=traceback.format_exc())
    finally:
        sys.argv = original_arguments
        refinement.refine_joint_mode, driver.capture_instrument_snapshot = original, capture
    report.update(elapsed_s=perf_counter()-started, implementation_unchanged=identity == solver_source_identity())
    with (output/"integrator_audit.json").open("x", encoding="utf-8") as stream:
        json.dump(thaw_json(freeze_json(report)), stream, indent=2)
    print(json.dumps({k: report[k] for k in ("status", "elapsed_s", "implementation_unchanged")}), flush=True)
    return int(report["status"] != "LOCAL_DIAGNOSTIC_COMPLETE_NOT_SOURCE" or not report["implementation_unchanged"])


if __name__ == "__main__":
    raise SystemExit(main())
