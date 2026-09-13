"""Audit completed same-implementation gun refinements, without admitting images."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from scripts.compare_gun_boundaries import compare_boundaries
from scripts.compare_surface_gun import compare


def summarize(first, second):
    paths = [Path(first), Path(second)]
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    for report in reports:
        if (report.get("status") != "DIAGNOSTIC_GUN_ONLY_NOT_IMAGES"
                or report.get("implementation_unchanged") is not True):
            raise ValueError("Refinement audit requires completed, unchanged-implementation gun runs")
    if reports[0]["implementation"] != reports[1]["implementation"]:
        raise ValueError("Numerical refinement requires the same solver implementation")
    ledger = []
    for a, b in zip(reports[0]["gun"]["mode_records"], reports[1]["gun"]["mode_records"]):
        signature = lambda mode: [tuple(row[key] for key in ("kind", "component", "z_nm", "radius_nm"))
                                 for row in mode["mask_losses"]]
        if signature(a) != signature(b):
            raise ValueError("Physical masks changed, moved or disappeared between refinements")
        ledger.append({"energy_ev": a["energy_ev"], "physical_mask_events": len(a["mask_losses"]),
            "physical_mask_absorbed_fractions": [a["mask_absorbed_fraction"], b["mask_absorbed_fraction"]],
            "unresolved_mask_fractions": [a["unresolved_mask_fraction"], b["unresolved_mask_fraction"]],
            "near_side_fractions": [a["near_flux"]["side"], b["near_flux"]["side"]]})
    # This enforces matching source/field/settings, energy identities, mixture
    # weights and near-tip axial phase references before numerical comparison.
    result = compare_boundaries(*paths, maximum_z_nm=100000.)
    fine = compare(*paths, samples=32769)
    quadrature_change = max(abs(x["relative_complex_l2"]-y["relative_complex_l2"])
        for x, y in zip(result["mode_comparisons"], fine["mode_comparisons"]))
    if quadrature_change > 2e-5:
        raise ValueError("Gun-exit complex comparison quadrature is unresolved")
    result.update(mode_comparisons=fine["mode_comparisons"], relative_complex_l2=fine["relative_complex_l2"],
        exit_comparison_quadrature_change=quadrature_change, radial_comparison_samples=32769,
        mask_ledger=ledger, thresholds={"relative_current": .01, "relative_complex_l2": .01},
        inputs=[{"report": str(p), "report_sha256": sha256(p.read_bytes()).hexdigest(),
                 "complex_states_sha256": sha256((p.parent/"complex_states.npz").read_bytes()).hexdigest()}
                for p in paths],
        comparison_driver_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        source_to_image_admission="NOT_GRANTED")
    result["coordinate_comparison_status"] = ("PASSES_THIS_PAIR_ONLY" if
        result["relative_current_change"] < .01 and result["relative_complex_l2"] < .01 else "FAILED")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.first, args.second)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ("current_a", "relative_current_change", "relative_complex_l2",
        "coordinate_comparison_status", "source_to_image_admission")}))


if __name__ == "__main__":
    main()
