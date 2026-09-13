"""Attribute current differences in executed physical-gun evidence.

No propagation, normalization, source creation or image admission is performed.
Finite-basis removal is deliberately separate from physical aperture absorption.
"""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path


def current_budget(report):
    if (report.get("status") != "DIAGNOSTIC_GUN_ONLY_NOT_IMAGES"
            or report.get("implementation_unchanged") is not True):
        raise ValueError("Current accounting requires an executed, unchanged-implementation gun")
    gun = report["gun"]
    reservoir = float(gun["record"]["source"]["emission"]["current_na"])
    modes = gun["mode_records"]
    if not math.isfinite(reservoir) or reservoir <= 0 or not modes:
        raise ValueError("A finite positive tip current and complete energy modes are required")
    weights = [float(m["mixture_weight"]) for m in modes]
    if any(not math.isfinite(w) or w < 0 for w in weights) or not math.isclose(sum(weights), 1., rel_tol=0., abs_tol=1e-12):
        raise ValueError("Retain the full energy mixture without renormalizing its weights")
    names = ("reflected_to_tip", "numerical_side_escape", "entering_gun",
             "physical_mask_absorption", "unresolved_mask_modes", "exit")
    totals = dict.fromkeys(names, 0.)
    components = defaultdict(lambda: {"physical_mask_absorption_na": 0., "unresolved_mask_modes_na": 0.,
                                      "positions_nm": set(), "events_per_mode": []})
    mode_rows = []
    signatures = []
    for mode, weight in zip(modes, weights):
        near, masks = mode["near_flux"], mode["mask_losses"]
        signature = [(row["component"], row["kind"], row["z_nm"], row["radius_nm"]) for row in masks]
        signatures.append(signature)
        if signature != signatures[0]:
            raise ValueError("Every energy mode must execute the same physical masks")
        fractions = dict(zip(names, (near["reflected"], near["side"], near["top"],
            sum(row["mask_absorbed"] for row in masks), sum(row["unresolved_transmitted"] for row in masks),
            mode["exit_fraction"])))
        if not all(math.isfinite(float(v)) for v in fractions.values()):
            raise ValueError("Current ledger contains nonfinite values")
        for name, fraction in fractions.items():
            totals[name] += reservoir*weight*fraction
        # Preserve signed residuals; never force conservation by rescaling.
        balance = near["top"]-fractions["exit"]-fractions["physical_mask_absorption"]-fractions["unresolved_mask_modes"]
        local = {"energy_ev": mode["energy_ev"], "weight": weight,
            "fractions": fractions, "near_balance_residual": near["incoming"]-near["reflected"]-near["side"]-near["top"],
            "gun_balance_residual": balance}
        if masks:
            first = min(row["z_nm"] for row in masks)
            samples = [row for row in mode.get("boundary_states", []) if row["z_nm"] < first]
            if samples:
                currents = [row["net_current_fraction"] for row in samples]
                local["pre_mask_sampled_current_range"] = max(currents)-min(currents)
                local["first_physical_mask_nm"] = first
        mode_rows.append(local)
        counts = defaultdict(int)
        for row in masks:
            item = components[row["component"]]
            item["physical_mask_absorption_na"] += reservoir*weight*row["mask_absorbed"]
            item["unresolved_mask_modes_na"] += reservoir*weight*row["unresolved_transmitted"]
            item["positions_nm"].add(row["z_nm"])
            counts[row["component"]] += 1
        for key, count in counts.items():
            components[key]["events_per_mode"].append(count)
    output_na = float(gun["current_a"])*1e9
    if not math.isclose(totals["exit"], output_na, rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError("Recorded exit current disagrees with the complete energy ledger")
    rows = [{"component": key, **row, "positions_nm": sorted(row["positions_nm"])}
            for key, row in components.items()]
    return {"reservoir_current_na": reservoir, "current_na": totals, "components": rows, "modes": mode_rows,
        "total_accounting_residual_na": reservoir-sum(v for key, v in totals.items() if key != "entering_gun"),
        "mask_signature": signatures[0],
        "scope": "Numerical evidence accounting; physical-mask numbers remain unqualified until wave convergence"}


def compare_budgets(first, second):
    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in (first, second)]
    if (not reports[0].get("instrument_settings_digest") or
            reports[0]["instrument_settings_digest"] != reports[1].get("instrument_settings_digest")):
        raise ValueError("Current differences require identical physical inputs")
    if reports[0]["implementation"] != reports[1]["implementation"]:
        raise ValueError("Use the same numerical implementation for coordinate attribution")
    budgets = [current_budget(report) for report in reports]
    if budgets[0]["mask_signature"] != budgets[1]["mask_signature"]:
        raise ValueError("Physical masks differ between runs")
    if (budgets[0]["reservoir_current_na"] != budgets[1]["reservoir_current_na"] or
            [(m["energy_ev"], m["weight"]) for m in budgets[0]["modes"]] !=
            [(m["energy_ev"], m["weight"]) for m in budgets[1]["modes"]]):
        raise ValueError("Tip current or energy mixture differs")
    changes = {key: budgets[1]["current_na"][key]-value for key, value in budgets[0]["current_na"].items()}
    components = []
    for a, b in zip(budgets[0]["components"], budgets[1]["components"]):
        components.append({"component": a["component"],
            "physical_mask_absorption_change_na": b["physical_mask_absorption_na"]-a["physical_mask_absorption_na"],
            "unresolved_mask_modes_change_na": b["unresolved_mask_modes_na"]-a["unresolved_mask_modes_na"]})
    for budget in budgets:
        del budget["mask_signature"]
    return {"inputs": [{"path": str(p), "sha256": sha256(Path(p).read_bytes()).hexdigest()} for p in (first, second)],
        "budgets": budgets, "second_minus_first_na": changes, "component_changes": components,
        "driver_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "causal_scope": "Accounts for current change; does not identify the exact discretization defect or certify images"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_budgets(args.first, args.second)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"totals_na": [b["current_na"] for b in result["budgets"]],
        "difference_na": result["second_minus_first_na"], "component_changes": result["component_changes"]}, indent=2))


if __name__ == "__main__":
    main()
