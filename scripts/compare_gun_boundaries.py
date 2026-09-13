"""Locate disagreement in executed gun boundary fields without phase fitting.

Only exact matching physical planes and energy modes are compared. Both
sides of an aperture are kept. No downstream source is constructed.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.compare_surface_gun import compare


def boundary_metrics(first, second, samples=16385):
    if first["z_nm"] != second["z_nm"]:
        raise ValueError("Boundary planes differ")
    coefficients = [np.asarray(row["coefficients_real"])+1j*np.asarray(row["coefficients_imag"])
                    for row in (first, second)]
    extent = max(row["width_nm"] for row in (first, second))*np.sqrt(4*max(map(len, coefficients))+160)
    radius = np.linspace(0., extent, samples)
    fields = []
    for row, values in zip((first, second), coefficients):
        x = (radius/row["width_nm"])**2
        envelope = np.zeros_like(radius, dtype=complex)
        # Recurrence for exponentially scaled polynomials avoids 0*inf.
        previous, current = np.zeros_like(x), np.exp(-x/2)
        for n, value in enumerate(values):
            envelope += value*current
            previous, current = current, ((2*n+1-x)*current-n*previous)/(n+1)
        fields.append(envelope*np.exp(.5j*row["reference_k_per_nm"]*row["curvature_per_nm"]*radius**2
                                     +1j*row.get("quartic_phase_per_nm4", 0.)*radius**4)
                      /(np.sqrt(np.pi)*row["width_nm"]))
    norm = np.trapezoid(2*np.pi*radius*abs(fields[1])**2, radius)
    if not norm > 0:
        raise ValueError("No reference field to compare")
    error = np.trapezoid(2*np.pi*radius*abs(fields[0]-fields[1])**2, radius)
    amplitude_error = np.trapezoid(2*np.pi*radius*(abs(fields[0])-abs(fields[1]))**2, radius)
    density_error = np.trapezoid(2*np.pi*radius*abs(abs(fields[0])**2-abs(fields[1])**2), radius)
    return {"relative_complex_l2": float(np.sqrt(error/norm)),
        "relative_amplitude_l2": float(np.sqrt(amplitude_error/norm)),
        "relative_density_l1": float(density_error/norm)}


def boundary_error(first, second, samples=16385):
    return boundary_metrics(first, second, samples)["relative_complex_l2"]


def compare_boundaries(first, second, maximum_z_nm=None):
    result = compare(first, second)  # Enforce physical/energy/reference identity.
    a, b = (json.loads(Path(name).read_text(encoding="utf-8")) for name in (first, second))
    rows = []
    for ma, mb in zip(a["gun"]["mode_records"], b["gun"]["mode_records"]):
        grouped = []
        for mode in (ma, mb):
            group = {}
            for state in mode["boundary_states"]:
                group.setdefault(state["z_nm"], []).append(state)
            grouped.append(group)
        for z in sorted(grouped[0].keys() & grouped[1].keys()):
            if maximum_z_nm is not None and z > maximum_z_nm:
                continue
            # First and last at a coincident set of physical masks are
            # unambiguous, unlike numerical boundary indices after refinement.
            sides = ("first", "last") if max(len(g[z]) for g in grouped) > 1 else ("first",)
            for side in sides:
                aa, bb = (g[z][0 if side == "first" else -1] for g in grouped)
                coarse = boundary_metrics(aa, bb)
                fine = boundary_metrics(aa, bb, 32769)
                change = max(abs(coarse[key]-fine[key]) for key in fine)
                if change > 2e-5:
                    raise ValueError(f"Comparison quadrature unresolved at {z} nm: {coarse}, {fine}")
                rows.append({"energy_ev": ma["energy_ev"], "z_nm": z, "side": side,
                    **fine, "comparison_quadrature_change": change,
                    "current_fractions": [aa["net_current_fraction"], bb["net_current_fraction"]]})
    result["boundary_comparisons"] = rows
    result["boundary_metric"] = "Physical field L2; exact matching planes; axial/radial phase retained; quadrature doubled"
    result["maximum_compared_boundary_z_nm"] = maximum_z_nm
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("first", type=Path)
    p.add_argument("second", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--maximum-z-nm", type=float)
    args = p.parse_args()
    result = compare_boundaries(args.first, args.second, args.maximum_z_nm)
    payload = json.dumps(result, indent=2)+"\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    else:
        print(payload)
