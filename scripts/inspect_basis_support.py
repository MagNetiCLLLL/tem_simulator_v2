"""Measure best-approximation loss of an executed field in another gun basis.

This is a representation diagnostic, not a transported or configurable source.
Complex phase is retained; coefficients are orthogonal projections, never a
fitted phase, amplitude rescaling or current correction. Physical settings and
energy identities are checked by the existing gun comparator.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from scripts.compare_surface_gun import compare


def sampled_basis(row, radius, count):
    x = (radius / row["width_nm"])**2
    columns = []
    previous, current = np.zeros_like(x), np.exp(-x/2)
    for n in range(count):
        columns.append(current)
        previous, current = current, ((2*n+1-x)*current-n*previous)/(n+1)
    phase = np.exp(.5j*row["reference_k_per_nm"]*row["curvature_per_nm"]*radius**2
                   +1j*row.get("quartic_phase_per_nm4", 0.)*radius**4)
    return np.stack(columns, axis=1)*phase[:, None]/(np.sqrt(np.pi)*row["width_nm"])


def projection_defect(executed, destination, samples=16385):
    if executed["z_nm"] != destination["z_nm"]:
        raise ValueError("Basis-support comparison requires the same physical plane")
    coefficients = np.asarray(executed["coefficients_real"])+1j*np.asarray(executed["coefficients_imag"])
    count = len(destination["coefficients_real"])
    extent = max(row["width_nm"]*np.sqrt(4*n+160)
                 for row, n in ((executed, len(coefficients)), (destination, count)))
    radius = np.linspace(0., extent, samples)
    weights = 2*np.pi*radius*(radius[1]-radius[0])
    weights[[0, -1]] *= .5
    wave = sampled_basis(executed, radius, len(coefficients))@coefficients
    basis = sampled_basis(destination, radius, count)
    gram = basis.conj().T@(weights[:, None]*basis)
    defect = float(np.linalg.norm(gram-np.eye(count), 2))
    # Use the quadrature Gram matrix for the least-squares projection. The
    # separately reported Gram defect prevents sampling errors masquerading
    # as physical or basis loss.
    projection = np.linalg.solve(gram, basis.conj().T@(weights*wave))
    remainder = wave-basis@projection
    norm = float(np.vdot(wave, weights*wave).real)
    if not norm > 0:
        raise ValueError("Executed wave has no finite norm")
    return {"best_approximation_relative_l2": float(np.sqrt(np.vdot(remainder, weights*remainder).real/norm)),
            "quadrature_gram_defect": defect}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--maximum-z-nm", type=float, default=100000.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.first, args.second)
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in (args.first, args.second)]
    rows = []
    with threadpool_limits(1):
        for modes in zip(*(report["gun"]["mode_records"] for report in reports)):
            groups = [{row["z_nm"]: row for row in mode["boundary_states"]}
                      for mode in modes]
            for z in sorted(groups[0].keys() & groups[1].keys()):
                if z > args.maximum_z_nm:
                    continue
                for i in (0, 1):
                    source, destination = groups[i][z], groups[1-i][z]
                    coarse = projection_defect(source, destination)
                    fine = projection_defect(source, destination, 32769)
                    change = abs(coarse["best_approximation_relative_l2"]-fine["best_approximation_relative_l2"])
                    if change > 2e-5 or fine["quadrature_gram_defect"] > 2e-5:
                        raise ValueError(f"Basis comparison quadrature unresolved at {z} nm")
                    rows.append({"energy_ev": modes[i]["energy_ev"], "z_nm": z,
                        "direction": "first_into_second" if i == 0 else "second_into_first",
                        **fine, "quadrature_change": change})
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({"scope": "BEST_APPROXIMATION_DIAGNOSTIC_NOT_TRANSPORT_OR_ACCEPTANCE",
            "first": str(args.first), "second": str(args.second), "rows": rows}, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
