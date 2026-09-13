"""Check the actual finite-element / radial interface, not gun current alone."""
import argparse
import json
from pathlib import Path

import numpy as np
from temsim.physics.radial_gun_wave import basis_values


def inspect(directory):
    report = json.loads((directory/"report.json").read_text())
    with np.load(directory/"complex_states.npz") as arrays:
        radius = arrays["near_r_nm"].copy()
    nodes, weights = np.polynomial.legendre.leggauss(8)
    rows = []
    for mode, payload in zip(report["gun"]["mode_records"], report["gun"]["radial_output_modes"]):
        far = mode["far_field"]
        n = len(payload["coefficients_real"])
        gamma = mode["near_flux"].get("radial_phase_gradient_nm2", 0.)
        curvature = 0. if gamma else far["initial_curvature_per_nm"]
        trace = basis_values(radius, far["initial_width_nm"], curvature,
                             far["reference_wave_number_per_nm"], n)
        gram = np.zeros((n, n), complex)
        for left in range(0, len(radius)-2, 2):
            r = radius[left:left+3]
            for u, w in zip((nodes+1)/2, weights/2):
                basis = np.array(((1-u)*(1-2*u), 4*u*(1-u), u*(2*u-1)))
                derivative = np.array((4*u-3, 4-8*u, 4*u-1))
                interpolated = basis@trace[left:left+3]
                gram += 2*np.pi*w*(basis@r)*(derivative@r)*np.outer(interpolated.conj(), interpolated)
        eigen = np.linalg.eigvalsh(gram)
        rows.append({"energy_ev": mode["energy_ev"], "radial_modes": n,
            "minimum_interface_gram_eigenvalue": float(eigen[0]),
            "maximum_interface_gram_eigenvalue": float(eigen[-1]),
            "interface_gram_error": float(max(abs(eigen-1))),
            "scope": "Full retained trace interpolation check; not output or image acceptance"})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.directory), indent=2))
