"""Compare actual electrode projections with independent positive quadrature.

This re-reads executed settings, never replaces a beam or optical component.
Only radial potential integration is audited, not full wave convergence.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from temsim.instrument_snapshot import InstrumentSnapshot
from temsim.physics.grounded_tip_field import grounded_field
from temsim.physics.radial_gun_wave import (piecewise_radial_potential, resolved_radial_quadrature,
    KINETIC_NM2_PER_EV, REST_EV)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    with threadpool_limits(1):
        for path in args.reports:
            report = json.loads(path.read_text(encoding="utf-8"))
            state = InstrumentSnapshot.from_dict(report["instrument_snapshot"]).restore()
            field = grounded_field(state.electron_gun)
            for mode in report["gun"]["mode_records"]:
                energy = mode["energy_ev"]
                for boundary in mode["boundary_states"]:
                    if boundary["z_nm"] not in (100., 1000., 10000.):
                        continue
                    width, z = boundary["width_nm"], boundary["z_nm"]
                    a = np.asarray(boundary["coefficients_real"])+1j*np.asarray(boundary["coefficients_imag"])
                    count, limit = len(a), width*np.sqrt(4*len(a)+160)
                    knots = field.r*1e9
                    if limit > knots[-1]:
                        raise ValueError("Basis reaches outside the solved electrode domain")
                    knots = np.r_[knots[knots < limit], limit]
                    def kinetic(radius):
                        xyz = np.column_stack((radius, np.zeros(len(radius)), np.full(len(radius), z)))*1e-9
                        return energy+field.potential_rise_v_at_global_positions(xyz)
                    axis = kinetic(np.array([0.]))[0]
                    actual = piecewise_radial_potential(width, count, knots, kinetic(knots), axis)
                    references = []
                    for order in (32, 64):
                        radius, weighted = resolved_radial_quadrature(width, count, knots, order)
                        local = kinetic(radius)
                        residual = KINETIC_NM2_PER_EV*(local-axis)*(1+(local+axis)/(2*REST_EV))
                        references.append((weighted*residual)@weighted.T)
                    error = actual-references[-1]
                    rows.append({"input": str(path), "energy_ev": energy, "z_nm": z,
                        "analytic_matrix_norm_per_nm2": float(np.linalg.norm(actual, 2)),
                        "matrix_error_norm_per_nm2": float(np.linalg.norm(error, 2)),
                        "relative_matrix_error": float(np.linalg.norm(error)/max(np.linalg.norm(references[-1]), 1e-300)),
                        "wave_action_error_per_nm2": float(np.linalg.norm(error@a)/np.linalg.norm(a)),
                        "quadrature_change_per_nm2": float(np.linalg.norm(references[1]-references[0], 2))})
    output = {"scope": "Actual-field radial integration audit, not transported source acceptance",
        "driver_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "inputs": [{"path": str(p), "sha256": sha256(p.read_bytes()).hexdigest()} for p in args.reports], "rows": rows}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps({"rows": len(rows), "maximum_relative_matrix_error": max(r["relative_matrix_error"] for r in rows),
        "maximum_wave_action_error_per_nm2": max(r["wave_action_error_per_nm2"] for r in rows)}))


if __name__ == "__main__":
    main()
