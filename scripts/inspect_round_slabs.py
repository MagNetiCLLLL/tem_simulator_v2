"""Compare actual gun slabs with an independent matrix-exponential route.

This diagnoses the same finite-basis operator; it is not source acceptance.
The observer does not change any returned production operator or input.
"""
import argparse
from dataclasses import replace
import inspect
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from temsim.optics.column import default_state
from temsim.optics.electron_gun.tip_surface import load_tip_surface_reference, SurfaceCoherence
from temsim.physics import radial_gun_wave as radial
from temsim.physics.covariant_boundary import _slab
from temsim.physics.adaptive_scattering import scattering_distance
from temsim.calculation_manifest import solver_source_identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    state = default_state()
    state.electron_gun.emitter.surface_model = replace(load_tip_surface_reference(), coherence=SurfaceCoherence())
    original = radial.hermitian_slab
    rows, arrays = [], {}
    failure = None
    started = perf_counter()
    identity = solver_source_identity()
    def observed(q, g, width, kappa, *, carrier_k=None):
        local = inspect.currentframe().f_back.f_locals
        index = local["index"]
        start, stop = local["z"][index:index+2]
        result, diagnostic = original(q, g, width, kappa, carrier_k=carrier_k)
        if 500 <= start < 20000:
            reference, detail = _slab(carrier_k**2*np.eye(len(q))+q, g, width, kappa, lambda: False)
            error = scattering_distance(result, reference)
            row = {"start_nm": float(start), "end_nm": float(stop), "complex_operator_difference": error,
                "method": diagnostic, "reference": detail}
            rows.append(row)
            if error > 1e-5:
                name = f"slab_{len(rows)}"
                row["array_key"] = name
                arrays[name+"_residual"] = q
                arrays[name+"_connection"] = g
                arrays[name+"_width_k"] = np.array([width, kappa, carrier_k])
            print(json.dumps(row), flush=True)
        return result, diagnostic
    radial.hermitian_slab = observed
    try:
        with threadpool_limits(1):
            radial.prepare_round_gun(state.electron_gun, 2., .18744544167837376,
                radial.RadialGunNumerics(radial_modes=64, potential_quadrature=128,
                    relative_axial_step=.1, field_step_mm=.5), column_state=state,
                cancelled=lambda: perf_counter()-started > 240)
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        radial.hermitian_slab = original
        np.savez(args.output/"slabs.npz", **arrays)
        (args.output/"report.json").write_text(json.dumps({"scope": "ACTUAL_OPERATOR_DIAGNOSTIC_NOT_SOURCE_ACCEPTANCE",
            "solver_identity": identity, "implementation_unchanged": identity == solver_source_identity(),
            "status": "FAILED" if failure else "OPERATOR_CHECK_COMPLETE_NOT_SOURCE_ACCEPTANCE",
            "failure": failure, "elapsed_s": perf_counter()-started, "slabs": rows}, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
