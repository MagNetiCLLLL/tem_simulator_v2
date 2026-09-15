"""Uncached production particle comparison of identical field evaluators.

Neither solver tolerances nor physical inputs change. This is implementation
parity, not field-mesh, source-sampling or microscope-design qualification.
"""
import argparse
from pathlib import Path
from time import perf_counter
import json

import numpy as np

from temsim.optics.column import default_state
from temsim.operating_modes import apply_operating_mode_pair
from temsim.optics.electron_gun.tracing import trace_feg_to_exit
from check_gun_matching_candidate import write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rays",type=int,default=49)
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("Choose a new scalar report")
    from temsim.physics.axis_field_interpolation import compiled_evaluate
    if compiled_evaluate is None:
        raise RuntimeError("Compiler unavailable; cannot report compiled parity")
    state = default_state()
    apply_operating_mode_pair(state,"nano_probe","diffraction")
    state.vacuum_map.enabled = False
    gun = state.electron_gun
    field = gun.electric_field
    original = field._regular.compiled
    runs = []
    try:
        for compiled in (False,True):
            field._regular.compiled = compiled
            start = perf_counter()
            result = trace_feg_to_exit(gun,args.rays)  # bypass trajectory cache
            elapsed = perf_counter()-start
            runs.append((result,elapsed))
            print(json.dumps(dict(compiled=compiled,seconds=elapsed,
                exit_alive=int(result.exit_bundle.alive.sum()))),flush=True)
    finally:
        field._regular.compiled = original
    reference,actual = (r[0] for r in runs)
    for name in ("alive","weight","energy_offset_ev"):
        np.testing.assert_array_equal(getattr(reference.exit_bundle,name),getattr(actual.exit_bundle,name))
    assert reference.blocked_key == actual.blocked_key
    np.testing.assert_allclose(reference.blocked_z_mm,actual.blocked_z_mm,rtol=0,atol=1e-6,equal_nan=True)
    differences = {}
    mask = actual.exit_bundle.alive
    for name in ("x_m","y_m","tx_rad","ty_rad"):
        a,b = (getattr(r.exit_bundle,name)[mask] for r in (reference,actual))
        differences[name] = float(np.max(abs(a-b),initial=0))
        np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-9 if name.endswith("_m") else 1e-8)
    write_report(dict(scope="UNCACHED_PARTICLE_IMPLEMENTATION_PARITY",rays=args.rays,
        numpy_seconds=runs[0][1],compiled_seconds=runs[1][1],
        speedup=runs[0][1]/runs[1][1],maximum_exit_differences=differences,
        exit_alive=int(mask.sum()),field=field.report,
        result="PASS; not mesh or design convergence"),args.output)


if __name__ == "__main__":
    main()
