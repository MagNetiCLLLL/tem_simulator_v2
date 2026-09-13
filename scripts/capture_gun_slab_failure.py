"""Observe a failing physical gun slab without changing its returned operator.

All arguments are forwarded to inspect_surface_column. Diagnostic outputs
are not accepted sources. This observer records its own hash separately.
"""
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np

from scripts import inspect_surface_column
from temsim.physics import radial_gun_wave


def main():
    output = Path(sys.argv[sys.argv.index("--output")+1])
    original = radial_gun_wave.hermitian_slab
    def observed(q, g, width, kappa, *, carrier_k=None):
        try:
            return original(q, g, width, kappa, carrier_k=carrier_k)
        except Exception as error:
            scope = inspect.currentframe().f_back.f_locals
            i = scope["index"]
            np.savez(output/"failed_slab.npz", residual=q, connection=g,
                     width_kappa_carrier=np.array([width, kappa, carrier_k]))
            record = {"observer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "start_nm": float(scope["z"][i]), "end_nm": float(scope["z"][i+1]),
                "energy_ev": float(scope["energy_ev"]), "method_error": str(error),
                "scope": "Unmodified failing operator; no accepted result or omitted channels"}
            with (output/"failed_slab.json").open("x", encoding="utf-8") as stream:
                json.dump(record, stream, indent=2)
            raise
    radial_gun_wave.hermitian_slab = observed
    try:
        return inspect_surface_column.main()
    finally:
        radial_gun_wave.hermitian_slab = original


if __name__ == "__main__":
    raise SystemExit(main())
