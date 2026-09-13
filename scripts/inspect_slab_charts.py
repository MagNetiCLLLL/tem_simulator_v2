"""Compare internal current-coordinate scales on a captured constant slab.

Connectors only change the numerical decomposition of the SAME field and
covariant derivative. They are not physical steps or additional sources.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from temsim.physics.covariant_boundary import _slab, _unitarity
from temsim.physics.scattering_load import compose
from temsim.physics.covariant_boundary import _solve, _compose
from scipy.linalg import expm


def connector(first, second, count):
    reflection = (first-second)/(first+second)
    transmission = 2*np.sqrt(first*second)/(first+second)
    eye = np.eye(count)
    return reflection*eye, transmission*eye, transmission*eye, -reflection*eye


def balanced(q, g, width, outer, inner):
    result, record = _slab(q, g, width, inner, lambda: False)
    result = compose(compose(connector(outer, inner, len(q)), result), connector(inner, outer, len(q)))
    return result, {**record, "outer_chart_unitarity": _unitarity(result)}


def growth_bounded(q, g, width, outer, inner, maximum_log_growth=.25):
    n=len(q); eye=np.eye(n)
    chart=np.block([[eye,eye],[1j*eye,-1j*eye]])
    inverse=.5*np.block([[eye,-1j*eye],[eye,1j*eye]])
    generator=inverse@np.block([[-1j*g,inner*eye],[-q/inner,-1j*g]])@chart
    growth=float(np.max(abs(np.linalg.eigvalsh((generator+generator.conj().T)/2))))
    doublings=max(0,int(np.ceil(np.log2(growth*width/maximum_log_growth)))) if growth else 0
    transfer=expm(generator*(width/2**doublings))
    a,b,c,d=transfer[:n,:n],transfer[:n,n:],transfer[n:,:n],transfer[n:,n:]
    inv_d=_solve(d,eye); reflected=-_solve(d,c)
    blocks=(reflected,inv_d,a+b@reflected,b@inv_d)
    error=_unitarity(blocks)
    for _ in range(doublings):
        blocks=_compose(blocks,blocks)
        error=max(error,_unitarity(blocks))
    blocks=compose(compose(connector(outer,inner,n),blocks),connector(inner,outer,n))
    return blocks,{"doublings":doublings,"maximum_unitarity_residual":error,"outer_chart_unitarity":_unitarity(blocks)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    with np.load(args.input) as f:
        width, kappa, carrier = f["width_kappa_carrier"]
        q = carrier**2*np.eye(len(f["residual"]))+f["residual"]
        g = f["connection"]
    scale = np.sqrt(np.linalg.norm(q, ord=2))
    rows, outputs = [], []
    with threadpool_limits(1):
        for inner in (kappa, scale/2, scale, scale*2):
            row = {"inner_kappa": float(inner), "outer_kappa": float(kappa)}
            try:
                result, record = balanced(q, g, width, kappa, inner)
                full = np.block([[result[0], result[1]], [result[2], result[3]]])
                row.update(record)
                if outputs:
                    row["relative_complex_matrix_difference"] = float(np.linalg.norm(full-outputs[0], ord=np.inf))
                outputs.append(full)
                row["status"] = "EXECUTED"
            except Exception as error:
                row.update(status="FAILED", error=str(error))
            rows.append(row)
            print(json.dumps(row), flush=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({"scope": "Numerical constant-slab comparison, not physical source acceptance",
            "input": str(args.input), "results": rows}, stream, indent=2)


if __name__ == "__main__":
    main()
