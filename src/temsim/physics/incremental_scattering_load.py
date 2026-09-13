"""Exact suffix reuse in the same right-to-left reflected-load algorithm.

Changed propagation operators invalidate every dependent upstream response.
Unchanged downstream responses are immutable and reusable. The source trace
is not cached: each request still solves and propagates its current full BVP.
"""
from hashlib import sha256
import math

import numpy as np
from scipy.linalg import eigh, solve

from temsim.physics.coupled_low_energy import _hermitian
from temsim.physics.scattering_load import ScatteringLoad
from temsim.physics.wave_execution import check_available_memory


def _digest(arrays):
    digest = sha256()
    for value in arrays:
        value = np.ascontiguousarray(value)
        digest.update(repr((value.shape, value.dtype.str)).encode())
        digest.update(value.view(np.uint8))
    return digest.hexdigest()


class IncrementalOutgoingLoad:
    def __init__(self, maximum_working_bytes):
        if type(maximum_working_bytes) is not int or maximum_working_bytes <= 0:
            raise ValueError("Reflected-load cache requires a positive numerical memory budget")
        self.maximum_working_bytes = maximum_working_bytes
        self._keys, self._exit_key, self._load = (), None, None
        self.last_recomputed_intervals = 0

    def build(self, operators, exit_q, kappa, *, cancelled=lambda: False, progress_callback=None):
        if cancelled():
            raise InterruptedError("Incremental reflected load cancelled")
        if not math.isfinite(kappa) or kappa <= 0:
            raise ValueError("A positive numerical current-chart wave number is required")
        _hermitian(exit_q, "Exit wave-number square")
        n, count = len(exit_q), len(operators)
        required = 4*(count+1)*n*n*16  # Old and replacement response generations.
        if required > self.maximum_working_bytes:
            raise MemoryError("Reflected-load cache exceeds its numerical working-memory budget")
        keys = []
        for operator in operators:
            if cancelled():
                raise InterruptedError("Incremental reflected load cancelled")
            if len(operator) != 4 or any(np.shape(v) != (n, n) or not np.all(np.isfinite(v)) for v in operator):
                raise ValueError("Reflected load requires four finite matching scattering blocks")
            keys.append(_digest(operator))
        keys = tuple(keys)
        exit_key = (float(kappa), _digest((exit_q,)))
        compatible = self._load is not None and exit_key == self._exit_key
        # Adaptive refinement inserts/removes numerical intervals. Responses
        # depend only on their exact downstream suffix, not its array indices.
        # Match from the physical exit; never reuse across a changed operator.
        suffix = 0
        if compatible:
            for old, new in zip(reversed(self._keys), reversed(keys)):
                if old != new:
                    break
                suffix += 1
        last = count-suffix-1
        if compatible and keys == self._keys:
            self.last_recomputed_intervals = 0
            return self._load
        check_available_memory(2*(last+2)*n*n*16)
        if compatible:
            old_start = len(self._keys)-suffix
            reflections = [None]*(count-suffix)+list(self._load.reflections[old_start:])
            forwards = [None]*(count-suffix)+list(self._load.forwards[old_start:])
            out = self._load.output_wave_number
            reflection = reflections[last+1]
        else:
            eigen, vectors = eigh(exit_q, check_finite=False)
            k = np.sqrt(eigen.astype(complex))
            out = (vectors*k)@vectors.conj().T
            reflection = (vectors*((kappa-k)/(kappa+k)))@vectors.conj().T
            reflections, forwards = [None]*(count+1), [None]*count
            reflections[-1] = reflection
        for i in range(last, -1, -1):
            if cancelled():
                raise InterruptedError("Incremental reflected load cancelled")
            rl, tr, tl, rr = operators[i]
            forwards[i] = solve(np.eye(n)-rr@reflection, tl, check_finite=False)
            reflection = rl+tr@reflection@forwards[i]
            reflections[i] = reflection
            if progress_callback is not None and i % 64 == 0:
                progress_callback(last-i+1, last+1, "Updating changed reflected-load prefix")
        admittance = 1j*kappa*solve((np.eye(n)+reflection).T, (np.eye(n)-reflection).T, check_finite=False).T
        for value in (admittance, out, *reflections, *forwards):
            value.setflags(write=False)
        result = ScatteringLoad(kappa, admittance, out, tuple(reflections), tuple(forwards), ())
        if cancelled():
            raise InterruptedError("Incremental reflected load cancelled before publication")
        self._keys, self._exit_key, self._load = keys, exit_key, result
        self.last_recomputed_intervals = last+1
        return result
