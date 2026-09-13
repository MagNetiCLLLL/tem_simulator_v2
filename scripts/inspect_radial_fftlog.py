"""Independent backend comparison for the same zero-order complex FFTLog.

The Gamma-ratio diagonal depends on grid spacing; cache it once, apply the
actual offset phase on every call. No wave coefficient or phase is removed.
This is a numerical benchmark, not source or image acceptance.
"""
import argparse
import json
from time import perf_counter

import numpy as np
from scipy.fft import fht, ifht, fhtoffset
from scipy.special import loggamma


class Hankel:
    def __init__(self, count, dln, backend):
        self.count = count
        if backend == "cuda":
            import cupy
            self.xp = cupy
            self.fft = cupy.fft
        else:
            import scipy.fft
            self.xp = np
            self.fft = scipy.fft
        # Mellin multiplier for J0: 2^(2iy) Gamma(1/2+iy)/Gamma(1/2-iy).
        y = np.linspace(0., np.pi*(count//2)/(count*dln), count//2+1)
        self.phase = self.xp.asarray(2*loggamma(.5+1j*y).imag + 2*y*np.log(2.))
        self.y = self.xp.asarray(y)

    def apply(self, value, offset, inverse=False):
        xp = self.xp
        multiplier = xp.exp(1j*(self.phase-2*self.y*offset))
        if self.count % 2 == 0:
            multiplier[-1] = multiplier[-1].real
        spectrum = self.fft.rfft(xp.stack((value.real, value.imag)), axis=-1)
        spectrum = spectrum/xp.conj(multiplier) if inverse else spectrum*multiplier
        result = self.fft.irfft(spectrum, n=self.count, axis=-1)[:, ::-1]
        return result[0]+1j*result[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=65536)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    n = args.samples
    r = np.geomspace(1e-10, 100., n, endpoint=False)
    dln = float(np.log(r[1]/r[0]))
    offset = fhtoffset(dln, 0., initial=np.log(r[0]*r[-1]))
    wave = r*np.exp(-r*r/2-1j*r**4/5)/np.sqrt(np.pi)
    start = perf_counter()
    reference = fht(wave.real, dln, 0., offset=offset)+1j*fht(wave.imag, dln, 0., offset=offset)
    scipy_s = perf_counter()-start
    hankel = Hankel(n, dln, args.backend)
    native = hankel.xp.asarray(wave)
    def sync():
        if args.backend == "cuda":
            hankel.xp.cuda.get_current_stream().synchronize()
    hankel.apply(native, offset); sync()
    start = perf_counter()
    for _ in range(3):
        actual = hankel.apply(native, offset)
    sync()
    elapsed = (perf_counter()-start)/3
    returned = hankel.apply(actual, offset, inverse=True)
    if args.backend == "cuda":
        actual = hankel.xp.asnumpy(actual)
        returned = hankel.xp.asnumpy(returned)
    print(json.dumps({"backend": args.backend, "samples": n, "scipy_uncached_s": scipy_s,
        "candidate_resident_s": elapsed, "relative_complex_difference": float(np.linalg.norm(actual-reference)/np.linalg.norm(reference)),
        "relative_roundtrip_difference": float(np.linalg.norm(returned-wave)/np.linalg.norm(wave)),
        "scope": "FFTLOG_KERNEL_ONLY_NOT_GUN_OR_IMAGES"}), flush=True)


if __name__ == "__main__":
    main()
