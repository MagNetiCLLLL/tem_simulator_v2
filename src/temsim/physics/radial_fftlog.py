"""Cached complex, zero-order FFTLog with explicit float64 CPU/CUDA choice.

Uses the Mellin multiplier of J0, with the low-ringing offset supplied by
the caller. This changes execution resources, not the wave representation,
physical effects, sampling tests or phase convention. No CUDA fallback is
silent. The cache stores only numerical transform coefficients, not beams.
"""
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
import math

import numpy as np
from scipy.special import loggamma


_BACKEND = ContextVar("radial_hankel_backend", default="cpu")
_KERNELS = ContextVar("radial_hankel_kernels", default=None)
_CACHE_BYTES = 256*1024**2


@contextmanager
def hankel_backend(backend):
    if backend not in ("cpu", "cuda"):
        raise ValueError("Radial Hankel backend must be cpu or cuda")
    if backend == "cuda":
        try:
            import cupy
            if cupy.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("No CUDA device is available")
        except Exception as error:
            raise RuntimeError("The explicitly selected radial CUDA backend is unavailable") from error
    selection = _BACKEND.set(backend)
    cache = _KERNELS.set(OrderedDict())
    try:
        yield
    finally:
        _KERNELS.reset(cache)
        _BACKEND.reset(selection)


class Hankel:
    def __init__(self, count, dln, backend):
        if type(count) is not int or count < 2 or not math.isfinite(dln) or dln <= 0:
            raise ValueError("FFTLog needs at least two samples and positive finite log spacing")
        self.count = count
        if backend == "cuda":
            import cupy
            self.xp, self.fft = cupy, cupy.fft
        elif backend == "cpu":
            import scipy.fft
            self.xp, self.fft = np, scipy.fft
        else:
            raise ValueError("Unknown FFTLog backend")
        # U(2iy)=2^(2iy) Gamma(1/2+iy)/Gamma(1/2-iy). Retain the
        # actual offset separately, so changing a transform interval does
        # not require recomputing the Gamma ratio for its same spacing.
        y = np.linspace(0., np.pi*(count//2)/(count*dln), count//2+1)
        self.phase = self.xp.asarray(2*loggamma(.5+1j*y).imag+2*y*np.log(2.))
        self.y = self.xp.asarray(y)

    @property
    def nbytes(self):
        return self.phase.nbytes+self.y.nbytes

    def apply(self, value, offset, inverse=False):
        if value.shape != (self.count,) or value.dtype != np.dtype("complex128"):
            raise ValueError("Radial complex FFTLog requires one complex128 field of the declared size")
        if not math.isfinite(offset):
            raise ValueError("Hankel offset must be finite")
        xp = self.xp
        multiplier = xp.exp(1j*(self.phase-2*self.y*offset))
        if self.count % 2 == 0:
            multiplier[-1] = multiplier[-1].real
        spectrum = self.fft.rfft(xp.stack((value.real, value.imag)), axis=-1)
        spectrum = spectrum/xp.conj(multiplier) if inverse else spectrum*multiplier
        result = self.fft.irfft(spectrum, n=self.count, axis=-1)[:, ::-1]
        return result[0]+1j*result[1]


def complex_hankel(value, dln, offset, *, inverse=False):
    backend = _BACKEND.get()
    kernels = _KERNELS.get()
    if kernels is None:
        kernels = OrderedDict()
        _KERNELS.set(kernels)
    device = None
    if backend == "cuda":
        import cupy
        device = cupy.cuda.runtime.getDevice()
        free, _ = cupy.cuda.runtime.memGetInfo()
        # Includes batched real channels, complex FFT work and a safety
        # allowance. Memory exhaustion must not change physical channels.
        if free < 320*len(value)+64*1024**2:
            raise MemoryError("Insufficient free GPU memory for the full complex radial transform")
    key = (backend, device, len(value), float(dln))
    if key not in kernels:
        kernel = Hankel(len(value), float(dln), backend)
        while kernels and sum(k.nbytes for k in kernels.values())+kernel.nbytes > _CACHE_BYTES:
            kernels.popitem(last=False)
        kernels[key] = kernel
    kernel = kernels[key]
    kernels.move_to_end(key)
    native = kernel.xp.asarray(value, dtype=kernel.xp.complex128)
    result = kernel.apply(native, offset, inverse=inverse)
    return result if backend == "cpu" else kernel.xp.asnumpy(result)
