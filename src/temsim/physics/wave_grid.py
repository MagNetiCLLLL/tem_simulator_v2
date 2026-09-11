"""Bounded refinement of an executed complex field, never a new source.

The periodic complex Fourier interpolant retains every represented frequency,
including the negative Nyquist bin of an even lattice. It adds no recovered
physical detail and performs no filtering, fitting or probability correction.
Analytical phase carriers stay factored and the physical period stays fixed.
"""
from dataclasses import dataclass, replace
import math

import numpy as np

from temsim.physics.wave_flux import check_lossless_norm
from temsim.physics.wave_execution import check_available_memory


class WaveSamplingError(ValueError):
    """An operator has not been applied because its grid needs refinement."""

    def __init__(self, message, required_scale=2.):
        super().__init__(message)
        self.required_scale = float(required_scale)


@dataclass(frozen=True)
class WaveGridNumerics:
    automatic_refinement: bool = True
    maximum_pixels: int = 32768
    maximum_working_bytes: int = 72*1024**3

    def validate(self):
        if not isinstance(self.automatic_refinement, bool):
            raise ValueError("Automatic wave refinement must be a boolean")
        if isinstance(self.maximum_pixels, bool) or not isinstance(self.maximum_pixels, int) or not 32 <= self.maximum_pixels <= 65536:
            raise ValueError("Maximum wave grid pixels must be an integer from 32 to 65536")
        if isinstance(self.maximum_working_bytes, bool) or not isinstance(self.maximum_working_bytes, int) or self.maximum_working_bytes <= 0:
            raise ValueError("Wave working memory budget must be a positive integer")
        return self

    def check(self, shape, *, retained_bytes=0):
        self.validate()
        # Covers immutable input/output copies, FFT scratch, coordinates,
        # polynomial phase arrays and field propagation work, not total RSS.
        required = int(retained_bytes)+256*math.prod(shape)
        if max(shape) > self.maximum_pixels or required > self.maximum_working_bytes:
            raise ValueError(f"Wave refinement budget exceeded: grid={tuple(shape)}, estimated working bytes={required}; "
                             f"maximum_pixels={self.maximum_pixels}, maximum_working_bytes={self.maximum_working_bytes}. "
                             "The unresolved optical operator was not applied; increase the numerical budget.")
        check_available_memory(required-int(retained_bytes))
        return required


def refine_plane_wave(wave, shape, *, numerics=WaveGridNumerics(), retained_bytes=0):
    """Unitary Fourier embedding at fixed physical period and carrier gauge."""
    old = wave.amplitude.shape
    if len(shape) != 2 or any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < o
                              for n, o in zip(shape, old)):
        raise ValueError("Refinement needs two integer dimensions no smaller than the input")
    shape = tuple(int(n) for n in shape)
    numerics.check(shape, retained_bytes=retained_bytes+wave.amplitude.nbytes)
    if shape == old:
        return wave
    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(wave.amplitude), norm="ortho"))
    padded = np.zeros(shape, dtype=np.complex128)
    start = tuple(n//2-o//2 for n, o in zip(shape, old))
    padded[start[0]:start[0]+old[0], start[1]:start[1]+old[1]] = spectrum
    amplitude = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(padded), norm="ortho"))
    basis = wave.basis_m@np.diag((old[1]/shape[1], old[0]/shape[0]))
    result = replace(wave, amplitude=amplitude, basis_m=basis)
    check_lossless_norm(wave.probability, result.probability, context="Fourier wave refinement")
    return result


def apply_resolved_operator(wave, operator, *, numerics=WaveGridNumerics(), retained_bytes=0,
                            cancelled=lambda: False):
    """Retry only a typed sampling failure, always from the pre-operator wave."""
    numerics.check(wave.amplitude.shape, retained_bytes=retained_bytes)
    rows = []
    while True:
        if cancelled():
            raise InterruptedError("Wave refinement cancelled")
        try:
            return operator(wave), rows
        except WaveSamplingError as error:
            if not numerics.automatic_refinement:
                raise
            if not math.isfinite(error.required_scale):
                raise ValueError("Wave operator requires an unbounded refinement") from error
            factor = 2**max(1, math.ceil(math.log2(max(1., error.required_scale))))
            shape = tuple(n*factor for n in wave.amplitude.shape)
            required = numerics.check(shape, retained_bytes=retained_bytes+wave.amplitude.nbytes)
            previous = wave.amplitude.shape
            wave = refine_plane_wave(wave, shape, numerics=numerics, retained_bytes=retained_bytes)
            rows.append({"from_shape": previous, "to_shape": shape, "reason": str(error),
                         "estimated_working_bytes": required, "probability": wave.probability,
                         "method": "unitary complex Fourier embedding; fixed physical period and analytical carriers"})
