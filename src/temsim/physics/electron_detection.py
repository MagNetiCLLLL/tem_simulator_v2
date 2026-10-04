"""Sample virtual electron arrivals from an executed observation-plane image.

The input is absolute probability density per reference electron, already
summed over independent coherent modes. An emitted electron either arrives in
one screen bin or is absent from this observation plane. Lost probability is
never restored by normalising the surviving image. This readout neither
propagates a wave nor calculates detector quantum efficiency or particle
tracks; bin-uniform positions describe only the executed display resolution.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np


MAX_EMITTED_ELECTRONS = 1_000_000
_DRAW_CHUNK = 65_536
# Matches the double-precision flux contract. Larger excesses are invalid
# probabilities, not a reason to normalise an executed beam automatically.
_PROBABILITY_ROUNDOFF_ATOL = 1e-10


def _readonly(array):
    """Detach numeric buffers with immutable backing, including empty arrays."""
    array = np.ascontiguousarray(array)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class ElectronDetectionResult:
    """Virtual arrivals at one selected screen for a fixed emission count.

    ``points_um`` has shape (detected_count, 2), columns (X,Y), in micrometres
    and emission order. ``counts`` has shape (nx, ny): the same X/Y display-bin
    convention as the input preview, not the native wave's Y/X convention.
    Both arrays are read-only. ``lost_count`` means not arriving in this
    probability image; it does not assign a physical loss mechanism.
    """

    points_um: np.ndarray
    counts: np.ndarray
    emitted_count: int
    detected_count: int
    lost_count: int
    detection_probability: float


def sample_electron_detections(density_per_um2, bounds_um, *, emitted_electrons: int,
                              seed: int = 0) -> ElectronDetectionResult:
    """Sample independent arrivals from an executed probability-density preview.

    ``density_per_um2`` is a finite, non-negative array of shape (nx, ny),
    measured per emitted/reference electron per square micrometre. Supply
    ``_Preview.density`` and ``_Preview.bounds_um`` from the executed coherent
    observation. Independent mode probabilities must already be summed;
    coherent amplitudes are not valid inputs. Bounds are ((xmin,xmax),
    (ymin,ymax)) in micrometres, and bins are uniform along each display axis.

    Integrated probability must be at most one, allowing 1e-10 roundoff.
    Sampling uses this absolute probability plus an explicit lost outcome.
    Conditional on arrival, a point is uniform within its display bin; it
    does not claim sub-bin wave detail or model a physical detector's response.

    Count and seed are integers (excluding booleans); the count is bounded by
    ``MAX_EMITTED_ELECTRONS`` and seed by the unsigned 64-bit range. Work is
    chunked, and the seed generates three variates per emitted electron, even
    when lost. Therefore increasing the count for identical inputs preserves
    every previously sampled hit as an exact prefix. The function never
    changes global random state or the probability input.
    """
    if (isinstance(emitted_electrons, (bool, np.bool_))
            or not isinstance(emitted_electrons, Integral)
            or not 0 <= emitted_electrons <= MAX_EMITTED_ELECTRONS):
        raise ValueError(f"emitted_electrons must be an integer from 0 to {MAX_EMITTED_ELECTRONS}")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral)
            or not 0 <= seed < 2**64):
        raise ValueError("seed must be an integer from 0 to 2**64 - 1")
    emitted_electrons, seed = int(emitted_electrons), int(seed)
    try:
        raw_density = np.asarray(density_per_um2)
        if np.iscomplexobj(raw_density):
            raise ValueError("complex amplitudes are not probability density")
        density = np.asarray(raw_density, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Screen probability density must be real numeric values") from exc
    if (density.ndim != 2 or 0 in density.shape or not np.all(np.isfinite(density))
            or np.any(density < 0.)):
        raise ValueError("Screen probability density must be a non-empty finite non-negative 2D array")
    try:
        if np.iscomplexobj(bounds_um):
            raise ValueError("complex bounds")
        bounds = np.asarray(bounds_um, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Screen bounds must be finite real X/Y edge pairs") from exc
    if bounds.shape != (2, 2) or not np.all(np.isfinite(bounds)):
        raise ValueError("Screen bounds must be finite X/Y edge pairs")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        widths = bounds[:, 1]-bounds[:, 0]
        bin_widths = widths/np.asarray(density.shape)
        bin_area = float(np.prod(bin_widths))
    if (not np.all(np.isfinite(widths)) or np.any(widths <= 0.)
            or not np.isfinite(bin_area) or bin_area <= 0.):
        raise ValueError("Screen bounds and bin area must be finite, ordered and positive")
    with np.errstate(over="ignore", invalid="ignore"):
        probabilities = density.ravel()*bin_area
        total = float(np.sum(probabilities, dtype=np.float64))
    if not np.isfinite(total) or total > 1.+_PROBABILITY_ROUNDOFF_ATOL:
        raise ValueError("Absolute screen probability per emitted electron must not exceed one")
    detection_probability = min(total, 1.)
    counts = np.zeros(density.shape, dtype=np.int64)
    if emitted_electrons == 0 or detection_probability == 0.:
        return ElectronDetectionResult(_readonly(np.empty((0, 2))), _readonly(counts),
                                       emitted_electrons, 0, emitted_electrons,
                                       detection_probability)

    cumulative = np.minimum(np.cumsum(probabilities), detection_probability)
    # Avoid a final summation-roundoff gap without altering genuine losses.
    cumulative[-1] = detection_probability
    random = np.random.Generator(np.random.PCG64(seed))
    points = np.empty((emitted_electrons, 2), dtype=np.float64)
    detected_count = 0
    flat_counts = counts.ravel()
    for start in range(0, emitted_electrons, _DRAW_CHUNK):
        draws = random.random((min(_DRAW_CHUNK, emitted_electrons-start), 3))
        arrived = draws[draws[:, 0] < detection_probability]
        bins = np.searchsorted(cumulative, arrived[:, 0], side="right")
        x_bins, y_bins = np.divmod(bins, density.shape[1])
        end = detected_count+len(bins)
        points[detected_count:end, 0] = bounds[0, 0]+(x_bins+arrived[:, 1])*bin_widths[0]
        points[detected_count:end, 1] = bounds[1, 0]+(y_bins+arrived[:, 2])*bin_widths[1]
        np.add.at(flat_counts, bins, 1)
        detected_count = end
    return ElectronDetectionResult(_readonly(points[:detected_count]), _readonly(counts),
                                   emitted_electrons, detected_count,
                                   emitted_electrons-detected_count, detection_probability)
