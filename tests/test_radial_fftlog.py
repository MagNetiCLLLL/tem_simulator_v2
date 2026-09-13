"""Complex CPU/CUDA agreement with the public SciPy FFTLog reference."""
import numpy as np
import pytest
from scipy.fft import fht, ifht, fhtoffset

from temsim.physics.radial_fftlog import complex_hankel, hankel_backend


@pytest.mark.parametrize("count", (127, 128, 4096))
@pytest.mark.parametrize("inverse", (False, True))
def test_cached_cpu_matches_scipy_complex_field_including_nyquist(count, inverse):
    rng = np.random.default_rng(837)
    wave = rng.normal(size=count)+1j*rng.normal(size=count)
    spacing = .01
    transform = ifht if inverse else fht
    with hankel_backend("cpu"):
        # Offset changes use the same grid Gamma ratio, not an old offset.
        for initial in (-5., .3, 2.):
            offset = fhtoffset(spacing, 0., initial=initial)
            expected = transform(wave.real, spacing, 0., offset=offset)+1j*transform(wave.imag, spacing, 0., offset=offset)
            actual = complex_hankel(wave, spacing, offset, inverse=inverse)
            assert np.linalg.norm(actual-expected)/np.linalg.norm(expected) < 2e-12


def test_explicit_cuda_has_same_complex_field_and_no_implicit_fallback():
    cupy = pytest.importorskip("cupy")
    try:
        available = cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError:
        pytest.skip("No accessible CUDA runtime")
    if not available:
        pytest.skip("No CUDA device")
    radius = np.geomspace(1e-8, 100., 131072, endpoint=False)
    wave = radius*np.exp(-radius**2/2-.2j*radius**4)/np.sqrt(np.pi)
    spacing = float(np.log(radius[1]/radius[0]))
    offset = fhtoffset(spacing, 0., initial=np.log(radius[0]*radius[-1]))
    with hankel_backend("cpu"):
        expected = complex_hankel(wave, spacing, offset)
    with hankel_backend("cuda"):
        actual = complex_hankel(wave, spacing, offset)
        returned = complex_hankel(actual, spacing, offset, inverse=True)
    assert np.linalg.norm(actual-expected)/np.linalg.norm(expected) < 2e-12
    assert np.linalg.norm(returned-wave)/np.linalg.norm(wave) < 2e-12


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="cpu or cuda"):
        with hankel_backend("pretend-gpu"):
            pass
