"""Complex64 CUDA Fresnel norm control on a non power-of-two grid."""
import numpy as np
import pytest
from temsim.physics import compute_backend
from temsim.physics.cuda_multislice_plan import _propagate


@pytest.mark.parametrize("bandlimited", [False, True])
def test_cuda_fresnel_keeps_incoming_norm_and_real_band_loss(bandlimited, record_property):
    if not compute_backend.cupy_capability().available:
        pytest.skip("CuPy CUDA backend unavailable")
    cp = compute_backend.cupy_module()
    n = 150
    y, x = np.indices((n,n), dtype=float)
    x -= n//2; y -= n//2
    wave = np.exp(-(x*x+y*y)/(2*11.**2) + .31j*x + .07j*y)
    wave /= np.linalg.norm(wave)
    # An already attenuated beam must not be restored to unit probability.
    incoming = np.stack((wave*np.sqrt(.37), wave*np.sqrt(.81))).astype(np.complex64)
    qx, qy = np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n))
    mask = qx*qx+qy*qy < .053**2 if bandlimited else np.ones((n,n),bool)
    propagator = mask*np.exp(-1j*.37*(qx*qx+qy*qy))
    reference = incoming.astype(np.complex128)
    actual = cp.asarray(incoming)
    device_propagator = cp.asarray(propagator,dtype=cp.complex64)
    steps = 32
    for _ in range(steps):
        reference = np.fft.ifft2(np.fft.fft2(reference,axes=(-2,-1))*propagator,axes=(-2,-1))
        actual = _propagate(actual,device_propagator,xp=cp)
    actual = cp.asnumpy(actual)
    norm = lambda a:np.sum(abs(a.astype(np.complex128))**2,axis=(-2,-1))
    # Each complex64 write rounds both components. Bound accumulated norm
    # error by one float32 epsilon per step; demanding double-precision
    # conservation here would misstate this accelerated path's precision.
    roundoff_bound = steps * np.finfo(np.float32).eps
    absolute_error = np.abs(norm(actual) - norm(reference))
    relative_error = absolute_error / norm(reference)
    record_property("maximum_norm_absolute_error", float(np.max(absolute_error)))
    record_property("maximum_norm_relative_error", float(np.max(relative_error)))
    record_property("accumulated_float32_roundoff_bound", float(roundoff_bound))
    np.testing.assert_allclose(norm(actual), norm(reference), rtol=roundoff_bound, atol=0.)
    np.testing.assert_allclose(actual,reference,rtol=3e-4,atol=2e-7)
    assert norm(actual)[0] < .38
    if bandlimited:
        assert np.all(norm(reference) < .9*norm(incoming))
    else:
        np.testing.assert_allclose(norm(actual), norm(incoming), rtol=roundoff_bound, atol=0.)
    # Fully rejected support remains zero, including after prior attenuation.
    np.testing.assert_array_equal(cp.asnumpy(_propagate(cp.asarray(incoming),cp.zeros((n,n),dtype=cp.complex64),xp=cp)),0.)
