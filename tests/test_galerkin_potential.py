"""Independent finite-basis references, not a complete tip-to-image test."""
import numpy as np
import pytest
from scipy.linalg import expm

from temsim.physics.galerkin_potential import potential_action


def fourier_basis(shape, quadrature_shape):
    # Explicit exponentials, no FFT implementation shared with the operator.
    coordinates = [np.arange(m)-m//2 for m in quadrature_shape]
    frequencies = [np.arange(n)-n//2 for n in shape]
    yy, xx = np.meshgrid(*coordinates, indexing="ij")
    ky, kx = np.meshgrid(*frequencies, indexing="ij")
    return np.exp(2j*np.pi*(yy.ravel()[:, None]*ky.ravel()/quadrature_shape[0]
                          + xx.ravel()[:, None]*kx.ravel()/quadrature_shape[1])) / np.sqrt(np.prod(quadrature_shape))


@pytest.mark.parametrize("shape,quadrature", [((4, 6), (8, 12)), ((3, 5), (7, 11))])
def test_matches_independent_dense_hermitian_exponential(shape, quadrature):
    rng = np.random.default_rng(9301)
    a = rng.normal(size=shape)+1j*rng.normal(size=shape)
    a /= np.linalg.norm(a)
    phase = rng.uniform(-3., 5., quadrature)
    embedding = fourier_basis(shape, quadrature)
    h = embedding.conj().T @ (phase.ravel()[:, None]*embedding)
    np.testing.assert_allclose(h, h.conj().T, atol=2e-15)
    basis = fourier_basis(shape, shape)
    reference = basis @ expm(1j*h) @ basis.conj().T @ a.ravel()
    result, record = potential_action(a, phase)
    np.testing.assert_allclose(result.ravel(), reference, rtol=2e-13, atol=2e-14)
    assert abs(np.linalg.norm(result)-1) < 1e-13
    assert record["exponential_error_bound"] < 1e-13
    assert not record["renormalised"]
    restored, _ = potential_action(result, -phase)
    np.testing.assert_allclose(restored, a, atol=3e-14)


def test_constant_phase_keeps_global_phase_and_all_frequency_bins():
    a = np.random.default_rng(30).normal(size=(7, 8)).astype(complex)
    result, record = potential_action(a, np.full((14, 16), 71.2))
    np.testing.assert_allclose(result, a*np.exp(71.2j), atol=3e-15)
    assert record["applications"] == 0


def test_unresolved_scattering_does_not_wrap_into_negative_frequency():
    # Start in kx=+3 on N=8; V=cos(x) couples it to +2 and +4.
    # +4 is outside the basis, NOT its circular alias -4. With a small phase,
    # a seven-hop path to -4 is negligible, whereas circular multiplication
    # invents a first-order amplitude there.
    n, m, strength = 8, 16, .01
    basis = fourier_basis((n, n), (n, n))
    coefficients = np.zeros((n, n), complex)
    coefficients[n//2, -1] = 1.
    a = (basis @ coefficients.ravel()).reshape((n, n))
    fine_x = 2*np.pi*(np.arange(m)-m//2)/m
    phase = np.broadcast_to(strength*np.cos(fine_x), (m, m))
    result, _ = potential_action(a, phase)
    result_k = (basis.conj().T @ result.ravel()).reshape((n, n))
    circular_k = (basis.conj().T @ (a*np.exp(1j*phase[::2, ::2])).ravel()).reshape((n, n))
    assert abs(result_k[n//2, 0]) < 1e-14
    assert abs(circular_k[n//2, 0]) > .0049
    assert abs(result_k[n//2, -2]) > .0049


def test_periodic_analytic_solution_converges_with_basis():
    errors = []
    for n in (8, 16, 32):
        x = 2*np.pi*(np.arange(n)-n//2)/n
        xf = 2*np.pi*(np.arange(2*n)-n)/(2*n)
        a = np.ones((n, n), complex)/n
        phase = np.broadcast_to(1.5*np.cos(xf), (2*n, 2*n))
        result, _ = potential_action(a, phase)
        exact = a*np.exp(1.5j*np.cos(x))[None, :]
        errors.append(np.linalg.norm(result-exact))
    assert errors[1] < errors[0]*.01
    assert errors[2] < errors[1]*.01
    assert errors[2] < 1e-13


def test_deterministic_no_random_state_consumed():
    a = np.ones((4, 4), complex)/4
    phase = np.arange(64).reshape(8, 8)*.1
    before = np.random.get_state()
    first, _ = potential_action(a, phase)
    second, _ = potential_action(a, phase)
    np.testing.assert_array_equal(first, second)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])


@pytest.mark.parametrize("budget,expected_workers", ((1, 1), (2, 2), (16, 4)))
def test_threaded_fft_preserves_full_complex_phase_and_respects_cpu_budget(monkeypatch, budget, expected_workers):
    from temsim.physics import galerkin_potential as operator
    from types import SimpleNamespace
    rng = np.random.default_rng(72451)
    amplitude = rng.normal(size=(9, 12))+1j*rng.normal(size=(9, 12))
    amplitude /= np.linalg.norm(amplitude)
    phase = rng.uniform(-1.4, 2.7, (20, 25))
    original_amplitude, original_phase = amplitude.copy(), phase.copy()
    actual_fft, calls = operator.fft, []
    monkeypatch.setattr(operator, "numerical_thread_budget", lambda: budget)
    def track(name):
        def execute(value, **kwargs):
            calls.append((name, kwargs["workers"], kwargs.get("norm")))
            return getattr(actual_fft, name)(value, **kwargs)
        return execute
    monkeypatch.setattr(operator, "fft", SimpleNamespace(fft2=track("fft2"), ifft2=track("ifft2")))
    result, record = operator.potential_action(amplitude, phase)
    assert calls and {item[1] for item in calls} == {expected_workers}
    assert {item[2] for item in calls} == {"ortho"}
    assert record["fft_workers"] == expected_workers
    # Exercise the previous NumPy transforms with the identical physical
    # operator; independent dense-exponential tests above remain the oracle.
    def numpy_fft(name):
        def execute(value, *, workers, **kwargs):
            return getattr(np.fft, name)(value, **kwargs)
        return execute
    monkeypatch.setattr(operator, "fft", SimpleNamespace(fft2=numpy_fft("fft2"), ifft2=numpy_fft("ifft2")))
    reference, previous = operator.potential_action(amplitude, phase)
    np.testing.assert_allclose(result, reference, rtol=2e-13, atol=2e-14)
    assert abs(np.linalg.norm(result)-1) < 1e-13
    assert record["applications"] == previous["applications"]
    assert not record["renormalised"]
    np.testing.assert_array_equal(amplitude, original_amplitude)
    np.testing.assert_array_equal(phase, original_phase)


def test_invalid_quadrature_cancel_and_budgets_fail_before_execution():
    a = np.ones((4, 4), complex)/4
    with pytest.raises(ValueError, match="twice"):
        potential_action(a, np.ones((4, 4)))
    with pytest.raises(ValueError, match="Real potential"):
        potential_action(a, np.ones((8, 8), complex))
    with pytest.raises(MemoryError):
        potential_action(a, np.ones((8, 8)), maximum_working_bytes=1)
    with pytest.raises(InterruptedError):
        potential_action(a, np.ones((8, 8)), cancelled=lambda: True)
    with pytest.raises(ValueError, match="application budget"):
        potential_action(a, np.arange(64).reshape(8, 8), maximum_applications=1)


def cupy_device_or_skip():
    cp = pytest.importorskip("cupy")
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as error:
        pytest.skip(f"CUDA runtime unavailable: {error}")
    if not count:
        pytest.skip("No CUDA device")
    return cp


@pytest.mark.parametrize("resident", (False, True))
def test_gpu_galerkin_matches_dense_complex_action_without_iteration_transfers(monkeypatch, resident):
    cp = cupy_device_or_skip()
    from temsim.physics import wave_device
    rng = np.random.default_rng(62142)
    shape, quadrature = (4, 6), (8, 12)
    a = rng.normal(size=shape)+1j*rng.normal(size=shape)
    a /= np.linalg.norm(a)
    phase = rng.uniform(-2., 3., quadrature)
    embedding, basis = fourier_basis(shape, quadrature), fourier_basis(shape, shape)
    h = embedding.conj().T @ (phase.ravel()[:, None]*embedding)
    expected = basis @ expm(1j*h) @ basis.conj().T @ a.ravel()
    transfers, fft_calls = [], []
    host_copy = wave_device.to_host
    def track_host(value):
        transfers.append(value.shape)
        return host_copy(value)
    monkeypatch.setattr(wave_device, "to_host", track_host)
    for name in ("fft2", "ifft2"):
        implementation = getattr(cp.fft, name)
        def track_fft(value, *args, _implementation=implementation, **kwargs):
            assert isinstance(value, cp.ndarray) and value.dtype == cp.complex128
            fft_calls.append(value.shape)
            return _implementation(value, *args, **kwargs)
        monkeypatch.setattr(cp.fft, name, track_fft)
    result, record = potential_action(cp.asarray(a) if resident else a,
        cp.asarray(phase) if resident else phase, backend="Require GPU",
        maximum_device_working_bytes=32*1024**2)
    assert isinstance(result, cp.ndarray if resident else np.ndarray)
    assert transfers == ([] if resident else [shape])
    actual = cp.asnumpy(result) if resident else result
    np.testing.assert_allclose(actual.ravel(), expected, rtol=2e-13, atol=2e-14)
    assert abs(np.linalg.norm(actual)-1) < 1e-13
    assert len(fft_calls) == 2+2*record["applications"]
    assert record["compute_backend"] == "cupy"
    assert record["numeric_precision"] == "complex128 / float64"
    assert not record["fallback_reason"] and not record["renormalised"]


def test_gpu_galerkin_resource_and_physics_failures_do_not_fallback():
    cp = cupy_device_or_skip()
    from temsim.physics.compute_backend import GPUExecutionError
    a = cp.ones((4, 4), dtype=cp.complex128)/4
    phase = cp.arange(64, dtype=cp.float64).reshape(8, 8)/64
    with pytest.raises(GPUExecutionError, match="out_of_memory"):
        potential_action(a, phase, backend="Require GPU", maximum_device_working_bytes=1)
    with pytest.raises(ValueError, match="application budget"):
        potential_action(a, phase*100, backend="Require GPU", maximum_applications=1)
    with pytest.raises(InterruptedError):
        potential_action(a, phase, backend="Require GPU", cancelled=lambda: True)


@pytest.mark.parametrize("policy,physical_error", (("Prefer GPU", False), ("Require GPU", False), ("Prefer GPU", True)))
def test_gpu_execution_retry_uses_original_host_wave_only_for_resource_failure(monkeypatch, policy, physical_error):
    cp = cupy_device_or_skip()
    from temsim.physics import galerkin_potential as operator
    from temsim.physics.compute_backend import GPUExecutionError
    rng = np.random.default_rng(14231)
    a = rng.normal(size=(4, 6))+1j*rng.normal(size=(4, 6))
    a /= np.linalg.norm(a)
    phase = rng.uniform(-1., 2., (8, 12))
    original_a, original_phase = a.copy(), phase.copy()
    expected, _ = operator.potential_action(a, phase)
    attempts, actual_operator = [], operator._resident_potential_action
    error = ValueError("controlled invalid physical phase") if physical_error else GPUExecutionError("out_of_memory", "controlled executed FFT allocation")
    def fail_gpu(amplitude, potential, *, xp, **kwargs):
        attempts.append("numpy" if xp is np else "cupy")
        if xp is cp:
            cp.fft.fft2(amplitude)  # The device stage has actually started.
            raise error
        return actual_operator(amplitude, potential, xp=xp, **kwargs)
    monkeypatch.setattr(operator, "_resident_potential_action", fail_gpu)
    if physical_error or policy == "Require GPU":
        with pytest.raises(ValueError if physical_error else GPUExecutionError):
            operator.potential_action(a, phase, backend=policy)
        assert attempts == ["cupy"]
    else:
        result, record = operator.potential_action(a, phase, backend=policy)
        np.testing.assert_array_equal(result, expected)
        assert attempts == ["cupy", "numpy"]
        assert record["compute_backend"] == "numpy" and "out_of_memory" in record["fallback_reason"]
    np.testing.assert_array_equal(a, original_a)
    np.testing.assert_array_equal(phase, original_phase)
