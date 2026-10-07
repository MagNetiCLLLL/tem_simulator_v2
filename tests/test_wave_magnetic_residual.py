"""Independent finite-matrix, Fourier, and real-device residual acceptance."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import expm

from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.wave_device import device_scope, to_host
from temsim.physics.wave_magnetic_residual import apply_magnetic_residual


def _wave(shape=(6, 8)):
    rng = np.random.default_rng(8126)
    amplitude = rng.normal(size=shape)+1j*rng.normal(size=shape)
    amplitude *= np.sqrt(.37/np.sum(abs(amplitude)**2))
    return PlaneWave(amplitude, np.array(((.8e-9, .13e-9), (-.2e-9, 1.1e-9))),
        np.array((2e-9, -3e-9)), np.array(((1.2e4, 2e3), (2e3, -9e3))),
        np.array((.0002, -.0003)))


def _frequencies(wave, wavelength):
    ny, nx = wave.amplitude.shape
    fy, fx = np.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nx), indexing="ij")
    return wavelength*np.einsum("ij,jyx->iyx", np.linalg.inv(wave.basis_m).T, np.stack((fx, fy)))


def _coefficients(shape):
    y, x = np.indices(shape)
    velocity = np.stack((.006*np.cos(2*np.pi*x/shape[1]),
                         .004*np.sin(2*np.pi*y/shape[0]+.3)))
    scalar = .0006*np.cos(2*np.pi*(x/shape[1]+y/shape[0]))
    return velocity, scalar


def test_constant_coefficients_match_exact_fourier_phase_and_preserve_carrier():
    wave, wavelength, distance = _wave(), 2e-12, 3e-9
    velocity = np.broadcast_to(np.array((.008, -.004))[:, None, None], (2, *wave.amplitude.shape))
    scalar = np.full(wave.amplitude.shape, .0004)
    actual, record = apply_magnetic_residual(wave, velocity, scalar, wavelength, distance,
                                            error_budget=2e-10)
    eigenvalues = scalar+np.sum(velocity*_frequencies(wave, wavelength), axis=0)
    expected = np.fft.ifft2(np.fft.fft2(wave.amplitude)*np.exp(-2j*np.pi*distance/wavelength*eigenvalues))
    np.testing.assert_allclose(actual.amplitude, expected, rtol=0., atol=2e-13)
    assert actual.probability == pytest.approx(wave.probability, rel=2e-13)
    assert record["substeps"] > 1
    assert record["series_error_bound"] <= 1e-12
    for key in ("basis_m", "origin_m", "curvature_m1", "tilt_rad"):
        np.testing.assert_array_equal(getattr(actual, key), getattr(wave, key))
    undo, _ = apply_magnetic_residual(actual, velocity, scalar, wavelength, -distance, error_budget=2e-10)
    np.testing.assert_allclose(undo.amplitude, wave.amplitude, rtol=0., atol=3e-13)


def test_variable_coefficients_match_independent_hermitian_matrix_exponential():
    wave, wavelength, distance = _wave((4, 6)), 2e-12, 2e-9
    velocity, scalar = _coefficients(wave.amplitude.shape)
    actual, record = apply_magnetic_residual(wave, velocity, scalar, wavelength, distance,
                                            error_budget=1e-13)
    ny, nx = wave.amplitude.shape
    def dft(n):
        return np.exp(-2j*np.pi*np.outer(np.arange(n), np.arange(n))/n)/np.sqrt(n)
    transform = np.kron(dft(ny), dft(nx))
    hamiltonian = np.diag(scalar.ravel()).astype(complex)
    for v, frequency in zip(velocity, _frequencies(wave, wavelength)):
        derivative = transform.conj().T@np.diag(frequency.ravel())@transform
        multiplication = np.diag(v.ravel())
        hamiltonian += .5*(multiplication@derivative+derivative@multiplication)
    np.testing.assert_allclose(hamiltonian, hamiltonian.conj().T, rtol=0., atol=1e-18)
    expected = expm(-2j*np.pi*distance/wavelength*hamiltonian)@wave.amplitude.ravel()
    np.testing.assert_allclose(actual.amplitude.ravel(), expected, rtol=0., atol=2e-14)
    assert actual.probability == pytest.approx(.37, rel=1e-13)
    assert record["series_error_bound"] <= 1e-13


def test_zero_correction_is_exact_identity_without_normalization():
    wave = _wave()
    actual, record = apply_magnetic_residual(wave, np.zeros((2, *wave.amplitude.shape)),
        np.zeros(wave.amplitude.shape), 2e-12, 1., error_budget=1e-12)
    np.testing.assert_array_equal(actual.amplitude, wave.amplitude)
    assert record["operator_applications"] == 0
    assert record["input_probability"] == record["output_probability"]


@pytest.mark.parametrize("budget", [0., -1., np.inf, np.nan])
def test_invalid_error_budget_is_rejected_before_input_changes(budget):
    wave = _wave()
    before = wave.amplitude.copy()
    velocity, scalar = _coefficients(wave.amplitude.shape)
    with pytest.raises(ValueError, match="error budget"):
        apply_magnetic_residual(wave, velocity, scalar, 2e-12, 1e-9, error_budget=budget)
    np.testing.assert_array_equal(wave.amplitude, before)


@pytest.mark.parametrize("budget", [1e-30, np.finfo(float).eps/2])
def test_submachine_precision_budget_is_rejected_for_nonzero_propagation(budget):
    wave = _wave()
    before = wave.amplitude.copy()
    velocity, scalar = _coefficients(wave.amplitude.shape)
    with pytest.raises(ValueError, match="below float64 machine precision"):
        apply_magnetic_residual(wave, velocity, scalar, 2e-12, 1e-9, error_budget=budget)
    np.testing.assert_array_equal(wave.amplitude, before)


def test_zero_distance_remains_exact_identity_at_submachine_budget():
    wave = _wave()
    velocity, scalar = _coefficients(wave.amplitude.shape)
    actual, record = apply_magnetic_residual(wave, velocity, scalar, 2e-12, 0., error_budget=1e-30)
    np.testing.assert_array_equal(actual.amplitude, wave.amplitude)
    assert record["operator_applications"] == 0
    assert record["series_error_bound"] == 0.


@pytest.mark.parametrize("invalid", ["complex", "shape", "nonfinite", "substeps"])
def test_invalid_coefficients_and_infeasible_work_are_rejected(invalid):
    wave = _wave()
    velocity, scalar = _coefficients(wave.amplitude.shape)
    match = {"complex": "real", "shape": "shapes", "nonfinite": "finite", "substeps": "substep budget"}[invalid]
    if invalid == "complex":
        velocity = velocity.astype(complex)
    elif invalid == "shape":
        scalar = scalar[:-1]
    elif invalid == "nonfinite":
        scalar[0, 0] = np.nan
    else:
        scalar[:] = 1e10
    with pytest.raises(ValueError, match=match):
        apply_magnetic_residual(wave, velocity, scalar, 2e-12, 1e-9, error_budget=1e-12)


def test_cancellation_during_series_preserves_input():
    wave = _wave()
    before = wave.amplitude.copy()
    velocity, scalar = _coefficients(wave.amplitude.shape)
    calls = 0
    def cancelled():
        nonlocal calls
        calls += 1
        return calls == 3
    with pytest.raises(InterruptedError, match="cancelled"):
        apply_magnetic_residual(wave, velocity, scalar, 2e-12, 3e-9,
            error_budget=1e-12, cancelled=cancelled)
    np.testing.assert_array_equal(wave.amplitude, before)


def test_actual_gpu_matches_cpu_full_complex_envelope():
    cp = pytest.importorskip("cupy")
    from temsim.physics.compute_backend import cupy_capability
    status = cupy_capability()
    if not status.available:
        pytest.skip(status.detail)
    wave = _wave()
    velocity, scalar = _coefficients(wave.amplitude.shape)
    expected, _ = apply_magnetic_residual(wave, velocity, scalar, 2e-12, 2e-9, error_budget=1e-13)
    with device_scope("Require GPU") as device:
        assert device.backend == "cupy"
        gpu_wave = replace(wave, amplitude=cp.asarray(wave.amplitude))
        actual, record = apply_magnetic_residual(gpu_wave, cp.asarray(velocity), cp.asarray(scalar),
            2e-12, 2e-9, error_budget=1e-13)
        assert record["compute_backend"] == "cupy"
        assert hasattr(actual.amplitude, "__cuda_array_interface__")
        np.testing.assert_allclose(to_host(actual.amplitude), expected.amplitude, rtol=0., atol=3e-14)
