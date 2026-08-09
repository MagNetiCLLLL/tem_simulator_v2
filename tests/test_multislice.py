import numpy as np
import pytest

from temsim.physics.multislice import propagate_multislice
from temsim.physics import compute_backend, multislice


def _normalised_plane_wave(size=32):
    return np.ones((size, size), dtype=np.complex128) / size


def _propagate(wave, potential, **overrides):
    parameters = {
        "pixel_size_angstrom": 0.25,
        "wavelength_angstrom": 0.0197,
        "interaction_constant_rad_per_v_angstrom": 0.0065,
        "total_thickness_angstrom": 20.0,
        "target_slice_thickness_angstrom": 2.0,
        "bandwidth_fraction": 1.0,
    }
    parameters.update(overrides)
    return propagate_multislice(wave, potential, **parameters)


def test_vacuum_multislice_preserves_plane_wave_and_intensity():
    wave = _normalised_plane_wave()
    result, diagnostics = _propagate(wave, np.zeros_like(wave.real))

    assert result == pytest.approx(wave, rel=1.0e-12, abs=1.0e-12)
    assert diagnostics.model == "continuous_column_multislice"
    assert diagnostics.slice_count == 10
    assert diagnostics.slice_thickness_angstrom == pytest.approx(2.0)
    assert diagnostics.initial_integrated_intensity == pytest.approx(1.0)
    assert diagnostics.final_integrated_intensity == pytest.approx(1.0)
    assert diagnostics.maximum_relative_intensity_change < 1.0e-12


def test_zero_thickness_is_identity_even_for_nonzero_projected_potential():
    wave = _normalised_plane_wave()
    potential = np.full(wave.shape, 12.0)
    result, diagnostics = _propagate(
        wave, potential, total_thickness_angstrom=0.0
    )

    assert result == pytest.approx(wave)
    assert diagnostics.slice_count == 0
    assert diagnostics.total_thickness_angstrom == 0.0


def test_uniform_potential_produces_the_expected_global_phase():
    wave = _normalised_plane_wave()
    potential = np.full(wave.shape, 8.0)
    result, diagnostics = _propagate(wave, potential)
    expected = wave * np.exp(1j * 0.0065 * 8.0)

    assert result == pytest.approx(expected, rel=1.0e-11, abs=1.0e-12)
    assert diagnostics.maximum_phase_per_slice_rad == pytest.approx(
        0.0065 * 8.0 / diagnostics.slice_count
    )


def test_multislice_supports_a_leading_probe_batch():
    wave = _normalised_plane_wave(24)
    batch = np.stack((wave, wave * 1j, -wave), axis=0)
    potential = np.zeros(wave.shape)
    result, diagnostics = _propagate(batch, potential)

    assert result.shape == batch.shape
    assert result == pytest.approx(batch, rel=1.0e-12, abs=1.0e-12)
    assert diagnostics.initial_integrated_intensity == pytest.approx(3.0)
    assert diagnostics.final_integrated_intensity == pytest.approx(3.0)


def test_vacuum_multislice_supports_rectangular_grid_and_anisotropic_sampling():
    wave = np.ones((20, 28), dtype=np.complex128)
    wave /= np.linalg.norm(wave)
    result, diagnostics = _propagate(
        wave,
        np.zeros(wave.shape),
        pixel_size_angstrom=(0.3, 0.2),
    )

    assert result.shape == (20, 28)
    assert result == pytest.approx(wave, rel=1.0e-12, abs=1.0e-12)
    assert diagnostics.pixel_size_y_angstrom == pytest.approx(0.3)
    assert diagnostics.pixel_size_x_angstrom == pytest.approx(0.2)


def test_explicit_atomic_slices_use_their_own_projected_potentials():
    wave = _normalised_plane_wave(24)
    slices = np.stack(
        (np.full(wave.shape, 3.0), np.full(wave.shape, 5.0)), axis=0
    )
    result, diagnostics = _propagate(
        wave,
        slices,
        total_thickness_angstrom=None,
        slice_thicknesses_angstrom=np.array([1.0, 1.5]),
    )

    expected = wave * np.exp(1j * 0.0065 * 8.0)
    assert result == pytest.approx(expected, rel=1.0e-11, abs=1.0e-12)
    assert diagnostics.model == "atomic_slice_multislice"
    assert diagnostics.slice_count == 2
    assert diagnostics.total_thickness_angstrom == pytest.approx(2.5)
    assert diagnostics.slice_thickness_angstrom == pytest.approx(1.5)


def test_symmetric_split_converges_as_slice_thickness_is_reduced():
    size = 48
    axis = (np.arange(size) - size // 2) * 0.25
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    potential = 30.0 * np.exp(-(xx**2 + yy**2) / (2.0 * 0.8**2))
    wave = _normalised_plane_wave(size)

    coarse, _ = _propagate(
        wave, potential, target_slice_thickness_angstrom=4.0
    )
    medium, _ = _propagate(
        wave, potential, target_slice_thickness_angstrom=2.0
    )
    reference, _ = _propagate(
        wave, potential, target_slice_thickness_angstrom=0.25
    )
    coarse_error = np.linalg.norm(coarse - reference)
    medium_error = np.linalg.norm(medium - reference)

    assert medium_error < coarse_error
    assert medium_error < 2.0e-4


@pytest.mark.parametrize(
    "potential, message",
    [
        (np.zeros((2, 3, 4, 5)), "shape"),
        (np.zeros((31, 32)), "spatial shapes"),
    ],
)
def test_multislice_rejects_invalid_potential_shapes(potential, message):
    with pytest.raises(ValueError, match=message):
        _propagate(_normalised_plane_wave(), potential)


def test_cupy_multislice_matches_complex128_cpu_reference():
    if not compute_backend.cupy_capability().available:
        pytest.skip("CuPy CUDA backend unavailable")
    size = 64
    axis = (np.arange(size) - size // 2) * 0.25
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    wave = np.exp(
        -((xx / 2.2) ** 2 + (yy / 1.8) ** 2)
        + 1j * (0.25 * xx - 0.12 * yy)
    )
    wave /= np.linalg.norm(wave)
    potential = 35.0 * np.exp(
        -(xx**2 + yy**2) / (2.0 * 0.75**2)
    )
    cpu, _ = _propagate(
        wave,
        potential,
        target_slice_thickness_angstrom=0.5,
        compute_backend="NumPy CPU",
    )
    gpu, diagnostics = _propagate(
        wave,
        potential,
        target_slice_thickness_angstrom=0.5,
        compute_backend="CuPy CUDA",
    )

    relative_l2_error = np.linalg.norm(gpu - cpu) / np.linalg.norm(cpu)
    assert diagnostics.compute_backend == "CuPy CUDA"
    assert diagnostics.numeric_precision == "complex64 / float32"
    assert diagnostics.fallback_reason is None
    assert relative_l2_error < 3.0e-5


def test_cupy_runtime_failure_retries_the_complex128_cpu_reference(monkeypatch):
    monkeypatch.setattr(
        multislice,
        "cupy_module",
        lambda: (_ for _ in ()).throw(RuntimeError("synthetic GPU failure")),
    )
    wave = _normalised_plane_wave(24)
    result, diagnostics = _propagate(
        wave,
        np.zeros(wave.shape),
        compute_backend="CuPy CUDA",
    )

    assert result == pytest.approx(wave, rel=1.0e-12, abs=1.0e-12)
    assert diagnostics.compute_backend == "NumPy CPU"
    assert diagnostics.numeric_precision == "complex128 / float64"
    assert "synthetic GPU failure" in diagnostics.fallback_reason
