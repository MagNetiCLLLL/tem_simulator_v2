"""Device-resident propagation with exact CPU physical-acceptance masks."""

from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics import compute_backend, stem_cuda_pipeline
from temsim.physics.multislice import propagate_multislice
from temsim.physics.stem_cuda_pipeline import (
    ResidentStemCudaResult,
    run_resident_stem_cuda,
)


def _inputs():
    shape = (24, 32)
    spacing = (0.31, 0.23)
    fy = np.fft.fftshift(np.fft.fftfreq(shape[0], d=spacing[0]))
    fx = np.fft.fftshift(np.fft.fftfreq(shape[1], d=spacing[1]))
    qx, qy = np.meshgrid(fx, fy, indexing="xy")
    spectrum = np.exp(-(qx**2 + qy**2) / 0.8**2 + 0.15j * qx * qy)
    y, x = np.indices(shape, dtype=float)
    x = (x - shape[1] // 2) * spacing[1]
    y = (y - shape[0] // 2) * spacing[0]
    potentials = tuple(
        np.stack([
            15.0 * np.exp(-((x - offset)**2 + y**2) / 0.6**2),
            22.0 * np.exp(-(x**2 + (y + offset)**2) / 0.8**2),
            12.0 * np.exp(-((x + offset)**2 + y**2) / 0.7**2),
        ])
        for offset in (0.1, 0.3)
    )
    valid = qx**2 + qy**2 < 0.75**2
    parameters = dict(
        base_spectrum=spectrum,
        frequencies_x=fx,
        frequencies_y=fy,
        scan_x_angstrom=np.linspace(-0.8, 0.6, 7),
        scan_y_angstrom=np.linspace(0.4, -0.2, 7),
        potential_configurations_v_angstrom=potentials,
        detector_masks={"left": valid & (qx < 0), "right": valid & (qx >= 0)},
        multislice_enabled=True,
        pixel_size_angstrom=spacing,
        wavelength_angstrom=0.0197,
        interaction_constant_rad_per_v_angstrom=0.0065,
        total_thickness_angstrom=3.0,
        target_slice_thickness_angstrom=1.0,
        slice_thicknesses_angstrom=np.array([0.7, 1.3, 1.0]),
        bandwidth_fraction=0.85,
        batch_size=3,
    )

    def masks(start, stop):
        thresholds = np.linspace(-0.18, 0.18, 7)[start:stop, None, None]
        left = (qx[None, :, :] < thresholds) & valid[None, :, :]
        right = ~left & valid[None, :, :]
        # A physical aperture rejects part of the remaining angular support.
        right &= qy[None, :, :] > -0.4
        return {"left": left, "right": right}

    return parameters, masks, valid


def _reference(parameters, masks, valid):
    fx, fy = np.meshgrid(
        parameters["frequencies_x"], parameters["frequencies_y"], indexing="xy",
    )
    x0 = parameters["scan_x_angstrom"][:, None, None]
    y0 = parameters["scan_y_angstrom"][:, None, None]
    shifted = parameters["base_spectrum"][None] * np.exp(-2j * np.pi * (fx * x0 + fy * y0))
    probes = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(shifted, axes=(-2, -1)), axes=(-2, -1)),
        axes=(-2, -1),
    )
    probes /= np.sqrt(np.sum(np.abs(probes)**2, axis=(-2, -1), keepdims=True))
    all_masks = masks(0, len(x0))
    values = {key: [] for key in all_masks}
    truncated = []
    for potential in parameters["potential_configurations_v_angstrom"]:
        wave, _ = propagate_multislice(
            probes, potential,
            pixel_size_angstrom=parameters["pixel_size_angstrom"],
            wavelength_angstrom=parameters["wavelength_angstrom"],
            interaction_constant_rad_per_v_angstrom=parameters["interaction_constant_rad_per_v_angstrom"],
            slice_thicknesses_angstrom=parameters["slice_thicknesses_angstrom"],
            bandwidth_fraction=parameters["bandwidth_fraction"],
        )
        diffraction = np.abs(np.fft.fftshift(np.fft.fft2(wave), axes=(-2, -1)))**2
        diffraction /= diffraction.sum(axis=(-2, -1), keepdims=True)
        for key, mask in all_masks.items():
            values[key].append(np.sum(diffraction * mask, axis=(-2, -1)))
        truncated.append(diffraction[:, ~valid].sum(axis=1))
    means = {key: np.mean(samples, axis=0) for key, samples in values.items()}
    sem = {
        key: np.sqrt(np.sum((np.std(samples, axis=0, ddof=1) / np.sqrt(2))**2)
                     / np.sum(means[key]**2))
        for key, samples in values.items()
    }
    return means, sem, np.mean(truncated, axis=0)


def test_resident_result_keeps_legacy_constructor_compatible():
    result = ResidentStemCudaResult({}, np.zeros(1), {}, None, SimpleNamespace(), {})
    assert result.truncated_flat is None


def test_validity_mask_rejects_wrong_shape_before_cuda_initialization(monkeypatch):
    parameters, _, _ = _inputs()
    monkeypatch.setattr(stem_cuda_pipeline, "cupy_module", lambda: pytest.fail("CUDA must not initialize"))
    with pytest.raises(ValueError, match="validity mask has the wrong shape"):
        run_resident_stem_cuda(**parameters, valid_reciprocal_mask=np.ones((2, 3), bool))


def test_resident_physical_masks_match_cpu_fraction_truncation_and_phonon_sem(monkeypatch):
    if not compute_backend.cupy_capability().available:
        pytest.skip("CuPy CUDA backend unavailable")
    cp = compute_backend.cupy_module()
    parameters, masks, valid = _inputs()
    means, sem, truncated = _reference(parameters, masks, valid)
    calls = []
    transfers = []
    progress = []
    original_asnumpy = cp.asnumpy

    def provider(start, stop):
        calls.append((start, stop))
        return masks(start, stop)

    def asnumpy(array, *args, **kwargs):
        transfers.append(array.nbytes)
        return original_asnumpy(array, *args, **kwargs)

    monkeypatch.setattr(cp, "asnumpy", asnumpy)
    result = run_resident_stem_cuda(
        **parameters, detector_mask_provider=provider, valid_reciprocal_mask=valid,
        progress_callback=lambda completed, total, _label: progress.append((completed, total)),
    )
    assert calls == [(0, 3), (3, 6), (6, 7)]
    assert progress == [(1, 3), (2, 3), (3, 3)]
    assert transfers == [4 * 7 * np.dtype(np.float64).itemsize]
    for key in means:
        np.testing.assert_allclose(result.fractions_flat[key], means[key], rtol=2e-4, atol=2e-7)
        assert result.detector_relative_standard_error[key] == pytest.approx(sem[key], rel=2e-3, abs=2e-7)
    np.testing.assert_allclose(result.truncated_flat, truncated, rtol=2e-4, atol=2e-7)
    np.testing.assert_allclose(result.uncollected_flat, 1 - sum(means.values()), rtol=2e-4, atol=2e-7)
    assert result.metrics["cuda_physical_detector_mask_upload_count"] == 6
    assert result.metrics["cuda_physical_detector_mask_upload_bytes"] == 2 * 7 * valid.size


@pytest.mark.parametrize("invalid", ["keys", "shape"])
def test_resident_rejects_invalid_physical_batch_masks(invalid):
    if not compute_backend.cupy_capability().available:
        pytest.skip("CuPy CUDA backend unavailable")
    parameters, masks, _ = _inputs()

    def provider(start, stop):
        result = masks(start, stop)
        if invalid == "keys":
            result.pop("right")
        else:
            result["right"] = result["right"][0]
        return result

    with pytest.raises(ValueError, match="mask keys|batch shape"):
        run_resident_stem_cuda(**parameters, detector_mask_provider=provider)
