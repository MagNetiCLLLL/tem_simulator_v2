"""Bounded transmission snapshots, raw-array mutation safety and CUDA parity."""

from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics import compute_backend
from temsim.physics.cuda_multislice_plan import (
    CuPyMultislicePlan,
    MAX_TRANSMISSION_CACHE_BYTES,
    _transmission_cache_budget,
)
from temsim.physics.multislice import propagate_multislice
from temsim.physics.stem_cuda_pipeline import run_resident_stem_cuda


class _NumpyDevice:
    """CPU-only cache/control-flow fixture, not evidence of GPU execution."""

    cuda = SimpleNamespace(
        get_current_stream=lambda: SimpleNamespace(synchronize=lambda: None),
        runtime=SimpleNamespace(memGetInfo=lambda: (8 * 1024**3, 16 * 1024**3)),
    )

    def __getattr__(self, name):
        return getattr(np, name)


@pytest.fixture(params=["numpy_control_flow", "actual_cuda"])
def device(request):
    if request.param == "numpy_control_flow":
        return _NumpyDevice()
    if not compute_backend.cupy_capability().available:
        pytest.skip("CuPy CUDA backend unavailable")
    return compute_backend.cupy_module()


def _host(device, array):
    return np.asarray(array) if isinstance(device, _NumpyDevice) else device.asnumpy(array)


def _fixture(explicit=True):
    y, x = np.indices((14, 18), dtype=float)
    x = (x - 9) * 0.2
    y = (y - 7) * 0.3
    base = 35 * np.exp(-(x**2 + y**2) / 0.8**2)
    potential = np.stack((base, np.roll(base, 1, axis=1), base * 0.7)) if explicit else base
    wave = np.exp(-(x**2 + y**2) / 1.4**2 + 0.2j * x)
    wave /= np.linalg.norm(wave)
    parameters = dict(
        pixel_size_angstrom=(0.3, 0.2),
        wavelength_angstrom=0.0197,
        interaction_constant_rad_per_v_angstrom=0.0065,
        bandwidth_fraction=0.85,
    )
    if explicit:
        parameters["slice_thicknesses_angstrom"] = np.array([0.7, 1.3, 1.0])
    else:
        parameters.update(total_thickness_angstrom=7.0, target_slice_thickness_angstrom=2.0)
    return wave, potential.astype(np.float32), parameters


def test_default_budget_is_conservative_and_unavailable_memory_disables_cache():
    assert _transmission_cache_budget(_NumpyDevice(), None) == MAX_TRANSMISSION_CACHE_BYTES
    small = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(memGetInfo=lambda: (8000, 9000))))
    assert _transmission_cache_budget(small, None) == 1000
    assert _transmission_cache_budget(object(), None) == 0
    assert _transmission_cache_budget(object(), 123) == 123
    with pytest.raises(ValueError, match="negative"):
        _transmission_cache_budget(object(), -1)


@pytest.mark.parametrize("explicit", [False, True])
def test_snapshot_removes_repeated_exponents_and_matches_cpu_reference(device, explicit, monkeypatch):
    wave, potential, parameters = _fixture(explicit)
    plan = CuPyMultislicePlan.build(potential, **parameters, cupy=device)
    source = device.asarray(potential)
    original_exp = device.exp
    exp_calls = []

    def count_exp(value):
        exp_calls.append(value.shape)
        return original_exp(value)

    monkeypatch.setattr(device, "exp", count_exp)
    snapshot = plan.prepare_transmission(source)
    grating_count = 3 if explicit else 1
    assert len(exp_calls) == grating_count
    assert plan.transmission_cache_bytes == grating_count * wave.size * 8
    assert plan.transmission_cache_bytes <= plan.transmission_cache_limit_bytes
    for batch_size in (2, 2, 1):
        incoming = device.asarray(np.repeat(wave[None], batch_size, axis=0))
        cached, cached_diagnostics = plan.propagate(incoming, snapshot)
        assert len(exp_calls) == grating_count
    raw, raw_diagnostics = plan.propagate(incoming, source)
    np.testing.assert_array_equal(_host(device, cached), _host(device, raw))
    assert cached_diagnostics == raw_diagnostics
    reference, _ = propagate_multislice(wave, potential, **parameters)
    error = np.linalg.norm(_host(device, cached)[0] - reference) / np.linalg.norm(reference)
    assert error < 3e-5
    assert plan.transmission_cache_build_count == 1
    assert plan.transmission_cache_hit_count == 2
    assert plan.transmission_cache_build_elapsed_s >= 0


def test_mutable_raw_potential_and_phonon_switch_cannot_change_snapshot(device):
    wave, potential, parameters = _fixture()
    plan = CuPyMultislicePlan.build(potential, **parameters, cupy=device)
    source = device.asarray(potential).copy()
    first = plan.prepare_transmission(source)
    old, _ = plan.propagate(device.asarray(wave), first)
    source *= 4
    second = plan.prepare_transmission(source)
    changed, _ = plan.propagate(device.asarray(wave), second)
    unchanged, _ = plan.propagate(device.asarray(wave), first)
    raw, _ = plan.propagate(device.asarray(wave), source)
    np.testing.assert_array_equal(_host(device, old), _host(device, unchanged))
    np.testing.assert_array_equal(_host(device, changed), _host(device, raw))
    assert np.linalg.norm(_host(device, changed) - _host(device, old)) > 1e-3
    assert plan.transmission_cache_build_count == 2
    assert plan.transmission_cache_hit_count == 1


def test_snapshot_cannot_cross_plan_or_changed_energy(device):
    wave, potential, parameters = _fixture()
    plan = CuPyMultislicePlan.build(potential, **parameters, cupy=device)
    other = CuPyMultislicePlan.build(potential, **parameters, cupy=device)
    snapshot = plan.prepare_transmission(potential)
    with pytest.raises(ValueError, match="different CUDA plan"):
        other.propagate(device.asarray(wave), snapshot)
    plan.interaction_constant_rad_per_v_angstrom *= 1.01
    with pytest.raises(ValueError, match="parameters changed"):
        plan.propagate(device.asarray(wave), snapshot)


@pytest.mark.parametrize("capacity", [0, 1, 3 * 14 * 18 * 8])
def test_budget_rejection_keeps_other_configuration_and_raw_fallback(device, capacity):
    wave, potential, parameters = _fixture()
    plan = CuPyMultislicePlan.build(
        potential, **parameters, cupy=device, transmission_cache_limit_bytes=capacity,
    )
    first = plan.prepare_transmission(potential)
    second = plan.prepare_transmission(potential * 2)
    assert second is None
    assert plan.transmission_cache_bytes <= capacity
    assert (first is not None) == (capacity == 3 * wave.size * 8)
    result, _ = plan.propagate(device.asarray(wave), potential * 2)
    reference, _ = propagate_multislice(wave, potential * 2, **parameters)
    assert np.linalg.norm(_host(device, result) - reference) / np.linalg.norm(reference) < 3e-5
    if first is not None:
        assert plan.propagate(device.asarray(wave), first)[0].shape == wave.shape


def test_cache_allocation_failure_is_optional_and_partial_entry_is_not_published(monkeypatch):
    device = _NumpyDevice()
    wave, potential, parameters = _fixture()
    plan = CuPyMultislicePlan.build(potential, **parameters, cupy=device)
    original_exp = device.exp
    calls = 0

    def allocation_failure(value):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise MemoryError("synthetic optional cache allocation failure")
        return original_exp(value)

    monkeypatch.setattr(device, "exp", allocation_failure)
    assert plan.prepare_transmission(potential) is None
    assert plan.transmission_cache_bytes == plan.transmission_cache_build_count == 0
    assert not plan._transmission_entries
    monkeypatch.setattr(device, "exp", original_exp)
    assert plan.propagate(wave, potential)[0].shape == wave.shape


def test_zero_thickness_does_not_allocate_transmission_and_invalid_potential_is_rejected():
    device = _NumpyDevice()
    wave, potential, parameters = _fixture(False)
    parameters["total_thickness_angstrom"] = 0
    plan = CuPyMultislicePlan.build(potential, **parameters, cupy=device)
    assert plan.prepare_transmission(potential) is None
    np.testing.assert_allclose(plan.propagate(wave, potential)[0], wave, rtol=2e-7, atol=1e-8)
    potential[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinity"):
        plan.prepare_transmission(potential)


def test_resident_cache_matches_disabled_partial_budget_and_phonon_sem():
    if not compute_backend.cupy_capability().available:
        pytest.skip("CuPy CUDA backend unavailable")
    wave, potential, parameters = _fixture()
    fy = np.fft.fftshift(np.fft.fftfreq(14, d=0.3))
    fx = np.fft.fftshift(np.fft.fftfreq(18, d=0.2))
    qx, qy = np.meshgrid(fx, fy, indexing="xy")
    inputs = dict(
        base_spectrum=np.exp(-(qx**2 + qy**2) / 0.8**2),
        frequencies_x=fx, frequencies_y=fy,
        scan_x_angstrom=np.linspace(-0.4, 0.4, 5),
        scan_y_angstrom=np.linspace(0.3, -0.3, 5),
        potential_configurations_v_angstrom=(potential, potential * 1.4),
        detector_masks={"bf": qx**2 + qy**2 < 0.3**2, "df": qx**2 + qy**2 >= 0.3**2},
        multislice_enabled=True, total_thickness_angstrom=3.0,
        target_slice_thickness_angstrom=1.0, batch_size=2, **parameters,
    )
    bytes_per_configuration = potential.size * 8
    results = []
    for limit in (0, bytes_per_configuration, 2 * bytes_per_configuration):
        results.append(run_resident_stem_cuda(**inputs, transmission_cache_limit_bytes=limit))
    for result, builds in zip(results, (0, 1, 2)):
        for key in result.fractions_flat:
            np.testing.assert_array_equal(result.fractions_flat[key], results[0].fractions_flat[key])
        np.testing.assert_array_equal(result.uncollected_flat, results[0].uncollected_flat)
        assert result.detector_relative_standard_error == results[0].detector_relative_standard_error
        assert result.multislice_diagnostics == results[0].multislice_diagnostics
        assert result.metrics["cuda_transmission_cache_builds"] == builds
        assert result.metrics["cuda_transmission_cache_hits"] == builds * 2
        assert result.metrics["cuda_transmission_cache_bypass_count"] == 2 - builds
        assert result.metrics["cuda_transmission_cache_bytes"] == builds * bytes_per_configuration
        assert result.metrics["cuda_bulk_host_transfer_count"] == 1
