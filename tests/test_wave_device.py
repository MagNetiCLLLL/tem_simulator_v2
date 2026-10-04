"""Device policy tests plus small, optional real CUDA column comparisons."""
from dataclasses import replace
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics import wave_device as device
from temsim.physics.compute_backend import GPUExecutionError, WAVE_BACKEND_CUPY


def test_cpu_scope_does_not_probe_optional_cuda(monkeypatch):
    monkeypatch.setattr(device, "cupy_module", lambda: pytest.fail("CPU must not probe CUDA"))
    with device.device_scope("cpu", work_items=100_000_000) as session:
        assert session.xp is np
        assert session.evidence() == {"compute_backend": "numpy",
            "numeric_precision": "complex128 / float64", "fallback_reason": None}
        device.check_device_memory(10**20)


def test_disabled_gpu_preference_and_requirement_remain_distinct():
    with device.device_scope("Prefer GPU", acceleration_enabled=False) as session:
        assert session.backend == "numpy" and "disabled" in session.fallback_reason
    with pytest.raises(GPUExecutionError, match="disabled"):
        with device.device_scope("Require GPU", acceleration_enabled=False):
            pytest.fail("disabled acceleration cannot satisfy Require GPU")


@pytest.mark.parametrize("policy,fallback", [("Auto", True), ("CUDA GPU", True), ("Require GPU", False)])
def test_gpu_admission_memory_policy(monkeypatch, policy, fallback):
    class Pool:
        def set_limit(self, **kwargs): pass
        def total_bytes(self): return 0
        def used_bytes(self): return 0
        def free_all_blocks(self): pass
    fake = SimpleNamespace(cuda=SimpleNamespace(MemoryPool=Pool,
        runtime=SimpleNamespace(memGetInfo=lambda: (1<<30, 2<<30))))
    monkeypatch.setattr(device, "choose_wave_backend", lambda *a, **k: (WAVE_BACKEND_CUPY, None))
    monkeypatch.setattr(device, "cupy_module", lambda: fake)
    if fallback:
        with device.device_scope(policy, maximum_working_bytes=2<<30, required_bytes=1<<30) as session:
            assert session.backend == "numpy"
            assert "out_of_memory" in session.fallback_reason
    else:
        with pytest.raises(GPUExecutionError, match="out_of_memory"):
            with device.device_scope(policy, required_bytes=1<<30):
                pytest.fail("GPU memory admission must fail")
    assert device._ACTIVE.get() is None


def test_invalid_gpu_execution_is_never_silently_retried(monkeypatch):
    from temsim.physics import column_wave as column
    from temsim.optics.column import default_state
    from test_tip_wave_pipeline import _quiet_prepared_column
    from test_wave_detector_readout import checkpoint
    monkeypatch.setattr(column, "_prepare_column", _quiet_prepared_column)
    monkeypatch.setattr(column, "_propagate_column_impl", lambda *a, **k:
        (_ for _ in ()).throw(GPUExecutionError("invalid_configuration", "bad operator")))
    with pytest.raises(GPUExecutionError, match="bad operator"):
        column._propagate_column(default_state(), checkpoint(), 2000.1)


def _cuda():
    cp = pytest.importorskip("cupy", reason="optional real CUDA verification requires CuPy")
    from temsim.physics.compute_backend import cupy_capability
    status = cupy_capability()
    if not status.available:
        pytest.skip(status.detail)
    return cp


@pytest.mark.parametrize("policy", ["Auto", "Require GPU"])
def test_real_gpu_column_matches_cpu_complex_field_and_host_checkpoint(policy):
    started = perf_counter()
    _cuda()
    from temsim.physics import column_wave as column
    from temsim.optics.column import default_state
    from temsim.physics.wave_grid import WaveGridNumerics
    from test_wave_detector_readout import checkpoint
    state = default_state()
    source = checkpoint(two=True)
    prepared = column._prepare_column(state, 2000., 2000.02, .01)
    cpu = column._propagate_column(state, source, 2000.02, _prepared=prepared)
    gpu = column._propagate_column(state, source, 2000.02, _prepared=prepared,
        grid_numerics=WaveGridNumerics(compute_backend=policy))
    assert all(row["compute_backend"] == "cupy" for row in gpu.record["modes"])
    assert gpu.digest
    for expected, actual in zip(cpu.beam.modes, gpu.beam.modes):
        assert isinstance(actual.plane.amplitude, np.ndarray)
        assert not actual.plane.amplitude.flags.writeable
        assert actual.weight_per_reference_electron == pytest.approx(expected.weight_per_reference_electron, abs=1e-12)
        np.testing.assert_allclose(actual.plane.amplitude, expected.plane.amplitude, atol=2e-12, rtol=2e-10)
        np.testing.assert_allclose(actual.plane.basis_m, expected.plane.basis_m, atol=1e-18, rtol=2e-10)
        np.testing.assert_allclose(actual.plane.curvature_m1, expected.plane.curvature_m1, atol=1e-10, rtol=2e-10)
        assert actual.axial_reference == expected.axial_reference
    if policy == "Require GPU":
        # Extend the actually executed 20-um column segment through the same
        # conservative preview used by the GUI, then virtual-screen arrivals.
        # This qualifies neither a complete tip-to-screen chain nor detector QE.
        from temsim.gui.coherent_beam import _intensity_preview
        from temsim.physics.electron_detection import sample_electron_detections
        previews = [_intensity_preview(result, bins=32) for result in (cpu, gpu)]
        np.testing.assert_allclose(previews[1].density, previews[0].density,
                                   atol=2e-13, rtol=2e-9)
        np.testing.assert_allclose(previews[1].bounds_um, previews[0].bounds_um,
                                   atol=1e-10, rtol=2e-10)
        arrivals = [sample_electron_detections(p.density, p.bounds_um,
                    emitted_electrons=3000, seed=0) for p in previews]
        for preview, hits in zip(previews, arrivals):
            area = float(np.prod(np.diff(preview.bounds_um, axis=1).ravel()/32))
            probability = preview.density*area
            assert preview.probability == pytest.approx(.8, abs=1e-12)
            assert hits.detection_probability == pytest.approx(.8, abs=1e-12)
            assert hits.emitted_count == hits.detected_count+hits.lost_count == 3000
            assert 0 < hits.lost_count < 3000
            assert abs(hits.detected_count-2400) < 6.*np.sqrt(3000*.8*.2)
            # A cell's count is binomial even though all counts jointly form
            # a multinomial. Include one count for the discrete low-p limit.
            assert np.all(np.abs(hits.counts-3000*probability)
                          <= 6.*np.sqrt(3000*probability*(1.-probability))+1.)
        # Seeds repeat each implementation's stream, but tiny probability
        # differences can move a draw across a categorical boundary. Compare
        # CPU/GPU arrival statistics, not exact arrays of random categories.
        expected = previews[0].density*float(np.prod(
            np.diff(previews[0].bounds_um, axis=1).ravel()/32))
        assert np.all(np.abs(arrivals[1].counts-arrivals[0].counts)
                      <= 6.*np.sqrt(2*3000*expected*(1.-expected))+1.)
        mean_difference = np.abs(arrivals[1].points_um.mean(axis=0)
                                 -arrivals[0].points_um.mean(axis=0))
        mean_standard_error = np.sqrt(sum(h.points_um.var(axis=0)/h.detected_count
                                           for h in arrivals))
        assert np.all(mean_difference <= 6.*mean_standard_error)
        print(f"Require GPU virtual-screen smoke: backend=cupy; "
              f"probability={arrivals[1].detection_probability:.12g}; emitted=3000; "
              f"detected={arrivals[1].detected_count}; lost={arrivals[1].lost_count}; "
              f"elapsed={perf_counter()-started:.3f}s")


def test_real_gpu_physical_annulus_and_column_aperture_masks():
    cp = _cuda()
    from temsim.physics.record_plane import PlaneStop
    from temsim.physics.column_wave import _clip
    from test_column_wave_transport import _off_axis_affine_clip_wave
    wave = _off_axis_affine_clip_wave()
    plane = PlaneStop("detector", "Ring", 2000., "detector", "annulus", outer_width_mm=.01,
                      inner_diameter_mm=.004, offset_x_mm=.001)
    xy = wave.coordinates_m()
    np.testing.assert_array_equal(cp.asnumpy(plane.hit_mask(cp.asarray(xy[0]), cp.asarray(xy[1]))),
                                  plane.hit_mask(*xy))
    aperture = SimpleNamespace(key="opening", radius_mm=.004, offset_x_mm=.001, offset_y_mm=-.001)
    expected, losses = _clip(wave, .007, (aperture,), 2000., .7)
    actual, device_losses = _clip(replace(wave, amplitude=cp.asarray(wave.amplitude)), .007, (aperture,), 2000., .7)
    np.testing.assert_allclose(cp.asnumpy(actual.amplitude), expected.amplitude, atol=1e-15)
    assert [r["lost_weight"] for r in device_losses] == pytest.approx([r["lost_weight"] for r in losses], abs=1e-14)
