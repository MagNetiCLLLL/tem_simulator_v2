"""WP-07 real CUDA capture and failure/transaction boundaries."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics import compute_backend, stem_wave_imaging
from temsim.physics.fourdstem import FourDSTEMCaptureSink, FourDSTEMCancelled, FourDSTEMWriter
from test_fourdstem import _calibration
from test_stem_cuda_pipeline import _state, _scan, _detectors, _incident_bundle


def test_actual_gpu_capture_matches_cpu_cube_and_detector_reintegration(tmp_path, monkeypatch, record_property):
    if not compute_backend.cupy_capability().available:
        pytest.skip("NOT_RUN: actual CUDA device unavailable")
    scan_x, scan_y = _scan()
    sim = SimpleNamespace(incident=_incident_bundle())
    cpu_sink = FourDSTEMCaptureSink(tmp_path / "cpu.npy", electrons_per_frame=100, store_raw_probability=True)
    cpu = stem_wave_imaging.simulate_angle_resolved_stem(_state("CPU", atomistic=True), sim, _detectors(), scan_x, scan_y, diffraction_sink=cpu_sink)
    def forbidden(*args, **kwargs):
        raise AssertionError("GPU capture must not repropagate or FFT on CPU")
    monkeypatch.setattr(stem_wave_imaging, "propagate_multislice", forbidden)
    monkeypatch.setattr(stem_wave_imaging, "stem_diffraction_intensity", forbidden)
    monkeypatch.setattr(stem_wave_imaging, "resident_stem_batch_size", lambda *a, **k: 2)
    gpu_state = _state("CUDA GPU", atomistic=True)
    gpu_state.sample.stem_execution_policy = "require_gpu"
    gpu_sink = FourDSTEMCaptureSink(tmp_path / "gpu.npy", electrons_per_frame=999, store_raw_probability=True)
    gpu = stem_wave_imaging.simulate_angle_resolved_stem(gpu_state, sim, _detectors(), scan_x, scan_y, diffraction_sink=gpu_sink)
    assert gpu.metrics["cuda_resident_pipeline"]
    assert gpu.metrics["cuda_diffraction_batch_transfer_count"] == 2
    assert gpu.metrics["cuda_diffraction_host_bound_bytes"] <= gpu.metrics["cuda_diffraction_host_budget_bytes"]
    np.testing.assert_allclose(gpu_sink.artifact.data, cpu_sink.artifact.data, atol=2e-7, rtol=2e-4)
    axes = gpu_sink.artifact.calibration
    radius = np.hypot(axes.angle_x_mrad, axes.angle_y_mrad)
    for detector in _detectors():
        mask = (radius >= detector.inner_mrad) & (radius < detector.outer_mrad)
        reduced = np.sum(gpu_sink.artifact.data * mask, axis=(-2, -1), dtype=np.float64)
        np.testing.assert_allclose(reduced, gpu.fractions[detector.key], atol=2e-7, rtol=2e-4)
        np.testing.assert_allclose(cpu.fractions[detector.key], gpu.fractions[detector.key], atol=2e-7, rtol=2e-4)
    provenance = gpu_sink.artifact.metadata["provenance"]
    assert provenance["actual_wave_backend"] == "CuPy CUDA"
    assert len(provenance["solver_source_sha256"]) == 64
    assert len(provenance["potential_sha256"]) == 2
    record_property("wp07_actual_gpu",json.dumps({"device":compute_backend.cupy_capability().detail,
        "maximum_probability_difference":float(np.max(np.abs(gpu_sink.artifact.data-cpu_sink.artifact.data))),
        "relative_tolerance":2e-4,"absolute_probability_tolerance":2e-7,
        "frame_count":int(scan_x.size),"configuration_count":2,
        "transfer_batches":gpu.metrics["cuda_diffraction_batch_transfer_count"],
        "host_bound_bytes":gpu.metrics["cuda_diffraction_host_bound_bytes"],
        "host_budget_bytes":gpu.metrics["cuda_diffraction_host_budget_bytes"],"cpu_repropagation":False,
        "scope":"Si thin frozen-phonon fixture; same potential, grid, source and dose-independent probabilities"}))


@pytest.mark.parametrize("policy", ["auto", "prefer_gpu", "require_gpu"])
def test_policy_only_resource_failures_may_fallback(policy, monkeypatch):
    monkeypatch.setattr(compute_backend, "cupy_capability", lambda: compute_backend.BackendCapability(False, "test absent"))
    if policy == "require_gpu":
        with pytest.raises(compute_backend.GPUExecutionError, match="unavailable"):
            compute_backend.choose_wave_backend(policy, acceleration_enabled=True, work_items=2_000_000)
        with pytest.raises(compute_backend.GPUExecutionError, match="out_of_memory"):
            compute_backend.gpu_retry_reason(compute_backend.GPUExecutionError("out_of_memory", "test"), policy)
    else:
        assert compute_backend.choose_wave_backend(policy, acceleration_enabled=True, work_items=2_000_000)[0] == compute_backend.WAVE_BACKEND_NUMPY
        assert "out_of_memory" in compute_backend.gpu_retry_reason(compute_backend.GPUExecutionError("out_of_memory", "test"), policy)
    for error in (ValueError("bad input"), FloatingPointError("norm drift"), OSError("disk failed"), AssertionError("bug"), FourDSTEMCancelled("cancel")):
        with pytest.raises(type(error), match=str(error)):
            compute_backend.gpu_retry_reason(error, policy)


def test_resume_detects_corruption_and_source_identity_changes(tmp_path):
    cal = _calibration((2, 2), (3, 3))
    path = tmp_path / "partial.npy"
    writer = FourDSTEMWriter(path, cal, provenance={"code": "a", "potential": "p", "modes": [1]})
    writer.write_frame(0, 0, np.ones((3, 3)))
    writer.close()
    with pytest.raises(ValueError, match="different source"):
        FourDSTEMWriter(path, cal, resume=True, provenance={"code": "b", "potential": "p", "modes": [1]})
    data = np.load(writer.partial_path, mmap_mode="r+")
    data[0, 0, 0, 0] = 2
    data.flush()
    data._mmap.close()
    with pytest.raises(ValueError, match="checksum"):
        FourDSTEMWriter(path, cal, resume=True, provenance={"code": "a", "potential": "p", "modes": [1]})


def test_failed_checkpoint_does_not_publish_uncommitted_frames(tmp_path, monkeypatch):
    import temsim.physics.fourdstem as module
    cal = _calibration((2, 2), (3, 3))
    path = tmp_path / "failure.npy"
    writer = FourDSTEMWriter(path, cal)
    writer.write_frame(0, 0, np.ones((3, 3)))
    writer.checkpoint()
    writer.write_frame(0, 1, np.ones((3, 3))*2)
    with monkeypatch.context() as context:
        def disk_failure(*args):
            raise OSError("disk full")
        context.setattr(module, "_atomic_json", disk_failure)
        with pytest.raises(OSError, match="disk full"):
            writer.checkpoint()
        writer._close_memmap()  # simulate process loss: no finalizer checkpoint
    resumed = FourDSTEMWriter(path, cal, resume=True)
    assert resumed.completed_frames == 1
    assert not resumed.frame_completed(0, 1)
    resumed.close()


def test_cancel_resume_writes_each_position_once(tmp_path):
    cal = _calibration((2, 2), (3, 3))
    path = tmp_path / "cancel.npy"
    cancelled = False
    sink = FourDSTEMCaptureSink(path, electrons_per_frame=10, store_raw_probability=True,
                               cancellation_requested=lambda: cancelled)
    sink.begin(cal, np.ones((3, 3), bool), maximum_isotropic_angle_mrad=8)
    frame = np.ones((3, 3))/9
    sink.write_frame(0, 0, frame)
    cancelled = True
    with pytest.raises(FourDSTEMCancelled):
        sink.write_frame(0, 1, frame)
    resumed = FourDSTEMCaptureSink(path, electrons_per_frame=10, store_raw_probability=True, resume=True)
    resumed.begin(cal, np.ones((3, 3), bool), maximum_isotropic_angle_mrad=8)
    assert resumed.writer.completed_frames == 1
    for y in range(2):
        for x in range(2):
            resumed.write_frame(y, x, frame)
    artifact = resumed.finish()
    assert artifact.metadata["completed_frames"] == 4
    np.testing.assert_allclose(artifact.data, 1/9)


def test_capture_host_budget_rejects_oversized_frame_before_gpu_work(tmp_path):
    state = _state("CPU")
    state.sample.stem_fourdstem_host_budget_mb = 0
    sink = FourDSTEMCaptureSink(tmp_path / "budget.npy", electrons_per_frame=1)
    with pytest.raises(MemoryError, match="budget"):
        stem_wave_imaging.simulate_angle_resolved_stem(state, SimpleNamespace(incident=_incident_bundle()), _detectors(), *_scan(), diffraction_sink=sink)
    sink.close_partial()


def test_stem_known_bandwidth_loss_keeps_absolute_probability(tmp_path):
    from temsim.physics.multislice import propagate_multislice
    from temsim.physics.wave_fft import stem_diffraction_intensity
    from temsim.physics.fourdstem import FourDSTEMCalibration
    n=32
    x=np.arange(n)
    wave=np.broadcast_to((np.sqrt(.2)+np.sqrt(.8)*np.exp(2j*np.pi*14*x/n))/n,(n,n)).copy()
    exit_wave,_=propagate_multislice(wave,np.zeros((n,n)),pixel_size_angstrom=.25,
        wavelength_angstrom=.0197,interaction_constant_rad_per_v_angstrom=.0065,total_thickness_angstrom=2.,
        target_slice_thickness_angstrom=1.,bandwidth_fraction=2/3,compute_backend="NumPy CPU")
    probability,_=stem_diffraction_intensity(exit_wave,reference_norm=1.)
    assert probability.sum()==pytest.approx(.2,abs=1e-12)
    sink=FourDSTEMCaptureSink(tmp_path/"loss.npy",electrons_per_frame=100,store_raw_probability=True)
    cal=FourDSTEMCalibration.from_rectilinear_axes([0.],[0.],np.arange(n),np.arange(n))
    sink.begin(cal,np.ones((n,n),bool),maximum_isotropic_angle_mrad=20)
    sink.write_frame(0,0,probability)
    assert sink.finish().data.sum()==pytest.approx(.2,abs=2e-8)
