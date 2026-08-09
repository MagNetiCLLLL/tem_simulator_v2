from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.physics import compute_backend, stem_wave_imaging
from temsim.physics.compute_backend import WAVE_BACKEND_CUPY
from temsim.physics.stem_wave_imaging import (
    AngularDetector,
    simulate_angle_resolved_stem,
)
from temsim.specimen.atomistic import atomistic_capability


def _incident_bundle():
    return SimpleNamespace(
        alive=np.ones(5, dtype=bool),
        ray_weight=np.asarray([0.6, 0.1, 0.1, 0.1, 0.1]),
        x=np.zeros((1, 5), dtype=float),
        y=np.zeros((1, 5), dtype=float),
        tx=np.asarray([[0.0, 2.0e-3, -2.0e-3, 0.0, 0.0]]),
        ty=np.asarray([[0.0, 0.0, 0.0, 2.0e-3, -2.0e-3]]),
    )


def _state(backend: str, *, atomistic: bool = False):
    state = default_state()
    state.illumination_mode = "STEM"
    state.acceleration_enabled = backend != "CPU"
    state.acceleration_backend = backend
    state.sample.specimen_preset_key = "si_110" if atomistic else "vacuum"
    state.sample.thickness_nm = 0.4
    state.sample.wave_grid_pixels = 32
    state.sample.wave_field_of_view_angstrom = 8.0 if atomistic else 16.0
    state.sample.wave_multislice_enabled = True
    state.sample.wave_slice_thickness_angstrom = 2.0
    state.sample.wave_atomistic_enabled = atomistic
    state.sample.wave_frozen_phonon_enabled = atomistic
    state.sample.wave_frozen_phonon_configurations = 2
    state.sample.wave_frozen_phonon_seed = 707
    return state


def _scan():
    return (
        np.asarray([[-5.0e-5, 5.0e-5], [-5.0e-5, 5.0e-5]]),
        np.asarray([[-5.0e-5, -5.0e-5], [5.0e-5, 5.0e-5]]),
    )


def _detectors():
    return (
        AngularDetector("bf", 0.0, 10.0),
        AngularDetector("df", 10.0, 24.0),
    )


def test_explicit_cuda_keeps_stem_arrays_resident_until_one_bulk_transfer(
    monkeypatch,
):
    if not compute_backend.cupy_capability().available:
        pytest.skip("CuPy CUDA backend unavailable")
    cp = compute_backend.cupy_module()
    original_asnumpy = cp.asnumpy
    transfer_count = 0

    def counted_asnumpy(*args, **kwargs):
        nonlocal transfer_count
        transfer_count += 1
        return original_asnumpy(*args, **kwargs)

    monkeypatch.setattr(cp, "asnumpy", counted_asnumpy)
    scan_x, scan_y = _scan()
    result = simulate_angle_resolved_stem(
        _state("CUDA GPU"),
        SimpleNamespace(incident=_incident_bundle()),
        _detectors(),
        scan_x,
        scan_y,
    )

    assert transfer_count == 1
    assert result.metrics["cuda_resident_pipeline"] is True
    assert result.metrics["cuda_bulk_host_transfer_count"] == 1
    assert result.metrics["cuda_bulk_host_transfer_bytes"] == (
        (len(_detectors()) + 1) * scan_x.size * np.dtype(np.float64).itemsize
    )
    assert result.metrics["cuda_potential_upload_count"] == 1
    assert result.metrics["cuda_probe_batch_count"] == 1
    assert result.metrics["cuda_multislice_plan_reused"] is True
    assert result.metrics["cuda_multislice_plan_build_count"] == 1
    assert result.metrics["cuda_multislice_plan_use_count"] == 1
    assert result.metrics["cuda_potential_phase_scan_count"] == 1
    assert result.metrics["wave_compute_backend"] == "CuPy CUDA"
    assert result.metrics["cuda_pipeline_fallback_reason"] is None


def test_resident_cuda_frozen_phonon_detector_signals_match_cpu_reference():
    if not compute_backend.cupy_capability().available:
        pytest.skip("CuPy CUDA backend unavailable")
    if not atomistic_capability().available:
        pytest.skip("Atomistic backend unavailable")
    scan_x, scan_y = _scan()
    simulation = SimpleNamespace(incident=_incident_bundle())
    cpu = simulate_angle_resolved_stem(
        _state("CPU", atomistic=True),
        simulation,
        _detectors(),
        scan_x,
        scan_y,
    )
    gpu = simulate_angle_resolved_stem(
        _state("CUDA GPU", atomistic=True),
        simulation,
        _detectors(),
        scan_x,
        scan_y,
    )

    for key in cpu.fractions:
        assert gpu.fractions[key] == pytest.approx(
            cpu.fractions[key], rel=2.0e-4, abs=2.0e-7
        )
    assert gpu.uncollected_fraction == pytest.approx(
        cpu.uncollected_fraction, rel=2.0e-4, abs=2.0e-7
    )
    assert gpu.metrics["cuda_resident_pipeline"] is True
    assert gpu.metrics["cuda_configuration_count"] == 2
    assert gpu.metrics["cuda_potential_upload_count"] == 2
    assert gpu.metrics["cuda_resident_potential_bytes"] > 0
    assert gpu.metrics["cuda_multislice_plan_build_count"] == 1
    assert gpu.metrics["cuda_multislice_plan_use_count"] == 2
    assert gpu.metrics["cuda_potential_phase_scan_count"] == 2
    assert gpu.metrics["specimen_model"] == (
        "atomistic_frozen_phonon_multislice"
    )
    for key, cpu_error in cpu.metrics[
        "detector_configuration_relative_standard_error"
    ].items():
        assert gpu.metrics[
            "detector_configuration_relative_standard_error"
        ][key] == pytest.approx(cpu_error, rel=2.0e-3, abs=2.0e-7)


def test_resident_cuda_projected_phase_object_matches_cpu_reference():
    if not compute_backend.cupy_capability().available:
        pytest.skip("CuPy CUDA backend unavailable")
    cpu_state = _state("CPU")
    gpu_state = _state("CUDA GPU")
    cpu_state.sample.wave_multislice_enabled = False
    gpu_state.sample.wave_multislice_enabled = False
    scan_x, scan_y = _scan()
    simulation = SimpleNamespace(incident=_incident_bundle())

    cpu = simulate_angle_resolved_stem(
        cpu_state,
        simulation,
        _detectors(),
        scan_x,
        scan_y,
    )
    gpu = simulate_angle_resolved_stem(
        gpu_state,
        simulation,
        _detectors(),
        scan_x,
        scan_y,
    )

    for key in cpu.fractions:
        assert gpu.fractions[key] == pytest.approx(
            cpu.fractions[key], rel=2.0e-4, abs=2.0e-7
        )
    assert gpu.metrics["cuda_resident_pipeline"] is True
    assert gpu.metrics["specimen_model"] == "projected_phase_object"
    assert gpu.metrics["specimen_compute_backend"] == "CuPy CUDA"
    assert gpu.metrics["wave_compute_backend"] == "CuPy CUDA"
    assert gpu.metrics["cuda_multislice_diagnostic_count"] == 0
    assert gpu.metrics["cuda_multislice_plan_reused"] is False
    assert gpu.metrics["cuda_multislice_plan_build_count"] == 0
    assert gpu.metrics["cuda_multislice_plan_use_count"] == 0
    assert gpu.metrics["cuda_potential_phase_scan_count"] == 0


def test_resident_cuda_failure_discards_partial_work_and_recomputes_on_cpu(
    monkeypatch,
):
    monkeypatch.setattr(
        stem_wave_imaging,
        "choose_wave_backend",
        lambda *_args, **_kwargs: (WAVE_BACKEND_CUPY, None),
    )
    monkeypatch.setattr(
        stem_wave_imaging,
        "run_resident_stem_cuda",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic resident pipeline failure")
        ),
    )
    monkeypatch.setattr(
        stem_wave_imaging,
        "release_cupy_memory_pools",
        lambda: None,
    )
    scan_x, scan_y = _scan()
    result = simulate_angle_resolved_stem(
        _state("CUDA GPU"),
        SimpleNamespace(incident=_incident_bundle()),
        _detectors(),
        scan_x,
        scan_y,
    )

    assert result.metrics["cuda_resident_pipeline"] is False
    assert "synthetic resident pipeline failure" in result.metrics[
        "cuda_pipeline_fallback_reason"
    ]
    assert result.metrics["specimen_compute_backend"] == "NumPy CPU"
    assert result.metrics["fft_compute_backend"] == "NumPy CPU"
    assert "synthetic resident pipeline failure" in result.metrics[
        "specimen_fallback_reason"
    ]
    assert all(
        np.all(np.isfinite(values)) for values in result.fractions.values()
    )
