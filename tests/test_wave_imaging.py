from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.physics import compute_backend
from temsim.physics.wave_imaging import (
    _weighted_ray_statistics,
    simulate_wave_image,
)
from temsim.physics.stem_wave_imaging import (
    AngularDetector,
    simulate_angle_resolved_stem,
)


def _incident_bundle(tx_rad, ty_rad, weights):
    count = len(weights)
    return SimpleNamespace(
        alive=np.ones(count, dtype=bool),
        ray_weight=np.asarray(weights, dtype=float),
        x=np.zeros((1, count), dtype=float),
        y=np.zeros((1, count), dtype=float),
        tx=np.asarray(tx_rad, dtype=float)[None, :],
        ty=np.asarray(ty_rad, dtype=float)[None, :],
    )


def test_weighted_convergence_uses_chief_ray_and_99_percent_semiangle():
    slope = 0.01
    incident = _incident_bundle(
        [0.0, slope, -slope, 0.0, 0.0],
        [0.0, 0.0, 0.0, slope, -slope],
        [0.98, 0.005, 0.005, 0.005, 0.005],
    )
    statistics = _weighted_ray_statistics(incident)
    edge_angle = np.arctan(slope)

    assert statistics["mean_tx_rad"] == pytest.approx(0.0, abs=1.0e-15)
    assert statistics["mean_ty_rad"] == pytest.approx(0.0, abs=1.0e-15)
    assert statistics["convergence_99_rad"] == pytest.approx(edge_angle)
    assert statistics["convergence_semiangle_rad"] == pytest.approx(edge_angle)
    assert statistics["convergence_edge_rad"] == pytest.approx(edge_angle)
    assert statistics["convergence_rms_rad"] == pytest.approx(
        np.sqrt(0.02) * edge_angle
    )


def test_tem_wave_image_reports_multislice_model_and_sampling_metrics():
    state = default_state()
    state.illumination_mode = "TEM"
    state.sample.specimen_preset_key = "vacuum"
    state.sample.thickness_nm = 2.0
    state.sample.wave_grid_pixels = 32
    state.sample.wave_field_of_view_angstrom = 16.0
    state.sample.wave_multislice_enabled = True
    state.sample.wave_slice_thickness_angstrom = 2.0
    incident = _incident_bundle(
        [0.0, 1.0e-4, -1.0e-4],
        [0.0, 0.0, 0.0],
        [0.8, 0.1, 0.1],
    )

    result = simulate_wave_image(state, SimpleNamespace(incident=incident))

    assert result.image_intensity.shape == (32, 32)
    assert result.exit_wave.shape == (32, 32)
    assert result.metrics["specimen_model"] == "continuous_column_multislice"
    assert result.metrics["specimen_slice_count"] == 10
    assert result.metrics["specimen_slice_thickness_angstrom"] == pytest.approx(2.0)
    assert result.metrics["convergence_semiangle_rad"] > 0.0
    assert result.metrics["specimen_maximum_relative_intensity_change"] < 1.0e-10
    assert result.metrics["specimen_compute_backend"] == "NumPy CPU"
    assert result.metrics["fft_compute_backend"] == "NumPy CPU"


def test_projected_phase_object_remains_available_as_preview_model():
    state = default_state()
    state.illumination_mode = "TEM"
    state.sample.specimen_preset_key = "vacuum"
    state.sample.thickness_nm = 2.0
    state.sample.wave_grid_pixels = 32
    state.sample.wave_field_of_view_angstrom = 16.0
    state.sample.wave_multislice_enabled = False
    incident = _incident_bundle(
        [0.0, 1.0e-4, -1.0e-4],
        [0.0, 0.0, 0.0],
        [0.8, 0.1, 0.1],
    )

    result = simulate_wave_image(state, SimpleNamespace(incident=incident))

    assert result.metrics["specimen_model"] == "projected_phase_object"
    assert result.metrics["specimen_slice_count"] == 1


def test_angle_resolved_stem_uses_the_same_multislice_specimen_model():
    state = default_state()
    state.illumination_mode = "STEM"
    state.sample.specimen_preset_key = "vacuum"
    state.sample.thickness_nm = 0.4
    state.sample.wave_grid_pixels = 32
    state.sample.wave_field_of_view_angstrom = 16.0
    state.sample.wave_multislice_enabled = True
    state.sample.wave_slice_thickness_angstrom = 2.0
    incident = _incident_bundle(
        [0.0, 2.0e-3, -2.0e-3, 0.0, 0.0],
        [0.0, 0.0, 0.0, 2.0e-3, -2.0e-3],
        [0.6, 0.1, 0.1, 0.1, 0.1],
    )
    scan_x = np.zeros((1, 1))
    scan_y = np.zeros((1, 1))

    result = simulate_angle_resolved_stem(
        state,
        SimpleNamespace(incident=incident),
        (AngularDetector("bf", 0.0, 10.0),),
        scan_x,
        scan_y,
    )

    assert result.metrics["model"] == "multislice_angle_resolved"
    assert result.metrics["specimen_model"] == "continuous_column_multislice"
    assert result.metrics["specimen_slice_count"] == 2
    assert result.fractions["bf"].shape == (1, 1)
    assert 0.0 <= result.fractions["bf"][0, 0] <= 1.0


def test_angle_resolved_stem_applies_per_probe_descan_detector_shift():
    state = default_state()
    state.illumination_mode = "STEM"
    state.sample.specimen_preset_key = "vacuum"
    state.sample.thickness_nm = 0.0
    state.sample.wave_grid_pixels = 32
    state.sample.wave_field_of_view_angstrom = 16.0
    state.sample.wave_multislice_enabled = False
    incident = _incident_bundle(
        [0.0, 2.0e-3, -2.0e-3, 0.0, 0.0],
        [0.0, 0.0, 0.0, 2.0e-3, -2.0e-3],
        [0.6, 0.1, 0.1, 0.1, 0.1],
    )
    scan_x = np.zeros((1, 2))
    scan_y = np.zeros((1, 2))

    result = simulate_angle_resolved_stem(
        state,
        SimpleNamespace(incident=incident),
        (AngularDetector("bf", 0.0, 5.0),),
        scan_x,
        scan_y,
        detector_center_shifts_mrad={
            "bf": (np.array([[0.0, 20.0]]), np.zeros((1, 2))),
        },
    )

    assert result.metrics["descan_detector_shift_applied"] is True
    assert result.fractions["bf"][0, 0] > result.fractions["bf"][0, 1]


def test_explicit_cuda_preference_reaches_tem_multislice_and_imaging_fft():
    if not compute_backend.cupy_capability().available:
        pytest.skip("CuPy CUDA backend unavailable")
    state = default_state()
    state.illumination_mode = "TEM"
    state.acceleration_enabled = True
    state.acceleration_backend = "CUDA GPU"
    state.sample.specimen_preset_key = "vacuum"
    state.sample.thickness_nm = 0.4
    state.sample.wave_grid_pixels = 32
    state.sample.wave_field_of_view_angstrom = 16.0
    state.sample.wave_multislice_enabled = True
    state.sample.wave_slice_thickness_angstrom = 2.0
    incident = _incident_bundle(
        [0.0, 1.0e-4, -1.0e-4],
        [0.0, 0.0, 0.0],
        [0.8, 0.1, 0.1],
    )

    result = simulate_wave_image(state, SimpleNamespace(incident=incident))

    assert result.metrics["specimen_compute_backend"] == "CuPy CUDA"
    assert result.metrics["fft_compute_backend"] == "CuPy CUDA"
    assert result.metrics["wave_compute_backend"] == "CuPy CUDA"
    assert result.metrics["specimen_fallback_reason"] is None
