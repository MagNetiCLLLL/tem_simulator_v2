from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.physics import compute_backend
from temsim.physics.wave_imaging import (
    _weighted_ray_statistics,
    effective_sample_thickness_nm,
    estimate_tem_wave_memory_bytes,
    simulate_wave_image,
    tem_wave_imaging_enabled,
)
from temsim.physics.stem_wave_imaging import (
    AngularDetector,
    probe_focus_aberrations,
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


def _incident_bundle_with_waist(waist_offset_nm):
    """Five rays whose paraxial waist is the requested distance from sample."""

    slope = 0.03
    tx = np.asarray([0.0, slope, -slope, 0.0, 0.0])
    ty = np.asarray([0.0, 0.0, 0.0, slope, -slope])
    waist_offset_m = float(waist_offset_nm) * 1.0e-9
    return SimpleNamespace(
        alive=np.ones(5, dtype=bool),
        ray_weight=np.full(5, 0.2),
        x=(-tx * waist_offset_m)[None, :],
        y=(-ty * waist_offset_m)[None, :],
        tx=tx[None, :],
        ty=ty[None, :],
    )


def test_retracted_sample_has_zero_interacting_wave_thickness():
    state = default_state()
    state.sample.thickness_nm = 250.0

    assert effective_sample_thickness_nm(state) == pytest.approx(250.0)
    state.sample.inserted = False
    assert effective_sample_thickness_nm(state) == 0.0


def test_tem_wave_observable_requires_tem_and_real_sample_modes():
    state = default_state()
    state.sample.wave_enabled = True
    state.illumination_mode = "TEM"
    state.sample.specimen_mode = "atomic"

    assert tem_wave_imaging_enabled(state) is True

    state.illumination_mode = "STEM"
    assert tem_wave_imaging_enabled(state) is False

    state.illumination_mode = "TEM"
    state.sample.specimen_mode = "virtual"
    assert tem_wave_imaging_enabled(state) is False

    state.sample.specimen_mode = "atomic"
    state.sample.inserted = False
    assert tem_wave_imaging_enabled(state) is True


def test_tem_wave_memory_estimate_accounts_for_large_fft_grid():
    state = default_state()
    state.illumination_mode = "TEM"
    state.sample.wave_enabled = True
    state.sample.wave_multislice_enabled = False
    state.sample.wave_grid_pixels = 8192

    estimate = estimate_tem_wave_memory_bytes(state)

    assert estimate > 15 * 1024**3


def test_retracted_sample_ignores_dormant_custom_cif_settings(tmp_path):
    state = default_state()
    state.sample.inserted = False
    state.sample.cif_path = str(tmp_path / "missing.cif")
    state.sample.wave_atomistic_enabled = False
    state.sample.wave_multislice_enabled = False
    state.sample.wave_grid_pixels = 32
    state.sample.wave_field_of_view_angstrom = 16.0
    incident = _incident_bundle(
        [0.0, 1.0e-4, -1.0e-4],
        [0.0, 0.0, 0.0],
        [0.8, 0.1, 0.1],
    )

    result = simulate_wave_image(state, SimpleNamespace(incident=incident))

    assert result.preset_key == "vacuum"
    assert result.metrics["specimen_sample_inserted"] is False
    assert result.metrics["specimen_sample_interaction_applied"] is False
    assert result.metrics["specimen_total_thickness_angstrom"] == 0.0


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


def test_probe_focus_uses_traced_waist_once_with_fresnel_sign():
    state = default_state()
    state.sample.wave_defocus_nm = 2.0
    statistics = _weighted_ray_statistics(_incident_bundle_with_waist(5.0))

    coefficients, focus = probe_focus_aberrations(state, statistics)

    assert statistics["waist_offset_m"] == pytest.approx(5.0e-9)
    assert focus.ray_defocus_mm == pytest.approx(-5.0e-6)
    assert focus.configured_defocus_mm == pytest.approx(2.0e-6)
    assert focus.effective_defocus_mm == pytest.approx(-3.0e-6)
    assert coefficients.c1_mm == pytest.approx(-3.0e-6)


def test_si_110_stem_detector_signals_respond_to_position_and_traced_defocus():
    state = default_state()
    state.acceleration_enabled = False
    state.illumination_mode = "STEM"
    state.sample.specimen_preset_key = "si_110"
    state.sample.thickness_nm = 10.0
    state.sample.wave_grid_pixels = 256
    state.sample.wave_field_of_view_angstrom = 40.0
    state.sample.wave_multislice_enabled = False
    state.sample.wave_atomistic_enabled = False
    state.objective_lens.cs_mm = 0.0
    state.objective_lens.cc_mm = 0.0
    state.probe_corrector_installed = False
    # Symmetric positions keep the ROI origin fixed: centre is a [110] column
    # and the two +/-1.92 A samples lie between the configured columns.
    scan_x_um = np.asarray([[-1.92e-4, 0.0, 1.92e-4]])
    scan_y_um = np.zeros_like(scan_x_um)
    detectors = (
        AngularDetector("bf", 0.0, 15.0),
        AngularDetector("df", 20.0, 40.0),
        AngularDetector("haadf", 40.0, 60.0),
    )

    focused = simulate_angle_resolved_stem(
        state,
        SimpleNamespace(incident=_incident_bundle_with_waist(0.0)),
        detectors,
        scan_x_um,
        scan_y_um,
    )
    defocused = simulate_angle_resolved_stem(
        state,
        SimpleNamespace(incident=_incident_bundle_with_waist(10.0)),
        detectors,
        scan_x_um,
        scan_y_um,
    )

    assert focused.metrics["probe_effective_defocus_nm"] == pytest.approx(0.0)
    assert defocused.metrics["probe_ray_waist_offset_nm"] == pytest.approx(10.0)
    assert defocused.metrics["probe_effective_defocus_nm"] == pytest.approx(-10.0)
    for key in ("bf", "df", "haadf"):
        assert np.ptp(focused.fractions[key]) > 0.0
        assert not np.allclose(
            defocused.fractions[key],
            focused.fractions[key],
            rtol=1.0e-8,
            atol=1.0e-12,
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
    assert result.linear_diffraction_probability.shape == (32, 32)
    assert np.sum(result.linear_diffraction_probability) == pytest.approx(1.0)
    assert result.spatial_frequency_inv_angstrom.shape == (32,)
    assert result.spatial_frequency_y_inv_angstrom.shape == (32,)
    assert result.metrics["specimen_model"] == "continuous_column_multislice"
    assert result.metrics["specimen_slice_count"] == 10
    assert result.metrics["specimen_slice_thickness_angstrom"] == pytest.approx(2.0)
    assert result.metrics["convergence_semiangle_rad"] > 0.0
    assert result.metrics["specimen_maximum_relative_intensity_change"] < 1.0e-10
    assert result.metrics["specimen_compute_backend"] == "NumPy CPU"
    assert result.metrics["fft_compute_backend"] == "NumPy CPU"
    assert 0.0 <= result.metrics[
        "elastic_exit_intensity_outside_incident_cone_fraction"
    ] <= 1.0
    assert 0.0 <= result.metrics[
        "elastic_incident_baseline_outside_cone_fraction"
    ] <= 1.0


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
