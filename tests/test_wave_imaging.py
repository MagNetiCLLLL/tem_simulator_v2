from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.physics import compute_backend
from temsim.physics.camera_wave import project_wave_to_recording_plane
from temsim.physics.wave_imaging import (
    _incident_wave,
    _weighted_ray_statistics,
    effective_sample_thickness_nm,
    estimate_tem_wave_memory_bytes,
    reproject_wave_image,
    simulate_wave_image,
    tem_wave_imaging_enabled,
)
from temsim.physics.recording_stop import (
    active_tem_recording_plane,
    tem_projection_reference_plane,
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


def _retract_stem_detectors(state):
    for detector in state.stem_detectors:
        detector.inserted = False
        detector.readout_enabled = False


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


def test_tem_wave_observable_accepts_reference_or_imported_real_cif():
    state = default_state()
    _retract_stem_detectors(state)
    state.sample.wave_enabled = True
    state.illumination_mode = "TEM"

    assert tem_wave_imaging_enabled(state) is True

    state.illumination_mode = "STEM"
    assert tem_wave_imaging_enabled(state) is False

    state.illumination_mode = "TEM"
    state.sample.specimen_mode = "atomic"
    state.sample.cif_path = ""
    assert tem_wave_imaging_enabled(state) is False

    state.sample.cif_path = "real-sample.cif"
    assert tem_wave_imaging_enabled(state) is True

    state.sample.inserted = False
    assert tem_wave_imaging_enabled(state) is True


def test_tem_recording_plane_stops_at_screen_and_alignment_falls_back_to_camera():
    state = default_state()
    _retract_stem_detectors(state)

    assert active_tem_recording_plane(state).key == "flu_screen"
    state.fluorescent_screen.inserted = False
    assert active_tem_recording_plane(state).key == "camera"
    state.camera.inserted = False

    with pytest.raises(ValueError, match="Insert the Fluorescent Screen or Camera"):
        active_tem_recording_plane(state)
    assert tem_projection_reference_plane(state).key == "camera"


def test_tem_recording_rejects_an_inserted_upstream_stem_detector():
    state = default_state()

    with pytest.raises(ValueError, match="Retract upstream STEM detector"):
        active_tem_recording_plane(state)


def test_projector_checkpoint_reprojection_matches_fresh_wave_calculation():
    state = default_state()
    _retract_stem_detectors(state)
    state.illumination_mode = "TEM"
    state.projector_mode = "image"
    state.fluorescent_screen.inserted = False
    state.camera.inserted = True
    state.sample.reference_sample_key = "si_110"
    state.sample.thickness_nm = 2.0
    state.sample.wave_grid_pixels = 32
    state.sample.wave_field_of_view_angstrom = 16.0
    state.sample.wave_multislice_enabled = True
    state.sample.wave_atomistic_enabled = True
    incident = _incident_bundle(
        [0.0, 1.0e-4, -1.0e-4],
        [0.0, 0.0, 0.0],
        [0.8, 0.1, 0.1],
    )
    simulation = SimpleNamespace(incident=incident)

    previous = simulate_wave_image(state, simulation)
    previous_exit_wave = previous.exit_wave.copy()
    previous_transfer = np.asarray(
        previous.metrics["projector_transfer_matrix"], dtype=float
    )
    state.intermediate_lens.percent += 2.0
    reprojected = reproject_wave_image(state, previous)
    fresh = simulate_wave_image(state, simulation)

    np.testing.assert_array_equal(previous.exit_wave, previous_exit_wave)
    np.testing.assert_allclose(
        reprojected.image_intensity, fresh.image_intensity, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        reprojected.camera_electron_optical_intensity,
        fresh.camera_electron_optical_intensity,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(reprojected.camera_x_mm, fresh.camera_x_mm)
    np.testing.assert_array_equal(reprojected.camera_y_mm, fresh.camera_y_mm)
    assert reprojected.metrics["projector_checkpoint_reused"] is True
    assert reprojected.metrics["recording_plane_key"] == "camera"
    assert not np.allclose(
        previous_transfer,
        np.asarray(reprojected.metrics["projector_transfer_matrix"], dtype=float),
    )
    checkpoint = previous.projector_checkpoint
    xx, _yy = np.meshgrid(
        checkpoint.x_angstrom,
        checkpoint.y_angstrom,
        indexing="xy",
    )
    first_wave = checkpoint.objective_wave_configurations[0]
    second_wave = first_wave * np.exp(1j * 2.0 * np.pi * xx / 4.0)
    forward = reproject_wave_image(
        state,
        replace(
            previous,
            projector_checkpoint=replace(
                checkpoint,
                objective_wave_configurations=(first_wave, second_wave),
            ),
        ),
    )
    reversed_order = reproject_wave_image(
        state,
        replace(
            previous,
            projector_checkpoint=replace(
                checkpoint,
                objective_wave_configurations=(second_wave, first_wave),
            ),
        ),
    )
    np.testing.assert_allclose(
        forward.image_intensity,
        reversed_order.image_intensity,
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    assert forward.metrics[
        "camera_collected_zero_loss_relative_intensity"
    ] == pytest.approx(
        reversed_order.metrics[
            "camera_collected_zero_loss_relative_intensity"
        ],
        rel=1.0e-14,
    )


def test_defocused_image_wave_uses_symplectic_specimen_canonical_transfer():
    from temsim.physics.simulation import run

    state = default_state()
    _retract_stem_detectors(state)
    # Isolate the single Collins transfer: inserted downstream apertures
    # correctly select the separate coherent multi-plane propagation path.
    for aperture in state.apertures:
        if float(aperture.z_mm) > float(state.sample.z_mm):
            aperture.inserted = False
    state.projector_mode = "image"
    state.fluorescent_screen.inserted = False
    state.camera.inserted = True
    state.intermediate_lens.percent += 2.0
    axis = np.linspace(-8.0, 8.0, 32)
    wave = np.ones((32, 32), dtype=np.complex128)

    projection = project_wave_to_recording_plane(
        state, wave, axis, axis, 0.0197
    )
    matrix = np.asarray(projection.transfer_matrix, dtype=float)
    symplectic_form = np.block([
        [np.zeros((2, 2)), np.eye(2)],
        [-np.eye(2), np.zeros((2, 2))],
    ])
    quadratic_phase = np.linalg.solve(
        matrix[:2, 2:], matrix[:2, :2]
    )

    assert projection.method == "collins_fft_linear_canonical_transform"
    assert projection.metrics["projector_transfer_input_basis"] == (
        "specimen_canonical_momentum"
    )
    assert np.linalg.norm(
        matrix.T @ symplectic_form @ matrix - symplectic_form, ord=2
    ) <= 1.0e-7
    assert np.linalg.norm(
        quadratic_phase - quadratic_phase.T, ord=2
    ) <= 1.0e-9
    state.acceleration_enabled = False
    state.step_mm = 5.0
    state.history_step_mm = 5.0
    emitter = getattr(state.electron_gun, "emitter", None)
    if emitter is None:
        state.electron_gun.ray_count = 9
    else:
        emitter.ray_count = 9
    simulation = run(state)
    np.testing.assert_allclose(
        simulation.metrics["j_img"],
        projection.metrics["projector_image_map"],
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_tem_wave_memory_estimate_accounts_for_large_fft_grid():
    state = default_state()
    _retract_stem_detectors(state)
    state.illumination_mode = "TEM"
    state.sample.wave_enabled = True
    state.sample.wave_multislice_enabled = False
    state.sample.wave_grid_pixels = 8192

    estimate = estimate_tem_wave_memory_bytes(state)

    assert estimate > 15 * 1024**3


def test_retracted_sample_ignores_dormant_custom_cif_settings(tmp_path):
    state = default_state()
    _retract_stem_detectors(state)
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


def test_tem_incident_wave_uses_traced_condenser_focus_curvature():
    state = default_state()
    state.illumination_mode = "TEM"
    frequencies = np.fft.fftshift(np.fft.fftfreq(32, d=0.5))
    focused_stats = _weighted_ray_statistics(
        _incident_bundle_with_waist(0.0)
    )
    defocused_stats = _weighted_ray_statistics(
        _incident_bundle_with_waist(10.0)
    )

    focused = _incident_wave(
        state, focused_stats, frequencies, frequencies, 0.025
    )
    defocused = _incident_wave(
        state, defocused_stats, frequencies, frequencies, 0.025
    )

    assert focused_stats["radial_wavefront_curvature_per_m"] == pytest.approx(
        0.0
    )
    assert abs(defocused_stats["radial_wavefront_curvature_per_m"]) > 0.0
    assert not np.allclose(focused, defocused)


def test_si_110_stem_detector_signals_respond_to_position_and_traced_defocus():
    state = default_state()
    state.acceleration_enabled = False
    state.illumination_mode = "STEM"
    state.sample.reference_sample_key = "si_110"
    state.sample.thickness_nm = 2.0
    state.sample.wave_grid_pixels = 256
    state.sample.wave_field_of_view_angstrom = 40.0
    state.sample.wave_multislice_enabled = True
    state.sample.wave_atomistic_enabled = True
    state.objective_lens.cs_mm = 0.0
    state.objective_lens.cc_mm = 0.0
    state.probe_corrector_installed = False
    # Symmetric positions keep the actual CIF ROI origin fixed while sampling
    # different positions within the projected crystal unit cell.
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
    _retract_stem_detectors(state)
    state.illumination_mode = "TEM"
    state.sample.reference_sample_key = "si_110"
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
    assert result.metrics["specimen_model"] == "atomistic_static_multislice"
    assert result.metrics["specimen_slice_count"] == 10
    assert result.metrics["specimen_slice_thickness_angstrom"] == pytest.approx(2.0)
    assert result.metrics["convergence_semiangle_rad"] > 0.0
    # Real IAM scattering can leave the finite reciprocal-space bandwidth.
    # Its diagnostic must account for that norm loss without creating charge.
    initial = result.metrics["specimen_initial_integrated_intensity"]
    final = result.metrics["specimen_final_integrated_intensity"]
    assert 0.0 < final <= initial
    assert result.metrics["specimen_maximum_relative_intensity_change"] == pytest.approx(
        (initial - final) / initial, abs=1.0e-12
    )
    assert result.metrics["specimen_compute_backend"] == "NumPy CPU"
    assert result.metrics["fft_compute_backend"] == "NumPy CPU"
    assert 0.0 <= result.metrics[
        "elastic_exit_intensity_outside_incident_cone_fraction"
    ] <= 1.0
    assert 0.0 <= result.metrics[
        "elastic_incident_baseline_outside_cone_fraction"
    ] <= 1.0


def test_reference_cif_rejects_projected_phase_preview():
    state = default_state()
    _retract_stem_detectors(state)
    state.illumination_mode = "TEM"
    state.sample.reference_sample_key = "si_110"
    state.sample.thickness_nm = 2.0
    state.sample.wave_grid_pixels = 32
    state.sample.wave_field_of_view_angstrom = 16.0
    state.sample.wave_multislice_enabled = False
    incident = _incident_bundle(
        [0.0, 1.0e-4, -1.0e-4],
        [0.0, 0.0, 0.0],
        [0.8, 0.1, 0.1],
    )

    with pytest.raises(ValueError, match="CIF requires multislice"):
        simulate_wave_image(state, SimpleNamespace(incident=incident))


def test_angle_resolved_stem_uses_the_same_multislice_specimen_model():
    state = default_state()
    state.illumination_mode = "STEM"
    state.sample.reference_sample_key = "si_110"
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
    assert result.metrics["specimen_model"] == "atomistic_static_multislice"
    assert result.metrics["specimen_slice_count"] == 2
    assert result.fractions["bf"].shape == (1, 1)
    assert 0.0 <= result.fractions["bf"][0, 0] <= 1.0


def test_angle_resolved_stem_reports_completed_cpu_probe_batches():
    state = default_state()
    state.acceleration_enabled = False
    state.illumination_mode = "STEM"
    state.sample.reference_sample_key = "si_110"
    state.sample.thickness_nm = 0.0
    state.sample.wave_grid_pixels = 32
    state.sample.wave_field_of_view_angstrom = 16.0
    state.sample.wave_multislice_enabled = False
    state.sample.wave_atomistic_enabled = False
    incident = _incident_bundle(
        [0.0, 2.0e-3, -2.0e-3, 0.0, 0.0],
        [0.0, 0.0, 0.0, 2.0e-3, -2.0e-3],
        [0.6, 0.1, 0.1, 0.1, 0.1],
    )
    scan_x = np.zeros((1, 9))
    scan_y = np.zeros((1, 9))
    progress = []

    simulate_angle_resolved_stem(
        state,
        SimpleNamespace(incident=incident),
        (AngularDetector("bf", 0.0, 10.0),),
        scan_x,
        scan_y,
        progress_callback=lambda completed, total, stage: progress.append(
            (completed, total, stage)
        ),
    )

    assert progress[0] == (0, 1, "Preparing STEM specimen potential")
    assert progress[1] == (1, 4, "STEM specimen potential ready")
    assert progress[2][:2] == (2, 4)
    assert "8/9" in progress[2][2]
    assert progress[3][:2] == (3, 4)
    assert "9/9" in progress[3][2]
    assert progress[-1] == (4, 4, "STEM detector frame complete")


def test_angle_resolved_stem_applies_per_probe_descan_detector_shift():
    state = default_state()
    state.illumination_mode = "STEM"
    state.sample.reference_sample_key = "si_110"
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
    _retract_stem_detectors(state)
    state.illumination_mode = "TEM"
    state.acceleration_enabled = True
    state.acceleration_backend = "CUDA GPU"
    state.sample.reference_sample_key = "si_110"
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
