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
    _simulate_angle_resolved_stem_single as simulate_local_stem_operator,
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

    focused = simulate_local_stem_operator(
        state,
        SimpleNamespace(incident=_incident_bundle_with_waist(0.0)),
        detectors,
        scan_x_um,
        scan_y_um,
    )
    defocused = simulate_local_stem_operator(
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

    result = simulate_local_stem_operator(
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

    simulate_local_stem_operator(
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

    result = simulate_local_stem_operator(
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






# This module tests supplied local fields; production admission remains active.
from local_wave_operator_fixture import supplied_local_probe


@pytest.mark.parametrize("entry", ["image", "internal_image", "incident", "reprojection"])
@pytest.mark.parametrize("inserted", [False, True])
def test_tem_requests_require_executed_coherent_source_before_any_operator(monkeypatch, entry, inserted):
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.physics import wave_imaging as wave
    from temsim.physics.source_admission import UnsupportedWaveSource
    state = default_state()
    _retract_stem_detectors(state)
    state.illumination_mode = "TEM"
    state.sample.inserted = inserted
    state.sample.wave_grid_pixels = 32
    before = capture_instrument_snapshot(state).digest
    forbidden = lambda *a, **kw: pytest.fail("Unqualified source must not execute any TEM operator")
    monkeypatch.setattr(wave, "prepare_specimen_potentials", forbidden)
    simulation = SimpleNamespace(incident=_incident_bundle_with_waist(10.0))
    frequencies = np.fft.fftshift(np.fft.fftfreq(32, d=.5))
    with pytest.raises(UnsupportedWaveSource, match="coherent phase is unavailable"):
        if entry == "image":
            wave.simulate_wave_image(state, simulation)
        elif entry == "internal_image":
            wave._simulate_wave_image(state, simulation)
        elif entry == "reprojection":
            wave.reproject_wave_image(state, SimpleNamespace(metrics={}))
        else:
            wave._incident_wave(state, _weighted_ray_statistics(simulation.incident), frequencies, frequencies, .025)
    assert capture_instrument_snapshot(state).digest == before
