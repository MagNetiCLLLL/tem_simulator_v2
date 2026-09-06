from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.detector import stem_signal
from temsim.gui.scan_panel import ScanControlView
from temsim.optics.column import default_state
from temsim.physics.fourdstem import (
    FourDSTEMCalibration,
    FourDSTEMWriter,
    RuntimeDetectorImages,
    annular_virtual_detector,
    integrate_virtual_detectors,
)


def _completed_cube(tmp_path):
    calibration = FourDSTEMCalibration.from_rectilinear_axes(
        np.asarray((-1.0, 1.0)),
        np.asarray((-2.0, 2.0)),
        np.asarray((-20.0, 0.0, 20.0)),
        np.asarray((-20.0, 0.0, 20.0)),
    )
    writer = FourDSTEMWriter(tmp_path / "existing.npy", calibration)
    for scan_y in range(2):
        for scan_x in range(2):
            writer.write_frame(
                scan_y,
                scan_x,
                np.arange(9, dtype=float).reshape(3, 3) + scan_y + scan_x,
            )
    return writer.finalize()


def test_scanning_controls_persist_4dstem_settings_and_exclude_both_policies(
    qtbot, tmp_path
):
    state = default_state()
    view = ScanControlView()
    qtbot.addWidget(view)
    view.set_state(state)

    assert state.sample.stem_fourdstem_enabled is False
    assert view.fourdstem_path.isEnabled() is False
    view.fourdstem_enabled.setChecked(True)
    assert state.sample.stem_fourdstem_enabled is True
    assert view.fourdstem_path.isEnabled() is True

    output = tmp_path / "scan.npy"
    view.fourdstem_path.setText(str(output))
    view._fourdstem_path_changed()
    assert state.sample.stem_fourdstem_output_path == str(output)

    view.fourdstem_overwrite.setChecked(True)
    assert state.sample.stem_fourdstem_overwrite is True
    view.fourdstem_resume.setChecked(True)
    assert state.sample.stem_fourdstem_resume is True
    assert state.sample.stem_fourdstem_overwrite is False
    assert view.fourdstem_overwrite.isChecked() is False

    view.fourdstem_response_mode.setCurrentIndex(
        view.fourdstem_response_mode.findData("adjustable")
    )
    qe = view.fourdstem_response_controls["quantum_efficiency"]
    qe.setValue(0.73)
    assert state.sample.stem_fourdstem_response_mode == "adjustable"
    assert state.sample.stem_fourdstem_quantum_efficiency == pytest.approx(0.73)


def test_scanning_result_summary_shows_completed_artifact_path_and_shape(
    qtbot, tmp_path
):
    state = default_state()
    view = ScanControlView()
    qtbot.addWidget(view)
    view.set_state(state)
    artifact = SimpleNamespace(
        path=tmp_path / "result.npy",
        data=np.zeros((3, 4, 16, 20), dtype=np.float32),
    )

    view._update_fourdstem_summary(
        SimpleNamespace(fourdstem_artifact=artifact)
    )

    assert str(Path(artifact.path).resolve()) in view.fourdstem_summary.text()
    assert "3 × 4 × 16 × 20" in view.fourdstem_summary.text()


def test_configured_capture_converts_user_detector_response(monkeypatch, tmp_path):
    state = default_state()
    sample = state.sample
    sample.stem_fourdstem_output_path = str(tmp_path / "configured.npy")
    sample.stem_fourdstem_response_mode = "adjustable"
    sample.stem_fourdstem_quantum_efficiency = 0.8
    sample.stem_fourdstem_charge_spread_sigma_px = 1.25
    sample.stem_fourdstem_read_noise_electrons_rms = 2.5
    sample.stem_fourdstem_poisson_enabled = True
    sample.stem_fourdstem_seed = 19
    captured = {}
    sentinel = object()

    def prepare(_state, _simulation, request):
        captured["request"] = request
        return sentinel

    monkeypatch.setattr(
        "temsim.physics.fourdstem_workflow.prepare_fourdstem_capture",
        prepare,
    )

    result = stem_signal._prepare_configured_fourdstem_capture(
        state, SimpleNamespace()
    )

    assert result is sentinel
    request = captured["request"]
    assert request.path == tmp_path / "configured.npy"
    assert request.response.quantum_efficiency == pytest.approx(0.8)
    assert request.response.charge_spread_sigma_x_px == pytest.approx(1.25)
    assert request.response.charge_spread_sigma_y_px == pytest.approx(1.25)
    assert request.response.read_noise_electrons_rms == pytest.approx(2.5)
    assert request.response.poisson_enabled is True
    assert request.response.seed == 19

    sample.stem_fourdstem_overwrite = True
    sample.stem_fourdstem_resume = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        stem_signal._prepare_configured_fourdstem_capture(
            state, SimpleNamespace()
        )


def test_acquisition_rejects_4dstem_without_explicit_wave_stem(monkeypatch):
    state = default_state()
    state.ac_deflector.enabled = True
    state.ac_deflector.scan_enabled = True
    state.sample.stem_fourdstem_enabled = True
    state.sample.stem_wave_enabled = False
    state.illumination_mode = "STEM"
    monkeypatch.setattr(stem_signal, "sample_illumination_absent", lambda *_: False)

    with pytest.raises(ValueError, match="requires STEM illumination.*wave",):
        stem_signal.acquire_stem_scan(SimpleNamespace(real_interactions=None), state)

    state.sample.stem_fourdstem_enabled = False
    with pytest.raises(ValueError, match="explicit 4D-STEM enablement"):
        stem_signal.acquire_stem_scan(
            SimpleNamespace(real_interactions=None),
            state,
            diffraction_sink=object(),
        )


def test_wave_acquisition_passes_sink_and_returns_artifact(monkeypatch, tmp_path):
    state = default_state()
    state.acceleration_enabled = False
    state.illumination_mode = "STEM"
    state.ac_deflector.enabled = True
    state.ac_deflector.scan_enabled = True
    state.ac_deflector.scan_pixels_x = 1
    state.ac_deflector.scan_lines = 1
    state.sample.stem_wave_enabled = True
    state.sample.stem_fourdstem_enabled = True
    state.sample.stem_fourdstem_output_path = str(tmp_path / "capture.npy")
    for candidate in state.stem_detectors:
        candidate.inserted = False
        candidate.readout_enabled = False
    detector = state.stem_detectors[-1]
    detector.inserted = True
    detector.readout_enabled = True
    detector_key = detector.key
    physical = SimpleNamespace(
        key=detector_key,
        detector=detector,
        inner_mrad=0.0,
        outer_mrad=10.0,
    )
    sink = object()
    prepared = SimpleNamespace(sink=sink)
    artifact = SimpleNamespace(
        path=tmp_path / "capture.npy",
        data=np.zeros((1, 1, 2, 2), dtype=np.float32),
        metadata={"detector_response": {"model": "ideal"}},
    )
    observed = {}
    zero = np.zeros((1, 1), dtype=float)

    monkeypatch.setattr(stem_signal, "sample_illumination_absent", lambda *_: False)
    monkeypatch.setattr(stem_signal, "calibrate_scan_system", lambda *_: None)
    monkeypatch.setattr(
        stem_signal,
        "raster_sample_grid",
        lambda *_args, **_kwargs: (zero, zero, zero),
    )
    monkeypatch.setattr(
        stem_signal, "paired_kick_response", lambda *_: np.zeros((2, 2))
    )
    monkeypatch.setattr(
        stem_signal,
        "physical_angular_detectors",
        lambda *_: ((physical,), {detector_key: None}),
    )
    monkeypatch.setattr(
        stem_signal, "_detector_center_shifts_mrad", lambda *_: None
    )
    monkeypatch.setattr(
        stem_signal,
        "_prepare_configured_fourdstem_capture",
        lambda *_: prepared,
    )

    def simulate(*_args, **kwargs):
        observed["sink"] = kwargs.get("diffraction_sink")
        return SimpleNamespace(
            scan_x_um=zero,
            scan_y_um=zero,
            fractions={detector_key: np.full((1, 1), 0.25)},
            maximum_isotropic_angle_mrad=10.0,
            uncollected_fraction=np.full((1, 1), 0.75),
            truncated_fraction=zero,
            metrics={"model": "thin_phase_angle_resolved"},
            fourdstem_artifact=artifact,
        )

    monkeypatch.setattr(stem_signal, "simulate_angle_resolved_stem", simulate)
    monkeypatch.setattr(
        stem_signal,
        "measure_sample_current",
        lambda *_: SimpleNamespace(fraction=1.0),
    )
    monkeypatch.setattr(
        stem_signal,
        "_real_high_angle_tail",
        lambda *_: (
            {detector_key: zero.copy()}, zero.copy(), zero.copy(), None
        ),
    )
    monkeypatch.setattr(stem_signal, "probe_state_from_simulation", lambda *_: None)

    result = stem_signal.acquire_stem_scan(
        SimpleNamespace(real_interactions=None), state
    )

    assert observed["sink"] is sink
    assert result.fourdstem_artifact is artifact
    assert result.metrics["fourdstem_artifact_path"] == str(
        Path(artifact.path).resolve()
    )
    assert result.metrics["fourdstem_artifact_shape"] == (1, 1, 2, 2)


def test_virtual_annulus_reintegrates_existing_cube_without_solver_signal(
    qtbot, tmp_path
):
    state = default_state()
    view = ScanControlView()
    qtbot.addWidget(view)
    view.set_state(state)
    artifact = _completed_cube(tmp_path)
    frame = SimpleNamespace(fourdstem_artifact=artifact, metrics={})
    view._stem_frame = frame
    view._load_fourdstem_cube(artifact.path)
    changed = []
    view.parameters_changed.connect(changed.append)

    view.fourdstem_virtual_inner_mrad.setValue(0.0)
    view.fourdstem_virtual_outer_mrad.setValue(10.0)
    detector = annular_virtual_detector(
        "reference", artifact.calibration, 0.0, 10.0
    )
    expected = integrate_virtual_detectors(
        artifact, (detector,), chunk_scan_points=1
    )["reference"]
    view._integrate_fourdstem_annulus()

    assert view._fourdstem_virtual_image == pytest.approx(expected)
    assert changed == []
    assert "specimen and multislice were not rerun" in (
        view.fourdstem_result_summary.text()
    )
    assert "fourdstem_virtual_detectors_signature" in frame.metrics
    assert state.sample.stem_fourdstem_virtual_outer_mrad == pytest.approx(10.0)


def test_physical_reintegration_uses_current_plan_without_rerunning_cube(
    qtbot, monkeypatch, tmp_path
):
    state = default_state()
    view = ScanControlView()
    qtbot.addWidget(view)
    view.set_state(state)
    artifact = _completed_cube(tmp_path)
    frame = SimpleNamespace(fourdstem_artifact=artifact, metrics={})
    view._stem_frame = frame
    view._load_fourdstem_cube(artifact.path)
    current_plan = SimpleNamespace(fingerprint="a" * 64)
    image = np.arange(4, dtype=float).reshape(2, 2)
    routed = RuntimeDetectorImages(
        plan_fingerprint=current_plan.fingerprint,
        calibration_fingerprint=artifact.calibration.digest,
        images={"camera": image},
        surviving_weight=np.zeros((2, 2)),
        balance_error=0.0,
        capture_plan_fingerprint="b" * 64,
        plan_changed_since_capture=True,
    )
    calls = []
    monkeypatch.setattr(
        "temsim.physics.record_plane.build_record_plane_plan",
        lambda active_state: calls.append(active_state) or current_plan,
    )
    monkeypatch.setattr(
        "temsim.physics.fourdstem.integrate_runtime_recording_planes",
        lambda active_artifact, calibration, plan, **_kwargs: (
            calls.append((active_artifact, calibration, plan)) or routed
        ),
    )
    changed = []
    view.parameters_changed.connect(changed.append)

    view._rederive_fourdstem_physical_detectors()

    assert calls[0] is state
    assert calls[1] == (view._fourdstem_artifact, None, current_plan)
    assert view._fourdstem_physical_result is routed
    assert changed == []
    assert "raw cube was not rerun" in view.fourdstem_result_summary.text()
    assert "fourdstem_physical_recording_signature" in frame.metrics
