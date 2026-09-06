from dataclasses import replace

import numpy as np
from PySide6.QtWidgets import QMessageBox

from temsim.calculation_cache import calculation_signatures
from temsim.detector.stem_signal import StemScanResult, DetectorSignal
from temsim.gui.scan_panel import ScanControlView
from temsim.optics.column import default_state
from temsim.physics.stem_sampling import detector_sampling_report


def frame_for(state):
    report = detector_sampling_report(
        {"bf": (0., 10.), "df": (16., 112.), "haadf": (60., 331.)},
        maximum_angle_mrad=42., wavelength_angstrom=.019687,
        requested_fov_angstrom=40., requested_grid_pixels=256,
        bandwidth_fraction=2/3, probe_semiangle_mrad=25.,
    )
    fractions = {key: np.arange(4, dtype=float).reshape(2, 2) / 4 for key in ("haadf", "df", "bf")}
    fractions["haadf"] = np.zeros((2, 2))
    return StemScanResult(
        scan_x_um=np.array([[0., .0001], [0., .0001]]),
        scan_y_um=np.array([[0., 0.], [.0001, .0001]]), fractions=fractions,
        detector_signals={k: DetectorSignal(k, k.upper(), float(v.mean()), 1., 2., 3., None)
                          for k, v in fractions.items()},
        metrics={"model": "multislice_angle_resolved", "detector_sampling": report,
                 "sampling_state_signature": calculation_signatures(state)["stem"]},
    )


def setup(qtbot):
    state = default_state()
    view = ScanControlView()
    qtbot.addWidget(view)
    view.set_state(state)
    frame = frame_for(state)
    view._set_stem_frame(frame)
    return view, state, frame


def test_outside_grid_is_not_displayed_as_physical_black_image(qtbot):
    view, state, frame = setup(qtbot)
    assert view.detector_image_items["haadf"].image is None
    assert "Not simulated" in view.detector_sampling_labels["haadf"].text()
    assert "Partial band" in view.detector_sampling_labels["df"].text()
    assert "Limited angular coverage" in view.image_model_notice.text()
    assert "HAADF not simulated" in view._stem_frame_summary("Done")
    assert "[partial band]" in view._stem_frame_summary("Done")
    assert np.all(frame.fractions["haadf"] == 0.)  # Preserve raw cached data.
    np.testing.assert_array_equal(view.detector_image_items["bf"].image, frame.fractions["bf"].T)
    assert tuple(view.detector_image_items["bf"].levels) == (0., .75)  # Not reversed.


def test_sampling_button_changes_grid_only_and_retains_frame(qtbot, monkeypatch):
    view, state, frame = setup(qtbot)
    before = calculation_signatures(state)
    changes = []
    view.parameters_changed.connect(changes.append)
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
    view._match_detector_sampling()
    assert state.sample.wave_grid_pixels == frame.metrics["detector_sampling"]["recommended_grid_pixels"]
    assert view._stem_frame is frame
    assert changes == ["sample.wave_grid_pixels"]
    assert not view.match_detector_sampling.isEnabled()
    after = calculation_signatures(state)
    for key in ("incident", "eds", "elastic", "sample_region"):
        assert before[key] == after[key]


def test_cancel_and_stale_proposal_do_not_mutate_state(qtbot, monkeypatch):
    view, state, frame = setup(qtbot)
    pixels = state.sample.wave_grid_pixels
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.No)
    view._match_detector_sampling()
    assert state.sample.wave_grid_pixels == pixels
    state.sample.wave_field_of_view_angstrom = 123.
    errors = []
    view.error.connect(errors.append)
    view._match_detector_sampling()
    assert "older state" in errors[-1]
    assert state.sample.wave_grid_pixels == pixels


def test_pause_keeps_sampling_labels_and_data_from_same_complete_frame(qtbot):
    view, state, frame = setup(qtbot)
    view.pause_image_refresh.setChecked(True)
    newer = replace(frame, metrics={"model": "geometric_detector_interception"})
    view._set_stem_frame(newer)
    assert "Limited angular coverage" in view.image_model_notice.text()
    assert view.detector_image_items["haadf"].image is None
    view.pause_image_refresh.setChecked(False)
    assert "Geometry preview only" in view.image_model_notice.text()
    assert view.detector_image_items["haadf"].image is not None
    assert view.detector_sampling_labels["haadf"].isHidden()


def test_illumination_outside_grid_suppresses_all_images(qtbot):
    view, state, frame = setup(qtbot)
    frame.metrics["detector_sampling"]["illumination_covered"] = False
    view._set_stem_frame(frame)
    assert all(item.image is None for item in view.detector_image_items.values())
