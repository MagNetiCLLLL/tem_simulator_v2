from dataclasses import replace
from types import SimpleNamespace

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


def test_low_angle_annulus_visibly_reports_illumination_overlap(qtbot):
    view, state, frame = setup(qtbot)
    frame.metrics["detector_sampling"] = detector_sampling_report(
        {"bf": (0., 1.), "df": (1., 7.), "haadf": (60., 120.)},
        maximum_angle_mrad=169., wavelength_angstrom=.019687,
        requested_fov_angstrom=40., requested_grid_pixels=1024,
        bandwidth_fraction=2/3, probe_semiangle_mrad=24.8,
    )
    view._set_stem_frame(frame)
    view.result_tabs.setCurrentIndex(1)
    view.resize(1000, 650)
    view.show()
    df = view.detector_sampling_labels["df"]
    haadf = view.detector_sampling_labels["haadf"]
    qtbot.waitUntil(df.isVisible)
    assert "Required: 1\u20137 mrad" in df.text()
    assert "Band covered" in df.text()
    assert "Overlaps illumination disk" in df.text()
    assert "probe 24.8 mrad" in df.text()
    assert haadf.isVisible()
    assert "Required: 60\u2013120 mrad" in haadf.text()
    assert "Band covered" in haadf.text()
    assert "Overlaps illumination disk" not in haadf.text()
    assert df.wordWrap()
    assert df.height() >= df.heightForWidth(df.width())
    assert view.width() == 1000  # The additional explanation must wrap, not widen the page.
    np.testing.assert_array_equal(view.detector_image_items["df"].image, frame.fractions["df"].T)


def test_tilted_direct_disk_exposes_df_adjustment_action(qtbot):
    view, state, frame = setup(qtbot)
    frame.metrics["detector_sampling"] = detector_sampling_report(
        {"bf": (0., 1.), "df": (4., 10.), "haadf": (60., 120.)},
        illumination_bounds_mrad={"bf": (6., 8.), "df": (0., 17.), "haadf": (53., 127.)},
        probe_center_mrad=(7., 0.), probe_semiangle_mrad=2.,
        maximum_angle_mrad=169., wavelength_angstrom=.019687,
        requested_fov_angstrom=40., requested_grid_pixels=1024,
        bandwidth_fraction=2/3,
    )
    view._set_stem_frame(frame)
    assert "Overlaps illumination disk" in view.detector_sampling_labels["df"].text()
    assert not view.exclude_direct_beam.isHidden()
    assert view.exclude_direct_beam.isEnabled()
    requested = []
    view.df_geometry_requested.connect(requested.append)
    view.exclude_direct_beam.click()
    assert requested == [frame]


def test_auto_contrast_exposes_nonzero_black_level_from_actual_displayed_frame(qtbot):
    view, state, frame = setup(qtbot)
    values = np.array([[.2, .4], [.6, .8]])
    frame = replace(frame, fractions={**frame.fractions, "df": values.copy()})
    view._set_stem_frame(frame)
    view.result_tabs.setCurrentIndex(1)
    view.resize(1000, 650)
    view.show()
    label = view.detector_contrast_labels["df"]
    qtbot.waitUntil(label.isVisible)
    assert label.text() == "Auto contrast: 0.2\u20130.8"
    assert "Black = minimum (0.2)" in label.toolTip()
    assert "white = maximum (0.8)" in label.toolTip()
    assert tuple(view.detector_image_items["df"].levels) == (.2, .8)
    np.testing.assert_array_equal(view.detector_image_items["df"].image, values.T)
    np.testing.assert_array_equal(frame.fractions["df"], values)
    assert view.detector_contrast_labels["haadf"].isHidden()  # Outside wave grid.

    view.pause_image_refresh.setChecked(True)
    newer = replace(frame, fractions={**frame.fractions, "df": values + 1.})
    view._set_stem_frame(newer)
    assert label.text() == "Auto contrast: 0.2\u20130.8"
    np.testing.assert_array_equal(view.detector_image_items["df"].image, values.T)

    bank = replace(frame, fractions={**frame.fractions, "df": values + 2.})
    view.set_bank_readout(SimpleNamespace(stem=bank))
    view.image_source.setCurrentIndex(view.image_source.findData("bank"))
    assert label.text() == "Auto contrast: 2.2\u20132.8"
    np.testing.assert_array_equal(view.detector_image_items["df"].image, (values + 2.).T)
    view.mark_bank_readout_pending("Another bank selection is still computing")
    assert label.text() == "Auto contrast: 2.2\u20132.8"
    view.set_bank_readout(None)
    assert label.isHidden() and label.text() == ""
    assert view.detector_image_items["df"].image is None

    view.image_source.setCurrentIndex(view.image_source.findData("current"))
    assert label.text() == "Auto contrast: 0.2\u20130.8"
    view.pause_image_refresh.setChecked(False)
    assert label.text() == "Auto contrast: 1.2\u20131.8"
    np.testing.assert_array_equal(view.detector_image_items["df"].image, (values + 1.).T)
    view.display_result(None, complete=True)
    assert all(item.isHidden() and item.text() == "" for item in view.detector_contrast_labels.values())
    view._set_stem_frame(frame)
    view.set_state(default_state())
    assert all(item.isHidden() and item.text() == "" for item in view.detector_contrast_labels.values())


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
