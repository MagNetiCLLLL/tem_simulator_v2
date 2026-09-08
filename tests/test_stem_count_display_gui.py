"""Stored STEM electron counts are presentation data, never resampled by Qt."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.detector.stem_signal import StemScanResult
from temsim.gui.scan_panel import ScanControlView
from temsim.optics.column import default_state
from temsim.physics.stem_sampling import detector_sampling_report


def count_frame(*, seed=7, dwell=.01, offset=0):
    x, y = np.meshgrid(np.arange(4) * .001, np.arange(3) * .001)
    expected = np.arange(12).reshape(3, 4) / 2. + offset
    counts = np.array([[0, 1, 0, 2], [1, 3, 3, 4], [2, 6, 4, 12]]) + offset
    return StemScanResult(
        scan_x_um=x, scan_y_um=y,
        fractions={key: (expected / 100).copy() for key in ("haadf", "df", "bf")},
        expected_electrons={key: expected.copy() for key in ("haadf", "df", "bf")},
        poisson_counts={key: counts.copy() for key in ("haadf", "df", "bf")},
        detector_signals={}, dwell_time_s=dwell,
        metrics={"model": "geometric_detector_interception", "poisson_seed": seed,
                 "poisson_noise_enabled": True, "dwell_time_s": dwell,
                 "scan_frame_period_s": 10., "scan_pixel_size_nm": 1.,
                 "scan_field_of_view_x_nm": 4., "scan_field_of_view_y_nm": 3.},
    )


@pytest.fixture
def view(qtbot):
    widget = ScanControlView()
    qtbot.addWidget(widget)
    widget.set_state(default_state())
    widget.resize(1100, 700)
    widget.result_tabs.setCurrentIndex(1)
    widget.show()
    yield widget
    widget._playback_timer.stop()


def choose(view, quantity):
    view.image_display_quantity.setCurrentIndex(view.image_display_quantity.findData(quantity))


def assert_counts(view, frame):
    for key, values in frame.poisson_counts.items():
        np.testing.assert_array_equal(view.detector_image_items[key].image, values.T)


def test_display_quantities_use_stored_arrays_and_shared_count_scale_without_calculating(view, monkeypatch):
    frame = count_frame()
    original = {field: {key: value.copy() for key, value in getattr(frame, field).items()}
                for field in ("fractions", "expected_electrons", "poisson_counts")}
    view._set_stem_frame(frame)
    assert view.image_display_quantity.currentData() == "ideal"
    assert tuple(view.detector_image_items["bf"].levels) == (0., 1.)
    changes = []
    view.parameters_changed.connect(changes.append)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Changing display quantity must neither solve nor sample")

    monkeypatch.setattr("temsim.detector.stem_signal.acquire_stem_scan", forbidden)
    monkeypatch.setattr("temsim.detector.stem_signal.reweight_stem_scan", forbidden)
    monkeypatch.setattr("temsim.simulation_pipeline.calculate_stem_scan_frame", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    choose(view, "expected")
    item = view.detector_image_items["bf"]
    np.testing.assert_array_equal(item.image, frame.expected_electrons["bf"].T)
    assert tuple(item.levels) == (0., 12.)  # Stored Poisson peak exceeds expected max 5.5.
    assert "electrons/pixel" in view.image_quantity_notice.text()
    assert "Shared count scale: 0–12" in view.detector_contrast_labels["bf"].text()
    choose(view, "poisson")
    assert_counts(view, frame)
    assert tuple(item.levels) == (0., 12.)
    assert np.all(item.image == np.floor(item.image))
    item.render()
    assert item.qimage.pixelColor(0, 0).red() == 0
    assert item.qimage.pixelColor(item.qimage.width() - 1, item.qimage.height() - 1).red() == 255
    assert "Seed 7" in view.image_quantity_notice.text()
    assert "Dwell 0.01 s/pixel" in view.image_quantity_notice.text()
    choose(view, "ideal")
    np.testing.assert_array_equal(item.image, frame.fractions["bf"].T)
    assert tuple(item.levels) == (0., 1.)
    assert changes == []
    for field, images in original.items():
        for key, values in images.items():
            np.testing.assert_array_equal(getattr(frame, field)[key], values)


def test_generate_selects_counts_but_old_frame_stays_explicitly_unavailable_until_new_result(view):
    old = replace(count_frame(), poisson_counts=None, expected_electrons=None,
                  dwell_time_s=None, metrics={"model": "geometric_detector_interception"})
    view._set_stem_frame(old)
    changes = []
    view.parameters_changed.connect(changes.append)
    view.poisson_enabled.setChecked(True)
    assert view.image_display_quantity.currentData() == "poisson"
    assert changes == ["sample.stem_poisson_enabled"]
    assert all(item.image is None for item in view.detector_image_items.values())
    assert "Unavailable" in view.image_quantity_notice.text()
    assert "run High accuracy" in view.image_quantity_notice.text()
    assert "Seed not recorded" in view.image_quantity_notice.text()
    assert "Dwell not recorded" in view.image_quantity_notice.text()
    assert all("not stored" in label.text() for label in view.detector_contrast_labels.values())
    choose(view, "expected")
    assert "Unavailable" in view.image_quantity_notice.text()
    choose(view, "poisson")
    new = count_frame(seed=25)
    view.display_result(None, new, complete=True)
    assert_counts(view, new)
    assert "Unavailable" not in view.image_quantity_notice.text()
    assert "Seed 25" in view.image_quantity_notice.text()
    # Deselecting generation does not erase or regenerate the completed data.
    view.poisson_enabled.setChecked(False)
    assert_counts(view, new)
    assert view.image_display_quantity.currentData() == "poisson"


def test_paused_stale_and_bank_counts_keep_captured_seed_dwell_and_arrays(view):
    old, new, bank = count_frame(), count_frame(seed=9, dwell=.5, offset=3), count_frame(seed=11, dwell=.125, offset=7)
    view._set_stem_frame(old)
    choose(view, "poisson")
    view.pause_image_refresh.setChecked(True)
    view._set_stem_frame(new)
    view._state.sample.stem_poisson_seed = 999
    view._state.ac_deflector.scan_frame_period_s = 99.
    assert_counts(view, old)
    assert "Seed 7" in view.image_quantity_notice.text()
    assert "Dwell 0.01 s/pixel" in view.image_quantity_notice.text()
    view.set_bank_readout(SimpleNamespace(stem=bank, state_snapshot=default_state()))
    view.image_source.setCurrentIndex(view.image_source.findData("bank"))
    assert_counts(view, bank)
    assert "Seed 11" in view.image_quantity_notice.text()
    assert "Dwell 0.125 s/pixel" in view.image_quantity_notice.text()
    view.mark_bank_readout_pending("Another bank frame is pending")
    view.mark_stem_frame_stale()
    assert_counts(view, bank)
    assert "Seed 11" in view.image_quantity_notice.text()
    view.image_source.setCurrentIndex(view.image_source.findData("current"))
    assert_counts(view, old)
    assert "inputs changed" in view.image_model_notice.text()
    assert "Seed 7" in view.image_quantity_notice.text()
    view.pause_image_refresh.setChecked(False)
    assert_counts(view, new)
    assert "Seed 9" in view.image_quantity_notice.text()
    assert "Dwell 0.5 s/pixel" in view.image_quantity_notice.text()


def test_missing_counts_in_paused_and_bank_frames_do_not_borrow_current_counts(view):
    old = replace(count_frame(), poisson_counts=None)
    view._set_stem_frame(old)
    view.pause_image_refresh.setChecked(True)
    view._set_stem_frame(count_frame(seed=8))
    choose(view, "poisson")
    assert all(item.image is None for item in view.detector_image_items.values())
    assert "Resume refresh" in view.image_quantity_notice.text()
    view.set_bank_readout(SimpleNamespace(stem=old))
    view.image_source.setCurrentIndex(view.image_source.findData("bank"))
    assert "bank frame" in view.image_quantity_notice.text()
    assert all(item.image is None for item in view.detector_image_items.values())


def test_line_playback_and_quantity_switch_reveal_one_fixed_count_realization(view, monkeypatch):
    frame = count_frame()
    view._set_stem_frame(frame)
    choose(view, "poisson")
    original = frame.poisson_counts["bf"].copy()
    monkeypatch.setattr(np.random, "default_rng", lambda *_: pytest.fail("Playback must not sample"))
    view._playback_started_s = 0.
    monkeypatch.setattr("temsim.gui.scan_panel.perf_counter", lambda: 1.)
    view._playback_tick()
    item = view.detector_image_items["bf"]
    first_line = item.image.copy()
    np.testing.assert_array_equal(first_line[:, 0], original[0])
    assert np.isnan(first_line[:, 1:]).all()
    choose(view, "expected")
    assert np.isnan(item.image[:, 1:]).all()
    choose(view, "poisson")
    np.testing.assert_array_equal(item.image, first_line)
    monkeypatch.setattr("temsim.gui.scan_panel.perf_counter", lambda: 8.)
    view._playback_tick()
    np.testing.assert_array_equal(item.image[:, :2], original[:2].T)
    assert tuple(item.levels) == (0., 12.)
    monkeypatch.setattr("temsim.gui.scan_panel.perf_counter", lambda: 11.)
    view._playback_tick()
    np.testing.assert_array_equal(item.image, first_line)
    np.testing.assert_array_equal(frame.poisson_counts["bf"], original)


@pytest.mark.parametrize("quantity", ["ideal", "expected", "poisson"])
def test_angular_mask_still_hides_counts_outside_simulated_wave_domain(view, quantity):
    frame = count_frame()
    frame.metrics["detector_sampling"] = detector_sampling_report(
        {"bf": (0., 10.), "df": (16., 112.), "haadf": (60., 331.)},
        maximum_angle_mrad=42., wavelength_angstrom=.019687,
        requested_fov_angstrom=40., requested_grid_pixels=256,
        bandwidth_fraction=2 / 3, probe_semiangle_mrad=25.,
    )
    view._set_stem_frame(frame)
    choose(view, quantity)
    assert view.detector_image_items["haadf"].image is None
    assert "outside simulated angular coverage" in view.image_quantity_notice.text()
    assert view.detector_image_items["bf"].image is not None


def test_clear_removes_count_data_and_metadata_while_zero_sample_remains_a_real_zero(view):
    frame = count_frame()
    frame.poisson_counts["bf"].fill(0)
    view._set_stem_frame(frame)
    choose(view, "poisson")
    item = view.detector_image_items["bf"]
    assert np.count_nonzero(item.image) == 0
    assert tuple(item.levels) == (0., 5.5)
    assert "Unavailable" not in view.image_quantity_notice.text()
    view.display_result(None, complete=True)
    assert all(item.image is None for item in view.detector_image_items.values())
    assert "Unavailable: no stored frame" in view.image_quantity_notice.text()
    assert "Seed 7" not in view.image_quantity_notice.text()


@pytest.mark.parametrize("bad", [np.full((3, 4), .5), np.full((2, 2), 1), np.full((3, 4), np.nan)])
def test_invalid_stored_count_channel_is_unavailable_without_replacing_it(view, bad):
    frame = count_frame()
    frame.poisson_counts["bf"] = bad
    view._set_stem_frame(frame)
    choose(view, "poisson")
    assert view.detector_image_items["bf"].image is None
    assert "BF (invalid" in view.image_quantity_notice.text()
    np.testing.assert_array_equal(frame.poisson_counts["bf"], bad)
