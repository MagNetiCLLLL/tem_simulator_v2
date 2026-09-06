"""Presentation-only STEM source selection; synthetic cached frames, no solves."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.detector.stem_signal import DetectorSignal, StemScanResult
from temsim.gui.scan_panel import ScanControlView
from temsim.optics.column import default_state
from temsim.physics.stem_sampling import detector_sampling_report


def frame(value=0.0, *, origin_um=0.0, model="multislice_angle_resolved"):
    x, y = np.meshgrid(np.arange(4) * 0.001 + origin_um,
                       np.arange(3) * 0.001 + origin_um)
    images = {key: np.arange(12).reshape(3, 4) / 20 + value
              for key in ("haadf", "df", "bf")}
    return StemScanResult(
        scan_x_um=x, scan_y_um=y, fractions=images,
        detector_signals={key: DetectorSignal(key, key.upper(), float(values.mean()),
                                             1.0, 2.0, 3.0, None)
                          for key, values in images.items()},
        metrics={"model": model, "scan_pixel_size_nm": 1.0,
                 "scan_field_of_view_x_nm": 4.0, "scan_field_of_view_y_nm": 3.0,
                 "scan_frame_period_s": 10.0},
    )


def readout(stem, state=None, *, notes=()):
    return SimpleNamespace(stem=stem, state_snapshot=state, notes=notes,
                           coordinates={"objective.excitation": 68.5})


@pytest.fixture
def view(qtbot):
    widget = ScanControlView()
    qtbot.addWidget(widget)
    widget.set_state(default_state())
    yield widget
    widget._playback_timer.stop()


def choose_bank(view):
    view.image_source.setCurrentIndex(view.image_source.findData("bank"))


def choose_current(view):
    view.image_source.setCurrentIndex(view.image_source.findData("current"))


def assert_image(view, expected):
    for key, values in expected.fractions.items():
        np.testing.assert_array_equal(view.detector_image_items[key].image, values.T)


def test_selector_presents_bank_without_changing_current_state_or_signals(view):
    current, bank = frame(), frame(2.0)
    view._set_stem_frame(current)
    state, geometry = view._state, object()
    view._result = geometry
    signals = []
    view.parameters_changed.connect(lambda *_: signals.append("parameters"))
    view.playback_time_changed.connect(lambda *_: signals.append("time"))
    view.playback_active_changed.connect(lambda *_: signals.append("active"))
    view.set_bank_readout(readout(bank))
    assert view.image_source.objectName() == "stemImageSource"
    assert view.image_source.currentData() == "current"
    assert_image(view, current)
    choose_bank(view)
    assert_image(view, bank)
    assert view._state is state and view._result is geometry
    assert view._stem_frame is current and view._paused_display_frame is None
    assert not view.pause_image_refresh.isEnabled()
    choose_current(view)
    assert_image(view, current)
    assert signals == []


def test_new_main_frame_and_playback_do_not_replace_bank_images(view):
    current, newer, bank = frame(), frame(3.0), frame(8.0)
    view._set_stem_frame(current)
    view.pause_image_refresh.setChecked(True)
    view.set_bank_readout(readout(bank))
    choose_bank(view)
    view._set_stem_frame(newer)
    view._set_playback_active(True)
    emitted_times = []
    view.playback_time_changed.connect(emitted_times.append)
    view._playback_tick()
    assert emitted_times
    assert view._playback_timer.isActive()
    assert_image(view, bank)
    assert "Advanced bank" in view.detector_playback_summary.text()
    assert view._stem_frame is newer and view._paused_display_frame is current
    choose_current(view)
    assert_image(view, current)
    assert "paused" in view.detector_playback_summary.text()
    view._playback_timer.stop()
    view.pause_image_refresh.setChecked(False)
    assert_image(view, newer)


def test_pending_error_retains_previous_bank_but_missing_product_clears_only_bank(view):
    current, bank = frame(), frame(2.0)
    view._set_stem_frame(current)
    view.set_bank_readout(readout(bank))
    choose_bank(view)
    view.mark_bank_readout_pending("Requested node is outside the computed range")
    assert_image(view, bank)
    assert "previous frame retained" in view.detector_playback_summary.text()
    assert "outside the computed range" in view.detector_playback_summary.toolTip()
    view.set_bank_readout(readout(None, notes=("STEM unavailable: raw angular frames required",)))
    assert all(item.image is None for item in view.detector_image_items.values())
    assert "STEM unavailable" in view.detector_playback_summary.text()
    assert "raw angular frames" in view.image_model_notice.toolTip()
    assert view._stem_frame is current
    choose_current(view)
    assert_image(view, current)


def test_bank_uses_own_scan_coordinates_and_geometry_without_changing_zoom(view):
    current, bank = frame(), frame(2.0, origin_um=0.25)
    view._set_stem_frame(current)
    snapshot = default_state()
    snapshot.stem_detectors[0].z_mm = 1234.5
    view.set_bank_readout(readout(bank, snapshot))
    before = {}
    for key, plot in view.detector_image_views.items():
        plot.setRange(xRange=(-0.4, 0.4), yRange=(-0.3, 0.3), padding=0)
        before[key] = np.asarray(plot.viewRange())
    choose_bank(view)
    item = view.detector_image_items["haadf"]
    rectangle = item.mapRectToParent(item.boundingRect())
    assert rectangle.left() == pytest.approx(0.2495)
    assert rectangle.right() == pytest.approx(0.2535)
    assert rectangle.height() == pytest.approx(0.003)
    assert "1234.5" in view.detector_geometry_labels[snapshot.stem_detectors[0].key].text()
    assert "objective.excitation: 68.5" in view.detector_playback_summary.toolTip()
    for key, plot in view.detector_image_views.items():
        np.testing.assert_allclose(plot.viewRange(), before[key])
    choose_current(view)
    for key, plot in view.detector_image_views.items():
        np.testing.assert_allclose(plot.viewRange(), before[key])


def test_bank_sampling_diagnostics_never_apply_to_current_settings(view, monkeypatch):
    current, bank = frame(), frame(2.0)
    report = detector_sampling_report(
        {"bf": (0.0, 10.0), "df": (16.0, 112.0), "haadf": (60.0, 331.0)},
        maximum_angle_mrad=42.0, wavelength_angstrom=0.019687,
        requested_fov_angstrom=40.0, requested_grid_pixels=256,
        bandwidth_fraction=2 / 3, probe_semiangle_mrad=25.0,
    )
    bank = replace(bank, metrics={**bank.metrics, "detector_sampling": report})
    view._set_stem_frame(current)
    view.set_bank_readout(readout(bank, notes=(
        "STEM: angle-resolved intensity routing approximation, not coherent arbitrary-plane imaging.",
    )))
    choose_bank(view)
    assert "approximation" in view.image_model_notice.text()
    assert "Limited angular coverage" in view.image_model_notice.text()
    assert view.detector_image_items["haadf"].image is None
    assert "Not simulated" in view.detector_sampling_labels["haadf"].text()
    assert not view.match_detector_sampling.isEnabled()
    assert view.match_detector_sampling.isHidden()
    monkeypatch.setattr("temsim.calculation_cache.calculation_signatures",
                        lambda *_: pytest.fail("Bank selection must not inspect or mutate live physics"))
    view.match_detector_sampling.setEnabled(True)  # The slot itself must also guard the source.
    view._match_detector_sampling()
    choose_current(view)
    assert_image(view, current)


def test_current_clear_or_state_change_does_not_clear_bank_or_publish_bank_to_fourdstem(view, monkeypatch):
    current, bank = frame(), frame(2.0)
    view._set_stem_frame(current)
    fourdstem_updates = []
    monkeypatch.setattr(view, "_update_fourdstem_summary", fourdstem_updates.append)
    view.set_bank_readout(readout(bank))
    choose_bank(view)
    assert fourdstem_updates == []
    view.display_result(None, complete=True)
    assert_image(view, bank)
    assert fourdstem_updates == [None]
    view.set_state(default_state())
    assert_image(view, bank)
    assert "Bank detector geometry unavailable" in view.detector_geometry_labels["bf"].text()
    choose_current(view)
    assert all(item.image is None for item in view.detector_image_items.values())


def test_stale_current_frame_status_restored_after_bank_selection(view):
    current, bank = frame(), frame(2.0)
    view._set_stem_frame(current)
    view.set_bank_readout(readout(bank))
    choose_bank(view)
    notice = view.image_model_notice.text()
    view.mark_stem_frame_stale()
    assert_image(view, bank)
    assert view.image_model_notice.text() == notice
    choose_current(view)
    assert_image(view, current)
    assert "inputs changed" in view.image_model_notice.text()


def test_invalid_bank_frame_retains_previous_and_never_overwrites_main(view):
    current, bank = frame(), frame(2.0)
    view._set_stem_frame(current)
    view.set_bank_readout(readout(bank))
    choose_bank(view)
    invalid = replace(bank, scan_y_um=np.full((3, 4), np.nan))
    view.set_bank_readout(readout(invalid))
    assert_image(view, bank)
    assert "previous frame retained" in view.detector_playback_summary.text()
    assert "coordinates" in view.detector_playback_summary.toolTip()
    assert view._stem_frame is current


def test_switching_cached_sources_does_not_reload_cif_files(view, monkeypatch):
    current, bank = frame(), frame(2.0)
    view._set_stem_frame(current)
    view._state.sample.specimen_mode = "atomic"
    view._state.sample.cif_path = "current.cif"
    snapshot = default_state()
    snapshot.sample.specimen_mode = "atomic"
    snapshot.sample.cif_path = "captured.cif"
    view.set_bank_readout(readout(bank, snapshot))
    reads = []
    monkeypatch.setattr("ase.io.read", lambda *args, **kwargs: reads.append(args))
    choose_bank(view)
    choose_current(view)
    choose_bank(view)
    assert reads == []
