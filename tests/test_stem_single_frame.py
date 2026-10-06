"""Deterministic single-frame presentation, without computing particle images."""
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.detector.stem_signal import StemScanResult
from temsim.gui.scan_panel import ScanControlView
from temsim.physics.scan_geometry import ScanGeometryResult


def cached_result(period=10., offset=0.):
    x, y = np.meshgrid(np.arange(4)*.001, np.arange(3)*.001)
    fractions = {key: (np.arange(12).reshape(3, 4)+offset)/100.
                 for key in ("haadf", "df", "bf")}
    frame = StemScanResult(scan_x_um=x, scan_y_um=y, fractions=fractions,
        detector_signals={}, metrics={"model": "geometric_detector_interception",
            "scan_frame_period_s": period, "scan_pixel_size_nm": 1.,
            "scan_field_of_view_x_nm": 4., "scan_field_of_view_y_nm": 3.})
    geometry = ScanGeometryResult(times_s=(np.arange(12).reshape(3, 4)+.5)*period/12,
        sample_x_um=x, sample_y_um=y, plane_positions_um={}, plane_names={},
        requested_pixels_x=4, requested_pixels_y=3, ac_enabled=True, descan_enabled=False,
        ac_drift_pivot_z_mm=None, descan_drift_pivot_z_mm=None,
        ac_lower_from_upper=-np.eye(2), ac_angular_residual=0.)
    return geometry, frame


@pytest.fixture
def playback(qtbot, monkeypatch):
    view = ScanControlView()
    qtbot.addWidget(view)
    view._state = SimpleNamespace(ac_deflector=SimpleNamespace(
        enabled=True, scan_enabled=True, scan_frame_period_s=10., scan_pixels_x=4, scan_lines=3))
    clock = [0.]
    monkeypatch.setattr("temsim.gui.scan_panel.perf_counter", lambda: clock[0])
    active, times, changes = [], [], []
    view.playback_active_changed.connect(active.append)
    view.playback_time_changed.connect(times.append)
    view.parameters_changed.connect(changes.append)
    yield view, clock, active, times, changes
    view._playback_timer.stop()


def assert_complete(view, frame):
    for key, values in frame.fractions.items():
        np.testing.assert_array_equal(view.detector_image_items[key].image, values.T)


@pytest.mark.parametrize("elapsed", [10., 35.])
def test_frame_boundary_stops_once_and_queued_ticks_cannot_clear_the_image(playback, elapsed):
    view, clock, active, times, changes = playback
    geometry, frame = cached_result()
    view.display_result(geometry, frame)
    assert view._playback_timer.isActive()
    clock[0] = elapsed
    view._playback_tick()
    assert not view._playback_timer.isActive()
    assert active == [True, False]
    assert_complete(view, frame)
    assert times[-1] == pytest.approx(10.*11.5/12.)
    assert times[-1] < 10.  # Do not wrap the physical raster back to pixel zero.
    count = len(times)
    for now in (elapsed+.1, elapsed+10., elapsed+25.):
        clock[0] = now
        view._playback_tick()
        assert_complete(view, frame)
    assert active == [True, False] and len(times) == count
    assert view._state.ac_deflector.scan_enabled and changes == []


def test_fast_frame_completes_immediately_without_background_timer(playback):
    view, _, active, times, _ = playback
    period = view._playback_timer.interval()*1e-3
    geometry, frame = cached_result(period)
    view.display_result(geometry, frame)
    assert not view._playback_timer.isActive()
    assert_complete(view, frame)
    assert active.count(False) == 1
    assert times[-1] == pytest.approx(period*11.5/12.)
    view._playback_tick()
    assert active.count(False) == 1


def test_republishing_same_frame_does_not_replay_but_manual_calculation_does(playback):
    view, clock, active, _, _ = playback
    geometry, frame = cached_result()
    view.display_result(geometry, frame, complete=True)
    clock[0] = 10.
    view._playback_tick()
    view.display_result(geometry, frame, complete=True)
    assert not view._playback_timer.isActive()
    assert active == [True, False]
    assert_complete(view, frame)
    view.display_result(geometry, frame, complete=True, explicit_calculation=True)
    assert view._playback_timer.isActive()
    assert active == [True, False, True]
    clock[0] = 20.
    view._playback_tick()
    assert active == [True, False, True, False]
    assert_complete(view, frame)
    geometry, new_frame = cached_result(offset=20.)
    view.display_result(geometry, new_frame, complete=True)
    assert view._playback_timer.isActive()
    clock[0] = 30.
    view._playback_tick()
    assert_complete(view, new_frame)
    assert active == [True, False, True, False, True, False]


def test_display_controls_and_bank_switches_do_not_restart_completed_frame(playback):
    view, clock, active, times, changes = playback
    geometry, current = cached_result()
    view.display_result(geometry, current)
    clock[0] = 10.
    view._playback_tick()
    _, bank = cached_result(offset=50.)
    view.set_bank_readout(SimpleNamespace(stem=bank))
    for source, expected in (("bank", bank), ("current", current)):
        view.image_source.setCurrentIndex(view.image_source.findData(source))
        assert_complete(view, expected)
        assert not view._playback_timer.isActive()
    view.image_display_quantity.setCurrentIndex(view.image_display_quantity.findData("expected"))
    view.image_display_quantity.setCurrentIndex(view.image_display_quantity.findData("ideal"))
    assert_complete(view, current)
    assert active == [True, False]
    assert changes == []


def test_clearing_a_result_stops_playback_and_queued_tick_is_safe(playback):
    view, clock, active, times, _ = playback
    view.display_result(*cached_result())
    view.display_result(None, complete=True)
    assert not view._playback_timer.isActive()
    assert active == [True, False]
    assert all(item.image is None for item in view.detector_image_items.values())
    before = len(times)
    clock[0] = 100.
    view._playback_tick()
    assert len(times) == before and active == [True, False]


def test_arming_raster_and_images_requests_one_frame_and_manual_button_can_request_another(playback, monkeypatch):
    from temsim.optics.ac_deflector import create_ac_deflector
    from temsim.optics.descan_deflector import create_descan_deflector
    view, _, _, _, _ = playback
    ac, descan = create_ac_deflector(), create_descan_deflector()
    ac.scan_enabled = False
    view._state = SimpleNamespace(ac_deflector=ac, descan_deflector=descan,
                                  sample=SimpleNamespace(stem_image_enabled=False))
    # The test observes UI requests only; field calibration is covered elsewhere.
    monkeypatch.setattr("temsim.gui.scan_panel.calibrate_scan_system", lambda _s: None)
    requests = []
    view.calculation_requested.connect(lambda: requests.append("one frame"))
    view._control_changed("ac", "scan_enabled", True)
    assert requests == []  # Image output is still disabled.
    view._image_readout_changed(True)
    assert requests == ["one frame"]
    view._control_changed("ac", "scan_pixel_size_nm", ac.scan_pixel_size_nm*2)
    assert requests == ["one frame"]
    assert view.calculate_button.text() == "Calculate STEM (single frame)"
    view.calculate_button.click()
    assert requests == ["one frame", "one frame"]
    view._updating = True
    view._image_readout_changed(True)
    view._updating = False
    assert requests == ["one frame", "one frame"]
    view._image_readout_changed(False)
    assert requests == ["one frame", "one frame"]
