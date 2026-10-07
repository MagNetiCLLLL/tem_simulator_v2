"""Receiver clocks follow the captured STEM frame without acquiring another."""

from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from temsim.detector.stem_signal import StemScanResult
from temsim.gui.scan_panel import ScanControlView
from temsim.gui.visualization import VisualizationWorkspace
from temsim.physics.scan_geometry import ScanGeometryResult


def captured_result(state, offset=0.0):
    x, y = np.meshgrid(np.arange(4) * 0.001, np.arange(3) * 0.001)
    frame = StemScanResult(
        scan_x_um=x, scan_y_um=y,
        fractions={key: (np.arange(12).reshape(3, 4) + offset) / 100.0
                   for key in ("haadf", "df", "bf")},
        detector_signals={},
        metrics={"model": "geometric_detector_interception", "scan_frame_period_s": 10.0},
    )
    geometry = ScanGeometryResult(
        times_s=(np.arange(12).reshape(3, 4) + 0.5) * 10.0 / 12,
        sample_x_um=x, sample_y_um=y, plane_positions_um={}, plane_names={},
        requested_pixels_x=4, requested_pixels_y=3, ac_enabled=True, descan_enabled=False,
        ac_drift_pivot_z_mm=None, descan_drift_pivot_z_mm=None,
        ac_lower_from_upper=-np.eye(2), ac_angular_residual=0.0,
    )
    return SimpleNamespace(stem_scan=frame, scan_geometry=geometry, state_snapshot=state)


@pytest.fixture
def linked_scan(qtbot, monkeypatch):
    scan = ScanControlView()
    qtbot.addWidget(scan)
    scan._state = SimpleNamespace(ac_deflector=SimpleNamespace(
        enabled=True, scan_enabled=True, scan_frame_period_s=10.0,
        scan_pixels_x=4, scan_lines=3))
    clock = [0.0]
    monkeypatch.setattr("temsim.gui.scan_panel.perf_counter", lambda: clock[0])
    receiver = Mock()
    view = SimpleNamespace(
        scan_control=scan, receiver_imaging=receiver,
        _last_result=None, _receiver_scan_source=None,
        _receiver_scan_pending_source=None, _receiver_scan_active=False,
        _receiver_scan_time_s=None,
    )
    for name in ("_publish_scan_control_result", "_receiver_scan_active_changed",
                 "_receiver_scan_time_changed", "_sync_receiver_scan_playback"):
        setattr(view, name, MethodType(getattr(VisualizationWorkspace, name), view))
    # PySide slots weak-reference bound-method owners; this lightweight
    # namespace is not a QObject or weak-referenceable instance.
    scan.playback_active_changed.connect(lambda active: view._receiver_scan_active_changed(active))
    scan.playback_time_changed.connect(lambda time_s: view._receiver_scan_time_changed(time_s))
    requests = []
    scan.calculation_requested.connect(lambda: requests.append("calculate"))
    scan.parameters_changed.connect(lambda *_args: requests.append("change"))
    yield view, clock, requests
    scan._playback_timer.stop()


def test_follow_clock_uses_stem_source_and_preserves_single_frame_completion(linked_scan):
    view, clock, requests = linked_scan
    result = captured_result(view.scan_control._state)
    view._publish_scan_control_result(result, complete=True)
    view.receiver_imaging.follow_scan_started.assert_called_once_with(result, True)
    view.receiver_imaging.follow_scan_time.assert_called_once_with(result, 0.0)

    # A different page can publish a new workspace result during the old scan.
    view._last_result = object()
    clock[0] = 10.0
    view.scan_control._playback_tick()
    assert not view.scan_control._playback_timer.isActive()
    view.receiver_imaging.follow_scan_started.assert_called_with(result, False)
    view.receiver_imaging.follow_scan_time.assert_called_with(result, 10.0 * 11.5 / 12)
    for key, values in result.stem_scan.fractions.items():
        np.testing.assert_array_equal(view.scan_control.detector_image_items[key].image, values.T)
    count = len(view.receiver_imaging.mock_calls)
    clock[0] = 100.0
    view.scan_control._playback_tick()
    view._sync_receiver_scan_playback()
    assert len(view.receiver_imaging.mock_calls) == count
    assert requests == []


@pytest.mark.parametrize("reuse_frame", (False, True))
def test_replacement_stop_uses_old_source_before_new_start(linked_scan, reuse_frame):
    view, clock, requests = linked_scan
    old = captured_result(view.scan_control._state)
    view._publish_scan_control_result(old, complete=True)
    new = (SimpleNamespace(**vars(old)) if reuse_frame
           else captured_result(view.scan_control._state, offset=25.0))
    clock[0] = 3.0
    view.receiver_imaging.reset_mock()

    view._publish_scan_control_result(new, complete=True, explicit_calculation=reuse_frame)

    calls = view.receiver_imaging.mock_calls
    assert calls[0].args == (old, False)
    assert calls[1].args == (new, True)
    assert calls[2].args == (new, 0.0)
    assert view._receiver_scan_source is new
    assert view._receiver_scan_pending_source is None
    assert requests == []


def test_cached_frame_keeps_its_source_and_catches_up_new_receiver_without_restarting(linked_scan):
    view, clock, requests = linked_scan
    original = captured_result(view.scan_control._state)
    view._publish_scan_control_result(original, complete=True)
    clock[0] = 4.0
    view.scan_control._playback_tick()
    started = view.scan_control._playback_started_s
    cached = SimpleNamespace(**vars(original))
    view._last_result = cached
    view.receiver_imaging.reset_mock()

    view._sync_receiver_scan_playback()
    view._publish_scan_control_result(cached, complete=True)

    view.receiver_imaging.follow_scan_started.assert_called_once_with(original, True)
    view.receiver_imaging.follow_scan_time.assert_called_once_with(original, 4.0)
    assert view._receiver_scan_source is original
    assert view.scan_control._playback_started_s == started
    assert view.scan_control._playback_timer.isActive()
    assert requests == []


def test_unassociated_frame_and_cleared_source_do_not_send_old_clock(linked_scan):
    view, _clock, _requests = linked_scan
    result = captured_result(view.scan_control._state)
    view._publish_scan_control_result(result, complete=True)
    view.receiver_imaging.reset_mock()
    original_frame = view.scan_control._stem_frame
    view.scan_control._stem_frame = object()
    view._receiver_scan_time_changed(5.0)
    view._sync_receiver_scan_playback()
    assert view.receiver_imaging.mock_calls == []

    view.scan_control._stem_frame = original_frame
    view._publish_scan_control_result(None, complete=True)
    view.receiver_imaging.follow_scan_started.assert_called_once_with(result, False)
    assert view._receiver_scan_source is None
    assert view._receiver_scan_time_s is None
    view._receiver_scan_time_changed(6.0)
    view.receiver_imaging.follow_scan_time.assert_not_called()
