"""Retained receiver playback uses a display clock, without new transport."""
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.detector import receiver_scan
from temsim.detector.receiver_image import ReceiverImage
from temsim.gui import receiver_imaging as module


class _Raster:
    key = "test_scan"
    enabled = True
    scan_enabled = True
    scan_pixels_x = 4
    scan_lines = 2

    def __init__(self, period=1.):
        self.scan_frame_period_s = period

    def scan_factors(self, time):
        line = (float(time) % self.scan_frame_period_s) / self.scan_frame_period_s * self.scan_lines
        row = min(int(line), self.scan_lines-1)
        return 2*(line-row)-1, 2*(row+.5)/self.scan_lines-1


def _capture(*, period=1.):
    driver = _Raster(period)
    rows, columns = np.arange(2), np.arange(4)
    times = (rows[:, None] + (columns[None, :]+.5)/4.)/2.*period
    xy = np.asarray([driver.scan_factors(time) for time in times.flat]).reshape(2, 4, 2)*1000.
    plane = SimpleNamespace(key="camera", name="Camera", z_mm=3., inserted=True,
        outer_width_mm=16., geometry="square",
        hit_mask=lambda x, y: (np.abs(x) <= 8.) & (np.abs(y) <= 8.))
    state = SimpleNamespace(recording_planes=[plane], projector_mode="image",
        ac_deflector=driver, descan_deflector=None, simulation_time_s=0.)
    geometry = SimpleNamespace(times_s=times, plane_positions_um={"camera": (xy[..., 0], xy[..., 1])},
        ac_enabled=True, descan_enabled=False, requested_pixels_x=4, requested_pixels_y=2,
        unavailable_planes={}, plane_roles={"camera": "image"})
    return SimpleNamespace(state_snapshot=state, scan_geometry=geometry, workflow="receiver",
        signatures={"column": "column-one", "sample_downstream": "sample-one"})


def _image(key, exposure_s):
    values = np.zeros((16, 16))
    values[8, 8] = .25
    return ReceiverImage(key=key, name="Camera", status="AVAILABLE", detail="Synthetic captured hits",
        ideal_probability=values, response_probability=values, x_mm=np.arange(16)-7.5,
        y_mm=np.arange(16)-7.5, exposure_s=exposure_s, expected_electrons=values*100.,
        current_pa=1., metrics={"capture_request_identity": "original-capture"})


@pytest.fixture
def make_playback(qtbot, monkeypatch):
    clock = SimpleNamespace(now=0.)
    monkeypatch.setattr(module, "perf_counter", lambda: clock.now)
    contexts = []

    def create(*, visible=True, period=1.):
        samples, requests = [], []

        def sample(result, key, **kwargs):
            samples.append((result, key))
            return _image(key, kwargs.get("exposure_s"))

        monkeypatch.setattr(module, "receiver_image", sample)
        monkeypatch.setattr(receiver_scan, "receiver_image", sample)
        view = module.ReceiverImagingView()
        qtbot.addWidget(view)
        view.calculation_requested.connect(lambda: requests.append(True))
        capture = _capture(period=period)
        view.display_result(capture)
        view.resize(1000, 650)
        context = SimpleNamespace(view=view, clock=clock, capture=capture,
                                  samples=samples, requests=requests)
        contexts.append(context)
        if visible:
            view.show()
            view._render()
            assert view._displayed_image.status == "GEOMETRIC_PREVIEW"
        return context

    yield create
    for context in contexts:
        context.view.pause_scan()
        context.view.hide()


def _advance(context, elapsed):
    context.clock.now += elapsed
    context.view._playback_tick()


def _mode(view, mode):
    view.mode.setCurrentIndex(view.mode.findData(mode))
    view._render()


@pytest.mark.parametrize("loop", [False, True])
def test_local_playback_completes_one_frame_and_only_loops_when_selected(make_playback, loop):
    context = make_playback()
    view = context.view
    view.loop_scan.setChecked(loop)
    view.play_scan()
    assert view.mode.currentData() == "position"
    assert view._playback_timer.isActive()
    _advance(context, .4)
    assert view.position.value() == 2
    _advance(context, .6)
    assert view.position.value() == 7
    assert view._displayed_image.metrics["scan_index"] == 7
    assert view._playback_timer.isActive() is loop
    assert "complete" in view.playback_status.text().lower()
    _advance(context, .125)
    assert view.position.value() == (0 if loop else 7)
    assert context.requests == []
    assert context.samples == [(context.capture, "camera")]


def test_next_reset_and_manual_position_pause_playback(make_playback):
    context = make_playback()
    view = context.view
    view.play_scan()
    _advance(context, .4)
    view.step_button.click()
    assert view.position.value() == 3
    assert not view._playback_timer.isActive()
    assert view._displayed_image.metrics["scan_index"] == 3
    view.play_scan()
    view.reset_button.click()
    assert view.position.value() == 0
    assert not view._playback_timer.isActive()
    view.play_scan()
    view.position.setValue(5)
    view._render()
    assert not view._playback_timer.isActive()
    assert view._displayed_image.metrics["scan_index"] == 5
    view.position.setValue(7)
    view.step_button.click()
    assert view.position.value() == 0
    view.loop_scan.setChecked(False)
    view.position.setValue(7)
    view.step_button.click()
    assert view.position.value() == 7
    assert context.requests == []


@pytest.mark.parametrize("log_display", [False, True])
def test_frame_build_up_grows_to_full_exposure_with_fixed_contrast(make_playback, log_display):
    context = make_playback()
    view = context.view
    view.log_scale.setChecked(log_display)
    _mode(view, "accumulate")
    full = view._prepared_scan().preview().image
    expected_peak = np.log10(1.+1e4) if log_display else full.response_probability.max()
    levels = np.asarray(view.image.getImageItem().getLevels()).copy()
    np.testing.assert_allclose(levels, (0., expected_peak))
    assert view._displayed_image.metrics["exposure_fraction"] == .125
    assert view._displayed_image.response_probability.sum() == pytest.approx(.25/8.)
    assert "12.5%" in view.summary.text()
    view.loop_scan.setChecked(False)
    view.play_scan()
    _advance(context, .5)
    assert view.position.value() == 3
    assert view._displayed_image.metrics["exposure_fraction"] == .5
    assert view._displayed_image.response_probability.sum() == pytest.approx(.25/2.)
    np.testing.assert_array_equal(view.image.getImageItem().getLevels(), levels)
    _advance(context, .5)
    assert not view._playback_timer.isActive()
    assert view._displayed_image.metrics["exposure_fraction"] == 1.
    np.testing.assert_array_equal(view._displayed_image.response_probability, full.response_probability)
    np.testing.assert_array_equal(view.image.getImageItem().getLevels(), levels)
    assert view._displayed_image.expected_electrons is None
    assert view._displayed_image.current_pa is None
    assert view._displayed_image.status == "GEOMETRIC_PREVIEW"
    assert not view.quantity.isEnabled()
    assert not view.exposure.isEnabled()
    assert context.requests == []
    assert context.samples == [(context.capture, "camera")]


def test_pause_resume_preserves_position_and_ignores_paused_wall_time(make_playback):
    context = make_playback()
    view = context.view
    view.play_scan()
    _advance(context, .4)
    view.pause_scan()
    retained = view._displayed_image
    _advance(context, 20.)
    assert view.position.value() == 2
    assert view._displayed_image is retained
    assert not view._playback_timer.isActive()
    view.play_scan()
    assert view.position.value() == 2
    _advance(context, .2)
    assert view.position.value() == 3
    assert view._playback_timer.isActive()


def test_fast_captured_frame_is_slowed_for_visible_build_up(make_playback):
    context = make_playback(period=.004)
    view = context.view
    _mode(view, "accumulate")
    view.loop_scan.setChecked(False)
    view.play_scan()
    _advance(context, .1)
    assert view._playback_timer.isActive()
    assert view.position.value() == 1
    assert view._displayed_image.metrics["exposure_fraction"] == .25
    assert "slowed for display" in view.playback_status.text()
    _advance(context, .1)
    assert view.position.value() == 3
    assert view._displayed_image.metrics["exposure_fraction"] == .5
    _advance(context, .21)
    assert view.position.value() == 7
    assert view._displayed_image.metrics["exposure_fraction"] == 1.
    assert not view._playback_timer.isActive()
    assert context.requests == []


def test_hidden_start_defers_all_image_work_until_visible(make_playback):
    context = make_playback(visible=False)
    view = context.view
    view.play_scan()
    _advance(context, 10.)
    assert context.samples == []
    assert view._displayed_image is None
    assert not view._timer.isActive()
    assert not view._playback_timer.isActive()
    view.show()
    assert view._playback_timer.isActive()
    assert view._displayed_image.status == "GEOMETRIC_PREVIEW"
    assert view.position.value() == 0
    assert context.samples == [(context.capture, "camera")]
    assert context.requests == []


def test_hide_stops_timers_and_preview_work_then_resumes_same_position(make_playback, monkeypatch):
    context = make_playback()
    view = context.view
    calls = []
    original_preview = receiver_scan.PreparedReceiverScan.preview

    def preview(prepared, *args, **kwargs):
        calls.append((args, kwargs))
        return original_preview(prepared, *args, **kwargs)

    monkeypatch.setattr(receiver_scan.PreparedReceiverScan, "preview", preview)
    view.play_scan()
    _advance(context, .4)
    view.hide()
    before = len(calls)
    retained = view._displayed_image
    _advance(context, 50.)
    assert not view._playback_timer.isActive()
    assert not view._timer.isActive()
    assert view._displayed_image is retained
    assert len(calls) == before
    view.show()
    assert view._playback_timer.isActive()
    assert view.position.value() == 2
    _advance(context, .2)
    assert view.position.value() == 3
    assert context.samples == [(context.capture, "camera")]


def test_changed_inputs_stop_playback_and_stale_capture_cannot_restart(make_playback):
    context = make_playback()
    view = context.view
    view.play_scan()
    _advance(context, .4)
    view.mark_result_stale()
    view._render()
    assert not view._playback_timer.isActive()
    assert not view.play_button.isEnabled()
    assert not view.step_button.isEnabled()
    assert "inputs changed" in view.summary.text()
    view.play_scan()
    view.follow_scan_started(context.capture, True)
    view.follow_scan_time(context.capture, .95)
    _advance(context, 2.)
    assert not view._playback_timer.isActive()
    assert not view._is_following()
    assert view.position.value() == 2
    assert context.requests == []


def test_preparation_error_stops_playback_without_requesting_transport(make_playback, monkeypatch):
    context = make_playback(visible=False)
    view = context.view

    def unavailable(*args, **kwargs):
        raise ValueError("Captured receiver axes disagree")

    monkeypatch.setattr(module, "prepare_receiver_scan", unavailable)
    view.show()
    view._render()
    view.play_scan()
    assert view._displayed_image is None
    assert not view._playback_timer.isActive()
    assert "unavailable" in view.summary.text().lower()
    assert "No usable scan reception" in view.playback_status.text()
    assert context.samples == []
    assert context.requests == []


@pytest.mark.parametrize("mismatch", ["geometry", "signature"])
def test_follow_uses_matching_stem_clock_and_ignores_unrelated_sources(make_playback, mismatch):
    context = make_playback()
    view = context.view
    view.speed.setCurrentIndex(view.speed.findData(.01))
    view.follow_scan_started(context.capture, True)
    view.follow_scan_time(context.capture, .55)
    view._render()
    assert view._is_following()
    assert not view._playback_timer.isActive()
    assert not view.speed.isEnabled()
    assert view.position.value() == 3
    assert view._displayed_image.metrics["scan_index"] == 3
    unrelated = _capture()
    if mismatch == "signature":
        unrelated.scan_geometry = context.capture.scan_geometry
        unrelated.signatures["sample_downstream"] = "different-sample"
    view.follow_scan_started(unrelated, True)
    view.follow_scan_time(unrelated, .99)
    view.follow_scan_started(unrelated, False)
    assert view._is_following()
    assert view.position.value() == 3
    assert view._follow_source is context.capture
    assert context.requests == []


def test_follow_accepts_shared_geometry_provenance_and_can_pause_then_rejoin(make_playback):
    context = make_playback()
    view = context.view
    source = SimpleNamespace(scan_geometry=context.capture.scan_geometry,
                             signatures=dict(context.capture.signatures))
    view.follow_scan_started(source, True)
    view.follow_scan_time(source, .4)
    view._render()
    assert view._is_following()
    assert view.position.value() == 2
    view.pause_scan()
    view.follow_scan_time(source, .8)
    assert view.position.value() == 2
    assert not view._is_following()
    view.play_scan()
    view._render()
    assert view._is_following()
    assert not view._playback_timer.isActive()
    assert view.position.value() == 5
    assert context.requests == []


def test_follow_while_hidden_records_latest_clock_without_rendering(make_playback, monkeypatch):
    context = make_playback()
    view = context.view
    _mode(view, "accumulate")
    view.follow_scan_started(context.capture, True)
    view.follow_scan_time(context.capture, .4)
    view._render()
    view.hide()
    retained = view._displayed_image
    calls = []
    original_preview = receiver_scan.PreparedReceiverScan.preview

    def preview(prepared, *args, **kwargs):
        calls.append((args, kwargs))
        return original_preview(prepared, *args, **kwargs)

    monkeypatch.setattr(receiver_scan.PreparedReceiverScan, "preview", preview)
    view.follow_scan_time(context.capture, .9)
    assert view.position.value() == 6
    assert view._displayed_image is retained
    assert calls == []
    assert not view._timer.isActive()
    assert not view._playback_timer.isActive()
    view.show()
    view._render()
    assert view._displayed_image.metrics["scan_index"] == 6
    assert view._displayed_image.metrics["exposure_fraction"] == .875
    assert view._is_following()
    assert not view._playback_timer.isActive()
    assert context.samples == [(context.capture, "camera")]


@pytest.mark.parametrize("loop", [False, True])
def test_stem_completion_only_continues_local_preview_when_loop_selected(make_playback, loop):
    context = make_playback()
    view = context.view
    _mode(view, "accumulate")
    view.loop_scan.setChecked(loop)
    view.follow_scan_started(context.capture, True)
    view.follow_scan_time(context.capture, .999)
    view._render()
    assert view.position.value() == 7
    completed = view._displayed_image
    view.follow_scan_started(context.capture, False)
    assert not view._follow_active
    assert not view._is_following()
    assert view._playback_timer.isActive() is loop
    assert view.position.value() == 7
    assert view._displayed_image is completed
    assert view._displayed_image.metrics["exposure_fraction"] == 1.
    _advance(context, .1)
    assert view.position.value() == (0 if loop else 7)
    _advance(context, .4)
    assert view.position.value() == (2 if loop else 7)
    assert not view._follow_active
    assert context.requests == []
    assert context.samples == [(context.capture, "camera")]
