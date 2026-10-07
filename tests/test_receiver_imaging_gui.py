"""Camera reception UI keeps capture provenance and never runs a scan loop."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.gui import receiver_imaging as module
from temsim.detector import receiver_scan
from temsim.detector.receiver_image import ReceiverImage


def _capture():
    planes = [SimpleNamespace(key=key, name=name, z_mm=z, inserted=True)
              for key, name, z in (("flu_screen", "Fluorescent Screen", 2.), ("camera", "Camera", 3.))]
    state = SimpleNamespace(recording_planes=planes, projector_mode="image")
    return SimpleNamespace(state_snapshot=state, scan_geometry=None)


def _image(key="camera", *, zero=False, status="AVAILABLE", exposure_s=1.):
    values = np.zeros((16, 16))
    if not zero:
        values[8, 9] = .25
    return ReceiverImage(key, key, status, "Captured physical hits", values, values,
        np.linspace(-1., 1., 16), np.linspace(-1., 1., 16), exposure_s,
        None if status != "AVAILABLE" else values * (exposure_s or 1.) * 100.,
        (0. if zero else 1.) if status == "AVAILABLE" else None,
        {"capture_request_identity": "original-capture"})


@pytest.fixture
def page(qtbot, monkeypatch):
    view = module.ReceiverImagingView()
    qtbot.addWidget(view)
    calls = []
    def sampled(result, key, **kw):
        calls.append((result, key, kw))
        return _image(key, exposure_s=kw.get("exposure_s"))
    monkeypatch.setattr(module, "receiver_image", sampled)
    monkeypatch.setattr(receiver_scan, "receiver_image", sampled)
    view._test_calls = calls
    return view


def test_opening_and_selecting_receiver_never_requests_transport(page, qtbot):
    capture = _capture()
    requested = []
    page.calculation_requested.connect(lambda: requested.append(True))
    page.set_state(capture.state_snapshot)
    page.resize(1000, 600)
    page.show()
    qtbot.wait(100)
    assert page._test_calls == []
    assert page._displayed_image is None
    assert "No reception calculated" in page.notice.text()
    page.receiver.setCurrentIndex(1)
    qtbot.wait(100)
    assert not requested and not page._test_calls
    page.calculate_button.click()
    assert requested == [True]


def test_hidden_result_is_lazy_switch_preserves_snapshot_and_stale_status(page, qtbot):
    capture = _capture()
    page.display_result(capture)
    assert not page._test_calls
    page.show()
    qtbot.waitUntil(lambda: page._displayed_image is not None)
    page.receiver.setCurrentIndex(1)
    qtbot.waitUntil(lambda: page._displayed_image.key == "camera")
    assert all(row[0] is capture for row in page._test_calls)
    live = _capture().state_snapshot
    live.recording_planes.clear()
    page.set_state(live)
    page.mark_result_stale()
    qtbot.waitUntil(lambda: "inputs changed" in page.summary.text())
    assert page.receiver.count() == 2
    assert all(plane.inserted for plane in capture.state_snapshot.recording_planes)
    assert "Inputs changed" in page.calculation_bar.status.text()
    page.display_result(None)
    qtbot.waitUntil(lambda: page._displayed_image is None)
    assert not page.export_button.isEnabled()


def test_real_zero_and_unavailable_are_distinct(page, monkeypatch):
    capture = _capture()
    monkeypatch.setattr(module, "receiver_image", lambda *a, **kw: _image(zero=True))
    page.display_result(capture)
    page._render()
    assert page._displayed_image is not None
    assert "No electrons received" in page.summary.text()
    assert page.export_button.isEnabled()
    assert len(page.sensor_outline.getData()[0]) > 0
    monkeypatch.setattr(module, "receiver_image", lambda *a, **kw: _image(status="NOT_REACHED"))
    page.display_result(capture)
    page._render()
    assert page._displayed_image is None
    assert "NOT REACHED" in page.summary.text()
    assert not page.export_button.isEnabled()
    assert len(page.sensor_outline.getData()[0]) == 0


def test_manual_scan_is_explicit_preview_and_export_has_no_dose(page, monkeypatch, tmp_path, qtbot):
    capture = _capture()
    times = np.array([[0., .25], [.5, .75]])
    capture.scan_geometry = SimpleNamespace(times_s=times)
    calls = []
    def scan(index=None, *, accumulate=False):
        key = page.receiver.currentData()
        calls.append(index)
        image = _image(key, status="GEOMETRIC_PREVIEW")
        return SimpleNamespace(key=key, image=image, status="GEOMETRIC_PREVIEW", detail="Preview only",
            times_s=times, displacement_x_mm=times, displacement_y_mm=times*0,
            trajectory_x_mm=times, trajectory_y_mm=times*0)
    prepared = SimpleNamespace(preview=scan, times_s=times,
        base_image=_image(), metrics={"frame_period_s": 1.})
    monkeypatch.setattr(module, "prepare_receiver_scan", lambda *a, **kw: prepared)
    page.display_result(capture)
    page.resize(1000, 600)
    page.show()
    qtbot.waitUntil(lambda: page._displayed_image is not None)
    assert page.mode.currentData() == "frame"
    assert calls == [None]
    qtbot.wait(150)
    assert calls == [None] and not page._timer.isActive()
    assert not page.exposure.isEnabled() and not page.quantity.isEnabled()
    assert "Geometry preview" in page.notice.text()
    page.mode.setCurrentIndex(2)
    page.position.setValue(3)
    qtbot.waitUntil(lambda: calls[-1] == 3)
    assert "4/4" in page.position_text.text()
    assert page.marker.getData()[0].size == 1
    path = tmp_path / "preview.npz"
    page.export_result(path)
    with np.load(path, allow_pickle=False) as data:
        assert "expected_electrons" not in data
        assert np.array_equal(data["times_s"], times)
        assert json.loads(str(data["metadata_json"]))["status"] == "GEOMETRIC_PREVIEW"


def test_scan_preview_never_keeps_an_electron_count_display_label(page, monkeypatch):
    page.quantity.setCurrentIndex(1)
    page.mode.setCurrentIndex(1)
    page._render()
    assert page.quantity.currentData() == "probability"
    assert not page.quantity.isEnabled()


def test_static_export_keeps_raw_values_after_log_display(page, tmp_path):
    page.display_result(_capture())
    page.log_scale.setChecked(True)
    page._render()
    path = tmp_path / "physical.npz"
    page.export_result(path)
    with np.load(path, allow_pickle=False) as data:
        assert data["response_probability"].sum() == .25
        assert data["expected_electrons"].sum() == 25.
        assert json.loads(str(data["metadata_json"]))["capture_request_identity"] == "original-capture"
    assert page.plot.getViewBox().state["yInverted"] is False


def test_later_optical_preview_keeps_completed_physical_receiver(page):
    capture = _capture()
    capture.workflow = "receiver"
    page.display_result(capture)
    page._render()
    expected = page._displayed_image
    optical = _capture()
    optical.workflow = "rays"
    optical.simulation = SimpleNamespace(metrics={"optical_tuning": True})
    page.display_result(optical)
    page._render()
    assert page._result is capture
    assert page._displayed_image is expected
    assert page._stale and "Previous calculation" in page.summary.text()


def test_small_screen_leaves_room_for_receiver_image(page, qtbot, qapp):
    from temsim.app import APPLICATION_STYLE
    old = qapp.styleSheet()
    qapp.setStyleSheet(APPLICATION_STYLE)
    try:
        page.display_result(_capture())
        page.resize(1000, 550)
        page.show()
        qtbot.waitUntil(lambda: page._displayed_image is not None)
        assert page.width() == 1000 and page.height() == 550
        assert page.image.height() >= 250
    finally:
        qapp.setStyleSheet(old)


def test_workspace_routes_receiver_calculation_without_wave_request(qtbot):
    from temsim.gui.visualization import VisualizationWorkspace
    workspace = VisualizationWorkspace()
    qtbot.addWidget(workspace)
    requests = []
    workspace.calculation_requested.connect(requests.append)
    assert workspace.illuminating_page.widget(0) is workspace.receiver_imaging
    assert workspace.illuminating_page.widget(2) is workspace.wave_imaging
    workspace.receiver_imaging.calculate_button.click()
    assert requests == ["receiver"]
    assert workspace.selected_plane_readout.shutdown()
    assert workspace.conjugate_planes.shutdown()
