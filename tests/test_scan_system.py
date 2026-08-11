from types import SimpleNamespace

import numpy as np
import pytest

from temsim.detector.stem_signal import DetectorSignal, StemScanResult
from temsim.gui.scan_panel import ScanControlView
from temsim.optics.ac_deflector import create_ac_deflector
from temsim.physics import scan_geometry
from temsim.physics.scan_geometry import ScanGeometryResult


def test_ac_lower_foil_is_derived_from_one_pure_shift_pair_gain():
    component = create_ac_deflector()
    component.wobble_enabled = False
    component.set_pure_shift_coupling(
        ((-1.25, 0.1), (-0.1, -1.25)),
        1.0e-12,
    )
    component.upper_coil_gain = 0.4
    component.kick_x_mrad = 2.0
    component.kick_y_mrad = -3.0

    assert component.lower_coil_gain == pytest.approx(-0.5)
    upper, lower = component.coil_kicks_mrad(2.0, -3.0)
    assert upper == pytest.approx((0.8, -1.2))
    assert lower == pytest.approx((-1.12, 1.42))
    events = component.kick_events()
    assert events[0][1:] == pytest.approx((0.8e-3, -1.2e-3))
    assert events[1][1:] == pytest.approx((-1.12e-3, 1.42e-3))


def test_ac_pure_shift_calibration_cancels_sample_angle(monkeypatch):
    component = create_ac_deflector()
    component.wobble_enabled = False
    component.scan_enabled = True
    upper_angle = np.array(((2.0, 0.4), (-0.4, 2.0)))
    lower_angle = np.array(((1.5, -0.2), (0.2, 1.5)))

    def phase_response(_state, start_z_mm, _observation_z_mm):
        angle = (
            upper_angle
            if float(start_z_mm) == pytest.approx(component.upper_z_mm)
            else lower_angle
        )
        return np.zeros((2, 2)), angle

    monkeypatch.setattr(
        scan_geometry,
        "transverse_kick_phase_space_response",
        phase_response,
    )
    state = SimpleNamespace(
        ac_deflector=component,
        sample=SimpleNamespace(z_mm=component.lower_z_mm + 10.0),
    )

    coupling, residual = scan_geometry.calibrate_ac_pure_shift(state)

    assert upper_angle + lower_angle @ coupling == pytest.approx(
        np.zeros((2, 2)),
        abs=1.0e-12,
    )
    assert residual < 1.0e-12


def test_paired_response_uses_both_planes_and_cross_axis_coupling(monkeypatch):
    component = SimpleNamespace(
        upper_z_mm=1.0,
        lower_z_mm=2.0,
        coil_kick_matrices=lambda: (
            ((0.5, 0.0), (0.0, 0.5)),
            ((-0.6, 0.2), (-0.2, -0.6)),
        ),
    )
    upper_response = np.array(((2.0, 0.0), (0.0, 3.0)))
    lower_response = np.array(((5.0, 1.0), (-1.0, 4.0)))

    monkeypatch.setattr(
        scan_geometry,
        "transverse_kick_response",
        lambda _state, start, _stop: (
            upper_response if float(start) == 1.0 else lower_response
        ),
    )

    actual = scan_geometry.paired_kick_response(None, component, 3.0)
    expected = (
        upper_response @ np.asarray(component.coil_kick_matrices()[0])
        + lower_response @ np.asarray(component.coil_kick_matrices()[1])
    )
    assert actual == pytest.approx(expected)


def test_scan_view_replays_one_cached_detector_frame_until_stopped(qtbot):
    view = ScanControlView()
    qtbot.addWidget(view)
    view._state = SimpleNamespace(
        ac_deflector=SimpleNamespace(
            enabled=True,
            scan_enabled=True,
            scan_frame_period_s=0.2,
        )
    )
    images = {
        key: np.arange(12, dtype=float).reshape(3, 4) / 12.0
        for key in ("haadf", "df", "bf")
    }
    signals = {
        key: DetectorSignal(key, key.upper(), 0.25, 1.0, 2.0, 3.0, None)
        for key in images
    }
    frame = StemScanResult(
        scan_x_um=np.zeros((3, 4)),
        scan_y_um=np.zeros((3, 4)),
        fractions=images,
        detector_signals=signals,
        metrics={"model": "test", "scan_frame_period_s": 0.2},
    )
    geometry = ScanGeometryResult(
        times_s=np.zeros((3, 4)),
        sample_x_um=np.zeros((3, 4)),
        sample_y_um=np.zeros((3, 4)),
        plane_positions_um={},
        plane_names={},
        requested_pixels_x=4,
        requested_pixels_y=3,
        ac_enabled=True,
        descan_enabled=False,
        ac_drift_pivot_z_mm=None,
        descan_drift_pivot_z_mm=None,
        ac_lower_from_upper=-np.eye(2),
        ac_angular_residual=0.0,
    )

    view.display_result(geometry, frame)

    assert view._playback_timer.isActive()
    assert "Scanning continuously" in view.detector_playback_summary.text()
    view._state.ac_deflector.scan_enabled = False
    view._set_playback_active(False)
    assert not view._playback_timer.isActive()
    assert "last frame retained" in view.detector_playback_summary.text()
