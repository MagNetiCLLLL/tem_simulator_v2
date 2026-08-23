import numpy as np
import pytest

from temsim.detector.recording_system import (
    restore_recording_system,
    serialise_recording_system,
)
from temsim.detector.stem_detector import create_stem_detectors
from temsim.detector.stem_signal import StemScanResult, _readout_view
from temsim.optics.column import default_state


def test_detector_insertion_readout_and_centre_offset_round_trip_independently():
    state = default_state()
    detector = state.haadf_detector
    detector.inserted = True
    detector.readout_enabled = False
    detector.centre_offset_x_mm = 0.125
    detector.centre_offset_y_mm = -0.25

    payload = serialise_recording_system(state)
    restored = default_state()
    restore_recording_system(restored, payload)
    result = restored.haadf_detector

    assert result.inserted is True
    assert result.readout_enabled is False
    assert result.centre_offset_x_mm == pytest.approx(0.125)
    assert result.centre_offset_y_mm == pytest.approx(-0.25)


def test_camera_length_has_no_detector_selector_in_serialised_state():
    state = default_state()
    payload = state.to_dict()

    assert "stem_diffraction_target_detector_key" not in payload


def test_legacy_detector_payload_keeps_old_inserted_equals_readout_semantics():
    state = default_state()
    restore_recording_system(state, [{"key": "haadf", "inserted": False}])

    assert state.haadf_detector.inserted is False
    assert state.haadf_detector.readout_enabled is False


def test_detector_centre_offset_moves_the_physical_hit_mask():
    detector = create_stem_detectors()[0]
    detector.centre_offset_x_mm = 1.0
    detector.centre_offset_y_mm = -2.0
    detector.inner_diameter_mm = 0.0
    detector.geometry = "disk"
    detector.outer_width_mm = 2.0

    assert detector.hit_mask(np.asarray([1.0]), np.asarray([-2.0]))[0]
    assert not detector.hit_mask(np.asarray([0.0]), np.asarray([0.0]))[0]


def test_readout_filter_preserves_unread_detector_interception_accounting():
    shape = (2, 2)
    result = StemScanResult(
        scan_x_um=np.zeros(shape),
        scan_y_um=np.zeros(shape),
        fractions={
            "haadf": np.full(shape, 0.2),
            "df": np.full(shape, 0.3),
        },
        detector_signals={"haadf": object(), "df": object()},
        metrics={"physical_detector_masks": True},
        current_pa={"haadf": np.ones(shape), "df": np.ones(shape)},
        uncollected_fraction=np.full(shape, 0.5),
    )

    filtered = _readout_view(
        result,
        inserted_keys=("haadf", "df"),
        readout_keys=("df",),
    )

    assert set(filtered.fractions) == {"df"}
    assert set(filtered.detector_signals) == {"df"}
    assert filtered.uncollected_fraction == pytest.approx(
        result.uncollected_fraction
    )
    assert filtered.metrics["unreadout_intercepted_mean_fraction"] == {
        "haadf": pytest.approx(0.2)
    }
