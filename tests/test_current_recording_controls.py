"""Current detector controls cannot be silently migrated or partially installed."""
from copy import deepcopy
import pytest
from temsim.optics.column import default_state
from temsim.detector.recording_system import restore_recording_system, serialise_recording_system


@pytest.mark.parametrize("change", ["duplicate", "missing", "old_key", "old_anchor", "missing_readout", "string_bool", "fractional_pixels", "nonfinite_offset"])
def test_invalid_records_leave_current_detectors_unchanged(change):
    state = default_state()
    records = serialise_recording_system(state)
    before = deepcopy(records)
    if change == "duplicate":
        records.append(deepcopy(records[0]))
    elif change == "missing":
        records.pop()
    elif change == "old_key":
        records[0]["key"] = "HAADF"
    elif change == "old_anchor":
        records[-1]["anchor_key"] = "old_image_reference"
    elif change == "missing_readout":
        records[0].pop("readout_enabled")
    elif change == "fractional_pixels":
        records[-1]["pixels"] = 10.5
    elif change == "nonfinite_offset":
        records[0]["centre_offset_x_mm"] = float("nan")
    else:
        records[0]["inserted"] = "false"
    objects = tuple(state.recording_planes)
    with pytest.raises(ValueError):
        restore_recording_system(state, records)
    assert serialise_recording_system(state) == before
    assert all(a is b for a, b in zip(objects, state.recording_planes))


def test_recording_roundtrip_preserves_insertion_independently_from_readout():
    state = default_state()
    state.recording_planes[0].inserted = True
    state.recording_planes[0].readout_enabled = False
    state.camera.pixels = 256
    records = serialise_recording_system(state)
    fresh = default_state()
    restore_recording_system(fresh, records)
    assert serialise_recording_system(fresh) == records
    assert fresh.camera is fresh.recording_planes[-1]
