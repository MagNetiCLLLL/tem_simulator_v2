"""Runtime camera edits and profiles must preserve clonable current state."""
from copy import deepcopy

import pytest

from temsim.gui.parameter_panel import ParameterPanel
from temsim.optics.column import default_state
from temsim.profile_io import apply_profile_values
from temsim.runtime_parameters import runtime_targets, validate_runtime_assignment


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_runtime_rejects_invalid_pixel_count_without_mutation(value):
    state = default_state()
    target = runtime_targets(state)["camera"]
    before = state.camera.pixels
    with pytest.raises(ValueError, match="positive|integer"):
        validate_runtime_assignment(target, "pixels", value)
    assert state.camera.pixels == before


@pytest.mark.parametrize("value", [0, -1])
def test_invalid_profile_pixels_leave_all_current_controls_unchanged(value):
    state = default_state()
    before = deepcopy(state.to_dict())
    with pytest.raises(ValueError, match="pixels.*positive"):
        apply_profile_values(state, {"camera": {"inserted": True, "pixels": value}})
    assert state.to_dict() == before
    assert type(state).from_dict(state.to_dict()).camera.pixels == before["recording_planes"][-1]["pixels"]


@pytest.mark.parametrize("text", ["0", "-1"])
def test_invalid_gui_pixels_restore_text_without_publishing_change(qtbot, text):
    state = default_state()
    before = state.camera.pixels
    panel = ParameterPanel()
    qtbot.addWidget(panel)
    panel.set_context("Camera", runtime_targets(state)["camera"], None, (), None)
    errors, changes = [], []
    panel.error.connect(errors.append)
    panel.runtime_changed.connect(changes.append)
    row = next(i for i in range(panel.runtime_table.rowCount())
               if panel.runtime_table.item(i, 0).text() == "pixels")
    item = panel.runtime_table.item(row, 1)
    item.setText(text)
    assert errors and "positive" in errors[-1]
    assert not changes
    assert item.text() == str(before)
    assert state.camera.pixels == before
    assert type(state).from_dict(state.to_dict()).camera.pixels == before


@pytest.mark.parametrize("pixels", [1, 256])
def test_valid_pixel_counts_survive_profile_and_complete_snapshot(pixels):
    state = default_state()
    apply_profile_values(state, {"camera": {"pixels": pixels}})
    assert state.camera.pixels == pixels
    assert type(state).from_dict(state.to_dict()).camera.pixels == pixels
