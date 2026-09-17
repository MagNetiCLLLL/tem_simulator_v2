"""Exact operating-only patch boundaries, stale prevention and GUI rollback."""
from dataclasses import replace

import pytest

from temsim.alignment_transaction import AlignmentCommitGate
from temsim.illumination_apply import prepare_illumination_patch
from temsim.instrument_snapshot import capture_instrument_snapshot, decode_instrument, encode_instrument
from temsim.optics.column import default_state
from temsim.runtime_parameters import runtime_targets
from temsim.working_point import WorkingPointCheckpoint, snapshot_changes


def donor_point(state):
    donor = decode_instrument(encode_instrument(state))
    runtime_targets(donor)["condenser_lens_1"].obj.percent = 36.12345678901234
    runtime_targets(donor)["condenser_aperture_2"].obj.radius_mm = .043210987654321
    # Donor downstream settings must never be copied by illumination Apply.
    donor.sample.thickness_nm += 2.345
    donor.step_mm *= 2
    return WorkingPointCheckpoint(capture_instrument_snapshot(donor), {}, donor.sample.z_mm,
        "input-candidate", {"package_kind": "INSTRUMENT_INPUTS_ONLY"})


def test_apply_exact_controls_keeps_complete_non_target_state_and_undo():
    state = default_state()
    donor = donor_point(state)
    before = capture_instrument_snapshot(state)
    patch = prepare_illumination_patch(state, donor, revision=7)
    assert len(patch.controls) == 2
    assert capture_instrument_snapshot(state).digest == before.digest
    gate = AlignmentCommitGate()
    updated = gate.apply_illumination(state, patch, revision=7, previous_checkpoint=donor)
    assert snapshot_changes(before, capture_instrument_snapshot(updated)) == patch.exact_changes
    assert len(patch.exact_changes) == 2
    assert updated.sample.thickness_nm == state.sample.thickness_nm
    assert updated.step_mm == state.step_mm
    assert encode_instrument(updated.electron_gun) == encode_instrument(state.electron_gun)
    restored, checkpoint = gate.peek_undo()
    assert capture_instrument_snapshot(restored).digest == before.digest
    assert checkpoint is donor


@pytest.mark.parametrize("mismatch", ["source", "mode", "insertion", "geometry", "implementation"])
def test_incompatible_candidate_rejected_without_mutation(mismatch):
    state = default_state()
    before = capture_instrument_snapshot(state)
    donor = donor_point(state).snapshot.restore()
    if mismatch == "source":
        donor.electron_gun.emitter.emission_current_na *= 2
    elif mismatch == "mode":
        donor.illumination_mode = "TEM"
    elif mismatch == "insertion":
        runtime_targets(donor)["condenser_lens_1"].obj.enabled = False
    elif mismatch == "geometry":
        runtime_targets(donor)["condenser_lens_1"].obj.a_mm *= 1.2
    snapshot = capture_instrument_snapshot(donor)
    if mismatch == "implementation":
        snapshot = replace(snapshot, implementation="historical")
    point = WorkingPointCheckpoint(snapshot, {}, donor.sample.z_mm, "mismatch", {})
    with pytest.raises(ValueError, match="Incompatible|implementation changed"):
        prepare_illumination_patch(state, point, revision=0)
    assert capture_instrument_snapshot(state).digest == before.digest


def test_stale_and_aba_apply_rejected_before_undo_history_changes():
    state = default_state()
    patch = prepare_illumination_patch(state, donor_point(state), revision=2)
    gate = AlignmentCommitGate()
    with pytest.raises(ValueError, match="STALE"):
        gate.apply_illumination(state, patch, revision=4)  # A -> B -> A.
    state.sample.thickness_nm += 1
    with pytest.raises(ValueError, match="STALE"):
        gate.apply_illumination(state, patch, revision=2)
    with pytest.raises(ValueError, match="No applied"):
        gate.peek_undo()


# Reuse the isolated main-window fixture and its explicit worker disposal.
from test_working_point_restore_gui import window


def test_gui_install_failure_preserves_results_state_and_undo(window, monkeypatch):
    state = window.state
    point = donor_point(state)
    window._active_working_checkpoint = point
    before = capture_instrument_snapshot(state)
    selection = window.selection
    revision = window._physical_revision
    ray_result = window.workspace._last_result
    high_result = window.workspace._high_accuracy_result
    patch = prepare_illumination_patch(state, point, revision=revision)
    original = window._sync_working_point_selectors

    def injected_failure():
        original()
        if window.state is not state:
            raise ValueError("Injected installation failure")

    monkeypatch.setattr(window, "_sync_working_point_selectors", injected_failure)
    errors = []
    monkeypatch.setattr(window, "_show_error", errors.append)
    window._apply_illumination_patch(patch)
    assert errors and "Injected installation failure" in errors[0]
    assert window.state is state and window.selection == selection
    assert capture_instrument_snapshot(window.state).digest == before.digest
    assert window._physical_revision == revision
    assert window._active_working_checkpoint is point
    assert window.workspace._last_result is ray_result
    assert window.workspace._high_accuracy_result is high_result
    with pytest.raises(ValueError, match="No applied"):
        window._alignment_commits.peek_undo()


def test_gui_apply_is_input_transaction_and_undo_restores_checkpoint(window, monkeypatch):
    point = donor_point(window.state)
    window._active_working_checkpoint = point
    before = capture_instrument_snapshot(window.state)
    patch = prepare_illumination_patch(window.state, point, revision=window._physical_revision)
    errors = []
    monkeypatch.setattr(window, "_show_error", errors.append)
    window._apply_illumination_patch(patch)
    assert not errors
    assert capture_instrument_snapshot(window.state).digest == patch.after.digest
    assert window._active_working_checkpoint is None
    assert not window.preview_timer.isActive()
    window._undo_alignment()
    assert not errors
    assert capture_instrument_snapshot(window.state).digest == before.digest
    assert window._active_working_checkpoint is point
