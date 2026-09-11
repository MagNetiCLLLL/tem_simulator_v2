"""Transaction/permission tests; synthetic candidates do not certify optics."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.alignment_transaction import AlignmentRequest, AlignmentCandidate, AlignmentCommitGate, AlignmentCancelled, solve_alignment_candidate
from temsim.assembly_catalog import AssemblyCatalog
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state
from temsim.optics.direct_alignment import apply_direct_alignment, DirectAlignmentResult
from temsim.operating_modes import direct_alignment_by_key
from temsim.working_point import WorkingPointCheckpoint


@pytest.fixture
def state():
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, catalog.default_selection())
    return state


def candidate_for(state, *, revision=0):
    request = AlignmentRequest.capture(state, "nanoprobe_convergence", 25., revision=revision)
    new = request.start_snapshot.restore()
    keys = direct_alignment_by_key(request.key).devices
    strengths = {}
    for lens in new.lenses:
        if lens.key in keys:
            lens.percent += .001
            strengths[lens.key] = lens.percent
    snapshot = capture_instrument_snapshot(new)
    cp = WorkingPointCheckpoint(snapshot, {}, state.sample.z_mm, "fixture", {"validation_status": "NOT_RUN"})
    result = DirectAlignmentResult(key=request.key, success=True, requested=request.target,
        achieved=24.99, unit="mrad", constraint_value=0., constraint_unit="mm", strengths=strengths,
        iterations=1, validation_step_mm=.1, numerical_spread=0., message="Synthetic transaction fixture")
    return AlignmentCandidate(request, result, cp,
        {"status": "PASS", "candidate_snapshot": snapshot.to_dict(), "forward_snapshot_id": snapshot.digest}, "READY_TO_APPLY")


def test_t227_rejects_unregistered_or_modified_catalog_definition(state):
    with pytest.raises(ValueError):
        AlignmentRequest.capture(state, "alpha99", 30., revision=0)
    definition = direct_alignment_by_key("nanoprobe_convergence")
    with pytest.raises(ValueError, match="registered"):
        apply_direct_alignment(state, definition.key, 25., definition=replace(definition, maximum=500.))
    with pytest.raises(ValueError, match="range"):
        AlignmentRequest.capture(state, definition.key, 1000., revision=0)


def test_t231_32_atomic_apply_idempotence_undo_and_aba_revision(state):
    initial = capture_instrument_snapshot(state)
    candidate = candidate_for(state, revision=3)
    gate = AlignmentCommitGate()
    # A -> B -> A leaves the same numerical hash, but is still stale.
    state.lenses[0].percent += 1
    state.lenses[0].percent -= 1
    with pytest.raises(ValueError, match="STALE"):
        gate.apply(state, candidate, revision=5)
    changed = gate.apply(state, candidate, revision=3)
    assert capture_instrument_snapshot(state).digest == initial.digest
    assert capture_instrument_snapshot(changed).digest != initial.digest
    assert gate.apply(changed, candidate, revision=4) is changed
    assert capture_instrument_snapshot(gate.undo()).digest == initial.digest
    assert candidate.request.start_snapshot.digest == initial.digest


def test_t229_30_failed_or_tampered_candidate_never_applies(state):
    initial = capture_instrument_snapshot(state).digest
    candidate = candidate_for(state)
    gate = AlignmentCommitGate()
    with pytest.raises(ValueError, match="complete"):
        gate.apply(state, replace(candidate, status="FAILED"), revision=0)
    with pytest.raises(TypeError):
        candidate.result.strengths["objective_lens"] = 10.
    candidate = replace(candidate, result=replace(candidate.result,
        strengths={**candidate.result.strengths, "objective_lens": 10.}))
    with pytest.raises(ValueError, match="exact registered"):
        gate.apply(state, candidate, revision=0)
    assert capture_instrument_snapshot(state).digest == initial


def test_display_plane_does_not_stale_alignment_or_get_reset(state):
    state.virtual_observation_z_mm = 1200.
    candidate = candidate_for(state, revision=3)
    initial = candidate.request.start_snapshot
    state.virtual_observation_z_mm = 1300.
    current = capture_instrument_snapshot(state)
    assert current.digest != initial.digest
    assert current.physical_digest == initial.physical_digest
    gate = AlignmentCommitGate()
    changed = gate.apply(state, candidate, revision=3)
    assert changed.virtual_observation_z_mm == 1300.
    changed.virtual_observation_z_mm = 1400.
    assert gate.apply(changed, candidate, revision=4) is changed
    assert candidate.request.start_snapshot.digest == initial.digest
    restored = gate.undo()
    assert restored.virtual_observation_z_mm == 1300.


def test_unknown_new_parameters_remain_physical_for_alignment(state):
    before = capture_instrument_snapshot(state)
    state.new_physics_control = 1.
    assert capture_instrument_snapshot(state).physical_digest != before.physical_digest


@pytest.mark.parametrize("equivalent", [False, True])
def test_projector_search_keeps_the_captured_field_model(state, equivalent):
    from temsim.optics.direct_alignment import (
        _ProjectorMeasurementModel, _EquivalentImageFirstOrderModel, _LiveFirstOrderModel)
    state.projector_mode = "image"
    state.equivalent_image_lenses_enabled = equivalent
    before = capture_instrument_snapshot(state).digest
    model = _ProjectorMeasurementModel(state, direct_alignment_by_key("image_magnification"), step_mm=.2)
    expected = _EquivalentImageFirstOrderModel if equivalent else _LiveFirstOrderModel
    assert isinstance(model.sample_model, expected)
    assert state.equivalent_image_lenses_enabled is equivalent
    assert capture_instrument_snapshot(state).digest == before


def test_t231_cancel_before_solving_never_runs_optimizer(state, monkeypatch):
    request = AlignmentRequest.capture(state, "nanoprobe_convergence", 25., revision=0)
    def forbidden(*args, **kwargs):
        raise AssertionError("Cancelled requests must not start computation")
    monkeypatch.setattr("temsim.optics.direct_alignment._solve_direct_alignment", forbidden)
    with pytest.raises(AlignmentCancelled):
        solve_alignment_candidate(request, cancelled=lambda: True)


def test_forward_and_observable_backends_are_not_conflated(state, monkeypatch):
    # Orchestration-only fixture: neither the GPU nor a projector calculation
    # runs here. A CPU observable must not be reported as a GPU validation.
    from temsim.optics.direct_alignment import DirectAlignmentMeasurement
    state.projector_mode = "imaging"
    request = AlignmentRequest.capture(state, "image_magnification", 100., revision=0)
    keys = direct_alignment_by_key(request.key).devices
    strengths = {lens.key: lens.percent for lens in state.lenses if lens.key in keys}
    solved = DirectAlignmentResult(request.key, True, 100., 100., "x", 0., "um",
        strengths, 1, .1, 0., "Synthetic backend-reporting fixture")
    monkeypatch.setattr("temsim.optics.direct_alignment._solve_direct_alignment", lambda *args, **kw: solved)
    def forward(working, **kwargs):
        working.active_backend = "FAKE_GPU"
        return object()
    monkeypatch.setattr("temsim.physics.simulation.run", forward)
    monkeypatch.setattr("temsim.optics.direct_alignment._validate_projector_production",
        lambda *args: DirectAlignmentMeasurement(request.key, 100., "x", 0., "um",
            execution_backend="CPU", relay_error_um=0.))
    def checkpoint(cls, result, **kwargs):
        return cls(result.calculation_manifest.instrument_snapshot, {}, state.sample.z_mm,
            "fixture", {"validation_status": "NOT_RUN"})
    monkeypatch.setattr(WorkingPointCheckpoint, "from_result", classmethod(checkpoint))
    candidate = solve_alignment_candidate(request)
    assert candidate.validation["forward_particle_backend"] == "FAKE_GPU"
    assert candidate.validation["actual_backend"] == "FAKE_GPU"
    assert candidate.validation["observable_validation_backend"] == "CPU"


def test_alignment_cancel_button_and_generation_discard_late_result(qtbot):
    from temsim.gui.direct_alignment_panel import DirectAlignmentPanel
    from temsim.gui.direct_alignment_controller import DirectAlignmentController
    panel = DirectAlignmentPanel()
    qtbot.addWidget(panel)
    controller = DirectAlignmentController()
    received = []
    controller.result_ready.connect(lambda *args: received.append(args))
    panel.cancellation_requested.connect(controller.invalidate_pending)
    panel.set_busy("nanoprobe_convergence")
    assert panel.cancel_button.isEnabled()
    old_generation = controller._generation
    panel.cancel_button.click()
    assert controller._cancellation.is_set()
    controller._accept_result(old_generation, "nanoprobe_convergence", object(), 1.)
    assert received == []
    panel.set_busy(None)
    assert not panel.cancel_button.isEnabled()


def test_t234_working_point_browser_never_reads_active_state(qtbot, state):
    from temsim.gui.working_point_panel import WorkingPointPanel
    cp = WorkingPointCheckpoint(capture_instrument_snapshot(state), {}, state.sample.z_mm,
                                "fixture", {"source_representation": "historical fixture"})
    panel = WorkingPointPanel()
    qtbot.addWidget(panel)
    def forbidden():
        raise AssertionError("Browsing a checkpoint must not evaluate the current state")
    panel.current_snapshot = forbidden
    panel.add_checkpoint(cp)
    assert panel.selected.digest == cp.digest
    assert panel.tree.topLevelItemCount() == len(cp.snapshot.graph["nodes"])
    assert "read-only" in panel.status.text()
    received = []
    panel.restore_requested.connect(lambda checkpoint, fork: received.append((checkpoint, fork)))
    panel._restore(False)
    assert received == [(cp, False)]


def test_t208_gui_restore_no_preset_and_no_recalculation(qtbot, monkeypatch):
    from temsim.gui.main_window import MainWindow
    window = MainWindow()
    qtbot.addWidget(window)
    window.preview_timer.stop()
    snapshot = capture_instrument_snapshot(window.state)
    cp = WorkingPointCheckpoint(snapshot, {}, window.state.sample.z_mm, "fixture", {})
    window.state.lenses[0].percent += 1
    errors = []
    monkeypatch.setattr(window, "_show_error", errors.append)
    window._restore_working_point(cp)
    assert not errors
    assert capture_instrument_snapshot(window.state).digest == snapshot.digest
    assert window._active_working_checkpoint is cp
    assert not window.preview_timer.isActive()
