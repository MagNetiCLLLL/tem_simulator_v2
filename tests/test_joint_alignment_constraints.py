"""Constraint/transaction fixtures are not physical alignment qualification."""
from dataclasses import replace
import numpy as np
import pytest

from temsim.alignment_constraints import BeamConstraint, ConstrainedAlignment, evaluate_incident
from temsim.alignment_transaction import AlignmentRequest, AlignmentCommitGate, solve_alignment_candidate
from temsim.assembly_catalog import AssemblyCatalog
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state


@pytest.fixture
def state():
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, catalog.default_selection())
    return state


def options(*constraints, **kwargs):
    return ConstrainedAlignment(tuple(constraints),
        {"condenser_lens_2": (0., 150.), "condenser_lens_3": (0., 150.)}, maximum_evaluations=8, **kwargs)


def response_fixture(monkeypatch, state, *, topology_match=True):
    initial = {lens.key: lens.percent for lens in state.lenses}
    def measure(scratch, **kwargs):
        controls = {lens.key: lens.percent for lens in scratch.lenses}
        metrics = dict(alpha95_mrad=25. + .1 * (controls["condenser_lens_2"] - initial["condenser_lens_2"]),
            alpha99_mrad=26., diameter95_um=.5, current_pa=10., centroid_x_um=0., centroid_y_um=0.,
            waist_offset_mm=1.e-4 * (controls["condenser_lens_3"] - initial["condenser_lens_3"]),
            curvature_per_m=0., effective_samples=32.)
        arrays = dict(x_m=np.zeros(32), y_m=np.zeros(32), tx_rad=np.zeros(32), ty_rad=np.zeros(32),
                      alive=np.ones(32, bool), weight=np.full(32, 1./32))
        summary = dict(plane_z_mm=scratch.sample.upper_surface_z_mm, source_current_a=1.e-11)
        topo = dict(reference_match=topology_match, status="SYNTHETIC_TEST") if kwargs.get("topology") else None
        return metrics, arrays, summary, topo
    monkeypatch.setattr("temsim.alignment_constraints.evaluate_incident", measure)


@pytest.mark.parametrize("constraint", [BeamConstraint("diameter95_um", maximum=.1),
                                       BeamConstraint("current_pa", minimum=20.)])
def test_angle_pass_does_not_override_spot_or_current_failure(state, monkeypatch, constraint):
    response_fixture(monkeypatch, state)
    before = capture_instrument_snapshot(state).digest
    request = AlignmentRequest.capture(state, "nanoprobe_convergence", 25., revision=0, options=options(constraint))
    candidate = solve_alignment_candidate(request)
    assert candidate.result.achieved == pytest.approx(25.)
    assert not candidate.result.success
    assert candidate.validation["status"] == "CONSTRAINT_VIOLATED"
    with pytest.raises(ValueError, match="complete"):
        AlignmentCommitGate().apply(state, candidate, revision=0)
    assert capture_instrument_snapshot(state).digest == before


def test_joint_pass_captures_source_and_controls_and_supports_undo(state, monkeypatch):
    response_fixture(monkeypatch, state)
    before = capture_instrument_snapshot(state).digest
    request = AlignmentRequest.capture(state, "nanoprobe_convergence", 25., revision=0,
                                      options=options(BeamConstraint("current_pa", minimum=5.)))
    candidate = solve_alignment_candidate(request)
    assert candidate.result.success
    assert candidate.validation["control_authority"]["rank"] == 2
    assert {r["axis"] for r in candidate.validation["refinements"]} == {"configured", "column_step", "gun_step"}
    assert candidate.result.iterations <= 8
    assert capture_instrument_snapshot(state).digest == before
    gate = AlignmentCommitGate()
    restored = gate.apply(state, candidate, revision=0)
    def emitter_graph(value):
        return next(node for node in capture_instrument_snapshot(value).graph["nodes"]
                    if node["type"].endswith(":ColdFieldEmitter"))
    assert emitter_graph(restored) == emitter_graph(state)
    assert capture_instrument_snapshot(gate.undo()).digest == before
    damaged = replace(candidate, validation={**candidate.validation, "options_id": "changed"})
    with pytest.raises(ValueError, match="constraints changed"):
        AlignmentCommitGate().apply(state, damaged, revision=0)


def test_topology_mismatch_is_a_constraint_failure(state, monkeypatch):
    response_fixture(monkeypatch, state, topology_match=False)
    request = AlignmentRequest.capture(state, "nanoprobe_convergence", 25., revision=0, options=options())
    # Synthetic topology response isolates acceptance; it does not install a reference.
    request = replace(request, options=replace(request.options, check_topology=True))
    candidate = solve_alignment_candidate(request)
    assert not candidate.result.success
    assert candidate.validation["status"] == "CONSTRAINT_VIOLATED"


def test_real_tip_incident_observation_is_exact_and_source_is_unchanged(state):
    state.electron_gun.emitter.ray_count = 9
    state.step_mm = 1.
    before = capture_instrument_snapshot(state)
    metrics, arrays, summary, topology = evaluate_incident(state)
    assert arrays["x_m"].dtype == np.float64
    assert arrays["weight"].shape == (9,)
    assert summary["plane_z_mm"] == state.sample.upper_surface_z_mm
    assert topology is None
    assert before.physical_digest == capture_instrument_snapshot(state).physical_digest


def test_joint_editor_captures_explicit_bounds_and_does_not_edit_state(qtbot, state):
    from temsim.gui.direct_alignment_panel import DirectAlignmentPanel
    panel = DirectAlignmentPanel()
    qtbot.addWidget(panel)
    panel.set_state(state)
    before = capture_instrument_snapshot(state).digest
    editor = panel._joint_editors["nanoprobe_convergence"]
    assert panel.constraint_options("nanoprobe_convergence") is None
    editor.enabled.setChecked(True)
    editor.add_constraint()
    editor.constraints.item(0, 2).setText(".5")
    request_options = panel.constraint_options("nanoprobe_convergence")
    assert request_options.constraints[0].maximum == .5
    assert request_options.maximum_evaluations == 32
    assert capture_instrument_snapshot(state).digest == before
