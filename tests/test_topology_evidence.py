"""Scoped design references and executed, bounded convergence integration."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from temsim.immutable_json import thaw_json
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state
from temsim.topology_evidence import topology_reference, topology_run, compare_topology
from temsim.working_point import WorkingPointCheckpoint


@pytest.fixture
def state():
    result = default_state()
    result.electron_gun.emitter.ray_count = 9
    result.step_mm = 1.
    return result


def test_micro_and_nano_references_are_distinct_read_only_targets(state, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Reference browsing must not restore or calculate")
    snapshot = capture_instrument_snapshot(state)
    monkeypatch.setattr(type(snapshot), "restore", forbidden)
    nano = topology_reference(snapshot)
    assert nano["status"] == "TARGET_ONLY" and nano["reference"]["count"] == 5
    state.illumination_mode = "TEM"
    micro = topology_reference(capture_instrument_snapshot(state))
    assert micro["reference"]["count"] == 6
    assert micro["reference"]["intervals"] != nano["reference"]["intervals"]
    assert nano["qualification"] == micro["qualification"] == "NOT_ESTABLISHED"
    # The packaged historical target agrees with the identified source record.
    path = Path(nano["reference"]["source"])
    content = path.read_bytes()
    assert sha256(content).hexdigest() == nano["reference"]["source_sha256"]
    original = json.loads(content)["reports"]["Flat"]
    from temsim.optics.beam_path_audit import crossover_intervals
    expected = crossover_intervals([r["z_mm"] for r in original["column_crossovers"]], original["component_planes"])
    assert thaw_json(nano["reference"]["intervals"]) == thaw_json(expected)


def test_reference_does_not_transfer_to_other_assembly(state):
    assembly = state._resolved_assembly
    paths = tuple((key, "column/C2.toml" if key == "column" else value) for key, value in assembly.selected_module_paths)
    state._resolved_assembly = replace(assembly, selected_module_paths=paths)
    # Capture labels directly: this detached test graph is not a physical assembly.
    from temsim.instrument_snapshot import InstrumentSnapshot, encode_instrument
    result = topology_reference(InstrumentSnapshot(encode_instrument(state)))
    assert result["status"] == "NO_SCOPED_REFERENCE"


def test_topology_comparison_preserves_order_multiplicity_and_terminal_root():
    components = [("C1", 0.), ("C2", 1.), ("sample", 3.)]
    roots = [dict(z_mm=z, status="EXECUTED_AT_FIXED_SAMPLING") for z in (.5, 2., 3.)]
    reference = dict(status="TARGET_ONLY", reference=dict(count=2,
        intervals=[["between", ["C1"], ["C2"]], ["between", ["C2"], ["sample"]]]))
    left = topology_run(roots, components, surface_mm=3., focus_tolerance_nm=1., reference=reference)
    assert left["intermediate_count"] == 2 and len(left["roots"]) == 3
    assert len(left["terminal_roots"]) == 1 and left["reference_match"]
    assert compare_topology(left, left)["status"] == "MATCH_FOR_CHECKED_AXIS"
    extra = topology_run([roots[0], dict(z_mm=.7), *roots[1:]], components,
        surface_mm=3., focus_tolerance_nm=1., reference=reference)
    assert not extra["reference_match"]
    assert compare_topology(left, extra)["status"] == "UNRESOLVED"
    failed = dict(left, status="UNRESOLVED")
    assert compare_topology(left, failed)["status"] == "UNRESOLVED"


def test_topology_budget_rejects_before_physics(state, monkeypatch):
    from temsim.sampling_convergence import ConvergenceRequest, run_convergence
    from temsim.optics import beam_path_audit
    point = WorkingPointCheckpoint(capture_instrument_snapshot(state), {}, state.sample.z_mm, "inputs", {})
    request = ConvergenceRequest(point, "column_step", check_topology=True,
        checkpoint_spacing_mm=.01, maximum_checkpoint_bytes=1024**2)
    monkeypatch.setattr(beam_path_audit, "incident_checkpoints", lambda *_a, **_k: pytest.fail("Budget must reject before tracing"))
    with pytest.raises(ValueError, match="budget"):
        run_convergence(request)


def test_bracket_axis_executes_actual_tip_chain_without_changing_transport(state):
    from temsim.sampling_convergence import ConvergenceRequest, run_convergence
    point = WorkingPointCheckpoint(capture_instrument_snapshot(state), {}, state.sample.z_mm, "inputs", {})
    report = run_convergence(ConvergenceRequest(point, "checkpoint_spacing", 9,
        check_topology=True, checkpoint_spacing_mm=40.))
    left, right = report["runs"]
    assert left["snapshot_id"] == right["snapshot_id"]
    assert left["numerical_identity"] != right["numerical_identity"]
    assert left["topology"]["bracket_spacing_mm"] == 40.
    assert right["topology"]["bracket_spacing_mm"] == 20.
    assert right["topology"]["checkpoint_count"] > left["topology"]["checkpoint_count"]
    assert right["input_changes"] == ()
    assert report["comparison"]["status"] == "UNRESOLVED"  # Nine emitted samples cannot meet the support screen.
    assert report["physical_validation"] == "NOT_ESTABLISHED"


def test_sampling_ui_reference_and_explicit_topology_settings(qtbot, state):
    from temsim.gui.sampling_panel import SamplingPanel
    panel = SamplingPanel()
    qtbot.addWidget(panel)
    point = WorkingPointCheckpoint(capture_instrument_snapshot(state), {}, state.sample.z_mm, "inputs", {})
    panel.set_checkpoint(point)
    assert "intermediate count 5" in panel.reference.text()
    assert not panel.topology.isChecked()
    assert panel.checkpoint_budget.maximum() == 32768
    assert panel.axis.findData("checkpoint_spacing") >= 0


def test_joint_step_check_requires_matching_separate_evidence_and_executes(state):
    from temsim.sampling_convergence import ConvergenceRequest, run_convergence
    point = WorkingPointCheckpoint(capture_instrument_snapshot(state), {}, state.sample.z_mm, "inputs", {})
    with pytest.raises(ValueError, match="separate"):
        ConvergenceRequest(point, "joint_steps", 9).prepare()
    separate = tuple(run_convergence(ConvergenceRequest(point, axis, 9)) for axis in ("gun_step", "column_step"))
    report = run_convergence(ConvergenceRequest(point, "joint_steps", 9, prior_evidence=separate))
    assert report["comparison_kind"] == "JOINT_TRANSPORT_STEPS"
    assert set(report["prior_evidence_ids"]) == {row["digest"] for row in separate}
    assert len(report["runs"][1]["input_changes"]) == 2
    assert report["comparison"]["status"] == "UNRESOLVED"
    state.objective_lens.percent += 1.
    changed = WorkingPointCheckpoint(capture_instrument_snapshot(state), {}, state.sample.z_mm, "inputs", {})
    with pytest.raises(ValueError, match="separate"):
        ConvergenceRequest(changed, "joint_steps", 9, prior_evidence=separate).prepare()
