"""Transport result publication with synthetic paths, not optics qualification."""
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.alignment_transaction import (
    AlignmentCommitGate, AlignmentRequest, solve_alignment_candidate,
)
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state
from temsim.optics.transport_matching import LENSES, target_plane
from temsim.physics.core import PropagationCheckpoints
from temsim.physics.simulation import Branch, Simulation


@pytest.fixture
def successful_transport(monkeypatch):
    """Keep real snapshots, measurements, checkpoints and result constructors."""
    state = default_state()
    request = AlignmentRequest.capture(state, "column_transport", .01, revision=4)
    strengths = np.array([7.9, 25.7, 40.3])
    count = 4
    zeros = np.zeros(count)
    x = np.linspace(-1e-5, 1e-5, count)
    weights = np.full(count, 1 / count)
    gun = SimpleNamespace(exit_bundle=SimpleNamespace(
        x_m=x, y_m=zeros, tx_rad=zeros, ty_rad=zeros, weight=weights,
        alive=np.ones(count, bool), ray_id=np.arange(count),
    ))
    calls = []

    def forward(working, *, resolved_layout, optical_only):
        assert optical_only
        assert resolved_layout is working._resolved_optics_layout
        assert [lens.percent for lens in working.lenses if lens.key in LENSES] == list(strengths)
        z = np.array([working.electron_gun.exit_plane_z_mm, working.sample.z_mm,
                      target_plane(working), target_plane(working) + 1])
        shape = (len(z), count)
        branch = Branch(
            name="synthetic transport", colour=(1., 1., 1.), z=z,
            x=np.tile(x, (len(z), 1)), y=np.zeros(shape),
            tx=np.zeros(shape), ty=np.zeros(shape), alive=np.ones(count, bool),
            blocked_z=np.full(count, np.nan), blocked_key=[""] * count,
            weight=1., energy_offset_ev=zeros, ray_weight=weights,
        )
        exact = PropagationCheckpoints(
            np.array([working.sample.z_mm]), x[None], zeros[None], zeros[None], zeros[None],
        )
        simulation = Simulation(branch, {"000": branch}, {}, gun_trace=gun,
                                incident_checkpoints=exact)
        calls.append((working.step_mm, simulation))
        return simulation

    monkeypatch.setattr("temsim.optics.electron_gun.source.trace_source_to_exit", lambda _: gun)
    monkeypatch.setattr("temsim.optics.transport_matching._candidate_vectors",
                        lambda *args: [strengths])
    monkeypatch.setattr("temsim.physics.simulation.run", forward)
    return state, request, calls


def test_successful_transport_publishes_refined_result_and_retains_manifest(successful_transport):
    from temsim.gui.result_readout import result_readout

    state, request, calls = successful_transport
    before = capture_instrument_snapshot(state).digest
    candidate = solve_alignment_candidate(request)
    assert candidate.status == "READY_TO_APPLY"
    assert candidate.result.success
    assert [step for step, _ in calls] == [.2, .1, .05]
    display = candidate.ray_result
    assert display.simulation is calls[-1][1]
    assert display.calculated_products == frozenset({"incident", "column"})
    adjustment = display.simulation.metrics["transport_adjustment"]
    assert adjustment["source_ray_count"] == len(display.simulation.gun_trace.exit_bundle.ray_id)
    assert adjustment["physics_scope"] == "optical_transport_only"
    before_values = {lens.key: lens.percent for lens in state.lenses if lens.key in LENSES}
    assert {row["key"]: row["before_percent"] for row in adjustment["controls"]} == before_values
    assert {row["key"]: row["after_percent"] for row in adjustment["controls"]} == dict(candidate.result.strengths)
    manifest = display.calculation_manifest
    assert isinstance(display.signatures, dict)
    assert display.signatures == dict(manifest.calculation_signatures)
    assert candidate.checkpoint.stage_signature == display.signatures["incident"]
    assert candidate.checkpoint.snapshot.digest == manifest.instrument_snapshot.digest
    assert result_readout(display, "Preview · transport validation")["result_id"] == display.signatures["request"]
    assert capture_instrument_snapshot(state).digest == before

    gate = AlignmentCommitGate()
    updated = gate.apply(state, candidate, revision=4)
    assert {lens.key: lens.percent for lens in updated.lenses if lens.key in LENSES} == dict(candidate.result.strengths)
    assert capture_instrument_snapshot(gate.undo()).digest == before

    # A mutable display record must never mutate the immutable provenance.
    original = manifest.calculation_signatures["request"]
    digest = manifest.digest
    display.signatures["request"] = "changed display fixture"
    assert manifest.calculation_signatures["request"] == original
    assert manifest.digest == digest
    with pytest.raises(TypeError):
        manifest.calculation_signatures["request"] = "forbidden"


def test_transport_worker_emits_completed_result_without_signature_error(qtbot, successful_transport):
    from temsim.gui.direct_alignment_controller import DirectAlignmentWorker
    from temsim.gui.result_readout import ResultReadout

    _, request, calls = successful_transport
    worker = DirectAlignmentWorker(3, request, Event())
    results, errors, finished = [], [], []
    worker.signals.result.connect(lambda *args: results.append(args))
    worker.signals.error.connect(lambda *args: errors.append(args))
    worker.signals.finished.connect(lambda *args: finished.append(args))
    worker.run()
    assert errors == []
    assert finished == [(3, "column_transport")]
    assert len(results) == 1
    generation, key, candidate, _ = results[0]
    assert (generation, key) == (3, "column_transport")
    assert candidate.ray_result.simulation is calls[-1][1]
    readout = ResultReadout()
    qtbot.addWidget(readout)
    readout.publish(candidate.ray_result, "Preview · transport validation")
    assert "Readout unavailable" not in readout.label.text()
    assert candidate.ray_result.signatures["request"][:12] in readout.label.text()
