"""Product contracts, distinct from physical or whole-chain qualification."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state
from temsim.sampling_diagnostics import sampling_summary, working_point_description
from temsim.sampling_convergence import (ConvergenceRequest, SamplingCancelled, compare_runs,
    run_convergence, write_evidence, _thresholds)
from temsim.working_point import WorkingPointCheckpoint


@pytest.fixture
def point():
    state = default_state()
    arrays = dict(x_m=np.array([0., 1e-6, -1e-6]), y_m=np.zeros(3),
        tx_rad=np.array([0., 1e-3, -1e-3]), ty_rad=np.zeros(3),
        weight=np.array([.2, .4, .4]), alive=np.array([True, True, False]))
    return WorkingPointCheckpoint(capture_instrument_snapshot(state), arrays, state.sample.z_mm,
        "fixture", {"source_current_a": 1e-9, "validation_status": "PASSED"})


def test_weighted_sampling_uses_full_source_population(point):
    row = sampling_summary(point.arrays, plane_z_mm=point.plane_z_mm, source_current_a=1e-9)
    assert row["emitted_samples"] == 3 and row["transmitted_samples"] == 2
    assert row["transmission"] == pytest.approx(.6)  # Not the count fraction 2/3.
    assert row["plane_current_a"] == pytest.approx(.6e-9)
    assert row["effective_samples"] == pytest.approx(1.8)
    assert row["centroid_x_m"] == pytest.approx(2e-6/3)
    assert row["convergence_status"] == "NOT_RUN"
    assert row["maximum_weight_fraction"] == pytest.approx(2/3)


@pytest.mark.parametrize("kind,expected", [("zero", "ZERO_TOTAL_WEIGHT"),
    ("empty", "NO_SAMPLED_SURVIVORS"), ("nonfinite", "NUMERICAL_FAILURE")])
def test_no_false_empty_or_finite_population_claims(point, kind, expected):
    arrays = {k: v.copy() for k, v in point.arrays.items()}
    if kind == "zero":
        arrays["weight"][:] = 0
    elif kind == "empty":
        arrays["alive"][:] = False
    else:
        arrays["x_m"][0] = np.nan
    row = sampling_summary(arrays, plane_z_mm=1, source_current_a=1e-9)
    assert row["status"] == expected
    assert row["diameter95_m"] is None and row["effective_samples"] is None
    if kind != "empty":
        assert row["transmission"] is None


def test_historical_pass_label_never_confers_current_qualification(point):
    row = working_point_description(point, current_implementation="changed")
    assert row["compatibility"] == "HISTORICAL_IMPLEMENTATION"
    assert row["numerical"] == "NOT_RUN" and row["physical_validation"] == "NOT_ESTABLISHED"


def test_publication_retains_exact_plane_not_plot_history(point):
    from temsim.calculation_manifest import capture_calculation_manifest
    from temsim.physics.core import PropagationCheckpoints
    state = default_state()
    manifest = capture_calculation_manifest(state)
    precise = np.array([[1.123456789012345e-6]])
    checkpoints = PropagationCheckpoints(np.array([state.sample.z_mm]), precise, precise*2, precise*3, precise*4)
    branch = SimpleNamespace(z=np.array([state.sample.z_mm]), x=precise.astype("f4"),
        y=precise.astype("f4"), tx=precise.astype("f4"), ty=precise.astype("f4"),
        ray_weight=np.ones(1), alive=np.ones(1, bool), energy_offset_ev=np.zeros(1))
    simulation = SimpleNamespace(incident=branch, incident_checkpoints=checkpoints,
        gun_trace=SimpleNamespace(exit_bundle=SimpleNamespace(ray_id=np.array([0]))))
    result = SimpleNamespace(simulation=simulation, calculation_manifest=manifest,
                             signatures=manifest.calculation_signatures)
    cp = WorkingPointCheckpoint.from_result(result)
    assert cp.arrays["x_m"].dtype == np.dtype("f8")
    assert np.array_equal(cp.arrays["x_m"], precise[0])
    assert cp.arrays["x_m"][0] != float(branch.x[0, 0])
    assert np.array_equal(cp.arrays["ty_rad"], (precise*4)[0])
    assert cp.metadata["source_current_a"] > 0
    simulation.incident_checkpoints = None
    with pytest.raises(ValueError, match="full-precision"):
        WorkingPointCheckpoint.from_result(result)


def test_pair_screen_rejects_underresolved_and_identifies_first_plane(point):
    row = dict(sampling_summary(point.arrays, plane_z_mm=1, source_current_a=1e-9))
    assert compare_runs([row], [row], _thresholds())["status"] == "UNRESOLVED"
    row.update(transmitted_samples=32, effective_samples=32)
    next_row = {**row, "plane_z_mm": 2}
    different = {**next_row, "diameter95_m": next_row["diameter95_m"]*2}
    report = compare_runs([row, next_row], [row, different], _thresholds())
    assert report["first_unresolved_plane_mm"] == 2
    assert report["planes"][0]["status"] == "STABLE_FOR_CHECKED_AXIS"
    assert "diameter95_m" in report["planes"][1]["failed_observables"]


def test_cancel_before_execution_and_invalid_budget_preserve_point(point):
    identity = point.digest
    with pytest.raises(SamplingCancelled):
        run_convergence(ConvergenceRequest(point, "column_step"), cancelled=lambda: True)
    with pytest.raises(ValueError, match="budget"):
        ConvergenceRequest(point, "emission_samples", maximum_rays=9).prepare()
    assert point.digest == identity


def test_vacuum_choice_is_preserved_when_audit_unsupported(point):
    state = point.snapshot.restore()
    state.vacuum_map.enabled = True
    snapshot = capture_instrument_snapshot(state)
    request = ConvergenceRequest(replace(point, snapshot=snapshot), "column_step")
    with pytest.raises(ValueError, match="vacuum"):
        request.prepare()
    assert state.vacuum_map.enabled


def test_actual_tip_chain_refinement_is_detached_scalar_and_unqualified(tmp_path):
    from temsim.assembly_catalog import AssemblyCatalog
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, catalog.default_selection())
    state.electron_gun.emitter.ray_count = 9
    state.step_mm = 1.
    state.history_step_mm = 5.
    state.acceleration_backend = "CPU"
    snapshot = capture_instrument_snapshot(state)
    point = WorkingPointCheckpoint(snapshot, {}, state.sample.z_mm, snapshot.physical_digest,
                                    {"package_kind": "INSTRUMENT_INPUTS_ONLY"})
    report = run_convergence(ConvergenceRequest(point, "emission_samples", maximum_rays=18))
    assert report["checkpoint_id"] == point.digest
    assert report["comparison"]["status"] == "UNRESOLVED"
    assert report["runs"][0]["planes"][-1]["emitted_samples"] == 9
    assert report["runs"][1]["planes"][-1]["emitted_samples"] == 18
    assert report["runs"][0]["sampling_rule"]["emitted_ray_budget"] == 9
    assert report["runs"][1]["sampling_rule"]["emitted_ray_budget"] == 18
    assert report["runs"][0]["sampling_rule"]["random_seed"] is None
    assert report["runs"][0]["snapshot_id"] != report["runs"][1]["snapshot_id"]
    assert len(report["runs"][1]["input_changes"]) == 1
    assert report["runs"][1]["input_changes"][0][0].endswith("/ray_count")
    assert report["runs"][0]["planes"][-1]["source_current_a"] == report["runs"][1]["planes"][-1]["source_current_a"]
    assert capture_instrument_snapshot(state).digest == snapshot.digest
    write_evidence(report, tmp_path / "scalar-evidence.json")
    assert (tmp_path / "scalar-evidence.json").stat().st_size < 250000


def test_browser_sort_filter_pin_and_readout_do_not_restore(qtbot, point, monkeypatch):
    from temsim.gui.working_point_panel import WorkingPointPanel
    from PySide6.QtCore import Qt
    panel = WorkingPointPanel()
    qtbot.addWidget(panel)
    def forbidden(*args, **kwargs):
        raise AssertionError("Read-only browsing attempted restoration")
    monkeypatch.setattr(type(point.snapshot), "restore", forbidden)
    from temsim.optics.electron_gun import source
    from temsim.physics import core, simulation
    monkeypatch.setattr(source, "trace_source_to_exit", forbidden)
    monkeypatch.setattr(core, "propagate", forbidden)
    monkeypatch.setattr(simulation, "run", forbidden)
    panel.add_checkpoint(point, label="Candidate A")
    panel._pin("A")
    second = replace(point, metadata={**point.metadata, "quality": "Different preset"})
    panel.add_checkpoint(second, label="Candidate B")
    panel._pin("B")
    panel.points.sortItems(0, Qt.SortOrder.DescendingOrder)
    assert panel.selected.digest == second.digest
    panel._compare_pins()
    assert "read-only" in panel.status.text()
    panel.filter.setText("no-matching-record")
    assert all(panel.points.isRowHidden(row) for row in range(2))
    panel.filter.setText("Candidate A")
    assert sum(not panel.points.isRowHidden(row) for row in range(2)) == 1
    panel.filter.setText(point.digest)
    assert sum(not panel.points.isRowHidden(row) for row in range(2)) == 1
    assert panel._summaries[point.digest]["transmission"] == pytest.approx(.6)
    assert panel.sampling.shutdown()


def test_cancelled_gui_result_never_replaces_previous_evidence(qtbot, point):
    from temsim.gui.sampling_panel import SamplingPanel
    panel = SamplingPanel()
    qtbot.addWidget(panel)
    panel.report = {"previous": True}
    panel.cancel()
    panel._result({"late": True})
    panel._finished()
    assert panel.report == {"previous": True}
    assert "cancelled" in panel.status.text()
    assert panel.shutdown()


def test_input_candidate_save_and_stale_evidence(qtbot, point):
    from temsim.gui.working_point_panel import WorkingPointPanel
    panel = WorkingPointPanel()
    qtbot.addWidget(panel)
    panel.current_snapshot = lambda: point.snapshot
    errors = []
    panel.error.connect(errors.append)
    panel._save_inputs()
    cp = panel.selected
    assert cp.is_input_design and cp.snapshot.digest == point.snapshot.digest
    panel._accept_evidence(dict(checkpoint_id=cp.digest, snapshot_id="stale", implementation=cp.snapshot.implementation))
    assert errors and "identity" in errors[-1]
    assert not panel._evidence
    assert panel.sampling.shutdown()


def test_async_worker_routes_evidence_without_restoring_live_state(qtbot, point, monkeypatch):
    from temsim.gui import sampling_panel
    from temsim.immutable_json import freeze_json
    report = freeze_json(dict(checkpoint_id=point.digest, snapshot_id=point.snapshot.digest,
        implementation=point.snapshot.implementation, axis="gun_step", comparison={"status": "UNRESOLVED"}))
    calls = []
    def execute(request, *, cancelled, progress):
        calls.append(request)
        progress("Bounded worker fixture")
        return report
    monkeypatch.setattr(sampling_panel, "run_convergence", execute)
    panel = sampling_panel.SamplingPanel()
    qtbot.addWidget(panel)
    panel.set_checkpoint(point)
    with qtbot.waitSignal(panel.evidence_ready, timeout=3000):
        panel.start()
    qtbot.waitUntil(lambda: not panel._busy)
    assert calls[0].checkpoint is point
    assert panel.report is report
    assert panel.run_button.isEnabled() and panel.export_button.isEnabled()
    assert "UNRESOLVED" in panel.status.text()
    assert panel.shutdown()
