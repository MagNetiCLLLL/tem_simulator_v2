"""Bounded workflow/analytic fixtures are not full-assembly qualification."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.immutable_json import freeze_json, thaw_json, json_digest
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state
from temsim.sampling_convergence import ConvergenceRequest
from temsim.sampling_qualification import (
    QualificationPlan, design_identity, load_journal, run_qualification,
)
from temsim.working_point import WorkingPointCheckpoint


@pytest.fixture
def point():
    state = default_state()
    state.electron_gun.emitter.ray_count = 32
    state.step_mm = 1.
    return WorkingPointCheckpoint(capture_instrument_snapshot(state), {},
        state.sample.upper_surface_z_mm, "qualification-fixture", {})


@pytest.fixture
def analytic_transport(monkeypatch):
    """An analytic ring in free space, explicitly not executed gun physics."""
    calls = []
    def execute(state, planes, *, step_mm):
        n = state.electron_gun.emitter.ray_count
        calls.append((n, step_mm, state.electron_gun.trace_step_mm))
        angle = np.arange(n)*2*np.pi/n
        shape = (len(planes), n)
        # Exactly radius 1 nm at every plane; zero slope. Equal source weights.
        checkpoints = SimpleNamespace(z_mm=np.array(planes),
            x_m=np.broadcast_to(1e-9*np.cos(angle), shape).copy(),
            y_m=np.broadcast_to(1e-9*np.sin(angle), shape).copy(),
            tx_rad=np.zeros(shape), ty_rad=np.zeros(shape))
        gun = SimpleNamespace(exit_bundle=SimpleNamespace(weight=np.ones(n)/n))
        return gun, checkpoints, np.ones(shape, bool)
    monkeypatch.setattr("temsim.optics.beam_path_audit.incident_checkpoints", execute)
    return calls


def test_three_levels_two_axes_numeric_only_detached_and_no_repeat(point, analytic_transport, tmp_path):
    plan = QualificationPlan(point, ("gun_step", "column_step"))
    path = tmp_path / "receipt.json"
    report = run_qualification(plan, path=path)
    assert report["status"] == "NUMERICALLY_CHECKED_FOR_DECLARED_SCOPE"
    assert report["physical_validation"] == "NOT_ESTABLISHED"
    assert len(report["completed"]) == 4 and len(analytic_transport) == 8
    for pair in report["completed"]:
        for run in pair["runs"]:
            assert run["planes"][0]["diameter95_m"] == pytest.approx(2e-9)
            assert run["planes"][0]["transmission"] == pytest.approx(1.)
            assert run["planes"][0]["chief_slope_x"] == pytest.approx(0.)
            assert "input_snapshot" in run
    restored = load_journal(path)
    again = run_qualification(plan, previous=restored, path=path)
    assert again["status"] == report["status"]
    assert len(analytic_transport) == 8
    assert point.snapshot.restore().step_mm == 1.
    assert QualificationPlan.from_manifest(point, restored["plan"]).identity == plan.identity


def test_nine_rays_never_qualify_even_exact_analytic_transport(point, analytic_transport):
    state = point.snapshot.restore()
    state.electron_gun.emitter.ray_count = 9
    point = replace(point, snapshot=capture_instrument_snapshot(state))
    result = run_qualification(QualificationPlan(point, ("column_step",)))
    assert result["status"] == "UNRESOLVED"
    assert len(result["completed"]) == 1
    assert result["completed"][0]["comparison"]["planes"][0]["status"] == "INSUFFICIENT_SUPPORT"


def test_budget_exhaustion_preserves_evidence_without_relaxing(point, analytic_transport):
    plan = QualificationPlan(point, ("gun_step", "column_step"), maximum_comparisons=1)
    result = run_qualification(plan)
    assert result["status"] == "BUDGET_EXHAUSTED"
    assert len(result["completed"]) == 1
    again = run_qualification(plan, previous=result)
    assert again["status"] == "BUDGET_EXHAUSTED" and len(analytic_transport) == 2
    assert result["completed"][0]["thresholds"] == plan.thresholds


def test_cancel_after_complete_pair_resumes_only_unfinished(point, analytic_transport, tmp_path):
    plan = QualificationPlan(point, ("column_step",))
    completed = []
    path = tmp_path / "resume.json"
    first = run_qualification(plan, path=path, cancelled=lambda: len(completed) == 1,
                             evidence=completed.append)
    assert first["status"] == "CANCELLED" and len(first["completed"]) == 1
    final = run_qualification(plan, previous=load_journal(path), path=path)
    assert final["status"] == "NUMERICALLY_CHECKED_FOR_DECLARED_SCOPE"
    assert len(analytic_transport) == 4


def test_cancel_during_pair_keeps_attempt_budget(point, analytic_transport, tmp_path):
    plan = QualificationPlan(point, ("column_step",), maximum_comparisons=3)
    first = run_qualification(plan, cancelled=lambda: len(analytic_transport) >= 1)
    assert first["status"] == "CANCELLED"
    assert not first["completed"] and len(first["attempts"]) == 1
    final = run_qualification(plan, previous=first)
    assert len(final["attempts"]) == 3 and len(analytic_transport) == 5


def test_wall_budget_uses_solver_cancellation_boundary(point, analytic_transport, monkeypatch):
    import temsim.sampling_qualification as module
    monkeypatch.setattr(module, "perf_counter", lambda: len(analytic_transport)*10.)
    result = run_qualification(QualificationPlan(point, ("column_step",), wall_seconds=1))
    assert result["status"] == "BUDGET_EXHAUSTED"
    assert not result["completed"] and len(analytic_transport) == 1
    assert result["elapsed_s"] == 10.


@pytest.mark.parametrize("change", ["tolerance", "budget", "lens", "seed_policy", "implementation"])
def test_mismatched_resume_is_rejected(point, analytic_transport, change):
    plan = QualificationPlan(point, ("column_step",), maximum_comparisons=1)
    previous = run_qualification(plan)
    if change == "tolerance":
        policy = thaw_json(plan.thresholds)
        policy["relative"] *= 2
        altered = replace(plan, thresholds=policy)
    elif change == "budget":
        altered = replace(plan, maximum_comparisons=2)
    elif change == "lens":
        state = point.snapshot.restore()
        state.lenses[0].percent += .1
        altered = replace(plan, checkpoint=replace(point, snapshot=capture_instrument_snapshot(state)))
    else:
        previous = thaw_json(previous)
        previous["plan"][change] = "changed"
        previous["digest"] = json_digest({k: v for k, v in previous.items() if k != "digest"})
        altered = plan
    count = len(analytic_transport)
    with pytest.raises(ValueError):
        run_qualification(altered, previous=previous)
    assert len(analytic_transport) == count


def test_changed_snapshot_or_forged_verdict_cannot_be_admitted(point, analytic_transport):
    plan = QualificationPlan(point, ("column_step",), maximum_comparisons=1)
    previous = thaw_json(run_qualification(plan))
    pair = previous["completed"][0]
    pair["runs"][1]["planes"][0]["diameter95_m"] *= 10
    pair["digest"] = json_digest({k: v for k, v in pair.items() if k != "digest"})
    previous["digest"] = json_digest({k: v for k, v in previous.items() if k != "digest"})
    with pytest.raises(ValueError, match="verdict"):
        run_qualification(plan, previous=previous)


def test_one_unstable_axis_stops_multi_axis_plan(point, analytic_transport, monkeypatch):
    import temsim.sampling_qualification as module
    actual = module.run_convergence
    def divergent(request, **kwargs):
        result = thaw_json(actual(request, **kwargs))
        if request.axis == "column_step":
            result["runs"][1]["planes"][0]["diameter95_m"] *= 2
            from temsim.sampling_convergence import compare_runs
            topology = result["comparison"]["topology"]
            result["comparison"] = compare_runs(result["runs"][0]["planes"], result["runs"][1]["planes"], request.threshold_policy)
            result["comparison"]["topology"] = topology
            result["digest"] = json_digest({k: v for k, v in result.items() if k != "digest"})
        return freeze_json(result)
    monkeypatch.setattr(module, "run_convergence", divergent)
    result = run_qualification(QualificationPlan(point, ("gun_step", "column_step")))
    assert result["status"] == "UNRESOLVED"
    assert len(result["completed"]) == 3


def test_product_energy_remains_conditional_and_method_changes_not_mixed(point, analytic_transport):
    plan = QualificationPlan(point, ("energies",), spatial_samples=3, direction_samples=3, energy_samples=3)
    result = run_qualification(plan)
    assert result["status"] == "NUMERICALLY_CHECKED_FOR_DECLARED_SCOPE"
    runs = result["completed"][1]["runs"]
    assert [run["sampling_rule"]["independent_product"]["energies"] for run in runs] == [6, 12]
    assert all("conditional" in run["sampling_rule"]["conditional_law"].lower()
               or "given local direction" in run["sampling_rule"]["conditional_law"] for run in runs)
    mixed = replace(plan, axes=("energies", "gun_step"))
    with pytest.raises(ValueError, match="different quadrature methods"):
        run_qualification(mixed)


def test_design_identity_excludes_numerics_not_physical_controls(point):
    state = point.snapshot.restore()
    initial = design_identity(point.snapshot)
    state.step_mm *= .5
    state.electron_gun.trace_step_mm *= .5
    state.electron_gun.emitter.ray_count *= 2
    assert design_identity(capture_instrument_snapshot(state)) == initial
    state.lenses[0].percent += .1
    assert design_identity(capture_instrument_snapshot(state)) != initial


def test_preflight_invalid_axis_level_and_memory_no_execution(point, analytic_transport):
    with pytest.raises(ValueError, match="supported"):
        QualificationPlan(point, ("domain_extent",))
    with pytest.raises(ValueError, match="three settings"):
        QualificationPlan(point, ("column_step",), refinements=1)
    with pytest.raises(ValueError, match="ray budget"):
        run_qualification(QualificationPlan(point, ("emission_samples",), maximum_rays=64))
    assert not analytic_transport


def test_no_overwrite_or_corrupt_resume(point, analytic_transport, tmp_path):
    path = tmp_path / "receipt.json"
    path.write_text("existing unrelated file")
    with pytest.raises(ValueError, match="exists"):
        run_qualification(QualificationPlan(point, ("column_step",)), path=path)
    assert path.read_text() == "existing unrelated file"


def test_plan_ui_worker_is_detached_and_retains_cancelled_evidence(qtbot, point, monkeypatch, tmp_path):
    from threading import Event
    from temsim.gui.sampling_panel import SamplingPanel
    from temsim.gui.qualification_dialog import QualificationDialog
    entered, release = Event(), Event()
    def lifecycle_fixture(plan, *, cancelled, **kwargs):
        entered.set()
        assert release.wait(5)
        assert cancelled()
        pairs = [dict(checkpoint_id=point.digest, axis="column_step", refinement_level=n,
                      comparison={"status": "UNRESOLVED"}) for n in (0, 1)]
        return freeze_json(dict(plan_id=plan.identity, status="CANCELLED", reason="Lifecycle fixture",
                                completed=pairs))
    monkeypatch.setattr("temsim.gui.qualification_dialog.run_qualification", lifecycle_fixture)
    dialog = QualificationDialog()
    qtbot.addWidget(dialog)
    assert {key for key, control in dialog.axes.items() if control.isChecked()} == {"gun_step", "column_step"}
    panel = SamplingPanel()
    qtbot.addWidget(panel)
    panel.set_checkpoint(point)
    plan = QualificationPlan(point, ("column_step",))
    panel._start_plan(plan, str(tmp_path / "gui.json"))
    try:
        qtbot.waitUntil(entered.is_set, timeout=5000)
        panel.cancel()
        release.set()
        qtbot.waitUntil(lambda: not panel._busy, timeout=5000)
        assert panel.plan_report["status"] == "CANCELLED"
        assert panel.plan_button.isEnabled() and panel.resume_button.isEnabled()
        assert {"column_step", "column_step/refinement-1"} == set(panel._history[point.digest])
    finally:
        release.set()
        assert panel.shutdown()


def test_active_field_mesh_and_domain_change_only_numerical_identity(point):
    from temsim.optics.electron_gun.tip_assembly import model_from_part
    state = point.snapshot.restore()
    emitter = state.electron_gun.emitter
    emitter.surface_model = model_from_part(state._resolved_assembly.part("feg_tip").data)
    point = replace(point, snapshot=capture_instrument_snapshot(state))
    for axis in ("field_radial", "field_domain"):
        a = ConvergenceRequest(point, axis, refinement_level=0).prepare()
        b = ConvergenceRequest(point, axis, refinement_level=1).prepare()
        assert design_identity(capture_instrument_snapshot(a)) == design_identity(capture_instrument_snapshot(b))
    with pytest.raises(ValueError, match="radial_nodes"):
        ConvergenceRequest(point, "field_radial", refinement_level=3).prepare()


def test_actual_nine_ray_tip_origin_plan_is_unresolved_and_bounded(point, tmp_path):
    """Real emitter/gun/column execution, separate from the analytic fixtures."""
    state = point.snapshot.restore()
    state.electron_gun.emitter.ray_count = 9
    point = replace(point, snapshot=capture_instrument_snapshot(state))
    result = run_qualification(QualificationPlan(point, ("column_step",), maximum_rays=9,
        wall_seconds=90), path=tmp_path / "actual-nine-rays.json")
    assert result["status"] == "UNRESOLVED"
    assert len(result["completed"]) == 1
    assert result["completed"][0]["runs"][0]["planes"][0]["emitted_samples"] == 9
    assert result["full_simulator_qualification"] == "NOT_ESTABLISHED"
