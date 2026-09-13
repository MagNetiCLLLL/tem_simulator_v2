"""Round diagnostics cannot mutate or replace a physical boundary result."""
import pytest

from temsim.immutable_json import thaw_json
from temsim.physics.refinement_progress import emit_refinement_record
from temsim.physics.occupied_axial_refinement import OccupiedAxialRefinement, refine_joint_mode
from test_occupied_axial_refinement import plan_and_boundary


def test_observer_receives_immutable_diagnostics_only():
    rows = []
    def callback(*args):
        pass
    callback.record_refinement = lambda method, row: rows.append((method, row))
    original = {"round": 1, "nested": {"value": [1, 2]}}
    emit_refinement_record(callback, "test", original)
    with pytest.raises(TypeError):
        rows[0][1]["nested"]["value"][0] = 9
    original["nested"]["value"][0] = 7
    assert rows[0][1]["nested"]["value"][0] == 1
    emit_refinement_record(None, "test", original)


@pytest.mark.parametrize("strategy", ["global_embedded", "spatial_embedded"])
def test_each_completed_round_is_observed_without_affecting_acceptance(strategy):
    rows = []
    def callback(*args):
        pass
    callback.record_refinement = lambda method, row: rows.append((method, row))
    plan, boundary, _, _ = plan_and_boundary()
    _, _, result = refine_joint_mode(plan, boundary, OccupiedAxialRefinement(
        enabled=True, integrator="cf6", strategy=strategy, maximum_rounds=32), progress_callback=callback)
    assert [thaw_json(row) for _, row in rows] == result["rounds"]
    assert all(method == strategy for method, _ in rows)
    assert rows[-1][1]["successive_uniform_checks"] == 2


def test_observer_failure_does_not_publish_success():
    def callback(*args):
        pass
    def failed(*args):
        raise OSError("Could not preserve round diagnostics")
    callback.record_refinement = failed
    plan, boundary, _, _ = plan_and_boundary()
    with pytest.raises(OSError, match="preserve round") as failure:
        refine_joint_mode(plan, boundary, OccupiedAxialRefinement(enabled=True, strategy="spatial_embedded"),
                          progress_callback=callback)
    assert failure.value.refinement_diagnostic["scope"].startswith("FAILED_")


def test_failure_diagnostics_publish_before_buffer_cleanup(tmp_path):
    import json
    from scripts.inspect_surface_column import _persist_report
    record = {"status": "FAILED", "diagnostics_saved_before_failed_buffer_cleanup": True,
              "refinement_diagnostic": {"scope": "FAILED_SPATIAL_MESH_COMPARISON_NOT_SOURCE", "coarse_mesh": []}}
    _persist_report(tmp_path, record)
    assert json.loads((tmp_path/"report.json").read_text()) == record
    assert not (tmp_path/"report.pending.json").exists()
