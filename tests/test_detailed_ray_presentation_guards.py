"""Presentation-only safeguards when detailed rays replace optical references."""

from types import SimpleNamespace

import numpy as np
import pytest

from temsim.gui.visualization import VisualizationWorkspace
from temsim.physics.ray_identity import source_identity, select_identity
from temsim.specimen.downstream_transport import GeometricSpecimenExit


def _branch(name, start, stop, count=3):
    x = np.tile(np.linspace(-1e-6, 1e-6, count), (2, 1))
    y = np.tile(np.linspace(1e-6, 2e-6, count), (2, 1))
    return SimpleNamespace(
        name=name, z=np.asarray((start, stop)), x=x, y=y,
        tx=np.zeros_like(x), ty=np.zeros_like(y),
        alive=np.ones(count, dtype=bool), blocked_z=np.full(count, np.nan),
        blocked_key=[""] * count, ray_weight=np.full(count, 1.0 / count),
        weight=1.0, colour=(.3, .9, .4),
        interaction_kind="incident" if name == "incident" else "sample_region_elastic",
    )


def _result():
    incident = _branch("incident", 0., 10.)
    incident.source_ray_id, incident.source_azimuth_rad = source_identity(incident)
    reference = _branch("000", 10., 20.)
    reference.source_ray_id = incident.source_ray_id
    reference.source_azimuth_rad = incident.source_azimuth_rad
    calls = []

    def scan_command(time_s):
        calls.append(time_s)
        return (float(time_s), 0.0)

    responses = {
        branch.name: (np.ones((2, 2, 2)), np.zeros((2, 2, 2)))
        for branch in (incident, reference)
    }
    paths = SimpleNamespace(
        frame_period_s=1.0, pixels_x=4, pixels_y=4,
        baseline_ac_command_mrad=np.zeros(2),
        baseline_descan_command_mrad=np.zeros(2),
        responses_m_per_rad=responses,
    )
    result = SimpleNamespace(
        simulation=SimpleNamespace(
            incident=incident, branches={"000": reference}, metrics={},
            gun_waist=None, c2c3_crossover=None, corrector_crossovers=(),
        ),
        state_snapshot=SimpleNamespace(
            sample=SimpleNamespace(z_mm=10.0),
            ac_deflector=SimpleNamespace(scan_kick_mrad=scan_command),
            descan_deflector=SimpleNamespace(enabled=False, scan_enabled=False),
        ),
        signatures={"sample_downstream": "current"},
        specimen_exit=None, sample_region=None,
        lens_crossovers=(), aperture_stops=(),
        assembly=SimpleNamespace(parts=(), vacuum_bore_segments=()),
        scan_ray_paths=paths,
    )
    return result, calls


def _exit(result, *, empty=False):
    children = ()
    if not empty:
        branch = _branch("finite_elastic", 10., 20., count=2)
        branch.source_ray_id, branch.source_azimuth_rad = select_identity(
            result.simulation.incident.source_ray_id,
            result.simulation.incident.source_azimuth_rad,
            np.asarray((2, 0)),
        )
        children = (branch,)
    return GeometricSpecimenExit(
        children,
        {
            "tracked_downstream_source_probability": float(not empty),
            "inelastic_absorbed_source_probability": float(empty),
        },
        dependency_signature="current",
    )


@pytest.fixture
def view(qtbot, monkeypatch):
    widget = VisualizationWorkspace()
    qtbot.addWidget(widget)
    monkeypatch.setattr(widget, "_sync_ray_static_layers", lambda _result: None)
    return widget


@pytest.mark.parametrize("empty", (False, True))
def test_detailed_downstream_budget_never_computes_reference_counts(view, monkeypatch, empty):
    import temsim.physics.interaction_budget as budget_module

    result, _calls = _result()
    result.specimen_exit = _exit(result, empty=empty)
    view._last_result = result
    view._selected_z_mm = 15.0
    original_budget = budget_module.plane_interaction_budget
    monkeypatch.setattr(
        budget_module, "plane_interaction_budget",
        lambda *_args: pytest.fail("Detailed exit must not use reference branch budget"),
    )
    view._update_interaction_detail()
    text = view.interaction_detail.toPlainText()
    assert "optical-reference budget does not describe this population" in text
    assert "Current reaching Z" not in text
    assert "before the sample" in view.interaction_detail.toolTip()

    monkeypatch.setattr(budget_module, "plane_interaction_budget", original_budget)
    view._selected_z_mm = 5.0
    view._update_interaction_detail()
    assert "upstream of sample" in view.interaction_detail.toPlainText()
    assert "100% of source" in view.interaction_detail.toPlainText()


@pytest.mark.parametrize("empty", (False, True))
def test_same_result_detailed_publication_clears_offsets_then_reference_playback_can_resume(
    view, monkeypatch, empty
):
    result, calls = _result()
    view._last_result = result
    view._last_quality = "High accuracy"
    view._prepare_scan_ray_playback(result)
    view._draw_ray_diagram(result, "High accuracy")
    view._scan_playback_time_changed(.25)
    assert view._scan_ray_offsets_m
    assert calls == [.25]
    source_x = result.simulation.incident.x.copy()
    response_arrays = tuple(
        array.copy()
        for pair in result.scan_ray_paths.responses_m_per_rad.values()
        for array in pair
    )
    paths = view._scan_ray_paths

    # Manual enrichment publishes into the same result without preparing new
    # playback: offsets must disappear immediately, before the next timer tick.
    result.specimen_exit = _exit(result, empty=empty)
    view._draw_ray_diagram(result, "High accuracy", preserve_view=True)
    assert view._scan_ray_offsets_m == {}
    assert view._scan_ray_paths is paths
    assert "captured rays; scan animation off" in view.heading.text()
    assert "STEM image playback is separate" in view.heading.toolTip()
    curves = [(item, item.getData()[1].copy()) for item, _payload in view._ray_bundle_records]

    monkeypatch.setattr(
        result.state_snapshot.ac_deflector, "scan_kick_mrad",
        lambda _t: pytest.fail("Detailed display must not apply optical scan offsets"),
    )
    view._scan_playback_time_changed(.75)
    for item, values in curves:
        np.testing.assert_array_equal(item.getData()[1], values)
    assert calls == [.25]
    np.testing.assert_array_equal(result.simulation.incident.x, source_x)
    for pair, originals in zip(
        result.scan_ray_paths.responses_m_per_rad.values(),
        zip(response_arrays[::2], response_arrays[1::2]), strict=True,
    ):
        for array, original in zip(pair, originals, strict=True):
            np.testing.assert_array_equal(array, original)

    result.specimen_exit = None
    monkeypatch.setattr(result.state_snapshot.ac_deflector, "scan_kick_mrad", lambda t: (t, 0.))
    view._draw_ray_diagram(result, "High accuracy", preserve_view=True)
    view._scan_playback_time_changed(.5)
    assert view._scan_ray_offsets_m
    assert "scan animation off" not in view.heading.text()
    assert view.heading.toolTip() == ""
