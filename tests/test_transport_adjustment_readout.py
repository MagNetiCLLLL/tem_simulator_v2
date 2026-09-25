"""Captured adjustment descriptions and display-only GUI state transitions."""
from copy import deepcopy
from types import SimpleNamespace

from PySide6.QtCore import Qt
import pytest

from temsim.gui.transport_adjustment_readout import TransportAdjustmentReadout


def adjustment():
    return dict(
        physics_scope="optical_transport_only", source_ray_count=193,
        validation_step_mm=.05, target_plane_z_mm=2586.9, source_fraction=.125,
        controls=tuple(dict(key=f"condenser_lens_{i}", label=f"C{i}",
                            before_percent=10.*i, after_percent=10.*i + .125)
                       for i in (1, 2, 3)),
    )


def result(record):
    return SimpleNamespace(simulation=SimpleNamespace(metrics={"transport_adjustment": record}))


def test_captured_changes_are_selectable_and_cleared_with_other_results(qtbot):
    view = TransportAdjustmentReadout()
    qtbot.addWidget(view)
    record = adjustment()
    before = deepcopy(record)
    view.display_result(result(record))
    assert not view.isHidden()
    assert "C1 10% → 10.125%" in view.text()
    assert "C3 30% → 30.125%" in view.text()
    assert "193 source samples" in view.text()
    assert "12.5% of source through Z 2586.9 mm" in view.text()
    assert "Specimen signals deferred" in view.text()
    assert "focus not calibrated" in view.text()
    assert view.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse
    assert record == before
    view.display_result(SimpleNamespace(simulation=SimpleNamespace(metrics={})))
    assert view.isHidden()
    assert view.text() == ""
    view.display_result(result(record))
    view.display_result(None)
    assert view.isHidden()


@pytest.mark.parametrize("record", [
    {}, {"physics_scope": "full_physics"},
    {**adjustment(), "controls": ()},
    {**adjustment(), "source_fraction": float("nan")},
])
def test_invalid_details_do_not_break_result_publication(qtbot, record):
    view = TransportAdjustmentReadout()
    qtbot.addWidget(view)
    view.display_result(result(record))
    assert "unavailable" in view.text()


def test_workspace_uses_captured_record_and_clears_new_overlays(make_workspace):
    from test_incremental_ray_scene import _result

    view = make_workspace()
    captured = _result()
    captured.simulation.metrics.update(optical_tuning=True, transport_adjustment=adjustment())
    view.display_result(captured, "Preview · transport validation")
    assert "C2 20% → 20.125%" in view.transport_adjustment_readout.text()
    view.display_result(_result(), "Preview")
    assert view.transport_adjustment_readout.isHidden()
    view.clear_result()
    assert view.transport_adjustment_readout.isHidden()


from test_incremental_ray_scene import make_workspace
