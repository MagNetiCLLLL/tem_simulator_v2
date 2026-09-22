"""Readout identity and exact data stay attached to their own displayed result."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.gui.result_readout import ResultReadout, result_readout


def result(identity, shift=0.):
    # Readout fixture, not a simulated or qualified microscope.
    x = np.array([[0., 1., 2.]])*1e-6+shift
    checkpoints = SimpleNamespace(z_mm=np.array([42.]), x_m=x, y_m=x*0,
        tx_rad=x*.1, ty_rad=x*0)
    state = SimpleNamespace(sample=SimpleNamespace(z_mm=42.),
        simulation_mode="Analytical fixture", illumination_mode="TEM", active_backend="CPU",
        electron_gun=SimpleNamespace(emitted_current_a=1e-9))
    incident = SimpleNamespace(ray_weight=np.array([.8, .1, .1]), alive=np.array([True, True, False]))
    return SimpleNamespace(state_snapshot=state, simulation=SimpleNamespace(
        incident_checkpoints=checkpoints, incident=incident), signatures={"request": identity})


def test_readout_uses_exact_plane_and_physical_weights(monkeypatch):
    import temsim.physics.beam_current as module
    monkeypatch.setattr(module, "effective_source_current_a", lambda state: state.electron_gun.emitted_current_a)
    row = result_readout(result("a"*64), "High accuracy")
    assert row["observations"]["transmission"] == pytest.approx(.9)
    assert row["observations"]["plane_current_a"] == pytest.approx(.9e-9)
    assert row["numerical_validation"] == "NOT_ESTABLISHED"
    assert row["observations"]["centroid_x_m"] == pytest.approx(1e-6/9)
    changed = result("b"*64)
    changed.state_snapshot.sample.z_mm = 43.
    assert result_readout(changed, "Preview")["observations"] is None


def test_selection_staleness_and_live_edits_do_not_mix_result_data(qtbot, monkeypatch):
    import temsim.physics.beam_current as module
    monkeypatch.setattr(module, "effective_source_current_a", lambda state: state.electron_gun.emitted_current_a)
    panel = ResultReadout()
    qtbot.addWidget(panel)
    high = result("a"*64)
    panel.publish(high, "High accuracy")
    saved = panel._records["high"]
    panel.mark_stale("high")
    panel.set_revision(12)
    preview = result("b"*64, 2e-6)
    panel.publish(preview, "Preview")
    assert panel._records["high"] is saved
    assert panel._records["ray"]["observations"]["centroid_x_m"] > saved["observations"]["centroid_x_m"]
    panel.selection.setCurrentIndex(1)
    assert "STALE" in panel.label.text() and "aaaaaaaaaaaa" in panel.label.text()
    assert "revision 12" in panel.label.text()
    high.state_snapshot.sample.z_mm = 99.
    high.simulation.incident_checkpoints.x_m[:] = 999.
    panel._refresh()
    assert panel._records["high"] is saved
    assert "Z 42 mm" in panel.label.text()


def test_missing_current_result_contract_is_reported_without_crashing(qtbot):
    panel = ResultReadout()
    qtbot.addWidget(panel)
    panel.publish(SimpleNamespace(), "Preview")
    assert "Readout unavailable" in panel.label.text()
    assert panel._records["ray"]["result_id"] == "UNRECORDED"


def test_current_result_without_signatures_has_no_invented_identity(qtbot):
    from temsim.simulation_pipeline import CalculationResult

    current = CalculationResult(simulation=SimpleNamespace(), energy_filter=None)
    assert current.signatures == {}
    row = result_readout(current, "Preview")
    assert row["result_id"] == "UNRECORDED"
    assert row["observations"] is None
    panel = ResultReadout()
    qtbot.addWidget(panel)
    panel.publish(current, "Preview")
    assert "UNRECORDED" in panel.label.text()
    assert "Readout unavailable" not in panel.label.text()


@pytest.mark.parametrize("signatures", [None, [], {"request": None}, {"request": 12}, {"request": ""}])
def test_malformed_result_identity_reports_error_without_replacing_plot(qtbot, signatures):
    panel = ResultReadout()
    qtbot.addWidget(panel)
    panel.publish(SimpleNamespace(signatures=signatures), "Preview")
    assert "Readout unavailable" in panel.label.text()
    assert panel._records["ray"]["result_id"] == "UNRECORDED"


def test_readout_error_keeps_the_valid_result_identity(qtbot):
    current = result("captured-request")
    current.simulation.incident_checkpoints.x_m = None
    panel = ResultReadout()
    qtbot.addWidget(panel)
    panel.publish(current, "Preview")
    assert panel._records["ray"]["result_id"] == "captured-request"
    assert "Readout unavailable" in panel.label.text()


@pytest.mark.parametrize("quality", ["Preview · transport validation", " Preview", "Medium", "Historical", "unknown"])
def test_non_high_labels_cannot_overwrite_retained_high(qtbot, quality):
    panel = ResultReadout()
    qtbot.addWidget(panel)
    panel.publish(SimpleNamespace(signatures={"request": "high"}), "High accuracy")
    panel.mark_stale("high")
    panel.publish(SimpleNamespace(signatures={"request": "other"}), quality)
    assert panel._records["high"]["result_id"] == "high"
    assert panel._stale["high"]
