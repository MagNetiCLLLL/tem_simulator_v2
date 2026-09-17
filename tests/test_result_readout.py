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


def test_missing_old_products_remain_readable(qtbot):
    panel = ResultReadout()
    qtbot.addWidget(panel)
    panel.publish(SimpleNamespace(), "Historical")
    assert "Unavailable" in panel.label.text()
    assert "NOT_ESTABLISHED" in panel.label.text()
