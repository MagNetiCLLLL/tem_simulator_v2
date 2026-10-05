"""Conjugacy navigation and display contracts; these do not qualify optics."""
from types import SimpleNamespace

import numpy as np
import pyqtgraph as pg

from temsim.gui.conjugate_plane_overlay import ConjugatePlaneOverlay
from temsim.gui.visualization import VisualizationWorkspace
from temsim.physics.selected_plane import SelectedPlaneDiagnostic


def _search():
    return SimpleNamespace(reference_z_mm=20., candidates=(
        SimpleNamespace(z_mm=60., kind="image", residual_m_per_rad=1.e-8,
                        detail="Synthetic first-order image"),
        SimpleNamespace(z_mm=80., kind="line_focus", residual_m_per_rad=.01,
                        detail="Synthetic one-direction focus"),
    ))


def test_markers_preserve_fit_bounds_and_survive_scene_reset(qtbot):
    plot = pg.PlotWidget()
    qtbot.addWidget(plot)
    plot.plot([0., 1.], [0., 1.])
    plot.setRange(xRange=(0., 1.), yRange=(0., 1.), padding=0.)
    before = np.asarray(plot.getViewBox().viewRange())
    before_bounds = np.asarray(plot.getViewBox().childrenBounds())
    overlay = ConjugatePlaneOverlay(plot)
    overlay.set_search(_search())
    overlay.setVisible(True)
    assert len(overlay.items) == 3
    assert all(item.isVisible() for item in overlay.items)
    assert "not a point image" in overlay.items[2].toolTip()
    np.testing.assert_allclose(plot.getViewBox().childrenBounds(), before_bounds)
    np.testing.assert_allclose(plot.getViewBox().viewRange(), before)
    plot.clear()
    overlay.refresh()
    assert all(item.scene() is plot.scene() for item in overlay.items)
    overlay.setVisible(False)
    assert all(not item.isVisible() for item in overlay.items)
    overlay.clear()
    assert not overlay.items
    assert overlay.search is None


def test_workspace_conjugacy_jump_preserves_view_and_toggle_never_calculates(qtbot, monkeypatch):
    workspace = VisualizationWorkspace()
    qtbot.addWidget(workspace)
    panel = workspace.conjugate_planes
    assert panel.isHidden()
    workspace.conjugate_planes_toggle.click()
    assert not panel.isHidden()
    assert panel._worker is None
    monkeypatch.setattr(workspace, "_simulation_x_limits", lambda: (0., 100.))
    monkeypatch.setattr(workspace, "_focus_transverse", lambda *_args: None)
    monkeypatch.setattr(workspace, "_update_interaction_detail", lambda: None)
    workspace.plot.setRange(xRange=(10., 90.), yRange=(-2., 2.), padding=0.)
    before = np.asarray(workspace.plot.getViewBox().viewRange())
    panel.search_changed.emit(_search())
    assert len(workspace.conjugate_plane_overlay.items) == 3
    panel.plane_selected.emit(60.)
    assert workspace._selected_z_mm == 60.
    assert workspace.axial_position.value() == 60.
    workspace.selected_plane_readout._display(SelectedPlaneDiagnostic(60., "mixed"))
    tooltip = workspace.selected_plane_readout.label.toolTip()
    assert "Mixed plane" in tooltip
    assert workspace.axial_cursor_item.toolTip() == tooltip
    assert workspace.axial_cursor_item.label.toolTip() == tooltip
    workspace.selected_plane_readout.clear()
    assert "Model only" in workspace.axial_cursor_item.toolTip()
    assert "Mixed plane" not in workspace.axial_cursor_item.toolTip()
    np.testing.assert_allclose(workspace.plot.getViewBox().viewRange(), before)
    assert panel._worker is None
    panel.mark_stale()
    assert not workspace.conjugate_plane_overlay.items
    assert panel.shutdown()
    assert workspace.selected_plane_readout.shutdown()
