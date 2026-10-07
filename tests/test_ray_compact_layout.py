"""Conjugate-plane browsing must leave rays usable on a small CPU laptop."""

from types import SimpleNamespace

import pytest
from PySide6.QtCore import QRect, QSize, Qt

from temsim.app import APPLICATION_STYLE
from temsim.gui.visualization import VisualizationWorkspace


def _search():
    return SimpleNamespace(
        reference_z_mm=3026.4,
        lower_z_mm=450.0,
        upper_z_mm=3026.4,
        candidates=tuple(
            SimpleNamespace(
                z_mm=496.55 + 250.0 * index,
                kind="approximate",
                residual_m_per_rad=1.4e-6,
                magnifications=(0.00148, 0.00148),
                rotation_deg=-55.06,
                mirrored=False,
                detail="Synthetic first-order conjugate; particle reach is not implied.",
            )
            for index in range(9)
        ),
        detail=(
            "Captured nominal zero-loss paraxial optics | 450–3026.4 mm | "
            "10308 cached nodes | Numba CPU. Uses installed lens, stigmator and "
            "deflector fields. Near-tip/gun propagation and curved energy-filter "
            "coordinates are excluded. Conjugacy does not guarantee particle "
            "transmission, absence of aberrations or material coherence."
        ),
    )


@pytest.fixture
def application_theme(qapp):
    previous = qapp.styleSheet()
    qapp.setStyleSheet(APPLICATION_STYLE)
    try:
        yield
    finally:
        qapp.setStyleSheet(previous)


@pytest.fixture
def workspace(qtbot, monkeypatch, application_theme):
    view = VisualizationWorkspace()
    qtbot.addWidget(view)
    view.transverse_beam_toggle.setChecked(False)
    view.heading.setText(
        "Electron ray paths — Preview | X projection at 0° | "
        "8 crossovers | 0 column-wall stops | Optical reference"
    )
    view.stop_detail.setText(
        "Selected axial position: Z 3026.4 mm | drag the cyan cursor or "
        "double-click another axial plot"
    )
    view.hint.setText(
        "Max physical X angle: 1.46° | Transverse display: 5.04× "
        "(angles not to scale) | Hue = fixed emitted-position azimuth | "
        "grey = undefined source azimuth | Blocked intercept rays stop at first "
        "intercept | Column walls use radial X/Y"
    )
    panel = view.conjugate_planes

    def forbidden(*_args, **_kwargs):
        pytest.fail("Changing layout or browsing retained results must not start work")

    monkeypatch.setattr(panel.pool, "start", forbidden)
    monkeypatch.setattr(view.selected_plane_readout.pool, "start", forbidden)
    panel.set_result(SimpleNamespace(name="captured synthetic optics"))
    panel.select_z(3026.4)
    panel.use_selected_z()
    panel._atlas = object()
    panel._solved(panel._generation, panel._result_generation, _search())
    yield view
    assert panel.shutdown()
    assert view.selected_plane_readout.shutdown()


def _visible_height(view, viewport):
    """Reject content that is merely below the surrounding scroll viewport."""
    clip = view.page_scroll.viewport()
    content_rect = QRect(viewport.mapTo(clip, viewport.rect().topLeft()), viewport.size())
    return content_rect.intersected(clip.rect()).height()


@pytest.mark.parametrize("height,interaction_visible", ((650, True), (550, False)))
def test_conjugate_results_leave_visible_rays_on_small_screen(
    workspace, qtbot, height, interaction_visible,
):
    assert workspace.interaction_detail_toggle.isChecked()
    workspace.interaction_detail_toggle.setChecked(interaction_visible)
    workspace.resize(1100, height)
    workspace.show()
    workspace.conjugate_planes_toggle.setChecked(True)
    qtbot.wait(20)

    assert workspace.size() == QSize(1100, height)
    assert workspace.plot.isVisible()
    assert workspace.plot.height() >= 180
    assert _visible_height(workspace, workspace.plot.viewport()) >= 110
    assert workspace.conjugate_planes.table.isVisible()
    assert _visible_height(workspace, workspace.conjugate_planes.table.viewport()) >= 30
    assert workspace.interaction_detail.isVisible() is interaction_visible
    assert workspace.ray_plot_details.isHidden()
    assert not workspace.conjugate_planes.details_toggle.isChecked()
    assert workspace.conjugate_planes.details.isHidden()
    assert workspace.conjugate_planes._worker is None


@pytest.mark.parametrize("height", (650, 550))
def test_conjugate_table_and_interaction_budget_can_release_space(workspace, qtbot, height):
    workspace.resize(1100, height)
    workspace.show()
    workspace.conjugate_planes_toggle.setChecked(True)
    assert workspace.interaction_detail_toggle.isChecked()
    workspace.interaction_detail_toggle.setChecked(False)
    qtbot.wait(20)
    assert workspace.interaction_detail.isHidden()

    splitter = workspace.ray_conjugate_splitter
    assert splitter.orientation() == Qt.Orientation.Vertical
    assert splitter.widget(0) is workspace.ray_plot_panel
    assert splitter.widget(1) is workspace.conjugate_planes
    assert splitter.handleWidth() >= 7
    splitter.setSizes((10000, 0))
    qtbot.wait(20)
    compact_height = workspace.conjugate_planes.table.height()
    splitter.setSizes((0, 10000))
    qtbot.wait(20)
    assert workspace.conjugate_planes.table.height() > compact_height
    assert workspace.conjugate_planes.table.maximumHeight() > 115
    assert workspace.plot.height() >= 180
    assert _visible_height(workspace, workspace.plot.viewport()) >= 110
    assert _visible_height(workspace, workspace.conjugate_planes.table.viewport()) >= 30
    assert workspace.size() == QSize(1100, height)

    workspace.interaction_detail_toggle.setChecked(True)
    qtbot.wait(20)
    assert workspace.interaction_detail.isVisible()
    assert workspace.conjugate_planes._worker is None


def test_hiding_details_or_results_retains_the_search_without_recalculation(workspace, qtbot):
    workspace.resize(1100, 650)
    workspace.show()
    workspace.conjugate_planes_toggle.setChecked(True)
    panel = workspace.conjugate_planes
    panel.table.setCurrentCell(3, 0)
    search, atlas = panel._search, panel._atlas
    cache = dict(panel._cache)
    details = panel.details.toPlainText()

    panel.details_toggle.setChecked(True)
    qtbot.wait(20)
    assert panel.details.isVisible()
    assert panel.details.toPlainText() == details
    panel.details_toggle.setChecked(False)
    assert panel.details.isHidden()
    workspace.conjugate_planes_toggle.setChecked(False)
    assert panel.isHidden()
    assert all(not marker.isVisible() for marker in workspace.conjugate_plane_overlay.items)
    workspace.conjugate_planes_toggle.setChecked(True)
    qtbot.wait(20)

    assert panel.isVisible()
    assert panel._search is search
    assert panel._atlas is atlas
    assert panel._cache == cache
    assert panel._reference_z_mm == 3026.4
    assert panel.table.rowCount() == 9
    assert panel.table.currentRow() == 3
    assert panel.details.toPlainText() == details
    assert all(marker.isVisible() for marker in workspace.conjugate_plane_overlay.items)
    assert panel._worker is None
    assert panel._pending is None
