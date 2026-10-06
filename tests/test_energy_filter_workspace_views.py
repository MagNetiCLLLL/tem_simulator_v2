"""Branch views reuse accepted products and geometry without another solver."""
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QSettings

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.gui.diagnostic_tabs import EnergyFilterView, PhysicalLayoutView
from temsim.gui.instrument_configuration_dialog import InstrumentConfigurationDialog, UNITS
from temsim.gui.visualization import VisualizationWorkspace
from temsim.optics.column import default_state


@pytest.fixture
def filter_context():
    catalog = AssemblyCatalog()
    state = default_state()
    assembly = catalog.apply(state, AssemblySelection("FEG", "C3", "Energy Filter"))
    return catalog, state, assembly


def _result(state, assembly):
    branch = SimpleNamespace(
        paths_u_mm=[np.array([0., 100., 220.])],
        paths_v_mm=[np.array([0., 0., -90.])], colours=["#22d3ee"],
        status="cached branch", eels_forward=None, eftem_image=None,
    )
    return SimpleNamespace(state_snapshot=state, assembly=assembly,
                           layout=state._resolved_optics_layout, energy_filter=branch,
                           calculated_products=frozenset({"energy_filter"}))


def test_branch_workspace_tabs_share_results_selection_and_staleness(qtbot, filter_context):
    _, state, assembly = filter_context
    result = _result(state, assembly)
    workspace = VisualizationWorkspace()
    qtbot.addWidget(workspace)
    owner, rays = workspace.energy_filter, workspace.energy_filter_rays
    structure = workspace.physical_layout.energy_filter_structure
    assert len({id(owner.plot), id(rays.plot), id(structure.plot)}) == 3
    assert workspace.ray_result_tabs.tabText(workspace.ray_result_tabs.indexOf(rays)) == "Energy Filter rays"
    assert workspace.physical_layout.tabs.tabText(workspace.physical_layout.tabs.indexOf(structure)) == "Energy Filter"
    assert [owner.output_tabs.tabText(i) for i in range(3)] == ["Physical + rays", "EELS spectrum", "EFTEM image"]
    for mirror in (rays, structure):
        assert not hasattr(mirror, "output_tabs")
        assert not hasattr(mirror, "calculate_button")
        assert not hasattr(mirror, "spectrum_plot")

    requests, selected = [], []
    workspace.calculation_requested.connect(requests.append)
    workspace.component_selected.connect(selected.append)
    # Existing ray-scope publication may carry a cached or freshly executed
    # filter product. Both routes must publish the identical object to the mirror.
    workspace._display_ray_scope_products(result, "High accuracy")
    workspace.physical_layout.display_result(result)
    assert owner._result is rays._result is structure._result is result
    np.testing.assert_array_equal(rays._ray_items[0].getData()[0], result.energy_filter.paths_u_mm[0])
    assert len(owner._ray_items) == len(rays._ray_items) == 1
    assert not structure._ray_items
    assert "Structure only" in structure.summary.text()
    assert len(structure._multipole_housing_items) == 20
    assert len(structure._device_body_items) == 12

    key = "energy_filter_multipole_04"
    rays._component_clicked(None, [SimpleNamespace(data=lambda: key)])
    assert selected == [key]
    assert all(view._selected_key == key and view._selection_item is not None
               for view in (owner, rays, structure))
    workspace.ray_result_tabs.setCurrentWidget(rays)
    workspace.physical_layout.tabs.setCurrentWidget(structure)
    assert not requests

    retained_curve = rays._ray_items[0]
    result_without_filter = SimpleNamespace(energy_filter=None)
    workspace._display_ray_scope_products(result_without_filter, "High accuracy")
    assert rays._ray_items[0] is retained_curve
    assert owner._result is rays._result is result
    assert "inputs changed" in rays.summary.text()
    assert "inputs changed" not in structure.summary.text()

    owner.display_result(None)
    assert rays._result is None and not rays._ray_items
    assert "not installed" in rays.summary.text()
    assert not requests


def test_filter_structure_is_available_in_configuration_review(qtbot, tmp_path, filter_context):
    catalog, state, assembly = filter_context
    before = state.to_dict()
    calls = []
    settings = QSettings(str(tmp_path / "review.ini"), QSettings.Format.IniFormat)
    dialog = InstrumentConfigurationDialog(catalog, lambda: state, calls.append, settings=settings)
    qtbot.addWidget(dialog)
    review = dialog.review
    structure = review.energy_filter_structure
    assert review.tabs.isTabVisible(review.tabs.indexOf(structure))
    assert review.model_editor is None
    assert structure._result is None
    assert not structure._multipole_housing_items
    assert review.assembly_3d.mesh_builds == 0
    assert not hasattr(structure, "calculate_button")

    filter_row = next(i for i, (key, _) in enumerate(UNITS) if key == "energy_filter")
    dialog.table.setCurrentCell(filter_row, 0)
    assert review.tabs.currentWidget() is structure
    assert structure._result.state_snapshot is state
    assert len(structure._multipole_housing_items) == 20
    assert not structure._ray_items
    assert review.reveal_component(assembly.part("energy_filter_slit"))
    assert structure._selected_key == "energy_filter_slit"
    assert structure._selection_item is not None
    assert not calls and state.to_dict() == before

    # Review's candidate is authoritative, but must not erase unrelated runtime
    # geometry such as the aperture opening inherited from its preview.
    review.assembly_3d.set_runtime_values({"objective_aperture": {"radius": 0.25}})
    state.energy_filter.energy_slit.gap_m = 0.0013
    state.energy_filter.energy_slit.centre_m = 0.0002
    dialog._show_preview(state)
    assert review.assembly_3d._runtime_values["energy_filter_slit"] == {"gap_m": 0.0013, "centre_m": 0.0002}
    assert review.assembly_3d._runtime_values["objective_aperture"] == {"radius": 0.25}


def test_physical_snapshot_fills_slit_geometry_without_overwriting_live_runtime(qtbot, filter_context):
    _, state, assembly = filter_context
    state.energy_filter.energy_slit.gap_m = 0.0013
    state.energy_filter.energy_slit.centre_m = 0.0002
    result = _result(state, assembly)
    view = PhysicalLayoutView()
    qtbot.addWidget(view)
    aperture = {"radius": 0.25, "x_shift": 0.01}
    view.assembly_3d.set_assembly(assembly, {"objective_aperture": aperture})
    view.display_result(result)
    assert view.assembly_3d._runtime_values == {
        "objective_aperture": aperture,
        "energy_filter_slit": {"gap_m": 0.0013, "centre_m": 0.0002},
    }
    live_slit = {"gap_m": 0.002, "centre_m": -0.0003}
    view.assembly_3d.set_runtime_values({"objective_aperture": aperture, "energy_filter_slit": live_slit})
    view.display_result(result)
    assert view.assembly_3d._assembly is assembly
    assert view.assembly_3d._runtime_values == {
        "objective_aperture": aperture, "energy_filter_slit": live_slit,
    }
    assert state.energy_filter.energy_slit.gap_m == 0.0013
    assert state.energy_filter.energy_slit.centre_m == 0.0002


def test_branch_mirrors_show_uninstalled_state_without_scientific_outputs(qtbot):
    for show_rays in (True, False):
        view = EnergyFilterView(show_rays=show_rays, show_outputs=False)
        qtbot.addWidget(view)
        view.display_result(SimpleNamespace(state_snapshot=default_state()))
        assert "not installed" in view.summary.text()
        assert not view._multipole_housing_items and not view._ray_items
        assert not hasattr(view, "eftem_plot")


def test_old_named_layouts_and_new_branch_tabs_restore_without_calculation(qtbot, monkeypatch, tmp_path):
    from temsim.gui import main_window as shell, interactive_calculation

    settings = QSettings(str(tmp_path / "layouts.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(shell, "QSettings", lambda: settings)
    monkeypatch.setattr(interactive_calculation, "QSettings", lambda: settings)
    monkeypatch.setattr(shell.MainWindow, "INITIAL_PREVIEW_DELAY_MS", 60000)
    monkeypatch.setattr(shell.MainWindow, "_apply_state_operating_modes", lambda *_: object())
    window = shell.MainWindow()
    window.preview_timer.stop()
    qtbot.addWidget(window)
    monkeypatch.setattr(window.calculations.pool, "start", lambda *_: pytest.fail("Layout must not calculate"))
    workspace, manager = window.workspace, window.workspace_layouts
    before = window.state.to_dict()
    requests = []
    workspace.calculation_requested.connect(requests.append)
    # Mirrors must not duplicate this object name or the layout manager would
    # silently stop saving/restoring the original EELS/EFTEM tab.
    assert manager.tabs["energyFilterOutputTabs"] is workspace.energy_filter.output_tabs
    old = manager._snapshot()
    for physical_title, expected in (("2D section", "2D"), ("3D model editor", "3D Parts"), ("3D", "3D")):
        old["tabs"].update(physicalLayoutTabs=physical_title,
                           rayResultTabs="Cached signals", energyFilterOutputTabs="EFTEM image")
        manager._apply(old)
        physical = workspace.physical_layout.tabs
        assert physical.tabText(physical.currentIndex()) == expected
        assert workspace.ray_result_tabs.currentWidget() is workspace.interactive_calculation.readout_panel
        assert workspace.energy_filter.output_tabs.currentIndex() == 2

    workspace.ray_result_tabs.setCurrentWidget(workspace.energy_filter_rays)
    workspace.physical_layout.tabs.setCurrentWidget(workspace.physical_layout.energy_filter_structure)
    saved = manager.save_as("Energy Filter views")
    manager.select("default")
    manager.select(saved)
    assert workspace.ray_result_tabs.currentWidget() is workspace.energy_filter_rays
    assert workspace.physical_layout.tabs.currentWidget() is workspace.physical_layout.energy_filter_structure
    assert not requests and not window.preview_timer.isActive()
    assert window.state.to_dict() == before
