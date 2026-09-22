"""Wire spatial coverage to displayed results, not cursor or requested planes."""
from types import SimpleNamespace

import numpy as np
from test_incremental_ray_scene import _result, make_workspace


def completed(target=16., resumable=None):
    # Explicit UI-only data: the physical restart path has separate tests.
    result = _result(end_z=target)
    result.simulation.metrics.update(section_target_z_mm=target,
        section_resumable_through_z_mm=target if resumable is None else resumable)
    result.simulation.section_checkpoint = SimpleNamespace(
        gun_trace=SimpleNamespace(z_mm=np.array([0., 4.])))
    return result


def test_result_cutoff_selection_cursor_and_staleness_stay_separate(make_workspace):
    view = make_workspace()
    page = view.interactive_calculation
    view.display_result(completed(resumable=12.), "Preview")
    bar = view.ray_calculation_extent
    assert bar.extent["completed_z_mm"] == 16.
    assert bar.extent["resumable_z_mm"] == 12.
    assert bar.extent["requested_z_mm"] is None
    assert not bar.extent["stale"]
    assert view.ray_primary_panel.layout().indexOf(bar) == view.ray_primary_panel.layout().indexOf(view.plot) + 1
    page.section_group.setChecked(True)
    page.section_z.setValue(19.)
    assert bar.extent["requested_z_mm"] == 19.
    assert bar.extent["completed_z_mm"] == 16.
    view.jump_to_ray_position(7., activate_tab=False)
    assert bar.extent["completed_z_mm"] == 16.
    # A changed physical input marks the old calculation, not the chosen plane.
    view.mark_ray_stale(SimpleNamespace(electron_gun=SimpleNamespace(
        type_key="fixture", display_name="Electron source")))
    assert bar.extent["stale"]
    view.display_result(completed(target=19.), "Preview")
    assert bar.extent["completed_z_mm"] == 19.
    assert not bar.extent["stale"]
    page.section_group.setChecked(False)
    assert bar.extent["requested_z_mm"] is None
    assert bar.extent["completed_z_mm"] == 19.


def test_request_without_completed_result_never_fills_bar(make_workspace):
    view = make_workspace()
    page = view.interactive_calculation
    page.section_group.setChecked(True)
    page.section_z.setValue(19.)
    assert view.ray_calculation_extent.extent["requested_z_mm"] == 19.
    assert view.ray_calculation_extent.extent["completed_z_mm"] is None
    # Historical diagram data are still displayed without inventing completion.
    view.display_result(_result(), "Preview")
    assert view.ray_calculation_extent.extent["completed_z_mm"] is None


def test_recapturing_settings_updates_the_requested_marker(make_workspace):
    from temsim.optics.column import default_state
    view = make_workspace()
    page = view.interactive_calculation
    page.section_group.setChecked(True)
    page.section_z.setValue(19.)
    page.set_source(default_state())
    assert view.ray_calculation_extent.extent["requested_z_mm"] == page.section_z.value()
    assert view.ray_calculation_extent.extent["completed_z_mm"] is None
