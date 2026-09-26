"""Overlay regressions use captured synthetic geometry; no physics execution."""

from copy import deepcopy
from types import SimpleNamespace as NS

import numpy as np
from PySide6.QtCore import Qt
import pyqtgraph as pg
import pytest

from temsim.gui.accelerator_gap_overlay import (
    AcceleratorGapOverlay, accelerator_gap_records,
)


def _result(*, surface=False, offset=1.25):
    gun = NS(
        type_key="cold_feg" if surface else "thermionic", emitter=NS(surface_model=object() if surface else None),
        accelerator=NS(field_center_offset_mm=offset, stages=[
            NS(center_from_tip_mm=60.0, soft_edge_mm=4.0),
            NS(center_from_tip_mm=90.0, soft_edge_mm=3.0),
        ]),
    )
    return NS(state_snapshot=NS(electron_gun=gun))


def test_analytic_centres_and_support_match_captured_transport_inputs():
    result = _result()
    records = accelerator_gap_records(result)
    assert [r.center_z_mm for r in records] == [61.25, 91.25]
    assert [(r.start_z_mm, r.end_z_mm) for r in records] == [
        (57.25, 65.25), (88.25, 94.25),
    ]
    assert "Accelerator gap 2" in records[1].tooltip
    assert "Analytic potential transition" in records[1].tooltip
    # Detached records remain accurate for the result used to create them.
    result.state_snapshot.electron_gun.accelerator.stages[0].center_from_tip_mm = 80.0
    assert records[0].center_z_mm == 61.25
    assert accelerator_gap_records(result)[0].center_z_mm == 81.25


def test_solved_field_has_electrodes_only_and_ignores_analytic_offsets():
    result = _result(surface=True, offset=999.0)
    result.state_snapshot.electron_gun.accelerator.stages[0].soft_edge_mm = 999.0
    records = accelerator_gap_records(result)
    assert [r.center_z_mm for r in records] == [60.0, 90.0]
    assert all(r.start_z_mm is None and r.end_z_mm is None for r in records)
    assert "Electrode centre" in records[0].tooltip
    assert "not a field boundary" in records[0].tooltip


def test_flat_cold_source_also_displays_electrode_positions_not_analytic_bands():
    result = _result(surface=True, offset=999.)
    result.state_snapshot.electron_gun.emitter.surface_model = None
    records = accelerator_gap_records(result)
    assert [row.center_z_mm for row in records] == [60., 90.]
    assert all(row.start_z_mm is None for row in records)


@pytest.mark.parametrize("result", [None, NS(), NS(state_snapshot=NS())])
def test_absent_historical_geometry_does_not_consult_live_state(result):
    assert accelerator_gap_records(result) == ()


def test_invalid_geometry_never_becomes_invented_band():
    result = _result()
    accelerator = result.state_snapshot.electron_gun.accelerator
    accelerator.stages[0].soft_edge_mm = float("nan")
    accelerator.stages[1].center_from_tip_mm = float("inf")
    assert accelerator_gap_records(result) == ()
    accelerator.stages[0].soft_edge_mm = 0.0
    assert accelerator_gap_records(result) == ()


def test_graphics_reuse_toggle_ranges_and_payload_identity(qtbot):
    plot = pg.PlotWidget()
    qtbot.addWidget(plot)
    x, y = np.array([0.0, 1.0, 2.0]), np.array([-1.0, 0.0, 1.0])
    ray = plot.plot(x, y)
    ray_data = ray.getData()
    plot.autoRange()
    before_range = np.asarray(plot.viewRange())
    overlay = AcceleratorGapOverlay(plot)
    result = _result()
    overlay.sync(result)
    original_items = overlay.items
    assert len(original_items) == 4
    assert all(item not in plot.getViewBox().addedItems for item in original_items)
    plot.autoRange()
    np.testing.assert_allclose(plot.viewRange(), before_range)
    for value in (False, True):
        overlay.setVisible(value)
        assert overlay.items == original_items
        assert all(item.isVisible() == value for item in original_items)
        np.testing.assert_allclose(plot.viewRange(), before_range)
    overlay.sync(deepcopy(result))
    assert overlay.items == original_items
    assert all(not item.movable for item in original_items)
    assert all(item.acceptedMouseButtons() == Qt.MouseButton.NoButton for item in original_items)
    assert all(a is b for a, b in zip(ray_data, ray.getData()))
    np.testing.assert_array_equal(ray.getData()[0], x)
    np.testing.assert_array_equal(ray.getData()[1], y)


def test_changed_snapshot_replaces_only_changed_stage_and_clear_releases(qtbot):
    plot = pg.PlotWidget()
    qtbot.addWidget(plot)
    overlay = AcceleratorGapOverlay(plot)
    result = _result()
    overlay.sync(result)
    old_items = overlay.items
    result.state_snapshot.electron_gun.accelerator.stages[0].soft_edge_mm = 2.0
    overlay.sync(result, visible=False)
    assert overlay.items[2:] == old_items[2:]
    assert all(item not in plot.plotItem.items for item in old_items[:2])
    assert all(not item.isVisible() for item in overlay.items)
    overlay.sync(_result(surface=True))
    assert len(overlay.items) == 2
    assert all(isinstance(item, pg.InfiniteLine) for item in overlay.items)
    assert all(item not in plot.plotItem.items for item in old_items)
    final_items = overlay.items
    overlay.clear()
    assert overlay.records == () and overlay.items == ()
    assert all(item not in plot.plotItem.items for item in final_items)


def test_plot_clear_reattaches_cached_overlay_without_field_access(qtbot):
    class CapturedGun:
        type_key = "cold_feg"
        emitter = NS(surface_model=None)
        accelerator = NS(field_center_offset_mm=0.0, stages=[
            NS(center_from_tip_mm=75.0, soft_edge_mm=4.0),
        ])

        @property
        def electric_field(self):
            raise AssertionError("Display must never instantiate a physical field")

    plot = pg.PlotWidget()
    qtbot.addWidget(plot)
    overlay = AcceleratorGapOverlay(plot)
    result = NS(state_snapshot=NS(electron_gun=CapturedGun()))
    overlay.sync(result)
    items = overlay.items
    plot.clear()
    assert all(item.scene() is None for item in items)
    overlay.sync(result)
    assert overlay.items == items
    assert all(item in plot.plotItem.items for item in items)
    assert all(item not in plot.getViewBox().addedItems for item in items)
