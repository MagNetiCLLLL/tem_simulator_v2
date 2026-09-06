"""Scalar live edits preserve validation without rebuilding the instrument."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QSettings

from temsim.interactive_calculation import (
    CalculationRange, available_controls, apply_live_tuning_values,
)
from temsim.optics.model import Aperture


@pytest.fixture
def scalar_state():
    return SimpleNamespace(
        lenses=[SimpleNamespace(key="lens", name="Test lens", percent=50., max_percent=100., enabled=True)],
        apertures=[Aperture("Test aperture", "aperture", 10., .5, enabled=True)],
        recording_planes=[], electron_gun=SimpleNamespace(),
        sample=SimpleNamespace(z_mm=20.), _resolved_assembly=object(),
    )


def axes(state):
    controls = available_controls(state)
    lens = next(c for c in controls if c.group == "lens")
    aperture = next(c for c in controls if c.field == "diameter_mm")
    return CalculationRange(lens, 0, 120), CalculationRange(aperture, -1, 2)


def test_scalar_edits_match_assignment_and_preserve_assembly(scalar_state):
    from temsim.interactive_calculation import _assign
    old = deepcopy(scalar_state)
    geometry = scalar_state._resolved_assembly
    edits = tuple(zip(axes(scalar_state), (65., .2)))
    for axis, value in edits:
        _assign(old, axis.control, value)
    apply_live_tuning_values(scalar_state, edits)
    assert vars(old.lenses[0]) == vars(scalar_state.lenses[0])
    assert vars(old.apertures[0]) == vars(scalar_state.apertures[0])
    assert scalar_state.apertures[0].radius_mm == .1
    assert scalar_state._resolved_assembly is geometry


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -10., 110., 200.])
def test_failed_batch_does_not_apply_an_earlier_valid_scalar(scalar_state, bad):
    lens, aperture = axes(scalar_state)
    with pytest.raises(ValueError):
        apply_live_tuning_values(scalar_state, ((aperture, .2), (lens, bad)))
    assert scalar_state.apertures[0].diameter_mm == 1.
    assert scalar_state.lenses[0].percent == 50.


@pytest.mark.parametrize("kind", ["disabled", "missing", "stage", "field", "detector", "duplicate", "negative_aperture"])
def test_live_control_scope_is_checked_without_snapshots(scalar_state, kind):
    lens, aperture = axes(scalar_state)
    edits = ((lens, 60.),)
    if kind == "disabled":
        scalar_state.lenses[0].enabled = False
    elif kind == "missing":
        lens = replace(lens, control=replace(lens.control, key="missing"))
        edits = ((lens, 60.),)
    elif kind in {"stage", "field", "detector"}:
        change = {"stage": {"stage": "readout"}, "field": {"field": "max_percent"}, "detector": {"group": "detector"}}[kind]
        lens = replace(lens, control=replace(lens.control, **change))
        edits = ((lens, 60.),)
    elif kind == "duplicate":
        edits = ((lens, 60.), (lens, 70.))
    else:
        edits = ((lens, 60.), (aperture, -.1))
    with pytest.raises(ValueError):
        apply_live_tuning_values(scalar_state, edits)
    assert scalar_state.lenses[0].percent == 50.
    assert scalar_state.apertures[0].diameter_mm == 1.


def test_main_window_live_edit_does_not_snapshot_or_rebuild_controls(qtbot, monkeypatch, tmp_path):
    from temsim.gui import main_window as shell, interactive_calculation as gui, calculation_controller as controller
    from temsim.runtime_parameters import runtime_targets
    settings = QSettings(str(tmp_path / "ui.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(shell, "QSettings", lambda: settings)
    monkeypatch.setattr(gui, "QSettings", lambda: settings)
    monkeypatch.setattr(controller, "default_artifact_cache_root", lambda: tmp_path / "cache")
    monkeypatch.setattr(shell.MainWindow, "INITIAL_PREVIEW_DELAY_MS", 60000)
    window = shell.MainWindow()
    qtbot.addWidget(window)
    window.preview_timer.stop()
    panel = window.parameter_panel
    key = "objective_lens"
    target = runtime_targets(window.state)[key]
    panel.set_context("Objective", target, None, (), None)
    before_widgets = dict(panel._quick_widgets)
    before_items = [panel.runtime_table.item(row, 1) for row in range(panel.runtime_table.rowCount())]
    old_high = object()
    window.workspace._high_accuracy_result = old_high
    window.workspace._high_accuracy_current = False
    control = next(c for c in available_controls(window.state) if c.key == key and c.field == "percent")
    axis = CalculationRange(control, 0., 100.)
    monkeypatch.setattr(controller.CalculationController, "_calculation_snapshot",
                        lambda *a: pytest.fail("A scalar slider update must not snapshot the instrument"))
    monkeypatch.setattr(panel, "_load_quick_controls", lambda: pytest.fail("Must retain quick widgets"))
    window._apply_interactive_tuning(((axis, 67.5),))
    window.preview_timer.stop()
    assert window.state.objective_lens.percent == 67.5
    assert panel.lens_excitation.value() == 67.5
    assert dict(panel._quick_widgets) == before_widgets
    assert [panel.runtime_table.item(row, 1) for row in range(panel.runtime_table.rowCount())] == before_items
    assert window.workspace._high_accuracy_result is old_high


def test_quick_aperture_values_update_in_place(qtbot, monkeypatch):
    from temsim.gui.parameter_panel import ParameterPanel
    from temsim.optics.column import default_state
    from temsim.runtime_parameters import runtime_targets
    state = default_state()
    control = next(c for c in available_controls(state) if c.group == "aperture" and c.field == "diameter_mm")
    target = runtime_targets(state)[control.key]
    panel = ParameterPanel()
    qtbot.addWidget(panel)
    panel.set_context("Aperture", target, None, (), None)
    widgets = dict(panel._quick_widgets)
    changed = []
    panel.runtime_changed.connect(changed.append)
    monkeypatch.setattr(panel, "_load_quick_controls", lambda: pytest.fail("Must retain editor widgets"))
    target.obj.diameter_mm = .002
    panel.refresh_live_values({control.key})
    assert panel._quick_widgets == widgets
    assert panel._quick_widgets["diameter_mm"].value() == pytest.approx(2.)
    assert changed == []


def test_configured_aperture_limit_rejects_entire_live_batch():
    from temsim.interactive_calculation import _assign
    from temsim.optics.column import default_state
    state = default_state()
    controls = available_controls(state)
    lens = next(c for c in controls if c.group == "lens")
    target = state.condenser_aperture_2
    aperture = next(c for c in controls if c.key == target.key and c.field == "diameter_mm")
    maximum = 2.0 * target.maximum_radius_mm
    old_lens = getattr(next(l for l in state.lenses if l.key == lens.key), "percent")
    old_diameter = target.diameter_mm
    lens_axis, aperture_axis = CalculationRange(lens, 0., 100.), CalculationRange(aperture, 0., 2. * maximum)
    with pytest.raises(ValueError, match="clear bore"):
        apply_live_tuning_values(state, ((lens_axis, 50.), (aperture_axis, 2. * maximum)))
    assert getattr(next(l for l in state.lenses if l.key == lens.key), "percent") == old_lens
    assert target.diameter_mm == old_diameter
    with pytest.raises(ValueError, match="clear bore"):
        _assign(state, aperture, 2. * maximum)
    apply_live_tuning_values(state, ((aperture_axis, maximum),))
    assert target.diameter_mm == maximum
