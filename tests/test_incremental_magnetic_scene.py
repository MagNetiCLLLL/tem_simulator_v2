"""Incremental Qt magnetic graphics; all field/transport outputs are synthetic."""

from types import SimpleNamespace

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtWidgets import QVBoxLayout, QWidget

from temsim.diagnostics import ImagePlaneRotationRecord, LensFieldRecord
from temsim.gui.diagnostic_tabs import MagneticFieldView


def result(*, amplitude=1.0, lens_keys=("c1", "objective"), sample_z=50.0,
           plane_keys=("image",), formula="gaussian", colour="#38bdf8"):
    state = SimpleNamespace(amplitude=amplitude, lens_keys=lens_keys,
                            sample=SimpleNamespace(z_mm=sample_z), plane_keys=plane_keys,
                            formula=formula, colour=colour)
    simulation = SimpleNamespace(incident=SimpleNamespace(z=np.array([0.0, 100.0])), branches={})
    return SimpleNamespace(state_snapshot=state, simulation=simulation)


def records_for(state, z_mm):
    records = []
    for index, key in enumerate(state.lens_keys):
        center = 25.0 + index * 30.0
        values = state.amplitude * np.exp(-((z_mm - center) / 4.0) ** 2)
        records.append(LensFieldRecord(
            key=key, name=key.title(), colour=state.colour,
            formula_key=state.formula, formula_label=f"{state.formula} field",
            formula_expression="Synthetic GUI test field", formula_colour=state.colour,
            enabled=True, excitation_percent=10.0 * state.amplitude, polarity=1,
            field_polarity_status="test", field_polarity_source="test",
            center_z_mm=center, z_mm=z_mm, field_t=values,
            peak_t=float(np.max(values)), support_mm=(center - 5, center + 5),
            support_definition="Synthetic support", field_model_status="Test only",
            geometry_material_coupling="Test only", field_at_sample_t=state.amplitude * 0.2,
            sample_inside_numerical_support=False, focal_length_mm=state.amplitude,
            signed_field_integral_t_m=state.amplitude * 0.001,
            larmor_rotation_deg=state.amplitude * 5,
            cumulative_column_rotation_deg=state.amplitude * 10,
            spherical_aberration_mm=0.5,
        ))
    return np.sum([row.field_t for row in records], axis=0) if records else np.zeros_like(z_mm), tuple(records)


def planes_for(state):
    return tuple(ImagePlaneRotationRecord(
        key=key, name=key.title(), z_mm=80.0 + index,
        magnification=2.0, image_rotation_from_sample_deg=state.amplitude * 7,
        larmor_rotation_from_sample_deg=state.amplitude * 5,
        conjugacy_error_m=0.0, anisotropy_ratio=1.0,
    ) for index, key in enumerate(state.plane_keys))


@pytest.fixture
def view(qtbot, monkeypatch):
    widget = MagneticFieldView()
    qtbot.addWidget(widget)
    calls = []

    def fields(state, z):
        calls.append(("fields", state))
        return records_for(state, z)

    def planes(state):
        calls.append(("planes", state))
        return planes_for(state)

    monkeypatch.setattr("temsim.gui.diagnostic_tabs.lens_field_records", fields)
    monkeypatch.setattr("temsim.gui.diagnostic_tabs.image_plane_rotation_records", planes)
    widget.test_provider_calls = calls
    yield widget
    if hasattr(widget, "test_container"):
        widget.plot.setXLink(None)
        widget.setParent(None)
        del widget.test_container


def test_new_snapshot_updates_existing_curves_legend_lines_and_labels(view, monkeypatch):
    first = result()
    view.display_result(first)
    view.focus_component(SimpleNamespace(key="c1"))
    curves = dict(view._curves)
    total = view._total_curve
    specimen_line = view._sample_field_items[0]
    annotations = dict(view._rotation_by_key)
    support = view._support_item
    legend_items = tuple(view.legend.items)
    formula_samples = tuple(view._formula_samples)
    owned_items = tuple(view.plot.plotItem.items)
    monkeypatch.setattr(view.plot, "clear", lambda: pytest.fail("Do not clear a completed magnetic scene"))
    monkeypatch.setattr(view.plot, "plot", lambda *_a, **_k: pytest.fail("Same lens keys must reuse curves"))
    monkeypatch.setattr(view.plot, "addItem", lambda *_a, **_k: pytest.fail("Same keys must reuse graphics"))

    latest = result(amplitude=2.0, sample_z=52.0)
    view.display_result(latest)

    assert view._curves == curves and view._total_curve is total
    assert view._sample_field_items == [specimen_line]
    assert view._rotation_by_key == annotations and view._support_item is support
    assert tuple(view.legend.items) == legend_items
    assert tuple(view._formula_samples) == formula_samples
    assert tuple(view.plot.plotItem.items) == owned_items
    assert specimen_line.value() == 52.0
    assert "+0.8 T" in specimen_line.toolTip()
    assert "20%" in curves["c1"].toolTip()
    assert "+10°" in annotations[("lens", "c1")].toPlainText()
    assert "+14°" in annotations[("plane_label", "image")].toPlainText()
    assert "excitation 20%" in view.diagnostic_text("c1")
    assert "20%" in view.summary.text()
    for row in view._records:
        np.testing.assert_array_equal(curves[row.key].getData()[1], row.field_t)
    assert view.test_provider_calls == [("fields", first.state_snapshot), ("planes", first.state_snapshot),
                                        ("fields", latest.state_snapshot), ("planes", latest.state_snapshot)]


def test_added_removed_lenses_planes_and_formulas_reconcile_without_scene_clear(view, monkeypatch):
    view.display_result(result())
    view.focus_component(SimpleNamespace(key="objective"))
    survivor = view._curves["c1"]
    removed = view._curves["objective"]
    removed_label = view._rotation_by_key[("lens", "objective")]
    removed_plane = view._rotation_by_key[("plane_line", "image")]
    removed_formula = view._formula_samples[0]
    selected_support = view._support_item
    monkeypatch.setattr(view.plot, "clear", lambda: pytest.fail("Key changes require reconciliation, not plot.clear"))

    view.display_result(result(lens_keys=("c1", "p1"), plane_keys=("camera",), formula="map"))

    assert set(view._curves) == {"c1", "p1"}
    assert view._curves["c1"] is survivor
    for item in (removed, removed_label, removed_plane, removed_formula, selected_support):
        assert item.scene() is None
    assert set(view._rotation_by_key) == {("lens", "c1"), ("lens", "p1"),
                                          ("plane_line", "camera"), ("plane_label", "camera")}
    assert view._selected_key is None and view._support_item is None
    assert len(view.legend.items) == 2
    assert view.legend.items[1][1].text == "map field"


def test_subsequent_publications_preserve_user_y_range_and_linked_ray_x(view, qtbot):
    source = pg.PlotWidget()
    container = QWidget()
    view.test_container = container  # Keep the reparented fixture's native owner alive through teardown.
    qtbot.addWidget(container)
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(source)
    layout.addWidget(view)
    container.resize(1600, 900)
    container.show()
    source.getAxis("left").setWidth(65)
    view.link_axial_axis(source)
    source.setXRange(15.0, 35.0, padding=0)
    qtbot.wait(10)
    view.display_result(result())
    first_y = tuple(view.plot.viewRange()[1])
    assert first_y[0] < 0.1 and first_y[1] > 0.9  # Initial Y fit still includes the total field.
    view.plot.setYRange(-0.15, 0.25, padding=0)
    source.setXRange(31.0, 39.0, padding=0)
    qtbot.wait(10)
    before_x = tuple(view.plot.viewRange()[0])
    view.display_result(result(amplitude=100.0))
    np.testing.assert_allclose(view.plot.viewRange()[1], (-0.15, 0.25), rtol=0, atol=1e-12)
    np.testing.assert_allclose(view.plot.viewRange()[0], before_x, rtol=0, atol=1e-12)
    np.testing.assert_allclose(view.plot.viewRange()[0], source.viewRange()[0], rtol=0, atol=1e-9)


def test_toggles_and_curve_click_connections_survive_repeated_updates(view):
    view.display_result(result())
    view.show_individual.setChecked(False)
    view.show_rotation_labels.setChecked(False)
    view.focus_component(SimpleNamespace(key="c1"))
    chosen = []
    view.component_selected.connect(chosen.append)
    curve = view._curves["c1"]
    count = len(view.plot.plotItem.items)
    for value in range(2, 8):
        view.display_result(result(amplitude=float(value)))
        assert len(view.plot.plotItem.items) == count
    assert view._curves["c1"] is curve and curve.isVisible()
    assert not view._curves["objective"].isVisible()
    assert all(not item.isVisible() for item in view._rotation_items)
    curve.sigClicked.emit(curve, None)
    assert chosen == ["c1"]  # No duplicate connection per published result.


def test_pending_diagnostics_and_clear_never_expose_stale_fields(view, monkeypatch):
    view.display_result(result(amplitude=123.0))
    total, old_items = view._total_curve, tuple(view.plot.plotItem.items)
    view.mark_presentation_pending()
    assert "pending" in view.diagnostic_text("c1")
    assert "123" not in view.diagnostic_text("c1")
    monkeypatch.setattr(view.plot, "clear", lambda: pytest.fail("Clear only magnetic-owned items"))
    view.display_result(SimpleNamespace(state_snapshot=None))
    assert view._records == () and view._plane_records == ()
    assert not view._presentation_pending and not view._has_field_scene
    assert view._total_curve is None and total.scene() is None
    assert view.legend.items == []
    assert "123" not in view.heading.text()
    assert all(item.scene() is None for item in old_items if item is not view.legend)
    view.display_result(result())
    assert view._has_field_scene and view._total_curve is not total


def test_formula_colour_update_reuses_legend_sample_with_latest_colour(view):
    view.display_result(result())
    curve, sample = view._curves["c1"], view._formula_samples[0]
    legend_items = tuple(view.legend.items)
    view.display_result(result(colour="#ef4444"))
    assert view._curves["c1"] is curve and view._formula_samples[0] is sample
    assert tuple(view.legend.items) == legend_items
    assert curve.opts["pen"].color().name() == "#ef4444"
    assert sample.opts["pen"].color().name() == "#ef4444"


def test_selected_support_and_curve_domain_follow_latest_geometry(view):
    view.display_result(result())
    view.focus_component(SimpleNamespace(key="c1"))
    support, curve = view._support_item, view._curves["c1"]
    assert support.getRegion() == (20.0, 30.0)
    latest = result(lens_keys=("objective", "c1"))
    latest.simulation.incident.z = np.array([-50.0, 150.0])
    view.display_result(latest)
    assert view._support_item is support and view._curves["c1"] is curve
    assert support.getRegion() == (50.0, 60.0)
    assert curve.getData()[0][0] == -50.0 and curve.getData()[0][-1] == 150.0
    assert view._rotation_by_key[("lens", "c1")].pos().x() == 55.0
