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
    from temsim.magnetic_field_scene import MagneticDiagnosticProfile, MagneticSourceRegion
    widget = MagneticFieldView()
    qtbot.addWidget(widget)
    calls = []

    def fields(state, z):
        calls.append(("fields", state))
        return records_for(state, z)

    def planes(state):
        calls.append(("planes", state))
        return planes_for(state)

    def prepare(state, *, z_limits_mm):
        calls.append(("prepare", state))
        regions = tuple(MagneticSourceRegion(key, key.title(), "lens", np.asarray(((-.001, -.001, .0), (.001, .001, .1))))
                        for key in state.lens_keys)
        regions += (MagneticSourceRegion("deflector", "Deflector", "deflector", np.asarray(((-.001, -.001, .03), (.001, .001, .04)))),
                    MagneticSourceRegion("stigmator", "Stigmator", "stigmator", np.asarray(((-.001, -.001, .06), (.001, .001, .07)))))
        return SimpleNamespace(state=state, source_keys=tuple(r.key for r in regions), source_regions=regions,
                               notes=("Synthetic combined field; not transport validation.",))

    def sample(scene, z):
        state = scene.state
        calls.append(("sample", state))
        bz, _ = records_for(state, z)
        bx = np.full(len(z), state.amplitude*.01)
        by = np.full(len(z), state.amplitude*-.02)
        if getattr(state, "zero_transverse_axis", False):
            bx[:] = by[:] = 0.
        rms = np.sqrt(bx*bx + by*by + (.003*state.amplitude)**2)
        return MagneticDiagnosticProfile(z, np.column_stack((bx, by, bz)), rms,
                                         np.full(len(z), state.amplitude*.1),
                                         np.full(len(z), 1e-5), np.ones(len(z), dtype=bool))

    monkeypatch.setattr("temsim.gui.diagnostic_tabs.lens_field_records", fields)
    monkeypatch.setattr("temsim.gui.diagnostic_tabs.image_plane_rotation_records", planes)
    monkeypatch.setattr("temsim.magnetic_field_scene.prepare_magnetic_scene", prepare)
    monkeypatch.setattr("temsim.magnetic_field_scene.sample_magnetic_diagnostic", sample)
    widget.test_provider_calls = calls
    yield widget
    widget.field_lines.set_active(False)
    qtbot.waitUntil(lambda: widget._profile_worker is None, timeout=5000)
    if hasattr(widget, "test_container"):
        widget.plot.setXLink(None)
        widget.setParent(None)
        del widget.test_container


def publish(view, qtbot, latest):
    view.display_result(latest)
    qtbot.waitUntil(lambda: view._profile_worker is None, timeout=5000)
    assert view._profile is not None, view.details.toPlainText()


def test_new_snapshot_updates_four_combined_curves_without_graphics_recreation(view, qtbot, monkeypatch):
    first = result()
    publish(view, qtbot, first)
    curves = dict(view._curves)
    specimen_line = view._sample_field_items[0]
    legend_items = tuple(view.legend.items)
    owned_items = tuple(view.plot.plotItem.items)
    monkeypatch.setattr(view.plot, "clear", lambda: pytest.fail("Do not clear a completed magnetic scene"))
    monkeypatch.setattr(view.plot, "plot", lambda *_a, **_k: pytest.fail("Combined curves must be reused"))
    monkeypatch.setattr(view.plot, "addItem", lambda *_a, **_k: pytest.fail("Existing graphics must be reused"))
    latest = result(amplitude=2., sample_z=52.)
    publish(view, qtbot, latest)
    assert view._curves == curves
    assert view._sample_field_items == [specimen_line]
    assert tuple(view.legend.items) == legend_items
    assert tuple(view.plot.plotItem.items) == owned_items
    assert specimen_line.value() == 52.
    assert set(curves) == {"bz", "bu", "bv", "transverse_rms"}
    np.testing.assert_allclose(curves["bu"].getData()[1], .02)
    np.testing.assert_allclose(curves["bv"].getData()[1], -.04)
    np.testing.assert_array_equal(curves["bz"].getData()[1], records_for(latest.state_snapshot, view._profile.z_mm)[0])
    assert "excitation 20%" in view.diagnostic_text("c1")
    assert len(view.test_provider_calls) == 8


def test_component_changes_and_focus_preserve_the_combined_scene(view, qtbot, monkeypatch):
    publish(view, qtbot, result())
    curves = dict(view._curves)
    data = {key: curve.getData()[1].copy() for key, curve in curves.items()}
    ranges = view.plot.viewRange()
    calls = len(view.test_provider_calls)
    view.focus_component(SimpleNamespace(key="c1"))
    view.focus_component(SimpleNamespace(key="deflector"))
    assert len(view.test_provider_calls) == calls
    assert view.plot.viewRange() == ranges
    assert all(curve.isVisible() for curve in curves.values())
    for key, curve in curves.items():
        np.testing.assert_array_equal(curve.getData()[1], data[key])
    assert "Deflector | deflector | included in the combined" in view.diagnostic_text("deflector")
    assert not hasattr(view, "show_individual")
    monkeypatch.setattr(view.plot, "clear", lambda: pytest.fail("Do not clear combined graphics"))
    publish(view, qtbot, result(lens_keys=("c1", "p1"), plane_keys=("camera",)))
    assert view._curves == curves
    assert len(view.legend.items) == 4
    assert {record.key for record in view._records} == {"c1", "p1"}


def test_projection_only_rotates_cached_xy_vectors_and_3d_view(view, qtbot, monkeypatch):
    publish(view, qtbot, result())
    calls = len(view.test_provider_calls)
    angles = []
    monkeypatch.setattr(view.field_lines, "set_projection_angle", angles.append)
    curves = dict(view._curves)
    bz = curves["bz"].getData()[1].copy()
    rms = curves["transverse_rms"].getData()[1].copy()
    view.set_projection_angle(90.)
    np.testing.assert_allclose(curves["bu"].getData()[1], -.02, atol=1e-16)
    np.testing.assert_allclose(curves["bv"].getData()[1], -.01, atol=1e-16)
    view.set_projection_angle(180.)
    np.testing.assert_allclose(curves["bu"].getData()[1], -.01, atol=1e-16)
    np.testing.assert_allclose(curves["bv"].getData()[1], .02, atol=1e-16)
    np.testing.assert_array_equal(curves["bz"].getData()[1], bz)
    np.testing.assert_array_equal(curves["transverse_rms"].getData()[1], rms)
    assert angles == [90., 180.]
    assert len(view.test_provider_calls) == calls
    assert view._curves == curves


def test_finite_radius_curve_exposes_stigmator_when_on_axis_transverse_field_is_zero(view, qtbot):
    latest = result()
    latest.state_snapshot.zero_transverse_axis = True
    publish(view, qtbot, latest)
    np.testing.assert_array_equal(view._curves["bu"].getData()[1], 0.)
    np.testing.assert_array_equal(view._curves["bv"].getData()[1], 0.)
    assert np.all(view._curves["transverse_rms"].getData()[1] > 0)
    assert "eight-point ring" in view._curves["transverse_rms"].toolTip()
    assert "10" in view.status.text()


def test_publications_preserve_y_range_and_both_views_follow_linked_ray_x(view, qtbot, monkeypatch):
    assert view.plot.getAxis("bottom").labelUnits == "m"
    assert view.plot.getAxis("bottom").scale == 1e-3
    source = pg.PlotWidget()
    container = QWidget()
    view.test_container = container
    qtbot.addWidget(container)
    layout = QVBoxLayout(container)
    layout.addWidget(source)
    layout.addWidget(view)
    container.resize(1200, 800)
    container.show()
    source.getAxis("left").setWidth(65)
    intervals = []
    monkeypatch.setattr(view.field_lines, "set_view_range_mm", lambda axial, transverse: intervals.append(tuple(axial)))
    view.link_axial_axis(source)
    source.setXRange(15., 35., padding=0)
    qtbot.wait(10)
    publish(view, qtbot, result())
    assert view.plot.viewRange()[1][1] > .9
    view.plot.setYRange(-.15, .25, padding=0)
    source.setXRange(31., 39., padding=0)
    qtbot.wait(10)
    before_x = tuple(view.plot.viewRange()[0])
    publish(view, qtbot, result(amplitude=100.))
    np.testing.assert_allclose(view.plot.viewRange()[1], (-.15, .25), atol=1e-12)
    np.testing.assert_allclose(view.plot.viewRange()[0], before_x, atol=1e-12)
    np.testing.assert_allclose(view.plot.viewRange()[0], source.viewRange()[0], atol=1e-9)
    np.testing.assert_allclose(intervals[-1], (31., 39.), atol=1e-9)
    view.plot.setXRange(42., 58., padding=0)
    qtbot.wait(10)
    np.testing.assert_allclose(source.viewRange()[0], (42., 58.), atol=1e-9)
    np.testing.assert_allclose(intervals[-1], (42., 58.), atol=1e-9)


def test_pending_and_clear_remove_stale_fields_without_plot_clear(view, qtbot, monkeypatch):
    publish(view, qtbot, result(amplitude=123.))
    old_items = tuple(view.plot.plotItem.items)
    view.mark_presentation_pending()
    assert "pending" in view.diagnostic_text("c1")
    assert "123" not in view.diagnostic_text("c1")
    assert view._profile is None
    monkeypatch.setattr(view.plot, "clear", lambda: pytest.fail("Remove only owned items"))
    view.display_result(SimpleNamespace(state_snapshot=None))
    assert view._records == () and view._plane_records == ()
    assert not view._presentation_pending and not view._has_field_scene
    assert view._curves == {} and view.legend.items == []
    assert all(item.scene() is None for item in old_items if item is not view.legend)
    publish(view, qtbot, result())
    assert view._has_field_scene


def test_advanced_controls_do_not_consume_plot_area_and_share_one_toolbar(view, qtbot):
    view.resize(1000, 500)
    view.show()
    qtbot.wait(10)
    assert not view.field_map_import.isVisible()
    assert not view.field_lines.reference.isVisible()
    assert view.field_lines.fit_button.isVisible()
    assert view.plot.height() > .8*view.height()
    view._show_advanced()
    qtbot.wait(10)
    assert view.field_map_import.isVisible()
    assert view.field_lines.reference.isVisible()
    assert view.field_lines.match_reference.isVisible()
    assert view.advanced_dialog.isAncestorOf(view.field_map_import)
    view.advanced_dialog.hide()
    controls = (view.display_mode, view.field_lines.fit_button, view.advanced_button)
    centers = [control.mapTo(view, control.rect().center()).y() for control in controls]
    assert max(centers)-min(centers) <= 1


def test_profile_errors_remain_readable_in_advanced(view, qtbot, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("Synthetic unavailable map")
    monkeypatch.setattr("temsim.magnetic_field_scene.prepare_magnetic_scene", fail)
    view.display_result(result())
    qtbot.waitUntil(lambda: view._profile_worker is None, timeout=5000)
    assert "unavailable" in view.status.text()
    view._show_advanced()
    assert "Synthetic unavailable map" in view.details.toPlainText()
    assert view._profile is None


def test_first_profile_fits_hidden_2d_curve_before_leaving_3d(view, qtbot):
    view.display_mode.setCurrentIndex(view.display_mode.findData("3d"))
    publish(view, qtbot, result(amplitude=5.))
    view.display_mode.setCurrentIndex(view.display_mode.findData("2d"))
    assert view.plot.viewRange()[1][1] > 4.9


def test_new_snapshot_discards_inflight_profile_before_publishing(view, qtbot):
    view.display_result(result(amplitude=1.))
    latest = result(amplitude=77.)
    view.display_result(latest)
    qtbot.waitUntil(lambda: view._profile_worker is None, timeout=5000)
    assert view._state_snapshot is latest.state_snapshot
    assert view._profile is not None
    np.testing.assert_allclose(view._profile.on_axis_t[:, 0], .77)
    assert "excitation 770%" in view.diagnostic_text("c1")


def test_diagnostic_signal_reports_pending_then_current_and_clear(view, qtbot):
    states = []
    view.diagnostics_updated.connect(lambda: states.append(view.diagnostic_text("c1")))
    publish(view, qtbot, result())
    assert "pending" in states[0]
    assert "peak |Bz|" in states[-1]
    view.display_result(None)
    assert "Recalculate" in states[-1]


def test_failed_captured_scene_never_invokes_a_diagnostic_field_solver(view, qtbot, monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise ValueError("Captured FEM cache missing")
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Diagnostics must not rebuild a missing FEM field")
    monkeypatch.setattr("temsim.magnetic_field_scene.prepare_magnetic_scene", unavailable)
    monkeypatch.setattr("temsim.gui.diagnostic_tabs.lens_field_records", forbidden)
    view.display_result(result())
    qtbot.waitUntil(lambda: view._profile_worker is None, timeout=5000)
    assert view._profile is None
    assert "Captured FEM cache missing" in view.details.toPlainText()
