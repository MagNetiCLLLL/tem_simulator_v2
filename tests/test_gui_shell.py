import numpy as np
from PySide6.QtWidgets import QDockWidget
import pyqtgraph as pg
import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.gui.main_window import MainWindow
from temsim.gui.visualization import VisualizationWorkspace
from temsim.optics.column import default_state
from temsim.physics.all_lens_crossovers import detect_all_lens_crossovers
from temsim.physics.simulation import run
from temsim.simulation_pipeline import (
    CalculationResult,
    aperture_stop_records,
)
from temsim.runtime_parameters import editable_parameters, runtime_targets


def test_ray_segments_stop_at_the_interpolated_blocking_plane():
    z_mm = np.array([0.0, 1.0, 2.0, 3.0])
    x_m = np.array([[0.0], [0.001], [0.002], [0.003]])

    plot_z, plot_x_mm = VisualizationWorkspace._bundle_lines(
        z_mm, x_m, 1, np.array([1.5])
    )
    finite = np.isfinite(plot_z)

    assert plot_z[finite].tolist() == pytest.approx([0.0, 1.0, 1.5])
    assert plot_x_mm[finite].tolist() == pytest.approx([0.0, 1.0, 1.5])


def test_transverse_projection_supports_arbitrary_view_angles():
    x = np.array([[1.0, 0.0], [-2.0, 3.0]])
    y = np.array([[0.0, 2.0], [4.0, -1.0]])

    assert VisualizationWorkspace._project_transverse_values(
        x, y, 0.0
    ) == pytest.approx(x)
    assert VisualizationWorkspace._project_transverse_values(
        x, y, 90.0
    ) == pytest.approx(y)
    assert VisualizationWorkspace._project_transverse_values(
        x, y, 45.0
    ) == pytest.approx((x + y) / np.sqrt(2.0))


def _find_tree_item(tree, key):
    for root_index in range(tree.topLevelItemCount()):
        root = tree.topLevelItem(root_index)
        for child_index in range(root.childCount()):
            child = root.child(child_index)
            if child.toolTip(0) == key:
                return child
    raise AssertionError(f"Missing tree item: {key}")


def test_main_window_contains_the_toml_backed_workspace(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    assert window.windowTitle() == "TEM Simulator v2"
    assert window.centralWidget().objectName() == "visualizationWorkspace"
    assert window.findChild(QDockWidget, "instrumentDock") is not None
    assert window.findChild(QDockWidget, "parameterDock") is not None
    assert window.findChild(QDockWidget, "logDock") is not None
    assert window.assembly_panel.tree.topLevelItemCount() >= 3


def test_lens_selection_exposes_live_excitation_control(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    tree = window.assembly_panel.tree
    objective = _find_tree_item(tree, "objective_lens")
    tree.setCurrentItem(objective)

    assert window.parameter_panel.title.text() == "Objective Lens Assembly"
    assert window.parameter_panel.lens_box.isHidden() is False
    assert window.parameter_panel.lens_cs.suffix() == " mm"
    assert window.parameter_panel.lens_field_direction.currentData() in (-1, 1)
    window.parameter_panel.lens_cs.setValue(0.85)
    assert window.state.objective_lens.cs_mm == pytest.approx(0.85)
    negative_index = window.parameter_panel.lens_field_direction.findData(-1)
    window.parameter_panel.lens_field_direction.setCurrentIndex(negative_index)
    assert window.state.objective_lens.polarity == -1
    assert window.parameter_panel.manifest_table.rowCount() > 10
    assert window.parameter_panel.anchor_table.rowCount() >= 6
    manifest_names = {
        window.parameter_panel.manifest_table.item(row, 0).text()
        for row in range(window.parameter_panel.manifest_table.rowCount())
    }
    assert "vacuum_inner_diameter_mm" in manifest_names
    assert not hasattr(window.parameter_panel, "column_diameter")


def test_aperture_selection_exposes_unit_aware_quick_controls(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    aperture = _find_tree_item(
        window.assembly_panel.tree, "condenser_aperture_2"
    )
    window.assembly_panel.tree.setCurrentItem(aperture)

    assert window.parameter_panel.quick_box.isHidden() is False
    assert set(window.parameter_panel._quick_widgets) == {
        "enabled", "radius_mm", "offset_x_mm", "offset_y_mm"
    }
    radius = window.parameter_panel._quick_widgets["radius_mm"]
    assert radius.suffix() == " µm"
    radius.setValue(75.0)
    assert window.state.condenser_aperture_2.radius_mm == pytest.approx(0.075)


def test_toml_geometry_is_not_exposed_as_a_runtime_value():
    state = default_state()
    targets = runtime_targets(state)

    simulation_names = {
        item.name for item in editable_parameters(targets["simulation"])
    }
    camera_names = {
        item.name for item in editable_parameters(targets["camera"])
    }
    objective_names = {
        item.name for item in editable_parameters(targets["objective_lens"])
    }

    assert "column_inner_diameter_mm" not in simulation_names
    assert "outer_width_mm" not in camera_names
    assert "inner_diameter_mm" not in camera_names
    assert "inner_face_gap_mm" not in objective_names
    assert "upper_a_mm" not in objective_names
    assert "lower_objective_lens_axial_length_mm" not in objective_names


def test_source_selection_exposes_primary_emission_controls(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    source = _find_tree_item(window.assembly_panel.tree, "feg_tip")
    window.assembly_panel.tree.setCurrentItem(source)

    assert {
        "emission_current_na",
        "virtual_source_fwhm_nm",
        "angular_cutoff_mrad",
        "energy_spread_fwhm_ev",
    }.issubset(window.parameter_panel._quick_widgets)


def test_ray_plot_marks_every_component_centre_and_detected_crossover(qtbot):
    state = default_state()
    catalog = AssemblyCatalog()
    assembly = catalog.apply(state, catalog.default_selection())
    state.electron_gun.emitter.ray_count = 25
    state.step_mm = 3.0
    state.sample.diffraction_enabled = False
    layout = apply_physical_layout_to_state(state)
    simulation = run(state, resolved_layout=layout)
    crossovers = detect_all_lens_crossovers(
        [simulation.incident, *simulation.branches.values()], state.lenses
    )
    result = CalculationResult(
        simulation=simulation,
        energy_filter=None,
        state_snapshot=state,
        layout=layout,
        assembly=assembly,
        lens_crossovers=tuple(crossovers),
        aperture_stops=aperture_stop_records(state),
    )
    window = MainWindow()
    qtbot.addWidget(window)
    window.preview_timer.stop()

    window.workspace.display_result(result, "Preview")
    qtbot.wait(20)

    assert len(window.workspace.component_marker_items) == len(assembly.parts)
    assert window.workspace.tabs.count() == 5
    assert len(window.workspace.physical_layout._records) == len(assembly.parts)
    c1 = assembly.part("condenser_lens_1")
    c2 = assembly.part("condenser_lens_2")
    c1_lower_pole = assembly.part("condenser_lens_1_lower_pole")
    c2_upper_pole = assembly.part("condenser_lens_2_upper_pole")
    c3 = assembly.part("condenser_lens_3")
    c3_upper_pole = assembly.part("condenser_lens_3_upper_pole")
    c3_lower_pole = assembly.part("condenser_lens_3_lower_pole")
    with pytest.raises(KeyError):
        assembly.part("condenser_lens_1_upper_pole")
    with pytest.raises(KeyError):
        assembly.part("condenser_lens_2_lower_pole")
    assert c1.end_z_mm == pytest.approx(c2.start_z_mm)
    assert c1_lower_pole.start_z_mm >= c1.start_z_mm
    assert c1_lower_pole.end_z_mm <= c1.end_z_mm
    assert c2_upper_pole.start_z_mm >= c2.start_z_mm
    assert c2_upper_pole.end_z_mm <= c2.end_z_mm
    assert c3.start_z_mm > c2.end_z_mm
    assert c3_upper_pole.start_z_mm >= c3.start_z_mm
    assert c3_lower_pole.end_z_mm <= c3.end_z_mm
    gap_start, gap_end, gap_midpoint = (
        window.workspace.physical_layout._c1_c2_pole_gap
    )
    assert gap_end - gap_start == pytest.approx(20.0)
    assert gap_midpoint == pytest.approx(c1.end_z_mm)
    assert len(window.workspace.physical_layout._design_reference_items) == 1
    assert len(window.workspace.physical_layout._vacuum_liner_items) == (
        2 * len(assembly.vacuum_liner_segments)
    )
    assert window.workspace.physical_layout._pole_face_at_end(
        "condenser_lens_1_lower_pole"
    )
    assert not window.workspace.physical_layout._pole_face_at_end(
        "condenser_lens_2_upper_pole"
    )
    assert len(window.workspace.magnetic_field._curves) == len(state.lenses)
    formula_records = window.workspace.magnetic_field._records
    assert len({record.formula_key for record in formula_records}) == 3
    for record in formula_records:
        curve = window.workspace.magnetic_field._curves[record.key]
        assert curve.opts["pen"].color().name() == record.formula_colour
    assert len(window.workspace.magnetic_field._formula_samples) == 3
    assert len(window.workspace.magnetic_field.legend.items) == 4
    assert window.workspace.transverse_beam._scatter is not None
    assert window.workspace.stop_marker_items
    assert len(window.workspace.crossover_marker_items) >= len(crossovers)
    assert len(window.workspace.column_wall_items) == 2
    wall_ranges = [item.getData()[1] for item in window.workspace.column_wall_items]
    expected_radii = {
        0.5 * float(segment.inner_diameter_mm)
        for segment in assembly.vacuum_bore_segments
    }
    assert expected_radii.issubset({
        abs(float(value))
        for values in wall_ranges
        for value in values
        if np.isfinite(value)
    })
    aperture_parts = [
        part for part in assembly.parts if "aperture" in part.key
    ]
    aperture_count = len(aperture_parts)
    assert len(window.workspace.aperture_marker_items) == aperture_count
    assert len(window.workspace.aperture_optical_plane_items) == aperture_count
    assert len(window.workspace.aperture_stop_segment_items) == 2 * aperture_count
    expected_optical_stops = sorted(
        window.workspace._aperture_optical_plane(part)
        for part in aperture_parts
    )
    actual_optical_stops = sorted(
        float(line.value())
        for line in window.workspace.aperture_optical_plane_items
    )
    assert actual_optical_stops == pytest.approx(expected_optical_stops)
    objective = assembly.part("objective_aperture")
    objective_stop = window.workspace._aperture_optical_plane(objective)
    assert objective_stop == pytest.approx(1613.4536907227932)
    assert objective_stop != pytest.approx(objective.center_z_mm)
    objective_blocked_z = [
        z_mm
        for key, z_mm in zip(
            simulation.branches["000"].blocked_key,
            simulation.branches["000"].blocked_z,
        )
        if key == "objective_aperture"
    ]
    assert objective_blocked_z
    assert objective_blocked_z == pytest.approx(
        [objective_stop] * len(objective_blocked_z)
    )
    lower, upper, centre_x_mm, radius_mm = next(
        item
        for item in window.workspace._aperture_span_records
        if float(item[0].value()) == pytest.approx(objective_stop)
    )
    window.workspace.plot.setYRange(-1.0, 1.0, padding=0.0)
    window.workspace._update_aperture_spans()
    y_min, y_max = window.workspace.plot.getViewBox().viewRange()[1]
    y_span = y_max - y_min
    displayed_opening_lower = y_min + float(lower.span[1]) * y_span
    displayed_opening_upper = y_min + float(upper.span[0]) * y_span
    assert centre_x_mm == pytest.approx(0.0)
    assert radius_mm == pytest.approx(0.05)
    assert displayed_opening_lower == pytest.approx(-0.05)
    assert displayed_opening_upper == pytest.approx(0.05)
    paired_parts = [
        part
        for part in assembly.parts
        if window.workspace._deflector_planes(part)
    ]
    expected_plane_lines = sum(
        1
        if len(set(window.workspace._deflector_planes(part))) == 1
        else 2
        for part in paired_parts
    )
    plane_lines = [
        item
        for item in window.workspace.deflector_pair_items
        if isinstance(item, pg.InfiniteLine)
    ]
    assert len(plane_lines) == expected_plane_lines
    expected_positions = []
    for part in paired_parts:
        planes = window.workspace._deflector_planes(part)
        expected_positions.extend(planes[:1] if len(set(planes)) == 1 else planes)
    actual_positions = [float(line.value()) for line in plane_lines]
    assert sorted(actual_positions) == pytest.approx(sorted(expected_positions))
    assert "crossovers" in window.workspace.heading.text()
    assert "angles not to scale" in window.workspace.hint.text()
    assert "Blocked rays stop at first intercept" in window.workspace.hint.text()
    assert all(
        button.sizeHint().height() >= 36
        for button in (
            window.workspace.auto_zoom,
            window.workspace.component_centres,
            window.workspace.crossovers,
            window.workspace.column_walls,
            window.workspace.fit_column,
        )
    )

    traced_result = window.workspace._last_result
    window.workspace._set_projection_angle(90.0)
    assert window.workspace._last_result is traced_result
    assert window.workspace.projection_slider.value() == 900
    assert window.workspace.projection_angle.value() == pytest.approx(90.0)
    assert window.workspace.projection_yz.isChecked()
    assert window.workspace.plot.getAxis("left").labelText == (
        "Projected displacement"
    )
    assert "Y projection" in window.workspace.heading.text()

    window.workspace._set_projection_angle(37.25)
    assert window.workspace._last_result is traced_result
    assert window.workspace.projection_angle.value() == pytest.approx(37.25)
    assert window.workspace.projection_slider.value() == 373
    assert "Projected displacement" == (
        window.workspace.plot.getAxis("left").labelText
    )
    window.workspace._set_projection_angle(0.0)

    window.workspace.tabs.setCurrentIndex(1)
    window.workspace.physical_layout.axial_position_selected.emit(910.0)
    assert window.workspace.tabs.currentIndex() == 0
    assert window.workspace._last_result is traced_result
    assert window.workspace._selected_z_mm == pytest.approx(910.0)
    assert window.workspace.axial_position.value() == pytest.approx(910.0)
    assert window.workspace.axial_cursor_item is not None
    assert float(window.workspace.axial_cursor_item.value()) == pytest.approx(
        910.0
    )
    linked_range = window.workspace.plot.getViewBox().viewRange()[0]
    assert 0.5 * (linked_range[0] + linked_range[1]) == pytest.approx(910.0)
    assert linked_range[1] - linked_range[0] <= 260.0

    window.workspace.tabs.setCurrentIndex(2)
    window.workspace.magnetic_field.axial_position_selected.emit(950.0)
    assert window.workspace.tabs.currentIndex() == 0
    assert window.workspace._selected_z_mm == pytest.approx(950.0)

    window.workspace.jump_to_ray_position(1200.0, window_mm=80.0)
    exact_range = window.workspace.plot.getViewBox().viewRange()[0]
    assert 0.5 * (exact_range[0] + exact_range[1]) == pytest.approx(1200.0)
    assert exact_range[1] - exact_range[0] == pytest.approx(80.0)
    window.workspace.plot.setXRange(-40.0, 40.0, padding=0.0)
    window.workspace.plot.setYRange(-0.123, 0.456, padding=0.0)
    qtbot.wait(10)

    def z_screen_metrics():
        view_box = window.workspace.plot.getViewBox()
        x_min, x_max = view_box.viewRange()[0]
        scene_bounds = view_box.sceneBoundingRect()
        pixels_per_mm = scene_bounds.width() / (x_max - x_min)
        zero_scene_x = scene_bounds.left() - x_min * pixels_per_mm
        return pixels_per_mm, zero_scene_x

    view_before_rotation = window.workspace.plot.getViewBox().viewRange()
    z_scale_before, z_zero_before = z_screen_metrics()
    window.workspace._set_projection_angle(90.0)
    qtbot.wait(10)
    view_after_rotation = window.workspace.plot.getViewBox().viewRange()
    z_scale_after, z_zero_after = z_screen_metrics()
    assert view_after_rotation[0] == pytest.approx(view_before_rotation[0])
    assert view_after_rotation[1] == pytest.approx(view_before_rotation[1])
    assert z_scale_after == pytest.approx(z_scale_before)
    assert z_zero_after == pytest.approx(z_zero_before)
    assert float(window.workspace.axial_cursor_item.value()) == pytest.approx(
        1200.0
    )
    window.workspace._set_projection_angle(0.0)
    qtbot.wait(10)
    assert window.workspace.plot.getViewBox().viewRange()[0] == pytest.approx(
        view_before_rotation[0]
    )
    assert window.workspace.plot.getViewBox().viewRange()[1] == pytest.approx(
        view_before_rotation[1]
    )
    z_scale_restored, z_zero_restored = z_screen_metrics()
    assert z_scale_restored == pytest.approx(z_scale_before)
    assert z_zero_restored == pytest.approx(z_zero_before)

    assert window.workspace.auto_zoom.isChecked() is False
    window.workspace.auto_zoom.setChecked(True)
    objective_item = _find_tree_item(
        window.assembly_panel.tree, "objective_lens"
    )
    window.assembly_panel.tree.setCurrentItem(objective_item)
    objective = assembly.part("objective_lens")
    focused_range = window.workspace.plot.getViewBox().viewRange()[0]
    assert focused_range[0] < objective.center_z_mm < focused_range[1]
    assert focused_range[1] - focused_range[0] <= 520.0
    assert "peak |Bz|" in window.parameter_panel.lens_diagnostics.text()
    assert "focal length" in window.workspace.magnetic_field.summary.text()
    assert "Larmor rotation" in window.workspace.magnetic_field.summary.text()
    assert "orientation relative" in window.workspace.transverse_beam.summary.text()

    window.workspace.physical_layout.component_selected.emit(
        "condenser_lens_1"
    )
    assert window.assembly_panel.tree.currentItem().toolTip(0) == (
        "condenser_lens_1"
    )

    window.workspace.auto_zoom.setChecked(False)
    window.workspace.plot.setXRange(900.0, 1100.0, padding=0.0)
    manual_range = window.workspace.plot.getViewBox().viewRange()[0]
    aperture_item = _find_tree_item(
        window.assembly_panel.tree, "condenser_aperture_2"
    )
    window.assembly_panel.tree.setCurrentItem(aperture_item)
    assert window.workspace.plot.getViewBox().viewRange()[0] == pytest.approx(
        manual_range
    )

    window.workspace.auto_zoom.setChecked(True)
    aperture = assembly.part("condenser_aperture_2")
    refocused_range = window.workspace.plot.getViewBox().viewRange()[0]
    assert refocused_range[0] < aperture.center_z_mm < refocused_range[1]
    assert refocused_range[1] - refocused_range[0] == pytest.approx(90.0)
