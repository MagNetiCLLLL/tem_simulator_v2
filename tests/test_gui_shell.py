import numpy as np
import threading
from types import SimpleNamespace
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QDockWidget
import pyqtgraph as pg
import pytest

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.gui.diagnostic_tabs import OpticalTransferView
from temsim.gui.direct_alignment_panel import DirectAlignmentPanel
from temsim.gui.main_window import MainWindow
from temsim.gui.visualization import VisualizationWorkspace
from temsim.optics.column import default_state
from temsim.optics.direct_alignment import DirectAlignmentResult
from temsim.operating_modes import apply_operating_mode_pair
from temsim.diagnostics import optical_transfer_records
from temsim.physics.core import electron
from temsim.physics.all_lens_crossovers import detect_all_lens_crossovers
from temsim.physics.simulation import run
from temsim.simulation_pipeline import (
    CalculationResult,
    aperture_stop_records,
)
from temsim.runtime_parameters import editable_parameters, runtime_targets


DIRECT_CONDENSER_KEYS = ("condenser_lens_2", "condenser_lens_3")


class _SceneClick:
    def __init__(self, scene_position):
        self._scene_position = scene_position
        self.accepted = False

    @staticmethod
    def button():
        return Qt.MouseButton.LeftButton

    @staticmethod
    def double():
        return False

    def scenePos(self):
        return self._scene_position

    def accept(self):
        self.accepted = True


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


def _tree_keys(tree):
    keys = set()
    pending = [
        tree.topLevelItem(index)
        for index in range(tree.topLevelItemCount())
    ]
    while pending:
        item = pending.pop(0)
        key = item.toolTip(0)
        if key:
            keys.add(key)
        pending.extend(
            item.child(index) for index in range(item.childCount())
        )
    return keys


def test_main_window_contains_the_toml_backed_workspace(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    assert window.windowTitle() == "TEM Simulator v2"
    assert window.centralWidget().objectName() == "visualizationWorkspace"
    assert window.findChild(QDockWidget, "instrumentDock") is not None
    assert window.findChild(QDockWidget, "parameterDock") is None
    assert window.findChild(QDockWidget, "logDock") is not None
    assert window.instrument_editor.indexOf(window.assembly_panel) == 0
    assert window.instrument_editor.indexOf(window.parameter_panel) == 1
    assert [
        window.assembly_panel.component_pages.tabText(index)
        for index in range(window.assembly_panel.component_pages.count())
    ] == ["Optical", "Mechanical", "Direct Alignment"]
    direct_alignment = window.assembly_panel.direct_alignment_panel
    assert {
        control.target.objectName()
        for control in direct_alignment.controls.values()
    } == {
        "nanoprobeConvergenceTarget",
        "microprobeIlluminationTarget",
        "imageMagnificationTarget",
        "cameraLengthTarget",
    }
    assert window.assembly_panel.optical_filter.currentData() == "all"
    assert window.assembly_panel.tree.topLevelItemCount() >= 3
    assert window.assembly_panel.probe_mode.currentData() == "nano_probe"
    assert window.assembly_panel.projector_mode.currentData() == "diffraction"
    assert window.compute_backend.objectName() == "computeBackend"
    assert window.compute_backend.currentData() == "Auto"
    assert "C2 + C3 + C2 aperture" in (
        window.assembly_panel.operating_mode_status.text()
    )


def test_workspace_action_buttons_fit_without_a_window_state_change(qtbot):
    workspace = VisualizationWorkspace()
    qtbot.addWidget(workspace)
    workspace.resize(640, 700)
    workspace.show()
    qtbot.wait(20)

    assert workspace.minimumSizeHint().width() < 900
    for button in (
        workspace.auto_zoom,
        workspace.fit_column,
        workspace.column_walls,
        workspace.component_centres,
        workspace.crossovers,
        workspace.jump_to_position,
    ):
        top_left = button.mapTo(workspace, button.rect().topLeft())
        bottom_right = button.mapTo(workspace, button.rect().bottomRight())
        assert button.isVisible()
        assert top_left.x() >= 0
        assert bottom_right.x() < workspace.width()


def test_direct_alignment_gui_gates_modes_and_emits_the_requested_target(qtbot):
    panel = DirectAlignmentPanel()
    qtbot.addWidget(panel)
    controls = panel.controls
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, catalog.default_selection())

    state.illumination_mode = "STEM"
    state.projector_mode = "diffraction"
    panel.set_state(state)
    assert controls["nanoprobe_convergence"].target.isEnabled()
    assert controls["nanoprobe_convergence"].apply_button.isEnabled()
    assert not controls["microprobe_illumination"].target.isEnabled()
    assert not controls["image_magnification"].target.isEnabled()
    assert controls["diffraction_camera_length"].target.isEnabled()

    controls["nanoprobe_convergence"].target.setValue(31.25)
    with qtbot.waitSignal(panel.adjustment_requested) as blocker:
        qtbot.mouseClick(
            controls["nanoprobe_convergence"].apply_button,
            Qt.MouseButton.LeftButton,
        )
    assert blocker.args == ["nanoprobe_convergence", 31.25]

    state.illumination_mode = "TEM"
    state.projector_mode = "image"
    panel.set_state(state)
    assert not controls["nanoprobe_convergence"].target.isEnabled()
    assert controls["microprobe_illumination"].target.isEnabled()
    assert controls["image_magnification"].target.isEnabled()
    assert not controls["diffraction_camera_length"].target.isEnabled()


def _successful_direct_alignment_result(state, key, target):
    lenses = {lens.key: lens for lens in state.lenses}
    strengths = {
        "condenser_lens_2": lenses["condenser_lens_2"].percent + 0.2,
        "condenser_lens_3": lenses["condenser_lens_3"].percent - 0.1,
    }
    return DirectAlignmentResult(
        key=key,
        success=True,
        requested=float(target),
        achieved=float(target),
        unit="mrad",
        constraint_value=0.0,
        constraint_unit="mm",
        strengths=strengths,
        iterations=1,
        validation_step_mm=0.05,
        numerical_spread=0.0,
        message="Background test solve passed.",
    )


def test_main_window_commits_a_current_background_alignment_atomically(
    qtbot, monkeypatch
):
    window = MainWindow()
    qtbot.addWidget(window)
    window.preview_timer.stop()
    captured_states = []

    def solve(snapshot, key, target):
        captured_states.append(snapshot)
        return _successful_direct_alignment_result(snapshot, key, target)

    monkeypatch.setattr(
        "temsim.gui.direct_alignment_controller.apply_direct_alignment",
        solve,
    )
    before = {lens.key: lens.percent for lens in window.state.lenses}
    with qtbot.waitSignal(window.direct_alignments.finished, timeout=5_000):
        window.apply_direct_alignment("nanoprobe_convergence", 30.0)
    window.preview_timer.stop()

    after = {lens.key: lens.percent for lens in window.state.lenses}
    assert captured_states and captured_states[0] is not window.state
    assert {
        key for key in before if before[key] != after[key]
    } == {"condenser_lens_2", "condenser_lens_3"}
    assert "Direct Alignment applied" in window.status_label.text()
    assert not window.progress.isVisible()


def test_main_window_discards_a_stale_background_alignment(
    qtbot, monkeypatch
):
    window = MainWindow()
    qtbot.addWidget(window)
    window.preview_timer.stop()
    worker_started = threading.Event()
    release_worker = threading.Event()

    def solve(snapshot, key, target):
        worker_started.set()
        assert release_worker.wait(timeout=5.0)
        return _successful_direct_alignment_result(snapshot, key, target)

    monkeypatch.setattr(
        "temsim.gui.direct_alignment_controller.apply_direct_alignment",
        solve,
    )
    window.apply_direct_alignment("nanoprobe_convergence", 30.0)
    assert worker_started.wait(timeout=5.0)
    lenses = {lens.key: lens for lens in window.state.lenses}
    manual_value = lenses["condenser_lens_2"].percent + 0.35
    lenses["condenser_lens_2"].percent = manual_value

    with qtbot.waitSignal(window.direct_alignments.finished, timeout=5_000):
        release_worker.set()
    window.preview_timer.stop()

    assert lenses["condenser_lens_2"].percent == pytest.approx(manual_value)
    assert "not applied" in window.status_label.text().lower()
    assert "stale result was discarded" in window.log_output.toPlainText()


def test_invalidating_a_running_alignment_clears_busy_progress_and_status(
    qtbot, monkeypatch
):
    window = MainWindow()
    qtbot.addWidget(window)
    window.preview_timer.stop()
    worker_started = threading.Event()
    release_worker = threading.Event()

    def solve(snapshot, key, target):
        worker_started.set()
        assert release_worker.wait(timeout=5.0)
        return _successful_direct_alignment_result(snapshot, key, target)

    monkeypatch.setattr(
        "temsim.gui.direct_alignment_controller.apply_direct_alignment",
        solve,
    )
    window.apply_direct_alignment("nanoprobe_convergence", 30.0)
    assert worker_started.wait(timeout=5.0)
    assert not window.progress.isHidden()

    window._runtime_parameter_changed("condenser_lens_2.percent")
    window.preview_timer.stop()

    assert window.assembly_panel.direct_alignment_panel._busy_key is None
    assert window.progress.isHidden()
    assert "cancelled" in (
        window.assembly_panel.direct_alignment_panel.result_status.text().lower()
    )
    release_worker.set()
    assert window.direct_alignments.pool.waitForDone(5_000)


@pytest.mark.parametrize(
    ("returned_key", "strength_keys"),
    (
        ("image_magnification", DIRECT_CONDENSER_KEYS),
        ("nanoprobe_convergence", ("condenser_lens_2",)),
        (
            "nanoprobe_convergence",
            (*DIRECT_CONDENSER_KEYS, "objective_lens"),
        ),
    ),
)
def test_background_commit_rejects_wrong_key_or_coupled_device_set(
    qtbot, returned_key, strength_keys
):
    window = MainWindow()
    qtbot.addWidget(window)
    window.preview_timer.stop()
    before = {lens.key: lens.percent for lens in window.state.lenses}
    strengths = {key: before[key] + 0.1 for key in strength_keys}
    result = DirectAlignmentResult(
        key=returned_key,
        success=True,
        requested=30.0,
        achieved=30.0,
        unit="mrad",
        constraint_value=0.0,
        constraint_unit="mm",
        strengths=strengths,
        iterations=1,
        validation_step_mm=0.05,
        numerical_spread=0.0,
        message="Invalid test result.",
    )
    window._direct_alignment_state_token = repr(window.state.to_dict())

    window._direct_alignment_ready(
        "nanoprobe_convergence", result, 0.01
    )

    assert {lens.key: lens.percent for lens in window.state.lenses} == before
    assert "exact coupled-device set" in window.log_output.toPlainText()


def test_direct_alignment_worker_error_clears_solving_status(qtbot, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    window.preview_timer.stop()
    errors = []
    monkeypatch.setattr(window, "_show_error", errors.append)

    def fail(_snapshot, _key, _target):
        raise RuntimeError("synthetic worker failure")

    monkeypatch.setattr(
        "temsim.gui.direct_alignment_controller.apply_direct_alignment",
        fail,
    )
    with qtbot.waitSignal(window.direct_alignments.finished, timeout=5_000):
        window.apply_direct_alignment("nanoprobe_convergence", 30.0)

    panel = window.assembly_panel.direct_alignment_panel
    assert panel._busy_key is None
    assert window.progress.isHidden()
    assert "failed" in panel.result_status.text().lower()
    assert errors and "synthetic worker failure" in errors[0]


def test_direct_alignment_and_calculation_progress_are_mutually_guarded(
    qtbot, monkeypatch
):
    window = MainWindow()
    qtbot.addWidget(window)
    window.preview_timer.stop()
    submissions = []
    monkeypatch.setattr(
        window.calculations,
        "submit",
        lambda *args, **kwargs: submissions.append((args, kwargs)),
    )
    window._direct_alignment_state_token = "busy"

    window.run_preview()
    window.run_high_accuracy()

    assert submissions == []
    assert "deferred until Direct Alignment" in window.status_label.text()
    window._set_progress_active("calculation", True)
    window._set_progress_active("direct_alignment", True)
    window._set_progress_active("calculation", False)
    assert not window.progress.isHidden()
    window._set_progress_active("direct_alignment", False)
    assert window.progress.isHidden()


def test_optical_transfer_view_pairs_image_and_diffraction_captures(qtbot):
    state = default_state()
    catalog = AssemblyCatalog()
    assembly = catalog.apply(state, catalog.default_selection())
    state.step_mm = 1.0
    state.history_step_mm = 1.0
    state.acceleration_enabled = False
    view = OpticalTransferView()
    qtbot.addWidget(view)

    def result_for(projector_key):
        apply_operating_mode_pair(
            state,
            "nano_probe",
            projector_key,
            column_name="C3 + Probe Corrector",
            recording_name="Energy Filter",
        )
        apply_physical_layout_to_state(state)
        simulation = SimpleNamespace(
            optical_transfers=optical_transfer_records(state),
            metrics={"lambda_nm": electron(state)[2]},
        )
        return SimpleNamespace(
            state_snapshot=state,
            simulation=simulation,
            assembly=assembly,
        )

    view.display_result(result_for("imaging"))
    view.target_plane.setCurrentIndex(view.target_plane.findData("camera"))
    qtbot.mouseClick(view.capture_current, Qt.MouseButton.LeftButton)
    assert "Captured modes: image" in view.pair_summary.text()

    view.display_result(result_for("diffraction"))
    view.target_plane.setCurrentIndex(view.target_plane.findData("camera"))
    qtbot.mouseClick(view.capture_current, Qt.MouseButton.LeftButton)

    assert "Normalised diffraction-vector" in view.pair_summary.text()
    assert "Absolute hardware orientation is NOT calibrated" in (
        view.pair_summary.text()
    )
    assert "Lens field polarities remain provisional" in (
        view.pair_summary.text()
    )


def test_gui_applies_calculated_probe_and_projection_modes(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.preview_timer.stop()
    panel = window.assembly_panel

    panel.probe_mode.setCurrentIndex(
        panel.probe_mode.findData("micro_probe")
    )
    panel.projector_mode.setCurrentIndex(
        panel.projector_mode.findData("imaging")
    )
    panel.apply_operating_mode_button.click()
    window.preview_timer.stop()

    by_key = {lens.key: lens for lens in window.state.lenses}
    assert window.state.illumination_mode == "TEM"
    assert window.state.projector_mode == "image"
    assert window.state.condenser_aperture_2.radius_um == pytest.approx(50.0)
    assert window.state.condenser_aperture_3.radius_um == pytest.approx(2000.0)
    assert by_key["condenser_lens_2"].percent == pytest.approx(70.0)
    assert by_key["condenser_lens_3"].percent == pytest.approx(
        33.7460694622
    )
    assert by_key["objective_lens"].percent == pytest.approx(70.0)
    assert by_key["diffraction_lens"].percent == pytest.approx(
        16.0105466828
    )
    assert "sample semi-angle 0.411 mrad" in panel.operating_mode_status.text()


def test_component_navigation_filters_only_the_active_assembly(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.preview_timer.stop()
    panel = window.assembly_panel

    optical_keys = _tree_keys(panel.tree)
    mechanical_keys = _tree_keys(panel.mechanical_tree)
    assert "objective_lens" in optical_keys
    assert "probe_tl12_lens" in optical_keys
    assert "image_tl12_lens" not in optical_keys
    assert "objective_lens_housing" in mechanical_keys
    assert "objective_lens" not in mechanical_keys

    panel.optical_filter.setCurrentIndex(
        panel.optical_filter.findData("lens")
    )
    lens_keys = _tree_keys(panel.tree)
    assert "objective_lens" in lens_keys
    assert "beam_deflector" not in lens_keys
    assert "probe_tl12_lens" not in lens_keys

    panel.optical_filter.setCurrentIndex(
        panel.optical_filter.findData("deflector")
    )
    deflector_keys = _tree_keys(panel.tree)
    assert {
        "beam_deflector",
        "ac_deflector",
        "image_diffraction_deflector",
    }.issubset(deflector_keys)
    assert "probe_dp11_deflector" not in deflector_keys

    panel.optical_filter.setCurrentIndex(
        panel.optical_filter.findData("aperture")
    )
    aperture_keys = _tree_keys(panel.tree)
    assert {
        "condenser_aperture_2",
        "objective_aperture",
        "selected_area_aperture",
    }.issubset(aperture_keys)
    assert "objective_lens" not in aperture_keys

    panel.optical_filter.setCurrentIndex(
        panel.optical_filter.findData("stigmator")
    )
    stigmator_keys = _tree_keys(panel.tree)
    assert {
        "feg_stigmator",
        "condenser_stigmator",
        "objective_stigmator",
        "diffraction_stigmator",
    }.issubset(stigmator_keys)

    panel.optical_filter.setCurrentIndex(
        panel.optical_filter.findData("corrector")
    )
    corrector_keys = _tree_keys(panel.tree)
    assert "probe_tl12_lens" in corrector_keys
    assert "probe_dp11_deflector" in corrector_keys
    assert "objective_lens" not in corrector_keys
    assert "image_tl12_lens" not in corrector_keys
    assert panel.tree.topLevelItemCount() == 1
    assert panel.tree.topLevelItem(0).text(0).startswith("Correctors")
    assert not panel.select_key("image_tl12_lens")

    assert panel.select_key("objective_lens_housing")
    assert panel.component_pages.currentIndex() == 1
    assert panel.select_key("objective_lens")
    assert panel.component_pages.currentIndex() == 0

    window.load_assembly(AssemblySelection(
        gun="FEG",
        column="C2",
        recording="No Energy Filter",
    ))
    window.preview_timer.stop()
    panel.optical_filter.setCurrentIndex(
        panel.optical_filter.findData("all")
    )
    c2_keys = _tree_keys(panel.tree)
    assert "condenser_lens_3" not in c2_keys
    assert "probe_tl12_lens" not in c2_keys
    assert "image_tl12_lens" not in c2_keys

    window.load_assembly(AssemblySelection(
        gun="FEG",
        column="C3 + Image Corrector",
        recording="No Energy Filter",
    ))
    window.preview_timer.stop()
    panel.optical_filter.setCurrentIndex(
        panel.optical_filter.findData("corrector")
    )
    image_corrector_keys = _tree_keys(panel.tree)
    assert "image_tl12_lens" in image_corrector_keys
    assert "image_dp11_deflector" in image_corrector_keys
    assert "probe_tl12_lens" not in image_corrector_keys


def test_layout_selection_opens_energy_slit_editor_and_updates_window(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.preview_timer.stop()
    panel = window.assembly_panel

    panel.optical_filter.setCurrentIndex(
        panel.optical_filter.findData("lens")
    )
    panel.component_pages.setCurrentIndex(1)
    window.instrument_dock.hide()

    window.workspace.component_selected.emit("energy_filter_slit")
    qtbot.wait(20)
    window.preview_timer.stop()

    assert not window.instrument_dock.isHidden()
    assert panel.component_pages.currentIndex() == 0
    assert panel.optical_filter.currentData() == "energy_filter"
    assert panel.tree.current_key() == "energy_filter_slit"
    assert window.parameter_panel.title.text() == (
        "Iliad XO Crossover / Optional EFTEM Energy-slit Assembly"
    )
    assert window.parameter_panel._runtime_target.obj is (
        window.state.energy_filter.energy_slit
    )
    assert window.parameter_panel._manifest_target.part_key == (
        "energy_filter_slit"
    )
    assert {
        field.label for field in window.parameter_panel._manifest_fields
    } >= {"path_center_mm", "clear_height_mm", "maximum_gap_mm"}
    assert window.parameter_panel.tabs.currentIndex() == 0
    assert set(window.parameter_panel._quick_widgets) == {
        "inserted",
        "requested_centre_loss_ev",
        "requested_width_ev",
    }
    runtime_names = {
        window.parameter_panel.runtime_table.item(row, 0).text()
        for row in range(window.parameter_panel.runtime_table.rowCount())
    }
    assert runtime_names == {
        "inserted",
        "requested_centre_loss_ev",
        "requested_width_ev",
    }

    slit = window.state.energy_filter.energy_slit
    requested_width = float(slit.requested_width_ev) + 5.0
    window.parameter_panel._quick_widgets[
        "requested_width_ev"
    ].setValue(requested_width)
    window.preview_timer.stop()
    assert slit.requested_width_ev == pytest.approx(requested_width)
    assert slit.gap_m == pytest.approx(
        abs(slit.calibrated_dispersion_um_per_ev)
        * requested_width
        * 1.0e-6
    )
    assert window.status_label.text() == (
        "Selected Iliad XO Crossover / Optional EFTEM Energy-slit "
        "Assembly from layout"
    )


def test_layout_selection_opens_unmodelled_iliad_component_toml(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.preview_timer.stop()

    key = "energy_filter_dynamic_focus_electrostatic_quadrupole"
    window.workspace.component_selected.emit(key)
    qtbot.wait(20)
    window.preview_timer.stop()

    assert window.assembly_panel.component_pages.currentIndex() == 0
    assert window.assembly_panel.optical_filter.currentData() == (
        "energy_filter"
    )
    assert window.assembly_panel.tree.current_key() == key
    assert window.parameter_panel._runtime_target is None
    assert window.parameter_panel._manifest_target.part_key == key
    assert window.parameter_panel.tabs.currentIndex() == 1
    fields = {
        field.label: field.value
        for field in window.parameter_panel._manifest_fields
    }
    assert fields["electrode_count"] == 4
    assert fields["mechanical_only"] is True
    assert fields["optical_model_status"] == (
        "mechanical_layout_only_dynamic_focus_field_not_implemented"
    )


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
    assert (
        window.parameter_panel.quick_form.labelForField(
            window.parameter_panel._quick_widgets["enabled"]
        ).text()
        == "Inserted"
    )
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


def test_objective_and_selected_area_apertures_start_retracted():
    state = default_state()

    assert state.objective_aperture.enabled is False
    assert state.selected_area_aperture.enabled is False


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


def test_sample_selection_exposes_multislice_controls(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.preview_timer.stop()
    sample_item = _find_tree_item(window.assembly_panel.tree, "sample")

    window.assembly_panel.tree.setCurrentItem(sample_item)

    widgets = window.parameter_panel._quick_widgets
    assert "wave_multislice_enabled" in widgets
    assert widgets["wave_multislice_enabled"].isChecked()
    assert "wave_atomistic_enabled" in widgets
    assert widgets["wave_atomistic_enabled"].isChecked()
    assert "wave_frozen_phonon_enabled" in widgets
    assert not widgets["wave_frozen_phonon_enabled"].isChecked()
    assert widgets["wave_frozen_phonon_configurations"].value() == 4
    assert widgets["wave_frozen_phonon_seed"].value() == 100
    assert widgets["wave_frozen_phonon_sigma_angstrom"].suffix() == " Å"
    assert not widgets["wave_frozen_phonon_configurations"].isEnabled()
    widgets["wave_frozen_phonon_enabled"].setChecked(True)
    assert widgets["wave_frozen_phonon_configurations"].isEnabled()
    assert widgets["wave_frozen_phonon_sigma_angstrom"].isEnabled()
    assert widgets["wave_frozen_phonon_seed"].isEnabled()
    widgets["wave_atomistic_enabled"].setChecked(False)
    assert not widgets["wave_frozen_phonon_enabled"].isEnabled()
    assert not widgets["wave_frozen_phonon_configurations"].isEnabled()
    assert "wave_slice_thickness_angstrom" in widgets
    assert widgets["wave_slice_thickness_angstrom"].value() == pytest.approx(2.0)
    assert widgets["wave_slice_thickness_angstrom"].suffix() == " Å"


def test_sample_parameters_are_immediately_scrollable_in_a_short_panel(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1100, 700)
    window.show()
    window.preview_timer.stop()
    window.instrument_editor.setSizes([220, 220])
    sample_item = _find_tree_item(window.assembly_panel.tree, "sample")

    window.assembly_panel.tree.setCurrentItem(sample_item)
    qtbot.wait(20)

    panel = window.parameter_panel
    scroll_bar = panel.scroll_area.verticalScrollBar()
    assert panel.minimumSizeHint().height() < (
        panel.scroll_content.minimumSizeHint().height()
    )
    assert scroll_bar.maximum() > 0

    last_control = panel._quick_widgets["wave_defocus_nm"]
    panel.scroll_area.ensureWidgetVisible(last_control)
    qtbot.wait(20)
    bottom_right = last_control.mapTo(
        panel.scroll_area.viewport(), last_control.rect().bottomRight()
    )
    assert 0 <= bottom_right.y() < panel.scroll_area.viewport().height()


def test_ray_plot_marks_every_component_centre_and_detected_crossover(
    qtbot, monkeypatch
):
    state = default_state()
    catalog = AssemblyCatalog()
    assembly = catalog.apply(state, catalog.default_selection())
    # This test exercises active aperture clipping and span rendering. The
    # user-facing default remains retracted and is verified separately.
    state.objective_aperture.enabled = True
    state.selected_area_aperture.enabled = True
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
    # This test injects a deterministic result directly. Prevent the window's
    # queued startup preview from arriving later and replacing its plot state.
    monkeypatch.setattr(
        window.calculations, "submit", lambda *_args, **_kwargs: None
    )
    window.preview_timer.stop()

    window.workspace.display_result(result, "Preview")
    qtbot.wait(20)

    axial_part_count = sum(
        not bool(part.data.get("branch_path_only", False))
        for part in assembly.parts
    )
    assert len(window.workspace.component_marker_items) == axial_part_count
    assert len(window.workspace.sample_marker_items) == 2
    sample_line, sample_axis_marker = window.workspace.sample_marker_items
    assert sample_line.value() == pytest.approx(
        assembly.part("sample").center_z_mm
    )
    assert "SAMPLE / SPECIMEN" in sample_line.label.toPlainText()
    assert "continuous ray boundary" in sample_line.toolTip()
    assert sample_line.zValue() > sample_axis_marker.zValue() - 2
    full_view_label_size = sample_line.label.textItem.font().pointSize()
    assert full_view_label_size == window.workspace.RAY_LABEL_BASE_PT
    assert all(
        window.workspace.plot.getAxis(axis_name)
        .style["tickFont"]
        .pointSize()
        == window.workspace.RAY_AXIS_TICK_PT
        for axis_name in ("bottom", "left")
    )
    assert (
        window.workspace.plot.plotItem.legend.opts["labelTextSize"]
        == f"{window.workspace.RAY_LEGEND_PT}pt"
    )
    window.workspace.component_centres.setChecked(False)
    assert len(window.workspace.sample_marker_items) == 2
    assert window.workspace.sample_marker_items[0].isVisible()
    window.workspace.component_centres.setChecked(True)
    assert window.workspace.tabs.count() == 7
    assert "Energy Filter" in {
        window.workspace.tabs.tabText(index)
        for index in range(window.workspace.tabs.count())
    }
    assert len(window.workspace.physical_layout._records) == sum(
        not bool(part.data.get("branch_path_only", False))
        for part in assembly.parts
    )
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
    objective_gap = window.workspace.physical_layout._objective_pole_gap()
    assert objective_gap == pytest.approx((
        assembly.part("objective_upper_pole").end_z_mm,
        assembly.part("objective_lower_pole").start_z_mm,
    ))
    objective_gap_start, objective_gap_end = objective_gap
    assert len(
        window.workspace.physical_layout._objective_lens_half_items
    ) == 12
    for item in window.workspace.physical_layout._objective_lens_half_items:
        rectangle = item.rect()
        assert (
            rectangle.x() + rectangle.width()
            <= objective_gap_start + 1.0e-9
            or rectangle.x() >= objective_gap_end - 1.0e-9
        )
    assert len(window.workspace.physical_layout._objective_lens_labels) == 2

    stage_items = window.workspace.physical_layout._sample_stage_items
    holder_items = window.workspace.physical_layout._sample_holder_items
    sample_plane_items = window.workspace.physical_layout._sample_plane_items
    assert len(stage_items) == 3
    assert len(holder_items) == 4
    assert len(sample_plane_items) == 2
    assert len(
        window.workspace.physical_layout._sample_plane_labels
    ) == 1
    stage_sleeve = stage_items[0].rect()
    holder_shaft = holder_items[0].rect()
    sample_z_mm = assembly.part("sample").center_z_mm
    assert stage_sleeve.left() <= holder_shaft.left()
    assert stage_sleeve.right() >= holder_shaft.right()
    assert holder_shaft.center().x() == pytest.approx(sample_z_mm)
    holder_tip = holder_items[1].polygon()[0]
    assert holder_tip.x() == pytest.approx(sample_z_mm)
    assert holder_tip.y() == pytest.approx(0.0)

    recording_items = (
        window.workspace.physical_layout._recording_device_items
    )
    assert set(recording_items) == {
        "haadf", "flu_screen", "df", "bf", "camera",
    }
    assert {key: len(items) for key, items in recording_items.items()} == {
        "haadf": 5,
        "flu_screen": 4,
        "df": 5,
        "bf": 5,
        "camera": 5,
    }
    assert len(
        window.workspace.physical_layout._recording_device_labels
    ) == 5
    haadf_head = recording_items["haadf"][0].rect()
    assert haadf_head.center().x() == pytest.approx(
        assembly.part("haadf").center_z_mm
    )
    screen_bounds = recording_items["flu_screen"][0].polygon().boundingRect()
    assert screen_bounds.top() > 0.0 or screen_bounds.bottom() < 0.0
    camera_sensor = recording_items["camera"][0].rect()
    assert camera_sensor.center().x() == pytest.approx(
        assembly.part("camera").center_z_mm
    )

    component_labels = (
        window.workspace.physical_layout._component_label_items
    )
    component_leaders = (
        window.workspace.physical_layout._component_label_leader_items
    )
    label_callouts = window.workspace.physical_layout._label_callouts
    visible_label_keys = (
        window.workspace.physical_layout._visible_component_label_keys
    )
    assert {
        "condenser_lens_1",
        "condenser_aperture_2",
        "beam_deflector",
        "diffraction_lens",
        "projector_lens_2",
    }.issubset(component_labels)
    assert set(component_leaders) == set(component_labels)
    assert "condenser_lens_1_housing" not in component_labels
    assert "objective_lens_housing" not in component_labels
    assert {
        "objective:upper",
        "objective:lower",
        "sample:stage",
        "sample:holder",
        "sample:specimen",
        "recording:camera",
        "recording:flu_screen",
    }.issubset(label_callouts)
    assert (
        window.workspace.physical_layout._label_rows_per_side
        >= window.workspace.physical_layout.LABEL_MIN_ROWS_PER_SIDE
    )
    assert visible_label_keys
    visible_labels = [
        component_labels[key] for key in visible_label_keys
    ]
    horizontally_offset_leaders = 0
    for index, first in enumerate(visible_labels):
        assert first.isVisible()
        key = visible_label_keys[index]
        leader = component_leaders[key]
        assert leader.isVisible()
        assert (
            leader.curve.opts["pen"].style()
            == Qt.PenStyle.DashLine
        )
        leader_x, leader_y = leader.getData()
        assert len(leader_x) == len(leader_y) == 3
        assert leader_x[0] == pytest.approx(
            window.workspace.physical_layout._record_by_key[
                key
            ].center_z_mm
        )
        assert leader_x[-1] == pytest.approx(first.pos().x())
        assert leader_y[-1] == pytest.approx(first.pos().y())
        if abs(float(leader_x[-1] - leader_x[0])) > 1.0e-9:
            horizontally_offset_leaders += 1
        first_bounds = first.sceneBoundingRect()
        assert all(
            not first_bounds.intersects(second.sceneBoundingRect())
            for second in visible_labels[index + 1:]
        )
        assert all(
            not first_bounds.intersects(special.sceneBoundingRect())
            for special in (
                window.workspace.physical_layout._special_label_items()
            )
            if special.isVisible()
        )
    assert horizontally_offset_leaders > 0
    for callout in label_callouts.values():
        assert callout.label.isVisible() == callout.leader.isVisible()
    assert len(window.workspace.magnetic_field._curves) == len(state.lenses)
    formula_records = window.workspace.magnetic_field._records
    assert len({record.formula_key for record in formula_records}) == 3
    for record in formula_records:
        curve = window.workspace.magnetic_field._curves[record.key]
        assert curve.opts["pen"].color().name() == record.formula_colour
    assert len(window.workspace.magnetic_field._formula_samples) == 3
    assert len(window.workspace.magnetic_field.legend.items) == 4
    assert window.workspace.magnetic_field.show_rotation_labels.isChecked()
    assert window.workspace.magnetic_field._rotation_items
    assert window.workspace.magnetic_field._plane_records
    assert "Objective image plane" in window.workspace.magnetic_field.summary.text()
    window.workspace.magnetic_field.show_rotation_labels.setChecked(False)
    assert all(
        not item.isVisible()
        for item in window.workspace.magnetic_field._rotation_items
    )
    window.workspace.magnetic_field.show_rotation_labels.setChecked(True)
    assert all(
        item.isVisible()
        for item in window.workspace.magnetic_field._rotation_items
    )
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
    assert objective_stop == pytest.approx(objective.center_z_mm)
    assert objective.start_z_mm <= objective_stop <= objective.end_z_mm
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
    assert window.workspace.plot.getViewBox().autoRangeEnabled() == [False, False]
    window.workspace.plot.setXRange(
        objective_stop - 10.0, objective_stop + 10.0, padding=0.0
    )
    window.workspace.plot.setYRange(-1.0, 1.0, padding=0.0)
    qtbot.wait(20)
    x_min, x_max = window.workspace.plot.getViewBox().viewRange()[0]
    assert x_min == pytest.approx(objective_stop - 10.0)
    assert x_max == pytest.approx(objective_stop + 10.0)
    zoomed_label_size = (
        window.workspace.sample_marker_items[0]
        .label.textItem.font()
        .pointSize()
    )
    assert full_view_label_size < zoomed_label_size
    assert zoomed_label_size <= window.workspace.RAY_LABEL_MAX_PT
    assert all(
        label.textItem.font().pointSize() == zoomed_label_size
        for label in window.workspace._ray_label_items
    )
    y_min, y_max = window.workspace.plot.getViewBox().viewRange()[1]
    y_span = y_max - y_min
    displayed_opening_lower = y_min + float(lower.span[1]) * y_span
    displayed_opening_upper = y_min + float(upper.span[0]) * y_span
    assert centre_x_mm == pytest.approx(0.0)
    assert radius_mm == pytest.approx(0.05)
    assert displayed_opening_lower == pytest.approx(-0.05)
    assert displayed_opening_upper == pytest.approx(0.05)

    # A second manual zoom must update the aperture gap without AutoRange
    # snapping back to the complete column or changing its physical diameter.
    window.workspace.plot.setYRange(-0.1, 0.1, padding=0.0)
    qtbot.wait(20)
    y_min, y_max = window.workspace.plot.getViewBox().viewRange()[1]
    y_span = y_max - y_min
    displayed_opening_lower = y_min + float(lower.span[1]) * y_span
    displayed_opening_upper = y_min + float(upper.span[0]) * y_span
    assert y_min == pytest.approx(-0.1)
    assert y_max == pytest.approx(0.1)
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

    # A recalculation of the same physical assembly (for example after a lens
    # excitation edit) must replace the rays without changing manual X/Y zoom.
    view_before_recalculation = (
        window.workspace.plot.getViewBox().viewRange()
    )
    window.workspace.display_result(result, "Preview update")
    qtbot.wait(10)
    view_after_recalculation = (
        window.workspace.plot.getViewBox().viewRange()
    )
    assert view_after_recalculation[0] == pytest.approx(
        view_before_recalculation[0]
    )
    assert view_after_recalculation[1] == pytest.approx(
        view_before_recalculation[1]
    )
    assert window.workspace.plot.getViewBox().autoRangeEnabled() == [
        False,
        False,
    ]

    window.workspace.crossovers.setChecked(False)
    view_without_crossovers = window.workspace.plot.getViewBox().viewRange()
    assert view_without_crossovers[0] == pytest.approx(
        view_before_recalculation[0]
    )
    assert view_without_crossovers[1] == pytest.approx(
        view_before_recalculation[1]
    )
    window.workspace.crossovers.setChecked(True)
    view_with_crossovers = window.workspace.plot.getViewBox().viewRange()
    assert view_with_crossovers[0] == pytest.approx(
        view_before_recalculation[0]
    )
    assert view_with_crossovers[1] == pytest.approx(
        view_before_recalculation[1]
    )

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
    assert "field direction" in window.parameter_panel.lens_diagnostics.text()
    assert "column cumulative" in window.parameter_panel.lens_diagnostics.text()
    assert "focal length" in window.workspace.magnetic_field.summary.text()
    assert "Larmor rotation" in window.workspace.magnetic_field.summary.text()
    assert "orientation relative" in window.workspace.transverse_beam.summary.text()
    window.workspace.optical_transfer.target_plane.setCurrentIndex(
        window.workspace.optical_transfer.target_plane.findData("camera")
    )
    assert "J_img" in window.workspace.optical_transfer.matrix_text.toPlainText()
    assert "J_diff" in window.workspace.optical_transfer.matrix_text.toPlainText()
    assert "UNCALIBRATED placeholder" in (
        window.workspace.optical_transfer.matrix_text.toPlainText()
    )

    c1_callout = (
        window.workspace.physical_layout._label_callouts[
            "component:condenser_lens_1"
        ]
    )
    assert c1_callout.label.isVisible()
    c1_click = _SceneClick(
        c1_callout.label.sceneBoundingRect().center()
    )
    window.workspace.physical_layout._component_item_clicked(c1_click)
    assert c1_click.accepted
    assert window.assembly_panel.tree.currentItem().toolTip(0) == (
        "condenser_lens_1"
    )
    assert window.assembly_panel.optical_filter.currentData() == "lens"

    window.workspace.auto_zoom.setChecked(False)
    stage_body = (
        window.workspace.physical_layout._sample_stage_items[0]
    )
    stage_bounds = stage_body.boundingRect()
    stage_click = _SceneClick(stage_body.mapToScene(QPointF(
        stage_bounds.left() + 0.15 * stage_bounds.width(),
        stage_bounds.top() + 0.10 * stage_bounds.height(),
    )))
    window.workspace.physical_layout._component_item_clicked(stage_click)
    assert stage_click.accepted
    assert window.assembly_panel.component_pages.currentIndex() == 1
    assert window.assembly_panel.mechanical_tree.current_key() == (
        "sample_stage"
    )
    assert window.parameter_panel.tabs.currentIndex() == 1

    window.assembly_panel.component_pages.setCurrentIndex(0)
    window.workspace.plot.setXRange(900.0, 1100.0, padding=0.0)
    manual_range = window.workspace.plot.getViewBox().viewRange()[0]
    window.assembly_panel.optical_filter.setCurrentIndex(
        window.assembly_panel.optical_filter.findData("aperture")
    )
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
