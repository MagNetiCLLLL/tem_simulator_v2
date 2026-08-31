from types import SimpleNamespace

import pytest

import temsim.gui.calculation_controller as calculation_controller_module
import temsim.simulation_pipeline as simulation_pipeline
from temsim.gui.calculation_controller import (
    CalculationController,
    HIGH_ACCURACY_MEMORY_BUDGET_BYTES,
    estimate_calculation_memory_bytes,
)
from temsim.optics.column import default_state


def test_high_accuracy_pipeline_reports_completed_real_stages(monkeypatch):
    state = default_state()
    state.sample.wave_enabled = False
    state.ac_deflector.scan_enabled = False
    simulation = SimpleNamespace(
        incident=SimpleNamespace(),
        branches={},
    )

    monkeypatch.setattr(
        simulation_pipeline,
        "ensure_recording_system",
        lambda _state: None,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "ensure_energy_filter",
        lambda _state: None,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "ensure_corrector_structure",
        lambda _state: None,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "normalise_component_names",
        lambda _state: None,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "apply_physical_layout_to_state",
        lambda _state: "layout",
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "run",
        lambda _state, resolved_layout: simulation,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "simulate_energy_filter",
        lambda _state, _simulation: "energy-filter",
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "calculate_scan_geometry",
        lambda _state: "scan-geometry",
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "calculate_scan_ray_paths",
        lambda _state, _simulation: "scan-rays",
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "detect_all_lens_crossovers",
        lambda _branches, _lenses: (),
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "aperture_stop_records",
        lambda _state: (),
    )

    progress = []
    result = simulation_pipeline.calculate(
        state,
        progress_callback=lambda completed, total, stage: progress.append(
            (completed, total, stage)
        ),
    )

    assert result.simulation is simulation
    assert progress == [
        (0, 6, "Preparing state and physical layout"),
        (1, 6, "Tracing the electron column"),
        (2, 6, "Tracing the energy filter"),
        (3, 6, "Solving scan geometry"),
        (4, 6, "Building scan-ray playback"),
        (5, 6, "Finalising optical diagnostics"),
        (6, 6, "Complete"),
    ]


def test_high_accuracy_pipeline_maps_stem_batches_inside_stage(monkeypatch):
    state = default_state()
    state.sample.wave_enabled = False
    state.ac_deflector.enabled = True
    state.ac_deflector.scan_enabled = True
    state.ac_deflector.scan_pixels_x = 32
    state.ac_deflector.scan_lines = 32
    state.sample.stem_wave_enabled = True
    state.sample.wave_atomistic_enabled = False
    simulation = SimpleNamespace(incident=SimpleNamespace(), branches={})

    monkeypatch.setattr(
        simulation_pipeline, "ensure_recording_system", lambda _state: None
    )
    monkeypatch.setattr(
        simulation_pipeline, "ensure_energy_filter", lambda _state: None
    )
    monkeypatch.setattr(
        simulation_pipeline, "ensure_corrector_structure", lambda _state: None
    )
    monkeypatch.setattr(
        simulation_pipeline, "normalise_component_names", lambda _state: None
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "apply_physical_layout_to_state",
        lambda _state: "layout",
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "run",
        lambda _state, resolved_layout: simulation,
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "simulate_energy_filter",
        lambda _state, _simulation: "energy-filter",
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "calculate_scan_geometry",
        lambda _state: "scan-geometry",
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "calculate_scan_ray_paths",
        lambda _state, _simulation: "scan-rays",
    )

    def fake_stem(_state, _simulation, *, progress_callback):
        progress_callback(0, 4, "Preparing STEM")
        progress_callback(2, 4, "STEM probes 16/32")
        progress_callback(4, 4, "STEM detector frame complete")
        return "stem-frame"

    monkeypatch.setattr(
        simulation_pipeline, "calculate_stem_scan_frame", fake_stem
    )
    monkeypatch.setattr(
        simulation_pipeline,
        "detect_all_lens_crossovers",
        lambda _branches, _lenses: (),
    )
    monkeypatch.setattr(
        simulation_pipeline, "aperture_stop_records", lambda _state: ()
    )

    progress = []
    result = simulation_pipeline.calculate(
        state,
        progress_callback=lambda completed, total, stage: progress.append(
            (completed, total, stage)
        ),
    )

    assert result.stem_scan == "stem-frame"
    nested = [item for item in progress if item[1] == 1_340_000]
    assert nested == [
        (50_000, 1_340_000, "Preparing STEM"),
        (690_000, 1_340_000, "STEM probes 16/32"),
        (1_330_000, 1_340_000, "STEM detector frame complete"),
    ]
    assert nested[0][0] / nested[0][1] < 0.04
    percentages = [completed / total for completed, total, _stage in progress]
    assert percentages == sorted(percentages)


def test_controller_forwards_only_current_high_accuracy_progress(monkeypatch):
    controller = CalculationController()
    workers = []
    controller.pool.start = workers.append
    updates = []
    controller.progress_changed.connect(
        lambda *values: updates.append(values)
    )

    def fake_calculate(_state, *, progress_callback):
        progress_callback(0, 2, "Preparing")
        progress_callback(1, 2, "Finalising")
        progress_callback(2, 2, "Complete")
        return object()

    monkeypatch.setattr(
        calculation_controller_module,
        "calculate",
        fake_calculate,
    )
    controller.submit(default_state(), "High accuracy", 25, 5.0)
    workers[0].run()

    assert updates == [
        ("High accuracy", 0, 2, "Preparing"),
        ("High accuracy", 1, 2, "Finalising"),
        ("High accuracy", 2, 2, "Complete"),
    ]


def test_preview_runs_off_the_gui_thread(qtbot):
    controller = CalculationController()
    state = default_state()
    state.objective_lens.cs_mm = 0.85
    state.objective_lens.polarity = -1

    with qtbot.waitSignal(controller.result_ready, timeout=30_000) as blocker:
        controller.submit(state, "Preview", 25, 3.0)

    quality, result, duration = blocker.args
    assert quality == "Preview"
    assert result.simulation.incident.x.shape[1] == 25
    assert "000" in result.simulation.branches
    assert {
        branch.interaction_kind
        for branch in result.simulation.branches.values()
    } == {"virtual_interactions_disabled"}
    assert result.simulation.metrics["branch_weights_are_absolute"] is False
    assert result.lens_crossovers
    assert all(item["verified"] for item in result.lens_crossovers)
    assert result.aperture_stops
    assert result.state_snapshot.objective_lens.cs_mm == 0.85
    assert result.state_snapshot.objective_lens.polarity == -1
    assert all("diameter_mm" in item for item in result.aperture_stops)
    assert duration > 0.0


def test_high_accuracy_defaults_fit_32_gib_budget_and_extreme_request_is_rejected():
    state = default_state()
    default_estimate = estimate_calculation_memory_bytes(
        state, "High accuracy", 15_000, 0.1
    )
    assert default_estimate < HIGH_ACCURACY_MEMORY_BUDGET_BYTES

    controller = CalculationController()
    with pytest.raises(ValueError, match="32 GiB workstation"):
        controller.submit(state, "High accuracy", 1_000_000, 0.01)


def test_high_accuracy_memory_guard_includes_tem_wave_grid():
    state = default_state()
    state.illumination_mode = "TEM"
    state.sample.wave_enabled = True
    state.sample.wave_multislice_enabled = False
    state.sample.wave_grid_pixels = 8192

    estimate = estimate_calculation_memory_bytes(
        state, "High accuracy", 15_000, 0.1
    )

    assert estimate > HIGH_ACCURACY_MEMORY_BUDGET_BYTES
    controller = CalculationController()
    with pytest.raises(ValueError, match="TEM wave grid"):
        controller.submit(state, "High accuracy", 15_000, 0.1)


def test_wave_imaging_is_disabled_only_for_preview():
    controller = CalculationController()
    captured = []
    controller.pool.start = captured.append
    state = default_state()
    state.illumination_mode = "TEM"
    state.sample.wave_enabled = True

    controller.submit(state, "High accuracy", 25, 5.0)
    controller.submit(state, "Preview", 25, 5.0)

    assert captured[0].state.sample.wave_enabled is True
    assert captured[1].state.sample.wave_enabled is False


def test_real_sample_preview_disables_synthetic_ray_scattering():
    controller = CalculationController()
    captured = []
    controller.pool.start = captured.append
    state = default_state()
    state.ac_deflector.wobble_enabled = False
    state.ac_deflector.scan_enabled = True
    state.sample.specimen_mode = "atomic"
    state.sample.cif_path = "real-sample.cif"
    state.sample.diffraction_enabled = True
    state.sample.stem_wave_enabled = True

    controller.submit(state, "Preview", 25, 5.0)

    snapshot = captured[0].state
    assert snapshot.sample.diffraction_enabled is False
    assert snapshot.sample.stem_wave_enabled is False


def test_virtual_sample_preview_keeps_explicit_interaction_channels():
    controller = CalculationController()
    captured = []
    controller.pool.start = captured.append
    state = default_state()
    state.sample.specimen_mode = "virtual"
    state.sample.diffraction_enabled = True

    controller.submit(state, "Preview", 25, 5.0)

    assert captured[0].state.sample.diffraction_enabled is True


def test_high_accuracy_preserves_selected_compute_backend():
    controller = CalculationController()
    captured = []
    controller.pool.start = captured.append
    state = default_state()
    state.acceleration_enabled = False
    state.acceleration_backend = "CPU"

    controller.submit(state, "High accuracy", 25, 5.0)

    assert captured[0].state.acceleration_enabled is False
    assert captured[0].state.acceleration_backend == "CPU"
