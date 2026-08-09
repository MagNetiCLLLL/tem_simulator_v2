import pytest

from temsim.gui.calculation_controller import (
    CalculationController,
    HIGH_ACCURACY_MEMORY_BUDGET_BYTES,
    estimate_calculation_memory_bytes,
)
from temsim.optics.column import default_state


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
    assert set(result.simulation.branches) == {"000"}
    assert result.lens_crossovers
    assert all(item["verified"] for item in result.lens_crossovers)
    assert result.aperture_stops
    assert result.state_snapshot.objective_lens.cs_mm == 0.85
    assert result.state_snapshot.objective_lens.polarity == -1
    assert all("radius_mm" in item for item in result.aperture_stops)
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


def test_wave_imaging_is_disabled_only_for_preview():
    controller = CalculationController()
    captured = []
    controller.pool.start = captured.append
    state = default_state()
    state.sample.wave_enabled = True

    controller.submit(state, "High accuracy", 25, 5.0)
    controller.submit(state, "Preview", 25, 5.0)

    assert captured[0].state.sample.wave_enabled is True
    assert captured[1].state.sample.wave_enabled is False


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
