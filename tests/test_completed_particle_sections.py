"""Full classical results retain executed restart data, without a new source."""
from dataclasses import replace

import numpy as np
import pytest

from temsim import simulation_pipeline as pipeline
from temsim.physics.particle_sections import validate_section_checkpoint


@pytest.fixture(scope="module")
def completed():
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.optics.column import default_state
    state = default_state()
    catalog = AssemblyCatalog()
    selection = catalog.default_selection()
    recording = next(o.name for o in catalog.recording_systems
                     if not bool(o.properties.get("energy_filter")))
    catalog.apply(state, replace(selection, recording=recording))
    state.electron_gun.emitter.ray_count = 9
    state.step_mm = 2.
    state.sample.inserted = False
    state.sample.eds_enabled = False
    state.acceleration_backend = "CPU"
    assert not state.sample.wave_enabled and not state.sample.stem_wave_enabled
    return pipeline.calculate(state)


def test_full_pipeline_retains_true_terminal_checkpoints(completed):
    simulation = completed.simulation
    cp = validate_section_checkpoint(simulation.section_checkpoint)
    assert tuple(s.name for s in cp.segments) == ("incident", "000")
    assert simulation.metrics["tuning_quality"] == "High accuracy"
    assert simulation.metrics["section_resumable_through_z_mm"] == cp.segments[-1].branch.z[-1]
    assert simulation.sample_to_analysis_transfer is not None  # Existing diagnostics survive.
    np.testing.assert_array_equal(simulation.incident.x[-1], cp.segments[0].checkpoints.x_m[-1])


def test_completed_full_result_reuses_gun_and_column_for_section(completed, monkeypatch):
    from temsim.instrument_snapshot import capture_instrument_snapshot
    state = capture_instrument_snapshot(completed.state_snapshot).restore()
    monkeypatch.setattr("temsim.optics.electron_gun.source.trace_source_to_exit",
                        lambda *a, **k: pytest.fail("Executed gun must not run again"))
    result = pipeline.calculate_particle_section(state,
        completed.simulation.metrics["section_target_z_mm"], existing_result=completed)
    assert result.simulation.metrics["section_gun_reused"]
    assert result.simulation.metrics["section_reused_prefix"]
    assert result.simulation.metrics["section_resume_z_mm"] == completed.simulation.metrics["section_target_z_mm"]
    for name in ("x", "y", "tx", "ty", "flight_time_s"):
        np.testing.assert_allclose(getattr(result.simulation.branches["000"], name)[-1],
            getattr(completed.simulation.branches["000"], name)[-1], rtol=1e-13, atol=1e-22)


def test_full_calculation_continues_saved_section_without_gun_retrace(completed, monkeypatch):
    from temsim.instrument_snapshot import capture_instrument_snapshot
    state = capture_instrument_snapshot(completed.state_snapshot).restore()
    first = pipeline.calculate_particle_section(state, state.sample.z_mm-20., existing_result=completed)
    monkeypatch.setattr("temsim.optics.electron_gun.source.trace_source_to_exit",
                        lambda *a, **k: pytest.fail("Section gun must be reused by normal full calculation"))
    full = pipeline.calculate(capture_instrument_snapshot(state).restore(), existing_result=first)
    assert full.simulation.metrics["column_segment_cache"]["hit"]
    assert full.simulation.metrics["column_segment_cache"]["resume_z_mm"] == state.sample.z_mm-20.
    validate_section_checkpoint(full.simulation.section_checkpoint)


def test_high_accuracy_archive_round_trip(completed, tmp_path, monkeypatch):
    from temsim.particle_section_io import save_section_result, load_section_result
    from temsim.instrument_snapshot import capture_instrument_snapshot
    path = tmp_path / "full-high.temsection"
    save_section_result(completed, path)
    loaded = load_section_result(path)
    assert loaded.simulation.metrics["tuning_quality"] == "High accuracy"
    assert loaded.state_snapshot.electron_gun.emitter.ray_count == 9
    monkeypatch.setattr("temsim.optics.electron_gun.source.trace_source_to_exit",
                        lambda *a, **k: pytest.fail("Loaded complete gun must be reused"))
    result = pipeline.calculate_particle_section(capture_instrument_snapshot(loaded.state_snapshot).restore(),
        loaded.simulation.metrics["section_target_z_mm"], existing_result=loaded)
    assert result.simulation.metrics["section_gun_reused"]
    assert result.simulation.metrics["section_reused_prefix"]
