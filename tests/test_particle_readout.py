"""Detector signals use actual stops and source-normalised particle weights."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from test_particle_sections import fixture
from temsim.simulation_pipeline import calculate_particle_section
from temsim.detector.particle_readout import measure_particle_detectors
from temsim.physics.beam_current import effective_source_current_pa


def result_at(fixture, monkeypatch, target):
    state, _, _ = fixture
    monkeypatch.setattr("temsim.simulation_pipeline.apply_physical_layout_to_state", lambda s: object())
    monkeypatch.setattr("temsim.simulation_pipeline.ensure_recording_system", lambda s: None)
    state.camera.set_optical_reference_z_mm(state.selected_area_aperture.z_mm, 455.)
    state.camera.inserted = True
    state.camera.readout_enabled = True
    state.camera.outer_width_mm = 100.
    state.energy_filter.enabled = False
    return calculate_particle_section(state, target)


def camera_row(result, **kwargs):
    return next(row for row in measure_particle_detectors(result, **kwargs)
                if row.key == result.state_snapshot.camera.key)


def test_counts_follow_absorption_and_weight_not_surviving_ray_count(fixture, monkeypatch):
    result = result_at(fixture, monkeypatch, 456.)
    assert not any(b.alive.any() for b in result.simulation.branches.values())
    row = camera_row(result, exposure_s=.002)
    assert row.status == "AVAILABLE"
    assert row.fraction == pytest.approx(1.)
    assert row.simulated_electrons == pytest.approx(result.state_snapshot.electron_gun.ray_count)
    assert row.current_pa == pytest.approx(effective_source_current_pa(result.state_snapshot))
    assert row.expected_electrons == pytest.approx(row.electrons_per_second*.002)
    # Deselecting readout cannot resurrect absorbed trajectories.
    result.state_snapshot.camera.readout_enabled = False
    assert camera_row(result).status == "READOUT_DISABLED"
    assert not any(b.alive.any() for b in result.simulation.branches.values())


def test_unreached_detectors_are_unavailable_instead_of_zero(fixture, monkeypatch):
    result = result_at(fixture, monkeypatch, 454.5)
    row = camera_row(result)
    assert row.status == "NOT_REACHED"
    assert row.simulated_electrons is None


def test_weighted_zero_and_fractional_numerical_counts_are_valid(fixture, monkeypatch):
    result = result_at(fixture, monkeypatch, 456.)
    branch = next(iter(result.simulation.branches.values()))
    result.simulation.branches = {"test": replace(branch,
        ray_weight=np.array([.1,.2,.7]), blocked_key=(result.state_snapshot.camera.key,"other","other"))}
    assert camera_row(result).fraction == pytest.approx(.1)
    result.simulation.branches = {"test": replace(branch, blocked_key=("other",)*3)}
    row = camera_row(result)
    assert row.status == "AVAILABLE" and row.simulated_electrons == 0.


def test_missing_material_transport_cannot_be_reported_as_detector_signal(fixture, monkeypatch):
    result = result_at(fixture, monkeypatch, 456.)
    result.state_snapshot.sample.inserted = True
    result.state_snapshot.sample.specimen_mode = "reference"
    assert camera_row(result).status == "NOT_CALCULATED"
    result.state_snapshot.sample.inserted = False
    result.simulation.metrics["optical_tuning"] = True
    assert camera_row(result).status == "NOT_CALCULATED"


def test_invalid_probability_and_exposure_fail_explicitly(fixture, monkeypatch):
    result = result_at(fixture, monkeypatch, 456.)
    branch = next(iter(result.simulation.branches.values()))
    result.simulation.branches = {"test": replace(branch, ray_weight=np.ones(3))}
    with pytest.raises(ValueError, match="exceeds"):
        camera_row(result)
    with pytest.raises(ValueError, match="exposure"):
        camera_row(result, exposure_s=-1.)


def test_filter_readout_requires_executed_bent_path_not_axial_extrapolation(fixture, monkeypatch):
    result = result_at(fixture, monkeypatch, 456.)
    result.state_snapshot.energy_filter_installed = True
    result.state_snapshot.energy_filter_mode = "energy_filter"
    result.state_snapshot.energy_filter.enabled = True
    result.state_snapshot.energy_filter.output_detector_inserted = True
    filtered = {r.key: r for r in measure_particle_detectors(result)}
    assert filtered["energy_filter_output_detector"].status == "NOT_REACHED"
    result.energy_filter = SimpleNamespace(camera_recorded_fraction=.2, eels_transmitted_fraction=.1)
    filtered = {r.key: r for r in measure_particle_detectors(result)}
    assert filtered["energy_filter_output_detector"].fraction == .2
    assert filtered["energy_loss_detector"].fraction == .1
