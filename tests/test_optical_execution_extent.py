"""Optical coverage receipts: presentation fixtures and bounded CPU execution.

These checks establish executed axial coverage only; the nine-ray run does not
qualify optical accuracy, specimen interactions, or an energy-filter trajectory.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.gui.ray_extent_data import completed_ray_extent
from temsim.physics.simulation import run
from test_particle_sections import fixture as optical_fixture
from test_segmented_column_cache import _small_vacuum_state


def optical_result():
    return SimpleNamespace(simulation=SimpleNamespace(
        section_checkpoint=None,
        metrics={
            "optical_tuning": True,
            "sample_scattering_applied": False,
            "optical_execution_extent": {
                "coordinate_system": "column_axial_z_mm",
                "physics_scope": "optical_reference_without_specimen_interactions",
                "start_z_mm": 0.,
                "completed_z_mm": 2800.5,
            },
        },
        branches={"tail": SimpleNamespace(z=np.array([500., 8000.]))},
    ))


def test_optical_receipt_supplies_coverage_without_admitting_a_restart():
    result = optical_result()
    result.simulation.metrics.update(
        section_target_z_mm=9000., section_resumable_through_z_mm=2700.)
    result.simulation.metrics["optical_execution_extent"]["resumable_z_mm"] = 2700.
    assert completed_ray_extent(result) == dict(
        start_z_mm=0., completed_z_mm=2800.5, resumable_z_mm=None)


@pytest.mark.parametrize("field,value", [
    ("start_z_mm", None), ("start_z_mm", float("nan")),
    ("start_z_mm", "0"), ("start_z_mm", True),
    ("completed_z_mm", None), ("completed_z_mm", float("inf")),
    ("completed_z_mm", float("nan")), ("completed_z_mm", "2800.5"),
    ("completed_z_mm", True), ("completed_z_mm", -1.),
    ("coordinate_system", "energy_filter_local_path_mm"),
    ("physics_scope", "particle_section"),
])
def test_invalid_optical_receipt_cannot_claim_coverage(field, value):
    result = optical_result()
    result.simulation.metrics["optical_execution_extent"][field] = value
    assert all(value is None for value in completed_ray_extent(result).values())


@pytest.mark.parametrize("field,value", [
    ("optical_tuning", False), ("optical_tuning", None),
    ("sample_scattering_applied", True), ("sample_scattering_applied", None),
    ("optical_execution_extent", None), ("optical_execution_extent", []),
])
def test_missing_or_inconsistent_optical_scope_remains_unavailable(field, value):
    result = optical_result()
    result.simulation.metrics[field] = value
    assert all(value is None for value in completed_ray_extent(result).values())


def test_actual_optical_execution_retains_endpoint_after_display_or_request_changes():
    from temsim.physics.particle_sections import section_limits

    state = _small_vacuum_state()
    state.energy_filter.enabled = False
    simulation = run(state, optical_only=True)
    completed = section_limits(state)[1]
    start = float(simulation.gun_trace.z_mm[0])
    assert simulation.section_checkpoint is None
    assert "section_target_z_mm" not in simulation.metrics
    assert simulation.metrics["sample_scattering_applied"] is False
    assert simulation.metrics["sample_scattering_model"] == "omitted_for_optical_tuning"
    assert simulation.metrics["tuning_quality"] == "requested_resolution"
    assert simulation.real_interactions is None
    assert simulation.incident_plan.z_mm[-1] == state.sample.z_mm
    assert simulation.branches["000"].z[-1] == completed

    # Coverage belongs to the completed solver invocation, even if presentation
    # data are thinned/replaced and the mutable current selection changes.
    simulation.branches["000"].z = np.array([state.sample.z_mm, completed + 1000.])
    simulation.gun_trace = SimpleNamespace(z_mm=np.array([start + 1., state.sample.z_mm]))
    state.energy_filter.enabled = True
    expected = dict(start_z_mm=start, completed_z_mm=completed, resumable_z_mm=None)
    assert completed_ray_extent(SimpleNamespace(simulation=simulation)) == expected


def test_optical_execution_receipt_uses_filter_entrance_in_bounded_drift_fixture(optical_fixture):
    # Synthetic source in an explicit short drift, with the actual column
    # propagation kernel. This checks the coverage boundary, not filter optics.
    state, _, _ = optical_fixture
    state.energy_filter_installed = True
    state.energy_filter_mode = "energy_filter"
    state.energy_filter.enabled = True
    state.energy_filter.entrance_z_mm = 456.
    simulation = run(state, optical_only=True, resolved_layout=object())
    endpoint = float(state.energy_filter.entrance_z_mm)
    assert simulation.branches["000"].z[-1] == endpoint
    extent = completed_ray_extent(SimpleNamespace(simulation=simulation))
    assert extent["completed_z_mm"] == endpoint
    assert extent["resumable_z_mm"] is None
    assert simulation.metrics["optical_execution_extent"]["coordinate_system"] == "column_axial_z_mm"
    assert simulation.section_checkpoint is None


def test_optical_execution_cannot_publish_an_unreached_requested_boundary(optical_fixture, monkeypatch):
    import temsim.physics.simulation as simulation_module

    state, _, _ = optical_fixture
    state.energy_filter_installed = True
    state.energy_filter_mode = "energy_filter"
    state.energy_filter.enabled = True
    state.energy_filter.entrance_z_mm = 456.
    propagate = simulation_module.propagate

    def missing_endpoint(*args, **kwargs):
        values = propagate(*args, **kwargs)
        # Simulate an incomplete solver response, before Branch creation. The
        # configured target alone must never establish successful execution.
        z = values[0].copy()
        z[-1] -= .1
        return (z, *values[1:])

    monkeypatch.setattr(simulation_module, "propagate", missing_endpoint)
    with pytest.raises(ValueError, match="did not reach its executed column boundary"):
        run(state, optical_only=True, resolved_layout=object())
