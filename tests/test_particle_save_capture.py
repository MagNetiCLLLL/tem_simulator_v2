"""Save requests pin mutable structure without copying executed numeric buffers."""
from dataclasses import dataclass

import numpy as np
import pytest

from temsim.instrument_snapshot import encode_instrument
from temsim.optics.column import default_state
from temsim.particle_section_io import capture_result_for_save
from temsim.physics.simulation import Branch, Simulation
from temsim.simulation_pipeline import CalculationResult


@pytest.fixture
def completed_result():
    state = default_state()
    state._tuning_quality = "Medium"
    state._tuning_step_mm = 0.12345678901234567
    history = np.arange(32., dtype=np.float64).reshape(4, 8)
    branch = Branch("incident", (1., 0., 0.), np.arange(4.), history, history,
                    history, history, np.ones(8, dtype=bool), np.full(8, np.inf),
                    [""] * 8, 1., np.zeros(8))
    branch.vacuum_report = {"settings": {"enabled": False}}
    simulation = Simulation(branch, {"000": branch}, {
        "tuning_quality": "Medium", "section_target_z_mm": 1234.56789012345,
        "section_component_keys": ["objective_lens"], "detail": {"count": 8}})
    return CalculationResult(simulation, None, state_snapshot=state,
        signatures={"request": "original"}, calculated_products=frozenset({"incident"}),
        performance={"stages": [{"seconds": 0.12345678901234567}]})


def test_capture_detaches_later_enrichment_nested_metadata_and_branch_controls(completed_result):
    result = completed_result
    captured = capture_result_for_save(result)
    result.sample_region = object()
    result.specimen_interactions = object()
    result.signatures["request"] = "enriched"
    result.calculated_products = frozenset({"incident", "sample_region"})
    result.performance["stages"][0]["seconds"] = 99.
    result.simulation.metrics["section_target_z_mm"] = 999.
    result.simulation.metrics["tuning_quality"] = "Preview"
    result.simulation.metrics["section_component_keys"].append("projector_lens_1")
    result.simulation.metrics["detail"]["count"] = 3
    result.simulation.incident.blocked_key[0] = "later-stop"
    result.simulation.incident.vacuum_report["settings"]["enabled"] = True
    result.simulation.branches.clear()

    assert captured is not result and captured.simulation is not result.simulation
    assert captured.sample_region is captured.specimen_interactions is None
    assert captured.signatures == {"request": "original"}
    assert captured.calculated_products == frozenset({"incident"})
    assert captured.performance == {"stages": [{"seconds": 0.12345678901234567}]}
    assert captured.simulation.metrics == {
        "tuning_quality": "Medium", "section_target_z_mm": 1234.56789012345,
        "section_component_keys": ["objective_lens"], "detail": {"count": 8}}
    assert captured.simulation.incident is captured.simulation.branches["000"]
    assert captured.simulation.incident.blocked_key == [""] * 8
    assert captured.simulation.incident.vacuum_report == {"settings": {"enabled": False}}
    assert captured.simulation.incident.x is result.simulation.incident.x


@pytest.mark.parametrize("readonly", [False, True])
def test_capture_preserves_exact_snapshot_and_shares_large_input_arrays(completed_result, readonly):
    result = completed_result
    # Non-contiguous, non-native input bytes remain untouched on the owner thread.
    backing = np.arange(65_536., dtype=">f8").reshape(256, 256)
    large = backing[:, ::2]
    large.setflags(write=not readonly)
    result.state_snapshot.capture_test_array = large
    result.state_snapshot.capture_test_alias = large
    before = encode_instrument(result.state_snapshot)
    captured = capture_result_for_save(result)

    assert captured.state_snapshot is not result.state_snapshot
    assert encode_instrument(captured.state_snapshot) == before
    assert captured.state_snapshot.capture_test_array is large
    assert captured.state_snapshot.capture_test_alias is large
    assert bool(large.flags.writeable) is not readonly
    result.state_snapshot.sample.inserted = not result.state_snapshot.sample.inserted
    result.state_snapshot._tuning_quality = "Preview"
    result.state_snapshot._tuning_step_mm = 2.
    result.state_snapshot.capture_test_array = np.zeros((1,))
    assert encode_instrument(captured.state_snapshot) == before


def test_capture_shares_completed_frozen_products_without_traversing_event_graphs(completed_result):
    @dataclass(frozen=True)
    class PublishedProduct:
        events: tuple

    product = PublishedProduct((object(), object()))
    completed_result.sample_region = product
    captured = capture_result_for_save(completed_result)
    assert captured.sample_region is product


def test_capture_rejects_unknown_result_contract_and_coherent_product(completed_result):
    with pytest.raises(ValueError, match="explicit archive contract"):
        capture_result_for_save(object())
    completed_result.wave_imaging = object()
    with pytest.raises(ValueError, match="wave archive"):
        capture_result_for_save(completed_result)
