"""Probability bookkeeping fixtures; field and downstream transport are stubbed."""
from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.specimen import downstream_transport as downstream
from temsim.specimen.elastic_transport import ElasticTerminalBundle
from test_downstream_transport import (
    _test_state, _test_transport, _test_inelastic_distribution, _stub_times,
)


@pytest.fixture
def ledger_transport(monkeypatch):
    def propagate(state, start, stop, x, tx, y, ty, events, energy, *, save_z_mm=(), **kwargs):
        z = np.asarray((start, *save_z_mm, stop))
        return (z, *(np.tile(values, (len(z), 1)) for values in (x, tx, y, ty)),
                _stub_times(z, tx, ty, energy, **kwargs))
    monkeypatch.setattr(downstream, "propagate", propagate)
    monkeypatch.setattr(downstream, "SpecimenFieldTransport", lambda state: SimpleNamespace(
        to_plane=lambda position, direction, plane, **kwargs: (position, direction, 0.)))
    for name in ("clip_recording_planes", "clip_column_wall"):
        monkeypatch.setattr(downstream, name,
            lambda state, z, x, y, alive, blocked, keys: (alive, blocked, keys))

    def run(*, scale=1., mixed=False, corrupt=None, distribution=None):
        count = 5000
        weights = np.full(count, 1. / count) * scale
        if corrupt is not None:
            weights[0] = corrupt
        outcomes = np.full(count, "transmitted", dtype=object)
        material = np.zeros(count)
        if mixed:
            outcomes[::5] = "backscattered"
            material[::3] = 10.
        source_ids = 3 * np.arange(count) + 7
        terminal = ElasticTerminalBundle(
            source_ray_index=np.arange(count), position_nm=np.zeros((count, 3)),
            direction=np.tile((0., 0., 1.), (count, 1)),
            kinetic_energy_ev=np.full(count, 200_000.), weight=weights,
            outcome=tuple(outcomes), event_count=np.zeros(count, dtype=int),
            has_scattered=(np.arange(count) % 2 == 0 if mixed else np.zeros(count, dtype=bool)),
            reference_time_offset_s=np.zeros(count), material_path_nm=material)
        simulation = SimpleNamespace(incident=SimpleNamespace(
            alive=np.ones(count, dtype=bool), ray_weight=np.full(count, 1. / count),
            z=np.array([1.]), flight_time_s=np.full((1, count), 1e-9),
            source_ray_id=source_ids, source_azimuth_rad=np.zeros(count)))
        original = weights.copy()
        result = downstream.build_geometric_specimen_exit(_test_state(), simulation,
            replace(_test_transport(), terminal_electrons=terminal),
            distribution if distribution is not None else _test_inelastic_distribution() if mixed else None,
            stop_z_mm=3., save_z_mm=(2.,), dependency_signature="ledger-fixture")
        np.testing.assert_array_equal(terminal.weight, original)
        return result, source_ids
    return run


def assert_bounded_ledger(result):
    assert downstream.validated_geometric_specimen_exit(result, "ledger-fixture") is result
    metrics = result.metrics
    keys = ("tracked_downstream_source_probability", "inelastic_absorbed_source_probability",
            "elastic_nontransmitted_source_probability", "pre_sample_lost_source_probability")
    assert all(0. <= float(metrics[key]) <= 1. for key in keys)
    assert math.fsum(metrics[key] for key in keys) == pytest.approx(1., abs=5e-12)
    assert math.fsum(branch.weight for branch in result.branches) == pytest.approx(
        metrics["tracked_downstream_source_probability"], abs=5e-12)
    assert 0. <= metrics["exit_plane_source_probability"] <= 1.
    assert 0. <= metrics["exit_plane_weight"] <= 1.


def test_5000_equal_weights_fully_transmitted_stay_inside_strict_ledger(ledger_transport):
    # The original Windows run summed this population to 1.0000000000000002.
    # The ledger contract must hold independently of NumPy's summation order.
    result, source_ids = ledger_transport()
    assert_bounded_ledger(result)
    assert result.metrics["tracked_downstream_source_probability"] == 1.
    assert result.metrics["elastic_forward_conditional_probability"] == 1.
    assert result.metrics["elastic_nontransmitted_conditional_probability"] == 0.
    assert result.metrics["elastic_nontransmitted_source_probability"] == 0.
    assert result.metrics["inelastic_absorbed_source_probability"] == 0.
    assert len(result.branches) == 1
    assert result.branches[0].weight == 1.
    assert sum(len(branch.source_ray_id) for branch in result.branches) == 5000
    np.testing.assert_array_equal(np.sort(np.concatenate([
        branch.source_ray_id for branch in result.branches])), source_ids)


def test_5000_mixed_material_misses_losses_and_nontransmitted_form_one_ledger(ledger_transport):
    result, _ = ledger_transport(mixed=True)
    assert_bounded_ledger(result)
    assert result.metrics["elastic_nontransmitted_conditional_probability"] == pytest.approx(.2)
    assert result.metrics["inelastic_absorbed_source_probability"] > 0.
    assert result.metrics["vacuum_miss_forward_conditional_probability"] > 0.
    assert all(math.fsum(branch.ray_weight) == pytest.approx(1., abs=2e-15)
               for branch in result.branches)


@pytest.mark.parametrize("scale", (1. + 1e-13, 1. - 1e-13))
def test_admitted_conditional_roundoff_uses_local_normalisation(ledger_transport, scale):
    result, _ = ledger_transport(scale=scale)
    assert_bounded_ledger(result)
    assert result.metrics["tracked_downstream_source_probability"] == pytest.approx(1., abs=2e-15)
    assert result.metrics["elastic_nontransmitted_source_probability"] == 0.


@pytest.mark.parametrize("scale", (1. + 1e-10, 1. - 1e-10))
def test_small_but_invalid_conditional_total_is_still_rejected(ledger_transport, scale):
    with pytest.raises(ValueError, match="conditional weights must sum to one"):
        ledger_transport(scale=scale)


@pytest.mark.parametrize("corrupt", (-1e-18, np.nan, np.inf))
def test_invalid_individual_weights_are_never_normalised_away(ledger_transport, corrupt):
    with pytest.raises(ValueError, match="Elastic terminal"):
        ledger_transport(corrupt=corrupt)


@pytest.mark.parametrize("value", (1. + 1e-10, -1e-10, np.nextafter(0., -np.inf), np.nan, np.inf))
def test_derived_bound_rejects_more_than_ledger_roundoff(value):
    with pytest.raises(ValueError, match="outside"):
        downstream._bounded_probability(value)


def test_derived_bound_only_repairs_endpoint_roundoff():
    assert downstream._bounded_probability(np.nextafter(1., np.inf)) == 1.
    assert downstream._bounded_probability(0.) == 0.


def test_small_negative_absorption_is_not_hidden_by_roundoff_repair(ledger_transport):
    distribution = _test_inelastic_distribution()
    distribution.channels = (SimpleNamespace(key="real_zero_loss", probability=1. + 1e-13,
        characteristic_angle_mrad=0., energy_loss_ev=0.),)
    distribution.tracked_probability = 1. + 1e-13
    distribution.absorbed_probability = -1e-13
    with pytest.raises(ValueError, match="outside"):
        ledger_transport(mixed=True, distribution=distribution)


@pytest.mark.parametrize("tracked", (1. + 1e-10, -1e-10))
def test_checkpoint_validator_remains_strict_for_invalid_input(tracked):
    result = downstream.GeometricSpecimenExit((SimpleNamespace(weight=max(tracked, 0.)),),
        {"tracked_downstream_source_probability": tracked,
         "inelastic_absorbed_source_probability": 0.}, dependency_signature="strict")
    assert downstream.validated_geometric_specimen_exit(result, "strict") is None
