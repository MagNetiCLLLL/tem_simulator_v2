from types import SimpleNamespace

import numpy as np
import pytest

import temsim.specimen.downstream_transport as downstream_transport
from temsim.specimen.downstream_transport import (
    GeometricSpecimenExit,
    build_geometric_specimen_exit,
    validated_geometric_specimen_exit,
)
from temsim.specimen.elastic_transport import (
    ElasticTerminalBundle,
    ElasticTransportResult,
)


def _test_state():
    return SimpleNamespace(
        beam_voltage_kv=200.0,
        chromatic_aberration_enabled=False,
        objective_lens=SimpleNamespace(cc_mm=2.0),
        sample=SimpleNamespace(z_mm=1.0),
        lenses=(),
        stigmators=(),
        condenser_system={},
        deflectors=(),
        corrector_elements=(),
    )


def _test_simulation():
    return SimpleNamespace(
        incident=SimpleNamespace(
            alive=np.asarray((True, False, True)),
            ray_weight=np.asarray((0.2, 0.3, 0.5)),
        )
    )


def _test_transport():
    second_direction = np.asarray((0.1, 0.0, np.sqrt(0.99)))
    terminal = ElasticTerminalBundle(
        source_ray_index=np.asarray((0, 2, 0)),
        position_nm=np.asarray(
            ((10.0, 0.0, 5.0), (0.0, 20.0, 10.0), (0.0, 0.0, 0.0))
        ),
        direction=np.asarray(
            ((0.0, 0.0, 1.0), second_direction, (0.0, 0.0, -1.0))
        ),
        kinetic_energy_ev=np.full(3, 200_000.0),
        weight=np.asarray((0.2, 0.5, 0.3)),
        outcome=("transmitted", "transmitted", "backscattered"),
        event_count=np.asarray((0, 1, 1)),
        has_scattered=np.asarray((False, True, True)),
    )
    return ElasticTransportResult(
        eds_tracks=(),
        trajectories=(),
        metrics={},
        terminal_electrons=terminal,
    )


def _test_inelastic_distribution():
    return SimpleNamespace(
        channels=(
            SimpleNamespace(
                key="real_zero_loss",
                probability=0.75,
                characteristic_angle_mrad=0.0,
                energy_loss_ev=0.0,
            ),
            SimpleNamespace(
                key="real_plasmon",
                probability=0.15,
                characteristic_angle_mrad=2.0,
                energy_loss_ev=20.0,
            ),
        ),
        tracked_probability=0.9,
        absorbed_probability=0.1,
    )


def _signed_exit(*, tracked=0.0, absorbed=0.0, branch_weights=()):
    return GeometricSpecimenExit(
        tuple(SimpleNamespace(weight=weight) for weight in branch_weights),
        {
            "tracked_downstream_source_probability": tracked,
            "inelastic_absorbed_source_probability": absorbed,
        },
        dependency_signature="current",
    )


def test_geometric_exit_multiplies_elastic_and_inelastic_probabilities(monkeypatch):
    def fake_propagate(
        _state,
        start_z,
        stop_z,
        x,
        tx,
        y,
        ty,
        _events,
        _energy,
        *,
        save_z_mm=(),
        include_initial_plane_kicks=True,
    ):
        z = np.asarray((start_z, *save_z_mm, stop_z), dtype=float)
        return (
            z,
            np.tile(np.asarray(x, dtype=float), (z.size, 1)),
            np.tile(np.asarray(tx, dtype=float), (z.size, 1)),
            np.tile(np.asarray(y, dtype=float), (z.size, 1)),
            np.tile(np.asarray(ty, dtype=float), (z.size, 1)),
        )

    monkeypatch.setattr(downstream_transport, "determine_tem_stop_z", lambda _s: 3.0)
    monkeypatch.setattr(downstream_transport, "propagate", fake_propagate)
    monkeypatch.setattr(
        downstream_transport,
        "clip_recording_planes",
        lambda _s, _z, _x, _y, alive, blocked, keys: (alive, blocked, keys),
    )
    monkeypatch.setattr(
        downstream_transport,
        "clip_column_wall",
        lambda _s, _z, _x, _y, alive, blocked, keys: (alive, blocked, keys),
    )

    result = build_geometric_specimen_exit(
        _test_state(),
        _test_simulation(),
        _test_transport(),
        _test_inelastic_distribution(),
        save_z_mm=(2.0,),
    )

    assert len(result.branches) == 4
    assert sum(branch.weight for branch in result.branches) == pytest.approx(0.441)
    assert result.metrics["sample_incident_source_probability"] == pytest.approx(0.7)
    assert result.metrics["elastic_forward_conditional_probability"] == pytest.approx(0.7)
    assert result.metrics["inelastic_tracked_conditional_probability"] == pytest.approx(0.9)
    assert result.metrics["tracked_downstream_source_probability"] == pytest.approx(0.441)
    assert result.metrics["inelastic_absorbed_source_probability"] == pytest.approx(0.049)
    assert result.metrics["elastic_nontransmitted_source_probability"] == pytest.approx(0.21)
    assert result.metrics["pre_sample_lost_source_probability"] == pytest.approx(0.3)
    assert result.metrics["source_probability_conserved"]
    assert result.metrics["exit_plane_weight"] == pytest.approx(0.63)


def test_geometric_exit_back_projects_terminal_line_to_common_sample_plane(
    monkeypatch,
):
    captured_x = []

    def fake_propagate(
        _state,
        start_z,
        stop_z,
        x,
        tx,
        y,
        ty,
        _events,
        _energy,
        *,
        save_z_mm=(),
        include_initial_plane_kicks=True,
    ):
        captured_x.extend(np.asarray(x, dtype=float).tolist())
        z = np.asarray((start_z, stop_z), dtype=float)
        return (
            z,
            np.tile(np.asarray(x, dtype=float), (2, 1)),
            np.tile(np.asarray(tx, dtype=float), (2, 1)),
            np.tile(np.asarray(y, dtype=float), (2, 1)),
            np.tile(np.asarray(ty, dtype=float), (2, 1)),
        )

    monkeypatch.setattr(downstream_transport, "determine_tem_stop_z", lambda _s: 3.0)
    monkeypatch.setattr(downstream_transport, "propagate", fake_propagate)
    monkeypatch.setattr(
        downstream_transport,
        "clip_recording_planes",
        lambda _s, _z, _x, _y, alive, blocked, keys: (alive, blocked, keys),
    )
    monkeypatch.setattr(
        downstream_transport,
        "clip_column_wall",
        lambda _s, _z, _x, _y, alive, blocked, keys: (alive, blocked, keys),
    )

    result = build_geometric_specimen_exit(
        _test_state(),
        _test_simulation(),
        _test_transport(),
    )

    expected_scattered_x_m = (
        -10.0 * 0.1 / np.sqrt(0.99)
    ) * 1.0e-9
    assert captured_x == pytest.approx((10.0e-9, expected_scattered_x_m))
    assert not result.metrics["terminal_state_projection_preserves_free_flight_line"]
    assert result.metrics["terminal_state_projection_model"] == (
        "inverse shared-vector-field reference-plane matching"
    )


def test_geometric_exit_requires_matching_internal_provenance():
    signed = GeometricSpecimenExit(
        (),
        {
            "tracked_downstream_source_probability": 0.0,
            "inelastic_absorbed_source_probability": 0.0,
        },
        dependency_signature="current",
    )

    assert validated_geometric_specimen_exit(signed, "current") is signed
    assert validated_geometric_specimen_exit(signed, "stale") is None
    assert validated_geometric_specimen_exit(
        GeometricSpecimenExit((), {}),
        "current",
    ) is None

    signed.metrics["sample_downstream_signature"] = "tampered"
    assert validated_geometric_specimen_exit(signed, "current") is None


def test_geometric_exit_rejects_conflicting_metric_signature():
    with pytest.raises(ValueError, match="cannot be relabelled"):
        GeometricSpecimenExit(
            (),
            {
                "sample_downstream_signature": "old",
                "tracked_downstream_source_probability": 0.0,
                "inelastic_absorbed_source_probability": 0.0,
            },
            dependency_signature="new",
        )


def test_geometric_exit_rejects_incomplete_probability_ledger():
    incomplete = GeometricSpecimenExit(
        (),
        {"tracked_downstream_source_probability": 0.0},
        dependency_signature="current",
    )

    assert validated_geometric_specimen_exit(incomplete, "current") is None


@pytest.mark.parametrize(
    ("tracked", "absorbed"),
    (
        (-0.1, 0.0),
        (1.1, 0.0),
        (float("nan"), 0.0),
        (float("inf"), 0.0),
        (0.0, -0.1),
        (0.0, 1.1),
        (0.0, float("nan")),
        (0.0, float("inf")),
    ),
)
def test_geometric_exit_rejects_invalid_source_probabilities(tracked, absorbed):
    checkpoint = _signed_exit(tracked=tracked, absorbed=absorbed)

    assert validated_geometric_specimen_exit(checkpoint, "current") is None


def test_geometric_exit_checks_probability_partition_with_roundoff_tolerance():
    within_tolerance = _signed_exit(
        tracked=0.6,
        absorbed=0.4 + 1.0e-12,
        branch_weights=(0.2, 0.4),
    )
    beyond_tolerance = _signed_exit(
        tracked=0.6,
        absorbed=0.4 + 1.0e-10,
        branch_weights=(0.2, 0.4),
    )

    assert validated_geometric_specimen_exit(within_tolerance, "current") is (
        within_tolerance
    )
    assert validated_geometric_specimen_exit(beyond_tolerance, "current") is None


@pytest.mark.parametrize("weight", (-0.1, float("nan"), float("inf")))
def test_geometric_exit_rejects_invalid_branch_weights(weight):
    checkpoint = _signed_exit(tracked=0.5, branch_weights=(0.5, weight))

    assert validated_geometric_specimen_exit(checkpoint, "current") is None


def test_geometric_exit_requires_branch_weights_to_match_tracked_probability():
    within_tolerance = _signed_exit(
        tracked=0.3,
        branch_weights=(0.1, 0.2 + 1.0e-12),
    )
    beyond_tolerance = _signed_exit(
        tracked=0.3,
        branch_weights=(0.1, 0.2 + 1.0e-8),
    )

    assert validated_geometric_specimen_exit(within_tolerance, "current") is (
        within_tolerance
    )
    assert validated_geometric_specimen_exit(beyond_tolerance, "current") is None
