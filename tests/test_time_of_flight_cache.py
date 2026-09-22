"""Persistent clock round trips and actual tip-origin column cache reuse."""
from dataclasses import replace
import numpy as np
import pytest

from temsim.artifact_store import ArtifactIntegrityError, ArtifactStore, CHECKPOINT_CODEC, INCIDENT_SEED_CODEC
from temsim.physics.simulation import run
from test_segmented_column_cache import _small_vacuum_state
from test_calculation_manifest_artifacts import _assembled_state, _manifest, _checkpoints


def test_clock_checkpoint_roundtrip_preserves_float64_and_unknowns(tmp_path):
    state, selection = _assembled_state()
    manifest = _manifest(state, selection)
    clock = np.tile([1e-8, 1e-8+2e-17, np.nan, 1e-8+4e-17], (3, 1))
    checkpoint = replace(_checkpoints(), flight_time_s=clock)
    store = ArtifactStore(tmp_path, quota_bytes=100_000_000)
    store.put_propagation_checkpoints(manifest, checkpoint)
    restored = store.get_propagation_checkpoints(manifest)
    assert restored.flight_time_s.dtype == np.float64
    np.testing.assert_array_equal(restored.flight_time_s, clock)
    assert not restored.flight_time_s.flags.writeable


def test_actual_tip_clock_survives_persistent_restart_and_changed_lens(tmp_path):
    from temsim.calculation_manifest import capture_calculation_manifest
    state = _small_vacuum_state()
    first = run(state, optical_only=True)
    manifest = capture_calculation_manifest(state, ray_count=9, step_mm=state.step_mm)
    store = ArtifactStore(tmp_path, quota_bytes=100_000_000)
    store.put_incident_simulation_seed(manifest, first)
    seed = store.get_incident_simulation_seed(manifest)
    for old, new in ((first.incident, seed.incident), (first.gun_trace, seed.gun_trace),
                     (first.gun_trace.exit_bundle, seed.gun_trace.exit_bundle),
                     (first.incident_checkpoints, seed.incident_checkpoints)):
        np.testing.assert_array_equal(old.flight_time_s, new.flight_time_s)
    np.testing.assert_array_equal(first.gun_trace.equal_time_history.time_s,
                                  seed.gun_trace.equal_time_history.time_s)
    for old, new in zip(first.gun_trace.plane_arrivals, seed.gun_trace.plane_arrivals):
        np.testing.assert_array_equal(old.time_s, new.time_s)
    resumed = run(state, existing_simulation=seed, optical_only=True)
    assert resumed.metrics["column_segment_cache"]["mode"] == "full_incident"
    np.testing.assert_array_equal(first.incident.flight_time_s, resumed.incident.flight_time_s)
    # A one-column cache clock must not broadcast the first particle's time
    # across all nine histories, even when the optical plan is unchanged.
    correct_clock = seed.incident.flight_time_s
    seed.incident.flight_time_s = correct_clock[:, :1]
    repaired = run(state, existing_simulation=seed, optical_only=True)
    assert repaired.metrics["column_segment_cache"]["mode"] == "none"
    np.testing.assert_array_equal(repaired.incident.flight_time_s, first.incident.flight_time_s)
    for branch in repaired.branches.values():
        np.testing.assert_array_equal(branch.flight_time_s[0],
                                      np.where(first.incident.alive, correct_clock[-1], np.nan))
    seed.incident.flight_time_s = correct_clock
    state.lenses[1].percent += .01
    seed.incident.flight_time_s = correct_clock.astype(np.float32)
    no_suffix = run(state, existing_simulation=seed, optical_only=True)
    assert no_suffix.metrics["column_segment_cache"]["mode"] == "none"
    seed.incident.flight_time_s = correct_clock
    changed = run(state, existing_simulation=seed, optical_only=True)
    fresh = run(state, optical_only=True)
    np.testing.assert_array_equal(changed.incident.flight_time_s, fresh.incident.flight_time_s)
    np.testing.assert_array_equal(no_suffix.incident.flight_time_s, fresh.incident.flight_time_s)
    assert np.all(np.isfinite(fresh.incident.flight_time_s[-1, fresh.incident.alive]))
    assert np.all(fresh.incident.flight_time_s[-1, fresh.incident.alive] > 0)
    # History-only old seeds remain readable, but cannot supply a new clock.
    seed.incident.flight_time_s = None
    no_clock = run(state, existing_simulation=seed, optical_only=True)
    assert no_clock.metrics["column_segment_cache"]["mode"] == "none"


@pytest.fixture(scope="module")
def serialized_clock_seed():
    from temsim.calculation_manifest import capture_calculation_manifest
    state = _small_vacuum_state()
    first = run(state, optical_only=True)
    manifest = capture_calculation_manifest(state, ray_count=9, step_mm=state.step_mm)
    return first, manifest


@pytest.mark.parametrize("field,damage", [
    ("incident.flight_time_s", "shape"), ("gun.flight_time_s", "shape"),
    ("gun_exit.flight_time_s", "shape"), ("checkpoint.flight_time_s", "shape"),
    ("incident.flight_time_s", "float32"), ("gun_exit.flight_time_s", "float32"),
    ("checkpoint.flight_time_s", "float32"), ("incident.flight_time_s", "infinite"),
    ("gun.flight_time_s", "negative"),
])
def test_structurally_invalid_serialized_clocks_are_rejected(tmp_path, serialized_clock_seed, field, damage):
    first, manifest = serialized_clock_seed
    store = ArtifactStore(tmp_path, quota_bytes=100_000_000)
    store.put_incident_simulation_seed(manifest, first)
    signature = str(manifest.calculation_signatures["incident"])
    bundle = store.get_array_bundle(manifest, product_key="incident", dependency_signature=signature,
                                     codec=INCIDENT_SEED_CODEC)
    arrays = dict(bundle.arrays)
    value = arrays[field].copy()
    if damage == "shape":
        value = value[..., :1]
    elif damage == "float32":
        value = value.astype(np.float32)
    elif damage == "infinite":
        value.flat[0] = np.inf
    else:
        value.flat[0] = -1.
    arrays[field] = value
    # Repack valid checksums: semantic shape/type validation must catch this,
    # not merely the storage checksum detector.
    store.put_array_bundle(manifest, product_key="incident", dependency_signature=signature,
        codec=INCIDENT_SEED_CODEC, arrays=arrays, metadata=bundle.metadata)
    with pytest.raises(ArtifactIntegrityError, match="Incident seed metadata"):
        store.get_incident_simulation_seed(manifest)


def test_persistence_does_not_silently_upgrade_a_float32_checkpoint_clock(tmp_path):
    state, selection = _assembled_state()
    manifest = _manifest(state, selection)
    checkpoint = _checkpoints()
    store = ArtifactStore(tmp_path, quota_bytes=100_000_000)
    arrays = {name: getattr(checkpoint, name) for name in ("z_mm", "x_m", "tx_rad", "y_m", "ty_rad")}
    arrays["flight_time_s"] = np.zeros(checkpoint.x_m.shape, dtype=np.float32)
    store.put_array_bundle(manifest, product_key="incident",
        dependency_signature=str(manifest.calculation_signatures["incident"]),
        codec=CHECKPOINT_CODEC, arrays=arrays, metadata={})
    with pytest.raises(ArtifactIntegrityError, match="clock"):
        store.get_propagation_checkpoints(manifest)


def test_aggregate_loss_channels_never_invent_an_interaction_depth_clock(monkeypatch):
    from types import SimpleNamespace
    from temsim.specimen import inelastic, source
    state = _small_vacuum_state()
    # Explicit synthetic channel fixture: test clock eligibility independently
    # of a material database fit or stochastic event-depth implementation.
    monkeypatch.setattr(source, "specimen_is_vacuum", lambda _sample: False)
    distribution = SimpleNamespace(mean_inelastic_events=.5, absorbed_probability=0., metrics=lambda: {})
    monkeypatch.setattr(inelastic, "real_inelastic_distribution", lambda _state: distribution)
    monkeypatch.setattr(inelastic, "real_inelastic_ray_branches", lambda *_a, **_k: (
        inelastic.RealInelasticRayBranch("zero", 0., 0., .7, "real_zero_loss", 0.),
        inelastic.RealInelasticRayBranch("loss", 0., 0., .3, "real_plasmon", 20.),
    ))
    result = run(state)
    assert np.isnan(result.branches["loss"].flight_time_s).all()
    zero = result.branches["zero"]
    assert np.any(np.isfinite(zero.flight_time_s[0]))
    np.testing.assert_array_equal(zero.flight_time_s[0],
                                  np.where(result.incident.alive, result.incident.flight_time_s[-1], np.nan))
