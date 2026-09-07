"""Quota admission must not destroy usable artifacts on a rejected write."""

from dataclasses import replace

import numpy as np
import pytest

from temsim.artifact_store import ArtifactStore, ArtifactTooLargeError
from temsim.calculation_manifest import CalculationManifest, SolverIdentity


@pytest.fixture
def manifest():
    return CalculationManifest(
        created_at_utc="2026-09-07T00:00:00+00:00",
        state_payload={},
        selection={},
        calculation_signatures={"incident": "first"},
        state_model_signature="model",
        external_model_signature="external",
        geometry_fingerprint="geometry",
        solver=SolverIdentity(package_version="test", state_schema_version=1),
        external_inputs=(),
    )


def _different_request(manifest, name):
    return replace(manifest, calculation_signatures={"incident": name})


def _put(store, manifest, values, **kwargs):
    return store.put_array_bundle(
        manifest,
        product_key="incident",
        dependency_signature=manifest.calculation_signatures["incident"],
        arrays={"values": values},
        **kwargs,
    )


def _get(store, manifest):
    return store.get_array_bundle(
        manifest,
        product_key="incident",
        dependency_signature=manifest.calculation_signatures["incident"],
    )


def _files(store):
    return {
        path.relative_to(store.root): path.read_bytes()
        for path in store.root.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("replace_existing", [False, True])
@pytest.mark.parametrize("oversized_part", ["array", "metadata"])
def test_rejected_write_preserves_all_existing_artifacts(
    tmp_path, manifest, replace_existing, oversized_part,
):
    store = ArtifactStore(tmp_path / "cache", quota_bytes=1_000_000)
    other = _different_request(manifest, "other")
    values = np.arange(32, dtype=np.float64)
    _put(store, manifest, values)
    _put(store, other, values + 1)
    store.set_quota_bytes(store._store_size())
    before = _files(store)
    incoming = manifest if replace_existing else _different_request(manifest, "new")
    arrays = np.zeros(store.quota_bytes, dtype=np.float64) if oversized_part == "array" else values
    metadata = {"note": "x" * (2 * store.quota_bytes)} if oversized_part == "metadata" else {}

    with pytest.raises(ArtifactTooLargeError):
        _put(store, incoming, arrays, metadata=metadata)

    assert _files(store) == before
    np.testing.assert_array_equal(_get(store, manifest).arrays["values"], values)
    np.testing.assert_array_equal(_get(store, other).arrays["values"], values + 1)
    if not replace_existing:
        assert _get(store, incoming) is None


def test_rejected_deduplicated_write_preserves_shared_content(tmp_path, manifest):
    store = ArtifactStore(tmp_path / "cache", quota_bytes=1_000_000)
    values = np.arange(128, dtype=np.float64)
    _put(store, manifest, values)
    standalone_size = store._store_size()
    other = _different_request(manifest, "other")
    _put(store, other, values)
    before = _files(store)
    store.set_quota_bytes(standalone_size - 1)

    # The content object already exists, so rejection must not unlink it.
    with pytest.raises(ArtifactTooLargeError):
        _put(store, manifest, values)

    assert _files(store) == before
    assert _get(store, manifest) is not None
    assert _get(store, other) is not None


def test_successful_replacement_reclaims_old_content_without_evicting_other_request(
    tmp_path, manifest,
):
    store = ArtifactStore(tmp_path / "cache", quota_bytes=1_000_000)
    values = np.arange(128, dtype=np.float64)
    other = _different_request(manifest, "other")
    _put(store, manifest, values)
    _put(store, other, values + 1)
    previous_objects = set(store.objects_root.glob("*/*/manifest.json"))
    store.set_quota_bytes(store._store_size())

    _put(store, manifest, values + 2)

    np.testing.assert_array_equal(_get(store, manifest).arrays["values"], values + 2)
    np.testing.assert_array_equal(_get(store, other).arrays["values"], values + 1)
    current_objects = set(store.objects_root.glob("*/*/manifest.json"))
    assert len(current_objects) == 2
    assert len(previous_objects - current_objects) == 1
    assert store._store_size() <= store.quota_bytes


def test_admission_counts_array_headers_manifests_and_lock_overhead(tmp_path, manifest):
    values = np.arange(64, dtype=np.float64)
    probe = ArtifactStore(tmp_path / "probe", quota_bytes=1_000_000)
    _put(probe, manifest, values, metadata={"units": "m"})
    required_bytes = probe._store_size()
    assert required_bytes > values.nbytes

    rejected = ArtifactStore(tmp_path / "rejected", quota_bytes=required_bytes - 1)
    with pytest.raises(ArtifactTooLargeError):
        _put(rejected, manifest, values, metadata={"units": "m"})
    assert list(rejected.references_root.glob("*.json")) == []
    assert list(rejected.objects_root.glob("*/*/manifest.json")) == []

    exact = ArtifactStore(tmp_path / "exact", quota_bytes=required_bytes)
    _put(exact, manifest, values, metadata={"units": "m"})
    # Same-identity, same-content replacement needs no second object or ref.
    _put(exact, manifest, values, metadata={"units": "m"})
    assert exact._store_size() == required_bytes
    np.testing.assert_array_equal(_get(exact, manifest).arrays["values"], values)


def test_deduplicated_content_survives_lru_reference_eviction(tmp_path, manifest):
    store = ArtifactStore(tmp_path / "cache", quota_bytes=1_000_000)
    values = np.arange(128, dtype=np.float64)
    # Equal-length signatures keep reference sizes identical at the boundary.
    first = _different_request(manifest, "first")
    second = _different_request(manifest, "other")
    third = _different_request(manifest, "third")
    _put(store, first, values)
    _put(store, second, values)
    store.set_quota_bytes(store._store_size())

    # Mark the second reference most recent through the public read operation.
    assert _get(store, second) is not None
    _put(store, third, values)

    assert _get(store, first) is None
    np.testing.assert_array_equal(_get(store, second).arrays["values"], values)
    np.testing.assert_array_equal(_get(store, third).arrays["values"], values)
    assert len(list(store.objects_root.glob("*/*/manifest.json"))) == 1
    assert store._store_size() <= store.quota_bytes


def test_failed_reference_commit_preserves_previous_identity(tmp_path, manifest, monkeypatch):
    store = ArtifactStore(tmp_path / "cache", quota_bytes=1_000_000)
    values = np.arange(64, dtype=np.float64)
    _put(store, manifest, values)
    before = _files(store)

    def fail_write(_path, _data):
        raise OSError("Simulated reference-write failure")

    monkeypatch.setattr(store, "_write_file_atomic", fail_write)
    with pytest.raises(OSError, match="reference-write failure"):
        _put(store, manifest, values + 1)

    assert _files(store) == before
    np.testing.assert_array_equal(_get(store, manifest).arrays["values"], values)


def test_quota_change_after_admission_applies_to_next_write(tmp_path, manifest, monkeypatch):
    store = ArtifactStore(tmp_path / "cache", quota_bytes=1_000_000)
    values = np.arange(64, dtype=np.float64)
    write_atomic = store._write_file_atomic

    def reduce_quota_during_commit(path, data):
        store.set_quota_bytes(1)
        write_atomic(path, data)

    monkeypatch.setattr(store, "_write_file_atomic", reduce_quota_during_commit)
    _put(store, manifest, values)
    before = _files(store)
    with pytest.raises(ArtifactTooLargeError):
        _put(store, manifest, values + 1)
    assert _files(store) == before
    np.testing.assert_array_equal(_get(store, manifest).arrays["values"], values)
