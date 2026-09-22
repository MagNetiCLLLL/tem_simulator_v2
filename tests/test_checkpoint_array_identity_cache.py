"""Content checksums are reused only for detached immutable numeric payloads."""
from hashlib import sha256
import json
from zipfile import ZIP_STORED, ZipFile

import numpy as np
import pytest

from temsim import working_point as module
from temsim.immutable_json import json_digest, thaw_json
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state


@pytest.fixture
def checkpoint():
    # Plain numeric storage exercises endian, stride, scalar and complex data;
    # no physical transport or coherent-source calculation is requested.
    arrays = {
        "history": np.arange(60, dtype=np.float64).reshape(6, 10)[:, ::2],
        "clock": np.array([1.25, 2.5], dtype=">f8"),
        "mask": np.array([True, False]),
        "complex": np.array([1 + 2j, -3 - 4j], dtype=np.complex128),
        "scalar": np.asarray(7, dtype=np.int32),
    }
    state = default_state()
    return module.WorkingPointCheckpoint(
        capture_instrument_snapshot(state), arrays, state.sample.z_mm,
        "storage-only-fixture", {"scope": "numeric package validation"},
    )


def _count_content_hashes(monkeypatch):
    original = module._array_identity
    calls = []

    def counted(value):
        calls.append((value.shape, value.dtype.str))
        return original(value)

    monkeypatch.setattr(module, "_array_identity", counted)
    return calls


def _legacy_identities(checkpoint):
    return {
        key: {"shape": list(value.shape), "dtype": value.dtype.str,
              "sha256": sha256(np.ascontiguousarray(value)).hexdigest()}
        for key, value in checkpoint.arrays.items()
    }


def _rewrite_archive(source, target, transform):
    with ZipFile(source) as original, ZipFile(target, "w") as output:
        for item in original.infolist():
            output.writestr(item.filename, transform(item.filename, original.read(item.filename)))


def test_digest_and_repeated_writes_hash_each_actual_array_once(checkpoint, monkeypatch, tmp_path):
    calls = _count_content_hashes(monkeypatch)
    expected_payload = json_digest(_legacy_identities(checkpoint))
    expected_digest = json_digest({
        "snapshot": checkpoint.snapshot.digest, "plane": checkpoint.plane_id,
        "signature": checkpoint.stage_signature, "payload": expected_payload,
        "metadata": checkpoint.metadata, "parent": checkpoint.parent_id,
    })
    assert checkpoint.payload_hash == expected_payload
    assert checkpoint.digest == expected_digest
    for name in ("first.temwp", "second.temwp"):
        path = tmp_path / name
        checkpoint.write_package(path, compression=ZIP_STORED)
        with ZipFile(path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
        assert manifest["digest"] == expected_digest
        assert {
            key: {name: row[name] for name in ("shape", "dtype", "sha256")}
            for key, row in manifest["arrays"].items()
        } == _legacy_identities(checkpoint)
    assert len(calls) == len(checkpoint.arrays)


def test_load_checks_actual_content_once_and_rewrite_reuses_verified_table(checkpoint, monkeypatch, tmp_path):
    path = tmp_path / "original.temwp"
    checkpoint.write_package(path)
    calls = _count_content_hashes(monkeypatch)
    loaded = module.WorkingPointCheckpoint.read_package(path)
    assert loaded.digest == checkpoint.digest
    for key, original in checkpoint.arrays.items():
        assert loaded.arrays[key].shape == original.shape
        assert loaded.arrays[key].dtype == original.dtype
        np.testing.assert_array_equal(loaded.arrays[key], original)
    loaded.write_package(tmp_path / "rewritten.temwp")
    assert len(calls) == len(checkpoint.arrays)


def test_identity_table_and_numeric_payload_are_detached_and_immutable(checkpoint):
    source = np.arange(12, dtype=np.float64).reshape(3, 4)
    point = module.WorkingPointCheckpoint(
        checkpoint.snapshot, {"history": source}, checkpoint.plane_z_mm,
        checkpoint.stage_signature, {},
    )
    before = point.digest
    expected = source.copy()
    source[:] = -1
    np.testing.assert_array_equal(point.arrays["history"], expected)
    with pytest.raises(ValueError):
        point.arrays["history"][0, 0] = 99
    with pytest.raises(ValueError):
        point.arrays["history"].setflags(write=True)
    with pytest.raises(TypeError):
        point._array_identities["history"]["sha256"] = "0" * 64
    with pytest.raises(TypeError):
        point._array_identities["history"]["shape"][0] = 99
    assert point.digest == before


def test_changed_numeric_bytes_are_rejected_even_with_valid_zip_crc(checkpoint, tmp_path):
    path, corrupt = tmp_path / "original.temwp", tmp_path / "corrupt.temwp"
    checkpoint.write_package(path)

    def corrupt_numeric(name, payload):
        if name == "arrays/0.npy":
            return payload[:-1] + bytes([payload[-1] ^ 1])
        return payload

    _rewrite_archive(path, corrupt, corrupt_numeric)
    with pytest.raises(ValueError, match="numeric content checksum"):
        module.WorkingPointCheckpoint.read_package(corrupt)


def test_consistent_forged_manifest_does_not_supply_verified_array_identities(checkpoint, tmp_path):
    path, corrupt = tmp_path / "original.temwp", tmp_path / "forged.temwp"
    checkpoint.write_package(path)

    def corrupt_manifest(name, payload):
        if name != "manifest.json":
            return payload
        data = json.loads(payload)
        next(iter(data["arrays"].values()))["sha256"] = "0" * 64
        identities = {
            key: {name: row[name] for name in ("shape", "dtype", "sha256")}
            for key, row in data["arrays"].items()
        }
        data["digest"] = json_digest({
            "snapshot": checkpoint.snapshot.digest, "plane": checkpoint.plane_id,
            "signature": data["stage_signature"], "payload": json_digest(identities),
            "metadata": data["metadata"], "parent": data["parent_id"],
        })
        return json.dumps(data).encode()

    _rewrite_archive(path, corrupt, corrupt_manifest)
    with pytest.raises(ValueError, match="numeric content checksum"):
        module.WorkingPointCheckpoint.read_package(corrupt)


def test_failed_write_keeps_original_archive_and_removes_temporary_file(checkpoint, monkeypatch, tmp_path):
    path = tmp_path / "original.temwp"
    checkpoint.write_package(path)
    original = path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("simulated storage failure")

    monkeypatch.setattr(module.np, "save", fail)
    with pytest.raises(OSError, match="storage failure"):
        checkpoint.write_package(path, overwrite=True)
    assert path.read_bytes() == original
    assert not tuple(tmp_path.glob(".working-point-*.tmp"))
    assert checkpoint.payload_hash == json_digest(thaw_json(checkpoint._array_identities))
