"""Atomic executed-wave publication and Windows file-sharing regressions.

Small independent complex arrays exercise storage only, without propagation.
"""
from contextlib import contextmanager
from dataclasses import asdict, replace
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import pytest

from temsim.immutable_json import json_digest
from temsim.optics.electron_gun.tip_coherence import TIP_REFERENCE
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.tip_gun_wave import TipGunCheckpoint
from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
from temsim.physics.wave_flux import BeamState, WaveMode
from temsim.physics.wave_reference import AxialWaveReference


def _checkpoint(phase=0.):
    amplitude = np.array(((1.+2j, -3.+1j), (2.-1j, 4.+3j)))
    amplitude = amplitude / np.linalg.norm(amplitude) * np.exp(1j*phase)
    plane = PlaneWave(amplitude, np.array(((2e-9, 3e-10), (-1e-10, 3e-9))),
        np.array((4e-9, -5e-9)), np.array(((7., 2.), (2., 11.))),
        np.array((1e-4, -2e-4)))
    mode = WaveMode(plane, .65, TIP_REFERENCE, "publication:0", 300.,
        AxialWaveReference(2e-8, 3e-22), ("independent-storage-fixture",))
    return TipGunCheckpoint(BeamState((mode,), TIP_REFERENCE), 1600., 2e-9,
        {"fixture": "exact complex storage without numerical propagation"})


@contextmanager
def _held_child_file_without_delete_sharing(path):
    """Model an external reader/indexer with a real Windows sharing handle."""
    if os.name != "nt":
        with path.open("rb"):
            yield
        return
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
        wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    create.restype = wintypes.HANDLE
    close = kernel.CloseHandle
    close.argtypes = (wintypes.HANDLE,)
    close.restype = wintypes.BOOL
    # GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, OPEN_EXISTING.
    handle = create(str(path), 0x80000000, 3, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        yield
    finally:
        if not close(handle):
            raise ctypes.WinError(ctypes.get_last_error())


def _assert_same_mode(actual, expected):
    for field in ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad"):
        np.testing.assert_array_equal(getattr(actual.plane, field), getattr(expected.plane, field))
        assert not getattr(actual.plane, field).flags.writeable
    assert actual.weight_per_reference_electron == expected.weight_per_reference_electron
    assert actual.energy_kev == expected.energy_kev
    assert actual.mode_id == expected.mode_id
    assert actual.axial_reference == expected.axial_reference
    assert actual.scattering_history == expected.scattering_history


def test_commit_with_windows_child_handle_without_delete_sharing(tmp_path):
    store = ExecutedWaveStore(tmp_path, "windows-share-fixture", 1<<20)
    expected = _checkpoint()
    key = store.key("executed-state")
    writer = store.writer(key, TIP_REFERENCE)
    try:
        writer.append(expected.beam.modes[0])
        child = writer.directory / writer.rows[0]["arrays"]["amplitude"]["file"]
        with _held_child_file_without_delete_sharing(child):
            actual = writer.finish(expected.plane_z_mm, expected.reference_current_a, expected.record)
            # Finally-style cleanup must preserve the accepted immutable files,
            # even while the external reader continues to hold its handle.
            writer.abort()
            _assert_same_mode(actual.beam.modes[0], expected.beam.modes[0])
        _assert_same_mode(store.get(key).beam.modes[0], expected.beam.modes[0])
    finally:
        writer.abort()


def test_unindexed_complete_arrays_and_manifest_are_not_a_checkpoint(tmp_path):
    store = ExecutedWaveStore(tmp_path, "visibility-fixture", 1<<20)
    key = store.key("private-write")
    writer = store.writer(key, TIP_REFERENCE)
    try:
        writer.append(_checkpoint().beam.modes[0])
        writer.append_auxiliary("mandatory_complex", np.array((1.+2j, 3.-4j)))
        assert store.get(key) is None
        assert store.auxiliary_arrays(key) is None
        restarted = ExecutedWaveStore(tmp_path, "visibility-fixture", 1<<20)
        assert restarted.get(key) is None
    finally:
        writer.abort()
    assert not list(tmp_path.iterdir())


def test_atomic_index_failure_keeps_previous_result_and_cleans_only_own_files(tmp_path, monkeypatch):
    import temsim.physics.wave_checkpoint_store as module
    store = ExecutedWaveStore(tmp_path, "index-failure-fixture", 1<<20)
    key = store.key("same-physical-dependency")
    expected = _checkpoint()
    prior = store.put(key, expected)
    prior_files = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    prior_index = (tmp_path / (key+".json")).read_bytes()
    writer = store.writer(key, TIP_REFERENCE)
    writer.append(_checkpoint(.6).beam.modes[0])
    def denied_index(source, destination):
        assert Path(destination) == tmp_path / (key+".json")
        raise PermissionError("independent atomic index publication failure")
    monkeypatch.setattr(module.os, "replace", denied_index)
    try:
        with pytest.raises(PermissionError, match="atomic index publication"):
            writer.finish(expected.plane_z_mm, expected.reference_current_a, expected.record)
    finally:
        writer.abort()
    assert (tmp_path / (key+".json")).read_bytes() == prior_index
    assert store.get(key).digest == prior.digest
    _assert_same_mode(store.get(key).beam.modes[0], expected.beam.modes[0])
    assert {path.relative_to(tmp_path) for path in tmp_path.rglob("*")} == prior_files
    assert store._used_bytes == sum(path.stat().st_size for path in tmp_path.rglob("*") if path.is_file())


def test_finished_writer_abort_is_safe_and_further_writes_are_rejected(tmp_path):
    store = ExecutedWaveStore(tmp_path, "writer-state-fixture", 1<<20)
    expected = _checkpoint()
    key = store.key("complete")
    writer = store.writer(key, TIP_REFERENCE)
    writer.append(expected.beam.modes[0])
    writer.finish(expected.plane_z_mm, expected.reference_current_a, expected.record)
    writer.abort()
    writer.abort()
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    for action in (
        lambda: writer.append(replace(expected.beam.modes[0], mode_id="another")),
        lambda: writer.append_auxiliary("another", np.ones(2)),
        lambda: writer.finish(expected.plane_z_mm, expected.reference_current_a, expected.record),
    ):
        with pytest.raises((ValueError, RuntimeError), match="published|closed|finished|completed|aborted"):
            action()
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
    _assert_same_mode(store.get(key).beam.modes[0], expected.beam.modes[0])


@pytest.mark.parametrize("key", ("../escape", "a"*63, "z"*64, "", None, 42))
def test_invalid_writer_key_does_not_create_private_directory(tmp_path, key):
    store = ExecutedWaveStore(tmp_path, "key-fixture", 1<<20)
    with pytest.raises(ValueError, match="key|digest"):
        store.writer(key, TIP_REFERENCE)
    assert not list(tmp_path.iterdir())


def test_abort_rejects_tampered_cleanup_path_without_deleting_user_file(tmp_path):
    store = ExecutedWaveStore(tmp_path / "cache", "cleanup-fixture", 1<<20)
    writer = store.writer(store.key("private"), TIP_REFERENCE)
    owned = writer.directory
    outside = tmp_path / "user-data"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_text("user data", encoding="utf-8")
    writer.directory = outside
    try:
        with pytest.raises(ValueError, match="cleanup|private|owned|path"):
            writer.abort()
        assert sentinel.read_text(encoding="utf-8") == "user data"
    finally:
        writer.directory = owned
        writer.abort()


def test_abort_rejects_symlink_cleanup_target_before_recursive_delete(tmp_path, monkeypatch):
    store = ExecutedWaveStore(tmp_path, "symlink-policy-fixture", 1<<20)
    writer = store.writer(store.key("private"), TIP_REFERENCE)
    original = Path.is_symlink
    def symlink_for_owned(path):
        return path == writer.directory or original(path)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "is_symlink", symlink_for_owned)
        with pytest.raises(ValueError, match="cleanup|symlink|private|owned|path"):
            writer.abort()
        assert writer.directory.exists()
    writer.abort()


def test_historical_disk_v1_manifest_is_readable_without_rewriting(tmp_path):
    dependency = "historical-codec-fixture"
    store = ExecutedWaveStore(tmp_path, dependency, 1<<20)
    key = store.key("preexisting-executed-checkpoint")
    directory = tmp_path / (key+"-"+"0"*32)
    directory.mkdir()
    expected = _checkpoint()
    mode = expected.beam.modes[0]
    arrays = {}
    for field in ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad"):
        value = getattr(mode.plane, field)
        path = directory / ("m0_"+field+".npy")
        np.save(path, value, allow_pickle=False)
        arrays[field] = {"file": path.name, "sha256": sha256(path.read_bytes()).hexdigest(),
            "shape": value.shape, "nbytes": value.nbytes}
    data = {"schema": "executed-tip-wave-disk-v1", "dependency": dependency, "key": key,
        "reference_plane": TIP_REFERENCE, "plane_z_mm": expected.plane_z_mm,
        "current_a": expected.reference_current_a, "modes": [{"mode_id": mode.mode_id,
            "weight": mode.weight_per_reference_electron, "energy_kev": mode.energy_kev,
            "axial_reference": asdict(mode.axial_reference), "scattering_history": list(mode.scattering_history),
            "arrays": arrays}], "record": dict(expected.record)}
    digest = json_digest(data)
    data["manifest_digest"] = digest
    (directory / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
    (tmp_path / (key+".json")).write_text(json.dumps({"directory": directory.name, "digest": digest}), encoding="utf-8")
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    restored = ExecutedWaveStore(tmp_path, dependency, 1<<20).get(key)
    _assert_same_mode(restored.beam.modes[0], mode)
    assert restored.plane_z_mm == expected.plane_z_mm
    assert restored.reference_current_a == expected.reference_current_a
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
