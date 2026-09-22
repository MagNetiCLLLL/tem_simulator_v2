"""Saved-state identity regressions using executed tip-origin particle sections."""
from pathlib import Path

import numpy as np
import pytest

from test_particle_section_io import executed_section
from temsim.gui.calculation_controller import CalculationController
from temsim.particle_section_io import (
    archive_section_result, load_section_result, save_section_result, section_archive_identity,
)


@pytest.fixture(scope="module")
def continued_section(executed_section):
    from temsim.instrument_snapshot import decode_instrument, encode_instrument
    from temsim.simulation_pipeline import calculate_particle_section
    return calculate_particle_section(
        decode_instrument(encode_instrument(executed_section.state_snapshot)),
        target_z_mm=executed_section.simulation.metrics["section_target_z_mm"] + 1.,
        component_keys=("objective_lens",), existing_result=executed_section)


@pytest.fixture
def dispatcher(qtbot, monkeypatch, tmp_path):
    owner = CalculationController(persistent_cache_enabled=False, artifact_cache_root=tmp_path)
    workers, events = [], []
    monkeypatch.setattr(owner.section_file_pool, "start", workers.append)
    owner.section_archive_changed.connect(events.append)
    return owner, workers, events


def _save(dispatcher, result, path):
    owner, workers, events = dispatcher
    owner.archive_completed_section(result, path=path)
    workers[-1].run()
    assert events[-1]["status"] == "saved", events[-1]


def test_manual_overwrite_cannot_be_reused_as_previous_result(
        dispatcher, executed_section, continued_section, tmp_path):
    owner, workers, events = dispatcher
    path = tmp_path / "same-path.temsection"
    _save(dispatcher, executed_section, path)
    _save(dispatcher, continued_section, path)
    assert section_archive_identity(load_section_result(path)) == section_archive_identity(continued_section)
    workers.clear()
    owner.archive_completed_section(executed_section)
    assert len(workers) == 1, "Previous identity must queue a new archive after its path was overwritten"
    assert events[-1]["status"] == "saving"
    workers[0].run()
    assert events[-1]["identity"] == section_archive_identity(executed_section)
    assert Path(events[-1]["path"]) != path


@pytest.mark.parametrize("alias", [False, True])
def test_pending_replacement_prevents_cached_reuse(
        dispatcher, executed_section, continued_section, tmp_path, alias):
    import os
    owner, workers, events = dispatcher
    path = tmp_path / "pending.temsection"
    _save(dispatcher, executed_section, path)
    destination = path
    if alias:
        destination = tmp_path / "hard-link.temsection"
        os.link(path, destination)
    owner.archive_completed_section(continued_section, path=destination)
    # Even a load that restores the original association during a pending write
    # cannot make that association safe for another immediate reuse.
    owner.retain_section_seed(load_section_result(path))
    queued = len(workers)
    owner.archive_completed_section(executed_section)
    assert len(workers) == queued + 1
    assert events[-1]["status"] == "saving"


@pytest.mark.parametrize("operation", ["save", "load"])
def test_delayed_completion_cannot_report_replaced_file_as_saved(
        dispatcher, executed_section, continued_section, tmp_path, operation):
    from temsim.particle_section_io import checked_section_archive_info
    from types import SimpleNamespace
    owner, _, events = dispatcher
    path = tmp_path / "delayed.temsection"
    package = save_section_result(executed_section, path)
    info = checked_section_archive_info(executed_section, path,
        maximum_unpacked_bytes=8 * 1024**3, expected_package_digest=package.digest)
    save_section_result(continued_section, path, overwrite=True)
    owner._section_file_jobs["delayed"] = SimpleNamespace(operation=operation)
    output = info if operation == "save" else {"result": executed_section, "info": info}
    owner._section_file_completed("delayed", output)
    assert events[-1]["status"] == "failed"
    assert "changed" in events[-1]["error"]
    assert not owner._section_archive_records


def test_unknown_existing_archive_is_verified_without_loading_arrays(
        executed_section, tmp_path, monkeypatch):
    from temsim import particle_section_io as codec
    first = archive_section_result(executed_section, tmp_path)
    def no_decode(*args, **kwargs):
        pytest.fail("Archive identity verification must not reconstruct large particle arrays")
    monkeypatch.setattr(codec, "load_section_result", no_decode)
    monkeypatch.setattr(np, "load", no_decode)
    second = archive_section_result(executed_section, tmp_path)
    assert second["identity"] == first["identity"] and second["reused"]
    assert second["_file_fingerprint"] == first["_file_fingerprint"]


def test_existing_automatic_archive_payload_corruption_is_not_reused(executed_section, tmp_path):
    from zipfile import ZipFile, ZIP_STORED
    from temsim import particle_section_io as codec
    saved = archive_section_result(executed_section, tmp_path)
    path = Path(saved["path"])
    with ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    numeric = next(name for name in entries if name.endswith(".npy"))
    data = entries[numeric]
    entries[numeric] = data[:-1] + bytes([data[-1] ^ 1])
    # Rewriting preserves a valid ZIP CRC but intentionally breaks the recorded
    # numeric SHA-256. The bounded verifier must check the numeric checksum too.
    with ZipFile(path, "w", compression=ZIP_STORED) as archive:
        for name, value in entries.items():
            archive.writestr(name, value)
    with pytest.raises(ValueError, match="checksum"):
        codec.archive_section_result(executed_section, tmp_path)


@pytest.mark.parametrize("change", ["replace", "delete", "corrupt"])
def test_external_archive_changes_invalidate_cached_reuse(
        dispatcher, executed_section, continued_section, tmp_path, change):
    owner, workers, events = dispatcher
    path = tmp_path / "external.temsection"
    _save(dispatcher, executed_section, path)
    if change == "replace":
        save_section_result(continued_section, path, overwrite=True)
    elif change == "delete":
        path.unlink()
    else:
        path.write_bytes(b"damaged saved section")
    workers.clear()
    owner.archive_completed_section(executed_section)
    assert len(workers) == 1, "Existence alone cannot establish archive identity"
    assert events[-1]["status"] == "saving"
