"""Archive size admission and GUI routing; no electron transport is executed."""
from threading import Event
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QPushButton

from temsim.gui.calculation_controller import CalculationController
from temsim.gui.working_point_loader import ArchiveLoader
from temsim.gui.working_point_panel import WorkingPointPanel
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state
from temsim.working_point import WorkingPointCheckpoint, WorkingPointArchiveIndex

GIB = 1024**3


@pytest.fixture
def package(tmp_path):
    state = default_state()
    point = WorkingPointCheckpoint(capture_instrument_snapshot(state),
        {"fixture": np.arange(8.)}, state.sample.z_mm, "archive-routing-fixture", {})
    path = tmp_path / "small.temwp"
    point.write_package(path)
    return point, path


def test_loader_reserves_validated_size_instead_of_user_limit(package):
    _, path = package
    index = WorkingPointArchiveIndex.read(path, maximum_unpacked_bytes=8*GIB)
    worker = ArchiveLoader(index, Event(), maximum_unpacked_bytes=4*GIB)
    assert worker.maximum_unpacked_bytes == 4*GIB
    assert worker.record.maximum_unpacked_bytes == index.unpacked_size_bytes
    assert worker.resource_claim.working_bytes == max(512*1024**2, 2*index.unpacked_size_bytes)
    assert worker.resource_claim.working_bytes < 4*GIB
    with pytest.raises(ValueError, match="memory budget"):
        ArchiveLoader(index, Event(), maximum_unpacked_bytes=index.unpacked_size_bytes-1)


def test_working_point_import_export_and_load_use_captured_shared_budget(qtbot, monkeypatch, package, tmp_path):
    from temsim.gui import working_point_panel as module
    point, path = package
    panel = WorkingPointPanel()
    qtbot.addWidget(panel)
    budget = 3*GIB
    monkeypatch.setattr(panel._archive_pool.coordinator, "ram_budget_bytes", budget)
    monkeypatch.setattr(module.QFileDialog, "getOpenFileName", lambda *args: (str(path), ""))
    panel._import()
    assert panel.selected.maximum_unpacked_bytes == budget
    workers = []
    monkeypatch.setattr(panel._archive_pool, "start", workers.append)
    panel._archive_cancel.set()  # A completed cancellation must not poison a new load.
    panel._load_selected()
    assert workers[0].maximum_unpacked_bytes == budget
    assert not workers[0].cancelled.is_set()
    panel._load_finished()
    panel.add_checkpoint(point)
    calls = []
    monkeypatch.setattr(WorkingPointCheckpoint, "write_package", lambda self, target, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(module.QInputDialog, "getItem", lambda *args: ("Complete inputs and retained results", True))
    monkeypatch.setattr(module.QFileDialog, "getSaveFileName", lambda *args: (str(tmp_path / "export.temwp"), ""))
    panel._export()
    assert calls[0]["maximum_unpacked_bytes"] == budget
    assert not any(button.text() == "Migrate inputs" for button in panel.findChildren(QPushButton))
    assert not hasattr(panel, "_migrate")


@pytest.mark.parametrize("operation", ["automatic", "manual", "load"])
def test_section_file_operations_capture_and_enforce_shared_budget(qtbot, monkeypatch, package, tmp_path, operation):
    from temsim import particle_section_io as codec
    _, path = package
    owner = CalculationController(persistent_cache_enabled=False, artifact_cache_root=tmp_path)
    budget = 3*GIB
    monkeypatch.setattr(owner.section_file_pool.coordinator, "ram_budget_bytes", budget)
    workers, calls, events = [], [], []
    monkeypatch.setattr(owner.section_file_pool, "start", workers.append)
    owner.section_archive_changed.connect(events.append)
    monkeypatch.setattr(codec, "section_archive_summary", lambda result: {"identity": "routing-fixture"})
    # This fixture tests worker admission/routing; real result capture has its own regressions.
    monkeypatch.setattr(codec, "capture_result_for_save", lambda result: result)
    payload = SimpleNamespace(state_snapshot=None)
    def receipt(target):
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"File worker routing fixture, not a particle calculation")
        return {"identity": "routing-fixture", "path": str(target),
                "_file_fingerprint": codec.section_file_fingerprint(target), "_package_digest": "routing-fixture"}
    def auto(result, directory, **kwargs):
        calls.append(kwargs)
        return receipt(directory / "automatic.temsection")
    def manual(result, target, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(digest="routing-fixture")
    def load(target, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(section_archive_info={"identity": "routing-fixture", "path": str(target),
            "_file_fingerprint": codec.section_file_fingerprint(target), "_package_digest": "routing-fixture"})
    monkeypatch.setattr(codec, "archive_section_result", auto)
    monkeypatch.setattr(codec, "save_section_result", manual)
    monkeypatch.setattr(codec, "checked_section_archive_info", lambda result, target, **kwargs: receipt(target))
    monkeypatch.setattr(codec, "load_section_result", load)
    monkeypatch.setattr(owner, "retain_section_seed", lambda result: None)
    if operation == "load":
        owner.load_section_archive(path)
    else:
        owner.archive_completed_section(payload, path=None if operation == "automatic" else tmp_path / "manual.temsection")
    worker = workers[0]
    assert worker.maximum_unpacked_bytes == budget
    monkeypatch.setattr(owner.section_file_pool.coordinator, "ram_budget_bytes", budget//2)
    worker.run()
    expected = WorkingPointArchiveIndex.read(path).unpacked_size_bytes if operation == "load" else budget
    assert calls[0]["maximum_unpacked_bytes"] == expected
    assert not any(row.get("status") == "failed" for row in events)
    if operation == "load":
        assert worker.resource_claim.working_bytes < budget
