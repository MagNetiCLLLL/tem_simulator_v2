"""Export scope cannot manufacture physical inputs or qualify omitted results."""
from dataclasses import replace
import json
from zipfile import ZipFile

import numpy as np
import pytest

from temsim.immutable_json import thaw_json
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state
from temsim.working_point import WorkingPointCheckpoint, WorkingPointArchiveIndex, migrate_working_point_inputs
from temsim.working_point_export import export_checkpoint


@pytest.fixture
def point():
    state = default_state()
    # A serialization asset, not a calibrated or active physical field.
    state.export_fixture = np.arange(32768, dtype=np.float64).reshape(128, 256)
    arrays = {key: np.zeros(32) for key in ("x_m", "y_m", "tx_rad", "ty_rad")}
    arrays.update(weight=np.ones(32), alive=np.ones(32, dtype=bool))
    return WorkingPointCheckpoint(capture_instrument_snapshot(state), arrays, state.sample.z_mm,
        "export-fixture", {"source_current_a": 2e-9, "validation_status": "HISTORICAL_LABEL"})


def test_complete_inputs_and_results_preserve_identity_and_all_exact_buffers(point, tmp_path):
    path = tmp_path / "full.temwp"
    point.write_package(path, mode="inputs_and_results")
    loaded = WorkingPointCheckpoint.read_package(path)
    assert loaded.digest == point.digest
    assert loaded.snapshot.to_dict() == point.snapshot.to_dict()
    for key, value in point.arrays.items():
        assert loaded.arrays[key].dtype == value.dtype
        assert loaded.arrays[key].tobytes() == value.tobytes()


def test_inputs_export_needs_no_result_load_and_keeps_exact_input_graph(point, tmp_path, monkeypatch):
    full = tmp_path / "full.temwp"
    point.write_package(full)
    index = WorkingPointArchiveIndex.read(full)
    def forbidden(*args, **kwargs):
        pytest.fail("Input export must not restore or read retained result arrays")
    monkeypatch.setattr(np, "load", forbidden)
    monkeypatch.setattr(type(point.snapshot), "restore", forbidden)
    target = tmp_path / "inputs.temwp"
    index.write_package(target, mode="inputs")
    loaded = WorkingPointCheckpoint.read_package(target)
    assert loaded.is_input_design and not loaded.arrays
    assert loaded.parent_id == point.digest and loaded.digest != point.digest
    assert loaded.snapshot.to_dict() == point.snapshot.to_dict()
    assert loaded.metadata["validation_status"] == "NOT_RUN"
    assert loaded.metadata["offline_reproducible"] is False
    assert loaded.metadata["restoration_policy"] == "VERIFY_ORIGINAL_DEPENDENCIES"


def test_metadata_omits_bulk_inputs_and_cannot_restore_apply_or_migrate(point, tmp_path, qtbot):
    from temsim.gui.working_point_panel import WorkingPointPanel
    from temsim.sampling_diagnostics import checkpoint_sampling_summary
    target = tmp_path / "metadata.temwp"
    point.write_package(target, mode="metadata")
    with ZipFile(target) as archive:
        assert archive.namelist() == ["manifest.json"]
        document = json.loads(archive.read("manifest.json"))
    assert '"array":' not in json.dumps(document)
    assert all(row["content_hex"] is None for row in document["snapshot"]["external_inputs"])
    assert len(json.dumps(document)) < len(json.dumps(point.snapshot.to_dict()))
    original_array = point.snapshot.graph["nodes"][0]["attributes"]["export_fixture"]
    assert original_array["array"] not in json.dumps(document)
    assert document["snapshot"]["graph"]["nodes"][0]["attributes"]["export_fixture"]["omitted_array"]["shape"] == [128, 256]
    index = WorkingPointArchiveIndex.read(target)
    assert index.is_metadata_only and not index.has_retained_payload and not index.is_input_design
    loaded = index.load()
    assert loaded.parent_id == point.digest
    assert checkpoint_sampling_summary(loaded)["status"] == "METADATA_ONLY"
    for operation in (loaded.compatible_state, loaded.snapshot.restore, lambda: migrate_working_point_inputs(loaded)):
        with pytest.raises(ValueError):
            operation()
    with pytest.raises(ValueError, match="complete input"):
        export_checkpoint(loaded, "inputs")
    panel = WorkingPointPanel()
    qtbot.addWidget(panel)
    restored, applied = [], []
    panel.restore_requested.connect(lambda *args: restored.append(args))
    panel.illumination_requested.connect(applied.append)
    panel.add_checkpoint(index)
    assert "Metadata summary" in panel.status.text()
    assert not panel.sampling.run_button.isEnabled()
    panel._restore(False)
    panel._restore(True)
    panel._apply_illumination()
    panel._migrate()
    panel._load_selected()
    assert not restored and not applied
    assert len(panel._points) == 1 and not panel._archive_loading
    assert panel.shutdown()


def test_missing_dependency_cannot_be_labelled_complete(point, tmp_path):
    rows = thaw_json(point.snapshot.external_inputs)
    rows[0]["content_hex"] = None
    incomplete = replace(point, snapshot=replace(point.snapshot, external_inputs=tuple(rows)))
    for mode in ("inputs", "inputs_and_results"):
        with pytest.raises(ValueError, match="missing"):
            incomplete.write_package(tmp_path / "missing.temwp", mode=mode)
    incomplete.write_package(tmp_path / "readable.temwp", mode="metadata")
    assert WorkingPointArchiveIndex.read(tmp_path / "readable.temwp").is_metadata_only


def test_failed_export_preserves_existing_file_and_removes_only_owned_temporary(point, tmp_path, monkeypatch):
    target = tmp_path / "existing.temwp"
    point.write_package(target)
    original = target.read_bytes()
    def fail(*args, **kwargs):
        raise OSError("Injected archive writer failure")
    monkeypatch.setattr(ZipFile, "writestr", fail)
    with pytest.raises(OSError, match="Injected"):
        point.write_package(target, mode="inputs", overwrite=True)
    assert target.read_bytes() == original
    assert set(tmp_path.iterdir()) == {target}


def test_export_dialog_routes_metadata_without_loading_or_changing_live_state(point, tmp_path, qtbot, monkeypatch):
    from temsim.gui.working_point_panel import WorkingPointPanel, QInputDialog, QFileDialog
    full, target = tmp_path / "full.temwp", tmp_path / "summary.temwp"
    point.write_package(full)
    panel = WorkingPointPanel()
    qtbot.addWidget(panel)
    panel.add_checkpoint(WorkingPointArchiveIndex.read(full))
    monkeypatch.setattr(QInputDialog, "getItem", lambda *a, **k: ("Metadata only; cannot restore", True))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), ""))
    monkeypatch.setattr(WorkingPointArchiveIndex, "load", lambda *a: pytest.fail("Metadata export loaded results"))
    errors = []
    panel.error.connect(errors.append)
    panel._export()
    assert not errors
    assert WorkingPointArchiveIndex.read(target).is_metadata_only
    assert panel.selected.digest == point.digest
    assert panel.shutdown()


def test_export_keeps_record_selected_before_dialog_even_if_new_result_arrives(point, tmp_path, qtbot, monkeypatch):
    from temsim.gui.working_point_panel import WorkingPointPanel, QInputDialog, QFileDialog
    panel = WorkingPointPanel()
    qtbot.addWidget(panel)
    panel.add_checkpoint(point)
    newer = replace(point, stage_signature="later-result")
    def choose(*args, **kwargs):
        panel.add_checkpoint(newer)
        return "Complete inputs and retained results", True
    target = tmp_path / "selected.temwp"
    monkeypatch.setattr(QInputDialog, "getItem", choose)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), ""))
    panel._export()
    assert panel.selected.digest == newer.digest
    assert WorkingPointArchiveIndex.read(target).digest == point.digest
    assert panel.shutdown()
