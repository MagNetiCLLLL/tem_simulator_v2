"""Lazy archives retain exact identities without accepting unverified products."""
from dataclasses import replace
from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np
import pytest

from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state
from temsim.working_point import WorkingPointCheckpoint, WorkingPointArchiveIndex, migrate_working_point_inputs


@pytest.fixture
def archived(tmp_path):
    state = default_state()
    arrays = {k: np.zeros(20) for k in ("x_m", "y_m", "tx_rad", "ty_rad")}
    arrays.update(weight=np.ones(20), alive=np.ones(20, dtype=bool))
    point = WorkingPointCheckpoint(capture_instrument_snapshot(state), arrays, state.sample.z_mm,
        "local-fixture", {"source_current_a": 1e-9})
    path = tmp_path / "fixture.temwp"
    point.write_package(path)
    return point, path


def test_index_and_gui_browse_do_not_load_arrays_or_restore(archived, monkeypatch, qtbot):
    from temsim.gui.working_point_panel import WorkingPointPanel
    point, path = archived
    def forbidden(*args, **kwargs):
        pytest.fail("Lazy browsing must not load numeric products or restore state")
    monkeypatch.setattr(np, "load", forbidden)
    monkeypatch.setattr(type(point.snapshot), "restore", forbidden)
    record = WorkingPointArchiveIndex.read(path)
    assert record.digest == point.digest
    assert record.has_retained_payload and not record.is_input_design
    panel = WorkingPointPanel()
    qtbot.addWidget(panel)
    panel.add_checkpoint(record, label="Indexed")
    panel._pin("A")
    panel._pin("B")
    panel._compare_pins()
    panel.filter.setText(point.digest)
    assert panel._summaries[point.digest]["status"] == "INDEX_ONLY"
    assert record.digest == point.digest
    panel.shutdown()


def test_multiple_axis_receipts_survive_package_and_explicit_verification(archived, qtbot):
    from temsim.gui.working_point_panel import WorkingPointPanel
    from temsim.immutable_json import freeze_json, json_digest
    point, path = archived
    reports = []
    for axis in ("gun_step", "column_step"):
        report = dict(schema="incident-convergence-evidence-v1", checkpoint_id=point.digest,
            snapshot_id=point.snapshot.digest, implementation=point.snapshot.implementation,
            axis=axis, comparison={"status": "UNRESOLVED"}, scope="Archive association fixture; not physical evidence")
        report["digest"] = json_digest(report)
        reports.append(freeze_json(report))
    panel = WorkingPointPanel()
    qtbot.addWidget(panel)
    panel.add_checkpoint(point)
    for report in reports:
        panel._accept_evidence(report)
    assert len(panel._imported_evidence[point.digest]) == 2
    point.write_package(path, overwrite=True, evidence=panel._imported_evidence[point.digest])
    indexed = WorkingPointArchiveIndex.read(path)
    assert len(indexed.evidence) == 2
    loaded = indexed.load()
    for report in indexed.evidence:
        panel._accept_imported_evidence(loaded, report)
    assert set(panel.sampling._history[point.digest]) == {"gun_step", "column_step"}
    panel.shutdown()


def test_lazy_load_verifies_all_payloads_and_detects_file_replacement(archived):
    point, path = archived
    index = WorkingPointArchiveIndex.read(path)
    loaded = index.load()
    assert loaded.digest == point.digest
    assert np.array_equal(loaded.arrays["weight"], point.arrays["weight"])
    replace(point, stage_signature="replaced").write_package(path, overwrite=True)
    with pytest.raises(ValueError, match="changed"):
        index.load()


def test_explicit_migration_is_new_inputs_only_identity_and_preserves_history(archived):
    point, _ = archived
    old = replace(point, snapshot=replace(point.snapshot, implementation="old-solver"))
    migrated = migrate_working_point_inputs(old)
    assert migrated.parent_id == old.digest
    assert migrated.is_input_design and not migrated.arrays
    assert migrated.digest != old.digest and old.arrays
    assert migrated.metadata["validation_status"] == "NOT_RUN"
    assert migrated.snapshot.graph == point.snapshot.graph
    assert migrated.snapshot.implementation == point.snapshot.implementation
    assert old.snapshot.implementation == "old-solver"


def test_migration_cannot_admit_historical_exit_source(archived):
    point, _ = archived
    state = point.snapshot.restore()
    state.electron_gun.source_representation = "effective_gaussian_schell"
    old = replace(point, snapshot=capture_instrument_snapshot(state))
    with pytest.raises(ValueError, match="Custom exit sources"):
        migrate_working_point_inputs(old)


@pytest.mark.parametrize("attack", ["duplicate", "traversal", "huge_shape", "undeclared"])
def test_package_inventory_and_npy_bounds_are_checked_before_allocation(archived, tmp_path, attack):
    _, path = archived
    corrupted = tmp_path / "bad.temwp"
    with ZipFile(path) as original, ZipFile(corrupted, "w") as target:
        document = json.loads(original.read("manifest.json"))
        for item in original.infolist():
            if item.filename == "manifest.json":
                continue
            content = original.read(item.filename)
            if attack == "huge_shape" and item.filename == "arrays/0.npy":
                buffer = BytesIO()
                np.lib.format.write_array_header_1_0(buffer,
                    dict(descr="|b1", fortran_order=False, shape=(2**60,)))
                content = buffer.getvalue()
            target.writestr(item.filename, content)
        if attack == "duplicate":
            target.writestr("arrays/0.npy", b"duplicate")
        if attack in {"traversal", "undeclared"}:
            target.writestr("../outside" if attack == "traversal" else "extra.dat", b"payload")
        target.writestr("manifest.json", json.dumps(document))
    with pytest.raises(ValueError):
        WorkingPointCheckpoint.read_package(corrupted)
