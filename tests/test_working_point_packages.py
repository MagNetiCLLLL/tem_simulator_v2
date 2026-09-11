"""Software acceptance for frozen browsing, package integrity and restore."""
from dataclasses import replace
from types import SimpleNamespace
import json
import subprocess
import sys
from zipfile import ZipFile

import numpy as np
import pytest

from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state
from temsim.working_point import WorkingPointCheckpoint, component_rows, snapshot_changes


@pytest.fixture
def checkpoint():
    state = default_state()
    arrays = {k: np.array([0., .01, -.01]) for k in ("x_m", "y_m", "tx_rad", "ty_rad")}
    arrays.update(weight=np.array([.2, .4, .4]), alive=np.array([True, True, False]))
    return WorkingPointCheckpoint(capture_instrument_snapshot(state), arrays, state.sample.z_mm,
                                  "exact-local-signature", {"source_representation": "test-fixture"})


def test_t209_12_frozen_observables_and_component_tables(checkpoint):
    cp = checkpoint
    old = cp.digest
    a = cp.observables.get("alpha95")
    state = cp.compatible_state()
    state.lenses[0].percent += 1
    assert a == cp.observables.get("alpha95")
    assert a.status == "AVAILABLE" and a.checkpoint_id == cp.digest
    assert cp.observables.get("alpha99").value >= a.value
    assert cp.observables.get("physical_angular_edge").status == "UNAVAILABLE"
    assert cp.observables.get("invented").value is None
    assert cp.observables.get("source_fraction").value == pytest.approx(.6)
    with pytest.raises((AttributeError, TypeError)):
        a.value = 0
    with pytest.raises(ValueError):
        cp.arrays["weight"].setflags(write=True)
    assert cp.digest == old
    assert list(component_rows(cp.snapshot))
    assert snapshot_changes(cp.snapshot, capture_instrument_snapshot(state))
    missing = replace(cp, arrays={})
    assert missing.observables.get("alpha95").status == "NOT_COMPUTED"


def test_t207_package_cross_process_and_exact_restore(checkpoint, tmp_path):
    path = tmp_path / "point.temwp"
    checkpoint.write_package(path)
    loaded = WorkingPointCheckpoint.read_package(path)
    assert loaded.digest == checkpoint.digest
    assert capture_instrument_snapshot(loaded.compatible_state()).digest == checkpoint.snapshot.digest
    code = "from temsim.working_point import WorkingPointCheckpoint as C; import sys; c=C.read_package(sys.argv[1]); c.compatible_state(); print(c.digest)"
    result = subprocess.run([sys.executable, "-c", code, str(path)], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == checkpoint.digest
    with pytest.raises(FileExistsError):
        checkpoint.write_package(path)


def test_t226_package_corruption_and_memory_limit(checkpoint, tmp_path):
    path = tmp_path / "point.temwp"
    checkpoint.write_package(path)
    with pytest.raises(ValueError, match="budget"):
        WorkingPointCheckpoint.read_package(path, maximum_unpacked_bytes=1)
    corrupt = tmp_path / "corrupt.temwp"
    with ZipFile(path) as original, ZipFile(corrupt, "w") as target:
        for item in original.infolist():
            data = original.read(item.filename)
            if item.filename == "manifest.json":
                document = json.loads(data)
                document["plane_z_mm"] += 1
                data = json.dumps(document).encode()
            target.writestr(item.filename, data)
    with pytest.raises(ValueError, match="checksum"):
        WorkingPointCheckpoint.read_package(corrupt)


def test_t234_incompatible_dependency_keeps_historical_reading(checkpoint, tmp_path):
    snapshot = replace(checkpoint.snapshot, external_inputs=({"role": "field map", "path": str(tmp_path / "missing"),
                                                           "sha256": "", "content_hex": None},))
    cp = replace(checkpoint, snapshot=snapshot)
    assert cp.observables.get("alpha95").status == "AVAILABLE"
    with pytest.raises(ValueError, match="Missing field map"):
        cp.compatible_state()


def test_working_point_from_actual_gun_execution():
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.physics.simulation import run
    from temsim.calculation_manifest import capture_calculation_manifest
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, catalog.default_selection())
    state.electron_gun.emitter.ray_count = 9
    state.step_mm = state.history_step_mm = 5.
    manifest = capture_calculation_manifest(state)
    simulation = run(state)
    cp = WorkingPointCheckpoint.from_result(SimpleNamespace(simulation=simulation,
        calculation_manifest=manifest, signatures=manifest.calculation_signatures))
    assert cp.plane_z_mm == state.sample.z_mm
    assert cp.arrays["gun_ray_id"].size == 9
    assert cp.observables.get("alpha95").status == "AVAILABLE"
    assert cp.metadata["phase_status"] == "NOT_COMPUTED"
