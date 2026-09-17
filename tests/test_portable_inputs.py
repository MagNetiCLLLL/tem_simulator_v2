"""Archived bytes must drive the same real source chain with originals absent."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import shutil
import sys
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from temsim import input_io
from temsim.immutable_json import thaw_json, json_digest
from temsim.instrument_snapshot import capture_instrument_snapshot, encode_instrument, decode_instrument
from temsim.optics.column import default_state
from temsim.working_point import WorkingPointCheckpoint, WorkingPointArchiveIndex
from temsim.working_point_export import make_portable_inputs


@pytest.fixture
def portable_fixture(tmp_path, monkeypatch):
    from temsim.paths import CONFIG_ROOT
    original_root = Path(CONFIG_ROOT)
    state = default_state()
    state.electron_gun.emitter.ray_count = 9
    state.step_mm = 1.
    graph = thaw_json(encode_instrument(state))
    copied_root = tmp_path / "copied-inputs" / "configs"
    shutil.copytree(original_root, copied_root)
    def relocate(value):
        if isinstance(value, str):
            for prefix in (str(original_root), original_root.as_posix()):
                if value == prefix or value.startswith(prefix + "\\") or value.startswith(prefix + "/"):
                    return str(copied_root) + value[len(prefix):]
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relocate(item) for item in value]
        return value
    from temsim.specimen.presets import _preset_index, load_specimen_preset
    from temsim.specimen import reference_catalog
    from temsim.specimen.support import load_support_catalog
    from temsim.detector.eds_atomic import load_bote_salvat_coefficients
    readers = (_preset_index, load_specimen_preset, load_support_catalog, load_bote_salvat_coefficients)
    # Redirect this test process only during capture. Later calculations have
    # ordinary live roots and must explicitly resolve the archived input set.
    with monkeypatch.context() as capture_patch:
        try:
            for module_name, module in tuple(sys.modules.items()):
                if not module_name.startswith("temsim") or module is None:
                    continue
                for name, value in tuple(vars(module).items()):
                    if isinstance(value, Path) and value.is_relative_to(original_root):
                        capture_patch.setattr(module, name, copied_root / value.relative_to(original_root))
            capture_patch.setenv("TEMSIM_PROJECT_ROOT", str(copied_root.parent))
            for reader in readers:
                reader.cache_clear()
            state = decode_instrument(relocate(graph))
            snapshot = capture_instrument_snapshot(state)
            source = WorkingPointCheckpoint(snapshot, {}, state.sample.z_mm, snapshot.physical_digest,
                                            {"package_kind": "INSTRUMENT_INPUTS_ONLY"})
            portable = make_portable_inputs(source)
        finally:
            # A lazily imported module can copy a patched constant at import
            # time. Restore those new aliases too, before yielding the fixture.
            for module_name, module in tuple(sys.modules.items()):
                if not module_name.startswith("temsim") or module is None:
                    continue
                for name, value in tuple(vars(module).items()):
                    if isinstance(value, Path) and value.is_relative_to(copied_root):
                        setattr(module, name, original_root / value.relative_to(copied_root))
            for reader in readers:
                reader.cache_clear()
    yield portable, source, copied_root
    for reader in readers:
        reader.cache_clear()


def _remove_fixture_location(root, tmp_path):
    destination = root.with_name("originals-moved-aside")
    assert root.resolve().is_relative_to(tmp_path.resolve())
    assert destination.resolve().is_relative_to(tmp_path.resolve())
    root.rename(destination)
    assert not root.exists()


def test_moved_archive_reproduces_actual_tip_gun_and_column_checkpoints(portable_fixture, tmp_path):
    from temsim.optics.beam_path_audit import incident_checkpoints
    from temsim.specimen.source import active_cif_path
    from temsim.specimen.cif_io import read_cif_atoms
    portable, source, root = portable_fixture
    state = portable.compatible_state()
    planes = [state.electron_gun.exit_plane_z_mm, state.sample.upper_surface_z_mm]
    gun_a, cp_a, mask_a = incident_checkpoints(state, planes, step_mm=1.)
    with input_io.input_scope(state):
        atoms_a = read_cif_atoms(active_cif_path(state.sample), index=0)
    path = tmp_path / "moved-package" / "input.temwp"
    path.parent.mkdir()
    from temsim.calculation_cache import calculation_signatures
    # Retain the actually executed exact entrance checkpoint alongside every
    # archived input. This is a bounded audit, not a validated working point.
    arrays = {name: getattr(cp_a, name)[-1] for name in ("x_m", "tx_rad", "y_m", "ty_rad")}
    arrays.update(alive=mask_a[-1], weight=gun_a.exit_bundle.weight,
                  energy_offset_ev=gun_a.exit_bundle.energy_offset_ev)
    complete = WorkingPointCheckpoint(portable.snapshot, arrays, planes[-1], calculation_signatures(state)["incident"],
        dict(source_representation="classical_particles", validation_status="NOT_RUN",
             scope="Executed nine-ray incident audit at specimen entrance"), portable.digest)
    complete.write_package(path, mode="inputs_and_results")
    _remove_fixture_location(root, tmp_path)
    with pytest.raises(ValueError, match="Missing"):
        source.snapshot.restore()
    loaded = WorkingPointArchiveIndex.read(path).load()
    assert loaded.digest == complete.digest
    for name, value in complete.arrays.items():
        assert loaded.arrays[name].dtype == value.dtype
        assert np.array_equal(loaded.arrays[name], value, equal_nan=True)
    restored = loaded.compatible_state()
    assert capture_instrument_snapshot(restored).digest == portable.snapshot.digest
    assert encode_instrument(restored) == portable.snapshot.graph
    gun_b, cp_b, mask_b = incident_checkpoints(restored, planes, step_mm=1.)
    for field in ("z_mm", "x_m", "tx_rad", "y_m", "ty_rad"):
        assert np.array_equal(getattr(cp_a, field), getattr(cp_b, field), equal_nan=True)
    assert np.array_equal(mask_a, mask_b)
    assert np.array_equal(gun_a.exit_bundle.weight, gun_b.exit_bundle.weight)
    assert np.array_equal(gun_a.exit_bundle.energy_offset_ev, gun_b.exit_bundle.energy_offset_ev)
    with input_io.input_scope(restored):
        atoms_b = read_cif_atoms(active_cif_path(restored.sample), index=0)
    assert np.array_equal(atoms_a.numbers, atoms_b.numbers)
    assert np.array_equal(atoms_a.positions, atoms_b.positions)
    assert np.array_equal(atoms_a.cell.array, atoms_b.cell.array)
    assert input_io.active_archive() is None


def test_archive_survives_full_and_preview_capture_without_originals(portable_fixture, tmp_path):
    from temsim.gui.calculation_request import CapturedCalculationRequest
    point, _, root = portable_fixture
    state = point.compatible_state()
    _remove_fixture_location(root, tmp_path)
    for quality in ("Preview", "High accuracy"):
        request = CapturedCalculationRequest.capture(state, quality, 9, 1.)
        try:
            prepared = request.prepare(Event())
            assert prepared.snapshot._archive_inputs == state._archive_inputs
            assert prepared.external_inputs
            assert prepared.snapshot.electron_gun.source_representation == "classical_particles"
        finally:
            if request._input_assets is not None:
                request._input_assets.close()


def test_archive_isolation_prevents_live_or_other_archive_cache_hits(portable_fixture):
    from temsim.specimen.presets import load_specimen_preset
    point, _, _ = portable_fixture
    state = point.compatible_state()
    first = thaw_json(state._archive_inputs)
    second = thaw_json(state._archive_inputs)
    row = next(row for row in second["files"] if row["config_relative"] == "specimens/10_si_110.toml")
    original = bytes.fromhex(row["content_hex"])
    # Change only the human-readable name to observe which parsed file is used.
    import re
    modified, count = re.subn(rb'(?m)^name\s*=\s*"[^"]*"', b'name = "Archived fixture B"', original, count=1)
    assert count == 1
    from hashlib import sha256
    row.update(content_hex=modified.hex(), sha256=sha256(modified).hexdigest())
    second["digest"] = json_digest({key: value for key, value in second.items() if key != "digest"})
    def read(payload):
        with input_io.input_scope(SimpleNamespace(_archive_inputs=payload)):
            return load_specimen_preset("si_110").name
    expected = read(first)
    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = list(pool.map(read, [first, second]))
    assert a == expected and b == "Archived fixture B" and a != b
    assert read(first) == expected
    assert input_io.active_archive() is None


@pytest.mark.parametrize("mutation", ["checksum", "traversal", "duplicate", "missing"])
def test_bad_archive_cannot_restore_or_fall_back_to_present_live_inputs(portable_fixture, mutation):
    point, _, _ = portable_fixture
    graph = thaw_json(point.snapshot.graph)
    archive = graph["archived_inputs"]
    if mutation == "checksum":
        archive["files"][0]["content_hex"] = "00"
    elif mutation == "traversal":
        archive["files"][0]["config_relative"] = "../outside.toml"
    elif mutation == "duplicate":
        archive["files"].append(archive["files"][0])
    else:
        required = point.snapshot.external_inputs[0]["path"]
        archive["files"] = [row for row in archive["files"] if row["path"] != required]
    archive["digest"] = json_digest({key: value for key, value in archive.items() if key != "digest"})
    broken = replace(point.snapshot, graph=graph)
    with pytest.raises((ValueError, FileNotFoundError)):
        broken.restore()
    assert input_io.active_archive() is None


def test_solver_drift_does_not_admit_old_archive_results(portable_fixture):
    point, _, _ = portable_fixture
    old = replace(point.snapshot, implementation="another-solver")
    with pytest.raises(ValueError, match="implementation changed"):
        old.restore()


def test_runtime_drift_requires_explicit_new_input_identity(portable_fixture):
    from temsim.working_point import migrate_working_point_inputs
    point, _, _ = portable_fixture
    graph = thaw_json(point.snapshot.graph)
    archive = graph["archived_inputs"]
    archive["runtime"]["libraries"]["scipy"] = "historical-fixture"
    archive["digest"] = json_digest({key: value for key, value in archive.items() if key != "digest"})
    old = replace(point, snapshot=replace(point.snapshot, graph=graph))
    with pytest.raises(ValueError, match="runtime dependencies changed"):
        old.snapshot.restore()
    migrated = migrate_working_point_inputs(old)
    assert migrated.digest != old.digest and migrated.parent_id == old.digest
    assert migrated.is_input_design and not migrated.arrays
    assert migrated.snapshot.graph["archived_inputs"]["runtime"] == input_io.runtime_identity()
    assert old.snapshot.graph["archived_inputs"]["runtime"]["libraries"]["scipy"] == "historical-fixture"


def test_main_window_restores_archived_configuration_without_live_file_writes(portable_fixture, tmp_path, monkeypatch, qtbot):
    from PySide6.QtCore import QSettings
    from temsim.gui import main_window as shell, interactive_calculation as gui, calculation_controller as controller
    point, _, root = portable_fixture
    settings = QSettings(str(tmp_path / "archive-window.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(shell, "QSettings", lambda: settings)
    monkeypatch.setattr(gui, "QSettings", lambda: settings)
    monkeypatch.setattr(controller, "default_artifact_cache_root", lambda: tmp_path / "cache")
    monkeypatch.setattr(shell.MainWindow, "INITIAL_PREVIEW_DELAY_MS", 600000)
    errors = []
    monkeypatch.setattr(shell.MainWindow, "_show_error", lambda self, message: errors.append(message))
    window = shell.MainWindow()
    window.preview_timer.stop()
    qtbot.addWidget(window)
    monkeypatch.setattr(window.calculations.pool, "start", lambda *a: pytest.fail("Restore must not start physics"))
    monkeypatch.setattr(window, "_show_error", errors.append)
    live = window.state
    live_snapshot = capture_instrument_snapshot(live)
    _remove_fixture_location(root, tmp_path)
    window._restore_working_point(point)
    assert not errors
    assert capture_instrument_snapshot(window.state).digest == point.snapshot.digest
    assert window.catalog.root == root / "instruments"
    assert not root.exists()
    with pytest.raises(ValueError, match="read-only"):
        window._save_model_document(root / "instruments" / "gun" / "FEG.toml", {})
    assert window._save_manifest_updates(None, {"test": 1}) is False
    assert errors and "read-only" in errors[-1]
    assert not root.exists()
    # Returning to an existing live input record restores its catalog as well.
    errors.clear()
    previous = WorkingPointCheckpoint(live_snapshot, {}, live.sample.z_mm, live_snapshot.physical_digest,
                                      {"package_kind": "INSTRUMENT_INPUTS_ONLY"})
    window._restore_working_point(previous)
    assert not errors
    assert capture_instrument_snapshot(window.state).digest == live_snapshot.digest
    assert window.assembly_panel.catalog is window.catalog
    window.preview_timer.stop()
    assert window.calculations.pool.waitForDone(3000)


def test_portable_copy_button_runs_capture_off_thread_and_preserves_selection(tmp_path, monkeypatch, qtbot):
    import gc
    gc.collect()
    from temsim.gui.working_point_panel import WorkingPointPanel
    state = default_state()
    snapshot = capture_instrument_snapshot(state)
    source = WorkingPointCheckpoint(snapshot, {}, state.sample.z_mm, snapshot.physical_digest,
                                    {"package_kind": "INSTRUMENT_INPUTS_ONLY"})
    panel = WorkingPointPanel()
    qtbot.addWidget(panel)
    panel.add_checkpoint(source)
    errors = []
    panel.error.connect(errors.append)
    panel._make_portable()
    qtbot.waitUntil(lambda: not panel._archive_loading, timeout=30000)
    assert not errors
    assert panel.selected.digest == source.digest
    assert len(panel._points) == 2
    copied = next(point for point in panel._points if point.digest != source.digest)
    assert copied.parent_id == source.digest
    assert copied.is_input_design and "archived_inputs" in copied.snapshot.graph
    assert copied.snapshot.graph["nodes"] == source.snapshot.graph["nodes"]
    assert panel.shutdown()
