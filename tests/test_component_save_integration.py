"""Saving inactive geometry must not install its preset or replace live state."""

import shutil
import tomllib

import pytest
import tomli_w
import numpy as np
from PySide6.QtCore import QSettings

from temsim.assembly_catalog import AssemblyCatalog
from temsim.component_operations import make_component
from temsim.gui import main_window
from temsim.manifest_editor import ManifestEditor
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT


@pytest.fixture
def window(qtbot, tmp_path, monkeypatch):
    root = tmp_path / "instruments"
    shutil.copytree(INSTRUMENT_CONFIG_ROOT, root)
    monkeypatch.setattr(main_window, "AssemblyCatalog", lambda: AssemblyCatalog(root))
    monkeypatch.setattr(main_window, "ManifestEditor", lambda: ManifestEditor(root))
    monkeypatch.setattr(main_window, "QSettings", lambda: QSettings(
        str(tmp_path / "window.ini"), QSettings.Format.IniFormat))
    monkeypatch.setattr(main_window.MainWindow, "schedule_preview", lambda *args: None)
    widget = main_window.MainWindow()
    qtbot.addWidget(widget)
    widget.preview_timer.stop()
    return widget, root


def test_inactive_catalog_save_keeps_current_state_and_selection(window):
    window, root = window
    path = root / "column/C2.toml"
    assert path.relative_to(root).as_posix() not in dict(window.assembly.selected_module_paths).values()
    page = window.workspace.physical_layout.model_editor
    assert page.open_path(path)
    page._selected_key = page.session.add_component(make_component(
        key="user_inactive_bracket", shape="box", center_z_mm=60, length_mm=4,
    ))
    state, assembly, selection = window.state, window.assembly, window.selection
    assert page.save(), page.status.text()
    assert window.state is state and window.assembly is assembly and window.selection == selection
    assert PartModelDocument(path).part("user_inactive_bracket")["length_mm"] == 4
    assert not page.session.dirty


def test_active_catalog_save_adds_part_to_resolved_assembly_without_optical_control(window):
    window, root = window
    path = root / dict(window.assembly.selected_module_paths)["column"]
    page = window.workspace.physical_layout.model_editor
    assert page.open_path(path)
    key = page.session.add_component(make_component(
        key="user_active_bracket", shape="box", center_z_mm=60, length_mm=4,
    ))
    page._selected_key = key
    previous_keys = {part.key for part in window.assembly.parts}
    assert page.save(), page.status.text()
    assert {part.key for part in window.assembly.parts} == previous_keys | {key}
    part = window.assembly.part(key)
    assert part.data["mechanical_only"] is True
    assert key not in window._runtime_targets
    assert not page.session.dirty
    from temsim.assembly_model_3d import assembly_model_from_assembly
    model = assembly_model_from_assembly(window.assembly)
    body = next(mesh for mesh in model.meshes if mesh.key == key)
    assert np.min(body.vertices[:, 2]) == pytest.approx(part.center_z_mm - 2)
    assert np.max(body.vertices[:, 2]) == pytest.approx(part.center_z_mm + 2)


def test_catalog_reload_failure_restores_destination_and_retains_draft(window, monkeypatch):
    window, root = window
    path = root / dict(window.assembly.selected_module_paths)["column"]
    page = window.workspace.physical_layout.model_editor
    assert page.open_path(path)
    original = path.read_bytes()
    page._selected_key = page.session.add_component(make_component(
        key="user_rollback_bracket", shape="box", center_z_mm=60, length_mm=4,
    ))
    def fail(*args, **kwargs):
        raise ValueError("Synthetic runtime reload failure")
    monkeypatch.setattr(window.catalog, "apply", fail)
    assert page.save() is False
    assert "Synthetic runtime reload failure" in page.status.text()
    assert page.session.dirty and path.read_bytes() == original


def test_windows_catalog_paths_still_reload_the_installed_assembly(window):
    window, root = window
    catalog_path = root / "catalog.toml"
    document = tomllib.loads(catalog_path.read_text(encoding="utf-8"))
    for group, rows in document.items():
        if group.endswith("_variants"):
            for row in rows:
                row["file"] = row["file"].replace("/", "\\")
    catalog_path.write_bytes(tomli_w.dumps(document).encode("utf-8"))
    # POSIX has no backslash path separator; this is a Windows catalog case.
    import os
    if os.name != "nt":
        pytest.skip("Windows native catalog paths")
    window.catalog = AssemblyCatalog(root)
    window.assembly = window.catalog.apply(window.state, window.selection, preserve_operating_parameters=True)
    window._refresh_assembly_views()
    page = window.workspace.physical_layout.model_editor
    source = dict(window.assembly.selected_module_paths)["column"]
    assert "\\" in source
    path = root / source
    assert path.relative_to(root).as_posix() in page._project_paths
    assert page.open_path(path)
    key = page.session.add_component(make_component(key="user_windows_path_bracket", shape="box"))
    page._selected_key = key
    assert page.save(), page.status.text()
    assert window.assembly.part(key).data["mechanical_only"] is True
