from dataclasses import replace
import threading
import shutil
import tomllib
import tomli_w

import pytest
from PySide6.QtWidgets import QMessageBox, QPushButton

from temsim.assembly_catalog import AssemblyCatalog
from temsim.gui.assembly_panel import AssemblyPanel
from temsim.gui.main_window import MainWindow
from temsim.gui.operating_preset_controller import OperatingPresetController
from temsim.gui.parameter_panel import ParameterPanel
from temsim.optics.column import default_state
from temsim.profile_io import save_profile, read_profile, apply_profile_values
from temsim.runtime_parameters import runtime_targets, editable_parameters


def test_nanopulser_profile_and_snapshot_restore_installed_operating_state(tmp_path):
    catalog = AssemblyCatalog()
    selection = replace(catalog.default_selection(), beam_blanker="NanoPulser")
    state = default_state()
    catalog.apply(state, selection)
    state.nanopulser.blanked = True
    state.nanopulser.voltage_v = 600.0
    state.nanopulser.azimuth_deg = 35.0
    payload = state.to_dict()
    assert set(payload["nanopulser"]) == {"installed", "blanked", "voltage_v", "azimuth_deg"}
    restored = type(state).from_dict(payload)
    assert restored.nanopulser.to_dict() == state.nanopulser.to_dict()
    assert restored.nanopulser.z_mm == state.nanopulser.z_mm
    assert restored.sample.z_mm == state.sample.z_mm

    path = tmp_path / "nanopulser.toml"
    save_profile(path, state, selection)
    selected, values = read_profile(path)
    fresh = default_state()
    catalog.apply(fresh, selected)
    apply_profile_values(fresh, values)
    assert selected == selection
    assert fresh.nanopulser.to_dict() == state.nanopulser.to_dict()


def test_legacy_profile_has_no_optional_nanopulser(tmp_path):
    path = tmp_path / "legacy.toml"
    path.write_text(
        'format_version=2\n[assembly]\ngun="FEG"\ncolumn="C3"\nrecording="Energy Filter"\n',
        encoding="utf-8",
    )
    selection, _ = read_profile(path)
    assert selection.beam_blanker == "None"


def test_assembly_selector_and_quick_blanking_controls(qtbot):
    catalog = AssemblyCatalog()
    panel = AssemblyPanel(catalog, catalog.default_selection())
    qtbot.addWidget(panel)
    assert panel.beam_blanker.currentText() == "None"
    panel.beam_blanker.setCurrentText("NanoPulser")
    selection = panel.current_selection()
    assert selection.beam_blanker == "NanoPulser"
    panel.reload_catalog(catalog, selection)
    assert panel.beam_blanker.count() == 2
    assert panel.beam_blanker.currentText() == "NanoPulser"

    state = default_state()
    catalog.apply(state, selection)
    gun_target = runtime_targets(state)[state.electron_gun.deflector.key]
    parameters = ParameterPanel()
    qtbot.addWidget(parameters)
    parameters.set_context(gun_target.label, gun_target, None, (), None)
    with qtbot.waitSignal(parameters.runtime_changed):
        parameters._quick_widgets["beam_blanked"].setChecked(True)
    assert state.beam_blanked
    assert not state.nanopulser.blanked

    target = runtime_targets(state)["nanopulser_deflector"]
    assert {p.name for p in editable_parameters(target)} == {"blanked", "voltage_v", "azimuth_deg"}
    parameters.set_context(target.label, target, None, (), None)
    with qtbot.waitSignal(parameters.runtime_changed):
        parameters._quick_widgets["blanked"].setChecked(True)
    assert state.nanopulser.blanked
    parameters._quick_widgets["voltage_v"].setValue(450.0)
    assert state.nanopulser.voltage_v == 450.0


def test_background_preset_cancellation_does_not_change_live_state(qtbot, monkeypatch):
    import temsim.gui.operating_preset_controller as controller_module

    state = default_state()
    original = next(lens.percent for lens in state.lenses if lens.key == "condenser_lens_2")
    catalog = AssemblyCatalog()
    entered = threading.Event()
    release = threading.Event()
    result_threads = []

    def solve(snapshot, *_args, **_kwargs):
        next(lens for lens in snapshot.lenses if lens.key == "condenser_lens_2").percent = 13.0
        result_threads.append(threading.get_ident())
        entered.set()
        assert release.wait(10.0)
        return None

    monkeypatch.setattr(controller_module, "apply_operating_mode_pair", solve)
    controller = OperatingPresetController()
    delivered = []
    controller.result_ready.connect(lambda *result: delivered.append(result))
    controller.submit(state, catalog, catalog.default_selection(), "nano_probe", "diffraction")
    try:
        qtbot.waitUntil(entered.is_set, timeout=30_000)
        assert len(result_threads) == 1
        assert result_threads[0] != threading.get_ident()
        assert next(lens.percent for lens in state.lenses if lens.key == "condenser_lens_2") == original
        controller.invalidate_pending()
    finally:
        release.set()
        assert controller.pool.waitForDone(30_000)
    qtbot.wait(20)
    assert delivered == []
    assert next(lens.percent for lens in state.lenses if lens.key == "condenser_lens_2") == original


def test_component_blank_and_pending_assembly_are_transactional(qtbot, monkeypatch):
    monkeypatch.setattr(MainWindow, "run_preview", lambda *_: None)
    monkeypatch.setattr(QMessageBox, "critical", lambda *_: None)
    window = MainWindow()
    qtbot.addWidget(window)
    window.preview_timer.stop()
    assert window.findChild(QPushButton, "blankBeamButton") is None
    assert window.findChild(QPushButton, "nanopulserBlankButton") is None

    def set_blank(key, field, value):
        window._select_component_from_workspace(key)
        window.parameter_panel._quick_widgets[field].setChecked(value)

    set_blank(window.state.electron_gun.deflector.key, "beam_blanked", True)
    assert window.state.beam_blanked
    assert not window.state.nanopulser.blanked
    set_blank(window.state.electron_gun.deflector.key, "beam_blanked", False)
    assert not window.state.beam_blanked
    original_state = window.state
    selection = replace(window.selection, beam_blanker="NanoPulser")
    submitted = []
    monkeypatch.setattr(window.operating_presets, "submit", lambda *args, **kwargs: submitted.append((args, kwargs)))
    window.load_assembly(selection)
    assert submitted
    assert window.state is original_state
    assert window._preset_state_token is not None

    candidate = type(window.state).from_dict(window.state.to_dict())
    window.catalog.apply(candidate, selection)
    window._operating_preset_ready(candidate, selection, None, 1.0)
    window._operating_preset_finished()
    assert window.state is candidate
    assert window.selection == selection
    assert window.findChild(QPushButton, "blankBeamButton") is None
    assert window.findChild(QPushButton, "nanopulserBlankButton") is None
    set_blank(candidate.electron_gun.deflector.key, "beam_blanked", True)
    assert candidate.beam_blanked
    assert not candidate.nanopulser.blanked
    set_blank("nanopulser_deflector", "blanked", True)
    assert candidate.nanopulser.blanked
    set_blank(candidate.electron_gun.deflector.key, "beam_blanked", False)
    assert not candidate.beam_blanked
    assert candidate.nanopulser.blanked
    set_blank("nanopulser_deflector", "blanked", False)
    assert not candidate.nanopulser.blanked

    window.apply_operating_modes("micro_probe", "imaging")
    assert window._preset_state_token is not None
    set_blank(candidate.electron_gun.deflector.key, "beam_blanked", True)
    assert window._preset_state_token is None
    assert candidate.beam_blanked
    assert not candidate.nanopulser.blanked
    window.preview_timer.stop()


def test_background_preset_solves_selected_catalog_geometry(qtbot, monkeypatch, tmp_path):
    import temsim.gui.operating_preset_controller as controller_module

    original = AssemblyCatalog()
    root = tmp_path / "instruments"
    shutil.copytree(original.root, root)
    path = root / "beam_blanker" / "NanoPulser.toml"
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    document["geometry"]["length_mm"] = 100.0
    document["ports"]["exit"]["local_z_mm"] = 100.0
    path.write_text(tomli_w.dumps(document), encoding="utf-8")
    catalog = AssemblyCatalog(root)
    selection = replace(catalog.default_selection(), beam_blanker="NanoPulser")
    state = default_state()
    catalog.apply(state, selection)
    expected = state.sample.z_mm
    seen = []

    def solve(snapshot, *_args, **_kwargs):
        seen.append(snapshot.sample.z_mm)
        assert snapshot._resolved_assembly.root == root.resolve()
        return None

    monkeypatch.setattr(controller_module, "apply_operating_mode_pair", solve)
    controller = OperatingPresetController()
    with qtbot.waitSignal(controller.result_ready, timeout=30_000) as result:
        controller.submit(state, catalog, selection, "nano_probe", "diffraction")
    assert controller.pool.waitForDone(30_000)
    assert seen == [expected]
    assert result.args[0].sample.z_mm == expected
