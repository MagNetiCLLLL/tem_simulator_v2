"""Unit assembly and draft transactions; no ray/wave calculation required."""
from itertools import product

import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.instrument_configuration import InstrumentUnits, check_instrument_configuration, unit_for_component
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state


@pytest.fixture
def state():
    return default_state()


def test_all_60_supported_unit_combinations_use_existing_assembly_interfaces(state):
    catalog = AssemblyCatalog()
    count = 0
    for source, mono in (("cold_feg", False), ("cold_feg", True), ("thermionic", False)):
        for c3, probe, image in ((False, False, False), (True, False, False),
                                 (True, True, False), (True, False, True), (True, True, True)):
            for blanker, energy_filter in product((False, True), repeat=2):
                units = InstrumentUnits(source, mono, blanker, c3, probe, image, energy_filter)
                selection = units.selection(catalog)
                assert InstrumentUnits.from_selection(catalog, selection) == units
                assembly = catalog.apply(state, selection, preserve_operating_parameters=True)
                assert catalog.selection_for_resolved(assembly) == selection
                assert state.electron_gun.type_key == source
                assert state.monochromator_installed == mono
                assert state.nanopulser.installed == blanker
                assert state.probe_corrector_installed == probe
                assert state.image_corrector_installed == image
                assert state.energy_filter_installed == energy_filter
                keys = {p.key for p in assembly.parts}
                assert ("condenser_lens_3" in keys) == c3
                assert ("energy_filter" in keys) == energy_filter
                assert {"sample", "objective_lens", "camera", "haadf", "projector_lens_2"} <= keys
                count += 1
    assert count == 60


@pytest.mark.parametrize("units,reason", [
    (InstrumentUnits(source="thermionic", monochromator=True), "requires Cold FEG"),
    (InstrumentUnits(c3_lens=False), "require C3"),
    (InstrumentUnits(c3_lens=False, probe_corrector=False, image_corrector=True), "require C3"),
    (InstrumentUnits(energy_filter="false"), "explicit on/off"),
])
def test_invalid_units_are_rejected_before_mutation(state, units, reason):
    before = capture_instrument_snapshot(state).digest
    with pytest.raises(ValueError, match=reason):
        check_instrument_configuration(state, AssemblyCatalog(), units)
    assert capture_instrument_snapshot(state).digest == before


def test_check_is_read_only_preserves_shared_controls_and_does_not_calculate(state, monkeypatch):
    catalog = AssemblyCatalog()
    state.electron_gun.emitter.curvature_nm_inv = 1e-8
    state.objective_lens.percent = 64.123
    before = capture_instrument_snapshot(state).digest
    monkeypatch.setattr("temsim.physics.simulation.run", lambda *a, **kw: pytest.fail("Check propagated electrons"))
    monkeypatch.setattr("temsim.operating_modes.apply_operating_mode_pair", lambda *a, **kw: pytest.fail("Check applied a preset"))
    checked = check_instrument_configuration(state, catalog, InstrumentUnits(energy_filter=False))
    assert capture_instrument_snapshot(state).digest == before
    candidate = checked.restore_for(state)
    assert candidate.energy_filter_installed is False
    assert candidate.electron_gun.emitter.curvature_nm_inv == 1e-8
    assert candidate.objective_lens.percent == 64.123
    assert candidate is not state
    original_parts = {p.key: p for p in state._resolved_assembly.parts}
    for part in candidate._resolved_assembly.parts:
        if part.key in original_parts:
            assert part.center_z_mm == original_parts[part.key].center_z_mm


def test_unchanged_check_never_resets_the_instrument_and_stale_check_rejected(state):
    state.objective_lens.percent = 61.25
    catalog = AssemblyCatalog()
    units = InstrumentUnits.from_selection(catalog, catalog.selection_for_resolved(state._resolved_assembly))
    checked = check_instrument_configuration(state, catalog, units)
    assert checked.original.digest == checked.candidate.digest
    state.objective_lens.percent += .01
    with pytest.raises(ValueError, match="changed"):
        checked.restore_for(state)


def test_units_include_mechanical_children_and_filter_entrance(state):
    from dataclasses import replace
    catalog = AssemblyCatalog()
    a = catalog.apply(state, replace(catalog.default_selection(), recording="Energy Filter"))
    assert unit_for_component(a, "condenser_lens_3_excitation_coil") == "c3_lens"
    assert unit_for_component(a, "feg_tip") == "source"
    assert unit_for_component(a, "energy_filter_entrance_aperture") == "energy_filter"
    assert unit_for_component(a, "sample") is None


def _dialog(qtbot, state, tmp_path, on_assemble=None):
    from PySide6.QtCore import QSettings, Qt
    from temsim.gui.instrument_configuration_dialog import InstrumentConfigurationDialog
    dialog = InstrumentConfigurationDialog(AssemblyCatalog(), lambda: state,
        on_assemble or (lambda checked: None), settings=QSettings(str(tmp_path/"ui.ini"), QSettings.Format.IniFormat))
    # Successful Assemble closes the dialog; pytest-qt still owns its wrapper
    # and closes it during teardown.
    dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
    qtbot.addWidget(dialog)
    return dialog


def test_dialog_dependencies_draft_and_linked_review(qtbot, state, tmp_path):
    d = _dialog(qtbot, state, tmp_path)
    before = capture_instrument_snapshot(state).digest
    assembly = d._assembly
    assert d.assemble_button.isEnabled()
    assert not hasattr(d, "check_button") and not hasattr(d, "check")
    assert d.review.model_editor is None
    d.options["c3_lens"].setChecked(False)
    for key in ("probe_corrector", "image_corrector"):
        assert not d.options[key].isEnabled() and not d.options[key].isChecked()
    d.options["monochromator"].setChecked(True)
    d.source.setCurrentIndex(d.source.findData("thermionic"))
    assert not d.options["monochromator"].isEnabled()
    assert not d.options["monochromator"].isChecked()
    d.options["energy_filter"].setChecked(False)
    assert d.assemble_button.isEnabled()
    assert d._assembly is assembly  # Read-only review retains the installed unit.
    assert capture_instrument_snapshot(state).digest == before
    d.select_component("feg_tip")
    assert d.table.currentRow() == 0
    d.options["c3_lens"].setChecked(True)
    assert d.options["probe_corrector"].isEnabled()
    assert d.assemble_button.isEnabled()
    assert "draft changed" in d.preview_status.text().lower()


def test_dialog_single_click_validates_current_state_and_applies_once(qtbot, state, tmp_path):
    from PySide6.QtWidgets import QDialog
    applied = []

    def apply(checked):
        assert not d.assemble_button.isEnabled()
        d.assemble()  # A nested event cannot apply the same draft again.
        applied.append(checked.restore_for(state))

    d = _dialog(qtbot, state, tmp_path, apply)
    d.options["energy_filter"].setChecked(True)
    # Live edits after opening are captured at this click, not rejected against
    # an obsolete separately checked draft.
    state.objective_lens.percent += .125
    before = capture_instrument_snapshot(state).digest
    d.assemble_button.click()
    assert len(applied) == 1
    assert applied[0].energy_filter_installed
    assert applied[0].objective_lens.percent == state.objective_lens.percent
    assert capture_instrument_snapshot(state).digest == before
    assert d.result() == QDialog.DialogCode.Accepted


def test_dialog_validation_failure_does_not_apply_and_can_retry(qtbot, state, tmp_path, monkeypatch):
    applied = []
    d = _dialog(qtbot, state, tmp_path, applied.append)
    before = capture_instrument_snapshot(state).digest
    valid_units = d.units
    monkeypatch.setattr(d, "units", lambda: InstrumentUnits(source="thermionic", monochromator=True))
    d.show()
    d.assemble_button.click()
    assert not applied and d.isVisible()
    assert "requires Cold FEG" in d.status.text()
    assert d.assemble_button.isEnabled()
    assert capture_instrument_snapshot(state).digest == before
    monkeypatch.setattr(d, "units", valid_units)
    d.assemble_button.click()
    assert len(applied) == 1


def test_dialog_failed_apply_stays_open_and_can_retry(qtbot, state, tmp_path):
    attempts = []

    def apply(checked):
        candidate = checked.restore_for(state)
        attempts.append(candidate)
        if len(attempts) == 1:
            raise RuntimeError("Installation rejected")

    d = _dialog(qtbot, state, tmp_path, apply)
    before = capture_instrument_snapshot(state).digest
    d.show()
    d.assemble_button.click()
    assert "Installation rejected" in d.status.text()
    assert d.isVisible() and d.assemble_button.isEnabled()
    assert capture_instrument_snapshot(state).digest == before
    d.assemble_button.click()
    assert len(attempts) == 2


def test_dialog_layout_is_saved_separately_and_draft_does_not_apply(qtbot, state, tmp_path):
    from PySide6.QtCore import QSettings, Qt
    from temsim.gui.instrument_configuration_dialog import InstrumentConfigurationDialog
    settings = QSettings(str(tmp_path/"layout.ini"), QSettings.Format.IniFormat)
    applied = []
    d = InstrumentConfigurationDialog(AssemblyCatalog(), lambda: state, applied.append,
        settings=settings, layout_id="review")
    # Keep the wrapper alive until pytest-qt closes its registered widgets.
    d.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
    qtbot.addWidget(d)
    # The offscreen screen is 800 px wide; restoreGeometry correctly clamps
    # windows saved larger than the current screen. Test a fitting width.
    d.resize(780, 700)
    d.show()
    qtbot.wait(20)
    d.splitter.setSizes([260, 500])
    expected = d.size()
    expected_split = d.splitter.sizes()
    d.options["energy_filter"].setChecked(not d.options["energy_filter"].isChecked())
    assert not applied
    d.reject()
    restored = InstrumentConfigurationDialog(AssemblyCatalog(), lambda: state, applied.append,
        settings=settings, layout_id="review")
    qtbot.addWidget(restored)
    restored.show()
    qtbot.wait(20)
    assert restored.size() == expected
    assert restored.splitter.sizes() == expected_split
    assert settings.value("instrument_configuration/v1/default/geometry") is None


def _window(qtbot, monkeypatch, tmp_path):
    from PySide6.QtCore import QSettings
    from temsim.gui import main_window as shell, interactive_calculation, instrument_configuration_dialog
    settings = QSettings(str(tmp_path / "window.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(shell, "QSettings", lambda: settings)
    monkeypatch.setattr(interactive_calculation, "QSettings", lambda: settings)
    monkeypatch.setattr(instrument_configuration_dialog, "QSettings", lambda: settings)
    monkeypatch.setattr(shell.MainWindow, "INITIAL_PREVIEW_DELAY_MS", 60000)
    monkeypatch.setattr(shell.MainWindow, "_apply_state_operating_modes", lambda *_: object())
    window = shell.MainWindow()
    qtbot.addWidget(window)
    window.preview_timer.stop()
    monkeypatch.setattr(window.calculations.pool, "start", lambda *_: pytest.fail("Configuration must not calculate"))
    return window


def test_main_window_configuration_applies_once_without_preset_or_ray_solve(qtbot, monkeypatch, tmp_path):
    window = _window(qtbot, monkeypatch, tmp_path)
    previews = []
    monkeypatch.setattr(window, "schedule_preview", lambda *a: previews.append(True))
    monkeypatch.setattr(window, "_start_operating_preset", lambda *a, **kw: pytest.fail("Automatic preset"))
    d = window.open_instrument_configuration()
    assert window.open_instrument_configuration() is d
    d.options["beam_blanker"].setChecked(True)
    d.options["energy_filter"].setChecked(True)
    d.assemble_button.click()
    assert window.state.nanopulser.installed
    assert window.state.energy_filter_installed
    assert window.selection.recording == "Energy Filter"
    assert window.assembly_panel.current_selection() == window.selection
    assert previews == [True]
    assert window.workspace.physical_layout._result.assembly is window.assembly
    assert window._configuration_dialog is None


def test_closing_main_window_saves_open_configuration_without_applying(qtbot, tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings
    window = _window(qtbot, monkeypatch, tmp_path)
    before = capture_instrument_snapshot(window.state).digest
    d = window.open_instrument_configuration()
    settings = QSettings(str(tmp_path / "close.ini"), QSettings.Format.IniFormat)
    d.settings = settings
    key = d.settings_key
    d.options["energy_filter"].setChecked(not d.options["energy_filter"].isChecked())
    window.close()
    assert settings.value(key + "/geometry") is not None
    assert settings.value(key + "/splitter") is not None
    assert window._configuration_dialog is None
    assert capture_instrument_snapshot(window.state).digest == before
