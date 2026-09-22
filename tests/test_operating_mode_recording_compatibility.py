"""Recording-module compatibility concerns the shared physical projector planes."""
from dataclasses import asdict, replace

import pytest

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.optics.column import default_state
from temsim.operating_modes import apply_operating_mode_pair, compatible_modes
from temsim.profile_io import apply_profile_values, read_profile, save_profile


@pytest.mark.parametrize("column", ["C2", "C3", "C3 + Probe Corrector"])
def test_recording_variants_share_projector_fields_and_upstream_reference_planes(column):
    catalog = AssemblyCatalog()
    states = []
    for recording in ("No Energy Filter", "Energy Filter"):
        state = default_state()
        catalog.apply(state, AssemblySelection("FEG", column, recording))
        states.append(state)
    without_filter, with_filter = states
    assert not without_filter.energy_filter.enabled
    assert with_filter.energy_filter.enabled
    # This guards the physical reason the same preset can be selected for both
    # modules. A future geometry/field change requires reassessing compatibility.
    for key in ("diffraction_lens", "intermediate_lens", "projector_lens_1", "projector_lens_2"):
        lenses = [next(lens for lens in state.lenses if lens.key == key) for state in states]
        assert asdict(lenses[0]) == asdict(lenses[1])
    assert without_filter.sample.z_mm == with_filter.sample.z_mm
    for attribute in ("camera", "fluorescent_screen"):
        planes = [getattr(state, attribute) for state in states]
        assert planes[0].z_mm == planes[1].z_mm
        assert planes[1].z_mm < with_filter.energy_filter.entrance_z_mm


@pytest.mark.parametrize("recording", ["No Energy Filter", "Energy Filter"])
@pytest.mark.parametrize("projector", ["imaging", "diffraction"])
def test_shared_projector_modes_are_explicitly_available_for_each_recording_module(recording, projector):
    modes = {mode.key: mode for mode in compatible_modes("projector", "C3 + Probe Corrector", recording)}
    assert projector in modes
    assert set(modes[projector].compatible_recording_systems) == {"Energy Filter", "No Energy Filter"}
    assert "downstream energy-filter camera" in modes[projector].calibration_reference
    assert "does not establish a calibration" in modes[projector].calibration_reference


@pytest.mark.parametrize("recording", ["No Energy Filter", "Energy Filter"])
def test_saved_recording_choice_survives_no_filter_startup_default(tmp_path, recording):
    catalog = AssemblyCatalog()
    selection = replace(catalog.default_selection(), recording=recording)
    assert catalog.default_selection().recording == "No Energy Filter"
    state = default_state()
    catalog.apply(state, selection)
    apply_operating_mode_pair(state, "micro_probe", "imaging",
                              column_name=selection.column, recording_name=selection.recording)
    path = tmp_path / "recording_choice.toml"
    save_profile(path, state, selection)
    loaded_selection, values = read_profile(path)
    assert loaded_selection == selection
    restored = default_state()
    catalog.apply(restored, loaded_selection)
    assert apply_profile_values(restored, values) is None
    assert restored.energy_filter.enabled is (recording == "Energy Filter")
    assert restored.projector_mode == "image"
    assert restored.camera.z_mm == state.camera.z_mm


def test_main_window_starts_with_default_unfiltered_recording(qtbot, monkeypatch, tmp_path):
    from PySide6.QtCore import QSettings
    from temsim.gui import main_window as shell, interactive_calculation as gui, calculation_controller as controller

    settings = QSettings(str(tmp_path / "startup.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(shell, "QSettings", lambda: settings)
    monkeypatch.setattr(gui, "QSettings", lambda: settings)
    monkeypatch.setattr(controller, "default_artifact_cache_root", lambda: tmp_path / "cache")
    monkeypatch.setattr(shell.MainWindow, "INITIAL_PREVIEW_DELAY_MS", 60000)
    window = shell.MainWindow()
    qtbot.addWidget(window)
    window.preview_timer.stop()
    assert window.selection.recording == "No Energy Filter"
    assert not window.state.energy_filter.enabled
    assert window.state.projector_mode in {"image", "diffraction"}
    window.close()
