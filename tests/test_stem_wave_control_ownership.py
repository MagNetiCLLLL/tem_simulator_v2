"""The STEM wave switch has one UI owner and retains its saved-state contract."""

from unittest.mock import Mock

import pytest
from PySide6.QtWidgets import QCheckBox, QGroupBox, QLabel

from temsim.assembly_catalog import AssemblyCatalog
from temsim.gui.sample_panel import SamplePage
from temsim.gui.scan_panel import ScanControlView
from temsim.optics.column import default_state
from temsim.profile_io import apply_profile_values, read_profile, save_profile


@pytest.fixture
def wave_panels(qtbot, monkeypatch):
    """Exercise GUI ownership without allowing a ray or wave calculation."""

    forbidden = Mock(side_effect=AssertionError("UI binding must not calculate physics"))
    monkeypatch.setattr("temsim.simulation_pipeline.calculate", forbidden)
    monkeypatch.setattr("temsim.gui.calculation_controller.calculate", forbidden)
    monkeypatch.setattr(
        "temsim.physics.stem_wave_imaging.simulate_angle_resolved_stem",
        forbidden,
    )
    sample_page = SamplePage()
    scan_view = ScanControlView()
    qtbot.addWidget(sample_page)
    qtbot.addWidget(scan_view)
    yield sample_page, scan_view
    forbidden.assert_not_called()


def test_stem_wave_toggle_has_only_scanning_image_owner(wave_panels):
    sample_page, scan_view = wave_panels
    state = default_state()
    sample_page.set_state(state)
    scan_view.set_state(state)

    toggles = [
        checkbox
        for panel in (sample_page, scan_view)
        for checkbox in panel.findChildren(QCheckBox)
        if checkbox.objectName() in {"sampleStemWaveEnabled", "stemWaveScanEnabled"}
    ]
    assert toggles == [scan_view.wave_scan_enabled]
    assert not hasattr(sample_page, "stem_wave_enabled")
    assert scan_view.wave_scan_enabled.text() == "Calculate STEM detector images (High accuracy)"
    assert sample_page.findChild(QCheckBox, "sampleTemWaveEnabled") is sample_page.tem_wave_enabled
    assert sample_page.findChild(QGroupBox, "sampleWaveControls").title() == "Wave imaging settings"
    location = sample_page.findChild(QLabel, "sampleStemImageLocation")
    assert location is not None
    assert "Scanning Image" in location.text()
    assert "Scanning Parameters" in location.text()


@pytest.mark.parametrize("enabled", [False, True])
def test_scanning_toggle_updates_shared_field_and_emits_once(wave_panels, enabled):
    sample_page, scan_view = wave_panels
    state = default_state()
    state.sample.stem_wave_enabled = not enabled
    sample_page.set_state(state)
    scan_view.set_state(state)
    scan_changes = []
    sample_changes = []
    scan_view.parameters_changed.connect(scan_changes.append)
    sample_page.parameters_changed.connect(sample_changes.append)

    scan_view.wave_scan_enabled.setChecked(enabled)

    assert state.sample.stem_wave_enabled is enabled
    assert scan_changes == ["sample.stem_wave_enabled"]
    assert sample_changes == []


@pytest.mark.parametrize("enabled", [False, True])
def test_panel_refresh_preserves_stem_request_without_emitting(wave_panels, enabled):
    sample_page, scan_view = wave_panels
    state = default_state()
    state.sample.stem_wave_enabled = enabled
    state.sample.wave_grid_pixels = 64
    state.sample.wave_slice_thickness_angstrom = 1.5
    scan_changes = []
    sample_changes = []
    scan_view.parameters_changed.connect(scan_changes.append)
    sample_page.parameters_changed.connect(sample_changes.append)

    for illumination in ("STEM", "TEM", "STEM"):
        state.illumination_mode = illumination
        sample_page.set_state(state)
        scan_view.set_state(state)
        assert state.sample.stem_wave_enabled is enabled
        assert scan_view.wave_scan_enabled.isChecked() is enabled

    assert state.sample.wave_grid_pixels == 64
    assert state.sample.wave_slice_thickness_angstrom == pytest.approx(1.5)
    assert scan_changes == []
    assert sample_changes == []


@pytest.mark.parametrize("enabled", [False, True])
def test_operating_profile_restores_stem_request_to_sole_toggle(
    wave_panels, tmp_path, enabled
):
    sample_page, scan_view = wave_panels
    state = default_state()
    state.sample.stem_wave_enabled = enabled
    selection = AssemblyCatalog().default_selection()
    path = tmp_path / "stem-wave-selection.toml"
    save_profile(path, state, selection)

    loaded_selection, values = read_profile(path)
    restored = default_state()
    restored.sample.stem_wave_enabled = not enabled
    assert apply_profile_values(restored, values) == []
    scan_changes = []
    scan_view.parameters_changed.connect(scan_changes.append)
    sample_page.set_state(restored)
    scan_view.set_state(restored)

    assert loaded_selection == selection
    assert restored.sample.stem_wave_enabled is enabled
    assert scan_view.wave_scan_enabled.isChecked() is enabled
    assert scan_changes == []


@pytest.mark.parametrize("illumination", ["TEM", "STEM"])
def test_scanning_wave_control_keeps_existing_illumination_support(
    wave_panels, illumination
):
    sample_page, scan_view = wave_panels
    state = default_state()
    state.illumination_mode = illumination
    state.sample.stem_wave_enabled = False
    sample_page.set_state(state)
    scan_view.set_state(state)

    assert scan_view.wave_scan_enabled.isEnabled()
    scan_view.wave_scan_enabled.setChecked(True)

    assert state.sample.stem_wave_enabled is True
    assert sample_page.tem_wave_enabled.isEnabled() is (illumination == "TEM")
