"""Current dialog constructors with optional hardware; no transport is run."""
import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.instrument_configuration import InstrumentUnits
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state


@pytest.mark.parametrize("configuration", ("default", "minimal", "all_optional"))
def test_current_physical_dialogs_open_without_mutating_or_calculating(
    qtbot, monkeypatch, configuration,
):
    from temsim.gui.aberration_dialog import AberrationSettingsDialog
    from temsim.gui.excitation_dialog import ExcitationDialog
    from temsim.gui.gun_source_dialog import GunSourceDialog
    from temsim.gui.part_feature_dialog import PartFeatureDialog
    from temsim.gui.qualification_dialog import QualificationDialog
    from temsim.excitation_calibration import ExcitationCalibration

    state = default_state()
    if configuration != "default":
        all_optional = configuration == "all_optional"
        units = InstrumentUnits(
            source="cold_feg" if all_optional else "thermionic",
            monochromator=all_optional, beam_blanker=all_optional,
            c3_lens=all_optional, probe_corrector=all_optional,
            image_corrector=all_optional, energy_filter=all_optional,
        )
        catalog = AssemblyCatalog()
        catalog.apply(state, units.selection(catalog), preserve_operating_parameters=True)
    before = capture_instrument_snapshot(state).digest

    def reject_calculation(*args, **kwargs):
        pytest.fail("Opening a current settings dialog must not run transport")

    monkeypatch.setattr("temsim.physics.simulation.run", reject_calculation)
    monkeypatch.setattr("temsim.operating_modes.apply_operating_mode_pair", reject_calculation)
    dialogs = [
        AberrationSettingsDialog(state.probe_aberrations, system="probe", state_step_mm=state.step_mm),
        AberrationSettingsDialog(state.image_aberrations, system="image", state_step_mm=state.step_mm),
        ExcitationDialog(ExcitationCalibration(250), state.lenses[0]),
        QualificationDialog(),
        PartFeatureDialog({"kind": "hole", "diameter_mm": 1., "depth_mm": 2.}),
        PartFeatureDialog({"kind": "slot", "width_mm": 1., "length_mm": 2., "depth_mm": 3.}),
    ]
    if state.electron_gun.type_key == "cold_feg":
        dialogs.append(GunSourceDialog(state.electron_gun, instrument_state=state))
    else:
        with pytest.raises(ValueError, match="FEG tip emission only"):
            GunSourceDialog(state.electron_gun, instrument_state=state)
    for dialog in dialogs:
        qtbot.addWidget(dialog)
        dialog.show()
        assert dialog.isVisible()
        dialog.reject()
    assert capture_instrument_snapshot(state).digest == before
