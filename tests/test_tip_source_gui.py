"""Source-editor routing and state ownership; not full-chain image validation."""
from dataclasses import replace
import math
import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.gui.gun_source_dialog import GunSourceDialog
from temsim.gui.model_inspector import ModelInspectorPage
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state


@pytest.fixture
def state():
    value = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(value, catalog.default_selection())
    value.electron_gun.emitter.surface_model = None  # historical source editor fixture
    value.electron_gun.emitter.ray_count = 9
    # Existing preview serialization snaps an aperture anchor by one ULP.
    # Establish that canonical layout before measuring source-edit ownership.
    value.to_dict()
    return value


def test_tip_button_shares_existing_controls_without_a_full_width_row(qtbot):
    page = ModelInspectorPage()
    qtbot.addWidget(page)
    page.resize(1000, 700)
    page.show()
    layout = page.layout()
    row = next(layout.itemAt(i).layout() for i in range(layout.count())
               if layout.itemAt(i).layout() is not None
               and layout.itemAt(i).layout().indexOf(page.lens) >= 0)
    assert row.indexOf(page.gun_source_button) >= 0
    assert layout.indexOf(page.gun_source_button) == -1
    assert page.gun_source_button.width() <= page.gun_source_button.sizeHint().width()
    assert "emission" in page.gun_source_button.toolTip()


def test_near_field_preview_clones_installed_assembly_and_keeps_live_source(state, qtbot, monkeypatch):
    from temsim.gui import surface_wave_dialog
    # Deliberately retain runtime edits instead of rebuilding from the manifest.
    state.objective_lens.percent = 61.23
    state.objective_lens.upper_b0_t += .0123
    state.electron_gun.extractor.voltage_kv = 4.1
    before = capture_instrument_snapshot(state).digest
    dialog = GunSourceDialog(state.electron_gun, instrument_state=state)
    qtbot.addWidget(dialog)
    dialog.surface_enabled.setChecked(True)
    dialog.surface_coherent.setChecked(True)
    dialog.surface_inputs["kinetic_mean_ev"].setText("0.42")
    opened = []

    class Viewer:
        """Offline GUI handoff recorder: no propagation is performed here."""
        def __init__(self, snapshot, parent):
            opened.append(snapshot)
            assert parent is dialog

        def exec(self):
            return 0

    monkeypatch.setattr(surface_wave_dialog, "SurfaceWaveDialog", Viewer)
    dialog._preview_surface()
    assert len(opened) == 1, dialog.error.text()
    draft = opened[0]
    assert draft is not state
    assert draft.electron_gun is not state.electron_gun
    assert draft.objective_lens.percent == state.objective_lens.percent
    assert draft.objective_lens.upper_b0_t == state.objective_lens.upper_b0_t
    assert draft.electron_gun.extractor.voltage_kv == 4.1
    assert draft.electron_gun.emitter.surface_model.shared_boundary
    assert draft.electron_gun.emitter.surface_model.emission.kinetic_mean_ev == .42
    assert draft._resolved_assembly is not None
    assert capture_instrument_snapshot(state).digest == before
    assert state.electron_gun.emitter.surface_model is None


def test_surface_status_reports_tip_phase_definition(state, qtbot):
    dialog = GunSourceDialog(state.electron_gun, instrument_state=state)
    qtbot.addWidget(dialog)
    dialog.surface_enabled.setChecked(True)
    assert "Tip phase is not defined" in dialog.surface_status.text()
    assert "Ray Diagram is available" in dialog.surface_status.text()
    dialog.surface_coherent.setChecked(True)
    assert "Ray Diagram" in dialog.surface_status.text()
    assert "TEM/STEM" in dialog.surface_status.text()
    assert "geometric launch" in dialog.surface_status.text()
    assert "under development" in dialog.surface_status.text()


def test_tip_editor_can_enable_phase_and_apply_all_emission_parameters(state, qtbot):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QDialogButtonBox, QScrollArea, QStyle, QStyleOptionButton
    from temsim.optics.electron_gun.tip_edit import candidate_tip_edit

    emitter = state.electron_gun.emitter
    emitter.coherence = None
    before = capture_instrument_snapshot(state).digest
    dialog = GunSourceDialog(state.electron_gun, instrument_state=state)
    qtbot.addWidget(dialog)
    dialog.show()
    assert dialog.windowTitle() == "Physical Layout · Tip parameters"
    assert not dialog.coherence_enabled.isChecked()
    assert dialog.coherence_enabled.isVisibleTo(dialog)
    assert dialog.coherence_description.isHidden()
    assert all(edit.isHidden() for edit in dialog.coherence_inputs.values())

    QApplication.processEvents()
    dialog.findChild(QScrollArea).ensureWidgetVisible(dialog.coherence_enabled)
    QApplication.processEvents()
    option = QStyleOptionButton()
    dialog.coherence_enabled.initStyleOption(option)
    indicator = dialog.coherence_enabled.style().subElementRect(
        QStyle.SubElement.SE_CheckBoxIndicator, option, dialog.coherence_enabled)
    qtbot.mouseClick(dialog.coherence_enabled, Qt.MouseButton.LeftButton, pos=indicator.center())
    assert dialog.coherence_enabled.isChecked()
    assert dialog.coherence_description.isVisibleTo(dialog)
    assert all(edit.isVisibleTo(dialog) for edit in dialog.coherence_inputs.values())
    assert not dialog.inputs["angular_rms_mrad"].isEnabled()
    assert not dialog.inputs["angular_cutoff_mrad"].isEnabled()

    # Bounded editor fixture: this checks publication, not propagation accuracy.
    emission = {
        "virtual_source_fwhm_nm": 100.0,
        "emission_current_na": 123.0,
        "emission_energy_ev": 0.3,
        "minimum_kinetic_energy_ev": 0.01,
        "energy_spread_fwhm_ev": 0.1,
    }
    phase = {
        "incoherent_angle_rms_mrad": 0.1,
        "curvature_x_m1": 1000.0,
        "curvature_xy_m1": 5.0,
        "curvature_y_m1": 900.0,
        "offset_x_nm": 1.0,
        "offset_y_nm": -2.0,
        "tilt_x_mrad": 0.2,
        "tilt_y_mrad": -0.3,
    }
    for key, value in emission.items():
        dialog.inputs[key].setText(str(value))
    for key, value in phase.items():
        dialog.coherence_inputs[key].setText(str(value))
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Apply).click()
    assert dialog.result() == dialog.DialogCode.Accepted, dialog.error.text()

    candidate = candidate_tip_edit(state.electron_gun, dialog.value())
    assert candidate.emitter.surface_model is None
    for key, value in emission.items():
        assert getattr(candidate.emitter, key) == value
    for key, value in phase.items():
        assert getattr(candidate.emitter.coherence, key) == value
    assert capture_instrument_snapshot(state).digest == before


def test_tip_phase_selector_remains_available_when_disabled(state, qtbot):
    from temsim.optics.electron_gun.tip_coherence import TipCoherence

    state.electron_gun.emitter.virtual_source_fwhm_nm = 100.0
    state.electron_gun.emitter.coherence = TipCoherence()
    dialog = GunSourceDialog(state.electron_gun, instrument_state=state)
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.coherence_enabled.setChecked(False)
    assert dialog.coherence_enabled.isVisibleTo(dialog)
    assert all(edit.isHidden() for edit in dialog.coherence_inputs.values())
    assert dialog.inputs["angular_rms_mrad"].isEnabled()
    assert dialog.inputs["angular_cutoff_mrad"].isEnabled()
    dialog.coherence_enabled.setChecked(True)
    assert all(edit.isVisibleTo(dialog) for edit in dialog.coherence_inputs.values())


def test_existing_shared_gaussian_source_edits_use_the_same_domain_guard(state, qtbot):
    from temsim.optics.electron_gun.tip_coherence import TipCoherence
    emitter = state.electron_gun.emitter
    emitter.virtual_source_fwhm_nm = 100.
    emitter.coherence = TipCoherence()
    before = capture_instrument_snapshot(state).digest
    dialog = GunSourceDialog(state.electron_gun, instrument_state=state)
    qtbot.addWidget(dialog)
    dialog.inputs["virtual_source_fwhm_nm"].setText("5")
    dialog.accept()
    with pytest.raises(ValueError, match="No tip emission values"):
        dialog.value()
    assert "source" in dialog.error.text().lower()
    assert capture_instrument_snapshot(state).digest == before
    dialog.inputs["virtual_source_fwhm_nm"].setText("100")
    dialog.accept()
    assert dialog.value()["virtual_source_fwhm_nm"] == 100.


def test_transport_matching_is_an_explicit_curved_particle_edit_choice(state, qtbot):
    dialog = GunSourceDialog(state.electron_gun, instrument_state=state)
    qtbot.addWidget(dialog)
    assert not dialog.match_transport_requested
    assert "Flat tip at 0" in dialog.model_change_summary.text()
    assert "uncalibrated" in dialog.model_change_summary.toolTip()
    dialog.surface_enabled.setChecked(True)
    assert dialog.match_transport_requested
    assert "electrode fields follow its geometry" in dialog.model_change_summary.text()
    dialog.match_transport.setChecked(False)
    assert not dialog.match_transport_requested
    dialog.match_transport.setChecked(True)
    dialog.surface_coherent.setChecked(True)
    assert not dialog.match_transport_requested


@pytest.mark.parametrize("coherent", [False, True])
@pytest.mark.parametrize("quality", ["Preview", "Medium", "High accuracy"])
def test_calculation_snapshots_preserve_applied_surface_selection(state, qtbot, coherent, quality):
    from temsim.gui.calculation_controller import CalculationController
    dialog = GunSourceDialog(state.electron_gun, instrument_state=state)
    qtbot.addWidget(dialog)
    dialog.surface_enabled.setChecked(True)
    dialog.surface_coherent.setChecked(coherent)
    dialog.accept()
    for key, value in dialog.value().items():
        setattr(state.electron_gun.emitter, key, value)
    before = capture_instrument_snapshot(state).digest
    snapshot = CalculationController._calculation_snapshot(state, quality, 9, .2)
    assert snapshot.electron_gun.emitter.surface_model == state.electron_gun.emitter.surface_model
    assert capture_instrument_snapshot(state).digest == before


def test_shared_tip_selection_owns_energy_and_explains_geometric_rays(state, qtbot):
    from temsim.optics.electron_gun.tip_surface import SharedSurfaceCoherence
    before = capture_instrument_snapshot(state).digest
    dialog = GunSourceDialog(state.electron_gun, instrument_state=state)
    qtbot.addWidget(dialog)
    dialog.surface_enabled.setChecked(True)
    dialog.surface_inputs["normal_mean_energy_ev"].setText("0.4")
    dialog.surface_inputs["tangential_mean_energy_ev"].setText("0.12")
    dialog.surface_coherent.setChecked(True)
    # Independent moments of two positive exponential energy components.
    assert float(dialog.surface_inputs["kinetic_mean_ev"].text()) == pytest.approx(.52)
    assert float(dialog.surface_inputs["kinetic_sigma_ev"].text()) == pytest.approx(math.hypot(.4, .12))
    assert dialog.quantum_inputs["mean_energy_ev"].isHidden()
    assert dialog.quantum_inputs["energy_rms_ev"].isHidden()
    assert dialog.quantum_inputs["edge_phase_rad"].isReadOnly()
    assert "geometric-optics approximation" in dialog.surface_derived.toolTip()
    assert "reflection" in dialog.surface_status.text() or "reflected" in dialog.surface_status.text()
    dialog.surface_inputs["kinetic_mean_ev"].setText("0.42")
    dialog.surface_inputs["kinetic_sigma_ev"].setText("0.07")
    dialog.surface_inputs["normal_mean_energy_ev"].setText("invalid inactive value")
    dialog.accept()
    model = dialog.value()["surface_model"]
    assert isinstance(model.coherence, SharedSurfaceCoherence)
    assert model.coherence.edge_phase_rad == 0
    assert model.emission.flux_profile == "cosine_cap"
    assert model.emission.maximum_angle_deg == 0
    assert model.mean_energy_ev == .42
    assert model.energy_sigma_ev == .07
    assert capture_instrument_snapshot(state).digest == before


def test_historical_reservoir_opens_read_only_without_conversion(state, qtbot):
    from temsim.optics.electron_gun.tip_surface import SurfaceCoherence, load_tip_surface_reference
    saved = replace(load_tip_surface_reference(), coherence=SurfaceCoherence(
        mean_energy_ev=.42, energy_rms_ev=.08, edge_phase_rad=.6))
    state.electron_gun.emitter.surface_model = saved
    before = capture_instrument_snapshot(state).digest
    dialog = GunSourceDialog(state.electron_gun, instrument_state=state)
    qtbot.addWidget(dialog)
    assert dialog.surface_coherent.isChecked()
    assert not dialog.surface_coherent.isEnabled()
    assert "read-only" in dialog.surface_status.text()
    assert all(edit.isReadOnly() for edit in dialog.surface_inputs.values())
    assert all(edit.isReadOnly() for edit in dialog.quantum_inputs.values())
    assert not dialog.dimensions_button.isEnabled()
    # Even a programmatic edit of a read-only historical readout is ignored.
    dialog.quantum_inputs["mean_energy_ev"].setText("7")
    dialog.surface_inputs["current_na"].setText("invalid")
    dialog.accept()
    assert dialog.value()["surface_model"] is saved
    assert capture_instrument_snapshot(state).digest == before


def test_historical_reservoir_replacement_is_explicit_and_keeps_tip_energy(state, qtbot):
    from temsim.optics.electron_gun.tip_surface import SharedSurfaceCoherence, SurfaceCoherence, load_tip_surface_reference
    saved = replace(load_tip_surface_reference(), coherence=SurfaceCoherence(
        mean_energy_ev=.42, energy_rms_ev=0., edge_phase_rad=.6))
    state.electron_gun.emitter.surface_model = saved
    before = capture_instrument_snapshot(state).digest
    dialog = GunSourceDialog(state.electron_gun, instrument_state=state)
    qtbot.addWidget(dialog)
    dialog.replace_reservoir_button.click()
    assert not dialog.surface_inputs["kinetic_mean_ev"].isReadOnly()
    assert dialog.energy_law.currentData() == "monoenergetic"
    assert float(dialog.surface_inputs["kinetic_mean_ev"].text()) == .42
    assert float(dialog.quantum_inputs["edge_phase_rad"].text()) == 0
    dialog.accept()
    model = dialog.value()["surface_model"]
    assert isinstance(model.coherence, SharedSurfaceCoherence)
    assert model.geometry == saved.geometry
    assert model.mean_energy_ev == .42
    assert model.energy_sigma_ev == 0
    assert capture_instrument_snapshot(state).digest == before


def test_shared_tip_phase_gradient_is_rejected_without_mutating_live_state(state, qtbot):
    before = capture_instrument_snapshot(state).digest
    dialog = GunSourceDialog(state.electron_gun, instrument_state=state)
    qtbot.addWidget(dialog)
    dialog.surface_enabled.setChecked(True)
    dialog.surface_coherent.setChecked(True)
    dialog.quantum_inputs["edge_phase_rad"].setText("0.2")
    dialog.accept()
    with pytest.raises(ValueError, match="No tip emission values"):
        dialog.value()
    assert "only constant surface phase" in dialog.error.text()
    assert capture_instrument_snapshot(state).digest == before
