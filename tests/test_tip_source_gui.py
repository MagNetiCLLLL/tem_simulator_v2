"""Source-editor routing and state ownership; not full-chain image validation."""
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
    dialog.quantum_inputs["mean_energy_ev"].setText("0.42")
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
    assert draft.electron_gun.emitter.surface_model.coherence.mean_energy_ev == .42
    assert draft._resolved_assembly is not None
    assert capture_instrument_snapshot(state).digest == before
    assert state.electron_gun.emitter.surface_model is None


def test_surface_status_distinguishes_ray_source_from_coherent_development(state, qtbot):
    dialog = GunSourceDialog(state.electron_gun, instrument_state=state)
    qtbot.addWidget(dialog)
    dialog.surface_enabled.setChecked(True)
    assert "Classical rays" in dialog.surface_status.text()
    assert "no coherent phase" in dialog.surface_status.text()
    dialog.surface_coherent.setChecked(True)
    assert "Ray Diagram" in dialog.surface_status.text()
    assert "TEM/STEM" in dialog.surface_status.text()
    assert "unavailable" in dialog.surface_status.text()


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
