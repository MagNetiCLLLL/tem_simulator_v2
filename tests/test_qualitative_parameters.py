"""Units and causal explanations for the first classical qualification batch."""
from copy import deepcopy

import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.parameter_registry import runtime_definition
from temsim.parameter_semantics import parameter_unit
from temsim.runtime_parameters import runtime_targets, editable_parameters


@pytest.mark.parametrize("name,unit", [
    ("emission_current_na", "nA"), ("b0_t", "T"),
    ("gradient_t_per_m", "T/m"), ("electric_field_v_per_m", "V/m"),
    ("electric_quadrupole_gradient_v_per_m2", "V/m²"),
    ("curvature_nm_inv", "nm⁻¹"), ("vacuum_pa", "Pa"),
    ("maximum_strength_m2", "m⁻²"), ("maximum_strength_m3", "m⁻³"),
    ("strength_m2", "m⁻²"), ("strength_m3", "m⁻³"), ("orientation_rad", "rad"),
    ("offset_x_mm", "mm"), ("energy_spread_fwhm_ev", "eV"),
])
def test_physical_units_are_not_dimensionless_or_truncated(name, unit):
    assert parameter_unit(("runtime", "component", name)) == unit


def test_primary_gun_controls_have_explanations_without_executing_physics(monkeypatch):
    from temsim.optics.electron_gun.field_emission import FieldEmissionGun
    state = default_state()
    before = deepcopy(state.to_dict())
    def forbidden(*args, **kwargs):
        pytest.fail("Parameter inspection must not run transport")
    monkeypatch.setattr(FieldEmissionGun, "trace_to_exit", forbidden)
    targets = runtime_targets(state)
    for key in ("electron_gun", "feg_tip", "feg_extractor", "feg_electrostatic_lens",
                "feg_accelerator", "feg_dpa_aperture", "feg_c1_aperture"):
        for parameter in editable_parameters(targets[key]):
            definition = runtime_definition(targets[key], parameter.name)
            assert definition is not None, (key, parameter.name)
            assert definition.description
    assert state.to_dict() == before


def test_source_metadata_is_not_advertised_as_an_active_physical_control():
    state = default_state()
    target = runtime_targets(state)["feg_tip"]
    original = target.obj.emit(49)
    for name, value in {"work_function_ev": 5., "vacuum_pa": 1e-6,
                        "tip_radius_nm": 150., "tip_cone_half_angle_deg": 7.,
                        "emitter_material": "user material"}.items():
        definition = runtime_definition(target, name)
        assert definition.category == "metadata"
        assert "Metadata only" in definition.tooltip()
        assert not definition.sweep_eligible
        setattr(target.obj, name, value)
    edited = target.obj.emit(49)
    for name in ("x_m", "y_m", "tx_rad", "ty_rad", "energy_offset_ev", "weight"):
        np.testing.assert_array_equal(getattr(edited, name), getattr(original, name))


def test_units_and_conditions_reach_existing_parameter_panel(qtbot):
    from temsim.gui.parameter_panel import ParameterPanel
    from temsim.gui.gun_source_dialog import GunSourceDialog
    state = default_state()
    panel = ParameterPanel()
    qtbot.addWidget(panel)
    target = runtime_targets(state)["feg_extractor"]
    panel.set_context(target.label, target, None, [], None)
    row = next(i for i in range(panel.runtime_table.rowCount())
               if panel.runtime_table.item(i, 0).text() == "voltage_kv")
    item = panel.runtime_table.item(row, 1)
    panel.runtime_table.setCurrentItem(item)
    assert "kV" in item.toolTip()
    assert "fixed final potential" in item.toolTip()
    assert "kV" in panel.parameter_details.text()
    dialog = GunSourceDialog(state.electron_gun)
    qtbot.addWidget(dialog)
    assert "nA" in dialog.inputs["emission_current_na"].toolTip()
    assert "space charge" in dialog.inputs["emission_current_na"].toolTip()


def test_editable_stigmator_multipole_and_scan_controls_have_causal_explanations():
    from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
    state = default_state()
    AssemblyCatalog().apply(state, AssemblySelection("FEG", "C3 + Probe Corrector + Image Corrector", "No Energy Filter"))
    targets = runtime_targets(state)
    before = deepcopy(state.to_dict())
    selected = {"condenser_stigmator", "objective_stigmator", "diffraction_stigmator", "ac_deflector", "descan_deflector"}
    selected.update(c.key for c in state.corrector_elements if c.kind in {"hexapole", "quadrupole"})
    for key in selected:
        for parameter in editable_parameters(targets[key]):
            definition = runtime_definition(targets[key], parameter.name)
            assert definition is not None, (key, parameter.name)
            assert definition.description and not definition.sweep_eligible
    assert state.to_dict() == before
    assert "inactive in ideal mode" in runtime_definition(targets["probe_hp2_hexapole"], "strength_m3").tooltip()
    for key in ("ac_deflector", "descan_deflector"):
        assert runtime_definition(targets[key], "scan_amplitude_x_mrad").category == "metadata"
        assert runtime_definition(targets[key], "effective_thickness_mm").category == "metadata"


def test_scan_and_corrector_explanations_reach_both_existing_editors(qtbot):
    from temsim.gui.parameter_panel import ParameterPanel
    from temsim.gui.scan_panel import ScanControlView
    state = default_state()
    panel = ParameterPanel()
    qtbot.addWidget(panel)
    target = runtime_targets(state)["probe_hp2_hexapole"]
    panel.set_context(target.label, target, None, [], None)
    row = next(i for i in range(panel.runtime_table.rowCount())
               if panel.runtime_table.item(i, 0).text() == "strength_m3")
    item = panel.runtime_table.item(row, 1)
    assert "m⁻³" in item.toolTip() and "quadratic" in item.toolTip()
    scan = ScanControlView()
    qtbot.addWidget(scan)
    assert "(N-1)" in scan.ac_controls["scan_pixel_size_nm"].toolTip()
    assert "normalized raster phase" in scan.descan_controls["scan_frame_period_s"].toolTip()
    assert "full 2x2 matrix" in scan.ac_controls["lower_coil_gain"].toolTip()


def test_detector_controls_have_shared_physical_and_readout_meanings(qtbot):
    from temsim.gui.parameter_panel import ParameterPanel
    from temsim.parameter_registry import parameter_definition
    state = default_state()
    before = deepcopy(state.to_dict())
    targets = runtime_targets(state)
    for detector in state.recording_planes:
        for parameter in editable_parameters(targets[detector.key]):
            definition = runtime_definition(targets[detector.key], parameter.name)
            assert definition is not None, (detector.key, parameter.name)
            assert not definition.sweep_eligible
    panel = ParameterPanel()
    qtbot.addWidget(panel)
    target = targets['bf']
    panel.set_context(target.label, target, None, [], None)
    row = next(i for i in range(panel.runtime_table.rowCount())
               if panel.runtime_table.item(i, 0).text() == 'readout_enabled')
    assert 'still intercepts' in panel.runtime_table.item(row, 1).toolTip()
    assert parameter_definition('sample', 'stem_fourdstem_gain_counts_per_electron').unit == 'counts/electron'
    assert parameter_definition('sample', 'stem_fourdstem_dark_electrons_per_pixel').unit == 'electrons/pixel/exposure'
    assert 'not DQE' in parameter_definition('sample', 'stem_fourdstem_quantum_efficiency').description
    assert state.to_dict() == before


def test_structural_detector_editor_shares_psf_meaning_without_reclassifying_it_as_material():
    from temsim.parameter_semantics import describe_parameter, CATEGORY_LABELS
    from temsim.parameter_registry import parameter_definition
    from temsim import module_manifest
    part = module_manifest.part_data('project_and_recording_system/NoEnergyFilter.toml', 'camera')
    for field, category in [('point_spread_sigma_x_mm', 'operating'),
                            ('point_spread_source', 'unknown'), ('outer_width_mm', 'physical')]:
        meaning = describe_parameter(part, ('parts', 'camera', field))
        assert meaning.category == category
        assert meaning.category_label == CATEGORY_LABELS[category]
        assert meaning.description == parameter_definition('camera', field).description
