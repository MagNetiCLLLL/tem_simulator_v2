"""Current widget edits commit validated filter controls without mode resets."""
from dataclasses import replace
import pytest
from temsim.assembly_catalog import AssemblyCatalog
from temsim.optics.column import default_state
from temsim.gui.parameter_panel import ParameterPanel
from temsim.runtime_parameters import runtime_targets


@pytest.fixture
def controls(qtbot):
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, replace(catalog.default_selection(), recording="Energy Filter"))
    ef = state.energy_filter
    ef.multi_eels_enabled = True
    ef.multi_eels_region_count = 5
    panel = ParameterPanel()
    qtbot.addWidget(panel)
    target = runtime_targets(state)["energy_filter"]
    panel.set_context(target.label, target, None, (), None)
    return panel, ef


def values(ef):
    return (ef.operating_mode, ef.multi_eels_enabled, ef.multi_eels_region_count,
            ef.selected_loss_ev, ef.slit_width_ev, ef.energy_slit.gap_m,
            ef.energy_slit.centre_m, ef.fast_shutter.open, ef.bias_tube.offset_ev,
            ef.camera_deflector.active_strip, ef.zebra_detector.alignment_mode,
            ef.energy_slit.inserted, ef.output_detector_inserted,
            ef.zebra_detector.inserted, ef.bias_tube.enabled)


def test_ordinary_edits_preserve_explicit_detector_and_physical_blade_states(controls):
    panel, ef = controls
    children = tuple(getattr(ef, n) for n in ("energy_slit", "bias_tube", "camera_deflector", "zebra_detector", "fast_shutter"))
    panel.energy_filter_shutter.setChecked(False)
    panel.energy_filter_bias.setValue(87.)
    panel.energy_filter_active_strip.setValue(4)
    panel.energy_filter_selected_loss.setValue(120.)
    assert not ef.fast_shutter.open
    assert ef.bias_tube.offset_ev == 87.
    assert ef.camera_deflector.active_strip == 4
    assert ef.energy_slit.derived_centre_loss_ev == pytest.approx(120.)
    # Physical blade adjustments and insertion settings are independent state.
    ef.energy_slit.gap_m = 17e-6
    ef.energy_slit.inserted = True
    ef.zebra_detector.inserted = False
    panel.energy_filter_alignment.setChecked(True)
    assert ef.energy_slit.gap_m == 17e-6
    assert ef.energy_slit.inserted and not ef.zebra_detector.inserted
    assert not ef.fast_shutter.open and ef.bias_tube.offset_ev == 87.
    assert ef.camera_deflector.active_strip == 4
    assert all(old is getattr(ef, name) for old, name in zip(children,
        ("energy_slit", "bias_tube", "camera_deflector", "zebra_detector", "fast_shutter")))


def test_explicit_mode_switch_updates_participation_but_preserves_user_values(controls):
    panel, ef = controls
    panel.energy_filter_shutter.setChecked(False)
    panel.energy_filter_bias.setValue(80.)
    panel.energy_filter_active_strip.setValue(4)
    panel.energy_filter_mode.setCurrentIndex(panel.energy_filter_mode.findData("eftem"))
    assert ef.energy_slit.inserted and ef.output_detector_inserted
    assert not ef.zebra_detector.inserted and not ef.fast_shutter.enabled
    panel.energy_filter_slit_width.setValue(15.)
    assert ef.energy_slit.derived_width_ev == pytest.approx(15.)
    panel.energy_filter_mode.setCurrentIndex(panel.energy_filter_mode.findData("eels"))
    assert not ef.energy_slit.inserted and not ef.output_detector_inserted
    assert ef.zebra_detector.inserted and ef.fast_shutter.enabled and ef.bias_tube.enabled
    assert not ef.fast_shutter.open and ef.bias_tube.offset_ev == 80.
    assert ef.camera_deflector.active_strip == 4
    assert ef.slit_width_ev == 15.


@pytest.mark.parametrize("control", ["bias", "window"])
def test_rejected_filter_edits_do_not_partly_mutate_any_live_component(controls, control):
    panel, ef = controls
    failures, accepted = [], []
    panel.error.connect(failures.append)
    panel.runtime_changed.connect(accepted.append)
    before = values(ef)
    if control == "bias":
        panel.energy_filter_bias.setMaximum(ef.bias_tube.maximum_abs_offset_ev*2)
        panel.energy_filter_bias.setValue(ef.bias_tube.maximum_abs_offset_ev*1.5)
    else:
        panel.energy_filter_slit_width.setMaximum(1e9)
        panel.energy_filter_slit_width.setValue(1e8)
    assert failures and not accepted
    assert values(ef) == before


def test_unrelated_widget_edit_does_not_round_full_precision_saved_values(controls):
    panel, ef = controls
    ef.selected_loss_ev = 123.4567890123
    ef.slit_width_ev = 15.987654321
    ef.energy_slit.gap_m = 17.123456789e-6
    ef.energy_slit.centre_m = 40.123456789e-6
    panel.set_context(panel._runtime_target.label, panel._runtime_target, None, (), None)
    before = (ef.selected_loss_ev, ef.slit_width_ev, ef.energy_slit.gap_m, ef.energy_slit.centre_m)
    panel.energy_filter_bias.setValue(87.)
    assert (ef.selected_loss_ev, ef.slit_width_ev, ef.energy_slit.gap_m, ef.energy_slit.centre_m) == before
