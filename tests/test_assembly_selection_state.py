"""Assembly state is explicit; configuring and displaying it are separate actions."""
from dataclasses import replace

import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.gui.assembly_panel import AssemblyPanel


@pytest.fixture
def panel(qtbot):
    catalog = AssemblyCatalog()
    panel = AssemblyPanel(catalog, catalog.default_selection())
    qtbot.addWidget(panel)
    return panel


def test_selection_and_summary_update_without_requesting_configuration(panel):
    requests = []
    panel.configuration_requested.connect(lambda: requests.append(True))
    selection = replace(panel.current_selection(), column="C3", gun="FEG + Mono",
                        recording="No Energy Filter")
    panel.set_selection(selection)
    assert panel.current_selection() == selection
    assert "Monochromator" in panel.configuration_summary.text()
    assert "Probe corrector" not in panel.configuration_summary.text()
    assert "Energy filter" not in panel.configuration_summary.text()
    assert not requests
    panel.configure_button.click()
    assert requests == [True]


def test_reloading_catalog_keeps_selected_hardware_and_compatible_modes(panel):
    selection = replace(panel.current_selection(), gun="FEG + Mono", recording="No Energy Filter")
    catalog = AssemblyCatalog()
    panel.reload_catalog(catalog, selection)
    assert panel.catalog is catalog
    assert panel.current_selection() == selection
    assert panel.probe_mode.currentData()
    assert panel.projector_mode.currentData()


def test_catalog_reload_does_not_retain_presets_for_an_unsupported_column(panel):
    selection = replace(panel.current_selection(), column="C3")
    panel.reload_catalog(AssemblyCatalog(), selection)
    assert panel.current_selection() == selection
    assert panel.probe_mode.count() == 0
    assert not panel.apply_operating_mode_button.isEnabled()


@pytest.mark.parametrize("reload", [False, True])
def test_invalid_selection_does_not_partially_replace_displayed_state(panel, reload):
    before, summary, catalog = panel.current_selection(), panel.configuration_summary.text(), panel.catalog
    invalid = replace(before, gun="unknown gun")
    with pytest.raises(ValueError):
        if reload:
            panel.reload_catalog(AssemblyCatalog(), invalid)
        else:
            panel.set_selection(invalid)
    assert panel.current_selection() == before
    assert panel.configuration_summary.text() == summary
    assert panel.catalog is catalog
