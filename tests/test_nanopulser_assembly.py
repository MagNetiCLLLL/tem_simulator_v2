from dataclasses import replace
from pathlib import Path
import shutil

import pytest

from temsim import module_manifest
from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.column.layout import build_optics_layout
from temsim.column.state_layout import layout_configuration_from_state
from temsim.component_keys import NANOPULSER_APERTURE, NANOPULSER_DEFLECTOR
from temsim.manifest_editor import ManifestEditor, ManifestTarget
from temsim.optics.column import default_state
from temsim.paths import INSTRUMENT_CONFIG_ROOT


def test_optional_blanker_inserts_one_module_and_removal_restores_baseline():
    catalog = AssemblyCatalog()
    assert [option.name for option in catalog.beam_blankers] == ["None", "NanoPulser"]
    state = default_state()
    selection = catalog.default_selection()
    baseline = catalog.apply(state, selection)
    baseline_parts = {part.key: part for part in baseline.parts}
    assert selection.beam_blanker == "None"
    assert [module.type for module in baseline.modules] == [
        "gun", "column", "project_and_recording_system",
    ]
    assert "beam_blanker" not in catalog.selected_paths(selection)

    enabled = replace(selection, beam_blanker="NanoPulser")
    installed = catalog.apply(state, enabled)
    assert [module.type for module in installed.modules] == [
        "gun", "beam_blanker", "column", "project_and_recording_system",
    ]
    assert installed.selected_path("beam_blanker") == "beam_blanker/NanoPulser.toml"
    assert state.nanopulser.installed
    assert state.sample.z_mm == pytest.approx(1679.2)
    for part in installed.parts:
        if part.key not in baseline_parts:
            assert part.key in {NANOPULSER_DEFLECTOR, NANOPULSER_APERTURE}
            continue
        original = baseline_parts[part.key]
        shift = 0.0 if original.module_key.startswith("gun:") else 80.0
        assert part.center_z_mm == pytest.approx(original.center_z_mm + shift)
        assert part.length_mm == original.length_mm
    layout = {item.key: item for item in state._resolved_optics_layout}
    assert layout[NANOPULSER_DEFLECTOR].optical_reference_plane_z_mm == 470.0
    assert layout[NANOPULSER_APERTURE].optical_reference_plane_z_mm == 510.0
    assert layout[NANOPULSER_APERTURE].downstream_key == "condenser_lens_1"

    restored = catalog.apply(state, selection)
    assert not state.nanopulser.installed
    assert [part.center_z_mm for part in restored.parts] == [
        part.center_z_mm for part in baseline.parts
    ]
    assert NANOPULSER_DEFLECTOR not in {
        item.key for item in state._resolved_optics_layout
    }


@pytest.mark.parametrize("gun", ("FEG", "FEG + Mono", "Thermionic"))
def test_blanker_geometry_follows_the_selected_gun_exit(gun):
    catalog = AssemblyCatalog()
    state = default_state()
    selection = AssemblySelection(gun, "C3", "Energy Filter", "NanoPulser")
    catalog.apply(state, selection)
    exit_z = state.electron_gun.exit_plane_z_mm
    assert state.nanopulser.z_mm == pytest.approx(exit_z + 20.0)
    assert state.nanopulser.stop_z_mm == pytest.approx(exit_z + 60.0)
    assert state.nanopulser.z_mm < state.nanopulser.stop_z_mm < (
        state.condenser_lens_1.optical_reference_from_tip_mm
    )
    assert state.nanopulser.plate_length_mm == 10.0
    assert state.nanopulser.plate_gap_mm == 1.0
    assert state.nanopulser.aperture_radius_mm == 0.1


def test_blanker_selection_survives_legacy_recording_normalisation():
    catalog = AssemblyCatalog()
    selection = AssemblySelection("FEG", "C3", "No Energy Filter", "NanoPulser")
    normal = catalog.normalise_selection(selection)
    assert normal.recording == "Energy Filter"
    assert normal.beam_blanker == "NanoPulser"
    with pytest.raises(ValueError, match="Unknown assembly option"):
        catalog.normalise_selection(replace(selection, beam_blanker="unknown"))


def test_physical_layout_exposes_installed_electrostatic_blanker():
    state = default_state()
    state.nanopulser.installed = True
    configuration = layout_configuration_from_state(state)
    layout = build_optics_layout(configuration)
    by_key = {item.key: item for item in layout}
    assert by_key[NANOPULSER_DEFLECTOR].field_model.kind == "electrostatic_field"
    assert by_key[NANOPULSER_APERTURE].field_model.kind == "hard_aperture"
    assert by_key[NANOPULSER_APERTURE].effective_aperture_radius_mm == 0.1


def test_blanker_toml_geometry_rebuild_preserves_operating_settings(tmp_path: Path):
    root = tmp_path / "instruments"
    shutil.copytree(INSTRUMENT_CONFIG_ROOT, root)
    catalog = AssemblyCatalog(root)
    state = default_state()
    selection = replace(catalog.default_selection(), beam_blanker="NanoPulser")
    catalog.apply(state, selection)
    state.nanopulser.blanked = True
    state.nanopulser.voltage_v = 425.0
    editor = ManifestEditor(root)
    editor.save(
        ManifestTarget("beam_blanker/NanoPulser.toml"),
        {
            ("parts", NANOPULSER_DEFLECTOR, "plate_length_mm"): 8.0,
            ("parts", NANOPULSER_DEFLECTOR, "plate_gap_mm"): 0.8,
            ("parts", NANOPULSER_APERTURE, "aperture_radius_mm"): 0.08,
            ("parts", NANOPULSER_APERTURE, "bore_diameter_mm"): 0.16,
        },
        layout_configuration_from_state(state, assembly_root=root),
    )
    catalog.apply(state, selection)
    assert state.nanopulser.plate_length_mm == 8.0
    assert state.nanopulser.plate_gap_mm == 0.8
    assert state.nanopulser.aperture_radius_mm == 0.08
    assert state.nanopulser.voltage_v == 425.0
    assert state.nanopulser.blanked is True


def test_blanker_manifest_rejects_an_upstream_stop_and_oem_dimension_claim():
    path = INSTRUMENT_CONFIG_ROOT / "beam_blanker" / "NanoPulser.toml"
    document = module_manifest.read_document(path)
    stop = document["parts"][1]
    for field in (
        "local_start_z_mm", "local_center_z_mm", "local_end_z_mm",
        "optical_reference_local_z_mm",
    ):
        stop[field] -= 40.0
    with pytest.raises(ValueError, match="downstream"):
        module_manifest.validate_document(document)
    document = module_manifest.read_document(path)
    document["module"]["geometry_status"] = "manufacturer_documented"
    with pytest.raises(ValueError, match="non-OEM"):
        module_manifest.validate_document(document)
