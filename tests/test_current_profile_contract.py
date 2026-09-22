"""Current profiles reject obsolete inputs without implicit model changes."""
from copy import deepcopy
import pytest
import tomli_w

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.optics.column import default_state
from temsim.profile_io import PROFILE_FORMAT_VERSION, apply_profile_values, read_profile, save_profile


@pytest.mark.parametrize("version", [0, 1, 4, 9, 10, 11, True, 12.0, "12"])
def test_only_exact_current_profile_version_is_accepted(tmp_path, version):
    path = tmp_path / "profile.toml"
    path.write_text(tomli_w.dumps({"format_version": version}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported operating-profile format"):
        read_profile(path)


@pytest.mark.parametrize("bad", [{"old_device": {"value": 2}},
    {"sample": {"eds_elastic_trajectory_count": 50}},
    {"sample": {"atomic_structure_source": "preset"}},
    {"simulation": {"old_field": 1}}, {"__profile_format_version__": 10}])
def test_unknown_and_retired_fields_are_transactionally_rejected(bad):
    state = default_state()
    before = deepcopy(state.to_dict())
    with pytest.raises(ValueError):
        apply_profile_values(state, {"objective_lens": {"percent": 12.5}, **bad})
    assert state.to_dict() == before


def test_partial_current_assignment_preserves_unrelated_settings():
    state = default_state()
    state.vacuum_map.enabled = True
    state.electron_gun.electrostatic_lens.voltage_reference = "tip"
    before = deepcopy(state.to_dict())
    assert apply_profile_values(state, {"objective_lens": {"percent": 12.5}}) is None
    assert state.objective_lens.percent == 12.5
    assert state.vacuum_map.enabled
    assert state.electron_gun.electrostatic_lens.voltage_reference == "tip"
    assert state.simulation_mode == before["simulation_mode"]


@pytest.mark.parametrize("gun", ["FEG", "FEG + Mono", "Thermionic"])
def test_current_profile_roundtrip_for_each_gun(tmp_path, gun):
    catalog = AssemblyCatalog()
    selection = AssemblySelection(gun, "C3 + Probe Corrector", "No Energy Filter")
    state = default_state()
    catalog.apply(state, selection)
    state.objective_lens.percent = 12.5
    path = tmp_path / "current.toml"
    save_profile(path, state, selection)
    selected, values = read_profile(path)
    fresh = default_state()
    catalog.apply(fresh, selected)
    apply_profile_values(fresh, values)
    assert selected == selection
    assert fresh.objective_lens.percent == 12.5


def test_active_lens_controls_have_one_owner_and_conflicts_are_rejected():
    state = default_state()
    key = state.objective_lens.key
    before = deepcopy(state.to_dict())
    with pytest.raises(ValueError, match="belongs only in profile devices"):
        apply_profile_values(state, {key: {"percent": 25}, "__simulation_model__": {
            "mode": state.simulation_mode, "settings": {
                "lens_excitation": {key: {"percent": 30, "polarity": 1}}}}})
    assert state.to_dict() == before


def test_partial_file_does_not_inject_missing_source_or_model_tables(tmp_path):
    from dataclasses import asdict
    path = tmp_path / "partial.toml"
    path.write_text(tomli_w.dumps({"format_version": PROFILE_FORMAT_VERSION,
        "assembly": asdict(AssemblyCatalog().default_selection()),
        "devices": {"objective_lens": {"percent": 25}}}), encoding="utf-8")
    _, values = read_profile(path)
    assert set(values) == {"__profile_format_version__", "objective_lens"}


def test_scan_and_wobble_are_validated_together_before_commit():
    state = default_state()
    before = deepcopy(state.to_dict())
    with pytest.raises(ValueError):
        apply_profile_values(state, {state.ac_deflector.key: {
            "scan_enabled": True, "wobble_enabled": True}})
    assert state.to_dict() == before


def test_save_refuses_mismatched_assembly_without_overwriting(tmp_path):
    from dataclasses import replace
    state = default_state()
    catalog = AssemblyCatalog()
    selection = catalog.default_selection()
    catalog.apply(state, selection)
    path = tmp_path / "existing.toml"
    path.write_text("existing input", encoding="utf-8")
    before = deepcopy(state.to_dict())
    with pytest.raises(ValueError, match="assembly selection"):
        save_profile(path, state, replace(selection, recording="Energy Filter"))
    assert path.read_text(encoding="utf-8") == "existing input"
    assert state.to_dict() == before


def test_save_unresolved_state_reports_contract_error(tmp_path):
    state = default_state()
    vars(state).pop("_resolved_assembly", None)
    with pytest.raises(ValueError, match="resolved instrument"):
        save_profile(tmp_path / "invalid.toml", state, AssemblyCatalog().default_selection())


def _filter_state():
    from dataclasses import replace
    state = default_state()
    catalog = AssemblyCatalog()
    selection = replace(catalog.default_selection(), recording="Energy Filter")
    catalog.apply(state, selection)
    return state, selection


def test_partial_filter_window_moves_physical_slit_atomically():
    state, _ = _filter_state()
    slit = state.energy_filter.energy_slit
    aperture = state.energy_filter._entrance_aperture_component
    apply_profile_values(state, {slit.key: {"requested_width_ev": 18.0, "requested_centre_loss_ev": 20.0}})
    assert state.energy_filter._entrance_aperture_component is aperture
    assert slit.derived_width_ev == pytest.approx(18.0)
    assert slit.derived_centre_loss_ev == pytest.approx(20.0)
    before = deepcopy(state.to_dict())
    with pytest.raises(ValueError, match="physical slit travel"):
        apply_profile_values(state, {"objective_lens": {"percent": 25.0},
                                    slit.key: {"requested_width_ev": 1e9}})
    assert state.to_dict() == before


def test_filter_profile_preserves_physical_controls_without_mode_reset(tmp_path):
    state, selection = _filter_state()
    ef = state.energy_filter
    ef.fast_shutter.open = False
    ef.fast_shutter.enabled = False
    ef.bias_tube.enabled = False
    ef.bias_tube.offset_ev = 120.0
    ef.multi_eels_enabled = True
    ef.multi_eels_region_count = 4
    ef.camera_deflector.active_strip = 3
    ef.camera_deflector.enabled = False
    ef.energy_slit.inserted = True
    ef.energy_slit.configure_energy_window(23.0, 17.0)
    ef.energy_slit.gap_m *= 0.75
    ef.energy_slit.centre_m += 1e-6
    from temsim.optics.energy_filter import serialise_energy_filter
    before = serialise_energy_filter(ef)
    path = tmp_path / "filter.toml"
    save_profile(path, state, selection)
    selected, values = read_profile(path)
    assert "slit_width_ev" not in values["energy_filter"]
    assert "selected_loss_ev" not in values["energy_filter"]
    fresh = default_state()
    AssemblyCatalog().apply(fresh, selected)
    apply_profile_values(fresh, values)
    after = serialise_energy_filter(fresh.energy_filter)
    assert after == before


def test_partial_mode_request_applies_physical_mode_controls():
    state, _ = _filter_state()
    apply_profile_values(state, {"energy_filter": {"operating_mode": "eftem"}})
    assert state.energy_filter.energy_slit.inserted
    assert state.energy_filter.output_detector_inserted
    assert not state.energy_filter.zebra_detector.inserted
