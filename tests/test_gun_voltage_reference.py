"""Electrode gauges, cache identity and historical voltage preservation."""
import json

import pytest

from temsim.optics.column import default_state
from temsim.physics.grounded_tip_field import field_request


@pytest.fixture
def surface_state():
    from temsim import module_manifest
    from temsim.optics.electron_gun.tip_assembly import model_from_part
    state = default_state()
    state.electron_gun.emitter.surface_model = model_from_part(
        module_manifest.part_data("gun/FEG.toml", "feg_tip"))
    assert state.electron_gun.emitter.surface_model.coherence is None
    return state


@pytest.mark.parametrize("reference,rise,ground",[
    ("tip",1100.,-298900.),("extractor",5600.,-294400.),("ground",301100.,1100.)])
def test_electrode_voltage_reference_is_not_a_focusing_multiplier(surface_state,reference,rise,ground):
    gun = surface_state.electron_gun
    gun.extractor.voltage_kv = 4.5
    gun.electrostatic_lens.voltage_kv = 1.1
    gun.electrostatic_lens.voltage_reference = reference
    request = field_request(gun)
    assert request["rings"][1][-1] == pytest.approx(rise)
    assert request["rings"][1][-1]-request["high_tension_v"] == pytest.approx(ground)
    assert request["gun_lens_voltage_reference"] == reference
    assert request["rings"][-1][-1] == 300000.


def test_invalid_reference_fails_before_field_solve(surface_state):
    from temsim.runtime_parameters import runtime_targets, validate_runtime_assignment
    state = surface_state
    lens = state.electron_gun.electrostatic_lens
    target = runtime_targets(state)[lens.key]
    with pytest.raises(ValueError,match="reference"):
        validate_runtime_assignment(target,"voltage_reference","arbitrary")
    lens.voltage_reference = "arbitrary"
    with pytest.raises(ValueError,match="reference"):
        field_request(state.electron_gun)


def test_cache_and_saved_gun_require_reference_without_injected_defaults(surface_state):
    from temsim.optics.electron_gun.field_emission import field_emission_gun_from_dict
    gun = surface_state.electron_gun
    before = gun._cache_key(9)
    gun.electrostatic_lens.voltage_reference = "tip"
    assert gun._cache_key(9) != before
    saved = gun.to_dict()
    assert field_emission_gun_from_dict(saved).electrostatic_lens.voltage_reference == "tip"
    saved["components"][gun.electrostatic_lens.key].pop("voltage_reference")
    with pytest.raises(ValueError, match="requires explicit voltage_reference"):
        field_emission_gun_from_dict(saved)


def test_snapshot_rejects_missing_voltage_reference_without_injecting_default():
    from temsim.instrument_snapshot import encode_instrument, decode_instrument
    from temsim.immutable_json import thaw_json
    state = default_state()
    graph = thaw_json(encode_instrument(state))
    node = next(n for n in graph["nodes"] if n["type"].endswith("ElectrostaticGunLens"))
    node["fields"].remove("voltage_reference")
    node["attributes"].pop("voltage_reference")
    before = json.dumps(graph,sort_keys=True)
    with pytest.raises(ValueError, match="Unsupported model schema"):
        decode_instrument(graph)
    assert json.dumps(graph,sort_keys=True) == before


def test_profile_round_trip_and_omitted_reference_is_unchanged(tmp_path, surface_state):
    from temsim.profile_io import save_profile,read_profile,apply_profile_values
    from temsim.assembly_catalog import AssemblyCatalog
    state = surface_state
    lens = state.electron_gun.electrostatic_lens
    lens.voltage_reference = "tip"
    path = tmp_path/"gun.toml"
    save_profile(path,state,AssemblyCatalog().default_selection())
    _,values = read_profile(path)
    other = default_state()
    assert apply_profile_values(other,values) is None
    assert other.electron_gun.electrostatic_lens.voltage_reference == "tip"
    other.electron_gun.validate()
    values[lens.key].pop("voltage_reference")
    assert apply_profile_values(other,values) is None
    assert other.electron_gun.electrostatic_lens.voltage_reference == "tip"
    assert "voltage_reference" not in values[lens.key]
