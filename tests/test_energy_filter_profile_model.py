"""Profiles retain all executed filter excitation and independent references."""
from copy import deepcopy
from dataclasses import replace
import tomllib

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.component_keys import ENERGY_FILTER_MULTIPOLE_KEYS
from temsim.optics.column import default_state
from temsim.optics.energy_filter import serialise_energy_filter
from temsim.optics.energy_filter_m12 import serialise_energy_filter_m12
from temsim.profile_io import save_profile, read_profile, apply_profile_values
from temsim.runtime_parameters import editable_parameters, runtime_targets


MODEL_KEY = "__energy_filter_model__"


def configured_filter():
    state = default_state()
    catalog = AssemblyCatalog()
    selection = replace(catalog.default_selection(), recording="Energy Filter")
    catalog.apply(state, selection)
    for index, element in enumerate(state.energy_filter.multipoles, start=1):
        for order in range(1, 7):
            element.multipole_field.set_component(order, normal=index + .25 * order, skew=-index - .375 * order)
        element.enabled = index % 2 == 0
        element.field_backend.fringe_expansion_order = index % 3
        calibration = element.calibration
        calibration.reference_voltage_kv = 100. + index
        calibration.reference_normal_coefficients[:] = np.arange(1., 7.) * (index + .5)
        calibration.reference_skew_coefficients[:] = -np.arange(1., 7.) * (index + .75)
        calibration.normal_trim_coefficients[:] = np.arange(1., 7.) * .01 * index
        calibration.skew_trim_coefficients[:] = -np.arange(1., 7.) * .02 * index
    return state, selection


def model_for(state):
    return {"multipoles": [serialise_energy_filter_m12(element) for element in state.energy_filter.multipoles]}


def test_profile_preserves_ten_current_fields_and_independent_calibrations(tmp_path):
    state, selection = configured_filter()
    before = serialise_energy_filter(state.energy_filter)
    path = tmp_path / "filter.toml"
    save_profile(path, state, selection)
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    assert document["energy_filter_model"] == model_for(state)
    assert not set(ENERGY_FILTER_MULTIPOLE_KEYS) & document["devices"].keys()
    selected, values = read_profile(path)
    fresh = default_state()
    AssemblyCatalog().apply(fresh, selected)
    original_filter = fresh.energy_filter
    original_list = original_filter.multipoles
    original_components = tuple(original_list)
    original_aperture = original_filter._entrance_aperture_component
    original_geometry = [(
        element.frame.origin_m.copy(), element.frame.rotation_local_to_global.copy(),
        deepcopy(element.field_backend.envelope), element.housing_length_m,
        element.outer_radius_m, element.pole_zero_angle_rad,
    ) for element in original_components]
    apply_profile_values(fresh, values)
    assert fresh.energy_filter is original_filter
    assert fresh.energy_filter.multipoles is original_list
    assert fresh.energy_filter._entrance_aperture_component is original_aperture
    for index, element in enumerate(fresh.energy_filter.multipoles):
        assert element is original_components[index]
        origin, rotation, envelope, length, radius, pole_angle = original_geometry[index]
        np.testing.assert_array_equal(element.frame.origin_m, origin)
        np.testing.assert_array_equal(element.frame.rotation_local_to_global, rotation)
        assert element.field_backend.envelope == envelope
        assert (element.housing_length_m, element.outer_radius_m, element.pole_zero_angle_rad) == (length, radius, pole_angle)
    assert serialise_energy_filter(fresh.energy_filter) == before


@pytest.mark.parametrize("damage", [
    lambda model: model.update(unknown=1),
    lambda model: model.update(multipoles=model["multipoles"][:-1]),
    lambda model: model["multipoles"][0].update(role="entrance"),
    lambda model: model["multipoles"][0].update(key=ENERGY_FILTER_MULTIPOLE_KEYS[1]),
    lambda model: model["multipoles"][0].update(name="Wrong component"),
    lambda model: model["multipoles"][0].update(enabled="false"),
    lambda model: model["multipoles"][0]["field"].pop("skew_coefficients"),
    lambda model: model["multipoles"][0]["field"].update(normal_coefficients=[1.] * 5),
    lambda model: model["multipoles"][0]["calibration"].update(skew_trim_coefficients=[float("nan")] * 6),
])
def test_invalid_multipole_model_does_not_partially_apply_any_control(damage):
    state, _ = configured_filter()
    before = deepcopy(state.to_dict())
    model = model_for(state)
    damage(model)
    with pytest.raises(ValueError):
        apply_profile_values(state, {"objective_lens": {"percent": 12.5}, MODEL_KEY: model})
    assert state.to_dict() == before


def test_duplicate_device_controls_conflict_with_complete_carrier_model():
    state, _ = configured_filter()
    before = deepcopy(state.to_dict())
    with pytest.raises(ValueError, match="both devices and energy_filter_model"):
        apply_profile_values(state, {MODEL_KEY: model_for(state),
            ENERGY_FILTER_MULTIPOLE_KEYS[0]: {"enabled": False}, "objective_lens": {"percent": 12.5}})
    assert state.to_dict() == before


def test_partial_current_multipole_switch_remains_a_physical_field_control():
    state, _ = configured_filter()
    component = state.energy_filter.multipoles[0]
    before = model_for(state)
    apply_profile_values(state, {component.key: {"enabled": True}})
    after = model_for(state)
    before["multipoles"][0]["enabled"] = True
    assert after == before
    assert component is state.energy_filter.multipoles[0]


def test_carrier_geometry_and_identity_are_not_operating_profile_inputs():
    state, _ = configured_filter()
    targets = runtime_targets(state)
    for key in ENERGY_FILTER_MULTIPOLE_KEYS:
        fields = {item.name for item in editable_parameters(targets[key])}
        assert fields == {"enabled"}
    before = deepcopy(state.to_dict())
    for name in ("role", "housing_length_m", "outer_radius_m", "pole_zero_angle_rad"):
        with pytest.raises(ValueError, match="Unknown operating-profile field"):
            apply_profile_values(state, {ENERGY_FILTER_MULTIPOLE_KEYS[0]: {name: 1.}})
        assert state.to_dict() == before


def test_filter_model_cannot_install_or_bypass_unselected_hardware():
    configured, _ = configured_filter()
    state = default_state()
    before = deepcopy(state.to_dict())
    with pytest.raises(ValueError, match="installed energy filter"):
        apply_profile_values(state, {MODEL_KEY: model_for(configured)})
    assert state.to_dict() == before


def test_profile_writer_rejects_installation_mismatch_before_overwriting(tmp_path):
    state, selection = configured_filter()
    state.energy_filter.enabled = False
    path = tmp_path / "protected.toml"
    path.write_text("existing profile", encoding="utf-8")
    with pytest.raises(ValueError, match="energy_filter.enabled"):
        save_profile(path, state, selection)
    assert path.read_text(encoding="utf-8") == "existing profile"
