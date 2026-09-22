"""Current filter state is explicit; no former two-carrier reconstruction."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.component_keys import ENERGY_FILTER_MULTIPOLE_KEYS
from temsim.optics.column import default_state
from temsim.optics.energy_filter import (
    energy_filter_from_dict, serialise_energy_filter, ensure_energy_filter,
)
from temsim.runtime_parameters import RuntimeTarget, editable_parameters, runtime_targets


def filter_state():
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, replace(catalog.default_selection(), recording="Energy Filter"))
    return state


def test_current_filter_roundtrip_preserves_nondefault_state_without_reconfiguration():
    state = filter_state()
    ef = state.energy_filter
    ef.energy_slit.inserted = True
    ef.energy_slit.gap_m = 11e-6
    ef.energy_slit.centre_m = 2e-6
    ef.energy_slit.zero_loss_offset_m = -3e-6
    ef.energy_slit.requested_width_ev = 12.
    ef.energy_slit.requested_centre_loss_ev = 19.
    ef.output_detector_inserted = True
    ef.zebra_detector.inserted = False
    ef.zebra_detector.enabled = False
    ef.bias_tube.enabled = True
    ef.bias_tube.offset_ev = 20.
    ef.fast_shutter.enabled = False
    ef.fast_shutter.open = False
    ef.camera_deflector.enabled = False
    ef.camera_deflector.active_strip = 3
    ef.multi_eels_region_count = 4
    for index, element in enumerate(ef.multipoles):
        element.multipole_field.set_component(2, normal=index + .125, skew=-index - .375)
        element.calibration.normal_trim_coefficients[1] = .0125 * index
        element.calibration.skew_trim_coefficients[2] = -.0025 * index
    before = deepcopy(vars(ef.energy_slit))
    payload = serialise_energy_filter(ef)
    assert vars(ef.energy_slit) == before
    assert not {"entrance_m12", "exit_m12", "m12_frames_placed", "optical_integration_enabled"} & payload.keys()
    assert [item["key"] for item in payload["multipoles"]] == list(ENERGY_FILTER_MULTIPOLE_KEYS)
    ensure_energy_filter(state)
    assert serialise_energy_filter(ef) == payload
    restored = energy_filter_from_dict(deepcopy(payload), state.beam_voltage_kv)
    assert serialise_energy_filter(restored) == payload
    for actual, expected in zip(restored.multipoles, ef.multipoles, strict=True):
        np.testing.assert_array_equal(actual.frame.origin_m, expected.frame.origin_m)
        np.testing.assert_array_equal(actual.multipole_field.normal_coefficients, expected.multipole_field.normal_coefficients)
        np.testing.assert_array_equal(actual.multipole_field.skew_coefficients, expected.multipole_field.skew_coefficients)


@pytest.mark.parametrize("change", [
    lambda p: p.pop("multipoles"),
    lambda p: p.update(multipoles=p["multipoles"][:2]),
    lambda p: p.update(entrance_m12=p["multipoles"][2]),
    lambda p: p.update(exit_m12=p["multipoles"][3]),
    lambda p: p.update(optical_integration_enabled=True),
    lambda p: p.update(slit_width_ev=10.),
    lambda p: p.update(selected_loss_ev=0.),
    lambda p: p["multipoles"][0].update(key=ENERGY_FILTER_MULTIPOLE_KEYS[1]),
    lambda p: p["multipoles"][0].update(role="entrance"),
    lambda p: p["multipoles"][0]["field"].pop("skew_coefficients"),
    lambda p: p["multipoles"][0]["calibration"].pop("normal_trim_coefficients"),
    lambda p: p["multipoles"][0].update(enabled="false"),
    lambda p: p.update(energy_slit=None),
    lambda p: p["energy_slit"].pop("centre_m"),
    lambda p: p["energy_slit"].update(inserted="false"),
    lambda p: p.update(fast_shutter=None),
    lambda p: p["fast_shutter"].update(open="false"),
    lambda p: p["camera_deflector"].update(active_strip=2.5),
    lambda p: p["zebra_detector"].update(unknown=1),
])
def test_incomplete_or_retired_filter_records_are_rejected(change):
    state = filter_state()
    payload = serialise_energy_filter(state.energy_filter)
    change(payload)
    with pytest.raises(ValueError):
        energy_filter_from_dict(payload, state.beam_voltage_kv)


def test_installed_filter_cannot_be_disabled_as_a_transport_bypass():
    state = filter_state()
    state.energy_filter.enabled = False
    with pytest.raises(ValueError, match="physical installation"):
        ensure_energy_filter(state)
    assert state.energy_filter.enabled is False


def test_virtual_layout_and_filter_installation_have_no_phantom_editable_controls():
    state = filter_state()
    targets = runtime_targets(state)
    virtual = [target for target in targets.values() if getattr(target.obj, "KIND", "") == "virtual_layout"]
    assert virtual
    assert all(editable_parameters(target) == () for target in virtual)
    fields = {p.name for p in editable_parameters(targets["energy_filter"])}
    assert "enabled" not in fields
    assert "optical_integration_enabled" not in fields
    physical = state.ac_deflector
    fields = {p.name for p in editable_parameters(RuntimeTarget(physical.key, physical.name, physical))}
    assert {"kick_x_mrad", "kick_y_mrad", "upper_coil_gain", "scan_enabled"} <= fields


def test_complete_current_state_retains_filter_operating_choices():
    from temsim.optics.model import State
    state = filter_state()
    state.energy_filter.fast_shutter.open = False
    state.energy_filter.fast_shutter.enabled = False
    state.energy_filter.energy_slit.inserted = True
    state.energy_filter.zebra_detector.inserted = False
    state.energy_filter.camera_deflector.active_strip = 3
    state.energy_filter.multi_eels_region_count = 4
    payload = state.to_dict()
    restored = State.from_dict(deepcopy(payload))
    assert restored.to_dict()["energy_filter"] == payload["energy_filter"]


def test_slit_owns_window_requests_and_physical_configuration():
    from temsim.optics.energy_filter import configure_energy_slit_from_software
    state = filter_state()
    ef = state.energy_filter
    ef.slit_width_ev = 17.
    ef.selected_loss_ev = 33.
    assert ef.energy_slit.requested_width_ev == 17.
    assert ef.energy_slit.requested_centre_loss_ev == 33.
    configure_energy_slit_from_software(ef)
    assert ef.energy_slit.derived_width_ev == pytest.approx(17.)
    assert ef.energy_slit.derived_centre_loss_ev == pytest.approx(33.)
    payload = serialise_energy_filter(ef)
    assert not {"slit_width_ev", "selected_loss_ev"} & payload.keys()
    assert payload["energy_slit"]["requested_width_ev"] == 17.
    assert payload["energy_slit"]["requested_centre_loss_ev"] == 33.


@pytest.mark.parametrize("dispersion", [0.75, -2.5])
@pytest.mark.parametrize("zero_offset", [0., 19e-6])
def test_slit_inverse_energy_window_uses_displacement_units_and_offset(dispersion, zero_offset):
    from temsim.optics.energy_filter_slit import create_energy_selection_slit
    slit = create_energy_selection_slit(dispersion_um_per_ev=dispersion)
    slit.zero_loss_offset_m = zero_offset
    slit.configure_energy_window(41., 7.)
    assert slit.derived_centre_loss_ev == pytest.approx(41.)
    assert slit.derived_width_ev == pytest.approx(7.)
    assert slit.centre_m == pytest.approx(zero_offset + dispersion * 41e-6)
