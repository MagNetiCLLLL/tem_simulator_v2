"""Current component identity checks; no ray propagation or calibration."""
from dataclasses import replace

import pytest

from temsim.component_keys import (
    require_current_component_key, require_current_binding_key,
    require_current_recording_plane_key,
)
from temsim.optics import probe_corrector as probe
from temsim.optics.condenser_deflector import create_condenser_deflector, condenser_deflector_from_dict
from temsim.optics.mini_condenser import create_mini_condenser, mini_condenser_from_dict
from temsim.optics.image_diffraction_deflector import create_image_diffraction_deflector, image_diffraction_deflector_from_dict


@pytest.mark.parametrize("key", ["c1", "tl22", "cond_def", "ic_hp1", "ceta", "adf", "dc_deflector", "energy_filter_entrance_m12"])
def test_retired_keys_are_rejected_without_translation(key):
    with pytest.raises(ValueError, match="Retired"):
        require_current_component_key(key)
    with pytest.raises(ValueError, match="Retired"):
        require_current_binding_key("lens:" + key)


def test_current_and_custom_identifiers_are_unchanged():
    assert require_current_component_key("condenser_lens_1") == "condenser_lens_1"
    assert require_current_component_key("custom_lens") == "custom_lens"
    assert require_current_recording_plane_key("camera") == "camera"
    assert require_current_binding_key("lens:custom_lens") == "lens:custom_lens"


_LOADERS = [
    (probe.create_adapter_lens, probe.adapter_lens_from_dict),
    (probe.create_dph2_deflector, probe.dph2_deflector_from_dict),
    (probe.create_dp22_deflector, probe.dp22_deflector_from_dict),
    (probe.create_qph2_quadrupole, probe.qph2_quadrupole_from_dict),
    (probe.create_hp2_hexapole, probe.hp2_hexapole_from_dict),
    (probe.create_hpc_hexapole, probe.hpc_hexapole_from_dict),
    (probe.create_tl21_lens, probe.tl21_lens_from_dict),
    (probe.create_tl22_lens, probe.tl22_lens_from_dict),
    (probe.create_dp12_scan_deflector, probe.dp12_scan_deflector_from_dict),
    (create_condenser_deflector, condenser_deflector_from_dict),
    (create_mini_condenser, mini_condenser_from_dict),
    (create_image_diffraction_deflector, image_diffraction_deflector_from_dict),
]


@pytest.mark.parametrize("factory,loader", _LOADERS, ids=lambda function: function.__name__)
def test_specific_loader_requires_its_exact_current_key(factory, loader):
    from dataclasses import asdict
    component = factory()
    payload = component.to_dict() if hasattr(component, "to_dict") else asdict(component)
    assert loader(payload).key == component.key
    for invalid in (None, "c1", "some_other_current_component"):
        with pytest.raises(ValueError, match="key|Retired"):
            loader({**payload, "key": invalid})
    with pytest.raises(ValueError, match="key"):
        loader({key: value for key, value in payload.items() if key != "key"})


def test_layout_only_dp12_does_not_accept_a_hidden_physical_kick():
    from dataclasses import asdict
    component = probe.create_dp12_scan_deflector()
    payload = asdict(component)
    for field in ("upper_x_mrad", "upper_y_mrad", "lower_x_mrad", "lower_y_mrad"):
        with pytest.raises(ValueError, match="no physical kick"):
            probe.dp12_scan_deflector_from_dict({**payload, field: .1})
    restored = probe.dp12_scan_deflector_from_dict({**payload, "upper_z_mm": 123., "lower_z_mm": 456.})
    assert restored.upper_z_mm == component.upper_z_mm
    assert restored.lower_z_mm == component.lower_z_mm
    assert restored.kick_events() == ()


def test_default_filter_installation_matches_catalog_and_explicit_installation_works():
    from temsim.optics.column import default_state
    from temsim.assembly_catalog import AssemblyCatalog
    state = default_state()
    assert not state.energy_filter_installed
    assert state.energy_filter_mode == "no_energy_filter"
    assert not state.energy_filter.enabled
    catalog = AssemblyCatalog()
    selected = catalog.default_selection()
    catalog.apply(state, replace(selected, recording="Energy Filter"))
    assert state.energy_filter_installed and state.energy_filter.enabled
    assert len(state.energy_filter.multipoles) == 10
    catalog.apply(state, selected)
    assert not state.energy_filter_installed and not state.energy_filter.enabled


@pytest.mark.parametrize("kind", ["retired", "duplicate", "wrong_type"])
def test_corrector_structure_rejects_bad_current_records_without_mutating_them(kind):
    from temsim.optics.column import default_state
    from temsim.optics.corrector_structure import CorrectorElement, ensure_corrector_structure
    state = default_state()
    if kind == "retired":
        state.corrector_elements[0].key = "dph2"
    elif kind == "duplicate":
        state.corrector_elements.append(state.corrector_elements[0])
    else:
        old = state.corrector_elements[0]
        state.corrector_elements[0] = CorrectorElement(old.key, old.name, old.z_mm, 0., "deflector", "probe", "#fff")
    previous = tuple(state.corrector_elements)
    with pytest.raises(ValueError, match="Retired|Duplicate|current corrector"):
        ensure_corrector_structure(state)
    assert tuple(state.corrector_elements) == previous
