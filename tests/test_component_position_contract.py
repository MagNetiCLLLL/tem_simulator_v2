"""Position edits keep mechanical/optical offsets and persistence semantics."""
from dataclasses import asdict, fields, replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.probe_corrector import (
    create_adapter_lens, create_dph2_deflector, create_hp2_hexapole,
    create_qph2_quadrupole, create_tl21_lens,
)
from temsim.optics.diffraction_lens import create_diffraction_lens
from temsim.optics.diffraction_stigmator import create_diffraction_stigmator

TIP_FACTORIES = [create_adapter_lens, create_tl21_lens, create_hp2_hexapole,
                 create_qph2_quadrupole, create_dph2_deflector]
INSTALLATION_FACTORIES = [create_diffraction_lens, create_diffraction_stigmator]


@pytest.mark.parametrize("factory", TIP_FACTORIES)
def test_tip_position_edits_preserve_offset_and_explicit_reference(factory):
    component = factory()
    centre, reference = component.mechanical_center_from_tip_mm, component.optical_reference_from_tip_mm
    component.mechanical_center_from_tip_mm = str(centre + 2.5)
    assert component.optical_reference_from_tip_mm == reference + 2.5
    assert component.z_mm == reference + 2.5
    component.optical_reference_from_tip_mm = str(reference + 1.25)
    assert component.z_mm == reference + 1.25
    assert component.mechanical_center_from_tip_mm == centre + 2.5
    # Direct z writes historically do not redefine the tip-relative reference.
    component.z_mm = str(reference + 9.)
    assert component.optical_reference_from_tip_mm == reference + 1.25
    assert component.apply_optical_position() is component
    assert component.z_mm == reference + 1.25


@pytest.mark.parametrize("factory", TIP_FACTORIES)
def test_dataclass_init_and_temporarily_disabled_coupling(factory):
    original = factory()
    component = replace(original, z_mm=12., mechanical_center_from_tip_mm=21.,
                        optical_reference_from_tip_mm=24.)
    assert (component.z_mm, component.mechanical_center_from_tip_mm,
            component.optical_reference_from_tip_mm) == (12., 21., 24.)
    object.__setattr__(component, "_position_coupling_ready", False)
    component.mechanical_center_from_tip_mm = 31.
    assert component.z_mm == 12. and component.optical_reference_from_tip_mm == 24.
    object.__setattr__(component, "_position_coupling_ready", True)
    component.mechanical_center_from_tip_mm = 32.
    assert component.z_mm == 25. and component.optical_reference_from_tip_mm == 25.
    assert asdict(type(component)(**asdict(component))) == asdict(component)


@pytest.mark.parametrize("factory", INSTALLATION_FACTORIES)
@pytest.mark.parametrize("active", ["standalone", "image_corrected"])
def test_installation_edit_switch_and_anchor_resolution(factory, active):
    component = factory().select_installation(active)
    inactive = "image_corrected" if active == "standalone" else "standalone"
    active_z = component.z_mm
    inactive_ref = getattr(component, f"{inactive}_optical_reference_z_mm")
    centre_name = f"{inactive}_mechanical_center_below_sample_mm"
    setattr(component, centre_name, getattr(component, centre_name) + 3.)
    assert component.z_mm == active_z
    assert getattr(component, f"{inactive}_optical_reference_z_mm") == inactive_ref + 3.
    component.z_mm = active_z + 7.
    assert getattr(component, f"{active}_optical_reference_z_mm") == active_z + 7.
    component.select_installation(inactive)
    assert component.z_mm == inactive_ref + 3.
    component.select_installation(active)
    assert component.z_mm == active_z + 7.
    setattr(component, f"{active}_optical_reference_z_mm", active_z + 8.)
    assert component.z_mm == active_z + 8.
    anchor = SimpleNamespace(mechanical_center_below_sample_mm=113., optical_reference_z_mm=221.)
    offset = component.mechanical_center_downstream_of_anchor_mm
    geometry = component.resolve_against(anchor)
    assert geometry.mechanical_center_below_sample_mm == 113. + offset
    assert geometry.optical_reference_z_mm == component.z_mm == 221. + offset
    assert component.optical_reference_downstream_of_anchor_mm == offset
    assert getattr(component, f"{inactive}_optical_reference_z_mm") == inactive_ref + 3.
    assert asdict(type(component)(**asdict(component))) == asdict(component)


def test_round_and_adapter_keep_distinct_field_normalisation_validation_and_schema():
    from temsim.optics.condenser_lens import AxialFieldTerm

    adapter, round_lens = create_adapter_lens(), create_tl21_lens()
    for component in (adapter, round_lens):
        component.gaussian = [AxialFieldTerm(amplitude=2., sigma=1., offset=0.)]
        component.normalise_profile_peak = True
        component.enabled = True
    np.testing.assert_allclose(adapter.magnetic_field_t([adapter.z_mm]),
                               [2 * adapter.polarity * adapter.scale()])
    np.testing.assert_allclose(round_lens.magnetic_field_t([round_lens.z_mm]),
                               [round_lens.polarity * round_lens.scale()])
    assert "corrector" not in {field.name for field in fields(adapter)}
    assert "corrector" in {field.name for field in fields(round_lens)}
    adapter.pole_gap_mm = adapter.mechanical_length_mm + 1.
    assert adapter.validate() is adapter
    round_lens.pole_gap_mm = round_lens.mechanical_length_mm + 1.
    with pytest.raises(ValueError, match="pole gap"):
        round_lens.validate()


def test_state_json_entry_preserves_current_component_state(tmp_path):
    import json
    from temsim.optics.column import default_state
    from temsim.state import save, load

    state = default_state()
    path = tmp_path / "state.json"
    save(state, path)
    restored = load(path)
    # JSON represents both tuple- and list-valued vectors as arrays.
    assert json.dumps(restored.to_dict(), sort_keys=True) == json.dumps(state.to_dict(), sort_keys=True)
