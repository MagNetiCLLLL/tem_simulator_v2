"""An explicitly unconfigured current sample must not acquire reference atoms."""
from copy import deepcopy

import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.optics.column import default_state
from temsim.optics.model import State
from temsim.profile_io import apply_profile_values, read_profile, save_profile
from temsim.specimen.geometry import quaternion_from_euler_xyz_deg
from temsim.specimen.source import active_cif_path, specimen_interactions_active


@pytest.mark.parametrize("version", [0, 52, 64, 70, 71, 76, 77, 79])
def test_noncurrent_state_is_rejected_without_mutating_input(version):
    data = default_state().to_dict()
    data["schema_version"] = version
    before = deepcopy(data)
    with pytest.raises(ValueError, match="schema"):
        State.from_dict(data)
    assert data == before


def test_current_empty_cif_roundtrips_state_and_profile_without_inventing_atoms(tmp_path):
    state = default_state()
    state.sample.specimen_mode = "atomic"
    state.sample.cif_path = ""
    orientation = quaternion_from_euler_xyz_deg((11., 2., -8.))
    state.sample.specimen_orientation_quaternion_wxyz = orientation
    restored = State.from_dict(state.to_dict())
    path = tmp_path / "empty-cif.toml"
    save_profile(path, restored, AssemblyCatalog().default_selection())
    loaded = default_state()
    apply_profile_values(loaded, read_profile(path)[1])
    for sample in (restored.sample, loaded.sample):
        assert sample.specimen_mode == "atomic"
        assert sample.cif_path == active_cif_path(sample) == ""
        assert not specimen_interactions_active(sample)
        assert sample.specimen_orientation_quaternion_wxyz == pytest.approx(orientation)
