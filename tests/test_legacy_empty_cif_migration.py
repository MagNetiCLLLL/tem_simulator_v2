"""An explicitly unconfigured imported sample must not acquire reference atoms."""

from types import SimpleNamespace

import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.optics.column import default_state
from temsim.optics.model import State
from temsim.profile_io import apply_profile_values, read_profile, save_profile
from temsim.specimen.geometry import quaternion_from_euler_xyz_deg, quaternion_multiply
from temsim.specimen.reference_catalog import apply_reference_sample
from temsim.specimen.source import active_cif_path, specimen_interactions_active


RELATIVE_ROTATION = (11.0, 2.0, -8.0)
RELATIVE_QUATERNION = quaternion_from_euler_xyz_deg(RELATIVE_ROTATION)


def _legacy_state(version, *, orientation=True):
    data = default_state().to_dict()
    data["schema_version"] = version
    sample = data["sample"]
    sample.update(specimen_mode="atomic", cif_path="", specimen_preset_key="si_110",
                  inserted=True, thickness_nm=5.0)
    sample.pop("reference_sample_key", None)
    for axis, angle in zip("xyz", RELATIVE_ROTATION):
        sample[f"specimen_rotation_{axis}_deg"] = angle if orientation else 0.0
    if orientation:
        sample["specimen_orientation_quaternion_wxyz"] = RELATIVE_QUATERNION
    else:
        sample.pop("specimen_orientation_quaternion_wxyz", None)
    return data


def _assert_unconfigured(sample, expected_orientation):
    assert sample.specimen_mode == "atomic"
    assert sample.cif_path == active_cif_path(sample) == ""
    assert not specimen_interactions_active(sample)
    assert sample.specimen_orientation_quaternion_wxyz == pytest.approx(expected_orientation)


@pytest.mark.parametrize("version", [71, 76])
@pytest.mark.parametrize("orientation", [False, True])
def test_post_selector_state_empty_atomic_path_stays_unconfigured(version, orientation):
    restored = State.from_dict(_legacy_state(version, orientation=orientation))
    expected = RELATIVE_QUATERNION if orientation else (1.0, 0.0, 0.0, 0.0)
    _assert_unconfigured(restored.sample, expected)
    _assert_unconfigured(State.from_dict(restored.to_dict()).sample, expected)


@pytest.mark.parametrize("version,mode,source", [
    (70, "atomic", ""),
    (70, None, ""),
    (76, "atomic", "preset"),
    (76, "virtual", ""),
])
def test_legacy_implicit_or_explicit_preset_acquires_reference_basis_once(version, mode, source):
    data = _legacy_state(version)
    if mode is None:
        data["sample"].pop("specimen_mode")
    else:
        data["sample"]["specimen_mode"] = mode
    if source:
        data["sample"]["atomic_structure_source"] = source
    restored = State.from_dict(data)
    basis = SimpleNamespace()
    apply_reference_sample(basis, "si_110")
    expected = quaternion_multiply(RELATIVE_QUATERNION, basis.specimen_orientation_quaternion_wxyz)
    assert restored.sample.specimen_mode == "reference"
    assert restored.sample.reference_sample_key == "si_110"
    assert active_cif_path(restored.sample)
    assert specimen_interactions_active(restored.sample)
    assert restored.sample.specimen_orientation_quaternion_wxyz == pytest.approx(expected)
    assert State.from_dict(restored.to_dict()).sample.specimen_orientation_quaternion_wxyz == pytest.approx(expected)


def test_explicit_legacy_cif_source_overrides_schema_70_implicit_preset():
    data = _legacy_state(70)
    data["sample"]["atomic_structure_source"] = "cif"
    _assert_unconfigured(State.from_dict(data).sample, RELATIVE_QUATERNION)


@pytest.mark.parametrize("version", [1, 4])
@pytest.mark.parametrize("source", ["", "cif"])
def test_legacy_profile_empty_atomic_path_remains_empty_after_upgrade(tmp_path, version, source):
    fields = {"specimen_mode": "atomic", "cif_path": "", "specimen_preset_key": "si_110"}
    if source:
        fields["atomic_structure_source"] = source
    values = {
        "__profile_format_version__": version,
        "sample": fields,
        "__sample_model__": {"orientation_quaternion_wxyz": list(RELATIVE_QUATERNION)},
    }
    restored = default_state()
    assert apply_profile_values(restored, values) == []
    _assert_unconfigured(restored.sample, RELATIVE_QUATERNION)
    path = tmp_path / "upgraded.toml"
    save_profile(path, restored, AssemblyCatalog().default_selection())
    reloaded = default_state()
    assert apply_profile_values(reloaded, read_profile(path)[1]) == []
    _assert_unconfigured(reloaded.sample, RELATIVE_QUATERNION)


def test_legacy_empty_cif_profile_without_orientation_keeps_legacy_identity():
    restored = default_state()
    assert apply_profile_values(restored, {
        "__profile_format_version__": 4,
        "sample": {"specimen_mode": "atomic", "cif_path": "", "specimen_preset_key": "si_110"},
    }) == []
    _assert_unconfigured(restored.sample, (1.0, 0.0, 0.0, 0.0))
