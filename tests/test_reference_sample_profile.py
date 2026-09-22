"""Profiles retain real sources, while legacy presets acquire their CIF basis once."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tomllib

import numpy as np
import pytest
import tomli_w

from temsim.assembly_catalog import AssemblyCatalog
from temsim.optics.column import default_state
from temsim.profile_io import PROFILE_FORMAT_VERSION, apply_profile_values, read_profile, save_profile
from temsim.runtime_parameters import editable_parameters, runtime_targets, validate_runtime_assignment
from temsim.specimen.geometry import quaternion_from_euler_xyz_deg, quaternion_multiply, quaternion_to_matrix, set_sample_orientation
from temsim.specimen.reference_catalog import apply_reference_sample
from temsim.specimen.source import active_cif_path


def _write(tmp_path, version, sample, model=None, **document_fields):
    selection = AssemblyCatalog().default_selection()
    document = {
        "format_version": version,
        "assembly": {"gun": selection.gun, "column": selection.column, "recording": selection.recording},
        "devices": {"sample": sample},
        **document_fields,
    }
    if model is not None:
        document["sample_model"] = model
    path = tmp_path / "profile.toml"
    path.write_text(tomli_w.dumps(document), encoding="utf-8")
    return path


@pytest.mark.parametrize("version", [PROFILE_FORMAT_VERSION])
@pytest.mark.parametrize("sigma", [float("nan"), float("inf"), -float("inf"), 0.0, -0.01])
def test_profile_rejects_invalid_element_rms_without_applying_other_fields(tmp_path, version, sigma):
    state = default_state()
    before = deepcopy(state.to_dict())
    path = _write(
        tmp_path, version, {"thickness_nm": 17.5, "specimen_mode": "reference"},
        {"frozen_phonon_sigma_by_element_angstrom": {"Si": sigma}},
    )
    with pytest.raises(ValueError, match="finite and positive"):
        apply_profile_values(state, read_profile(path)[1])
    assert state.to_dict() == before


@pytest.mark.parametrize("version", range(1, PROFILE_FORMAT_VERSION))
def test_old_profiles_are_rejected_without_rewriting_the_file(tmp_path, version):
    path = _write(tmp_path, version, {"specimen_mode": "virtual"})
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Unsupported operating-profile format"):
        read_profile(path)
    assert path.read_bytes() == before








@pytest.mark.parametrize("version", [PROFILE_FORMAT_VERSION])
def test_external_cif_keeps_path_and_absolute_orientation(tmp_path, version):
    relative = quaternion_from_euler_xyz_deg((11.0, 2.0, -8.0))
    path = _write(tmp_path, version, {
        "specimen_mode": "atomic", "cif_path": "external-retained.cif",
    }, {"orientation_quaternion_wxyz": list(relative)})
    state = default_state()
    apply_profile_values(state, read_profile(path)[1])
    assert active_cif_path(state.sample) == "external-retained.cif"
    assert state.sample.specimen_mode == "atomic"
    assert state.sample.specimen_orientation_quaternion_wxyz == pytest.approx(relative)





def test_sample_runtime_hides_retired_controls_and_source_workflow():
    target = runtime_targets(default_state())["sample"]
    names = {parameter.name for parameter in editable_parameters(target)}
    assert not any(name.startswith("virtual_") for name in names)
    assert not names & {"specimen_mode", "reference_sample_key", "cif_path", "specimen_preset_key",
                        "diffraction_enabled", "g_inv_nm", "excitation_error_inv_nm", "rocking_width_inv_nm",
                        "diffuse_broadening_mrad"}
    assert {"real_tail_material_source", "real_tail_screening_source", "thickness_nm"} <= names


@pytest.mark.parametrize("field,valid,invalid", [
    ("specimen_mode", "reference", "virtual"),
    ("real_tail_material_source", "structure", "preset"),
    ("real_tail_screening_source", "moliere", "auto"),
])
def test_runtime_source_enums_reject_invalid_values(field, valid, invalid):
    target = runtime_targets(default_state())["sample"]
    assert validate_runtime_assignment(target, field, valid) == valid
    with pytest.raises(ValueError):
        validate_runtime_assignment(target, field, invalid)


def test_format_five_explicit_virtual_mode_is_rejected(tmp_path):
    path = _write(tmp_path, PROFILE_FORMAT_VERSION, {"specimen_mode": "virtual"})
    state = default_state()
    before = deepcopy(state.to_dict())
    with pytest.raises(ValueError, match="atomic or reference"):
        apply_profile_values(state, read_profile(path)[1])
    assert state.to_dict() == before





@pytest.mark.parametrize("material,screening", [("structure", "moliere"), ("manual", "manual")])
def test_new_profile_preserves_tail_source_choices_and_manual_overrides(tmp_path, material, screening):
    state = default_state()
    state.sample.real_tail_material_source = material
    state.sample.real_tail_screening_source = screening
    state.sample.real_tail_atomic_number = 29
    state.sample.real_tail_areal_density_atoms_nm2 = 152.0
    state.sample.real_tail_screening_angle_mrad = 4.0
    path = tmp_path / "saved.toml"
    save_profile(path, state, AssemblyCatalog().default_selection())
    restored = default_state()
    assert apply_profile_values(restored, read_profile(path)[1]) is None
    assert restored.sample.real_tail_material_source == material
    assert restored.sample.real_tail_screening_source == screening
    assert restored.sample.real_tail_atomic_number == 29
    assert restored.sample.real_tail_areal_density_atoms_nm2 == 152.0
    assert restored.sample.real_tail_screening_angle_mrad == 4.0





@pytest.mark.parametrize("values", [{}, {"sample": {"real_tail_max_angle_mrad": 200.0}}])
def test_partial_current_profile_without_source_or_rotation_retains_current_orientation(values):
    state = default_state()
    expected = quaternion_from_euler_xyz_deg((18.0, 5.0, -8.0))
    set_sample_orientation(state.sample, expected)
    assert apply_profile_values(state, {"__profile_format_version__": PROFILE_FORMAT_VERSION, **values}) is None
    assert state.sample.specimen_orientation_quaternion_wxyz == pytest.approx(expected)
