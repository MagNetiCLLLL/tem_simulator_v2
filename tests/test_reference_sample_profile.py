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


@pytest.mark.parametrize("version", [2, PROFILE_FORMAT_VERSION])
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


@pytest.mark.parametrize("version", [1, 2, 3, 4])
@pytest.mark.parametrize("key", ["si_110", "au_001"])
def test_legacy_preset_profile_has_one_reference_basis_and_manual_tail(tmp_path, version, key):
    relative = quaternion_from_euler_xyz_deg((11.0, 2.0, -8.0))
    sample = {
        "specimen_mode": "virtual", "specimen_preset_key": key,
        "specimen_rotation_x_deg": 11.0, "specimen_rotation_y_deg": 2.0,
        "specimen_rotation_z_deg": -8.0,
        "real_tail_atomic_number": 29, "real_tail_areal_density_atoms_nm2": 123.5,
        "real_tail_screening_angle_mrad": 7.25,
        "virtual_diffraction_relative_weight": 0.5, "diffraction_enabled": True,
    }
    model = {"orientation_quaternion_wxyz": list(relative), "virtual_regions": [{"density": 0.2}]}
    path = _write(tmp_path, version, sample, model)
    state = default_state()
    _, values = read_profile(path)
    assert apply_profile_values(state, values) == []
    basis = SimpleNamespace()
    apply_reference_sample(basis, key)
    expected = quaternion_multiply(relative, basis.specimen_orientation_quaternion_wxyz)
    assert state.sample.specimen_mode == "reference"
    assert state.sample.reference_sample_key == key
    assert Path(active_cif_path(state.sample)).is_file()
    assert np.allclose(quaternion_to_matrix(state.sample.specimen_orientation_quaternion_wxyz), quaternion_to_matrix(expected))
    assert state.sample.real_tail_material_source == "manual"
    assert state.sample.real_tail_screening_source == "manual"
    assert state.sample.real_tail_atomic_number == 29
    assert state.sample.real_tail_areal_density_atoms_nm2 == 123.5
    assert state.sample.real_tail_screening_angle_mrad == 7.25
    assert state.sample.virtual_interactions == state.sample.virtual_regions == []

    # Saving upgrades to the absolute CIF orientation; another load must not
    # compose the reference basis a second time.
    save_profile(path, state, AssemblyCatalog().default_selection())
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    assert document["format_version"] == PROFILE_FORMAT_VERSION == 6
    assert not any(name.startswith("virtual_") for name in document["devices"]["sample"])
    assert "diffraction_enabled" not in document["devices"]["sample"]
    assert "specimen_preset_key" not in document["devices"]["sample"]
    assert "virtual_regions" not in document["sample_model"]
    restored = default_state()
    assert apply_profile_values(restored, read_profile(path)[1]) == []
    assert restored.sample.specimen_orientation_quaternion_wxyz == pytest.approx(state.sample.specimen_orientation_quaternion_wxyz)


def test_legacy_missing_relative_orientation_does_not_reuse_new_reference_default(tmp_path):
    path = _write(tmp_path, 1, {"specimen_mode": "virtual", "specimen_preset_key": "si_110"})
    state = default_state()
    expected = SimpleNamespace()
    apply_reference_sample(expected, "si_110")
    apply_profile_values(state, read_profile(path)[1])
    assert state.sample.specimen_orientation_quaternion_wxyz == pytest.approx(expected.specimen_orientation_quaternion_wxyz)


def test_old_atomic_preset_source_is_converted_after_sample_model_rotation(tmp_path):
    relative = quaternion_from_euler_xyz_deg((7.0, -13.0, 0.0))
    path = _write(tmp_path, 4, {
        "specimen_mode": "atomic", "atomic_structure_source": "preset", "specimen_preset_key": "si_110",
    }, {"orientation_quaternion_wxyz": list(relative)})
    state = default_state()
    apply_profile_values(state, read_profile(path)[1])
    base = SimpleNamespace()
    apply_reference_sample(base, "si_110")
    assert state.sample.specimen_orientation_quaternion_wxyz == pytest.approx(
        quaternion_multiply(relative, base.specimen_orientation_quaternion_wxyz)
    )


@pytest.mark.parametrize("version", [1, 4, 5])
def test_external_cif_keeps_path_and_absolute_orientation(tmp_path, version):
    relative = quaternion_from_euler_xyz_deg((11.0, 2.0, -8.0))
    path = _write(tmp_path, version, {
        "specimen_mode": "atomic", "cif_path": "external-retained.cif",
        "specimen_rotation_x_deg": 11.0, "specimen_rotation_y_deg": 2.0,
        "specimen_rotation_z_deg": -8.0,
    }, {"orientation_quaternion_wxyz": list(relative)})
    state = default_state()
    apply_profile_values(state, read_profile(path)[1])
    assert active_cif_path(state.sample) == "external-retained.cif"
    assert state.sample.specimen_mode == "atomic"
    assert state.sample.specimen_orientation_quaternion_wxyz == pytest.approx(relative)


def test_unavailable_legacy_reference_does_not_partially_apply_profile(tmp_path):
    path = _write(tmp_path, 4, {"specimen_mode": "virtual", "specimen_preset_key": "unavailable-material"})
    state = default_state()
    state.objective_lens.cs_mm = 4.5
    before = deepcopy(state.to_dict())
    values = read_profile(path)[1]
    values["objective_lens"] = {"cs_mm": None}
    with pytest.raises(ValueError, match="unavailable"):
        apply_profile_values(state, values)
    assert state.to_dict() == before


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
    path = _write(tmp_path, 5, {"specimen_mode": "virtual"})
    state = default_state()
    before = deepcopy(state.to_dict())
    with pytest.raises(ValueError, match="atomic or reference"):
        apply_profile_values(state, read_profile(path)[1])
    assert state.to_dict() == before


def test_format_four_none_coefficients_survive_reference_migration(tmp_path):
    path = _write(tmp_path, 4, {"specimen_mode": "virtual"},
                  none_values={"objective_lens": ["cs_mm", "cc_mm"]})
    state = default_state()
    state.objective_lens.cs_mm = 3.0
    state.objective_lens.cc_mm = 4.0
    assert apply_profile_values(state, read_profile(path)[1]) == []
    assert state.objective_lens.cs_mm is None
    assert state.objective_lens.cc_mm is None
    assert state.sample.specimen_mode == "reference"


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
    assert apply_profile_values(restored, read_profile(path)[1]) == []
    assert restored.sample.real_tail_material_source == material
    assert restored.sample.real_tail_screening_source == screening
    assert restored.sample.real_tail_atomic_number == 29
    assert restored.sample.real_tail_areal_density_atoms_nm2 == 152.0
    assert restored.sample.real_tail_screening_angle_mrad == 4.0


@pytest.mark.parametrize("version", [1, 4])
@pytest.mark.parametrize("fields", [
    {"specimen_mode": "atomic", "cif_path": "unchanged.cif"},
    {"specimen_mode": "virtual", "atomic_structure_source": "cif", "cif_path": "unchanged.cif"},
    {"cif_path": "unchanged.cif"},
])
def test_old_imported_cif_without_rotation_has_legacy_identity(tmp_path, version, fields):
    path = _write(tmp_path, version, fields)
    state = default_state()
    assert apply_profile_values(state, read_profile(path)[1]) == []
    assert state.sample.specimen_mode == "atomic"
    assert state.sample.cif_path == "unchanged.cif"
    assert state.sample.specimen_orientation_quaternion_wxyz == pytest.approx((1.0, 0.0, 0.0, 0.0))


@pytest.mark.parametrize("values", [{}, {"sample": {"real_tail_max_angle_mrad": 200.0}}])
def test_partial_legacy_profile_without_source_or_rotation_retains_current_orientation(values):
    state = default_state()
    expected = quaternion_from_euler_xyz_deg((18.0, 5.0, -8.0))
    set_sample_orientation(state.sample, expected)
    assert apply_profile_values(state, {"__profile_format_version__": 4, **values}) == []
    assert state.sample.specimen_orientation_quaternion_wxyz == pytest.approx(expected)
